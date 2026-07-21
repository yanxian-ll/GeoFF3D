# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Script to benchmark the image calibration performance

Modified:
- get_all_info_for_metric_computation is adapted from dense_n_view_benchmark
- Add ABSOLUTE depth metrics (z-depth, in meters):
  * z_depth_abs_mae, z_depth_abs_rmse
  * inlier ratios within thresholds: 0.5m / 1m / 2m / 5m

Added in this version:
- Save ALL visualization results (for every sample and every view):
  * RGB / MVS depth / Pred depth / GT depth
  * error maps clipped by thresholds (0.5/1/2/5m)
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf

from mapanything.datasets import get_test_many_ar_data_loader
from mapanything.models import init_model
from mapanything.utils.geometry import (
    geotrf,
    inv,
    quaternion_to_rotation_matrix,
    transform_pose_using_quats_and_trans_2_to_1,
)
from mapanything.utils.metrics import (
    l2_distance_of_unit_ray_directions_to_angular_error,
)
from mapanything.utils.misc import StreamToLogger

from mapanything.utils.image import rgb

log = logging.getLogger(__name__)


def save_triplet_vis(
    save_path: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    mvs: np.ndarray,
    pred: np.ndarray,
    err: np.ndarray,
    mask: np.ndarray,
    title_prefix: str = "",
    cmap_depth: str = "viridis",
    cmap_err: str = "magma",
    thr_list=(0.5, 1.0, 2.0, 5.0),
):
    assert mvs.shape == pred.shape == err.shape == mask.shape
    mask = mask.astype(bool)

    if mask.sum() > 0:
        valid_depth = np.concatenate([gt[mask].reshape(-1), pred[mask].reshape(-1)])
        vmin = float(np.nanpercentile(valid_depth, 1)) - 10
        vmax = float(np.nanpercentile(valid_depth, 99)) + 10
    else:
        vmin, vmax = 0.0, 1.0
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    mvs_show = mvs.astype(np.float32, copy=True)
    gt_show = gt.astype(np.float32, copy=True)
    pred_show = pred.astype(np.float32, copy=True)
    err_show = err.astype(np.float32, copy=True)

    gt_show[~mask] = np.nan
    err_show[~mask] = np.nan
    mvs_show[mvs_show <= 0.1] = np.nan

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    fig, axes = plt.subplots(2, 4, figsize=(22, 10), dpi=200, constrained_layout=True)
    cbar_kw = dict(orientation="vertical", fraction=0.04, pad=0.02, shrink=0.95)

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"{title_prefix}RGB")
    axes[0, 0].axis("off")

    ax_mvs = axes[0, 1]
    im_mvs = ax_mvs.imshow(mvs_show, vmin=vmin, vmax=vmax, cmap=cmap_depth)
    ax_mvs.set_title(f"{title_prefix}MVS depth")
    ax_mvs.axis("off")
    fig.colorbar(im_mvs, ax=ax_mvs, **cbar_kw).set_label("Depth")

    ax_pred = axes[0, 2]
    im_pred = ax_pred.imshow(pred_show, vmin=vmin, vmax=vmax, cmap=cmap_depth)
    ax_pred.set_title(f"{title_prefix}Pred depth")
    ax_pred.axis("off")
    fig.colorbar(im_pred, ax=ax_pred, **cbar_kw).set_label("Depth")

    ax_gt = axes[0, 3]
    im_gt = ax_gt.imshow(gt_show, vmin=vmin, vmax=vmax, cmap=cmap_depth)
    ax_gt.set_title(f"{title_prefix}GT depth")
    ax_gt.axis("off")
    fig.colorbar(im_gt, ax=ax_gt, **cbar_kw).set_label("Depth")

    total_valid = int(mask.sum())
    for j, thr in enumerate(thr_list):
        ax = axes[1, j]
        thr = float(thr)
        err_clip = np.clip(err.astype(np.float32, copy=False), 0.0, thr)
        err_clip_show = err_clip.copy()
        err_clip_show[~mask] = np.nan

        if total_valid > 0:
            within = ((err <= thr) & mask).sum()
            pct = 100.0 * float(within) / float(total_valid)
        else:
            pct = 0.0

        im_thr = ax.imshow(err_clip_show, vmin=0.0, vmax=thr, cmap=cmap_err)
        ax.set_title(rf"$|e|\leq{thr:g}$ clip  ({pct:.1f}%)")
        ax.axis("off")
        fig.colorbar(im_thr, ax=ax, **cbar_kw).set_label("Abs error (clipped)")

    if title_prefix:
        fig.suptitle(title_prefix.strip(), fontsize=12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def global_scale_from_pointmaps(
    gt_pts_list,
    pr_pts_list,
    masks_list,
    max_samples_per_view=20000,
    eps=1e-8,
):
    """
    Estimate ONE global scale s such that pr * s matches gt scale.
    Uses ratio of point norms (no correspondences required):
        s = median( ||gt|| / ||pr|| ) over valid points across all views.

    gt_pts_list/pr_pts_list: list of (H,W,3) torch tensors (same frame: view0)
    masks_list: list of (H,W) torch tensors/bool
    return: s (float)
    """
    ratios = []
    for gt, pr, m in zip(gt_pts_list, pr_pts_list, masks_list):
        gt = gt.detach().cpu().numpy()
        pr = pr.detach().cpu().numpy()
        m = m.detach().cpu().numpy().astype(bool)

        if gt.ndim == 3:
            gt_v = gt[m].reshape(-1, 3)
            pr_v = pr[m].reshape(-1, 3)
        else:
            gt_v = gt.reshape(-1, 3)
            pr_v = pr.reshape(-1, 3)

        if gt_v.shape[0] == 0:
            continue

        if gt_v.shape[0] > max_samples_per_view:
            idx = np.random.choice(gt_v.shape[0], max_samples_per_view, replace=False)
            gt_v = gt_v[idx]
            pr_v = pr_v[idx]

        gt_n = np.linalg.norm(gt_v, axis=1)
        pr_n = np.linalg.norm(pr_v, axis=1)

        valid = (gt_n > eps) & (pr_n > eps) & np.isfinite(gt_n) & np.isfinite(pr_n)
        if valid.sum() == 0:
            continue

        r = gt_n[valid] / pr_n[valid]
        ratios.append(r)

    if len(ratios) == 0:
        return 1.0

    ratios = np.concatenate(ratios, axis=0)
    if ratios.size == 0:
        return 1.0

    s = float(np.median(ratios))
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    return s


def compute_depth_abs_metrics_np(gt_z, pr_z, mask, thresholds=(0.5, 1.0, 2.0, 5.0)):
    """
    Compute absolute depth metrics on z-depth maps (meters).
    gt_z, pr_z: (H,W) numpy arrays
    mask: (H,W) bool, valid pixels
    thresholds: meters

    Returns: mae, rmse, inlier_ratios dict
    """
    if gt_z.ndim == 3 and gt_z.shape[-1] == 1:
        gt_z = gt_z[..., 0]
    if pr_z.ndim == 3 and pr_z.shape[-1] == 1:
        pr_z = pr_z[..., 0]

    m = mask.astype(bool)
    finite = np.isfinite(gt_z) & np.isfinite(pr_z)
    positive = gt_z > 0
    v = m & finite & positive

    if v.sum() == 0:
        mae = np.nan
        rmse = np.nan
        inliers = {float(t): np.nan for t in thresholds}
        return mae, rmse, inliers

    e = np.abs(pr_z - gt_z)[v]
    mae = float(np.mean(e))
    rmse = float(np.sqrt(np.mean(e * e)))

    inliers = {}
    for t in thresholds:
        t = float(t)
        inliers[t] = float(np.mean(e <= t))
    return mae, rmse, inliers


def get_all_info_for_metric_computation(batch, preds):
    """
    Adapted from dense_n_view_benchmark.get_all_info_for_metric_computation
    Goal for calibration benchmark:
      - Provide ray directions (for original metric)
      - Provide ABSOLUTE z-depth maps (meters) for GT and Pred,
        where Pred depth is aligned to GT scale using a global scale estimated
        from multi-view pointmaps in view0 frame.

    Returns:
      gt_info: dict with ray_directions, z_depths_abs (list of (B,H,W,1) on CPU)
      pr_info: dict with ray_directions, z_depths_abs_aligned (list of (B,H,W,1) on CPU)
      valid_masks: list of (B,H,W) on CPU
      scale_factors: dict with pr_to_gt_scale on CPU
    """
    n_views = len(batch)
    batch_size = batch[0]["camera_pose"].shape[0]

    # Everything in view0 frame
    in_camera0 = inv(batch[0]["camera_pose"])

    # Pred camera0
    pred_camera0 = torch.eye(4, device=preds[0]["cam_quats"].device).unsqueeze(0)
    pred_camera0 = pred_camera0.repeat(batch_size, 1, 1)
    pred_camera0[..., :3, :3] = quaternion_to_rotation_matrix(preds[0]["cam_quats"].clone())
    pred_camera0[..., :3, 3] = preds[0]["cam_trans"].clone()
    pred_in_camera0 = inv(pred_camera0)

    # lists
    no_norm_gt_pts_view0 = []
    no_norm_pr_pts_view0 = []
    no_norm_gt_pts3d_cam = []
    no_norm_pr_pts3d_cam = []
    no_norm_pr_pose_trans_view0 = []
    pr_pose_quats_view0 = []
    valid_masks = []
    gt_ray_directions = []
    pr_ray_directions = []

    for i in range(n_views):
        # GT pointmap in view0 frame
        no_norm_gt_pts_view0.append(geotrf(in_camera0, batch[i]["pts3d"]))
        no_norm_gt_pts3d_cam.append(batch[i]["pts3d_cam"])
        valid_masks.append(batch[i]["valid_mask"].clone())
        gt_ray_directions.append(batch[i]["ray_directions_cam"])

        # Pred pose transform to view0
        pr_pose_quats_in_view0, pr_pose_trans_in_view0 = transform_pose_using_quats_and_trans_2_to_1(
            preds[0]["cam_quats"],
            preds[0]["cam_trans"],
            preds[i]["cam_quats"],
            preds[i]["cam_trans"],
        )
        pr_pose_quats_view0.append(pr_pose_quats_in_view0)
        pr_ray_directions.append(preds[i]["ray_directions"])

        # Pred pointmap in view0 frame
        pr_pts3d_in_view0 = geotrf(pred_in_camera0, preds[i]["pts3d"])

        # Handle metric_scaling_factor exactly like dense script
        if "metric_scaling_factor" in preds[i].keys():
            curr_no_norm_pr_pts = pr_pts3d_in_view0 / preds[i]["metric_scaling_factor"].unsqueeze(-1).unsqueeze(-1)
            curr_no_norm_pr_pts3d_cam = preds[i]["pts3d_cam"] / preds[i]["metric_scaling_factor"].unsqueeze(-1).unsqueeze(-1)
            curr_no_norm_pr_pose_trans = pr_pose_trans_in_view0 / preds[i]["metric_scaling_factor"]
        else:
            curr_no_norm_pr_pts = pr_pts3d_in_view0
            curr_no_norm_pr_pts3d_cam = preds[i]["pts3d_cam"]
            curr_no_norm_pr_pose_trans = pr_pose_trans_in_view0

        no_norm_pr_pts_view0.append(curr_no_norm_pr_pts)
        no_norm_pr_pts3d_cam.append(curr_no_norm_pr_pts3d_cam)
        no_norm_pr_pose_trans_view0.append(curr_no_norm_pr_pose_trans)

    # Move valid masks to CPU for later numpy usage
    valid_masks_cpu = [m.cpu() for m in valid_masks]

    # Estimate global scale per batch element using all views (absolute, no normalization)
    pr_to_gt_scales = torch.ones(
        (batch_size,), device=no_norm_pr_pts_view0[0].device, dtype=no_norm_pr_pts_view0[0].dtype
    )
    for b in range(batch_size):
        gt_list_b = [no_norm_gt_pts_view0[v][b] for v in range(n_views)]
        pr_list_b = [no_norm_pr_pts_view0[v][b] for v in range(n_views)]
        m_list_b = [valid_masks[v][b] for v in range(n_views)]
        s_b = global_scale_from_pointmaps(gt_list_b, pr_list_b, m_list_b)
        pr_to_gt_scales[b] = pr_to_gt_scales[b] * float(s_b)

    # Build ABSOLUTE z-depth maps (GT) and aligned ABSOLUTE z-depth maps (Pred)
    gt_z_depths_abs = []
    pr_z_depths_abs_aligned = []

    for i in range(n_views):
        # GT absolute z-depth from pts3d_cam z
        gt_z = no_norm_gt_pts3d_cam[i][..., 2:].detach().cpu()  # (B,H,W,1)
        gt_z_depths_abs.append(gt_z)

        # Pred raw absolute z-depth from pts3d_cam z, then scale-align
        s_map = pr_to_gt_scales[:, None, None, None]  # (B,1,1,1)
        pr_z_aligned = (no_norm_pr_pts3d_cam[i][..., 2:] * s_map).detach().cpu()  # (B,H,W,1)
        pr_z_depths_abs_aligned.append(pr_z_aligned)

    gt_info = {
        "ray_directions": gt_ray_directions,
        "z_depths_abs": gt_z_depths_abs,
    }
    pr_info = {
        "ray_directions": pr_ray_directions,
        "z_depths_abs_aligned": pr_z_depths_abs_aligned,
    }
    scale_factors = {
        "pr_to_gt_scale": pr_to_gt_scales.detach().cpu(),
    }

    return gt_info, pr_info, valid_masks_cpu, scale_factors


def build_dataset(dataset, batch_size, num_workers):
    """
    Builds data loaders for testing.
    """
    print("Building data loader for dataset: ", dataset)
    loader = get_test_many_ar_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        drop_last=False,
    )
    print("Dataset length: ", len(loader))
    return loader


@torch.no_grad()
def benchmark(args):
    print("Output Directory: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    # Determine the mixed precision floating point type
    if args.amp:
        if args.amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif args.amp_dtype == "bf16":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                warnings.warn("bf16 is not supported on this device. Using fp16 instead.")
                amp_dtype = torch.float16
        elif args.amp_dtype == "fp32":
            amp_dtype = torch.float32
        else:
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    # Depth thresholds (meters)
    depth_thresholds = (0.5, 1.0, 2.0, 5.0)

    # Visualization: save ALL images (every sample, every view)
    vis_enabled = bool(getattr(args, "vis_enabled", True))
    vis_root = os.path.join(args.output_dir, "vis_depth_all")

    # Init Test Datasets and Dataloaders
    print("Building test dataset {:s}".format(args.dataset.test_dataset))
    data_loaders = {
        dataset.split("(")[0]: build_dataset(dataset, args.batch_size, args.dataset.num_workers)
        for dataset in args.dataset.test_dataset.split("+")
        if "(" in dataset
    }

    # Load Model
    model = init_model(args.model.model_str, args.model.model_config, torch_hub_force_reload=False)
    model.to(device)

    # Load pretrained model
    if args.model.pretrained:
        print("Loading pretrained: ", args.model.pretrained)
        ckpt = torch.load(args.model.pretrained, map_location=device, weights_only=False)
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt

    per_dataset_results = {}

    global_sample_counter = 0  # unique id across entire run

    for benchmark_dataset_name, data_loader in data_loaders.items():
        print("Benchmarking dataset: ", benchmark_dataset_name)
        data_loader.dataset.set_epoch(0)

        per_scene_results = {}
        for dataset_scene in data_loader.dataset.dataset.scenes:
            per_scene_results[dataset_scene] = {
                # original
                "ray_dirs_err_deg": [],
                # NEW: absolute depth metrics (meters) using ABS z-depth
                "z_depth_abs_mae": [],
                "z_depth_abs_rmse": [],
                "z_depth_inlier_ratio_0p5m": [],
                "z_depth_inlier_ratio_1m": [],
                "z_depth_inlier_ratio_2m": [],
                "z_depth_inlier_ratio_5m": [],
                # (optional) record scale
                "pr_to_gt_scale": [],
            }

        for batch in data_loader:
            n_views = len(batch)

            # Remove unnecessary indices
            for view in batch:
                view["idx"] = view["idx"][2:]

            # Transfer batch to device
            # NOTE: we need pts3d/pts3d_cam etc. so do NOT ignore them
            ignore_keys = set(
                [
                    "dataset",
                    "label",
                    "instance",
                    "idx",
                    "true_shape",
                    "rng",
                    "data_norm_type",
                ]
            )
            for view in batch:
                for name in list(view.keys()):
                    if name in ignore_keys:
                        continue
                    if torch.is_tensor(view[name]):
                        view[name] = view[name].to(device, non_blocking=True)

            # Inference
            with torch.autocast("cuda", enabled=bool(args.amp), dtype=amp_dtype):
                preds = model(batch)

            # Build absolute depth info (aligned pred)
            gt_info, pr_info, valid_masks, scale_factors = get_all_info_for_metric_computation(batch, preds)

            batch_size = batch[0]["img"].shape[0]
            for batch_idx in range(batch_size):
                scene = batch[0]["label"][batch_idx]

                # ---- ray metric across views (original) ----
                ray_dirs_err_deg_across_views = []

                # ---- depth metrics across views (absolute z-depth aligned) ----
                depth_mae_across_views = []
                depth_rmse_across_views = []
                inlier_0p5_across_views = []
                inlier_1_across_views = []
                inlier_2_across_views = []
                inlier_5_across_views = []

                # scale for this sample
                scale_val = float(scale_factors["pr_to_gt_scale"][batch_idx].item())

                for view_idx in range(n_views):
                    # ray
                    ray_dirs_l2 = torch.norm(
                        gt_info["ray_directions"][view_idx][batch_idx]
                        - pr_info["ray_directions"][view_idx][batch_idx],
                        dim=-1,
                    )
                    ray_deg = l2_distance_of_unit_ray_directions_to_angular_error(ray_dirs_l2)
                    ray_dirs_err_deg_across_views.append(float(torch.mean(ray_deg).detach().cpu().numpy()))

                    # depth (ABSOLUTE)
                    m = valid_masks[view_idx][batch_idx].numpy().astype(bool)

                    gt_z = gt_info["z_depths_abs"][view_idx][batch_idx].numpy()  # (H,W,1)
                    pr_z = pr_info["z_depths_abs_aligned"][view_idx][batch_idx].numpy()  # (H,W,1)

                    mae, rmse, inl = compute_depth_abs_metrics_np(
                        gt_z=gt_z,
                        pr_z=pr_z,
                        mask=m,
                        thresholds=depth_thresholds,
                    )

                    depth_mae_across_views.append(mae)
                    depth_rmse_across_views.append(rmse)
                    inlier_0p5_across_views.append(inl[0.5])
                    inlier_1_across_views.append(inl[1.0])
                    inlier_2_across_views.append(inl[2.0])
                    inlier_5_across_views.append(inl[5.0])

                    # -------------------------
                    # Save ALL visualizations
                    # -------------------------
                    if vis_enabled:
                        # squeeze gt/pr to (H,W)
                        if gt_z.ndim == 3 and gt_z.shape[-1] == 1:
                            gt_z_hw = gt_z[..., 0]
                        else:
                            gt_z_hw = gt_z
                        if pr_z.ndim == 3 and pr_z.shape[-1] == 1:
                            pr_z_hw = pr_z[..., 0]
                        else:
                            pr_z_hw = pr_z

                        err_hw = np.abs(pr_z_hw - gt_z_hw).astype(np.float32)

                        # RGB
                        rgb_u8 = (rgb(batch[view_idx]["img"][batch_idx], norm_type=batch[view_idx]["data_norm_type"][batch_idx]) * 255.0).astype(np.uint8)

                        H, W = gt_z_hw.shape[:2]

                        # ensure shapes (H,W)
                        gt_z_hw = gt_z_hw.astype(np.float32)
                        pr_z_hw = pr_z_hw.astype(np.float32)
                        mask_hw = m.astype(bool)

                        # output path: vis_depth_all/<dataset>/<scene>/sample_<global>_b<batch>_i<idx>_v<view>.png
                        vis_dir = os.path.join(vis_root, benchmark_dataset_name, str(scene))
                        os.makedirs(vis_dir, exist_ok=True)

                        save_path = os.path.join(
                            vis_dir,
                            f"sample_{global_sample_counter:08d}_b{batch_idx:03d}_v{view_idx:02d}_scale{scale_val:.4f}.png",
                        )

                        title_prefix = f"{benchmark_dataset_name} | {scene} | sample={global_sample_counter} | v{view_idx} | scale={scale_val:.4f}  "

                        save_triplet_vis(
                            save_path=save_path,
                            rgb=rgb_u8,
                            gt=gt_z_hw,
                            mvs=gt_z_hw,
                            pred=pr_z_hw,
                            err=err_hw,
                            mask=mask_hw,
                            title_prefix=title_prefix,
                            thr_list=depth_thresholds,
                        )

                # average across views
                ray_dirs_err_deg_curr_set = float(np.mean(ray_dirs_err_deg_across_views))

                z_mae = float(np.nanmean(depth_mae_across_views))
                z_rmse = float(np.nanmean(depth_rmse_across_views))
                inl_0p5 = float(np.nanmean(inlier_0p5_across_views))
                inl_1 = float(np.nanmean(inlier_1_across_views))
                inl_2 = float(np.nanmean(inlier_2_across_views))
                inl_5 = float(np.nanmean(inlier_5_across_views))

                per_scene_results[scene]["ray_dirs_err_deg"].append(ray_dirs_err_deg_curr_set)
                per_scene_results[scene]["z_depth_abs_mae"].append(z_mae)
                per_scene_results[scene]["z_depth_abs_rmse"].append(z_rmse)
                per_scene_results[scene]["z_depth_inlier_ratio_0p5m"].append(inl_0p5)
                per_scene_results[scene]["z_depth_inlier_ratio_1m"].append(inl_1)
                per_scene_results[scene]["z_depth_inlier_ratio_2m"].append(inl_2)
                per_scene_results[scene]["z_depth_inlier_ratio_5m"].append(inl_5)
                per_scene_results[scene]["pr_to_gt_scale"].append(scale_val)

                global_sample_counter += 1

        # Save per-scene
        with open(os.path.join(args.output_dir, f"{benchmark_dataset_name}_per_scene_results.json"), "w") as f:
            json.dump(per_scene_results, f, indent=4)

        # Aggregate across scenes
        across_dataset_results = {}
        for scene in per_scene_results.keys():
            for metric in per_scene_results[scene].keys():
                if metric not in across_dataset_results:
                    across_dataset_results[metric] = []
                across_dataset_results[metric].extend(per_scene_results[scene][metric])

        # Mean across all scenes (nan-safe)
        for metric in across_dataset_results.keys():
            across_dataset_results[metric] = float(np.nanmean(across_dataset_results[metric]))

        # Save dataset avg
        with open(os.path.join(args.output_dir, f"{benchmark_dataset_name}_avg_across_all_scenes.json"), "w") as f:
            json.dump(across_dataset_results, f, indent=4)

        # Print
        print("Average results across all scenes for dataset: ", benchmark_dataset_name)
        for metric in across_dataset_results.keys():
            print(f"{metric}: {across_dataset_results[metric]}")

        per_dataset_results[benchmark_dataset_name] = across_dataset_results

    # Average across datasets
    average_results = {}
    first_key = next(iter(per_dataset_results))
    for metric in per_dataset_results[first_key].keys():
        vals = [per_dataset_results[d][metric] for d in per_dataset_results]
        average_results[metric] = float(np.nanmean(vals))
    per_dataset_results["Average"] = average_results

    print("Benchmarking Done! ...")
    print("Average results across all datasets:")
    for metric in average_results.keys():
        print(f"{metric}: {average_results[metric]}")

    with open(os.path.join(args.output_dir, "per_dataset_results.json"), "w") as f:
        json.dump(per_dataset_results, f, indent=4)


@hydra.main(version_base=None, config_path="../../configs", config_name="calibration_benchmark")
def execute_benchmarking(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)

    benchmark(cfg)


if __name__ == "__main__":
    execute_benchmarking()  # noqa
