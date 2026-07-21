import os
import json
import logging
import sys
import cv2
import numpy as np
import torch
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from mapanything.models import init_model
from mapanything.utils.wai.core import load_data, load_frame

import PIL
import torchvision.transforms as tvf
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
from scipy.spatial.transform import Rotation

from mapanything.utils.geometry import (
    depthmap_to_camera_coordinates,
    get_absolute_pointmaps_and_rays_info,
)

from mapanything.utils.misc import StreamToLogger
log = logging.getLogger(__name__)


def build_mapanything_transform(transform: str = "imgnorm", data_norm_type: str = "dinov2"):
    if data_norm_type in IMAGE_NORMALIZATION_DICT:
        image_norm = IMAGE_NORMALIZATION_DICT[data_norm_type]
        norm_t = tvf.Normalize(mean=image_norm.mean, std=image_norm.std)
    elif data_norm_type == "identity":
        norm_t = None
    else:
        raise ValueError(
            f"Unknown data_norm_type: {data_norm_type}. "
            f"Available: identity or {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )

    def _ensure_rgb(img):
        return img.convert("RGB") if isinstance(img, PIL.Image.Image) and img.mode != "RGB" else img

    to_tensor = tvf.ToTensor()

    if transform == "imgnorm":
        t = [tvf.Lambda(_ensure_rgb), to_tensor]
        if norm_t is not None:
            t.append(norm_t)
        return tvf.Compose(t)
    else:
        raise ValueError(
            'Unknown transform. Available options: "imgnorm", "colorjitter", "colorjitter+grayscale+gaublur"'
        )

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
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1.0

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

    axes[0, 0].imshow(rgb); axes[0, 0].set_title(f"{title_prefix}RGB"); axes[0, 0].axis("off")

    ax_mvs = axes[0, 1]
    im_mvs = ax_mvs.imshow(mvs_show, vmin=vmin, vmax=vmax, cmap=cmap_depth)
    ax_mvs.set_title(f"{title_prefix}MVS depth"); ax_mvs.axis("off")
    fig.colorbar(im_mvs, ax=ax_mvs, **cbar_kw).set_label("Depth")

    ax_pred = axes[0, 2]
    im_pred = ax_pred.imshow(pred_show, vmin=vmin, vmax=vmax, cmap=cmap_depth)
    ax_pred.set_title(f"{title_prefix}Pred depth"); ax_pred.axis("off")
    fig.colorbar(im_pred, ax=ax_pred, **cbar_kw).set_label("Depth")

    ax_gt = axes[0, 3]
    im_gt = ax_gt.imshow(gt_show, vmin=vmin, vmax=vmax, cmap=cmap_depth)
    ax_gt.set_title(f"{title_prefix}GT depth"); ax_gt.axis("off")
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


def align_depth_least_square_np(gt_arr: np.ndarray, pred_arr: np.ndarray, valid_mask_arr: np.ndarray):
    m = valid_mask_arr.astype(bool)
    if m.sum() < 10:
        return pred_arr, 1.0, 0.0

    x = pred_arr[m].reshape(-1).astype(np.float64)
    y = gt_arr[m].reshape(-1).astype(np.float64)

    A = np.stack([x, np.ones_like(x)], axis=1)
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    pred_aligned = (a * pred_arr + b).astype(np.float32)
    return pred_aligned, a, b


def save_exr(path: str, depth: np.ndarray):
    """depth: (H,W) float32"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    depth = depth.astype(np.float32)
    # OpenCV EXR 写入需要启用 OpenEXR
    # export OPENCV_IO_ENABLE_OPENEXR=1
    ok = cv2.imwrite(path, depth)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for EXR: {path}. Try setting OPENCV_IO_ENABLE_OPENEXR=1")


def compute_depth_abs_metrics_np(gt_z, pr_z, mask, thresholds=(0.5, 1.0, 2.0, 5.0)):
    m = mask.astype(bool)
    finite = np.isfinite(gt_z) & np.isfinite(pr_z)
    positive = gt_z > 0
    v = m & finite & positive

    if v.sum() == 0:
        return np.nan, np.nan, {float(t): np.nan for t in thresholds}

    e = np.abs(pr_z - gt_z)[v]
    mae = float(np.mean(e))
    rmse = float(np.sqrt(np.mean(e * e)))
    inliers = {float(t): float(np.mean(e <= float(t))) for t in thresholds}
    return mae, rmse, inliers


def resize_depth_and_mask(depth, mask, target_wh):
    """depth:(H,W) float, mask:(H,W) bool -> resize to (Ht,Wt)"""
    Wt, Ht = target_wh
    d = cv2.resize(depth, (Wt, Ht), interpolation=cv2.INTER_NEAREST if depth.dtype == np.bool_ else cv2.INTER_LINEAR)
    m = cv2.resize(mask.astype(np.uint8), (Wt, Ht), interpolation=cv2.INTER_NEAREST) > 0
    return d, m


def extract_pred_depth_z(preds0: dict):
    """
    从模型输出提取 z-depth（单位：米）。
    你现在的 pipeline 里模型有 pts3d_cam: (B,H,W,3) 或 (H,W,3)
    """
    if "pts3d_cam" in preds0:
        pts = preds0["pts3d_cam"]
        # torch tensor
        if pts.ndim == 4:
            z = pts[..., 2]  # (B,H,W)
        else:
            z = pts[..., 2]  # (H,W)
        return z
    if "depth" in preds0:
        d = preds0["depth"]
        return d[..., 0] if d.ndim == 4 else d
    raise KeyError(f"Cannot find depth output key in preds[0]. keys={list(preds0.keys())}")

def build_view_like_basedataset(
    rgb_uint8: np.ndarray,        # (H,W,3) uint8
    depth_float: np.ndarray,      # (H,W) float32, metric depth z (meters)
    intr_3x3: np.ndarray,         # (3,3) float32
    c2w_4x4: np.ndarray,          # (4,4) float32
    dataset_name: str,
    label: str,
    instance: str,
    img_transform,
    data_norm_type: str,
    rng: np.random.Generator = None,
):
    """
    生成的 view 尽量对齐 BaseDataset.__getitem__ 末尾返回的字段与约束：
    - img: torch.float32 (3,H,W)
    - depthmap: (H,W,1) float32
    - valid_mask: (H,W) bool
    - pts3d, pts3d_cam, ray_directions_cam, depth_along_ray 等齐全
    - non_ambiguous_mask: (H,W) bool
    - true_shape: np.int32((H,W))
    - camera_pose_quats/camera_pose_trans
    """

    # ---- 基础字段（注意：BaseDataset 里是 PIL img 先 transform，再算几何）----
    pil_img = PIL.Image.fromarray(rgb_uint8)
    img_t = img_transform(pil_img)  # torch (3,H,W)

    depthmap = depth_float.astype(np.float32)
    depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    view = dict(
        img=img_t,  
        depthmap=depthmap,  
        camera_intrinsics=intr_3x3.astype(np.float32),
        camera_pose=c2w_4x4.astype(np.float32),
        dataset=dataset_name,
        label=label,
        instance=instance,
        data_norm_type=data_norm_type,
        is_metric_scale=np.array(False, dtype=np.bool_),
        is_synthetic=np.array(False, dtype=np.bool_),
    )

    H, W = depthmap.shape
    view["true_shape"] = np.int32((H, W))

    # ---- 计算 pointmaps / rays / depth_along_ray / pts3d_cam：和 BaseDataset 一致 ----
    (
        pts3d,
        valid_mask,
        ray_origins_world,
        ray_directions_world,
        depth_along_ray,
        ray_directions_cam,
        pts3d_cam,
    ) = get_absolute_pointmaps_and_rays_info(**view)

    view["pts3d"] = pts3d
    view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)
    view["depth_along_ray"] = depth_along_ray
    view["ray_directions_cam"] = ray_directions_cam
    view["pts3d_cam"] = pts3d_cam

    # ---- non_ambiguous_mask 对齐 BaseDataset ----
    ambiguous_mask = view["depthmap"] <= 0
    view["non_ambiguous_mask"] = (~ambiguous_mask).astype(view["valid_mask"].dtype)

    # ---- expand depthmap last dim：BaseDataset 在最后做 ----
    view["depthmap"] = view["depthmap"][..., None]  # (H,W,1)

    # ---- pose 的 quat/trans：BaseDataset 存的是 (x,y,z,w) ----
    view["camera_pose_quats"] = (
        Rotation.from_matrix(view["camera_pose"][:3, :3]).as_quat().astype(np.float32)
    )
    view["camera_pose_trans"] = view["camera_pose"][:3, 3].astype(np.float32)

    # ---- rng 字段（可选，但对齐一下）----
    if rng is not None:
        view["rng"] = int.from_bytes(rng.bytes(4), "big")
    else:
        view["rng"] = 0

    return view


def get_scene_list(dataset_root):
    """返回 dataset_root 下所有子目录名（场景名），排除非目录和隐藏文件夹"""
    scenes = []
    for item in os.listdir(dataset_root):
        full_path = os.path.join(dataset_root, item)
        if os.path.isdir(full_path) and not item.startswith('.'):
            # 可选：检查是否包含 scene_meta.json，若不包含则跳过
            if os.path.exists(os.path.join(full_path, 'scene_meta.json')):
                scenes.append(item)
            else:
                log.warning(f"Directory {item} does not contain scene_meta.json, skipping.")
    return sorted(scenes, reverse=False)


def should_skip_scene(scene_dir, scene_name):
    """检查场景是否已经完成：depth_complete 中的 exr 文件数是否等于 images 中的 png 文件数"""
    images_dir = os.path.join(scene_dir, 'images')
    depth_dir = os.path.join(scene_dir, 'depth_complete')
    if not os.path.exists(images_dir):
        return False  # 没有 images 目录，肯定不能跳过
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]
    if not image_files:
        return False
    if not os.path.exists(depth_dir):
        return False
    exr_files = [f for f in os.listdir(depth_dir) if f.lower().endswith('.exr')]
    # 如果数量相等，且都大于0，认为已完成
    return len(exr_files) == len(image_files) and len(exr_files) > 0


def process_single_scene(cfg, model, device, amp_dtype, scene_name):
    """处理单个场景，模型已加载好"""
    dataset_root = cfg["dataset_root"]
    scene_dir = os.path.join(dataset_root, scene_name)
    out_dir = scene_dir  # 结果直接保存在场景目录下

    # 跳过检查
    if should_skip_scene(scene_dir, scene_name):
        log.info(f"Scene {scene_name} already completed, skipping.")
        return

    log.info(f"Processing scene: {scene_name}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "depth_complete").mkdir(parents=True, exist_ok=True)

    transform_name = cfg.get("transform", "imgnorm")
    data_norm_type = cfg.get("data_norm_type", "dinov2")
    img_transform = build_mapanything_transform(transform_name, data_norm_type)

    infer_res = tuple(cfg["resolution"])  # [W,H]
    batch_size = int(cfg["batch_size"])
    align_mode = cfg.get("align_mode", "scale")
    thresholds = tuple(cfg.get("depth_thresholds", [0.5, 1.0, 2.0, 5.0]))
    save_exr_flag = bool(cfg.get("save_exr", True))
    save_png_flag = bool(cfg.get("save_png", True))

    if save_png_flag:
        (Path(out_dir) / "depth_complete_vis").mkdir(parents=True, exist_ok=True)

    # scene meta
    scene_meta = load_data(os.path.join(scene_dir, "scene_meta.json"), "scene_meta")
    frame_names = list(scene_meta["frame_names"].keys())
    frame_names = sorted(frame_names)

    per_frame = {}
    all_mae, all_rmse = [], []
    all_inliers = {float(t): [] for t in thresholds}

    # mini-batch iterator
    def iter_batches(lst, bs):
        for i in range(0, len(lst), bs):
            yield i, lst[i:i+bs]

    for start_idx, batch_frames in tqdm(iter_batches(frame_names, batch_size), desc=f"Scene {scene_name}"):
        views = []
        orig_sizes = []
        rgbs_orig = []
        gt_depth_orig = []
        mask_orig = []

        for fn in batch_frames:
            vd = load_frame(
                scene_dir,
                fn,
                modalities=cfg.get("modalities", ["image", "depth", "mask"]),
                scene_meta=scene_meta,
            )
            rgb = vd["image"].permute(1, 2, 0).cpu().numpy()
            rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)

            depth = vd["depth"].cpu().numpy().astype(np.float32)  # (H,W)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            if "mask" in vd:
                m = vd["mask"].cpu().numpy().astype(bool)
            else:
                m = depth > 0

            if np.any(m):
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    m.astype(np.uint8), connectivity=8
                )
                if num_labels > 1:
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    keep_labels = [i+1 for i, area in enumerate(areas) if area >= cfg.get("min_region_area", 500)]
                    if keep_labels:
                        new_mask_binary = np.isin(labels, keep_labels)
                        depth[~new_mask_binary] = 0.0
                        m = new_mask_binary

            intr = vd["intrinsics"].cpu().numpy().astype(np.float32)
            c2w = vd["extrinsics"].cpu().numpy().astype(np.float32)

            H0, W0 = depth.shape
            orig_sizes.append((W0, H0))
            rgbs_orig.append(rgb)
            gt_depth_orig.append(depth)
            mask_orig.append(m)

            Wt, Ht = infer_res
            if (W0, H0) != (Wt, Ht):
                rgb_in = cv2.resize(rgb, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
                depth_in = cv2.resize(depth, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
                m_in = cv2.resize(m.astype(np.uint8), (Wt, Ht), interpolation=cv2.INTER_NEAREST) > 0

                sx = Wt / float(W0)
                sy = Ht / float(H0)
                intr_in = intr.copy()
                intr_in[0, 0] *= sx
                intr_in[1, 1] *= sy
                intr_in[0, 2] *= sx
                intr_in[1, 2] *= sy
            else:
                rgb_in, depth_in, m_in, intr_in = rgb, depth, m, intr

            if not hasattr(process_single_scene, "_rng"):
                process_single_scene._rng = np.random.default_rng(seed=cfg.seed)
            rng = process_single_scene._rng

            view = build_view_like_basedataset(
                rgb_uint8=rgb_in,
                depth_float=depth_in,
                intr_3x3=intr_in,
                c2w_4x4=c2w,
                dataset_name="A3DScenes",
                label=scene_name,
                instance=str(fn),
                img_transform=img_transform,
                data_norm_type=data_norm_type,
                rng=rng,
            )
            views.append(view)

        def to_torch(x):
            if torch.is_tensor(x):
                return x
            if isinstance(x, np.ndarray):
                if x.dtype == np.bool_:
                    return torch.from_numpy(x.astype(np.bool_))
                return torch.from_numpy(x)
            return x

        tensor_keys = [
            "img","depthmap","valid_mask","pts3d","pts3d_cam","ray_directions_cam",
            "depth_along_ray","non_ambiguous_mask","camera_intrinsics","camera_pose",
            "true_shape","camera_pose_quats","camera_pose_trans",
            "is_metric_scale","is_synthetic",
        ]

        for v in views:
            for kk in tensor_keys:
                v[kk] = to_torch(v[kk])
                if torch.is_tensor(v[kk]):
                    v[kk] = v[kk].unsqueeze(0).to(device, non_blocking=True)

        merged = {}
        for k in views[0].keys():
            if torch.is_tensor(views[0][k]):
                merged[k] = torch.cat([vv[k] for vv in views], dim=0)  # 已经在 device 上
            else:
                merged[k] = [vv[k] for vv in views]
        model_input = [merged]

        if cfg.amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                preds = model(model_input)
        else:
            preds = model(model_input)

        pred_z = preds[0]["pts3d_cam"][..., 2].detach().float().cpu().numpy()  # (B,H,W)
        gt_z = merged["pts3d_cam"][..., 2].detach().cpu().numpy().astype(np.float32)
        vm = merged["valid_mask"].detach().cpu().numpy().astype(bool)

        for i, fn in enumerate(batch_frames):
            W0, H0 = orig_sizes[i]
            gt_i_infer = gt_z[i]
            pr_i_infer = pred_z[i]
            m_i_infer = vm[i] & (gt_i_infer > 0) & np.isfinite(gt_i_infer) & np.isfinite(pr_i_infer)

            if align_mode == "affine":
                pr_aligned_infer, a, b = align_depth_least_square_np(gt_i_infer, pr_i_infer, m_i_infer)
                align_info = {"mode": "affine", "a": float(a), "b": float(b)}
            else:
                if m_i_infer.sum() < 10:
                    s = 1.0
                else:
                    ratio = gt_i_infer[m_i_infer] / np.clip(pr_i_infer[m_i_infer], 1e-8, None)
                    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
                    s = float(np.median(ratio)) if ratio.size > 0 else 1.0
                pr_aligned_infer = (pr_i_infer * s).astype(np.float32)
                align_info = {"mode": "scale", "s": float(s)}

            mae, rmse, inl = compute_depth_abs_metrics_np(gt_i_infer, pr_aligned_infer, m_i_infer, thresholds=thresholds)

            per_frame[str(fn)] = dict(
                mae=mae, rmse=rmse,
                inliers={str(k): float(v) for k, v in inl.items()},
                align=align_info,
            )
            all_mae.append(mae)
            all_rmse.append(rmse)
            for t in thresholds:
                all_inliers[float(t)].append(inl[float(t)])

            if save_exr_flag:
                pr_aligned_orig = cv2.resize(pr_aligned_infer, (W0, H0), interpolation=cv2.INTER_LINEAR)
                exr_path = os.path.join(out_dir, "depth_complete", f"{Path(str(fn)).stem}.exr")
                save_exr(exr_path, pr_aligned_orig)

            if save_png_flag:
                gt0 = gt_depth_orig[i].astype(np.float32)
                pr0 = cv2.resize(pr_aligned_infer, (W0, H0), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                m0 = mask_orig[i].astype(bool) & (gt0 > 0) & np.isfinite(gt0) & np.isfinite(pr0)
                err0 = np.abs(pr0 - gt0).astype(np.float32)
                png_path = os.path.join(out_dir, "depth_complete_vis", f"{Path(str(fn)).stem}.png")
                save_triplet_vis(
                    save_path=png_path,
                    rgb=rgbs_orig[i],
                    gt=gt0,
                    mvs=gt0,
                    pred=pr0,
                    err=err0,
                    mask=m0,
                    title_prefix=f"{scene_name}/{fn}  ",
                    thr_list=thresholds,
                )

        # 清理当前 batch 的大变量，帮助回收显存
        del views, merged, model_input, preds, pred_z, gt_z, vm
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "scene": scene_name,
        "num_frames": len(frame_names),
        "resolution_infer": list(infer_res),
        "align_mode": align_mode,
        "mae_mean": float(np.nanmean(all_mae)) if len(all_mae) else np.nan,
        "rmse_mean": float(np.nanmean(all_rmse)) if len(all_rmse) else np.nan,
        "inlier_mean": {str(t): float(np.nanmean(all_inliers[t])) for t in all_inliers},
    }

    with open(os.path.join(out_dir, "per_frame_metrics.json"), "w") as f:
        json.dump(per_frame, f, indent=2)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Scene {scene_name} finished.")

    # 场景结束后清理显存
    if device.type == "cuda":
        torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="../configs", config_name="depth_completion")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    # 重定向日志（保持原功能）
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)

    # 设置随机种子
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 获取设备
    gpu_id = cfg.get("gpu_id", 0)
    if torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
            log.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            log.warning(f"Specified GPU ID {gpu_id} is not available (only {torch.cuda.device_count()} GPUs). Using default GPU 0.")
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        log.info("CUDA not available, using CPU.")

    cudnn.benchmark = not cfg.disable_cudnn_benchmark

    # 混合精度类型
    if cfg.amp:
        if cfg.amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif cfg.amp_dtype == "bf16":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                warnings.warn("bf16 is not supported on this device. Using fp16 instead.")
                amp_dtype = torch.float16
        elif cfg.amp_dtype == "fp32":
            amp_dtype = torch.float32
        else:
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    # 初始化模型（只加载一次）
    log.info("Loading model...")
    model = init_model(
        cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False
    )
    model.to(device)
    if cfg.model.pretrained:
        log.info(f"Loading pretrained weights from {cfg.model.pretrained}")
        ckpt = torch.load(cfg.model.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        del ckpt
    model.eval()
    log.info("Model loaded.")

    # 获取所有场景
    dataset_root = cfg["dataset_root"]
    if not os.path.isdir(dataset_root):
        log.error(f"dataset_root {dataset_root} is not a directory.")
        return

    scene_list = get_scene_list(dataset_root)
    if not scene_list:
        log.warning("No scenes found.")
        return

    log.info(f"Found {len(scene_list)} scenes: {scene_list}")

    # 依次处理每个场景
    for scene_name in scene_list:
        process_single_scene(cfg, model, device, amp_dtype, scene_name)

    log.info("All scenes processed.")


if __name__ == "__main__":
    main()