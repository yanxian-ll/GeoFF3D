#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run predict_scene_to_rrd after random world-frame Sim(3) input augmentation.

This script is a thin evaluation wrapper around ``scripts/predict_scene_to_rrd.py``.
It applies the same style of train-time world-frame augmentation used by
``WorldFrameAugmentDataset`` before model inference:

    recenter -> random translation -> random scale -> random xyz rotation

Images, intrinsics, masks, and ray directions stay unchanged. Camera-frame
metric values such as depth and ``pts3d_cam`` follow the sampled scale. World
points and camera poses are transformed into the augmented frame seen by the
model. The RRD is saved in that augmented coordinate system: transformed GT and
raw model predictions are logged directly in the same frame.

Example:
    python scripts/predict_scene_to_rrd_world_aug.py \
      --scene_dir /opt/data/private/dataset/data/usegeo/dataset1 \
      --model geoff3d \
      --checkpoint experiments/dom/uav_training/geoff3d_no_metric_scale_8v_6d_16ipg_2g_gravity/checkpoint-last.pth \
      --output_rrd outputs/scene_world_aug.rrd \
      --num_views 30 \
      --max_side 518 \
      --align scale \
      --aug_rotation_deg 0 0 180 \
      --aug_translation_range 20 \
      --aug_scale_range 0.5 1.5 \
      --aug_recenter_mode mean_camera
      
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import predict_scene_to_rrd as base


RECENTER_CHOICES = ("first_camera", "mean_camera", "scene_points")
CAMERA_SCALE_KEYS = (
    "depthmap",
    "depth_along_ray",
    "prior_depth_along_ray",
    "pts3d_cam",
)


def _parse_float_list(values: Optional[Sequence[float]], default: Sequence[float]) -> List[float]:
    if values is None:
        return [float(v) for v in default]
    return [float(v) for v in values]


def _parse_rotation_range(
    rotation_deg: Optional[Sequence[float]],
    x_rotation_deg: float,
    y_rotation_deg: float,
    z_rotation_deg: float,
) -> np.ndarray:
    values = _parse_float_list(rotation_deg, [x_rotation_deg, y_rotation_deg, z_rotation_deg])
    if len(values) == 1:
        arr = np.asarray([values[0], values[0], values[0]], dtype=np.float64)
    elif len(values) == 3:
        arr = np.asarray(values, dtype=np.float64)
    else:
        raise ValueError("--aug_rotation_deg must be one scalar or three values: x y z")
    if np.any(arr < 0):
        raise ValueError("--aug_rotation_deg values must be non-negative")
    return arr


def _parse_translation_range(values: Sequence[float]) -> np.ndarray:
    vals = [float(v) for v in values]
    if len(vals) == 1:
        arr = np.asarray([vals[0], vals[0], vals[0]], dtype=np.float64)
    elif len(vals) == 2:
        arr = np.asarray([vals[0], vals[0], vals[1]], dtype=np.float64)
    elif len(vals) == 3:
        arr = np.asarray(vals, dtype=np.float64)
    else:
        raise ValueError("--aug_translation_range must be scalar, xy z, or x y z")
    if np.any(arr < 0):
        raise ValueError("--aug_translation_range values must be non-negative")
    return arr


def _parse_scale_range(values: Sequence[float]) -> np.ndarray:
    vals = [float(v) for v in values]
    if len(vals) == 1:
        arr = np.asarray([vals[0], vals[0]], dtype=np.float64)
    elif len(vals) == 2:
        arr = np.asarray(vals, dtype=np.float64)
    else:
        raise ValueError("--aug_scale_range must be scalar or min max")
    if np.any(arr <= 0):
        raise ValueError("--aug_scale_range values must be positive")
    if arr[0] > arr[1]:
        raise ValueError("--aug_scale_range min must be <= max")
    return arr


def parse_aug_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--disable_world_aug",
        action="store_true",
        help="Disable this wrapper's random world-frame augmentation.",
    )
    parser.add_argument(
        "--aug_rotation_deg",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Random Euler rotation range in degrees. Provide one scalar or x y z. "
            "Default follows training config split keys: 180 180 180."
        ),
    )
    parser.add_argument("--aug_x_rotation_deg", type=float, default=180.0)
    parser.add_argument("--aug_y_rotation_deg", type=float, default=180.0)
    parser.add_argument("--aug_z_rotation_deg", type=float, default=180.0)
    parser.add_argument(
        "--aug_translation_range",
        type=float,
        nargs="+",
        default=[20.0],
        help="Uniform translation range. Provide scalar, xy z, or x y z. Default: 20.",
    )
    parser.add_argument(
        "--aug_scale_range",
        type=float,
        nargs="+",
        default=[0.5, 1.5],
        help="Uniform scale range. Provide fixed scalar or min max. Default: 0.5 1.5.",
    )
    parser.add_argument(
        "--aug_recenter",
        action="store_true",
        default=True,
        help="Recenter before random translation/scale/rotation. Enabled by default.",
    )
    parser.add_argument(
        "--no_aug_recenter",
        action="store_false",
        dest="aug_recenter",
        help="Do not recenter before random translation/scale/rotation.",
    )
    parser.add_argument(
        "--aug_recenter_mode",
        choices=RECENTER_CHOICES,
        default="mean_camera",
        help="Anchor used when recentering. Default: mean_camera.",
    )
    parser.add_argument(
        "--aug_seed",
        type=int,
        default=None,
        help="Random seed for the sampled world-frame augmentation. Default: base --seed.",
    )
    aug_args, remaining = parser.parse_known_args(list(argv))

    aug_args.rotation_range = _parse_rotation_range(
        aug_args.aug_rotation_deg,
        aug_args.aug_x_rotation_deg,
        aug_args.aug_y_rotation_deg,
        aug_args.aug_z_rotation_deg,
    )
    aug_args.translation_range = _parse_translation_range(aug_args.aug_translation_range)
    aug_args.scale_range = _parse_scale_range(aug_args.aug_scale_range)
    return aug_args, remaining


def _has_camera_pose_priors(views: Sequence[Dict[str, object]]) -> bool:
    return bool(views) and all("camera_pose" in view for view in views)


def _torch_anchor_from_views(
    views: Sequence[Dict[str, object]],
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    if mode == "first_camera":
        return views[0]["camera_pose"][0, :3, 3].to(device=device, dtype=torch.float32)

    if mode == "mean_camera":
        centers = [view["camera_pose"][0, :3, 3].to(device=device, dtype=torch.float32) for view in views]
        return torch.stack(centers, dim=0).mean(dim=0)

    if mode == "scene_points":
        pts_all = []
        for view in views:
            pts = view.get("pts3d", None)
            valid = view.get("valid_mask", None)
            if pts is None or valid is None:
                continue
            pts = pts.to(device=device, dtype=torch.float32)
            valid = valid.to(device=device).bool()
            finite = torch.isfinite(pts).all(dim=-1)
            keep = valid & finite
            if keep.any():
                pts_all.append(pts[keep].reshape(-1, 3))
        if pts_all:
            return torch.cat(pts_all, dim=0).mean(dim=0)
        return views[0]["camera_pose"][0, :3, 3].to(device=device, dtype=torch.float32)

    raise ValueError(f"Unsupported recenter mode: {mode!r}")


def _sample_world_aug(
    aug_args: argparse.Namespace,
    views: Sequence[Dict[str, object]],
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    rotation_range = np.asarray(aug_args.rotation_range, dtype=np.float64)
    translation_range = np.asarray(aug_args.translation_range, dtype=np.float64)
    scale_range = np.asarray(aug_args.scale_range, dtype=np.float64)

    if np.any(rotation_range > 0):
        angles_deg = rng.uniform(-rotation_range, rotation_range)
        rot_np = Rotation.from_euler("xyz", angles_deg, degrees=True).as_matrix().astype(np.float32)
    else:
        angles_deg = np.zeros(3, dtype=np.float64)
        rot_np = np.eye(3, dtype=np.float32)

    if np.any(translation_range > 0):
        local_trans_np = rng.uniform(-translation_range, translation_range).astype(np.float32)
    else:
        local_trans_np = np.zeros(3, dtype=np.float32)

    if np.allclose(scale_range[0], scale_range[1]):
        scale = float(scale_range[0])
    else:
        scale = float(rng.uniform(scale_range[0], scale_range[1]))

    if bool(aug_args.aug_recenter):
        anchor = _torch_anchor_from_views(views, aug_args.aug_recenter_mode, device=device)
    else:
        anchor = torch.zeros(3, dtype=torch.float32, device=device)

    return {
        "seed": int(seed),
        "rotation_deg_range": rotation_range.tolist(),
        "sampled_euler_xyz_deg": angles_deg.tolist(),
        "rotation_matrix": rot_np.tolist(),
        "translation_range": translation_range.tolist(),
        "sampled_local_translation": local_trans_np.tolist(),
        "scale_range": scale_range.tolist(),
        "sampled_scale": scale,
        "recenter": bool(aug_args.aug_recenter),
        "recenter_mode": str(aug_args.aug_recenter_mode),
        "anchor": anchor.detach().cpu().numpy().astype(np.float32).tolist(),
        "_rot_torch": torch.from_numpy(rot_np).to(device=device, dtype=torch.float32),
        "_local_trans_torch": torch.from_numpy(local_trans_np).to(device=device, dtype=torch.float32),
        "_anchor_torch": anchor.to(device=device, dtype=torch.float32),
        "_scale_torch": torch.tensor(scale, device=device, dtype=torch.float32),
    }


def _transform_world_points_torch(
    points: torch.Tensor,
    rot: torch.Tensor,
    local_trans: torch.Tensor,
    anchor: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return ((points - anchor.to(points.device, points.dtype) + local_trans.to(points.device, points.dtype)) * scale.to(points.device, points.dtype)) @ rot.to(points.device, points.dtype).T


def _transform_pose_torch(
    pose: torch.Tensor,
    rot: torch.Tensor,
    local_trans: torch.Tensor,
    anchor: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    pose_out = pose.clone()
    rot = rot.to(pose.device, pose.dtype)
    pose_out[..., :3, :3] = rot.view(1, 3, 3) @ pose[..., :3, :3]
    pose_out[..., :3, 3] = _transform_world_points_torch(
        pose[..., :3, 3],
        rot=rot,
        local_trans=local_trans,
        anchor=anchor,
        scale=scale,
    )
    return pose_out


def _quat_from_pose_torch(pose: torch.Tensor) -> torch.Tensor:
    quats = []
    for R_t in pose[..., :3, :3]:
        quat = base.numpy_quat_xyzw_from_rotmat(R_t.detach().cpu().numpy())
        quats.append(torch.from_numpy(quat).to(device=pose.device, dtype=pose.dtype))
    return torch.stack(quats, dim=0)


def _apply_world_aug_to_views_inplace(
    views: Sequence[Dict[str, object]],
    aug_meta: Dict[str, object],
) -> None:
    rot = aug_meta["_rot_torch"]
    local_trans = aug_meta["_local_trans_torch"]
    anchor = aug_meta["_anchor_torch"]
    scale = aug_meta["_scale_torch"]

    for view in views:
        pose = _transform_pose_torch(view["camera_pose"], rot, local_trans, anchor, scale)
        view["camera_pose"] = pose
        view["camera_pose_trans"] = pose[..., :3, 3]
        view["camera_pose_quats"] = _quat_from_pose_torch(pose)
        if "world_translation" in view:
            view["world_translation"] = pose[..., :3, 3]

        if "pts3d" in view:
            view["pts3d"] = _transform_world_points_torch(
                view["pts3d"],
                rot=rot,
                local_trans=local_trans,
                anchor=anchor,
                scale=scale,
            )

        for key in CAMERA_SCALE_KEYS:
            if key in view:
                value = view[key]
                view[key] = value * scale.to(value.device, value.dtype)


def _transform_world_points_np(points: np.ndarray, aug_meta: Dict[str, object]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points
    shape = points.shape
    rot = np.asarray(aug_meta["rotation_matrix"], dtype=np.float64)
    anchor = np.asarray(aug_meta["anchor"], dtype=np.float64).reshape(3)
    local_trans = np.asarray(aug_meta["sampled_local_translation"], dtype=np.float64).reshape(3)
    scale = float(aug_meta["sampled_scale"])
    flat = points.reshape(-1, 3).astype(np.float64)
    transformed = ((flat - anchor.reshape(1, 3) + local_trans.reshape(1, 3)) * scale) @ rot.T
    return transformed.astype(np.float32).reshape(shape)


def _transform_point_maps_np(point_maps: Sequence[np.ndarray], aug_meta: Dict[str, object]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for point_map in point_maps:
        if point_map.size == 0:
            out.append(point_map)
        else:
            out.append(_transform_world_points_np(point_map, aug_meta))
    return out


def _transform_cameras_np(
    cams: Sequence[Dict[str, object]],
    aug_meta: Dict[str, object],
) -> List[Dict[str, object]]:
    rot = np.asarray(aug_meta["rotation_matrix"], dtype=np.float64)
    out: List[Dict[str, object]] = []
    for cam in cams:
        T_in = np.asarray(cam["T_c2w"], dtype=np.float64)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = (rot @ T_in[:3, :3]).astype(np.float32)
        T[:3, 3] = _transform_world_points_np(T_in[:3, 3].reshape(1, 3), aug_meta).reshape(3)
        out.append({**cam, "T_c2w": T})
    return out


def _apply_world_aug_to_meta_inplace(meta: Dict[str, object], aug_meta: Dict[str, object]) -> None:
    meta["gt_maps"] = _transform_point_maps_np(meta["gt_maps"], aug_meta)
    meta["gt_points"] = _transform_world_points_np(meta["gt_points"], aug_meta)

    cams = meta.get("cams", {})
    transformed_cams = _transform_cameras_np(
        [{"stem": stem, "T_c2w": np.asarray(cam["T_c2w"], dtype=np.float32)} for stem, cam in cams.items()],
        aug_meta,
    )
    by_stem = {str(cam["stem"]): cam["T_c2w"] for cam in transformed_cams}
    for stem, cam in cams.items():
        if stem in by_stem:
            T_c2w = np.asarray(by_stem[stem], dtype=np.float32)
            cam["T_c2w"] = T_c2w
            cam["T_w2c"] = np.linalg.inv(T_c2w.astype(np.float64)).astype(np.float32)


def _strip_private_aug_meta(aug_meta: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in aug_meta.items() if not k.startswith("_")}


def _append_world_aug_sidecar(output_rrd: str, aug_meta: Dict[str, object]) -> None:
    sidecar = Path(output_rrd).expanduser().resolve().with_suffix(".json")
    if not sidecar.exists():
        return
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] failed to read sidecar metadata for world-aug update: {exc}")
        return
    payload["world_frame_augmentation"] = _strip_private_aug_meta(aug_meta)
    sidecar.write_text(json.dumps(base.json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated sidecar world-frame augmentation metadata: {sidecar}")


@torch.no_grad()
def main() -> None:
    aug_args, remaining_argv = parse_aug_args(sys.argv[1:])

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining_argv]
        args = base.parse_args()
    finally:
        sys.argv = old_argv

    device = base.resolve_device(args.device)
    views, meta = base.build_views_from_scene(args, device=device)

    can_augment = _has_camera_pose_priors(views)
    applied = bool(not aug_args.disable_world_aug and can_augment)
    if applied:
        aug_seed = int(args.seed if aug_args.aug_seed is None else aug_args.aug_seed)
        aug_meta = _sample_world_aug(aug_args, views, seed=aug_seed, device=device)
        _apply_world_aug_to_views_inplace(views, aug_meta)
        _apply_world_aug_to_meta_inplace(meta, aug_meta)
        printable = _strip_private_aug_meta(aug_meta)
        print(
            "Applying world-frame input augmentation: "
            f"seed={printable['seed']}, "
            f"euler_xyz_deg={np.array(printable['sampled_euler_xyz_deg'])}, "
            f"local_translation={np.array(printable['sampled_local_translation'])}, "
            f"scale={printable['sampled_scale']:.6g}, "
            f"anchor={np.array(printable['anchor'])}"
        )
    else:
        aug_meta = {
            "seed": int(args.seed if aug_args.aug_seed is None else aug_args.aug_seed),
            "applied": False,
            "reason": (
                "disabled by --disable_world_aug"
                if aug_args.disable_world_aug
                else "not all selected views have camera_pose priors"
            ),
            "coordinate_policy": "no world-frame augmentation was applied",
        }
        if not aug_args.disable_world_aug and not can_augment:
            print("[WARN] world-frame augmentation requested but not all views have camera_pose; using raw coordinates")

    aug_meta["applied"] = applied
    if applied:
        aug_meta["coordinate_policy"] = (
            "GT priors are transformed to the augmented coordinate system; "
            "model predictions are logged directly in that same frame"
        )

    gt_cams = [
        {"stem": stem, "T_c2w": np.asarray(meta["cams"][stem]["T_c2w"], dtype=np.float32)}
        for stem in meta["stems"]
        if stem in meta["cams"]
    ]

    model, _ = base.init_model_from_hydra(
        model_name=args.model,
        machine=args.machine,
        hydra_overrides=args.hydra_override,
        device=device,
    )
    base.load_optional_checkpoint(model, args.checkpoint)
    model.eval()

    run_label = "world-augmented" if applied else "raw"
    print(f"Running model={args.model} on {len(views)} {run_label} views ...")
    preds = model(views)

    raw_pred_points, raw_pred_colors, pred_maps, pred_valid_masks, raw_pred_cams = base.collect_pred_outputs(
        preds=preds,
        rgbs=meta["rgbs"],
        args=args,
        stems=meta["stems"],
    )

    print(f"Raw prediction summary: points={raw_pred_points.shape[0]}, cameras={len(raw_pred_cams)}")

    pred_points, pred_colors, pred_maps_aligned, pred_cams, align_meta = base.estimate_and_apply_alignment(
        args=args,
        meta=meta,
        pred_points=raw_pred_points,
        pred_colors=raw_pred_colors,
        pred_maps=pred_maps,
        pred_valid_masks=pred_valid_masks,
        pred_cams=raw_pred_cams,
    )
    print(f"Logged prediction summary: points={pred_points.shape[0]}, cameras={len(pred_cams)}, align={align_meta['mode']}")

    base.save_rrd(
        args=args,
        meta=meta,
        gt_cams=gt_cams,
        pred_points=pred_points,
        pred_colors=pred_colors,
        pred_cams=pred_cams,
        raw_pred_points=raw_pred_points,
        raw_pred_colors=raw_pred_colors,
        raw_pred_cams=raw_pred_cams,
        align_meta=align_meta,
    )
    _append_world_aug_sidecar(args.output_rrd, aug_meta)


if __name__ == "__main__":
    main()
