#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run chunked scene prediction and save all chunks into one Rerun .rrd.

This is a lightweight SLAM smoke-test script.  It keeps the same scene/model
inputs as ``predict_scene_to_rrd.py`` and changes inference into chunked
sliding windows.  By default, each chunk is independently scale+translation
aligned to GT for visualization; no rotation is estimated or applied.  Optional
pose-translation alignment modes can instead align every chunk's predicted
camera translations to the input pose translations with scale only, or with
scale followed by XY-plane yaw around Z plus translation.

Example:

      --scene_dir /opt/data/private/dataset/data/usegeo/dataset1 \
      --scene_dir /opt/data/private/dataset/data/NPU_Dronemap/gopro-npu-kfs \


    python scripts/predict_scene_to_rrd_slam_chunks.py \
      --model geoff3d \
      --scene_dir /opt/data/private/dataset/data/usegeo/dataset1 \
      --checkpoint experiments/dom/uav_training/pi3x_zup_translation_yaw_aug_8v_6d_16ipg_2g/checkpoint-last.pth \
      --output_rrd outputs/pi3x_zup_translation_yaw_aug_8v_6d_16ipg_2g/usegeo-dataset1-translation-rotation.rrd \
      --num_views 0 \
      --stride 1 \
      --max_side 518 \
      --chunk_size 30 \
      --chunk_overlap 2 \
      --align pose_scale_yaw_translation \
      --slam_overlap_point_align none \
      --use_world_translation_prior \
      --use_world_rotation_prior  \
      --no_ray_prior \
      --no_depth_prior

    
    python scripts/predict_scene_to_rrd_slam_chunks.py \
      --model pi3x \
      --scene_dir /opt/data/private/dataset/data/NPU_Dronemap/gopro-npu-kfs \
      --checkpoint experiments/mapanything/uav_training/pi3x_finetuning_16v_6d_16ipg_2g_mvs/checkpoint-best.pth \
      --output_rrd outputs/gopro_scene_slam_chunks_sim3_pi3x.rrd \
      --num_views 0 \
      --stride 1 \
      --max_side 518 \
      --chunk_size 30 \
      --chunk_overlap 2 \
      --align pose_sim3 \
      --slam_overlap_point_align none \
      --use_world_translation_prior \
      --use_world_rotation_prior  \
      --no_ray_prior \
      --no_depth_prior


Camera entities are grouped under ``world/cameras`` in the saved recording, so
Rerun can hide/show all GT cameras or all predicted cameras from one parent
entity instead of toggling every chunk.

Open:
    rerun outputs/scene_slam_chunks.rrd
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import predict_scene_to_rrd as base
from mapanything.utils.geometry import recover_pinhole_intrinsics_from_ray_directions


ALIGN_MODES = {
    "none",
    "scale",
    "pose_scale",
    "pose_scale_yaw_translation",
    "pose_sim3",
}
ALIGN_ALIASES = {
    "pose-scale": "pose_scale",
    "pose_scale_only": "pose_scale",
    "pose-scale-only": "pose_scale",
    "pose_scale_yaw_trans": "pose_scale_yaw_translation",
    "pose-scale-yaw-trans": "pose_scale_yaw_translation",
    "pose_scale_xy": "pose_scale_yaw_translation",
    "pose-scale-xy": "pose_scale_yaw_translation",
    "pose_scale_xy_rot_trans": "pose_scale_yaw_translation",
    "pose-scale-xy-rot-trans": "pose_scale_yaw_translation",
    "pose_full": "pose_sim3",
    "pose-full": "pose_sim3",
    "pose_scale_rotation_translation": "pose_sim3",
    "pose-scale-rotation-translation": "pose_sim3",
    "pose_srt": "pose_sim3",
    "pose-srt": "pose_sim3",
}
POSE_TRANSLATION_ALIGN_MODES = {"pose_scale", "pose_scale_yaw_translation", "pose_sim3"}
CHUNK_MODES = {"sequence", "pose_space"}
CHUNK_MODE_ALIASES = {
    "order": "sequence",
    "ordered": "sequence",
    "sequential": "sequence",
    "pose": "pose_space",
    "pose-space": "pose_space",
    "spatial": "pose_space",
    "space": "pose_space",
}
WORLD_TRANSLATION_PRIOR_MODELS = {"geoff3d"}
WORLD_ROTATION_PRIOR_MODELS = {"geoff3d"}
FOV_PRIOR_MODELS = set()
PI3X_PRIOR_MODELS = {"geoff3d"}
VGGT_OMEGA_MODELS = {"vggt_omega"}


def normalize_align_mode(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    key = str(mode).strip().lower()
    key = ALIGN_ALIASES.get(key, key)
    if key not in ALIGN_MODES:
        raise ValueError(
            f"Unknown --align mode: {mode}. Supported modes are: "
            "none, scale, pose_scale, pose_scale_yaw_translation, pose_sim3"
        )
    return key


def has_cli_option(argv: Sequence[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in argv)


def normalize_chunk_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    key = CHUNK_MODE_ALIASES.get(key, key)
    if key not in CHUNK_MODES:
        raise ValueError("--chunk_mode must be one of: sequence, pose_space")
    return key


def parse_chunk_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--align",
        dest="align",
        default=None,
        help=(
            "Alignment mode. none: raw chunks; scale: each chunk independently aligns to GT; "
            "pose_scale: align each chunk to input pose translations using predicted pose translations, scale only; "
            "pose_scale_yaw_translation: pose_scale plus yaw rotation in the XY plane and translation; "
            "pose_sim3: pose translation alignment with scale, full 3D rotation, and translation."
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8,
        help="Number of selected frames per model inference chunk.",
    )
    parser.add_argument(
        "--chunk_interval",
        "--chunk_stride",
        dest="chunk_interval",
        type=int,
        default=None,
        help=(
            "Distance, in selected-frame indices, between consecutive chunk starts. "
            "Default: chunk_size - chunk_overlap."
        ),
    )
    parser.add_argument(
        "--chunk_overlap",
        "--overlap",
        dest="chunk_overlap",
        type=int,
        default=0,
        help="Target number of overlapping images between neighboring chunks.",
    )
    parser.add_argument(
        "--chunk_mode",
        "--chunk_partition",
        dest="chunk_mode",
        default="sequence",
        help=(
            "Chunking mode. sequence keeps the selected frame order. pose_space partitions "
            "input camera centers into a spatial grid, with overlap images from neighboring cells."
        ),
    )
    parser.add_argument(
        "--pose_grid_size",
        type=float,
        default=0.0,
        help=(
            "Grid cell side length for --chunk_mode pose_space, in pose-coordinate units. "
            "<=0 estimates a square/cubic cell size from --chunk_size."
        ),
    )
    parser.add_argument(
        "--pose_grid_axes",
        default="xy",
        help="Pose axes used for grid partitioning: xy, xz, yz, or xyz. Default: xy.",
    )
    parser.add_argument(
        "--pose_grid_neighbor_radius",
        type=int,
        default=1,
        help="Neighbor-cell radius used to choose overlap images for --chunk_mode pose_space.",
    )
    parser.add_argument(
        "--min_chunk_size",
        type=int,
        default=1,
        help="Skip the final chunk if it contains fewer frames than this value.",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=0,
        help="Optional cap on the number of chunks to run; <=0 disables the cap.",
    )
    parser.add_argument(
        "--max_pred_points_per_chunk",
        type=int,
        default=None,
        help="Optional point logging cap per chunk. Default: reuse --max_pred_points.",
    )
    parser.add_argument(
        "--slam_overlap_point_align",
        nargs="?",
        const="sim3",
        default="none",
        type=str.lower,
        choices=["none", "sim3"],
        help=(
            "Optional second-stage SLAM alignment after each chunk's normal alignment. "
            "none disables it. sim3 aligns each later chunk to the previous connected chunk "
            "using one-to-one predicted point-map correspondences from overlapping images. "
            "Passing the flag without a value means sim3."
        ),
    )
    parser.add_argument(
        "--slam_overlap_align_min_corr",
        type=int,
        default=100,
        help="Minimum overlap point correspondences required for --slam_overlap_point_align.",
    )
    parser.add_argument(
        "--slam_overlap_align_max_corr",
        type=int,
        default=20000,
        help="Maximum overlap point correspondences sampled for --slam_overlap_point_align.",
    )
    parser.add_argument(
        "--use_ray_prior",
        dest="use_ray_prior",
        action="store_true",
        default=None,
        help="For geoff3d, force ray/intrinsics prior on via model.task.ray_dirs_prob=1.",
    )
    parser.add_argument(
        "--no_ray_prior",
        dest="use_ray_prior",
        action="store_false",
        help=(
            "For geoff3d, run chunk 0 without external ray/intrinsics prior, "
            "then use average intrinsics recovered from chunk 0 predicted rays for later chunks."
        ),
    )
    parser.add_argument(
        "--use_depth_prior",
        dest="use_depth_prior",
        action="store_true",
        default=None,
        help="For geoff3d, force depth prior on via model.task.depth_prob=1.",
    )
    parser.add_argument(
        "--no_depth_prior",
        dest="use_depth_prior",
        action="store_false",
        help=(
            "For geoff3d, disable external depth prior and feed predicted "
            "depth from overlapping previous-chunk frames into later chunks."
        ),
    )
    parser.add_argument(
        "--use_world_rotation_prior",
        dest="use_world_rotation_prior",
        action="store_true",
        default=None,
        help=(
            "For world-prior models, force world rotation prior on via "
            "model.task.world_rotation_prob=1."
        ),
    )
    parser.add_argument(
        "--no_world_rotation_prior",
        dest="use_world_rotation_prior",
        action="store_false",
        help=(
            "For world-prior models, disable external world rotation prior and feed "
            "predicted rotation from overlapping previous-chunk frames into later chunks."
        ),
    )
    parser.add_argument(
        "--use_world_translation_prior",
        dest="use_world_translation_prior",
        action="store_true",
        default=True,
        help=(
            "For world-prior models, force world translation prior on for all views "
            "via model.task.world_translation_prob=1. This is the default."
        ),
    )
    parser.add_argument(
        "--no_world_translation_prior",
        dest="use_world_translation_prior",
        action="store_false",
        help=(
            "For world-prior models, force world translation prior off via "
            "model.task.world_translation_prob=0."
        ),
    )
    parser.add_argument(
        "--use_fov_prior",
        dest="use_fov_prior",
        action="store_true",
        default=None,
        help="Force FOV prior on via model.task.fov_prob=1.",
    )
    parser.add_argument(
        "--no_fov_prior",
        dest="use_fov_prior",
        action="store_false",
        help="Force FOV prior off via model.task.fov_prob=0.",
    )
    chunk_args, remaining = parser.parse_known_args(list(argv))
    return chunk_args, remaining


def append_world_prior_overrides(args: argparse.Namespace, chunk_args: argparse.Namespace) -> None:
    model_name = str(args.model)
    if model_name not in WORLD_TRANSLATION_PRIOR_MODELS:
        return

    overrides = list(getattr(args, "hydra_override", []) or [])

    def add_bool_override(enabled: bool, task_key: str, model_config_key: str | None = None) -> None:
        value = "1.0" if enabled else "0.0"
        overrides.append(f"model.task.{task_key}={value}")
        if model_config_key is not None:
            overrides.append(f"model.model_config.{model_config_key}={'true' if enabled else 'false'}")

    # Translation is the stable default anchor for this smoke-test script. In eval
    # mode, probability 1.0 means every selected view is used by the model mask.
    add_bool_override(
        bool(chunk_args.use_world_translation_prior),
        "world_translation_prob",
        "use_world_translation_prior",
    )

    if model_name in PI3X_PRIOR_MODELS and chunk_args.use_ray_prior is not None:
        add_bool_override(bool(chunk_args.use_ray_prior), "ray_dirs_prob")
    elif chunk_args.use_ray_prior is not None:
        print(f"[WARN] --use/--no_ray_prior is ignored for model={model_name}.")

    if model_name in PI3X_PRIOR_MODELS and chunk_args.use_depth_prior is not None:
        add_bool_override(bool(chunk_args.use_depth_prior), "depth_prob")
    elif chunk_args.use_depth_prior is not None:
        print(f"[WARN] --use/--no_depth_prior is ignored for model={model_name}.")

    if chunk_args.use_world_rotation_prior is not None:
        add_bool_override(bool(chunk_args.use_world_rotation_prior), "world_rotation_prob")
        overrides.append("model.model_config.use_world_rotation_prior=true")

    if model_name in FOV_PRIOR_MODELS and chunk_args.use_fov_prior is not None:
        add_bool_override(bool(chunk_args.use_fov_prior), "fov_prob", "use_fov_prior")
    elif chunk_args.use_fov_prior is not None:
        print(f"[WARN] --use/--no_fov_prior is ignored for model={model_name}.")

    args.hydra_override = overrides

    print(
        f"[INFO] {model_name} prior overrides: "
        + ", ".join(overrides[-8:])
    )


def set_model_task_value(model: torch.nn.Module, key: str, value: float) -> None:
    cfg = getattr(model, "geometric_input_config", None)
    if cfg is None:
        return
    try:
        from omegaconf import open_dict

        with open_dict(cfg):
            cfg[key] = float(value)
        return
    except Exception:
        pass
    try:
        cfg[key] = float(value)
    except Exception:
        try:
            setattr(cfg, key, float(value))
        except Exception:
            print(f"[WARN] Could not update model.geometric_input_config.{key} at runtime.")


def set_model_task_prob(model: torch.nn.Module, key: str, enabled: bool) -> None:
    set_model_task_value(model, key, 1.0 if enabled else 0.0)


def set_pi3x_ray_prior_prob(model: torch.nn.Module, enabled: bool) -> None:
    set_model_task_prob(model, "ray_dirs_prob", enabled)


def recover_average_intrinsics_from_pred_rays(preds: Sequence[Dict[str, torch.Tensor]]) -> Optional[torch.Tensor]:
    intrinsics: List[torch.Tensor] = []
    for pred in preds:
        rays = pred.get("ray_directions", None)
        if rays is None:
            continue
        K = recover_pinhole_intrinsics_from_ray_directions(
            rays.float(),
            use_geometric_calculation=True,
        )
        if K.dim() == 2:
            K = K.unsqueeze(0)
        finite = torch.isfinite(K).flatten(1).all(dim=1)
        if bool(finite.any()):
            intrinsics.append(K[finite])

    if not intrinsics:
        return None

    K_mean = torch.cat(intrinsics, dim=0).mean(dim=0)
    if not bool(torch.isfinite(K_mean).all()):
        return None
    return K_mean


def apply_bootstrap_intrinsics_to_views(
    views: Sequence[Dict[str, object]],
    intrinsics: torch.Tensor,
    device: torch.device,
) -> List[Dict[str, object]]:
    K = intrinsics.to(device=device, dtype=torch.float32).unsqueeze(0)
    return [{**view, "camera_intrinsics": K.clone()} for view in views]


def pred_depth_prior(pred: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    pts_cam = pred.get("pts3d_cam", None)
    if pts_cam is None:
        return None
    depth = pts_cam[..., 2]
    if depth.dim() == 3:
        depth = depth[:1]
    elif depth.dim() == 2:
        depth = depth.unsqueeze(0)
    else:
        return None
    depth = depth.detach().float()
    valid = torch.isfinite(depth) & (depth > 0)
    if not bool(valid.any()):
        return None
    return torch.where(valid, depth, torch.zeros_like(depth))


def pred_world_rotation_prior(pred: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    quats = pred.get("cam_quats", None)
    if quats is None:
        return None
    if quats.dim() == 1:
        quats = quats.unsqueeze(0)
    if quats.shape[-1] != 4:
        return None
    rotation = base.quaternion_to_rotation_matrix(quats.float())
    if rotation.dim() == 2:
        rotation = rotation.unsqueeze(0)
    rotation = rotation[:1].detach().float()
    if not bool(torch.isfinite(rotation).all()):
        return None
    return rotation


def build_prediction_prior_cache(
    preds: Sequence[Dict[str, torch.Tensor]],
    indices: Sequence[int],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    depth_cache: Dict[int, torch.Tensor] = {}
    rotation_cache: Dict[int, torch.Tensor] = {}
    for global_idx, pred in zip(indices, preds):
        depth = pred_depth_prior(pred)
        if depth is not None:
            depth_cache[int(global_idx)] = depth.detach().cpu()
        rotation = pred_world_rotation_prior(pred)
        if rotation is not None:
            rotation_cache[int(global_idx)] = rotation.detach().cpu()
    return depth_cache, rotation_cache


def apply_overlap_depth_priors_to_views(
    views: Sequence[Dict[str, object]],
    indices: Sequence[int],
    overlap_indices: Sequence[int],
    depth_cache: Dict[int, torch.Tensor],
    device: torch.device,
) -> Tuple[List[Dict[str, object]], int]:
    overlap = set(int(i) for i in overlap_indices)
    matched = [int(i) for i in indices if int(i) in overlap and int(i) in depth_cache]
    if not matched:
        return list(views), 0

    template = depth_cache[matched[0]].to(device=device, dtype=torch.float32)
    zero_depth = torch.zeros_like(template)
    out: List[Dict[str, object]] = []
    used = 0
    for view, global_idx in zip(views, indices):
        prior = depth_cache.get(int(global_idx)) if int(global_idx) in overlap else None
        next_view = dict(view)
        if prior is None:
            next_view["depthmap"] = zero_depth.clone()
            next_view["depth_prior_mask"] = torch.zeros((1,), device=device, dtype=torch.bool)
        else:
            next_view["depthmap"] = prior.to(device=device, dtype=torch.float32)
            next_view["depth_prior_mask"] = torch.ones((1,), device=device, dtype=torch.bool)
            used += 1
        out.append(next_view)
    return out, used


def apply_overlap_rotation_priors_to_views(
    views: Sequence[Dict[str, object]],
    indices: Sequence[int],
    overlap_indices: Sequence[int],
    rotation_cache: Dict[int, torch.Tensor],
    device: torch.device,
) -> Tuple[List[Dict[str, object]], int]:
    overlap = set(int(i) for i in overlap_indices)
    matched = [int(i) for i in indices if int(i) in overlap and int(i) in rotation_cache]
    if not matched:
        return list(views), 0

    identity = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0)
    out: List[Dict[str, object]] = []
    used = 0
    for view, global_idx in zip(views, indices):
        prior = rotation_cache.get(int(global_idx)) if int(global_idx) in overlap else None
        next_view = dict(view)
        if prior is None:
            next_view["world_rotation"] = identity.clone()
            next_view["world_rotation_mask"] = torch.zeros((1,), device=device, dtype=torch.bool)
        else:
            next_view["world_rotation"] = prior.to(device=device, dtype=torch.float32)
            next_view["world_rotation_mask"] = torch.ones((1,), device=device, dtype=torch.bool)
            used += 1
        out.append(next_view)
    return out, used


def validate_chunk_args(chunk_args: argparse.Namespace) -> int:
    chunk_args.align = normalize_align_mode(chunk_args.align)
    chunk_args.chunk_mode = normalize_chunk_mode(chunk_args.chunk_mode)
    chunk_args.pose_grid_axes = str(chunk_args.pose_grid_axes).strip().lower()
    if chunk_args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")
    if chunk_args.chunk_overlap < 0:
        raise ValueError("--chunk_overlap must be non-negative")
    if chunk_args.chunk_overlap >= chunk_args.chunk_size:
        raise ValueError("--chunk_overlap must be smaller than --chunk_size")
    if chunk_args.min_chunk_size <= 0:
        raise ValueError("--min_chunk_size must be positive")
    if chunk_args.slam_overlap_align_min_corr <= 0:
        raise ValueError("--slam_overlap_align_min_corr must be positive")
    if chunk_args.slam_overlap_align_max_corr <= 0:
        raise ValueError("--slam_overlap_align_max_corr must be positive")
    if chunk_args.slam_overlap_align_max_corr < chunk_args.slam_overlap_align_min_corr:
        raise ValueError("--slam_overlap_align_max_corr must be >= --slam_overlap_align_min_corr")
    if chunk_args.pose_grid_axes not in {"xy", "xz", "yz", "xyz"}:
        raise ValueError("--pose_grid_axes must be one of: xy, xz, yz, xyz")
    if chunk_args.pose_grid_neighbor_radius < 0:
        raise ValueError("--pose_grid_neighbor_radius must be non-negative")

    if chunk_args.chunk_interval is None:
        interval = int(chunk_args.chunk_size) - int(chunk_args.chunk_overlap)
    else:
        interval = int(chunk_args.chunk_interval)
    if interval <= 0:
        raise ValueError("--chunk_interval must be positive")
    return interval


def parse_args() -> Tuple[argparse.Namespace, argparse.Namespace, int]:
    chunk_args, remaining_argv = parse_chunk_args(sys.argv[1:])
    interval = validate_chunk_args(chunk_args)

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining_argv]
        args = base.parse_args()
    finally:
        sys.argv = old_argv

    if str(args.model) in VGGT_OMEGA_MODELS and not has_cli_option(remaining_argv, "--size_multiple"):
        args.size_multiple = 16
        print(
            "[INFO] VGGT-Omega model selected; using --size_multiple 16 "
            "because no explicit --size_multiple was provided."
        )

    if chunk_args.align is None:
        chunk_args.runtime_align = str(args.align).lower()
    else:
        chunk_args.runtime_align = str(chunk_args.align).lower()

    if str(chunk_args.runtime_align).lower() == "scale":
        print("[INFO] Each chunk will be independently scale+translation aligned to GT; no rotation is applied.")
    elif str(chunk_args.runtime_align).lower() == "pose_scale":
        print("[INFO] Each chunk will be aligned to input pose translations using predicted pose translations: scale only.")
    elif str(chunk_args.runtime_align).lower() == "pose_scale_yaw_translation":
        print(
            "[INFO] Each chunk will be aligned to input pose translations using predicted pose translations: "
            "scale, then XY-plane yaw rotation around Z and translation."
        )
    elif str(chunk_args.runtime_align).lower() == "pose_sim3":
        print(
            "[INFO] Each chunk will be aligned to input pose translations using predicted pose translations: "
            "scale, full 3D rotation, and translation."
        )
    elif str(chunk_args.runtime_align).lower() == "none":
        print("[WARN] --align none selected; raw chunk predictions will be logged.")
    else:
        raise ValueError(f"Unknown --align mode after parsing: {chunk_args.runtime_align}")

    if chunk_args.max_pred_points_per_chunk is None:
        chunk_args.max_pred_points_per_chunk = int(args.max_pred_points)
    append_world_prior_overrides(args, chunk_args)
    return args, chunk_args, interval


def iter_chunk_slices(
    num_frames: int,
    chunk_size: int,
    interval: int,
    min_chunk_size: int,
    max_chunks: int,
) -> List[Tuple[int, int, List[int], List[int]]]:
    chunks: List[Tuple[int, int, List[int], List[int]]] = []
    previous: set[int] = set()
    covered_end = 0

    start = 0
    while start < num_frames:
        end = min(start + int(chunk_size), num_frames)
        if end <= covered_end:
            break
        indices = list(range(start, end))
        if len(indices) < int(min_chunk_size):
            break
        overlap_indices = sorted(previous.intersection(indices))
        chunks.append((start, end, indices, overlap_indices))
        previous = set(indices)
        covered_end = max(covered_end, end)
        if max_chunks > 0 and len(chunks) >= int(max_chunks):
            break
        start += int(interval)

    return chunks


def pose_centers_from_meta(meta: Dict[str, object]) -> np.ndarray:
    cams = meta.get("cams", {})
    stems = list(meta.get("stems", []))
    centers: List[np.ndarray] = []
    missing: List[str] = []

    for stem in stems:
        cam = cams.get(stem) if isinstance(cams, dict) else None
        if cam is None:
            missing.append(str(stem))
            continue
        T = np.asarray(cam.get("T_c2w"), dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T[:3, 3]).all():
            missing.append(str(stem))
            continue
        centers.append(T[:3, 3].astype(np.float64))

    if missing:
        shown = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise ValueError(
            "--chunk_mode pose_space requires a finite input camera pose for every selected frame; "
            f"missing/invalid pose for: {shown}{suffix}"
        )

    if not centers:
        raise ValueError("--chunk_mode pose_space requires at least one selected frame with an input camera pose.")

    return np.stack(centers, axis=0)


def pose_grid_axis_indices(axes: str) -> Tuple[int, ...]:
    mapping = {"x": 0, "y": 1, "z": 2}
    return tuple(mapping[ch] for ch in str(axes).lower())


def estimate_pose_grid_size(coords: np.ndarray, target_core_size: int) -> float:
    pts = np.asarray(coords, dtype=np.float64).reshape(-1, coords.shape[-1])
    if pts.shape[0] <= 1:
        return 1.0
    target_cells = max(1, int(np.ceil(float(pts.shape[0]) / max(1, int(target_core_size)))))
    cells_per_axis = max(1, int(np.ceil(target_cells ** (1.0 / float(pts.shape[1])))))
    extent = pts.max(axis=0) - pts.min(axis=0)
    max_extent = float(np.max(extent))
    if not np.isfinite(max_extent) or max_extent <= 0:
        return 1.0
    return max(max_extent / float(cells_per_axis), 1e-6)


def pose_grid_cell_order(cell_keys: Sequence[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    keys = sorted(tuple(int(v) for v in key) for key in cell_keys)
    if not keys or len(keys[0]) != 2:
        return keys

    rows: Dict[int, List[Tuple[int, int]]] = {}
    for x, y in keys:
        rows.setdefault(y, []).append((x, y))

    ordered: List[Tuple[int, int]] = []
    for row_i, y in enumerate(sorted(rows)):
        row = sorted(rows[y], key=lambda key: key[0], reverse=bool(row_i % 2))
        ordered.extend(row)
    return ordered


def build_pose_grid_chunks(
    centers: np.ndarray,
    chunk_args: argparse.Namespace,
) -> Tuple[List[Tuple[int, int, List[int], List[int]]], List[int]]:
    axis_indices = pose_grid_axis_indices(str(chunk_args.pose_grid_axes))
    coords = np.asarray(centers, dtype=np.float64)[:, axis_indices]
    origin = coords.min(axis=0)
    grid_size = float(chunk_args.pose_grid_size)
    if grid_size <= 0:
        grid_size = estimate_pose_grid_size(coords, int(chunk_args.chunk_size))

    cell_coords = np.floor((coords - origin[None, :]) / grid_size).astype(np.int64)
    cell_to_core: Dict[Tuple[int, ...], List[int]] = {}
    for frame_idx, cell in enumerate(cell_coords):
        key = tuple(int(v) for v in cell)
        cell_to_core.setdefault(key, []).append(int(frame_idx))

    ordered_cells = pose_grid_cell_order(cell_to_core.keys())
    chunks: List[Tuple[int, int, List[int], List[int]]] = []
    previous: set[int] = set()
    core_order: List[int] = []
    overlap_count = max(0, int(chunk_args.chunk_overlap))
    neighbor_radius = int(chunk_args.pose_grid_neighbor_radius)

    cell_centers = {
        key: coords[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        for key, indices in cell_to_core.items()
    }

    for cell_order_idx, cell_key in enumerate(ordered_cells):
        core_indices = sorted(cell_to_core[cell_key])
        if len(core_indices) < int(chunk_args.min_chunk_size):
            continue

        neighbor_candidates: List[int] = []
        if overlap_count > 0 and neighbor_radius > 0:
            for other_key, other_indices in cell_to_core.items():
                if other_key == cell_key:
                    continue
                if max(abs(a - b) for a, b in zip(cell_key, other_key)) > neighbor_radius:
                    continue
                neighbor_candidates.extend(int(i) for i in other_indices)

        overlap_extra: List[int] = []
        if neighbor_candidates:
            center = cell_centers[cell_key]
            unique_candidates = sorted(set(neighbor_candidates) - set(core_indices))
            candidate_coords = coords[np.asarray(unique_candidates, dtype=np.int64)]
            distances = np.linalg.norm(candidate_coords - center[None, :], axis=1)
            order = np.argsort(distances, kind="stable")
            overlap_extra = [unique_candidates[int(i)] for i in order[:overlap_count]]

        indices = sorted(set(core_indices + overlap_extra))
        overlap_indices = sorted(previous.intersection(indices))
        chunks.append((cell_order_idx, cell_order_idx + 1, indices, overlap_indices))
        previous = set(indices)
        core_order.extend(core_indices)

        if int(chunk_args.max_chunks) > 0 and len(chunks) >= int(chunk_args.max_chunks):
            break

    chunk_args.pose_grid_effective_size = float(grid_size)
    chunk_args.pose_grid_origin = origin.astype(float).tolist()
    chunk_args.pose_grid_num_occupied_cells = int(len(ordered_cells))
    return chunks, core_order


def build_chunk_slices(
    meta: Dict[str, object],
    chunk_args: argparse.Namespace,
    interval: int,
) -> Tuple[List[Tuple[int, int, List[int], List[int]]], List[int]]:
    num_frames = len(meta.get("stems", []))
    if str(chunk_args.chunk_mode) == "sequence":
        ordered_indices = list(range(num_frames))
        chunks = iter_chunk_slices(
            num_frames=num_frames,
            chunk_size=int(chunk_args.chunk_size),
            interval=int(interval),
            min_chunk_size=int(chunk_args.min_chunk_size),
            max_chunks=int(chunk_args.max_chunks),
        )
        return chunks, ordered_indices

    centers = pose_centers_from_meta(meta)
    return build_pose_grid_chunks(centers, chunk_args)


def gt_cameras_for_stems(meta: Dict[str, object], stems: Sequence[str]) -> List[Dict[str, object]]:
    cams = meta.get("cams", {})
    out = []
    for stem in stems:
        if stem not in cams:
            continue
        out.append({"stem": stem, "T_c2w": np.asarray(cams[stem]["T_c2w"], dtype=np.float32)})
    return out


def make_chunk_meta(meta: Dict[str, object], indices: Sequence[int]) -> Dict[str, object]:
    chunk_meta = dict(meta)
    chunk_meta["stems"] = [meta["stems"][i] for i in indices]
    chunk_meta["rgbs"] = [meta["rgbs"][i] for i in indices]
    chunk_meta["gt_maps"] = [meta["gt_maps"][i] for i in indices]
    chunk_meta["valid_masks"] = [meta["valid_masks"][i] for i in indices]
    return chunk_meta


def sample_chunk_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    return base.sample_points_and_colors(points, colors, max_points=max_points, seed=seed)


def camera_centers_by_stem(cams: Sequence[Dict[str, object]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for cam in cams:
        stem = str(cam.get("stem", ""))
        if not stem:
            continue
        T = np.asarray(cam.get("T_c2w"), dtype=np.float64)
        if T.shape != (4, 4):
            continue
        center = T[:3, 3]
        if np.isfinite(center).all():
            out[stem] = center.astype(np.float64)
    return out


def matched_pose_translation_correspondences(
    reference_cams_by_stem: Dict[str, np.ndarray],
    current_cams: Sequence[Dict[str, object]],
    target_stems: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    current_by_stem = camera_centers_by_stem(current_cams)
    ref_corr: List[np.ndarray] = []
    cur_corr: List[np.ndarray] = []
    matched_stems: List[str] = []
    for stem in target_stems:
        stem = str(stem)
        ref_center = reference_cams_by_stem.get(stem)
        cur_center = current_by_stem.get(stem)
        if ref_center is None or cur_center is None:
            continue
        ref_corr.append(ref_center)
        cur_corr.append(cur_center)
        matched_stems.append(stem)

    if not ref_corr:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            [],
        )
    return (
        np.asarray(ref_corr, dtype=np.float32).reshape(-1, 3),
        np.asarray(cur_corr, dtype=np.float32).reshape(-1, 3),
        matched_stems,
    )


def yaw_rotation_from_xy_correspondences(src_xy: np.ndarray, dst_xy: np.ndarray) -> Tuple[float, bool, str]:
    src = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
    if src.shape[0] < 2 or dst.shape[0] != src.shape[0]:
        return 0.0, False, "not enough correspondences to estimate yaw"

    src_centered = src - np.mean(src, axis=0, keepdims=True)
    dst_centered = dst - np.mean(dst, axis=0, keepdims=True)
    if np.linalg.norm(src_centered) < 1e-8 or np.linalg.norm(dst_centered) < 1e-8:
        return 0.0, False, "not enough non-zero XY baselines to estimate yaw"

    cross = float(np.sum(src_centered[:, 0] * dst_centered[:, 1] - src_centered[:, 1] * dst_centered[:, 0]))
    dot = float(np.sum(src_centered[:, 0] * dst_centered[:, 0] + src_centered[:, 1] * dst_centered[:, 1]))
    yaw = float(np.arctan2(cross, dot))
    if not np.isfinite(yaw):
        return 0.0, False, "yaw solve produced non-finite angle"
    return yaw, True, "yaw estimated from XY pose baselines"


def rotation_matrix_z(yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_similarity_umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    estimate_scale: bool = True,
    eps: float = 1e-12,
) -> Tuple[float, np.ndarray, np.ndarray, bool, str]:
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] < 3 or dst.shape != src.shape:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), False, (
            "not enough 3D correspondences to estimate full similarity"
        )

    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[finite]
    dst = dst[finite]
    if src.shape[0] < 3:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), False, (
            "not enough finite 3D correspondences to estimate full similarity"
        )

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst
    src_var = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if src_var <= eps:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), False, (
            "source pose translations are degenerate"
        )

    cov = (dst_centered.T @ src_centered) / float(src.shape[0])
    try:
        U, singular_values, Vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), False, "SVD failed"

    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if estimate_scale:
        scale = float(np.sum(singular_values * np.diag(S)) / src_var)
    else:
        scale = 1.0
    t = mu_dst - scale * (R @ mu_src)

    if not np.isfinite(scale) or scale <= eps or not np.isfinite(R).all() or not np.isfinite(t).all():
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), False, (
            "full similarity solve produced non-finite transform"
        )
    return scale, R, t, True, "scale+rotation+translation estimated from 3D pose correspondences"


def apply_similarity_to_points(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts
    out = (float(scale) * pts.astype(np.float64)) @ np.asarray(R, dtype=np.float64).T
    out += np.asarray(t, dtype=np.float64)[None, :]
    return out.astype(np.float32)


def apply_similarity_to_point_maps(
    point_maps: Sequence[np.ndarray],
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> List[np.ndarray]:
    out = []
    for point_map in point_maps:
        shape = point_map.shape
        if point_map.size == 0:
            out.append(point_map)
            continue
        out.append(apply_similarity_to_points(point_map.reshape(-1, 3), scale, R, t).reshape(shape))
    return out


def apply_similarity_to_cameras(
    cams: Sequence[Dict[str, object]],
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> List[Dict[str, object]]:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    aligned = []
    for cam in cams:
        T = np.asarray(cam["T_c2w"], dtype=np.float64)
        T_al = np.eye(4, dtype=np.float32)
        T_al[:3, :3] = (R @ T[:3, :3]).astype(np.float32)
        T_al[:3, 3] = (R @ (float(scale) * T[:3, 3]) + t).astype(np.float32)
        aligned.append({**cam, "T_c2w": T_al})
    return aligned


def identity_pose_alignment_meta(mode: str, chunk_id: int, note: str, valid: bool) -> Dict[str, object]:
    return {
        "mode": mode,
        "valid": bool(valid),
        "source": "pose_translations",
        "num_corr": 0,
        "num_scale_pairs": 0,
        "matched_camera_stems": [],
        "scale": 1.0,
        "yaw_degrees": 0.0,
        "yaw_valid": False,
        "R": np.eye(3, dtype=np.float32).tolist(),
        "t": np.zeros(3, dtype=np.float32).tolist(),
        "median_residual": float("nan"),
        "note": note,
        "chunk_id": int(chunk_id),
    }


def estimate_chunk_pose_alignment(
    mode: str,
    chunk_id: int,
    reference_cams_by_stem: Dict[str, np.ndarray],
    raw_pred_cams: Sequence[Dict[str, object]],
    target_stems: Sequence[str],
    seed: int,
) -> Dict[str, object]:
    ref_corr, cur_corr, matched_stems = matched_pose_translation_correspondences(
        reference_cams_by_stem=reference_cams_by_stem,
        current_cams=raw_pred_cams,
        target_stems=target_stems,
    )
    if ref_corr.shape[0] == 0:
        return identity_pose_alignment_meta(
            mode,
            chunk_id,
            "no predicted pose translations matched input pose translations for this chunk; using raw chunk coordinates",
            valid=False,
        )

    scale, num_scale_pairs, scale_valid = base.estimate_scale_from_random_baselines(
        pr_corr=cur_corr,
        gt_corr=ref_corr,
        seed=seed,
    )
    scale_note = "scale estimated from matched pose-translation baselines"
    if not scale_valid:
        scale = 1.0
        scale_note = "scale fallback to 1.0; not enough non-zero pose-translation baselines"

    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    yaw = 0.0
    yaw_valid = False
    yaw_note = "yaw not requested"
    if mode == "pose_sim3":
        scale, R, t, sim3_valid, sim3_note = estimate_similarity_umeyama(
            src=cur_corr,
            dst=ref_corr,
            estimate_scale=True,
        )
        if not sim3_valid:
            return identity_pose_alignment_meta(
                mode,
                chunk_id,
                f"{sim3_note}; using raw chunk coordinates",
                valid=False,
            )
        scale_note = sim3_note
        yaw_note = "full 3D rotation estimated"
    elif mode == "pose_scale_yaw_translation":
        scaled_cur = float(scale) * cur_corr.astype(np.float64)
        yaw, yaw_valid, yaw_note = yaw_rotation_from_xy_correspondences(
            src_xy=scaled_cur[:, :2],
            dst_xy=ref_corr[:, :2],
        )
        R = rotation_matrix_z(yaw)
        transformed = scaled_cur @ R.T
        t = np.median(ref_corr.astype(np.float64) - transformed, axis=0)

    if not np.isfinite(scale) or scale <= 1e-12 or not np.isfinite(R).all() or not np.isfinite(t).all():
        return identity_pose_alignment_meta(
            mode,
            chunk_id,
            "pose alignment solve produced non-finite transform; using raw chunk coordinates",
            valid=False,
        )

    transformed_corr = apply_similarity_to_points(cur_corr, scale, R, t)
    residual = np.linalg.norm(transformed_corr.astype(np.float64) - ref_corr.astype(np.float64), axis=1)
    median_residual = float(np.median(residual)) if residual.size else float("nan")
    note = f"{scale_note}; {yaw_note}"
    if mode == "pose_scale":
        note += "; no rotation or translation applied"
    elif mode == "pose_sim3":
        note += "; full 3D rotation and translation applied"
    else:
        note += "; translation estimated after scale+yaw"

    return {
        "mode": mode,
        "valid": True,
        "source": "pose_translations",
        "num_corr": int(ref_corr.shape[0]),
        "num_scale_pairs": int(num_scale_pairs),
        "matched_camera_stems": matched_stems,
        "scale": float(scale),
        "yaw_degrees": float(np.degrees(yaw)),
        "yaw_valid": bool(yaw_valid),
        "R": R.astype(np.float32).tolist(),
        "t": t.astype(np.float32).tolist(),
        "median_residual": median_residual,
        "note": note,
        "chunk_id": int(chunk_id),
    }


def apply_chunk_pose_alignment(
    mode: str,
    chunk_id: int,
    reference_cams_by_stem: Dict[str, np.ndarray],
    raw_pred_points: np.ndarray,
    raw_pred_colors: np.ndarray,
    pred_maps: Sequence[np.ndarray],
    raw_pred_cams: Sequence[Dict[str, object]],
    target_stems: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Dict[str, object]], Dict[str, object]]:
    align_meta = estimate_chunk_pose_alignment(
        mode=mode,
        chunk_id=chunk_id,
        reference_cams_by_stem=reference_cams_by_stem,
        raw_pred_cams=raw_pred_cams,
        target_stems=target_stems,
        seed=seed,
    )
    scale = float(align_meta["scale"])
    R = np.asarray(align_meta["R"], dtype=np.float64)
    t = np.asarray(align_meta["t"], dtype=np.float64)
    pred_points_aligned = apply_similarity_to_points(raw_pred_points, scale, R, t)
    pred_maps_aligned = apply_similarity_to_point_maps(pred_maps, scale, R, t)
    pred_cams_aligned = apply_similarity_to_cameras(raw_pred_cams, scale, R, t)
    return pred_points_aligned, raw_pred_colors, pred_maps_aligned, pred_cams_aligned, align_meta


def identity_slam_overlap_alignment_meta(mode: str, chunk_id: int, note: str, valid: bool) -> Dict[str, object]:
    return {
        "mode": mode,
        "alignment_type": "none" if mode == "none" else "identity",
        "valid": bool(valid),
        "source": "overlap_pointmaps",
        "num_corr": 0,
        "num_pose_corr": 0,
        "num_inliers": 0,
        "matched_stems": [],
        "matched_pose_stems": [],
        "scale": 1.0,
        "R": np.eye(3, dtype=np.float32).tolist(),
        "t": np.zeros(3, dtype=np.float32).tolist(),
        "median_residual": float("nan"),
        "initial_median_residual": float("nan"),
        "inlier_median_residual": float("nan"),
        "inlier_threshold": float("nan"),
        "inlier_refined": False,
        "note": note,
        "chunk_id": int(chunk_id),
    }


def invert_transform_matrix(T: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(T, dtype=np.float64).reshape(4, 4))


def similarity_components_from_matrix(T: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, bool, str]:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    A = T[:3, :3]
    det = float(np.linalg.det(A))
    if not np.isfinite(det) or abs(det) <= 1e-12:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), False, (
            "overlap pose transform has degenerate linear part"
        )
    scale = float(np.cbrt(det))
    if not np.isfinite(scale) or abs(scale) <= 1e-12:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), False, (
            "overlap pose transform has invalid scale"
        )
    R = A / scale
    t = T[:3, 3]
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), False, (
            "overlap pose transform contains non-finite values"
        )
    return scale, R, t, True, "scale+rotation+translation extracted from overlap pose transform"


def estimate_slam_overlap_pose_alignment(
    mode: str,
    chunk_id: int,
    overlap_stems: Sequence[str],
    reference_cams_by_stem: Dict[str, np.ndarray],
    current_cams_by_stem: Dict[str, np.ndarray],
    prefix_note: str = "",
) -> Dict[str, object]:
    model_centers: List[np.ndarray] = []
    world_centers: List[np.ndarray] = []
    pose_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    matched_stems: List[str] = []
    for stem in overlap_stems:
        stem = str(stem)
        reference_cam = reference_cams_by_stem.get(stem)
        current_cam = current_cams_by_stem.get(stem)
        if reference_cam is None or current_cam is None:
            continue
        reference_cam = np.asarray(reference_cam, dtype=np.float64).reshape(4, 4)
        current_cam = np.asarray(current_cam, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(reference_cam).all() or not np.isfinite(current_cam).all():
            continue
        world_centers.append(reference_cam[:3, 3])
        model_centers.append(current_cam[:3, 3])
        pose_pairs.append((reference_cam, current_cam))
        matched_stems.append(stem)

    note_prefix = f"{prefix_note}; " if prefix_note else ""
    if len(model_centers) >= 3:
        scale, R, t, valid, note = estimate_similarity_umeyama(
            src=np.asarray(model_centers, dtype=np.float32),
            dst=np.asarray(world_centers, dtype=np.float32),
            estimate_scale=True,
        )
        if valid:
            transformed = apply_similarity_to_points(np.asarray(model_centers, dtype=np.float32), scale, R, t)
            residual = np.linalg.norm(
                transformed.astype(np.float64) - np.asarray(world_centers, dtype=np.float64),
                axis=1,
            )
            return {
                "mode": mode,
                "alignment_type": "sim3_overlap_pose",
                "valid": True,
                "source": "overlap_pose",
                "num_corr": 0,
                "num_pose_corr": int(len(model_centers)),
                "num_inliers": 0,
                "matched_stems": [],
                "matched_pose_stems": matched_stems,
                "scale": float(scale),
                "R": R.astype(np.float32).tolist(),
                "t": t.astype(np.float32).tolist(),
                "median_residual": float(np.median(residual)) if residual.size else float("nan"),
                "initial_median_residual": float("nan"),
                "inlier_median_residual": float("nan"),
                "inlier_threshold": float("nan"),
                "inlier_refined": False,
                "note": f"{note_prefix}fallback to VGGT-SLAM-style sim3_overlap_pose; {note}",
                "chunk_id": int(chunk_id),
            }

    if pose_pairs:
        reference_cam, current_cam = pose_pairs[0]
        try:
            T = reference_cam @ invert_transform_matrix(current_cam)
        except np.linalg.LinAlgError:
            return identity_slam_overlap_alignment_meta(
                mode,
                chunk_id,
                f"{note_prefix}overlap pose fallback failed because current pose is singular",
                valid=False,
            )
        scale, R, t, valid, note = similarity_components_from_matrix(T)
        if valid:
            return {
                "mode": mode,
                "alignment_type": "overlap_pose",
                "valid": True,
                "source": "overlap_pose",
                "num_corr": 0,
                "num_pose_corr": int(len(pose_pairs)),
                "num_inliers": 0,
                "matched_stems": [],
                "matched_pose_stems": matched_stems[:1],
                "scale": float(scale),
                "R": R.astype(np.float32).tolist(),
                "t": t.astype(np.float32).tolist(),
                "median_residual": float("nan"),
                "initial_median_residual": float("nan"),
                "inlier_median_residual": float("nan"),
                "inlier_threshold": float("nan"),
                "inlier_refined": False,
                "note": f"{note_prefix}fallback to VGGT-SLAM-style single-frame overlap_pose; {note}",
                "chunk_id": int(chunk_id),
            }

    return identity_slam_overlap_alignment_meta(
        mode,
        chunk_id,
        f"{note_prefix}no usable overlap points or poses; using first-stage chunk alignment only",
        valid=False,
    )


def sample_overlap_pointmap_correspondences(
    overlap_stems: Sequence[str],
    reference_maps_by_stem: Dict[str, np.ndarray],
    reference_masks_by_stem: Dict[str, np.ndarray],
    current_maps_by_stem: Dict[str, np.ndarray],
    current_masks_by_stem: Dict[str, np.ndarray],
    max_corr: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    valid_records: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    total_valid = 0

    for stem in overlap_stems:
        stem = str(stem)
        ref_map = reference_maps_by_stem.get(stem)
        cur_map = current_maps_by_stem.get(stem)
        if ref_map is None or cur_map is None:
            continue
        ref_map = np.asarray(ref_map, dtype=np.float32)
        cur_map = np.asarray(cur_map, dtype=np.float32)
        if ref_map.shape != cur_map.shape or ref_map.ndim != 3 or ref_map.shape[-1] != 3:
            continue

        ref_mask = np.asarray(reference_masks_by_stem.get(stem, np.ones(ref_map.shape[:2], dtype=bool)), dtype=bool)
        cur_mask = np.asarray(current_masks_by_stem.get(stem, np.ones(cur_map.shape[:2], dtype=bool)), dtype=bool)
        if ref_mask.shape != ref_map.shape[:2] or cur_mask.shape != cur_map.shape[:2]:
            continue
        valid = (
            ref_mask
            & cur_mask
            & np.isfinite(ref_map).all(axis=-1)
            & np.isfinite(cur_map).all(axis=-1)
        )
        flat_idx = np.flatnonzero(valid.reshape(-1))
        if flat_idx.size == 0:
            continue

        valid_records.append((stem, ref_map, cur_map, flat_idx))
        total_valid += int(flat_idx.size)

    if not valid_records:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            [],
        )

    if total_valid > int(max_corr):
        keep_global = np.linspace(0, total_valid - 1, int(max_corr)).round().astype(np.int64)
    else:
        keep_global = np.arange(total_valid, dtype=np.int64)

    ref_parts: List[np.ndarray] = []
    cur_parts: List[np.ndarray] = []
    matched_stems: List[str] = []
    offset = 0
    for stem, ref_map, cur_map, flat_idx in valid_records:
        next_offset = offset + int(flat_idx.size)
        take = keep_global[(keep_global >= offset) & (keep_global < next_offset)] - offset
        offset = next_offset
        if take.size == 0:
            continue
        selected = flat_idx[take]
        ref_parts.append(ref_map.reshape(-1, 3)[selected])
        cur_parts.append(cur_map.reshape(-1, 3)[selected])
        matched_stems.append(stem)

    if not ref_parts:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            [],
        )

    ref_corr = np.concatenate(ref_parts, axis=0)
    cur_corr = np.concatenate(cur_parts, axis=0)
    return ref_corr.astype(np.float32), cur_corr.astype(np.float32), matched_stems


def estimate_slam_overlap_point_alignment(
    mode: str,
    chunk_id: int,
    overlap_stems: Sequence[str],
    reference_maps_by_stem: Dict[str, np.ndarray],
    reference_masks_by_stem: Dict[str, np.ndarray],
    reference_cams_by_stem: Dict[str, np.ndarray],
    current_maps_by_stem: Dict[str, np.ndarray],
    current_masks_by_stem: Dict[str, np.ndarray],
    current_cams_by_stem: Dict[str, np.ndarray],
    min_corr: int,
    max_corr: int,
    seed: int,
) -> Dict[str, object]:
    if mode == "none":
        return identity_slam_overlap_alignment_meta(
            mode,
            chunk_id,
            "SLAM overlap point alignment disabled",
        valid=True,
    )
    if chunk_id == 0 or not overlap_stems:
        return identity_slam_overlap_alignment_meta(
            mode,
            chunk_id,
            "no previous connected overlap chunk; using first-stage chunk alignment only",
            valid=True,
    )
    if mode != "sim3":
        raise ValueError(f"Unknown SLAM overlap point alignment mode: {mode}")

    ref_corr, cur_corr, matched_stems = sample_overlap_pointmap_correspondences(
        overlap_stems=overlap_stems,
        reference_maps_by_stem=reference_maps_by_stem,
        reference_masks_by_stem=reference_masks_by_stem,
        current_maps_by_stem=current_maps_by_stem,
        current_masks_by_stem=current_masks_by_stem,
        max_corr=max_corr,
        seed=seed,
    )
    if ref_corr.shape[0] < int(min_corr):
        return estimate_slam_overlap_pose_alignment(
            mode,
            chunk_id,
            overlap_stems=overlap_stems,
            reference_cams_by_stem=reference_cams_by_stem,
            current_cams_by_stem=current_cams_by_stem,
            prefix_note=f"not enough overlap point correspondences: {ref_corr.shape[0]} < {int(min_corr)}",
        )

    scale, R, t, valid, note = estimate_similarity_umeyama(
        src=cur_corr,
        dst=ref_corr,
        estimate_scale=True,
    )
    if not valid:
        return estimate_slam_overlap_pose_alignment(
            mode,
            chunk_id,
            overlap_stems=overlap_stems,
            reference_cams_by_stem=reference_cams_by_stem,
            current_cams_by_stem=current_cams_by_stem,
            prefix_note=f"{note}; point overlap alignment failed",
        )

    transformed = apply_similarity_to_points(cur_corr, scale, R, t)
    residual = np.linalg.norm(transformed.astype(np.float64) - ref_corr.astype(np.float64), axis=1)
    median_residual = float(np.median(residual)) if residual.size else float("nan")
    return {
        "mode": mode,
        "alignment_type": "sim3_overlap_points",
        "valid": True,
        "source": "overlap_pointmaps",
        "num_corr": int(ref_corr.shape[0]),
        "num_pose_corr": 0,
        "num_inliers": 0,
        "matched_stems": matched_stems,
        "matched_pose_stems": [],
        "scale": float(scale),
        "R": R.astype(np.float32).tolist(),
        "t": t.astype(np.float32).tolist(),
        "median_residual": median_residual,
        "initial_median_residual": median_residual,
        "inlier_median_residual": float("nan"),
        "inlier_threshold": float("nan"),
        "inlier_refined": False,
        "note": "VGGT-SLAM-style sim3_overlap_points estimated from one-to-one 3D pointmap correspondences",
        "chunk_id": int(chunk_id),
    }


def apply_slam_overlap_point_alignment(
    mode: str,
    chunk_id: int,
    overlap_stems: Sequence[str],
    reference_maps_by_stem: Dict[str, np.ndarray],
    reference_masks_by_stem: Dict[str, np.ndarray],
    reference_cams_by_stem: Dict[str, np.ndarray],
    current_maps_by_stem: Dict[str, np.ndarray],
    current_masks_by_stem: Dict[str, np.ndarray],
    current_cams_by_stem: Dict[str, np.ndarray],
    pred_points: np.ndarray,
    pred_colors: np.ndarray,
    pred_maps: Sequence[np.ndarray],
    pred_cams: Sequence[Dict[str, object]],
    min_corr: int,
    max_corr: int,
    seed: int,
    transform_point_maps: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Dict[str, object]], Dict[str, object]]:
    align_meta = estimate_slam_overlap_point_alignment(
        mode=mode,
        chunk_id=chunk_id,
        overlap_stems=overlap_stems,
        reference_maps_by_stem=reference_maps_by_stem,
        reference_masks_by_stem=reference_masks_by_stem,
        reference_cams_by_stem=reference_cams_by_stem,
        current_maps_by_stem=current_maps_by_stem,
        current_masks_by_stem=current_masks_by_stem,
        current_cams_by_stem=current_cams_by_stem,
        min_corr=min_corr,
        max_corr=max_corr,
        seed=seed,
    )
    scale = float(align_meta["scale"])
    R = np.asarray(align_meta["R"], dtype=np.float64)
    t = np.asarray(align_meta["t"], dtype=np.float64)
    pred_points_aligned = apply_similarity_to_points(pred_points, scale, R, t)
    pred_maps_aligned = apply_similarity_to_point_maps(pred_maps, scale, R, t) if transform_point_maps else list(pred_maps)
    pred_cams_aligned = apply_similarity_to_cameras(pred_cams, scale, R, t)
    return pred_points_aligned, pred_colors, pred_maps_aligned, pred_cams_aligned, align_meta


def maps_by_stem(stems: Sequence[str], maps: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    return {str(stem): np.asarray(point_map, dtype=np.float32) for stem, point_map in zip(stems, maps)}


def masks_by_stem(stems: Sequence[str], masks: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    return {str(stem): np.asarray(mask, dtype=bool) for stem, mask in zip(stems, masks)}


def cams_by_stem(cams: Sequence[Dict[str, object]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for cam in cams:
        stem = cam.get("stem")
        T = cam.get("T_c2w")
        if stem is None or T is None:
            continue
        out[str(stem)] = np.asarray(T, dtype=np.float32)
    return out


def filter_dict_by_stems(values_by_stem: Dict[str, np.ndarray], stems: Sequence[str]) -> Dict[str, np.ndarray]:
    return {str(stem): values_by_stem[str(stem)] for stem in stems if str(stem) in values_by_stem}


def transformed_map_cache_by_stem(
    maps_by_stem_: Dict[str, np.ndarray],
    stems: Sequence[str],
    align_meta: Dict[str, object],
) -> Dict[str, np.ndarray]:
    scale = float(align_meta["scale"])
    R = np.asarray(align_meta["R"], dtype=np.float64)
    t = np.asarray(align_meta["t"], dtype=np.float64)
    out: Dict[str, np.ndarray] = {}
    for stem in stems:
        stem = str(stem)
        point_map = maps_by_stem_.get(stem)
        if point_map is None:
            continue
        shape = point_map.shape
        out[stem] = apply_similarity_to_points(np.asarray(point_map, dtype=np.float32).reshape(-1, 3), scale, R, t).reshape(shape)
    return out


def alignment_note(mode: str) -> str:
    if mode == "scale":
        return (
            "scale means each chunk is independently aligned to GT with scale+translation only; "
            "no rotation or chunk-to-chunk alignment is applied"
        )
    if mode == "pose_scale":
        return (
            "pose_scale means every chunk is scaled to its input pose translations using matched predicted "
            "pose translations only, with no rotation or translation applied"
        )
    if mode == "pose_scale_yaw_translation":
        return (
            "pose_scale_yaw_translation means every chunk is aligned to its input pose translations using matched "
            "predicted pose translations by estimating scale, then yaw rotation in the XY plane around the Z axis "
            "and translation"
        )
    if mode == "pose_sim3":
        return (
            "pose_sim3 means every chunk is aligned to its input pose translations using matched predicted "
            "pose translations by estimating a full 3D similarity transform: scale, rotation, and translation"
        )
    return "none means raw chunk predictions are logged without alignment"


def log_world_axes_marker_no_labels(
    points_for_bbox: np.ndarray,
    origin_mode: str,
    axis_size: float,
    axis_size_ratio: float,
    min_axis_size: float,
    up_axis: str,
    up_offset_ratio: float,
    radius: float,
) -> None:
    pts = np.asarray(points_for_bbox, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        return

    pts = pts[finite]
    bbox_min = pts.min(axis=0).astype(np.float64)
    bbox_max = pts.max(axis=0).astype(np.float64)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    diag = float(np.linalg.norm(bbox_max - bbox_min))

    if axis_size > 0:
        size = float(axis_size)
    else:
        size = max(diag * float(axis_size_ratio), float(min_axis_size), 1e-6)

    if origin_mode == "zero":
        origin = np.zeros(3, dtype=np.float64)
    elif origin_mode == "scene_center":
        origin = bbox_center + base.parse_signed_axis(up_axis) * size * float(up_offset_ratio)
    else:
        raise ValueError("world_axes_origin must be 'zero' or 'scene_center'")

    x_end = origin + np.array([size, 0.0, 0.0], dtype=np.float64)
    y_end = origin + np.array([0.0, size, 0.0], dtype=np.float64)
    z_end = origin + np.array([0.0, 0.0, size], dtype=np.float64)
    strips = [
        np.stack([origin, x_end]).astype(np.float32),
        np.stack([origin, y_end]).astype(np.float32),
        np.stack([origin, z_end]).astype(np.float32),
    ]
    colors = [
        np.array([255, 0, 0], dtype=np.uint8),
        np.array([0, 220, 0], dtype=np.uint8),
        np.array([40, 80, 255], dtype=np.uint8),
    ]
    kwargs = {"strips": strips, "colors": colors}
    if radius > 0:
        kwargs["radii"] = float(radius)
    base.rr.log("world/world_axes", base.rr.LineStrips3D(**kwargs))


def log_chunk_prediction(
    chunk_entity: str,
    chunk_id: int,
    chunk_stems: Sequence[str],
    chunk_rgbs: Sequence[np.ndarray],
    chunk_gt_cams: Sequence[Dict[str, object]],
    pred_points: np.ndarray,
    pred_colors: np.ndarray,
    pred_cams: Sequence[Dict[str, object]],
    raw_pred_points: np.ndarray,
    raw_pred_colors: np.ndarray,
    raw_pred_cams: Sequence[Dict[str, object]],
    camera_axis_size: float,
    chunk_args: argparse.Namespace,
    args: argparse.Namespace,
) -> Tuple[int, int]:
    runtime_align = str(getattr(chunk_args, "runtime_align", args.align)).lower()
    pred_points, pred_colors = sample_chunk_points(
        pred_points,
        pred_colors,
        max_points=int(chunk_args.max_pred_points_per_chunk),
        seed=int(args.seed) + 1009 * (chunk_id + 1),
    )

    pred_root = "pred_aligned" if runtime_align != "none" else "pred"
    base.log_points(f"{chunk_entity}/{pred_root}/points", pred_points, pred_colors, args.point_radius)

    if args.log_raw_when_aligned and runtime_align != "none":
        raw_points, raw_colors = sample_chunk_points(
            raw_pred_points,
            raw_pred_colors,
            max_points=int(chunk_args.max_pred_points_per_chunk),
            seed=int(args.seed) + 2003 * (chunk_id + 1),
        )
        base.log_points(f"{chunk_entity}/pred_raw/points", raw_points, raw_colors, args.point_radius)

    chunk_name = f"chunk_{int(chunk_id):03d}"
    pred_axis_colors = ((255, 0, 255), (255, 180, 0), (0, 220, 255))
    base.log_camera_axes(
        f"world/cameras/{pred_root}/by_chunk/{chunk_name}/axes",
        pred_cams,
        camera_axis_size,
        args.camera_axis_radius,
        pred_axis_colors,
    )

    if args.log_raw_when_aligned and runtime_align != "none":
        raw_axis_colors = ((150, 150, 150), (180, 180, 180), (210, 210, 210))
        base.log_camera_axes(
            f"world/cameras/pred_raw/by_chunk/{chunk_name}/axes",
            raw_pred_cams,
            camera_axis_size,
            args.camera_axis_radius,
            raw_axis_colors,
        )

    if args.log_images:
        for local_i, (rgb, stem) in enumerate(zip(chunk_rgbs, chunk_stems)):
            safe_stem = base.sanitize_name(stem)
            base.rr.log(f"{chunk_entity}/inputs/view_{local_i:03d}_{safe_stem}/rgb", base.rr.Image(rgb))

    return int(pred_points.shape[0]), int(len(pred_cams))


def save_chunked_rrd(
    args: argparse.Namespace,
    chunk_args: argparse.Namespace,
    interval: int,
    meta: Dict[str, object],
    chunk_records: Sequence[Dict[str, object]],
    chunk_order_indices: Sequence[int],
) -> None:
    output_rrd = Path(args.output_rrd).expanduser().resolve()
    output_rrd.parent.mkdir(parents=True, exist_ok=True)

    scene_name = base.sanitize_name(Path(args.scene_dir).resolve().name)
    recording_id = f"slam_chunks_{scene_name}_{base.sanitize_name(args.model)}"
    base.rr_init_save_compat("predict_scene_to_rrd_slam_chunks", recording_id, output_rrd)
    base.rr_set_time_compat("frame", 0)
    base.log_view_coordinates(args.view_coordinates)
    base.send_blueprint(background=tuple(args.background), hide_grid=args.hide_grid)

    gt_points, gt_colors = base.sample_points_and_colors(
        meta["gt_points"],
        meta["gt_colors"],
        args.max_gt_points,
        args.seed,
    )
    base.log_points("world/gt/points", gt_points, gt_colors, args.point_radius)
    all_gt_cams = gt_cameras_for_stems(meta, meta["stems"])
    all_pred_points_for_axes = [
        np.asarray(record["aligned_pred_points"], dtype=np.float32).reshape(-1, 3)
        for record in chunk_records
    ]
    axis_size = base.estimate_axis_size(all_pred_points_for_axes, args.camera_axis_size)
    gt_axis_colors = ((255, 0, 0), (0, 220, 0), (40, 80, 255))
    base.log_camera_axes("world/cameras/gt/axes", all_gt_cams, axis_size, args.camera_axis_radius, gt_axis_colors)

    for record in chunk_records:
        chunk_entity = f"world/chunks/chunk_{int(record['chunk_id']):03d}"
        base.rr_set_time_compat("chunk", int(record["chunk_id"]))
        pred_points_logged, pred_cams_logged = log_chunk_prediction(
            chunk_entity=chunk_entity,
            chunk_id=int(record["chunk_id"]),
            chunk_stems=record["stems"],
            chunk_rgbs=record["rgbs"],
            chunk_gt_cams=record["gt_cams"],
            pred_points=record["aligned_pred_points"],
            pred_colors=record["aligned_pred_colors"],
            pred_cams=record["aligned_pred_cams"],
            raw_pred_points=record["raw_pred_points"],
            raw_pred_colors=record["raw_pred_colors"],
            raw_pred_cams=record["raw_pred_cams"],
            camera_axis_size=axis_size,
            chunk_args=chunk_args,
            args=args,
        )
        record["num_pred_points_logged"] = pred_points_logged
        record["num_pred_cameras_logged"] = pred_cams_logged

    if args.show_world_axes:
        bbox_points = gt_points
        if bbox_points.shape[0] == 0 and all_pred_points_for_axes:
            bbox_points = np.concatenate(all_pred_points_for_axes, axis=0)
        log_world_axes_marker_no_labels(
            bbox_points,
            origin_mode=args.world_axes_origin,
            axis_size=args.world_axis_size,
            axis_size_ratio=args.world_axis_size_ratio,
            min_axis_size=args.world_axis_min_size,
            up_axis=args.world_up_axis,
            up_offset_ratio=args.world_axis_up_offset_ratio,
            radius=args.world_axis_radius,
        )

    base.rr_disconnect_compat()
    print(f"Saved Rerun recording: {output_rrd}")

    sidecar = output_rrd.with_suffix(".json")
    payload = {
        "scene_dir": str(Path(args.scene_dir).resolve()),
        "model": args.model,
        "checkpoint": args.checkpoint,
        "output_rrd": str(output_rrd),
        "stems": list(meta["stems"]),
        "target_size": {"height": int(meta["target_h"]), "width": int(meta["target_w"])},
        "chunking": {
            "chunk_mode": str(chunk_args.chunk_mode),
            "chunk_order_indices": [int(i) for i in chunk_order_indices],
            "chunk_order_stems": [str(meta["stems"][int(i)]) for i in chunk_order_indices],
            "pose_grid_axes": str(getattr(chunk_args, "pose_grid_axes", "")),
            "pose_grid_size": float(getattr(chunk_args, "pose_grid_size", 0.0)),
            "pose_grid_effective_size": float(getattr(chunk_args, "pose_grid_effective_size", 0.0)),
            "pose_grid_origin": getattr(chunk_args, "pose_grid_origin", None),
            "pose_grid_neighbor_radius": int(getattr(chunk_args, "pose_grid_neighbor_radius", 0)),
            "pose_grid_num_occupied_cells": int(getattr(chunk_args, "pose_grid_num_occupied_cells", 0)),
            "chunk_size": int(chunk_args.chunk_size),
            "chunk_interval": int(interval),
            "chunk_overlap_requested": int(chunk_args.chunk_overlap),
            "min_chunk_size": int(chunk_args.min_chunk_size),
            "max_chunks": int(chunk_args.max_chunks),
            "num_chunks": int(len(chunk_records)),
            "alignment": str(getattr(chunk_args, "runtime_align", args.align)),
            "alignment_cli": str(chunk_args.align),
            "base_args_alignment": str(args.align),
            "alignment_note": alignment_note(str(getattr(chunk_args, "runtime_align", args.align)).lower()),
            "slam_overlap_point_align": str(chunk_args.slam_overlap_point_align),
            "slam_overlap_align_min_corr": int(chunk_args.slam_overlap_align_min_corr),
            "slam_overlap_align_max_corr": int(chunk_args.slam_overlap_align_max_corr),
        },
        "prior_control": {
            "ray_prior_cli": chunk_args.use_ray_prior,
            "depth_prior_cli": chunk_args.use_depth_prior,
            "world_rotation_prior_cli": chunk_args.use_world_rotation_prior,
            "world_translation_prior_cli": chunk_args.use_world_translation_prior,
            "fov_prior_cli": chunk_args.use_fov_prior,
            "bootstrap_ray_prior_from_chunk0": bool(getattr(chunk_args, "bootstrap_ray_prior_from_chunk0", False)),
            "bootstrap_depth_prior_from_overlap": bool(getattr(chunk_args, "bootstrap_depth_prior_from_overlap", False)),
            "bootstrap_rotation_prior_from_overlap": bool(getattr(chunk_args, "bootstrap_rotation_prior_from_overlap", False)),
            "bootstrap_intrinsics": getattr(chunk_args, "bootstrap_intrinsics", None),
        },
        "num_gt_points_logged": int(gt_points.shape[0]),
        "num_gt_cameras": int(len(all_gt_cams)),
        "chunks": [
            {
                "chunk_id": int(record["chunk_id"]),
                "start": int(record["start"]),
                "end": int(record["end"]),
                "indices": [int(i) for i in record["indices"]],
                "stems": list(record["stems"]),
                "overlap_indices": [int(i) for i in record["overlap_indices"]],
                "overlap_stems": list(record["overlap_stems"]),
                "num_pred_points_raw": int(record["num_pred_points_raw"]),
                "num_pred_points_logged": int(record.get("num_pred_points_logged", 0)),
                "num_pred_cameras_raw": int(record["num_pred_cameras_raw"]),
                "num_pred_cameras_logged": int(record.get("num_pred_cameras_logged", 0)),
                "overlap_depth_priors_used": int(record.get("overlap_depth_priors_used", 0)),
                "overlap_rotation_priors_used": int(record.get("overlap_rotation_priors_used", 0)),
                "alignment": record["align_meta"],
                "slam_overlap_alignment": record.get("slam_overlap_align_meta", {}),
            }
            for record in chunk_records
        ],
    }
    sidecar.write_text(json.dumps(base.json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved sidecar metadata: {sidecar}")


@torch.no_grad()
def main() -> None:
    args, chunk_args, interval = parse_args()
    device = base.resolve_device(args.device)

    views, meta = base.build_views_from_scene(args, device=device)
    chunks, chunk_order_indices = build_chunk_slices(meta, chunk_args, interval)
    if not chunks:
        raise RuntimeError("No chunks generated. Check --chunk_size/--min_chunk_size/--num_views.")

    interval_text = f"{interval}" if str(chunk_args.chunk_mode) == "sequence" else "n/a"
    print(
        f"Running chunked SLAM smoke test: frames={len(views)}, chunks={len(chunks)}, "
        f"chunk_mode={chunk_args.chunk_mode}, chunk_size={chunk_args.chunk_size}, "
        f"interval={interval_text}, requested_overlap={chunk_args.chunk_overlap}"
    )
    if str(chunk_args.chunk_mode) == "pose_space":
        ordered_stems = [str(meta["stems"][i]) for i in chunk_order_indices]
        print(
            "[INFO] Pose-space grid chunking: "
            f"axes={chunk_args.pose_grid_axes}, "
            f"cell_size={float(getattr(chunk_args, 'pose_grid_effective_size', 0.0)):.6g}, "
            f"occupied_cells={int(getattr(chunk_args, 'pose_grid_num_occupied_cells', 0))}, "
            f"overlap_images_per_cell={int(chunk_args.chunk_overlap)}, "
            "core_order="
            + ", ".join(ordered_stems[:12])
            + ("" if len(ordered_stems) <= 12 else f", ... ({len(ordered_stems)} total)")
        )
    if len(chunks) > 1 and len(chunks[-1][2]) < int(chunk_args.chunk_size):
        tail_len = len(chunks[-1][2])
        if tail_len < max(3, int(chunk_args.chunk_size) // 2):
            print(
                f"[WARN] Final chunk is short: n={tail_len} < chunk_size={chunk_args.chunk_size}. "
                "This can make overlap point alignment unstable. Use --min_chunk_size to drop short tails, "
                "--max_chunks to cap chunks, or choose chunk_size/overlap so the last chunk is not tiny."
            )

    model, _ = base.init_model_from_hydra(
        model_name=args.model,
        machine=args.machine,
        hydra_overrides=args.hydra_override,
        device=device,
    )
    base.load_optional_checkpoint(model, args.checkpoint)
    model.eval()

    model_name = str(args.model)
    bootstrap_ray_prior = model_name in PI3X_PRIOR_MODELS and chunk_args.use_ray_prior is False
    bootstrap_depth_prior = model_name in PI3X_PRIOR_MODELS and chunk_args.use_depth_prior is False
    bootstrap_rotation_prior = (
        model_name in WORLD_ROTATION_PRIOR_MODELS and chunk_args.use_world_rotation_prior is False
    )
    bootstrapped_intrinsics: Optional[torch.Tensor] = None
    previous_depth_priors: Dict[int, torch.Tensor] = {}
    previous_rotation_priors: Dict[int, torch.Tensor] = {}
    if bootstrap_ray_prior:
        chunk_args.bootstrap_ray_prior_from_chunk0 = True
        chunk_args.bootstrap_intrinsics = None
        set_pi3x_ray_prior_prob(model, enabled=False)
        print(
            "[INFO] --no_ray_prior selected: chunk 0 runs without ray/intrinsics prior; "
            "later chunks will use average intrinsics recovered from chunk 0 predicted rays."
        )
    if bootstrap_depth_prior:
        chunk_args.bootstrap_depth_prior_from_overlap = True
        set_model_task_prob(model, "depth_prob", enabled=False)
        set_model_task_value(model, "sparse_depth_prob", 0.0)
        print(
            "[INFO] --no_depth_prior selected: external depth prior is disabled; "
            "later chunks will use predicted depth from overlapping previous-chunk frames."
        )
    if bootstrap_rotation_prior:
        chunk_args.bootstrap_rotation_prior_from_overlap = True
        set_model_task_prob(model, "world_rotation_prob", enabled=False)
        print(
            "[INFO] --no_world_rotation_prior selected: external rotation prior is disabled; "
            "later chunks will use predicted rotation from overlapping previous-chunk frames."
        )

    align_mode = str(getattr(chunk_args, "runtime_align", args.align)).lower()
    slam_overlap_point_align_mode = str(chunk_args.slam_overlap_point_align).lower()
    input_pose_cams_by_stem = camera_centers_by_stem(gt_cameras_for_stems(meta, meta["stems"]))
    if align_mode in POSE_TRANSLATION_ALIGN_MODES and not input_pose_cams_by_stem:
        print(
            "[WARN] --align pose_* selected, but no input pose translations were found in scene metadata; "
            "pose alignment will fall back to raw chunk coordinates."
        )
    if slam_overlap_point_align_mode != "none":
        print(
            "[INFO] SLAM overlap point alignment enabled: "
            f"mode={slam_overlap_point_align_mode}, "
            f"min_corr={chunk_args.slam_overlap_align_min_corr}, "
            f"max_corr={chunk_args.slam_overlap_align_max_corr}"
        )

    previous_final_maps_by_stem: Dict[str, np.ndarray] = {}
    previous_final_masks_by_stem: Dict[str, np.ndarray] = {}
    previous_final_cams_by_stem: Dict[str, np.ndarray] = {}
    chunk_records: List[Dict[str, object]] = []
    for chunk_id, (start, end, indices, overlap_indices) in enumerate(chunks):
        chunk_views = [views[i] for i in indices]
        chunk_stems = [meta["stems"][i] for i in indices]
        chunk_rgbs = [meta["rgbs"][i] for i in indices]
        overlap_stems = [meta["stems"][i] for i in overlap_indices]
        next_overlap_indices = chunks[chunk_id + 1][3] if chunk_id + 1 < len(chunks) else []
        next_overlap_stems = [meta["stems"][i] for i in next_overlap_indices]
        if bootstrap_ray_prior and chunk_id > 0:
            if bootstrapped_intrinsics is None:
                set_pi3x_ray_prior_prob(model, enabled=False)
                print(
                    f"[chunk {chunk_id:03d}] bootstrap ray prior unavailable; "
                    "continuing without ray/intrinsics prior."
                )
            else:
                set_pi3x_ray_prior_prob(model, enabled=True)
                chunk_views = apply_bootstrap_intrinsics_to_views(
                    chunk_views,
                    intrinsics=bootstrapped_intrinsics,
                    device=device,
                )
        overlap_depth_priors_used = 0
        if bootstrap_depth_prior:
            chunk_views, overlap_depth_priors_used = apply_overlap_depth_priors_to_views(
                chunk_views,
                indices=indices,
                overlap_indices=overlap_indices,
                depth_cache=previous_depth_priors,
                device=device,
            )
            set_model_task_prob(model, "depth_prob", enabled=overlap_depth_priors_used > 0)
        overlap_rotation_priors_used = 0
        if bootstrap_rotation_prior:
            chunk_views, overlap_rotation_priors_used = apply_overlap_rotation_priors_to_views(
                chunk_views,
                indices=indices,
                overlap_indices=overlap_indices,
                rotation_cache=previous_rotation_priors,
                device=device,
            )
            set_model_task_prob(model, "world_rotation_prob", enabled=overlap_rotation_priors_used > 0)
        print(
            f"[chunk {chunk_id:03d}] window={start}:{end}, n={len(indices)}, "
            f"actual_overlap={len(overlap_indices)}, "
            f"overlap_depth_priors={overlap_depth_priors_used}, "
            f"overlap_rotation_priors={overlap_rotation_priors_used}, "
            f"stems={chunk_stems[0]}..{chunk_stems[-1]}"
        )

        preds = model(chunk_views)
        if bootstrap_ray_prior and chunk_id == 0:
            bootstrapped_intrinsics = recover_average_intrinsics_from_pred_rays(preds)
            if bootstrapped_intrinsics is None:
                print("[WARN] Failed to recover bootstrap intrinsics from chunk 0 predicted rays.")
            else:
                K_np = bootstrapped_intrinsics.detach().cpu().numpy()
                chunk_args.bootstrap_intrinsics = {
                    "fx": float(K_np[0, 0]),
                    "fy": float(K_np[1, 1]),
                    "cx": float(K_np[0, 2]),
                    "cy": float(K_np[1, 2]),
                    "matrix": K_np.tolist(),
                }
                print(
                    "[INFO] Recovered bootstrap intrinsics from chunk 0 predicted rays: "
                    f"fx={K_np[0, 0]:.3f}, fy={K_np[1, 1]:.3f}, "
                    f"cx={K_np[0, 2]:.3f}, cy={K_np[1, 2]:.3f}"
                )
        if bootstrap_depth_prior or bootstrap_rotation_prior:
            new_depth_priors, new_rotation_priors = build_prediction_prior_cache(preds, indices)
            if bootstrap_depth_prior:
                previous_depth_priors = new_depth_priors
            if bootstrap_rotation_prior:
                previous_rotation_priors = new_rotation_priors
        raw_pred_points, raw_pred_colors, pred_maps, pred_valid_masks, raw_pred_cams = base.collect_pred_outputs(
            preds=preds,
            rgbs=chunk_rgbs,
            args=args,
            stems=chunk_stems,
        )
        chunk_meta = make_chunk_meta(meta, indices)
        if align_mode in POSE_TRANSLATION_ALIGN_MODES:
            aligned_pred_points, aligned_pred_colors, _aligned_pred_maps, aligned_pred_cams, align_meta = (
                apply_chunk_pose_alignment(
                    mode=align_mode,
                    chunk_id=chunk_id,
                    reference_cams_by_stem=input_pose_cams_by_stem,
                    raw_pred_points=raw_pred_points,
                    raw_pred_colors=raw_pred_colors,
                    pred_maps=pred_maps,
                    raw_pred_cams=raw_pred_cams,
                    target_stems=chunk_stems,
                    seed=int(args.seed) + 70001 + int(chunk_id),
                )
            )
            print(
                f"[chunk {chunk_id:03d}] pose-translation alignment: mode={align_meta['mode']}, "
                f"valid={align_meta['valid']}, num_corr={align_meta['num_corr']}, "
                f"scale={float(align_meta['scale']):.6g}, "
                f"yaw={float(align_meta.get('yaw_degrees', 0.0)):.3f}deg, "
                f"median_residual={float(align_meta.get('median_residual', float('nan'))):.6g}"
            )
        else:
            alignment_args = argparse.Namespace(**vars(args))
            alignment_args.align = align_mode
            aligned_pred_points, aligned_pred_colors, _aligned_pred_maps, aligned_pred_cams, align_meta = (
                base.estimate_and_apply_alignment(
                    args=alignment_args,
                    meta=chunk_meta,
                    pred_points=raw_pred_points,
                    pred_colors=raw_pred_colors,
                    pred_maps=pred_maps,
                    pred_valid_masks=pred_valid_masks,
                    pred_cams=raw_pred_cams,
                )
            )

        if slam_overlap_point_align_mode == "none":
            slam_overlap_align_meta = identity_slam_overlap_alignment_meta(
                "none",
                chunk_id,
                "SLAM overlap point alignment disabled",
                valid=True,
            )
        else:
            current_maps_by_stem = maps_by_stem(chunk_stems, _aligned_pred_maps)
            current_masks_by_stem = masks_by_stem(chunk_stems, pred_valid_masks)
            current_cams_by_stem = cams_by_stem(aligned_pred_cams)
            aligned_pred_points, aligned_pred_colors = sample_chunk_points(
                aligned_pred_points,
                aligned_pred_colors,
                max_points=int(chunk_args.max_pred_points_per_chunk),
                seed=int(args.seed) + 3001 * (chunk_id + 1),
            )
            aligned_pred_points, aligned_pred_colors, _aligned_pred_maps, aligned_pred_cams, slam_overlap_align_meta = (
                apply_slam_overlap_point_alignment(
                    mode=slam_overlap_point_align_mode,
                    chunk_id=chunk_id,
                    overlap_stems=overlap_stems,
                    reference_maps_by_stem=previous_final_maps_by_stem,
                    reference_masks_by_stem=previous_final_masks_by_stem,
                    reference_cams_by_stem=previous_final_cams_by_stem,
                    current_maps_by_stem=current_maps_by_stem,
                    current_masks_by_stem=current_masks_by_stem,
                    current_cams_by_stem=current_cams_by_stem,
                    pred_points=aligned_pred_points,
                    pred_colors=aligned_pred_colors,
                    pred_maps=_aligned_pred_maps,
                    pred_cams=aligned_pred_cams,
                    min_corr=int(chunk_args.slam_overlap_align_min_corr),
                    max_corr=int(chunk_args.slam_overlap_align_max_corr),
                    seed=int(args.seed) + 90001 + int(chunk_id),
                    transform_point_maps=False,
                )
            )
            print(
                f"[chunk {chunk_id:03d}] SLAM overlap point alignment: "
                f"mode={slam_overlap_align_meta['mode']}, "
                f"type={slam_overlap_align_meta.get('alignment_type', 'unknown')}, "
                f"valid={slam_overlap_align_meta['valid']}, "
                f"num_corr={slam_overlap_align_meta['num_corr']}, "
                f"num_pose_corr={slam_overlap_align_meta.get('num_pose_corr', 0)}, "
                f"scale={float(slam_overlap_align_meta['scale']):.6g}, "
                f"median_residual={float(slam_overlap_align_meta.get('median_residual', float('nan'))):.6g}, "
                f"stems={slam_overlap_align_meta.get('matched_stems', [])}"
            )
            previous_final_maps_by_stem = transformed_map_cache_by_stem(
                current_maps_by_stem,
                next_overlap_stems,
                slam_overlap_align_meta,
            )
            previous_final_masks_by_stem = filter_dict_by_stems(current_masks_by_stem, next_overlap_stems)
            previous_final_cams_by_stem = filter_dict_by_stems(cams_by_stem(aligned_pred_cams), next_overlap_stems)

        chunk_records.append(
            {
                "chunk_id": int(chunk_id),
                "start": int(start),
                "end": int(end),
                "indices": list(indices),
                "stems": chunk_stems,
                "overlap_indices": list(overlap_indices),
                "overlap_stems": overlap_stems,
                "rgbs": chunk_rgbs,
                "gt_cams": gt_cameras_for_stems(meta, chunk_stems),
                "raw_pred_points": raw_pred_points,
                "raw_pred_colors": raw_pred_colors,
                "raw_pred_cams": raw_pred_cams,
                "aligned_pred_points": aligned_pred_points,
                "aligned_pred_colors": aligned_pred_colors,
                "aligned_pred_cams": aligned_pred_cams,
                "align_meta": align_meta,
                "slam_overlap_align_meta": slam_overlap_align_meta,
                "num_pred_points_raw": int(raw_pred_points.shape[0]),
                "num_pred_cameras_raw": int(len(raw_pred_cams)),
                "overlap_depth_priors_used": int(overlap_depth_priors_used),
                "overlap_rotation_priors_used": int(overlap_rotation_priors_used),
            }
        )
        print(
            f"[chunk {chunk_id:03d}] raw prediction: points={raw_pred_points.shape[0]}, "
            f"cameras={len(raw_pred_cams)}, align={align_meta['mode']}, valid={align_meta['valid']}"
        )

    save_chunked_rrd(
        args=args,
        chunk_args=chunk_args,
        interval=interval,
        meta=meta,
        chunk_records=chunk_records,
        chunk_order_indices=chunk_order_indices,
    )


if __name__ == "__main__":
    main()
