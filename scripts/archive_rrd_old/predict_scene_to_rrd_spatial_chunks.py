#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run footprint/pose-space spatial chunk reconstruction and save Rerun .rrd files.

This script is for spatial chunk reconstruction, not SLAM. It partitions input
views into connected spatial chunks, runs each chunk independently, and reuses
predicted depth from shared seam images as priors for later adjacent chunks. No
overlap Sim3 chaining or SLAM pose propagation is performed.

Example:
    python scripts/predict_scene_to_rrd_spatial_chunks.py \
      --scene_dir /path/to/scene \
      --model geoff3d \
      --checkpoint /path/to/checkpoint.pth \
      --output_rrd outputs/scene_spatial_chunks.rrd \
      --num_views 0 \
      --max_side 518 \
      --pose_grid_axes xy \
      --max_chunk_size 12 \
      --align none
    
    python scripts/predict_scene_to_rrd_spatial_chunks.py \
      --model geoff3d \
      --scene_dir /opt/data/private/dataset/data/UAVFF3D-Real/xiaoxiang_ndir2 \
      --checkpoint experiments/dom/uav_training/pi3x_zup_translation_yaw_aug_8v_6d_16ipg_2g/checkpoint-last.pth \
      --output_rrd outputs/pi3x_zup_translation_yaw_aug_8v_6d_16ipg_2g/xiaoxiang_ndir2-translation-rotation-scene_spatial_chunks.rrd \
      --num_views 0 \
      --max_side 518 \
      --pose_grid_axes xy \
      --max_chunk_size 50 \
      --align pose_scale_yaw_translation \
      --slam_overlap_point_align none \
      --use_world_translation_prior \
      --use_world_rotation_prior  \
      --no_ray_prior \
      --no_depth_prior


    python scripts/predict_scene_to_rrd_spatial_chunks.py \
      --model pi3x \
      --scene_dir /opt/data/private/dataset/data/usegeo/dataset1 \
      --checkpoint experiments/mapanything/uav_training/pi3_finetuning_16v_6d_16ipg_2g/checkpoint-best.pth \
      --output_rrd outputs/pi3x/dataset1-translation-rotation-scene_spatial_chunks.rrd \
      --num_views 0 \
      --max_side 518 \
      --pose_grid_axes xy \
      --max_chunk_size 50 \
      --align pose_sim3 \
      --slam_overlap_point_align none \
      --use_world_translation_prior \
      --use_world_rotation_prior  \
      --no_ray_prior \
      --no_depth_prior

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

import predict_scene_to_rrd as base
import predict_scene_to_rrd_slam_chunks as slam_base


VGGT_OMEGA_MODELS = {"vggt_omega"}
PI3X_PRIOR_MODELS = {"geoff3d"}
POSE_TRANSLATION_ALIGN_MODES = {"pose_scale", "pose_scale_yaw_translation", "pose_sim3"}
SPATIAL_PARTITIONS = {"footprint_tree", "pose_grid"}
FOOTPRINT_SOURCES = {"auto", "depth", "lookat", "center"}


def has_cli_option(argv: Sequence[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in argv)


def parse_spatial_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--align",
        dest="align",
        default=None,
        help=(
            "Alignment mode. Supports none, scale, pose_scale, "
            "pose_scale_yaw_translation, and pose_sim3."
        ),
    )
    parser.add_argument(
        "--pose_grid_size",
        type=float,
        default=0.0,
        help=(
            "Optional manual grid cell side length in pose-coordinate units. "
            "The default <=0 automatically estimates it from --max_chunk_size."
        ),
    )
    parser.add_argument(
        "--spatial_partition",
        "--pose_partition",
        dest="spatial_partition",
        default="footprint_tree",
        help=(
            "Spatial partition method. footprint_tree adaptively partitions observed XY/XZ/YZ footprints "
            "from depth/pose and is the default. pose_grid keeps the old fixed pose-center grid."
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
        help="Neighbor-cell radius used to choose overlap images.",
    )
    parser.add_argument(
        "--footprint_source",
        choices=sorted(FOOTPRINT_SOURCES),
        default="auto",
        help=(
            "Source for footprint_tree. auto uses depth backprojected footprints when available, "
            "then lookat points, then camera centers."
        ),
    )
    parser.add_argument(
        "--footprint_sample_stride",
        type=int,
        default=16,
        help="Pixel stride for sampling depth-derived footprints.",
    )
    parser.add_argument(
        "--footprint_min_points",
        type=int,
        default=32,
        help="Minimum valid sampled depth points required to trust one image footprint.",
    )
    parser.add_argument(
        "--footprint_quantile_min",
        type=float,
        default=0.02,
        help="Lower quantile for robust depth footprint bounding boxes.",
    )
    parser.add_argument(
        "--footprint_quantile_max",
        type=float,
        default=0.98,
        help="Upper quantile for robust depth footprint bounding boxes.",
    )
    parser.add_argument(
        "--footprint_lookat_distance",
        type=float,
        default=0.0,
        help=(
            "Fallback look-at distance for images without depth. <=0 estimates it from the "
            "ground plane or scene/camera extent."
        ),
    )
    parser.add_argument(
        "--footprint_neighbor_margin",
        type=float,
        default=0.0,
        help="Extra pose-coordinate margin used when deciding adjacent footprint-tree leaves.",
    )
    parser.add_argument(
        "--max_chunk_size",
        type=int,
        default=8,
        help="Maximum number of images passed to the model for one spatial chunk.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Deprecated alias for --max_chunk_size.",
    )
    parser.add_argument(
        "--chunk_overlap",
        "--overlap",
        dest="chunk_overlap",
        type=int,
        default=None,
        help=(
            "Deprecated and ignored. Seam overlap is now determined automatically "
            "from connected spatial cells and bounded by --max_chunk_size."
        ),
    )
    parser.add_argument(
        "--min_chunk_size",
        type=int,
        default=1,
        help="Skip grid cells with fewer core images than this value.",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=0,
        help="Optional cap on number of grid chunks; <=0 disables the cap.",
    )
    parser.add_argument(
        "--max_pred_points_per_chunk",
        type=int,
        default=None,
        help="Optional point logging cap per chunk. Default: reuse --max_pred_points.",
    )
    parser.add_argument(
        "--output_chunks_rrd",
        default=None,
        help="Optional output .rrd for per-chunk predictions. Default: <output_rrd stem>_chunks.rrd.",
    )
    parser.add_argument(
        "--slam_overlap_point_align",
        nargs="?",
        const="sim3",
        default=None,
        help="Accepted for CLI compatibility with slam_chunks, but ignored by this spatial script.",
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
        help="For geoff3d, force external depth prior on via model.task.depth_prob=1.",
    )
    parser.add_argument(
        "--no_depth_prior",
        dest="use_depth_prior",
        action="store_false",
        help=(
            "For geoff3d, disable external depth prior; seam depth priors "
            "from previously reconstructed shared images are still fed to later chunks."
        ),
    )
    parser.add_argument(
        "--use_world_rotation_prior",
        dest="use_world_rotation_prior",
        action="store_true",
        default=None,
        help="For world-prior models, force world rotation prior on via model.task.world_rotation_prob=1.",
    )
    parser.add_argument(
        "--no_world_rotation_prior",
        dest="use_world_rotation_prior",
        action="store_false",
        help="For world-prior models, force world rotation prior off via model.task.world_rotation_prob=0.",
    )
    parser.add_argument(
        "--use_world_translation_prior",
        dest="use_world_translation_prior",
        action="store_true",
        default=True,
        help="For world-prior models, force world translation prior on via model.task.world_translation_prob=1.",
    )
    parser.add_argument(
        "--no_world_translation_prior",
        dest="use_world_translation_prior",
        action="store_false",
        help="For world-prior models, force world translation prior off via model.task.world_translation_prob=0.",
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
    return parser.parse_known_args(list(argv))


def validate_spatial_args(spatial_args: argparse.Namespace) -> None:
    spatial_args.align = slam_base.normalize_align_mode(spatial_args.align)
    if spatial_args.chunk_size is not None:
        spatial_args.max_chunk_size = int(spatial_args.chunk_size)
        print("[WARN] --chunk_size is deprecated for spatial chunks; use --max_chunk_size.")
    if spatial_args.chunk_overlap is not None:
        print("[WARN] --chunk_overlap is deprecated for spatial chunks and will be ignored.")
    if spatial_args.slam_overlap_point_align is not None:
        print("[WARN] --slam_overlap_point_align is ignored by spatial chunks.")
    spatial_args.spatial_partition = str(spatial_args.spatial_partition).strip().lower().replace("-", "_")
    if spatial_args.spatial_partition in {"grid", "pose"}:
        spatial_args.spatial_partition = "pose_grid"
    if spatial_args.spatial_partition in {"tree", "footprint", "adaptive_tree", "footprint-tree"}:
        spatial_args.spatial_partition = "footprint_tree"
    if spatial_args.spatial_partition not in SPATIAL_PARTITIONS:
        raise ValueError("--spatial_partition must be one of: footprint_tree, pose_grid")
    spatial_args.footprint_source = str(spatial_args.footprint_source).strip().lower()
    if spatial_args.footprint_source not in FOOTPRINT_SOURCES:
        raise ValueError("--footprint_source must be one of: auto, depth, lookat, center")
    spatial_args.pose_grid_axes = str(spatial_args.pose_grid_axes).strip().lower()
    if spatial_args.pose_grid_axes not in {"xy", "xz", "yz", "xyz"}:
        raise ValueError("--pose_grid_axes must be one of: xy, xz, yz, xyz")
    if spatial_args.pose_grid_neighbor_radius < 0:
        raise ValueError("--pose_grid_neighbor_radius must be non-negative")
    if spatial_args.footprint_sample_stride <= 0:
        raise ValueError("--footprint_sample_stride must be positive")
    if spatial_args.footprint_min_points <= 0:
        raise ValueError("--footprint_min_points must be positive")
    if not 0.0 <= float(spatial_args.footprint_quantile_min) < float(spatial_args.footprint_quantile_max) <= 1.0:
        raise ValueError("--footprint_quantile_min/max must satisfy 0 <= min < max <= 1")
    if spatial_args.footprint_neighbor_margin < 0:
        raise ValueError("--footprint_neighbor_margin must be non-negative")
    if spatial_args.max_chunk_size <= 0:
        raise ValueError("--max_chunk_size must be positive")
    if spatial_args.min_chunk_size <= 0:
        raise ValueError("--min_chunk_size must be positive")
    if spatial_args.max_pred_points_per_chunk is not None and spatial_args.max_pred_points_per_chunk < 0:
        raise ValueError("--max_pred_points_per_chunk must be non-negative")


def parse_args() -> Tuple[argparse.Namespace, argparse.Namespace]:
    spatial_args, remaining_argv = parse_spatial_args(sys.argv[1:])
    validate_spatial_args(spatial_args)

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining_argv]
        args = base.parse_args()
    finally:
        sys.argv = old_argv

    if spatial_args.align is None:
        spatial_args.runtime_align = str(args.align).lower()
    else:
        spatial_args.runtime_align = str(spatial_args.align).lower()
        args.align = str(spatial_args.runtime_align)

    if str(args.model) in VGGT_OMEGA_MODELS and not has_cli_option(remaining_argv, "--size_multiple"):
        args.size_multiple = 16
        print(
            "[INFO] VGGT-Omega model selected; using --size_multiple 16 "
            "because no explicit --size_multiple was provided."
        )

    if spatial_args.max_pred_points_per_chunk is None:
        spatial_args.max_pred_points_per_chunk = int(args.max_pred_points)
    args.output_chunks_rrd = spatial_args.output_chunks_rrd
    slam_base.append_world_prior_overrides(args, spatial_args)
    return args, spatial_args


def pose_centers_from_meta(meta: Dict[str, object]) -> np.ndarray:
    cams = meta.get("cams", {})
    centers: List[np.ndarray] = []
    missing: List[str] = []

    for stem in meta.get("stems", []):
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
            "Spatial chunking requires a finite input camera pose for every selected frame; "
            f"missing/invalid pose for: {shown}{suffix}"
        )
    if not centers:
        raise ValueError("Spatial chunking requires at least one selected frame with an input camera pose.")
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


def auto_core_target_size(max_chunk_size: int) -> int:
    max_chunk_size = max(1, int(max_chunk_size))
    if max_chunk_size <= 2:
        return max_chunk_size
    # Leave room for seam images from connected neighboring chunks.
    return max(1, min(max_chunk_size - 1, int(np.floor(float(max_chunk_size) * 0.7))))


def build_cell_to_core(
    coords: np.ndarray,
    origin: np.ndarray,
    grid_size: float,
) -> Dict[Tuple[int, ...], List[int]]:
    cell_coords = np.floor((coords - origin[None, :]) / float(grid_size)).astype(np.int64)
    cell_to_core: Dict[Tuple[int, ...], List[int]] = {}
    for frame_idx, cell in enumerate(cell_coords):
        key = tuple(int(v) for v in cell)
        cell_to_core.setdefault(key, []).append(int(frame_idx))
    return cell_to_core


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


def are_neighbor_cells(a: Tuple[int, ...], b: Tuple[int, ...], radius: int) -> bool:
    if a == b:
        return False
    return max(abs(int(x) - int(y)) for x, y in zip(a, b)) <= int(radius)


def robust_bbox(points: np.ndarray, qmin: float, qmax: float) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, points.shape[-1])
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] == 0:
        raise ValueError("Cannot build bbox from empty points")
    lo = np.quantile(pts, float(qmin), axis=0)
    hi = np.quantile(pts, float(qmax), axis=0)
    return lo.astype(np.float64), hi.astype(np.float64)


def camera_forward_z(T_c2w: np.ndarray) -> np.ndarray:
    T = np.asarray(T_c2w, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    fwd = T[:3, 2].astype(np.float64)
    n = float(np.linalg.norm(fwd))
    if not np.isfinite(n) or n <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return fwd / n


def estimate_fallback_lookat_distance(centers: np.ndarray, gt_points: np.ndarray) -> float:
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt_points, dtype=np.float64).reshape(-1, 3)
    if gt.shape[0] > 0 and np.isfinite(gt).all(axis=1).any():
        gt = gt[np.isfinite(gt).all(axis=1)]
        scene_center = np.median(gt, axis=0)
        d = np.linalg.norm(centers - scene_center[None, :], axis=1)
        d = d[np.isfinite(d) & (d > 1e-6)]
        if d.size > 0:
            return float(np.median(d))

    extent = centers.max(axis=0) - centers.min(axis=0) if centers.shape[0] > 1 else np.ones(3)
    span = float(np.linalg.norm(extent))
    if np.isfinite(span) and span > 1e-6:
        return span
    return 1.0


def lookat_point_from_pose(
    T_c2w: np.ndarray,
    centers: np.ndarray,
    gt_points: np.ndarray,
    fallback_distance: float,
    requested_distance: float,
) -> np.ndarray:
    T = np.asarray(T_c2w, dtype=np.float64)
    center = T[:3, 3].astype(np.float64)
    forward = camera_forward_z(T)

    gt = np.asarray(gt_points, dtype=np.float64).reshape(-1, 3)
    if gt.shape[0] > 0 and np.isfinite(gt).all(axis=1).any():
        z_ground = float(np.quantile(gt[np.isfinite(gt).all(axis=1), 2], 0.05))
    else:
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        z_ground = float(np.min(centers[:, 2]))

    denom = float(forward[2])
    if abs(denom) > 1e-8:
        t_ground = (z_ground - float(center[2])) / denom
        if np.isfinite(t_ground) and t_ground > 1e-6:
            return center + forward * t_ground

    distance = float(requested_distance) if requested_distance > 0 else float(fallback_distance)
    return center + forward * max(distance, 1e-6)


def point_bbox(point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(point, dtype=np.float64).reshape(-1)
    return p.copy(), p.copy()


def footprint_features_from_meta(
    meta: Dict[str, object],
    spatial_args: argparse.Namespace,
    axis_indices: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    stems = list(meta["stems"])
    pose_centers = pose_centers_from_meta(meta)
    gt_points = np.asarray(meta.get("gt_points", np.empty((0, 3), np.float32)), dtype=np.float64).reshape(-1, 3)
    fallback_distance = estimate_fallback_lookat_distance(pose_centers, gt_points)
    source_requested = str(spatial_args.footprint_source)
    counts = {"depth": 0, "lookat": 0, "center": 0}

    centers: List[np.ndarray] = []
    bbox_mins: List[np.ndarray] = []
    bbox_maxs: List[np.ndarray] = []
    sources: List[str] = []
    stride = int(spatial_args.footprint_sample_stride)

    for i, stem in enumerate(stems):
        source_used: Optional[str] = None
        bbox_min: Optional[np.ndarray] = None
        bbox_max: Optional[np.ndarray] = None
        center_coord: Optional[np.ndarray] = None

        if source_requested in {"auto", "depth"}:
            point_map = np.asarray(meta["gt_maps"][i], dtype=np.float64)
            valid_mask = np.asarray(meta["valid_masks"][i], dtype=bool)
            if point_map.ndim == 3 and point_map.shape[-1] == 3 and valid_mask.shape == point_map.shape[:2]:
                sampled = point_map[::stride, ::stride]
                sampled_valid = valid_mask[::stride, ::stride]
                pts = sampled[sampled_valid & np.isfinite(sampled).all(axis=-1)]
                if pts.shape[0] >= int(spatial_args.footprint_min_points):
                    coords = pts[:, axis_indices]
                    bbox_min, bbox_max = robust_bbox(
                        coords,
                        float(spatial_args.footprint_quantile_min),
                        float(spatial_args.footprint_quantile_max),
                    )
                    center_coord = np.median(coords, axis=0).astype(np.float64)
                    source_used = "depth"

        if source_used is None and source_requested in {"auto", "lookat"}:
            cam = meta.get("cams", {}).get(stem)
            if cam is not None and cam.get("T_c2w") is not None:
                lookat = lookat_point_from_pose(
                    np.asarray(cam["T_c2w"], dtype=np.float64),
                    centers=pose_centers,
                    gt_points=gt_points,
                    fallback_distance=fallback_distance,
                    requested_distance=float(spatial_args.footprint_lookat_distance),
                )
                point = lookat[list(axis_indices)]
                bbox_min, bbox_max = point_bbox(point)
                center_coord = point
                source_used = "lookat"

        if source_used is None:
            point = pose_centers[i, list(axis_indices)]
            bbox_min, bbox_max = point_bbox(point)
            center_coord = point
            source_used = "center"

        centers.append(np.asarray(center_coord, dtype=np.float64))
        bbox_mins.append(np.asarray(bbox_min, dtype=np.float64))
        bbox_maxs.append(np.asarray(bbox_max, dtype=np.float64))
        sources.append(source_used)
        counts[source_used] += 1

    feature_meta = {
        "footprint_source_requested": source_requested,
        "footprint_source_counts": counts,
        "footprint_sources": sources,
        "fallback_lookat_distance": float(fallback_distance),
        "sample_stride": int(stride),
        "min_points": int(spatial_args.footprint_min_points),
        "quantile_min": float(spatial_args.footprint_quantile_min),
        "quantile_max": float(spatial_args.footprint_quantile_max),
    }
    return (
        np.stack(centers, axis=0).astype(np.float64),
        np.stack(bbox_mins, axis=0).astype(np.float64),
        np.stack(bbox_maxs, axis=0).astype(np.float64),
        feature_meta,
    )


def expand_degenerate_region(region_min: np.ndarray, region_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(region_min, dtype=np.float64).copy()
    hi = np.asarray(region_max, dtype=np.float64).copy()
    extent = hi - lo
    scale = max(float(np.max(np.abs(extent))), 1.0)
    eps = scale * 1e-6
    for axis in range(lo.shape[0]):
        if not np.isfinite(extent[axis]) or extent[axis] <= 1e-9:
            lo[axis] -= eps
            hi[axis] += eps
    return lo, hi


def split_adaptive_tree(
    indices: Sequence[int],
    centers: np.ndarray,
    region_min: np.ndarray,
    region_max: np.ndarray,
    target_core_size: int,
    path: Tuple[int, ...],
) -> List[Dict[str, object]]:
    idxs = [int(i) for i in indices]
    if len(idxs) <= int(target_core_size):
        return [
            {
                "cell_key": path if path else (0,),
                "core_indices": sorted(idxs),
                "region_min": np.asarray(region_min, dtype=np.float64),
                "region_max": np.asarray(region_max, dtype=np.float64),
            }
        ]

    pts = centers[np.asarray(idxs, dtype=np.int64)]
    extent = np.nanmax(pts, axis=0) - np.nanmin(pts, axis=0)
    axis = int(np.nanargmax(extent)) if np.isfinite(extent).any() else 0
    order = sorted(idxs, key=lambda idx: (float(centers[int(idx), axis]), int(idx)))
    mid = len(order) // 2
    left = order[:mid]
    right = order[mid:]
    if not left or not right:
        left = order[: max(1, len(order) // 2)]
        right = order[len(left) :]
    if not left or not right:
        return [
            {
                "cell_key": path if path else (0,),
                "core_indices": sorted(idxs),
                "region_min": np.asarray(region_min, dtype=np.float64),
                "region_max": np.asarray(region_max, dtype=np.float64),
            }
        ]

    left_max = float(max(centers[int(i), axis] for i in left))
    right_min = float(min(centers[int(i), axis] for i in right))
    threshold = 0.5 * (left_max + right_min)
    if not np.isfinite(threshold) or threshold <= float(region_min[axis]) or threshold >= float(region_max[axis]):
        threshold = float(centers[int(right[0]), axis])
    if not np.isfinite(threshold) or threshold <= float(region_min[axis]) or threshold >= float(region_max[axis]):
        threshold = 0.5 * (float(region_min[axis]) + float(region_max[axis]))

    left_min = np.asarray(region_min, dtype=np.float64).copy()
    left_max_region = np.asarray(region_max, dtype=np.float64).copy()
    right_min_region = np.asarray(region_min, dtype=np.float64).copy()
    right_max = np.asarray(region_max, dtype=np.float64).copy()
    left_max_region[axis] = threshold
    right_min_region[axis] = threshold

    return (
        split_adaptive_tree(left, centers, left_min, left_max_region, target_core_size, (*path, 0))
        + split_adaptive_tree(right, centers, right_min_region, right_max, target_core_size, (*path, 1))
    )


def region_gap(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
) -> np.ndarray:
    return np.maximum(np.maximum(a_min - b_max, b_min - a_max), 0.0)


def regions_are_adjacent(a: Dict[str, object], b: Dict[str, object], margin: float) -> bool:
    if a is b:
        return False
    gap = region_gap(
        np.asarray(a["region_min"], dtype=np.float64),
        np.asarray(a["region_max"], dtype=np.float64),
        np.asarray(b["region_min"], dtype=np.float64),
        np.asarray(b["region_max"], dtype=np.float64),
    )
    return bool(np.all(gap <= float(margin) + 1e-9))


def bbox_distance_to_region(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    region_min: np.ndarray,
    region_max: np.ndarray,
) -> float:
    gap = region_gap(
        np.asarray(bbox_min, dtype=np.float64),
        np.asarray(bbox_max, dtype=np.float64),
        np.asarray(region_min, dtype=np.float64),
        np.asarray(region_max, dtype=np.float64),
    )
    return float(np.linalg.norm(gap))


def order_tree_leaves(leaves: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    def key_fn(leaf: Dict[str, object]) -> Tuple[float, ...]:
        lo = np.asarray(leaf["region_min"], dtype=np.float64)
        hi = np.asarray(leaf["region_max"], dtype=np.float64)
        center = 0.5 * (lo + hi)
        return tuple(float(v) for v in center)

    return sorted(leaves, key=key_fn)


def build_footprint_tree_chunks(
    meta: Dict[str, object],
    spatial_args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    axis_indices = pose_grid_axis_indices(spatial_args.pose_grid_axes)
    centers, bbox_mins, bbox_maxs, feature_meta = footprint_features_from_meta(meta, spatial_args, axis_indices)
    target_core_size = auto_core_target_size(int(spatial_args.max_chunk_size))
    region_min, region_max = expand_degenerate_region(
        np.nanmin(bbox_mins, axis=0),
        np.nanmax(bbox_maxs, axis=0),
    )
    leaves = split_adaptive_tree(
        indices=list(range(len(meta["stems"]))),
        centers=centers,
        region_min=region_min,
        region_max=region_max,
        target_core_size=target_core_size,
        path=(),
    )
    ordered_leaves = order_tree_leaves(leaves)
    for order_idx, leaf in enumerate(ordered_leaves):
        leaf["cell_order"] = int(order_idx)

    chunks: List[Dict[str, object]] = []
    total_dropped_seam_images = 0
    margin = float(spatial_args.footprint_neighbor_margin)
    leaf_core_sets = [set(int(i) for i in leaf["core_indices"]) for leaf in ordered_leaves]

    for leaf_idx, leaf in enumerate(ordered_leaves):
        core_indices = sorted(int(i) for i in leaf["core_indices"])
        if len(core_indices) < int(spatial_args.min_chunk_size):
            continue

        seam_candidates: List[Tuple[int, float, int, Tuple[int, ...]]] = []
        for other_idx, other in enumerate(ordered_leaves):
            if other_idx == leaf_idx:
                continue
            if not regions_are_adjacent(leaf, other, margin):
                continue
            for idx in sorted(leaf_core_sets[other_idx]):
                dist = bbox_distance_to_region(
                    bbox_mins[int(idx)],
                    bbox_maxs[int(idx)],
                    np.asarray(leaf["region_min"], dtype=np.float64),
                    np.asarray(leaf["region_max"], dtype=np.float64),
                )
                intersects = int(dist <= margin + 1e-9)
                seam_candidates.append((0 if intersects else 1, dist, int(idx), tuple(int(v) for v in other["cell_key"])))

        seam_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        candidate_indices = []
        seen_candidates = set()
        for _priority, _dist, idx, _cell in seam_candidates:
            if idx in seen_candidates:
                continue
            seen_candidates.add(idx)
            candidate_indices.append(idx)

        budget = max(0, int(spatial_args.max_chunk_size) - len(core_indices))
        overlap_indices = sorted(candidate_indices[:budget])
        dropped_seam_images = max(0, len(candidate_indices) - len(overlap_indices))
        total_dropped_seam_images += dropped_seam_images

        indices = sorted(set(core_indices + overlap_indices))
        core_set = set(core_indices)
        core_local_indices = [local_i for local_i, global_i in enumerate(indices) if int(global_i) in core_set]
        chunks.append(
            {
                "chunk_id": int(len(chunks)),
                "cell_order": int(leaf["cell_order"]),
                "cell_key": tuple(int(v) for v in leaf["cell_key"]),
                "indices": indices,
                "core_indices": core_indices,
                "overlap_indices": overlap_indices,
                "core_local_indices": core_local_indices,
                "num_seam_candidates": int(len(candidate_indices)),
                "num_dropped_seam_images": int(dropped_seam_images),
            }
        )
        if int(spatial_args.max_chunks) > 0 and len(chunks) >= int(spatial_args.max_chunks):
            break

    grid_meta = {
        "partition": "footprint_tree",
        "axes": str(spatial_args.pose_grid_axes),
        "grid_size_requested": float(spatial_args.pose_grid_size),
        "grid_size_effective": 0.0,
        "origin": region_min.astype(float).tolist(),
        "region_min": region_min.astype(float).tolist(),
        "region_max": region_max.astype(float).tolist(),
        "num_occupied_cells": int(len(ordered_leaves)),
        "num_grid_refinements": int(max(0, len(ordered_leaves) - 1)),
        "auto_core_target_size": int(target_core_size),
        "total_dropped_seam_images": int(total_dropped_seam_images),
        "footprint": feature_meta,
    }
    return chunks, grid_meta


def build_pose_grid_chunks(
    meta: Dict[str, object],
    spatial_args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    centers = pose_centers_from_meta(meta)
    axis_indices = pose_grid_axis_indices(spatial_args.pose_grid_axes)
    coords = centers[:, axis_indices]
    origin = coords.min(axis=0)
    grid_size = float(spatial_args.pose_grid_size)
    if grid_size <= 0:
        target_core_size = auto_core_target_size(int(spatial_args.max_chunk_size))
        grid_size = estimate_pose_grid_size(coords, target_core_size)
        cell_to_core = build_cell_to_core(coords, origin, grid_size)
        num_grid_refinements = 0
    else:
        target_core_size = int(spatial_args.max_chunk_size)
        num_grid_refinements = 0
        cell_to_core = build_cell_to_core(coords, origin, grid_size)
        print(
            "[WARN] --pose_grid_size was set explicitly; automatic grid estimation from "
            "--max_chunk_size is bypassed."
        )

    oversized_cells = {
        key: len(indices)
        for key, indices in cell_to_core.items()
        if len(indices) > int(spatial_args.max_chunk_size)
    }
    if oversized_cells:
        key, count = max(oversized_cells.items(), key=lambda item: item[1])
        raise ValueError(
            f"Pose grid cell {key} contains {count} core images, exceeding "
            f"--max_chunk_size={spatial_args.max_chunk_size}. Reduce --pose_grid_size "
            "or increase --max_chunk_size."
        )

    ordered_cells = pose_grid_cell_order(cell_to_core.keys())
    cell_centers = {
        key: coords[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        for key, indices in cell_to_core.items()
    }

    chunks: List[Dict[str, object]] = []
    neighbor_radius = int(spatial_args.pose_grid_neighbor_radius)
    total_dropped_seam_images = 0

    for cell_order_idx, cell_key in enumerate(ordered_cells):
        core_indices = sorted(cell_to_core[cell_key])
        if len(core_indices) < int(spatial_args.min_chunk_size):
            continue

        seam_candidates: List[Tuple[float, int, Tuple[int, ...]]] = []
        if neighbor_radius > 0:
            center = cell_centers[cell_key]
            core_set = set(core_indices)
            for other_key in ordered_cells:
                if not are_neighbor_cells(cell_key, other_key, neighbor_radius):
                    continue
                for idx in cell_to_core[other_key]:
                    if int(idx) in core_set:
                        continue
                    dist = float(np.linalg.norm(coords[int(idx)] - center))
                    seam_candidates.append((dist, int(idx), tuple(int(v) for v in other_key)))

        seam_candidates.sort(key=lambda item: (item[0], item[1]))
        budget = max(0, int(spatial_args.max_chunk_size) - len(core_indices))
        overlap_indices = sorted({idx for _dist, idx, _cell in seam_candidates[:budget]})
        dropped_seam_images = max(0, len({idx for _dist, idx, _cell in seam_candidates}) - len(overlap_indices))
        total_dropped_seam_images += dropped_seam_images

        indices = sorted(set(core_indices + overlap_indices))
        core_set = set(core_indices)
        core_local_indices = [local_i for local_i, global_i in enumerate(indices) if int(global_i) in core_set]
        chunks.append(
            {
                "chunk_id": int(len(chunks)),
                "cell_order": int(cell_order_idx),
                "cell_key": tuple(int(v) for v in cell_key),
                "indices": indices,
                "core_indices": core_indices,
                "overlap_indices": overlap_indices,
                "core_local_indices": core_local_indices,
                "num_seam_candidates": int(len({idx for _dist, idx, _cell in seam_candidates})),
                "num_dropped_seam_images": int(dropped_seam_images),
            }
        )
        if int(spatial_args.max_chunks) > 0 and len(chunks) >= int(spatial_args.max_chunks):
            break

    grid_meta = {
        "partition": "pose_grid",
        "axes": str(spatial_args.pose_grid_axes),
        "grid_size_requested": float(spatial_args.pose_grid_size),
        "grid_size_effective": float(grid_size),
        "origin": origin.astype(float).tolist(),
        "num_occupied_cells": int(len(ordered_cells)),
        "num_grid_refinements": int(num_grid_refinements),
        "auto_core_target_size": int(target_core_size),
        "total_dropped_seam_images": int(total_dropped_seam_images),
    }
    return chunks, grid_meta


def build_spatial_chunks(
    meta: Dict[str, object],
    spatial_args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if str(spatial_args.spatial_partition) == "pose_grid":
        return build_pose_grid_chunks(meta, spatial_args)
    if float(spatial_args.pose_grid_size) > 0:
        print("[WARN] --pose_grid_size is ignored by --spatial_partition footprint_tree.")
    return build_footprint_tree_chunks(meta, spatial_args)


def make_chunk_meta(meta: Dict[str, object], indices: Sequence[int]) -> Dict[str, object]:
    chunk_meta = dict(meta)
    chunk_meta["stems"] = [meta["stems"][i] for i in indices]
    chunk_meta["rgbs"] = [meta["rgbs"][i] for i in indices]
    chunk_meta["gt_maps"] = [meta["gt_maps"][i] for i in indices]
    chunk_meta["valid_masks"] = [meta["valid_masks"][i] for i in indices]
    return chunk_meta


def gt_cameras_for_stems(meta: Dict[str, object], stems: Sequence[str]) -> List[Dict[str, object]]:
    cams = meta.get("cams", {})
    out = []
    for stem in stems:
        if stem not in cams:
            continue
        out.append({"stem": stem, "T_c2w": np.asarray(cams[stem]["T_c2w"], dtype=np.float32)})
    return out


def input_pose_centers_by_stem(meta: Dict[str, object]) -> Dict[str, np.ndarray]:
    return slam_base.camera_centers_by_stem(gt_cameras_for_stems(meta, meta["stems"]))


def points_from_maps(
    pred_maps: Sequence[np.ndarray],
    pred_valid_masks: Sequence[np.ndarray],
    rgbs: Sequence[np.ndarray],
    local_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    points_all: List[np.ndarray] = []
    colors_all: List[np.ndarray] = []
    for local_i in local_indices:
        point_map = np.asarray(pred_maps[int(local_i)], dtype=np.float32)
        if point_map.ndim != 3 or point_map.shape[-1] != 3:
            continue
        mask = np.asarray(pred_valid_masks[int(local_i)], dtype=bool)
        if mask.shape != point_map.shape[:2]:
            continue
        mask = mask & np.isfinite(point_map).all(axis=-1)
        if not mask.any():
            continue
        rgb = rgbs[int(local_i)]
        if rgb.shape[:2] != point_map.shape[:2]:
            rgb = cv2.resize(rgb, (point_map.shape[1], point_map.shape[0]), interpolation=cv2.INTER_AREA)
        points_all.append(point_map[mask].reshape(-1, 3).astype(np.float32))
        colors_all.append(rgb[mask].reshape(-1, 3).astype(np.uint8))

    points = np.concatenate(points_all, axis=0) if points_all else np.empty((0, 3), np.float32)
    colors = np.concatenate(colors_all, axis=0) if colors_all else np.empty((0, 3), np.uint8)
    return points, colors


def log_chunk_outputs(
    chunk_entity: str,
    pred_root: str,
    pred_points: np.ndarray,
    pred_colors: np.ndarray,
    pred_cams: Sequence[Dict[str, object]],
    axis_size: float,
    args: argparse.Namespace,
    spatial_args: argparse.Namespace,
    seed: int,
) -> int:
    points, colors = base.sample_points_and_colors(
        pred_points,
        pred_colors,
        int(spatial_args.max_pred_points_per_chunk),
        seed,
    )
    base.log_points(f"{chunk_entity}/{pred_root}/points", points, colors, args.point_radius)
    pred_axis_colors = ((255, 0, 255), (255, 180, 0), (0, 220, 255))
    base.log_camera_axes(
        f"{chunk_entity}/{pred_root}/cameras/axes",
        pred_cams,
        axis_size,
        args.camera_axis_radius,
        pred_axis_colors,
    )
    base.log_camera_labels(f"{chunk_entity}/{pred_root}/cameras/labels", pred_cams, (255, 120, 40))
    return int(points.shape[0])


def dedupe_cameras_by_stem(cams: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen: set[str] = set()
    for cam in cams:
        stem = str(cam.get("stem", ""))
        key = stem or f"pred_index_{cam.get('pred_index', len(out))}"
        if key in seen:
            continue
        seen.add(key)
        out.append(cam)
    return out


def chunks_rrd_path(args: argparse.Namespace, output_rrd: Path) -> Path:
    explicit = getattr(args, "output_chunks_rrd", None)
    if explicit:
        return Path(explicit).expanduser().resolve()
    suffix = output_rrd.suffix or ".rrd"
    return output_rrd.with_name(f"{output_rrd.stem}_chunks{suffix}")


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


def pred_depth_prior(pred: Dict[str, torch.Tensor]) -> torch.Tensor | None:
    pts_cam = pred.get("pts3d_cam", None)
    if pts_cam is None or pts_cam.shape[-1] != 3:
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


def build_depth_prior_cache(
    preds: Sequence[Dict[str, torch.Tensor]],
    indices: Sequence[int],
) -> Dict[int, torch.Tensor]:
    cache: Dict[int, torch.Tensor] = {}
    for global_idx, pred in zip(indices, preds):
        depth = pred_depth_prior(pred)
        if depth is not None:
            cache[int(global_idx)] = depth.detach().cpu()
    return cache


def apply_cached_depth_priors_to_views(
    views: Sequence[Dict[str, object]],
    indices: Sequence[int],
    depth_cache: Dict[int, torch.Tensor],
    device: torch.device,
) -> Tuple[List[Dict[str, object]], int]:
    matched = [int(i) for i in indices if int(i) in depth_cache]
    if not matched:
        return list(views), 0

    template = depth_cache[matched[0]].to(device=device, dtype=torch.float32)
    zero_depth = torch.zeros_like(template)
    out: List[Dict[str, object]] = []
    used = 0
    for view, global_idx in zip(views, indices):
        prior = depth_cache.get(int(global_idx))
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


def save_final_eval_outputs(
    eval_dir: Path,
    pred_cams: Sequence[Dict[str, object]],
    pred_points: np.ndarray,
    pred_colors: np.ndarray,
    meta: Dict[str, object],
) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)

    stems: List[str] = []
    Ts: List[np.ndarray] = []
    valid: List[bool] = []

    for cam in pred_cams:
        stem = str(cam.get("stem", ""))
        T = np.asarray(cam.get("T_c2w"), dtype=np.float32)
        ok = T.shape == (4, 4) and np.isfinite(T).all()

        stems.append(stem)
        Ts.append(T if ok else np.full((4, 4), np.nan, dtype=np.float32))
        valid.append(bool(ok))

    np.savez_compressed(
        eval_dir / "pred_cameras.npz",
        stems=np.asarray(stems, dtype=str),
        T_c2w=np.stack(Ts, axis=0).astype(np.float32) if Ts else np.empty((0, 4, 4), dtype=np.float32),
        valid=np.asarray(valid, dtype=bool),
    )

    points = np.asarray(pred_points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(pred_colors, dtype=np.uint8).reshape(-1, 3)
    if colors.shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]

    np.savez_compressed(
        eval_dir / "pred_points.npz",
        points=points.astype(np.float32),
        colors=colors.astype(np.uint8),
    )

    meta = dict(meta)
    meta["num_cameras"] = int(len(stems))
    meta["num_valid_cameras"] = int(np.asarray(valid, dtype=bool).sum())
    meta["num_points"] = int(points.shape[0])

    (eval_dir / "meta.json").write_text(
        json.dumps(base.json_safe(meta), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved final eval outputs: {eval_dir}")


def save_spatial_rrd(
    args: argparse.Namespace,
    spatial_args: argparse.Namespace,
    meta: Dict[str, object],
    grid_meta: Dict[str, object],
    chunk_records: Sequence[Dict[str, object]],
) -> None:
    output_rrd = Path(args.output_rrd).expanduser().resolve()
    output_chunks_rrd = chunks_rrd_path(args, output_rrd)
    if output_chunks_rrd == output_rrd:
        raise ValueError("--output_chunks_rrd must be different from --output_rrd.")
    output_rrd.parent.mkdir(parents=True, exist_ok=True)
    output_chunks_rrd.parent.mkdir(parents=True, exist_ok=True)

    scene_name = base.sanitize_name(Path(args.scene_dir).resolve().name)
    gt_points, gt_colors = base.sample_points_and_colors(
        meta["gt_points"],
        meta["gt_colors"],
        args.max_gt_points,
        args.seed,
    )
    all_gt_cams = gt_cameras_for_stems(meta, meta["stems"])

    aggregate_points = np.concatenate(
        [np.asarray(record["core_pred_points"], dtype=np.float32).reshape(-1, 3) for record in chunk_records],
        axis=0,
    ) if chunk_records else np.empty((0, 3), np.float32)
    aggregate_colors = np.concatenate(
        [np.asarray(record["core_pred_colors"], dtype=np.uint8).reshape(-1, 3) for record in chunk_records],
        axis=0,
    ) if chunk_records else np.empty((0, 3), np.uint8)
    aggregate_points, aggregate_colors = base.sample_points_and_colors(
        aggregate_points,
        aggregate_colors,
        args.max_pred_points,
        args.seed + 17,
    )

    pred_root = "pred_spatial_aligned" if args.align != "none" else "pred_spatial"

    axis_size = base.estimate_axis_size([aggregate_points, gt_points], args.camera_axis_size)
    gt_axis_colors = ((255, 0, 0), (0, 220, 0), (40, 80, 255))

    pred_axis_colors = ((255, 0, 255), (255, 180, 0), (0, 220, 255))
    all_pred_cams = dedupe_cameras_by_stem([cam for record in chunk_records for cam in record["pred_cams"]])

    save_final_eval_outputs(
        eval_dir=output_rrd.with_suffix("") / "eval",
        pred_cams=all_pred_cams,
        pred_points=aggregate_points,
        pred_colors=aggregate_colors,
        meta={
            "schema": "final_eval_v1",
            "script": "scripts/predict_scene_to_rrd_spatial_chunks.py",
            "scene_dir": str(Path(args.scene_dir).expanduser().resolve()),
            "method": args.model,
            "checkpoint": args.checkpoint,
            "pose_convention": "T_c2w",
            "points_coordinate": "same_as_pred_cameras",
            "post_align": {
                "enabled": False,
                "type": "none",
                "target": "none",
                "valid": True,
                "note": "Our method output is saved directly without extra final Sim3 alignment.",
            },
            "aggregation": {
                "points": "core_pred_points",
                "cameras": "dedupe_by_stem",
                "num_chunks": int(len(chunk_records)),
            },
        },
    )

    overall_recording_id = (
        f"spatial_overall_{scene_name}_{base.sanitize_name(args.model)}_{base.sanitize_name(args.align)}"
    )
    base.rr_init_save_compat("predict_scene_to_rrd_spatial_overall", overall_recording_id, output_rrd)
    base.rr_set_time_compat("frame", 0)
    base.log_view_coordinates(args.view_coordinates)
    base.send_blueprint(background=tuple(args.background), hide_grid=args.hide_grid)

    base.log_points("world/gt/points", gt_points, gt_colors, args.point_radius)
    base.log_points(f"world/{pred_root}/points", aggregate_points, aggregate_colors, args.point_radius)
    base.log_camera_axes("world/cameras/gt/axes", all_gt_cams, axis_size, args.camera_axis_radius, gt_axis_colors)
    base.log_camera_axes(f"world/cameras/{pred_root}/axes", all_pred_cams, axis_size, args.camera_axis_radius, pred_axis_colors)

    if args.show_world_axes:
        bbox_points = gt_points if gt_points.shape[0] > 0 else aggregate_points
        base.log_world_axes_marker(
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
    print(f"Saved overall Rerun recording: {output_rrd}")

    chunks_recording_id = (
        f"spatial_chunks_{scene_name}_{base.sanitize_name(args.model)}_{base.sanitize_name(args.align)}"
    )
    base.rr_init_save_compat("predict_scene_to_rrd_spatial_chunks", chunks_recording_id, output_chunks_rrd)
    base.rr_set_time_compat("frame", 0)
    base.log_view_coordinates(args.view_coordinates)
    base.send_blueprint(background=tuple(args.background), hide_grid=args.hide_grid)

    for record in chunk_records:
        chunk_id = int(record["chunk_id"])
        base.rr_set_time_compat("chunk", chunk_id)
        chunk_entity = f"world/spatial_chunks/chunk_{chunk_id:03d}"
        num_logged = log_chunk_outputs(
            chunk_entity=chunk_entity,
            pred_root=pred_root,
            pred_points=record["chunk_pred_points"],
            pred_colors=record["chunk_pred_colors"],
            pred_cams=record["pred_cams"],
            axis_size=axis_size,
            args=args,
            spatial_args=spatial_args,
            seed=int(args.seed) + 1009 * (chunk_id + 1),
        )
        record["num_chunk_pred_points_logged"] = int(num_logged)
        if args.log_images:
            for local_i, (rgb, stem) in enumerate(zip(record["rgbs"], record["stems"])):
                base.rr.log(
                    f"{chunk_entity}/inputs/view_{local_i:03d}_{base.sanitize_name(stem)}/rgb",
                    base.rr.Image(rgb),
                )

    base.rr_disconnect_compat()
    print(f"Saved chunk Rerun recording: {output_chunks_rrd}")

    sidecar = output_rrd.with_suffix(".json")
    payload = {
        "scene_dir": str(Path(args.scene_dir).resolve()),
        "model": args.model,
        "checkpoint": args.checkpoint,
        "output_rrd": str(output_rrd),
        "output_chunks_rrd": str(output_chunks_rrd),
        "stems": list(meta["stems"]),
        "target_size": {"height": int(meta["target_h"]), "width": int(meta["target_w"])},
        "grid": grid_meta,
        "chunking": {
            "max_chunk_size": int(spatial_args.max_chunk_size),
            "min_chunk_size": int(spatial_args.min_chunk_size),
            "max_chunks": int(spatial_args.max_chunks),
            "num_chunks": int(len(chunk_records)),
            "max_pred_points_per_chunk": int(spatial_args.max_pred_points_per_chunk),
            "note": (
                "Seam overlaps are selected automatically from connected neighboring cells "
                "and capped by max_chunk_size."
            ),
        },
        "alignment": str(args.align),
        "prior_control": {
            "ray_prior_cli": spatial_args.use_ray_prior,
            "depth_prior_cli": spatial_args.use_depth_prior,
            "world_rotation_prior_cli": spatial_args.use_world_rotation_prior,
            "world_translation_prior_cli": spatial_args.use_world_translation_prior,
            "fov_prior_cli": spatial_args.use_fov_prior,
            "bootstrap_ray_prior_from_chunk0": bool(getattr(spatial_args, "bootstrap_ray_prior_from_chunk0", False)),
            "bootstrap_depth_prior_from_seams": bool(getattr(spatial_args, "bootstrap_depth_prior_from_seams", False)),
            "bootstrap_intrinsics": getattr(spatial_args, "bootstrap_intrinsics", None),
            "seam_depth_priors": bool(getattr(spatial_args, "bootstrap_depth_prior_from_seams", False)),
        },
        "num_gt_points_logged": int(gt_points.shape[0]),
        "num_pred_core_points_logged": int(aggregate_points.shape[0]),
        "num_gt_cameras": int(len(all_gt_cams)),
        "num_pred_cameras": int(len(all_pred_cams)),
        "chunks": [
            {
                "chunk_id": int(record["chunk_id"]),
                "cell_key": list(record["cell_key"]),
                "indices": [int(i) for i in record["indices"]],
                "core_indices": [int(i) for i in record["core_indices"]],
                "overlap_indices": [int(i) for i in record["overlap_indices"]],
                "stems": list(record["stems"]),
                "core_stems": list(record["core_stems"]),
                "overlap_stems": list(record["overlap_stems"]),
                "num_seam_candidates": int(record["num_seam_candidates"]),
                "num_dropped_seam_images": int(record["num_dropped_seam_images"]),
                "num_depth_priors_used": int(record["num_depth_priors_used"]),
                "num_chunk_pred_points_raw": int(record["num_chunk_pred_points_raw"]),
                "num_core_pred_points": int(record["core_pred_points"].shape[0]),
                "num_chunk_pred_points_logged": int(record.get("num_chunk_pred_points_logged", 0)),
                "num_pred_cameras": int(len(record["pred_cams"])),
                "alignment": record["align_meta"],
            }
            for record in chunk_records
        ],
    }
    sidecar.write_text(json.dumps(base.json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved sidecar metadata: {sidecar}")


@torch.no_grad()
def main() -> None:
    args, spatial_args = parse_args()
    device = base.resolve_device(args.device)

    views, meta = base.build_views_from_scene(args, device=device)
    chunks, grid_meta = build_spatial_chunks(meta, spatial_args)
    if not chunks:
        raise RuntimeError("No spatial chunks generated. Check --pose_grid_size/--min_chunk_size/--num_views.")

    print(
        f"Running spatial chunk reconstruction: frames={len(views)}, chunks={len(chunks)}, "
        f"partition={grid_meta.get('partition', 'pose_grid')}, axes={grid_meta['axes']}, "
        f"cell_size={float(grid_meta['grid_size_effective']):.6g}, "
        f"max_chunk_size={spatial_args.max_chunk_size}, "
        f"auto_core_target={grid_meta['auto_core_target_size']}, "
        f"dropped_seam_images={grid_meta['total_dropped_seam_images']}"
    )
    if "footprint" in grid_meta:
        counts = grid_meta["footprint"].get("footprint_source_counts", {})
        print(
            "[INFO] footprint sources: "
            f"depth={int(counts.get('depth', 0))}, "
            f"lookat={int(counts.get('lookat', 0))}, "
            f"center={int(counts.get('center', 0))}"
        )
    if int(grid_meta["total_dropped_seam_images"]) > 0:
        print(
            "[WARN] Some adjacent seam images were not duplicated because "
            "--max_chunk_size left no room. Increase --max_chunk_size or reduce the core target pressure."
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
    align_mode = str(getattr(spatial_args, "runtime_align", args.align)).lower()
    input_pose_cams_by_stem = input_pose_centers_by_stem(meta)
    if align_mode in POSE_TRANSLATION_ALIGN_MODES and not input_pose_cams_by_stem:
        print(
            "[WARN] --align pose_* selected, but no input pose translations were found; "
            "pose alignment will fall back to raw chunk coordinates."
        )

    bootstrap_ray_prior = model_name in PI3X_PRIOR_MODELS and spatial_args.use_ray_prior is False
    bootstrap_depth_prior = model_name in PI3X_PRIOR_MODELS and spatial_args.use_depth_prior is False
    bootstrapped_intrinsics: torch.Tensor | None = None
    if bootstrap_ray_prior:
        spatial_args.bootstrap_ray_prior_from_chunk0 = True
        spatial_args.bootstrap_intrinsics = None
        slam_base.set_pi3x_ray_prior_prob(model, enabled=False)
        print(
            "[INFO] --no_ray_prior selected: chunk 0 runs without ray/intrinsics prior; "
            "later chunks will use average intrinsics recovered from chunk 0 predicted rays."
        )
    if bootstrap_depth_prior:
        spatial_args.bootstrap_depth_prior_from_seams = True
        set_model_task_prob(model, "depth_prob", enabled=False)
        set_model_task_value(model, "sparse_depth_prob", 0.0)
        print(
            "[INFO] --no_depth_prior selected: external depth prior is disabled; "
            "later adjacent chunks will use predicted seam-image depth from earlier chunks."
        )

    depth_prior_cache: Dict[int, torch.Tensor] = {}
    chunk_records: List[Dict[str, object]] = []
    for chunk in chunks:
        chunk_id = int(chunk["chunk_id"])
        indices = [int(i) for i in chunk["indices"]]
        core_indices = [int(i) for i in chunk["core_indices"]]
        overlap_indices = [int(i) for i in chunk["overlap_indices"]]
        chunk_views = [views[i] for i in indices]
        chunk_stems = [meta["stems"][i] for i in indices]
        chunk_rgbs = [meta["rgbs"][i] for i in indices]
        chunk_meta = make_chunk_meta(meta, indices)
        if bootstrap_ray_prior and chunk_id > 0:
            if bootstrapped_intrinsics is None:
                slam_base.set_pi3x_ray_prior_prob(model, enabled=False)
                print(
                    f"[chunk {chunk_id:03d}] bootstrap ray prior unavailable; "
                    "continuing without ray/intrinsics prior."
                )
            else:
                slam_base.set_pi3x_ray_prior_prob(model, enabled=True)
                chunk_views = slam_base.apply_bootstrap_intrinsics_to_views(
                    chunk_views,
                    intrinsics=bootstrapped_intrinsics,
                    device=device,
                )
        num_depth_priors_used = 0
        if bootstrap_depth_prior:
            chunk_views, num_depth_priors_used = apply_cached_depth_priors_to_views(
                chunk_views,
                indices=indices,
                depth_cache=depth_prior_cache,
                device=device,
            )
            set_model_task_prob(model, "depth_prob", enabled=num_depth_priors_used > 0)
            if num_depth_priors_used > 0:
                set_model_task_value(model, "sparse_depth_prob", 0.0)

        print(
            f"[chunk {chunk_id:03d}] cell={chunk['cell_key']}, "
            f"core={len(core_indices)}, overlap={len(overlap_indices)}, total={len(indices)}, "
            f"depth_priors={num_depth_priors_used}, stems={chunk_stems[0]}..{chunk_stems[-1]}"
        )

        preds = model(chunk_views)
        if bootstrap_ray_prior and chunk_id == 0:
            bootstrapped_intrinsics = slam_base.recover_average_intrinsics_from_pred_rays(preds)
            if bootstrapped_intrinsics is None:
                print("[WARN] Failed to recover bootstrap intrinsics from chunk 0 predicted rays.")
            else:
                K_np = bootstrapped_intrinsics.detach().cpu().numpy()
                spatial_args.bootstrap_intrinsics = {
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
        if bootstrap_depth_prior:
            depth_prior_cache.update(build_depth_prior_cache(preds, indices))
        raw_points, raw_colors, pred_maps, pred_valid_masks, raw_cams = base.collect_pred_outputs(
            preds=preds,
            rgbs=chunk_rgbs,
            args=args,
            stems=chunk_stems,
        )
        if align_mode in POSE_TRANSLATION_ALIGN_MODES:
            pred_points, pred_colors, pred_maps_aligned, pred_cams, align_meta = (
                slam_base.apply_chunk_pose_alignment(
                    mode=align_mode,
                    chunk_id=chunk_id,
                    reference_cams_by_stem=input_pose_cams_by_stem,
                    raw_pred_points=raw_points,
                    raw_pred_colors=raw_colors,
                    pred_maps=pred_maps,
                    raw_pred_cams=raw_cams,
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
            pred_points, pred_colors, pred_maps_aligned, pred_cams, align_meta = base.estimate_and_apply_alignment(
                args=alignment_args,
                meta=chunk_meta,
                pred_points=raw_points,
                pred_colors=raw_colors,
                pred_maps=pred_maps,
                pred_valid_masks=pred_valid_masks,
                pred_cams=raw_cams,
            )

        core_points, core_colors = points_from_maps(
            pred_maps=pred_maps_aligned,
            pred_valid_masks=pred_valid_masks,
            rgbs=chunk_rgbs,
            local_indices=chunk["core_local_indices"],
        )

        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "cell_key": tuple(chunk["cell_key"]),
                "indices": indices,
                "core_indices": core_indices,
                "overlap_indices": overlap_indices,
                "stems": chunk_stems,
                "core_stems": [meta["stems"][i] for i in core_indices],
                "overlap_stems": [meta["stems"][i] for i in overlap_indices],
                "num_seam_candidates": int(chunk["num_seam_candidates"]),
                "num_dropped_seam_images": int(chunk["num_dropped_seam_images"]),
                "num_depth_priors_used": int(num_depth_priors_used),
                "rgbs": chunk_rgbs,
                "chunk_pred_points": pred_points,
                "chunk_pred_colors": pred_colors,
                "core_pred_points": core_points,
                "core_pred_colors": core_colors,
                "pred_cams": pred_cams,
                "align_meta": align_meta,
                "num_chunk_pred_points_raw": int(raw_points.shape[0]),
            }
        )
        print(
            f"[chunk {chunk_id:03d}] prediction: raw_points={raw_points.shape[0]}, "
            f"core_points={core_points.shape[0]}, cameras={len(pred_cams)}, "
            f"align={align_meta['mode']}, valid={align_meta['valid']}"
        )

    save_spatial_rrd(
        args=args,
        spatial_args=spatial_args,
        meta=meta,
        grid_meta=grid_meta,
        chunk_records=chunk_records,
    )


if __name__ == "__main__":
    main()
