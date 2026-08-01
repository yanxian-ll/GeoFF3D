# -*- coding: utf-8 -*-
"""Footprint estimation strategies used before spatial chunking."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from geoff3d.slrf.chunking import (
    build_temporal_chunks,
    order_spatial_chunks,
    robust_bbox,
)
from geoff3d.slrf.geometry_align import (
    apply_chunk_pose_alignment,
    restore_predictions_from_recenter,
)
from geoff3d.slrf.model_runner import (
    collect_pred_outputs,
    filter_views_for_prior_policy,
)
from geoff3d.slrf.rrd_writer import input_pose_centers_by_stem
from geoff3d.slrf.scene_io import (
    DEPTH_MAX_METERS,
    DEPTH_MIN_METERS,
    load_chunk_views_from_scene,
    read_depth,
    resize_depth_to_target,
    scale_K_to_target,
)

FOOTPRINT_SAMPLE_STRIDE = 8
FOOTPRINT_MIN_POINTS = 32
FOOTPRINT_QUANTILE_MIN = 0.02
FOOTPRINT_QUANTILE_MAX = 0.98


def _prior_footprint_worker(
    args: Tuple[object, ...],
) -> Tuple[
    int,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    str,
]:
    (
        frame_index, depth_path, K, T_c2w, cam_width, cam_height,
        target_h, target_w, axis_indices, stride, min_points,
        quantile_min, quantile_max,
    ) = args
    try:
        depth_raw = read_depth(Path(str(depth_path)))
        K_scaled = scale_K_to_target(
            K=np.asarray(K, dtype=np.float64),
            cam_width=cam_width,
            cam_height=cam_height,
            source_h=depth_raw.shape[0],
            source_w=depth_raw.shape[1],
            target_h=int(target_h),
            target_w=int(target_w),
        )
        depth = resize_depth_to_target(
            depth_raw, target_h=int(target_h), target_w=int(target_w)
        )
        stride = max(1, int(stride))
        ys = np.arange(0, depth.shape[0], stride, dtype=np.float64)
        xs = np.arange(0, depth.shape[1], stride, dtype=np.float64)
        u, v = np.meshgrid(xs, ys)
        z = depth[::stride, ::stride].astype(np.float64)
        valid = (
            np.isfinite(z)
            & (z > DEPTH_MIN_METERS)
            & (z < DEPTH_MAX_METERS)
        )
        if np.count_nonzero(valid) < int(min_points):
            return int(frame_index), None, None, None, "not_enough_valid_depth"

        fx, fy = float(K_scaled[0, 0]), float(K_scaled[1, 1])
        cx, cy = float(K_scaled[0, 2]), float(K_scaled[1, 2])
        if abs(fx) < 1e-12 or abs(fy) < 1e-12:
            return int(frame_index), None, None, None, "invalid_intrinsics"

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_camera = np.stack([x, y, z], axis=-1)[valid].reshape(-1, 3)
        T = np.asarray(T_c2w, dtype=np.float64)
        points_world = points_camera @ T[:3, :3].T + T[:3, 3]
        points_world = points_world[np.isfinite(points_world).all(axis=1)]
        if points_world.shape[0] < int(min_points):
            return int(frame_index), None, None, None, "not_enough_finite_points"

        coords = points_world[:, tuple(axis_indices)]
        bbox_min, bbox_max = robust_bbox(coords, quantile_min, quantile_max)
        center = np.median(coords, axis=0).astype(np.float64)
        return int(frame_index), center, bbox_min, bbox_max, "ok"
    except Exception as exc:
        return int(frame_index), None, None, None, f"error:{exc}"


def estimate_footprints_from_prior(
    *,
    meta: Dict[str, object],
    axis_indices: Tuple[int, ...],
    workers: int,
) -> Dict[str, object]:
    """Estimate every footprint from input camera and metric-depth priors."""
    stems = list(meta["stems"])
    cams = meta.get("cams", {})
    depth_paths = meta.get("depth_paths", {})
    jobs: List[Tuple[object, ...]] = []
    for index, stem in enumerate(stems):
        cam = cams.get(stem) if isinstance(cams, dict) else None
        depth_path = depth_paths.get(stem) if isinstance(depth_paths, dict) else None
        if cam is None or not depth_path:
            raise ValueError(
                f"Cannot estimate prior footprint for {stem}: missing camera or depth."
            )
        jobs.append((
            index, str(depth_path), np.asarray(cam["K"]),
            np.asarray(cam["T_c2w"]), cam.get("width"), cam.get("height"),
            int(meta["target_h"]), int(meta["target_w"]), axis_indices,
            FOOTPRINT_SAMPLE_STRIDE, FOOTPRINT_MIN_POINTS,
            FOOTPRINT_QUANTILE_MIN, FOOTPRINT_QUANTILE_MAX,
        ))

    if int(workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            results = list(pool.map(_prior_footprint_worker, jobs))
    else:
        results = [_prior_footprint_worker(job) for job in jobs]

    centers = np.full((len(stems), len(axis_indices)), np.nan, np.float64)
    bbox_mins = np.full_like(centers, np.nan)
    bbox_maxs = np.full_like(centers, np.nan)
    failures: List[str] = []
    for index, center, bbox_min, bbox_max, status in results:
        if status != "ok" or center is None or bbox_min is None or bbox_max is None:
            failures.append(f"{stems[index]} ({status})")
            continue
        centers[index], bbox_mins[index], bbox_maxs[index] = center, bbox_min, bbox_max
    if failures:
        raise RuntimeError(
            "Prior footprint estimation failed: " + ", ".join(failures[:8])
        )
    return {
        "centers": centers,
        "bbox_mins": bbox_mins,
        "bbox_maxs": bbox_maxs,
        "meta": {
            "estimation": "prior",
            "coordinate_axes": list(axis_indices),
            "source_counts": {"prior": len(stems)},
            "sources": ["prior"] * len(stems),
            "sample_stride": FOOTPRINT_SAMPLE_STRIDE,
            "min_points": FOOTPRINT_MIN_POINTS,
            "quantile_min": FOOTPRINT_QUANTILE_MIN,
            "quantile_max": FOOTPRINT_QUANTILE_MAX,
            "workers": int(workers),
        },
    }


def _footprint_from_prediction(
    point_map: np.ndarray,
    valid_mask: np.ndarray,
    aligned_camera: Optional[Dict[str, object]],
    sample_stride: int,
    min_points: int,
    quantile_min: float,
    quantile_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
    points = np.asarray(point_map, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    stride = max(1, int(sample_stride))
    sampled = points[::stride, ::stride]
    sampled_mask = mask[::stride, ::stride]
    if sampled.ndim == 3 and sampled.shape[-1] == 3 and sampled_mask.shape == sampled.shape[:2]:
        sampled_mask &= np.isfinite(sampled).all(axis=-1)
        xy = sampled[..., :2][sampled_mask]
    else:
        xy = np.empty((0, 2), dtype=np.float64)
    if xy.shape[0] > 0:
        bbox_min, bbox_max = robust_bbox(xy, quantile_min, quantile_max)
        source = "predicted" if xy.shape[0] >= int(min_points) else "predicted_sparse"
        return np.median(xy, axis=0), bbox_min, bbox_max, source, len(xy)
    if aligned_camera is not None:
        T = np.asarray(aligned_camera.get("T_c2w"), dtype=np.float64)
        if T.shape == (4, 4) and np.isfinite(T[:2, 3]).all():
            center = T[:2, 3].copy()
            return center, center.copy(), center.copy(), "camera_fallback", 0
    invalid = np.full(2, np.nan)
    return invalid, invalid.copy(), invalid.copy(), "invalid", 0


def estimate_footprints_sequentially(
    *, model: torch.nn.Module, model_name: str,
    views: Sequence[Dict[str, object]], meta: Dict[str, object],
    prior_policy: Dict[str, object], device: torch.device,
    recenter_state: np.ndarray, norm_type: str, args: SimpleNamespace,
) -> Dict[str, object]:
    """Estimate footprints with an ordered, non-overlapping model pass."""
    chunks, _ = build_temporal_chunks(
        meta=meta, max_chunk_size=int(args.max_chunk_size),
        min_chunk_size=int(args.min_chunk_size), max_chunks=0,
        overlap_ratio=0.0, merge_small_tail=True,
    )
    chunks, _ = order_spatial_chunks(chunks, meta=meta, strategy="sequential")
    align_mode = "scale_yaw_translation" if model_name == "geoff3d" else "sim3"
    reference_cams = input_pose_centers_by_stem(meta)
    if not reference_cams:
        raise ValueError("Sequential footprint estimation requires camera poses.")

    count = len(meta["stems"])
    centers = np.full((count, 2), np.nan)
    bbox_mins, bbox_maxs = np.full_like(centers, np.nan), np.full_like(centers, np.nan)
    sources, point_counts, alignments = ["not_processed"] * count, [0] * count, []
    iterator = tqdm(chunks, desc="Estimate footprints") if tqdm else chunks
    for chunk in iterator:
        chunk_id = int(chunk["chunk_id"])
        indices = [int(i) for i in chunk["indices"]]
        stems = [meta["stems"][i] for i in indices]
        chunk_views_raw, rgbs = load_chunk_views_from_scene(
            lightweight_views=views, meta=meta, indices=indices,
            prior_policy=prior_policy, device=device,
            recenter_anchor=recenter_state, num_workers=args.scene_io_workers,
            norm_type=norm_type,
        )
        chunk_views = filter_views_for_prior_policy(chunk_views_raw, prior_policy)
        with torch.no_grad():
            preds = model(chunk_views)
        _, _, maps, masks, cameras = collect_pred_outputs(
            preds=preds, rgbs=rgbs, pred_min_depth=args.pred_min_depth,
            conf_quantile=0.0, stems=stems, collect_point_indices=[],
        )
        empty_xyz = np.empty((0, 3), np.float32)
        empty_rgb = np.empty((0, 3), np.uint8)
        _, _, maps, cameras = restore_predictions_from_recenter(
            empty_xyz, empty_rgb, maps, cameras, recenter_state
        )
        _, _, maps, cameras, alignment = apply_chunk_pose_alignment(
            mode=align_mode, chunk_id=chunk_id,
            reference_cams_by_stem=reference_cams,
            raw_pred_points=empty_xyz, raw_pred_colors=empty_rgb,
            pred_maps=maps, raw_pred_cams=cameras, target_stems=stems,
            seed=61001 + chunk_id,
        )
        if not alignment.get("valid", False):
            raise RuntimeError(f"Sequential footprint alignment failed: {alignment['note']}")
        alignments.append(alignment)
        cameras_by_stem = {str(cam.get("stem")): cam for cam in cameras}
        for local_index, global_index in enumerate(indices):
            result = _footprint_from_prediction(
                maps[local_index], masks[local_index],
                cameras_by_stem.get(stems[local_index]),
                FOOTPRINT_SAMPLE_STRIDE,
                FOOTPRINT_MIN_POINTS,
                FOOTPRINT_QUANTILE_MIN,
                FOOTPRINT_QUANTILE_MAX,
            )
            centers[global_index], bbox_mins[global_index], bbox_maxs[global_index] = result[:3]
            sources[global_index], point_counts[global_index] = result[3], result[4]
    invalid = [meta["stems"][i] for i, source in enumerate(sources) if source == "invalid" or source == "not_processed"]
    if invalid:
        raise RuntimeError(f"Sequential footprint estimation failed for: {invalid[:8]}")
    return {
        "centers": centers, "bbox_mins": bbox_mins, "bbox_maxs": bbox_maxs,
        "meta": {
            "estimation": "sequential", "coordinate_axes": [0, 1],
            "alignment_mode": align_mode,
            "source_counts": {s: sources.count(s) for s in set(sources)},
            "sources": sources, "point_counts": point_counts,
            "alignments": alignments,
        },
    }
