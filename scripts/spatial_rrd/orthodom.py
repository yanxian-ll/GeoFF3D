# -*- coding: utf-8 -*-
"""Orthographic-like DOM rendering from spatial chunk predictions using gsplat.

This module builds 3D Gaussians from predicted world point maps and renders
a top-down DOM in tiles. The final DOM is stitched into one image.

Two modes:
  1. Fused final point cloud → top-down DOM/DSM by max-elevation rasterization.
  2. Optimized 3DGS gaussians.npz → top-down DOM/DSM (if --gsplat_refine enabled).

Assumptions:
  - pred_maps / gaussians are in a metric/world coordinate system.
  - default horizontal plane is XY, up axis is Z.
  - gsplat only provides perspective rasterization, so we approximate
    orthographic rendering using a high top-down pinhole camera.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from spatial_rrd.chunk_cache import (
    get_cached_colors,
    get_cached_sequence,
)
from spatial_rrd.chunk_transform import (
    get_transformed_cached_point_maps,
    get_transformed_cached_points,
    get_transformed_cameras,
)
from spatial_rrd.dsm_visualization import (
    save_dsm_contour_visualization,
    save_dsm_elevation_visualization,
)
from spatial_rrd.rrd_writer import load_point_cloud_ply


_AXIS_TO_IDX = {"x": 0, "y": 1, "z": 2}


def _axis_unit(axis: str) -> np.ndarray:
    out = np.zeros(3, dtype=np.float32)
    out[_AXIS_TO_IDX[axis]] = 1.0
    return out


def _parse_dom_axes(axes: str, up_axis: str) -> Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    axes = str(axes).lower().strip()
    up_axis = str(up_axis).lower().strip()

    if len(axes) != 2 or axes[0] not in _AXIS_TO_IDX or axes[1] not in _AXIS_TO_IDX:
        raise ValueError(f"--dom_axes must look like 'xy', 'xz', or 'yz', got {axes!r}")
    if axes[0] == axes[1]:
        raise ValueError(f"--dom_axes cannot use duplicate axes: {axes!r}")
    if up_axis not in _AXIS_TO_IDX:
        raise ValueError(f"--dom_up_axis must be x/y/z, got {up_axis!r}")
    if up_axis in axes:
        raise ValueError(
            f"--dom_up_axis={up_axis!r} must not be inside --dom_axes={axes!r}"
        )

    u_idx = _AXIS_TO_IDX[axes[0]]
    v_idx = _AXIS_TO_IDX[axes[1]]
    up_idx = _AXIS_TO_IDX[up_axis]

    u_vec = _axis_unit(axes[0])
    v_vec = _axis_unit(axes[1])
    up_vec = _axis_unit(up_axis)
    return u_idx, v_idx, up_idx, u_vec, v_vec, up_vec


def _safe_uint8_rgb(rgb01: np.ndarray) -> np.ndarray:
    rgb01 = np.asarray(rgb01, dtype=np.float32)
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    return (rgb01 * 255.0 + 0.5).astype(np.uint8)


def _resize_rgb_to_hw(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    if rgb.shape[:2] != (int(h), int(w)):
        rgb = cv2.resize(rgb, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    return rgb


def _collect_dom_points_from_records(
    chunk_records: Sequence[Dict[str, object]],
    source: str = "core",
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """Collect point/color samples from chunk records.

    source:
      - core: use only each chunk's core local indices. Recommended.
      - all: use all views inside each chunk. May include duplicated seam views.
    """
    source = str(source).lower().strip()
    if source not in {"core", "all"}:
        raise ValueError(f"unknown dom source: {source}")

    pts_all: List[np.ndarray] = []
    cols_all: List[np.ndarray] = []
    records_meta: List[Dict[str, object]] = []

    for record in chunk_records:
        pred_maps = get_transformed_cached_point_maps(record, "_pred_maps")
        pred_valid_masks = get_cached_sequence(record, "_pred_valid_masks")
        rgbs = get_cached_sequence(record, "rgbs")

        if not pred_maps:
            # Fallback: current code always has core_pred_points/colors.
            p = get_transformed_cached_points(record, "core_pred_points")
            c = get_cached_colors(record, "core_pred_colors")
            if p.size > 0 and c.size > 0:
                pts_all.append(p.reshape(-1, 3))
                cols_all.append(c.reshape(-1, 3))
            continue

        if source == "core":
            local_indices = [int(i) for i in get_cached_sequence(record, "_core_local_indices")]
            if not local_indices:
                local_indices = list(range(len(pred_maps)))
        else:
            local_indices = list(range(len(pred_maps)))

        chunk_pts = 0
        for local_i in local_indices:
            local_i = int(local_i)
            if local_i < 0 or local_i >= len(pred_maps):
                continue

            pmap = np.asarray(pred_maps[local_i], dtype=np.float32)
            if pmap.ndim != 3 or pmap.shape[-1] != 3:
                continue

            mask = np.asarray(pred_valid_masks[local_i], dtype=bool)
            if mask.shape != pmap.shape[:2]:
                continue

            rgb = np.asarray(rgbs[local_i])
            rgb = _resize_rgb_to_hw(rgb, pmap.shape[0], pmap.shape[1])

            valid = mask & np.isfinite(pmap).all(axis=-1)
            if not valid.any():
                continue

            pts = pmap[valid].reshape(-1, 3).astype(np.float32)
            cols = rgb[valid].reshape(-1, 3).astype(np.uint8)

            pts_all.append(pts)
            cols_all.append(cols)
            chunk_pts += int(pts.shape[0])

        records_meta.append(
            {
                "chunk_id": int(record.get("chunk_id", -1)),
                "num_points_used": int(chunk_pts),
                "source": source,
            }
        )

    if not pts_all:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            records_meta,
        )

    pts = np.concatenate(pts_all, axis=0).astype(np.float32)
    cols = np.concatenate(cols_all, axis=0).astype(np.uint8)

    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    cols = cols[finite]

    return pts, cols, records_meta


def _collect_optimized_gaussians_from_summary(
    gsplat_summary: Dict[str, object],
) -> Dict[str, np.ndarray]:
    """Load optimized 3DGS bundles and convert local normalized coords to world coords."""
    means_all: List[np.ndarray] = []
    colors_all: List[np.ndarray] = []
    scales_all: List[np.ndarray] = []
    opacities_all: List[np.ndarray] = []
    quats_all: List[np.ndarray] = []

    for bundle_meta in gsplat_summary.get("bundles", []):
        if not isinstance(bundle_meta, dict):
            continue
        if not bool(bundle_meta.get("enabled", False)):
            continue

        npz_path = bundle_meta.get("gaussians_npz", None)
        if not npz_path:
            continue

        npz_path = Path(str(npz_path))
        if not npz_path.exists():
            print(f"[DOM][WARN] optimized gaussians not found: {npz_path}")
            continue

        data = np.load(npz_path)

        means_local = np.asarray(data["means_local"], dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(data["colors"], dtype=np.float32).reshape(-1, 3)

        if "scales_local" in data:
            scales_local = np.asarray(data["scales_local"], dtype=np.float32).reshape(-1, 3)
        else:
            log_scales_local = np.asarray(data["log_scales_local"], dtype=np.float32).reshape(-1, 3)
            scales_local = np.exp(log_scales_local).astype(np.float32)

        if "opacities" in data:
            opacities = np.asarray(data["opacities"], dtype=np.float32).reshape(-1)
        else:
            logits = np.asarray(data["opacity_logits"], dtype=np.float32).reshape(-1)
            opacities = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

        quats = np.asarray(data["quats"], dtype=np.float32).reshape(-1, 4)

        world_center = np.asarray(data["world_center"], dtype=np.float32).reshape(1, 3)
        world_scale = float(np.asarray(data["world_scale"], dtype=np.float32))

        means_world = means_local * world_scale + world_center
        scales_world = scales_local * world_scale

        finite = (
            np.isfinite(means_world).all(axis=1)
            & np.isfinite(colors).all(axis=1)
            & np.isfinite(scales_world).all(axis=1)
            & np.isfinite(opacities)
            & np.isfinite(quats).all(axis=1)
            & (opacities > 1e-4)
        )

        if not finite.any():
            continue

        means_all.append(means_world[finite].astype(np.float32))
        colors_all.append(np.clip(colors[finite], 0.0, 1.0).astype(np.float32))
        scales_all.append(scales_world[finite].astype(np.float32))
        opacities_all.append(opacities[finite].astype(np.float32))
        quats_all.append(quats[finite].astype(np.float32))

    if not means_all:
        return {
            "means_world": np.empty((0, 3), dtype=np.float32),
            "colors": np.empty((0, 3), dtype=np.float32),
            "scales_world": np.empty((0, 3), dtype=np.float32),
            "opacities": np.empty((0,), dtype=np.float32),
            "quats": np.empty((0, 4), dtype=np.float32),
        }

    return {
        "means_world": np.concatenate(means_all, axis=0).astype(np.float32),
        "colors": np.concatenate(colors_all, axis=0).astype(np.float32),
        "scales_world": np.concatenate(scales_all, axis=0).astype(np.float32),
        "opacities": np.concatenate(opacities_all, axis=0).astype(np.float32),
        "quats": np.concatenate(quats_all, axis=0).astype(np.float32),
    }


def _estimate_gsd_from_chunk_records(
    chunk_records: Sequence[Dict[str, object]],
    axes: Tuple[int, int],
    stride: int = 8,
    max_samples: int = 300000,
) -> Tuple[float, Dict[str, object]]:
    """Estimate world-unit-per-pixel GSD from adjacent predicted point samples."""
    stride = max(1, int(stride))
    distances: List[np.ndarray] = []
    total = 0

    for record in chunk_records:
        pred_maps = get_transformed_cached_point_maps(record, "_pred_maps")
        pred_valid_masks = get_cached_sequence(record, "_pred_valid_masks")
        if not pred_maps or not pred_valid_masks:
            continue

        local_indices = [int(i) for i in get_cached_sequence(record, "_core_local_indices")]
        if not local_indices:
            local_indices = list(range(len(pred_maps)))

        for local_i in local_indices:
            pmap = np.asarray(pred_maps[int(local_i)], dtype=np.float32)
            mask = np.asarray(pred_valid_masks[int(local_i)], dtype=bool)
            if pmap.ndim != 3 or pmap.shape[-1] != 3 or mask.shape != pmap.shape[:2]:
                continue

            p = pmap[::stride, ::stride]
            m = mask[::stride, ::stride] & np.isfinite(p).all(axis=-1)

            if p.shape[1] > 1:
                valid_x = m[:, 1:] & m[:, :-1]
                if valid_x.any():
                    d = p[:, 1:, list(axes)] - p[:, :-1, list(axes)]
                    d = np.linalg.norm(d, axis=-1) / float(stride)
                    d = d[valid_x]
                    d = d[np.isfinite(d) & (d > 0)]
                    if d.size:
                        distances.append(d.astype(np.float32))
                        total += int(d.size)

            if p.shape[0] > 1:
                valid_y = m[1:, :] & m[:-1, :]
                if valid_y.any():
                    d = p[1:, :, list(axes)] - p[:-1, :, list(axes)]
                    d = np.linalg.norm(d, axis=-1) / float(stride)
                    d = d[valid_y]
                    d = d[np.isfinite(d) & (d > 0)]
                    if d.size:
                        distances.append(d.astype(np.float32))
                        total += int(d.size)

            if total >= int(max_samples):
                break
        if total >= int(max_samples):
            break

    if not distances:
        return 0.1, {
            "valid": False,
            "reason": "no adjacent valid point pairs; fallback to 0.1",
            "gsd": 0.1,
            "num_samples": 0,
        }

    vals = np.concatenate(distances, axis=0)
    if vals.size > int(max_samples):
        rng = np.random.default_rng(2027)
        vals = vals[rng.choice(vals.size, size=int(max_samples), replace=False)]

    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 0.1, {
            "valid": False,
            "reason": "all adjacent distances invalid; fallback to 0.1",
            "gsd": 0.1,
            "num_samples": 0,
        }

    # Use median because point-map edges and facades can produce large outliers.
    gsd = float(np.median(vals))
    p10 = float(np.percentile(vals, 10.0))
    p90 = float(np.percentile(vals, 90.0))

    return gsd, {
        "valid": True,
        "method": "median_adjacent_world_distance",
        "gsd": float(gsd),
        "p10": p10,
        "p90": p90,
        "num_samples": int(vals.size),
        "stride": int(stride),
    }


def _estimate_gsd_from_fused_points(
    points: np.ndarray,
    axes: Tuple[int, int],
    qmin: float,
    qmax: float,
) -> Tuple[float, Dict[str, object]]:
    p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if p.shape[0] == 0:
        return 0.1, {
            "valid": False,
            "reason": "no finite fused points; fallback to 0.1",
            "gsd": 0.1,
            "num_points": 0,
        }

    u = p[:, int(axes[0])]
    v = p[:, int(axes[1])]
    u_min = float(np.percentile(u, float(qmin)))
    u_max = float(np.percentile(u, float(qmax)))
    v_min = float(np.percentile(v, float(qmin)))
    v_max = float(np.percentile(v, float(qmax)))
    area = max((u_max - u_min) * (v_max - v_min), 0.0)
    if not np.isfinite(area) or area <= 0.0:
        return 0.1, {
            "valid": False,
            "reason": "invalid fused point area; fallback to 0.1",
            "gsd": 0.1,
            "num_points": int(p.shape[0]),
        }

    # One fused point per output pixel is a practical DSM/DOM default for
    # unstructured point clouds. Users can still override via --dom_gsd.
    gsd = float(math.sqrt(area / max(float(p.shape[0]), 1.0)))
    return gsd, {
        "valid": True,
        "method": "sqrt_robust_area_per_fused_point",
        "gsd": float(gsd),
        "num_points": int(p.shape[0]),
        "area": float(area),
        "qmin": float(qmin),
        "qmax": float(qmax),
    }


def _estimate_projected_point_spacing(
    points: np.ndarray,
    axes: Tuple[int, int],
    *,
    max_samples: int = 100000,
    seed: int = 2029,
) -> Tuple[float, Dict[str, object]]:
    p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if p.shape[0] < 2:
        return 0.0, {
            "valid": False,
            "reason": "not enough finite points",
            "num_points": int(p.shape[0]),
        }

    xy = p[:, [int(axes[0]), int(axes[1])]].astype(np.float64)
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    if xy.shape[0] < 2:
        return 0.0, {
            "valid": False,
            "reason": "not enough finite projected points",
            "num_points": int(xy.shape[0]),
        }

    if xy.shape[0] > int(max_samples):
        rng = np.random.default_rng(int(seed))
        keep = rng.choice(xy.shape[0], size=int(max_samples), replace=False)
        xy = xy[keep]

    try:
        from scipy.spatial import cKDTree

        dist, _idx = cKDTree(xy).query(xy, k=2, workers=-1)
        nn = np.asarray(dist[:, 1], dtype=np.float64)
        method = "cKDTree_nearest_neighbor"
    except Exception:
        # Fallback: robust area-per-point spacing. Less local, but avoids
        # requiring scipy in lightweight environments.
        u = xy[:, 0]
        v = xy[:, 1]
        area = (
            float(np.percentile(u, 99.0) - np.percentile(u, 1.0))
            * float(np.percentile(v, 99.0) - np.percentile(v, 1.0))
        )
        spacing = math.sqrt(max(area, 0.0) / max(float(xy.shape[0]), 1.0))
        return spacing, {
            "valid": bool(np.isfinite(spacing) and spacing > 0.0),
            "method": "sqrt_area_per_sampled_point",
            "spacing": float(spacing),
            "num_samples": int(xy.shape[0]),
        }

    nn = nn[np.isfinite(nn) & (nn > 0.0)]
    if nn.size == 0:
        return 0.0, {
            "valid": False,
            "reason": "nearest-neighbor distances are empty",
            "method": method,
            "num_samples": int(xy.shape[0]),
        }

    p50 = float(np.percentile(nn, 50.0))
    p75 = float(np.percentile(nn, 75.0))
    p90 = float(np.percentile(nn, 90.0))
    spacing = p75
    return spacing, {
        "valid": True,
        "method": method,
        "spacing": float(spacing),
        "p50": p50,
        "p75": p75,
        "p90": p90,
        "num_samples": int(xy.shape[0]),
    }


def _disk_offsets(radius_px: int) -> List[Tuple[int, int]]:
    radius_px = max(0, int(radius_px))
    offsets: List[Tuple[int, int]] = []
    r2 = radius_px * radius_px
    for dy in range(-radius_px, radius_px + 1):
        for dx in range(-radius_px, radius_px + 1):
            if dx * dx + dy * dy <= r2:
                offsets.append((dx, dy))
    offsets.sort(key=lambda item: item[0] * item[0] + item[1] * item[1])
    return offsets


def _smooth_dsm_nanaware(
    dsm: np.ndarray,
    *,
    radius_px: int,
    sigma: float = 0.0,
    iterations: int = 1,
    min_weight: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, object]]:
    dsm = np.asarray(dsm, dtype=np.float32)
    meta: Dict[str, object] = {
        "enabled": False,
        "radius_px": int(radius_px),
        "sigma": float(sigma),
        "iterations": int(iterations),
        "min_weight": float(min_weight),
        "valid_pixels_before": int(np.isfinite(dsm).sum()),
        "valid_pixels_after": int(np.isfinite(dsm).sum()),
        "filled_pixels": 0,
    }
    if dsm.ndim != 2 or dsm.size == 0 or int(radius_px) <= 0 or int(iterations) <= 0:
        return dsm.astype(np.float32, copy=True), meta

    current = dsm.astype(np.float32, copy=True)
    original_valid = np.isfinite(current)
    kernel = max(1, int(radius_px) * 2 + 1)
    min_weight = float(np.clip(min_weight, 1e-6, 1.0))

    for _ in range(int(iterations)):
        valid = np.isfinite(current)
        if not bool(valid.any()):
            break
        values = np.where(valid, current, 0.0).astype(np.float32)
        weights = valid.astype(np.float32)

        if float(sigma) > 0.0:
            sum_values = cv2.GaussianBlur(
                values,
                (kernel, kernel),
                float(sigma),
                borderType=cv2.BORDER_REFLECT101,
            )
            sum_weights = cv2.GaussianBlur(
                weights,
                (kernel, kernel),
                float(sigma),
                borderType=cv2.BORDER_REFLECT101,
            )
        else:
            sum_values = cv2.blur(
                values,
                (kernel, kernel),
                borderType=cv2.BORDER_REFLECT101,
            )
            sum_weights = cv2.blur(
                weights,
                (kernel, kernel),
                borderType=cv2.BORDER_REFLECT101,
            )

        next_dsm = np.full_like(current, np.nan, dtype=np.float32)
        smooth_valid = sum_weights >= min_weight
        next_dsm[smooth_valid] = (
            sum_values[smooth_valid] / np.maximum(sum_weights[smooth_valid], 1e-6)
        ).astype(np.float32)
        current = next_dsm

    final_valid = np.isfinite(current)
    meta.update(
        {
            "enabled": True,
            "valid_pixels_after": int(final_valid.sum()),
            "filled_pixels": int((final_valid & ~original_valid).sum()),
            "coverage_before": float(original_valid.mean()),
            "coverage_after": float(final_valid.mean()),
        }
    )
    return current.astype(np.float32), meta


def _load_fused_points(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".ply":
        return load_point_cloud_ply(path)

    with np.load(path, allow_pickle=True) as data:
        points = np.asarray(data["points"], dtype=np.float32).reshape(-1, 3)
        if "colors" in data:
            colors = np.asarray(data["colors"], dtype=np.uint8).reshape(-1, 3)
        else:
            colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)
    if colors.shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    return points.astype(np.float32), colors.astype(np.uint8)


def _robust_bounds(
    points: np.ndarray,
    u_idx: int,
    v_idx: int,
    qmin: float,
    qmax: float,
    padding: float,
) -> Tuple[float, float, float, float]:
    p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if p.shape[0] == 0:
        raise RuntimeError("Cannot compute DOM bounds: no finite points.")

    u = p[:, u_idx]
    v = p[:, v_idx]
    u_min = float(np.percentile(u, float(qmin)))
    u_max = float(np.percentile(u, float(qmax)))
    v_min = float(np.percentile(v, float(qmin)))
    v_max = float(np.percentile(v, float(qmax)))

    if not np.isfinite([u_min, u_max, v_min, v_max]).all() or u_max <= u_min or v_max <= v_min:
        raise RuntimeError("Invalid DOM bounds from point cloud.")

    pad = float(max(padding, 0.0))
    return u_min - pad, u_max + pad, v_min - pad, v_max + pad


def _camera_centers_from_chunk_records(
    chunk_records: Sequence[Dict[str, object]],
) -> np.ndarray:
    centers: List[np.ndarray] = []
    for record in chunk_records:
        seen: set[str] = set()
        for cam in get_transformed_cameras(record):
            if not isinstance(cam, dict):
                continue
            stem = str(cam.get("stem", cam.get("pred_index", len(seen))))
            if stem in seen:
                continue
            T = np.asarray(cam.get("T_c2w", None), dtype=np.float32)
            if T.shape != (4, 4) or not np.isfinite(T).all():
                continue
            centers.append(T[:3, 3].astype(np.float32))
            seen.add(stem)

    if not centers:
        return np.empty((0, 3), dtype=np.float32)
    return np.stack(centers, axis=0).astype(np.float32)


def _camera_centers_from_gsplat_summary(
    gsplat_summary: Dict[str, object],
) -> np.ndarray:
    centers: List[np.ndarray] = []
    seen: set[Tuple[str, int]] = set()

    for bundle_i, bundle_meta in enumerate(gsplat_summary.get("bundles", [])):
        if not isinstance(bundle_meta, dict):
            continue
        if not bool(bundle_meta.get("enabled", False)):
            continue

        cam_npz_path = bundle_meta.get("cameras_npz", None)
        if not cam_npz_path:
            continue

        cam_npz_path = Path(str(cam_npz_path))
        if not cam_npz_path.exists():
            continue

        data = np.load(cam_npz_path)
        if "T_c2w_world" not in data:
            continue

        Ts = np.asarray(data["T_c2w_world"], dtype=np.float32).reshape(-1, 4, 4)
        global_indices = data.get("global_indices", np.arange(Ts.shape[0]))
        global_indices = np.asarray(global_indices).reshape(-1)

        for i, T in enumerate(Ts):
            global_i = int(global_indices[i]) if i < global_indices.shape[0] else int(i)
            key = (str(bundle_i), global_i)
            if key in seen:
                continue
            if T.shape == (4, 4) and np.isfinite(T).all():
                centers.append(T[:3, 3].astype(np.float32))
                seen.add(key)

    if not centers:
        return np.empty((0, 3), dtype=np.float32)
    return np.stack(centers, axis=0).astype(np.float32)


def _estimate_camera_height_from_centers(
    camera_centers: np.ndarray,
    *,
    up_idx: int,
    z_ref: float,
) -> Tuple[float, Dict[str, object]]:
    centers = np.asarray(camera_centers, dtype=np.float32).reshape(-1, 3)
    if centers.shape[0] == 0:
        raise RuntimeError("Cannot estimate DOM camera height: no valid camera centers.")

    up = centers[:, int(up_idx)]
    signed = up - float(z_ref)
    signed = signed[np.isfinite(signed)]
    positive = signed[signed > 1e-6]

    if positive.size > 0:
        height = float(np.median(positive))
        return height, {
            "method": "median_camera_center_minus_z_ref",
            "height": height,
            "num_cameras": int(centers.shape[0]),
            "num_positive": int(positive.size),
            "p10": float(np.percentile(positive, 10.0)),
            "p50": height,
            "p90": float(np.percentile(positive, 90.0)),
            "z_ref": float(z_ref),
        }

    abs_height = np.abs(signed)
    abs_height = abs_height[np.isfinite(abs_height) & (abs_height > 1e-6)]
    if abs_height.size > 0:
        height = float(np.median(abs_height))
        return height, {
            "method": "median_abs_camera_center_minus_z_ref",
            "reason": "no positive camera height; used absolute offset",
            "height": height,
            "num_cameras": int(centers.shape[0]),
            "z_ref": float(z_ref),
        }

    raise RuntimeError(
        "Cannot estimate DOM camera height: camera centers have zero/invalid "
        "height relative to z_ref."
    )


def _make_topdown_c2w_local(
    camera_height: float,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    up_vec: np.ndarray,
) -> np.ndarray:
    """Build local c2w for a top-down pinhole camera.

    Camera convention:
      +x = image right = +u
      +y = image down  = -v
      +z = forward     = -up
    """
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = u_vec.astype(np.float32)
    c2w[:3, 1] = -v_vec.astype(np.float32)
    c2w[:3, 2] = -up_vec.astype(np.float32)
    c2w[:3, 3] = up_vec.astype(np.float32) * float(camera_height)
    return c2w


def _sample_tile_indices_stratified(
    pts: np.ndarray,
    *,
    tile_u_min: float,
    tile_v_max: float,
    gsd: float,
    width: int,
    height: int,
    u_idx: int,
    v_idx: int,
    max_gaussians: int,
    seed: int,
    cell_px: int = 2,
) -> np.ndarray:
    """Grid-stratified sampling for DOM tile.

    Random sampling easily creates holes. This keeps spatial coverage first.
    """
    n = int(pts.shape[0])
    if max_gaussians <= 0 or n <= int(max_gaussians):
        return np.arange(n, dtype=np.int64)

    u = pts[:, u_idx]
    v = pts[:, v_idx]

    px = np.floor((u - float(tile_u_min)) / max(float(gsd), 1e-9)).astype(np.int64)
    py = np.floor((float(tile_v_max) - v) / max(float(gsd), 1e-9)).astype(np.int64)

    px = np.clip(px, 0, int(width) - 1)
    py = np.clip(py, 0, int(height) - 1)

    cell_px = max(1, int(cell_px))
    gx = px // cell_px
    gy = py // cell_px
    grid_w = int(np.ceil(float(width) / float(cell_px)))
    linear = gy * grid_w + gx

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)

    _, first = np.unique(linear[order], return_index=True)
    keep = order[first]

    max_gaussians = int(max_gaussians)
    if keep.size > max_gaussians:
        keep = rng.choice(keep, size=max_gaussians, replace=False)
    elif keep.size < max_gaussians:
        used = np.zeros(n, dtype=bool)
        used[keep] = True
        rest = np.nonzero(~used)[0]
        need = min(max_gaussians - keep.size, rest.size)
        if need > 0:
            add = rng.choice(rest, size=need, replace=False)
            keep = np.concatenate([keep, add], axis=0)

    return keep.astype(np.int64)


def _render_dom_tile_gsplat(
    *,
    points_world: np.ndarray,
    colors_u8: np.ndarray,
    tile_u_min: float,
    tile_u_max: float,
    tile_v_min: float,
    tile_v_max: float,
    z_ref: float,
    camera_height: float,
    gsd: float,
    width: int,
    height: int,
    u_idx: int,
    v_idx: int,
    up_idx: int,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    up_vec: np.ndarray,
    scale: float,
    opacity: float,
    max_gaussians: int,
    seed: int,
    device: torch.device,
    rasterize_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one DOM tile and return RGB01, alpha, DSM."""
    try:
        from gsplat.rendering import rasterization  # type: ignore[import-untyped]
    except Exception as e:
        raise RuntimeError(
            "--render_dom requires gsplat. Install it first, e.g. `pip install gsplat`."
        ) from e

    pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    cols = np.asarray(colors_u8, dtype=np.uint8).reshape(-1, 3)

    # Local origin: tile center in u/v and z_ref in up axis.
    origin = np.zeros(3, dtype=np.float32)
    origin[u_idx] = 0.5 * (float(tile_u_min) + float(tile_u_max))
    origin[v_idx] = 0.5 * (float(tile_v_min) + float(tile_v_max))
    origin[up_idx] = float(z_ref)

    pts_local = (pts - origin[None, :]).astype(np.float32)

    num_candidates = int(pts_local.shape[0])
    subsampled = False
    if max_gaussians > 0 and num_candidates > int(max_gaussians):
        print(
            f"[DOM][WARN] Subsampling raw tile Gaussians: "
            f"{num_candidates} -> {int(max_gaussians)}. "
            f"This may create holes. Set --dom_max_gaussians_per_tile 0 "
            f"to keep all Gaussians."
        )
        rng = np.random.default_rng(int(seed))
        keep = rng.choice(num_candidates, size=int(max_gaussians), replace=False)
        pts_local = pts_local[keep]
        cols = cols[keep]
        subsampled = True

    if pts_local.shape[0] == 0:
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        alpha = np.zeros((height, width), dtype=np.float32)
        dsm = np.full((height, width), np.nan, dtype=np.float32)
        return rgb, alpha, dsm

    means = torch.from_numpy(pts_local).to(device=device, dtype=torch.float32)
    colors = torch.from_numpy(cols.astype(np.float32) / 255.0).to(device=device)

    n = int(means.shape[0])
    quats = torch.zeros((n, 4), device=device, dtype=torch.float32)
    quats[:, 0] = 1.0  # gsplat wxyz identity

    splat_scale = max(float(scale), 1e-6)
    scales = torch.full((n, 3), splat_scale, device=device, dtype=torch.float32)
    opacities = torch.full((n,), float(opacity), device=device, dtype=torch.float32)

    c2w = _make_topdown_c2w_local(
        camera_height=float(camera_height),
        u_vec=u_vec,
        v_vec=v_vec,
        up_vec=up_vec,
    )
    c2w_t = torch.from_numpy(c2w[None]).to(device=device, dtype=torch.float32)
    viewmats = torch.linalg.inv(c2w_t)

    # Pinhole approximation to orthographic:
    # ground pixel size ~= camera_height / focal = gsd
    focal = float(camera_height) / max(float(gsd), 1e-9)
    K = np.asarray(
        [
            [focal, 0.0, (float(width) - 1.0) * 0.5],
            [0.0, focal, (float(height) - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    Ks = torch.from_numpy(K[None]).to(device=device, dtype=torch.float32)

    far = max(float(camera_height) * 2.5, 100.0)

    with torch.no_grad():
        rendering, alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=int(width),
            height=int(height),
            near_plane=0.01,
            far_plane=float(far),
            radius_clip=0.0,
            render_mode="RGB+D",
            packed=True,
            sparse_grad=False,
            rasterize_mode=str(rasterize_mode),
        )

    rgb = rendering[0, ..., :3].detach().float().clamp(0, 1).cpu().numpy()
    depth = rendering[0, ..., 3].detach().float().cpu().numpy()

    if alpha.ndim == 4:
        a = alpha[0, ..., 0].detach().float().cpu().numpy()
    elif alpha.ndim == 3:
        a = alpha[0].detach().float().cpu().numpy()
    else:
        a = alpha.detach().float().cpu().numpy().reshape(height, width)

    # Since camera forward is -up, z/up coordinate = z_ref + camera_height - depth.
    dsm = (float(z_ref) + float(camera_height) - depth).astype(np.float32)
    dsm[~np.isfinite(depth) | (a <= 1e-4)] = np.nan

    return rgb.astype(np.float32), a.astype(np.float32), dsm


def _render_dom_tile_optimized_gsplat(
    *,
    means_world: np.ndarray,
    colors01: np.ndarray,
    scales_world: np.ndarray,
    opacities: np.ndarray,
    quats: np.ndarray,
    tile_u_min: float,
    tile_u_max: float,
    tile_v_min: float,
    tile_v_max: float,
    z_ref: float,
    camera_height: float,
    gsd: float,
    width: int,
    height: int,
    u_idx: int,
    v_idx: int,
    up_idx: int,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    up_vec: np.ndarray,
    scale_multiplier: float,
    opacity_multiplier: float,
    max_gaussians: int,
    seed: int,
    device: torch.device,
    rasterize_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one DOM tile from optimized 3DGS parameters."""
    try:
        from gsplat.rendering import rasterization  # type: ignore[import-untyped]
    except Exception as e:
        raise RuntimeError(
            "--render_dom with --gsplat_refine requires gsplat. "
            "Install it first, e.g. `pip install gsplat`."
        ) from e

    means_world = np.asarray(means_world, dtype=np.float32).reshape(-1, 3)
    colors01 = np.asarray(colors01, dtype=np.float32).reshape(-1, 3)
    scales_world = np.asarray(scales_world, dtype=np.float32).reshape(-1, 3)
    opacities = np.asarray(opacities, dtype=np.float32).reshape(-1)
    quats = np.asarray(quats, dtype=np.float32).reshape(-1, 4)

    num_candidates = int(means_world.shape[0])
    if num_candidates == 0:
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        alpha = np.zeros((height, width), dtype=np.float32)
        dsm = np.full((height, width), np.nan, dtype=np.float32)
        return rgb, alpha, dsm

    subsampled = False
    if max_gaussians > 0 and num_candidates > int(max_gaussians):
        print(
            f"[DOM][WARN] Subsampling optimized tile Gaussians: "
            f"{num_candidates} -> {int(max_gaussians)}. "
            f"This may make optimized DOM worse. Set "
            f"--dom_max_gaussians_per_tile 0 to keep all optimized Gaussians."
        )
        subsampled = True

    keep = _sample_tile_indices_stratified(
        means_world,
        tile_u_min=tile_u_min,
        tile_v_max=tile_v_max,
        gsd=gsd,
        width=width,
        height=height,
        u_idx=u_idx,
        v_idx=v_idx,
        max_gaussians=int(max_gaussians),
        seed=int(seed),
        cell_px=2,
    )

    means_world = means_world[keep]
    colors01 = colors01[keep]
    scales_world = scales_world[keep]
    opacities = opacities[keep]
    quats = quats[keep]

    origin = np.zeros(3, dtype=np.float32)
    origin[u_idx] = 0.5 * (float(tile_u_min) + float(tile_u_max))
    origin[v_idx] = 0.5 * (float(tile_v_min) + float(tile_v_max))
    origin[up_idx] = float(z_ref)

    means_local = (means_world - origin[None, :]).astype(np.float32)

    means = torch.from_numpy(means_local).to(device=device, dtype=torch.float32)
    colors = torch.from_numpy(np.clip(colors01, 0.0, 1.0)).to(device=device, dtype=torch.float32)

    # Optimized scale may be too small. DOM is a top-down composite, allow a multiplier.
    scales = torch.from_numpy(scales_world).to(device=device, dtype=torch.float32)
    scales = (scales * float(scale_multiplier)).clamp_min(1e-6)

    opacities_t = torch.from_numpy(opacities).to(device=device, dtype=torch.float32)
    opacities_t = (opacities_t * float(opacity_multiplier)).clamp(0.0, 1.0)

    quats_t = torch.from_numpy(quats).to(device=device, dtype=torch.float32)
    quats_t = quats_t / quats_t.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    c2w = _make_topdown_c2w_local(
        camera_height=float(camera_height),
        u_vec=u_vec,
        v_vec=v_vec,
        up_vec=up_vec,
    )
    c2w_t = torch.from_numpy(c2w[None]).to(device=device, dtype=torch.float32)
    viewmats = torch.linalg.inv(c2w_t)

    focal = float(camera_height) / max(float(gsd), 1e-9)
    K = np.asarray(
        [
            [focal, 0.0, (float(width) - 1.0) * 0.5],
            [0.0, focal, (float(height) - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    Ks = torch.from_numpy(K[None]).to(device=device, dtype=torch.float32)

    far = max(float(camera_height) * 2.5, 100.0)

    with torch.no_grad():
        rendering, alpha, _ = rasterization(
            means=means,
            quats=quats_t,
            scales=scales,
            opacities=opacities_t,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=int(width),
            height=int(height),
            near_plane=0.01,
            far_plane=float(far),
            radius_clip=0.0,
            render_mode="RGB+D",
            packed=True,
            sparse_grad=False,
            rasterize_mode=str(rasterize_mode),
        )

    rgb = rendering[0, ..., :3].detach().float().clamp(0, 1).cpu().numpy()
    depth = rendering[0, ..., 3].detach().float().cpu().numpy()

    if alpha.ndim == 4:
        a = alpha[0, ..., 0].detach().float().cpu().numpy()
    elif alpha.ndim == 3:
        a = alpha[0].detach().float().cpu().numpy()
    else:
        a = alpha.detach().float().cpu().numpy().reshape(height, width)

    dsm = (float(z_ref) + float(camera_height) - depth).astype(np.float32)
    dsm[~np.isfinite(depth) | (a <= 1e-4)] = np.nan

    return rgb.astype(np.float32), a.astype(np.float32), dsm


def render_orthodom_from_fused_points(
    *,
    fused_points_path: Path,
    output_dir: Path,
    dom_gsd: float = 0.0,
    dom_axes: str = "xy",
    dom_up_axis: str = "z",
    dom_splat_scale: float = 1.0,
    dom_dsm_smooth_radius_px: int = 2,
    dom_dsm_smooth_sigma: float = 0.0,
    dom_dsm_smooth_iterations: int = 1,
    dom_dsm_smooth_min_weight: float = 0.05,
    dom_save_contours: bool = False,
    dom_tile_px: int = 1024,
    dom_bounds_quantile_min: float = 0.5,
    dom_bounds_quantile_max: float = 99.5,
    dom_padding_m: float = 0.0,
    dom_max_pixels: int = 160_000_000,
    dom_allow_large: bool = False,
    dom_save_tiles: bool = True,
    dom_epsg: Optional[int] = None,
) -> Dict[str, object]:
    """Build DOM/DSM directly from the fused final point cloud.

    Each output pixel keeps the highest fused point along dom_up_axis. Its
    height becomes DSM and its color becomes DOM.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fused_points_path = Path(fused_points_path).expanduser().resolve()

    u_idx, v_idx, up_idx, _u_vec, _v_vec, _up_vec = _parse_dom_axes(
        dom_axes,
        dom_up_axis,
    )

    if not fused_points_path.exists():
        meta = {
            "enabled": False,
            "reason": f"fused point cloud not found: {fused_points_path}",
            "output_dir": str(output_dir),
            "source": "fused_points",
            "fused_points_path": str(fused_points_path),
        }
        (output_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return meta

    points, colors = _load_fused_points(fused_points_path)
    if points.shape[0] == 0:
        meta = {
            "enabled": False,
            "reason": "no valid fused points for DOM/DSM",
            "output_dir": str(output_dir),
            "source": "fused_points",
            "fused_points_path": str(fused_points_path),
        }
        (output_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return meta

    if float(dom_gsd) > 0:
        gsd = float(dom_gsd)
        gsd_meta = {
            "valid": True,
            "method": "user",
            "gsd": gsd,
        }
    else:
        gsd, gsd_meta = _estimate_gsd_from_fused_points(
            points,
            axes=(u_idx, v_idx),
            qmin=float(dom_bounds_quantile_min),
            qmax=float(dom_bounds_quantile_max),
        )

    if not np.isfinite(gsd) or gsd <= 0:
        raise RuntimeError(f"Invalid fused-point DOM GSD: {gsd}")

    padding = float(dom_padding_m)
    if padding <= 0:
        padding = max(4.0 * gsd, 0.0)

    u_min, u_max, v_min, v_max = _robust_bounds(
        points=points,
        u_idx=u_idx,
        v_idx=v_idx,
        qmin=float(dom_bounds_quantile_min),
        qmax=float(dom_bounds_quantile_max),
        padding=padding,
    )

    full_w = int(math.ceil((u_max - u_min) / gsd))
    full_h = int(math.ceil((v_max - v_min) / gsd))
    if full_w <= 0 or full_h <= 0:
        raise RuntimeError(f"Invalid fused-point DOM image size: {full_w}x{full_h}")

    num_pixels = int(full_w) * int(full_h)
    if num_pixels > int(dom_max_pixels) and not bool(dom_allow_large):
        raise RuntimeError(
            f"Fused-point DOM would be too large: {full_w}x{full_h}={num_pixels} pixels. "
            f"Increase --dom_gsd, set --dom_max_pixels, or pass --dom_allow_large."
        )

    u = points[:, u_idx]
    v = points[:, v_idx]
    z = points[:, up_idx]
    px = np.floor((u - float(u_min)) / max(float(gsd), 1e-12)).astype(np.int64)
    py = np.floor((float(v_max) - v) / max(float(gsd), 1e-12)).astype(np.int64)
    in_canvas = (
        (px >= 0)
        & (px < int(full_w))
        & (py >= 0)
        & (py < int(full_h))
        & np.isfinite(z)
    )
    if not bool(in_canvas.any()):
        raise RuntimeError("No fused points fall inside DOM canvas.")

    px = px[in_canvas]
    py = py[in_canvas]
    z = z[in_canvas].astype(np.float32)
    colors_in = colors[in_canvas]

    spacing, spacing_meta = _estimate_projected_point_spacing(
        points[in_canvas],
        axes=(u_idx, v_idx),
    )
    if not np.isfinite(spacing) or spacing <= 0.0:
        spacing = float(gsd)
    if float(dom_splat_scale) <= 0.0:
        splat_radius_px = 0
    else:
        splat_radius_px = int(
            math.ceil(max(1.0, float(dom_splat_scale) * float(spacing) / float(gsd)))
        )
    splat_radius_px = int(np.clip(splat_radius_px, 0, 8))
    splat_offsets = _disk_offsets(splat_radius_px)

    rgb_canvas = np.zeros((full_h, full_w, 3), dtype=np.uint8)
    alpha_canvas = np.zeros((full_h, full_w), dtype=np.uint8)
    dsm_canvas = np.full((full_h, full_w), np.nan, dtype=np.float32)

    flat_rgb = rgb_canvas.reshape(-1, 3)
    flat_alpha = alpha_canvas.reshape(-1)
    flat_dsm = dsm_canvas.reshape(-1)

    num_candidate_splats = 0
    for dx, dy in splat_offsets:
        tx = px + int(dx)
        ty = py + int(dy)
        valid = (tx >= 0) & (tx < int(full_w)) & (ty >= 0) & (ty < int(full_h))
        if not bool(valid.any()):
            continue

        target_linear = (ty[valid] * int(full_w) + tx[valid]).astype(np.int64)
        src_idx = np.nonzero(valid)[0]
        z_valid = z[src_idx].astype(np.float64)
        order = np.lexsort((-z_valid, target_linear))
        unique_linear, first = np.unique(target_linear[order], return_index=True)
        chosen = src_idx[order[first]]
        update = (~np.isfinite(flat_dsm[unique_linear])) | (
            z[chosen] > flat_dsm[unique_linear]
        )
        if bool(update.any()):
            dst = unique_linear[update]
            src = chosen[update]
            flat_rgb[dst] = colors_in[src]
            flat_alpha[dst] = 255
            flat_dsm[dst] = z[src]
        num_candidate_splats += int(target_linear.shape[0])

    dsm_canvas, dsm_smoothing_meta = _smooth_dsm_nanaware(
        dsm_canvas,
        radius_px=int(dom_dsm_smooth_radius_px),
        sigma=float(dom_dsm_smooth_sigma),
        iterations=int(dom_dsm_smooth_iterations),
        min_weight=float(dom_dsm_smooth_min_weight),
    )

    z_vals = z[np.isfinite(z)]
    z_ref = float(np.nanmedian(z_vals)) if z_vals.size else 0.0
    z_min = float(np.nanpercentile(z_vals, 1.0)) if z_vals.size else z_ref
    z_max = float(np.nanpercentile(z_vals, 99.0)) if z_vals.size else z_ref

    tile_meta: List[Dict[str, object]] = []
    tile_px = max(128, int(dom_tile_px))
    num_tiles_x = int(math.ceil(full_w / tile_px))
    num_tiles_y = int(math.ceil(full_h / tile_px))
    tiles_dir = output_dir / "tiles"
    if bool(dom_save_tiles):
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for ty in range(num_tiles_y):
            y0 = ty * tile_px
            y1 = min(full_h, y0 + tile_px)
            for tx in range(num_tiles_x):
                x0 = tx * tile_px
                x1 = min(full_w, x0 + tile_px)
                rgb_tile = rgb_canvas[y0:y1, x0:x1]
                alpha_tile = alpha_canvas[y0:y1, x0:x1]
                cv2.imwrite(
                    str(tiles_dir / f"tile_y{ty:04d}_x{tx:04d}.png"),
                    cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(tiles_dir / f"tile_y{ty:04d}_x{tx:04d}_alpha.png"),
                    alpha_tile,
                )
                tile_meta.append(
                    {
                        "tx": int(tx),
                        "ty": int(ty),
                        "x0": int(x0),
                        "y0": int(y0),
                        "width": int(x1 - x0),
                        "height": int(y1 - y0),
                        "num_pixels": int((alpha_tile > 0).sum()),
                        "skipped": False,
                    }
                )

    rgb_path = output_dir / "dom_rgb.png"
    alpha_path = output_dir / "dom_alpha.png"
    dsm_path = output_dir / "dom_dsm.npy"
    meta_path = output_dir / "meta.json"

    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_canvas, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(alpha_path), alpha_canvas)
    np.save(str(dsm_path), dsm_canvas.astype(np.float32))
    dsm_vis_meta = save_dsm_elevation_visualization(
        dsm_canvas,
        output_dir=output_dir,
        u_min=float(u_min),
        u_max=float(u_max),
        v_min=float(v_min),
        v_max=float(v_max),
        dom_axes=str(dom_axes),
    )
    dsm_contour_meta: Dict[str, object] = {"saved": False, "disabled": True}
    if bool(dom_save_contours):
        dsm_contour_meta = save_dsm_contour_visualization(
            dsm_canvas,
            output_dir=output_dir,
            u_min=float(u_min),
            u_max=float(u_max),
            v_min=float(v_min),
            v_max=float(v_max),
            dom_axes=str(dom_axes),
        )

    geotiff_path = None
    if dom_epsg is not None:
        try:
            import rasterio
            from rasterio.transform import from_origin

            geotiff_path = output_dir / "dom_rgb.tif"
            transform = from_origin(float(u_min), float(v_max), float(gsd), float(gsd))
            with rasterio.open(
                geotiff_path,
                "w",
                driver="GTiff",
                height=int(full_h),
                width=int(full_w),
                count=3,
                dtype=rgb_canvas.dtype,
                crs=f"EPSG:{int(dom_epsg)}",
                transform=transform,
                compress="deflate",
            ) as dst:
                dst.write(rgb_canvas[:, :, 0], 1)
                dst.write(rgb_canvas[:, :, 1], 2)
                dst.write(rgb_canvas[:, :, 2], 3)
        except Exception as e:
            print(f"[DOM][WARN] Failed to save fused-point GeoTIFF: {e}")
            geotiff_path = None

    alpha_valid = alpha_canvas > 0
    alpha_coverage = float(alpha_valid.mean())
    num_occupied_pixels = int(alpha_valid.sum())
    meta = {
        "enabled": True,
        "method": "fused_point_topdown_max_elevation_splat",
        "output_dir": str(output_dir),
        "source": "fused_points",
        "fused_points_path": str(fused_points_path),
        "rgb_path": str(rgb_path),
        "alpha_path": str(alpha_path),
        "dsm_npy_path": str(dsm_path),
        "dsm_elevation_png_path": dsm_vis_meta.get("png_path", None),
        "dsm_elevation_svg_path": dsm_vis_meta.get("svg_path", None),
        "dsm_elevation_visualization": dsm_vis_meta,
        "dsm_contour_png_path": dsm_contour_meta.get("png_path", None),
        "dsm_contour_svg_path": dsm_contour_meta.get("svg_path", None),
        "dsm_contour_visualization": dsm_contour_meta,
        "geotiff_path": str(geotiff_path) if geotiff_path is not None else None,
        "dom_axes": str(dom_axes),
        "dom_up_axis": str(dom_up_axis),
        "gsd": float(gsd),
        "gsd_meta": gsd_meta,
        "point_spacing": float(spacing),
        "point_spacing_meta": spacing_meta,
        "splat_scale": float(dom_splat_scale),
        "splat_radius_px": int(splat_radius_px),
        "num_splat_offsets": int(len(splat_offsets)),
        "num_candidate_splats": int(num_candidate_splats),
        "dsm_smoothing": dsm_smoothing_meta,
        "alpha_coverage": float(alpha_coverage),
        "bbox": {
            "u_min": float(u_min),
            "u_max": float(u_max),
            "v_min": float(v_min),
            "v_max": float(v_max),
            "z_ref": float(z_ref),
            "z_min_p01": float(z_min),
            "z_max_p99": float(z_max),
        },
        "image_size": {
            "width": int(full_w),
            "height": int(full_h),
        },
        "tile_px": int(tile_px),
        "num_tiles_x": int(num_tiles_x),
        "num_tiles_y": int(num_tiles_y),
        "num_fused_points": int(points.shape[0]),
        "num_points_in_canvas": int(in_canvas.sum()),
        "num_occupied_pixels": int(num_occupied_pixels),
        "tiles": tile_meta,
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[DOM] saved fused-point DOM/DSM: rgb={rgb_path}, "
        f"dsm={dsm_path}, occupied={num_occupied_pixels}/{num_pixels}, "
        f"coverage={alpha_coverage:.4f}, "
        f"spacing={float(spacing):.6g}, radius_px={int(splat_radius_px)}"
    )
    print(f"[DOM] saved fused-point meta: {meta_path}")
    return meta


def render_orthodom_from_chunks(
    *,
    chunk_records: Sequence[Dict[str, object]],
    output_dir: Path,
    device: torch.device,
    dom_gsd: float = 0.0,
    dom_axes: str = "xy",
    dom_up_axis: str = "z",
    dom_source: str = "core",
    dom_tile_px: int = 1024,
    dom_margin_px: int = 32,
    dom_max_gaussians_per_tile: int = 500000,
    dom_splat_scale: float = 0.7,
    dom_opacity: float = 0.95,
    dom_gsd_stride: int = 8,
    dom_bounds_quantile_min: float = 0.5,
    dom_bounds_quantile_max: float = 99.5,
    dom_padding_m: float = 0.0,
    dom_max_pixels: int = 160_000_000,
    dom_allow_large: bool = False,
    dom_rasterize_mode: str = "classic",
    dom_save_tiles: bool = True,
    dom_save_contours: bool = False,
    dom_epsg: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    u_idx, v_idx, up_idx, u_vec, v_vec, up_vec = _parse_dom_axes(dom_axes, dom_up_axis)

    points, colors, source_meta = _collect_dom_points_from_records(
        chunk_records=chunk_records,
        source=dom_source,
    )
    if points.shape[0] == 0:
        meta = {
            "enabled": False,
            "reason": "no valid predicted points for DOM",
            "output_dir": str(output_dir),
        }
        (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    if float(dom_gsd) > 0:
        gsd = float(dom_gsd)
        gsd_meta = {
            "valid": True,
            "method": "user",
            "gsd": gsd,
        }
    else:
        gsd, gsd_meta = _estimate_gsd_from_chunk_records(
            chunk_records=chunk_records,
            axes=(u_idx, v_idx),
            stride=int(dom_gsd_stride),
        )

    if not np.isfinite(gsd) or gsd <= 0:
        raise RuntimeError(f"Invalid DOM GSD: {gsd}")

    padding = float(dom_padding_m)
    if padding <= 0:
        padding = max(4.0 * gsd, 0.0)

    u_min, u_max, v_min, v_max = _robust_bounds(
        points=points,
        u_idx=u_idx,
        v_idx=v_idx,
        qmin=float(dom_bounds_quantile_min),
        qmax=float(dom_bounds_quantile_max),
        padding=padding,
    )

    full_w = int(math.ceil((u_max - u_min) / gsd))
    full_h = int(math.ceil((v_max - v_min) / gsd))

    if full_w <= 0 or full_h <= 0:
        raise RuntimeError(f"Invalid DOM image size: {full_w}x{full_h}")

    num_pixels = int(full_w) * int(full_h)
    if num_pixels > int(dom_max_pixels) and not bool(dom_allow_large):
        raise RuntimeError(
            f"DOM would be too large: {full_w}x{full_h}={num_pixels} pixels. "
            f"Increase --dom_gsd, set --dom_max_pixels, or pass --dom_allow_large."
        )

    z_vals = points[:, up_idx]
    z_vals = z_vals[np.isfinite(z_vals)]
    z_ref = float(np.nanmedian(z_vals)) if z_vals.size else 0.0
    z_min = float(np.nanpercentile(z_vals, 1.0)) if z_vals.size else z_ref
    z_max = float(np.nanpercentile(z_vals, 99.0)) if z_vals.size else z_ref
    relief = max(z_max - z_min, 1.0)

    tile_px = max(128, int(dom_tile_px))
    margin_world = max(0.0, float(dom_margin_px) * float(gsd))

    camera_height, camera_height_meta = _estimate_camera_height_from_centers(
        _camera_centers_from_chunk_records(chunk_records),
        up_idx=up_idx,
        z_ref=z_ref,
    )

    rgb_canvas = np.zeros((full_h, full_w, 3), dtype=np.uint8)
    alpha_canvas = np.zeros((full_h, full_w), dtype=np.uint8)
    dsm_canvas = np.full((full_h, full_w), np.nan, dtype=np.float32)

    tiles_dir = output_dir / "tiles"
    if bool(dom_save_tiles):
        tiles_dir.mkdir(parents=True, exist_ok=True)

    tile_meta: List[Dict[str, object]] = []

    num_tiles_x = int(math.ceil(full_w / tile_px))
    num_tiles_y = int(math.ceil(full_h / tile_px))

    print(
        f"[DOM] rendering DOM: size={full_w}x{full_h}, "
        f"gsd={gsd:.6g}, tiles={num_tiles_x}x{num_tiles_y}, "
        f"points={points.shape[0]}, output={output_dir}"
    )

    u_all = points[:, u_idx]
    v_all = points[:, v_idx]

    for ty in range(num_tiles_y):
        y0 = ty * tile_px
        y1 = min(full_h, y0 + tile_px)
        tile_h = y1 - y0

        # row 0 corresponds to v_max
        tile_v_max = v_max - float(y0) * gsd
        tile_v_min = v_max - float(y1) * gsd

        for tx in range(num_tiles_x):
            x0 = tx * tile_px
            x1 = min(full_w, x0 + tile_px)
            tile_w = x1 - x0

            tile_u_min = u_min + float(x0) * gsd
            tile_u_max = u_min + float(x1) * gsd

            select = (
                (u_all >= tile_u_min - margin_world)
                & (u_all <= tile_u_max + margin_world)
                & (v_all >= tile_v_min - margin_world)
                & (v_all <= tile_v_max + margin_world)
                & np.isfinite(points).all(axis=1)
            )
            idx = np.nonzero(select)[0]
            if idx.size == 0:
                tile_meta.append(
                    {
                        "tx": int(tx),
                        "ty": int(ty),
                        "x0": int(x0),
                        "y0": int(y0),
                        "width": int(tile_w),
                        "height": int(tile_h),
                        "num_points": 0,
                        "skipped": True,
                    }
                )
                continue

            rgb01, alpha, dsm = _render_dom_tile_gsplat(
                points_world=points[idx],
                colors_u8=colors[idx],
                tile_u_min=tile_u_min,
                tile_u_max=tile_u_max,
                tile_v_min=tile_v_min,
                tile_v_max=tile_v_max,
                z_ref=z_ref,
                camera_height=camera_height,
                gsd=gsd,
                width=tile_w,
                height=tile_h,
                u_idx=u_idx,
                v_idx=v_idx,
                up_idx=up_idx,
                u_vec=u_vec,
                v_vec=v_vec,
                up_vec=up_vec,
                scale=float(gsd) * float(dom_splat_scale),
                opacity=float(dom_opacity),
                max_gaussians=int(dom_max_gaussians_per_tile),
                seed=int(seed) + ty * 10007 + tx,
                device=device,
                rasterize_mode=dom_rasterize_mode,
            )

            rgb_tile_u8 = _safe_uint8_rgb(rgb01)
            alpha_u8 = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)

            rgb_canvas[y0:y1, x0:x1] = rgb_tile_u8
            alpha_canvas[y0:y1, x0:x1] = alpha_u8
            dsm_canvas[y0:y1, x0:x1] = dsm

            if bool(dom_save_tiles):
                cv2.imwrite(
                    str(tiles_dir / f"tile_y{ty:04d}_x{tx:04d}.png"),
                    cv2.cvtColor(rgb_tile_u8, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(tiles_dir / f"tile_y{ty:04d}_x{tx:04d}_alpha.png"),
                    alpha_u8,
                )

            tile_meta.append(
                {
                    "tx": int(tx),
                    "ty": int(ty),
                    "x0": int(x0),
                    "y0": int(y0),
                    "width": int(tile_w),
                    "height": int(tile_h),
                    "num_points": int(idx.size),
                    "skipped": False,
                }
            )

        print(f"[DOM] row {ty + 1}/{num_tiles_y} done")

    rgb_path = output_dir / "dom_rgb.png"
    alpha_path = output_dir / "dom_alpha.png"
    dsm_path = output_dir / "dom_dsm.npy"
    meta_path = output_dir / "meta.json"

    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_canvas, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(alpha_path), alpha_canvas)
    np.save(str(dsm_path), dsm_canvas.astype(np.float32))
    dsm_vis_meta = save_dsm_elevation_visualization(
        dsm_canvas,
        output_dir=output_dir,
        u_min=float(u_min),
        u_max=float(u_max),
        v_min=float(v_min),
        v_max=float(v_max),
        dom_axes=str(dom_axes),
    )
    dsm_contour_meta: Dict[str, object] = {"saved": False, "disabled": True}
    if bool(dom_save_contours):
        dsm_contour_meta = save_dsm_contour_visualization(
            dsm_canvas,
            output_dir=output_dir,
            u_min=float(u_min),
            u_max=float(u_max),
            v_min=float(v_min),
            v_max=float(v_max),
            dom_axes=str(dom_axes),
        )

    geotiff_path = None
    if dom_epsg is not None:
        try:
            import rasterio
            from rasterio.transform import from_origin

            geotiff_path = output_dir / "dom_rgb.tif"
            transform = from_origin(float(u_min), float(v_max), float(gsd), float(gsd))
            with rasterio.open(
                geotiff_path,
                "w",
                driver="GTiff",
                height=int(full_h),
                width=int(full_w),
                count=3,
                dtype=rgb_canvas.dtype,
                crs=f"EPSG:{int(dom_epsg)}",
                transform=transform,
                compress="deflate",
            ) as dst:
                # rasterio expects band-first RGB
                dst.write(rgb_canvas[:, :, 0], 1)
                dst.write(rgb_canvas[:, :, 1], 2)
                dst.write(rgb_canvas[:, :, 2], 3)
        except Exception as e:
            print(f"[DOM][WARN] Failed to save GeoTIFF: {e}")
            geotiff_path = None

    meta = {
        "enabled": True,
        "method": "gsplat_topdown_pinhole_orthodom",
        "output_dir": str(output_dir),
        "rgb_path": str(rgb_path),
        "alpha_path": str(alpha_path),
        "dsm_npy_path": str(dsm_path),
        "dsm_elevation_png_path": dsm_vis_meta.get("png_path", None),
        "dsm_elevation_svg_path": dsm_vis_meta.get("svg_path", None),
        "dsm_elevation_visualization": dsm_vis_meta,
        "dsm_contour_png_path": dsm_contour_meta.get("png_path", None),
        "dsm_contour_svg_path": dsm_contour_meta.get("svg_path", None),
        "dsm_contour_visualization": dsm_contour_meta,
        "geotiff_path": str(geotiff_path) if geotiff_path is not None else None,
        "dom_axes": str(dom_axes),
        "dom_up_axis": str(dom_up_axis),
        "source": str(dom_source),
        "gsd": float(gsd),
        "gsd_meta": gsd_meta,
        "bbox": {
            "u_min": float(u_min),
            "u_max": float(u_max),
            "v_min": float(v_min),
            "v_max": float(v_max),
            "z_ref": float(z_ref),
            "z_min_p01": float(z_min),
            "z_max_p99": float(z_max),
        },
        "image_size": {
            "width": int(full_w),
            "height": int(full_h),
        },
        "tile_px": int(tile_px),
        "num_tiles_x": int(num_tiles_x),
        "num_tiles_y": int(num_tiles_y),
        "num_points": int(points.shape[0]),
        "camera_height": float(camera_height),
        "camera_height_meta": camera_height_meta,
        "splat_scale": float(gsd) * float(dom_splat_scale),
        "opacity": float(dom_opacity),
        "max_gaussians_per_tile": int(dom_max_gaussians_per_tile),
        "records": source_meta,
        "tiles": tile_meta,
    }

    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[DOM] saved RGB:   {rgb_path}")
    print(f"[DOM] saved alpha: {alpha_path}")
    print(f"[DOM] saved DSM:   {dsm_path}")
    print(f"[DOM] saved meta:  {meta_path}")

    return meta


def render_orthodom_from_gsplat_summary(
    *,
    gsplat_summary: Dict[str, object],
    fallback_chunk_records: Sequence[Dict[str, object]],
    output_dir: Path,
    device: torch.device,
    fallback_fused_points_path: Optional[Path] = None,
    dom_gsd: float = 0.0,
    dom_axes: str = "xy",
    dom_up_axis: str = "z",
    dom_tile_px: int = 1024,
    dom_margin_px: int = 32,
    dom_max_gaussians_per_tile: int = 500000,
    dom_splat_scale: float = 2.0,
    dom_dsm_smooth_radius_px: int = 2,
    dom_dsm_smooth_sigma: float = 0.0,
    dom_dsm_smooth_iterations: int = 1,
    dom_dsm_smooth_min_weight: float = 0.05,
    dom_save_contours: bool = False,
    dom_opacity: float = 1.0,
    dom_gsd_stride: int = 8,
    dom_bounds_quantile_min: float = 0.5,
    dom_bounds_quantile_max: float = 99.5,
    dom_padding_m: float = 0.0,
    dom_max_pixels: int = 160_000_000,
    dom_allow_large: bool = False,
    dom_rasterize_mode: str = "classic",
    dom_save_tiles: bool = True,
    dom_epsg: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, object]:
    """Render DOM from optimized 3DGS bundles.

    Falls back to the fused final point cloud if no optimized Gaussians are
    available and a fused point-cloud path is provided.
    """
    gs = _collect_optimized_gaussians_from_summary(gsplat_summary)
    means = gs["means_world"]
    if means.shape[0] == 0:
        if fallback_fused_points_path is not None:
            print("[DOM][WARN] No optimized Gaussians found; fallback to fused-point DOM/DSM.")
            return render_orthodom_from_fused_points(
                fused_points_path=Path(fallback_fused_points_path),
                output_dir=output_dir,
                dom_gsd=dom_gsd,
                dom_axes=dom_axes,
                dom_up_axis=dom_up_axis,
                dom_splat_scale=dom_splat_scale,
                dom_dsm_smooth_radius_px=dom_dsm_smooth_radius_px,
                dom_dsm_smooth_sigma=dom_dsm_smooth_sigma,
                dom_dsm_smooth_iterations=dom_dsm_smooth_iterations,
                dom_dsm_smooth_min_weight=dom_dsm_smooth_min_weight,
                dom_save_contours=dom_save_contours,
                dom_tile_px=dom_tile_px,
                dom_bounds_quantile_min=dom_bounds_quantile_min,
                dom_bounds_quantile_max=dom_bounds_quantile_max,
                dom_padding_m=dom_padding_m,
                dom_max_pixels=dom_max_pixels,
                dom_allow_large=dom_allow_large,
                dom_save_tiles=dom_save_tiles,
                dom_epsg=dom_epsg,
            )
        print("[DOM][WARN] No optimized Gaussians found; fallback to raw chunk DOM.")
        return render_orthodom_from_chunks(
            chunk_records=fallback_chunk_records,
            output_dir=output_dir,
            device=device,
            dom_gsd=dom_gsd,
            dom_axes=dom_axes,
            dom_up_axis=dom_up_axis,
            dom_source="all",
            dom_tile_px=dom_tile_px,
            dom_margin_px=dom_margin_px,
            dom_max_gaussians_per_tile=dom_max_gaussians_per_tile,
            dom_splat_scale=dom_splat_scale,
            dom_opacity=dom_opacity,
            dom_gsd_stride=dom_gsd_stride,
            dom_bounds_quantile_min=dom_bounds_quantile_min,
            dom_bounds_quantile_max=dom_bounds_quantile_max,
            dom_padding_m=dom_padding_m,
            dom_max_pixels=dom_max_pixels,
            dom_allow_large=dom_allow_large,
            dom_rasterize_mode=dom_rasterize_mode,
            dom_save_tiles=dom_save_tiles,
            dom_save_contours=dom_save_contours,
            dom_epsg=dom_epsg,
            seed=seed,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    u_idx, v_idx, up_idx, u_vec, v_vec, up_vec = _parse_dom_axes(dom_axes, dom_up_axis)

    # GSD is still best estimated from raw dense point maps, since optimized
    # Gaussian means are sampled points whose nearest-neighbor distances
    # may not match the true image GSD.
    if float(dom_gsd) > 0:
        gsd = float(dom_gsd)
        gsd_meta = {"valid": True, "method": "user", "gsd": gsd}
    else:
        gsd, gsd_meta = _estimate_gsd_from_chunk_records(
            chunk_records=fallback_chunk_records,
            axes=(u_idx, v_idx),
            stride=int(dom_gsd_stride),
        )

    if not np.isfinite(gsd) or gsd <= 0:
        raise RuntimeError(f"Invalid DOM GSD: {gsd}")

    padding = float(dom_padding_m)
    if padding <= 0:
        padding = max(4.0 * gsd, 0.0)

    u_min, u_max, v_min, v_max = _robust_bounds(
        points=means,
        u_idx=u_idx,
        v_idx=v_idx,
        qmin=float(dom_bounds_quantile_min),
        qmax=float(dom_bounds_quantile_max),
        padding=padding,
    )

    full_w = int(math.ceil((u_max - u_min) / gsd))
    full_h = int(math.ceil((v_max - v_min) / gsd))
    if full_w <= 0 or full_h <= 0:
        raise RuntimeError(f"Invalid DOM image size: {full_w}x{full_h}")

    num_pixels = int(full_w) * int(full_h)
    if num_pixels > int(dom_max_pixels) and not bool(dom_allow_large):
        raise RuntimeError(
            f"DOM would be too large: {full_w}x{full_h}={num_pixels} pixels. "
            f"Increase --dom_gsd, set --dom_max_pixels, or pass --dom_allow_large."
        )

    z_vals = means[:, up_idx]
    z_vals = z_vals[np.isfinite(z_vals)]
    z_ref = float(np.nanmedian(z_vals)) if z_vals.size else 0.0
    z_min = float(np.nanpercentile(z_vals, 1.0)) if z_vals.size else z_ref
    z_max = float(np.nanpercentile(z_vals, 99.0)) if z_vals.size else z_ref
    relief = max(z_max - z_min, 1.0)

    tile_px = max(128, int(dom_tile_px))
    margin_world = max(0.0, float(dom_margin_px) * float(gsd))

    camera_centers = _camera_centers_from_gsplat_summary(gsplat_summary)
    if camera_centers.shape[0] == 0:
        camera_centers = _camera_centers_from_chunk_records(fallback_chunk_records)
    camera_height, camera_height_meta = _estimate_camera_height_from_centers(
        camera_centers,
        up_idx=up_idx,
        z_ref=z_ref,
    )

    rgb_canvas = np.zeros((full_h, full_w, 3), dtype=np.uint8)
    alpha_canvas = np.zeros((full_h, full_w), dtype=np.uint8)
    dsm_canvas = np.full((full_h, full_w), np.nan, dtype=np.float32)

    tiles_dir = output_dir / "tiles"
    if bool(dom_save_tiles):
        tiles_dir.mkdir(parents=True, exist_ok=True)

    tile_meta: List[Dict[str, object]] = []

    num_tiles_x = int(math.ceil(full_w / tile_px))
    num_tiles_y = int(math.ceil(full_h / tile_px))

    print(
        f"[DOM] rendering optimized gsplat DOM: size={full_w}x{full_h}, "
        f"gsd={gsd:.6g}, tiles={num_tiles_x}x{num_tiles_y}, "
        f"gaussians={means.shape[0]}, output={output_dir}"
    )

    u_all = means[:, u_idx]
    v_all = means[:, v_idx]

    for ty in range(num_tiles_y):
        y0 = ty * tile_px
        y1 = min(full_h, y0 + tile_px)
        tile_h = y1 - y0

        tile_v_max = v_max - float(y0) * gsd
        tile_v_min = v_max - float(y1) * gsd

        for tx in range(num_tiles_x):
            x0 = tx * tile_px
            x1 = min(full_w, x0 + tile_px)
            tile_w = x1 - x0

            tile_u_min = u_min + float(x0) * gsd
            tile_u_max = u_min + float(x1) * gsd

            select = (
                (u_all >= tile_u_min - margin_world)
                & (u_all <= tile_u_max + margin_world)
                & (v_all >= tile_v_min - margin_world)
                & (v_all <= tile_v_max + margin_world)
                & np.isfinite(means).all(axis=1)
            )
            idx = np.nonzero(select)[0]

            if idx.size == 0:
                tile_meta.append(
                    {
                        "tx": int(tx),
                        "ty": int(ty),
                        "x0": int(x0),
                        "y0": int(y0),
                        "width": int(tile_w),
                        "height": int(tile_h),
                        "num_gaussians": 0,
                        "skipped": True,
                    }
                )
                continue

            rgb01, alpha, dsm = _render_dom_tile_optimized_gsplat(
                means_world=means[idx],
                colors01=gs["colors"][idx],
                scales_world=gs["scales_world"][idx],
                opacities=gs["opacities"][idx],
                quats=gs["quats"][idx],
                tile_u_min=tile_u_min,
                tile_u_max=tile_u_max,
                tile_v_min=tile_v_min,
                tile_v_max=tile_v_max,
                z_ref=z_ref,
                camera_height=camera_height,
                gsd=gsd,
                width=tile_w,
                height=tile_h,
                u_idx=u_idx,
                v_idx=v_idx,
                up_idx=up_idx,
                u_vec=u_vec,
                v_vec=v_vec,
                up_vec=up_vec,
                scale_multiplier=float(dom_splat_scale),
                opacity_multiplier=float(dom_opacity),
                max_gaussians=int(dom_max_gaussians_per_tile),
                seed=int(seed) + ty * 10007 + tx,
                device=device,
                rasterize_mode=dom_rasterize_mode,
            )

            rgb_tile_u8 = _safe_uint8_rgb(rgb01)
            alpha_u8 = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)

            rgb_canvas[y0:y1, x0:x1] = rgb_tile_u8
            alpha_canvas[y0:y1, x0:x1] = alpha_u8
            dsm_canvas[y0:y1, x0:x1] = dsm

            if bool(dom_save_tiles):
                cv2.imwrite(
                    str(tiles_dir / f"tile_y{ty:04d}_x{tx:04d}.png"),
                    cv2.cvtColor(rgb_tile_u8, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(tiles_dir / f"tile_y{ty:04d}_x{tx:04d}_alpha.png"),
                    alpha_u8,
                )

            tile_meta.append(
                {
                    "tx": int(tx),
                    "ty": int(ty),
                    "x0": int(x0),
                    "y0": int(y0),
                    "width": int(tile_w),
                    "height": int(tile_h),
                    "num_gaussians": int(idx.size),
                    "skipped": False,
                }
            )

        print(f"[DOM] optimized row {ty + 1}/{num_tiles_y} done")

    rgb_path = output_dir / "dom_rgb.png"
    alpha_path = output_dir / "dom_alpha.png"
    dsm_path = output_dir / "dom_dsm.npy"
    meta_path = output_dir / "meta.json"

    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_canvas, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(alpha_path), alpha_canvas)
    np.save(str(dsm_path), dsm_canvas.astype(np.float32))
    dsm_vis_meta = save_dsm_elevation_visualization(
        dsm_canvas,
        output_dir=output_dir,
        u_min=float(u_min),
        u_max=float(u_max),
        v_min=float(v_min),
        v_max=float(v_max),
        dom_axes=str(dom_axes),
    )
    dsm_contour_meta: Dict[str, object] = {"saved": False, "disabled": True}
    if bool(dom_save_contours):
        dsm_contour_meta = save_dsm_contour_visualization(
            dsm_canvas,
            output_dir=output_dir,
            u_min=float(u_min),
            u_max=float(u_max),
            v_min=float(v_min),
            v_max=float(v_max),
            dom_axes=str(dom_axes),
        )

    alpha_valid = alpha_canvas > 8
    alpha_coverage = float(alpha_valid.mean())
    print(
        f"[DOM] optimized alpha coverage={alpha_coverage:.4f}, "
        f"valid_pixels={int(alpha_valid.sum())}/{int(alpha_valid.size)}"
    )

    geotiff_path = None
    if dom_epsg is not None:
        try:
            import rasterio
            from rasterio.transform import from_origin

            geotiff_path = output_dir / "dom_rgb.tif"
            transform = from_origin(float(u_min), float(v_max), float(gsd), float(gsd))
            with rasterio.open(
                geotiff_path,
                "w",
                driver="GTiff",
                height=int(full_h),
                width=int(full_w),
                count=3,
                dtype=rgb_canvas.dtype,
                crs=f"EPSG:{int(dom_epsg)}",
                transform=transform,
                compress="deflate",
            ) as dst:
                dst.write(rgb_canvas[:, :, 0], 1)
                dst.write(rgb_canvas[:, :, 1], 2)
                dst.write(rgb_canvas[:, :, 2], 3)
        except Exception as e:
            print(f"[DOM][WARN] Failed to save GeoTIFF: {e}")
            geotiff_path = None

    meta = {
        "enabled": True,
        "method": "optimized_gsplat_topdown_pinhole_orthodom",
        "output_dir": str(output_dir),
        "rgb_path": str(rgb_path),
        "alpha_path": str(alpha_path),
        "dsm_npy_path": str(dsm_path),
        "dsm_elevation_png_path": dsm_vis_meta.get("png_path", None),
        "dsm_elevation_svg_path": dsm_vis_meta.get("svg_path", None),
        "dsm_elevation_visualization": dsm_vis_meta,
        "dsm_contour_png_path": dsm_contour_meta.get("png_path", None),
        "dsm_contour_svg_path": dsm_contour_meta.get("svg_path", None),
        "dsm_contour_visualization": dsm_contour_meta,
        "geotiff_path": str(geotiff_path) if geotiff_path is not None else None,
        "dom_axes": str(dom_axes),
        "dom_up_axis": str(dom_up_axis),
        "source": "optimized_gsplat",
        "gsd": float(gsd),
        "gsd_meta": gsd_meta,
        "alpha_coverage": float(alpha_coverage),
        "bbox": {
            "u_min": float(u_min),
            "u_max": float(u_max),
            "v_min": float(v_min),
            "v_max": float(v_max),
            "z_ref": float(z_ref),
            "z_min_p01": float(z_min),
            "z_max_p99": float(z_max),
        },
        "image_size": {
            "width": int(full_w),
            "height": int(full_h),
        },
        "tile_px": int(tile_px),
        "num_tiles_x": int(num_tiles_x),
        "num_tiles_y": int(num_tiles_y),
        "num_gaussians": int(means.shape[0]),
        "camera_height": float(camera_height),
        "camera_height_meta": camera_height_meta,
        "scale_multiplier": float(dom_splat_scale),
        "opacity_multiplier": float(dom_opacity),
        "max_gaussians_per_tile": int(dom_max_gaussians_per_tile),
        "tiles": tile_meta,
    }

    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[DOM] saved optimized RGB:   {rgb_path}")
    print(f"[DOM] saved optimized alpha: {alpha_path}")
    print(f"[DOM] saved optimized DSM:   {dsm_path}")
    print(f"[DOM] saved optimized meta:  {meta_path}")

    return meta
