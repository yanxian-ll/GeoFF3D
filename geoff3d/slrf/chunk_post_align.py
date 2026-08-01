# -*- coding: utf-8 -*-
"""Deferred hierarchical post-alignment for spatial chunk predictions.

This module runs after all feed-forward chunks have been predicted. It aligns
neighboring chunk/tree nodes using overlapping seam-view geometry, and records
the resulting per-chunk transforms. Dense chunk caches stay in the raw model
coordinate frame; downstream stages apply transforms on demand.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

from geoff3d.slrf.geometry_align import (
    apply_similarity_to_points,
    estimate_similarity_umeyama,
)
from geoff3d.slrf.chunk_cache import (
    get_cached_points,
    get_cached_sequence,
)
from geoff3d.slrf.chunk_transform import (
    apply_record_similarity_to_points,
    compose_record_similarity,
    ensure_record_similarity,
    get_record_similarity,
)


@dataclass
class _AlignNode:
    key: Tuple[int, ...]
    record_ids: Set[int]
    global_indices: Set[int]
    global_to_record_ids: Dict[int, Set[int]]


_RuntimeCache = Dict[Tuple[int, str], object]


def _progress_bar(total: int, enabled: bool):
    if enabled and tqdm is not None and total > 0:
        return tqdm(
            total=int(total),
            desc="Chunk post-align",
            unit="edge",
            dynamic_ncols=True,
            file=sys.__stderr__,
        )
    return None


def _as_key(x, fallback: int) -> Tuple[int, ...]:
    try:
        key = tuple(int(v) for v in x)
        return key if key else (int(fallback),)
    except Exception:
        return (int(fallback),)


def _record_index_map(record: Dict[str, object]) -> Dict[int, int]:
    cached = record.get("_global_to_local_index", None)
    if isinstance(cached, dict):
        return cached
    mapping = {int(g): int(l) for l, g in enumerate(record.get("indices", []))}
    record["_global_to_local_index"] = mapping
    return mapping


def _make_leaf_node(
    key: Tuple[int, ...],
    rid: int,
    record: Dict[str, object],
) -> _AlignNode:
    indices = [int(i) for i in record.get("indices", [])]
    g2r: Dict[int, Set[int]] = {}
    for gi in indices:
        g2r.setdefault(int(gi), set()).add(int(rid))
    return _AlignNode(
        key=key,
        record_ids={int(rid)},
        global_indices=set(indices),
        global_to_record_ids=g2r,
    )


def _copy_node_with_key(node: _AlignNode, key: Tuple[int, ...]) -> _AlignNode:
    return _AlignNode(
        key=key,
        record_ids=set(node.record_ids),
        global_indices=set(node.global_indices),
        global_to_record_ids={
            int(gi): set(int(rid) for rid in rids)
            for gi, rids in node.global_to_record_ids.items()
        },
    )


def _merge_node_index_inplace(anchor: _AlignNode, child: _AlignNode) -> None:
    anchor.record_ids.update(child.record_ids)
    anchor.global_indices.update(child.global_indices)
    for gi, rids in child.global_to_record_ids.items():
        anchor.global_to_record_ids.setdefault(int(gi), set()).update(
            int(rid) for rid in rids
        )


def _cached_sequence(
    record: Dict[str, object],
    rid: int,
    key: str,
    runtime_cache: Optional[_RuntimeCache],
) -> List[object]:
    cache_key = (int(rid), str(key))
    if runtime_cache is not None and cache_key in runtime_cache:
        value = runtime_cache[cache_key]
    else:
        value = get_cached_sequence(record, key)
        if runtime_cache is not None:
            runtime_cache[cache_key] = value
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return [value[i] for i in range(value.shape[0])]
    return list(value)


def _cached_points(
    record: Dict[str, object],
    rid: int,
    key: str,
    runtime_cache: Optional[_RuntimeCache],
) -> np.ndarray:
    cache_key = (int(rid), str(key))
    if runtime_cache is not None and cache_key in runtime_cache:
        return np.asarray(runtime_cache[cache_key], dtype=np.float32).reshape(-1, 3)
    value = get_cached_points(record, key)
    if runtime_cache is not None:
        runtime_cache[cache_key] = value
    return value


def _node_global_indices(
    records: Sequence[Dict[str, object]],
    node: _AlignNode,
) -> Set[int]:
    return set(node.global_indices)


def _select_node_view_prediction(
    records: Sequence[Dict[str, object]],
    node: _AlignNode,
    global_idx: int,
    runtime_cache: Optional[_RuntimeCache],
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, np.ndarray, np.ndarray]]]:
    """Return point map and valid mask for a view in a node.

    Prefer core predictions over seam predictions because core views are the
    predictions used for final aggregation.
    """
    best = None
    best_score = -1

    candidate_rids = node.global_to_record_ids.get(int(global_idx), set())
    if not candidate_rids:
        return None

    for rid in sorted(candidate_rids):
        record = records[int(rid)]
        index_map = _record_index_map(record)
        local_i = index_map.get(int(global_idx), None)
        if local_i is None:
            continue
        pred_maps = _cached_sequence(record, int(rid), "_pred_maps", runtime_cache)
        pred_masks = _cached_sequence(
            record,
            int(rid),
            "_pred_valid_masks",
            runtime_cache,
        )
        if not pred_maps or not pred_masks:
            continue
        if local_i >= len(pred_maps) or local_i >= len(pred_masks):
            continue

        point_map = np.asarray(pred_maps[local_i], dtype=np.float32)
        valid_mask = np.asarray(pred_masks[local_i], dtype=bool)
        if point_map.ndim != 3 or point_map.shape[-1] != 3:
            continue
        if valid_mask.shape != point_map.shape[:2]:
            continue

        score = 1 if int(global_idx) in set(record.get("core_indices", [])) else 0
        if score > best_score:
            best = (point_map, valid_mask, get_record_similarity(record))
            best_score = score

    return best


def _sample_indices(
    n: int,
    max_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0:
        return np.empty((0,), dtype=np.int64)
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=int(max_n), replace=False).astype(np.int64)


def _spatially_balanced_sample_correspondences(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    rng: np.random.Generator,
    enabled: bool,
    grid_size: int,
    max_points_total: int,
    min_points_per_cell: int,
    max_points_per_cell: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Balance correspondence sampling over the ground XY plane.

    Bins are computed from correspondence midpoints in the current aligned
    coordinate frame. This prevents dense repeated views at one location from
    dominating the final transform estimate.
    """
    src = np.asarray(src, dtype=np.float32).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float32).reshape(-1, 3)
    n = int(min(src.shape[0], dst.shape[0]))
    src = src[:n]
    dst = dst[:n]

    meta: Dict[str, object] = {
        "enabled": bool(enabled),
        "num_corr_before": int(n),
        "num_corr_after": int(n),
    }
    if not enabled or n <= 0:
        meta["reason"] = "disabled" if not enabled else "empty"
        return src, dst, meta

    grid = int(grid_size)
    if grid <= 1:
        meta["reason"] = "grid_size <= 1"
        return src, dst, meta

    mid_xy = 0.5 * (src[:, :2].astype(np.float64) + dst[:, :2].astype(np.float64))
    finite = np.isfinite(mid_xy).all(axis=1)
    if not bool(finite.any()):
        meta["reason"] = "no finite XY midpoints"
        return src, dst, meta

    src = src[finite]
    dst = dst[finite]
    mid_xy = mid_xy[finite]
    n = int(src.shape[0])

    xy_min = np.min(mid_xy, axis=0)
    xy_max = np.max(mid_xy, axis=0)
    span = xy_max - xy_min
    if not np.isfinite(span).all() or float(np.max(span)) <= 1e-9:
        meta.update(
            {
                "reason": "degenerate XY span",
                "num_corr_after": int(n),
                "num_finite_corr": int(n),
            }
        )
        return src, dst, meta

    # Avoid division by zero for line-like distributions.
    span = np.maximum(span, 1e-9)
    rel = (mid_xy - xy_min[None, :]) / span[None, :]
    bins = np.floor(rel * float(grid)).astype(np.int64)
    bins = np.clip(bins, 0, grid - 1)
    cell_ids = bins[:, 0] * grid + bins[:, 1]
    unique_cells = np.unique(cell_ids)

    explicit_cap = int(max_points_per_cell)
    if explicit_cap > 0:
        per_cell_cap = explicit_cap
    elif int(max_points_total) > 0:
        per_cell_cap = int(
            max(
                int(min_points_per_cell),
                np.ceil(float(max_points_total) / float(max(grid * grid, 1))),
            )
        )
    else:
        per_cell_cap = max(int(min_points_per_cell), 1)
    per_cell_cap = max(1, int(per_cell_cap))

    selected_parts: List[np.ndarray] = []
    cell_counts: List[int] = []
    for cell_id in unique_cells:
        ids = np.flatnonzero(cell_ids == int(cell_id))
        cell_counts.append(int(ids.size))
        take = _sample_indices(ids.size, per_cell_cap, rng)
        selected_parts.append(ids[take])

    if not selected_parts:
        meta["reason"] = "no occupied cells"
        return src, dst, meta

    selected = np.concatenate(selected_parts, axis=0).astype(np.int64, copy=False)
    if selected.size > 1:
        selected = rng.permutation(selected)

    meta.update(
        {
            "reason": "balanced",
            "grid_size": int(grid),
            "num_occupied_cells": int(unique_cells.size),
            "per_cell_cap": int(per_cell_cap),
            "min_cell_count_before": int(min(cell_counts)) if cell_counts else 0,
            "max_cell_count_before": int(max(cell_counts)) if cell_counts else 0,
            "num_finite_corr": int(n),
            "num_corr_after": int(selected.size),
        }
    )
    return src[selected], dst[selected], meta


def _collect_common_view_correspondences(
    records: Sequence[Dict[str, object]],
    anchor: _AlignNode,
    child: _AlignNode,
    rng: np.random.Generator,
    max_points_per_view: int,
    max_points_total: int,
    spatial_balance: bool,
    spatial_grid_size: int,
    spatial_min_points_per_cell: int,
    spatial_max_points_per_cell: int,
    runtime_cache: Optional[_RuntimeCache],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Collect one-to-one correspondences from common seam/overlap views.

    Returns src points from child node and dst points from anchor node.
    """
    anchor_indices = _node_global_indices(records, anchor)
    child_indices = _node_global_indices(records, child)
    common = sorted(anchor_indices & child_indices)

    src_all: List[np.ndarray] = []
    dst_all: List[np.ndarray] = []
    used_views = 0

    for global_idx in common:
        a = _select_node_view_prediction(records, anchor, int(global_idx), runtime_cache)
        b = _select_node_view_prediction(records, child, int(global_idx), runtime_cache)
        if a is None or b is None:
            continue

        map_a, mask_a, tfm_a = a
        map_b, mask_b, tfm_b = b
        if map_a.shape != map_b.shape or mask_a.shape != mask_b.shape:
            continue

        valid = (
            mask_a
            & mask_b
            & np.isfinite(map_a).all(axis=-1)
            & np.isfinite(map_b).all(axis=-1)
        )
        flat_ids = np.flatnonzero(valid.reshape(-1))
        if flat_ids.size == 0:
            continue

        take_local = _sample_indices(
            flat_ids.size, int(max_points_per_view), rng
        )
        flat_ids = flat_ids[take_local]

        pts_a = map_a.reshape(-1, 3)[flat_ids]
        pts_b = map_b.reshape(-1, 3)[flat_ids]
        pts_a = apply_similarity_to_points(pts_a, *tfm_a)
        pts_b = apply_similarity_to_points(pts_b, *tfm_b)

        dst_all.append(pts_a.astype(np.float32, copy=False))
        src_all.append(pts_b.astype(np.float32, copy=False))
        used_views += 1

    if not src_all:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            {
                "source": "common_views",
                "num_common_views": int(len(common)),
                "num_used_views": 0,
                "num_corr": 0,
            },
        )

    src = np.concatenate(src_all, axis=0)
    dst = np.concatenate(dst_all, axis=0)

    src, dst, spatial_meta = _spatially_balanced_sample_correspondences(
        src,
        dst,
        rng=rng,
        enabled=bool(spatial_balance),
        grid_size=int(spatial_grid_size),
        max_points_total=int(max_points_total),
        min_points_per_cell=int(spatial_min_points_per_cell),
        max_points_per_cell=int(spatial_max_points_per_cell),
    )

    if max_points_total > 0 and src.shape[0] > int(max_points_total):
        take = _sample_indices(src.shape[0], int(max_points_total), rng)
        src = src[take]
        dst = dst[take]

    return (
        src.astype(np.float32, copy=False),
        dst.astype(np.float32, copy=False),
        {
            "source": "common_views",
            "num_common_views": int(len(common)),
            "num_used_views": int(used_views),
            "num_corr": int(src.shape[0]),
            "spatial_balance": spatial_meta,
        },
    )


def _record_points_for_shared_views(
    record: Dict[str, object],
    rid: int,
    shared_indices: Sequence[int],
    runtime_cache: Optional[_RuntimeCache],
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Collect a bounded, view-balanced overlap sample for one chunk.

    Sampling before concatenation and similarity transformation avoids
    materializing millions of dense overlap pixels only to discard them later.
    """
    index_map = _record_index_map(record)
    pred_maps = _cached_sequence(record, rid, "_pred_maps", runtime_cache)
    pred_masks = _cached_sequence(record, rid, "_pred_valid_masks", runtime_cache)
    parts: List[np.ndarray] = []
    valid_shared = [
        (int(global_idx), index_map.get(int(global_idx), None))
        for global_idx in shared_indices
    ]
    valid_shared = [
        (global_idx, local_idx)
        for global_idx, local_idx in valid_shared
        if local_idx is not None
        and local_idx < len(pred_maps)
        and local_idx < len(pred_masks)
    ]
    per_view_budget = (
        max(1, int(np.ceil(float(max_points) / max(1, len(valid_shared)))))
        if int(max_points) > 0
        else 0
    )
    for _global_idx, local_idx in valid_shared:
        point_map = np.asarray(pred_maps[local_idx], dtype=np.float32)
        valid = np.asarray(pred_masks[local_idx], dtype=bool)
        if point_map.ndim != 3 or point_map.shape[-1] != 3 or valid.shape != point_map.shape[:2]:
            continue
        valid = valid & np.isfinite(point_map).all(axis=-1)
        if bool(valid.any()):
            valid_flat = np.flatnonzero(valid.reshape(-1))
            take = _sample_indices(valid_flat.shape[0], per_view_budget, rng)
            parts.append(point_map.reshape(-1, 3)[valid_flat[take]])
    if not parts:
        return np.empty((0, 3), dtype=np.float32)
    points = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    take = _sample_indices(points.shape[0], int(max_points), rng)
    points = points[take]
    return apply_record_similarity_to_points(record, points).astype(np.float32, copy=False)


def _directed_nn_medians(src: np.ndarray, dst: np.ndarray) -> Tuple[float, float]:
    src = np.asarray(src, dtype=np.float32).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float32).reshape(-1, 3)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return float("nan"), float("nan")
    try:
        from scipy.spatial import cKDTree

        dists, nearest = cKDTree(dst.astype(np.float64)).query(
            src.astype(np.float64), k=1, workers=-1
        )
    except Exception:
        dists_parts: List[np.ndarray] = []
        nearest_parts: List[np.ndarray] = []
        dst64 = dst.astype(np.float64)
        for start in range(0, src.shape[0], 256):
            query = src[start : start + 256].astype(np.float64)
            dist2 = np.sum((query[:, None, :] - dst64[None, :, :]) ** 2, axis=-1)
            nearest_block = np.argmin(dist2, axis=1)
            nearest_parts.append(nearest_block)
            dists_parts.append(np.sqrt(dist2[np.arange(nearest_block.shape[0]), nearest_block]))
        dists = np.concatenate(dists_parts)
        nearest = np.concatenate(nearest_parts)
    finite = np.isfinite(dists)
    if not bool(finite.any()):
        return float("nan"), float("nan")
    z_error = np.abs(src[:, 2].astype(np.float64) - dst[np.asarray(nearest), 2].astype(np.float64))
    finite_z = finite & np.isfinite(z_error)
    return (
        float(np.median(np.asarray(dists)[finite])),
        float(np.median(z_error[finite_z])) if bool(finite_z.any()) else float("nan"),
    )


def compute_adjacent_chunk_seam_error(
    chunk_records: List[Dict[str, object]],
    max_points_per_edge: int = 20000,
    seed: int = 0,
) -> Dict[str, object]:
    """Macro-average symmetric median NN error over adjacent chunk pairs."""
    edge_set: Set[Tuple[int, int]] = set()
    for rid, record in enumerate(chunk_records):
        for neighbor in record.get("adjacent_chunk_ids", []):
            a, b = sorted((int(rid), int(neighbor)))
            if a != b and 0 <= a < len(chunk_records) and 0 <= b < len(chunk_records):
                edge_set.add((a, b))

    per_edge: List[Dict[str, object]] = []
    runtime_cache: _RuntimeCache = {}
    for edge_idx, (a, b) in enumerate(sorted(edge_set)):
        record_a, record_b = chunk_records[a], chunk_records[b]
        shared = sorted(
            {int(v) for v in record_a.get("indices", [])}
            & {int(v) for v in record_b.get("indices", [])}
        )
        rng = np.random.default_rng(int(seed) + 7919 * (edge_idx + 1))
        points_a = _record_points_for_shared_views(
            record_a,
            a,
            shared,
            runtime_cache,
            max_points=int(max_points_per_edge),
            rng=rng,
        )
        points_b = _record_points_for_shared_views(
            record_b,
            b,
            shared,
            runtime_cache,
            max_points=int(max_points_per_edge),
            rng=rng,
        )
        ab, z_ab = _directed_nn_medians(points_a, points_b)
        ba, z_ba = _directed_nn_medians(points_b, points_a)
        seam = 0.5 * (ab + ba) if np.isfinite(ab) and np.isfinite(ba) else float("nan")
        seam_z = 0.5 * (z_ab + z_ba) if np.isfinite(z_ab) and np.isfinite(z_ba) else float("nan")
        per_edge.append(
            {
                "chunk_a": int(a),
                "chunk_b": int(b),
                "num_shared_views": int(len(shared)),
                "num_points_a": int(points_a.shape[0]),
                "num_points_b": int(points_b.shape[0]),
                "median_nn_a_to_b": float(ab),
                "median_nn_b_to_a": float(ba),
                "seam_error": float(seam),
                "median_z_a_to_b": float(z_ab),
                "median_z_b_to_a": float(z_ba),
                "seam_error_z": float(seam_z),
            }
        )
    valid = [item for item in per_edge if np.isfinite(float(item["seam_error"]))]
    valid_z = [item for item in per_edge if np.isfinite(float(item["seam_error_z"]))]
    return {
        "enabled": True,
        "aggregation": "unweighted_macro_average_over_adjacent_chunk_edges",
        "distance": "symmetric_median_nearest_neighbor",
        "num_adjacency_edges": int(len(per_edge)),
        "num_valid_edges": int(len(valid)),
        "max_points_per_edge_per_side": int(max_points_per_edge),
        "seam_error": float(np.mean([item["seam_error"] for item in valid])) if valid else float("nan"),
        "seam_error_z": float(np.mean([item["seam_error_z"] for item in valid_z])) if valid_z else float("nan"),
        "per_edge": per_edge,
    }


def _estimate_yaw_translation_from_3d_correspondences(
    src: np.ndarray,
    dst: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, np.ndarray, np.ndarray, bool, str]:
    """
    Estimate transform mapping src -> dst with only yaw rotation around Z and
    3D translation.

        x_dst ~= Rz(yaw) @ x_src + t

    Assumption:
    - The input and prediction are already in a Z-up world frame.
    - Pitch/roll should not be estimated during post chunk alignment.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)

    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[finite]
    dst = dst[finite]

    if src.shape[0] < 2:
        return (
            0.0,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            False,
            "not enough correspondences to estimate yaw+translation",
        )

    src_xy = src[:, :2]
    dst_xy = dst[:, :2]

    mu_src_xy = np.mean(src_xy, axis=0)
    mu_dst_xy = np.mean(dst_xy, axis=0)

    src_c = src_xy - mu_src_xy[None, :]
    dst_c = dst_xy - mu_dst_xy[None, :]

    src_energy = float(np.sum(src_c * src_c))
    dst_energy = float(np.sum(dst_c * dst_c))
    if src_energy <= eps or dst_energy <= eps:
        # If XY baselines are degenerate, yaw is not observable.
        # Fall back to translation only.
        R = np.eye(3, dtype=np.float64)
        t = np.median(dst - src, axis=0)
        valid = bool(np.isfinite(t).all())
        return (
            0.0,
            R,
            t.astype(np.float64),
            valid,
            "yaw not observable from degenerate XY baselines; fallback to translation",
        )

    # 2D Kabsch closed form.
    # Minimize || R(theta) src_c - dst_c ||^2.
    cross = float(np.sum(src_c[:, 0] * dst_c[:, 1] - src_c[:, 1] * dst_c[:, 0]))
    dot = float(np.sum(src_c[:, 0] * dst_c[:, 0] + src_c[:, 1] * dst_c[:, 1]))
    yaw = float(np.arctan2(cross, dot))

    if not np.isfinite(yaw):
        return (
            0.0,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            False,
            "yaw solve produced non-finite angle",
        )

    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    R = np.asarray(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # 3D translation. This keeps Z translation but forbids pitch/roll.
    src_rot = src @ R.T
    t = np.median(dst - src_rot, axis=0)

    if not np.isfinite(t).all():
        return (
            yaw,
            R,
            np.zeros(3, dtype=np.float64),
            False,
            "yaw+translation solve produced non-finite translation",
        )

    return (
        yaw,
        R,
        t.astype(np.float64),
        True,
        "yaw around Z + 3D translation estimated from overlap geometry",
    )


def _estimate_transform(
    src: np.ndarray,
    dst: np.ndarray,
    mode: str,
    min_corr: int,
) -> Dict[str, object]:
    """Estimate transform mapping src -> dst."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[finite]
    dst = dst[finite]

    if src.shape[0] < int(min_corr):
        return {
            "valid": False,
            "scale": 1.0,
            "R": np.eye(3, dtype=np.float32).tolist(),
            "t": np.zeros(3, dtype=np.float32).tolist(),
            "num_corr": int(src.shape[0]),
            "note": f"not enough correspondences: {src.shape[0]} < {min_corr}",
        }

    mode = str(mode).lower()
    if mode == "translation":
        t = np.median(dst - src, axis=0)
        scale = 1.0
        R = np.eye(3, dtype=np.float64)
        note = "translation estimated from overlap geometry"
        valid = bool(np.isfinite(t).all())
    elif mode in {"yaw_translation", "yaw", "se2"}:
        yaw, R, t, valid, note = _estimate_yaw_translation_from_3d_correspondences(
            src=src,
            dst=dst,
        )
        scale = 1.0
    elif mode == "rigid":
        scale, R, t, valid, note = estimate_similarity_umeyama(
            src=src,
            dst=dst,
            estimate_scale=False,
        )
    elif mode == "sim3":
        scale, R, t, valid, note = estimate_similarity_umeyama(
            src=src,
            dst=dst,
            estimate_scale=True,
        )
    else:
        raise ValueError(
            f"Unknown post chunk align mode: {mode}. "
            "Use translation, yaw_translation, rigid, or sim3."
        )

    if not valid:
        return {
            "valid": False,
            "scale": 1.0,
            "R": np.eye(3, dtype=np.float32).tolist(),
            "t": np.zeros(3, dtype=np.float32).tolist(),
            "num_corr": int(src.shape[0]),
            "note": note,
        }

    transformed = apply_similarity_to_points(src.astype(np.float32), scale, R, t)
    residual = np.linalg.norm(transformed.astype(np.float64) - dst, axis=1)

    yaw_degrees = float("nan")
    if mode in {"yaw_translation", "yaw", "se2"}:
        yaw_degrees = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    return {
        "valid": True,
        "scale": float(scale),
        "R": np.asarray(R, dtype=np.float32).tolist(),
        "t": np.asarray(t, dtype=np.float32).tolist(),
        "num_corr": int(src.shape[0]),
        "yaw_degrees": yaw_degrees,
        "median_residual": float(np.median(residual)) if residual.size else float("nan"),
        "mean_residual": float(np.mean(residual)) if residual.size else float("nan"),
        "note": note,
    }


def _compose_transform_to_record(
    record: Dict[str, object],
    rid: int,
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
    edge_meta: Dict[str, object],
    runtime_cache: Optional[_RuntimeCache],
) -> None:
    """Compose src->dst transform into one chunk record without rewriting cache."""
    compose_record_similarity(record, scale, R, t)

    metas = record.setdefault("post_chunk_align_edges", [])
    if isinstance(metas, list):
        metas.append(edge_meta)


def _merge_child_into_anchor(
    records: Sequence[Dict[str, object]],
    anchor: _AlignNode,
    child: _AlignNode,
    args,
    rng: np.random.Generator,
    level: int,
    runtime_cache: Optional[_RuntimeCache],
) -> Dict[str, object]:
    src, dst, corr_meta = _collect_common_view_correspondences(
        records=records,
        anchor=anchor,
        child=child,
        rng=rng,
        max_points_per_view=int(args.post_chunk_align_max_corr_per_view),
        max_points_total=int(args.post_chunk_align_max_corr),
        spatial_balance=bool(args.post_chunk_align_spatial_balance),
        spatial_grid_size=16,
        spatial_min_points_per_cell=64,
        spatial_max_points_per_cell=0,
        runtime_cache=runtime_cache,
    )

    tfm = _estimate_transform(
        src=src,
        dst=dst,
        mode=str(args.post_chunk_align_mode),
        min_corr=int(args.post_chunk_align_min_corr),
    )

    edge_meta: Dict[str, object] = {
        "level": int(level),
        "anchor_key": tuple(anchor.key),
        "child_key": tuple(child.key),
        "anchor_record_ids": sorted(int(i) for i in anchor.record_ids),
        "child_record_ids": sorted(int(i) for i in child.record_ids),
        "correspondence": corr_meta,
        **tfm,
    }

    if not bool(tfm.get("valid", False)):
        return edge_meta

    scale = float(tfm["scale"])
    R = np.asarray(tfm["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(tfm["t"], dtype=np.float64).reshape(3)

    for rid in sorted(child.record_ids):
        _compose_transform_to_record(
            records[int(rid)],
            rid=int(rid),
            scale=scale,
            R=R,
            t=t,
            edge_meta=edge_meta,
            runtime_cache=runtime_cache,
        )

    return edge_meta


def _stable_key_seed(key: Tuple[int, ...]) -> int:
    value = 2166136261
    for item in key:
        value ^= int(item) & 0xFFFFFFFF
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def _process_alignment_group(
    records: Sequence[Dict[str, object]],
    parent_key: Tuple[int, ...],
    children: Sequence[_AlignNode],
    args,
    base_seed: int,
    level: int,
) -> Tuple[Tuple[int, ...], _AlignNode, List[Dict[str, object]]]:
    children = sorted(
        children,
        key=lambda n: (-len(n.record_ids), n.key),
    )
    anchor = _copy_node_with_key(children[0], parent_key)
    runtime_cache: _RuntimeCache = {}
    rng = np.random.default_rng(
        int(base_seed) + 104729 * int(level + 1) + _stable_key_seed(parent_key)
    )
    edges: List[Dict[str, object]] = []

    for child in children[1:]:
        edge = _merge_child_into_anchor(
            records=records,
            anchor=anchor,
            child=child,
            args=args,
            rng=rng,
            level=level,
            runtime_cache=runtime_cache,
        )
        edges.append(edge)
        # Merge even if transform failed, so the tree can keep moving up.
        _merge_node_index_inplace(anchor, child)

    return parent_key, anchor, edges


def _apply_parent_graph_alignment(
    chunk_records: List[Dict[str, object]],
    args,
    show_progress: bool,
) -> Dict[str, object]:
    """Align a rooted chunk parent graph from leaves toward its root."""
    nodes = {
        rid: _make_leaf_node(key=(rid,), rid=rid, record=record)
        for rid, record in enumerate(chunk_records)
    }
    for record in chunk_records:
        ensure_record_similarity(record)
        record["post_chunk_align_edges"] = []

    parent = {
        rid: record.get("align_parent_id", None)
        for rid, record in enumerate(chunk_records)
    }
    levels = {
        rid: int(record.get("align_level", 0))
        for rid, record in enumerate(chunk_records)
    }
    all_edges: List[Dict[str, object]] = []
    pbar = _progress_bar(max(0, len(nodes) - 1), bool(show_progress))
    base_seed = int(getattr(args, "seed", 0)) + 880031

    try:
        for level in range(max(levels.values(), default=0), 0, -1):
            child_ids = sorted(rid for rid, value in levels.items() if value == level)
            for child_id in child_ids:
                parent_id_raw = parent.get(child_id, None)
                if parent_id_raw is None:
                    continue
                parent_id = int(parent_id_raw)
                if parent_id not in nodes or child_id not in nodes:
                    continue
                anchor = nodes[parent_id]
                child = nodes[child_id]
                runtime_cache: _RuntimeCache = {}
                rng = np.random.default_rng(
                    base_seed + 104729 * (level + 1) + 1009 * child_id
                )
                edge = _merge_child_into_anchor(
                    records=chunk_records,
                    anchor=anchor,
                    child=child,
                    args=args,
                    rng=rng,
                    level=level,
                    runtime_cache=runtime_cache,
                )
                edge["topology"] = "parent_graph"
                edge["parent_record_id"] = parent_id
                edge["child_record_id"] = child_id
                all_edges.append(edge)
                _merge_node_index_inplace(anchor, child)
                nodes.pop(child_id, None)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        level=level,
                        valid=sum(bool(item.get("valid", False)) for item in all_edges),
                        corr=int(edge.get("num_corr", 0)),
                    )
    finally:
        if pbar is not None:
            pbar.close()

    valid_edges = [edge for edge in all_edges if bool(edge.get("valid", False))]
    summary = {
        "enabled": True,
        "mode": str(args.post_chunk_align_mode),
        "topology": "parent_graph",
        "transform_storage": "record_metadata_lazy",
        "cache_rewrites": 0,
        "num_chunks": int(len(chunk_records)),
        "num_edges": int(len(all_edges)),
        "num_valid_edges": int(len(valid_edges)),
        "min_corr": int(args.post_chunk_align_min_corr),
        "workers": 1,
        "edges": all_edges,
    }
    for record in chunk_records:
        record["post_chunk_align_meta"] = {
            "enabled": True,
            "mode": str(args.post_chunk_align_mode),
            "topology": "parent_graph",
            "transform_storage": "record_metadata_lazy",
            "cache_rewrites": 0,
            "num_edges_applied": int(len(record.get("post_chunk_align_edges", []))),
        }
    print(
        "[INFO] Deferred chunk post-alignment: "
        f"mode={summary['mode']}, topology=parent_graph, "
        f"chunks={summary['num_chunks']}, edges={summary['num_edges']}, "
        f"valid={summary['num_valid_edges']}, transform_storage=lazy_meta, "
        "cache_rewrites=0"
    )
    return summary


def apply_deferred_chunk_post_alignment(
    chunk_records: List[Dict[str, object]],
    args,
    show_progress: bool = True,
) -> Dict[str, object]:
    """Hierarchically align chunk records in-place.

    The alignment follows the stored spatial tree path in record["cell_key"].
    At each tree level, sibling nodes are aligned using overlapping seam-view
    geometry and merged into a parent node.
    """
    if not bool(getattr(args, "post_chunk_align", False)):
        return {"enabled": False}

    if len(chunk_records) <= 1:
        return {
            "enabled": True,
            "reason": "single chunk",
            "num_chunks": int(len(chunk_records)),
            "num_edges": 0,
        }

    if all(
        str(record.get("alignment_topology", "")) == "parent_graph"
        for record in chunk_records
    ):
        summary = _apply_parent_graph_alignment(
            chunk_records=chunk_records,
            args=args,
            show_progress=show_progress,
        )
        return summary

    nodes: Dict[Tuple[int, ...], _AlignNode] = {}
    for rid, record in enumerate(chunk_records):
        ensure_record_similarity(record)
        key = _as_key(record.get("cell_key", (rid,)), rid)
        if key in nodes:
            # Extremely defensive fallback for duplicate keys.
            key = (*key, int(rid))
        nodes[key] = _make_leaf_node(key=key, rid=int(rid), record=record)
        record["post_chunk_align_edges"] = []

    all_edges: List[Dict[str, object]] = []
    level = 0
    max_iter = max(1, max(len(k) for k in nodes.keys()) + 4)
    pbar = _progress_bar(max(0, len(nodes) - 1), bool(show_progress))
    base_seed = int(getattr(args, "seed", 0)) + 880031
    max_workers = max(1, int(getattr(args, "post_chunk_align_workers", 1)))
    valid_count = 0

    def _record_edges(edges: Sequence[Dict[str, object]]) -> None:
        nonlocal valid_count
        all_edges.extend(edges)
        valid_count += sum(1 for e in edges if bool(e.get("valid", False)))
        if pbar is not None and edges:
            pbar.update(len(edges))
            last = edges[-1]
            pbar.set_postfix(
                level=level,
                valid=valid_count,
                corr=int(last.get("num_corr", 0)),
                workers=max_workers,
            )

    try:
        while len(nodes) > 1 and level < max_iter:
            groups: Dict[Tuple[int, ...], List[_AlignNode]] = {}
            for key, node in nodes.items():
                parent_key = key[:-1] if len(key) > 0 else key
                groups.setdefault(parent_key, []).append(node)

            next_nodes: Dict[Tuple[int, ...], _AlignNode] = {}
            group_items = sorted(groups.items(), key=lambda kv: kv[0])
            if max_workers > 1 and len(group_items) > 1:
                with ThreadPoolExecutor(max_workers=min(max_workers, len(group_items))) as pool:
                    futures = [
                        pool.submit(
                            _process_alignment_group,
                            chunk_records,
                            parent_key,
                            children,
                            args,
                            base_seed,
                            level,
                        )
                        for parent_key, children in group_items
                    ]
                    for future in as_completed(futures):
                        parent_key, anchor, edges = future.result()
                        next_nodes[parent_key] = anchor
                        _record_edges(edges)
            else:
                for parent_key, children in group_items:
                    parent_key, anchor, edges = _process_alignment_group(
                        records=chunk_records,
                        parent_key=parent_key,
                        children=children,
                        args=args,
                        base_seed=base_seed,
                        level=level,
                    )
                    next_nodes[parent_key] = anchor
                    _record_edges(edges)

            # If all nodes only changed keys but no merge happened, stop to avoid loops.
            if set(next_nodes.keys()) == set(nodes.keys()):
                break

            nodes = next_nodes
            level += 1
    finally:
        if pbar is not None:
            pbar.close()

    valid_edges = [e for e in all_edges if bool(e.get("valid", False))]
    summary = {
        "enabled": True,
        "mode": str(args.post_chunk_align_mode),
        "transform_storage": "record_metadata_lazy",
        "cache_rewrites": 0,
        "num_chunks": int(len(chunk_records)),
        "num_edges": int(len(all_edges)),
        "num_valid_edges": int(len(valid_edges)),
        "min_corr": int(args.post_chunk_align_min_corr),
        "workers": int(max_workers),
        "edges": all_edges,
    }

    for record in chunk_records:
        record["post_chunk_align_meta"] = {
            "enabled": True,
            "mode": str(args.post_chunk_align_mode),
            "transform_storage": "record_metadata_lazy",
            "cache_rewrites": 0,
            "num_edges_applied": int(
                len(record.get("post_chunk_align_edges", []))
            ),
        }

    print(
        "[INFO] Deferred chunk post-alignment: "
        f"mode={summary['mode']}, chunks={summary['num_chunks']}, "
        f"edges={summary['num_edges']}, valid={summary['num_valid_edges']}, "
        f"workers={summary['workers']}, transform_storage=lazy_meta, "
        "cache_rewrites=0"
    )
    return summary
