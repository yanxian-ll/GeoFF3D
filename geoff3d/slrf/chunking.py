# -*- coding: utf-8 -*-
"""Spatial and temporal chunking for large-scene reconstruction."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

SPATIAL_PARTITIONS = {"footprint_tree", "temporal"}
FOOTPRINT_SOURCES = {"auto", "prior", "lookat", "center", "sequential"}
CHUNK_ORDER_STRATEGIES = {
    "spatial_center_bfs",
    "sequential",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
            "Spatial chunking requires a finite input camera pose for every "
            f"selected frame; missing/invalid pose for: {shown}{suffix}"
        )
    if not centers:
        raise ValueError(
            "Spatial chunking requires at least one selected frame with an "
            "input camera pose."
        )
    return np.stack(centers, axis=0)


def spatial_axis_indices(axes: str) -> Tuple[int, ...]:
    mapping = {"x": 0, "y": 1, "z": 2}
    return tuple(mapping[ch] for ch in str(axes).lower())


def infer_spatial_axes(meta: Dict[str, object]) -> str:
    """Auto-detect best two axes for spatial partitioning from camera centers."""
    try:
        centers = pose_centers_from_meta(meta)
        extent = np.nanmax(centers, axis=0) - np.nanmin(centers, axis=0)
        axes = np.asarray(["x", "y", "z"])
        top2 = np.argsort(extent)[-2:]
        return "".join(axes[np.sort(top2)].tolist())
    except Exception:
        return "xy"


def infer_footprint_stride(meta: Dict[str, object]) -> int:
    h = int(meta.get("target_h", 0))
    w = int(meta.get("target_w", 0))
    return max(1, int(round(max(h, w) / 64.0)))


def auto_core_target_size(max_chunk_size: int) -> int:
    max_chunk_size = max(1, int(max_chunk_size))
    if max_chunk_size <= 2:
        return max_chunk_size
    return max(
        1, min(max_chunk_size - 1, int(np.floor(float(max_chunk_size) * 0.7)))
    )


def make_chunk_meta(
    meta: Dict[str, object], indices: Sequence[int]
) -> Dict[str, object]:
    chunk_meta = dict(meta)
    chunk_meta["stems"] = [meta["stems"][i] for i in indices]
    if "image_paths" in meta:
        chunk_meta["image_paths"] = {
            stem: meta["image_paths"][stem]
            for stem in chunk_meta["stems"]
            if stem in meta["image_paths"]
        }
    if "depth_paths" in meta:
        chunk_meta["depth_paths"] = {
            stem: meta["depth_paths"][stem]
            for stem in chunk_meta["stems"]
            if stem in meta["depth_paths"]
        }
    return chunk_meta


def _as_chunk_key(chunk: Dict[str, object]) -> Tuple[int, ...]:
    key = chunk.get("cell_key", ())
    if isinstance(key, tuple):
        return tuple(int(v) for v in key)
    if isinstance(key, list):
        return tuple(int(v) for v in key)
    return (int(chunk.get("chunk_id", 0)),)


def _chunk_core_center(
    chunk: Dict[str, object],
    centers: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(chunk.get("core_indices", []), dtype=np.int64)
    if indices.size == 0:
        indices = np.asarray(chunk.get("indices", []), dtype=np.int64)
    if indices.size == 0:
        return np.zeros((centers.shape[1],), dtype=np.float64)
    valid = indices[(indices >= 0) & (indices < centers.shape[0])]
    if valid.size == 0:
        return np.zeros((centers.shape[1],), dtype=np.float64)
    return np.nanmean(centers[valid], axis=0).astype(np.float64)


def _chunk_start_index(
    chunks: Sequence[Dict[str, object]],
    centers: np.ndarray,
) -> int:
    if not chunks:
        return 0
    chunk_centers = np.stack(
        [_chunk_core_center(chunk, centers) for chunk in chunks],
        axis=0,
    )
    scene_center = np.nanmean(centers, axis=0)
    dist = np.linalg.norm(chunk_centers - scene_center[None, :], axis=1)
    return int(np.nanargmin(dist))


def _chunk_adjacency(
    chunks: Sequence[Dict[str, object]],
    centers: np.ndarray,
) -> Tuple[List[List[int]], np.ndarray]:
    n = len(chunks)
    chunk_centers = np.stack(
        [_chunk_core_center(chunk, centers) for chunk in chunks],
        axis=0,
    ) if n else np.empty((0, centers.shape[1]), dtype=np.float64)
    core_sets: List[Set[int]] = [
        {int(v) for v in chunk.get("core_indices", [])}
        for chunk in chunks
    ]
    index_sets: List[Set[int]] = [
        {int(v) for v in chunk.get("indices", [])}
        for chunk in chunks
    ]
    overlap_sets: List[Set[int]] = [
        {int(v) for v in chunk.get("overlap_indices", [])}
        for chunk in chunks
    ]
    keys = [_as_chunk_key(chunk) for chunk in chunks]

    adjacency: List[Set[int]] = [set() for _ in chunks]
    for i in range(n):
        for j in range(i + 1, n):
            shared = index_sets[i] & index_sets[j]
            cross = (overlap_sets[i] & core_sets[j]) | (overlap_sets[j] & core_sets[i])
            sibling = (
                str(chunks[i].get("partition", "footprint_tree")) == "footprint_tree"
                and str(chunks[j].get("partition", "footprint_tree")) == "footprint_tree"
                and len(keys[i]) > 0
                and len(keys[j]) > 0
                and keys[i][:-1] == keys[j][:-1]
            )
            if shared or cross or sibling:
                adjacency[i].add(j)
                adjacency[j].add(i)

    if n > 1:
        for i in range(n):
            if adjacency[i]:
                continue
            dist = np.linalg.norm(chunk_centers - chunk_centers[i][None, :], axis=1)
            dist[i] = np.inf
            j = int(np.nanargmin(dist))
            if np.isfinite(dist[j]):
                adjacency[i].add(j)
                adjacency[j].add(i)

    ordered_neighbors: List[List[int]] = []
    for i, neighbors in enumerate(adjacency):
        ordered_neighbors.append(
            sorted(
                (int(j) for j in neighbors),
                key=lambda j: (
                    float(np.linalg.norm(chunk_centers[j] - chunk_centers[i])),
                    int(chunks[j].get("cell_order", chunks[j].get("chunk_id", j))),
                    j,
                ),
            )
        )
    return ordered_neighbors, chunk_centers


def _renumber_chunks(
    chunks: Sequence[Dict[str, object]],
    order: Sequence[int],
    strategy: str,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for new_id, old_i in enumerate(order):
        chunk = dict(chunks[int(old_i)])
        chunk["source_chunk_id"] = int(chunk.get("chunk_id", old_i))
        chunk["chunk_id"] = int(new_id)
        chunk["execution_order"] = int(new_id)
        chunk["chunk_order_strategy"] = str(strategy)
        out.append(chunk)
    return out


def _attach_adjacency_metadata(
    ordered: List[Dict[str, object]],
    source_order: Sequence[int],
    adjacency: Sequence[Sequence[int]],
) -> None:
    old_to_new = {int(old): int(new) for new, old in enumerate(source_order)}
    for new_id, old_id in enumerate(source_order):
        ordered[new_id]["adjacent_chunk_ids"] = sorted(
            old_to_new[int(neighbor)]
            for neighbor in adjacency[int(old_id)]
            if int(neighbor) in old_to_new
        )


def _attach_sequential_chain_topology(
    ordered: List[Dict[str, object]],
) -> None:
    """Attach a strict temporal chain: chunk i is aligned to chunk i-1."""
    for chunk_id, chunk in enumerate(ordered):
        chunk["alignment_topology"] = "parent_graph"
        chunk["align_parent_id"] = None if chunk_id == 0 else int(chunk_id - 1)
        chunk["align_level"] = int(chunk_id)
        chunk["adjacent_chunk_ids"] = [
            neighbor
            for neighbor in (chunk_id - 1, chunk_id + 1)
            if 0 <= neighbor < len(ordered)
        ]


def _append_disconnected_component(
    queue: deque,
    visited: Set[int],
    chunk_centers: np.ndarray,
    scene_center: np.ndarray,
) -> None:
    unvisited = [i for i in range(chunk_centers.shape[0]) if i not in visited]
    if not unvisited:
        return
    if visited:
        anchor = np.nanmean(chunk_centers[list(visited)], axis=0)
    else:
        anchor = scene_center
    next_i = min(
        unvisited,
        key=lambda i: (
            float(np.linalg.norm(chunk_centers[i] - anchor)),
            float(np.linalg.norm(chunk_centers[i] - scene_center)),
            i,
        ),
    )
    queue.append(next_i)


def order_spatial_chunks(
    chunks: Sequence[Dict[str, object]],
    meta: Dict[str, object],
    strategy: str = "spatial_center_bfs",
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    strategy = str(strategy).strip().lower()
    if strategy not in CHUNK_ORDER_STRATEGIES:
        raise ValueError(
            f"Unknown chunk order strategy: {strategy}. "
            f"Expected one of: {sorted(CHUNK_ORDER_STRATEGIES)}"
        )
    if not chunks:
        return [], {"strategy": strategy, "order": []}

    if strategy == "sequential":
        order = sorted(
            range(len(chunks)),
            key=lambda i: (
                int(chunks[i].get("temporal_order", chunks[i].get("cell_order", i))),
                int(chunks[i].get("chunk_id", i)),
            ),
        )
        ordered = _renumber_chunks(chunks, order, strategy)
        _attach_sequential_chain_topology(ordered)
        return ordered, {
            "strategy": strategy,
            "order": [int(i) for i in order],
            "source_chunk_ids": [int(chunks[i].get("chunk_id", i)) for i in order],
            "start_source_chunk_id": int(chunks[order[0]].get("chunk_id", order[0])),
            "num_adjacency_edges": max(0, len(ordered) - 1),
            "alignment_topology": "sequential_parent_chain",
        }

    centers = pose_centers_from_meta(meta)
    adjacency, chunk_centers = _chunk_adjacency(chunks, centers)
    scene_center = np.nanmean(centers, axis=0)
    start = _chunk_start_index(chunks, centers)
    queue: deque = deque([int(start)])
    visited: Set[int] = set()
    order: List[int] = []
    while len(order) < len(chunks):
        if not queue:
            _append_disconnected_component(
                queue,
                visited,
                chunk_centers,
                scene_center,
            )
            if not queue:
                break
        current = int(queue.popleft())
        if current in visited:
            continue
        visited.add(current)
        order.append(current)
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                queue.append(neighbor)

    ordered = _renumber_chunks(chunks, order, strategy)
    _attach_adjacency_metadata(ordered, order, adjacency)
    return ordered, {
        "strategy": strategy,
        "order": [int(i) for i in order],
        "source_chunk_ids": [int(chunks[i].get("chunk_id", i)) for i in order],
        "start_source_chunk_id": int(chunks[order[0]].get("chunk_id", order[0])),
        "num_adjacency_edges": int(sum(len(v) for v in adjacency) // 2),
    }


# ---------------------------------------------------------------------------
# Footprint-tree partition
# ---------------------------------------------------------------------------
def robust_bbox(
    points: np.ndarray, qmin: float, qmax: float
) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, points.shape[-1])
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


def estimate_fallback_lookat_distance(
    centers: np.ndarray, gt_points: np.ndarray
) -> float:
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt_points, dtype=np.float64).reshape(-1, 3)
    if gt.shape[0] > 0 and np.isfinite(gt).all(axis=1).any():
        gt = gt[np.isfinite(gt).all(axis=1)]
        scene_center = np.median(gt, axis=0)
        d = np.linalg.norm(centers - scene_center[None, :], axis=1)
        d = d[np.isfinite(d) & (d > 1e-6)]
        if d.size > 0:
            return float(np.median(d))

    extent = (
        centers.max(axis=0) - centers.min(axis=0)
        if centers.shape[0] > 1
        else np.ones(3)
    )
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
        z_ground = float(
            np.quantile(gt[np.isfinite(gt).all(axis=1), 2], 0.05)
        )
    else:
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        z_ground = float(np.min(centers[:, 2]))

    denom = float(forward[2])
    if abs(denom) > 1e-8:
        t_ground = (z_ground - float(center[2])) / denom
        if np.isfinite(t_ground) and t_ground > 1e-6:
            return center + forward * t_ground

    distance = (
        float(requested_distance) if requested_distance > 0 else float(fallback_distance)
    )
    return center + forward * max(distance, 1e-6)


def point_bbox(point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(point, dtype=np.float64).reshape(-1)
    return p.copy(), p.copy()


def footprint_features_from_meta(
    meta: Dict[str, object],
    axis_indices: Tuple[int, ...],
    footprint_source: str = "auto",
    footprint_sample_stride: int = 16,
    footprint_min_points: int = 32,
    footprint_quantile_min: float = 0.02,
    footprint_quantile_max: float = 0.98,
    footprint_lookat_distance: float = 0.0,
    footprint_workers: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    stems = list(meta["stems"])
    pose_centers = pose_centers_from_meta(meta)
    gt_points = np.asarray(
        meta.get("gt_points", np.empty((0, 3), np.float32)),
        dtype=np.float64,
    ).reshape(-1, 3)
    fallback_distance = estimate_fallback_lookat_distance(
        pose_centers, gt_points
    )
    source_requested = str(footprint_source)
    counts = {"prior": 0, "lookat": 0, "center": 0, "predicted": 0}
    if source_requested in {"prior", "sequential"}:
        estimated = meta.get("estimated_footprints", {})
        centers = np.asarray(estimated.get("centers", []), dtype=np.float64)
        bbox_mins = np.asarray(estimated.get("bbox_mins", []), dtype=np.float64)
        bbox_maxs = np.asarray(estimated.get("bbox_maxs", []), dtype=np.float64)
        expected_shape = (len(stems), len(axis_indices))
        if (
            centers.shape != expected_shape
            or bbox_mins.shape != expected_shape
            or bbox_maxs.shape != expected_shape
            or not np.isfinite(centers).all()
            or not np.isfinite(bbox_mins).all()
            or not np.isfinite(bbox_maxs).all()
        ):
            raise ValueError(
                "Estimated footprint arrays are missing or invalid: "
                f"expected shape {expected_shape}, got centers={centers.shape}, "
                f"bbox_mins={bbox_mins.shape}, bbox_maxs={bbox_maxs.shape}."
            )
        feature_meta = dict(estimated.get("meta", {}))
        feature_meta.update(
            {
                "footprint_source_requested": source_requested,
                "footprint_source_counts": {source_requested: len(stems)},
                "footprint_sources": [source_requested] * len(stems),
            }
        )
        return centers, bbox_mins, bbox_maxs, feature_meta

    foot_centers: List[np.ndarray] = []
    bbox_mins: List[np.ndarray] = []
    bbox_maxs: List[np.ndarray] = []
    sources: List[str] = []
    stride = int(footprint_sample_stride)

    for i, stem in enumerate(stems):
        source_used: Optional[str] = None
        bbox_min: Optional[np.ndarray] = None
        bbox_max: Optional[np.ndarray] = None
        center_coord: Optional[np.ndarray] = None

        if source_used is None and source_requested in {"auto", "lookat"}:
            cam = meta.get("cams", {}).get(stem)
            if cam is not None and cam.get("T_c2w") is not None:
                lookat = lookat_point_from_pose(
                    np.asarray(cam["T_c2w"], dtype=np.float64),
                    centers=pose_centers,
                    gt_points=gt_points,
                    fallback_distance=fallback_distance,
                    requested_distance=float(footprint_lookat_distance),
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

        foot_centers.append(np.asarray(center_coord, dtype=np.float64))
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
        "min_points": int(footprint_min_points),
        "quantile_min": float(footprint_quantile_min),
        "quantile_max": float(footprint_quantile_max),
    }
    return (
        np.stack(foot_centers, axis=0).astype(np.float64),
        np.stack(bbox_mins, axis=0).astype(np.float64),
        np.stack(bbox_maxs, axis=0).astype(np.float64),
        feature_meta,
    )


def expand_degenerate_region(
    region_min: np.ndarray, region_max: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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


def _safe_unit_2d(v: np.ndarray, fallback: Tuple[float, float] = (1.0, 0.0)) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 1e-12:
        return np.asarray(fallback, dtype=np.float64)
    return v / n


def estimate_main_flight_frame_from_pose(
    meta: Dict[str, object],
    axis_indices: Tuple[int, ...],
) -> Dict[str, object]:
    """
    Estimate a 2D flight-aligned frame from input pose centers.

    local_x: dominant flight direction estimated from consecutive pose motion.
    local_y: perpendicular cross-track direction.

    Coordinates transform:
        p_local = (p_world_xy - origin) @ rot_to_flight.T
        p_world_xy = p_local @ rot_to_flight + origin
    """
    pose_centers_3d = pose_centers_from_meta(meta)
    pose_xy = pose_centers_3d[:, tuple(axis_indices)].astype(np.float64)

    finite_pose = np.isfinite(pose_xy).all(axis=1)
    pose_xy_valid = pose_xy[finite_pose]

    if pose_xy_valid.shape[0] == 0:
        origin = np.zeros((2,), dtype=np.float64)
        main_dir = np.array([1.0, 0.0], dtype=np.float64)
        method = "fallback_empty"
    else:
        origin = np.nanmedian(pose_xy_valid, axis=0).astype(np.float64)
        main_dir: Optional[np.ndarray] = None
        method = "fallback"

        # Prefer consecutive pose displacements because they follow the flight path.
        if pose_xy_valid.shape[0] >= 2:
            deltas = pose_xy_valid[1:] - pose_xy_valid[:-1]
            lens = np.linalg.norm(deltas, axis=1)
            good = np.isfinite(lens) & (lens > 1e-8)

            if int(np.count_nonzero(good)) >= 1:
                # Use normalized displacement directions so long turns/jumps do not dominate.
                units = deltas[good] / lens[good, None]

                # Sign-invariant direction estimate: v and -v represent the same flight line.
                cov = units.T @ units
                try:
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    main_dir = eigvecs[:, int(np.argmax(eigvals))]
                    method = "pose_delta_orientation"
                except Exception:
                    main_dir = None

        # Fallback: PCA of pose positions.
        if main_dir is None and pose_xy_valid.shape[0] >= 2:
            centered = pose_xy_valid - np.nanmean(pose_xy_valid, axis=0, keepdims=True)
            cov = centered.T @ centered
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                main_dir = eigvecs[:, int(np.argmax(eigvals))]
                method = "pose_position_pca"
            except Exception:
                main_dir = None

        if main_dir is None:
            main_dir = np.array([1.0, 0.0], dtype=np.float64)
            method = "fallback_identity"

        main_dir = _safe_unit_2d(main_dir)

        # Make the sign stable for easier debugging. The split result is sign-invariant.
        if pose_xy_valid.shape[0] >= 2:
            overall = pose_xy_valid[-1] - pose_xy_valid[0]
            if np.isfinite(overall).all() and float(np.dot(main_dir, overall)) < 0.0:
                main_dir = -main_dir

    cross_dir = np.array([-main_dir[1], main_dir[0]], dtype=np.float64)
    cross_dir = _safe_unit_2d(cross_dir, fallback=(0.0, 1.0))

    rot_to_flight = np.stack([main_dir, cross_dir], axis=0).astype(np.float64)

    angle_rad = float(np.arctan2(main_dir[1], main_dir[0]))
    angle_deg = float(np.degrees(angle_rad))

    return {
        "type": "pose_main_flight_aligned",
        "method": method,
        "axes": "".join("xyz"[int(a)] for a in axis_indices),
        "origin": origin.astype(float).tolist(),
        "main_direction": main_dir.astype(float).tolist(),
        "cross_direction": cross_dir.astype(float).tolist(),
        "rot_to_flight": rot_to_flight.astype(float).tolist(),
        "angle_rad": angle_rad,
        "angle_deg": angle_deg,
    }


def transform_points_to_flight_frame(
    points: np.ndarray,
    flight_frame: Dict[str, object],
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    shape = pts.shape
    pts = pts.reshape(-1, 2)

    origin = np.asarray(flight_frame["origin"], dtype=np.float64).reshape(2)
    rot_to_flight = np.asarray(flight_frame["rot_to_flight"], dtype=np.float64).reshape(2, 2)

    out = (pts - origin[None, :]) @ rot_to_flight.T
    return out.reshape(shape).astype(np.float64)


def transform_points_from_flight_frame(
    points: np.ndarray,
    flight_frame: Dict[str, object],
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    shape = pts.shape
    pts = pts.reshape(-1, 2)

    origin = np.asarray(flight_frame["origin"], dtype=np.float64).reshape(2)
    rot_to_flight = np.asarray(flight_frame["rot_to_flight"], dtype=np.float64).reshape(2, 2)

    out = pts @ rot_to_flight + origin[None, :]
    return out.reshape(shape).astype(np.float64)


def transform_bboxes_to_flight_frame(
    bbox_mins: np.ndarray,
    bbox_maxs: np.ndarray,
    flight_frame: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform original axis-aligned 2D bboxes to flight-frame axis-aligned bboxes.

    Because rotation changes axis alignment, each bbox is transformed by its four corners.
    """
    bmin = np.asarray(bbox_mins, dtype=np.float64).reshape(-1, 2)
    bmax = np.asarray(bbox_maxs, dtype=np.float64).reshape(-1, 2)

    corners = np.stack(
        [
            np.stack([bmin[:, 0], bmin[:, 1]], axis=1),
            np.stack([bmin[:, 0], bmax[:, 1]], axis=1),
            np.stack([bmax[:, 0], bmin[:, 1]], axis=1),
            np.stack([bmax[:, 0], bmax[:, 1]], axis=1),
        ],
        axis=1,
    )
    local = transform_points_to_flight_frame(corners, flight_frame)
    return (
        np.nanmin(local, axis=1).astype(np.float64),
        np.nanmax(local, axis=1).astype(np.float64),
    )


def split_adaptive_tree(
    indices: Sequence[int],
    centers: np.ndarray,
    region_min: np.ndarray,
    region_max: np.ndarray,
    target_core_size: int,
    path: Tuple[int, ...],
    split_records: Optional[List[Dict[str, object]]] = None,
    preferred_root_axis: Optional[int] = None,
) -> List[Dict[str, object]]:
    """
    Faster adaptive binary tree split.

    Compared with the old recursive version:
    - avoids full Python sorting at every node
    - uses np.argpartition for median split
    - uses an explicit stack to avoid recursion overhead
    """
    centers = np.asarray(centers, dtype=np.float64)
    target_core_size = max(1, int(target_core_size))

    leaves: List[Dict[str, object]] = []
    stack = [
        (
            np.asarray(indices, dtype=np.int64),
            np.asarray(region_min, dtype=np.float64),
            np.asarray(region_max, dtype=np.float64),
            tuple(path),
        )
    ]

    while stack:
        idxs, rmin, rmax, node_path = stack.pop()

        if idxs.size <= target_core_size:
            leaves.append(
                {
                    "cell_key": node_path if node_path else (0,),
                    "core_indices": sorted(int(i) for i in idxs.tolist()),
                    "region_min": rmin,
                    "region_max": rmax,
                }
            )
            continue

        pts = centers[idxs]
        extent = np.nanmax(pts, axis=0) - np.nanmin(pts, axis=0)
        if not np.isfinite(extent).any():
            leaves.append(
                {
                    "cell_key": node_path if node_path else (0,),
                    "core_indices": sorted(int(i) for i in idxs.tolist()),
                    "region_min": rmin,
                    "region_max": rmax,
                }
            )
            continue

        if preferred_root_axis is not None and len(node_path) == 0:
            candidate_axis = int(preferred_root_axis)
            vals_candidate = centers[idxs, candidate_axis]
            finite_candidate = np.isfinite(vals_candidate)
            if (
                0 <= candidate_axis < centers.shape[1]
                and int(np.count_nonzero(finite_candidate)) >= 2
                and np.isfinite(extent[candidate_axis])
                and float(extent[candidate_axis]) > 1e-9
            ):
                axis = candidate_axis
            else:
                axis = int(np.nanargmax(extent))
        else:
            axis = int(np.nanargmax(extent))
        vals = centers[idxs, axis]
        finite = np.isfinite(vals)
        if finite.sum() < 2:
            leaves.append(
                {
                    "cell_key": node_path if node_path else (0,),
                    "core_indices": sorted(int(i) for i in idxs.tolist()),
                    "region_min": rmin,
                    "region_max": rmax,
                }
            )
            continue

        # Put non-finite values at the end, then split finite part.
        finite_idxs = idxs[finite]
        finite_vals = vals[finite]
        nonfinite_idxs = idxs[~finite]

        mid = finite_idxs.size // 2
        order = np.argpartition(finite_vals, mid)
        left = finite_idxs[order[:mid]]
        right = finite_idxs[order[mid:]]

        # Attach non-finite entries to the smaller side.
        for x in nonfinite_idxs:
            if left.size <= right.size:
                left = np.append(left, x)
            else:
                right = np.append(right, x)

        if left.size == 0 or right.size == 0:
            leaves.append(
                {
                    "cell_key": node_path if node_path else (0,),
                    "core_indices": sorted(int(i) for i in idxs.tolist()),
                    "region_min": rmin,
                    "region_max": rmax,
                }
            )
            continue

        left_max = float(np.nanmax(centers[left, axis]))
        right_min = float(np.nanmin(centers[right, axis]))
        threshold = 0.5 * (left_max + right_min)

        if (
            not np.isfinite(threshold)
            or threshold <= float(rmin[axis])
            or threshold >= float(rmax[axis])
        ):
            threshold = 0.5 * (float(rmin[axis]) + float(rmax[axis]))

        left_min = rmin.copy()
        left_max_region = rmax.copy()
        right_min_region = rmin.copy()
        right_max = rmax.copy()

        left_max_region[axis] = threshold
        right_min_region[axis] = threshold

        if split_records is not None:
            split_records.append(
                {
                    "order": int(len(split_records)),
                    "path": tuple(int(v) for v in node_path),
                    "depth": int(len(node_path)),
                    "axis": int(axis),
                    "threshold": float(threshold),
                    "region_min": np.asarray(rmin, dtype=np.float64).astype(float).tolist(),
                    "region_max": np.asarray(rmax, dtype=np.float64).astype(float).tolist(),
                    "left_key": tuple(int(v) for v in (*node_path, 0)),
                    "right_key": tuple(int(v) for v in (*node_path, 1)),
                    "num_indices": int(idxs.size),
                    "num_left": int(left.size),
                    "num_right": int(right.size),
                }
            )

        # Push right first so left is processed first.
        stack.append((right, right_min_region, right_max, (*node_path, 1)))
        stack.append((left, left_min, left_max_region, (*node_path, 0)))

    return leaves


def region_gap(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
) -> np.ndarray:
    return np.maximum(np.maximum(a_min - b_max, b_min - a_max), 0.0)


def regions_are_adjacent(
    a: Dict[str, object], b: Dict[str, object], margin: float
) -> bool:
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


def order_tree_leaves(
    leaves: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    def key_fn(leaf: Dict[str, object]) -> Tuple[float, ...]:
        lo = np.asarray(leaf["region_min"], dtype=np.float64)
        hi = np.asarray(leaf["region_max"], dtype=np.float64)
        center = 0.5 * (lo + hi)
        return tuple(float(v) for v in center)

    return sorted(leaves, key=key_fn)


def build_footprint_tree_chunks(
    meta: Dict[str, object],
    axes: str = "xy",
    max_chunk_size: int = 32,
    min_chunk_size: int = 1,
    max_chunks: int = 0,
    footprint_source: str = "auto",
    footprint_sample_stride: int = 16,
    footprint_min_points: int = 32,
    footprint_quantile_min: float = 0.02,
    footprint_quantile_max: float = 0.98,
    footprint_lookat_distance: float = 0.0,
    footprint_neighbor_margin: float = 0.0,
    footprint_workers: int = 0,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    axis_indices = spatial_axis_indices(axes)
    centers_world, bbox_mins_world, bbox_maxs_world, feature_meta = footprint_features_from_meta(
        meta,
        axis_indices=axis_indices,
        footprint_source=footprint_source,
        footprint_sample_stride=footprint_sample_stride,
        footprint_min_points=footprint_min_points,
        footprint_quantile_min=footprint_quantile_min,
        footprint_quantile_max=footprint_quantile_max,
        footprint_lookat_distance=footprint_lookat_distance,
        footprint_workers=footprint_workers,
    )

    # Estimate main flight direction from input poses, then rotate footprint
    # centers/bboxes into a flight-aligned 2D coordinate system.
    flight_frame = estimate_main_flight_frame_from_pose(
        meta=meta,
        axis_indices=axis_indices,
    )
    centers = transform_points_to_flight_frame(centers_world, flight_frame)
    bbox_mins, bbox_maxs = transform_bboxes_to_flight_frame(
        bbox_mins_world,
        bbox_maxs_world,
        flight_frame,
    )

    target_core_size = auto_core_target_size(int(max_chunk_size))
    region_min, region_max = expand_degenerate_region(
        np.nanmin(bbox_mins, axis=0),
        np.nanmax(bbox_maxs, axis=0),
    )
    footprint_split_lines: List[Dict[str, object]] = []
    leaves = split_adaptive_tree(
        indices=list(range(len(meta["stems"]))),
        centers=centers,
        region_min=region_min,
        region_max=region_max,
        target_core_size=target_core_size,
        path=(),
        split_records=footprint_split_lines,
        preferred_root_axis=0,
    )
    ordered_leaves = order_tree_leaves(leaves)
    for order_idx, leaf in enumerate(ordered_leaves):
        leaf["cell_order"] = int(order_idx)

    chunks: List[Dict[str, object]] = []
    total_dropped_seam_images = 0
    margin = float(footprint_neighbor_margin)
    leaf_core_sets = [
        set(int(i) for i in leaf["core_indices"]) for leaf in ordered_leaves
    ]

    # Pre-build numpy arrays for vectorized seam search
    leaf_region_min = np.stack(
        [np.asarray(leaf["region_min"], dtype=np.float64) for leaf in ordered_leaves],
        axis=0,
    )
    leaf_region_max = np.stack(
        [np.asarray(leaf["region_max"], dtype=np.float64) for leaf in ordered_leaves],
        axis=0,
    )
    bbox_mins_np = np.asarray(bbox_mins, dtype=np.float64)
    bbox_maxs_np = np.asarray(bbox_maxs, dtype=np.float64)


    def adjacent_leaf_indices(leaf_idx: int) -> np.ndarray:
        a_min = leaf_region_min[leaf_idx][None, :]
        a_max = leaf_region_max[leaf_idx][None, :]

        gap = np.maximum(
            np.maximum(a_min - leaf_region_max, leaf_region_min - a_max),
            0.0,
        )
        adjacent = np.all(gap <= margin + 1e-9, axis=1)
        adjacent[leaf_idx] = False
        return np.nonzero(adjacent)[0]


    def bbox_distances_to_leaf_region(candidate_indices: np.ndarray, leaf_idx: int) -> np.ndarray:
        rmin = leaf_region_min[leaf_idx][None, :]
        rmax = leaf_region_max[leaf_idx][None, :]
        bmin = bbox_mins_np[candidate_indices]
        bmax = bbox_maxs_np[candidate_indices]

        gap = np.maximum(
            np.maximum(bmin - rmax, rmin - bmax),
            0.0,
        )
        return np.linalg.norm(gap, axis=1)


    for leaf_idx, leaf in enumerate(ordered_leaves):
        core_indices = sorted(int(i) for i in leaf["core_indices"])
        if len(core_indices) < int(min_chunk_size):
            continue

        adjacent_leaf_idxs = adjacent_leaf_indices(leaf_idx)

        candidate_chunks = []
        for other_idx in adjacent_leaf_idxs:
            other_core = np.asarray(
                sorted(leaf_core_sets[int(other_idx)]),
                dtype=np.int64,
            )
            if other_core.size > 0:
                candidate_chunks.append(other_core)

        if candidate_chunks:
            candidate_arr = np.unique(np.concatenate(candidate_chunks, axis=0))
        else:
            candidate_arr = np.empty((0,), dtype=np.int64)

        if candidate_arr.size > 0:
            dists = bbox_distances_to_leaf_region(candidate_arr, leaf_idx)
            priority = (dists > margin + 1e-9).astype(np.int64)
            order = np.lexsort((candidate_arr, dists, priority))
            candidate_indices = candidate_arr[order].astype(np.int64).tolist()
        else:
            candidate_indices = []

        budget = max(0, int(max_chunk_size) - len(core_indices))
        overlap_indices = sorted(candidate_indices[:budget])
        dropped_seam_images = max(
            0, len(candidate_indices) - len(overlap_indices)
        )
        total_dropped_seam_images += dropped_seam_images

        indices = sorted(set(core_indices + overlap_indices))
        core_set = set(core_indices)
        core_local_indices = [
            local_i
            for local_i, global_i in enumerate(indices)
            if int(global_i) in core_set
        ]
        chunks.append(
            {
                "chunk_id": int(len(chunks)),
                "partition": "footprint_tree",
                "cell_order": int(leaf["cell_order"]),
                "cell_key": tuple(int(v) for v in leaf["cell_key"]),
                "indices": indices,
                "core_indices": core_indices,
                "raw_num_core_images": int(len(core_indices)),
                "overlap_indices": overlap_indices,
                "core_local_indices": core_local_indices,
                "num_seam_candidates": int(len(candidate_indices)),
                "num_dropped_seam_images": int(dropped_seam_images),
            }
        )
        if int(max_chunks) > 0 and len(chunks) >= int(max_chunks):
            break

    grid_meta = {
        "partition": "footprint_tree",
        "axes": str(axes),
        "grid_size_requested": 0.0,
        "grid_size_effective": 0.0,
        "origin": region_min.astype(float).tolist(),
        "region_min": region_min.astype(float).tolist(),
        "region_max": region_max.astype(float).tolist(),
        "num_occupied_cells": int(len(ordered_leaves)),
        "num_grid_refinements": int(max(0, len(ordered_leaves) - 1)),
        "auto_core_target_size": int(target_core_size),
        "max_chunk_size": int(max_chunk_size),
        "min_chunk_size": int(min_chunk_size),
        "total_dropped_seam_images": int(total_dropped_seam_images),
        "alignment_topology": "tree_path",
        "core_size_stats": {
            "min": int(min((len(v) for v in leaf_core_sets), default=0)),
            "max": int(max((len(v) for v in leaf_core_sets), default=0)),
            "mean": float(np.mean([len(v) for v in leaf_core_sets])) if leaf_core_sets else 0.0,
            "std": float(np.std([len(v) for v in leaf_core_sets])) if leaf_core_sets else 0.0,
            "num_under_min": int(sum(len(v) < int(min_chunk_size) for v in leaf_core_sets)),
        },
        "footprint": feature_meta,

        # For visualization/debugging.
        # Original selected spatial axes coordinates, e.g. original xy.
        "footprint_centers": np.asarray(centers_world, dtype=np.float64).astype(float).tolist(),
        "footprint_bbox_mins": np.asarray(bbox_mins_world, dtype=np.float64).astype(float).tolist(),
        "footprint_bbox_maxs": np.asarray(bbox_maxs_world, dtype=np.float64).astype(float).tolist(),

        # Flight-aligned local coordinates actually used by adaptive tree splitting.
        "footprint_centers_flight": np.asarray(centers, dtype=np.float64).astype(float).tolist(),
        "footprint_bbox_mins_flight": np.asarray(bbox_mins, dtype=np.float64).astype(float).tolist(),
        "footprint_bbox_maxs_flight": np.asarray(bbox_maxs, dtype=np.float64).astype(float).tolist(),
        "footprint_split_lines": footprint_split_lines,
        "footprint_split_lines_frame": "flight_aligned",
        "footprint_flight_frame": flight_frame,
    }
    return chunks, grid_meta


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------
def build_temporal_chunks(
    meta: Dict[str, object],
    max_chunk_size: int = 32,
    min_chunk_size: int = 1,
    max_chunks: int = 0,
    overlap_ratio: float = 0.25,
    merge_small_tail: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Build chronological sliding-window chunks with adjacent overlap only."""
    num_frames = int(len(meta.get("stems", [])))
    chunk_size = max(2, int(max_chunk_size))
    ratio = float(np.clip(overlap_ratio, 0.0, 0.95))
    overlap_size = int(np.ceil(float(chunk_size) * ratio)) if ratio > 0 else 0
    overlap_size = min(max(0, overlap_size), chunk_size - 1)
    step = chunk_size - overlap_size

    chunks: List[Dict[str, object]] = []
    start = 0
    previous_end = 0
    while start < num_frames:
        end = min(num_frames, start + chunk_size)
        indices = list(range(start, end))
        overlap_end = min(previous_end, end)
        overlap_indices = list(range(start, overlap_end)) if chunks else []
        overlap_set = set(overlap_indices)
        core_indices = [index for index in indices if index not in overlap_set]
        if not core_indices:
            break

        chunks.append(
            {
                "chunk_id": int(len(chunks)),
                "partition": "temporal",
                "temporal_order": int(len(chunks)),
                "cell_order": int(len(chunks)),
                "cell_key": (int(len(chunks)),),
                "indices": indices,
                "core_indices": core_indices,
                "overlap_indices": overlap_indices,
                "core_local_indices": [
                    local_i
                    for local_i, global_i in enumerate(indices)
                    if global_i not in overlap_set
                ],
                "num_seam_candidates": int(len(overlap_indices)),
                "num_dropped_seam_images": 0,
            }
        )
        previous_end = end
        if end >= num_frames or (int(max_chunks) > 0 and len(chunks) >= int(max_chunks)):
            break
        start += step

    tail_merged_into_previous = False
    if (
        bool(merge_small_tail)
        and len(chunks) > 1
        and len(chunks[-1]["core_indices"]) < int(min_chunk_size)
    ):
        tail = chunks.pop()
        previous = chunks[-1]
        previous["indices"] = list(previous["indices"]) + list(tail["indices"])
        previous["core_indices"] = (
            list(previous["core_indices"]) + list(tail["core_indices"])
        )
        previous["core_local_indices"] = list(range(len(previous["indices"])))
        previous["merged_tail_size"] = int(len(tail["core_indices"]))
        tail_merged_into_previous = True

    tail_under_min = bool(
        chunks
        and len(chunks[-1]["core_indices"]) < int(min_chunk_size)
        and len(chunks) > 1
    )

    covered_core = {
        int(index) for chunk in chunks for index in chunk.get("core_indices", [])
    }
    expected_core = set(range(num_frames))
    if int(max_chunks) <= 0 and covered_core != expected_core:
        raise RuntimeError(
            "Temporal chunking lost frames: "
            f"covered={len(covered_core)}/{num_frames}, "
            f"first_missing={sorted(expected_core - covered_core)[:8]}"
        )

    core_sizes = [len(chunk["core_indices"]) for chunk in chunks]
    return chunks, {
        "partition": "temporal",
        "axes": "time",
        "num_chunks": int(len(chunks)),
        "num_input_images": num_frames,
        "num_core_images": int(sum(core_sizes)),
        "core_coverage_ratio": float(len(covered_core) / max(1, num_frames)),
        "auto_core_target_size": int(step),
        "max_chunk_size": int(chunk_size),
        "temporal_overlap_ratio_requested": float(overlap_ratio),
        "temporal_overlap_size": int(overlap_size),
        "temporal_step": int(step),
        "alignment_topology": "sequential_parent_chain",
        "total_dropped_seam_images": 0,
        "tail_under_min_chunk_size": bool(tail_under_min),
        "tail_merged_into_previous": bool(tail_merged_into_previous),
        "effective_max_chunk_size": int(
            max((len(chunk["indices"]) for chunk in chunks), default=0)
        ),
        "core_size_stats": {
            "min": int(min(core_sizes, default=0)),
            "max": int(max(core_sizes, default=0)),
            "mean": float(np.mean(core_sizes)) if core_sizes else 0.0,
            "std": float(np.std(core_sizes)) if core_sizes else 0.0,
            "num_under_min": int(sum(size < int(min_chunk_size) for size in core_sizes)),
        },
    }


def build_spatial_chunks(
    meta: Dict[str, object],
    spatial_partition: str = "footprint_tree",
    axes: str = "auto",  # auto: infer from camera centers
    max_chunk_size: int = 32,
    min_chunk_size: int = 1,
    max_chunks: int = 0,
    footprint_source: str = "auto",
    footprint_sample_stride: int = -1,  # auto: infer from image resolution
    footprint_min_points: int = 32,
    footprint_quantile_min: float = 0.02,
    footprint_quantile_max: float = 0.98,
    footprint_lookat_distance: float = 0.0,
    footprint_neighbor_margin: float = 0.0,
    footprint_workers: int = 0,
    temporal_overlap_ratio: float = 0.25,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    partition = str(spatial_partition).strip().lower()
    if partition not in SPATIAL_PARTITIONS:
        raise ValueError(
            f"Unknown spatial partition: {spatial_partition!r}. "
            f"Supported modes: {sorted(SPATIAL_PARTITIONS)}"
        )

    if partition == "temporal":
        return build_temporal_chunks(
            meta=meta,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            max_chunks=max_chunks,
            overlap_ratio=temporal_overlap_ratio,
        )

    if footprint_source == "sequential":
        axes = "xy"
    elif axes == "auto":
        axes = infer_spatial_axes(meta)
        print(f"[INFO] Auto spatial axes: {axes}")

    if footprint_sample_stride <= 0:
        footprint_sample_stride = infer_footprint_stride(meta)

    return build_footprint_tree_chunks(
        meta=meta,
        axes=axes,
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        max_chunks=max_chunks,
        footprint_source=footprint_source,
        footprint_sample_stride=footprint_sample_stride,
        footprint_min_points=footprint_min_points,
        footprint_quantile_min=footprint_quantile_min,
        footprint_quantile_max=footprint_quantile_max,
        footprint_lookat_distance=footprint_lookat_distance,
        footprint_neighbor_margin=footprint_neighbor_margin,
        footprint_workers=footprint_workers,
    )
