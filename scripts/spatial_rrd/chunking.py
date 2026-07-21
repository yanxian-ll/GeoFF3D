# -*- coding: utf-8 -*-
"""Spatial and temporal chunking for large-scene reconstruction."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from spatial_rrd.scene_io import (
    read_depth,
    resize_depth_to_target,
    scale_K_to_target,
)


SPATIAL_PARTITIONS = {"footprint_tree", "footprint_grid", "pose_grid", "temporal"}
FOOTPRINT_SOURCES = {"auto", "depth", "lookat", "center"}
CHUNK_ORDER_STRATEGIES = {
    "spatial_sort",
    "spatial_center_bfs",
    "first_frame_bfs",
    "depth_prior_greedy",
    "dfs",
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


def pose_grid_axis_indices(axes: str) -> Tuple[int, ...]:
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


def estimate_pose_grid_size(coords: np.ndarray, target_core_size: int) -> float:
    pts = np.asarray(coords, dtype=np.float64).reshape(-1, coords.shape[-1])
    if pts.shape[0] <= 1:
        return 1.0
    target_cells = max(
        1, int(np.ceil(float(pts.shape[0]) / max(1, int(target_core_size))))
    )
    cells_per_axis = max(
        1, int(np.ceil(target_cells ** (1.0 / float(pts.shape[1]))))
    )
    extent = pts.max(axis=0) - pts.min(axis=0)
    max_extent = float(np.max(extent))
    if not np.isfinite(max_extent) or max_extent <= 0:
        return 1.0
    return max(max_extent / float(cells_per_axis), 1e-6)


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
    strategy: str,
) -> int:
    if not chunks:
        return 0
    if strategy == "first_frame_bfs":
        for key_name in ("core_indices", "indices", "overlap_indices"):
            for i, chunk in enumerate(chunks):
                if 0 in {int(v) for v in chunk.get(key_name, [])}:
                    return int(i)

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


def _attach_parent_graph_topology(
    ordered: List[Dict[str, object]],
    source_order: Sequence[int],
    adjacency: Sequence[Sequence[int]],
) -> None:
    """Attach a rooted parent graph without changing execution order.

    This is used by fixed-grid ablations. Each chunk selects an already
    executed adjacent chunk as parent, preferring the one sharing the most
    input views. The adaptive-tree path remains untouched.
    """
    old_to_new = {int(old): int(new) for new, old in enumerate(source_order)}
    levels: Dict[int, int] = {}
    for new_id, old_id in enumerate(source_order):
        chunk = ordered[new_id]
        chunk["alignment_topology"] = "parent_graph"
        if new_id == 0:
            chunk["align_parent_id"] = None
            chunk["align_level"] = 0
            levels[new_id] = 0
            continue

        current_indices = {int(v) for v in chunk.get("indices", [])}
        candidates = [
            old_to_new[int(neighbor)]
            for neighbor in adjacency[int(old_id)]
            if int(neighbor) in old_to_new
            and old_to_new[int(neighbor)] < new_id
        ]
        if not candidates:
            candidates = list(range(new_id))
        parent_id = max(
            candidates,
            key=lambda candidate: (
                len(
                    current_indices
                    & {int(v) for v in ordered[candidate].get("indices", [])}
                ),
                -candidate,
            ),
        )
        chunk["align_parent_id"] = int(parent_id)
        chunk["align_level"] = int(levels[parent_id] + 1)
        levels[new_id] = int(levels[parent_id] + 1)


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
    lifo: bool,
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
    if lifo:
        queue.append(next_i)
    else:
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

    if strategy == "spatial_sort":
        order = list(range(len(chunks)))
        ordered = _renumber_chunks(chunks, order, strategy)
        centers = pose_centers_from_meta(meta)
        adjacency, _chunk_centers = _chunk_adjacency(chunks, centers)
        _attach_adjacency_metadata(ordered, order, adjacency)
        if any(
            str(chunk.get("partition", "")) in {"footprint_grid", "pose_grid"}
            for chunk in ordered
        ):
            _attach_parent_graph_topology(ordered, order, adjacency)
        return ordered, {
            "strategy": strategy,
            "order": order,
            "source_chunk_ids": [int(chunks[i].get("chunk_id", i)) for i in order],
            "num_adjacency_edges": int(sum(len(v) for v in adjacency) // 2),
        }

    centers = pose_centers_from_meta(meta)
    adjacency, chunk_centers = _chunk_adjacency(chunks, centers)
    scene_center = np.nanmean(centers, axis=0)

    if strategy == "depth_prior_greedy":
        start = _chunk_start_index(chunks, centers, "spatial_center_bfs")
        visited: Set[int] = set()
        predicted_frames: Set[int] = set()
        order: List[int] = []
        current = int(start)
        while len(order) < len(chunks):
            visited.add(current)
            order.append(current)
            predicted_frames.update(int(v) for v in chunks[current].get("indices", []))

            candidates = [i for i in range(len(chunks)) if i not in visited]
            if not candidates:
                break
            neighbor_set = set()
            for v in visited:
                neighbor_set.update(adjacency[v])
            current = max(
                candidates,
                key=lambda i: (
                    len({int(v) for v in chunks[i].get("overlap_indices", [])} & predicted_frames),
                    int(i in neighbor_set),
                    -float(np.linalg.norm(chunk_centers[i] - chunk_centers[current])),
                    -float(np.linalg.norm(chunk_centers[i] - scene_center)),
                    -i,
                ),
            )
    else:
        start = _chunk_start_index(chunks, centers, strategy)
        lifo = strategy == "dfs"
        queue: deque = deque([int(start)])
        visited = set()
        order = []
        while len(order) < len(chunks):
            if not queue:
                _append_disconnected_component(
                    queue,
                    visited,
                    chunk_centers,
                    scene_center,
                    lifo=lifo,
                )
                if not queue:
                    break
            current = int(queue.pop() if lifo else queue.popleft())
            if current in visited:
                continue
            visited.add(current)
            order.append(current)
            neighbors = [n for n in adjacency[current] if n not in visited]
            if lifo:
                for neighbor in reversed(neighbors):
                    queue.append(neighbor)
            else:
                for neighbor in neighbors:
                    queue.append(neighbor)

    ordered = _renumber_chunks(chunks, order, strategy)
    _attach_adjacency_metadata(ordered, order, adjacency)
    if any(
        str(chunk.get("partition", "")) in {"footprint_grid", "pose_grid"}
        for chunk in ordered
    ):
        _attach_parent_graph_topology(ordered, order, adjacency)
    return ordered, {
        "strategy": strategy,
        "order": [int(i) for i in order],
        "source_chunk_ids": [int(chunks[i].get("chunk_id", i)) for i in order],
        "start_source_chunk_id": int(chunks[order[0]].get("chunk_id", order[0])),
        "num_adjacency_edges": int(sum(len(v) for v in adjacency) // 2),
    }


# ---------------------------------------------------------------------------
# Pose-grid partition (legacy)
# ---------------------------------------------------------------------------
def build_cell_to_core(
    coords: np.ndarray,
    origin: np.ndarray,
    grid_size: float,
) -> Dict[Tuple[int, ...], List[int]]:
    cell_coords = np.floor(
        (coords - origin[None, :]) / float(grid_size)
    ).astype(np.int64)
    cell_to_core: Dict[Tuple[int, ...], List[int]] = {}
    for frame_idx, cell in enumerate(cell_coords):
        key = tuple(int(v) for v in cell)
        cell_to_core.setdefault(key, []).append(int(frame_idx))
    return cell_to_core


def are_neighbor_cells(
    a: Tuple[int, ...], b: Tuple[int, ...], radius: int
) -> bool:
    if a == b:
        return False
    return max(abs(int(x) - int(y)) for x, y in zip(a, b)) <= int(radius)


def pose_grid_cell_order(
    cell_keys: Sequence[Tuple[int, ...]],
) -> List[Tuple[int, ...]]:
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
    meta: Dict[str, object],
    axes: str = "xy",
    pose_grid_size: float = 0.0,
    max_chunk_size: int = 32,
    min_chunk_size: int = 1,
    max_chunks: int = 0,
    pose_grid_neighbor_radius: int = 1,
    coords_override: Optional[np.ndarray] = None,
    partition_name: str = "pose_grid",
    extra_grid_meta: Optional[Dict[str, object]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if coords_override is None:
        centers = pose_centers_from_meta(meta)
        axis_indices = pose_grid_axis_indices(axes)
        coords = centers[:, axis_indices]
    else:
        coords = np.asarray(coords_override, dtype=np.float64)
    origin = coords.min(axis=0)
    grid_size = float(pose_grid_size)
    target_core_size = auto_core_target_size(int(max_chunk_size))

    if grid_size <= 0:
        grid_size = estimate_pose_grid_size(coords, target_core_size)
        cell_to_core = build_cell_to_core(coords, origin, grid_size)
        num_grid_refinements = 0
    else:
        num_grid_refinements = 0
        cell_to_core = build_cell_to_core(coords, origin, grid_size)
        print(
            "[WARN] --pose_grid_size was set explicitly; automatic grid estimation "
            "from --max_chunk_size is bypassed."
        )

    oversized_cells = {
        key: len(indices)
        for key, indices in cell_to_core.items()
        if len(indices) > int(target_core_size)
    }
    if oversized_cells:
        print(
            "[WARN] Fixed grid contains dense cells; keep the scene-wide grid "
            "size and split only their image lists to respect model capacity: "
            f"oversized_cells={len(oversized_cells)}, "
            f"max_cell_images={max(oversized_cells.values())}, "
            f"core_capacity={target_core_size}."
        )

    ordered_cells = pose_grid_cell_order(cell_to_core.keys())
    cell_centers = {
        key: coords[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        for key, indices in cell_to_core.items()
    }

    chunks: List[Dict[str, object]] = []
    neighbor_radius = int(pose_grid_neighbor_radius)
    total_dropped_seam_images = 0

    chunk_order_idx = 0
    num_split_cells = 0
    for cell_order_idx, cell_key in enumerate(ordered_cells):
        cell_core_indices = sorted(cell_to_core[cell_key])
        core_groups = [
            cell_core_indices[start : start + int(target_core_size)]
            for start in range(0, len(cell_core_indices), int(target_core_size))
        ]
        if len(core_groups) > 1:
            num_split_cells += 1

        # The spatial grid remains fixed. A dense cell may produce multiple
        # capacity-bounded chunks, but it is not spatially refined as a tree.
        for cell_split_index, core_indices in enumerate(core_groups):

            seam_candidates: List[Tuple[float, int, Tuple[int, ...]]] = []
            if neighbor_radius > 0:
                center = coords[np.asarray(core_indices, dtype=np.int64)].mean(axis=0)
                core_set = set(core_indices)
                for other_key in ordered_cells:
                    if (
                        other_key != cell_key
                        and not are_neighbor_cells(cell_key, other_key, neighbor_radius)
                    ):
                        continue
                    for idx in cell_to_core[other_key]:
                        if int(idx) in core_set:
                            continue
                        dist = float(
                            np.linalg.norm(coords[int(idx)] - center)
                        )
                        seam_candidates.append(
                            (dist, int(idx), tuple(int(v) for v in other_key))
                        )

            seam_candidates.sort(key=lambda item: (item[0], item[1]))
            budget = max(0, int(max_chunk_size) - len(core_indices))
            overlap_indices = sorted(
                {idx for _dist, idx, _cell in seam_candidates[:budget]}
            )
            dropped_seam_images = max(
                0,
                len({idx for _dist, idx, _cell in seam_candidates})
                - len(overlap_indices),
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
                    "partition": str(partition_name),
                    "cell_order": int(chunk_order_idx),
                    "grid_cell_order": int(cell_order_idx),
                    "cell_split_index": int(cell_split_index),
                    "num_cell_splits": int(len(core_groups)),
                    "cell_key": tuple(int(v) for v in cell_key),
                    "indices": indices,
                    "core_indices": core_indices,
                    "raw_num_core_images": int(len(cell_core_indices)),
                    "overlap_indices": overlap_indices,
                    "core_local_indices": core_local_indices,
                    "num_seam_candidates": int(
                        len({idx for _dist, idx, _cell in seam_candidates})
                    ),
                    "num_dropped_seam_images": int(dropped_seam_images),
                }
            )
            chunk_order_idx += 1
            if int(max_chunks) > 0 and len(chunks) >= int(max_chunks):
                break
        if int(max_chunks) > 0 and len(chunks) >= int(max_chunks):
            break

    grid_meta = {
        "partition": str(partition_name),
        "axes": str(axes),
        "grid_size_requested": float(pose_grid_size),
        "grid_size_effective": float(grid_size),
        "origin": origin.astype(float).tolist(),
        "num_occupied_cells": int(len(ordered_cells)),
        "num_retained_cells": int(
            len({tuple(chunk["cell_key"]) for chunk in chunks})
        ),
        "num_chunks": int(len(chunks)),
        "num_split_cells": int(num_split_cells),
        "num_oversized_cells": int(len(oversized_cells)),
        "num_core_images": int(sum(len(chunk["core_indices"]) for chunk in chunks)),
        "num_input_images": int(len(meta["stems"])),
        "core_coverage_ratio": float(
            sum(len(chunk["core_indices"]) for chunk in chunks)
            / max(1, len(meta["stems"]))
        ),
        "num_grid_refinements": int(num_grid_refinements),
        "auto_core_target_size": int(target_core_size),
        "total_dropped_seam_images": int(total_dropped_seam_images),
        "alignment_topology": "parent_graph",
        "core_size_stats": {
            "min": int(min((len(v) for v in cell_to_core.values()), default=0)),
            "max": int(max((len(v) for v in cell_to_core.values()), default=0)),
            "mean": float(np.mean([len(v) for v in cell_to_core.values()])) if cell_to_core else 0.0,
            "std": float(np.std([len(v) for v in cell_to_core.values()])) if cell_to_core else 0.0,
            "num_under_min": int(sum(len(v) < int(min_chunk_size) for v in cell_to_core.values())),
        },
    }
    if extra_grid_meta:
        grid_meta.update(extra_grid_meta)
    num_under_min = int(grid_meta["core_size_stats"]["num_under_min"])
    if num_under_min > 0:
        print(
            "[WARN] Fixed footprint grid retained under-filled cells to preserve "
            f"full scene coverage: {num_under_min}/{len(cell_to_core)} cells have "
            f"fewer than min_chunk_size={int(min_chunk_size)} core images."
        )
    covered_core = {
        int(index)
        for chunk in chunks
        for index in chunk.get("core_indices", [])
    }
    expected_core = set(range(len(meta["stems"])))
    if int(max_chunks) <= 0 and covered_core != expected_core:
        missing = sorted(expected_core - covered_core)
        raise RuntimeError(
            "Fixed-grid chunking lost core images: "
            f"covered={len(covered_core)}/{len(expected_core)}, "
            f"first_missing={missing[:8]}"
        )
    return chunks, grid_meta


def build_footprint_grid_chunks(
    meta: Dict[str, object],
    axes: str = "xy",
    pose_grid_size: float = 0.0,
    max_chunk_size: int = 32,
    min_chunk_size: int = 1,
    max_chunks: int = 0,
    pose_grid_neighbor_radius: int = 1,
    footprint_source: str = "auto",
    footprint_sample_stride: int = 16,
    footprint_min_points: int = 32,
    footprint_quantile_min: float = 0.02,
    footprint_quantile_max: float = 0.98,
    footprint_lookat_distance: float = 0.0,
    footprint_workers: int = 0,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    axis_indices = pose_grid_axis_indices(axes)
    centers_world, _bbox_min, _bbox_max, feature_meta = footprint_features_from_meta(
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
    flight_frame = estimate_main_flight_frame_from_pose(meta, axis_indices)
    centers = transform_points_to_flight_frame(centers_world, flight_frame)
    return build_pose_grid_chunks(
        meta=meta,
        axes=axes,
        pose_grid_size=pose_grid_size,
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        max_chunks=max_chunks,
        pose_grid_neighbor_radius=pose_grid_neighbor_radius,
        coords_override=centers,
        partition_name="footprint_grid",
        extra_grid_meta={
            "footprint": feature_meta,
            "footprint_centers": centers_world.astype(float).tolist(),
            "footprint_centers_flight": centers.astype(float).tolist(),
            "footprint_flight_frame": flight_frame,
        },
    )


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


def _depth_footprint_worker(args: Tuple[object, ...]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str]:
    (
        frame_index,
        depth_path,
        K,
        T_c2w,
        cam_width,
        cam_height,
        target_h,
        target_w,
        depth_scale,
        depth_min,
        depth_max,
        axis_indices,
        stride,
        footprint_min_points,
        footprint_quantile_min,
        footprint_quantile_max,
    ) = args
    try:
        depth_raw = read_depth(Path(str(depth_path)), depth_scale=float(depth_scale))
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
            depth_raw,
            target_h=int(target_h),
            target_w=int(target_w),
        )

        stride = max(1, int(stride))
        ys = np.arange(0, depth.shape[0], stride, dtype=np.float64)
        xs = np.arange(0, depth.shape[1], stride, dtype=np.float64)
        u, v = np.meshgrid(xs, ys)
        z = depth[::stride, ::stride].astype(np.float64)
        valid = (
            np.isfinite(z)
            & (z > float(depth_min))
            & (z < float(depth_max))
        )
        if int(np.count_nonzero(valid)) < int(footprint_min_points):
            return int(frame_index), None, None, None, "not_enough_valid_depth"

        fx, fy = float(K_scaled[0, 0]), float(K_scaled[1, 1])
        cx, cy = float(K_scaled[0, 2]), float(K_scaled[1, 2])
        if abs(fx) < 1e-12 or abs(fy) < 1e-12:
            return int(frame_index), None, None, None, "invalid_intrinsics"

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=-1)
        pts_cam = pts_cam[valid].reshape(-1, 3)
        T = np.asarray(T_c2w, dtype=np.float64)
        pts_world = pts_cam @ T[:3, :3].T + T[:3, 3][None, :]
        finite = np.isfinite(pts_world).all(axis=1)
        pts_world = pts_world[finite]
        if pts_world.shape[0] < int(footprint_min_points):
            return int(frame_index), None, None, None, "not_enough_finite_points"

        coords = pts_world[:, tuple(axis_indices)].astype(np.float32, copy=False)
        bbox_min, bbox_max = robust_bbox(
            coords,
            float(footprint_quantile_min),
            float(footprint_quantile_max),
        )
        center_coord = np.median(coords, axis=0).astype(np.float64)
        return int(frame_index), center_coord, bbox_min, bbox_max, "ok"
    except Exception as exc:
        return int(frame_index), None, None, None, f"error:{exc}"


def _depth_footprints_from_meta(
    meta: Dict[str, object],
    axis_indices: Tuple[int, ...],
    footprint_sample_stride: int,
    footprint_min_points: int,
    footprint_quantile_min: float,
    footprint_quantile_max: float,
    footprint_workers: int,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    stems = list(meta["stems"])
    cams = meta.get("cams", {})
    depth_paths = meta.get("depth_paths", {})
    jobs: List[Tuple[object, ...]] = []
    for i, stem in enumerate(stems):
        cam = cams.get(stem)
        depth_path = depth_paths.get(stem) if isinstance(depth_paths, dict) else None
        if cam is None or not depth_path:
            continue
        jobs.append(
            (
                int(i),
                str(depth_path),
                np.asarray(cam["K"], dtype=np.float64),
                np.asarray(cam["T_c2w"], dtype=np.float64),
                cam.get("width"),
                cam.get("height"),
                int(meta["target_h"]),
                int(meta["target_w"]),
                float(meta.get("depth_scale", 1.0)),
                float(meta.get("depth_min", 1e-6)),
                float(meta.get("depth_max", 1e6)),
                tuple(int(a) for a in axis_indices),
                int(footprint_sample_stride),
                int(footprint_min_points),
                float(footprint_quantile_min),
                float(footprint_quantile_max),
            )
        )

    if not jobs:
        return {}

    workers = int(footprint_workers)
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_depth_footprint_worker, jobs))
    else:
        results = [_depth_footprint_worker(job) for job in jobs]

    out: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    failed = 0
    for frame_index, center, bbox_min, bbox_max, status in results:
        if status == "ok" and center is not None and bbox_min is not None and bbox_max is not None:
            out[int(frame_index)] = (
                np.asarray(center, dtype=np.float64),
                np.asarray(bbox_min, dtype=np.float64),
                np.asarray(bbox_max, dtype=np.float64),
            )
        else:
            failed += 1
    if failed:
        print(
            f"[INFO] footprint depth sampling skipped/fell back for "
            f"{failed}/{len(results)} frames."
        )
    return out


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
    counts = {"depth": 0, "lookat": 0, "center": 0}
    depth_footprints = {}
    if source_requested in {"auto", "depth"}:
        depth_footprints = _depth_footprints_from_meta(
            meta=meta,
            axis_indices=axis_indices,
            footprint_sample_stride=footprint_sample_stride,
            footprint_min_points=footprint_min_points,
            footprint_quantile_min=footprint_quantile_min,
            footprint_quantile_max=footprint_quantile_max,
            footprint_workers=footprint_workers,
        )

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

        if source_requested in {"auto", "depth"}:
            fp = depth_footprints.get(int(i))
            if fp is not None:
                center_coord, bbox_min, bbox_max = fp
                source_used = "depth"

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
        "depth_sampling_workers": int(footprint_workers),
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
    axis_indices = pose_grid_axis_indices(axes)
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

    if chunks and len(chunks[-1]["core_indices"]) < int(min_chunk_size) and len(chunks) > 1:
        # Keep the tail rather than dropping frames; temporal ablations require
        # complete sequence coverage. Record the under-filled tail explicitly.
        tail_under_min = True
    else:
        tail_under_min = False

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
    pose_grid_size: float = 0.0,
    pose_grid_neighbor_radius: int = 1,
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

    if partition == "temporal":
        return build_temporal_chunks(
            meta=meta,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            max_chunks=max_chunks,
            overlap_ratio=temporal_overlap_ratio,
        )

    if axes == "auto":
        axes = infer_spatial_axes(meta)
        print(f"[INFO] Auto spatial axes: {axes}")

    if footprint_sample_stride <= 0:
        footprint_sample_stride = infer_footprint_stride(meta)

    if partition == "pose_grid":
        return build_pose_grid_chunks(
            meta=meta,
            axes=axes,
            pose_grid_size=pose_grid_size,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            max_chunks=max_chunks,
            pose_grid_neighbor_radius=pose_grid_neighbor_radius,
        )

    if partition == "footprint_grid":
        return build_footprint_grid_chunks(
            meta=meta,
            axes=axes,
            pose_grid_size=pose_grid_size,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            max_chunks=max_chunks,
            pose_grid_neighbor_radius=pose_grid_neighbor_radius,
            footprint_source=footprint_source,
            footprint_sample_stride=footprint_sample_stride,
            footprint_min_points=footprint_min_points,
            footprint_quantile_min=footprint_quantile_min,
            footprint_quantile_max=footprint_quantile_max,
            footprint_lookat_distance=footprint_lookat_distance,
            footprint_workers=footprint_workers,
        )

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
