# -*- coding: utf-8 -*-
"""Build larger 3DGS optimization bundles from feed-forward spatial chunks.

Feed-forward chunks are for model inference. 3DGS bundles are for rendering
optimization and can merge multiple feed-forward chunks.

Rules:
  - If the scene has <= single_max_images unique render images:
      use one scene-level 3DGS bundle.
  - Else:
      greedily cluster nearby feed-forward chunks into larger bundles.
  - 3DGS point initialization uses only core views.
  - 3DGS rendering supervision uses core + seam views.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from geoff3d.slrf.chunk_cache import get_cached_sequence
from geoff3d.slrf.chunk_transform import (
    get_transformed_cached_point_maps,
    get_transformed_cameras,
)


def _camera_by_local_index(
    pred_cams: Sequence[Dict[str, object]],
) -> Dict[int, Dict[str, object]]:
    out: Dict[int, Dict[str, object]] = {}
    for cam in pred_cams:
        idx = int(cam.get("pred_index", -1))
        if idx >= 0:
            out[idx] = cam
    return out


def _chunk_center(record: Dict[str, object]) -> np.ndarray:
    """Estimate spatial center of a feed-forward chunk from core cameras."""
    core_set = set(int(i) for i in record.get("core_indices", []))
    indices = [int(i) for i in record.get("indices", [])]
    cam_by_local = _camera_by_local_index(get_transformed_cameras(record))

    centers: List[np.ndarray] = []

    for local_i, global_i in enumerate(indices):
        if global_i not in core_set:
            continue
        cam = cam_by_local.get(local_i)
        if cam is None:
            continue
        T = np.asarray(cam.get("T_c2w"), dtype=np.float32)
        if T.shape == (4, 4) and np.isfinite(T).all():
            centers.append(T[:3, 3].astype(np.float32))

    if not centers:
        for cam in get_transformed_cameras(record):
            T = np.asarray(cam.get("T_c2w"), dtype=np.float32)
            if T.shape == (4, 4) and np.isfinite(T).all():
                centers.append(T[:3, 3].astype(np.float32))

    if not centers:
        return np.zeros(3, dtype=np.float32)

    return np.stack(centers, axis=0).mean(axis=0).astype(np.float32)


def _record_sets(record: Dict[str, object]) -> Tuple[Set[int], Set[int]]:
    core = set(int(i) for i in record.get("core_indices", []))
    render = set(int(i) for i in record.get("indices", []))
    return core, render


def build_gsplat_optimization_bundles(
    chunk_records: Sequence[Dict[str, object]],
    *,
    single_max_images: int,
    max_images_per_bundle: int,
    min_core_images_per_bundle: int,
) -> List[Dict[str, object]]:
    """Cluster feed-forward chunks into 3DGS optimization bundles."""
    records = list(chunk_records)
    if not records:
        return []

    all_render: Set[int] = set()
    for rec in records:
        _, render = _record_sets(rec)
        all_render.update(render)

    single_max_images = int(single_max_images)
    max_images_per_bundle = max(1, int(max_images_per_bundle))
    min_core_images_per_bundle = max(1, int(min_core_images_per_bundle))

    if single_max_images <= 0 or len(all_render) <= single_max_images:
        all_core: Set[int] = set()
        all_render_s = set()
        for rec in records:
            core, render = _record_sets(rec)
            all_core.update(core)
            all_render_s.update(render)

        return [
            {
                "bundle_id": 0,
                "name": "scene",
                "record_ids": list(range(len(records))),
                "core_global_indices": sorted(all_core),
                "render_global_indices": sorted(all_render_s),
                "split": False,
                "num_core_images": int(len(all_core)),
                "num_render_images": int(len(all_render_s)),
            }
        ]

    centers = np.stack([_chunk_center(rec) for rec in records], axis=0)
    remaining: Set[int] = set(range(len(records)))
    bundles: List[Dict[str, object]] = []

    while remaining:
        seed_id = min(remaining)
        remaining.remove(seed_id)

        bundle_ids = [seed_id]
        core_set, render_set = _record_sets(records[seed_id])
        center = centers[seed_id].copy()

        while remaining:
            candidates = sorted(
                remaining,
                key=lambda rid: float(np.linalg.norm(centers[rid] - center)),
            )

            chosen: Optional[int] = None
            chosen_core: Set[int] = set()
            chosen_render: Set[int] = set()

            for rid in candidates:
                rec_core, rec_render = _record_sets(records[rid])
                new_render = render_set | rec_render
                if len(new_render) <= max_images_per_bundle:
                    chosen = rid
                    chosen_core = rec_core
                    chosen_render = rec_render
                    break

            if chosen is None:
                if len(core_set) < min_core_images_per_bundle and candidates:
                    chosen = candidates[0]
                    chosen_core, chosen_render = _record_sets(records[chosen])
                else:
                    break

            remaining.remove(chosen)
            bundle_ids.append(chosen)
            core_set.update(chosen_core)
            render_set.update(chosen_render)
            center = centers[bundle_ids].mean(axis=0)

            if (
                len(core_set) >= min_core_images_per_bundle
                and len(render_set) >= max_images_per_bundle
            ):
                break

        bundle_id = len(bundles)
        bundles.append(
            {
                "bundle_id": int(bundle_id),
                "name": f"bundle_{bundle_id:03d}",
                "record_ids": [int(i) for i in bundle_ids],
                "core_global_indices": sorted(int(i) for i in core_set),
                "render_global_indices": sorted(int(i) for i in render_set),
                "split": True,
                "num_core_images": int(len(core_set)),
                "num_render_images": int(len(render_set)),
            }
        )

    return bundles


def collect_gsplat_bundle_inputs(
    chunk_records: Sequence[Dict[str, object]],
    bundle: Dict[str, object],
) -> Dict[str, object]:
    """Collect init points and render views for one 3DGS bundle.

    Initialization: only core views from selected feed-forward chunks.
    Rendering: core + seam views from selected feed-forward chunks.
    """
    record_ids = [int(i) for i in bundle["record_ids"]]
    core_global = set(int(i) for i in bundle["core_global_indices"])
    render_global = set(int(i) for i in bundle["render_global_indices"])

    init_maps: List[np.ndarray] = []
    init_masks: List[np.ndarray] = []
    init_rgbs: List[np.ndarray] = []
    init_global_indices: List[int] = []

    render_views_raw: List[Dict[str, object]] = []
    render_rgbs: List[np.ndarray] = []
    render_cams: List[Dict[str, object]] = []
    render_global_indices: List[int] = []

    seen_init: Set[int] = set()
    seen_render: Set[int] = set()

    # 1. Point initialization: only core views.
    for rid in record_ids:
        rec = chunk_records[rid]
        pred_maps = get_transformed_cached_point_maps(rec, "_pred_maps")
        pred_masks = get_cached_sequence(rec, "_pred_valid_masks")
        rgbs = get_cached_sequence(rec, "rgbs")
        if not pred_maps or not pred_masks or not rgbs:
            continue

        indices = [int(i) for i in rec["indices"]]
        core_set = set(int(i) for i in rec["core_indices"])

        for local_i, global_i in enumerate(indices):
            if global_i not in core_set:
                continue
            if global_i not in core_global:
                continue
            if global_i in seen_init:
                continue
            if local_i >= len(pred_maps) or local_i >= len(pred_masks):
                continue

            init_maps.append(pred_maps[local_i])
            init_masks.append(pred_masks[local_i])
            init_rgbs.append(rgbs[local_i])
            init_global_indices.append(global_i)
            seen_init.add(global_i)

    # 2. Rendering: core views first, then seam.
    def try_add_render(rec: Dict[str, object], local_i: int, global_i: int) -> None:
        if global_i not in render_global:
            return
        if global_i in seen_render:
            return

        intrinsics = get_cached_sequence(rec, "_chunk_intrinsics")
        if intrinsics is None or local_i >= len(intrinsics):
            return
        K = intrinsics[local_i]
        if K is None:
            return

        cam_by_local = _camera_by_local_index(get_transformed_cameras(rec))
        cam = cam_by_local.get(local_i)
        if cam is None:
            return
        T = np.asarray(cam.get("T_c2w"), dtype=np.float32)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            return

        stem = str(rec["stems"][local_i])
        render_idx = len(render_rgbs)

        render_views_raw.append(
            {
                "camera_intrinsics": np.asarray(K, dtype=np.float32),
                "label": stem,
                "global_index": int(global_i),
            }
        )
        rgbs = get_cached_sequence(rec, "rgbs")
        if local_i >= len(rgbs):
            return
        render_rgbs.append(rgbs[local_i])
        render_cams.append(
            {
                "stem": stem,
                "pred_index": int(render_idx),
                "global_index": int(global_i),
                "T_c2w": T.astype(np.float32),
            }
        )
        render_global_indices.append(int(global_i))
        seen_render.add(global_i)

    for rid in record_ids:
        rec = chunk_records[rid]
        indices = [int(i) for i in rec["indices"]]
        core_set = set(int(i) for i in rec["core_indices"])
        for local_i, global_i in enumerate(indices):
            if global_i in core_set:
                try_add_render(rec, local_i, global_i)

    for rid in record_ids:
        rec = chunk_records[rid]
        indices = [int(i) for i in rec["indices"]]
        for local_i, global_i in enumerate(indices):
            try_add_render(rec, local_i, global_i)

    return {
        "init_pred_maps": init_maps,
        "init_pred_valid_masks": init_masks,
        "init_rgbs": init_rgbs,
        "init_global_indices": init_global_indices,
        "render_views_raw": render_views_raw,
        "render_rgbs": render_rgbs,
        "render_cams": render_cams,
        "render_global_indices": render_global_indices,
    }
