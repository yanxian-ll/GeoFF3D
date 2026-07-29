# -*- coding: utf-8 -*-
"""Lazy per-chunk post-alignment transforms.

Chunk cache files store raw model outputs. Deferred chunk alignment records a
similarity transform per chunk, and downstream stages apply it on demand when
they need final world coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence as AbcSequence
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from geoff3d.spatial_rrd.chunk_cache import get_cached_points, get_cached_sequence
from geoff3d.spatial_rrd.geometry_align import (
    apply_similarity_to_cameras,
    apply_similarity_to_point_maps,
    apply_similarity_to_points,
)


def identity_similarity() -> Tuple[float, np.ndarray, np.ndarray]:
    return (
        1.0,
        np.eye(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )


def similarity_to_meta(
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> Dict[str, object]:
    return {
        "scale": float(scale),
        "R": np.asarray(R, dtype=np.float32).reshape(3, 3).tolist(),
        "t": np.asarray(t, dtype=np.float32).reshape(3).tolist(),
        "applied_to_cache": False,
    }


def get_record_similarity(
    record: Mapping[str, object],
) -> Tuple[float, np.ndarray, np.ndarray]:
    meta = record.get("post_chunk_align_transform", None)
    if not isinstance(meta, Mapping):
        return identity_similarity()
    try:
        scale = float(meta.get("scale", 1.0))
        R = np.asarray(meta.get("R", np.eye(3)), dtype=np.float64).reshape(3, 3)
        t = np.asarray(meta.get("t", np.zeros(3)), dtype=np.float64).reshape(3)
    except Exception:
        return identity_similarity()
    if not np.isfinite(scale) or not np.isfinite(R).all() or not np.isfinite(t).all():
        return identity_similarity()
    return scale, R, t


def set_record_similarity(
    record: Dict[str, object],
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> None:
    record["post_chunk_align_transform"] = similarity_to_meta(scale, R, t)


def ensure_record_similarity(record: Dict[str, object]) -> None:
    if "post_chunk_align_transform" not in record:
        scale, R, t = identity_similarity()
        set_record_similarity(record, scale, R, t)


def is_identity_similarity(
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
    atol: float = 1e-8,
) -> bool:
    return (
        abs(float(scale) - 1.0) <= float(atol)
        and np.allclose(R, np.eye(3), atol=float(atol))
        and np.allclose(t, np.zeros(3), atol=float(atol))
    )


def compose_similarity(
    first: Tuple[float, np.ndarray, np.ndarray],
    second: Tuple[float, np.ndarray, np.ndarray],
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return second(first(x)) for similarities x -> s R x + t."""
    s1, R1, t1 = first
    s2, R2, t2 = second
    s = float(s2) * float(s1)
    R = np.asarray(R2, dtype=np.float64) @ np.asarray(R1, dtype=np.float64)
    t = (
        float(s2)
        * (np.asarray(R2, dtype=np.float64) @ np.asarray(t1, dtype=np.float64))
        + np.asarray(t2, dtype=np.float64)
    )
    return s, R, t


def compose_record_similarity(
    record: Dict[str, object],
    delta_scale: float,
    delta_R: np.ndarray,
    delta_t: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    current = get_record_similarity(record)
    delta = (
        float(delta_scale),
        np.asarray(delta_R, dtype=np.float64).reshape(3, 3),
        np.asarray(delta_t, dtype=np.float64).reshape(3),
    )
    composed = compose_similarity(current, delta)
    set_record_similarity(record, *composed)
    return composed


def apply_record_similarity_to_points(
    record: Mapping[str, object],
    points: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    scale, R, t = get_record_similarity(record)
    if points.size == 0 or is_identity_similarity(scale, R, t):
        return points
    return apply_similarity_to_points(points, scale, R, t)


def apply_record_similarity_to_point_maps(
    record: Mapping[str, object],
    pred_maps: Sequence[object],
) -> List[np.ndarray]:
    maps = [np.asarray(m, dtype=np.float32) for m in pred_maps]
    scale, R, t = get_record_similarity(record)
    if not maps or is_identity_similarity(scale, R, t):
        return maps
    return apply_similarity_to_point_maps(maps, scale, R, t)


def apply_record_similarity_to_cameras(
    record: Mapping[str, object],
    pred_cams: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    scale, R, t = get_record_similarity(record)
    if not pred_cams or is_identity_similarity(scale, R, t):
        return [dict(cam) for cam in pred_cams]
    return apply_similarity_to_cameras(pred_cams, scale, R, t)


def get_transformed_cached_points(
    record: Mapping[str, object],
    key: str,
) -> np.ndarray:
    return apply_record_similarity_to_points(record, get_cached_points(record, key))


def get_transformed_cached_point_maps(
    record: Mapping[str, object],
    key: str = "_pred_maps",
) -> List[np.ndarray]:
    return apply_record_similarity_to_point_maps(record, get_cached_sequence(record, key))


def get_transformed_cameras(
    record: Mapping[str, object],
) -> List[Dict[str, object]]:
    refined = record.get("ba_refined_cameras", None)
    if isinstance(refined, AbcSequence):
        return [dict(cam) for cam in refined]  # type: ignore[arg-type]
    cams = record.get("pred_cams", [])
    if not isinstance(cams, AbcSequence):
        return []
    return apply_record_similarity_to_cameras(record, cams)  # type: ignore[arg-type]
