# -*- coding: utf-8 -*-
"""Disk-backed storage for per-chunk spatial predictions.

Chunk records should stay lightweight. Dense point maps, RGBs, and point clouds
live in per-chunk npz files and are loaded only by stages that need them.
"""

from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ARRAY_KEYS = {
    "chunk_pred_points",
    "chunk_pred_colors",
    "core_pred_points",
    "core_pred_colors",
    "_pred_maps",
    "_pred_valid_masks",
    "_pred_valid_masks_unmasked",
    "rgbs",
    "_chunk_intrinsics",
    "_chunk_point_local_indices",
    "_core_local_indices",
}


def chunk_cache_dir_for_output(output_path: Path) -> Path:
    return Path(output_path).expanduser().resolve() / "chunk_cache"


def chunk_cache_path(cache_dir: Path, chunk_id: int) -> Path:
    return Path(cache_dir) / f"chunk_{int(chunk_id):03d}.npz"


def _pack_sequence(values: Sequence[object]) -> np.ndarray:
    values = list(values)
    if not values:
        return np.empty((0,), dtype=object)
    arrays = [np.asarray(v) if v is not None else None for v in values]
    if all(a is not None for a in arrays):
        shapes = [a.shape for a in arrays if a is not None]
        dtypes = [a.dtype for a in arrays if a is not None]
        if len(set(shapes)) == 1 and len(set(str(d) for d in dtypes)) == 1:
            return np.stack([a for a in arrays if a is not None], axis=0)
    out = np.empty((len(values),), dtype=object)
    for i, value in enumerate(values):
        out[i] = value
    return out


def _unpack_array(value: np.ndarray):
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [value[i] for i in range(value.shape[0])]
    return value


def _as_int_array(values: Sequence[int]) -> np.ndarray:
    return np.asarray([int(v) for v in values], dtype=np.int64)


def _payload_keys(arrays: Mapping[str, object]) -> List[str]:
    return sorted(str(k) for k, v in arrays.items() if v is not None)


def _build_payload(arrays: Mapping[str, object]) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    for key, value in arrays.items():
        if value is None:
            continue
        if key in {"_pred_maps", "_pred_valid_masks", "_pred_valid_masks_unmasked", "rgbs", "_chunk_intrinsics"}:
            payload[key] = _pack_sequence(value)  # type: ignore[arg-type]
        elif key in {"_chunk_point_local_indices", "_core_local_indices"}:
            payload[key] = _as_int_array(value)  # type: ignore[arg-type]
        else:
            payload[key] = np.asarray(value)
    return payload


def _prepare_record_for_cache(
    record: Dict[str, object],
    cache_dir: Path,
    arrays: Mapping[str, object],
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = chunk_cache_path(cache_dir, int(record["chunk_id"]))
    record["chunk_cache_path"] = str(path)
    record["chunk_cache_keys"] = _payload_keys(arrays)
    return path


def _write_chunk_cache_npz(path: Path, arrays: Mapping[str, object]) -> None:
    payload = _build_payload(arrays)
    np.savez(path, **payload)


def save_chunk_cache(
    record: Dict[str, object],
    cache_dir: Path,
    arrays: Mapping[str, object],
) -> Path:
    path = _prepare_record_for_cache(record, cache_dir, arrays)
    _write_chunk_cache_npz(path, arrays)
    return path


class AsyncChunkCacheWriter:
    """Bounded background writer for large per-chunk cache payloads."""

    def __init__(
        self,
        cache_dir: Path,
        max_workers: int = 1,
        max_pending: int = 0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max(0, int(max_workers))
        if max_pending <= 0:
            max_pending = max(1, self.max_workers)
        self.max_pending = max(1, int(max_pending))
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending: List[Future[None]] = []
        if self.max_workers > 0:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="chunk-cache-writer",
            )

    def submit(
        self,
        record: Dict[str, object],
        arrays: Mapping[str, object],
    ) -> Path:
        arrays = dict(arrays)
        path = _prepare_record_for_cache(record, self.cache_dir, arrays)
        if self._executor is None:
            _write_chunk_cache_npz(path, arrays)
            return path

        self._raise_finished()
        while len(self._pending) >= self.max_pending:
            self._drain_one()
        self._pending.append(self._executor.submit(_write_chunk_cache_npz, path, arrays))
        return path

    def _raise_finished(self) -> None:
        still_pending: List[Future[None]] = []
        for future in self._pending:
            if future.done():
                future.result()
            else:
                still_pending.append(future)
        self._pending = still_pending

    def _drain_one(self) -> None:
        if not self._pending:
            return
        done, pending = wait(self._pending, return_when=FIRST_COMPLETED)
        for future in done:
            future.result()
        self._pending = list(pending)

    def wait(self) -> None:
        try:
            while self._pending:
                self._drain_one()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None


def load_chunk_cache(
    record: Mapping[str, object],
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    path = record.get("chunk_cache_path", None)
    if not path:
        return {}
    path = Path(str(path))
    if not path.exists():
        raise FileNotFoundError(f"Chunk cache not found: {path}")

    want = set(keys) if keys is not None else None
    out: Dict[str, object] = {}
    with np.load(path, allow_pickle=True) as data:
        for key in data.files:
            if want is not None and key not in want:
                continue
            out[key] = _unpack_array(data[key])
    return out


def update_chunk_cache(
    record: Dict[str, object],
    updates: Mapping[str, object],
) -> None:
    path = record.get("chunk_cache_path", None)
    if not path:
        raise ValueError("record does not have chunk_cache_path")
    path = Path(str(path))
    existing = load_chunk_cache(record)
    existing.update({k: v for k, v in updates.items() if v is not None})
    save_chunk_cache(record, path.parent, existing)


def get_cached_array(record: Mapping[str, object], key: str, default=None):
    if key in record:
        return record[key]
    return load_chunk_cache(record, keys=[key]).get(key, default)


def get_cached_points(
    record: Mapping[str, object],
    key: str,
) -> np.ndarray:
    value = get_cached_array(record, key, np.empty((0, 3), np.float32))
    return np.asarray(value, dtype=np.float32).reshape(-1, 3)


def get_cached_colors(
    record: Mapping[str, object],
    key: str,
) -> np.ndarray:
    value = get_cached_array(record, key, np.empty((0, 3), np.uint8))
    return np.asarray(value, dtype=np.uint8).reshape(-1, 3)


def get_cached_sequence(record: Mapping[str, object], key: str) -> List[object]:
    value = get_cached_array(record, key, [])
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return [value[i] for i in range(value.shape[0])]
    return list(value)


def cached_shape(record: Mapping[str, object], key: str) -> Tuple[int, ...]:
    value = get_cached_array(record, key, None)
    if value is None:
        return (0,)
    return tuple(np.asarray(value).shape)


def strip_array_payload(record: Dict[str, object]) -> None:
    for key in list(ARRAY_KEYS):
        record.pop(key, None)
    record.pop("_global_to_local_index", None)


def write_chunk_record_manifest(records: Sequence[Mapping[str, object]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for record in records:
        payload.append(
            {
                "chunk_id": int(record.get("chunk_id", -1)),
                "indices": [int(i) for i in record.get("indices", [])],
                "core_indices": [int(i) for i in record.get("core_indices", [])],
                "overlap_indices": [int(i) for i in record.get("overlap_indices", [])],
                "adjacent_chunk_ids": [
                    int(i) for i in record.get("adjacent_chunk_ids", [])
                ],
                "alignment_topology": record.get("alignment_topology", None),
                "align_parent_id": record.get("align_parent_id", None),
                "align_level": int(record.get("align_level", 0)),
                "chunk_cache_path": record.get("chunk_cache_path", None),
                "chunk_cache_keys": list(record.get("chunk_cache_keys", [])),
                "post_chunk_align_transform": record.get(
                    "post_chunk_align_transform", None
                ),
                "post_chunk_align_meta": record.get(
                    "post_chunk_align_meta", None
                ),
                "post_chunk_align_edges": record.get(
                    "post_chunk_align_edges", []
                ),
            }
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
