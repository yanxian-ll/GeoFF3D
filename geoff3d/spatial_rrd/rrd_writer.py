# -*- coding: utf-8 -*-
"""Rerun RRD output: logging, saving, eval exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import rerun as rr

try:
    import rerun.blueprint as rrb
except Exception:
    rrb = None

from geoff3d.spatial_rrd.scene_io import (
    load_gt_points_from_meta,
    sanitize_name,
    sample_points_and_colors,
    voxel_downsample,
)
from geoff3d.spatial_rrd.chunk_cache import (
    get_cached_colors,
    get_cached_sequence,
)
from geoff3d.spatial_rrd.chunk_transform import (
    get_transformed_cached_points,
    get_transformed_cached_point_maps,
    get_transformed_cameras,
)
from geoff3d.spatial_rrd.chunk_artifacts import (
    make_chunk_color_lookup,
    save_chunk_footprint_xy_visualization,
)


# ---------------------------------------------------------------------------
# Rerun compatibility
# ---------------------------------------------------------------------------
def rr_set_time_compat(name: str, sequence: int) -> None:
    try:
        rr.set_time(name, sequence=sequence)
    except AttributeError:
        rr.set_time_sequence(name, sequence)


def rr_disconnect_compat() -> None:
    disconnect_fn = getattr(rr, "disconnect", None)
    shutdown_fn = getattr(rr, "shutdown", None)
    try:
        if callable(disconnect_fn):
            disconnect_fn()
        elif callable(shutdown_fn):
            shutdown_fn()
    except Exception:
        pass


def rr_init_save_compat(
    app_id: str, recording_id: str, save_rrd: Path
) -> None:
    try:
        rr.init(app_id, recording_id=recording_id, spawn=False)
    except TypeError:
        rr.init(app_id, spawn=False)
    rr.save(str(save_rrd))


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------
def send_blueprint(
    background=(255, 255, 255), hide_grid: bool = False
) -> None:
    if rrb is None:
        return
    try:
        line_grid = rrb.LineGrid3D(visible=not hide_grid)
        blueprint = rrb.Blueprint(
            rrb.Spatial3DView(
                origin="/world",
                name="Prediction Scene",
                background=list(background),
                line_grid=line_grid,
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)
    except Exception as e:
        print(f"[WARN] failed to send Rerun blueprint: {e}")


# ---------------------------------------------------------------------------
# View coordinates
# ---------------------------------------------------------------------------
def log_view_coordinates(mode: str) -> None:
    mode = str(mode)
    candidates = [mode]
    if mode == "RIGHT_HAND_Z_UP":
        candidates.append("RFU")
    for name in candidates:
        obj = getattr(rr.ViewCoordinates, name, None)
        if obj is None:
            continue
        try:
            rr.log(
                "world", obj() if callable(obj) else obj, static=True
            )
            return
        except Exception:
            continue
    rr.log("world", rr.ViewCoordinates.RDF, static=True)


# ---------------------------------------------------------------------------
# Point clouds
# ---------------------------------------------------------------------------
def log_points(
    entity: str,
    points: np.ndarray,
    colors: np.ndarray,
    radius: float,
) -> None:
    if points.shape[0] == 0:
        return
    kwargs = {
        "positions": points.astype(np.float32),
        "colors": colors.astype(np.uint8),
    }
    if radius > 0:
        kwargs["radii"] = float(radius)
    rr.log(entity, rr.Points3D(**kwargs))


# ---------------------------------------------------------------------------
# Camera axes
# ---------------------------------------------------------------------------
def make_camera_axes_strips(
    cams: Sequence[Dict[str, object]],
    axis_size: float,
    colors_xyz,
):
    strips: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    for cam in cams:
        T = np.asarray(cam["T_c2w"], dtype=np.float64)
        R = T[:3, :3]
        o = T[:3, 3]
        strips.extend(
            [
                np.stack([o, o + R[:, 0] * axis_size], axis=0).astype(
                    np.float32
                ),
                np.stack([o, o + R[:, 1] * axis_size], axis=0).astype(
                    np.float32
                ),
                np.stack([o, o + R[:, 2] * axis_size], axis=0).astype(
                    np.float32
                ),
            ]
        )
        colors.extend([np.asarray(c, dtype=np.uint8) for c in colors_xyz])
    return strips, colors


def log_camera_axes(
    entity: str,
    cams: Sequence[Dict[str, object]],
    axis_size: float,
    radius: float,
    colors_xyz,
) -> None:
    strips, colors = make_camera_axes_strips(cams, axis_size, colors_xyz)
    if not strips:
        return
    kwargs = {"strips": strips, "colors": colors}
    if radius > 0:
        kwargs["radii"] = float(radius)
    rr.log(entity, rr.LineStrips3D(**kwargs))


def log_camera_labels(
    entity: str,
    cams: Sequence[Dict[str, object]],
    color,
) -> None:
    if not cams:
        return
    centers = np.asarray(
        [np.asarray(c["T_c2w"])[:3, 3] for c in cams], dtype=np.float32
    )
    labels = [
        str(c.get("stem", f"cam_{i:03d}")) for i, c in enumerate(cams)
    ]
    colors = np.repeat(
        np.asarray([color], dtype=np.uint8), len(cams), axis=0
    )
    try:
        rr.log(
            entity,
            rr.Points3D(
                positions=centers, colors=colors, labels=labels, radii=0.0
            ),
        )
    except TypeError:
        rr.log(
            entity,
            rr.Points3D(
                positions=centers, colors=colors, labels=labels
            ),
        )


# ---------------------------------------------------------------------------
# Axis size
# ---------------------------------------------------------------------------
def estimate_axis_size(
    point_arrays: Sequence[np.ndarray], explicit: float
) -> float:
    if explicit > 0:
        return float(explicit)
    valid = []
    for pts in point_arrays:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        finite = np.isfinite(pts).all(axis=1)
        if finite.any():
            valid.append(pts[finite])
    if not valid:
        return 0.1
    pts = np.concatenate(valid, axis=0)
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    if not np.isfinite(diag) or diag <= 0:
        diag = 1.0
    return max(diag * 0.03, 1e-4)


# ---------------------------------------------------------------------------
# World axes marker
# ---------------------------------------------------------------------------
def parse_signed_axis(axis: str) -> np.ndarray:
    axis = axis.strip().lower()
    sign = -1.0 if axis.startswith("-") else 1.0
    name = axis[1:] if axis.startswith("-") else axis
    if name == "x":
        return np.array([sign, 0.0, 0.0], dtype=np.float64)
    if name == "y":
        return np.array([0.0, sign, 0.0], dtype=np.float64)
    if name == "z":
        return np.array([0.0, 0.0, sign], dtype=np.float64)
    raise ValueError(f"Invalid axis {axis}; expected x/y/z/-x/-y/-z")


def log_world_axes_marker(
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
        origin = (
            bbox_center
            + parse_signed_axis(up_axis) * size * float(up_offset_ratio)
        )
    else:
        raise ValueError(
            "world_axes_origin must be 'zero' or 'scene_center'"
        )

    x_end = origin + np.array([size, 0.0, 0.0], dtype=np.float64)
    y_end = origin + np.array([0.0, size, 0.0], dtype=np.float64)
    z_end = origin + np.array([0.0, 0.0, size], dtype=np.float64)
    strips = [
        np.stack([origin, x_end]).astype(np.float32),
        np.stack([origin, y_end]).astype(np.float32),
        np.stack([origin, z_end]).astype(np.float32),
    ]
    axis_colors = [
        np.array([255, 0, 0], dtype=np.uint8),
        np.array([0, 220, 0], dtype=np.uint8),
        np.array([40, 80, 255], dtype=np.uint8),
    ]
    kwargs = {
        "strips": strips,
        "colors": axis_colors,
        "labels": ["world +X", "world +Y", "world +Z"],
    }
    if radius > 0:
        kwargs["radii"] = float(radius)
    rr.log("world/world_axes", rr.LineStrips3D(**kwargs))
    rr.log(
        "world/world_axes/labels",
        rr.Points3D(
            positions=np.stack(
                [origin, x_end, y_end, z_end]
            ).astype(np.float32),
            colors=np.asarray(
                [[20, 20, 20], [255, 0, 0], [0, 220, 0], [40, 80, 255]],
                dtype=np.uint8,
            ),
            labels=["world axes", "+X", "+Y", "+Z"],
        ),
    )


# ---------------------------------------------------------------------------
# Log input images
# ---------------------------------------------------------------------------
def log_input_images(
    rgbs: Sequence[np.ndarray], stems: Sequence[str]
) -> None:
    for i, (rgb, stem) in enumerate(zip(rgbs, stems)):
        rr.log(
            f"inputs/view_{i:03d}_{sanitize_name(stem)}/rgb",
            rr.Image(rgb),
        )


# ---------------------------------------------------------------------------
# JSON-safe conversion
# ---------------------------------------------------------------------------
def json_safe(obj):
    if torch.is_tensor(obj):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Dedupe cameras
# ---------------------------------------------------------------------------
def dedupe_cameras_by_stem(
    cams: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen: set = set()
    for cam in cams:
        stem = str(cam.get("stem", ""))
        key = stem or f"pred_index_{cam.get('pred_index', len(out))}"
        if key in seen:
            continue
        seen.add(key)
        out.append(cam)
    return out


# ---------------------------------------------------------------------------
# Eval outputs
# ---------------------------------------------------------------------------
def save_point_cloud_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if colors.shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]

    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(
                f"{float(p[0]):.8g} {float(p[1]):.8g} {float(p[2]):.8g} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def load_point_cloud_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as f:
        line = f.readline().decode("ascii", errors="strict").strip()
        if line != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        vertex_count = None
        fmt = "ascii"
        properties: List[Tuple[str, str]] = []
        in_vertex = False
        while True:
            line = f.readline()
            if line == b"":
                raise ValueError(f"Unexpected EOF in PLY header: {path}")
            line = line.decode("ascii", errors="strict").strip()
            if line == "end_header":
                break
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "format":
                fmt = parts[1]
                continue
            if len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                continue
            if in_vertex and len(parts) >= 3 and parts[0] == "property":
                if parts[1] == "list":
                    raise ValueError(f"Unsupported list property in vertex PLY: {path}")
                properties.append((parts[1], parts[2]))

        if vertex_count is None:
            raise ValueError(f"PLY missing vertex element: {path}")

        names = [name for _typ, name in properties]
        name_to_idx = {name: i for i, name in enumerate(names)}
        required = {"x", "y", "z"}
        if not required.issubset(name_to_idx):
            raise ValueError(f"PLY missing x/y/z properties: {path}")

        if fmt == "ascii":
            rows = []
            for _ in range(int(vertex_count)):
                line = f.readline()
                if line == b"":
                    break
                rows.append(line.decode("ascii", errors="ignore").split())

            points = np.empty((len(rows), 3), dtype=np.float32)
            colors = np.full((len(rows), 3), 220, dtype=np.uint8)
            for i, row in enumerate(rows):
                points[i, 0] = float(row[name_to_idx["x"]])
                points[i, 1] = float(row[name_to_idx["y"]])
                points[i, 2] = float(row[name_to_idx["z"]])
                if {"red", "green", "blue"}.issubset(name_to_idx):
                    colors[i, 0] = int(float(row[name_to_idx["red"]]))
                    colors[i, 1] = int(float(row[name_to_idx["green"]]))
                    colors[i, 2] = int(float(row[name_to_idx["blue"]]))
        elif fmt in {"binary_little_endian", "binary_big_endian"}:
            endian = "<" if fmt == "binary_little_endian" else ">"
            dtype_map = {
                "char": "i1",
                "int8": "i1",
                "uchar": "u1",
                "uint8": "u1",
                "short": endian + "i2",
                "int16": endian + "i2",
                "ushort": endian + "u2",
                "uint16": endian + "u2",
                "int": endian + "i4",
                "int32": endian + "i4",
                "uint": endian + "u4",
                "uint32": endian + "u4",
                "float": endian + "f4",
                "float32": endian + "f4",
                "double": endian + "f8",
                "float64": endian + "f8",
            }
            dtype_fields = []
            for typ, name in properties:
                if typ not in dtype_map:
                    raise ValueError(f"Unsupported PLY property type {typ!r}: {path}")
                dtype_fields.append((name, dtype_map[typ]))
            vertex_dtype = np.dtype(dtype_fields)
            data = np.fromfile(f, dtype=vertex_dtype, count=int(vertex_count))
            points = np.stack(
                [
                    np.asarray(data["x"], dtype=np.float32),
                    np.asarray(data["y"], dtype=np.float32),
                    np.asarray(data["z"], dtype=np.float32),
                ],
                axis=1,
            )
            colors = np.full((data.shape[0], 3), 220, dtype=np.uint8)
            if {"red", "green", "blue"}.issubset(name_to_idx):
                colors[:, 0] = np.asarray(data["red"], dtype=np.uint8)
                colors[:, 1] = np.asarray(data["green"], dtype=np.uint8)
                colors[:, 2] = np.asarray(data["blue"], dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported PLY format {fmt!r}: {path}")

    required = {"x", "y", "z"}
    finite = np.isfinite(points).all(axis=1)
    return points[finite].astype(np.float32), colors[finite].astype(np.uint8)


def save_final_eval_outputs(
    eval_dir: Path,
    pred_cams: Sequence[Dict[str, object]],
    gt_cams: Sequence[Dict[str, object]],
    pred_points: np.ndarray,
    pred_colors: np.ndarray,
    gt_points: np.ndarray,
    gt_colors: np.ndarray,
    meta: Dict[str, object],
    compress: bool = False,
) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)

    savez = np.savez_compressed if compress else np.savez

    stems: List[str] = []
    Ts: List[np.ndarray] = []
    valid_flags: List[bool] = []

    for cam in pred_cams:
        stem = str(cam.get("stem", ""))
        T = np.asarray(cam.get("T_c2w"), dtype=np.float32)
        ok = T.shape == (4, 4) and np.isfinite(T).all()

        stems.append(stem)
        Ts.append(
            T if ok else np.full((4, 4), np.nan, dtype=np.float32)
        )
        valid_flags.append(bool(ok))

    savez(
        eval_dir / "pred_cameras.npz",
        stems=np.asarray(stems, dtype=str),
        T_c2w=np.stack(Ts, axis=0).astype(np.float32)
        if Ts
        else np.empty((0, 4, 4), dtype=np.float32),
        valid=np.asarray(valid_flags, dtype=bool),
    )

    gt_stems: List[str] = []
    gt_Ts: List[np.ndarray] = []
    gt_valid_flags: List[bool] = []
    for cam in gt_cams:
        stem = str(cam.get("stem", ""))
        T = np.asarray(cam.get("T_c2w"), dtype=np.float32)
        ok = T.shape == (4, 4) and np.isfinite(T).all()

        gt_stems.append(stem)
        gt_Ts.append(
            T if ok else np.full((4, 4), np.nan, dtype=np.float32)
        )
        gt_valid_flags.append(bool(ok))

    savez(
        eval_dir / "gt_cameras.npz",
        stems=np.asarray(gt_stems, dtype=str),
        T_c2w=np.stack(gt_Ts, axis=0).astype(np.float32)
        if gt_Ts
        else np.empty((0, 4, 4), dtype=np.float32),
        valid=np.asarray(gt_valid_flags, dtype=bool),
    )

    points = np.asarray(pred_points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(pred_colors, dtype=np.uint8).reshape(-1, 3)
    if colors.shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]

    save_point_cloud_ply(
        eval_dir / "pred_points.ply",
        points=points,
        colors=colors,
    )
    save_point_cloud_ply(
        eval_dir / "gt_points.ply",
        points=gt_points,
        colors=gt_colors,
    )

    meta = dict(meta)
    meta["num_cameras"] = int(len(stems))
    meta["num_valid_cameras"] = int(
        np.asarray(valid_flags, dtype=bool).sum()
    )
    meta["num_gt_cameras"] = int(len(gt_stems))
    meta["num_valid_gt_cameras"] = int(
        np.asarray(gt_valid_flags, dtype=bool).sum()
    )
    meta["num_points"] = int(points.shape[0])
    meta["num_gt_points"] = int(
        np.asarray(gt_points, dtype=np.float32).reshape(-1, 3).shape[0]
    )
    meta["pred_points_path"] = "pred_points.ply"
    meta["gt_points_path"] = "gt_points.ply"
    meta["pred_cameras_path"] = "pred_cameras.npz"
    meta["gt_cameras_path"] = "gt_cameras.npz"

    (eval_dir / "meta.json").write_text(
        json.dumps(json_safe(meta), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved final eval outputs: {eval_dir}")


# ---------------------------------------------------------------------------
# GT cameras helper
# ---------------------------------------------------------------------------
def gt_cameras_for_stems(
    meta: Dict[str, object], stems: Sequence[str]
) -> List[Dict[str, object]]:
    cams = meta.get("gt_cams", meta.get("cams", {}))
    out = []
    for stem in stems:
        if stem not in cams:
            continue
        out.append(
            {
                "stem": stem,
                "T_c2w": np.asarray(
                    cams[stem]["T_c2w"], dtype=np.float32
                ),
            }
        )
    return out


def input_pose_centers_by_stem(
    meta: Dict[str, object],
) -> Dict[str, np.ndarray]:
    from geoff3d.spatial_rrd.geometry_align import camera_centers_by_stem

    cams = meta.get("cams", {})
    out = []
    if isinstance(cams, dict):
        for stem in meta.get("stems", []):
            stem = str(stem)
            cam = cams.get(stem)
            if cam is None:
                continue
            out.append(
                {
                    "stem": stem,
                    "T_c2w": np.asarray(cam["T_c2w"], dtype=np.float32),
                }
            )
    return camera_centers_by_stem(out)


# ---------------------------------------------------------------------------
# XY fill helpers
# ---------------------------------------------------------------------------
def points_from_cached_point_maps(
    point_maps: Sequence[np.ndarray],
    valid_masks: Sequence[np.ndarray],
    rgbs: Sequence[np.ndarray],
    local_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    points_all: List[np.ndarray] = []
    colors_all: List[np.ndarray] = []

    for local_i in local_indices:
        local_i = int(local_i)
        if local_i < 0 or local_i >= len(point_maps) or local_i >= len(valid_masks):
            continue

        point_map = np.asarray(point_maps[local_i], dtype=np.float32)
        if point_map.ndim != 3 or point_map.shape[-1] != 3:
            continue

        mask = np.asarray(valid_masks[local_i], dtype=bool)
        if mask.shape != point_map.shape[:2]:
            continue

        mask = mask & np.isfinite(point_map).all(axis=-1)
        if not mask.any():
            continue

        if local_i >= len(rgbs):
            colors = np.full((int(mask.sum()), 3), 220, dtype=np.uint8)
        else:
            rgb = np.asarray(rgbs[local_i], dtype=np.uint8)
            if rgb.shape[:2] != point_map.shape[:2]:
                rgb = cv2.resize(
                    rgb,
                    (point_map.shape[1], point_map.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            colors = rgb[mask].reshape(-1, 3).astype(np.uint8)

        points_all.append(point_map[mask].reshape(-1, 3).astype(np.float32))
        colors_all.append(colors)

    if not points_all:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)

    return np.concatenate(points_all, axis=0), np.concatenate(colors_all, axis=0)


def xy_occupied_keys(points: np.ndarray, grid_size: float) -> set:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return set()
    keys = np.floor(points[finite, :2].astype(np.float64) / float(grid_size)).astype(np.int64)
    return {tuple(int(v) for v in key) for key in keys.tolist()}


def filter_points_in_uncovered_xy_cells(
    base_points: np.ndarray,
    candidate_points: np.ndarray,
    candidate_colors: np.ndarray,
    grid_size: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    base_points = np.asarray(base_points, dtype=np.float32).reshape(-1, 3)
    candidate_points = np.asarray(candidate_points, dtype=np.float32).reshape(-1, 3)
    candidate_colors = np.asarray(candidate_colors, dtype=np.uint8).reshape(-1, 3)

    if candidate_points.shape[0] == 0:
        return candidate_points, candidate_colors, {
            "grid_size": float(grid_size),
            "num_base_points": int(base_points.shape[0]),
            "num_candidates": 0,
            "num_added": 0,
            "num_base_xy_cells": 0,
            "num_added_xy_cells": 0,
        }

    if candidate_colors.shape[0] != candidate_points.shape[0]:
        candidate_colors = np.full((candidate_points.shape[0], 3), 220, dtype=np.uint8)

    grid_size = float(grid_size)
    if not np.isfinite(grid_size) or grid_size <= 0:
        raise ValueError(f"xy fill grid_size must be positive, got {grid_size}")

    occupied = xy_occupied_keys(base_points, grid_size)
    candidate_finite = np.isfinite(candidate_points).all(axis=1)
    candidate_keys = np.floor(candidate_points[:, :2].astype(np.float64) / grid_size).astype(np.int64)

    keep = np.zeros((candidate_points.shape[0],), dtype=bool)
    added_cells = set()

    for i, key_arr in enumerate(candidate_keys):
        if not candidate_finite[i]:
            continue
        key = tuple(int(v) for v in key_arr.tolist())
        if key in occupied:
            continue
        keep[i] = True
        added_cells.add(key)

    out_points = candidate_points[keep]
    out_colors = candidate_colors[keep]

    return out_points, out_colors, {
        "grid_size": float(grid_size),
        "num_base_points": int(base_points.shape[0]),
        "num_candidates": int(candidate_points.shape[0]),
        "num_added": int(out_points.shape[0]),
        "num_base_xy_cells": int(len(occupied)),
        "num_added_xy_cells": int(len(added_cells)),
        "mode": "xy_uncovered_only",
    }


def aggregate_unmasked_xy_fill_points_streaming(
    chunk_records: Sequence[Dict[str, object]],
    base_points: np.ndarray,
    grid_size: float,
    max_points_per_chunk: int,
    voxel_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    fill_points_all: List[np.ndarray] = []
    fill_colors_all: List[np.ndarray] = []

    total_candidates = 0
    total_added_before_concat = 0
    total_chunks_used = 0

    for record in chunk_records:
        chunk_id = int(record["chunk_id"])

        point_maps = get_transformed_cached_point_maps(record, "_pred_maps")
        valid_masks = get_cached_sequence(record, "_pred_valid_masks_unmasked")
        rgbs = get_cached_sequence(record, "rgbs")
        core_local_indices = get_cached_sequence(record, "_core_local_indices")

        if not point_maps or not valid_masks or not core_local_indices:
            continue

        cand_points, cand_colors = points_from_cached_point_maps(
            point_maps=point_maps,
            valid_masks=valid_masks,
            rgbs=rgbs,
            local_indices=[int(v) for v in core_local_indices],
        )

        cand_points, cand_colors = sample_points_and_colors(
            cand_points,
            cand_colors,
            max_points=int(max_points_per_chunk),
            seed=int(seed) + 10007 * (chunk_id + 1),
        )

        if float(voxel_size) > 0:
            cand_points, cand_colors = voxel_downsample(cand_points, cand_colors, float(voxel_size))

        selected_points, selected_colors, detail = filter_points_in_uncovered_xy_cells(
            base_points=base_points,
            candidate_points=cand_points,
            candidate_colors=cand_colors,
            grid_size=float(grid_size),
        )

        total_candidates += int(detail.get("num_candidates", 0))
        total_added_before_concat += int(selected_points.shape[0])

        if selected_points.shape[0] > 0:
            fill_points_all.append(selected_points)
            fill_colors_all.append(selected_colors)
            total_chunks_used += 1

            # 更新 base_points，避免后续 chunk 在同一 XY 空洞重复补点
            base_points = np.concatenate([base_points, selected_points], axis=0)

    if not fill_points_all:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8), {
            "enabled": True,
            "grid_size": float(grid_size),
            "num_candidates": int(total_candidates),
            "num_added": 0,
            "num_chunks_used": int(total_chunks_used),
        }

    fill_points = np.concatenate(fill_points_all, axis=0)
    fill_colors = np.concatenate(fill_colors_all, axis=0)

    return fill_points, fill_colors, {
        "enabled": True,
        "grid_size": float(grid_size),
        "num_candidates": int(total_candidates),
        "num_added": int(fill_points.shape[0]),
        "num_chunks_used": int(total_chunks_used),
        "source": "_pred_maps + _pred_valid_masks_unmasked",
        "mode": "xy_uncovered_only",
    }


# ---------------------------------------------------------------------------
# Main spatial RRD save
# ---------------------------------------------------------------------------
def aggregate_core_points_streaming(
    chunk_records: Sequence[Dict[str, object]],
    max_points_per_chunk: int,
    voxel_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pts_list: List[np.ndarray] = []
    col_list: List[np.ndarray] = []

    for record in chunk_records:
        chunk_id = int(record["chunk_id"])
        pts = get_transformed_cached_points(record, "core_pred_points")
        cols = get_cached_colors(record, "core_pred_colors")

        pts, cols = sample_points_and_colors(
            pts,
            cols,
            max_points=max_points_per_chunk,
            seed=int(seed) + 7919 * (chunk_id + 1),
        )

        if voxel_size > 0:
            pts, cols = voxel_downsample(pts, cols, voxel_size)

        if pts.shape[0] > 0:
            pts_list.append(pts)
            col_list.append(cols)

    if not pts_list:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)

    return np.concatenate(pts_list, axis=0), np.concatenate(col_list, axis=0)


def save_chunk_point_cloud_artifacts(
    chunk_records: Sequence[Dict[str, object]],
    output_dir: Path,
    chunk_rgb_by_id: Dict[int, Tuple[int, int, int]],
    max_points: int,
    voxel_size: float,
    seed: int,
) -> List[Dict[str, object]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: List[Dict[str, object]] = []

    for record in chunk_records:
        chunk_id = int(record["chunk_id"])
        points = get_transformed_cached_points(record, "chunk_pred_points")
        rgb = np.asarray(
            chunk_rgb_by_id.get(chunk_id, (220, 220, 220)),
            dtype=np.uint8,
        ).reshape(1, 3)
        colors = np.repeat(rgb, int(points.shape[0]), axis=0)

        points, colors = sample_points_and_colors(
            points,
            colors,
            max_points=int(max_points),
            seed=int(seed) + 1009 * (chunk_id + 1),
        )
        if float(voxel_size) > 0:
            points, colors = voxel_downsample(points, colors, float(voxel_size))

        ply_path = output_dir / f"chunk_{chunk_id:03d}_points.ply"
        save_point_cloud_ply(ply_path, points=points, colors=colors)
        artifacts.append(
            {
                "chunk_id": int(chunk_id),
                "ply_path": str(ply_path),
                "num_points": int(points.shape[0]),
                "color_rgb": [int(v) for v in rgb.reshape(3).tolist()],
            }
        )
        record["num_chunk_pred_points_logged"] = int(points.shape[0])
        record["chunk_points_ply"] = str(ply_path)
        record["chunk_artifact_color_rgb"] = [
            int(v) for v in rgb.reshape(3).tolist()
        ]

    return artifacts


def save_spatial_rrd(
    output_rrd: Path,
    scene_dir: str,
    model_name: str,
    checkpoint: Optional[str],
    meta: Dict[str, object],
    grid_meta: Dict[str, object],
    chunk_records: Sequence[Dict[str, object]],
    align: str,
    view_coordinates: str = "RDF",
    background: Sequence[int] = (255, 255, 255),
    hide_grid: bool = False,
    point_radius: float = 0.0,
    camera_axis_size: float = 0.0,
    camera_axis_radius: float = 0.0,
    max_points_per_view: int = 250000,
    voxel_size: float = 0.5,
    point_downsample: bool = True,
    seed: int = 0,
    log_images: bool = False,
    show_world_axes: bool = True,
    world_axes_origin: str = "scene_center",
    world_axis_size: float = 0.0,
    world_axis_size_ratio: float = 0.12,
    world_axis_min_size: float = 0.1,
    world_up_axis: str = "z",
    world_axis_up_offset_ratio: float = 1.2,
    world_axis_radius: float = 0.0,
    prior_policy: Optional[Dict[str, bool]] = None,
    recenter_anchor: Optional[np.ndarray] = None,
    log_chunks_rrd: bool = True,
    compress_eval: bool = False,
    processing_time: Optional[Dict[str, object]] = None,
    gt_io_workers: int = 0,
    xy_fill_unmasked: bool = False,
    xy_fill_grid_size: float = 0.0,
    xy_fill_max_points_per_chunk: int = 50000,
) -> None:
    output_rrd = Path(output_rrd).expanduser().resolve()
    output_rrd.parent.mkdir(parents=True, exist_ok=True)

    scene_name = sanitize_name(Path(scene_dir).resolve().name)
    effective_max_points = int(max_points_per_view) if bool(point_downsample) else 0
    gt_points, gt_colors = load_gt_points_from_meta(
        meta,
        effective_max_points,
        seed,
        num_workers=gt_io_workers,
    )
    if bool(point_downsample):
        gt_points, gt_colors = voxel_downsample(gt_points, gt_colors, voxel_size)
    all_gt_cams = gt_cameras_for_stems(meta, meta["stems"])
    effective_voxel_size = float(voxel_size) if bool(point_downsample) else 0.0

    if bool(point_downsample):
        per_chunk_cap = max(1, int(max_points_per_view // max(1, len(chunk_records))))
        per_chunk_cap = max(per_chunk_cap, 50000)
    else:
        per_chunk_cap = 0

    aggregate_points, aggregate_colors = aggregate_core_points_streaming(
        chunk_records=chunk_records,
        max_points_per_chunk=per_chunk_cap,
        voxel_size=effective_voxel_size,
        seed=seed + 17,
    )

    xy_fill_meta: Dict[str, object] = {
        "enabled": False,
        "reason": "xy_fill_unmasked is disabled",
    }

    if bool(xy_fill_unmasked):
        fill_grid_size = float(xy_fill_grid_size)
        if fill_grid_size <= 0:
            if effective_voxel_size > 0:
                fill_grid_size = max(float(effective_voxel_size) * 5.0, 0.05)
            else:
                fill_grid_size = 0.05

        fill_points, fill_colors, xy_fill_meta = aggregate_unmasked_xy_fill_points_streaming(
            chunk_records=chunk_records,
            base_points=aggregate_points,
            grid_size=fill_grid_size,
            max_points_per_chunk=int(xy_fill_max_points_per_chunk),
            voxel_size=effective_voxel_size,
            seed=seed + 2027,
        )

        if fill_points.shape[0] > 0:
            aggregate_points = np.concatenate([aggregate_points, fill_points], axis=0)
            aggregate_colors = np.concatenate([aggregate_colors, fill_colors], axis=0)

        print(
            "[INFO] XY unmasked fill: "
            f"enabled={bool(xy_fill_unmasked)}, "
            f"grid={fill_grid_size:.6g}, "
            f"candidates={int(xy_fill_meta.get('num_candidates', 0))}, "
            f"added={int(xy_fill_meta.get('num_added', 0))}"
        )

    aggregate_points, aggregate_colors = sample_points_and_colors(
        aggregate_points,
        aggregate_colors,
        effective_max_points,
        seed + 17,
    )

    if bool(point_downsample):
        # 再做一次全局 voxel，去掉不同 chunk 边界重复点。
        aggregate_points, aggregate_colors = voxel_downsample(
            aggregate_points, aggregate_colors, voxel_size,
        )

    pred_root = (
        "pred_spatial_aligned" if align != "none" else "pred_spatial"
    )

    axis_size = estimate_axis_size(
        [aggregate_points, gt_points], camera_axis_size
    )
    gt_axis_colors = ((255, 0, 0), (0, 220, 0), (40, 80, 255))
    pred_axis_colors = ((255, 0, 255), (255, 180, 0), (0, 220, 255))
    all_pred_cams = dedupe_cameras_by_stem(
        [cam for record in chunk_records for cam in get_transformed_cameras(record)]
    )
    post_align_enabled = any(
        bool(record.get("post_chunk_align_meta", {}).get("enabled", False))
        for record in chunk_records
    )

    save_final_eval_outputs(
        eval_dir=output_rrd.with_suffix("") / "eval",
        pred_cams=all_pred_cams,
        gt_cams=all_gt_cams,
        pred_points=aggregate_points,
        pred_colors=aggregate_colors,
        gt_points=gt_points,
        gt_colors=gt_colors,
        meta={
            "schema": "final_eval_v1",
            "script": "scripts/predict_scene_to_rrd_spatial.py",
            "scene_dir": str(Path(scene_dir).expanduser().resolve()),
            "method": model_name,
            "checkpoint": checkpoint,
            "processing_time": processing_time or {},
            "pose_convention": "T_c2w",
            "points_coordinate": "same_as_pred_cameras",
            "post_align": {
                "enabled": bool(post_align_enabled),
                "type": "deferred_chunk_transform" if post_align_enabled else "none",
                "target": "chunk_records" if post_align_enabled else "none",
                "valid": True,
                "note": (
                    "Per-chunk post-alignment transforms are applied lazily "
                    "when exporting final cameras and points."
                    if post_align_enabled
                    else "Our method output is saved directly without extra final Sim3 alignment."
                ),
            },
            "aggregation": {
                "points": "core_pred_points",
                "xy_fill": xy_fill_meta,
                "cameras": "dedupe_by_stem",
                "num_chunks": int(len(chunk_records)),
                "max_points_per_view": int(max_points_per_view),
                "effective_max_points_per_view": int(effective_max_points),
                "voxel_downsample": float(voxel_size),
                "effective_voxel_downsample": float(effective_voxel_size),
                "point_downsample": bool(point_downsample),
            },
        },
        compress=compress_eval,
    )

    overall_recording_id = (
        f"spatial_overall_{scene_name}_"
        f"{sanitize_name(model_name)}_{sanitize_name(align)}"
    )
    rr_init_save_compat(
        "predict_scene_to_rrd_spatial_overall",
        overall_recording_id,
        output_rrd,
    )
    rr_set_time_compat("frame", 0)
    log_view_coordinates(view_coordinates)
    send_blueprint(
        background=tuple(background), hide_grid=hide_grid
    )

    log_points(
        "world/gt/points", gt_points, gt_colors, point_radius
    )
    log_points(
        f"world/{pred_root}/points",
        aggregate_points,
        aggregate_colors,
        point_radius,
    )
    log_camera_axes(
        "world/cameras/gt/axes",
        all_gt_cams,
        axis_size,
        camera_axis_radius,
        gt_axis_colors,
    )
    log_camera_axes(
        f"world/cameras/{pred_root}/axes",
        all_pred_cams,
        axis_size,
        camera_axis_radius,
        pred_axis_colors,
    )

    if show_world_axes:
        bbox_points = (
            gt_points if gt_points.shape[0] > 0 else aggregate_points
        )
        log_world_axes_marker(
            bbox_points,
            origin_mode=world_axes_origin,
            axis_size=world_axis_size,
            axis_size_ratio=world_axis_size_ratio,
            min_axis_size=world_axis_min_size,
            up_axis=world_up_axis,
            up_offset_ratio=world_axis_up_offset_ratio,
            radius=world_axis_radius,
        )

    rr_disconnect_compat()
    print(f"Saved overall Rerun recording: {output_rrd}")

    chunk_artifacts_dir: Optional[Path] = None
    chunk_point_artifacts: List[Dict[str, object]] = []
    chunk_footprint_path: Optional[Path] = None
    if log_chunks_rrd:
        chunk_artifacts_dir = output_rrd.with_suffix("") / "chunk_outputs"
        rgba_by_chunk_id, rgb_by_chunk_id = make_chunk_color_lookup(chunk_records)
        chunk_footprint_path = save_chunk_footprint_xy_visualization(
            meta=meta,
            grid_meta=grid_meta,
            chunks=chunk_records,
            output_dir=chunk_artifacts_dir,
            file_stem="chunk_footprint_xy",
            rgba_by_chunk_id=rgba_by_chunk_id,
        )
        chunk_point_artifacts = save_chunk_point_cloud_artifacts(
            chunk_records=chunk_records,
            output_dir=chunk_artifacts_dir / "ply",
            chunk_rgb_by_id=rgb_by_chunk_id,
            max_points=effective_max_points,
            voxel_size=effective_voxel_size,
            seed=int(seed),
        )
        print(f"[INFO] Saved chunk artifacts: {chunk_artifacts_dir}")
    else:
        output_chunks_path = None
        for record in chunk_records:
            record["num_chunk_pred_points_logged"] = 0
        print("[INFO] Skip chunk artifacts because log_chunks_rrd=False.")

    sidecar = output_rrd.with_suffix(".json")
    payload = {
        "scene_dir": str(Path(scene_dir).resolve()),
        "model": model_name,
        "checkpoint": checkpoint,
        "output_rrd": str(output_rrd),
        "chunk_artifacts_dir": str(chunk_artifacts_dir) if chunk_artifacts_dir is not None else None,
        "chunk_footprint_path": str(chunk_footprint_path) if chunk_footprint_path is not None else None,
        "chunk_point_artifacts": chunk_point_artifacts,
        "chunk_cache_dir": str(output_rrd.with_suffix("") / "chunk_cache"),
        "stems": list(meta["stems"]),
        "target_size": {
            "height": int(meta["target_h"]),
            "width": int(meta["target_w"]),
        },
        "grid": grid_meta,
        "chunking": {
            "max_chunk_size": int(grid_meta.get("max_chunk_size", 0)),
            "min_chunk_size": int(grid_meta.get("min_chunk_size", 0)),
            "auto_core_target_size": int(grid_meta.get("auto_core_target_size", 0)),
            "num_chunks": int(len(chunk_records)),
            "max_points_per_view": int(max_points_per_view),
            "effective_max_points_per_view": int(effective_max_points),
            "voxel_downsample": float(voxel_size),
            "effective_voxel_downsample": float(effective_voxel_size),
            "point_downsample": bool(point_downsample),
            "note": (
                "Seam overlaps are selected automatically from connected "
                "neighboring cells and capped by max_chunk_size."
            ),
        },
        "alignment": align,
        "prior_policy": prior_policy,
        "processing_time": processing_time or {},
        "recenter": (
            recenter_anchor.tolist()
            if recenter_anchor is not None
            else None
        ),
        "num_gt_points_logged": int(gt_points.shape[0]),
        "num_pred_core_points_logged": int(aggregate_points.shape[0]),
        "num_gt_cameras": int(len(all_gt_cams)),
        "num_pred_cameras": int(len(all_pred_cams)),
        "chunks": [
            {
                "chunk_id": int(record["chunk_id"]),
                "chunk_cache_path": record.get("chunk_cache_path", None),
                "chunk_cache_keys": list(record.get("chunk_cache_keys", [])),
                "cell_key": list(record["cell_key"]),
                "indices": [int(i) for i in record["indices"]],
                "core_indices": [
                    int(i) for i in record["core_indices"]
                ],
                "overlap_indices": [
                    int(i) for i in record["overlap_indices"]
                ],
                "stems": list(record["stems"]),
                "core_stems": list(record["core_stems"]),
                "overlap_stems": list(record["overlap_stems"]),
                "num_seam_candidates": int(
                    record["num_seam_candidates"]
                ),
                "num_dropped_seam_images": int(
                    record["num_dropped_seam_images"]
                ),
                "num_depth_priors_used": int(
                    record["num_depth_priors_used"]
                ),
                "num_chunk_pred_points_raw": int(
                    record["num_chunk_pred_points_raw"]
                ),
                "num_core_pred_points": int(
                    record.get("num_core_pred_points", 0)
                ),
                "num_chunk_pred_points_logged": int(
                    record.get("num_chunk_pred_points_logged", 0)
                ),
                "chunk_points_ply": record.get("chunk_points_ply", None),
                "chunk_artifact_color_rgb": record.get(
                    "chunk_artifact_color_rgb",
                    None,
                ),
                "num_pred_cameras": int(
                    len(record["pred_cams"])
                ),
                "alignment": record["align_meta"],
                "depth_conf_filter": record.get(
                    "depth_conf_filter_meta",
                    None,
                ),
                "post_chunk_align_transform": record.get(
                    "post_chunk_align_transform",
                    None,
                ),
                "post_chunk_align_meta": record.get(
                    "post_chunk_align_meta",
                    None,
                ),
            }
            for record in chunk_records
        ],
    }
    sidecar.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved sidecar metadata: {sidecar}")
