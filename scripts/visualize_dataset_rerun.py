#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless Rerun exporter for an A3D/WAI-style dataset.

Dataset layout:
    dataset_root/
      scene_000/
        images/  *.jpg | *.jpeg | *.png | *.bmp | *.tif | *.tiff
        cams/    *.txt
        depth/   *.exr | *.npy | *.png | *.tif | *.tiff  (optional)
      scene_001/
        images/
        cams/
        depth/
      ...

For each valid scene folder, this script saves:
    output_dir/<scene_name>.rrd

What each .rrd logs:
  - world/points: colored point cloud reconstructed from uniformly sampled RGB-D frames, when depth is available
  - world/cameras/axes: three-axis camera markers for all cameras
      x = red, y = green, z = blue
  - world/cameras/centers: camera centers
  - world/cameras/trajectory: optional trajectory line
  - world/cameras/frustums/view_xxxx: optional pinhole frustums
  - world/world_axes: optional large world-axis marker placed above the point-cloud center
      +X = red, +Y = green, +Z = blue
    When depth is unavailable, the camera-center bounding box is used as the reference
    for auto axis sizing and world-axis placement.

Remote usage:
    python visualize_a3d_dataset_rerun.py \
      --dataset_root /path/to/A3D-Syn-S \
      --output_dir experiments/dataset_viz/A3D-Syn-S \
      --max_point_views 80 \
      --max_side 960 \
      --target_points 800000 \
      --show_traj \
      --show_world_axes

Open one result locally:
    rerun experiments/dataset_viz/A3D-Syn-S/<scene_name>.rrd
"""

from __future__ import annotations

import argparse
import fnmatch
import gc
import math
import os
import re
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Must be set before importing cv2 when OpenCV EXR support is available.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

try:
    import cv2
except Exception:
    cv2 = None

try:
    import open3d as o3d
except Exception:
    o3d = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEPTH_EXTS = {".exr", ".npy", ".png", ".tif", ".tiff"}
CAM_EXTS = {".txt"}


# -----------------------------------------------------------------------------
# Dataset / file collection
# -----------------------------------------------------------------------------
def collect_stem_to_path(folder: Path, exts: Iterable[str]) -> Dict[str, Path]:
    exts = {e.lower() for e in exts}
    if not folder.exists():
        return {}
    out: Dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out[p.stem] = p
    return out


def is_scene_dir(scene_dir: Path, images_dir: str, cams_dir: str, depth_dir: str, require_depth: bool) -> bool:
    return (
        scene_dir.is_dir()
        and (scene_dir / images_dir).is_dir()
        and (scene_dir / cams_dir).is_dir()
        and (not require_depth or (scene_dir / depth_dir).is_dir())
    )


def find_scene_dirs(
    dataset_root: Path,
    images_dir: str,
    cams_dir: str,
    depth_dir: str,
    scene_glob: str,
    recursive: bool,
    require_depth: bool,
) -> List[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    if recursive:
        candidates = [p for p in dataset_root.rglob("*") if p.is_dir()]
    else:
        candidates = [p for p in dataset_root.iterdir() if p.is_dir()]

    scenes: List[Path] = []
    for p in sorted(candidates):
        if not is_scene_dir(p, images_dir, cams_dir, depth_dir, require_depth=require_depth):
            continue
        rel = str(p.relative_to(dataset_root))
        name_match = fnmatch.fnmatch(p.name, scene_glob) or fnmatch.fnmatch(rel, scene_glob)
        if scene_glob and scene_glob != "*" and not name_match:
            continue
        scenes.append(p)
    return scenes


def sanitize_filename(name: str) -> str:
    """Keep scene name readable while avoiding problematic filename characters."""
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|\s]+", "_", name)
    return name or "scene"


def evenly_select(items: Sequence[str], max_count: int) -> List[str]:
    """Uniformly select at most max_count items, preserving full scene coverage."""
    items = list(items)
    n = len(items)
    if max_count <= 0 or n <= max_count:
        return items
    idx = np.linspace(0, n - 1, num=max_count)
    idx = np.round(idx).astype(np.int64)
    idx = np.unique(idx)
    return [items[int(i)] for i in idx]


# -----------------------------------------------------------------------------
# Camera txt parser
# -----------------------------------------------------------------------------
def _float_tokens(line: str) -> Optional[List[float]]:
    try:
        vals = [float(x) for x in line.replace(",", " ").split()]
        return vals if vals else None
    except ValueError:
        return None


def _find_line(lines: Sequence[str], prefixes: Sequence[str]) -> int:
    prefixes = tuple(p.lower().rstrip(":") for p in prefixes)
    for i, line in enumerate(lines):
        l = line.strip().lower().rstrip(":")
        if any(l.startswith(p) for p in prefixes):
            return i
    return -1


def _read_numeric_rows(lines: Sequence[str], start: int, n_rows: int, n_cols: int, path: Path) -> np.ndarray:
    rows: List[List[float]] = []
    for j in range(start, len(lines)):
        vals = _float_tokens(lines[j])
        if vals is None or len(vals) < n_cols:
            continue
        rows.append(vals[:n_cols])
        if len(rows) == n_rows:
            break
    if len(rows) != n_rows:
        raise ValueError(f"Cannot read {n_rows}x{n_cols} numeric matrix from {path}")
    return np.asarray(rows, dtype=np.float64)


def parse_cam_txt(cam_path: Path) -> Dict[str, object]:
    with open(cam_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    idx_ext = _find_line(lines, ["extrinsic"])
    idx_int = _find_line(lines, ["intrinsic"])
    if idx_ext < 0 or idx_int < 0:
        raise ValueError(f"Invalid camera txt, missing extrinsic/intrinsic: {cam_path}")

    T_w2c = _read_numeric_rows(lines, idx_ext + 1, 4, 4, cam_path)
    K = _read_numeric_rows(lines, idx_int + 1, 3, 3, cam_path)

    idx_hwf = -1
    for i, ln in enumerate(lines):
        tokens = ln.lower().replace(":", " ").split()
        if "h" in tokens and "w" in tokens and ("fov" in tokens or "hfov" in tokens):
            idx_hwf = i
            break

    height: Optional[int] = None
    width: Optional[int] = None
    fov: Optional[float] = None
    if idx_hwf >= 0:
        vals = None
        for j in range(idx_hwf + 1, len(lines)):
            vals = _float_tokens(lines[j])
            if vals is not None and len(vals) >= 2:
                break
        if vals is not None and len(vals) >= 2:
            height = int(round(vals[0]))
            width = int(round(vals[1]))
            if len(vals) >= 3:
                fov = float(vals[2])

    return {
        "stem": cam_path.stem,
        "path": cam_path,
        "K": K,
        "T_w2c": T_w2c,
        "T_c2w": np.linalg.inv(T_w2c),
        "height": height,
        "width": width,
        "fov": fov,
    }


# -----------------------------------------------------------------------------
# RGB-D loading and point generation
# -----------------------------------------------------------------------------
def read_rgb(path: Path) -> np.ndarray:
    if cv2 is not None:
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def read_depth(path: Path, depth_scale: float = 1.0) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(str(path))
    else:
        if cv2 is None:
            raise ImportError("cv2 is required to read non-npy depth files.")
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(
                f"Cannot read depth: {path}. For EXR, check whether OpenCV was built with OpenEXR support."
            )
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if depth_scale != 1.0:
        depth = depth / float(depth_scale)
    return depth


def align_and_downsample_rgb_depth_K(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    cam_width: Optional[int],
    cam_height: Optional[int],
    max_side: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize RGB/depth and scale K consistently."""
    K = K.astype(np.float64).copy()

    depth_h, depth_w = depth.shape[:2]
    if cam_width is None or cam_height is None:
        cam_height, cam_width = rgb.shape[:2]

    # Scale K from camera-file resolution to depth resolution if needed.
    if int(cam_width) != int(depth_w) or int(cam_height) != int(depth_h):
        sx = float(depth_w) / float(cam_width)
        sy = float(depth_h) / float(cam_height)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    if rgb.shape[0] != depth_h or rgb.shape[1] != depth_w:
        if cv2 is None:
            raise ImportError("cv2 is required to resize RGB/depth arrays for point-cloud generation.")
        rgb = cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_AREA)

    if max_side > 0 and max(depth_h, depth_w) > max_side:
        if cv2 is None:
            raise ImportError("cv2 is required to resize RGB/depth arrays for point-cloud generation.")
        scale = float(max_side) / float(max(depth_h, depth_w))
        new_w = max(1, int(round(depth_w * scale)))
        new_h = max(1, int(round(depth_h * scale)))
        sx = float(new_w) / float(depth_w)
        sy = float(new_h) / float(depth_h)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    return rgb, depth, K


def depth_to_world_points(
    depth: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    T_c2w: np.ndarray,
    depth_min: float,
    depth_max: float,
    pixel_stride: int,
    max_points_per_view: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if pixel_stride < 1:
        pixel_stride = 1

    valid = np.isfinite(depth) & (depth > depth_min) & (depth < depth_max)
    if pixel_stride > 1:
        stride_mask = np.zeros_like(valid, dtype=bool)
        stride_mask[::pixel_stride, ::pixel_stride] = True
        valid &= stride_mask

    v, u = np.nonzero(valid)
    if v.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    if max_points_per_view > 0 and v.size > max_points_per_view:
        sel = rng.choice(v.size, size=max_points_per_view, replace=False)
        v = v[sel]
        u = u[sel]

    z = depth[v, u].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)

    R = T_c2w[:3, :3]
    t = T_c2w[:3, 3]
    pts_world = (R @ pts_cam.T).T + t[None, :]

    colors = rgb[v, u].astype(np.uint8)
    return pts_world.astype(np.float32), colors


def build_point_cloud_arrays(
    scene_root: Path,
    selected_stems: Sequence[str],
    images: Dict[str, Path],
    depths: Dict[str, Path],
    cams: Dict[str, Dict[str, object]],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    all_pts: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []

    if not selected_stems:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    for k, stem in enumerate(selected_stems):
        try:
            cam = cams[stem]
            rgb = read_rgb(images[stem])
            depth = read_depth(depths[stem], depth_scale=args.depth_scale)
            rgb, depth, K = align_and_downsample_rgb_depth_K(
                rgb=rgb,
                depth=depth,
                K=np.asarray(cam["K"], dtype=np.float64),
                cam_width=cam.get("width"),
                cam_height=cam.get("height"),
                max_side=args.max_side,
            )
            pts, cols = depth_to_world_points(
                depth=depth,
                rgb=rgb,
                K=K,
                T_c2w=np.asarray(cam["T_c2w"], dtype=np.float64),
                depth_min=args.depth_min,
                depth_max=args.depth_max,
                pixel_stride=args.pixel_stride,
                max_points_per_view=args.max_points_per_view,
                rng=rng,
            )
            if pts.shape[0] > 0:
                all_pts.append(pts)
                all_cols.append(cols)
            print(f"    [{k + 1:03d}/{len(selected_stems):03d}] {stem}: points={pts.shape[0]}")
        except Exception as e:
            print(f"    [WARN] skip frame {stem}: {e}")

    if len(all_pts) == 0:
        print(f"    [WARN] no valid depth points generated from scene: {scene_root}")
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    cols = np.concatenate(all_cols, axis=0).astype(np.uint8)
    return pts, cols


def voxel_downsample_arrays(
    pts: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    target_points: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if pts.shape[0] == 0:
        return pts, colors, 0.0

    if voxel_size <= 0 and (target_points <= 0 or pts.shape[0] <= target_points):
        return pts, colors, 0.0

    if o3d is None:
        raise ImportError("open3d is required for voxel downsampling. Please install open3d or disable downsampling.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    if voxel_size > 0:
        ds = pcd.voxel_down_sample(float(voxel_size))
        out_pts = np.asarray(ds.points, dtype=np.float32)
        out_cols = np.clip(np.asarray(ds.colors) * 255.0, 0, 255).astype(np.uint8)
        return out_pts, out_cols, float(voxel_size)

    if target_points <= 0 or pts.shape[0] <= target_points:
        return pts, colors, 0.0

    bbox = pcd.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(np.asarray(bbox.get_extent())))
    voxel = max(diag / 1000.0, 1e-8)
    best = pcd

    for _ in range(12):
        ds = pcd.voxel_down_sample(voxel)
        n = len(ds.points)
        best = ds
        if n <= target_points or n == 0:
            break
        ratio = max(float(n) / float(target_points), 1.01)
        voxel *= math.pow(ratio, 1.0 / 3.0) * 1.08

    out_pts = np.asarray(best.points, dtype=np.float32)
    out_cols = np.clip(np.asarray(best.colors) * 255.0, 0, 255).astype(np.uint8)
    return out_pts, out_cols, float(voxel)


# -----------------------------------------------------------------------------
# Rerun helpers
# -----------------------------------------------------------------------------
def rr_set_time_compat(name: str, sequence: int) -> None:
    try:
        rr.set_time(name, sequence=sequence)
    except AttributeError:
        rr.set_time_sequence(name, sequence)


def rr_disconnect_compat() -> None:
    try:
        rr.disconnect()
    except Exception:
        pass


def rr_init_save_compat(app_id: str, recording_id: str, save_rrd: Path) -> None:
    """Initialize a fresh recording and attach a file sink."""
    try:
        rr.init(app_id, recording_id=recording_id, spawn=False)
    except TypeError:
        rr.init(app_id, spawn=False)
    rr.save(str(save_rrd))


def log_point_cloud(entity_path: str, pts: np.ndarray, colors: np.ndarray, point_radius: Optional[float]) -> None:
    if pts.shape[0] == 0:
        return
    kwargs = dict(positions=pts, colors=colors)
    if point_radius is not None and point_radius > 0:
        kwargs["radii"] = float(point_radius)
    rr.log(entity_path, rr.Points3D(**kwargs))


def make_camera_axes_strips(
    cams_ordered: Sequence[Dict[str, object]],
    axis_size: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    strips: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    color_x = np.array([255, 0, 0], dtype=np.uint8)
    color_y = np.array([0, 220, 0], dtype=np.uint8)
    color_z = np.array([40, 80, 255], dtype=np.uint8)

    for cam in cams_ordered:
        T_c2w = np.asarray(cam["T_c2w"], dtype=np.float64)
        R = T_c2w[:3, :3]
        o = T_c2w[:3, 3]
        x_end = o + R[:, 0] * axis_size
        y_end = o + R[:, 1] * axis_size
        z_end = o + R[:, 2] * axis_size
        strips.extend([
            np.stack([o, x_end], axis=0).astype(np.float32),
            np.stack([o, y_end], axis=0).astype(np.float32),
            np.stack([o, z_end], axis=0).astype(np.float32),
        ])
        colors.extend([color_x, color_y, color_z])

    return strips, colors


def log_camera_axes(
    entity_path: str,
    cams_ordered: Sequence[Dict[str, object]],
    axis_size: float,
    radius: float,
) -> None:
    strips, colors = make_camera_axes_strips(cams_ordered, axis_size)
    if not strips:
        return
    kwargs = dict(strips=strips, colors=colors)
    if radius > 0:
        kwargs["radii"] = float(radius)
    rr.log(entity_path, rr.LineStrips3D(**kwargs))


def camera_centers_array(cams_ordered: Sequence[Dict[str, object]]) -> np.ndarray:
    if not cams_ordered:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray([np.asarray(c["T_c2w"], dtype=np.float64)[:3, 3] for c in cams_ordered], dtype=np.float32)


def reference_points_for_scene(pts: np.ndarray, cams_ordered: Sequence[Dict[str, object]]) -> np.ndarray:
    if pts.shape[0] > 0:
        return pts
    return camera_centers_array(cams_ordered)


def infer_scene_diag(reference_pts: np.ndarray) -> float:
    if reference_pts.shape[0] == 0:
        return 1.0
    extent = reference_pts.max(axis=0) - reference_pts.min(axis=0)
    diag = float(np.linalg.norm(extent))
    return diag if diag > 1e-8 else 1.0



def parse_signed_axis(axis: str) -> np.ndarray:
    """Parse axis string like 'z' or '-y' into a world-space unit vector."""
    axis = axis.strip().lower()
    sign = -1.0 if axis.startswith("-") else 1.0
    name = axis[1:] if axis.startswith("-") else axis

    if name == "x":
        return np.array([sign, 0.0, 0.0], dtype=np.float64)
    if name == "y":
        return np.array([0.0, sign, 0.0], dtype=np.float64)
    if name == "z":
        return np.array([0.0, 0.0, sign], dtype=np.float64)

    raise ValueError(f"Invalid axis '{axis}'. Expected one of: x, y, z, -x, -y, -z.")


def log_world_axes_marker(
    pts: np.ndarray,
    axis_size: float,
    axis_size_ratio: float,
    min_axis_size: float,
    up_axis: str,
    up_offset: float,
    up_offset_ratio: float,
    radius: float,
) -> None:
    """
    Draw a large world-axis marker above the center of a reference bounding box.

    Important:
      This marker is NOT the true world origin unless the reference center plus offset
      happens to be [0, 0, 0]. It is a reference frame whose directions are aligned with
      the world coordinate axes:
        +X = red, +Y = green, +Z = blue.
    """
    if pts.shape[0] == 0:
        return

    bbox_min = pts.min(axis=0).astype(np.float64)
    bbox_max = pts.max(axis=0).astype(np.float64)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))

    if axis_size > 0:
        size = float(axis_size)
    else:
        size = max(float(bbox_diag) * float(axis_size_ratio), float(min_axis_size), 1e-6)

    if up_offset > 0:
        offset = float(up_offset)
    else:
        offset = size * float(up_offset_ratio)

    up_dir = parse_signed_axis(up_axis)
    origin = bbox_center + up_dir * offset

    color_x = np.array([255, 0, 0], dtype=np.uint8)
    color_y = np.array([0, 220, 0], dtype=np.uint8)
    color_z = np.array([40, 80, 255], dtype=np.uint8)
    color_origin = np.array([20, 20, 20], dtype=np.uint8)

    origin_f32 = origin.astype(np.float32)
    x_end = (origin + np.array([size, 0.0, 0.0], dtype=np.float64)).astype(np.float32)
    y_end = (origin + np.array([0.0, size, 0.0], dtype=np.float64)).astype(np.float32)
    z_end = (origin + np.array([0.0, 0.0, size], dtype=np.float64)).astype(np.float32)

    strips = [
        np.stack([origin_f32, x_end], axis=0),
        np.stack([origin_f32, y_end], axis=0),
        np.stack([origin_f32, z_end], axis=0),
    ]
    colors = [color_x, color_y, color_z]
    labels = ["world +X", "world +Y", "world +Z"]

    line_kwargs = dict(strips=strips, colors=colors, labels=labels)
    if radius > 0:
        line_kwargs["radii"] = float(radius)
    rr.log("world/world_axes", rr.LineStrips3D(**line_kwargs))

    label_positions = np.stack([origin_f32, x_end, y_end, z_end], axis=0)
    label_colors = np.stack([color_origin, color_x, color_y, color_z], axis=0)
    rr.log(
        "world/world_axes/labels",
        rr.Points3D(
            positions=label_positions,
            colors=label_colors,
            labels=["world axes marker", "+X", "+Y", "+Z"],
        ),
    )

    print(
        "  world axes marker:"
        f" origin={origin.tolist()} size={size:.6g} up_axis={up_axis} offset={offset:.6g}"
    )


def log_camera_centers_and_traj(
    cams_ordered: Sequence[Dict[str, object]],
    center_radius: float,
    traj_radius: float,
    show_traj: bool,
) -> None:
    centers = camera_centers_array(cams_ordered)
    labels = [str(c.get("stem", f"cam_{i:04d}")) for i, c in enumerate(cams_ordered)]
    if len(centers) == 0:
        return

    center_kwargs = dict(
        positions=centers,
        colors=np.repeat(np.array([[180, 180, 180]], dtype=np.uint8), len(centers), axis=0),
        labels=labels,
    )
    if center_radius > 0:
        center_kwargs["radii"] = float(center_radius)
    rr.log("world/cameras/centers", rr.Points3D(**center_kwargs))

    if show_traj and len(centers) >= 2:
        traj_kwargs = dict(
            strips=[centers],
            colors=[np.array([255, 230, 0], dtype=np.uint8)],
            labels=["camera trajectory"],
        )
        if traj_radius > 0:
            traj_kwargs["radii"] = float(traj_radius)
        rr.log("world/cameras/trajectory", rr.LineStrips3D(**traj_kwargs))


def infer_hw_from_cam(cam: Dict[str, object], default_w: int, default_h: int) -> Tuple[int, int]:
    h = cam.get("height")
    w = cam.get("width")
    if h is not None and w is not None:
        return int(h), int(w)

    K = np.asarray(cam["K"], dtype=np.float64)
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    ww = int(round(max(default_w, 2.0 * cx))) if cx > 1e-6 else default_w
    hh = int(round(max(default_h, 2.0 * cy))) if cy > 1e-6 else default_h
    return hh, ww


def log_transform3d_compat(entity_base: str, c2w: np.ndarray, axis_length: float) -> None:
    try:
        rr.log(
            entity_base,
            rr.Transform3D(
                translation=c2w[:3, 3],
                mat3x3=c2w[:3, :3],
                axis_length=axis_length,
            ),
        )
    except TypeError:
        rr.log(
            entity_base,
            rr.Transform3D(
                translation=c2w[:3, 3],
                mat3x3=c2w[:3, :3],
            ),
        )


def make_pinhole_compat(K: np.ndarray, h: int, w: int, color: np.ndarray, line_width: float):
    base_kwargs = dict(
        image_from_camera=K,
        height=int(h),
        width=int(w),
        camera_xyz=rr.ViewCoordinates.RDF,
    )
    try:
        return rr.Pinhole(**base_kwargs, color=color, line_width=line_width)
    except TypeError:
        pass
    try:
        return rr.Pinhole(**base_kwargs, color=color)
    except TypeError:
        pass
    return rr.Pinhole(**base_kwargs)


def log_camera_frustums(
    cams_ordered: Sequence[Dict[str, object]],
    default_w: int,
    default_h: int,
    axis_length: float,
) -> None:
    color = np.array([90, 120, 255], dtype=np.uint8)
    for i, cam in enumerate(cams_ordered):
        c2w = np.asarray(cam["T_c2w"], dtype=np.float64)
        K = np.asarray(cam["K"], dtype=np.float64)
        h, w = infer_hw_from_cam(cam, default_w=default_w, default_h=default_h)
        base = f"world/cameras/frustums/view_{i:04d}"
        log_transform3d_compat(base, c2w=c2w, axis_length=axis_length)
        rr.log(base + "/pinhole", make_pinhole_compat(K, h=h, w=w, color=color, line_width=0.003))


def maybe_log_selected_images(
    selected_stems: Sequence[str],
    images: Dict[str, Path],
    max_images: int,
) -> None:
    if max_images <= 0:
        return
    stems = evenly_select(selected_stems, max_images)
    for i, stem in enumerate(stems):
        try:
            rgb = read_rgb(images[stem])
            rr_set_time_compat("frame", i)
            rr.log(f"world/selected_images/{stem}", rr.Image(rgb))
        except Exception as e:
            print(f"    [WARN] failed to log image {stem}: {e}")


# -----------------------------------------------------------------------------
# Per-scene export
# -----------------------------------------------------------------------------
def export_one_scene(scene_root: Path, save_rrd: Path, args: argparse.Namespace) -> None:
    image_dir = scene_root / args.images_dir
    cam_dir = scene_root / args.cams_dir
    depth_dir = scene_root / args.depth_dir

    images = collect_stem_to_path(image_dir, IMAGE_EXTS)
    depths = collect_stem_to_path(depth_dir, DEPTH_EXTS)
    cam_files = collect_stem_to_path(cam_dir, CAM_EXTS)

    image_cam_stems = sorted(set(images) & set(cam_files))
    if not image_cam_stems:
        raise RuntimeError(
            "No common stems among images/cams.\n"
            f"  images: {image_dir} ({len(images)})\n"
            f"  cams:   {cam_dir} ({len(cam_files)})\n"
            f"  depth:  {depth_dir} ({len(depths)})"
        )

    cams: Dict[str, Dict[str, object]] = {}
    for stem in sorted(cam_files):
        try:
            cams[stem] = parse_cam_txt(cam_files[stem])
        except Exception as e:
            print(f"  [WARN] failed to parse camera {stem}: {e}")

    image_cam_stems = [s for s in image_cam_stems if s in cams]
    if not image_cam_stems:
        raise RuntimeError("No valid parsed camera files for common RGB/camera frames.")

    depth_stems = sorted(set(image_cam_stems) & set(depths))
    selected_stems = evenly_select(depth_stems, args.max_point_views)
    image_log_stems = selected_stems if selected_stems else evenly_select(image_cam_stems, args.max_point_views)

    print(
        f"  images={len(images)} cams={len(cam_files)} depths={len(depths)} "
        f"image+cam={len(image_cam_stems)} rgbd={len(depth_stems)}"
    )
    if depth_stems:
        print(f"  frames used for point cloud: {len(selected_stems)} / {len(depth_stems)}")
        if len(selected_stems) > 1:
            print(f"  selected range: {selected_stems[0]} ... {selected_stems[-1]}")
    else:
        print("  [INFO] no matching depth frames; exporting camera/frustum/trajectory/RGB visualization only")

    pts, colors = build_point_cloud_arrays(scene_root, selected_stems, images, depths, cams, args)
    print(f"  raw points: {pts.shape[0]}")

    pts, colors, used_voxel = voxel_downsample_arrays(
        pts=pts,
        colors=colors,
        voxel_size=args.voxel_size,
        target_points=args.target_points,
    )
    print(f"  visualized points: {pts.shape[0]} | voxel_size={used_voxel:.8g}")

    if args.save_ply_dir and pts.shape[0] > 0:
        if o3d is None:
            raise ImportError("open3d is required to save point clouds as PLY. Please install open3d.")
        save_ply_dir = Path(args.save_ply_dir)
        save_ply_dir.mkdir(parents=True, exist_ok=True)
        out_ply = save_ply_dir / f"{sanitize_filename(scene_root.name)}.ply"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(out_ply), pcd)
        print(f"  saved point cloud: {out_ply}")
    elif args.save_ply_dir:
        print("  [INFO] skip PLY export because no point cloud was generated")

    cams_ordered = [cams[s] for s in sorted(cams)]
    reference_pts = reference_points_for_scene(pts, cams_ordered)

    # Camera axis scale from point-cloud bbox, or camera-center bbox when depth is unavailable.
    diag = infer_scene_diag(reference_pts)
    axis_size = float(args.axis_size) if args.axis_size > 0 else max(diag * 0.015, 1e-4)
    frustum_axis_len = max(axis_size, 1e-4)

    print(f"  all cameras logged: {len(cams_ordered)} | axis_size={axis_size:.6g}")

    save_rrd.parent.mkdir(parents=True, exist_ok=True)
    rr_init_save_compat(
        app_id=args.app_id,
        recording_id=sanitize_filename(scene_root.name),
        save_rrd=save_rrd,
    )
    rr_set_time_compat("stable_time", args.time_sequence)

    # Keep the same convention as OpenCV camera files: x right, y down, z forward.
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    send_clean_screenshot_blueprint(args)

    log_point_cloud("world/points", pts, colors, point_radius=args.point_radius)

    if args.show_world_axes:
        world_axes_radius = float(args.world_axes_radius) if args.world_axes_radius > 0 else float(args.axis_radius)
        log_world_axes_marker(
            pts=reference_pts,
            axis_size=args.world_axes_size,
            axis_size_ratio=args.world_axes_size_ratio,
            min_axis_size=args.world_axes_min_size,
            up_axis=args.world_axes_up_axis,
            up_offset=args.world_axes_offset,
            up_offset_ratio=args.world_axes_offset_ratio,
            radius=world_axes_radius,
        )

    log_camera_axes("world/cameras/axes", cams_ordered, axis_size=axis_size, radius=args.axis_radius)
    log_camera_centers_and_traj(
        cams_ordered,
        center_radius=args.center_radius,
        traj_radius=args.traj_radius,
        show_traj=args.show_traj,
    )

    if args.show_frustum:
        log_camera_frustums(
            cams_ordered,
            default_w=args.default_width,
            default_h=args.default_height,
            axis_length=frustum_axis_len,
        )

    maybe_log_selected_images(image_log_stems, images, max_images=args.log_selected_images)
    rr_disconnect_compat()

    print(f"  [DONE] Saved RRD to: {save_rrd}")


def send_clean_screenshot_blueprint(args: argparse.Namespace) -> None:
    try:
        line_grid = rrb.LineGrid3D(visible=not args.hide_grid)

        blueprint = rrb.Blueprint(
            rrb.Spatial3DView(
                origin="/world",
                name="A3D Scene",
                background=list(args.background),
                line_grid=line_grid,
            ),
            collapse_panels=bool(args.collapse_panels),
        )
        rr.send_blueprint(blueprint)
    except Exception as e:
        print(f"[WARN] failed to send Rerun blueprint: {e}")

# -----------------------------------------------------------------------------
# Args / Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export every A3D/WAI-style scene under a dataset root to separate Rerun .rrd files."
    )
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Dataset root containing multiple scene folders.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory. Each scene is saved as <scene_name>.rrd.")

    parser.add_argument("--images_dir", type=str, default="images")
    parser.add_argument("--cams_dir", type=str, default="cams")
    parser.add_argument("--depth_dir", type=str, default="depth")
    parser.add_argument("--require_depth", action="store_true",
                        help="Only treat folders with a depth directory as valid scenes.")

    parser.add_argument("--scene_glob", type=str, default="*",
                        help="Filter scene folder names, e.g. '0b*' or '*building*'.")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively find scene folders under dataset_root. Default only checks immediate children.")
    parser.add_argument("--start_scene", type=str, default="",
                        help="Start from the first sorted scene name >= this value.")
    parser.add_argument("--max_scenes", type=int, default=0,
                        help="Max number of scenes to export after filtering/start_scene. <=0 means all.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip scenes whose output .rrd already exists.")
    parser.add_argument("--stop_on_error", action="store_true",
                        help="Stop immediately when one scene fails. Default continues and prints a summary.")

    parser.add_argument("--max_point_views", type=int, default=5,
                        help="Max RGB-D frames used to build point cloud. Uniformly sampled over the whole scene.")
    parser.add_argument("--max_side", type=int, default=250,
                        help="Resize image/depth so max(H,W)<=max_side before backprojection. <=0 disables.")
    parser.add_argument("--pixel_stride", type=int, default=1,
                        help="Optional extra pixel stride after max_side resize. 1 means use all valid pixels.")
    parser.add_argument("--max_points_per_view", type=int, default=0,
                        help="Randomly cap valid points per selected view before global voxel sampling. <=0 disables.")

    parser.add_argument("--voxel_size", type=float, default=0.0,
                        help="Voxel size for final point cloud. >0 fixed; <=0 auto by target_points.")
    parser.add_argument("--target_points", type=int, default=800000,
                        help="When voxel_size<=0, auto voxel downsample to roughly <= this point count. <=0 disables.")

    parser.add_argument("--depth_scale", type=float, default=1.0,
                        help="Depth divisor. EXR/npy metric depth usually uses 1. uint16 mm depth uses 1000.")
    parser.add_argument("--depth_min", type=float, default=1e-6)
    parser.add_argument("--depth_max", type=float, default=1e8)

    parser.add_argument("--axis_size", type=float, default=0.0,
                        help="Camera axis length in world units. <=0 auto from point-cloud bbox.")
    parser.add_argument("--axis_radius", type=float, default=0.0,
                        help="Camera axis line radius. <=0 lets Rerun choose default.")
    parser.add_argument("--center_radius", type=float, default=0.0,
                        help="Camera center point radius. <=0 lets Rerun choose default.")
    parser.add_argument("--traj_radius", type=float, default=0.0,
                        help="Trajectory line radius. <=0 lets Rerun choose default.")
    parser.add_argument("--point_radius", type=float, default=0.0,
                        help="Point radius for Points3D. <=0 lets Rerun choose default.")

    parser.add_argument(
        "--show_world_axes",
        action="store_true",
        help=(
            "Log a large world-axis marker above the point-cloud bbox center. "
            "+X=red, +Y=green, +Z=blue. This is a reference marker, not necessarily the true world origin."
        ),
    )
    parser.add_argument(
        "--world_axes_size",
        type=float,
        default=0.0,
        help="World-axis marker length in world units. <=0 uses bbox diagonal * --world_axes_size_ratio.",
    )
    parser.add_argument(
        "--world_axes_size_ratio",
        type=float,
        default=0.12,
        help="Auto world-axis marker length as a ratio of point-cloud bbox diagonal.",
    )
    parser.add_argument(
        "--world_axes_min_size",
        type=float,
        default=1e-4,
        help="Minimum auto world-axis marker length.",
    )
    parser.add_argument(
        "--world_axes_up_axis",
        type=str,
        default="z",
        choices=["x", "y", "z", "-x", "-y", "-z"],
        help=(
            "World direction used as 'above' when placing the world-axis marker. "
            "Use z for Z-up datasets, -y for OpenCV/RDF-style visual up, or y for Y-up datasets."
        ),
    )
    parser.add_argument(
        "--world_axes_offset",
        type=float,
        default=0.0,
        help="Offset from point-cloud bbox center along --world_axes_up_axis. <=0 uses size * --world_axes_offset_ratio.",
    )
    parser.add_argument(
        "--world_axes_offset_ratio",
        type=float,
        default=0.75,
        help="Auto offset from bbox center as a ratio of world-axis marker size.",
    )
    parser.add_argument(
        "--world_axes_radius",
        type=float,
        default=0.0,
        help="World-axis line radius. <=0 reuses --axis_radius, and <=0 there lets Rerun choose default.",
    )

    parser.add_argument("--show_traj", action="store_true", help="Log yellow trajectory line through all camera centers.")
    parser.add_argument("--show_frustum", action="store_true", help="Also log Rerun Pinhole camera frustums.")
    parser.add_argument("--log_selected_images", type=int, default=0,
                        help="Optionally log up to N selected RGB images to each .rrd. 0 disables.")
    parser.add_argument("--default_width", type=int, default=640,
                        help="Fallback image width for Pinhole frustums when camera txt lacks h/w.")
    parser.add_argument("--default_height", type=int, default=480,
                        help="Fallback image height for Pinhole frustums when camera txt lacks h/w.")

    parser.add_argument("--app_id", type=str, default="visualize_a3d_dataset_rerun")
    parser.add_argument("--time_sequence", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_ply_dir", type=str, default="",
                        help="Optional directory to also save final voxelized point clouds as <scene_name>.ply.")
    parser.add_argument(
        "--background",
        type=int,
        nargs=3,
        default=[255, 255, 255],
        help="3D view background RGB, e.g. --background 255 255 255",
    )
    parser.add_argument(
        "--hide_grid",
        action="store_true",
        help="Hide Rerun 3D grid for cleaner screenshots.",
    )
    parser.add_argument(
        "--collapse_panels",
        action="store_true",
        help="Collapse side panels in Rerun viewer for cleaner screenshots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = find_scene_dirs(
        dataset_root=dataset_root,
        images_dir=args.images_dir,
        cams_dir=args.cams_dir,
        depth_dir=args.depth_dir,
        scene_glob=args.scene_glob,
        recursive=args.recursive,
        require_depth=args.require_depth,
    )

    if args.start_scene:
        scenes = [s for s in scenes if s.name >= args.start_scene or str(s.relative_to(dataset_root)) >= args.start_scene]

    if args.max_scenes > 0:
        scenes = scenes[:args.max_scenes]

    if not scenes:
        raise RuntimeError(
            f"No valid scene folders found under {dataset_root}. "
            f"Expected each scene to contain {args.images_dir}/ and {args.cams_dir}/"
            f"{', plus ' + args.depth_dir + '/' if args.require_depth else ''}."
        )

    print(f"[INFO] dataset_root: {dataset_root}")
    print(f"[INFO] output_dir:   {output_dir}")
    print(f"[INFO] scenes found: {len(scenes)}")

    failed: List[Tuple[str, str]] = []
    skipped = 0
    done = 0

    for i, scene_root in enumerate(scenes):
        scene_file = sanitize_filename(scene_root.name) + ".rrd"
        save_rrd = output_dir / scene_file

        print("\n" + "=" * 100)
        print(f"[SCENE {i + 1:04d}/{len(scenes):04d}] {scene_root.name}")
        print(f"[OUT] {save_rrd}")
        print("=" * 100)

        if args.skip_existing and save_rrd.exists():
            print(f"[SKIP] existing file: {save_rrd}")
            skipped += 1
            continue

        try:
            export_one_scene(scene_root, save_rrd=save_rrd, args=args)
            done += 1
        except Exception as e:
            rr_disconnect_compat()
            msg = f"{type(e).__name__}: {e}"
            failed.append((scene_root.name, msg))
            print(f"[ERROR] failed scene {scene_root.name}: {msg}")
            if args.stop_on_error:
                traceback.print_exc()
                raise
        finally:
            gc.collect()

    print("\n" + "=" * 100)
    print("[SUMMARY]")
    print(f"  done:    {done}")
    print(f"  skipped: {skipped}")
    print(f"  failed:  {len(failed)}")
    if failed:
        for name, msg in failed:
            print(f"  - {name}: {msg}")
    print(f"  output:  {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()

"""

python scripts/visualize_dataset_rerun.py --dataset_root /opt/data/private/dataset/data/UAVFF3D-Syn-S \
    --show_frustum --show_world_axes --output_dir experiments/dataset_viz/UAVFF3D-Syn-S

python scripts/visualize_dataset_rerun.py --dataset_root /opt/data/private/dataset/data/blendedmvs \
    --show_frustum --show_world_axes --output_dir experiments/dataset_viz/blendedmvs

python scripts/visualize_dataset_rerun.py --dataset_root /opt/data/private/dataset/data/uavscenes \
    --show_frustum --show_world_axes --output_dir experiments/dataset_viz/uavscenes

python scripts/visualize_dataset_rerun.py --dataset_root /opt/data/private/dataset/data/whu_whuomvs \
    --show_frustum --show_world_axes --output_dir experiments/dataset_viz/whu_whuomvs

python scripts/visualize_dataset_rerun.py --dataset_root /opt/data/private/dataset/data/UAVFF3D-Syn-L \
    --show_frustum --show_world_axes --output_dir experiments/dataset_viz/UAVFF3D-Syn-L

python scripts/visualize_dataset_rerun.py --dataset_root /opt/data/private/dataset/data/UAVFF3D-Real \
    --show_frustum --show_world_axes --output_dir experiments/dataset_viz/UAVFF3D-Real

    
"""
