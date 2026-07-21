#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a MapAnything-compatible model on one RGB-D scene and save Rerun .rrd.

Expected scene layout:
    scene_dir/
      images/  *.jpg | *.jpeg | *.png | *.bmp | *.tif | *.tiff  # required
      cams/    *.txt   # optional A3D/WAI-style extrinsic + intrinsic priors
      depth/   *.exr | *.npy | *.png | *.tif | *.tiff           # optional depth priors

The script builds a MapAnything-style list of views, runs the selected model,
and writes a single Rerun recording containing:
  - optional GT point cloud and cameras when cams/depth priors are available
  - predicted point cloud and predicted cameras, optionally aligned to GT
  - a visible world-axis marker
  - optional input images

Alignment modes:
  none   raw prediction, useful for checking whether a world-frame prior works
  scale  scale+translate prediction into the GT frame without rotating it.
         Scale is estimated from GT RGB-D point correspondences when available,
         otherwise from matched GT/predicted camera centers.

python scripts/predict_scene_to_rrd.py \
    --scene_dir /opt/data/private/dataset/data/NPU_Dronemap/gopro-npu-kfs \
    --model geoff3d  \
    --checkpoint experiments/dom/uav_training/pi3x_zup_translation_yaw_aug_8v_6d_16ipg_2g/checkpoint-best.pth \
    --output_rrd outputs/gopro.rrd \
    --num_views 10 \
    --stride 1 \
    --max_side 518 \
    --align none

python scripts/predict_scene_to_rrd.py \
  --scene_dir /opt/data/private/dataset/data/enrich/aerial_ndiir2 \
  --model vggt_omega \
  --checkpoint checkpoints/vggt-omega/vggt_omega_1b_512.pt \
  --output_rrd outputs/debug_vggt_omega.rrd \
  --num_views 30 \
  --stride 1 \
  --max_side 512 \
  --size_multiple 16 \
  --align scale

Example:
    python scripts/predict_scene_to_rrd.py \
      --scene_dir /path/to/scene_000 \
      --model geoff3d \
      --output_rrd outputs/scene_000_geoff3d.rrd \
      --num_views 8 \
      --max_side 518 \
      --align none

    python scripts/predict_scene_to_rrd.py \
      --scene_dir /path/to/scene_000 \
      --model pi3 \
      --output_rrd outputs/scene_000_pi3_scale.rrd \
      --align scale \
      --log_raw_when_aligned

Open:
    rerun outputs/scene_000_geoff3d.rrd
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Must be set before importing cv2 when OpenCV EXR support is available.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import hydra
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import rerun as rr

try:
    import rerun.blueprint as rrb
except Exception:
    rrb = None

from mapanything.models import init_model
from mapanything.utils.geometry import (
    get_rays_in_camera_frame,
    quaternion_to_rotation_matrix,
)
from mapanything.utils.torch_hub_setup import configure_torch_hub


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEPTH_EXTS = {".exr", ".npy", ".png", ".tif", ".tiff"}
CAM_EXTS = {".txt"}


# -----------------------------------------------------------------------------
# File collection / camera parsing
# -----------------------------------------------------------------------------
def collect_stem_to_path(folder: Path, exts: Iterable[str]) -> Dict[str, Path]:
    exts = {e.lower() for e in exts}
    if not folder.exists():
        return {}
    return {
        p.stem: p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    }


def sanitize_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[\\/:*?\"<>|\s]+", "_", name)
    return name or "scene"


def contiguous_select(items: Sequence[str], max_count: int) -> List[str]:
    """Select a continuous window from the already-filtered frame list.

    The input list is assumed to have already been sorted and filtered by
    --frame_glob, --start, and --stride. Unlike uniform/even sampling, this keeps
    neighboring selected views adjacent in that filtered order.
    """
    items = list(items)
    if max_count <= 0 or len(items) <= max_count:
        return items
    return items[: int(max_count)]


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

    height: Optional[int] = None
    width: Optional[int] = None
    fov: Optional[float] = None
    idx_hwf = -1
    for i, ln in enumerate(lines):
        tokens = ln.lower().replace(":", " ").split()
        if "h" in tokens and "w" in tokens and ("fov" in tokens or "hfov" in tokens):
            idx_hwf = i
            break
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
        "path": str(cam_path),
        "K": K,
        "T_w2c": T_w2c,
        "T_c2w": np.linalg.inv(T_w2c),
        "height": height,
        "width": width,
        "fov": fov,
    }


# -----------------------------------------------------------------------------
# RGB-D IO and geometry
# -----------------------------------------------------------------------------
def read_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def read_depth(path: Path, depth_scale: float = 1.0) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(str(path))
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(
                f"Cannot read depth: {path}. For EXR, check OPENCV_IO_ENABLE_OPENEXR/OpenCV build."
            )
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if depth_scale != 1.0:
        depth = depth / float(depth_scale)
    return depth


def _round_down_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return int(x)
    return max(m, int(x) // m * m)


def compute_target_hw(depth_h: int, depth_w: int, max_side: int, multiple: int) -> Tuple[int, int]:
    h, w = int(depth_h), int(depth_w)
    if max_side > 0 and max(h, w) > max_side:
        scale = float(max_side) / float(max(h, w))
        h = max(1, int(round(h * scale)))
        w = max(1, int(round(w * scale)))
    h = _round_down_to_multiple(h, multiple)
    w = _round_down_to_multiple(w, multiple)
    return h, w


def resize_rgb_depth_K(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    cam_width: Optional[int],
    cam_height: Optional[int],
    target_h: int,
    target_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize RGB/depth to target size and scale K consistently."""
    K = K.astype(np.float64).copy()
    depth_h, depth_w = depth.shape[:2]

    if cam_width is None or cam_height is None:
        cam_height, cam_width = rgb.shape[:2]

    # First scale intrinsics from camera-file resolution to depth resolution.
    if int(cam_width) != int(depth_w) or int(cam_height) != int(depth_h):
        sx = float(depth_w) / float(cam_width)
        sy = float(depth_h) / float(cam_height)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    if rgb.shape[0] != depth_h or rgb.shape[1] != depth_w:
        rgb = cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_AREA)

    if int(depth_h) != int(target_h) or int(depth_w) != int(target_w):
        sx = float(target_w) / float(depth_w)
        sy = float(target_h) / float(depth_h)
        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    return rgb, depth.astype(np.float32), K



def resize_rgb_to_target(rgb: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize RGB to target size."""
    if rgb.shape[0] != int(target_h) or rgb.shape[1] != int(target_w):
        rgb = cv2.resize(rgb, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)
    return rgb


def resize_depth_to_target(depth: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize depth to target size without changing metric values."""
    if depth.shape[0] != int(target_h) or depth.shape[1] != int(target_w):
        depth = cv2.resize(depth, (int(target_w), int(target_h)), interpolation=cv2.INTER_NEAREST)
    return depth.astype(np.float32)


def scale_K_to_target(
    K: np.ndarray,
    cam_width: Optional[int],
    cam_height: Optional[int],
    source_h: int,
    source_w: int,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Scale intrinsics from camera-file resolution to source image/depth, then to target size."""
    K = K.astype(np.float64).copy()

    if cam_width is None or cam_height is None:
        cam_height, cam_width = int(source_h), int(source_w)

    if int(cam_width) != int(source_w) or int(cam_height) != int(source_h):
        sx = float(source_w) / float(cam_width)
        sy = float(source_h) / float(cam_height)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    if int(source_h) != int(target_h) or int(source_w) != int(target_w):
        sx = float(target_w) / float(source_w)
        sy = float(target_h) / float(source_h)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    return K


def depth_to_world_points_numpy(depth: np.ndarray, K: np.ndarray, T_c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)
    R = T_c2w[:3, :3]
    t = T_c2w[:3, 3]
    pts_world = np.einsum("ij,hwj->hwi", R, pts_cam) + t[None, None, :]
    return pts_cam.astype(np.float32), pts_world.astype(np.float32)


def np_to_torch_img(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
    return x.to(device)


def numpy_quat_xyzw_from_rotmat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to XYZW quaternion."""
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    if q[3] < 0:
        q = -q
    return q


def build_views_from_scene(args: argparse.Namespace, device: torch.device):
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    images_dir = scene_dir / args.images_dir
    cams_dir = scene_dir / args.cams_dir
    depth_dir = scene_dir / args.depth_dir

    images = collect_stem_to_path(images_dir, IMAGE_EXTS)
    depths = collect_stem_to_path(depth_dir, DEPTH_EXTS)
    cam_paths = collect_stem_to_path(cams_dir, CAM_EXTS)

    if not images:
        raise RuntimeError(f"No images found under {images_dir}.")

    selected_stems = sorted(images)
    if args.frame_glob and args.frame_glob != "*":
        selected_stems = [s for s in selected_stems if fnmatch.fnmatch(s, args.frame_glob)]
    if args.start > 0:
        selected_stems = selected_stems[int(args.start) :]
    if args.stride > 1:
        selected_stems = selected_stems[:: int(args.stride)]
    selected_stems = contiguous_select(selected_stems, int(args.num_views))

    if len(selected_stems) == 0:
        raise RuntimeError(f"No frames selected under {images_dir}.")

    has_cams_dir = cams_dir.exists()
    has_depth_dir = depth_dir.exists()
    num_cam_matches = sum(1 for stem in selected_stems if stem in cam_paths)
    num_depth_matches = sum(1 for stem in selected_stems if stem in depths)
    print(
        f"Selected {len(selected_stems)} image frames. "
        f"Camera priors: {'enabled' if num_cam_matches else 'disabled'} "
        f"({num_cam_matches}/{len(selected_stems)} matched; dir_exists={has_cams_dir}). "
        f"Depth priors: {'enabled' if num_depth_matches else 'disabled'} "
        f"({num_depth_matches}/{len(selected_stems)} matched; dir_exists={has_depth_dir})."
    )

    cams: Dict[str, Dict[str, object]] = {}
    for stem in selected_stems:
        if stem not in cam_paths:
            continue
        try:
            cams[stem] = parse_cam_txt(cam_paths[stem])
        except Exception as e:
            print(f"[WARN] failed to parse camera prior for {stem}: {e}; skipping camera prior for this frame")

    # Use the first available depth as size reference when present; otherwise use the first RGB image.
    first_rgb = read_rgb(images[selected_stems[0]])
    ref_depth = None
    for stem in selected_stems:
        if stem in depths:
            try:
                ref_depth = read_depth(depths[stem], depth_scale=args.depth_scale)
                break
            except Exception as e:
                print(f"[WARN] failed to read reference depth for {stem}: {e}; trying next depth")
    if ref_depth is not None:
        ref_h, ref_w = ref_depth.shape[:2]
    else:
        ref_h, ref_w = first_rgb.shape[:2]

    target_h, target_w = compute_target_hw(
        depth_h=ref_h,
        depth_w=ref_w,
        max_side=args.max_side,
        multiple=args.size_multiple,
    )

    views: List[Dict[str, object]] = []
    gt_points_all: List[np.ndarray] = []
    gt_colors_all: List[np.ndarray] = []
    gt_maps: List[np.ndarray] = []
    valid_masks_np: List[np.ndarray] = []
    resized_rgbs: List[np.ndarray] = []
    resized_stems: List[str] = []

    for i, stem in enumerate(selected_stems):
        rgb_raw = read_rgb(images[stem])
        cam = cams.get(stem)

        depth_raw: Optional[np.ndarray] = None
        if stem in depths:
            try:
                depth_raw = read_depth(depths[stem], depth_scale=args.depth_scale)
            except Exception as e:
                print(f"[WARN] failed to read depth prior for {stem}: {e}; skipping depth prior for this frame")
                depth_raw = None

        K: Optional[np.ndarray] = None
        rgb = resize_rgb_to_target(rgb_raw, target_h=target_h, target_w=target_w)
        depth: Optional[np.ndarray] = None

        if cam is not None:
            if cam.get("width") is None or cam.get("height") is None:
                cam["height"], cam["width"] = rgb_raw.shape[:2]

            # If depth exists, match the historical behavior: first scale K to depth resolution.
            # Otherwise scale K from the camera-file resolution to the RGB resolution.
            if depth_raw is not None:
                rgb, depth, K = resize_rgb_depth_K(
                    rgb=rgb_raw,
                    depth=depth_raw,
                    K=np.asarray(cam["K"], dtype=np.float64),
                    cam_width=cam.get("width"),
                    cam_height=cam.get("height"),
                    target_h=target_h,
                    target_w=target_w,
                )
            else:
                K = scale_K_to_target(
                    K=np.asarray(cam["K"], dtype=np.float64),
                    cam_width=cam.get("width"),
                    cam_height=cam.get("height"),
                    source_h=rgb_raw.shape[0],
                    source_w=rgb_raw.shape[1],
                    target_h=target_h,
                    target_w=target_w,
                )
        elif depth_raw is not None:
            depth = resize_depth_to_target(depth_raw, target_h=target_h, target_w=target_w)

        img_t = np_to_torch_img(rgb, device=device)
        view: Dict[str, object] = {
            "img": img_t,
            "is_metric_scale": torch.ones((1,), dtype=torch.bool, device=device),
            "is_synthetic": torch.zeros((1,), dtype=torch.bool, device=device),
            "true_shape": torch.tensor([[target_h, target_w]], dtype=torch.int64, device=device),
            "data_norm_type": ["identity"],
            "label": [scene_dir.name],
            "instance": [stem],
            "idx": [f"{scene_dir.name}/{stem}"],
        }

        valid_np: Optional[np.ndarray] = None
        pts_world_np: Optional[np.ndarray] = None

        if depth is not None:
            valid_np = (
                np.isfinite(depth)
                & (depth > float(args.depth_min))
                & (depth < float(args.depth_max))
            )
            depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0).to(device)
            valid_t = torch.from_numpy(valid_np).unsqueeze(0).to(device)
            view["depthmap"] = depth_t
            view["valid_mask"] = valid_t
            view["non_ambiguous_mask"] = valid_t.clone()

        if cam is not None and K is not None:
            T_c2w = np.asarray(cam["T_c2w"], dtype=np.float64)
            K_t = torch.from_numpy(K.astype(np.float32)).unsqueeze(0).to(device)
            pose_t = torch.from_numpy(T_c2w.astype(np.float32)).unsqueeze(0).to(device)
            pose_quat = numpy_quat_xyzw_from_rotmat(T_c2w[:3, :3])
            pose_quat_t = torch.from_numpy(pose_quat).unsqueeze(0).to(device)
            pose_trans_t = pose_t[..., :3, 3]

            view["camera_intrinsics"] = K_t
            view["camera_pose"] = pose_t
            view["camera_pose_quats"] = pose_quat_t
            view["camera_pose_trans"] = pose_trans_t
            view["world_translation"] = pose_trans_t

            if depth is not None:
                pts_cam_np, pts_world_np = depth_to_world_points_numpy(depth, K, T_c2w)
                depth_along_ray_np = np.linalg.norm(pts_cam_np, axis=-1, keepdims=True).astype(np.float32)
                if valid_np is None:
                    valid_np = np.isfinite(depth)
                valid_np = (
                    valid_np
                    & np.isfinite(pts_world_np).all(axis=-1)
                    & np.isfinite(pts_cam_np).all(axis=-1)
                )
                valid_t = torch.from_numpy(valid_np).unsqueeze(0).to(device)
                pts_world_t = torch.from_numpy(pts_world_np).unsqueeze(0).to(device)
                pts_cam_t = torch.from_numpy(pts_cam_np).unsqueeze(0).to(device)
                depth_along_ray_t = torch.from_numpy(depth_along_ray_np).unsqueeze(0).to(device)
                _, ray_dirs_t = get_rays_in_camera_frame(K_t, target_h, target_w, normalize_to_unit_sphere=True)

                view["pts3d"] = pts_world_t
                view["pts3d_cam"] = pts_cam_t
                view["depth_along_ray"] = depth_along_ray_t
                view["ray_directions_cam"] = ray_dirs_t
                view["valid_mask"] = valid_t
                view["non_ambiguous_mask"] = valid_t.clone()

        views.append(view)
        gt_maps.append(pts_world_np if pts_world_np is not None else np.empty((0, 0, 3), np.float32))
        valid_masks_np.append(valid_np if valid_np is not None else np.zeros((0, 0), dtype=bool))

        if pts_world_np is not None and valid_np is not None and valid_np.any():
            gt_points_all.append(pts_world_np[valid_np].reshape(-1, 3))
            gt_colors_all.append(rgb[valid_np].reshape(-1, 3).astype(np.uint8))

        resized_rgbs.append(rgb)
        resized_stems.append(stem)

        priors = []
        if cam is not None and K is not None:
            priors.append("cam")
        if depth is not None:
            priors.append("depth")
        if pts_world_np is not None:
            priors.append("pts3d")
        prior_text = ",".join(priors) if priors else "none"
        valid_count = int(valid_np.sum()) if valid_np is not None else 0
        print(
            f"[{i + 1:03d}/{len(selected_stems):03d}] {stem}: "
            f"size={target_w}x{target_h}, priors={prior_text}, valid_gt_points={valid_count}"
        )

    meta = {
        "scene_dir": str(scene_dir),
        "stems": resized_stems,
        "target_h": target_h,
        "target_w": target_w,
        "cams": cams,
        "rgbs": resized_rgbs,
        "gt_maps": gt_maps,
        "valid_masks": valid_masks_np,
        "gt_points": np.concatenate(gt_points_all, axis=0) if gt_points_all else np.empty((0, 3), np.float32),
        "gt_colors": np.concatenate(gt_colors_all, axis=0) if gt_colors_all else np.empty((0, 3), np.uint8),
        "num_cam_priors": int(sum(1 for stem in resized_stems if stem in cams)),
        "num_depth_priors": int(sum(1 for stem in resized_stems if stem in depths)),
        "num_gt_rgbd_priors": int(sum(1 for m in gt_maps if m.size > 0)),
    }
    return views, meta


# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------
def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device_arg == "cuda":
        device_arg = "cuda:0"
    elif device_arg.isdigit():
        device_arg = f"cuda:{int(device_arg)}"

    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print(f"Using device {device}: {torch.cuda.get_device_name(device.index)}")
    else:
        print(f"Using device {device}")
    return device


def init_model_from_hydra(
    model_name: str,
    machine: str,
    hydra_overrides: Sequence[str],
    device: torch.device,
):
    GlobalHydra.instance().clear()
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "configs"
    hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir))
    overrides = [f"model={model_name}", f"machine={machine}"] + list(hydra_overrides)
    cfg = hydra.compose(config_name="train", overrides=overrides)
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    configure_torch_hub(cfg.machine)
    model = init_model(
        cfg.model.model_str,
        cfg.model.model_config,
        torch_hub_force_reload=cfg.model.torch_hub_force_reload,
    )

    pretrained = getattr(cfg.model, "pretrained", None)
    if pretrained:
        print(f"Loading checkpoint from config model.pretrained={pretrained}")
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        print(model.load_state_dict(state, strict=False))
        del ckpt

    model.to(device)
    model.eval()
    return model, cfg


def load_optional_checkpoint(model: torch.nn.Module, checkpoint: Optional[str]) -> None:
    if not checkpoint:
        return
    print(f"Loading checkpoint override: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    print(model.load_state_dict(state, strict=False))
    del ckpt


# -----------------------------------------------------------------------------
# Prediction post-processing
# -----------------------------------------------------------------------------
def torch_to_np(x, dtype=np.float32):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach()
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        x = x.cpu().numpy()
    return np.asarray(x, dtype=dtype)


def pred_pose_to_c2w(pred: Dict[str, torch.Tensor]) -> Optional[np.ndarray]:
    if "cam_trans" not in pred or "cam_quats" not in pred:
        return None
    trans = pred["cam_trans"]
    quat = pred["cam_quats"]
    if trans.ndim == 2:
        trans = trans[0]
    if quat.ndim == 2:
        quat = quat[0]
    R_t = quaternion_to_rotation_matrix(quat.float().unsqueeze(0))[0]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = torch_to_np(R_t)
    T[:3, 3] = torch_to_np(trans.float())
    return T


def collect_pred_outputs(
    preds: List[Dict[str, torch.Tensor]],
    rgbs: Sequence[np.ndarray],
    args: argparse.Namespace,
    stems: Optional[Sequence[str]] = None,
):
    pred_points_all: List[np.ndarray] = []
    pred_colors_all: List[np.ndarray] = []
    pred_maps: List[np.ndarray] = []
    pred_valid_masks: List[np.ndarray] = []
    pred_cams: List[Dict[str, object]] = []

    for i, pred in enumerate(preds):
        pred_stem = str(stems[i]) if stems is not None and i < len(stems) else f"pred_{i:03d}"

        # Collect predicted camera pose even if the model does not return pts3d for this view.
        T = pred_pose_to_c2w(pred)
        if T is not None and np.isfinite(T).all():
            pred_cams.append({"stem": pred_stem, "pred_index": int(i), "T_c2w": T})

        pts = pred.get("pts3d", None)
        if pts is None:
            pred_maps.append(np.empty((0, 0, 3), np.float32))
            pred_valid_masks.append(np.zeros((0, 0), dtype=bool))
            continue

        pts_np = torch_to_np(pts[0] if pts.ndim == 4 else pts, dtype=np.float32)
        h, w = pts_np.shape[:2]
        rgb = rgbs[i]
        if rgb.shape[0] != h or rgb.shape[1] != w:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

        finite = np.isfinite(pts_np).all(axis=-1)
        if "pts3d_cam" in pred:
            pts_cam = torch_to_np(pred["pts3d_cam"][0], dtype=np.float32)
            if pts_cam.shape[:2] == finite.shape:
                finite &= np.isfinite(pts_cam).all(axis=-1)
                finite &= pts_cam[..., 2] > float(args.pred_min_depth)

        if "conf" in pred and args.conf_quantile > 0:
            conf = torch_to_np(pred["conf"][0], dtype=np.float32)
            if conf.shape[:2] == finite.shape:
                good = np.isfinite(conf) & finite
                if good.any():
                    thr = np.quantile(conf[good], float(args.conf_quantile))
                    finite &= conf >= thr

        pred_maps.append(pts_np)
        pred_valid_masks.append(finite)
        pred_points_all.append(pts_np[finite].reshape(-1, 3))
        pred_colors_all.append(rgb[finite].reshape(-1, 3).astype(np.uint8))

    points = np.concatenate(pred_points_all, axis=0) if pred_points_all else np.empty((0, 3), np.float32)
    colors = np.concatenate(pred_colors_all, axis=0) if pred_colors_all else np.empty((0, 3), np.uint8)
    return points, colors, pred_maps, pred_valid_masks, pred_cams


def sample_points_and_colors(points: np.ndarray, colors: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    return points, colors


# -----------------------------------------------------------------------------
# Optional scale alignment
# -----------------------------------------------------------------------------
def sample_alignment_correspondences(
    gt_maps: Sequence[np.ndarray],
    pred_maps: Sequence[np.ndarray],
    gt_valid_masks: Sequence[np.ndarray],
    pred_valid_masks: Sequence[np.ndarray],
    max_samples_per_view: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample matched GT/predicted 3D points at the same pixels.

    These correspondences are used only to estimate a global scale and translation.
    No rotation is estimated or applied.
    """
    rng = np.random.default_rng(seed)
    gt_corr_all: List[np.ndarray] = []
    pr_corr_all: List[np.ndarray] = []

    for view_idx, (gt, pr, gt_valid, pr_valid) in enumerate(zip(gt_maps, pred_maps, gt_valid_masks, pred_valid_masks)):
        if gt.shape[:2] != pr.shape[:2] or gt.ndim != 3 or pr.ndim != 3:
            print(f"[WARN] skip scale alignment view {view_idx}: shape mismatch gt={gt.shape}, pred={pr.shape}")
            continue

        valid = (
            gt_valid.astype(bool)
            & pr_valid.astype(bool)
            & np.isfinite(gt).all(axis=-1)
            & np.isfinite(pr).all(axis=-1)
        )
        v, u = np.nonzero(valid)
        if v.size == 0:
            continue
        if max_samples_per_view > 0 and v.size > max_samples_per_view:
            sel = rng.choice(v.size, size=max_samples_per_view, replace=False)
            v = v[sel]
            u = u[sel]
        gt_corr_all.append(gt[v, u].reshape(-1, 3).astype(np.float32))
        pr_corr_all.append(pr[v, u].reshape(-1, 3).astype(np.float32))

    if not gt_corr_all:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
    return np.concatenate(gt_corr_all, axis=0), np.concatenate(pr_corr_all, axis=0)


def sample_camera_center_correspondences(
    meta: Dict[str, object],
    pred_cams: Sequence[Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return matched GT and predicted camera centers.

    Prefer exact stem matching. Fall back to input order only when exact matching
    fails, so scenes using old pred_000-style labels still get a usable alignment.
    """
    cams_by_stem = meta.get("cams", {})
    stems = list(meta.get("stems", []))

    gt_corr_all: List[np.ndarray] = []
    pr_corr_all: List[np.ndarray] = []
    matched_stems: List[str] = []

    # Preferred path: pred camera stem is the original input image stem.
    for pred_cam in pred_cams:
        stem = str(pred_cam.get("stem", ""))
        if stem not in cams_by_stem:
            continue
        gt_T = np.asarray(cams_by_stem[stem]["T_c2w"], dtype=np.float64)
        pr_T = np.asarray(pred_cam["T_c2w"], dtype=np.float64)
        gt_center = gt_T[:3, 3]
        pr_center = pr_T[:3, 3]
        if np.isfinite(gt_center).all() and np.isfinite(pr_center).all():
            gt_corr_all.append(gt_center.astype(np.float32))
            pr_corr_all.append(pr_center.astype(np.float32))
            matched_stems.append(stem)

    # Backward-compatible fallback: align by order.
    if not gt_corr_all:
        gt_items = [(stem, cams_by_stem[stem]) for stem in stems if stem in cams_by_stem]
        for pred_cam, (stem, gt_cam) in zip(pred_cams, gt_items):
            gt_T = np.asarray(gt_cam["T_c2w"], dtype=np.float64)
            pr_T = np.asarray(pred_cam["T_c2w"], dtype=np.float64)
            gt_center = gt_T[:3, 3]
            pr_center = pr_T[:3, 3]
            if np.isfinite(gt_center).all() and np.isfinite(pr_center).all():
                gt_corr_all.append(gt_center.astype(np.float32))
                pr_corr_all.append(pr_center.astype(np.float32))
                matched_stems.append(stem)

    if not gt_corr_all:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32), []
    return (
        np.stack(gt_corr_all, axis=0).astype(np.float32),
        np.stack(pr_corr_all, axis=0).astype(np.float32),
        matched_stems,
    )


def estimate_scale_from_random_baselines(
    pr_corr: np.ndarray,
    gt_corr: np.ndarray,
    seed: int,
    max_pairs: int = 20000,
) -> Tuple[float, int, bool]:
    """Estimate scale from matched points using random pairwise baseline ratios.

    The estimate is invariant to translation and rotation, but the transform we apply
    later is scale+translation only: X_aligned = scale * X_pred + t.
    """
    pr = np.asarray(pr_corr, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt_corr, dtype=np.float64).reshape(-1, 3)
    n = min(pr.shape[0], gt.shape[0])
    if n < 2:
        return 1.0, 0, False

    rng = np.random.default_rng(seed)
    if n == 2:
        i = np.asarray([0], dtype=np.int64)
        j = np.asarray([1], dtype=np.int64)
    else:
        num_pairs = min(int(max_pairs), n * (n - 1) // 2)
        i = rng.integers(0, n, size=num_pairs, endpoint=False)
        j = rng.integers(0, n, size=num_pairs, endpoint=False)
        keep = i != j
        i = i[keep]
        j = j[keep]

    if i.size == 0:
        return 1.0, 0, False

    d_pr = np.linalg.norm(pr[i] - pr[j], axis=1)
    d_gt = np.linalg.norm(gt[i] - gt[j], axis=1)
    valid = np.isfinite(d_pr) & np.isfinite(d_gt) & (d_pr > 1e-8) & (d_gt > 0)
    if not valid.any():
        return 1.0, 0, False

    ratios = d_gt[valid] / d_pr[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios > 1e-12)]
    if ratios.size == 0:
        return 1.0, 0, False
    return float(np.median(ratios)), int(ratios.size), True


def estimate_scale_translation(
    pr_corr: np.ndarray,
    gt_corr: np.ndarray,
    seed: int,
    source: str,
) -> Tuple[float, np.ndarray, bool, str, int]:
    """Estimate pred->GT scale and translation, without rotation."""
    pr = np.asarray(pr_corr, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt_corr, dtype=np.float64).reshape(-1, 3)
    if pr.shape[0] == 0 or gt.shape[0] != pr.shape[0]:
        return 1.0, np.zeros(3, dtype=np.float32), False, f"no matched {source} correspondences", 0

    scale, num_scale_pairs, scale_valid = estimate_scale_from_random_baselines(
        pr_corr=pr,
        gt_corr=gt,
        seed=seed,
    )
    if not scale_valid:
        scale = 1.0
        note = f"{source} translation-only fallback; not enough non-zero baselines to estimate scale"
    else:
        note = f"{source} scale+translation alignment using median pairwise baseline ratios"

    t = np.median(gt - float(scale) * pr, axis=0).astype(np.float32)
    valid = np.isfinite(scale) and scale > 1e-12 and np.isfinite(t).all()
    return float(scale), t, bool(valid), note, int(num_scale_pairs)


def apply_scale_translation_to_points(points: np.ndarray, scale: float, t: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return points
    out = float(scale) * points.astype(np.float64) + np.asarray(t, dtype=np.float64)[None, :]
    return out.astype(np.float32)


def apply_scale_translation_to_point_maps(point_maps: Sequence[np.ndarray], scale: float, t: np.ndarray) -> List[np.ndarray]:
    out = []
    for m in point_maps:
        shape = m.shape
        if m.size == 0:
            out.append(m)
            continue
        out.append(apply_scale_translation_to_points(m.reshape(-1, 3), scale, t).reshape(shape))
    return out


def apply_scale_translation_to_cameras(cams: Sequence[Dict[str, object]], scale: float, t: np.ndarray) -> List[Dict[str, object]]:
    t = np.asarray(t, dtype=np.float64)
    aligned = []
    for cam in cams:
        T = np.asarray(cam["T_c2w"], dtype=np.float64)
        T_al = np.eye(4, dtype=np.float32)
        # Scale alignment does not estimate/apply rotation.
        T_al[:3, :3] = T[:3, :3].astype(np.float32)
        T_al[:3, 3] = (float(scale) * T[:3, 3] + t).astype(np.float32)
        aligned.append({**cam, "T_c2w": T_al})
    return aligned


def estimate_and_apply_alignment(
    args: argparse.Namespace,
    meta: Dict[str, object],
    pred_points: np.ndarray,
    pred_colors: np.ndarray,
    pred_maps: Sequence[np.ndarray],
    pred_valid_masks: Sequence[np.ndarray],
    pred_cams: Sequence[Dict[str, object]],
):
    mode = str(args.align).lower()
    identity = {
        "mode": mode,
        "valid": False,
        "source": "none",
        "num_corr": 0,
        "num_point_corr": 0,
        "num_camera_corr": 0,
        "num_scale_pairs": 0,
        "scale": 1.0,
        "R": np.eye(3, dtype=np.float32).tolist(),
        "t": np.zeros(3, dtype=np.float32).tolist(),
        "note": "no alignment applied",
    }
    if mode == "none":
        identity["valid"] = True
        return pred_points, pred_colors, list(pred_maps), list(pred_cams), identity
    if mode != "scale":
        raise ValueError(f"Unknown align mode: {mode}. Supported modes are: none, scale")

    # 1) Preferred path: dense predicted point map -> GT RGB-D point map correspondences.
    gt_corr, pr_corr = sample_alignment_correspondences(
        gt_maps=meta["gt_maps"],
        pred_maps=pred_maps,
        gt_valid_masks=meta["valid_masks"],
        pred_valid_masks=pred_valid_masks,
        max_samples_per_view=args.align_max_samples_per_view,
        seed=args.seed + 12345,
    )
    source = "points"
    identity["num_point_corr"] = int(gt_corr.shape[0])

    # 2) Fallback path: when there is no depth-derived GT point cloud, use camera centers.
    # This is enough to recover global scale/placement for visualization when cams exist.
    matched_camera_stems: List[str] = []
    if gt_corr.shape[0] < int(args.align_min_corr):
        cam_gt_corr, cam_pr_corr, matched_camera_stems = sample_camera_center_correspondences(meta, pred_cams)
        identity["num_camera_corr"] = int(cam_gt_corr.shape[0])
        if cam_gt_corr.shape[0] > 0:
            print(
                f"[WARN] not enough point correspondences for scale alignment: "
                f"{gt_corr.shape[0]} < {args.align_min_corr}; falling back to camera-center scale alignment "
                f"with {cam_gt_corr.shape[0]} matched cameras"
            )
            gt_corr, pr_corr = cam_gt_corr, cam_pr_corr
            source = "camera_centers"
        else:
            identity["num_corr"] = int(gt_corr.shape[0])
            identity["note"] = (
                f"not enough point correspondences: {gt_corr.shape[0]} < {args.align_min_corr}; "
                "no matched camera centers available for scale fallback"
            )
            print(f"[WARN] {identity['note']}; using raw prediction")
            return pred_points, pred_colors, list(pred_maps), list(pred_cams), identity

    identity["source"] = source
    identity["num_corr"] = int(gt_corr.shape[0])

    min_corr = 1 if source == "camera_centers" else int(args.align_min_corr)
    if gt_corr.shape[0] < min_corr:
        identity["note"] = f"not enough {source} correspondences: {gt_corr.shape[0]} < {min_corr}"
        print(f"[WARN] {identity['note']}; using raw prediction")
        return pred_points, pred_colors, list(pred_maps), list(pred_cams), identity

    scale, t, valid, solve_note, num_scale_pairs = estimate_scale_translation(
        pr_corr=pr_corr,
        gt_corr=gt_corr,
        seed=args.seed + 54321,
        source=source,
    )

    if not valid:
        identity["note"] = f"failed to solve scale alignment from {source}: {solve_note}"
        print(f"[WARN] {identity['note']}; using raw prediction")
        return pred_points, pred_colors, list(pred_maps), list(pred_cams), identity

    pred_points_aligned = apply_scale_translation_to_points(pred_points, scale, t)
    pred_maps_aligned = apply_scale_translation_to_point_maps(pred_maps, scale, t)
    pred_cams_aligned = apply_scale_translation_to_cameras(pred_cams, scale, t)

    # Basic diagnostics on the sampled correspondences.
    pr_corr_aligned = apply_scale_translation_to_points(pr_corr, scale, t)
    residual = np.linalg.norm(pr_corr_aligned.astype(np.float64) - gt_corr.astype(np.float64), axis=1)
    median_residual = float(np.median(residual)) if residual.size else float("nan")

    align_meta = {
        "mode": mode,
        "valid": True,
        "source": source,
        "num_corr": int(gt_corr.shape[0]),
        "num_point_corr": int(identity["num_point_corr"]),
        "num_camera_corr": int(identity["num_camera_corr"]),
        "num_scale_pairs": int(num_scale_pairs),
        "matched_camera_stems": matched_camera_stems,
        "scale": float(scale),
        "R": np.eye(3, dtype=np.float32).tolist(),
        "t": np.asarray(t, dtype=np.float32).tolist(),
        "median_residual": median_residual,
        "note": f"predictions transformed by pred->GT scale+translation before logging; source={source}; {solve_note}; no rotation applied",
    }
    print(
        f"Alignment scale: valid=True, source={source}, num_corr={align_meta['num_corr']}, "
        f"num_scale_pairs={align_meta['num_scale_pairs']}, scale={align_meta['scale']:.6g}, "
        f"median_residual={median_residual:.6g}, rotation=identity"
    )
    return pred_points_aligned, pred_colors, pred_maps_aligned, pred_cams_aligned, align_meta


# -----------------------------------------------------------------------------
# Rerun logging
# -----------------------------------------------------------------------------
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


def rr_init_save_compat(app_id: str, recording_id: str, save_rrd: Path) -> None:
    try:
        rr.init(app_id, recording_id=recording_id, spawn=False)
    except TypeError:
        rr.init(app_id, spawn=False)
    rr.save(str(save_rrd))


def send_blueprint(background=(255, 255, 255), hide_grid: bool = False) -> None:
    if rrb is None:
        return
    try:
        line_grid = rrb.LineGrid3D(visible=not hide_grid)
        blueprint = rrb.Blueprint(
            rrb.Spatial3DView(origin="/world", name="Prediction Scene", background=list(background), line_grid=line_grid),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)
    except Exception as e:
        print(f"[WARN] failed to send Rerun blueprint: {e}")


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
            rr.log("world", obj() if callable(obj) else obj, static=True)
            return
        except Exception:
            continue
    rr.log("world", rr.ViewCoordinates.RDF, static=True)


def log_points(entity: str, points: np.ndarray, colors: np.ndarray, radius: float) -> None:
    if points.shape[0] == 0:
        return
    kwargs = {"positions": points.astype(np.float32), "colors": colors.astype(np.uint8)}
    if radius > 0:
        kwargs["radii"] = float(radius)
    rr.log(entity, rr.Points3D(**kwargs))


def make_camera_axes_strips(cams: Sequence[Dict[str, object]], axis_size: float, colors_xyz):
    strips: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    for cam in cams:
        T = np.asarray(cam["T_c2w"], dtype=np.float64)
        R = T[:3, :3]
        o = T[:3, 3]
        strips.extend(
            [
                np.stack([o, o + R[:, 0] * axis_size], axis=0).astype(np.float32),
                np.stack([o, o + R[:, 1] * axis_size], axis=0).astype(np.float32),
                np.stack([o, o + R[:, 2] * axis_size], axis=0).astype(np.float32),
            ]
        )
        colors.extend([np.asarray(c, dtype=np.uint8) for c in colors_xyz])
    return strips, colors


def log_camera_axes(entity: str, cams: Sequence[Dict[str, object]], axis_size: float, radius: float, colors_xyz) -> None:
    strips, colors = make_camera_axes_strips(cams, axis_size, colors_xyz)
    if not strips:
        return
    kwargs = {"strips": strips, "colors": colors}
    if radius > 0:
        kwargs["radii"] = float(radius)
    rr.log(entity, rr.LineStrips3D(**kwargs))


def log_camera_labels(entity: str, cams: Sequence[Dict[str, object]], color) -> None:
    if not cams:
        return
    centers = np.asarray([np.asarray(c["T_c2w"])[:3, 3] for c in cams], dtype=np.float32)
    labels = [str(c.get("stem", f"cam_{i:03d}")) for i, c in enumerate(cams)]
    colors = np.repeat(np.asarray([color], dtype=np.uint8), len(cams), axis=0)
    try:
        rr.log(entity, rr.Points3D(positions=centers, colors=colors, labels=labels, radii=0.0))
    except TypeError:
        rr.log(entity, rr.Points3D(positions=centers, colors=colors, labels=labels))


def estimate_axis_size(point_arrays: Sequence[np.ndarray], explicit: float) -> float:
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
        origin = bbox_center + parse_signed_axis(up_axis) * size * float(up_offset_ratio)
    else:
        raise ValueError("world_axes_origin must be 'zero' or 'scene_center'")

    x_end = origin + np.array([size, 0.0, 0.0], dtype=np.float64)
    y_end = origin + np.array([0.0, size, 0.0], dtype=np.float64)
    z_end = origin + np.array([0.0, 0.0, size], dtype=np.float64)
    strips = [
        np.stack([origin, x_end]).astype(np.float32),
        np.stack([origin, y_end]).astype(np.float32),
        np.stack([origin, z_end]).astype(np.float32),
    ]
    colors = [
        np.array([255, 0, 0], dtype=np.uint8),
        np.array([0, 220, 0], dtype=np.uint8),
        np.array([40, 80, 255], dtype=np.uint8),
    ]
    kwargs = {"strips": strips, "colors": colors, "labels": ["world +X", "world +Y", "world +Z"]}
    if radius > 0:
        kwargs["radii"] = float(radius)
    rr.log("world/world_axes", rr.LineStrips3D(**kwargs))
    rr.log(
        "world/world_axes/labels",
        rr.Points3D(
            positions=np.stack([origin, x_end, y_end, z_end]).astype(np.float32),
            colors=np.asarray([[20, 20, 20], [255, 0, 0], [0, 220, 0], [40, 80, 255]], dtype=np.uint8),
            labels=["world axes", "+X", "+Y", "+Z"],
        ),
    )


def log_input_images(rgbs: Sequence[np.ndarray], stems: Sequence[str]) -> None:
    for i, (rgb, stem) in enumerate(zip(rgbs, stems)):
        rr.log(f"inputs/view_{i:03d}_{sanitize_name(stem)}/rgb", rr.Image(rgb))


def json_safe(obj):
    """Recursively convert numpy/torch values into JSON-serializable Python values."""
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


def save_rrd(
    args: argparse.Namespace,
    meta: Dict[str, object],
    gt_cams: Sequence[Dict[str, object]],
    pred_points: np.ndarray,
    pred_colors: np.ndarray,
    pred_cams: Sequence[Dict[str, object]],
    raw_pred_points: Optional[np.ndarray],
    raw_pred_colors: Optional[np.ndarray],
    raw_pred_cams: Optional[Sequence[Dict[str, object]]],
    align_meta: Dict[str, object],
) -> None:
    output_rrd = Path(args.output_rrd).expanduser().resolve()
    output_rrd.parent.mkdir(parents=True, exist_ok=True)
    scene_name = sanitize_name(Path(args.scene_dir).resolve().name)
    recording_id = f"predict_{scene_name}_{sanitize_name(args.model)}_{sanitize_name(args.align)}"

    rr_init_save_compat("predict_scene_to_rrd", recording_id, output_rrd)
    rr_set_time_compat("frame", 0)
    log_view_coordinates(args.view_coordinates)
    send_blueprint(background=tuple(args.background), hide_grid=args.hide_grid)

    gt_points, gt_colors = sample_points_and_colors(meta["gt_points"], meta["gt_colors"], args.max_gt_points, args.seed)
    pred_points, pred_colors = sample_points_and_colors(pred_points, pred_colors, args.max_pred_points, args.seed + 17)

    log_points("world/gt/points", gt_points, gt_colors, args.point_radius)
    pred_entity = "world/pred_aligned/points" if args.align != "none" else "world/pred/points"
    log_points(pred_entity, pred_points, pred_colors, args.point_radius)

    if args.log_raw_when_aligned and args.align != "none" and raw_pred_points is not None and raw_pred_colors is not None:
        raw_pts, raw_cols = sample_points_and_colors(raw_pred_points, raw_pred_colors, args.max_pred_points, args.seed + 31)
        log_points("world/pred_raw/points", raw_pts, raw_cols, args.point_radius)
    else:
        raw_pts = None

    axis_size = estimate_axis_size([gt_points, pred_points], args.camera_axis_size)
    gt_axis_colors = ((255, 0, 0), (0, 220, 0), (40, 80, 255))
    pred_axis_colors = ((255, 0, 255), (255, 180, 0), (0, 220, 255))
    raw_axis_colors = ((150, 150, 150), (180, 180, 180), (210, 210, 210))

    log_camera_axes("world/gt/cameras/axes", gt_cams, axis_size, args.camera_axis_radius, gt_axis_colors)
    pred_cam_entity = "world/pred_aligned/cameras/axes" if args.align != "none" else "world/pred/cameras/axes"
    pred_label_entity = "world/pred_aligned/cameras/labels" if args.align != "none" else "world/pred/cameras/labels"
    log_camera_axes(pred_cam_entity, pred_cams, axis_size, args.camera_axis_radius, pred_axis_colors)
    log_camera_labels("world/gt/cameras/labels", gt_cams, (60, 200, 120))
    log_camera_labels(pred_label_entity, pred_cams, (255, 120, 40))

    if args.log_raw_when_aligned and args.align != "none" and raw_pred_cams is not None:
        log_camera_axes("world/pred_raw/cameras/axes", raw_pred_cams, axis_size, args.camera_axis_radius, raw_axis_colors)
        log_camera_labels("world/pred_raw/cameras/labels", raw_pred_cams, (180, 180, 180))

    if args.show_world_axes:
        bbox_points = gt_points if gt_points.shape[0] > 0 else pred_points
        log_world_axes_marker(
            bbox_points,
            origin_mode=args.world_axes_origin,
            axis_size=args.world_axis_size,
            axis_size_ratio=args.world_axis_size_ratio,
            min_axis_size=args.world_axis_min_size,
            up_axis=args.world_up_axis,
            up_offset_ratio=args.world_axis_up_offset_ratio,
            radius=args.world_axis_radius,
        )

    if args.log_images:
        log_input_images(meta["rgbs"], meta["stems"])

    rr_disconnect_compat()
    print(f"Saved Rerun recording: {output_rrd}")

    sidecar = output_rrd.with_suffix(".json")
    payload = {
        "scene_dir": str(Path(args.scene_dir).resolve()),
        "model": args.model,
        "output_rrd": str(output_rrd),
        "stems": list(meta["stems"]),
        "target_size": {"height": int(meta["target_h"]), "width": int(meta["target_w"])},
        "num_gt_points_logged": int(gt_points.shape[0]),
        "num_pred_points_logged": int(pred_points.shape[0]),
        "num_raw_pred_points_logged": int(raw_pts.shape[0]) if raw_pts is not None else 0,
        "num_gt_cameras": int(len(gt_cams)),
        "num_pred_cameras": int(len(pred_cams)),
        "num_cam_priors": int(meta.get("num_cam_priors", 0)),
        "num_depth_priors": int(meta.get("num_depth_priors", 0)),
        "num_gt_rgbd_priors": int(meta.get("num_gt_rgbd_priors", 0)),
        "alignment": align_meta,
        "prediction_note": (
            "Predictions are logged raw when align=none; otherwise world/pred_aligned contains pred->GT aligned geometry. "
            "Use --log_raw_when_aligned to also save world/pred_raw."
        ),
    }
    sidecar.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved sidecar metadata: {sidecar}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_dir", required=True, help="Scene folder containing images and optional cams/depth.")
    parser.add_argument("--output_rrd", required=True, help="Output .rrd path.")
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Hydra model config name, e.g. pi3, vggt, geoff3d."
        ),
    )
    parser.add_argument("--machine", default="default", help="Hydra machine config name.")
    parser.add_argument(
        "--hydra_override",
        action="append",
        default=[],
        help="Extra Hydra override. Repeatable, e.g. --hydra_override model.model_config.load_pretrained_weights=false",
    )
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override loaded after model init.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or numeric CUDA index.")

    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--cams_dir", default="cams")
    parser.add_argument("--depth_dir", default="depth")
    parser.add_argument("--frame_glob", default="*")
    parser.add_argument("--num_views", type=int, default=8, help="Select a continuous window of at most this many views after --frame_glob/--start/--stride; <=0 means all.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_side", type=int, default=518)
    parser.add_argument("--size_multiple", type=int, default=14)
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--depth_min", type=float, default=1e-6)
    parser.add_argument("--depth_max", type=float, default=1e6)
    parser.add_argument("--pred_min_depth", type=float, default=1e-6)
    parser.add_argument("--conf_quantile", type=float, default=0.0, help="Drop predicted points below this confidence quantile; 0 disables.")

    parser.add_argument("--align", choices=["none", "scale"], default="none")
    parser.add_argument("--align_max_samples_per_view", type=int, default=4096)
    parser.add_argument("--align_min_corr", type=int, default=64)
    parser.add_argument("--log_raw_when_aligned", action="store_true")

    parser.add_argument("--max_gt_points", type=int, default=800000)
    parser.add_argument("--max_pred_points", type=int, default=800000)
    parser.add_argument("--point_radius", type=float, default=0.0)

    parser.add_argument("--view_coordinates", default="RDF", help="Rerun ViewCoordinates name, e.g. RDF or RIGHT_HAND_Z_UP.")
    parser.add_argument("--background", type=int, nargs=3, default=[255, 255, 255])
    parser.add_argument("--hide_grid", action="store_true")
    parser.add_argument("--log_images", action="store_true")

    parser.add_argument("--camera_axis_size", type=float, default=0.0)
    parser.add_argument("--camera_axis_radius", type=float, default=0.0)
    parser.add_argument("--show_world_axes", action="store_true", default=True)
    parser.add_argument("--no_world_axes", action="store_false", dest="show_world_axes")
    parser.add_argument("--world_axes_origin", choices=["zero", "scene_center"], default="scene_center")
    parser.add_argument("--world_up_axis", default="z")
    parser.add_argument("--world_axis_size", type=float, default=0.0)
    parser.add_argument("--world_axis_size_ratio", type=float, default=0.12)
    parser.add_argument("--world_axis_min_size", type=float, default=0.1)
    parser.add_argument("--world_axis_up_offset_ratio", type=float, default=1.2)
    parser.add_argument("--world_axis_radius", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    views, meta = build_views_from_scene(args, device=device)
    gt_cams = [
        {"stem": stem, "T_c2w": np.asarray(meta["cams"][stem]["T_c2w"], dtype=np.float32)}
        for stem in meta["stems"]
        if stem in meta["cams"]
    ]

    model, _ = init_model_from_hydra(
        model_name=args.model,
        machine=args.machine,
        hydra_overrides=args.hydra_override,
        device=device,
    )
    load_optional_checkpoint(model, args.checkpoint)
    model.eval()

    print(f"Running model={args.model} on {len(views)} views ...")
    preds = model(views)

    raw_pred_points, raw_pred_colors, pred_maps, pred_valid_masks, raw_pred_cams = collect_pred_outputs(
        preds=preds,
        rgbs=meta["rgbs"],
        args=args,
        stems=meta["stems"],
    )
    print(f"Raw prediction summary: points={raw_pred_points.shape[0]}, cameras={len(raw_pred_cams)}")

    pred_points, pred_colors, pred_maps_aligned, pred_cams, align_meta = estimate_and_apply_alignment(
        args=args,
        meta=meta,
        pred_points=raw_pred_points,
        pred_colors=raw_pred_colors,
        pred_maps=pred_maps,
        pred_valid_masks=pred_valid_masks,
        pred_cams=raw_pred_cams,
    )
    print(f"Logged prediction summary: points={pred_points.shape[0]}, cameras={len(pred_cams)}, align={align_meta['mode']}")

    save_rrd(
        args=args,
        meta=meta,
        gt_cams=gt_cams,
        pred_points=pred_points,
        pred_colors=pred_colors,
        pred_cams=pred_cams,
        raw_pred_points=raw_pred_points,
        raw_pred_colors=raw_pred_colors,
        raw_pred_cams=raw_pred_cams,
        align_meta=align_meta,
    )


if __name__ == "__main__":
    main()
