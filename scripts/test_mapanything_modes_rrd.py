#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-scene multi-model inference + Rerun export.

Fixed input format:
    images/*.png
    cams/*.txt
        OpenCV RDF world2camera extrinsic
        3x3 intrinsic
        h w hfov
    depth/*.exr
        z-depth, same order/stem as images

Modes:
    images_only:
        image only
    csfm:
        image + intrinsics prior
    psfm:
        image + pose prior
    mvs:
        image + intrinsics + pose prior
    two_pass_psfm:
        first pass: pose prior
        estimate focal correction from GT depth pointmap
        second pass: pose + corrected intrinsics prior
    all:
        run all modes

This script is intentionally written in the benchmark.py style:
    - initialize model from configs/model/*.yaml via init_model_from_config
    - run inference through model(views)
    - normalize different model outputs before RRD export
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Any

# Must be set before importing cv2.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from mapanything.utils.torch_hub_setup import configure_torch_hub

configure_torch_hub(
    SimpleNamespace(
        torch_hub_disable_download=True,
        torch_hub_dir="./checkpoints/torch_cache/hub",
        local_dino_repo="./checkpoints/torch_cache/hub/facebookresearch_dinov2_main",
    )
)

from mapanything.models import init_model_from_config
from mapanything.utils.image import preprocess_inputs, rgb
from mapanything.utils.geometry import recover_pinhole_intrinsics_from_ray_directions
from benchmarking.dense_n_view.benchmark import save_repro_bundle_rrd


try:
    from mapanything.models.external.vggt.utils.rotation import quat_to_mat as _quat_to_mat
except Exception:
    _quat_to_mat = None


# -----------------------------------------------------------------------------
# Basic IO
# -----------------------------------------------------------------------------


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def sorted_files(root: Path, suffix: str) -> List[Path]:
    paths = sorted(root.glob(f"*{suffix}"), key=lambda p: natural_key(p.name))
    if not paths:
        raise FileNotFoundError(f"No {suffix} files found in {root}")
    return paths


def sample_paths(paths: List[Path], stride: int, max_views: int) -> List[Path]:
    if stride > 1:
        paths = paths[::stride]

    if max_views > 0 and len(paths) > max_views:
        paths = paths[:max_views]

    return paths


def next_nonempty_line(lines: List[str], start_idx: int, path: Path) -> Tuple[int, str]:
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if line:
            return i, line

    raise ValueError(f"Unexpected end of cam file while parsing {path}")


def parse_float_row(line: str, expected_cols: int, path: Path, line_idx: int) -> List[float]:
    parts = line.split()
    if len(parts) != expected_cols:
        raise ValueError(
            f"{path}:{line_idx + 1} expected {expected_cols} numeric values, "
            f"got {len(parts)}: {line!r}"
        )

    try:
        return [float(x) for x in parts]
    except ValueError as exc:
        raise ValueError(
            f"{path}:{line_idx + 1} contains non-numeric values: {line!r}"
        ) from exc


def find_header(lines_lower: List[str], prefix: str, path: Path) -> int:
    for i, line in enumerate(lines_lower):
        if line.startswith(prefix):
            return i

    raise ValueError(f"{path} missing header starting with {prefix!r}")


def read_cam_txt(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, float]]:
    """
    Read fixed cam txt format.

    Required format:

        extrinsic
        opencv(x Right, y Down, z Forward) world2camera
        r11 r12 r13 t1
        r21 r22 r23 t2
        r31 r32 r33 t3
        0   0   0   1

        intrinsic:
        fx fy cx cy (pixel)
        fx 0  cx
        0  fy cy
        0  0  1

        h w hfov
        H W HFOV

    The extrinsic block is OpenCV RDF world-to-camera.
    This function returns camera-to-world pose.
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines_lower = [line.strip().lower() for line in lines]

    ext_idx = find_header(lines_lower, "extrinsic", path)
    int_idx = find_header(lines_lower, "intrinsic", path)
    meta_idx = find_header(lines_lower, "h w hfov", path)

    ext_rows: List[List[float]] = []
    cursor = ext_idx + 1
    for _ in range(4):
        line_idx, line = next_nonempty_line(lines, cursor, path)
        ext_rows.append(parse_float_row(line, 4, path, line_idx))
        cursor = line_idx + 1

    E_w2c = np.asarray(ext_rows, dtype=np.float32)

    int_rows: List[List[float]] = []
    cursor = int_idx + 1
    for _ in range(3):
        line_idx, line = next_nonempty_line(lines, cursor, path)
        int_rows.append(parse_float_row(line, 3, path, line_idx))
        cursor = line_idx + 1

    K = np.asarray(int_rows, dtype=np.float32)

    line_idx, meta_line = next_nonempty_line(lines, meta_idx + 1, path)
    meta_vals = parse_float_row(meta_line, 3, path, line_idx)

    H = int(round(meta_vals[0]))
    W = int(round(meta_vals[1]))
    hfov = float(meta_vals[2])

    T_c2w = np.linalg.inv(E_w2c).astype(np.float32)
    return K, T_c2w, (H, W, hfov)


def read_exr_z_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to read EXR depth: {path}")

    if depth.ndim == 3:
        depth = depth[..., 0]

    return depth.astype(np.float32)


def load_scene(
    image_dir: Path,
    cams_dir: Path,
    depth_dir: Path,
    stride: int,
    max_views: int,
) -> List[Dict]:
    image_paths = [p for p in sorted_files(image_dir, ".png") if "mask" not in p.stem]
    cam_paths = sorted_files(cams_dir, ".txt")
    depth_paths = sorted_files(depth_dir, ".exr")

    if not (len(image_paths) == len(cam_paths) == len(depth_paths)):
        raise ValueError(
            f"images/cams/depth count mismatch: "
            f"{len(image_paths)} png, {len(cam_paths)} txt, {len(depth_paths)} exr"
        )

    image_paths = sample_paths(image_paths, stride, max_views)
    cam_paths = sample_paths(cam_paths, stride, max_views)
    depth_paths = sample_paths(depth_paths, stride, max_views)

    views: List[Dict] = []
    for idx, (img_path, cam_path, depth_path) in enumerate(
        zip(image_paths, cam_paths, depth_paths)
    ):
        K, T_c2w, cam_meta = read_cam_txt(cam_path)
        img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        depth_z = read_exr_z_depth(depth_path)

        H_cam, W_cam, hfov = cam_meta

        if img.size != (W_cam, H_cam):
            print(
                f"[warn] {img_path.name}: image size {img.size} "
                f"!= cam h/w {(W_cam, H_cam)}"
            )

        if depth_z.shape[:2] != (H_cam, W_cam):
            print(
                f"[warn] {depth_path.name}: depth shape {depth_z.shape[:2]} "
                f"!= cam h/w {(H_cam, W_cam)}"
            )

        views.append(
            {
                "img": img,
                "intrinsics": K,
                "camera_poses": T_c2w,
                "depth_z": depth_z,
                "is_metric_scale": torch.tensor([True], dtype=torch.bool),
                "idx": idx,
                "instance": img_path.stem,
                "hfov": hfov,
            }
        )

    return views


def strip_depth(views: List[Dict]) -> List[Dict]:
    out = []
    for v in views:
        d = dict(v)
        d.pop("depth_z", None)
        out.append(d)

    return out


# -----------------------------------------------------------------------------
# Mode config / model compatibility
# -----------------------------------------------------------------------------


MODE_GEOMETRIC_CONFIG = {
    "images_only": {
        "overall_prob": 0.0,
        "dropout_prob": 1.0,
        "ray_dirs_prob": 0.0,
        "depth_prob": 0.0,
        "cam_prob": 0.0,
        "sparse_depth_prob": 0.0,
        "sparsification_removal_percent": 0.0,
    },
    "csfm": {
        "overall_prob": 1.0,
        "dropout_prob": 0.0,
        "ray_dirs_prob": 1.0,
        "depth_prob": 0.0,
        "cam_prob": 0.0,
        "sparse_depth_prob": 0.0,
        "sparsification_removal_percent": 0.0,
    },
    "psfm": {
        "overall_prob": 1.0,
        "dropout_prob": 0.0,
        "ray_dirs_prob": 0.0,
        "depth_prob": 0.0,
        "cam_prob": 1.0,
        "sparse_depth_prob": 0.0,
        "sparsification_removal_percent": 0.0,
    },
    "mvs": {
        "overall_prob": 1.0,
        "dropout_prob": 0.0,
        "ray_dirs_prob": 1.0,
        "depth_prob": 0.0,
        "cam_prob": 1.0,
        "sparse_depth_prob": 0.0,
        "sparsification_removal_percent": 0.0,
    },
    "two_pass_psfm_first": {
        "overall_prob": 1.0,
        "dropout_prob": 0.0,
        "ray_dirs_prob": 0.0,
        "depth_prob": 0.0,
        "cam_prob": 1.0,
        "sparse_depth_prob": 0.0,
        "sparsification_removal_percent": 0.0,
    },
    "two_pass_psfm_second": {
        "overall_prob": 1.0,
        "dropout_prob": 0.0,
        "ray_dirs_prob": 1.0,
        "depth_prob": 0.0,
        "cam_prob": 1.0,
        "sparse_depth_prob": 0.0,
        "sparsification_removal_percent": 0.0,
    },
}


def is_identity_norm_model(model_config_name: str) -> bool:
    name = str(model_config_name).lower()
    return name in {"pi3", "pi3x"}


def configure_model_for_mode(model: torch.nn.Module, mode: str) -> None:
    """
    Configure prior usage for models that expose geometric_input_config.

    This is intentionally recursive because some wrappers keep the config on the
    root module, while some MapAnything variants may keep similar config on
    submodules.
    """
    if mode not in MODE_GEOMETRIC_CONFIG:
        return

    patch = MODE_GEOMETRIC_CONFIG[mode]
    visited = set()

    def _patch_module(m: torch.nn.Module):
        obj_id = id(m)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if hasattr(m, "geometric_input_config"):
            try:
                cfg = dict(getattr(m, "geometric_input_config"))
                cfg.update(patch)
                setattr(m, "geometric_input_config", cfg)
            except Exception as exc:
                print(
                    f"[warn] failed to patch geometric_input_config on "
                    f"{m.__class__.__name__}: {exc}"
                )

        for child in m.children():
            _patch_module(child)

    _patch_module(model)


def select_mode_inputs(processed_views: List[Dict], mode: str) -> List[Dict]:
    """
    Create model input views for a specific inference mode.

    We keep both key conventions:
        MapAnything / benchmark style:
            intrinsics, camera_poses
        Pi3X / some external wrappers:
            camera_intrinsics, camera_pose
    """
    common_keys = {
        "img",
        "data_norm_type",
        "true_shape",
        "idx",
        "instance",
        "is_metric_scale",
    }

    use_calib = mode in {"csfm", "mvs", "two_pass_psfm_second"}
    use_pose = mode in {
        "psfm",
        "mvs",
        "two_pass_psfm_first",
        "two_pass_psfm_second",
    }

    out: List[Dict] = []

    for v in processed_views:
        d = {k: v[k] for k in common_keys if k in v}

        if use_calib:
            d["intrinsics"] = v["intrinsics"]
            d["camera_intrinsics"] = v["intrinsics"]

        if use_pose:
            d["camera_poses"] = v["camera_poses"]
            d["camera_pose"] = v["camera_poses"]

        out.append(d)

    return out


def move_views_to_device(views: List[Dict], device: torch.device) -> List[Dict]:
    keep_cpu = {
        "data_norm_type",
        "idx",
        "instance",
        "true_shape",
        "hfov",
    }

    moved = []
    for v in views:
        d = {}
        for k, val in v.items():
            if k in keep_cpu:
                d[k] = val
            elif isinstance(val, torch.Tensor):
                d[k] = val.to(device, non_blocking=True)
            else:
                d[k] = val

        moved.append(d)

    return moved


def load_model(args, device: torch.device):
    model = init_model_from_config(
        args.model_config,
        device=str(device),
        machine=args.machine,
    )

    model = model.to(device)

    if args.checkpoint:
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(args.checkpoint, map_location="cpu")

        if isinstance(ckpt, dict):
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        else:
            state = ckpt

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[checkpoint] loaded {args.checkpoint}")
        print(f"[checkpoint] missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    return model


# -----------------------------------------------------------------------------
# Prediction standardization
# -----------------------------------------------------------------------------


def amp_dtype_from_arg(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32

    raise ValueError(f"Unsupported amp dtype: {name}")


def chw_to_hwc_if_needed(img: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(img) and img.ndim == 4 and img.shape[1] in (1, 3):
        return img.permute(0, 2, 3, 1).contiguous()
    return img


def quat_to_mat_safe(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.

    Prefer the repository's VGGT helper. The fallback assumes scalar-first
    quaternion layout [w, x, y, z], which matches common VGGT-style helpers.
    """
    if _quat_to_mat is not None:
        return _quat_to_mat(q.float())

    q = q.float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    w, x, y, z = q.unbind(dim=-1)

    B = q.shape[0]
    R = torch.empty(B, 3, 3, device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)

    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)

    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def pose_from_quat_trans(cam_quats: torch.Tensor, cam_trans: torch.Tensor) -> torch.Tensor:
    B = cam_trans.shape[0]
    T = (
        torch.eye(4, device=cam_trans.device, dtype=cam_trans.dtype)
        .reshape(1, 4, 4)
        .repeat(B, 1, 1)
    )

    R = quat_to_mat_safe(cam_quats).to(dtype=cam_trans.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = cam_trans

    return T


def recover_intrinsics_from_pred(pred: Dict, view: Dict) -> torch.Tensor:
    """
    Get model-predicted intrinsics if available.

    Priority:
        1. pred["intrinsics"]
        2. pred["camera_intrinsics"]
        3. recover K from pred["ray_directions"]
        4. fallback to input K, only to keep RRD export functional
    """
    if "intrinsics" in pred:
        return pred["intrinsics"]

    if "camera_intrinsics" in pred:
        return pred["camera_intrinsics"]

    if "ray_directions" in pred:
        K = recover_pinhole_intrinsics_from_ray_directions(
            pred["ray_directions"].float(),
            use_geometric_calculation=True,
        )
        return K.to(device=pred["ray_directions"].device)

    if "camera_intrinsics" in view:
        return view["camera_intrinsics"]

    if "intrinsics" in view:
        return view["intrinsics"]

    raise KeyError("Cannot find or recover intrinsics from prediction or input view.")


def add_img_no_norm(pred: Dict, view: Dict) -> None:
    if "img_no_norm" in pred:
        return

    if "img" not in view:
        return

    img = view["img"]

    try:
        img_np = rgb(img, view["data_norm_type"][0])
        img_t = torch.from_numpy(np.asarray(img_np)).to(img.device if torch.is_tensor(img) else "cpu")

        if img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)

        pred["img_no_norm"] = img_t.float()
        return
    except Exception:
        pass

    if torch.is_tensor(img):
        pred["img_no_norm"] = chw_to_hwc_if_needed(img)


def standardize_preds_for_rrd(preds: List[Dict], views: List[Dict]) -> List[Dict]:
    out: List[Dict] = []

    for pred, view in zip(preds, views):
        q = dict(pred)

        if "camera_poses" not in q:
            if "cam_quats" in q and "cam_trans" in q:
                q["camera_poses"] = pose_from_quat_trans(q["cam_quats"], q["cam_trans"])
            elif "camera_pose" in q:
                q["camera_poses"] = q["camera_pose"]
            elif "poses" in q:
                q["camera_poses"] = q["poses"]

        if "intrinsics" not in q:
            q["intrinsics"] = recover_intrinsics_from_pred(q, view)

        if "mask" not in q and "pts3d" in q:
            q["mask"] = torch.isfinite(q["pts3d"]).all(dim=-1, keepdim=True)

        add_img_no_norm(q, view)

        out.append(q)

    return out


@torch.no_grad()
def run_infer(
    model: torch.nn.Module,
    views: List[Dict],
    infer_mode: str,
    args,
    device: torch.device,
) -> List[Dict]:
    views = move_views_to_device(views, device)

    configure_model_for_mode(model, infer_mode)

    amp_dtype = amp_dtype_from_arg(args.amp_dtype)
    amp_enabled = (
        not args.no_amp
        and device.type == "cuda"
        and amp_dtype != torch.float32
    )

    with torch.autocast(
        device_type=device.type,
        enabled=amp_enabled,
        dtype=amp_dtype,
    ):
        preds = model(views)

    preds = standardize_preds_for_rrd(preds, views)
    return preds


# -----------------------------------------------------------------------------
# Alignment
# -----------------------------------------------------------------------------


def camera_centers_from_views(views: List[Dict]) -> torch.Tensor:
    return torch.cat([v["camera_poses"][:, :3, 3] for v in views], dim=0)


def camera_centers_from_preds(preds: List[Dict]) -> torch.Tensor:
    centers = []

    for p in preds:
        if "camera_poses" in p:
            centers.append(p["camera_poses"][:, :3, 3])
        elif "cam_trans" in p:
            centers.append(p["cam_trans"])
        else:
            raise KeyError("Prediction has neither camera_poses nor cam_trans.")

    return torch.cat(centers, dim=0)


def umeyama_sim3(src: torch.Tensor, dst: torch.Tensor):
    """
    Solve dst ~= s * R @ src + t.
    """
    src = src.float()
    dst = dst.float()

    valid = torch.isfinite(src).all(dim=-1) & torch.isfinite(dst).all(dim=-1)
    src = src[valid]
    dst = dst[valid]

    if src.shape[0] < 3:
        return (
            torch.tensor(1.0, device=dst.device),
            torch.eye(3, device=dst.device),
            torch.zeros(3, device=dst.device),
        )

    mu_x = src.mean(dim=0)
    mu_y = dst.mean(dim=0)

    X = src - mu_x
    Y = dst - mu_y

    cov = (Y.T @ X) / src.shape[0]

    U, S, Vh = torch.linalg.svd(cov)

    D = torch.eye(3, device=src.device, dtype=src.dtype)
    if torch.linalg.det(U @ Vh) < 0:
        D[-1, -1] = -1

    R = U @ D @ Vh

    var_x = (X * X).sum() / src.shape[0]
    s = torch.sum(S * torch.diag(D)) / torch.clamp(var_x, min=1e-8)

    t = mu_y - s * (R @ mu_x)

    return s, R, t


def apply_sim3_to_points(
    points: torch.Tensor,
    s: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    return s * torch.einsum("ij,...j->...i", R, points) + t


def apply_sim3_to_preds(
    preds: List[Dict],
    s: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> List[Dict]:
    out = []

    for p in preds:
        q = dict(p)

        if "pts3d" in q:
            q["pts3d"] = apply_sim3_to_points(q["pts3d"], s, R, t)

        if "camera_poses" in q:
            T = q["camera_poses"].clone()
            T[:, :3, 3] = apply_sim3_to_points(T[:, :3, 3], s, R, t)
            T[:, :3, :3] = torch.einsum("ij,bjk->bik", R, T[:, :3, :3])
            q["camera_poses"] = T
            q["cam_trans"] = T[:, :3, 3]
        elif "cam_trans" in q:
            q["cam_trans"] = apply_sim3_to_points(q["cam_trans"], s, R, t)

        out.append(q)

    return out


def align_preds_to_gt_pose(
    preds: List[Dict],
    gt_views: List[Dict],
) -> Tuple[List[Dict], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if "pts3d" in preds[0]:
        device = preds[0]["pts3d"].device
    else:
        device = camera_centers_from_preds(preds).device

    gt_centers = camera_centers_from_views(move_views_to_device(gt_views, device))
    pred_centers = camera_centers_from_preds(preds)

    s, R, t = umeyama_sim3(pred_centers, gt_centers)
    return apply_sim3_to_preds(preds, s, R, t), (s, R, t)


# -----------------------------------------------------------------------------
# GT pointmaps and two-pass focal correction
# -----------------------------------------------------------------------------


def build_gt_pointmaps(processed_views: List[Dict], device: torch.device):
    pointmaps, masks = [], []

    for v in move_views_to_device(processed_views, device):
        K = v["intrinsics"].float()
        T = v["camera_poses"].float()
        depth = v["depth_z"].float()

        if depth.ndim == 2:
            depth = depth[None]

        if depth.ndim == 4:
            depth = depth[..., 0]

        B, H, W = depth.shape

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        pix = (
            torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
            .reshape(1, H, W, 3)
            .repeat(B, 1, 1, 1)
        )

        rays = torch.einsum("bij,bhwj->bhwi", torch.linalg.inv(K), pix)
        pts_cam = rays * depth[..., None]
        pts_world = (
            torch.einsum("bij,bhwj->bhwi", T[:, :3, :3], pts_cam)
            + T[:, None, None, :3, 3]
        )

        valid = (
            torch.isfinite(pts_world).all(dim=-1, keepdim=True)
            & torch.isfinite(depth[..., None])
            & (depth[..., None] > 1e-6)
        )

        pointmaps.append(pts_world)
        masks.append(valid)

    return pointmaps, masks


def focal_alpha_from_gt_pointmaps(
    preds_aligned: List[Dict],
    processed_views: List[Dict],
    gt_pointmaps: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    min_points: int,
    global_alpha: bool,
) -> List[float]:
    alphas = []
    all_ratios = []

    device = preds_aligned[0]["pts3d"].device
    gt_views_dev = move_views_to_device(processed_views, device)

    for pred, view, gt_pts, gt_mask in zip(
        preds_aligned,
        gt_views_dev,
        gt_pointmaps,
        gt_masks,
    ):
        pred_pts = pred["pts3d"].detach().float()
        gt_pts = gt_pts.detach().float().to(device)
        gt_mask = gt_mask.detach().bool().to(device)

        mask = (
            gt_mask
            & torch.isfinite(pred_pts).all(dim=-1, keepdim=True)
            & torch.isfinite(gt_pts).all(dim=-1, keepdim=True)
        )

        if "mask" in pred:
            mask = mask & pred["mask"].detach().bool()

        if int(mask.sum()) < min_points:
            alphas.append(1.0)
            continue

        T = view["camera_poses"].float()
        Rcw = T[:, :3, :3].transpose(1, 2)
        tcw = T[:, :3, 3]

        pred_cam = torch.einsum(
            "bij,bhwj->bhwi",
            Rcw,
            pred_pts - tcw[:, None, None, :],
        )
        gt_cam = torch.einsum(
            "bij,bhwj->bhwi",
            Rcw,
            gt_pts - tcw[:, None, None, :],
        )

        pred_lat = torch.linalg.norm(pred_cam[..., :2], dim=-1)
        gt_lat = torch.linalg.norm(gt_cam[..., :2], dim=-1)

        valid = (
            mask[..., 0]
            & (gt_lat > 1e-6)
            & torch.isfinite(pred_lat)
            & torch.isfinite(gt_lat)
        )

        ratios = (pred_lat[valid] / gt_lat[valid]).detach().cpu().numpy()

        if ratios.size < min_points:
            alpha = 1.0
        else:
            lo, hi = np.percentile(ratios, [5, 95])
            ratios = ratios[(ratios >= lo) & (ratios <= hi)]
            alpha = float(np.median(ratios)) if ratios.size else 1.0

        all_ratios.extend(ratios.tolist())
        alphas.append(alpha)

    if global_alpha:
        if all_ratios:
            alpha = float(np.median(all_ratios))
        else:
            alpha = float(np.median(alphas))
        return [alpha for _ in alphas]

    return alphas


def make_corrected_intrinsics(
    preds_first: List[Dict],
    alphas: List[float],
    clip_min: float,
    clip_max: float,
):
    corrected = []

    for pred, alpha in zip(preds_first, alphas):
        K = pred["intrinsics"].detach().clone().float().cpu()
        a = float(np.clip(alpha, clip_min, clip_max))

        K[:, 0, 0] *= a
        K[:, 1, 1] *= a

        corrected.append(K)

    return corrected


# -----------------------------------------------------------------------------
# RRD collection
# -----------------------------------------------------------------------------


def tensor_to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def collect_pred_points(preds: List[Dict], max_points: int):
    pts_all, colors_all = [], []

    for p in preds:
        if "pts3d" not in p:
            continue

        pts = tensor_to_numpy(p["pts3d"])[0].reshape(-1, 3)

        if "img_no_norm" in p:
            colors = tensor_to_numpy(p["img_no_norm"])[0].reshape(-1, 3)
            if colors.max() <= 1.5:
                colors = colors * 255.0
        else:
            colors = np.full_like(pts, 180.0)

        valid = np.isfinite(pts).all(axis=1)

        if "mask" in p:
            valid = valid & (tensor_to_numpy(p["mask"])[0].reshape(-1) > 0)

        pts_all.append(pts[valid].astype(np.float32))
        colors_all.append(np.clip(colors[valid], 0, 255).astype(np.uint8))

    pts = np.concatenate(pts_all, axis=0) if pts_all else np.zeros((0, 3), np.float32)
    colors = (
        np.concatenate(colors_all, axis=0)
        if colors_all
        else np.zeros((0, 3), np.uint8)
    )

    if max_points > 0 and pts.shape[0] > max_points:
        ids = np.random.default_rng(0).choice(pts.shape[0], max_points, replace=False)
        pts, colors = pts[ids], colors[ids]

    return pts, colors


def collect_gt_points(
    gt_pointmaps: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    processed_views: List[Dict],
    max_points: int,
):
    pts_all, colors_all = [], []

    for pts_t, mask_t, view in zip(gt_pointmaps, gt_masks, processed_views):
        pts = tensor_to_numpy(pts_t)[0].reshape(-1, 3)
        mask = tensor_to_numpy(mask_t)[0].reshape(-1) > 0

        img = rgb(view["img"], view["data_norm_type"][0])
        img = np.asarray(img)

        if img.ndim == 4:
            img = img[0]

        colors = np.clip(img.reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)

        valid = mask & np.isfinite(pts).all(axis=1)

        pts_all.append(pts[valid].astype(np.float32))
        colors_all.append(colors[valid])

    pts = np.concatenate(pts_all, axis=0) if pts_all else np.zeros((0, 3), np.float32)
    colors = (
        np.concatenate(colors_all, axis=0)
        if colors_all
        else np.zeros((0, 3), np.uint8)
    )

    if max_points > 0 and pts.shape[0] > max_points:
        ids = np.random.default_rng(1).choice(pts.shape[0], max_points, replace=False)
        pts, colors = pts[ids], colors[ids]

    return pts, colors


def info_from_views(views: List[Dict]) -> Dict[str, List[torch.Tensor]]:
    return {
        "poses": [v["camera_poses"].detach().cpu() for v in views],
        "intrinsics": [v["intrinsics"].detach().cpu() for v in views],
    }


def info_from_preds(preds: List[Dict]) -> Dict[str, List[torch.Tensor]]:
    poses, intrinsics = [], []

    for p in preds:
        if "camera_poses" in p:
            poses.append(p["camera_poses"].detach().cpu())
        elif "cam_trans" in p:
            T = (
                torch.eye(4)
                .reshape(1, 4, 4)
                .repeat(p["cam_trans"].shape[0], 1, 1)
            )
            T[:, :3, 3] = p["cam_trans"].detach().cpu()
            poses.append(T)
        else:
            T = torch.eye(4).reshape(1, 4, 4)
            poses.append(T)

        if "intrinsics" in p:
            intrinsics.append(p["intrinsics"].detach().cpu())
        elif "camera_intrinsics" in p:
            intrinsics.append(p["camera_intrinsics"].detach().cpu())
        else:
            intrinsics.append(torch.eye(3).reshape(1, 3, 3))

    return {
        "poses": poses,
        "intrinsics": intrinsics,
    }


def save_rrd(
    out_path: Path,
    mode: str,
    scene: str,
    processed_views: List[Dict],
    preds: List[Dict],
    gt_points,
    gt_colors,
    args,
):
    pred_points, pred_colors = collect_pred_points(preds, args.max_rrd_points)

    batch_views = [{"instance": v.get("instance", "")} for v in processed_views]

    save_repro_bundle_rrd(
        rrd_path=out_path,
        benchmark_dataset_name=f"{args.model_config}_{mode}",
        scene=scene,
        set_idx=0,
        batch_views=batch_views,
        batch_idx=0,
        gt_info_abs=info_from_views(processed_views),
        pr_info_abs_aligned=info_from_preds(preds),
        gt_points=gt_points,
        gt_colors=gt_colors,
        pred_points=pred_points,
        pred_colors=pred_colors,
        background=tuple(args.background),
        hide_grid=args.hide_grid,
        collapse_panels=True,
        point_radius=args.point_radius,
        axis_size=args.axis_size,
        axis_radius=args.axis_radius,
        show_center_labels=not args.hide_center_labels,
    )

    print(f"[rrd] saved {out_path}")


# -----------------------------------------------------------------------------
# Mode runner
# -----------------------------------------------------------------------------


def run_mode(
    mode: str,
    model: torch.nn.Module,
    processed_views: List[Dict],
    gt_pointmaps: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    gt_points,
    gt_colors,
    args,
    device: torch.device,
    scene: str,
):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode in {"images_only", "csfm", "psfm", "mvs"}:
        views = select_mode_inputs(processed_views, mode)
        preds = run_infer(model, views, mode, args, device)

        if args.align_for_rrd == "pose":
            preds, sim3 = align_preds_to_gt_pose(preds, processed_views)
            print(f"[{mode}] pose-align scale={float(sim3[0].detach().cpu()):.6f}")

        save_rrd(
            out_dir / f"{scene}_{args.model_config}_{mode}.rrd",
            mode,
            scene,
            processed_views,
            preds,
            gt_points,
            gt_colors,
            args,
        )
        return

    if mode != "two_pass_psfm":
        raise ValueError(mode)

    views1 = select_mode_inputs(processed_views, "two_pass_psfm_first")
    preds1 = run_infer(model, views1, "two_pass_psfm_first", args, device)

    preds1_aligned, sim3 = align_preds_to_gt_pose(preds1, processed_views)
    print(
        f"[two_pass_psfm] first-pass pose-align "
        f"scale={float(sim3[0].detach().cpu()):.6f}"
    )

    alphas = focal_alpha_from_gt_pointmaps(
        preds1_aligned,
        processed_views,
        gt_pointmaps,
        gt_masks,
        min_points=args.two_pass_min_points,
        global_alpha=(args.two_pass_alpha_mode == "global"),
    )

    print(
        f"[two_pass_psfm] alpha mode={args.two_pass_alpha_mode}, "
        f"median={float(np.median(alphas)):.6f}, "
        f"min={float(np.min(alphas)):.6f}, "
        f"max={float(np.max(alphas)):.6f}"
    )

    corrected_K = make_corrected_intrinsics(
        preds1,
        alphas,
        clip_min=args.two_pass_alpha_clip_min,
        clip_max=args.two_pass_alpha_clip_max,
    )

    views2 = []
    for v, K in zip(processed_views, corrected_K):
        d = {
            k: v[k]
            for k in [
                "img",
                "data_norm_type",
                "true_shape",
                "idx",
                "instance",
                "is_metric_scale",
            ]
            if k in v
        }

        d["camera_poses"] = v["camera_poses"]
        d["camera_pose"] = v["camera_poses"]

        d["intrinsics"] = K
        d["camera_intrinsics"] = K

        views2.append(d)

    preds2 = run_infer(model, views2, "two_pass_psfm_second", args, device)

    if args.align_for_rrd == "pose":
        preds2, sim3_2 = align_preds_to_gt_pose(preds2, processed_views)
        print(
            f"[two_pass_psfm] second-pass pose-align "
            f"scale={float(sim3_2[0].detach().cpu()):.6f}"
        )

    save_rrd(
        out_dir / f"{scene}_{args.model_config}_two_pass_psfm.rrd",
        "two_pass_psfm",
        scene,
        processed_views,
        preds2,
        gt_points,
        gt_colors,
        args,
    )

    if args.save_first_pass_rrd:
        save_rrd(
            out_dir / f"{scene}_{args.model_config}_two_pass_psfm_first_pass.rrd",
            "two_pass_psfm_first_pass",
            scene,
            processed_views,
            preds1_aligned,
            gt_points,
            gt_colors,
            args,
        )

    summary = {
        "model_config": args.model_config,
        "mode": "two_pass_psfm",
        "alpha_mode": args.two_pass_alpha_mode,
        "alphas": [float(a) for a in alphas],
        "median_alpha": float(np.median(alphas)),
        "alpha_clip": [
            args.two_pass_alpha_clip_min,
            args.two_pass_alpha_clip_max,
        ],
    }

    json_path = out_dir / f"{scene}_{args.model_config}_two_pass_psfm_intrinsics_correction.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[json] saved {json_path}")


# -----------------------------------------------------------------------------
# Args / main
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Single-scene multi-model PNG/TXT/EXR RRD test script"
    )

    p.add_argument(
        "--image_dir",
        default="/opt/data/private/dataset/data/blendedmvs/5a77b46b318efe6c6736e68a/images",
        type=Path,
    )
    p.add_argument(
        "--cams_dir",
        default="/opt/data/private/dataset/data/blendedmvs/5a77b46b318efe6c6736e68a/cams",
        type=Path,
    )
    p.add_argument(
        "--depth_dir",
        default="/opt/data/private/dataset/data/blendedmvs/5a77b46b318efe6c6736e68a/depth",
        type=Path,
    )

    p.add_argument("--out_dir", default="output/rrd/test_scene", type=Path)

    p.add_argument(
        "--mode",
        default="psfm",
        choices=[
            "images_only",
            "csfm",
            "psfm",
            "mvs",
            "two_pass_psfm",
            "all",
        ],
    )

    p.add_argument("--scene_name", default=None)
    p.add_argument("--stride", default=1, type=int)
    p.add_argument("--max_views", default=16, type=int)

    p.add_argument("--resolution_set", default=518, type=int)
    p.add_argument("--norm_type", default="dinov2")
    p.add_argument("--patch_size", default=14, type=int)

    p.add_argument("--device", default="cuda:0")

    p.add_argument(
        "--model_config",
        default="mapanything",
        help=(
            "Hydra model config name under configs/model, "
            "for example: mapanything, pi3x, pi3, vggt, da3."
        ),
    )
    p.add_argument("--machine", default="default")

    p.add_argument(
        "--checkpoint",
        default="",
        help=(
            "Optional checkpoint to load after model construction. "
            "Leave empty for external models that load their own pretrained weights."
        ),
    )

    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    p.add_argument("--two_pass_alpha_mode", default="global", choices=["global", "per_view"])
    p.add_argument("--two_pass_min_points", default=512, type=int)
    p.add_argument("--two_pass_alpha_clip_min", default=0.5, type=float)
    p.add_argument("--two_pass_alpha_clip_max", default=2.0, type=float)
    p.add_argument("--save_first_pass_rrd", action="store_true")

    p.add_argument("--align_for_rrd", default="pose", choices=["pose", "none"])

    p.add_argument("--max_rrd_points", default=600_000, type=int)
    p.add_argument("--background", nargs=3, type=int, default=[255, 255, 255])
    p.add_argument("--hide_grid", action="store_true")
    p.add_argument("--hide_center_labels", action="store_true")
    p.add_argument("--point_radius", default=0.0, type=float)
    p.add_argument("--axis_size", default=0.0, type=float)
    p.add_argument("--axis_radius", default=0.0, type=float)

    return p.parse_args()


def main():
    args = parse_args()

    scene = args.scene_name or args.image_dir.parent.name or args.image_dir.name
    device = torch.device(args.device)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    if is_identity_norm_model(args.model_config) and args.norm_type != "identity":
        print(
            f"[norm] {args.model_config} expects identity normalization: "
            f"{args.norm_type} -> identity"
        )
        args.norm_type = "identity"

    print(f"[scene] {scene}")
    print(f"[device] {device}")
    print(f"[model_config] {args.model_config}")
    print(f"[norm_type] {args.norm_type}")

    raw_views = load_scene(
        args.image_dir,
        args.cams_dir,
        args.depth_dir,
        args.stride,
        args.max_views,
    )
    print(f"[input] loaded {len(raw_views)} views")

    processed_with_depth = preprocess_inputs(
        raw_views,
        resize_mode="fixed_mapping",
        size=None,
        norm_type=args.norm_type,
        patch_size=args.patch_size,
        resolution_set=args.resolution_set,
        verbose=True,
    )

    processed_views = strip_depth(processed_with_depth)

    gt_pointmaps, gt_masks = build_gt_pointmaps(processed_with_depth, device)
    gt_points, gt_colors = collect_gt_points(
        gt_pointmaps,
        gt_masks,
        processed_with_depth,
        args.max_rrd_points,
    )

    print(f"[gt-depth] GT point count for RRD: {gt_points.shape[0]}")

    model = load_model(args, device)

    modes = (
        ["images_only", "csfm", "psfm", "mvs", "two_pass_psfm"]
        if args.mode == "all"
        else [args.mode]
    )

    for mode in modes:
        print(f"\n========== Running mode: {mode} ==========")
        run_mode(
            mode,
            model,
            processed_views,
            gt_pointmaps,
            gt_masks,
            gt_points,
            gt_colors,
            args,
            device,
            scene,
        )


if __name__ == "__main__":
    main()
    