# -*- coding: utf-8 -*-
"""Optional gsplat refinement for spatial RRD chunks.

This module is imported only when --gsplat_refine is enabled. It initializes
3D Gaussians from predicted world point maps, renders chunk RGB images with
gsplat, and jointly optimizes Gaussian parameters and camera extrinsics.

The bundle-level optimizer follows the official gsplat trainer structure:
  - ParameterDict (means/scales/quats/opacities/sh0/shN)
  - Per-parameter Adam optimizers
  - DefaultStrategy / MCMCStrategy for densification & pruning
  - L1 + SSIM loss with optional opacity/scale/pose/mean regularization
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from gsplat.losses import (
        l1_loss,
        ssim_loss,
        opacity_reg_loss,
        scale_reg_loss,
    )
except Exception:
    l1_loss = None
    ssim_loss = None
    opacity_reg_loss = None
    scale_reg_loss = None

try:
    from gsplat.strategy import DefaultStrategy, MCMCStrategy
except Exception:
    DefaultStrategy = None
    MCMCStrategy = None

try:
    from gsplat.optimizers import SelectiveAdam
except Exception:
    SelectiveAdam = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def _skew(v: torch.Tensor) -> torch.Tensor:
    """v: [...,3] -> [...,3,3]."""
    z = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack(
        [
            torch.stack([z, -vz, vy], dim=-1),
            torch.stack([vz, z, -vx], dim=-1),
            torch.stack([-vy, vx, z], dim=-1),
        ],
        dim=-2,
    )


def so3_exp(rotvec: torch.Tensor) -> torch.Tensor:
    """Differentiable SO(3) exponential map.

    Args:
        rotvec: [...,3], axis-angle vector.

    Returns:
        R: [...,3,3]
    """
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp_min(1e-12)
    K = _skew(rotvec / theta)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    eye = eye.expand(rotvec.shape[:-1] + (3, 3))

    theta_e = theta[..., None]
    sin_t = torch.sin(theta_e)
    cos_t = torch.cos(theta_e)
    R = eye + sin_t * K + (1.0 - cos_t) * (K @ K)

    # Small-angle fallback improves numerical stability.
    small = (theta[..., 0] < 1e-5)[..., None, None]
    R_small = eye + _skew(rotvec)
    return torch.where(small, R_small, R)


def se3_delta_to_matrix(delta: torch.Tensor) -> torch.Tensor:
    """Convert small SE(3) camera deltas to 4x4 matrices.

    delta layout: [tx, ty, tz, rx, ry, rz].
    The transform is applied in world frame:
        T_c2w_refined = exp(delta) @ T_c2w_base
    """
    trans = delta[..., :3]
    rot = delta[..., 3:6]
    R = so3_exp(rot)

    T = torch.eye(4, dtype=delta.dtype, device=delta.device)
    T = T.expand(delta.shape[:-1] + (4, 4)).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = trans
    return T


def _extract_intrinsics_from_views(
    views: Sequence[Dict[str, object]],
) -> List[Optional[np.ndarray]]:
    out: List[Optional[np.ndarray]] = []
    for view in views:
        K = view.get("camera_intrinsics", None)
        if K is None:
            out.append(None)
            continue
        if torch.is_tensor(K):
            K = K.detach().cpu().numpy()
        K = np.asarray(K, dtype=np.float32)
        if K.ndim == 3:
            K = K[0]
        if K.shape == (3, 3) and np.isfinite(K).all():
            out.append(K.astype(np.float32))
        else:
            out.append(None)
    return out


def _cams_by_pred_index(
    pred_cams: Sequence[Dict[str, object]],
    num_views: int,
) -> List[Optional[np.ndarray]]:
    out: List[Optional[np.ndarray]] = [None for _ in range(num_views)]
    for cam in pred_cams:
        idx = int(cam.get("pred_index", -1))
        if idx < 0 or idx >= num_views:
            continue
        T = np.asarray(cam.get("T_c2w"), dtype=np.float32)
        if T.shape == (4, 4) and np.isfinite(T).all():
            out[idx] = T.astype(np.float32)
    return out


def _resize_rgb_to_hw(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    return rgb


def _sample_gaussians_from_pointmaps(
    pred_maps: Sequence[np.ndarray],
    pred_valid_masks: Sequence[np.ndarray],
    rgbs: Sequence[np.ndarray],
    max_gaussians: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    points_all: List[np.ndarray] = []
    colors_all: List[np.ndarray] = []
    view_ids_all: List[np.ndarray] = []
    ys_all: List[np.ndarray] = []
    xs_all: List[np.ndarray] = []

    for i, (pmap, mask, rgb) in enumerate(zip(pred_maps, pred_valid_masks, rgbs)):
        pmap = np.asarray(pmap, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if pmap.ndim != 3 or pmap.shape[-1] != 3:
            continue
        if mask.shape != pmap.shape[:2]:
            continue

        h, w = pmap.shape[:2]
        rgb = _resize_rgb_to_hw(rgb, h, w)
        valid = mask & np.isfinite(pmap).all(axis=-1)
        if not valid.any():
            continue

        ys, xs = np.nonzero(valid)
        pts = pmap[ys, xs].reshape(-1, 3).astype(np.float32)
        cols = rgb[ys, xs].reshape(-1, 3).astype(np.float32) / 255.0

        points_all.append(pts)
        colors_all.append(cols)
        view_ids_all.append(np.full((pts.shape[0],), i, dtype=np.int64))
        ys_all.append(ys.astype(np.int64))
        xs_all.append(xs.astype(np.int64))

    if not points_all:
        return {
            "points": np.empty((0, 3), dtype=np.float32),
            "colors": np.empty((0, 3), dtype=np.float32),
            "view_ids": np.empty((0,), dtype=np.int64),
            "ys": np.empty((0,), dtype=np.int64),
            "xs": np.empty((0,), dtype=np.int64),
        }

    points = np.concatenate(points_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)
    view_ids = np.concatenate(view_ids_all, axis=0)
    ys = np.concatenate(ys_all, axis=0)
    xs = np.concatenate(xs_all, axis=0)

    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    view_ids, ys, xs = view_ids[finite], ys[finite], xs[finite]

    if max_gaussians > 0 and points.shape[0] > int(max_gaussians):
        rng = np.random.default_rng(int(seed))
        keep = rng.choice(points.shape[0], size=int(max_gaussians), replace=False)
        points = points[keep]
        colors = colors[keep]
        view_ids = view_ids[keep]
        ys = ys[keep]
        xs = xs[keep]

    return {
        "points": points.astype(np.float32),
        "colors": colors.astype(np.float32),
        "view_ids": view_ids.astype(np.int64),
        "ys": ys.astype(np.int64),
        "xs": xs.astype(np.int64),
    }


def _prepare_view_tensors(
    pred_cams: Sequence[Dict[str, object]],
    intrinsics: Sequence[Optional[np.ndarray]],
    rgbs: Sequence[np.ndarray],
    point_hw: Tuple[int, int],
    render_scale: float,
    device: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[int],
    int,
    int,
]:
    """Prepare local view tensors for gsplat rendering.

    Returns:
        base_c2w: [C,4,4]
        Ks:       [C,3,3]
        images:   [C,H,W,3]
        local_ids:[C]
        height,width: render size
    """
    h0, w0 = int(point_hw[0]), int(point_hw[1])
    scale = float(render_scale)
    if scale <= 0:
        scale = 1.0
    h = max(16, int(round(h0 * scale)))
    w = max(16, int(round(w0 * scale)))

    cams_by_idx = _cams_by_pred_index(pred_cams, len(rgbs))

    c2w_list = []
    K_list = []
    img_list = []
    local_ids: List[int] = []

    for i, rgb in enumerate(rgbs):
        if i >= len(intrinsics):
            continue
        K = intrinsics[i]
        T = cams_by_idx[i]
        if K is None or T is None:
            continue

        K = np.asarray(K, dtype=np.float32).copy()
        K[0, :] *= float(w) / float(w0)
        K[1, :] *= float(h) / float(h0)

        rgb = _resize_rgb_to_hw(rgb, h0, w0)
        if (h, w) != (h0, w0):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        rgb_f = rgb.astype(np.float32) / 255.0

        c2w_list.append(T.astype(np.float32))
        K_list.append(K.astype(np.float32))
        img_list.append(rgb_f.astype(np.float32))
        local_ids.append(i)

    if not c2w_list:
        raise RuntimeError("No valid cameras/intrinsics for gsplat refinement.")

    base_c2w = torch.from_numpy(np.stack(c2w_list, axis=0)).to(device)
    Ks = torch.from_numpy(np.stack(K_list, axis=0)).to(device)
    images = torch.from_numpy(np.stack(img_list, axis=0)).to(device)

    return base_c2w, Ks, images, torch.tensor(local_ids, device=device), local_ids, h, w


def _estimate_initial_scale(points: np.ndarray, ratio: float, min_scale: float) -> float:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] < 2:
        return float(max(min_scale, 1e-4))
    diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if not np.isfinite(diag) or diag <= 0:
        diag = 1.0
    return float(max(diag * float(ratio), float(min_scale), 1e-6))


def _compute_gsplat_world_normalization(
    points: np.ndarray,
    c2w: np.ndarray,
    percentile: float = 95.0,
    min_scale: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Compute robust world -> local similarity normalization.

    x_local = (x_world - center) / scale
    Uses both Gaussian init points and camera centers.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    cams = np.asarray(c2w, dtype=np.float32).reshape(-1, 4, 4)[:, :3, 3]

    samples = []
    if pts.size > 0:
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] > 0:
            samples.append(pts)
    if cams.size > 0:
        cams = cams[np.isfinite(cams).all(axis=1)]
        if cams.shape[0] > 0:
            samples.append(cams)

    if not samples:
        center = np.zeros(3, dtype=np.float32)
        scale = np.float32(1.0)
    else:
        all_pts = np.concatenate(samples, axis=0).astype(np.float32)
        center = np.median(all_pts, axis=0).astype(np.float32)
        dist = np.linalg.norm(all_pts - center[None, :], axis=1)
        dist = dist[np.isfinite(dist)]
        if dist.size == 0:
            scale = np.float32(1.0)
        else:
            scale = np.float32(
                max(float(np.percentile(dist, float(percentile))), float(min_scale))
            )

    return {
        "center": center.astype(np.float32),
        "scale": np.asarray(scale, dtype=np.float32),
    }


def _points_world_to_local(points: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    center = np.asarray(norm["center"], dtype=np.float32).reshape(1, 3)
    scale = float(np.asarray(norm["scale"], dtype=np.float32))
    return ((np.asarray(points, dtype=np.float32) - center) / scale).astype(np.float32)


def _points_local_to_world(points: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    center = np.asarray(norm["center"], dtype=np.float32).reshape(1, 3)
    scale = float(np.asarray(norm["scale"], dtype=np.float32))
    return (np.asarray(points, dtype=np.float32) * scale + center).astype(np.float32)


def _c2w_world_to_local(c2w: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    out = np.asarray(c2w, dtype=np.float32).copy()
    center = np.asarray(norm["center"], dtype=np.float32).reshape(3)
    scale = float(np.asarray(norm["scale"], dtype=np.float32))
    out[..., :3, 3] = (out[..., :3, 3] - center[None, :]) / scale
    return out.astype(np.float32)


def _c2w_local_to_world(c2w: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    out = np.asarray(c2w, dtype=np.float32).copy()
    center = np.asarray(norm["center"], dtype=np.float32).reshape(3)
    scale = float(np.asarray(norm["scale"], dtype=np.float32))
    out[..., :3, 3] = out[..., :3, 3] * scale + center[None, :]
    return out.astype(np.float32)


def _replace_selected_points_in_maps(
    pred_maps: Sequence[np.ndarray],
    pred_valid_masks: Sequence[np.ndarray],
    selected: Dict[str, np.ndarray],
    refined_means: np.ndarray,
    optimized_only: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if optimized_only:
        maps_out = [
            np.zeros_like(np.asarray(m, dtype=np.float32))
            if np.asarray(m).ndim == 3
            else np.empty((0, 0, 3), dtype=np.float32)
            for m in pred_maps
        ]
        masks_out = [
            np.zeros_like(np.asarray(mask, dtype=bool))
            for mask in pred_valid_masks
        ]
    else:
        maps_out = [np.asarray(m, dtype=np.float32).copy() for m in pred_maps]
        masks_out = [np.asarray(mask, dtype=bool).copy() for mask in pred_valid_masks]

    view_ids = selected["view_ids"]
    ys = selected["ys"]
    xs = selected["xs"]

    for k in range(refined_means.shape[0]):
        vi = int(view_ids[k])
        y = int(ys[k])
        x = int(xs[k])
        if vi < 0 or vi >= len(maps_out):
            continue
        if maps_out[vi].ndim != 3 or maps_out[vi].shape[-1] != 3:
            continue
        if y < 0 or x < 0 or y >= maps_out[vi].shape[0] or x >= maps_out[vi].shape[1]:
            continue
        maps_out[vi][y, x] = refined_means[k].astype(np.float32)
        masks_out[vi][y, x] = True

    return maps_out, masks_out


def _update_pred_cameras(
    pred_cams: Sequence[Dict[str, object]],
    local_ids: Sequence[int],
    refined_c2w: np.ndarray,
) -> List[Dict[str, object]]:
    by_local = {
        int(local_id): refined_c2w[k].astype(np.float32)
        for k, local_id in enumerate(local_ids)
    }

    out: List[Dict[str, object]] = []
    for cam in pred_cams:
        next_cam = dict(cam)
        idx = int(next_cam.get("pred_index", -1))
        if idx in by_local:
            next_cam["T_c2w"] = by_local[idx]
        out.append(next_cam)
    return out


def refine_chunk_with_gsplat(
    pred_maps: Sequence[np.ndarray],
    pred_valid_masks: Sequence[np.ndarray],
    pred_cams: Sequence[Dict[str, object]],
    chunk_views_raw: Sequence[Dict[str, object]],
    rgbs: Sequence[np.ndarray],
    *,
    device: torch.device,
    steps: int = 200,
    max_gaussians: int = 120000,
    batch_views: int = 2,
    render_scale: float = 0.5,
    seed: int = 0,
    lr_means: float = 1e-4,
    lr_scales: float = 5e-3,
    lr_opacities: float = 5e-2,
    lr_colors: float = 1e-2,
    lr_pose: float = 1e-5,
    optimize_means: bool = True,
    optimize_pose: bool = True,
    pose_reg: float = 1e-4,
    opacity_reg: float = 1e-4,
    scale_init_ratio: float = 0.001,
    min_scale: float = 1e-4,
    max_scale: float = 1.0,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    rasterize_mode: str = "classic",
    optimized_only: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, object]], Dict[str, object]]:
    """Refine a chunk with gsplat.

    Returns:
        refined_pred_maps
        refined_pred_valid_masks
        refined_pred_cams
        gsplat_meta
    """
    if int(steps) <= 0:
        return list(pred_maps), list(pred_valid_masks), list(pred_cams), {
            "enabled": False,
            "reason": "steps <= 0",
        }

    try:
        from gsplat.rendering import rasterization  # type: ignore[import-untyped]
    except Exception as e:
        raise RuntimeError(
            "gsplat refinement is enabled, but gsplat cannot be imported. "
            "Install it first, e.g. `pip install gsplat`, or disable "
            "--gsplat_refine."
        ) from e

    # Find point-map size.
    point_hw = None
    for m in pred_maps:
        arr = np.asarray(m)
        if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[0] > 0 and arr.shape[1] > 0:
            point_hw = arr.shape[:2]
            break
    if point_hw is None:
        return list(pred_maps), list(pred_valid_masks), list(pred_cams), {
            "enabled": False,
            "reason": "no valid point map",
        }

    selected = _sample_gaussians_from_pointmaps(
        pred_maps=pred_maps,
        pred_valid_masks=pred_valid_masks,
        rgbs=rgbs,
        max_gaussians=int(max_gaussians),
        seed=int(seed),
    )
    points_np = selected["points"]
    colors_np = selected["colors"]
    if points_np.shape[0] == 0:
        return list(pred_maps), list(pred_valid_masks), list(pred_cams), {
            "enabled": False,
            "reason": "no valid gaussians",
        }

    intrinsics = _extract_intrinsics_from_views(chunk_views_raw)
    base_c2w, Ks, target_images, _local_id_tensor, local_ids, height, width = (
        _prepare_view_tensors(
            pred_cams=pred_cams,
            intrinsics=intrinsics,
            rgbs=rgbs,
            point_hw=point_hw,
            render_scale=float(render_scale),
            device=device,
        )
    )

    num_views = int(base_c2w.shape[0])
    num_gaussians = int(points_np.shape[0])

    # Normalize world coordinates to a compact local frame.
    normalization = _compute_gsplat_world_normalization(
        points=points_np,
        c2w=base_c2w.detach().cpu().numpy(),
        percentile=95.0,
        min_scale=1e-6,
    )
    points_local_np = _points_world_to_local(points_np, normalization)
    base_c2w_local_np = _c2w_world_to_local(
        base_c2w.detach().cpu().numpy(), normalization
    )
    base_c2w = torch.from_numpy(base_c2w_local_np).to(device=device, dtype=torch.float32)

    means0 = torch.from_numpy(points_local_np).to(device=device, dtype=torch.float32)
    colors0 = torch.from_numpy(colors_np).to(device=device, dtype=torch.float32)

    init_scale = _estimate_initial_scale(
        points_local_np,
        ratio=float(scale_init_ratio),
        min_scale=float(min_scale),
    )
    init_scale = min(init_scale, float(max_scale))

    with torch.enable_grad():
        means = torch.nn.Parameter(means0.clone(), requires_grad=bool(optimize_means))
        color_logits = torch.nn.Parameter(_safe_logit(colors0))
        log_scales = torch.nn.Parameter(
            torch.full((num_gaussians, 3), math.log(init_scale), device=device, dtype=torch.float32)
        )
        opacity_logits = torch.nn.Parameter(
            torch.full((num_gaussians,), 0.0, device=device, dtype=torch.float32)
        )
        # Identity quaternion in gsplat wxyz convention.
        quats = torch.zeros((num_gaussians, 4), device=device, dtype=torch.float32)
        quats[:, 0] = 1.0

        pose_delta = torch.nn.Parameter(
            torch.zeros((num_views, 6), device=device, dtype=torch.float32),
            requires_grad=bool(optimize_pose),
        )

        param_groups = []
        if optimize_means:
            param_groups.append({"params": [means], "lr": float(lr_means)})
        param_groups.extend(
            [
                {"params": [color_logits], "lr": float(lr_colors)},
                {"params": [log_scales], "lr": float(lr_scales)},
                {"params": [opacity_logits], "lr": float(lr_opacities)},
            ]
        )
        if optimize_pose:
            param_groups.append({"params": [pose_delta], "lr": float(lr_pose)})

        if not param_groups:
            return list(pred_maps), list(pred_valid_masks), list(pred_cams), {
                "enabled": False,
                "reason": "no trainable gsplat parameters",
            }

        optimizer = torch.optim.Adam(param_groups)

        rng = torch.Generator(device=device)
        rng.manual_seed(int(seed) + 99173)

        last_loss = float("nan")
        last_photo = float("nan")

        print(
            f"[gsplat] grad_enabled={torch.is_grad_enabled()}, "
            f"optimize_pose={optimize_pose}, optimize_means={optimize_means}, "
            f"num_gaussians={num_gaussians}, num_views={num_views}"
        )

        for _step in range(int(steps)):
            if int(batch_views) > 0 and int(batch_views) < num_views:
                perm = torch.randperm(num_views, generator=rng, device=device)
                view_ids = perm[: int(batch_views)]
            else:
                view_ids = torch.arange(num_views, device=device)

            delta_T = se3_delta_to_matrix(pose_delta)
            refined_c2w_all = torch.matmul(delta_T, base_c2w)
            refined_c2w = refined_c2w_all[view_ids]
            viewmats = torch.linalg.inv(refined_c2w)

            scales = torch.exp(log_scales).clamp(float(min_scale), float(max_scale))
            opacities = torch.sigmoid(opacity_logits)
            colors = torch.sigmoid(color_logits)

            renders, alphas, _ = rasterization(  # type: ignore[call-arg]
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmats,
                Ks=Ks[view_ids],
                width=int(width),
                height=int(height),
                near_plane=float(near_plane),
                far_plane=float(far_plane),
                radius_clip=float(radius_clip),
                render_mode="RGB",
                packed=True,
                sparse_grad=False,
                rasterize_mode=str(rasterize_mode),
            )

            target = target_images[view_ids]
            photo_loss = (renders[..., :3] - target).abs().mean()
            pose_reg_loss = torch.zeros((), device=device, dtype=torch.float32)
            opacity_reg_loss = torch.zeros((), device=device, dtype=torch.float32)

            loss = photo_loss

            if optimize_pose and float(pose_reg) > 0:
                pose_reg_loss = float(pose_reg) * (pose_delta ** 2).mean()
                loss = loss + pose_reg_loss

            if float(opacity_reg) > 0:
                opacity_reg_loss = float(opacity_reg) * (opacities ** 2).mean()
                loss = loss + opacity_reg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            last_loss = float(loss.detach().item())
            last_photo = float(photo_loss.detach().item())

        with torch.no_grad():
            final_delta_T = se3_delta_to_matrix(pose_delta)
            final_c2w = torch.matmul(final_delta_T, base_c2w)
            refined_means = means.detach().cpu().numpy().astype(np.float32)
            refined_c2w_np = final_c2w.detach().cpu().numpy().astype(np.float32)
            pose_delta_np = pose_delta.detach().cpu().numpy().astype(np.float32)

    maps_out, masks_out = _replace_selected_points_in_maps(
        pred_maps=pred_maps,
        pred_valid_masks=pred_valid_masks,
        selected=selected,
        refined_means=refined_means,
        optimized_only=bool(optimized_only),
    )
    cams_out = _update_pred_cameras(
        pred_cams=pred_cams,
        local_ids=local_ids,
        refined_c2w=refined_c2w_np,
    )

    trans_norm = np.linalg.norm(pose_delta_np[:, :3], axis=1)
    rot_norm = np.linalg.norm(pose_delta_np[:, 3:6], axis=1)

    meta = {
        "enabled": True,
        "method": "gsplat_chunk_refine",
        "steps": int(steps),
        "num_gaussians": int(num_gaussians),
        "num_views": int(num_views),
        "render_size": [int(height), int(width)],
        "render_scale": float(render_scale),
        "batch_views": int(batch_views),
        "optimize_means": bool(optimize_means),
        "optimize_pose": bool(optimize_pose),
        "optimized_only": bool(optimized_only),
        "init_scale": float(init_scale),
        "final_loss": float(last_loss),
        "final_photo_loss": float(last_photo),
        "pose_delta_trans_median": float(np.median(trans_norm)) if trans_norm.size else 0.0,
        "pose_delta_trans_max": float(np.max(trans_norm)) if trans_norm.size else 0.0,
        "pose_delta_rot_median_rad": float(np.median(rot_norm)) if rot_norm.size else 0.0,
        "pose_delta_rot_max_rad": float(np.max(rot_norm)) if rot_norm.size else 0.0,
    }
    return maps_out, masks_out, cams_out, meta


# ---------------------------------------------------------------------------
# 3DGS PLY/NPZ save helpers
# ---------------------------------------------------------------------------
def _save_loss_history(
    bundle_dir: Path,
    loss_history: Sequence[Dict[str, object]],
) -> Dict[str, str]:
    """Save gsplat optimization loss history as CSV and JSON."""
    if not loss_history:
        return {}

    bundle_dir.mkdir(parents=True, exist_ok=True)
    csv_path = bundle_dir / "loss_history.csv"
    json_path = bundle_dir / "loss_history.json"

    fieldnames = [
        "step",
        "loss",
        "photo_loss",
        "l1_loss",
        "ssim_loss",
        "pose_reg_loss",
        "opacity_reg_loss",
        "scale_reg_loss",
        "num_gaussians",
        "opacity_mean",
        "opacity_min",
        "opacity_max",
        "scale_mean",
        "scale_min",
        "scale_max",
        "pose_delta_trans_median",
        "pose_delta_rot_median_rad",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in loss_history:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    json_path.write_text(
        json.dumps(list(loss_history), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "loss_history_csv": str(csv_path),
        "loss_history_json": str(json_path),
    }


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


def _save_rgb_png(path: Path, rgb01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb8 = _to_uint8_rgb(rgb01)
    bgr8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr8)


def _depth_to_color(depth: np.ndarray, alpha: Optional[np.ndarray] = None) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)

    if alpha is not None:
        alpha = np.asarray(alpha, dtype=np.float32)
        valid = valid & np.isfinite(alpha) & (alpha > 1e-4)

    if not valid.any():
        return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)

    vals = depth[valid]
    lo = float(np.percentile(vals, 2.0))
    hi = float(np.percentile(vals, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())
    if hi <= lo:
        hi = lo + 1e-6

    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)

    gray = (norm * 255.0 + 0.5).astype(np.uint8)
    color_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    color_rgb[~valid] = 0
    return color_rgb


@torch.no_grad()
def _save_gsplat_render_previews(
    *,
    bundle_dir: Path,
    rasterization_fn,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    c2w: torch.Tensor,
    Ks: torch.Tensor,
    target_images: torch.Tensor,
    render_stems: Sequence[str],
    render_global_indices: Sequence[int],
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    rasterize_mode: str,
    max_views: int,
    stride: int,
) -> List[Dict[str, object]]:
    """Render optimized 3DGS to RGB+D and save preview PNG/NPY files."""
    render_dir = bundle_dir / "renders"
    rgb_dir = render_dir / "rgb"
    depth_dir = render_dir / "depth"
    target_dir = render_dir / "target"
    error_dir = render_dir / "error"

    stride = max(1, int(stride))
    view_indices = list(range(0, int(c2w.shape[0]), stride))
    if int(max_views) > 0:
        view_indices = view_indices[: int(max_views)]

    artifacts: List[Dict[str, object]] = []

    viewmats_all = torch.linalg.inv(c2w)

    for vi in view_indices:
        stem = str(render_stems[vi]) if vi < len(render_stems) else f"view_{vi:03d}"
        global_idx = (
            int(render_global_indices[vi])
            if vi < len(render_global_indices)
            else int(vi)
        )

        safe_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)
        prefix = f"{vi:04d}_g{global_idx:06d}_{safe_stem}"

        rendering, alpha, _ = rasterization_fn(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats_all[vi : vi + 1],
            Ks=Ks[vi : vi + 1],
            width=int(width),
            height=int(height),
            near_plane=float(near_plane),
            far_plane=float(far_plane),
            radius_clip=float(radius_clip),
            render_mode="RGB+D",
            packed=True,
            sparse_grad=False,
            rasterize_mode=str(rasterize_mode),
        )

        rgb = rendering[0, ..., :3].detach().float().clamp(0, 1).cpu().numpy()
        depth = rendering[0, ..., 3].detach().float().cpu().numpy()
        a = alpha[0, ..., 0].detach().float().cpu().numpy() if alpha.ndim == 4 else alpha[0].detach().float().cpu().numpy()
        target = target_images[vi].detach().float().clamp(0, 1).cpu().numpy()

        err = np.mean(np.abs(rgb - target), axis=-1)
        err_rgb = np.repeat(np.clip(err[..., None] / max(float(err.max()), 1e-6), 0.0, 1.0), 3, axis=-1)

        rgb_path = rgb_dir / f"{prefix}_rgb.png"
        target_path = target_dir / f"{prefix}_target.png"
        err_path = error_dir / f"{prefix}_error.png"
        depth_png_path = depth_dir / f"{prefix}_depth.png"
        depth_npy_path = depth_dir / f"{prefix}_depth.npy"

        _save_rgb_png(rgb_path, rgb)
        _save_rgb_png(target_path, target)
        _save_rgb_png(err_path, err_rgb)

        depth_color = _depth_to_color(depth, alpha=a)
        _save_rgb_png(depth_png_path, depth_color.astype(np.float32) / 255.0)
        np.save(depth_npy_path, depth.astype(np.float32))

        artifacts.append(
            {
                "view_id": int(vi),
                "global_index": int(global_idx),
                "stem": stem,
                "rgb": str(rgb_path),
                "target": str(target_path),
                "error": str(err_path),
                "depth_png": str(depth_png_path),
                "depth_npy": str(depth_npy_path),
            }
        )

    return artifacts


# ---------------------------------------------------------------------------
# PLY helpers
# ---------------------------------------------------------------------------
def _rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    c0 = 0.28209479177387814
    rgb = np.asarray(rgb, dtype=np.float32)
    return (rgb - 0.5) / c0


def _write_3dgs_ply(
    ply_path: Path,
    means: np.ndarray,
    colors: np.ndarray,
    log_scales: np.ndarray,
    opacity_logits: np.ndarray,
    quats: np.ndarray,
) -> None:
    """Write 3DGS-compatible binary PLY (x y z, normals, f_dc, opacity, scale, rot)."""
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    means = np.asarray(means, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
    log_scales = np.asarray(log_scales, dtype=np.float32).reshape(-1, 3)
    opacity_logits = np.asarray(opacity_logits, dtype=np.float32).reshape(-1)
    quats = np.asarray(quats, dtype=np.float32).reshape(-1, 4)

    n = int(means.shape[0])
    f_dc = _rgb_to_sh_dc(np.clip(colors, 0.0, 1.0)).astype(np.float32)
    normals = np.zeros((n, 3), dtype=np.float32)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    vertex = np.empty(n, dtype=dtype)

    vertex["x"] = means[:, 0]
    vertex["y"] = means[:, 1]
    vertex["z"] = means[:, 2]
    vertex["nx"] = normals[:, 0]
    vertex["ny"] = normals[:, 1]
    vertex["nz"] = normals[:, 2]
    vertex["f_dc_0"] = f_dc[:, 0]
    vertex["f_dc_1"] = f_dc[:, 1]
    vertex["f_dc_2"] = f_dc[:, 2]
    vertex["opacity"] = opacity_logits
    vertex["scale_0"] = log_scales[:, 0]
    vertex["scale_1"] = log_scales[:, 1]
    vertex["scale_2"] = log_scales[:, 2]
    vertex["rot_0"] = quats[:, 0]
    vertex["rot_1"] = quats[:, 1]
    vertex["rot_2"] = quats[:, 2]
    vertex["rot_3"] = quats[:, 3]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )

    with ply_path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertex.tofile(f)


def _save_3dgs_bundle_outputs(
    output_dir: Path,
    bundle_name: str,
    means_local: np.ndarray,
    colors: np.ndarray,
    log_scales_local: np.ndarray,
    opacity_logits: np.ndarray,
    quats: np.ndarray,
    refined_c2w_local: np.ndarray,
    refined_c2w_world: np.ndarray,
    pose_delta: np.ndarray,
    render_stems: Sequence[str],
    render_global_indices: Sequence[int],
    normalization: Dict[str, np.ndarray],
    meta: Dict[str, object],
    loss_history: Optional[Sequence[Dict[str, object]]] = None,
    render_artifacts: Optional[Sequence[Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Save optimized 3DGS (in local coords) and refined cameras (local + world)."""
    bundle_dir = output_dir / str(bundle_name)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    ply_path = bundle_dir / "point_cloud.ply"
    npz_path = bundle_dir / "gaussians.npz"
    cam_npz_path = bundle_dir / "cameras.npz"
    meta_path = bundle_dir / "meta.json"

    _write_3dgs_ply(
        ply_path=ply_path,
        means=means_local,
        colors=colors,
        log_scales=log_scales_local,
        opacity_logits=opacity_logits,
        quats=quats,
    )

    opacities = 1.0 / (1.0 + np.exp(-np.asarray(opacity_logits, dtype=np.float32)))
    scales_local = np.exp(np.asarray(log_scales_local, dtype=np.float32))

    np.savez(
        npz_path,
        means_local=np.asarray(means_local, dtype=np.float32),
        colors=np.asarray(colors, dtype=np.float32),
        log_scales_local=np.asarray(log_scales_local, dtype=np.float32),
        scales_local=scales_local.astype(np.float32),
        opacity_logits=np.asarray(opacity_logits, dtype=np.float32),
        opacities=opacities.astype(np.float32),
        quats=np.asarray(quats, dtype=np.float32),
        world_center=np.asarray(normalization["center"], dtype=np.float32),
        world_scale=np.asarray(normalization["scale"], dtype=np.float32),
    )

    np.savez(
        cam_npz_path,
        stems=np.asarray(list(render_stems), dtype=str),
        global_indices=np.asarray(list(render_global_indices), dtype=np.int64),
        T_c2w_local=np.asarray(refined_c2w_local, dtype=np.float32),
        T_c2w_world=np.asarray(refined_c2w_world, dtype=np.float32),
        pose_delta=np.asarray(pose_delta, dtype=np.float32),
    )

    loss_paths = _save_loss_history(bundle_dir, loss_history or [])

    save_meta = dict(meta)
    save_meta.update(
        {
            "saved": True,
            "format": "3dgs_ply",
            "coordinate_system": "local_normalized",
            "world_from_local": {
                "x_world": "x_local * world_scale + world_center",
                "world_center": np.asarray(normalization["center"], dtype=np.float32).tolist(),
                "world_scale": float(np.asarray(normalization["scale"], dtype=np.float32)),
            },
            "bundle_dir": str(bundle_dir),
            "ply_path": str(ply_path),
            "gaussians_npz": str(npz_path),
            "cameras_npz": str(cam_npz_path),
            "meta_json": str(meta_path),
            "loss_history": loss_paths,
            "render_artifacts": list(render_artifacts or []),
            "render_preview_dir": str(bundle_dir / "renders") if render_artifacts else None,
        }
    )

    meta_path.write_text(
        json.dumps(save_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return save_meta


# ---------------------------------------------------------------------------
# Official-style gsplat trainer helpers
# ---------------------------------------------------------------------------

_SH_C0 = 0.28209479177387814


def _rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB in [0, 1] to SH DC coefficient."""
    return (rgb - 0.5) / _SH_C0


def _sh0_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    """Convert SH DC coefficient back to RGB in [0, 1]."""
    return torch.clamp(sh0 * _SH_C0 + 0.5, 0.0, 1.0)


def _safe_adam(
    params,
    *,
    lr: float,
    eps: float,
    betas=(0.9, 0.999),
    fused: bool = False,
):
    """Create Adam with optional fused fallback for older PyTorch/CUDA combos."""
    try:
        return torch.optim.Adam(params, lr=lr, eps=eps, betas=betas, fused=fused)
    except TypeError:
        return torch.optim.Adam(params, lr=lr, eps=eps, betas=betas)


def _estimate_log_scales_from_points(
    points: torch.Tensor,
    *,
    init_scale: float,
    max_exact_knn_points: int = 200000,
    sample_points: int = 50000,
) -> torch.Tensor:
    """Official-style KNN scale initialization.

    Official gsplat trainer initializes Gaussian size from the average distance
    of 3 nearest neighbors. Dense UAV pointmaps may contain millions of points,
    so exact cdist is only used below max_exact_knn_points. For larger sets,
    estimate a global median neighbor distance from a random subset and apply
    it to all points.
    """
    points = points.float()
    n = int(points.shape[0])
    device = points.device

    if n <= 1:
        return torch.full((n, 3), math.log(1e-4), device=device, dtype=torch.float32)

    with torch.no_grad():
        if n <= int(max_exact_knn_points):
            dist = torch.cdist(points, points)
            dist.fill_diagonal_(float("inf"))
            k = min(4, n - 1)
            knn_dist = torch.topk(dist, k=k, dim=1, largest=False).values
            if k > 1:
                dist_avg = knn_dist[:, : min(3, k)].mean(dim=-1)
            else:
                dist_avg = knn_dist[:, 0]
            dist_avg = dist_avg.clamp_min(1e-8)
            scales = torch.log(dist_avg * float(init_scale)).unsqueeze(-1).repeat(1, 3)
            return scales.float()

        # Large point cloud fallback: estimate one robust scale from subset.
        m = min(int(sample_points), n)
        perm = torch.randperm(n, device=device)[:m]
        pts = points[perm]
        dist = torch.cdist(pts, pts)
        dist.fill_diagonal_(float("inf"))
        k = min(4, m - 1)
        knn_dist = torch.topk(dist, k=k, dim=1, largest=False).values
        if k > 1:
            dist_avg = knn_dist[:, : min(3, k)].mean(dim=-1)
        else:
            dist_avg = knn_dist[:, 0]
        s = torch.median(dist_avg).clamp_min(1e-8) * float(init_scale)
        return torch.full((n, 3), torch.log(s).item(), device=device, dtype=torch.float32)


def _create_splats_with_optimizers_from_points(
    *,
    points_local_np: np.ndarray,
    rgbs_np: np.ndarray,
    init_opacity: float,
    init_scale: float,
    means_lr: float,
    scales_lr: float,
    opacities_lr: float,
    quats_lr: float,
    sh0_lr: float,
    shN_lr: float,
    scene_scale: float,
    sh_degree: int,
    sparse_grad: bool,
    visible_adam: bool,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """Adapted from gsplat example trainer's create_splats_with_optimizers().

    Difference:
      - input points/colors come from our feed-forward pointmaps;
      - we keep one-GPU bundle optimization;
      - SH fields are kept official-style: sh0 / shN.
    """
    points = torch.from_numpy(
        np.asarray(points_local_np, dtype=np.float32).reshape(-1, 3)
    ).to(device=device, dtype=torch.float32)

    rgbs = torch.from_numpy(
        np.asarray(rgbs_np, dtype=np.float32).reshape(-1, 3)
    ).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

    n = int(points.shape[0])
    if n <= 0:
        raise RuntimeError("Cannot initialize gsplat splats from empty point cloud.")

    scales = _estimate_log_scales_from_points(
        points,
        init_scale=float(init_scale),
    )

    quats = torch.rand((n, 4), device=device, dtype=torch.float32)
    opacities = torch.logit(
        torch.full(
            (n,),
            float(init_opacity),
            device=device,
            dtype=torch.float32,
        ).clamp(1e-6, 1.0 - 1e-6)
    )

    sh_degree = int(sh_degree)
    num_sh = (sh_degree + 1) ** 2
    colors = torch.zeros((n, num_sh, 3), device=device, dtype=torch.float32)
    colors[:, 0, :] = _rgb_to_sh0(rgbs)

    params = [
        ("means", torch.nn.Parameter(points), float(means_lr) * float(scene_scale)),
        ("scales", torch.nn.Parameter(scales), float(scales_lr)),
        ("quats", torch.nn.Parameter(quats), float(quats_lr)),
        ("opacities", torch.nn.Parameter(opacities), float(opacities_lr)),
        ("sh0", torch.nn.Parameter(colors[:, :1, :].contiguous()), float(sh0_lr)),
        ("shN", torch.nn.Parameter(colors[:, 1:, :].contiguous()), float(shN_lr)),
    ]

    splats = torch.nn.ParameterDict({name: value for name, value, _lr in params}).to(device)

    bs = max(1, int(batch_size))
    sqrt_bs = math.sqrt(float(bs))

    optimizer_class = torch.optim.Adam
    if bool(sparse_grad):
        optimizer_class = torch.optim.SparseAdam
    elif bool(visible_adam):
        if SelectiveAdam is None:
            raise RuntimeError("visible_adam=True requires gsplat.optimizers.SelectiveAdam.")
        optimizer_class = SelectiveAdam

    optimizers: Dict[str, torch.optim.Optimizer] = {}
    for name, _value, lr in params:
        lr_scaled = float(lr) * sqrt_bs
        betas = (1 - bs * (1 - 0.9), 1 - bs * (1 - 0.999))
        betas = (
            max(0.0, min(0.999, betas[0])),
            max(0.0, min(0.9999, betas[1])),
        )
        param_group = [{"params": splats[name], "lr": lr_scaled, "name": name}]

        if bool(sparse_grad):
            optimizers[name] = torch.optim.SparseAdam(
                param_group,
                lr=lr_scaled,
                eps=1e-15 / sqrt_bs,
                betas=betas,
            )
        elif bool(visible_adam):
            if SelectiveAdam is None:
                raise RuntimeError("visible_adam=True requires gsplat.optimizers.SelectiveAdam.")
            optimizers[name] = SelectiveAdam(
                param_group,
                lr=lr_scaled,
                eps=1e-15 / sqrt_bs,
                betas=betas,
            )
        else:
            optimizers[name] = _safe_adam(
                param_group,
                lr=lr_scaled,
                eps=1e-15 / sqrt_bs,
                betas=betas,
                fused=False,
            )

    return splats, optimizers


def _rasterize_splats_official_style(
    *,
    splats: torch.nn.ParameterDict,
    camtoworlds: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int,
    near_plane: float,
    far_plane: float,
    packed: bool,
    sparse_grad: bool,
    absgrad: bool,
    antialiased: bool,
    render_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Official-style rasterization wrapper.

    Uses:
      means      = splats["means"]
      scales     = exp(splats["scales"])
      opacities  = sigmoid(splats["opacities"])
      colors     = concat(sh0, shN)
    """
    from gsplat.rendering import rasterization  # type: ignore[import-untyped]

    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)

    rasterize_mode = "antialiased" if bool(antialiased) else "classic"

    renders, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),
        Ks=Ks,
        width=int(width),
        height=int(height),
        packed=bool(packed),
        absgrad=bool(absgrad),
        sparse_grad=bool(sparse_grad),
        rasterize_mode=rasterize_mode,
        render_mode=str(render_mode),
        sh_degree=int(sh_degree),
        near_plane=float(near_plane),
        far_plane=float(far_plane),
    )
    return renders, alphas, info


def _make_gsplat_strategy(
    *,
    strategy_name: str,
    max_gaussians: int,
    refine_start_iter: int,
    refine_stop_iter: int,
    refine_every: int,
    reset_every: int,
    prune_opa: float,
    grow_grad2d: float,
    grow_scale3d: float,
    grow_scale2d: float,
    prune_scale3d: float,
    prune_scale2d: float,
    absgrad: bool,
    verbose: bool,
):
    """Create official gsplat densification strategy."""
    name = str(strategy_name).lower().strip()

    if name in {"none", "fixed", "off"}:
        return None

    if name == "default":
        if DefaultStrategy is None:
            raise RuntimeError("gsplat.strategy.DefaultStrategy is unavailable.")
        return DefaultStrategy(
            prune_opa=float(prune_opa),
            grow_grad2d=float(grow_grad2d),
            grow_scale3d=float(grow_scale3d),
            grow_scale2d=float(grow_scale2d),
            prune_scale3d=float(prune_scale3d),
            prune_scale2d=float(prune_scale2d),
            refine_start_iter=int(refine_start_iter),
            refine_stop_iter=int(refine_stop_iter),
            refine_every=int(refine_every),
            reset_every=int(reset_every),
            absgrad=bool(absgrad),
            verbose=bool(verbose),
        )

    if name == "mcmc":
        if MCMCStrategy is None:
            raise RuntimeError("gsplat.strategy.MCMCStrategy is unavailable.")

        # MCMCStrategy signatures vary slightly across gsplat versions.
        kwargs = {
            "refine_start_iter": int(refine_start_iter),
            "refine_stop_iter": int(refine_stop_iter),
            "refine_every": int(refine_every),
            "min_opacity": float(prune_opa),
            "verbose": bool(verbose),
        }
        if int(max_gaussians) > 0:
            kwargs["cap_max"] = int(max_gaussians)
        try:
            return MCMCStrategy(**kwargs)
        except TypeError:
            kwargs.pop("cap_max", None)
            return MCMCStrategy(**kwargs)

    raise ValueError(f"Unknown gsplat strategy: {strategy_name!r}")


def _strategy_initialize_state(strategy, *, scene_scale: float):
    if strategy is None:
        return None
    if DefaultStrategy is not None and isinstance(strategy, DefaultStrategy):
        return strategy.initialize_state(scene_scale=float(scene_scale))
    if MCMCStrategy is not None and isinstance(strategy, MCMCStrategy):
        return strategy.initialize_state()
    return strategy.initialize_state()


def _apply_sparse_grad_if_needed(
    *,
    splats: torch.nn.ParameterDict,
    info: Dict[str, torch.Tensor],
    sparse_grad: bool,
    packed: bool,
    num_views: int,
) -> None:
    """Same idea as official trainer: turn dense gradients into SparseTensor."""
    if not bool(sparse_grad):
        return
    if not bool(packed):
        raise RuntimeError("sparse_grad=True requires packed=True.")

    gaussian_ids = info.get("gaussian_ids", None)
    if gaussian_ids is None:
        return

    for key in splats.keys():
        grad = splats[key].grad
        if grad is None or grad.is_sparse:
            continue
        splats[key].grad = torch.sparse_coo_tensor(
            indices=gaussian_ids[None],
            values=grad[gaussian_ids],
            size=splats[key].size(),
            is_coalesced=int(num_views) == 1,
        )


def _optimizer_step_all(
    *,
    optimizers: Dict[str, torch.optim.Optimizer],
    visible_adam: bool,
    visibility_mask: Optional[torch.Tensor] = None,
) -> None:
    for optimizer in optimizers.values():
        if bool(visible_adam):
            if visibility_mask is None:
                optimizer.step()
            else:
                optimizer.step(visibility_mask)
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def _call_strategy_pre_backward(
    strategy,
    *,
    params: torch.nn.ParameterDict,
    optimizers: Dict[str, torch.optim.Optimizer],
    state,
    step: int,
    info: Dict[str, torch.Tensor],
) -> None:
    if strategy is None:
        return
    strategy.step_pre_backward(
        params=params,
        optimizers=optimizers,
        state=state,
        step=int(step),
        info=info,
    )


def _call_strategy_post_backward(
    strategy,
    *,
    params: torch.nn.ParameterDict,
    optimizers: Dict[str, torch.optim.Optimizer],
    state,
    step: int,
    info: Dict[str, torch.Tensor],
    packed: bool,
    lr: float,
) -> None:
    if strategy is None:
        return

    if DefaultStrategy is not None and isinstance(strategy, DefaultStrategy):
        strategy.step_post_backward(
            params=params,
            optimizers=optimizers,
            state=state,
            step=int(step),
            info=info,
            packed=bool(packed),
        )
        return

    if MCMCStrategy is not None and isinstance(strategy, MCMCStrategy):
        strategy.step_post_backward(
            params=params,
            optimizers=optimizers,
            state=state,
            step=int(step),
            info=info,
            lr=float(lr),
        )
        return

    strategy.step_post_backward(
        params=params,
        optimizers=optimizers,
        state=state,
        step=int(step),
        info=info,
    )


# ---------------------------------------------------------------------------
# Bundle-level 3DGS optimization (official gsplat trainer style)
# ---------------------------------------------------------------------------
def optimize_and_save_gsplat_bundle(
    init_pred_maps: Sequence[np.ndarray],
    init_pred_valid_masks: Sequence[np.ndarray],
    init_rgbs: Sequence[np.ndarray],
    render_views_raw: Sequence[Dict[str, object]],
    render_rgbs: Sequence[np.ndarray],
    render_cams: Sequence[Dict[str, object]],
    *,
    output_dir: Path,
    bundle_name: str,
    bundle_meta: Dict[str, object],
    device: torch.device,
    steps: int = 30000,
    max_gaussians: int = 120000,
    batch_views: int = 1,
    render_scale: float = 0.5,
    seed: int = 0,

    # Official-style Gaussian params.
    sh_degree: int = 3,
    sh_degree_interval: int = 1000,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    ssim_lambda: float = 0.2,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20.0,

    # Strategy.
    strategy_name: str = "default",
    refine_start_iter: int = 500,
    refine_stop_iter: int = 15000,
    refine_every: int = 100,
    reset_every: int = 3000,
    prune_opa: float = 0.005,
    grow_grad2d: float = 0.0002,
    grow_scale3d: float = 0.01,
    grow_scale2d: float = 0.05,
    prune_scale3d: float = 0.1,
    prune_scale2d: float = 0.15,
    absgrad: bool = False,
    strategy_verbose: bool = True,

    # Optimization behavior.
    optimize_means: bool = True,
    optimize_pose: bool = False,
    pose_lr: float = 1e-5,
    pose_reg: float = 1e-6,
    opacity_reg: float = 0.0,
    scale_reg: float = 0.0,

    # Rasterization / optimizer mode.
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    packed: bool = False,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    antialiased: bool = False,
    random_bkgd: bool = False,

    # Logging.
    log_every: int = 50,
    use_tqdm: bool = True,
    save_rendered_views: bool = False,
    render_output_max_views: int = 12,
    render_output_stride: int = 1,
) -> Dict[str, object]:
    """Optimize one 3DGS bundle using official-style gsplat training.

    This replaces the old hand-written fixed-Gaussian loop.

    Main properties:
      - ParameterDict keys follow gsplat examples:
          means / scales / quats / opacities / sh0 / shN
      - One optimizer per parameter.
      - DefaultStrategy / MCMCStrategy can add/prune Gaussians.
      - Loss is L1 + SSIM, plus optional opacity/scale/pose/mean regularization.
      - Output format remains compatible with current DOM renderer:
          gaussians.npz contains means_local, colors, scales, opacities, quats,
          world_center, world_scale.
    """
    if int(steps) <= 0:
        return {
            "enabled": False,
            "reason": "steps <= 0",
            **dict(bundle_meta),
        }

    try:
        from gsplat.rendering import rasterization as _rasterization_check  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "gsplat refinement is enabled, but gsplat cannot be imported. "
            "Install gsplat first or disable --gsplat_refine."
        ) from e

    # ------------------------------------------------------------------
    # 1. Prepare init points from feed-forward pointmaps
    # ------------------------------------------------------------------
    point_hw = None
    for m in init_pred_maps:
        arr = np.asarray(m)
        if (
            arr.ndim == 3
            and arr.shape[-1] == 3
            and arr.shape[0] > 0
            and arr.shape[1] > 0
        ):
            point_hw = arr.shape[:2]
            break

    if point_hw is None:
        return {
            "enabled": False,
            "reason": "no valid init point map",
            **dict(bundle_meta),
        }

    selected = _sample_gaussians_from_pointmaps(
        pred_maps=init_pred_maps,
        pred_valid_masks=init_pred_valid_masks,
        rgbs=init_rgbs,
        max_gaussians=int(max_gaussians),
        seed=int(seed),
    )
    points_np = np.asarray(selected["points"], dtype=np.float32).reshape(-1, 3)
    colors_np = np.asarray(selected["colors"], dtype=np.float32).reshape(-1, 3)

    if points_np.shape[0] == 0:
        return {
            "enabled": False,
            "reason": "no valid init gaussians",
            **dict(bundle_meta),
        }

    # ------------------------------------------------------------------
    # 2. Prepare render cameras and target images
    # ------------------------------------------------------------------
    render_intrinsics = _extract_intrinsics_from_views(render_views_raw)
    base_c2w, Ks, target_images, _local_id_tensor, local_ids, height, width = (
        _prepare_view_tensors(
            pred_cams=render_cams,
            intrinsics=render_intrinsics,
            rgbs=render_rgbs,
            point_hw=point_hw,
            render_scale=float(render_scale),
            device=device,
        )
    )

    render_stems: List[str] = []
    render_global_indices: List[int] = []
    for local_id in local_ids:
        local_id = int(local_id)
        if 0 <= local_id < len(render_cams):
            render_stems.append(
                str(render_cams[local_id].get("stem", f"view_{local_id:03d}"))
            )
            render_global_indices.append(
                int(render_cams[local_id].get("global_index", local_id))
            )
        else:
            render_stems.append(f"view_{local_id:03d}")
            render_global_indices.append(local_id)

    num_views = int(base_c2w.shape[0])
    if num_views <= 0:
        return {
            "enabled": False,
            "reason": "no render views",
            **dict(bundle_meta),
        }

    # ------------------------------------------------------------------
    # 3. Normalize world coordinates to local coordinate system
    # ------------------------------------------------------------------
    normalization = _compute_gsplat_world_normalization(
        points=points_np,
        c2w=base_c2w.detach().cpu().numpy(),
        percentile=95.0,
        min_scale=1e-6,
    )
    points_local_np = _points_world_to_local(points_np, normalization)
    base_c2w_local_np = _c2w_world_to_local(
        base_c2w.detach().cpu().numpy(),
        normalization,
    )
    base_c2w = torch.from_numpy(base_c2w_local_np).to(device=device, dtype=torch.float32)

    scene_scale_world = float(np.asarray(normalization["scale"], dtype=np.float32))
    scene_scale_local = 1.0

    # ------------------------------------------------------------------
    # 4. Create official-style splats and optimizers
    # ------------------------------------------------------------------
    splats, optimizers = _create_splats_with_optimizers_from_points(
        points_local_np=points_local_np,
        rgbs_np=colors_np,
        init_opacity=float(init_opacity),
        init_scale=float(init_scale),
        means_lr=float(means_lr),
        scales_lr=float(scales_lr),
        opacities_lr=float(opacities_lr),
        quats_lr=float(quats_lr),
        sh0_lr=float(sh0_lr),
        shN_lr=float(shN_lr),
        scene_scale=scene_scale_local,
        sh_degree=int(sh_degree),
        sparse_grad=bool(sparse_grad),
        visible_adam=bool(visible_adam),
        batch_size=max(1, int(batch_views)),
        device=device,
    )

    if not bool(optimize_means):
        splats["means"].requires_grad_(False)
        optimizers.pop("means", None)

    # Pose optimization is kept separate from gsplat strategy.
    pose_delta = torch.nn.Parameter(
        torch.zeros((num_views, 6), device=device, dtype=torch.float32),
        requires_grad=bool(optimize_pose),
    )
    pose_optimizer = None
    pose_scheduler = None
    if bool(optimize_pose):
        pose_optimizer = _safe_adam(
            [{"params": pose_delta, "lr": float(pose_lr), "name": "pose_delta"}],
            lr=float(pose_lr),
            eps=1e-15,
            betas=(0.9, 0.999),
            fused=False,
        )
        pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            pose_optimizer,
            gamma=0.01 ** (1.0 / max(int(steps), 1)),
        )

    means_scheduler = None
    if "means" in optimizers:
        means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"],
            gamma=0.01 ** (1.0 / max(int(steps), 1)),
        )

    # ------------------------------------------------------------------
    # 5. Official densification/pruning strategy
    # ------------------------------------------------------------------
    strategy = _make_gsplat_strategy(
        strategy_name=str(strategy_name),
        max_gaussians=int(max_gaussians),
        refine_start_iter=int(refine_start_iter),
        refine_stop_iter=int(refine_stop_iter),
        refine_every=int(refine_every),
        reset_every=int(reset_every),
        prune_opa=float(prune_opa),
        grow_grad2d=float(grow_grad2d),
        grow_scale3d=float(grow_scale3d),
        grow_scale2d=float(grow_scale2d),
        prune_scale3d=float(prune_scale3d),
        prune_scale2d=float(prune_scale2d),
        absgrad=bool(absgrad),
        verbose=bool(strategy_verbose),
    )

    strategy_state = None
    if strategy is not None:
        strategy.check_sanity(splats, optimizers)
        strategy_state = _strategy_initialize_state(
            strategy,
            scene_scale=scene_scale_local,
        )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed) + 99173)

    log_every_i = int(log_every)
    if log_every_i <= 0:
        log_every_i = int(steps) + 1

    sh_degree = int(sh_degree)
    sh_degree_interval = max(1, int(sh_degree_interval))

    loss_history: List[Dict[str, object]] = []
    last_loss = float("nan")
    last_photo = float("nan")
    last_l1 = float("nan")
    last_ssim = float("nan")

    print(
        f"[gsplat:{bundle_name}] official-style refine: "
        f"init_gaussians={int(splats['means'].shape[0])}, "
        f"views={num_views}, steps={int(steps)}, "
        f"batch_views={int(batch_views)}, render_size={height}x{width}, "
        f"strategy={strategy_name}, sh_degree={sh_degree}, "
        f"pose_opt={bool(optimize_pose)}, mean_opt={bool(optimize_means)}, "
        f"packed={bool(packed)}, sparse_grad={bool(sparse_grad)}"
    )

    step_iter = range(int(steps))
    pbar = None
    if bool(use_tqdm):
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(
                step_iter,
                desc=f"gsplat:{bundle_name}",
                dynamic_ncols=True,
                leave=True,
            )
            step_iter = pbar
        except Exception:
            pbar = None

    for step in step_iter:
        step_i = int(step)

        if int(batch_views) > 0 and int(batch_views) < num_views:
            perm = torch.randperm(num_views, generator=rng, device=device)
            view_ids = perm[: int(batch_views)]
        else:
            view_ids = torch.arange(num_views, device=device)

        if bool(optimize_pose):
            delta_T = se3_delta_to_matrix(pose_delta)
            c2w_all = torch.matmul(delta_T, base_c2w)
        else:
            c2w_all = base_c2w

        c2w = c2w_all[view_ids]
        target = target_images[view_ids]

        sh_degree_to_use = min(step_i // sh_degree_interval, sh_degree)

        renders, alphas, info = _rasterize_splats_official_style(
            splats=splats,
            camtoworlds=c2w,
            Ks=Ks[view_ids],
            width=int(width),
            height=int(height),
            sh_degree=int(sh_degree_to_use),
            near_plane=float(near_plane),
            far_plane=float(far_plane),
            packed=bool(packed),
            sparse_grad=bool(sparse_grad),
            absgrad=bool(absgrad) if str(strategy_name).lower() == "default" else False,
            antialiased=bool(antialiased),
            render_mode="RGB",
        )

        colors = renders[..., :3]

        if bool(random_bkgd):
            bkgd = torch.rand((1, 1, 1, 3), device=device, dtype=colors.dtype)
            colors = colors + bkgd * (1.0 - alphas)

        _call_strategy_pre_backward(
            strategy,
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step_i,
            info=info,
        )

        if l1_loss is not None:
            l1 = l1_loss(colors, target).mean()
        else:
            l1 = (colors - target).abs().mean()

        if ssim_loss is not None:
            ssim_l = ssim_loss(
                colors.permute(0, 3, 1, 2),
                target.permute(0, 3, 1, 2),
            )
        else:
            ssim_l = l1

        photo_loss = torch.lerp(l1, ssim_l, float(ssim_lambda))
        loss = photo_loss

        pose_reg_loss = torch.zeros((), device=device, dtype=torch.float32)
        opacity_reg_loss_value = torch.zeros((), device=device, dtype=torch.float32)
        scale_reg_loss_value = torch.zeros((), device=device, dtype=torch.float32)

        if bool(optimize_pose) and float(pose_reg) > 0:
            pose_reg_loss = float(pose_reg) * (pose_delta ** 2).mean()
            loss = loss + pose_reg_loss

        if float(opacity_reg) > 0:
            if opacity_reg_loss is not None:
                opacity_reg_loss_value = float(opacity_reg) * opacity_reg_loss(splats["opacities"])
            else:
                opacity_reg_loss_value = float(opacity_reg) * torch.sigmoid(splats["opacities"]).mean()
            loss = loss + opacity_reg_loss_value

        if float(scale_reg) > 0:
            if scale_reg_loss is not None:
                scale_reg_loss_value = float(scale_reg) * scale_reg_loss(splats["scales"])
            else:
                scale_reg_loss_value = float(scale_reg) * torch.exp(splats["scales"]).mean()
            loss = loss + scale_reg_loss_value

        loss.backward()

        _apply_sparse_grad_if_needed(
            splats=splats,
            info=info,
            sparse_grad=bool(sparse_grad),
            packed=bool(packed),
            num_views=int(view_ids.numel()),
        )

        visibility_mask = None
        if bool(visible_adam):
            if bool(packed):
                gaussian_ids = info.get("gaussian_ids", None)
                visibility_mask = torch.zeros_like(splats["opacities"], dtype=torch.bool)
                if gaussian_ids is not None:
                    visibility_mask.scatter_(0, gaussian_ids, 1)
            else:
                radii = info.get("radii", None)
                if radii is not None:
                    visibility_mask = (radii > 0).all(-1).any(0)

        _optimizer_step_all(
            optimizers=optimizers,
            visible_adam=bool(visible_adam),
            visibility_mask=visibility_mask,
        )

        if pose_optimizer is not None:
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)

        if means_scheduler is not None:
            means_scheduler.step()
        if pose_scheduler is not None:
            pose_scheduler.step()

        current_means_lr = (
            means_scheduler.get_last_lr()[0]
            if means_scheduler is not None
            else float(means_lr)
        )

        _call_strategy_post_backward(
            strategy,
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step_i,
            info=info,
            packed=bool(packed),
            lr=float(current_means_lr),
        )

        last_loss = float(loss.detach().item())
        last_photo = float(photo_loss.detach().item())
        last_l1 = float(l1.detach().item())
        last_ssim = float(ssim_l.detach().item())

        should_log = (
            step_i == 0
            or step_i == int(steps) - 1
            or ((step_i + 1) % log_every_i == 0)
        )
        if should_log:
            with torch.no_grad():
                opa = torch.sigmoid(splats["opacities"])
                sc = torch.exp(splats["scales"])
                if bool(optimize_pose):
                    pose_trans = torch.linalg.norm(pose_delta[:, :3], dim=-1)
                    pose_rot = torch.linalg.norm(pose_delta[:, 3:6], dim=-1)
                    pose_trans_med = float(torch.median(pose_trans).item())
                    pose_rot_med = float(torch.median(pose_rot).item())
                else:
                    pose_trans_med = 0.0
                    pose_rot_med = 0.0

                row = {
                    "step": step_i + 1,
                    "loss": last_loss,
                    "photo_loss": last_photo,
                    "l1_loss": last_l1,
                    "ssim_loss": last_ssim,
                    "pose_reg_loss": float(pose_reg_loss.detach().item()),
                    "opacity_reg_loss": float(opacity_reg_loss_value.detach().item()),
                    "scale_reg_loss": float(scale_reg_loss_value.detach().item()),
                    "num_gaussians": int(splats["means"].shape[0]),
                    "opacity_mean": float(opa.mean().item()) if opa.numel() else 0.0,
                    "opacity_min": float(opa.min().item()) if opa.numel() else 0.0,
                    "opacity_max": float(opa.max().item()) if opa.numel() else 0.0,
                    "scale_mean": float(sc.mean().item()) if sc.numel() else 0.0,
                    "scale_min": float(sc.min().item()) if sc.numel() else 0.0,
                    "scale_max": float(sc.max().item()) if sc.numel() else 0.0,
                    "pose_delta_trans_median": pose_trans_med,
                    "pose_delta_rot_median_rad": pose_rot_med,
                }
                loss_history.append(row)

            if pbar is not None:
                pbar.set_postfix(
                    {
                        "loss": f"{row['loss']:.4g}",
                        "gs": int(row["num_gaussians"]),
                        "opa": f"{row['opacity_mean']:.3f}",
                        "sc": f"{row['scale_mean']:.3g}",
                        "sh": int(sh_degree_to_use),
                    }
                )
            else:
                print(
                    f"[gsplat:{bundle_name}] "
                    f"step={row['step']:05d}/{int(steps):05d} "
                    f"loss={row['loss']:.6g} "
                    f"l1={row['l1_loss']:.6g} "
                    f"ssim={row['ssim_loss']:.6g} "
                    f"num_gs={row['num_gaussians']} "
                    f"opacity={row['opacity_mean']:.3f} "
                    f"scale={row['scale_mean']:.3g} "
                    f"sh={int(sh_degree_to_use)}"
                )

    # ------------------------------------------------------------------
    # 7. Export optimized parameters
    # ------------------------------------------------------------------
    with torch.no_grad():
        if bool(optimize_pose):
            final_delta_T = se3_delta_to_matrix(pose_delta)
            final_c2w_local = torch.matmul(final_delta_T, base_c2w)
            pose_delta_np = pose_delta.detach().cpu().numpy().astype(np.float32)
        else:
            final_c2w_local = base_c2w
            pose_delta_np = np.zeros((num_views, 6), dtype=np.float32)

        means_local_np = splats["means"].detach().cpu().numpy().astype(np.float32)
        log_scales_local_np = splats["scales"].detach().cpu().numpy().astype(np.float32)
        opacity_logits_np = splats["opacities"].detach().cpu().numpy().astype(np.float32)
        quats_np = splats["quats"].detach().cpu().numpy().astype(np.float32)

        sh0_t = splats["sh0"].detach()
        colors_np_final = _sh0_to_rgb(sh0_t[:, 0, :]).cpu().numpy().astype(np.float32)

        final_c2w_local_np = final_c2w_local.detach().cpu().numpy().astype(np.float32)
        final_c2w_world_np = _c2w_local_to_world(final_c2w_local_np, normalization)

        splats_state = {
            key: value.detach().cpu()
            for key, value in splats.state_dict().items()
        }

    num_gaussians_final = int(means_local_np.shape[0])
    trans_norm = (
        np.linalg.norm(pose_delta_np[:, :3], axis=1)
        if pose_delta_np.size
        else np.zeros((0,), dtype=np.float32)
    )
    rot_norm = (
        np.linalg.norm(pose_delta_np[:, 3:6], axis=1)
        if pose_delta_np.size
        else np.zeros((0,), dtype=np.float32)
    )

    bundle_dir = Path(output_dir) / str(bundle_name)
    splats_pt_path = bundle_dir / "splats.pt"

    render_artifacts: List[Dict[str, object]] = []
    if bool(save_rendered_views):
        try:
            from gsplat.rendering import rasterization as rasterization_fn

            render_artifacts = _save_gsplat_render_previews(
                bundle_dir=bundle_dir,
                rasterization_fn=rasterization_fn,
                means=torch.from_numpy(means_local_np).to(device=device),
                quats=torch.from_numpy(quats_np).to(device=device),
                scales=torch.exp(torch.from_numpy(log_scales_local_np).to(device=device)),
                opacities=torch.sigmoid(torch.from_numpy(opacity_logits_np).to(device=device)),
                colors=torch.from_numpy(colors_np_final).to(device=device),
                c2w=final_c2w_local.detach(),
                Ks=Ks,
                target_images=target_images,
                render_stems=render_stems,
                render_global_indices=render_global_indices,
                width=int(width),
                height=int(height),
                near_plane=float(near_plane),
                far_plane=float(far_plane),
                radius_clip=0.0,
                rasterize_mode="antialiased" if bool(antialiased) else "classic",
                max_views=int(render_output_max_views),
                stride=int(render_output_stride),
            )
        except Exception as e:
            print(f"[gsplat:{bundle_name}][WARN] failed to save render previews: {e}")

    meta = {
        "enabled": True,
        "method": "gsplat_official_style_strategy_refine",
        "strategy": str(strategy_name),
        "steps": int(steps),
        "num_gaussians_init": int(points_np.shape[0]),
        "num_gaussians_final": int(num_gaussians_final),
        "num_gaussians": int(num_gaussians_final),
        "num_render_views": int(num_views),
        "render_size": [int(height), int(width)],
        "render_scale": float(render_scale),
        "batch_views": int(batch_views),
        "sh_degree": int(sh_degree),
        "sh_degree_interval": int(sh_degree_interval),
        "init_opacity": float(init_opacity),
        "init_scale": float(init_scale),
        "ssim_lambda": float(ssim_lambda),
        "optimize_means": bool(optimize_means),
        "optimize_pose": bool(optimize_pose),
        "packed": bool(packed),
        "sparse_grad": bool(sparse_grad),
        "visible_adam": bool(visible_adam),
        "antialiased": bool(antialiased),
        "random_bkgd": bool(random_bkgd),
        "final_loss": float(last_loss),
        "final_photo_loss": float(last_photo),
        "final_l1_loss": float(last_l1),
        "final_ssim_loss": float(last_ssim),
        "scene_scale_world": float(scene_scale_world),
        "pose_delta_trans_median": float(np.median(trans_norm)) if trans_norm.size else 0.0,
        "pose_delta_trans_max": float(np.max(trans_norm)) if trans_norm.size else 0.0,
        "pose_delta_rot_median_rad": float(np.median(rot_norm)) if rot_norm.size else 0.0,
        "pose_delta_rot_max_rad": float(np.max(rot_norm)) if rot_norm.size else 0.0,
        "splats_pt": str(splats_pt_path),
        "loss_history_tail": loss_history[-200:],
        **dict(bundle_meta),
    }

    save_meta = _save_3dgs_bundle_outputs(
        output_dir=Path(output_dir),
        bundle_name=str(bundle_name),
        means_local=means_local_np,
        colors=colors_np_final,
        log_scales_local=log_scales_local_np,
        opacity_logits=opacity_logits_np,
        quats=quats_np,
        refined_c2w_local=final_c2w_local_np,
        refined_c2w_world=final_c2w_world_np,
        pose_delta=pose_delta_np,
        render_stems=render_stems,
        render_global_indices=render_global_indices,
        normalization=normalization,
        meta=meta,
        loss_history=loss_history,
        render_artifacts=render_artifacts,
    )

    bundle_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "gsplat_official_style_splats",
            "bundle_name": str(bundle_name),
            "splats": splats_state,
            "normalization": {
                "world_center": np.asarray(normalization["center"], dtype=np.float32),
                "world_scale": np.asarray(normalization["scale"], dtype=np.float32),
            },
            "meta": save_meta,
        },
        splats_pt_path,
    )

    return save_meta
