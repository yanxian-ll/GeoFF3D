# scripts/spatial_rrd/multiview_consistency.py
# -*- coding: utf-8 -*-
"""MVSNet-style multi-view consistency filtering for spatial RRD point maps.

This module works directly on predicted world point maps:
    pred_maps:        List[H,W,3] world point maps
    pred_valid_masks: List[H,W] current valid masks
    pred_cams:        List dicts with pred_index and T_c2w
    intrinsics:       List[3,3] input intrinsics for each local chunk view

It computes geometric support by projecting each source point map into nearby
target views, sampling target depth / target world point, and checking
depth + 3D consistency.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _as_numpy(x, dtype=np.float32):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach()
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        x = x.cpu().numpy()
    return np.asarray(x, dtype=dtype)


def intrinsics_from_views(
    views: Sequence[Dict[str, object]],
) -> List[Optional[np.ndarray]]:
    """Extract per-view 3x3 intrinsics from chunk views.

    Use chunk_views_raw, not filtered chunk_views, because prior-policy filtering
    may remove camera_intrinsics from the model input while post-processing still
    needs K for projection.
    """
    out: List[Optional[np.ndarray]] = []
    for view in views:
        K = view.get("camera_intrinsics", None)
        K_np = _as_numpy(K, dtype=np.float32)
        if K_np is None:
            out.append(None)
            continue
        if K_np.ndim == 3:
            K_np = K_np[0]
        if K_np.shape == (3, 3) and np.isfinite(K_np).all():
            out.append(K_np.astype(np.float32))
        else:
            out.append(None)
    return out


def _cameras_by_pred_index(
    pred_cams: Sequence[Dict[str, object]],
    num_views: int,
) -> List[Optional[np.ndarray]]:
    """Return local-index ordered T_c2w list."""
    Ts: List[Optional[np.ndarray]] = [None for _ in range(num_views)]
    for cam in pred_cams:
        idx = int(cam.get("pred_index", -1))
        if idx < 0 or idx >= num_views:
            continue
        T = np.asarray(cam.get("T_c2w"), dtype=np.float32)
        if T.shape == (4, 4) and np.isfinite(T).all():
            Ts[idx] = T.astype(np.float32)
    return Ts


def _prepare_stacks(
    pred_maps: Sequence[np.ndarray],
    pred_valid_masks: Sequence[np.ndarray],
    pred_cams: Sequence[Dict[str, object]],
    intrinsics: Sequence[Optional[np.ndarray]],
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Stack valid same-resolution views into torch tensors.

    Returns:
        points: [N,H,W,3]
        masks:  [N,H,W]
        Ks:     [N,3,3]
        Ts:     [N,4,4] cam2world
        valid_views: [N] bool
    """
    n = len(pred_maps)
    if n == 0:
        return None

    # Find reference shape.
    ref_hw = None
    for m in pred_maps:
        arr = np.asarray(m)
        if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[0] > 0 and arr.shape[1] > 0:
            ref_hw = arr.shape[:2]
            break
    if ref_hw is None:
        return None

    h, w = int(ref_hw[0]), int(ref_hw[1])
    Ts_np = _cameras_by_pred_index(pred_cams, n)

    points_np = np.zeros((n, h, w, 3), dtype=np.float32)
    masks_np = np.zeros((n, h, w), dtype=bool)
    Ks_np = np.tile(np.eye(3, dtype=np.float32)[None], (n, 1, 1))
    T_np = np.tile(np.eye(4, dtype=np.float32)[None], (n, 1, 1))
    valid_views_np = np.zeros((n,), dtype=bool)

    for i in range(n):
        P = np.asarray(pred_maps[i], dtype=np.float32)
        M = np.asarray(pred_valid_masks[i], dtype=bool)

        K = intrinsics[i] if i < len(intrinsics) else None
        T = Ts_np[i]

        if (
            P.shape != (h, w, 3)
            or M.shape != (h, w)
            or K is None
            or T is None
        ):
            continue

        finite = np.isfinite(P).all(axis=-1)
        M = M & finite
        if not M.any():
            continue

        points_np[i] = P
        masks_np[i] = M
        Ks_np[i] = np.asarray(K, dtype=np.float32)
        T_np[i] = np.asarray(T, dtype=np.float32)
        valid_views_np[i] = True

    points = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)
    masks = torch.from_numpy(masks_np).to(device=device, dtype=torch.bool)
    Ks = torch.from_numpy(Ks_np).to(device=device, dtype=torch.float32)
    Ts = torch.from_numpy(T_np).to(device=device, dtype=torch.float32)
    valid_views = torch.from_numpy(valid_views_np).to(device=device, dtype=torch.bool)

    return points, masks, Ks, Ts, valid_views


def _select_knn_neighbors(
    Ts: torch.Tensor,
    valid_views: torch.Tensor,
    max_neighbors: int,
) -> List[List[int]]:
    """Select nearest camera-center neighbors for each view."""
    n = int(Ts.shape[0])
    max_neighbors = max(0, int(max_neighbors))
    centers = Ts[:, :3, 3]  # [N,3]

    out: List[List[int]] = []
    for i in range(n):
        if not bool(valid_views[i]):
            out.append([])
            continue

        diff = centers - centers[i : i + 1]
        dist = torch.linalg.norm(diff, dim=-1)
        dist[i] = float("inf")
        dist = torch.where(valid_views, dist, torch.full_like(dist, float("inf")))

        order = torch.argsort(dist)
        neigh = []
        for j in order.tolist():
            if len(neigh) >= max_neighbors:
                break
            if j == i:
                continue
            if not torch.isfinite(dist[j]):
                continue
            neigh.append(int(j))
        out.append(neigh)
    return out


def _project_world_to_view_grid(
    points_world: torch.Tensor,
    K: torch.Tensor,
    T_c2w: torch.Tensor,
    height: int,
    width: int,
    min_depth: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project source world points into target view.

    Args:
        points_world: [1,H,W,3]
        K: [3,3]
        T_c2w: [4,4]

    Returns:
        grid: [1,H,W,2], normalized coords for grid_sample
        z_expected: [1,H,W]
        valid_projection: [1,H,W]
    """
    device = points_world.device
    dtype = points_world.dtype

    ones = torch.ones_like(points_world[..., :1])
    points_h = torch.cat([points_world, ones], dim=-1)  # [1,H,W,4]

    T_w2c = torch.linalg.inv(T_c2w.to(device=device, dtype=dtype))
    pts_cam_h = torch.einsum("ij,bhwj->bhwi", T_w2c, points_h)
    pts_cam = pts_cam_h[..., :3]

    x = pts_cam[..., 0]
    y = pts_cam[..., 1]
    z = pts_cam[..., 2]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    z_safe = torch.clamp(z, min=1e-6)
    u = fx * (x / z_safe) + cx
    v = fy * (y / z_safe) + cy

    grid_x = 2.0 * u / max(float(width - 1), 1.0) - 1.0
    grid_y = 2.0 * v / max(float(height - 1), 1.0) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)

    valid_projection = (
        torch.isfinite(grid).all(dim=-1)
        & torch.isfinite(z)
        & (z > float(min_depth))
        & (grid_x >= -1.0)
        & (grid_x <= 1.0)
        & (grid_y >= -1.0)
        & (grid_y <= 1.0)
    )

    return grid, z, valid_projection


@torch.no_grad()
def apply_mvsnet_style_multiview_filter(
    pred_maps: Sequence[np.ndarray],
    pred_valid_masks: Sequence[np.ndarray],
    pred_cams: Sequence[Dict[str, object]],
    intrinsics: Sequence[Optional[np.ndarray]],
    *,
    device: torch.device,
    enabled: bool = True,
    min_support: int = 1,
    max_neighbors: int = 4,
    conf_threshold: float = 0.0,
    depth_abs_tol: float = 0.05,
    depth_rel_tol: float = 0.02,
    point_abs_tol: float = 0.25,
    point_rel_tol: float = 0.02,
    min_depth: float = 1e-6,
    use_point_check: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    """Apply MVSNet-style multi-view consistency filtering.

    A source point is supported by a target view if:
      1. source point projects inside target image,
      2. target has a valid sampled point/depth,
      3. expected target depth matches sampled target depth,
      4. optionally, sampled target world point is close to source world point.

    final keep:
        original_mask
        & support >= min_support
        & support / evidence >= conf_threshold

    If a view has fewer than min_support neighbor views, that view is kept
    unchanged instead of being deleted entirely.
    """
    if (not enabled) or int(min_support) <= 0:
        return [np.asarray(m, dtype=bool) for m in pred_valid_masks], {
            "enabled": False,
            "reason": "disabled or min_support <= 0",
        }

    n = len(pred_maps)
    prepared = _prepare_stacks(
        pred_maps=pred_maps,
        pred_valid_masks=pred_valid_masks,
        pred_cams=pred_cams,
        intrinsics=intrinsics,
        device=device,
    )
    if prepared is None:
        return [np.asarray(m, dtype=bool) for m in pred_valid_masks], {
            "enabled": False,
            "reason": "no valid point maps",
        }

    points, masks, Ks, Ts, valid_views = prepared
    n, h, w, _ = points.shape

    if int(valid_views.sum().item()) < 2:
        return [np.asarray(m, dtype=bool) for m in pred_valid_masks], {
            "enabled": False,
            "reason": "fewer than two valid camera views",
            "valid_views": int(valid_views.sum().item()),
        }

    neighbors = _select_knn_neighbors(
        Ts=Ts,
        valid_views=valid_views,
        max_neighbors=max_neighbors,
    )

    support = torch.zeros((n, h, w), device=device, dtype=torch.float32)
    evidence = torch.zeros((n, h, w), device=device, dtype=torch.float32)

    for src_idx in range(n):
        if not bool(valid_views[src_idx]):
            continue

        src_neighbors = neighbors[src_idx]
        if len(src_neighbors) == 0:
            continue

        src_points = points[src_idx : src_idx + 1]  # [1,H,W,3]
        src_mask = masks[src_idx : src_idx + 1]     # [1,H,W]

        for tgt_idx in src_neighbors:
            tgt_points = points[tgt_idx : tgt_idx + 1]  # [1,H,W,3]
            tgt_mask = masks[tgt_idx : tgt_idx + 1]     # [1,H,W]

            grid, z_expected, valid_projection = _project_world_to_view_grid(
                points_world=src_points,
                K=Ks[tgt_idx],
                T_c2w=Ts[tgt_idx],
                height=h,
                width=w,
                min_depth=min_depth,
            )

            tgt_points_chw = tgt_points.permute(0, 3, 1, 2).contiguous()
            sampled_points = F.grid_sample(
                tgt_points_chw,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).permute(0, 2, 3, 1).contiguous()  # [1,H,W,3]

            tgt_mask_chw = tgt_mask.float().unsqueeze(1)
            sampled_mask = F.grid_sample(
                tgt_mask_chw,
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=True,
            )[:, 0] > 0.5  # [1,H,W]

            # Convert sampled target world points to target camera z.
            ones = torch.ones_like(sampled_points[..., :1])
            sampled_h = torch.cat([sampled_points, ones], dim=-1)
            T_w2c_tgt = torch.linalg.inv(Ts[tgt_idx])
            sampled_cam_h = torch.einsum(
                "ij,bhwj->bhwi", T_w2c_tgt, sampled_h
            )
            sampled_z = sampled_cam_h[..., 2]

            valid_pair = (
                src_mask
                & valid_projection
                & sampled_mask
                & torch.isfinite(z_expected)
                & torch.isfinite(sampled_z)
                & (sampled_z > float(min_depth))
                & torch.isfinite(sampled_points).all(dim=-1)
            )

            depth_tol = float(depth_abs_tol) + float(depth_rel_tol) * torch.clamp(
                torch.abs(z_expected), min=1.0
            )
            depth_ok = torch.abs(z_expected - sampled_z) <= depth_tol

            if use_point_check:
                dist3d = torch.linalg.norm(src_points - sampled_points, dim=-1)
                point_tol = float(point_abs_tol) + float(point_rel_tol) * torch.clamp(
                    torch.abs(z_expected), min=1.0
                )
                point_ok = dist3d <= point_tol
            else:
                point_ok = torch.ones_like(depth_ok, dtype=torch.bool)

            inlier = valid_pair & depth_ok & point_ok

            evidence[src_idx] += valid_pair[0].float()
            support[src_idx] += inlier[0].float()

    conf = support / torch.clamp(evidence, min=1.0)

    filtered_masks: List[np.ndarray] = []
    per_view = []
    total_before = 0
    total_after = 0

    conf_threshold = float(conf_threshold)
    min_support = int(min_support)

    for i in range(n):
        orig = masks[i]
        before = int(orig.sum().item())
        total_before += before

        if not bool(valid_views[i]):
            keep = orig
            skipped = True
            reason = "invalid camera/intrinsics/pointmap"
        elif len(neighbors[i]) < min_support:
            # Avoid deleting an entire view when it cannot be verified.
            keep = orig
            skipped = True
            reason = "not enough neighbor views"
        else:
            keep = (
                orig
                & (support[i] >= float(min_support))
                & (conf[i] >= conf_threshold)
            )
            skipped = False
            reason = ""

        after = int(keep.sum().item())
        total_after += after
        filtered_masks.append(keep.detach().cpu().numpy().astype(bool))

        valid_evidence = evidence[i][orig]
        valid_support = support[i][orig]
        per_view.append(
            {
                "view": int(i),
                "neighbors": [int(j) for j in neighbors[i]],
                "skipped": bool(skipped),
                "reason": reason,
                "before": before,
                "after": after,
                "keep_ratio": float(after / max(before, 1)),
                "mean_evidence": float(valid_evidence.mean().item())
                if valid_evidence.numel() > 0 else 0.0,
                "mean_support": float(valid_support.mean().item())
                if valid_support.numel() > 0 else 0.0,
            }
        )

    meta = {
        "enabled": True,
        "method": "mvsnet_style_pointmap_consistency",
        "num_views": int(n),
        "valid_views": int(valid_views.sum().item()),
        "min_support": int(min_support),
        "max_neighbors": int(max_neighbors),
        "conf_threshold": float(conf_threshold),
        "depth_abs_tol": float(depth_abs_tol),
        "depth_rel_tol": float(depth_rel_tol),
        "point_abs_tol": float(point_abs_tol),
        "point_rel_tol": float(point_rel_tol),
        "use_point_check": bool(use_point_check),
        "total_before": int(total_before),
        "total_after": int(total_after),
        "keep_ratio": float(total_after / max(total_before, 1)),
        "per_view": per_view,
    }
    return filtered_masks, meta