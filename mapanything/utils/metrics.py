# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Utils for metrics.

Metric design:
1. Relative metrics on normalized geometry / normalized poses.
2. Absolute metrics after exactly one global Sim(3) alignment.
3. Depth absolute metrics with sequence-level scale-only alignment.

Important rule:
- Relative pose ATE may perform an internal trajectory alignment.
- Absolute pose ATE must NOT perform any extra alignment. It must use only the
  initial global alignment that was already applied upstream.
- Optionally, an additional separately aligned absolute Pose ATE can be reported
  for protocol comparison. This is disabled by default and is not used as the
  main absolute pose metric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mapanything.utils.image import rgb


# ============================================================
# Basic helpers (numpy)
# ============================================================

def valid_mean(arr, mask, axis=None, keepdims=np._NoValue):
    mask = mask.astype(arr.dtype) if mask.dtype == bool else mask
    num_valid = np.sum(mask, axis=axis, keepdims=keepdims)
    masked_arr = arr * mask
    masked_arr_sum = np.sum(masked_arr, axis=axis, keepdims=keepdims)

    with np.errstate(divide="ignore", invalid="ignore"):
        vm = masked_arr_sum / num_valid
        is_valid = np.isfinite(vm)
        vm = np.nan_to_num(vm, nan=0, posinf=0, neginf=0)

    return vm, is_valid


def thresh_inliers(gt, pred, thresh=1.03, mask=None, output_scaling_factor=1.0):
    gt_norm = np.linalg.norm(gt, axis=-1)
    pred_norm = np.linalg.norm(pred, axis=-1)

    gt_norm_valid = gt_norm > 0
    combined_mask = (mask & gt_norm_valid) if mask is not None else gt_norm_valid

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_1 = np.nan_to_num(gt_norm / pred_norm, nan=thresh + 1, posinf=thresh + 1, neginf=thresh + 1)
        rel_2 = np.nan_to_num(pred_norm / gt_norm, nan=0, posinf=0, neginf=0)

    max_rel = np.maximum(rel_1, rel_2)
    inliers = ((0 < max_rel) & (max_rel < thresh)).astype(np.float32)

    inlier_ratio, valid = valid_mean(inliers, combined_mask)
    inlier_ratio = inlier_ratio * output_scaling_factor
    inlier_ratio = inlier_ratio if valid else np.nan
    return inlier_ratio


def m_rel_ae(gt, pred, mask=None, output_scaling_factor=1.0):
    error_norm = np.linalg.norm(pred - gt, axis=-1)
    gt_norm = np.linalg.norm(gt, axis=-1)

    gt_norm_valid = gt_norm > 0
    combined_mask = (mask & gt_norm_valid) if mask is not None else gt_norm_valid

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_ae = np.nan_to_num(error_norm / gt_norm, nan=0, posinf=0, neginf=0)

    out, valid = valid_mean(rel_ae, combined_mask)
    out = out * output_scaling_factor
    out = out if valid else np.nan
    return out


# ============================================================
# Depth metrics with sequence-level scale-only alignment
# ============================================================

def depth_abs_rel(gt, pred, mask):
    valid = mask & np.isfinite(gt) & np.isfinite(pred) & (gt > 1e-12) & (pred > 1e-12)
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(pred[valid] - gt[valid]) / gt[valid]))


def depth_rmse(gt, pred, mask):
    valid = mask & np.isfinite(gt) & np.isfinite(pred)
    if valid.sum() == 0:
        return np.nan
    err = pred[valid] - gt[valid]
    return float(np.sqrt(np.mean(err * err)))


def depth_mae(gt, pred, mask):
    valid = mask & np.isfinite(gt) & np.isfinite(pred)
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(pred[valid] - gt[valid])))


def depth_delta(gt, pred, mask, thresh=1.25):
    valid = mask & np.isfinite(gt) & np.isfinite(pred) & (gt > 1e-12) & (pred > 1e-12)
    if valid.sum() == 0:
        return np.nan
    ratio = np.maximum(gt[valid] / pred[valid], pred[valid] / gt[valid])
    return float(np.mean(ratio < thresh))


def solve_sequence_depth_scale(gt_depth_list, pr_depth_list, mask_list, max_samples_total=200000, eps=1e-12):
    """
    Solve one scale s for a multi-view set so that:
        pred_depth_aligned = s * pred_depth
    using robust median of gt/pred over all valid pixels from the set.
    """
    ratios = []
    total_valid = sum(int(m.sum()) for m in mask_list)
    if total_valid <= 0:
        return 1.0

    for gt, pr, m in zip(gt_depth_list, pr_depth_list, mask_list):
        valid = m & np.isfinite(gt) & np.isfinite(pr) & (gt > eps) & (pr > eps)
        if valid.sum() == 0:
            continue

        gt_v = gt[valid].reshape(-1)
        pr_v = pr[valid].reshape(-1)

        k = int(max_samples_total * (gt_v.shape[0] / total_valid))
        k = min(max(k, 1), gt_v.shape[0])
        if gt_v.shape[0] > k:
            sel = np.random.choice(gt_v.shape[0], k, replace=False)
            gt_v = gt_v[sel]
            pr_v = pr_v[sel]

        ratios.append(gt_v / np.maximum(pr_v, eps))

    if len(ratios) == 0:
        return 1.0

    ratios = np.concatenate(ratios, axis=0)
    if ratios.size == 0:
        return 1.0

    s = float(np.median(ratios))
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    return s


# ============================================================
# Angular error for unit ray directions
# ============================================================

def l2_distance_of_unit_ray_directions_to_angular_error(l2_distance: torch.Tensor) -> torch.Tensor:
    angular_error_radians = 2 * torch.asin(l2_distance / 2)
    angular_error_degrees = angular_error_radians * 180.0 / math.pi
    return angular_error_degrees


# ============================================================
# Pose metrics (ATE / AUC)
# ============================================================

def align(model, data):
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3, -1))
    data_zerocentered = data - data.mean(1).reshape((3, -1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1).reshape((3, -1)) - rot * model.mean(1).reshape((3, -1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]
    return rot, trans, trans_error


def align_scale_then_rt(model, data, eps: float = 1e-12, return_no_worse: bool = True):
    """
    Align a 3xN trajectory to another 3xN trajectory with Sim(3).

    The transform applied to ``model`` is:
        model_aligned = scale * rot @ model + trans

    This is the intended "scale first, then R,t" alignment for the optional
    absolute-pose ATE.  It differs from ``align`` above, which only estimates
    rigid R,t and therefore does not correct trajectory scale.

    Args:
        model: 3xN estimated/predicted trajectory points.
        data:  3xN ground-truth trajectory points.
        eps: small value for degeneracy handling.
        return_no_worse: guard against numerical/degenerate cases. When True,
            return the unaligned error if the estimated Sim(3) is worse than no
            extra alignment.

    Returns:
        scale, rot, trans, trans_error
    """
    np.set_printoptions(precision=3, suppress=True)

    model = np.asarray(model, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)

    if model.shape != data.shape or model.ndim != 2 or model.shape[0] != 3 or model.shape[1] == 0:
        scale = 1.0
        rot = np.matrix(np.identity(3))
        trans = np.zeros((3, 1), dtype=np.float64)
        return scale, rot, trans, np.asarray([], dtype=np.float64)

    no_align_error = np.linalg.norm(model - data, axis=0)

    n = model.shape[1]
    model_mean = model.mean(1).reshape((3, 1))
    data_mean = data.mean(1).reshape((3, 1))

    model_zerocentered = model - model_mean
    data_zerocentered = data - data_mean

    model_var = float(np.sum(model_zerocentered * model_zerocentered) / n)
    if not np.isfinite(model_var) or model_var <= eps:
        # Degenerate predicted trajectory. Only translation is meaningful.
        scale = 1.0
        rot = np.matrix(np.identity(3))
        trans = data_mean - model_mean
        model_aligned = model + trans
        trans_error = np.linalg.norm(model_aligned - data, axis=0)
        if return_no_worse and no_align_error.size > 0 and trans_error.mean() > no_align_error.mean():
            trans_error = no_align_error
            trans = np.zeros((3, 1), dtype=np.float64)
        return scale, rot, trans, trans_error

    # Umeyama-style proper Sim(3), mapping model -> data.  The operation order
    # is scale first, then rotation + translation: s * R @ model + t.
    covariance = (data_zerocentered @ model_zerocentered.T) / n
    U, singular_values, Vh = np.linalg.svd(covariance)

    S = np.matrix(np.identity(3))
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1

    rot = np.matrix(U) * S * np.matrix(Vh)
    scale = float(np.sum(np.asarray(S).diagonal() * singular_values) / model_var)

    if not np.isfinite(scale) or scale <= eps:
        scale = 1.0

    trans = data_mean - scale * rot * model_mean

    model_aligned = scale * rot * model + trans
    alignment_error = model_aligned - data
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    if return_no_worse and no_align_error.size > 0 and trans_error.mean() > no_align_error.mean():
        scale = 1.0
        rot = np.matrix(np.identity(3))
        trans = np.zeros((3, 1), dtype=np.float64)
        trans_error = no_align_error

    return scale, rot, trans, trans_error


def _stack_traj_points(gt_traj, est_traj):
    gt_traj_pts = [gt_traj[idx][:3, 3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3, 3] for idx in range(len(est_traj))]

    gt_traj_pts = torch.stack(gt_traj_pts).detach().cpu().numpy().T
    est_traj_pts = torch.stack(est_traj_pts).detach().cpu().numpy().T
    return gt_traj_pts, est_traj_pts


def evaluate_ate(gt_traj, est_traj, align_trajectories: bool = True):
    """
    Average translation error over a trajectory.

    Args:
        gt_traj: list[Tensor(4,4)] ground-truth poses.
        est_traj: list[Tensor(4,4)] predicted poses.
        align_trajectories:
            True  -> perform an internal rigid alignment before computing ATE.
                     This is used for relative pose evaluation.
            False -> do not apply any extra alignment. This is used for
                     absolute pose evaluation after the upstream global Sim(3).
    """
    gt_traj_pts, est_traj_pts = _stack_traj_points(gt_traj, est_traj)

    if align_trajectories:
        _, _, trans_error = align(gt_traj_pts, est_traj_pts)
    else:
        trans_error = np.linalg.norm(est_traj_pts - gt_traj_pts, axis=0)

    return float(trans_error.mean()) if trans_error.size > 0 else float("nan")


def evaluate_ate_scale_then_rt(gt_traj, est_traj, return_no_worse: bool = True):
    """
    Average translation error after an additional trajectory Sim(3) alignment.

    This function is intended for ``abs_pose_ate_separate_aligned`` after the
    upstream dense-geometry Sim(3) has already produced the absolute poses used
    by ``abs_pose_ate``. It therefore aligns ``est_traj`` directly on top of the
    same input trajectory used by ``abs_pose_ate``.

    The applied transform is scale first, then R,t:
        X_aligned = s * R @ X_est + t
    """
    gt_traj_pts, est_traj_pts = _stack_traj_points(gt_traj, est_traj)
    _, _, _, trans_error = align_scale_then_rt(
        est_traj_pts,
        gt_traj_pts,
        return_no_worse=return_no_worse,
    )
    return float(trans_error.mean()) if trans_error.size > 0 else float("nan")


def build_pair_index(N, B=1):
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    out = out[..., [1, 2, 3, 0]]
    out = standardize_quaternion(out)
    return out


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)
    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)
    return rel_rangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))
    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred) * 180.0 / np.pi
    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())
    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)
    return rel_tangle_deg


def calculate_auc_np(r_error, t_error, max_threshold=30):
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def closed_form_inverse_se3(se3, R=None, T=None):
    is_numpy = isinstance(se3, np.ndarray)
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    if is_numpy:
        R_transposed = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)
        top_right = -torch.bmm(R_transposed, T)
        inverted_matrix = torch.eye(4, 4, device=R.device, dtype=R.dtype)[None].repeat(len(R), 1, 1)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right
    return inverted_matrix


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)
    relative_pose_gt = closed_form_inverse_se3(gt_se3[pair_idx_i1]).bmm(gt_se3[pair_idx_i2])
    relative_pose_pred = closed_form_inverse_se3(pred_se3[pair_idx_i1]).bmm(pred_se3[pair_idx_i2])

    rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
    rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3])
    return rel_rangle_deg, rel_tangle_deg


@torch.no_grad()
def rotation_mae_deg(gt_se3: torch.Tensor, pred_se3: torch.Tensor) -> float:
    """
    Absolute rotation MAE in degrees.

    Important:
        gt_se3 and pred_se3 must already be in the same coordinate system.
        Do NOT align them again here.
    """
    if gt_se3.shape[0] == 0:
        return float("nan")

    r_err = rotation_angle(
        rot_gt=gt_se3[:, :3, :3],
        rot_pred=pred_se3[:, :3, :3],
    )

    return float(r_err.mean().item())


@torch.no_grad()
def pose_auc_deg(gt_se3: torch.Tensor, pred_se3: torch.Tensor, max_threshold: int = 5) -> float:
    """
    Pairwise relative pose AUC.

    This uses the poses as provided. No extra alignment is performed.
    """
    if gt_se3.shape[0] < 2:
        return float("nan")

    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
        pred_se3=pred_se3,
        gt_se3=gt_se3,
        num_frames=pred_se3.shape[0],
    )

    r_error = rel_rangle_deg.detach().cpu().numpy()
    t_error = rel_tangle_deg.detach().cpu().numpy()

    auc, _ = calculate_auc_np(r_error, t_error, max_threshold=max_threshold)
    return float(auc * 100.0)

# ============================================================
# Point cloud utils (merge / voxel / nn / ICP / Chamfer)
# ============================================================

def merge_masked_points_list(pts_list_np, rgb_list_u8, masks_list_np):
    all_pts, all_col = [], []
    for pts, col, m in zip(pts_list_np, rgb_list_u8, masks_list_np):
        if pts is None or col is None:
            continue
        valid = m.astype(bool)
        p = pts[valid].reshape(-1, 3)
        c = col[valid].reshape(-1, 3)
        if p.size > 0:
            all_pts.append(p.astype(np.float32))
            all_col.append(c.astype(np.uint8))
    if len(all_pts) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    return np.concatenate(all_pts, axis=0), np.concatenate(all_col, axis=0)


def voxel_downsample_np(points, colors_u8, voxel_size):
    if points.shape[0] == 0 or voxel_size <= 0:
        return points.astype(np.float32), colors_u8.astype(np.uint8)

    grid = np.floor(points / voxel_size).astype(np.int64)
    uniq, inv = np.unique(grid, axis=0, return_inverse=True)

    counts = np.bincount(inv)
    M = uniq.shape[0]

    pts_sum = np.zeros((M, 3), dtype=np.float64)
    col_sum = np.zeros((M, 3), dtype=np.float64)

    np.add.at(pts_sum, inv, points.astype(np.float64))
    np.add.at(col_sum, inv, colors_u8.astype(np.float64))

    pts_ds = (pts_sum / counts[:, None]).astype(np.float32)
    col_ds = np.clip(col_sum / counts[:, None], 0, 255)
    col_ds = (col_ds + 0.5).astype(np.uint8)
    return pts_ds, col_ds


@torch.no_grad()
def _nn_distance_torch(src: torch.Tensor, dst: torch.Tensor, src_chunk: int = 2048, dst_chunk: int = 2048):
    Ns = src.shape[0]
    device = src.device
    dtype = src.dtype

    best_d = torch.full((Ns,), float("inf"), device=device, dtype=dtype)
    best_j = torch.zeros((Ns,), device=device, dtype=torch.long)

    Nd = dst.shape[0]
    for i in range(0, Ns, src_chunk):
        s = src[i : i + src_chunk]
        cs = s.shape[0]

        bd = torch.full((cs,), float("inf"), device=device, dtype=dtype)
        bj = torch.zeros((cs,), device=device, dtype=torch.long)

        for j0 in range(0, Nd, dst_chunk):
            dblk = dst[j0 : j0 + dst_chunk]
            d = torch.cdist(s, dblk, p=2)
            md, mj = torch.min(d, dim=1)

            better = md < bd
            if better.any():
                bd[better] = md[better]
                bj[better] = mj[better] + j0

        best_d[i : i + cs] = bd
        best_j[i : i + cs] = bj

    return best_d, best_j


@torch.no_grad()
def kabsch_se3_torch(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-9):
    N = A.shape[0]
    if N < 3:
        R = torch.eye(3, device=A.device, dtype=A.dtype)
        t = B.mean(dim=0) - A.mean(dim=0)
        return R, t

    muA = A.mean(dim=0)
    muB = B.mean(dim=0)
    AA = A - muA
    BB = B - muB

    H = AA.t() @ BB
    U, S, Vt = torch.linalg.svd(H, full_matrices=False)
    R = Vt.t() @ U.t()

    if torch.det(R) < 0:
        Vt = Vt.clone()
        Vt[-1, :] *= -1
        R = Vt.t() @ U.t()

    t = muB - (R @ muA)
    return R, t


@torch.no_grad()
def icp_se3_torch(
    src: torch.Tensor,
    dst: torch.Tensor,
    iters: int = 20,
    max_corr_dist: float | None = None,
    nn_src_chunk: int = 2048,
    nn_dst_chunk: int = 2048,
    max_src_corr: int = 30000,
    trimmed_ratio: float | None = 0.8,
):
    device = src.device
    dtype = src.dtype

    R_tot = torch.eye(3, device=device, dtype=dtype)
    t_tot = torch.zeros(3, device=device, dtype=dtype)

    X = src
    for _ in range(iters):
        if X.shape[0] > max_src_corr:
            idx = torch.randperm(X.shape[0], device=device)[:max_src_corr]
            Xc = X[idx]
        else:
            Xc = X

        d, j = _nn_distance_torch(Xc, dst, src_chunk=nn_src_chunk, dst_chunk=nn_dst_chunk)
        Yc = dst[j]

        if max_corr_dist is not None:
            keep = d < max_corr_dist
            if keep.sum() < 50:
                break
            Xk = Xc[keep]
            Yk = Yc[keep]
            dk = d[keep]
        else:
            Xk, Yk, dk = Xc, Yc, d

        if trimmed_ratio is not None and 0 < trimmed_ratio < 1:
            m = Xk.shape[0]
            k_keep = max(int(m * trimmed_ratio), 50)
            _, order = torch.topk(dk, k=k_keep, largest=False)
            A = Xk[order]
            B = Yk[order]
        else:
            A, B = Xk, Yk

        dR, dt = kabsch_se3_torch(A, B)

        R_tot = dR @ R_tot
        t_tot = dR @ t_tot + dt
        X = (dR @ X.t()).t() + dt[None, :]

    return R_tot, t_tot, X


@torch.no_grad()
def chamfer_prf_torch(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float,
    nn_src_chunk: int = 2048,
    nn_dst_chunk: int = 2048,
    max_eval: int = 20000,
):
    """
    Fused point cloud metrics after the shared global Sim(3) alignment.

    pred, gt are already in the same GT world coordinate system.
    No extra alignment is performed here.

    Returns:
        chamfer_l1: 0.5 * (mean pred->gt NN distance + mean gt->pred NN distance)
        precision: fraction of predicted points within threshold to GT
        recall: fraction of GT points covered by prediction within threshold
        f1: harmonic mean of precision and recall
    """
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    if pred.shape[0] > max_eval:
        pred = pred[torch.randperm(pred.shape[0], device=pred.device)[:max_eval]]

    if gt.shape[0] > max_eval:
        gt = gt[torch.randperm(gt.shape[0], device=gt.device)[:max_eval]]

    d_pred_to_gt, _ = _nn_distance_torch(
        pred,
        gt,
        src_chunk=nn_src_chunk,
        dst_chunk=nn_dst_chunk,
    )

    d_gt_to_pred, _ = _nn_distance_torch(
        gt,
        pred,
        src_chunk=nn_src_chunk,
        dst_chunk=nn_dst_chunk,
    )

    chamfer_l1 = 0.5 * (d_pred_to_gt.mean() + d_gt_to_pred.mean())

    precision = (d_pred_to_gt < threshold).float().mean()
    recall = (d_gt_to_pred < threshold).float().mean()
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

    return (
        float(chamfer_l1.item()),
        float(precision.item()),
        float(recall.item()),
        float(f1.item()),
    )


# ============================================================
# Global Sim(3) from correspondences
# ============================================================

@torch.no_grad()
def umeyama_sim3_torch(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-9):
    N = A.shape[0]
    if N < 3:
        s = torch.tensor(1.0, device=A.device, dtype=A.dtype)
        R = torch.eye(3, device=A.device, dtype=A.dtype)
        t = B.mean(dim=0) - A.mean(dim=0)
        return s, R, t

    muA = A.mean(dim=0)
    muB = B.mean(dim=0)
    AA = A - muA
    BB = B - muB

    varA = (AA * AA).sum(dim=1).mean().clamp(min=eps)
    Sigma = (BB.t() @ AA) / N

    U, S, Vt = torch.linalg.svd(Sigma, full_matrices=False)
    det = torch.det(U @ Vt)
    d3 = torch.sign(det)
    D = torch.diag(torch.tensor([1.0, 1.0, d3.item()], device=A.device, dtype=A.dtype))
    R = U @ D @ Vt

    s = (S * torch.diag(D)).sum() / varA
    t = muB - s * (R @ muA)
    return s, R, t


@torch.no_grad()
def sim3_from_correspondences_robust(
    pr_corr: torch.Tensor,
    gt_corr: torch.Tensor,
    trim_ratio: float = 0.8,
    iters: int = 2,
    eps: float = 1e-9,
):
    if pr_corr is None or gt_corr is None or pr_corr.shape[0] < 10:
        device = gt_corr.device if gt_corr is not None else "cpu"
        s = torch.tensor(1.0, device=device)
        R = torch.eye(3, device=device)
        t = torch.zeros(3, device=device)
        return s, R, t

    s, R, t = umeyama_sim3_torch(pr_corr, gt_corr, eps=eps)

    if trim_ratio is None or not (0 < trim_ratio < 1):
        return s, R, t

    X = pr_corr
    Y = gt_corr
    for _ in range(max(iters, 1)):
        X_al = (s * (R @ X.t())).t() + t[None, :]
        err = torch.norm(X_al - Y, dim=1)
        N = err.shape[0]
        k = max(int(N * trim_ratio), 2000) if N >= 2000 else max(int(N * trim_ratio), 10)
        _, idx = torch.topk(err, k=k, largest=False)
        Xk, Yk = X[idx], Y[idx]
        s, R, t = umeyama_sim3_torch(Xk, Yk, eps=eps)

    return s, R, t


# ============================================================
# Misc torch helpers
# ============================================================

@torch.no_grad()
def random_sample_points_torch(x: torch.Tensor, max_n: int) -> torch.Tensor:
    if max_n is None or max_n <= 0 or x.shape[0] <= max_n:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:max_n]
    return x[idx]


# ============================================================
# High-level metrics for one set
# ============================================================

@dataclass
class FusedPCDebug:
    gt_ds: np.ndarray
    gt_colors_ds: np.ndarray
    pr_ds: np.ndarray
    pr_colors_ds: np.ndarray
    chamfer_l1: float


@torch.no_grad()
def camera_center_scale_metrics(
    gt_centers: torch.Tensor,
    pred_centers: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute trajectory-scale metrics directly from corresponding centers.

    Scale is the mean Euclidean distance from the trajectory centroid.  It is
    invariant to global translation and rotation, while deliberately retaining
    any scale mismatch. No alignment or model-provided scale value is used.
    """
    gt = torch.as_tensor(gt_centers, dtype=torch.float64).reshape(-1, 3)
    pred = torch.as_tensor(pred_centers, dtype=torch.float64).reshape(-1, 3)
    if gt.shape != pred.shape or gt.shape[0] < 2:
        return {
            "camera_scale_gt": float("nan"),
            "camera_scale_pred": float("nan"),
            "camera_scale_ratio": float("nan"),
            "camera_scale_rel_error": float("nan"),
            "camera_scale_log_error": float("nan"),
        }

    finite = torch.isfinite(gt).all(dim=-1) & torch.isfinite(pred).all(dim=-1)
    gt = gt[finite]
    pred = pred[finite]
    if gt.shape[0] < 2:
        return {
            "camera_scale_gt": float("nan"),
            "camera_scale_pred": float("nan"),
            "camera_scale_ratio": float("nan"),
            "camera_scale_rel_error": float("nan"),
            "camera_scale_log_error": float("nan"),
        }

    gt_scale = torch.linalg.vector_norm(gt - gt.mean(dim=0), dim=-1).mean()
    pred_scale = torch.linalg.vector_norm(pred - pred.mean(dim=0), dim=-1).mean()
    if gt_scale <= eps or pred_scale <= eps:
        ratio = rel_error = log_error = float("nan")
    else:
        ratio = float((pred_scale / gt_scale).item())
        rel_error = abs(ratio - 1.0)
        log_error = abs(math.log(ratio))

    return {
        "camera_scale_gt": float(gt_scale.item()),
        "camera_scale_pred": float(pred_scale.item()),
        "camera_scale_ratio": ratio,
        "camera_scale_rel_error": rel_error,
        "camera_scale_log_error": log_error,
    }


@torch.no_grad()
def compute_set_metrics(
    batch_views: List[Dict[str, Any]],
    batch_idx: int,
    gt_info: Dict[str, Any],
    pr_info: Dict[str, Any],
    valid_masks: List[torch.Tensor],
    gt_info_abs: Dict[str, Any],
    pr_info_abs: Dict[str, Any],
    scale_factors: Dict[str, torch.Tensor],
    device: torch.device,
    voxel: float = 0.1,
    icp_iters: int = 0,
    trim_ratio: float = 0.8,
    return_fused_debug: bool = False,
    compute_abs_metrics: bool = False,
    compute_separate_align_pose_ate: bool = False,
) -> Tuple[Dict[str, float], Optional[FusedPCDebug]]:
    """
    Returns metrics for one multi-view set.

    Naming convention:
    - rel_*: relative metrics on normalized geometry / poses.
    - abs_*: absolute metrics in the globally aligned world frame.
    - abs_pose_ate_separate_aligned: optional ATE after extra scale+R,t alignment on abs_pose_ate poses.
    - *_scale_aligned: evaluated after the sequence-level depth-only scale fit.
    """
    n_views = len(batch_views)

    rel_pointmap_abs_list = []
    rel_pointmap_delta_1p03_list = []
    rel_depth_abs_list = []
    rel_depth_delta_1p03_list = []
    ray_dir_mean_angle_deg_list = []
    gt_poses_rel_set = []
    pr_poses_rel_set = []

    abs_pointmap_mae_list = []
    abs_pointmap_rmse_list = []
    gt_poses_abs_set = []
    pr_poses_abs_set = []

    gt_depth_abs_list = []
    pr_depth_abs_list = []
    depth_mask_list = []

    rgb_list_u8 = []
    gt_pts_list_abs = []
    pr_pts_list_abs = []
    masks_list = []

    for view_idx in range(n_views):
        valid_mask = valid_masks[view_idx][batch_idx].cpu().numpy().astype(bool)

        gt_pts_rel = gt_info["pts3d"][view_idx][batch_idx].numpy()
        pr_pts_rel = pr_info["pts3d"][view_idx][batch_idx].numpy()
        gt_z_rel = gt_info["z_depths"][view_idx][batch_idx].numpy()
        pr_z_rel = pr_info["z_depths"][view_idx][batch_idx].numpy()

        rel_pointmap_abs_list.append(float(m_rel_ae(gt=gt_pts_rel, pred=pr_pts_rel, mask=valid_mask)))
        rel_pointmap_delta_1p03_list.append(
            float(thresh_inliers(gt=gt_pts_rel, pred=pr_pts_rel, mask=valid_mask, thresh=1.03))
        )
        rel_depth_abs_list.append(float(m_rel_ae(gt=gt_z_rel, pred=pr_z_rel, mask=valid_mask)))
        rel_depth_delta_1p03_list.append(
            float(thresh_inliers(gt=gt_z_rel, pred=pr_z_rel, mask=valid_mask, thresh=1.03))
        )

        ray_dirs_l2 = torch.norm(
            gt_info["ray_directions"][view_idx][batch_idx] - pr_info["ray_directions"][view_idx][batch_idx],
            dim=-1,
        )
        ray_dir_mean_angle_deg_list.append(
            float(l2_distance_of_unit_ray_directions_to_angular_error(ray_dirs_l2).mean().item())
        )

        gt_poses_rel_set.append(gt_info["poses"][view_idx][batch_idx])
        pr_poses_rel_set.append(pr_info["poses"][view_idx][batch_idx])

        gt_pts_abs_v = gt_info_abs["pts3d"][view_idx][batch_idx].cpu().numpy()
        pr_pts_abs_v = pr_info_abs["pts3d"][view_idx][batch_idx].cpu().numpy()

        if compute_abs_metrics:
            e3d = np.linalg.norm(pr_pts_abs_v - gt_pts_abs_v, axis=-1)
            e3d_valid = e3d[valid_mask]
            if e3d_valid.size == 0:
                abs_pointmap_mae_list.append(np.nan)
                abs_pointmap_rmse_list.append(np.nan)
            else:
                abs_pointmap_mae_list.append(float(np.mean(e3d_valid)))
                abs_pointmap_rmse_list.append(float(np.sqrt(np.mean(e3d_valid ** 2))))

        gt_poses_abs_set.append(gt_info_abs["poses"][view_idx][batch_idx])
        pr_poses_abs_set.append(pr_info_abs["poses"][view_idx][batch_idx])
        gt_z_abs_v = gt_info_abs["z_depths"][view_idx][batch_idx].cpu().numpy()[..., 0]
        pr_z_abs_v = pr_info_abs["z_depths"][view_idx][batch_idx].cpu().numpy()[..., 0]
        gt_depth_abs_list.append(gt_z_abs_v)
        pr_depth_abs_list.append(pr_z_abs_v)
        depth_mask_list.append(valid_mask)

        if compute_abs_metrics or return_fused_debug:
            masks_list.append(valid_mask)
            gt_pts_list_abs.append(gt_pts_abs_v)
            pr_pts_list_abs.append(pr_pts_abs_v)
            rgb_list_u8.append(
                (rgb(batch_views[view_idx]["img"][batch_idx], batch_views[view_idx]["data_norm_type"][batch_idx]) * 255.0)
                .astype(np.uint8)
            )

    metrics: Dict[str, float] = {}
    metrics["rel_pointmap_abs"] = float(np.nanmean(rel_pointmap_abs_list))
    metrics["rel_pointmap_delta_1p03"] = float(np.nanmean(rel_pointmap_delta_1p03_list))
    metrics["rel_depth_abs"] = float(np.nanmean(rel_depth_abs_list))
    metrics["rel_depth_delta_1p03"] = float(np.nanmean(rel_depth_delta_1p03_list))
    metrics["ray_dir_mean_angle_deg"] = float(np.nanmean(ray_dir_mean_angle_deg_list))

    pose_ate_rel = evaluate_ate(gt_traj=gt_poses_rel_set, est_traj=pr_poses_rel_set, align_trajectories=True)
    metrics["rel_pose_ate"] = float(pose_ate_rel)

    gt_poses_rel_set_t = torch.stack(gt_poses_rel_set)
    pr_poses_rel_set_t = torch.stack(pr_poses_rel_set)
    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
        pred_se3=pr_poses_rel_set_t,
        gt_se3=gt_poses_rel_set_t,
        num_frames=pr_poses_rel_set_t.shape[0],
    )
    r_error = rel_rangle_deg.cpu().numpy()
    t_error = rel_tangle_deg.cpu().numpy()
    pose_auc_5, _ = calculate_auc_np(r_error, t_error, max_threshold=5)
    metrics["rel_pose_auc_5deg"] = float(pose_auc_5 * 100.0)

    sim3_scale = float(scale_factors["pr_to_gt_scale"][batch_idx].item())
    sim3_valid = bool(scale_factors["sim3_valid"][batch_idx].item())
    pose_sim3_valid = bool(scale_factors.get("pose_sim3_valid", scale_factors["sim3_valid"])[batch_idx].item())
    metrics["sim3_scale"] = sim3_scale
    metrics["sim3_valid"] = 1.0 if sim3_valid else 0.0
    metrics["sim3_num_corr"] = float(scale_factors["sim3_num_corr"][batch_idx].item())
    metrics["sim3_median_residual"] = float(scale_factors["sim3_median_residual"][batch_idx].item())
    metrics["sim3_inlier_ratio"] = float(scale_factors["sim3_inlier_ratio"][batch_idx].item())
    metrics["points_sim3_scale"] = sim3_scale
    metrics["points_sim3_valid"] = 1.0 if sim3_valid else 0.0
    metrics["points_sim3_num_corr"] = metrics["sim3_num_corr"]
    metrics["points_sim3_median_residual"] = metrics["sim3_median_residual"]
    metrics["points_sim3_inlier_ratio"] = metrics["sim3_inlier_ratio"]
    metrics["pose_sim3_scale"] = float(
        scale_factors.get("pose_sim3_scale", scale_factors["pr_to_gt_scale"])[batch_idx].item()
    )
    metrics["pose_sim3_valid"] = 1.0 if pose_sim3_valid else 0.0
    metrics["pose_sim3_num_corr"] = float(
        scale_factors.get("pose_sim3_num_corr", scale_factors["sim3_num_corr"])[batch_idx].item()
    )
    metrics["pose_sim3_median_residual"] = float(
        scale_factors.get("pose_sim3_median_residual", scale_factors["sim3_median_residual"])[batch_idx].item()
    )
    metrics["pose_sim3_inlier_ratio"] = float(
        scale_factors.get("pose_sim3_inlier_ratio", scale_factors["sim3_inlier_ratio"])[batch_idx].item()
    )

    fused_debug: Optional[FusedPCDebug] = None

    if compute_abs_metrics:
        if sim3_valid:
            metrics["abs_pointmap_mae"] = float(np.nanmean(abs_pointmap_mae_list))
            metrics["abs_pointmap_rmse"] = float(np.nanmean(abs_pointmap_rmse_list))

            depth_scale = solve_sequence_depth_scale(gt_depth_abs_list, pr_depth_abs_list, depth_mask_list)
            abs_depth_mae_list, abs_depth_rmse_list = [], []
            abs_depth_rel_list, abs_depth_delta1_list = [], []
            for gt_d, pr_d, m in zip(gt_depth_abs_list, pr_depth_abs_list, depth_mask_list):
                pr_al = depth_scale * pr_d
                abs_depth_mae_list.append(depth_mae(gt_d, pr_al, m))
                abs_depth_rmse_list.append(depth_rmse(gt_d, pr_al, m))
                abs_depth_rel_list.append(depth_abs_rel(gt_d, pr_al, m))
                abs_depth_delta1_list.append(depth_delta(gt_d, pr_al, m, thresh=1.25))

            metrics["abs_depth_mae_scale_aligned"] = float(np.nanmean(abs_depth_mae_list))
            metrics["abs_depth_rmse_scale_aligned"] = float(np.nanmean(abs_depth_rmse_list))
            metrics["abs_depth_rel_scale_aligned"] = float(np.nanmean(abs_depth_rel_list))
            metrics["abs_depth_delta1_scale_aligned"] = float(np.nanmean(abs_depth_delta1_list))
        else:
            metrics["abs_pointmap_mae"] = float("nan")
            metrics["abs_pointmap_rmse"] = float("nan")
            metrics["abs_depth_mae_scale_aligned"] = float("nan")
            metrics["abs_depth_rmse_scale_aligned"] = float("nan")
            metrics["abs_depth_rel_scale_aligned"] = float("nan")
            metrics["abs_depth_delta1_scale_aligned"] = float("nan")

        if pose_sim3_valid:
            # Main absolute pose ATE: no extra alignment is allowed here.
            pose_ate_abs = evaluate_ate(
                gt_traj=gt_poses_abs_set,
                est_traj=pr_poses_abs_set,
                align_trajectories=False,
            )
            metrics["abs_pose_ate"] = float(pose_ate_abs)

            # Optional extra-aligned absolute pose ATE.
            # Important: this is now computed directly from the same predicted
            # poses as abs_pose_ate, i.e. pr_poses_abs_set, not from raw
            # poses_unaligned. Alignment order: scale first, then R,t.
            if compute_separate_align_pose_ate:
                try:
                    pose_ate_sep = evaluate_ate_scale_then_rt(
                        gt_traj=gt_poses_abs_set,
                        est_traj=pr_poses_abs_set,
                        return_no_worse=True,
                    )
                    metrics["abs_pose_ate_separate_aligned"] = float(pose_ate_sep)
                except Exception:
                    metrics["abs_pose_ate_separate_aligned"] = float("nan")

            gt_poses_abs_set_t = torch.stack(gt_poses_abs_set)
            pr_poses_abs_set_t = torch.stack(pr_poses_abs_set)
            metrics["abs_pose_auc_5deg"] = pose_auc_deg(
                gt_se3=gt_poses_abs_set_t,
                pred_se3=pr_poses_abs_set_t,
                max_threshold=5,
            )
            metrics["abs_pose_rot_mae_deg"] = rotation_mae_deg(
                gt_se3=gt_poses_abs_set_t,
                pred_se3=pr_poses_abs_set_t,
            )

        else:
            metrics["abs_pose_ate"] = float("nan")
            if compute_separate_align_pose_ate:
                metrics["abs_pose_ate_separate_aligned"] = float("nan")
            metrics["abs_pose_auc_5deg"] = float("nan")
            metrics["abs_pose_rot_mae_deg"] = float("nan")

    if compute_abs_metrics or return_fused_debug:
        gt_merged_abs, gt_colors = merge_masked_points_list(gt_pts_list_abs, rgb_list_u8, masks_list)
        pr_merged_abs, pr_colors = merge_masked_points_list(pr_pts_list_abs, rgb_list_u8, masks_list)

        gt_ds, gt_colors_ds = voxel_downsample_np(gt_merged_abs, gt_colors, voxel)
        pr_ds, pr_colors_ds = voxel_downsample_np(pr_merged_abs, pr_colors, voxel)

        if compute_abs_metrics:
            if sim3_valid:
                gt_t = torch.from_numpy(gt_ds).to(device=device, dtype=torch.float32)
                pr_t = torch.from_numpy(pr_ds).to(device=device, dtype=torch.float32)

                if icp_iters > 0 and gt_t.shape[0] > 50 and pr_t.shape[0] > 50:
                    icp_gate = voxel * 3.0
                    _, _, pr_t = icp_se3_torch(
                        pr_t,
                        gt_t,
                        iters=icp_iters,
                        max_corr_dist=icp_gate,
                        nn_src_chunk=2048,
                        nn_dst_chunk=2048,
                        max_src_corr=1500,
                        trimmed_ratio=trim_ratio,
                    )

                pc_threshold = voxel * 2.0
                (
                    abs_chamfer_l1,
                    abs_precision,
                    abs_recall,
                    abs_f1,
                ) = chamfer_prf_torch(
                    pr_t,
                    gt_t,
                    threshold=pc_threshold,
                    nn_src_chunk=2048,
                    nn_dst_chunk=2048,
                    max_eval=20000,
                )

                metrics["abs_fused_pc_chamfer_l1"] = (
                    float(abs_chamfer_l1) if np.isfinite(abs_chamfer_l1) else float("nan")
                )
                metrics["abs_fused_pc_precision"] = (
                    float(abs_precision) if np.isfinite(abs_precision) else float("nan")
                )
                metrics["abs_fused_pc_recall"] = (
                    float(abs_recall) if np.isfinite(abs_recall) else float("nan")
                )
                metrics["abs_fused_pc_f1"] = (
                    float(abs_f1) if np.isfinite(abs_f1) else float("nan")
                )

                if return_fused_debug:
                    fused_debug = FusedPCDebug(
                        gt_ds=gt_ds,
                        gt_colors_ds=gt_colors_ds,
                        pr_ds=pr_t.detach().cpu().numpy(),
                        pr_colors_ds=pr_colors_ds,
                        chamfer_l1=float(abs_chamfer_l1),
                    )
            else:
                metrics["abs_fused_pc_chamfer_l1"] = float("nan")
                metrics["abs_fused_pc_precision"] = float("nan")
                metrics["abs_fused_pc_recall"] = float("nan")
                metrics["abs_fused_pc_f1"] = float("nan")

                if return_fused_debug:
                    fused_debug = FusedPCDebug(
                        gt_ds=gt_ds,
                        gt_colors_ds=gt_colors_ds,
                        pr_ds=pr_ds,
                        pr_colors_ds=pr_colors_ds,
                        chamfer_l1=float("nan"),
                    )
        elif return_fused_debug:
            fused_debug = FusedPCDebug(
                gt_ds=gt_ds,
                gt_colors_ds=gt_colors_ds,
                pr_ds=pr_ds,
                pr_colors_ds=pr_colors_ds,
                chamfer_l1=float("nan"),
            )

    return metrics, fused_debug
