import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_3.utils.geometry import affine_inverse
from mapanything.utils.geometry import (
    geotrf,
    get_rays_in_world_frame,
    normalize_multiple_pointclouds,
)

try:
    import rerun as rr
except Exception:
    rr = None

EPS = 1e-8


def _resize_bhw1(x: torch.Tensor, target_hw: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(-1)
    if x.shape[-1] != 1:
        raise ValueError(f"Expected [B,H,W,1], got {tuple(x.shape)}")
    h, w = x.shape[1:3]
    th, tw = target_hw
    if (h, w) == (th, tw):
        return x
    x_nchw = x.permute(0, 3, 1, 2)
    x_nchw = F.interpolate(
        x_nchw,
        size=(th, tw),
        mode=mode,
        align_corners=False if mode != "nearest" else None,
    )
    return x_nchw.permute(0, 2, 3, 1)


def _resize_bhwc(x: torch.Tensor, target_hw: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    h, w = x.shape[1:3]
    th, tw = target_hw
    if (h, w) == (th, tw):
        return x
    x_nchw = x.permute(0, 3, 1, 2)
    x_nchw = F.interpolate(
        x_nchw,
        size=(th, tw),
        mode=mode,
        align_corners=False if mode != "nearest" else None,
    )
    return x_nchw.permute(0, 2, 3, 1)


def _resize_mask(mask: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    h, w = mask.shape[-2:]
    th, tw = target_hw
    if (h, w) == (th, tw):
        return mask
    x = mask.float().unsqueeze(1)
    x = F.interpolate(x, size=(th, tw), mode="nearest")
    return x[:, 0] > 0.5


def _pad_to_4x4(mat: torch.Tensor) -> torch.Tensor:
    if mat.shape[-2:] == (4, 4):
        return mat
    if mat.shape[-2:] == (3, 4):
        out = torch.zeros(*mat.shape[:-2], 4, 4, device=mat.device, dtype=mat.dtype)
        out[..., :3, :] = mat
        out[..., 3, 3] = 1.0
        return out
    raise ValueError(f"Unexpected matrix shape: {tuple(mat.shape)}")


def _intrinsics_to_fov(intrinsics: torch.Tensor, height: int, width: int) -> torch.Tensor:
    fx = intrinsics[..., 0, 0].clamp_min(EPS)
    fy = intrinsics[..., 1, 1].clamp_min(EPS)
    width_t = torch.as_tensor(width, device=intrinsics.device, dtype=intrinsics.dtype)
    height_t = torch.as_tensor(height, device=intrinsics.device, dtype=intrinsics.dtype)
    fov_x = 2.0 * torch.atan(width_t / (2.0 * fx))
    fov_y = 2.0 * torch.atan(height_t / (2.0 * fy))
    return torch.stack([fov_x, fov_y], dim=-1)


def _transform_dirs(rot: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,bhwj->bhwi", rot, dirs)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    if x.dim() == mask.dim() + 1:
        mask = mask.unsqueeze(-1)
    mask = mask.float()
    denom = mask.sum().clamp_min(eps)
    return (x * mask).sum() / denom


def _masked_smooth_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, gt, reduction="none", beta=beta)
    return _masked_mean(loss, mask)


def _matrix_to_quaternion(rot: torch.Tensor) -> torch.Tensor:
    m = rot
    batch_shape = m.shape[:-2]
    q = torch.empty(*batch_shape, 4, device=m.device, dtype=m.dtype)

    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cond = trace > 0

    if cond.any():
        s = torch.sqrt((trace[cond] + 1.0).clamp_min(EPS)) * 2.0
        q[cond, 0] = 0.25 * s
        q[cond, 1] = (m[cond, 2, 1] - m[cond, 1, 2]) / s
        q[cond, 2] = (m[cond, 0, 2] - m[cond, 2, 0]) / s
        q[cond, 3] = (m[cond, 1, 0] - m[cond, 0, 1]) / s

    cond1 = (~cond) & (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2])
    if cond1.any():
        s = torch.sqrt((1.0 + m[cond1, 0, 0] - m[cond1, 1, 1] - m[cond1, 2, 2]).clamp_min(EPS)) * 2.0
        q[cond1, 0] = (m[cond1, 2, 1] - m[cond1, 1, 2]) / s
        q[cond1, 1] = 0.25 * s
        q[cond1, 2] = (m[cond1, 0, 1] + m[cond1, 1, 0]) / s
        q[cond1, 3] = (m[cond1, 0, 2] + m[cond1, 2, 0]) / s

    cond2 = (~cond) & (~cond1) & (m[..., 1, 1] > m[..., 2, 2])
    if cond2.any():
        s = torch.sqrt((1.0 + m[cond2, 1, 1] - m[cond2, 0, 0] - m[cond2, 2, 2]).clamp_min(EPS)) * 2.0
        q[cond2, 0] = (m[cond2, 0, 2] - m[cond2, 2, 0]) / s
        q[cond2, 1] = (m[cond2, 0, 1] + m[cond2, 1, 0]) / s
        q[cond2, 2] = 0.25 * s
        q[cond2, 3] = (m[cond2, 1, 2] + m[cond2, 2, 1]) / s

    cond3 = (~cond) & (~cond1) & (~cond2)
    if cond3.any():
        s = torch.sqrt((1.0 + m[cond3, 2, 2] - m[cond3, 0, 0] - m[cond3, 1, 1]).clamp_min(EPS)) * 2.0
        q[cond3, 0] = (m[cond3, 1, 0] - m[cond3, 0, 1]) / s
        q[cond3, 1] = (m[cond3, 0, 2] + m[cond3, 2, 0]) / s
        q[cond3, 2] = (m[cond3, 1, 2] + m[cond3, 2, 1]) / s
        q[cond3, 3] = 0.25 * s

    q = q / q.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return q


def _quat_huber_loss(pred_q: torch.Tensor, gt_q: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    pred_q = pred_q / pred_q.norm(dim=-1, keepdim=True).clamp_min(EPS)
    gt_q = gt_q / gt_q.norm(dim=-1, keepdim=True).clamp_min(EPS)
    loss_pos = F.smooth_l1_loss(pred_q, gt_q, reduction="none", beta=beta).mean(dim=-1)
    loss_neg = F.smooth_l1_loss(pred_q, -gt_q, reduction="none", beta=beta).mean(dim=-1)
    return torch.minimum(loss_pos, loss_neg).mean()


def _normalized_intrinsics_repr(intrinsics: torch.Tensor, height: int, width: int) -> torch.Tensor:
    width_t = torch.as_tensor(width, device=intrinsics.device, dtype=intrinsics.dtype)
    height_t = torch.as_tensor(height, device=intrinsics.device, dtype=intrinsics.dtype)
    fx = intrinsics[..., 0, 0] / width_t
    fy = intrinsics[..., 1, 1] / height_t
    cx = intrinsics[..., 0, 2] / width_t
    cy = intrinsics[..., 1, 2] / height_t
    return torch.stack([fx, fy, cx, cy], dim=-1)


def _finite_diff_x(x: torch.Tensor) -> torch.Tensor:
    return x[..., :, 1:] - x[..., :, :-1]


def _finite_diff_y(x: torch.Tensor) -> torch.Tensor:
    return x[..., 1:, :] - x[..., :-1, :]


def _gradient_matching_depth_loss(
    pred_depth_z: torch.Tensor,
    gt_depth_z: torch.Tensor,
    valid_mask: torch.Tensor,
    scales=(1, 2, 4),
) -> torch.Tensor:
    if pred_depth_z.dim() == 4:
        pred_depth_z = pred_depth_z[..., 0]
    if gt_depth_z.dim() == 4:
        gt_depth_z = gt_depth_z[..., 0]

    pred_depth = pred_depth_z.clamp_min(EPS)
    gt_depth = gt_depth_z.clamp_min(EPS)
    total = pred_depth.new_tensor(0.0)
    count = 0

    for s in scales:
        if s > 1:
            pred_s = pred_depth[:, ::s, ::s]
            gt_s = gt_depth[:, ::s, ::s]
            mask_s = valid_mask[:, ::s, ::s]
        else:
            pred_s = pred_depth
            gt_s = gt_depth
            mask_s = valid_mask

        mask_x = mask_s[:, :, 1:] & mask_s[:, :, :-1]
        mask_y = mask_s[:, 1:, :] & mask_s[:, :-1, :]

        grad_pred_x = _finite_diff_x(pred_s)
        grad_gt_x = _finite_diff_x(gt_s)
        grad_pred_y = _finite_diff_y(pred_s)
        grad_gt_y = _finite_diff_y(gt_s)

        loss_x = _masked_mean(torch.abs(grad_pred_x - grad_gt_x), mask_x)
        loss_y = _masked_mean(torch.abs(grad_pred_y - grad_gt_y), mask_y)
        total = total + loss_x + loss_y
        count += 2

    return total / max(count, 1)


def _confidence_aware_depth_loss(
    pred_depth_z: torch.Tensor,
    gt_depth_z: torch.Tensor,
    conf: torch.Tensor,
    valid_mask: torch.Tensor,
    beta: float = 0.05,
    conf_reg_alpha: float = 0.1,
) -> torch.Tensor:
    if pred_depth_z.dim() == 4:
        pred_depth_z = pred_depth_z[..., 0]
    if gt_depth_z.dim() == 4:
        gt_depth_z = gt_depth_z[..., 0]
    if conf.dim() == 4:
        conf = conf[..., 0]

    pred = pred_depth_z.clamp_min(EPS)
    gt = gt_depth_z.clamp_min(EPS)
    residual = F.smooth_l1_loss(pred, gt, reduction="none", beta=beta)

    conf = conf.clamp_min(EPS)
    loss_map = conf * residual - conf_reg_alpha * torch.log(conf)
    return _masked_mean(loss_map, valid_mask)

def _masked_batch_smooth_l1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    sample_mask: torch.Tensor,
    beta: float = 0.05,
) -> torch.Tensor:
    """
    pred, gt: [B, ...]
    sample_mask: [B] bool
    """
    loss = F.smooth_l1_loss(pred, gt, reduction="none", beta=beta)
    mask = sample_mask.float()
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(EPS)
    return (loss * mask).sum() / denom


def _build_scale_ok_mask(
    gt_scale: torch.Tensor,
    pr_scale: torch.Tensor,
    scale_valid_min: float,
    scale_valid_max: float,
) -> torch.Tensor:
    """
    gt_scale, pr_scale: [B]
    return: [B] bool
    """
    gt_ok = torch.isfinite(gt_scale) & (gt_scale > scale_valid_min) & (gt_scale < scale_valid_max)
    pr_ok = torch.isfinite(pr_scale) & (pr_scale > scale_valid_min) & (pr_scale < scale_valid_max)
    return gt_ok & pr_ok


def _make_c2w_norm(c2w_raw: torch.Tensor, trans_norm: torch.Tensor) -> torch.Tensor:
    out = c2w_raw.clone()
    out[:, :3, 3] = trans_norm
    return out


def _tensor_to_np(x: torch.Tensor):
    return x.detach().float().cpu().numpy()


class DA3FineTuneLoss(nn.Module):
    def __init__(
        self,
        lambda_depth=1.0,
        lambda_depth_grad=1.0,
        lambda_ray=1.0,
        lambda_ray_origin=1.0,
        lambda_ray_direction=1.0,
        lambda_point=1.0,
        lambda_camera=1.0,
        lambda_camera_rot=1.0,
        lambda_camera_trans=1.0,
        lambda_camera_intr=1.0,
        lambda_camera_fov=1.0,
        norm_mode="avg_dis",
        robust_beta=0.05,
        conf_reg_alpha=0.1,
        grad_scales=(1, 2, 4),
        scale_valid_min=0.01,
        scale_valid_max=1000,
        debug_rrd=False,
        debug_rrd_dir="./loss_rrd_debug",
        debug_rrd_every=0,
        debug_rrd_on_large_loss=True,
        debug_rrd_loss_threshold=1000.0,
        debug_rrd_max_samples=1,
        debug_rrd_max_views=2,
    ):
        super().__init__()
        self.lambda_depth = lambda_depth
        self.lambda_depth_grad = lambda_depth_grad
        self.lambda_ray = lambda_ray
        self.lambda_ray_origin = lambda_ray_origin
        self.lambda_ray_direction = lambda_ray_direction
        self.lambda_point = lambda_point
        self.lambda_camera = lambda_camera
        self.lambda_camera_rot = lambda_camera_rot
        self.lambda_camera_trans = lambda_camera_trans
        self.lambda_camera_intr = lambda_camera_intr
        self.lambda_camera_fov = lambda_camera_fov
        self.norm_mode = norm_mode
        self.robust_beta = robust_beta
        self.conf_reg_alpha = conf_reg_alpha
        self.grad_scales = grad_scales

        self.debug_rrd = debug_rrd
        self.debug_rrd_dir = debug_rrd_dir
        self.debug_rrd_every = debug_rrd_every
        self.debug_rrd_on_large_loss = debug_rrd_on_large_loss
        self.debug_rrd_loss_threshold = debug_rrd_loss_threshold
        self.debug_rrd_max_samples = debug_rrd_max_samples
        self.debug_rrd_max_views = debug_rrd_max_views
        self._debug_call_idx = 0
        
        self.scale_valid_min = scale_valid_min
        self.scale_valid_max = scale_valid_max

    def _build_gt_info(self, batch, height, width):
        n_views = len(batch)
        device = batch[0]["img"].device

        gt_c2w_all = torch.stack([view["camera_pose"].to(device) for view in batch], dim=1)
        gt_w2c_all = affine_inverse(gt_c2w_all)
        gt_w2c0 = gt_w2c_all[:, 0]

        no_norm_gt_pts = []
        no_norm_gt_depth_z = []
        no_norm_gt_ray_map = []
        no_norm_gt_pose_trans = []
        gt_c2w_in_view0 = []
        gt_intrinsics = []
        gt_fov = []
        valid_masks = []

        for i, view in enumerate(batch):
            valid = view["valid_mask"].to(device).bool()
            if "non_ambiguous_mask" in view:
                valid = valid & view["non_ambiguous_mask"].to(device).bool()
            valid_masks.append(valid)

            gt_pts_world = view["pts3d"].to(device)
            gt_pts_v0 = geotrf(gt_w2c0, gt_pts_world)
            no_norm_gt_pts.append(gt_pts_v0)

            gt_pts_cam_i = geotrf(gt_w2c_all[:, i], gt_pts_world)
            gt_depth_z = gt_pts_cam_i[..., 2:3].clamp_min(EPS)
            no_norm_gt_depth_z.append(gt_depth_z)

            gt_c2w_i_v0 = gt_w2c0 @ gt_c2w_all[:, i]
            gt_c2w_in_view0.append(gt_c2w_i_v0)
            no_norm_gt_pose_trans.append(gt_c2w_i_v0[:, :3, 3])

            gt_ray_o_v0, gt_ray_d_v0 = get_rays_in_world_frame(
                intrinsics=view["camera_intrinsics"].to(device),
                height=height,
                width=width,
                normalize_to_unit_sphere=False,
                camera_pose=gt_c2w_i_v0,
            )
            no_norm_gt_ray_map.append(torch.cat([gt_ray_o_v0, gt_ray_d_v0], dim=-1))
            gt_intrinsics.append(view["camera_intrinsics"].to(device))
            gt_fov.append(_intrinsics_to_fov(view["camera_intrinsics"].to(device), height, width))
        
        gt_norm_output = normalize_multiple_pointclouds(
            no_norm_gt_pts,
            valid_masks,
            self.norm_mode,
            ret_factor=True,
        )
        gt_pts_norm = gt_norm_output[:-1]
        gt_norm_factor = gt_norm_output[-1]   # [B,1,1,1]

        gt_depth_z_norm = []
        gt_ray_map_norm = []
        gt_pose_trans_norm = []
        gt_quat = []
        gt_intr_norm = []

        for i in range(n_views):
            scale = gt_norm_factor[:, 0, 0, 0][:, None, None, None]
            gt_depth_z_norm.append(no_norm_gt_depth_z[i] / scale)

            gt_ray_o = no_norm_gt_ray_map[i][..., :3] / scale
            gt_ray_d = no_norm_gt_ray_map[i][..., 3:]
            gt_ray_map_norm.append(torch.cat([gt_ray_o, gt_ray_d], dim=-1))

            gt_pose_trans_norm.append(no_norm_gt_pose_trans[i] / gt_norm_factor[:, 0, 0, 0][:, None])
            gt_quat.append(_matrix_to_quaternion(gt_c2w_in_view0[i][:, :3, :3]))
            gt_intr_norm.append(_normalized_intrinsics_repr(gt_intrinsics[i], height, width))

        return {
            "pts_raw": no_norm_gt_pts,
            "pts_norm": gt_pts_norm,
            "depth_z_raw": no_norm_gt_depth_z,
            "depth_z_norm": gt_depth_z_norm,
            "ray_map_raw": no_norm_gt_ray_map,
            "ray_map_norm": gt_ray_map_norm,
            "c2w_view0": gt_c2w_in_view0,
            "quat_view0": gt_quat,
            "pose_trans_raw": no_norm_gt_pose_trans,
            "pose_trans_norm": gt_pose_trans_norm,
            "intrinsics": gt_intrinsics,
            "intr_norm": gt_intr_norm,
            "fov": gt_fov,
            "scale_factor": gt_norm_factor[:, 0, 0, 0],
            "valid_masks": valid_masks,
        }

    def _build_pred_info(self, preds, valid_masks, height, width):
        """
        NOTE:
        Here we assume prediction outputs are ALREADY expressed in the predicted view0 frame.
        Therefore, we must NOT transform pts/rays/poses by pred_w2c0 again.
        """
        n_views = len(preds)

        # pred_extrinsics are assumed to be w2c in predicted view0 frame already
        pred_w2c_all = torch.stack(
            [_pad_to_4x4(preds[i]["pred_extrinsics"]) for i in range(n_views)],
            dim=1,
        )
        pred_c2w_all = affine_inverse(pred_w2c_all)

        no_norm_pr_pts = []
        no_norm_pr_depth_z = []
        no_norm_pr_ray_map = []
        no_norm_pr_pose_trans = []
        pr_c2w_in_view0 = []
        pr_intrinsics = []
        pr_intr_norm = []
        pr_fov = []
        pr_conf = []

        for i in range(n_views):
            # ---------------------------------------------------------
            # preds are already in predicted view0 frame
            # ---------------------------------------------------------
            pr_pts_v0 = preds[i]["pts3d"]
            no_norm_pr_pts.append(pr_pts_v0)

            no_norm_pr_depth_z.append(preds[i]["depth_z"])
            pr_conf.append(preds[i]["conf"])

            pr_ray_o_v0 = preds[i]["ray_origins"]
            pr_ray_d_v0 = preds[i]["ray_directions"]
            no_norm_pr_ray_map.append(torch.cat([pr_ray_o_v0, pr_ray_d_v0], dim=-1))

            # pred pose is already pose-in-view0-frame
            pr_c2w_i_v0 = pred_c2w_all[:, i]
            pr_c2w_in_view0.append(pr_c2w_i_v0)
            no_norm_pr_pose_trans.append(pr_c2w_i_v0[:, :3, 3])

            pr_intrinsics.append(preds[i]["pred_intrinsics"])
            pr_intr_norm.append(
                _normalized_intrinsics_repr(preds[i]["pred_intrinsics"], height, width)
            )
            pr_fov.append(preds[i]["pred_fov"])

        pr_norm_output = normalize_multiple_pointclouds(
            no_norm_pr_pts,
            valid_masks,
            self.norm_mode,
            ret_factor=True,
        )
        pr_pts_norm = pr_norm_output[:-1]
        pr_norm_factor = pr_norm_output[-1]

        pr_depth_z_norm = []
        pr_ray_map_norm = []
        pr_pose_trans_norm = []
        pr_quat = []

        for i in range(n_views):
            scale = pr_norm_factor[:, 0, 0, 0][:, None, None, None]
            pr_depth_z_norm.append(no_norm_pr_depth_z[i] / scale)

            pr_ray_o = no_norm_pr_ray_map[i][..., :3] / scale
            pr_ray_d = no_norm_pr_ray_map[i][..., 3:]
            pr_ray_map_norm.append(torch.cat([pr_ray_o, pr_ray_d], dim=-1))

            pr_pose_trans_norm.append(
                no_norm_pr_pose_trans[i] / pr_norm_factor[:, 0, 0, 0][:, None]
            )
            pr_quat.append(_matrix_to_quaternion(pr_c2w_in_view0[i][:, :3, :3]))

        return {
            "pts_raw": no_norm_pr_pts,
            "pts_norm": pr_pts_norm,
            "depth_z_raw": no_norm_pr_depth_z,
            "depth_z_norm": pr_depth_z_norm,
            "ray_map_raw": no_norm_pr_ray_map,
            "ray_map_norm": pr_ray_map_norm,
            "c2w_view0": pr_c2w_in_view0,
            "quat_view0": pr_quat,
            "pose_trans_raw": no_norm_pr_pose_trans,
            "pose_trans_norm": pr_pose_trans_norm,
            "intrinsics": pr_intrinsics,
            "intr_norm": pr_intr_norm,
            "fov": pr_fov,
            "conf": pr_conf,
            "scale_factor": pr_norm_factor[:, 0, 0, 0],
        }

    def _should_dump_rrd(self, total_loss_value: float) -> bool:
        if not self.debug_rrd:
            return False
        should = False
        if self.debug_rrd_every > 0 and (self._debug_call_idx % self.debug_rrd_every == 0):
            should = True
        if self.debug_rrd_on_large_loss and total_loss_value > self.debug_rrd_loss_threshold:
            should = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            should = should and (torch.distributed.get_rank() == 0)
        return should

    def _log_one_camera(self, base_name: str, c2w: torch.Tensor, intrinsics: torch.Tensor, depth: torch.Tensor, pts3d: torch.Tensor, valid_mask: torch.Tensor):
        if rr is None:
            return
        rr.log(
            base_name,
            rr.Transform3D(
                translation=_tensor_to_np(c2w[:3, 3]),
                mat3x3=_tensor_to_np(c2w[:3, :3]),
            ),
        )
        h, w = depth.shape[:2]
        rr.log(
            f"{base_name}/pinhole",
            rr.Pinhole(
                image_from_camera=_tensor_to_np(intrinsics),
                height=h,
                width=w,
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )
        rr.log(f"{base_name}/pinhole/depth_z", rr.DepthImage(_tensor_to_np(depth)))
        rr.log(
            f"{base_name}/pinhole/valid_mask",
            rr.SegmentationImage(_tensor_to_np(valid_mask.to(torch.uint8))),
        )
        pts = pts3d[valid_mask]
        if pts.numel() > 0:
            rr.log(
                f"{base_name}_pointcloud",
                rr.Points3D(positions=_tensor_to_np(pts.reshape(-1, 3))),
            )

    def _dump_rrd_debug(self, batch, gt, pr, details, total_loss_value: float, height: int, width: int):
        if rr is None:
            print("[DA3FineTuneLoss] rerun is not installed, skip RRD dump.")
            return

        out_dir = Path(self.debug_rrd_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rrd_path = out_dir / f"loss_debug_step_{self._debug_call_idx:07d}.rrd"
        meta_path = out_dir / f"loss_debug_step_{self._debug_call_idx:07d}.json"

        rr.init(f"DA3LossDebug_{self._debug_call_idx}", spawn=False)
        rr.save(str(rrd_path))
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

        batch_size = min(self.debug_rrd_max_samples, batch[0]["img"].shape[0])
        num_views = min(self.debug_rrd_max_views, len(batch))

        meta = {
            "step": int(self._debug_call_idx),
            "total_loss": float(total_loss_value),
            "gt_scale_factor": [_tensor_to_np(gt["scale_factor"]).tolist()],
            "pred_scale_factor": [_tensor_to_np(pr["scale_factor"]).tolist()],
            "details": details,
        }

        for b in range(batch_size):
            rr.set_time("sample", sequence=b)
            for i in range(num_views):
                valid = gt["valid_masks"][i][b]

                gt_c2w_raw = gt["c2w_view0"][i][b]
                gt_c2w_norm = _make_c2w_norm(gt["c2w_view0"][i][b:b+1], gt["pose_trans_norm"][i][b:b+1])[0]
                pr_c2w_raw = pr["c2w_view0"][i][b]
                pr_c2w_norm = _make_c2w_norm(pr["c2w_view0"][i][b:b+1], pr["pose_trans_norm"][i][b:b+1])[0]

                self._log_one_camera(
                    f"sample_{b}/gt_raw/view_{i}",
                    gt_c2w_raw,
                    gt["intrinsics"][i][b],
                    gt["depth_z_raw"][i][b, ..., 0],
                    gt["pts_raw"][i][b],
                    valid,
                )
                self._log_one_camera(
                    f"sample_{b}/gt_norm/view_{i}",
                    gt_c2w_norm,
                    gt["intrinsics"][i][b],
                    gt["depth_z_norm"][i][b, ..., 0],
                    gt["pts_norm"][i][b],
                    valid,
                )
                self._log_one_camera(
                    f"sample_{b}/pred_raw/view_{i}",
                    pr_c2w_raw,
                    pr["intrinsics"][i][b],
                    pr["depth_z_raw"][i][b, ..., 0],
                    pr["pts_raw"][i][b],
                    valid,
                )
                self._log_one_camera(
                    f"sample_{b}/pred_norm/view_{i}",
                    pr_c2w_norm,
                    pr["intrinsics"][i][b],
                    pr["depth_z_norm"][i][b, ..., 0],
                    pr["pts_norm"][i][b],
                    valid,
                )

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[DA3FineTuneLoss] Saved RRD debug to: {rrd_path}")
        print(f"[DA3FineTuneLoss] Saved JSON debug to: {meta_path}")

    def forward(self, batch, preds, **kwargs):
        self._debug_call_idx += 1
        _, _, height, width = batch[0]["img"].shape
        gt = self._build_gt_info(batch, height, width)
        pr = self._build_pred_info(preds, gt["valid_masks"], height, width)

        gt_scale = gt["scale_factor"]   # [B]
        pr_scale = pr["scale_factor"]   # [B]

        scale_ok_mask = _build_scale_ok_mask(
            gt_scale,
            pr_scale,
            self.scale_valid_min,
            self.scale_valid_max,
        )

        total_loss = preds[0]["depth_z"].new_tensor(0.0)
        details = {}

        details["scale_valid_min"] = float(self.scale_valid_min)
        details["scale_valid_max"] = float(self.scale_valid_max)
        details["gt_scale_min"] = float(gt_scale.min().detach())
        details["gt_scale_max"] = float(gt_scale.max().detach())
        details["pred_scale_min"] = float(pr_scale.min().detach())
        details["pred_scale_max"] = float(pr_scale.max().detach())
        details["scale_ok_count"] = int(scale_ok_mask.sum().item())
        details["scale_skip_count"] = int((~scale_ok_mask).sum().item())

        sum_depth = 0.0
        sum_depth_w = 0.0
        sum_grad = 0.0
        sum_grad_w = 0.0
        sum_ray_origin = 0.0
        sum_ray_origin_w = 0.0
        sum_ray_direction = 0.0
        sum_ray_direction_w = 0.0
        sum_ray = 0.0
        sum_ray_w = 0.0
        sum_point = 0.0
        sum_point_w = 0.0
        sum_camera_rot = 0.0
        sum_camera_rot_w = 0.0
        sum_camera_trans = 0.0
        sum_camera_trans_w = 0.0
        sum_camera_intr = 0.0
        sum_camera_intr_w = 0.0
        sum_camera_fov = 0.0
        sum_camera_fov_w = 0.0
        sum_camera = 0.0
        sum_camera_w = 0.0
        sum_view_total = 0.0

        n_views = len(batch)

        for i in range(n_views):
            valid = gt["valid_masks"][i]
            valid_scaled = valid & scale_ok_mask[:, None, None]

            l_depth = _confidence_aware_depth_loss(
                pr["depth_z_norm"][i],
                gt["depth_z_norm"][i],
                pr["conf"][i],
                valid_scaled,
                beta=self.robust_beta,
                conf_reg_alpha=self.conf_reg_alpha,
            )
            l_grad = _gradient_matching_depth_loss(
                pr["depth_z_norm"][i],
                gt["depth_z_norm"][i],
                valid_scaled,
                scales=self.grad_scales,
            )

            l_ray_origin = _masked_smooth_l1(
                pr["ray_map_norm"][i][..., :3],
                gt["ray_map_norm"][i][..., :3],
                valid_scaled,
                beta=self.robust_beta,
            )
            l_ray_direction = _masked_smooth_l1(
                pr["ray_map_norm"][i][..., 3:],
                gt["ray_map_norm"][i][..., 3:],
                valid,
                beta=self.robust_beta,
            )
            l_ray = self.lambda_ray_origin * l_ray_origin + self.lambda_ray_direction * l_ray_direction

            l_point = _masked_smooth_l1(
                pr["pts_norm"][i],
                gt["pts_norm"][i],
                valid_scaled,
                beta=self.robust_beta,
            )

            l_rot = _quat_huber_loss(pr["quat_view0"][i], gt["quat_view0"][i], beta=self.robust_beta)
            l_trans = _masked_batch_smooth_l1(
                pr["pose_trans_norm"][i],
                gt["pose_trans_norm"][i],
                scale_ok_mask,
                beta=self.robust_beta,
            )
            # l_intr = F.smooth_l1_loss(
            #     pr["intr_norm"][i],
            #     gt["intr_norm"][i],
            #     reduction="mean",
            #     beta=self.robust_beta,
            # )
            l_fov = F.smooth_l1_loss(
                pr["fov"][i],
                gt["fov"][i],
                reduction="mean",
                beta=self.robust_beta,
            )
            l_camera = (
                self.lambda_camera_rot * l_rot
                + self.lambda_camera_trans * l_trans
                # + self.lambda_camera_intr * l_intr
                + self.lambda_camera_fov * l_fov
            )

            l_depth_w = self.lambda_depth * l_depth
            l_grad_w = self.lambda_depth_grad * l_grad
            l_ray_origin_w = self.lambda_ray * self.lambda_ray_origin * l_ray_origin
            l_ray_direction_w = self.lambda_ray * self.lambda_ray_direction * l_ray_direction
            l_ray_w = self.lambda_ray * l_ray
            l_point_w = self.lambda_point * l_point
            l_rot_w = self.lambda_camera * self.lambda_camera_rot * l_rot
            l_trans_w = self.lambda_camera * self.lambda_camera_trans * l_trans
            # l_intr_w = self.lambda_camera * self.lambda_camera_intr * l_intr
            l_fov_w = self.lambda_camera * self.lambda_camera_fov * l_fov
            l_camera_w = self.lambda_camera * l_camera

            view_loss = l_depth_w + l_grad_w + l_ray_w + l_point_w + l_camera_w
            total_loss = total_loss + view_loss

            details[f"da3_depth_view{i+1}"] = float(l_depth.detach())
            details[f"da3_grad_view{i+1}"] = float(l_grad.detach())
            details[f"da3_ray_origin_view{i+1}"] = float(l_ray_origin.detach())
            details[f"da3_ray_direction_view{i+1}"] = float(l_ray_direction.detach())
            details[f"da3_ray_view{i+1}"] = float(l_ray.detach())
            details[f"da3_point_view{i+1}"] = float(l_point.detach())
            details[f"da3_camera_rot_view{i+1}"] = float(l_rot.detach())
            details[f"da3_camera_trans_view{i+1}"] = float(l_trans.detach())
            # details[f"da3_camera_intr_view{i+1}"] = float(l_intr.detach())
            details[f"da3_camera_fov_view{i+1}"] = float(l_fov.detach())
            details[f"da3_camera_view{i+1}"] = float(l_camera.detach())
            details[f"da3_depth_weighted_view{i+1}"] = float(l_depth_w.detach())
            details[f"da3_grad_weighted_view{i+1}"] = float(l_grad_w.detach())
            details[f"da3_ray_origin_weighted_view{i+1}"] = float(l_ray_origin_w.detach())
            details[f"da3_ray_direction_weighted_view{i+1}"] = float(l_ray_direction_w.detach())
            details[f"da3_ray_weighted_view{i+1}"] = float(l_ray_w.detach())
            details[f"da3_point_weighted_view{i+1}"] = float(l_point_w.detach())
            details[f"da3_camera_rot_weighted_view{i+1}"] = float(l_rot_w.detach())
            details[f"da3_camera_trans_weighted_view{i+1}"] = float(l_trans_w.detach())
            # details[f"da3_camera_intr_weighted_view{i+1}"] = float(l_intr_w.detach())
            details[f"da3_camera_fov_weighted_view{i+1}"] = float(l_fov_w.detach())
            details[f"da3_camera_weighted_view{i+1}"] = float(l_camera_w.detach())
            details[f"da3_total_view{i+1}"] = float(view_loss.detach())

            sum_depth += float(l_depth.detach())
            sum_depth_w += float(l_depth_w.detach())
            sum_grad += float(l_grad.detach())
            sum_grad_w += float(l_grad_w.detach())
            sum_ray_origin += float(l_ray_origin.detach())
            sum_ray_origin_w += float(l_ray_origin_w.detach())
            sum_ray_direction += float(l_ray_direction.detach())
            sum_ray_direction_w += float(l_ray_direction_w.detach())
            sum_ray += float(l_ray.detach())
            sum_ray_w += float(l_ray_w.detach())
            sum_point += float(l_point.detach())
            sum_point_w += float(l_point_w.detach())
            sum_camera_rot += float(l_rot.detach())
            sum_camera_rot_w += float(l_rot_w.detach())
            sum_camera_trans += float(l_trans.detach())
            sum_camera_trans_w += float(l_trans_w.detach())
            # sum_camera_intr += float(l_intr.detach())
            # sum_camera_intr_w += float(l_intr_w.detach())
            sum_camera_fov += float(l_fov.detach())
            sum_camera_fov_w += float(l_fov_w.detach())
            sum_camera += float(l_camera.detach())
            sum_camera_w += float(l_camera_w.detach())
            sum_view_total += float(view_loss.detach())

        details["da3_depth_sum"] = sum_depth
        details["da3_depth_weighted_sum"] = sum_depth_w
        details["da3_grad_sum"] = sum_grad
        details["da3_grad_weighted_sum"] = sum_grad_w
        details["da3_ray_origin_sum"] = sum_ray_origin
        details["da3_ray_origin_weighted_sum"] = sum_ray_origin_w
        details["da3_ray_direction_sum"] = sum_ray_direction
        details["da3_ray_direction_weighted_sum"] = sum_ray_direction_w
        details["da3_ray_sum"] = sum_ray
        details["da3_ray_weighted_sum"] = sum_ray_w
        details["da3_point_sum"] = sum_point
        details["da3_point_weighted_sum"] = sum_point_w
        details["da3_camera_rot_sum"] = sum_camera_rot
        details["da3_camera_rot_weighted_sum"] = sum_camera_rot_w
        details["da3_camera_trans_sum"] = sum_camera_trans
        details["da3_camera_trans_weighted_sum"] = sum_camera_trans_w
        details["da3_camera_intr_sum"] = sum_camera_intr
        details["da3_camera_intr_weighted_sum"] = sum_camera_intr_w
        details["da3_camera_fov_sum"] = sum_camera_fov
        details["da3_camera_fov_weighted_sum"] = sum_camera_fov_w
        details["da3_camera_sum"] = sum_camera
        details["da3_camera_weighted_sum"] = sum_camera_w
        details["da3_total_sum"] = sum_view_total
        details["da3_depth_mean"] = sum_depth / n_views
        details["da3_depth_weighted_mean"] = sum_depth_w / n_views
        details["da3_grad_mean"] = sum_grad / n_views
        details["da3_grad_weighted_mean"] = sum_grad_w / n_views
        details["da3_ray_origin_mean"] = sum_ray_origin / n_views
        details["da3_ray_origin_weighted_mean"] = sum_ray_origin_w / n_views
        details["da3_ray_direction_mean"] = sum_ray_direction / n_views
        details["da3_ray_direction_weighted_mean"] = sum_ray_direction_w / n_views
        details["da3_ray_mean"] = sum_ray / n_views
        details["da3_ray_weighted_mean"] = sum_ray_w / n_views
        details["da3_point_mean"] = sum_point / n_views
        details["da3_point_weighted_mean"] = sum_point_w / n_views
        details["da3_camera_rot_mean"] = sum_camera_rot / n_views
        details["da3_camera_rot_weighted_mean"] = sum_camera_rot_w / n_views
        details["da3_camera_trans_mean"] = sum_camera_trans / n_views
        details["da3_camera_trans_weighted_mean"] = sum_camera_trans_w / n_views
        details["da3_camera_intr_mean"] = sum_camera_intr / n_views
        details["da3_camera_intr_weighted_mean"] = sum_camera_intr_w / n_views
        details["da3_camera_fov_mean"] = sum_camera_fov / n_views
        details["da3_camera_fov_weighted_mean"] = sum_camera_fov_w / n_views
        details["da3_camera_mean"] = sum_camera / n_views
        details["da3_camera_weighted_mean"] = sum_camera_w / n_views
        details["da3_total_mean"] = sum_view_total / n_views

        # total_loss_value = float(total_loss.detach())
        # if total_loss_value > 500:
        #     self._dump_rrd_debug(batch, gt, pr, details, total_loss_value, height, width)

        return total_loss, details

