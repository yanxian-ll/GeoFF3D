"""World-frame variants of factored geometry losses."""
from __future__ import annotations

import torch

from geoff3d.loss.losses import (
    apply_log_to_norm,
    Criterion,
    FactoredGeometryRegr3D,
    FactoredGeometryRegr3DPlusNormalGMLoss,
    MultiLoss,
    Sum,
)
from geoff3d.utils.geometry import (
    normalize_multiple_pointclouds,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
)


def _world_points_from_detached_geometry(
    pts3d_cam: torch.Tensor,
    cam_quats: torch.Tensor,
    cam_trans: torch.Tensor,
    detach_translation: bool = True,
    geometry_scale_factor: torch.Tensor | None = None,
) -> torch.Tensor:
    rot_c2w = quaternion_to_rotation_matrix(cam_quats)
    pts3d_cam_sg = pts3d_cam.detach()
    if geometry_scale_factor is not None:
        scale = geometry_scale_factor
        while scale.ndim < pts3d_cam_sg.ndim:
            scale = scale.unsqueeze(-1)
        pts3d_cam_sg = pts3d_cam_sg * scale
    cam_trans_for_loss = cam_trans.detach() if detach_translation else cam_trans
    return (
        torch.einsum("bij,bhwj->bhwi", rot_c2w, pts3d_cam_sg)
        + cam_trans_for_loss[:, None, None, :]
    )


def _metric_scale_mask_from_camera_points(self, batch, valid_masks, gt_pts_cam):
    if self.norm_all:
        return torch.ones_like(batch[0]["is_metric_scale"])

    if self.max_metric_scale:
        B = valid_masks[0].shape[0]
        dists_to_cam1 = []
        for i in range(len(valid_masks)):
            dists_to_cam1.append(
                torch.where(
                    valid_masks[i],
                    torch.norm(gt_pts_cam[i], dim=-1),
                    0,
                ).view(B, -1)
            )

        batch[0]["is_metric_scale"] = batch[0]["is_metric_scale"]
        for dist in dists_to_cam1:
            batch[0]["is_metric_scale"] &= dist.max(dim=-1).values < self.max_metric_scale

        for i in range(1, len(valid_masks)):
            batch[i]["is_metric_scale"] = batch[0]["is_metric_scale"]

    return ~batch[0]["is_metric_scale"]


def _match_assignment_dtype(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return src.to(dtype=dst.dtype) if src.dtype != dst.dtype else src


def _get_dataset_world_frame_all_info(self, batch, preds, dist_clip=None):
    if self.convert_predictions_to_view0_frame:
        raise ValueError(
            "World-frame loss expects predictions to already be in dataset "
            "world frame. Set convert_predictions_to_view0_frame=False."
        )

    n_views = len(batch)
    detach_world_frame_points_geometry = getattr(
        self, "detach_world_frame_points_geometry", False
    )
    detach_world_frame_points_translation = getattr(
        self, "detach_world_frame_points_translation", True
    )

    no_norm_gt_pts = []
    no_norm_gt_pts_cam = []
    no_norm_gt_depth = []
    no_norm_gt_pose_trans = []
    valid_masks = []
    gt_ray_directions = []
    gt_pose_quats = []

    no_norm_pr_pts = []
    no_norm_pr_pts_cam = []
    no_norm_pr_depth = []
    no_norm_pr_pose_trans = []
    pr_ray_directions = []
    pr_pose_quats = []

    for i in range(n_views):
        no_norm_gt_pts.append(batch[i]["pts3d"])
        valid_masks.append(batch[i]["valid_mask"].clone())
        no_norm_gt_pts_cam.append(batch[i]["pts3d_cam"])
        gt_ray_directions.append(batch[i]["ray_directions_cam"])
        if self.depth_type_for_loss == "depth_along_ray":
            no_norm_gt_depth.append(batch[i]["depth_along_ray"])
        elif self.depth_type_for_loss == "depth_z":
            no_norm_gt_depth.append(batch[i]["pts3d_cam"][..., 2:])

        gt_pose_quats.append(batch[i]["camera_pose_quats"])
        no_norm_gt_pose_trans.append(batch[i]["camera_pose_trans"])

        if detach_world_frame_points_geometry:
            local_points = preds[i].get(
                "pts3d_cam_unscaled",
                preds[i]["pts3d_cam"],
            )
            no_norm_pr_pts.append(
                _world_points_from_detached_geometry(
                    local_points,
                    preds[i]["cam_quats"],
                    preds[i]["cam_trans"],
                    detach_translation=detach_world_frame_points_translation,
                    geometry_scale_factor=preds[i].get(
                        "geometry_scale_factor"
                    ),
                )
            )
        else:
            no_norm_pr_pts.append(preds[i]["pts3d"])
        no_norm_pr_pose_trans.append(preds[i]["cam_trans"])
        pr_pose_quats.append(preds[i]["cam_quats"])

        no_norm_pr_pts_cam.append(preds[i]["pts3d_cam"])
        pr_ray_directions.append(preds[i]["ray_directions"])
        if self.depth_type_for_loss == "depth_along_ray":
            no_norm_pr_depth.append(preds[i]["depth_along_ray"])
        elif self.depth_type_for_loss == "depth_z":
            no_norm_pr_depth.append(preds[i]["pts3d_cam"][..., 2:])

    if dist_clip is not None:
        for i in range(n_views):
            dis = no_norm_gt_pts_cam[i].norm(dim=-1)
            valid_masks[i] = valid_masks[i] & (dis <= dist_clip)

    non_metric_scale_mask = _metric_scale_mask_from_camera_points(
        self, batch, valid_masks, no_norm_gt_pts_cam
    )

    gt_pts = [torch.zeros_like(pts) for pts in no_norm_gt_pts]
    gt_pts_cam = [torch.zeros_like(pts_cam) for pts_cam in no_norm_gt_pts_cam]
    gt_depth = [torch.zeros_like(depth) for depth in no_norm_gt_depth]
    gt_pose_trans = [torch.zeros_like(trans) for trans in no_norm_gt_pose_trans]

    pr_pts = [torch.zeros_like(pts) for pts in no_norm_pr_pts]
    pr_pts_cam = [torch.zeros_like(pts_cam) for pts_cam in no_norm_pr_pts_cam]
    pr_depth = [torch.zeros_like(depth) for depth in no_norm_pr_depth]
    pr_pose_trans = [torch.zeros_like(trans) for trans in no_norm_pr_pose_trans]

    if self.norm_mode and non_metric_scale_mask.any():
        pr_normalization_output = normalize_multiple_pointclouds(
            [pts[non_metric_scale_mask] for pts in no_norm_pr_pts],
            [mask[non_metric_scale_mask] for mask in valid_masks],
            self.norm_mode,
            ret_factor=True,
        )
        pr_pts_norm = pr_normalization_output[:-1]
        pr_norm_factor = pr_normalization_output[-1]

        if detach_world_frame_points_geometry:
            # Do not let normalization-scale gradients leak into the detached
            # local geometry. The world-point residual itself still controls the
            # non-detached pose components selected above.
            pr_norm_factor = pr_norm_factor.detach()
            pr_pts_norm = [
                pts[non_metric_scale_mask] / pr_norm_factor
                for pts in no_norm_pr_pts
            ]

        for i in range(n_views):
            pr_pts[i][non_metric_scale_mask] = _match_assignment_dtype(
                pr_pts_norm[i], pr_pts[i]
            )
            pr_pts_cam[i][non_metric_scale_mask] = (
                no_norm_pr_pts_cam[i][non_metric_scale_mask] / pr_norm_factor
            ).to(dtype=pr_pts_cam[i].dtype)
            pr_depth[i][non_metric_scale_mask] = (
                no_norm_pr_depth[i][non_metric_scale_mask] / pr_norm_factor
            ).to(dtype=pr_depth[i].dtype)
            pr_pose_trans[i][non_metric_scale_mask] = (
                no_norm_pr_pose_trans[i][non_metric_scale_mask]
                / pr_norm_factor[:, :, 0, 0]
            ).to(dtype=pr_pose_trans[i].dtype)
    elif non_metric_scale_mask.any():
        for i in range(n_views):
            pr_pts[i][non_metric_scale_mask] = no_norm_pr_pts[i][non_metric_scale_mask]
            pr_pts_cam[i][non_metric_scale_mask] = no_norm_pr_pts_cam[i][
                non_metric_scale_mask
            ]
            pr_depth[i][non_metric_scale_mask] = no_norm_pr_depth[i][
                non_metric_scale_mask
            ]
            pr_pose_trans[i][non_metric_scale_mask] = no_norm_pr_pose_trans[i][
                non_metric_scale_mask
            ]

    metric_scale_mask = ~non_metric_scale_mask
    if self.norm_mode and not self.gt_scale:
        gt_normalization_output = normalize_multiple_pointclouds(
            no_norm_gt_pts, valid_masks, self.norm_mode, ret_factor=True
        )
        gt_pts_norm = gt_normalization_output[:-1]
        norm_factor = gt_normalization_output[-1]

        for i in range(n_views):
            gt_pts[i] = gt_pts_norm[i]
            gt_pts_cam[i] = no_norm_gt_pts_cam[i] / norm_factor
            gt_depth[i] = no_norm_gt_depth[i] / norm_factor
            gt_pose_trans[i] = no_norm_gt_pose_trans[i] / norm_factor[:, :, 0, 0]

            pr_pts[i][metric_scale_mask] = (
                no_norm_pr_pts[i][metric_scale_mask]
                / norm_factor[metric_scale_mask]
            ).to(dtype=pr_pts[i].dtype)
            pr_pts_cam[i][metric_scale_mask] = (
                no_norm_pr_pts_cam[i][metric_scale_mask]
                / norm_factor[metric_scale_mask]
            ).to(dtype=pr_pts_cam[i].dtype)
            pr_depth[i][metric_scale_mask] = (
                no_norm_pr_depth[i][metric_scale_mask]
                / norm_factor[metric_scale_mask]
            ).to(dtype=pr_depth[i].dtype)
            pr_pose_trans[i][metric_scale_mask] = (
                no_norm_pr_pose_trans[i][metric_scale_mask]
                / norm_factor[metric_scale_mask][:, :, 0, 0]
            ).to(dtype=pr_pose_trans[i].dtype)
    elif metric_scale_mask.any():
        for i in range(n_views):
            gt_pts[i] = no_norm_gt_pts[i]
            gt_pts_cam[i] = no_norm_gt_pts_cam[i]
            gt_depth[i] = no_norm_gt_depth[i]
            gt_pose_trans[i] = no_norm_gt_pose_trans[i]
            pr_pts[i][metric_scale_mask] = no_norm_pr_pts[i][metric_scale_mask]
            pr_pts_cam[i][metric_scale_mask] = no_norm_pr_pts_cam[i][metric_scale_mask]
            pr_depth[i][metric_scale_mask] = no_norm_pr_depth[i][metric_scale_mask]
            pr_pose_trans[i][metric_scale_mask] = no_norm_pr_pose_trans[i][
                metric_scale_mask
            ]
    else:
        for i in range(n_views):
            gt_pts[i] = no_norm_gt_pts[i]
            gt_pts_cam[i] = no_norm_gt_pts_cam[i]
            gt_depth[i] = no_norm_gt_depth[i]
            gt_pose_trans[i] = no_norm_gt_pose_trans[i]

    ambiguous_masks = []
    for i in range(n_views):
        ambiguous_masks.append((~batch[i]["non_ambiguous_mask"]) & (~valid_masks[i]))

    gt_info = []
    pred_info = []
    for i in range(n_views):
        gt_info.append(
            {
                "ray_directions": gt_ray_directions[i],
                self.depth_type_for_loss: gt_depth[i],
                "pose_trans": gt_pose_trans[i],
                "pose_quats": gt_pose_quats[i],
                "pts3d": gt_pts[i],
                "pts3d_cam": gt_pts_cam[i],
            }
        )
        pred_info.append(
            {
                "ray_directions": pr_ray_directions[i],
                self.depth_type_for_loss: pr_depth[i],
                "pose_trans": pr_pose_trans[i],
                "pose_quats": pr_pose_quats[i],
                "pts3d": pr_pts[i],
                "pts3d_cam": pr_pts_cam[i],
            }
        )

    return gt_info, pred_info, valid_masks, ambiguous_masks


class WorldFramePointsRegr3D(Criterion, MultiLoss):
    """True dataset-world-frame pointmap loss only.

    This is meant to be added on top of the original Pi3/Pi3X view0-relative
    geometry loss. It does not compute camera-frame geometry, ray, depth, normal,
    GM, or pose losses. The optional detach flags only affect this true-world
    point loss.
    """

    def __init__(
        self,
        criterion,
        norm_mode="?avg_dis",
        gt_scale=False,
        ambiguous_loss_value=0,
        max_metric_scale=False,
        loss_in_log=True,
        flatten_across_image_only=False,
        detach_world_frame_points_geometry=False,
        detach_world_frame_points_translation=True,
        world_frame_points_loss_weight=1.0,
    ):
        super().__init__(criterion)
        if norm_mode.startswith("?"):
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.ambiguous_loss_value = ambiguous_loss_value
        self.max_metric_scale = max_metric_scale
        self.loss_in_log = loss_in_log
        self.flatten_across_image_only = flatten_across_image_only
        self.detach_world_frame_points_geometry = detach_world_frame_points_geometry
        self.detach_world_frame_points_translation = detach_world_frame_points_translation
        self.world_frame_points_loss_weight = world_frame_points_loss_weight

    def get_name(self):
        return f"WorldFramePointsRegr3D({self.criterion})"

    def get_all_info(self, batch, preds, dist_clip=None):
        n_views = len(batch)
        no_norm_gt_pts = []
        no_norm_gt_pts_cam = []
        no_norm_pr_pts = []
        valid_masks = []

        for i in range(n_views):
            no_norm_gt_pts.append(batch[i]["pts3d"])
            no_norm_gt_pts_cam.append(batch[i]["pts3d_cam"])
            valid_masks.append(batch[i]["valid_mask"].clone())
            if self.detach_world_frame_points_geometry:
                local_points = preds[i].get(
                    "pts3d_cam_unscaled",
                    preds[i]["pts3d_cam"],
                )
                no_norm_pr_pts.append(
                    _world_points_from_detached_geometry(
                        local_points,
                        preds[i]["cam_quats"],
                        preds[i]["cam_trans"],
                        detach_translation=self.detach_world_frame_points_translation,
                        geometry_scale_factor=preds[i].get(
                            "geometry_scale_factor"
                        ),
                    )
                )
            else:
                no_norm_pr_pts.append(preds[i]["pts3d"])

        if dist_clip is not None:
            for i in range(n_views):
                dis = no_norm_gt_pts_cam[i].norm(dim=-1)
                valid_masks[i] = valid_masks[i] & (dis <= dist_clip)

        non_metric_scale_mask = _metric_scale_mask_from_camera_points(
            self, batch, valid_masks, no_norm_gt_pts_cam
        )
        metric_scale_mask = ~non_metric_scale_mask

        gt_pts = [torch.zeros_like(pts) for pts in no_norm_gt_pts]
        pr_pts = [torch.zeros_like(pts) for pts in no_norm_pr_pts]

        if self.norm_mode and non_metric_scale_mask.any():
            pr_normalization_output = normalize_multiple_pointclouds(
                [pts[non_metric_scale_mask] for pts in no_norm_pr_pts],
                [mask[non_metric_scale_mask] for mask in valid_masks],
                self.norm_mode,
                ret_factor=True,
            )
            pr_pts_norm = pr_normalization_output[:-1]
            pr_norm_factor = pr_normalization_output[-1]

            if self.detach_world_frame_points_geometry:
                pr_norm_factor = pr_norm_factor.detach()
                pr_pts_norm = [
                    pts[non_metric_scale_mask] / pr_norm_factor
                    for pts in no_norm_pr_pts
                ]

            for i in range(n_views):
                pr_pts[i][non_metric_scale_mask] = _match_assignment_dtype(
                    pr_pts_norm[i], pr_pts[i]
                )
        elif non_metric_scale_mask.any():
            for i in range(n_views):
                pr_pts[i][non_metric_scale_mask] = no_norm_pr_pts[i][non_metric_scale_mask]

        if self.norm_mode and not self.gt_scale:
            gt_normalization_output = normalize_multiple_pointclouds(
                no_norm_gt_pts, valid_masks, self.norm_mode, ret_factor=True
            )
            gt_pts_norm = gt_normalization_output[:-1]
            norm_factor = gt_normalization_output[-1]
            for i in range(n_views):
                gt_pts[i] = gt_pts_norm[i]
                pr_pts[i][metric_scale_mask] = (
                    no_norm_pr_pts[i][metric_scale_mask]
                    / norm_factor[metric_scale_mask]
                ).to(dtype=pr_pts[i].dtype)
        elif metric_scale_mask.any():
            for i in range(n_views):
                gt_pts[i] = no_norm_gt_pts[i]
                pr_pts[i][metric_scale_mask] = no_norm_pr_pts[i][metric_scale_mask]
        else:
            for i in range(n_views):
                gt_pts[i] = no_norm_gt_pts[i]

        ambiguous_masks = []
        for i in range(n_views):
            ambiguous_masks.append((~batch[i]["non_ambiguous_mask"]) & (~valid_masks[i]))

        return gt_pts, pr_pts, valid_masks, ambiguous_masks

    def compute_loss(self, batch, preds, **kw):
        gt_pts, pred_pts, masks, ambiguous_masks = self.get_all_info(
            batch, preds, **kw
        )
        n_views = len(batch)
        losses = []
        details = {}
        running_avg_dict = {}
        self_name = type(self).__name__

        if self.ambiguous_loss_value > 0:
            assert self.criterion.reduction == "none", (
                "ambiguous_loss_value should be 0 if no conf/exclusion loss"
            )
            masks = [mask | amb_mask for mask, amb_mask in zip(masks, ambiguous_masks)]

        if not self.flatten_across_image_only:
            for view_idx in range(n_views):
                pred = pred_pts[view_idx][masks[view_idx]]
                gt = gt_pts[view_idx][masks[view_idx]]
                if self.loss_in_log:
                    pred = apply_log_to_norm(pred)
                    gt = apply_log_to_norm(gt)
                loss = self.criterion(pred, gt, factor="points")
                loss = loss * self.world_frame_points_loss_weight
                losses.append((loss, masks[view_idx], "world_pts3d"))
                if loss.numel() > 0:
                    loss_mean = float(loss.mean())
                    details[f"{self_name}_world_pts3d_view{view_idx + 1}"] = loss_mean
                    avg_key = f"{self_name}_world_pts3d_avg"
                    if avg_key not in details:
                        details[avg_key] = loss_mean
                        running_avg_dict[f"{self_name}_world_pts3d_valid_views"] = 1
                    else:
                        valid_views = (
                            running_avg_dict[f"{self_name}_world_pts3d_valid_views"] + 1
                        )
                        running_avg_dict[f"{self_name}_world_pts3d_valid_views"] = valid_views
                        details[avg_key] += (loss_mean - details[avg_key]) / valid_views
        else:
            batch_size, _, _, dim = gt_pts[0].shape
            for view_idx in range(n_views):
                gt = gt_pts[view_idx].view(batch_size, -1, dim)
                pred = pred_pts[view_idx].view(batch_size, -1, dim)
                view_mask = masks[view_idx].view(batch_size, -1)
                amb_mask = ambiguous_masks[view_idx].view(batch_size, -1)
                if self.loss_in_log:
                    pred = apply_log_to_norm(pred)
                    gt = apply_log_to_norm(gt)
                loss = self.criterion(pred, gt, factor="points")
                loss = loss * self.world_frame_points_loss_weight
                if self.ambiguous_loss_value > 0:
                    loss = torch.where(amb_mask, self.ambiguous_loss_value, loss)
                losses.append((loss, view_mask, "world_pts3d"))
                loss_after_masking = loss[view_mask]
                if loss_after_masking.numel() > 0:
                    loss_mean = float(loss_after_masking.mean())
                    details[f"{self_name}_world_pts3d_view{view_idx + 1}"] = loss_mean
                    avg_key = f"{self_name}_world_pts3d_avg"
                    if avg_key not in details:
                        details[avg_key] = loss_mean
                        running_avg_dict[f"{self_name}_world_pts3d_valid_views"] = 1
                    else:
                        valid_views = (
                            running_avg_dict[f"{self_name}_world_pts3d_valid_views"] + 1
                        )
                        running_avg_dict[f"{self_name}_world_pts3d_valid_views"] = valid_views
                        details[avg_key] += (loss_mean - details[avg_key]) / valid_views

        return Sum(*losses), details


class WorldFramePoseLoss(Criterion, MultiLoss):
    """Explicit camera pose loss in the dataset world frame.

    This loss supervises predicted ``cam_trans`` and ``cam_quats`` against the
    dataset world-frame ``camera_pose_trans`` and ``camera_pose_quats``. It is
    intended to complement the dense world-point loss because pose is a global
    per-view quantity and should not be processed by pixel-level top-percent
    filtering.
    """

    def __init__(
        self,
        criterion,
        norm_mode="?avg_dis",
        gt_scale=False,
        max_metric_scale=False,
        depth_type_for_loss="depth_z",
        pose_quats_loss_weight=1.0,
        pose_trans_loss_weight=1.0,
        compute_absolute_pose_loss=True,
        compute_pairwise_relative_pose_loss=False,
    ):
        super().__init__(criterion)
        if norm_mode.startswith("?"):
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.max_metric_scale = max_metric_scale
        self.depth_type_for_loss = depth_type_for_loss
        self.pose_quats_loss_weight = pose_quats_loss_weight
        self.pose_trans_loss_weight = pose_trans_loss_weight
        self.compute_absolute_pose_loss = compute_absolute_pose_loss
        self.compute_pairwise_relative_pose_loss = compute_pairwise_relative_pose_loss
        self.convert_predictions_to_view0_frame = False
        self.detach_world_frame_points_geometry = False
        self.detach_world_frame_points_translation = True
        if not (self.compute_absolute_pose_loss or self.compute_pairwise_relative_pose_loss):
            raise ValueError(
                "At least one of compute_absolute_pose_loss or "
                "compute_pairwise_relative_pose_loss must be True."
            )

    def get_name(self):
        return f"WorldFramePoseLoss({self.criterion})"

    def get_all_info(self, batch, preds, dist_clip=None):
        return _get_dataset_world_frame_all_info(self, batch, preds, dist_clip)

    def _criterion_values(self, pred, gt, factor):
        if pred.numel() == 0 or gt.numel() == 0:
            return pred.new_zeros(())
        if hasattr(self.criterion, "distance"):
            return self.criterion.distance(pred, gt, factor=factor)
        return self.criterion(pred, gt, factor=factor)

    def _mean_factor_loss(self, pred, gt, factor):
        values = self._criterion_values(pred, gt, factor=factor)
        return values.mean() if values.ndim > 0 else values

    def _mean_quat_loss(self, pred_quats, gt_quats):
        if pred_quats.numel() == 0 or gt_quats.numel() == 0:
            return pred_quats.new_zeros(())
        pos_values = self._criterion_values(pred_quats, gt_quats, factor="pose_quats")
        neg_values = self._criterion_values(pred_quats, -gt_quats, factor="pose_quats")
        values = torch.minimum(pos_values, neg_values)
        return values.mean() if values.ndim > 0 else values

    @staticmethod
    def _inverse_pose_from_quats_trans(quats, trans):
        inv_quats = quaternion_inverse(quats)
        inv_rot = quaternion_to_rotation_matrix(inv_quats)
        inv_trans = -torch.einsum("bij,bj->bi", inv_rot, trans)
        return inv_quats, inv_rot, inv_trans

    @staticmethod
    def _relative_pose(inv_ref_quats, inv_ref_rot, inv_ref_trans, quats, trans):
        rel_quats = quaternion_multiply(inv_ref_quats, quats)
        rel_trans = torch.einsum("bij,bj->bi", inv_ref_rot, trans) + inv_ref_trans
        return rel_quats, rel_trans

    def compute_loss(self, batch, preds, **kw):
        gt_info, pred_info, valid_masks, _ = self.get_all_info(batch, preds, **kw)
        n_views = len(batch)
        valid_pose_masks = [mask.sum(dim=(1, 2)) > 0 for mask in valid_masks]
        self_name = type(self).__name__
        details = {}
        total_loss = pred_info[0]["pose_trans"].new_zeros(())

        abs_trans_details = []
        abs_quats_details = []
        rel_trans_details = []
        rel_quats_details = []
        pose_trans_loss_weight = self.pose_trans_loss_weight
        details[f"{self_name}_pose_trans_loss_weight"] = float(pose_trans_loss_weight)

        if self.compute_absolute_pose_loss:
            for view_idx in range(n_views):
                valid = valid_pose_masks[view_idx]
                pred_trans = pred_info[view_idx]["pose_trans"][valid]
                gt_trans = gt_info[view_idx]["pose_trans"][valid]
                pred_quats = pred_info[view_idx]["pose_quats"][valid]
                gt_quats = gt_info[view_idx]["pose_quats"][valid]

                trans_loss = self._mean_factor_loss(
                    pred_trans, gt_trans, factor="pose_trans"
                ) * pose_trans_loss_weight
                quats_loss = self._mean_quat_loss(
                    pred_quats, gt_quats
                ) * self.pose_quats_loss_weight
                total_loss = total_loss + trans_loss + quats_loss

                if trans_loss.numel() > 0:
                    trans_value = float(trans_loss.detach())
                    details[f"{self_name}_world_pose_trans_abs_view{view_idx + 1}"] = trans_value
                    abs_trans_details.append(trans_value)
                if quats_loss.numel() > 0:
                    quats_value = float(quats_loss.detach())
                    details[f"{self_name}_world_pose_quats_abs_view{view_idx + 1}"] = quats_value
                    abs_quats_details.append(quats_value)

        if self.compute_pairwise_relative_pose_loss and n_views > 1:
            for ref_idx in range(n_views):
                pred_inv_ref_quats, pred_inv_ref_rot, pred_inv_ref_trans = (
                    self._inverse_pose_from_quats_trans(
                        pred_info[ref_idx]["pose_quats"],
                        pred_info[ref_idx]["pose_trans"],
                    )
                )
                gt_inv_ref_quats, gt_inv_ref_rot, gt_inv_ref_trans = (
                    self._inverse_pose_from_quats_trans(
                        gt_info[ref_idx]["pose_quats"],
                        gt_info[ref_idx]["pose_trans"],
                    )
                )

                pred_rel_quats = []
                pred_rel_trans = []
                gt_rel_quats = []
                gt_rel_trans = []
                for other_idx in range(n_views):
                    if other_idx == ref_idx:
                        continue
                    valid = valid_pose_masks[ref_idx] & valid_pose_masks[other_idx]
                    curr_pred_rel_quats, curr_pred_rel_trans = self._relative_pose(
                        pred_inv_ref_quats,
                        pred_inv_ref_rot,
                        pred_inv_ref_trans,
                        pred_info[other_idx]["pose_quats"],
                        pred_info[other_idx]["pose_trans"],
                    )
                    curr_gt_rel_quats, curr_gt_rel_trans = self._relative_pose(
                        gt_inv_ref_quats,
                        gt_inv_ref_rot,
                        gt_inv_ref_trans,
                        gt_info[other_idx]["pose_quats"],
                        gt_info[other_idx]["pose_trans"],
                    )
                    pred_rel_quats.append(curr_pred_rel_quats[valid])
                    pred_rel_trans.append(curr_pred_rel_trans[valid])
                    gt_rel_quats.append(curr_gt_rel_quats[valid])
                    gt_rel_trans.append(curr_gt_rel_trans[valid])

                if pred_rel_trans:
                    pred_rel_quats = torch.cat(pred_rel_quats, dim=0)
                    pred_rel_trans = torch.cat(pred_rel_trans, dim=0)
                    gt_rel_quats = torch.cat(gt_rel_quats, dim=0)
                    gt_rel_trans = torch.cat(gt_rel_trans, dim=0)

                    trans_loss = self._mean_factor_loss(
                        pred_rel_trans, gt_rel_trans, factor="pose_trans"
                    ) * pose_trans_loss_weight
                    quats_loss = self._mean_quat_loss(
                        pred_rel_quats, gt_rel_quats
                    ) * self.pose_quats_loss_weight
                    total_loss = total_loss + trans_loss + quats_loss

                    trans_value = float(trans_loss.detach())
                    quats_value = float(quats_loss.detach())
                    details[f"{self_name}_world_pose_trans_rel_ref{ref_idx + 1}"] = trans_value
                    details[f"{self_name}_world_pose_quats_rel_ref{ref_idx + 1}"] = quats_value
                    rel_trans_details.append(trans_value)
                    rel_quats_details.append(quats_value)

        if abs_trans_details:
            details[f"{self_name}_world_pose_trans_abs_avg"] = sum(abs_trans_details) / len(abs_trans_details)
        if abs_quats_details:
            details[f"{self_name}_world_pose_quats_abs_avg"] = sum(abs_quats_details) / len(abs_quats_details)
        if rel_trans_details:
            details[f"{self_name}_world_pose_trans_rel_avg"] = sum(rel_trans_details) / len(rel_trans_details)
        if rel_quats_details:
            details[f"{self_name}_world_pose_quats_rel_avg"] = sum(rel_quats_details) / len(rel_quats_details)

        return total_loss, details


class CameraFOVLoss(MultiLoss):
    """Supervise CameraHead [fov_x, fov_y] directly from intrinsics."""

    def __init__(
        self,
        fov_loss_weight=1.0,
        loss_type="l1",
        only_valid_depth=True,
        only_prior_views=False,
        average_across_views=False,
    ):
        super().__init__()
        if loss_type not in {"l1", "l2"}:
            raise ValueError("loss_type must be 'l1' or 'l2'")
        self.fov_loss_weight = float(fov_loss_weight)
        self.loss_type = loss_type
        self.only_valid_depth = bool(only_valid_depth)
        self.only_prior_views = bool(only_prior_views)
        self.average_across_views = bool(average_across_views)

    def get_name(self):
        return f"CameraFOVLoss({self.loss_type})"

    def compute_loss(self, batch, preds, **kw):
        total_loss = preds[0]["camera_fov_hw"].sum() * 0.0
        details = {}
        view_values = []

        for view_idx, (gt_view, pred_view) in enumerate(
            zip(batch, preds)
        ):
            if "camera_intrinsics" not in gt_view:
                raise KeyError(
                    "CameraFOVLoss requires batch[i]['camera_intrinsics']"
                )
            if "camera_fov_hw" not in pred_view:
                raise KeyError(
                    "CameraFOVLoss requires preds[i]['camera_fov_hw']"
                )

            intrinsics = gt_view["camera_intrinsics"]
            image_height, image_width = gt_view["img"].shape[-2:]
            fx = intrinsics[..., 0, 0].clamp_min(1e-6)
            fy = intrinsics[..., 1, 1].clamp_min(1e-6)
            gt_fov = torch.stack(
                [
                    2.0
                    * torch.atan((0.5 * float(image_width)) / fx),
                    2.0
                    * torch.atan((0.5 * float(image_height)) / fy),
                ],
                dim=-1,
            )
            pred_fov = pred_view["camera_fov_hw"]
            valid = torch.ones(
                pred_fov.shape[0],
                device=pred_fov.device,
                dtype=torch.bool,
            )
            if self.only_valid_depth and "valid_mask" in gt_view:
                valid &= gt_view["valid_mask"].sum(dim=(-2, -1)) > 0
            if self.only_prior_views:
                if "fov_prior_mask" not in pred_view:
                    valid &= False
                else:
                    valid &= pred_view["fov_prior_mask"].bool()
            if not bool(valid.any()):
                continue

            residual = pred_fov[valid] - gt_fov[valid]
            if self.loss_type == "l1":
                view_loss = residual.abs().sum(dim=-1).mean()
            else:
                view_loss = residual.norm(dim=-1).mean()
            view_loss = view_loss * self.fov_loss_weight
            total_loss = total_loss + view_loss
            value = float(view_loss.detach())
            details[f"CameraFOVLoss_view{view_idx + 1}"] = value
            view_values.append(value)

        if view_values and self.average_across_views:
            total_loss = total_loss / len(view_values)
        details["CameraFOVLoss_avg"] = (
            sum(view_values) / len(view_values) if view_values else 0.0
        )
        return total_loss, details


class WorldFrameFactoredGeometryRegr3D(FactoredGeometryRegr3D):
    """Factored geometry loss with global points/poses in dataset world frame."""

    def __init__(
        self,
        *args,
        detach_world_frame_points_geometry=False,
        detach_world_frame_points_translation=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.detach_world_frame_points_geometry = detach_world_frame_points_geometry
        self.detach_world_frame_points_translation = detach_world_frame_points_translation

    def get_all_info(self, batch, preds, dist_clip=None):
        return _get_dataset_world_frame_all_info(self, batch, preds, dist_clip)


class WorldFrameFactoredGeometryRegr3DPlusNormalGMLoss(
    FactoredGeometryRegr3DPlusNormalGMLoss
):
    """PlusNormalGM variant with global points/poses in dataset world frame."""

    def __init__(
        self,
        *args,
        detach_world_frame_points_geometry=False,
        detach_world_frame_points_translation=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.detach_world_frame_points_geometry = detach_world_frame_points_geometry
        self.detach_world_frame_points_translation = detach_world_frame_points_translation

    def get_all_info(self, batch, preds, dist_clip=None):
        return _get_dataset_world_frame_all_info(self, batch, preds, dist_clip)
