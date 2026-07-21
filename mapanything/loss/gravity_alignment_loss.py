"""Gravity/up-vector alignment loss for z-up world-frame supervision."""

import torch
import torch.nn.functional as F

from mapanything.utils.geometry import quaternion_to_rotation_matrix

from .losses import MultiLoss


class GravityAlignmentLoss(MultiLoss):
    """Constrain only the world up direction, not the full absolute yaw.

    This loss compares the direction of the world up vector expressed in each
    camera frame:

        up_cam = R_wc^T @ world_up

    where R_wc is the cam2world rotation. If a whole scene is rotated by an
    arbitrary yaw around the z-up axis, this quantity stays unchanged. Therefore
    the loss fixes roll/pitch / gravity alignment while leaving the horizontal
    x/y axis direction unconstrained.

    Expected keys:
        batch[i]["camera_pose_quats"]: GT cam2world quaternion, shape (B, 4)
        preds[i]["cam_quats"]: predicted cam2world quaternion, shape (B, 4)
        batch[i]["valid_mask"]: optional validity mask, shape (B, H, W)
    """

    def __init__(
        self,
        gravity_loss_weight=1.0,
        loss_type="cosine",
        world_up=(0.0, 0.0, 1.0),
        only_valid_depth=True,
        skip_first_view=False,
        average_across_views=False,
        eps=1e-6,
    ):
        super().__init__()
        if loss_type not in {"cosine", "angle", "l1", "l2"}:
            raise ValueError("loss_type must be one of {'cosine', 'angle', 'l1', 'l2'}")
        self.gravity_loss_weight = gravity_loss_weight
        self.loss_type = loss_type
        self.world_up = tuple(float(x) for x in world_up)
        self.only_valid_depth = only_valid_depth
        self.skip_first_view = skip_first_view
        self.average_across_views = average_across_views
        self.eps = eps

    def get_name(self):
        return f"GravityAlignmentLoss({self.loss_type})"

    def _zero_loss(self, preds):
        for pred in preds:
            if "cam_quats" in pred:
                return pred["cam_quats"].sum() * 0.0
        raise KeyError("GravityAlignmentLoss requires preds[i]['cam_quats']")

    def _valid_batch_mask(self, gt_view):
        if not self.only_valid_depth or "valid_mask" not in gt_view:
            return None
        return gt_view["valid_mask"].sum(dim=(-2, -1)) > 0

    def _up_in_camera(self, quats):
        quats = F.normalize(quats.float(), dim=-1, eps=self.eps)
        rot_c2w = quaternion_to_rotation_matrix(quats)
        world_up = torch.tensor(
            self.world_up,
            dtype=rot_c2w.dtype,
            device=rot_c2w.device,
        )
        world_up = F.normalize(world_up, dim=0, eps=self.eps)
        up_cam = torch.einsum("bij,j->bi", rot_c2w.transpose(-1, -2), world_up)
        return F.normalize(up_cam, dim=-1, eps=self.eps)

    def _gravity_residual(self, pred_quats, gt_quats):
        pred_up_cam = self._up_in_camera(pred_quats)
        gt_up_cam = self._up_in_camera(gt_quats)

        if self.loss_type == "cosine":
            return 1.0 - (pred_up_cam * gt_up_cam).sum(dim=-1).clamp(-1.0, 1.0)
        if self.loss_type == "angle":
            cosine = (pred_up_cam * gt_up_cam).sum(dim=-1).clamp(
                -1.0 + self.eps,
                1.0 - self.eps,
            )
            return torch.acos(cosine)
        if self.loss_type == "l1":
            return (pred_up_cam - gt_up_cam).abs().sum(dim=-1)
        return (pred_up_cam - gt_up_cam).norm(dim=-1)

    def compute_loss(self, batch, preds, **kw):
        n_views = len(batch)
        start_view = 1 if self.skip_first_view else 0

        total_loss = self._zero_loss(preds)
        details = {}
        valid_views = 0
        view_means = []

        for view_idx in range(start_view, n_views):
            if "camera_pose_quats" not in batch[view_idx]:
                raise KeyError("GravityAlignmentLoss requires batch[i]['camera_pose_quats']")
            if "cam_quats" not in preds[view_idx]:
                raise KeyError("GravityAlignmentLoss requires preds[i]['cam_quats']")

            gravity_loss = self._gravity_residual(
                preds[view_idx]["cam_quats"],
                batch[view_idx]["camera_pose_quats"],
            )
            valid_mask = self._valid_batch_mask(batch[view_idx])
            if valid_mask is not None:
                gravity_loss = gravity_loss[valid_mask]
            if gravity_loss.numel() == 0:
                continue

            view_loss = gravity_loss.mean() * self.gravity_loss_weight
            total_loss = total_loss + view_loss
            valid_views += 1
            view_means.append(float(view_loss.detach()))
            details[f"GravityAlignmentLoss_view{view_idx + 1}"] = float(view_loss.detach())

        if valid_views == 0:
            details["GravityAlignmentLoss_avg"] = 0.0
            return total_loss, details

        if self.average_across_views:
            total_loss = total_loss / valid_views
        details["GravityAlignmentLoss_avg"] = sum(view_means) / len(view_means)
        return total_loss, details
