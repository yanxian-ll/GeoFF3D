# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Multi-view geometric losses for training 3D reconstruction models.

References: DUSt3R & MASt3R
"""

import math
from copy import copy, deepcopy

import einops as ein
import torch
import torch.nn as nn
import torch.nn.functional as F

from mapanything.utils.geometry import (
    angle_diff_vec3,
    apply_log_to_norm,
    closed_form_pose_inverse,
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    geotrf,
    normalize_multiple_pointclouds,
    normalize_pose_translations,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    transform_pose_using_quats_and_trans_2_to_1,
)


def get_loss_terms_and_details(
    losses_dict, valid_masks, self_name, n_views, flatten_across_image_only
):
    """
    Helper function to generate loss terms and details for different loss types.

    Args:
        losses_dict (dict): Dictionary mapping loss types to their values.
            Format: {
                'loss_type': {
                    'values': list_of_loss_tensors or single_tensor,
                    'use_mask': bool,
                    'is_multi_view': bool
                }
            }
        valid_masks (list): List of valid masks for each view.
        self_name (str): Name of the loss class.
        n_views (int): Number of views.
        flatten_across_image_only (bool): Whether flattening was done across image only.

    Returns:
        tuple: (loss_terms, details) where loss_terms is a list of tuples (loss, mask, type)
               and details is a dictionary of loss details.
    """
    loss_terms = []
    details = {}

    for loss_type, loss_info in losses_dict.items():
        values = loss_info["values"]
        use_mask = loss_info["use_mask"]
        is_multi_view = loss_info["is_multi_view"]
        if is_multi_view:
            # Handle multi-view losses (list of tensors)
            view_loss_details = []
            for i in range(n_views):
                mask = valid_masks[i] if use_mask else None
                loss_terms.append((values[i], mask, loss_type))

                # Add details for individual view
                if not flatten_across_image_only or not use_mask:
                    values_after_masking = values[i]
                else:
                    values_after_masking = values[i][mask]

                if values_after_masking.numel() > 0:
                    view_loss_detail = float(values_after_masking.mean())
                    if view_loss_detail > 0:
                        details[f"{self_name}_{loss_type}_view{i + 1}"] = (
                            view_loss_detail
                        )
                        view_loss_details.append(view_loss_detail)
            # Add average across views
            if len(view_loss_details) > 0:
                details[f"{self_name}_{loss_type}_avg"] = sum(view_loss_details) / len(
                    view_loss_details
                )
        else:
            # Handle single tensor losses
            if values is not None:
                loss_terms.append((values, None, loss_type))
                if values.numel() > 0:
                    loss_detail = float(values.mean())
                    if loss_detail > 0:
                        details[f"{self_name}_{loss_type}"] = loss_detail

    return loss_terms, details


def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def _camera_intrinsics_to_fov_hw(
    camera_intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """
    Convert pinhole intrinsics to [fov_x, fov_y] in radians.
    camera_intrinsics: (B, 3, 3)
    returns: (B, 2)
    """
    fx = camera_intrinsics[:, 0, 0].clamp_min(1e-6)
    fy = camera_intrinsics[:, 1, 1].clamp_min(1e-6)

    fov_x = 2.0 * torch.atan((0.5 * float(image_width)) / fx)
    fov_y = 2.0 * torch.atan((0.5 * float(image_height)) / fy)

    return torch.stack([fov_x, fov_y], dim=-1)


def _normalize_ray_directions(ray_directions: torch.Tensor) -> torch.Tensor:
    """Normalize ray directions with a small epsilon for stability."""
    return F.normalize(ray_directions, dim=-1, eps=1e-6)


def _ray_directions_to_azimuth_elevation(ray_directions: torch.Tensor) -> torch.Tensor:
    """Convert ray directions (..., 3) into azimuth/elevation angles (..., 2)."""
    ray_directions = _normalize_ray_directions(ray_directions)
    ray_x, ray_y, ray_z = ray_directions.unbind(dim=-1)
    azimuth = torch.atan2(ray_x, ray_z)
    elevation = torch.atan2(
        ray_y,
        torch.sqrt(ray_x.square() + ray_z.square()).clamp_min(1e-6),
    )
    return torch.stack((azimuth, elevation), dim=-1)


def _wrap_angle_difference(angle_difference: torch.Tensor) -> torch.Tensor:
    """Wrap angular residuals into [-pi, pi]."""
    return torch.atan2(torch.sin(angle_difference), torch.cos(angle_difference))


def _ray_direction_angle_residual(
    predicted_ray_directions: torch.Tensor,
    target_ray_directions: torch.Tensor,
) -> torch.Tensor:
    """Return wrapped azimuth/elevation residuals for ray direction supervision."""
    pred_angles = _ray_directions_to_azimuth_elevation(predicted_ray_directions)
    target_angles = _ray_directions_to_azimuth_elevation(target_ray_directions)
    return _wrap_angle_difference(pred_angles - target_angles)


def compute_normal_loss(points, gt_points, mask):
    """
    Compute the normal loss between the predicted and ground truth points.
    References:
    https://github.com/microsoft/MoGe/blob/a8c37341bc0325ca99b9d57981cc3bb2bd3e255b/moge/train/losses.py#L205

    Args:
        points (torch.Tensor): Predicted points. Shape: (..., H, W, 3).
        gt_points (torch.Tensor): Ground truth points. Shape: (..., H, W, 3).
        mask (torch.Tensor): Mask indicating valid points. Shape: (..., H, W).

    Returns:
        torch.Tensor: Normal loss.
    """
    height, width = points.shape[-3:-1]

    leftup, rightup, leftdown, rightdown = (
        points[..., :-1, :-1, :],
        points[..., :-1, 1:, :],
        points[..., 1:, :-1, :],
        points[..., 1:, 1:, :],
    )
    upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
    leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
    downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
    rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

    gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = (
        gt_points[..., :-1, :-1, :],
        gt_points[..., :-1, 1:, :],
        gt_points[..., 1:, :-1, :],
        gt_points[..., 1:, 1:, :],
    )
    gt_upxleft = torch.cross(
        gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1
    )
    gt_leftxdown = torch.cross(
        gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1
    )
    gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
    gt_rightxup = torch.cross(
        gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1
    )

    mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = (
        mask[..., :-1, :-1],
        mask[..., :-1, 1:],
        mask[..., 1:, :-1],
        mask[..., 1:, 1:],
    )
    mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
    mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
    mask_downxright = mask_leftdown & mask_rightup & mask_leftup
    mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

    loss = (
        mask_upxleft
        * _smooth(
            angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE),
            beta=BETA_RAD,
        )
        + mask_leftxdown
        * _smooth(
            angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE),
            beta=BETA_RAD,
        )
        + mask_downxright
        * _smooth(
            angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE),
            beta=BETA_RAD,
        )
        + mask_rightxup
        * _smooth(
            angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE),
            beta=BETA_RAD,
        )
    )

    total_valid_mask = mask_upxleft | mask_leftxdown | mask_downxright | mask_rightxup
    valid_count = total_valid_mask.sum()
    if valid_count > 0:
        loss = loss.sum() / (valid_count * (4 * max(points.shape[-3:-1])))
    else:
        loss = 0 * loss.sum()

    return loss


def compute_gradient_loss(prediction, gt_target, mask):
    """
    Compute the gradient loss between the prediction and GT target at valid points.
    References:
    https://docs.nerf.studio/_modules/nerfstudio/model_components/losses.html#GradientLoss
    https://github.com/autonomousvision/monosdf/blob/main/code/model/loss.py

    Args:
        prediction (torch.Tensor): Predicted scene representation. Shape: (B, H, W, C).
        gt_target (torch.Tensor): Ground truth scene representation. Shape: (B, H, W, C).
        mask (torch.Tensor): Mask indicating valid points. Shape: (B, H, W).
    """
    # Expand mask to match number of channels in prediction
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    summed_mask = torch.sum(mask, (1, 2, 3))

    # Compute the gradient of the prediction and GT target
    diff = prediction - gt_target
    diff = torch.mul(mask, diff)

    # Gradient in x direction
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Gradient in y direction
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp the outlier gradients
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Compute the total loss
    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    num_valid_pixels = torch.sum(summed_mask)
    if num_valid_pixels > 0:
        image_loss = torch.sum(image_loss) / num_valid_pixels
    else:
        image_loss = 0 * torch.sum(image_loss)

    return image_loss


def compute_gradient_matching_loss(prediction, gt_target, mask, scales=4):
    """
    Compute the multi-scale gradient matching loss between the prediction and GT target at valid points.
    This loss biases discontinuities to be sharp and to coincide with discontinuities in the ground truth.
    More info in MiDAS: https://arxiv.org/pdf/1907.01341.pdf; Equation 11
    References:
    https://docs.nerf.studio/_modules/nerfstudio/model_components/losses.html#GradientLoss
    https://github.com/autonomousvision/monosdf/blob/main/code/model/loss.py

    Args:
        prediction (torch.Tensor): Predicted scene representation. Shape: (B, H, W, C).
        gt_target (torch.Tensor): Ground truth scene representation. Shape: (B, H, W, C).
        mask (torch.Tensor): Mask indicating valid points. Shape: (B, H, W).
        scales (int): Number of scales to compute the loss at. Default: 4.
    """
    # Define total loss
    total_loss = 0.0

    # Compute the gradient loss at different scales
    for scale in range(scales):
        step = pow(2, scale)
        grad_loss = compute_gradient_loss(
            prediction[:, ::step, ::step],
            gt_target[:, ::step, ::step],
            mask[:, ::step, ::step],
        )
        total_loss += grad_loss

    return total_loss


def Sum(*losses_and_masks):
    """
    Aggregates multiple losses into a single loss value or returns the original losses.

    Args:
        *losses_and_masks: Variable number of tuples, each containing (loss, mask, rep_type)
            - loss: Tensor containing loss values
            - mask: Mask indicating valid pixels/regions
            - rep_type: String indicating the type of representation (e.g., 'pts3d', 'depth')

    Returns:
        If the first loss has dimensions > 0:
            Returns the original list of (loss, mask, rep_type) tuples
        Otherwise:
            Returns a scalar tensor that is the sum of all loss values
    """
    loss, mask, rep_type = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2, rep_type2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    "Base Criterion to support different reduction methods"

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction


class LLoss(BaseCriterion):
    "L-norm loss"

    def forward(self, a, b, **kwargs):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 4, (
            f"Bad shape = {a.shape}"
        )
        dist = self.distance(a, b, **kwargs)
        assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(self, a, b, **kwargs):
        raise NotImplementedError()


class L1Loss(LLoss):
    "L1 distance"

    def distance(self, a, b, **kwargs):
        return torch.abs(a - b).sum(dim=-1)


class L2Loss(LLoss):
    "Euclidean (L2 Norm) distance"

    def distance(self, a, b, **kwargs):
        return torch.norm(a - b, dim=-1)


class GenericLLoss(LLoss):
    "Criterion that supports different L-norms"

    def distance(self, a, b, loss_type, **kwargs):
        if loss_type == "l1":
            # L1 distance
            return torch.abs(a - b).sum(dim=-1)
        elif loss_type == "l2":
            # Euclidean (L2 norm) distance
            return torch.norm(a - b, dim=-1)
        else:
            raise ValueError(
                f"Unsupported loss type: {loss_type}. Supported types are 'l1' and 'l2'."
            )


class FactoredLLoss(LLoss):
    "Criterion that supports different L-norms for the factored loss functions"

    def __init__(
        self,
        reduction="mean",
        points_loss_type="l2",
        depth_loss_type="l1",
        ray_directions_loss_type="l1",
        pose_quats_loss_type="l1",
        pose_trans_loss_type="l1",
        scale_loss_type="l1",
        fov_hw_loss_type="l1",
    ):
        super().__init__(reduction)
        self.points_loss_type = points_loss_type
        self.depth_loss_type = depth_loss_type
        self.ray_directions_loss_type = ray_directions_loss_type
        self.pose_quats_loss_type = pose_quats_loss_type
        self.pose_trans_loss_type = pose_trans_loss_type
        self.scale_loss_type = scale_loss_type
        self.fov_hw_loss_type = fov_hw_loss_type

    def _distance(self, a, b, loss_type):
        if loss_type == "l1":
            # L1 distance
            return torch.abs(a - b).sum(dim=-1)
        elif loss_type == "l2":
            # Euclidean (L2 norm) distance
            return torch.norm(a - b, dim=-1)
        elif loss_type == "angle_l1":
            angle_residual = _ray_direction_angle_residual(a, b)
            return torch.abs(angle_residual).sum(dim=-1)
        elif loss_type == "angle_l2":
            angle_residual = _ray_direction_angle_residual(a, b)
            return torch.norm(angle_residual, dim=-1)
        elif loss_type == "angle_huber":
            angle_residual = _ray_direction_angle_residual(a, b)
            return _smooth(torch.abs(angle_residual), beta=math.radians(1.0)).sum(
                dim=-1
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}.")

    def distance(self, a, b, factor, **kwargs):
        if factor == "points":
            return self._distance(a, b, self.points_loss_type)
        elif factor == "depth":
            return self._distance(a, b, self.depth_loss_type)
        elif factor == "ray_directions":
            return self._distance(a, b, self.ray_directions_loss_type)
        elif factor == "pose_quats":
            return self._distance(a, b, self.pose_quats_loss_type)
        elif factor == "pose_trans":
            return self._distance(a, b, self.pose_trans_loss_type)
        elif factor == "scale":
            return self._distance(a, b, self.scale_loss_type)
        elif factor == "fov_hw":
            return self._distance(a, b, self.fov_hw_loss_type)
        else:
            raise ValueError(f"Unsupported factor type: {factor}.")


class RobustRegressionLoss(LLoss):
    """
    Generalized Robust Loss introduced in https://arxiv.org/abs/1701.03077.
    """

    def __init__(self, alpha=0.5, scaling_c=0.25, reduction="mean", ray_directions_in_angle_space=False,):
        """
        Initialize the Robust Regression Loss.

        Args:
            alpha (float): Shape parameter controlling the robustness of the loss.
                Lower values make the loss more robust to outliers. Default: 0.5.
            scaling_c (float): Scale parameter controlling the transition between
                quadratic and robust behavior. Default: 0.1.
            reduction (str): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Default: 'mean'.
            ray_directions_in_angle_space (bool): Whether the ray directions are represented in angle space. Default: False.
        """
        super().__init__(reduction)
        self.alpha = alpha
        self.scaling_c = scaling_c
        self.ray_directions_in_angle_space = ray_directions_in_angle_space

    def distance(self, a, b, factor=None, **kwargs):
        # only apply the angle wrapping for ray directions if they are represented in angle space, otherwise we assume they are in vector form and can be directly subtracted
        if factor == "ray_directions" and self.ray_directions_in_angle_space:
            residual = _ray_direction_angle_residual(a, b)
        else:
            residual = a - b

        error_scaled = torch.sum((residual / self.scaling_c) ** 2, dim=-1)
        robust_loss = (abs(self.alpha - 2) / self.alpha) * (
            torch.pow((error_scaled / abs(self.alpha - 2)) + 1, self.alpha / 2) - 1
        )
        return robust_loss


class BCELoss(BaseCriterion):
    """Binary Cross Entropy loss"""

    def forward(self, predicted_logits, reference_mask):
        """
        Args:
            predicted_logits: (B, H, W) tensor of predicted logits for the mask
            reference_mask: (B, H, W) tensor of reference mask

        Returns:
            loss: scalar tensor of the BCE loss
        """
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predicted_logits, reference_mask.float()
        )

        return bce_loss


class Criterion(nn.Module):
    """
    Base class for all criterion modules that wrap a BaseCriterion.

    This class serves as a wrapper around BaseCriterion objects, providing
    additional functionality like naming and reduction mode control.

    Args:
        criterion (BaseCriterion): The base criterion to wrap.
    """

    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, BaseCriterion), (
            f"{criterion} is not a proper criterion!"
        )
        self.criterion = copy(criterion)

    def get_name(self):
        """
        Returns a string representation of this criterion.

        Returns:
            str: A string containing the class name and the wrapped criterion.
        """
        return f"{type(self).__name__}({self.criterion})"

    def with_reduction(self, mode="none"):
        """
        Creates a deep copy of this criterion with the specified reduction mode.

        This method recursively sets the reduction mode for this criterion and
        any chained MultiLoss criteria.

        Args:
            mode (str): The reduction mode to set. Default: "none".

        Returns:
            Criterion: A new criterion with the specified reduction mode.
        """
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss(nn.Module):
    """
    Base class for combinable loss functions with automatic tracking of individual loss values.

    This class enables easy combination of multiple loss functions through arithmetic operations:
        loss = MyLoss1() + 0.1*MyLoss2()

    The combined loss functions maintain their individual weights and the forward pass
    automatically computes and aggregates all losses while tracking individual loss values.

    Usage:
        Inherit from this class and override get_name() and compute_loss() methods.

    Attributes:
        _alpha (float): Weight multiplier for this loss component.
        _loss2 (MultiLoss): Reference to the next loss in the chain, if any.
    """

    def __init__(self):
        """Initialize the MultiLoss with default weight of 1 and no chained loss."""
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        """
        Compute the loss value for this specific loss component.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor or tuple: Either the loss tensor or a tuple of (loss, details_dict).

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def get_name(self):
        """
        Get the name of this loss component.

        Returns:
            str: The name of the loss.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def __mul__(self, alpha):
        """
        Multiply the loss by a scalar weight.

        Args:
            alpha (int or float): The weight to multiply the loss by.

        Returns:
            MultiLoss: A new loss object with the updated weight.

        Raises:
            AssertionError: If alpha is not a number.
        """
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res

    __rmul__ = __mul__  # Support both loss*alpha and alpha*loss

    def __add__(self, loss2):
        """
        Add another loss to this loss, creating a chain of losses.

        Args:
            loss2 (MultiLoss): Another loss to add to this one.

        Returns:
            MultiLoss: A new loss object representing the combined losses.

        Raises:
            AssertionError: If loss2 is not a MultiLoss.
        """
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # Find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        """
        Create a string representation of the loss, including weights and chained losses.

        Returns:
            str: String representation of the loss.
        """
        name = self.get_name()
        if self._alpha != 1:
            name = f"{self._alpha:g}*{name}"
        if self._loss2:
            name = f"{name} + {self._loss2}"
        return name

    def forward(self, *args, **kwargs):
        """
        Compute the weighted loss and aggregate with any chained losses.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The total weighted loss.
                - dict: Details about individual loss components.
        """
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class ScheduledLoss(MultiLoss):
    """Wrap a MultiLoss and scale it with a training-step schedule.

    This is intended for terms such as raw metric/world-coordinate losses whose
    gradient scale can destabilize early fine-tuning.  The training loop updates
    ``current_step`` through ``set_schedule_step`` before each forward pass.
    """

    def __init__(
        self,
        loss,
        start_weight=0.0,
        end_weight=1.0,
        warmup_steps=1000,
        start_step=0,
        schedule="linear",
        name=None,
    ):
        super().__init__()
        if not isinstance(loss, MultiLoss):
            raise TypeError("ScheduledLoss expects a MultiLoss instance")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.loss = loss
        self.start_weight = float(start_weight)
        self.end_weight = float(end_weight)
        self.warmup_steps = int(warmup_steps)
        self.start_step = int(start_step)
        self.schedule = str(schedule)
        self.name = name
        self.current_step = 0
        self.current_epoch = 0.0

    def get_name(self):
        if self.name:
            return f"ScheduledLoss({self.name})"
        return f"ScheduledLoss({self.loss.get_name()})"

    def set_schedule_step(self, step, epoch=None):
        self.current_step = int(step)
        if epoch is not None:
            self.current_epoch = float(epoch)

    def _progress(self):
        if self.warmup_steps <= 0:
            return 1.0
        progress = (self.current_step - self.start_step) / float(self.warmup_steps)
        return max(0.0, min(1.0, progress))

    def get_scheduled_weight(self):
        progress = self._progress()
        if self.schedule == "linear":
            scale = progress
        elif self.schedule == "cosine":
            scale = 0.5 - 0.5 * math.cos(math.pi * progress)
        elif self.schedule == "constant":
            scale = 1.0
        else:
            raise ValueError(f"Unknown ScheduledLoss schedule: {self.schedule}")
        return self.start_weight + (self.end_weight - self.start_weight) * scale

    def compute_loss(self, *args, **kwargs):
        loss, details = self.loss(*args, **kwargs)
        weight = self.get_scheduled_weight()
        scheduled_loss = loss * weight
        details = dict(details)
        details[f"{self.get_name()}_weight"] = float(weight)
        details[f"{self.get_name()}_step"] = float(self.current_step)
        if loss.detach().ndim == 0:
            details[f"{self.get_name()}_unweighted"] = float(loss.detach())
            details[f"{self.get_name()}_weighted"] = float(scheduled_loss.detach())
        return scheduled_loss, details


class PriorDepthConsistencyLoss(MultiLoss):
    """Keep prior-conditioned depth aligned with frozen clean geometry."""

    def __init__(
        self,
        prediction_key="prior_branch_depth_z",
        target_key="geometry_branch_depth_z",
        loss_type="smooth_l1",
        log_space=True,
        beta=0.05,
        eps=1e-6,
        only_valid_depth=True,
        detach_target=True,
    ):
        super().__init__()
        if loss_type not in {"l1", "smooth_l1", "l2"}:
            raise ValueError(
                "PriorDepthConsistencyLoss loss_type must be one of "
                "['l1', 'smooth_l1', 'l2']"
            )
        self.prediction_key = str(prediction_key)
        self.target_key = str(target_key)
        self.loss_type = str(loss_type)
        self.log_space = bool(log_space)
        self.beta = float(beta)
        self.eps = float(eps)
        self.only_valid_depth = bool(only_valid_depth)
        self.detach_target = bool(detach_target)

    def get_name(self):
        return "PriorDepthConsistencyLoss"

    @staticmethod
    def _squeeze_depth_channel(depth):
        if depth.ndim >= 4 and depth.shape[-1] == 1:
            return depth.squeeze(-1)
        return depth

    def compute_loss(self, batch, preds, **kw):
        view_losses = []
        details = {}
        for view_idx, pred in enumerate(preds):
            if (
                self.prediction_key not in pred
                or self.target_key not in pred
            ):
                continue

            prediction = self._squeeze_depth_channel(
                pred[self.prediction_key].float()
            )
            target = self._squeeze_depth_channel(
                pred[self.target_key].float()
            )
            if prediction.shape != target.shape:
                raise ValueError(
                    "PriorDepthConsistencyLoss expects matching depth "
                    f"shapes, got {tuple(prediction.shape)} and "
                    f"{tuple(target.shape)} for view {view_idx + 1}"
                )
            if self.detach_target:
                target = target.detach()

            valid = (
                torch.isfinite(prediction)
                & torch.isfinite(target)
                & (prediction > self.eps)
                & (target > self.eps)
            )
            if self.only_valid_depth and "valid_mask" in batch[view_idx]:
                valid = valid & batch[view_idx]["valid_mask"].bool()

            if not bool(valid.any()):
                view_loss = prediction.sum() * 0.0
            else:
                prediction_values = prediction[valid]
                target_values = target[valid]
                if self.log_space:
                    prediction_values = torch.log(
                        prediction_values.clamp_min(self.eps)
                    )
                    target_values = torch.log(
                        target_values.clamp_min(self.eps)
                    )

                if self.loss_type == "l1":
                    view_loss = F.l1_loss(
                        prediction_values,
                        target_values,
                    )
                elif self.loss_type == "l2":
                    view_loss = F.mse_loss(
                        prediction_values,
                        target_values,
                    )
                else:
                    view_loss = F.smooth_l1_loss(
                        prediction_values,
                        target_values,
                        beta=self.beta,
                    )

            view_losses.append(view_loss)
            details[
                f"{self.get_name()}_view{view_idx + 1}"
            ] = float(view_loss.detach())

        if not view_losses:
            reference = preds[0].get("pts3d")
            if reference is None:
                reference = next(
                    value
                    for value in preds[0].values()
                    if torch.is_tensor(value)
                )
            total = reference.sum() * 0.0
        else:
            total = torch.stack(view_losses).mean()
        details[f"{self.get_name()}_avg"] = float(total.detach())
        return total, details


class DepthNormalFromDepthLoss(MultiLoss):
    """Compare normals estimated from predicted z-depth and GT z-depth."""

    def __init__(
        self,
        pred_depth_key="pts3d_cam",
        gt_depth_key="depthmap",
        intrinsics_keys=("camera_intrinsics", "intrinsics"),
        loss_type="cosine",
        detach_target=True,
        use_valid_mask=True,
        use_depth_mask=True,
        depth_mask_keys=(
            "depthmap_mask",
            "depth_mask",
            "depth_valid_mask",
            "depthmap_valid_mask",
        ),
        min_depth=1.0e-6,
        eps=1.0e-6,
    ):
        super().__init__()
        if loss_type not in {"cosine", "l1", "cosine_l1"}:
            raise ValueError(
                "DepthNormalFromDepthLoss loss_type must be one of "
                "['cosine', 'l1', 'cosine_l1']"
            )
        if isinstance(intrinsics_keys, str):
            intrinsics_keys = (intrinsics_keys,)
        if isinstance(depth_mask_keys, str):
            depth_mask_keys = (depth_mask_keys,)
        self.pred_depth_key = str(pred_depth_key)
        self.gt_depth_key = str(gt_depth_key)
        self.intrinsics_keys = tuple(intrinsics_keys)
        self.loss_type = str(loss_type)
        self.detach_target = bool(detach_target)
        self.use_valid_mask = bool(use_valid_mask)
        self.use_depth_mask = bool(use_depth_mask)
        self.depth_mask_keys = tuple(depth_mask_keys)
        self.min_depth = float(min_depth)
        self.eps = float(eps)

    def get_name(self):
        return "DepthNormalFromDepthLoss"

    @staticmethod
    def _ensure_bhw(x, name="tensor"):
        if x.dim() == 3:
            return x
        if x.dim() == 4 and x.shape[-1] == 1:
            return x[..., 0]
        if x.dim() == 4 and x.shape[1] == 1:
            return x[:, 0]
        if x.dim() == 4 and x.shape[-1] == 3:
            return x[..., 2]
        if x.dim() == 4 and x.shape[1] == 3:
            return x[:, 2]
        raise ValueError(f"Cannot convert {name} with shape {tuple(x.shape)} to [B,H,W].")

    def _prediction_depth(self, pred, view_idx):
        if self.pred_depth_key not in pred:
            raise KeyError(
                f"DepthNormalFromDepthLoss requires pred[{view_idx}]"
                f"['{self.pred_depth_key}']."
            )
        depth = pred[self.pred_depth_key].float()
        return self._ensure_bhw(
            depth,
            name=f"pred[{view_idx}]['{self.pred_depth_key}']",
        )

    def _gt_depth(self, view, view_idx, device):
        if self.gt_depth_key not in view:
            raise KeyError(
                f"DepthNormalFromDepthLoss requires batch[{view_idx}]"
                f"['{self.gt_depth_key}']."
            )
        return self._ensure_bhw(
            view[self.gt_depth_key].to(device=device).float(),
            name=f"batch[{view_idx}]['{self.gt_depth_key}']",
        )

    def _intrinsics(self, view, view_idx, device):
        for key in self.intrinsics_keys:
            if key in view:
                intrinsics = view[key].to(device=device).float()
                if intrinsics.dim() != 3 or intrinsics.shape[-2:] != (3, 3):
                    raise ValueError(
                        f"DepthNormalFromDepthLoss expected batch[{view_idx}]"
                        f"['{key}'] to be [B,3,3], got {tuple(intrinsics.shape)}"
                    )
                return intrinsics
        raise KeyError(
            "DepthNormalFromDepthLoss requires one of "
            f"{self.intrinsics_keys} in batch[{view_idx}]."
        )

    def _depth_mask(self, view, depth, view_idx, device):
        mask = torch.ones_like(depth, dtype=torch.bool)
        if self.use_valid_mask and "valid_mask" in view:
            valid = self._ensure_bhw(
                view["valid_mask"].to(device=device),
                name=f"batch[{view_idx}]['valid_mask']",
            ).bool()
            if valid.shape[-2:] != depth.shape[-2:]:
                valid = self._resize_nearest(valid.float(), depth.shape[-2:]).bool()
            mask = mask & valid

        if self.use_depth_mask:
            for key in self.depth_mask_keys:
                if key not in view:
                    continue
                depth_mask = self._ensure_bhw(
                    view[key].to(device=device),
                    name=f"batch[{view_idx}]['{key}']",
                ).bool()
                if depth_mask.shape[-2:] != depth.shape[-2:]:
                    depth_mask = self._resize_nearest(
                        depth_mask.float(),
                        depth.shape[-2:],
                    ).bool()
                mask = mask & depth_mask
                break
        return mask

    @staticmethod
    def _resize_nearest(x, size):
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x[:, None].float(), size=size, mode="nearest")[:, 0]

    def _pixel_intrinsics(self, intrinsics, h, w):
        intrinsics = intrinsics.clone()
        fx = intrinsics[:, 0, 0].detach().abs().median()
        fy = intrinsics[:, 1, 1].detach().abs().median()
        cx = intrinsics[:, 0, 2].detach().abs().median()
        cy = intrinsics[:, 1, 2].detach().abs().median()
        looks_normalized = bool(fx <= 4.0 and fy <= 4.0 and cx <= 2.0 and cy <= 2.0)
        if looks_normalized:
            intrinsics[:, 0, :] = intrinsics[:, 0, :] * float(w)
            intrinsics[:, 1, :] = intrinsics[:, 1, :] * float(h)
        return intrinsics

    def _normal_from_depth(self, depth, intrinsics):
        b, h, w = depth.shape
        intrinsics = self._pixel_intrinsics(intrinsics, h=h, w=w)
        fx = intrinsics[:, 0, 0].clamp_min(self.eps)
        fy = intrinsics[:, 1, 1].clamp_min(self.eps)
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]

        x, y = torch.meshgrid(
            torch.arange(w, device=depth.device, dtype=depth.dtype),
            torch.arange(h, device=depth.device, dtype=depth.dtype),
            indexing="xy",
        )
        x = x.view(1, h, w).expand(b, h, w)
        y = y.view(1, h, w).expand(b, h, w)

        x_cam = (x - cx[:, None, None] + 0.5) * depth / fx[:, None, None]
        y_cam = (y - cy[:, None, None] + 0.5) * depth / fy[:, None, None]
        points = torch.stack((x_cam, y_cam, depth), dim=-1)

        normals = torch.zeros_like(points)
        dx = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
        dy = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
        normals[:, 1:-1, 1:-1] = F.normalize(
            torch.cross(dx, dy, dim=-1),
            dim=-1,
            eps=self.eps,
        )
        return normals

    def _normal_loss(self, pred_normal, target_normal, valid):
        normal_valid = (
            valid
            & torch.isfinite(pred_normal).all(dim=-1)
            & torch.isfinite(target_normal).all(dim=-1)
            & (pred_normal.norm(dim=-1) > self.eps)
            & (target_normal.norm(dim=-1) > self.eps)
        )
        if not bool(normal_valid.any()):
            return pred_normal.sum() * 0.0, normal_valid

        pred_values = pred_normal[normal_valid]
        target_values = target_normal[normal_valid]
        if self.detach_target:
            target_values = target_values.detach()

        cosine_loss = 1.0 - (pred_values * target_values).sum(dim=-1)
        if self.loss_type == "cosine":
            loss = cosine_loss.mean()
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred_values, target_values)
        else:
            loss = 0.5 * cosine_loss.mean() + 0.5 * F.l1_loss(
                pred_values,
                target_values,
            )
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0), normal_valid

    def compute_loss(self, batch, preds, **kw):
        losses = []
        details = {}
        self_name = self.get_name()

        for view_idx, (view, pred) in enumerate(zip(batch, preds)):
            pred_depth = self._prediction_depth(pred, view_idx)
            device = pred_depth.device
            gt_depth = self._gt_depth(view, view_idx, device=device)
            if gt_depth.shape[-2:] != pred_depth.shape[-2:]:
                gt_depth = self._resize_nearest(gt_depth, pred_depth.shape[-2:])
            intrinsics = self._intrinsics(view, view_idx, device=device)

            valid = (
                torch.isfinite(pred_depth)
                & torch.isfinite(gt_depth)
                & (pred_depth > self.min_depth)
                & (gt_depth > self.min_depth)
            )
            valid = valid & self._depth_mask(view, gt_depth, view_idx, device=device)

            pred_normal = self._normal_from_depth(pred_depth, intrinsics)
            gt_normal = self._normal_from_depth(gt_depth, intrinsics)
            loss, normal_valid = self._normal_loss(pred_normal, gt_normal, valid)
            losses.append(loss)
            details[f"{self_name}_view{view_idx + 1}"] = float(loss.detach())
            details[f"{self_name}_valid_ratio_view{view_idx + 1}"] = float(
                normal_valid.float().mean().detach()
            )

        if not losses:
            reference = next(
                value
                for value in preds[0].values()
                if torch.is_tensor(value)
            )
            total = reference.sum() * 0.0
        else:
            total = torch.stack(losses).mean()
        details[f"{self_name}_avg"] = float(total.detach())
        return total, details


class NonAmbiguousMaskLoss(Criterion, MultiLoss):
    """
    Loss on non-ambiguous mask prediction logits.
    """

    def __init__(self, criterion):
        super().__init__(criterion)

    def compute_loss(self, batch, preds, **kw):
        """
        Args:
            batch: list of dicts with the gt data
            preds: list of dicts with the predictions

        Returns:
            loss: Sum class of the lossses for N-views and the loss details
        """
        # Init loss list to keep track of individual losses for each view
        loss_list = []
        mask_loss_details = {}
        mask_loss_total = 0
        self_name = type(self).__name__

        # Loop over the views
        for view_idx, (gt, pred) in enumerate(zip(batch, preds)):
            # Get the GT non-ambiguous masks
            gt_non_ambiguous_mask = gt["non_ambiguous_mask"]

            # Get the predicted non-ambiguous mask logits
            pred_non_ambiguous_mask_logits = pred["non_ambiguous_mask_logits"]

            # Compute the loss for the current view
            loss = self.criterion(pred_non_ambiguous_mask_logits, gt_non_ambiguous_mask)

            # Add the loss to the list
            loss_list.append((loss, None, "non_ambiguous_mask"))

            # Add the loss details to the dictionary
            mask_loss_details[f"{self_name}_mask_view{view_idx + 1}"] = float(loss)
            mask_loss_total += float(loss)

        # Compute the average loss across all views
        mask_loss_details[f"{self_name}_mask_avg"] = mask_loss_total / len(batch)

        return Sum(*loss_list), (mask_loss_details | {})


class ConfLoss(MultiLoss):
    """
    Applies confidence-weighted regression loss using model-predicted confidence values.

    The confidence-weighted loss has the form:
        conf_loss = raw_loss * conf - alpha * log(conf)

    Where:
    - raw_loss is the original per-pixel loss
    - conf is the predicted confidence (higher values = higher confidence)
    - alpha is a hyperparameter controlling the regularization strength

    This loss can be selectively applied to specific loss components in factored and multi-view settings.
    """

    def __init__(self, pixel_loss, alpha=1, loss_set_indices=None):
        """
        Args:
            pixel_loss (MultiLoss): The pixel-level regression loss to be used.
            alpha (float): Hyperparameter controlling the confidence regularization strength.
            loss_set_indices (list or None): Indices of the loss sets to apply confidence weighting to.
                Each index selects a specific loss set across all views (with the same rep_type).
                If None, defaults to [0] which applies to the first loss set only.
        """
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction("none")
        self.loss_set_indices = [0] if loss_set_indices is None else loss_set_indices

    def get_name(self):
        return f"ConfLoss({self.pixel_loss})"

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, batch, preds, **kw):
        # Init loss list and details
        total_loss = 0
        conf_loss_details = {}
        running_avg_dict = {}
        self_name = type(self.pixel_loss).__name__
        n_views = len(batch)

        # Compute per-pixel loss for each view
        losses, pixel_loss_details = self.pixel_loss(batch, preds, **kw)

        # Select specific loss sets based on indices
        selected_losses = []
        processed_indices = set()
        for idx in self.loss_set_indices:
            start_idx = idx * n_views
            end_idx = min((idx + 1) * n_views, len(losses))
            selected_losses.extend(losses[start_idx:end_idx])
            processed_indices.update(range(start_idx, end_idx))

        # Process selected losses with confidence weighting
        for loss_idx, (loss, msk, rep_type) in enumerate(selected_losses):
            view_idx = loss_idx % n_views  # Map to corresponding view index

            if loss.numel() == 0:
                # print(f"NO VALID VALUES in loss idx {loss_idx} (Rep Type: {rep_type}, Num Views: {n_views})", force=True)
                continue

            # Get the confidence and log confidence
            if (
                hasattr(self.pixel_loss, "flatten_across_image_only")
                and self.pixel_loss.flatten_across_image_only
            ):
                # Reshape confidence to match the flattened dimensions
                conf_reshaped = preds[view_idx]["conf"].view(
                    preds[view_idx]["conf"].shape[0], -1
                )
                conf, log_conf = self.get_conf_log(conf_reshaped[msk])
                loss = loss[msk]
            else:
                conf, log_conf = self.get_conf_log(preds[view_idx]["conf"][msk])

            # Weight the loss by the confidence
            conf_loss = loss * conf - self.alpha * log_conf

            # Only add to total loss and store details if there are valid elements
            if conf_loss.numel() > 0:
                conf_loss = conf_loss.mean()
                total_loss = total_loss + conf_loss

                # Store details
                conf_loss_details[
                    f"{self_name}_{rep_type}_conf_loss_view{view_idx + 1}"
                ] = float(conf_loss)

                # Initialize or update running average directly
                avg_key = f"{self_name}_{rep_type}_conf_loss_avg"
                if avg_key not in conf_loss_details:
                    conf_loss_details[avg_key] = float(conf_loss)
                    running_avg_dict[
                        f"{self_name}_{rep_type}_conf_loss_valid_views"
                    ] = 1
                else:
                    valid_views = (
                        running_avg_dict[
                            f"{self_name}_{rep_type}_conf_loss_valid_views"
                        ]
                        + 1
                    )
                    running_avg_dict[
                        f"{self_name}_{rep_type}_conf_loss_valid_views"
                    ] = valid_views
                    conf_loss_details[avg_key] += (
                        float(conf_loss) - conf_loss_details[avg_key]
                    ) / valid_views

        # Add unmodified losses for sets not in selected_losses
        for idx, (loss, msk, rep_type) in enumerate(losses):
            if idx not in processed_indices:
                if msk is not None:
                    loss_after_masking = loss[msk]
                else:
                    loss_after_masking = loss
                if loss_after_masking.numel() > 0:
                    loss_mean = loss_after_masking.mean()
                else:
                    # print(f"NO VALID VALUES in loss idx {idx} (Rep Type: {rep_type}, Num Views: {n_views})", force=True)
                    loss_mean = 0
                total_loss = total_loss + loss_mean

        return total_loss, dict(**conf_loss_details, **pixel_loss_details)


class ExcludeTopNPercentPixelLoss(MultiLoss):
    """
    Pixel-level regression loss where for each instance in a batch the top N% of per-pixel loss values are ignored
    for the mean loss computation.
    Allows selecting which pixel-level regression loss sets to apply the exclusion to.
    """

    def __init__(
        self,
        pixel_loss,
        top_n_percent=5,
        apply_to_real_data_only=True,
        loss_set_indices=None,
    ):
        """
        Args:
            pixel_loss (MultiLoss): The pixel-level regression loss to be used.
            top_n_percent (float): The percentage of top per-pixel loss values to ignore. Range: [0, 100]. Default: 5.
            apply_to_real_data_only (bool): Whether to apply the loss only to real world data. Default: True.
            loss_set_indices (list or None): Indices of the loss sets to apply the exclusion to.
                Each index selects a specific loss set across all views (with the same rep_type).
                If None, defaults to [0] which applies to the first loss set only.
        """
        super().__init__()
        self.pixel_loss = pixel_loss.with_reduction("none")
        self.top_n_percent = top_n_percent
        self.bottom_n_percent = 100 - top_n_percent
        self.apply_to_real_data_only = apply_to_real_data_only
        self.loss_set_indices = [0] if loss_set_indices is None else loss_set_indices

    def get_name(self):
        return f"ExcludeTopNPercentPixelLoss({self.pixel_loss})"

    def keep_bottom_n_percent(self, tensor, mask, bottom_n_percent):
        """
        Function to compute the mask for keeping the bottom n percent of per-pixel loss values.

        Args:
            tensor (torch.Tensor): The tensor containing the per-pixel loss values.
                                   Shape: (B, N) where B is the batch size and N is the number of total pixels.
            mask (torch.Tensor): The mask indicating valid pixels. Shape: (B, N).

        Returns:
            torch.Tensor: Flattened tensor containing the bottom n percent of per-pixel loss values.
        """
        B, N = tensor.shape

        # Calculate the number of valid elements (where mask is True)
        num_valid = mask.sum(dim=1)

        # Calculate the number of elements to keep (n% of valid elements)
        num_keep = (num_valid * bottom_n_percent / 100).long()

        # Create a mask for the bottom n% elements
        keep_mask = torch.arange(N, device=tensor.device).unsqueeze(
            0
        ) < num_keep.unsqueeze(1)

        # Create a tensor with inf where mask is False
        masked_tensor = torch.where(
            mask, tensor, torch.tensor(float("inf"), device=tensor.device)
        )

        # Sort the masked tensor along the N dimension
        sorted_tensor, _ = torch.sort(masked_tensor, dim=1, descending=False)

        # Get the bottom n% elements
        bottom_n_percent_elements = sorted_tensor[keep_mask]

        return bottom_n_percent_elements

    def compute_loss(self, batch, preds, **kw):
        # Compute per-pixel loss
        losses, details = self.pixel_loss(batch, preds, **kw)
        n_views = len(batch)

        # Select specific loss sets based on indices
        selected_losses = []
        processed_indices = set()
        for idx in self.loss_set_indices:
            start_idx = idx * n_views
            end_idx = min((idx + 1) * n_views, len(losses))
            selected_losses.extend(losses[start_idx:end_idx])
            processed_indices.update(range(start_idx, end_idx))

        # Initialize total loss
        total_loss = 0.0
        loss_details = {}
        running_avg_dict = {}
        self_name = type(self.pixel_loss).__name__

        # Process selected losses with top N percent exclusion
        for loss_idx, (loss, msk, rep_type) in enumerate(selected_losses):
            view_idx = loss_idx % n_views  # Map to corresponding view index

            if loss.numel() == 0:
                # print(f"NO VALID VALUES in loss idx {loss_idx} (Rep Type: {rep_type}, Num Views: {n_views})", force=True)
                continue

            # Create empty list for current view's aggregated tensors
            aggregated_losses = []

            if self.apply_to_real_data_only:
                # Get the synthetic and real world data mask
                synthetic_mask = batch[view_idx]["is_synthetic"]
                real_data_mask = ~batch[view_idx]["is_synthetic"]
            else:
                # Apply the filtering to all data
                synthetic_mask = torch.zeros_like(batch[view_idx]["is_synthetic"])
                real_data_mask = torch.ones_like(batch[view_idx]["is_synthetic"])

            # Process synthetic data
            if synthetic_mask.any():
                synthetic_loss = loss[synthetic_mask]
                synthetic_msk = msk[synthetic_mask]
                aggregated_losses.append(synthetic_loss[synthetic_msk])

            # Process real data
            if real_data_mask.any():
                real_loss = loss[real_data_mask]
                real_msk = msk[real_data_mask]
                real_bottom_n_percent_loss = self.keep_bottom_n_percent(
                    real_loss, real_msk, self.bottom_n_percent
                )
                aggregated_losses.append(real_bottom_n_percent_loss)

            # Compute view loss
            view_loss = torch.cat(aggregated_losses, dim=0)

            # Only add to total loss and store details if there are valid elements
            if view_loss.numel() > 0:
                view_loss = view_loss.mean()
                total_loss = total_loss + view_loss

                # Store details
                loss_details[
                    f"{self_name}_{rep_type}_bot{self.bottom_n_percent}%_loss_view{view_idx + 1}"
                ] = float(view_loss)

                # Initialize or update running average directly
                avg_key = f"{self_name}_{rep_type}_bot{self.bottom_n_percent}%_loss_avg"
                if avg_key not in loss_details:
                    loss_details[avg_key] = float(view_loss)
                    running_avg_dict[
                        f"{self_name}_{rep_type}_bot{self.bottom_n_percent}%_valid_views"
                    ] = 1
                else:
                    valid_views = (
                        running_avg_dict[
                            f"{self_name}_{rep_type}_bot{self.bottom_n_percent}%_valid_views"
                        ]
                        + 1
                    )
                    running_avg_dict[
                        f"{self_name}_{rep_type}_bot{self.bottom_n_percent}%_valid_views"
                    ] = valid_views
                    loss_details[avg_key] += (
                        float(view_loss) - loss_details[avg_key]
                    ) / valid_views

        # Add unmodified losses for sets not in selected_losses
        for idx, (loss, msk, rep_type) in enumerate(losses):
            if idx not in processed_indices:
                if msk is not None:
                    loss_after_masking = loss[msk]
                else:
                    loss_after_masking = loss
                if loss_after_masking.numel() > 0:
                    loss_mean = loss_after_masking.mean()
                else:
                    # print(f"NO VALID VALUES in loss idx {idx} (Rep Type: {rep_type}, Num Views: {n_views})", force=True)
                    loss_mean = 0
                total_loss = total_loss + loss_mean

        return total_loss, dict(**loss_details, **details)


class ConfAndExcludeTopNPercentPixelLoss(MultiLoss):
    """
    Combined loss that applies ConfLoss to one set of pixel-level regression losses
    and ExcludeTopNPercentPixelLoss to another set of pixel-level regression losses.
    """

    def __init__(
        self,
        pixel_loss,
        conf_alpha=1,
        top_n_percent=5,
        apply_to_real_data_only=True,
        conf_loss_set_indices=None,
        exclude_loss_set_indices=None,
    ):
        """
        Args:
        pixel_loss (MultiLoss): The pixel-level regression loss to be used.
        conf_alpha (float): Alpha parameter for ConfLoss. Default: 1.
        top_n_percent (float): Percentage of top per-pixel loss values to ignore. Range: [0, 100]. Default: 5.
        apply_to_real_data_only (bool): Whether to apply the exclude loss only to real world data. Default: True.
        conf_loss_set_indices (list or None): Indices of the loss sets to apply confidence weighting to.
            Each index selects a specific loss set across all views (with the same rep_type).
            If None, defaults to [0] which applies to the first loss set only.
        exclude_loss_set_indices (list or None): Indices of the loss sets to apply top N percent exclusion to.
            Each index selects a specific loss set across all views (with the same rep_type).
            If None, defaults to [1] which applies to the second loss set only.
        """
        super().__init__()
        self.pixel_loss = pixel_loss.with_reduction("none")
        assert conf_alpha > 0
        self.conf_alpha = conf_alpha
        self.top_n_percent = top_n_percent
        self.bottom_n_percent = 100 - top_n_percent
        self.apply_to_real_data_only = apply_to_real_data_only
        self.conf_loss_set_indices = (
            [0] if conf_loss_set_indices is None else conf_loss_set_indices
        )
        self.exclude_loss_set_indices = (
            [1] if exclude_loss_set_indices is None else exclude_loss_set_indices
        )

    def get_name(self):
        return f"ConfAndExcludeTopNPercentPixelLoss({self.pixel_loss})"

    def get_conf_log(self, x):
        return x, torch.log(x)

    def keep_bottom_n_percent(self, tensor, mask, bottom_n_percent):
        """
        Function to compute the mask for keeping the bottom n percent of per-pixel loss values.
        """
        B, N = tensor.shape

        # Calculate the number of valid elements (where mask is True)
        num_valid = mask.sum(dim=1)

        # Calculate the number of elements to keep (n% of valid elements)
        num_keep = (num_valid * bottom_n_percent / 100).long()

        # Create a mask for the bottom n% elements
        keep_mask = torch.arange(N, device=tensor.device).unsqueeze(
            0
        ) < num_keep.unsqueeze(1)

        # Create a tensor with inf where mask is False
        masked_tensor = torch.where(
            mask, tensor, torch.tensor(float("inf"), device=tensor.device)
        )

        # Sort the masked tensor along the N dimension
        sorted_tensor, _ = torch.sort(masked_tensor, dim=1, descending=False)

        # Get the bottom n% elements
        bottom_n_percent_elements = sorted_tensor[keep_mask]

        return bottom_n_percent_elements

    def compute_loss(self, batch, preds, **kw):
        # Compute per-pixel loss
        losses, pixel_loss_details = self.pixel_loss(batch, preds, **kw)
        n_views = len(batch)

        # Select specific loss sets for confidence weighting
        conf_selected_losses = []
        conf_processed_indices = set()
        for idx in self.conf_loss_set_indices:
            start_idx = idx * n_views
            end_idx = min((idx + 1) * n_views, len(losses))
            conf_selected_losses.extend(losses[start_idx:end_idx])
            conf_processed_indices.update(range(start_idx, end_idx))

        # Select specific loss sets for top N percent exclusion
        exclude_selected_losses = []
        exclude_processed_indices = set()
        for idx in self.exclude_loss_set_indices:
            start_idx = idx * n_views
            end_idx = min((idx + 1) * n_views, len(losses))
            exclude_selected_losses.extend(losses[start_idx:end_idx])
            exclude_processed_indices.update(range(start_idx, end_idx))

        # Initialize total loss and details
        total_loss = 0
        loss_details = {}
        running_avg_dict = {}
        self_name = type(self.pixel_loss).__name__

        # Process selected losses with confidence weighting
        for loss_idx, (loss, msk, rep_type) in enumerate(conf_selected_losses):
            view_idx = loss_idx % n_views  # Map to corresponding view index

            if loss.numel() == 0:
                # print(f"NO VALID VALUES in loss idx {loss_idx} (Rep Type: {rep_type}, Num Views: {n_views}) for conf loss", force=True)
                continue

            # Get the confidence and log confidence
            if (
                hasattr(self.pixel_loss, "flatten_across_image_only")
                and self.pixel_loss.flatten_across_image_only
            ):
                # Reshape confidence to match the flattened dimensions
                conf_reshaped = preds[view_idx]["conf"].view(
                    preds[view_idx]["conf"].shape[0], -1
                )
                conf, log_conf = self.get_conf_log(conf_reshaped[msk])
                loss = loss[msk]
            else:
                conf, log_conf = self.get_conf_log(preds[view_idx]["conf"][msk])

            # Weight the loss by the confidence
            conf_loss = loss * conf - self.conf_alpha * log_conf

            # Only add to total loss and store details if there are valid elements
            if conf_loss.numel() > 0:
                conf_loss = conf_loss.mean()
                total_loss = total_loss + conf_loss

                # Store details
                loss_details[f"{self_name}_{rep_type}_conf_loss_view{view_idx + 1}"] = (
                    float(conf_loss)
                )

                # Initialize or update running average directly
                avg_key = f"{self_name}_{rep_type}_conf_loss_avg"
                if avg_key not in loss_details:
                    loss_details[avg_key] = float(conf_loss)
                    running_avg_dict[
                        f"{self_name}_{rep_type}_conf_loss_valid_views"
                    ] = 1
                else:
                    valid_views = (
                        running_avg_dict[
                            f"{self_name}_{rep_type}_conf_loss_valid_views"
                        ]
                        + 1
                    )
                    running_avg_dict[
                        f"{self_name}_{rep_type}_conf_loss_valid_views"
                    ] = valid_views
                    loss_details[avg_key] += (
                        float(conf_loss) - loss_details[avg_key]
                    ) / valid_views

        # Process selected losses with top N percent exclusion
        for loss_idx, (loss, msk, rep_type) in enumerate(exclude_selected_losses):
            view_idx = loss_idx % n_views  # Map to corresponding view index

            if loss.numel() == 0:
                # print(f"NO VALID VALUES in loss idx {loss_idx} (Rep Type: {rep_type}, Num Views: {n_views}) for exclude loss", force=True)
                continue

            # Create empty list for current view's aggregated tensors
            aggregated_losses = []

            if self.apply_to_real_data_only:
                # Get the synthetic and real world data mask
                synthetic_mask = batch[view_idx]["is_synthetic"]
                real_data_mask = ~batch[view_idx]["is_synthetic"]
            else:
                # Apply the filtering to all data
                synthetic_mask = torch.zeros_like(batch[view_idx]["is_synthetic"])
                real_data_mask = torch.ones_like(batch[view_idx]["is_synthetic"])

            # Process synthetic data
            if synthetic_mask.any():
                synthetic_loss = loss[synthetic_mask]
                synthetic_msk = msk[synthetic_mask]
                aggregated_losses.append(synthetic_loss[synthetic_msk])

            # Process real data
            if real_data_mask.any():
                real_loss = loss[real_data_mask]
                real_msk = msk[real_data_mask]
                real_bottom_n_percent_loss = self.keep_bottom_n_percent(
                    real_loss, real_msk, self.bottom_n_percent
                )
                aggregated_losses.append(real_bottom_n_percent_loss)

            # Compute view loss
            view_loss = torch.cat(aggregated_losses, dim=0)

            # Only add to total loss and store details if there are valid elements
            if view_loss.numel() > 0:
                view_loss = view_loss.mean()
                total_loss = total_loss + view_loss

                # Store details
                loss_details[
                    f"{self_name}_{rep_type}_bot{self.bottom_n_percent}%_loss_view{view_idx + 1}"
                ] = float(view_loss)

                # Initialize or update running average directly
                avg_key = f"{self_name}_{rep_type}_bot{self.bottom_n_percent}%_loss_avg"
                if avg_key not in loss_details:
                    loss_details[avg_key] = float(view_loss)
                    running_avg_dict[
                        f"{self_name}_{rep_type}_bot{self.bottom_n_percent}%_valid_views"
                    ] = 1
                else:
                    valid_views = (
                        running_avg_dict[
                            f"{self_name}_{rep_type}_bot{self.bottom_n_percent}%_valid_views"
                        ]
                        + 1
                    )
                    running_avg_dict[
                        f"{self_name}_{rep_type}_bot{self.bottom_n_percent}%_valid_views"
                    ] = valid_views
                    loss_details[avg_key] += (
                        float(view_loss) - loss_details[avg_key]
                    ) / valid_views

        # Add unmodified losses for sets not processed with either confidence or exclusion
        all_processed_indices = conf_processed_indices.union(exclude_processed_indices)
        for idx, (loss, msk, rep_type) in enumerate(losses):
            if idx not in all_processed_indices:
                if msk is not None:
                    loss_after_masking = loss[msk]
                else:
                    loss_after_masking = loss
                if loss_after_masking.numel() > 0:
                    loss_mean = loss_after_masking.mean()
                else:
                    # print(f"NO VALID VALUES in loss idx {idx} (Rep Type: {rep_type}, Num Views: {n_views})", force=True)
                    loss_mean = 0
                total_loss = total_loss + loss_mean

        return total_loss, dict(**loss_details, **pixel_loss_details)


class Regr3D(Criterion, MultiLoss):
    """
    Regression Loss for World Frame Pointmaps.
    Asymmetric loss where view 1 is supposed to be the anchor.

    For each view i:
    Pi = RTi @ Di
    lossi = (RTi1 @ pred_Di) - (RT1^-1 @ RTi @ Di)
    where RT1 is the anchor view camera pose
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
    ):
        """
        Initialize the loss criterion for World Frame Pointmaps.

        Args:
            criterion (BaseCriterion): The base criterion to use for computing the loss.
            norm_mode (str): Normalization mode for scene representation. Default: "?avg_dis".
                If prefixed with "?", normalization is only applied to non-metric scale data.
            gt_scale (bool): If True, enforce predictions to have the same scale as ground truth.
                If False, both GT and predictions are normalized independently. Default: False.
            ambiguous_loss_value (float): Value to use for ambiguous pixels in the loss.
                If 0, ambiguous pixels are ignored. Default: 0.
            max_metric_scale (float): Maximum scale for metric scale data. If data exceeds this
                value, it will be treated as non-metric. Default: False (no limit).
            loss_in_log (bool): If True, apply logarithmic transformation to input before
                computing the loss for pointmaps. Default: True.
            flatten_across_image_only (bool): If True, flatten H x W dimensions only when computing
                the loss. If False, flatten across batch and spatial dimensions. Default: False.
        """
        super().__init__(criterion)
        if norm_mode.startswith("?"):
            # Do no norm pts from metric scale datasets
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

    def get_all_info(self, batch, preds, dist_clip=None):
        n_views = len(batch)
        in_camera0 = closed_form_pose_inverse(batch[0]["camera_pose"])

        # Initialize lists to store points and masks
        no_norm_gt_pts = []
        valid_masks = []

        # Process ground truth points and valid masks
        for view_idx in range(n_views):
            no_norm_gt_pts.append(
                geotrf(in_camera0, batch[view_idx]["pts3d"])
            )  # B,H,W,3
            valid_masks.append(batch[view_idx]["valid_mask"].clone())

        if dist_clip is not None:
            # Points that are too far-away == invalid
            for view_idx in range(n_views):
                dis = no_norm_gt_pts[view_idx].norm(dim=-1)  # (B, H, W)
                valid_masks[view_idx] = valid_masks[view_idx] & (dis <= dist_clip)

        # Get predicted points
        no_norm_pr_pts = []
        for view_idx in range(n_views):
            no_norm_pr_pts.append(preds[view_idx]["pts3d"])

        if not self.norm_all:
            if self.max_metric_scale:
                B = valid_masks[0].shape[0]
                # Calculate distances to camera for all views
                dists_to_cam1 = []
                for view_idx in range(n_views):
                    dist = torch.where(
                        valid_masks[view_idx],
                        torch.norm(no_norm_gt_pts[view_idx], dim=-1),
                        0,
                    ).view(B, -1)
                    dists_to_cam1.append(dist)

                # Update metric scale flags
                metric_scale_mask = batch[0]["is_metric_scale"]
                for dist in dists_to_cam1:
                    metric_scale_mask = metric_scale_mask & (
                        dist.max(dim=-1).values < self.max_metric_scale
                    )

                for view_idx in range(n_views):
                    batch[view_idx]["is_metric_scale"] = metric_scale_mask

            non_metric_scale_mask = ~batch[0]["is_metric_scale"]
        else:
            non_metric_scale_mask = torch.ones_like(batch[0]["is_metric_scale"])

        # Initialize normalized points
        gt_pts = [torch.zeros_like(pts) for pts in no_norm_gt_pts]
        pr_pts = [torch.zeros_like(pts) for pts in no_norm_pr_pts]

        # Normalize 3d points
        if self.norm_mode and non_metric_scale_mask.any():
            normalized_pr_pts = normalize_multiple_pointclouds(
                [pts[non_metric_scale_mask] for pts in no_norm_pr_pts],
                [mask[non_metric_scale_mask] for mask in valid_masks],
                self.norm_mode,
            )
            for i in range(n_views):
                pr_pts[i][non_metric_scale_mask] = normalized_pr_pts[i]
        elif non_metric_scale_mask.any():
            for i in range(n_views):
                pr_pts[i][non_metric_scale_mask] = no_norm_pr_pts[i][
                    non_metric_scale_mask
                ]

        if self.norm_mode and not self.gt_scale:
            gt_normalization_output = normalize_multiple_pointclouds(
                no_norm_gt_pts, valid_masks, self.norm_mode, ret_factor=True
            )
            normalized_gt_pts = gt_normalization_output[:-1]
            norm_factor = gt_normalization_output[-1]
            for i in range(n_views):
                gt_pts[i] = normalized_gt_pts[i]
                pr_pts[i][~non_metric_scale_mask] = (
                    no_norm_pr_pts[i][~non_metric_scale_mask]
                    / norm_factor[~non_metric_scale_mask]
                )
        elif ~non_metric_scale_mask.any():
            for i in range(n_views):
                gt_pts[i] = no_norm_gt_pts[i]
                pr_pts[i][~non_metric_scale_mask] = no_norm_pr_pts[i][
                    ~non_metric_scale_mask
                ]
        else:
            for i in range(n_views):
                gt_pts[i] = no_norm_gt_pts[i]

        # Get ambiguous masks
        ambiguous_masks = []
        for view_idx in range(n_views):
            ambiguous_masks.append(
                (~batch[view_idx]["non_ambiguous_mask"]) & (~valid_masks[view_idx])
            )

        return gt_pts, pr_pts, valid_masks, ambiguous_masks, {}

    def compute_loss(self, batch, preds, **kw):
        gt_pts, pred_pts, masks, ambiguous_masks, monitoring = self.get_all_info(
            batch, preds, **kw
        )
        n_views = len(batch)

        if self.ambiguous_loss_value > 0:
            assert self.criterion.reduction == "none", (
                "ambiguous_loss_value should be 0 if no conf loss"
            )
            # Add the ambiguous pixels as "valid" pixels
            masks = [mask | amb_mask for mask, amb_mask in zip(masks, ambiguous_masks)]

        losses = []
        details = {}
        running_avg_dict = {}
        self_name = type(self).__name__

        if not self.flatten_across_image_only:
            for view_idx in range(n_views):
                pred = pred_pts[view_idx][masks[view_idx]]
                gt = gt_pts[view_idx][masks[view_idx]]

                if self.loss_in_log:
                    pred = apply_log_to_norm(pred)
                    gt = apply_log_to_norm(gt)

                loss = self.criterion(pred, gt)

                if self.ambiguous_loss_value > 0:
                    loss = torch.where(
                        ambiguous_masks[view_idx][masks[view_idx]],
                        self.ambiguous_loss_value,
                        loss,
                    )

                losses.append((loss, masks[view_idx], "pts3d"))
                if loss.numel() > 0:
                    loss_mean = float(loss.mean())
                    details[f"{self_name}_pts3d_view{view_idx + 1}"] = loss_mean
                    # Initialize or update running average directly
                    avg_key = f"{self_name}_pts3d_avg"
                    if avg_key not in details:
                        details[avg_key] = loss_mean
                        running_avg_dict[f"{self_name}_pts3d_valid_views"] = 1
                    else:
                        valid_views = (
                            running_avg_dict[f"{self_name}_pts3d_valid_views"] + 1
                        )
                        running_avg_dict[f"{self_name}_pts3d_valid_views"] = valid_views
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

                loss = self.criterion(pred, gt)

                if self.ambiguous_loss_value > 0:
                    loss = torch.where(amb_mask, self.ambiguous_loss_value, loss)

                losses.append((loss, view_mask, "pts3d"))
                loss_after_masking = loss[view_mask]
                if loss_after_masking.numel() > 0:
                    loss_mean = float(loss_after_masking.mean())
                    details[f"{self_name}_pts3d_view{view_idx + 1}"] = loss_mean
                    # Initialize or update running average directly
                    avg_key = f"{self_name}_pts3d_avg"
                    if avg_key not in details:
                        details[avg_key] = loss_mean
                        running_avg_dict[f"{self_name}_pts3d_valid_views"] = 1
                    else:
                        valid_views = (
                            running_avg_dict[f"{self_name}_pts3d_valid_views"] + 1
                        )
                        running_avg_dict[f"{self_name}_pts3d_valid_views"] = valid_views
                        details[avg_key] += (loss_mean - details[avg_key]) / valid_views

        return Sum(*losses), (details | monitoring)


class PointsPlusScaleRegr3D(Criterion, MultiLoss):
    """
    Regression Loss for World Frame Pointmaps & Scale.
    """

    def __init__(
        self,
        criterion,
        norm_predictions=True,
        norm_mode="avg_dis",
        ambiguous_loss_value=0,
        loss_in_log=True,
        flatten_across_image_only=False,
        world_frame_points_loss_weight=1,
        scale_loss_weight=1,
    ):
        """
        Initialize the loss criterion for World Frame Pointmaps & Scale.
        The predicted scene representation is always normalized w.r.t. the frame of view0.
        Loss is applied between the predicted metric scale and the ground truth metric scale.

        Args:
            criterion (BaseCriterion): The base criterion to use for computing the loss.
            norm_predictions (bool): If True, normalize the predictions before computing the loss.
            norm_mode (str): Normalization mode for the gt and predicted (optional) scene representation. Default: "avg_dis".
            ambiguous_loss_value (float): Value to use for ambiguous pixels in the loss.
                If 0, ambiguous pixels are ignored. Default: 0.
            loss_in_log (bool): If True, apply logarithmic transformation to input before
                computing the loss for depth, pointmaps and scale. Default: True.
            flatten_across_image_only (bool): If True, flatten H x W dimensions only when computing
                the loss. If False, flatten across batch and spatial dimensions. Default: False.
            world_frame_points_loss_weight (float): Weight to use for the world frame pointmap loss. Default: 1.
            scale_loss_weight (float): Weight to use for the scale loss. Default: 1.
        """
        super().__init__(criterion)
        self.norm_predictions = norm_predictions
        self.norm_mode = norm_mode
        self.ambiguous_loss_value = ambiguous_loss_value
        self.loss_in_log = loss_in_log
        self.flatten_across_image_only = flatten_across_image_only
        self.world_frame_points_loss_weight = world_frame_points_loss_weight
        self.scale_loss_weight = scale_loss_weight

    def get_all_info(self, batch, preds, dist_clip=None):
        """
        Function to get all the information needed to compute the loss.
        Returns all quantities normalized w.r.t. camera of view0.
        """
        n_views = len(batch)

        # Everything is normalized w.r.t. camera of view0
        # Initialize lists to store data for all views
        # Ground truth quantities
        in_camera0 = closed_form_pose_inverse(batch[0]["camera_pose"])
        no_norm_gt_pts = []
        valid_masks = []
        # Predicted quantities
        no_norm_pr_pts = []
        metric_pr_pts_to_compute_scale = []

        # Get ground truth & prediction info for all views
        for i in range(n_views):
            # Get the ground truth
            no_norm_gt_pts.append(geotrf(in_camera0, batch[i]["pts3d"]))
            valid_masks.append(batch[i]["valid_mask"].clone())

            # Get predictions for normalized loss
            if "metric_scaling_factor" in preds[i].keys():
                # Divide by the predicted metric scaling factor to get the raw predicted points, depth_along_ray, and pose_trans
                # This detaches the predicted metric scaling factor from the geometry based loss
                curr_view_no_norm_pr_pts = preds[i]["pts3d"] / preds[i][
                    "metric_scaling_factor"
                ].unsqueeze(-1).unsqueeze(-1)
            else:
                curr_view_no_norm_pr_pts = preds[i]["pts3d"]
            no_norm_pr_pts.append(curr_view_no_norm_pr_pts)

            # Get the predicted metric scale points
            if "metric_scaling_factor" in preds[i].keys():
                # Detach the raw predicted points so that the scale loss is only applied to the scaling factor
                curr_view_metric_pr_pts_to_compute_scale = (
                    curr_view_no_norm_pr_pts.detach()
                    * preds[i]["metric_scaling_factor"].unsqueeze(-1).unsqueeze(-1)
                )
            else:
                curr_view_metric_pr_pts_to_compute_scale = (
                    curr_view_no_norm_pr_pts.clone()
                )
            metric_pr_pts_to_compute_scale.append(
                curr_view_metric_pr_pts_to_compute_scale
            )

        if dist_clip is not None:
            # Points that are too far-away == invalid
            for i in range(n_views):
                dis = no_norm_gt_pts[i].norm(dim=-1)
                valid_masks[i] = valid_masks[i] & (dis <= dist_clip)

        # Initialize normalized tensors
        gt_pts = [torch.zeros_like(pts) for pts in no_norm_gt_pts]
        pr_pts = [torch.zeros_like(pts) for pts in no_norm_pr_pts]

        # Normalize the predicted points if specified
        if self.norm_predictions:
            pr_normalization_output = normalize_multiple_pointclouds(
                no_norm_pr_pts,
                valid_masks,
                self.norm_mode,
                ret_factor=True,
            )
            pr_pts_norm = pr_normalization_output[:-1]

        # Normalize the ground truth points
        gt_normalization_output = normalize_multiple_pointclouds(
            no_norm_gt_pts, valid_masks, self.norm_mode, ret_factor=True
        )
        gt_pts_norm = gt_normalization_output[:-1]
        gt_norm_factor = gt_normalization_output[-1]

        for i in range(n_views):
            if self.norm_predictions:
                # Assign the normalized predictions
                pr_pts[i] = pr_pts_norm[i]
            else:
                pr_pts[i] = no_norm_pr_pts[i]
            # Assign the normalized ground truth quantities
            gt_pts[i] = gt_pts_norm[i]

        # Get the mask indicating ground truth metric scale quantities
        metric_scale_mask = batch[0]["is_metric_scale"]
        valid_gt_norm_factor_mask = (
            gt_norm_factor[:, 0, 0, 0] > 1e-8
        )  # Mask out cases where depth for all views is invalid
        valid_metric_scale_mask = metric_scale_mask & valid_gt_norm_factor_mask

        if valid_metric_scale_mask.any():
            # Compute the scale norm factor using the predicted metric scale points
            metric_pr_normalization_output = normalize_multiple_pointclouds(
                metric_pr_pts_to_compute_scale,
                valid_masks,
                self.norm_mode,
                ret_factor=True,
            )
            pr_metric_norm_factor = metric_pr_normalization_output[-1]

            # Get the valid ground truth and predicted scale norm factors for the metric ground truth quantities
            gt_metric_norm_factor = gt_norm_factor[valid_metric_scale_mask]
            pr_metric_norm_factor = pr_metric_norm_factor[valid_metric_scale_mask]
        else:
            gt_metric_norm_factor = None
            pr_metric_norm_factor = None

        # Get ambiguous masks
        ambiguous_masks = []
        for i in range(n_views):
            ambiguous_masks.append(
                (~batch[i]["non_ambiguous_mask"]) & (~valid_masks[i])
            )

        # Pack into info dicts
        gt_info = []
        pred_info = []
        for i in range(n_views):
            gt_info.append(
                {
                    "pts3d": gt_pts[i],
                }
            )
            pred_info.append(
                {
                    "pts3d": pr_pts[i],
                }
            )

        return (
            gt_info,
            pred_info,
            valid_masks,
            ambiguous_masks,
            gt_metric_norm_factor,
            pr_metric_norm_factor,
        )

    def compute_loss(self, batch, preds, **kw):
        (
            gt_info,
            pred_info,
            valid_masks,
            ambiguous_masks,
            gt_metric_norm_factor,
            pr_metric_norm_factor,
        ) = self.get_all_info(batch, preds, **kw)
        n_views = len(batch)

        if self.ambiguous_loss_value > 0:
            assert self.criterion.reduction == "none", (
                "ambiguous_loss_value should be 0 if no conf loss"
            )
            # Add the ambiguous pixel as "valid" pixels...
            valid_masks = [
                mask | ambig_mask
                for mask, ambig_mask in zip(valid_masks, ambiguous_masks)
            ]

        pts3d_losses = []

        for i in range(n_views):
            # Get the predicted dense quantities
            if not self.flatten_across_image_only:
                # Flatten the points across the entire batch with the masks
                pred_pts3d = pred_info[i]["pts3d"][valid_masks[i]]
                gt_pts3d = gt_info[i]["pts3d"][valid_masks[i]]
            else:
                # Flatten the H x W dimensions to H*W
                batch_size, _, _, pts_dim = gt_info[i]["pts3d"].shape
                gt_pts3d = gt_info[i]["pts3d"].view(batch_size, -1, pts_dim)
                pred_pts3d = pred_info[i]["pts3d"].view(batch_size, -1, pts_dim)
                valid_masks[i] = valid_masks[i].view(batch_size, -1)

            # Apply loss in log space if specified
            if self.loss_in_log:
                gt_pts3d = apply_log_to_norm(gt_pts3d)
                pred_pts3d = apply_log_to_norm(pred_pts3d)

            # Compute point loss
            pts3d_loss = self.criterion(pred_pts3d, gt_pts3d, factor="points")
            pts3d_loss = pts3d_loss * self.world_frame_points_loss_weight
            pts3d_losses.append(pts3d_loss)

            # Handle ambiguous pixels
            if self.ambiguous_loss_value > 0:
                if not self.flatten_across_image_only:
                    pts3d_losses[i] = torch.where(
                        ambiguous_masks[i][valid_masks[i]],
                        self.ambiguous_loss_value,
                        pts3d_losses[i],
                    )
                else:
                    pts3d_losses[i] = torch.where(
                        ambiguous_masks[i].view(ambiguous_masks[i].shape[0], -1),
                        self.ambiguous_loss_value,
                        pts3d_losses[i],
                    )

        # Compute the scale loss
        if gt_metric_norm_factor is not None:
            if self.loss_in_log:
                gt_metric_norm_factor = apply_log_to_norm(gt_metric_norm_factor)
                pr_metric_norm_factor = apply_log_to_norm(pr_metric_norm_factor)
            scale_loss = (
                self.criterion(
                    pr_metric_norm_factor, gt_metric_norm_factor, factor="scale"
                )
                * self.scale_loss_weight
            )
        else:
            scale_loss = None

        # Use helper function to generate loss terms and details

        losses_dict = {
            "pts3d": {
                "values": pts3d_losses,
                "use_mask": True,
                "is_multi_view": True,
            },
            "scale": {
                "values": scale_loss,
                "use_mask": False,
                "is_multi_view": False,
            },
        }

        loss_terms, details = get_loss_terms_and_details(
            losses_dict,
            valid_masks,
            type(self).__name__,
            n_views,
            self.flatten_across_image_only,
        )
        losses = Sum(*loss_terms)

        return losses, (details | {})


class NormalGMLoss(MultiLoss):
    """
    Normal & Gradient Matching Loss for Monocular Depth Training.
    """

    def __init__(
        self,
        norm_predictions=True,
        norm_mode="avg_dis",
        apply_normal_and_gm_loss_to_synthetic_data_only=True,
    ):
        """
        Initialize the loss criterion for Normal & Gradient Matching Loss (currently only valid for 1 view).
        Computes:
        (1) Normal Loss over the PointMap (naturally will be in local frame) in euclidean coordinates,
        (2) Gradient Matching (GM) Loss over the Depth Z in log space. (MiDAS applied GM loss in disparity space)

        Args:
            norm_predictions (bool): If True, normalize the predictions before computing the loss.
            norm_mode (str): Normalization mode for the gt and predicted (optional) scene representation. Default: "avg_dis".
            apply_normal_and_gm_loss_to_synthetic_data_only (bool): If True, apply the normal and gm loss only to synthetic data.
                If False, apply the normal and gm loss to all data. Default: True.
        """
        super().__init__()
        self.norm_predictions = norm_predictions
        self.norm_mode = norm_mode
        self.apply_normal_and_gm_loss_to_synthetic_data_only = (
            apply_normal_and_gm_loss_to_synthetic_data_only
        )

    def get_all_info(self, batch, preds, dist_clip=None):
        """
        Function to get all the information needed to compute the loss.
        Returns all quantities normalized.
        """
        n_views = len(batch)
        assert n_views == 1, (
            "Normal & Gradient Matching Loss Class only supports 1 view"
        )

        # Everything is normalized w.r.t. camera of view1
        in_camera1 = closed_form_pose_inverse(batch[0]["camera_pose"])

        # Initialize lists to store data for all views
        no_norm_gt_pts = []
        valid_masks = []
        no_norm_pr_pts = []

        # Get ground truth & prediction info for all views
        for i in range(n_views):
            # Get ground truth
            no_norm_gt_pts.append(geotrf(in_camera1, batch[i]["pts3d"]))
            valid_masks.append(batch[i]["valid_mask"].clone())

            # Get predictions for normalized loss
            if "metric_scaling_factor" in preds[i].keys():
                # Divide by the predicted metric scaling factor to get the raw predicted points
                # This detaches the predicted metric scaling factor from the geometry based loss
                curr_view_no_norm_pr_pts = preds[i]["pts3d"] / preds[i][
                    "metric_scaling_factor"
                ].unsqueeze(-1).unsqueeze(-1)
            else:
                curr_view_no_norm_pr_pts = preds[i]["pts3d"]
            no_norm_pr_pts.append(curr_view_no_norm_pr_pts)

        if dist_clip is not None:
            # Points that are too far-away == invalid
            for i in range(n_views):
                dis = no_norm_gt_pts[i].norm(dim=-1)
                valid_masks[i] = valid_masks[i] & (dis <= dist_clip)

        # Initialize normalized tensors
        gt_pts = [torch.zeros_like(pts) for pts in no_norm_gt_pts]
        pr_pts = [torch.zeros_like(pts) for pts in no_norm_pr_pts]

        # Normalize the predicted points if specified
        if self.norm_predictions:
            pr_normalization_output = normalize_multiple_pointclouds(
                no_norm_pr_pts,
                valid_masks,
                self.norm_mode,
                ret_factor=True,
            )
            pr_pts_norm = pr_normalization_output[:-1]

        # Normalize the ground truth points
        gt_normalization_output = normalize_multiple_pointclouds(
            no_norm_gt_pts, valid_masks, self.norm_mode, ret_factor=True
        )
        gt_pts_norm = gt_normalization_output[:-1]

        for i in range(n_views):
            if self.norm_predictions:
                # Assign the normalized predictions
                pr_pts[i] = pr_pts_norm[i]
            else:
                # Assign the raw predicted points
                pr_pts[i] = no_norm_pr_pts[i]
            # Assign the normalized ground truth
            gt_pts[i] = gt_pts_norm[i]

        return gt_pts, pr_pts, valid_masks

    def compute_loss(self, batch, preds, **kw):
        gt_pts, pred_pts, valid_masks = self.get_all_info(batch, preds, **kw)
        n_views = len(batch)
        assert n_views == 1, (
            "Normal & Gradient Matching Loss Class only supports 1 view"
        )

        normal_losses = []
        gradient_matching_losses = []
        details = {}
        running_avg_dict = {}
        self_name = type(self).__name__

        for i in range(n_views):
            # Get the local frame points, log space depth_z & valid masks
            pred_local_pts3d = pred_pts[i]
            pred_depth_z = pred_local_pts3d[..., 2:]
            pred_depth_z = apply_log_to_norm(pred_depth_z)
            gt_local_pts3d = gt_pts[i]
            gt_depth_z = gt_local_pts3d[..., 2:]
            gt_depth_z = apply_log_to_norm(gt_depth_z)
            valid_mask_for_normal_gm_loss = valid_masks[i].clone()

            # Update the validity mask for normal & gm loss based on the synthetic data mask if required
            if self.apply_normal_and_gm_loss_to_synthetic_data_only:
                synthetic_mask = batch[i]["is_synthetic"]  # (B, )
                synthetic_mask = synthetic_mask.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                synthetic_mask = synthetic_mask.expand(
                    -1, pred_depth_z.shape[1], pred_depth_z.shape[2]
                )  # (B, H, W)
                valid_mask_for_normal_gm_loss = (
                    valid_mask_for_normal_gm_loss & synthetic_mask
                )

            # Compute the normal loss
            normal_loss = compute_normal_loss(
                pred_local_pts3d, gt_local_pts3d, valid_mask_for_normal_gm_loss.clone()
            )
            normal_losses.append(normal_loss)

            # Compute the gradient matching loss
            gradient_matching_loss = compute_gradient_matching_loss(
                pred_depth_z, gt_depth_z, valid_mask_for_normal_gm_loss.clone()
            )
            gradient_matching_losses.append(gradient_matching_loss)

            # Add loss details if only valid values are present
            # Initialize or update running average directly
            # Normal loss details
            if float(normal_loss) > 0:
                details[f"{self_name}_normal_view{i + 1}"] = float(normal_loss)
                normal_avg_key = f"{self_name}_normal_avg"
                if normal_avg_key not in details:
                    details[normal_avg_key] = float(normal_losses[i])
                    running_avg_dict[f"{self_name}_normal_valid_views"] = 1
                else:
                    normal_valid_views = (
                        running_avg_dict[f"{self_name}_normal_valid_views"] + 1
                    )
                    running_avg_dict[f"{self_name}_normal_valid_views"] = (
                        normal_valid_views
                    )
                    details[normal_avg_key] += (
                        float(normal_losses[i]) - details[normal_avg_key]
                    ) / normal_valid_views

            # Gradient Matching loss details
            if float(gradient_matching_loss) > 0:
                details[f"{self_name}_gradient_matching_view{i + 1}"] = float(
                    gradient_matching_loss
                )
                # For gradient matching loss
                gm_avg_key = f"{self_name}_gradient_matching_avg"
                if gm_avg_key not in details:
                    details[gm_avg_key] = float(gradient_matching_losses[i])
                    running_avg_dict[f"{self_name}_gm_valid_views"] = 1
                else:
                    gm_valid_views = running_avg_dict[f"{self_name}_gm_valid_views"] + 1
                    running_avg_dict[f"{self_name}_gm_valid_views"] = gm_valid_views
                    details[gm_avg_key] += (
                        float(gradient_matching_losses[i]) - details[gm_avg_key]
                    ) / gm_valid_views

        # Put the losses together
        loss_terms = []
        for i in range(n_views):
            loss_terms.append((normal_losses[i], None, "normal"))
            loss_terms.append((gradient_matching_losses[i], None, "gradient_matching"))
        losses = Sum(*loss_terms)

        return losses, details


class FactoredGeometryRegr3D(Criterion, MultiLoss):
    """
    Regression Loss for Factored Geometry.
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
        depth_type_for_loss="depth_along_ray",
        cam_frame_points_loss_weight=1,
        depth_loss_weight=1,
        ray_directions_loss_weight=1,
        pose_quats_loss_weight=1,
        pose_trans_loss_weight=1,
        compute_absolute_pose_loss=True,
        compute_pairwise_relative_pose_loss=False,
        convert_predictions_to_view0_frame=False,
        compute_world_frame_points_loss=True,
        world_frame_points_loss_weight=1,
    ):
        """
        Initialize the loss criterion for Factored Geometry (Ray Directions, Depth, Pose),
        and the Collective Geometry i.e. Local Frame Pointmaps & optionally World Frame Pointmaps.
        If world-frame pointmap loss is computed, the pixel-level losses are computed in the following order:
        (1) world points, (2) cam points, (3) depth, (4) ray directions, (5) pose quats, (6) pose trans.
        Else, the pixel-level losses are returned in the following order:
        (1) cam points, (2) depth, (3) ray directions, (4) pose quats, (5) pose trans.

        Args:
            criterion (BaseCriterion): The base criterion to use for computing the loss.
            norm_mode (str): Normalization mode for scene representation. Default: "?avg_dis".
                If prefixed with "?", normalization is only applied to non-metric scale data.
            gt_scale (bool): If True, enforce predictions to have the same scale as ground truth.
                If False, both GT and predictions are normalized independently. Default: False.
            ambiguous_loss_value (float): Value to use for ambiguous pixels in the loss.
                If 0, ambiguous pixels are ignored. Default: 0.
            max_metric_scale (float): Maximum scale for metric scale data. If data exceeds this
                value, it will be treated as non-metric. Default: False (no limit).
            loss_in_log (bool): If True, apply logarithmic transformation to input before
                computing the loss for depth and pointmaps. Default: True.
            flatten_across_image_only (bool): If True, flatten H x W dimensions only when computing
                the loss. If False, flatten across batch and spatial dimensions. Default: False.
            depth_type_for_loss (str): Type of depth to use for loss computation. Default: "depth_along_ray".
                Options: "depth_along_ray", "depth_z"
            cam_frame_points_loss_weight (float): Weight to use for the camera frame pointmap loss. Default: 1.
            depth_loss_weight (float): Weight to use for the depth loss. Default: 1.
            ray_directions_loss_weight (float): Weight to use for the ray directions loss. Default: 1.
            pose_quats_loss_weight (float): Weight to use for the pose quats loss. Default: 1.
            pose_trans_loss_weight (float): Weight to use for the pose trans loss. Default: 1.
            compute_absolute_pose_loss (bool): If True, compute the absolute pose loss. Default: True.
            compute_pairwise_relative_pose_loss (bool): If True, the pose loss is computed on the
                exhaustive pairwise relative poses. Default: False.
            convert_predictions_to_view0_frame (bool): If True, convert predictions to view0 frame.
                Use this if the predictions are not already in the view0 frame. Default: False.
            compute_world_frame_points_loss (bool): If True, compute the world frame pointmap loss. Default: True.
            world_frame_points_loss_weight (float): Weight to use for the world frame pointmap loss. Default: 1.
        """
        super().__init__(criterion)
        if norm_mode.startswith("?"):
            # Do no norm pts from metric scale datasets
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
        self.depth_type_for_loss = depth_type_for_loss
        assert self.depth_type_for_loss in [
            "depth_along_ray",
            "depth_z",
        ], "depth_type_for_loss must be one of ['depth_along_ray', 'depth_z']"
        self.cam_frame_points_loss_weight = cam_frame_points_loss_weight
        self.depth_loss_weight = depth_loss_weight
        self.ray_directions_loss_weight = ray_directions_loss_weight
        self.pose_quats_loss_weight = pose_quats_loss_weight
        self.pose_trans_loss_weight = pose_trans_loss_weight
        self.compute_absolute_pose_loss = compute_absolute_pose_loss
        self.compute_pairwise_relative_pose_loss = compute_pairwise_relative_pose_loss
        self.convert_predictions_to_view0_frame = convert_predictions_to_view0_frame
        self.compute_world_frame_points_loss = compute_world_frame_points_loss
        self.world_frame_points_loss_weight = world_frame_points_loss_weight

    def get_all_info(self, batch, preds, dist_clip=None):
        """
        Function to get all the information needed to compute the loss.
        Returns all quantities normalized w.r.t. camera of view0.
        """
        n_views = len(batch)

        # Everything is normalized w.r.t. camera of view0
        # Initialize lists to store data for all views
        # Ground truth quantities
        in_camera0 = closed_form_pose_inverse(batch[0]["camera_pose"])
        no_norm_gt_pts = []
        no_norm_gt_pts_cam = []
        no_norm_gt_depth = []
        no_norm_gt_pose_trans = []
        valid_masks = []
        gt_ray_directions = []
        gt_pose_quats = []
        # Predicted quantities
        if self.convert_predictions_to_view0_frame:
            # Get the camera transform to convert quantities to view0 frame
            pred_camera0 = torch.eye(4, device=preds[0]["cam_quats"].device).unsqueeze(
                0
            )
            batch_size = preds[0]["cam_quats"].shape[0]
            pred_camera0 = pred_camera0.repeat(batch_size, 1, 1)
            pred_camera0_rot = quaternion_to_rotation_matrix(
                preds[0]["cam_quats"].clone()
            )
            pred_camera0[..., :3, :3] = pred_camera0_rot
            pred_camera0[..., :3, 3] = preds[0]["cam_trans"].clone()
            pred_in_camera0 = closed_form_pose_inverse(pred_camera0)
        no_norm_pr_pts = []
        no_norm_pr_pts_cam = []
        no_norm_pr_depth = []
        no_norm_pr_pose_trans = []
        pr_ray_directions = []
        pr_pose_quats = []

        # Get ground truth & prediction info for all views
        for i in range(n_views):
            # Get ground truth
            no_norm_gt_pts.append(geotrf(in_camera0, batch[i]["pts3d"]))
            valid_masks.append(batch[i]["valid_mask"].clone())
            no_norm_gt_pts_cam.append(batch[i]["pts3d_cam"])
            gt_ray_directions.append(batch[i]["ray_directions_cam"])
            if self.depth_type_for_loss == "depth_along_ray":
                no_norm_gt_depth.append(batch[i]["depth_along_ray"])
            elif self.depth_type_for_loss == "depth_z":
                no_norm_gt_depth.append(batch[i]["pts3d_cam"][..., 2:])
            if i == 0:
                # For view0, initialize identity pose
                gt_pose_quats.append(
                    torch.tensor(
                        [0, 0, 0, 1],
                        dtype=gt_ray_directions[0].dtype,
                        device=gt_ray_directions[0].device,
                    )
                    .unsqueeze(0)
                    .repeat(gt_ray_directions[0].shape[0], 1)
                )
                no_norm_gt_pose_trans.append(
                    torch.tensor(
                        [0, 0, 0],
                        dtype=gt_ray_directions[0].dtype,
                        device=gt_ray_directions[0].device,
                    )
                    .unsqueeze(0)
                    .repeat(gt_ray_directions[0].shape[0], 1)
                )
            else:
                # For other views, transform pose to view0's frame
                gt_pose_quats_world = batch[i]["camera_pose_quats"]
                no_norm_gt_pose_trans_world = batch[i]["camera_pose_trans"]
                gt_pose_quats_in_view0, no_norm_gt_pose_trans_in_view0 = (
                    transform_pose_using_quats_and_trans_2_to_1(
                        batch[0]["camera_pose_quats"],
                        batch[0]["camera_pose_trans"],
                        gt_pose_quats_world,
                        no_norm_gt_pose_trans_world,
                    )
                )
                gt_pose_quats.append(gt_pose_quats_in_view0)
                no_norm_gt_pose_trans.append(no_norm_gt_pose_trans_in_view0)

            # Get the local predictions
            no_norm_pr_pts_cam.append(preds[i]["pts3d_cam"])
            pr_ray_directions.append(preds[i]["ray_directions"])
            if self.depth_type_for_loss == "depth_along_ray":
                no_norm_pr_depth.append(preds[i]["depth_along_ray"])
            elif self.depth_type_for_loss == "depth_z":
                no_norm_pr_depth.append(preds[i]["pts3d_cam"][..., 2:])

            # Get the predicted global predictions in view0's frame
            if self.convert_predictions_to_view0_frame:
                # Convert predictions to view0 frame
                pr_pts3d_in_view0 = geotrf(pred_in_camera0, preds[i]["pts3d"])
                pr_pose_quats_in_view0, pr_pose_trans_in_view0 = (
                    transform_pose_using_quats_and_trans_2_to_1(
                        preds[0]["cam_quats"],
                        preds[0]["cam_trans"],
                        preds[i]["cam_quats"],
                        preds[i]["cam_trans"],
                    )
                )
                no_norm_pr_pts.append(pr_pts3d_in_view0)
                no_norm_pr_pose_trans.append(pr_pose_trans_in_view0)
                pr_pose_quats.append(pr_pose_quats_in_view0)
            else:
                # Predictions are already in view0 frame
                no_norm_pr_pts.append(preds[i]["pts3d"])
                no_norm_pr_pose_trans.append(preds[i]["cam_trans"])
                pr_pose_quats.append(preds[i]["cam_quats"])

        if dist_clip is not None:
            # Points that are too far-away == invalid
            for i in range(n_views):
                dis = no_norm_gt_pts[i].norm(dim=-1)
                valid_masks[i] = valid_masks[i] & (dis <= dist_clip)

        # Handle metric scale
        if not self.norm_all:
            if self.max_metric_scale:
                B = valid_masks[0].shape[0]
                dists_to_cam1 = []
                for i in range(n_views):
                    dists_to_cam1.append(
                        torch.where(
                            valid_masks[i], torch.norm(no_norm_gt_pts[i], dim=-1), 0
                        ).view(B, -1)
                    )

                batch[0]["is_metric_scale"] = batch[0]["is_metric_scale"]
                for dist in dists_to_cam1:
                    batch[0]["is_metric_scale"] &= (
                        dist.max(dim=-1).values < self.max_metric_scale
                    )

                for i in range(1, n_views):
                    batch[i]["is_metric_scale"] = batch[0]["is_metric_scale"]

            non_metric_scale_mask = ~batch[0]["is_metric_scale"]
        else:
            non_metric_scale_mask = torch.ones_like(batch[0]["is_metric_scale"])

        # Initialize normalized tensors
        gt_pts = [torch.zeros_like(pts) for pts in no_norm_gt_pts]
        gt_pts_cam = [torch.zeros_like(pts_cam) for pts_cam in no_norm_gt_pts_cam]
        gt_depth = [torch.zeros_like(depth) for depth in no_norm_gt_depth]
        gt_pose_trans = [torch.zeros_like(trans) for trans in no_norm_gt_pose_trans]

        pr_pts = [torch.zeros_like(pts) for pts in no_norm_pr_pts]
        pr_pts_cam = [torch.zeros_like(pts_cam) for pts_cam in no_norm_pr_pts_cam]
        pr_depth = [torch.zeros_like(depth) for depth in no_norm_pr_depth]
        pr_pose_trans = [torch.zeros_like(trans) for trans in no_norm_pr_pose_trans]

        # Normalize points
        if self.norm_mode and non_metric_scale_mask.any():
            pr_normalization_output = normalize_multiple_pointclouds(
                [pts[non_metric_scale_mask] for pts in no_norm_pr_pts],
                [mask[non_metric_scale_mask] for mask in valid_masks],
                self.norm_mode,
                ret_factor=True,
            )
            pr_pts_norm = pr_normalization_output[:-1]
            pr_norm_factor = pr_normalization_output[-1]

            for i in range(n_views):
                pr_pts[i][non_metric_scale_mask] = pr_pts_norm[i]
                pr_pts_cam[i][non_metric_scale_mask] = (
                    no_norm_pr_pts_cam[i][non_metric_scale_mask] / pr_norm_factor
                )
                pr_depth[i][non_metric_scale_mask] = (
                    no_norm_pr_depth[i][non_metric_scale_mask] / pr_norm_factor
                )
                pr_pose_trans[i][non_metric_scale_mask] = (
                    no_norm_pr_pose_trans[i][non_metric_scale_mask]
                    / pr_norm_factor[:, :, 0, 0]
                )

        elif non_metric_scale_mask.any():
            for i in range(n_views):
                pr_pts[i][non_metric_scale_mask] = no_norm_pr_pts[i][
                    non_metric_scale_mask
                ]
                pr_pts_cam[i][non_metric_scale_mask] = no_norm_pr_pts_cam[i][
                    non_metric_scale_mask
                ]
                pr_depth[i][non_metric_scale_mask] = no_norm_pr_depth[i][
                    non_metric_scale_mask
                ]
                pr_pose_trans[i][non_metric_scale_mask] = no_norm_pr_pose_trans[i][
                    non_metric_scale_mask
                ]

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

                pr_pts[i][~non_metric_scale_mask] = (
                    no_norm_pr_pts[i][~non_metric_scale_mask]
                    / norm_factor[~non_metric_scale_mask]
                )
                pr_pts_cam[i][~non_metric_scale_mask] = (
                    no_norm_pr_pts_cam[i][~non_metric_scale_mask]
                    / norm_factor[~non_metric_scale_mask]
                )
                pr_depth[i][~non_metric_scale_mask] = (
                    no_norm_pr_depth[i][~non_metric_scale_mask]
                    / norm_factor[~non_metric_scale_mask]
                )
                pr_pose_trans[i][~non_metric_scale_mask] = (
                    no_norm_pr_pose_trans[i][~non_metric_scale_mask]
                    / norm_factor[~non_metric_scale_mask][:, :, 0, 0]
                )

        elif ~non_metric_scale_mask.any():
            for i in range(n_views):
                gt_pts[i] = no_norm_gt_pts[i]
                gt_pts_cam[i] = no_norm_gt_pts_cam[i]
                gt_depth[i] = no_norm_gt_depth[i]
                gt_pose_trans[i] = no_norm_gt_pose_trans[i]
                pr_pts[i][~non_metric_scale_mask] = no_norm_pr_pts[i][
                    ~non_metric_scale_mask
                ]
                pr_pts_cam[i][~non_metric_scale_mask] = no_norm_pr_pts_cam[i][
                    ~non_metric_scale_mask
                ]
                pr_depth[i][~non_metric_scale_mask] = no_norm_pr_depth[i][
                    ~non_metric_scale_mask
                ]
                pr_pose_trans[i][~non_metric_scale_mask] = no_norm_pr_pose_trans[i][
                    ~non_metric_scale_mask
                ]
        else:
            for i in range(n_views):
                gt_pts[i] = no_norm_gt_pts[i]
                gt_pts_cam[i] = no_norm_gt_pts_cam[i]
                gt_depth[i] = no_norm_gt_depth[i]
                gt_pose_trans[i] = no_norm_gt_pose_trans[i]

        # Get ambiguous masks
        ambiguous_masks = []
        for i in range(n_views):
            ambiguous_masks.append(
                (~batch[i]["non_ambiguous_mask"]) & (~valid_masks[i])
            )

        # Pack into info dicts
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

    def compute_loss(self, batch, preds, **kw):
        gt_info, pred_info, valid_masks, ambiguous_masks = self.get_all_info(
            batch, preds, **kw
        )
        n_views = len(batch)

        # Mask out samples in the batch where the gt depth validity mask is entirely zero
        valid_norm_factor_masks = [
            mask.sum(dim=(1, 2)) > 0 for mask in valid_masks
        ]  # List of (B,)

        if self.ambiguous_loss_value > 0:
            assert self.criterion.reduction == "none", (
                "ambiguous_loss_value should be 0 if no conf loss"
            )
            # Add the ambiguous pixel as "valid" pixels...
            valid_masks = [
                mask | ambig_mask
                for mask, ambig_mask in zip(valid_masks, ambiguous_masks)
            ]

        pose_trans_losses = []
        pose_quats_losses = []
        ray_directions_losses = []
        depth_losses = []
        cam_pts3d_losses = []
        if self.compute_world_frame_points_loss:
            pts3d_losses = []

        for i in range(n_views):
            # Get the predicted dense quantities
            if not self.flatten_across_image_only:
                # Flatten the points across the entire batch with the masks
                pred_ray_directions = pred_info[i]["ray_directions"]
                gt_ray_directions = gt_info[i]["ray_directions"]
                pred_depth = pred_info[i][self.depth_type_for_loss][valid_masks[i]]
                gt_depth = gt_info[i][self.depth_type_for_loss][valid_masks[i]]
                pred_cam_pts3d = pred_info[i]["pts3d_cam"][valid_masks[i]]
                gt_cam_pts3d = gt_info[i]["pts3d_cam"][valid_masks[i]]
                if self.compute_world_frame_points_loss:
                    pred_pts3d = pred_info[i]["pts3d"][valid_masks[i]]
                    gt_pts3d = gt_info[i]["pts3d"][valid_masks[i]]
            else:
                # Flatten the H x W dimensions to H*W
                batch_size, _, _, direction_dim = gt_info[i]["ray_directions"].shape
                gt_ray_directions = gt_info[i]["ray_directions"].view(
                    batch_size, -1, direction_dim
                )
                pred_ray_directions = pred_info[i]["ray_directions"].view(
                    batch_size, -1, direction_dim
                )
                depth_dim = gt_info[i][self.depth_type_for_loss].shape[-1]
                gt_depth = gt_info[i][self.depth_type_for_loss].view(
                    batch_size, -1, depth_dim
                )
                pred_depth = pred_info[i][self.depth_type_for_loss].view(
                    batch_size, -1, depth_dim
                )
                cam_pts_dim = gt_info[i]["pts3d_cam"].shape[-1]
                gt_cam_pts3d = gt_info[i]["pts3d_cam"].view(batch_size, -1, cam_pts_dim)
                pred_cam_pts3d = pred_info[i]["pts3d_cam"].view(
                    batch_size, -1, cam_pts_dim
                )
                if self.compute_world_frame_points_loss:
                    pts_dim = gt_info[i]["pts3d"].shape[-1]
                    gt_pts3d = gt_info[i]["pts3d"].view(batch_size, -1, pts_dim)
                    pred_pts3d = pred_info[i]["pts3d"].view(batch_size, -1, pts_dim)
                valid_masks[i] = valid_masks[i].view(batch_size, -1)

            # Apply loss in log space for depth if specified
            if self.loss_in_log:
                gt_depth = apply_log_to_norm(gt_depth)
                pred_depth = apply_log_to_norm(pred_depth)
                gt_cam_pts3d = apply_log_to_norm(gt_cam_pts3d)
                pred_cam_pts3d = apply_log_to_norm(pred_cam_pts3d)
                if self.compute_world_frame_points_loss:
                    gt_pts3d = apply_log_to_norm(gt_pts3d)
                    pred_pts3d = apply_log_to_norm(pred_pts3d)

            # Compute pose loss
            if (
                self.compute_absolute_pose_loss
                and self.compute_pairwise_relative_pose_loss
            ):
                # Compute the absolute pose loss
                # Get the pose info for the current view
                pred_pose_trans = pred_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                gt_pose_trans = gt_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                pred_pose_quats = pred_info[i]["pose_quats"]
                gt_pose_quats = gt_info[i]["pose_quats"]

                # Compute pose translation loss
                abs_pose_trans_loss = self.criterion(
                    pred_pose_trans, gt_pose_trans, factor="pose_trans"
                )
                abs_pose_trans_loss = abs_pose_trans_loss * self.pose_trans_loss_weight

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                abs_pose_quats_loss = torch.minimum(
                    self.criterion(pred_pose_quats, gt_pose_quats, factor="pose_quats"),
                    self.criterion(
                        pred_pose_quats, -gt_pose_quats, factor="pose_quats"
                    ),
                )
                abs_pose_quats_loss = abs_pose_quats_loss * self.pose_quats_loss_weight

                # Compute the pairwise relative pose loss
                # Get the inverse of current view predicted pose
                pred_inv_curr_view_pose_quats = quaternion_inverse(
                    pred_info[i]["pose_quats"]
                )
                pred_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    pred_inv_curr_view_pose_quats
                )
                pred_inv_curr_view_pose_trans = -1 * ein.einsum(
                    pred_inv_curr_view_pose_rot_mat,
                    pred_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the inverse of the current view GT pose
                gt_inv_curr_view_pose_quats = quaternion_inverse(
                    gt_info[i]["pose_quats"]
                )
                gt_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    gt_inv_curr_view_pose_quats
                )
                gt_inv_curr_view_pose_trans = -1 * ein.einsum(
                    gt_inv_curr_view_pose_rot_mat,
                    gt_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the other N-1 relative poses using the current pose as reference frame
                pred_rel_pose_quats = []
                pred_rel_pose_trans = []
                gt_rel_pose_quats = []
                gt_rel_pose_trans = []
                for ov_idx in range(n_views):
                    if ov_idx == i:
                        continue
                    # Get the relative predicted pose
                    pred_ov_rel_pose_quats = quaternion_multiply(
                        pred_inv_curr_view_pose_quats, pred_info[ov_idx]["pose_quats"]
                    )
                    pred_ov_rel_pose_trans = (
                        ein.einsum(
                            pred_inv_curr_view_pose_rot_mat,
                            pred_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + pred_inv_curr_view_pose_trans
                    )

                    # Get the relative GT pose
                    gt_ov_rel_pose_quats = quaternion_multiply(
                        gt_inv_curr_view_pose_quats, gt_info[ov_idx]["pose_quats"]
                    )
                    gt_ov_rel_pose_trans = (
                        ein.einsum(
                            gt_inv_curr_view_pose_rot_mat,
                            gt_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + gt_inv_curr_view_pose_trans
                    )

                    # Get the valid translations using valid_norm_factor_masks for current view and other view
                    overall_valid_mask_for_trans = (
                        valid_norm_factor_masks[i] & valid_norm_factor_masks[ov_idx]
                    )

                    # Append the relative poses
                    pred_rel_pose_quats.append(pred_ov_rel_pose_quats)
                    pred_rel_pose_trans.append(
                        pred_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )
                    gt_rel_pose_quats.append(gt_ov_rel_pose_quats)
                    gt_rel_pose_trans.append(
                        gt_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )

                # Cat the N-1 relative poses along the batch dimension
                pred_rel_pose_quats = torch.cat(pred_rel_pose_quats, dim=0)
                pred_rel_pose_trans = torch.cat(pred_rel_pose_trans, dim=0)
                gt_rel_pose_quats = torch.cat(gt_rel_pose_quats, dim=0)
                gt_rel_pose_trans = torch.cat(gt_rel_pose_trans, dim=0)

                # Compute pose translation loss
                rel_pose_trans_loss = self.criterion(
                    pred_rel_pose_trans, gt_rel_pose_trans, factor="pose_trans"
                )
                rel_pose_trans_loss = rel_pose_trans_loss * self.pose_trans_loss_weight

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                rel_pose_quats_loss = torch.minimum(
                    self.criterion(
                        pred_rel_pose_quats, gt_rel_pose_quats, factor="pose_quats"
                    ),
                    self.criterion(
                        pred_rel_pose_quats, -gt_rel_pose_quats, factor="pose_quats"
                    ),
                )
                rel_pose_quats_loss = rel_pose_quats_loss * self.pose_quats_loss_weight

                # Concatenate the absolute and relative pose losses together
                pose_trans_loss = torch.cat(
                    [abs_pose_trans_loss, rel_pose_trans_loss], dim=0
                )
                pose_quats_loss = torch.cat(
                    [abs_pose_quats_loss, rel_pose_quats_loss], dim=0
                )
                pose_trans_losses.append(pose_trans_loss)
                pose_quats_losses.append(pose_quats_loss)
            elif self.compute_absolute_pose_loss:
                # Get the pose info for the current view
                pred_pose_trans = pred_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                gt_pose_trans = gt_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                pred_pose_quats = pred_info[i]["pose_quats"]
                gt_pose_quats = gt_info[i]["pose_quats"]

                # Compute pose translation loss
                pose_trans_loss = self.criterion(
                    pred_pose_trans, gt_pose_trans, factor="pose_trans"
                )
                pose_trans_loss = pose_trans_loss * self.pose_trans_loss_weight
                pose_trans_losses.append(pose_trans_loss)

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                pose_quats_loss = torch.minimum(
                    self.criterion(pred_pose_quats, gt_pose_quats, factor="pose_quats"),
                    self.criterion(
                        pred_pose_quats, -gt_pose_quats, factor="pose_quats"
                    ),
                )
                pose_quats_loss = pose_quats_loss * self.pose_quats_loss_weight
                pose_quats_losses.append(pose_quats_loss)
            elif self.compute_pairwise_relative_pose_loss:
                # Get the inverse of current view predicted pose
                pred_inv_curr_view_pose_quats = quaternion_inverse(
                    pred_info[i]["pose_quats"]
                )
                pred_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    pred_inv_curr_view_pose_quats
                )
                pred_inv_curr_view_pose_trans = -1 * ein.einsum(
                    pred_inv_curr_view_pose_rot_mat,
                    pred_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the inverse of the current view GT pose
                gt_inv_curr_view_pose_quats = quaternion_inverse(
                    gt_info[i]["pose_quats"]
                )
                gt_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    gt_inv_curr_view_pose_quats
                )
                gt_inv_curr_view_pose_trans = -1 * ein.einsum(
                    gt_inv_curr_view_pose_rot_mat,
                    gt_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the other N-1 relative poses using the current pose as reference frame
                pred_rel_pose_quats = []
                pred_rel_pose_trans = []
                gt_rel_pose_quats = []
                gt_rel_pose_trans = []
                for ov_idx in range(n_views):
                    if ov_idx == i:
                        continue
                    # Get the relative predicted pose
                    pred_ov_rel_pose_quats = quaternion_multiply(
                        pred_inv_curr_view_pose_quats, pred_info[ov_idx]["pose_quats"]
                    )
                    pred_ov_rel_pose_trans = (
                        ein.einsum(
                            pred_inv_curr_view_pose_rot_mat,
                            pred_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + pred_inv_curr_view_pose_trans
                    )

                    # Get the relative GT pose
                    gt_ov_rel_pose_quats = quaternion_multiply(
                        gt_inv_curr_view_pose_quats, gt_info[ov_idx]["pose_quats"]
                    )
                    gt_ov_rel_pose_trans = (
                        ein.einsum(
                            gt_inv_curr_view_pose_rot_mat,
                            gt_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + gt_inv_curr_view_pose_trans
                    )

                    # Get the valid translations using valid_norm_factor_masks for current view and other view
                    overall_valid_mask_for_trans = (
                        valid_norm_factor_masks[i] & valid_norm_factor_masks[ov_idx]
                    )

                    # Append the relative poses
                    pred_rel_pose_quats.append(pred_ov_rel_pose_quats)
                    pred_rel_pose_trans.append(
                        pred_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )
                    gt_rel_pose_quats.append(gt_ov_rel_pose_quats)
                    gt_rel_pose_trans.append(
                        gt_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )

                # Cat the N-1 relative poses along the batch dimension
                pred_rel_pose_quats = torch.cat(pred_rel_pose_quats, dim=0)
                pred_rel_pose_trans = torch.cat(pred_rel_pose_trans, dim=0)
                gt_rel_pose_quats = torch.cat(gt_rel_pose_quats, dim=0)
                gt_rel_pose_trans = torch.cat(gt_rel_pose_trans, dim=0)

                # Compute pose translation loss
                pose_trans_loss = self.criterion(
                    pred_rel_pose_trans, gt_rel_pose_trans, factor="pose_trans"
                )
                pose_trans_loss = pose_trans_loss * self.pose_trans_loss_weight
                pose_trans_losses.append(pose_trans_loss)

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                pose_quats_loss = torch.minimum(
                    self.criterion(
                        pred_rel_pose_quats, gt_rel_pose_quats, factor="pose_quats"
                    ),
                    self.criterion(
                        pred_rel_pose_quats, -gt_rel_pose_quats, factor="pose_quats"
                    ),
                )
                pose_quats_loss = pose_quats_loss * self.pose_quats_loss_weight
                pose_quats_losses.append(pose_quats_loss)
            else:
                # Error
                raise ValueError(
                    "compute_absolute_pose_loss and compute_pairwise_relative_pose_loss cannot both be False"
                )

            # Compute ray direction loss
            ray_directions_loss = self.criterion(
                pred_ray_directions, gt_ray_directions, factor="ray_directions"
            )
            ray_directions_loss = ray_directions_loss * self.ray_directions_loss_weight
            ray_directions_losses.append(ray_directions_loss)

            # Compute depth loss
            depth_loss = self.criterion(pred_depth, gt_depth, factor="depth")
            depth_loss = depth_loss * self.depth_loss_weight
            depth_losses.append(depth_loss)

            # Compute camera frame point loss
            cam_pts3d_loss = self.criterion(
                pred_cam_pts3d, gt_cam_pts3d, factor="points"
            )
            cam_pts3d_loss = cam_pts3d_loss * self.cam_frame_points_loss_weight
            cam_pts3d_losses.append(cam_pts3d_loss)

            if self.compute_world_frame_points_loss:
                # Compute point loss
                pts3d_loss = self.criterion(pred_pts3d, gt_pts3d, factor="points")
                pts3d_loss = pts3d_loss * self.world_frame_points_loss_weight
                pts3d_losses.append(pts3d_loss)

            # Handle ambiguous pixels
            if self.ambiguous_loss_value > 0:
                if not self.flatten_across_image_only:
                    depth_losses[i] = torch.where(
                        ambiguous_masks[i][valid_masks[i]],
                        self.ambiguous_loss_value,
                        depth_losses[i],
                    )
                    cam_pts3d_losses[i] = torch.where(
                        ambiguous_masks[i][valid_masks[i]],
                        self.ambiguous_loss_value,
                        cam_pts3d_losses[i],
                    )
                    if self.compute_world_frame_points_loss:
                        pts3d_losses[i] = torch.where(
                            ambiguous_masks[i][valid_masks[i]],
                            self.ambiguous_loss_value,
                            pts3d_losses[i],
                        )
                else:
                    depth_losses[i] = torch.where(
                        ambiguous_masks[i].view(ambiguous_masks[i].shape[0], -1),
                        self.ambiguous_loss_value,
                        depth_losses[i],
                    )
                    cam_pts3d_losses[i] = torch.where(
                        ambiguous_masks[i].view(ambiguous_masks[i].shape[0], -1),
                        self.ambiguous_loss_value,
                        cam_pts3d_losses[i],
                    )
                    if self.compute_world_frame_points_loss:
                        pts3d_losses[i] = torch.where(
                            ambiguous_masks[i].view(ambiguous_masks[i].shape[0], -1),
                            self.ambiguous_loss_value,
                            pts3d_losses[i],
                        )

        # Use helper function to generate loss terms and details
        if self.compute_world_frame_points_loss:
            losses_dict = {
                "pts3d": {
                    "values": pts3d_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
            }
        else:
            losses_dict = {}
        losses_dict.update(
            {
                "cam_pts3d": {
                    "values": cam_pts3d_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                self.depth_type_for_loss: {
                    "values": depth_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                "ray_directions": {
                    "values": ray_directions_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
                "pose_quats": {
                    "values": pose_quats_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
                "pose_trans": {
                    "values": pose_trans_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
            }
        )
        loss_terms, details = get_loss_terms_and_details(
            losses_dict,
            valid_masks,
            type(self).__name__,
            n_views,
            self.flatten_across_image_only,
        )
        losses = Sum(*loss_terms)

        return losses, (details | {})


class FactoredGeometryRegr3DPlusNormalGMLoss(FactoredGeometryRegr3D):
    """
    Regression, Normals & Gradient Matching Loss for Factored Geometry.
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
        depth_type_for_loss="depth_along_ray",
        cam_frame_points_loss_weight=1,
        depth_loss_weight=1,
        ray_directions_loss_weight=1,
        pose_quats_loss_weight=1,
        pose_trans_loss_weight=1,
        compute_absolute_pose_loss=True,
        compute_pairwise_relative_pose_loss=False,
        convert_predictions_to_view0_frame=False,
        compute_world_frame_points_loss=True,
        world_frame_points_loss_weight=1,
        apply_normal_and_gm_loss_to_synthetic_data_only=True,
        normal_loss_weight=1,
        gm_loss_weight=1,
    ):
        """
        Initialize the loss criterion for Factored Geometry (see parent class for details).
        Additionally computes:
        (1) Normal Loss over the Camera Frame Pointmaps in euclidean coordinates,
        (2) Gradient Matching (GM) Loss over the Depth Z in log space. (MiDAS applied GM loss in disparity space)

        Args:
            criterion (BaseCriterion): The base criterion to use for computing the loss.
            norm_mode (str): Normalization mode for scene representation. Default: "avg_dis".
                If prefixed with "?", normalization is only applied to non-metric scale data.
            gt_scale (bool): If True, enforce predictions to have the same scale as ground truth.
                If False, both GT and predictions are normalized independently. Default: False.
            ambiguous_loss_value (float): Value to use for ambiguous pixels in the loss.
                If 0, ambiguous pixels are ignored. Default: 0.
            max_metric_scale (float): Maximum scale for metric scale data. If data exceeds this
                value, it will be treated as non-metric. Default: False (no limit).
            loss_in_log (bool): If True, apply logarithmic transformation to input before
                computing the loss for depth and pointmaps. Default: True.
            flatten_across_image_only (bool): If True, flatten H x W dimensions only when computing
                the loss. If False, flatten across batch and spatial dimensions. Default: False.
            depth_type_for_loss (str): Type of depth to use for loss computation. Default: "depth_along_ray".
                Options: "depth_along_ray", "depth_z"
            cam_frame_points_loss_weight (float): Weight to use for the camera frame pointmap loss. Default: 1.
            depth_loss_weight (float): Weight to use for the depth loss. Default: 1.
            ray_directions_loss_weight (float): Weight to use for the ray directions loss. Default: 1.
            pose_quats_loss_weight (float): Weight to use for the pose quats loss. Default: 1.
            pose_trans_loss_weight (float): Weight to use for the pose trans loss. Default: 1.
            compute_absolute_pose_loss (bool): If True, compute the absolute pose loss. Default: True.
            compute_pairwise_relative_pose_loss (bool): If True, the pose loss is computed on the
                exhaustive pairwise relative poses. Default: False.
            convert_predictions_to_view0_frame (bool): If True, convert predictions to view0 frame.
                Use this if the predictions are not already in the view0 frame. Default: False.
            compute_world_frame_points_loss (bool): If True, compute the world frame pointmap loss. Default: True.
            world_frame_points_loss_weight (float): Weight to use for the world frame pointmap loss. Default: 1.
            apply_normal_and_gm_loss_to_synthetic_data_only (bool): If True, apply the normal and gm loss only to synthetic data.
                If False, apply the normal and gm loss to all data. Default: True.
            normal_loss_weight (float): Weight to use for the normal loss. Default: 1.
            gm_loss_weight (float): Weight to use for the gm loss. Default: 1.
        """
        super().__init__(
            criterion=criterion,
            norm_mode=norm_mode,
            gt_scale=gt_scale,
            ambiguous_loss_value=ambiguous_loss_value,
            max_metric_scale=max_metric_scale,
            loss_in_log=loss_in_log,
            flatten_across_image_only=flatten_across_image_only,
            depth_type_for_loss=depth_type_for_loss,
            cam_frame_points_loss_weight=cam_frame_points_loss_weight,
            depth_loss_weight=depth_loss_weight,
            ray_directions_loss_weight=ray_directions_loss_weight,
            pose_quats_loss_weight=pose_quats_loss_weight,
            pose_trans_loss_weight=pose_trans_loss_weight,
            compute_absolute_pose_loss=compute_absolute_pose_loss,
            compute_pairwise_relative_pose_loss=compute_pairwise_relative_pose_loss,
            convert_predictions_to_view0_frame=convert_predictions_to_view0_frame,
            compute_world_frame_points_loss=compute_world_frame_points_loss,
            world_frame_points_loss_weight=world_frame_points_loss_weight,
        )
        self.apply_normal_and_gm_loss_to_synthetic_data_only = (
            apply_normal_and_gm_loss_to_synthetic_data_only
        )
        self.normal_loss_weight = normal_loss_weight
        self.gm_loss_weight = gm_loss_weight

    def compute_loss(self, batch, preds, **kw):
        gt_info, pred_info, valid_masks, ambiguous_masks = self.get_all_info(
            batch, preds, **kw
        )
        n_views = len(batch)

        # Mask out samples in the batch where the gt depth validity mask is entirely zero
        valid_norm_factor_masks = [
            mask.sum(dim=(1, 2)) > 0 for mask in valid_masks
        ]  # List of (B,)

        if self.ambiguous_loss_value > 0:
            assert self.criterion.reduction == "none", (
                "ambiguous_loss_value should be 0 if no conf loss"
            )
            # Add the ambiguous pixel as "valid" pixels...
            valid_masks = [
                mask | ambig_mask
                for mask, ambig_mask in zip(valid_masks, ambiguous_masks)
            ]

        normal_losses = []
        gradient_matching_losses = []
        pose_trans_losses = []
        pose_quats_losses = []
        ray_directions_losses = []
        depth_losses = []
        cam_pts3d_losses = []
        if self.compute_world_frame_points_loss:
            pts3d_losses = []

        for i in range(n_views):
            # Get the camera frame points, log space depth_z & valid masks
            pred_local_pts3d = pred_info[i]["pts3d_cam"]
            pred_depth_z = pred_local_pts3d[..., 2:]
            pred_depth_z = apply_log_to_norm(pred_depth_z)
            gt_local_pts3d = gt_info[i]["pts3d_cam"]
            gt_depth_z = gt_local_pts3d[..., 2:]
            gt_depth_z = apply_log_to_norm(gt_depth_z)
            valid_mask_for_normal_gm_loss = valid_masks[i].clone()

            # Update the validity mask for normal & gm loss based on the synthetic data mask if required
            if self.apply_normal_and_gm_loss_to_synthetic_data_only:
                synthetic_mask = batch[i]["is_synthetic"]  # (B, )
                synthetic_mask = synthetic_mask.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                synthetic_mask = synthetic_mask.expand(
                    -1, pred_depth_z.shape[1], pred_depth_z.shape[2]
                )  # (B, H, W)
                valid_mask_for_normal_gm_loss = (
                    valid_mask_for_normal_gm_loss & synthetic_mask
                )

            # Compute the normal loss
            normal_loss = compute_normal_loss(
                pred_local_pts3d, gt_local_pts3d, valid_mask_for_normal_gm_loss.clone()
            )
            normal_loss = normal_loss * self.normal_loss_weight
            normal_losses.append(normal_loss)

            # Compute the gradient matching loss
            gradient_matching_loss = compute_gradient_matching_loss(
                pred_depth_z, gt_depth_z, valid_mask_for_normal_gm_loss.clone()
            )
            gradient_matching_loss = gradient_matching_loss * self.gm_loss_weight
            gradient_matching_losses.append(gradient_matching_loss)

            # Get the predicted dense quantities
            if not self.flatten_across_image_only:
                # Flatten the points across the entire batch with the masks
                pred_ray_directions = pred_info[i]["ray_directions"]
                gt_ray_directions = gt_info[i]["ray_directions"]
                pred_depth = pred_info[i][self.depth_type_for_loss][valid_masks[i]]
                gt_depth = gt_info[i][self.depth_type_for_loss][valid_masks[i]]
                pred_cam_pts3d = pred_info[i]["pts3d_cam"][valid_masks[i]]
                gt_cam_pts3d = gt_info[i]["pts3d_cam"][valid_masks[i]]
                if self.compute_world_frame_points_loss:
                    pred_pts3d = pred_info[i]["pts3d"][valid_masks[i]]
                    gt_pts3d = gt_info[i]["pts3d"][valid_masks[i]]
            else:
                # Flatten the H x W dimensions to H*W
                batch_size, _, _, direction_dim = gt_info[i]["ray_directions"].shape
                gt_ray_directions = gt_info[i]["ray_directions"].view(
                    batch_size, -1, direction_dim
                )
                pred_ray_directions = pred_info[i]["ray_directions"].view(
                    batch_size, -1, direction_dim
                )
                depth_dim = gt_info[i][self.depth_type_for_loss].shape[-1]
                gt_depth = gt_info[i][self.depth_type_for_loss].view(
                    batch_size, -1, depth_dim
                )
                pred_depth = pred_info[i][self.depth_type_for_loss].view(
                    batch_size, -1, depth_dim
                )
                cam_pts_dim = gt_info[i]["pts3d_cam"].shape[-1]
                gt_cam_pts3d = gt_info[i]["pts3d_cam"].view(batch_size, -1, cam_pts_dim)
                pred_cam_pts3d = pred_info[i]["pts3d_cam"].view(
                    batch_size, -1, cam_pts_dim
                )
                if self.compute_world_frame_points_loss:
                    pts_dim = gt_info[i]["pts3d"].shape[-1]
                    gt_pts3d = gt_info[i]["pts3d"].view(batch_size, -1, pts_dim)
                    pred_pts3d = pred_info[i]["pts3d"].view(batch_size, -1, pts_dim)
                valid_masks[i] = valid_masks[i].view(batch_size, -1)

            # Apply loss in log space for depth if specified
            if self.loss_in_log:
                gt_depth = apply_log_to_norm(gt_depth)
                pred_depth = apply_log_to_norm(pred_depth)
                gt_cam_pts3d = apply_log_to_norm(gt_cam_pts3d)
                pred_cam_pts3d = apply_log_to_norm(pred_cam_pts3d)
                if self.compute_world_frame_points_loss:
                    gt_pts3d = apply_log_to_norm(gt_pts3d)
                    pred_pts3d = apply_log_to_norm(pred_pts3d)

            # Compute pose loss
            if (
                self.compute_absolute_pose_loss
                and self.compute_pairwise_relative_pose_loss
            ):
                # Compute the absolute pose loss
                # Get the pose info for the current view
                pred_pose_trans = pred_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                gt_pose_trans = gt_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                pred_pose_quats = pred_info[i]["pose_quats"]
                gt_pose_quats = gt_info[i]["pose_quats"]

                # Compute pose translation loss
                abs_pose_trans_loss = self.criterion(
                    pred_pose_trans, gt_pose_trans, factor="pose_trans"
                )
                abs_pose_trans_loss = abs_pose_trans_loss * self.pose_trans_loss_weight

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                abs_pose_quats_loss = torch.minimum(
                    self.criterion(pred_pose_quats, gt_pose_quats, factor="pose_quats"),
                    self.criterion(
                        pred_pose_quats, -gt_pose_quats, factor="pose_quats"
                    ),
                )
                abs_pose_quats_loss = abs_pose_quats_loss * self.pose_quats_loss_weight

                # Compute the pairwise relative pose loss
                # Get the inverse of current view predicted pose
                pred_inv_curr_view_pose_quats = quaternion_inverse(
                    pred_info[i]["pose_quats"]
                )
                pred_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    pred_inv_curr_view_pose_quats
                )
                pred_inv_curr_view_pose_trans = -1 * ein.einsum(
                    pred_inv_curr_view_pose_rot_mat,
                    pred_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the inverse of the current view GT pose
                gt_inv_curr_view_pose_quats = quaternion_inverse(
                    gt_info[i]["pose_quats"]
                )
                gt_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    gt_inv_curr_view_pose_quats
                )
                gt_inv_curr_view_pose_trans = -1 * ein.einsum(
                    gt_inv_curr_view_pose_rot_mat,
                    gt_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the other N-1 relative poses using the current pose as reference frame
                pred_rel_pose_quats = []
                pred_rel_pose_trans = []
                gt_rel_pose_quats = []
                gt_rel_pose_trans = []
                for ov_idx in range(n_views):
                    if ov_idx == i:
                        continue
                    # Get the relative predicted pose
                    pred_ov_rel_pose_quats = quaternion_multiply(
                        pred_inv_curr_view_pose_quats, pred_info[ov_idx]["pose_quats"]
                    )
                    pred_ov_rel_pose_trans = (
                        ein.einsum(
                            pred_inv_curr_view_pose_rot_mat,
                            pred_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + pred_inv_curr_view_pose_trans
                    )

                    # Get the relative GT pose
                    gt_ov_rel_pose_quats = quaternion_multiply(
                        gt_inv_curr_view_pose_quats, gt_info[ov_idx]["pose_quats"]
                    )
                    gt_ov_rel_pose_trans = (
                        ein.einsum(
                            gt_inv_curr_view_pose_rot_mat,
                            gt_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + gt_inv_curr_view_pose_trans
                    )

                    # Get the valid translations using valid_norm_factor_masks for current view and other view
                    overall_valid_mask_for_trans = (
                        valid_norm_factor_masks[i] & valid_norm_factor_masks[ov_idx]
                    )

                    # Append the relative poses
                    pred_rel_pose_quats.append(pred_ov_rel_pose_quats)
                    pred_rel_pose_trans.append(
                        pred_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )
                    gt_rel_pose_quats.append(gt_ov_rel_pose_quats)
                    gt_rel_pose_trans.append(
                        gt_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )

                # Cat the N-1 relative poses along the batch dimension
                pred_rel_pose_quats = torch.cat(pred_rel_pose_quats, dim=0)
                pred_rel_pose_trans = torch.cat(pred_rel_pose_trans, dim=0)
                gt_rel_pose_quats = torch.cat(gt_rel_pose_quats, dim=0)
                gt_rel_pose_trans = torch.cat(gt_rel_pose_trans, dim=0)

                # Compute pose translation loss
                rel_pose_trans_loss = self.criterion(
                    pred_rel_pose_trans, gt_rel_pose_trans, factor="pose_trans"
                )
                rel_pose_trans_loss = rel_pose_trans_loss * self.pose_trans_loss_weight

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                rel_pose_quats_loss = torch.minimum(
                    self.criterion(
                        pred_rel_pose_quats, gt_rel_pose_quats, factor="pose_quats"
                    ),
                    self.criterion(
                        pred_rel_pose_quats, -gt_rel_pose_quats, factor="pose_quats"
                    ),
                )
                rel_pose_quats_loss = rel_pose_quats_loss * self.pose_quats_loss_weight

                # Concatenate the absolute and relative pose losses together
                pose_trans_loss = torch.cat(
                    [abs_pose_trans_loss, rel_pose_trans_loss], dim=0
                )
                pose_quats_loss = torch.cat(
                    [abs_pose_quats_loss, rel_pose_quats_loss], dim=0
                )
                pose_trans_losses.append(pose_trans_loss)
                pose_quats_losses.append(pose_quats_loss)

            elif self.compute_absolute_pose_loss:
                # Get the pose info for the current view
                pred_pose_trans = pred_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                gt_pose_trans = gt_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                pred_pose_quats = pred_info[i]["pose_quats"]
                gt_pose_quats = gt_info[i]["pose_quats"]

                # Compute pose translation loss
                pose_trans_loss = self.criterion(
                    pred_pose_trans, gt_pose_trans, factor="pose_trans"
                )
                pose_trans_loss = pose_trans_loss * self.pose_trans_loss_weight
                pose_trans_losses.append(pose_trans_loss)

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                pose_quats_loss = torch.minimum(
                    self.criterion(pred_pose_quats, gt_pose_quats, factor="pose_quats"),
                    self.criterion(
                        pred_pose_quats, -gt_pose_quats, factor="pose_quats"
                    ),
                )
                pose_quats_loss = pose_quats_loss * self.pose_quats_loss_weight
                pose_quats_losses.append(pose_quats_loss)

            elif self.compute_pairwise_relative_pose_loss:
                # Get the inverse of current view predicted pose
                pred_inv_curr_view_pose_quats = quaternion_inverse(
                    pred_info[i]["pose_quats"]
                )
                pred_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    pred_inv_curr_view_pose_quats
                )
                pred_inv_curr_view_pose_trans = -1 * ein.einsum(
                    pred_inv_curr_view_pose_rot_mat,
                    pred_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the inverse of the current view GT pose
                gt_inv_curr_view_pose_quats = quaternion_inverse(
                    gt_info[i]["pose_quats"]
                )
                gt_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    gt_inv_curr_view_pose_quats
                )
                gt_inv_curr_view_pose_trans = -1 * ein.einsum(
                    gt_inv_curr_view_pose_rot_mat,
                    gt_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the other N-1 relative poses using the current pose as reference frame
                pred_rel_pose_quats = []
                pred_rel_pose_trans = []
                gt_rel_pose_quats = []
                gt_rel_pose_trans = []
                for ov_idx in range(n_views):
                    if ov_idx == i:
                        continue
                    # Get the relative predicted pose
                    pred_ov_rel_pose_quats = quaternion_multiply(
                        pred_inv_curr_view_pose_quats, pred_info[ov_idx]["pose_quats"]
                    )
                    pred_ov_rel_pose_trans = (
                        ein.einsum(
                            pred_inv_curr_view_pose_rot_mat,
                            pred_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + pred_inv_curr_view_pose_trans
                    )

                    # Get the relative GT pose
                    gt_ov_rel_pose_quats = quaternion_multiply(
                        gt_inv_curr_view_pose_quats, gt_info[ov_idx]["pose_quats"]
                    )
                    gt_ov_rel_pose_trans = (
                        ein.einsum(
                            gt_inv_curr_view_pose_rot_mat,
                            gt_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + gt_inv_curr_view_pose_trans
                    )

                    # Get the valid translations using valid_norm_factor_masks for current view and other view
                    overall_valid_mask_for_trans = (
                        valid_norm_factor_masks[i] & valid_norm_factor_masks[ov_idx]
                    )

                    # Append the relative poses
                    pred_rel_pose_quats.append(pred_ov_rel_pose_quats)
                    pred_rel_pose_trans.append(
                        pred_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )
                    gt_rel_pose_quats.append(gt_ov_rel_pose_quats)
                    gt_rel_pose_trans.append(
                        gt_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )

                # Cat the N-1 relative poses along the batch dimension
                pred_rel_pose_quats = torch.cat(pred_rel_pose_quats, dim=0)
                pred_rel_pose_trans = torch.cat(pred_rel_pose_trans, dim=0)
                gt_rel_pose_quats = torch.cat(gt_rel_pose_quats, dim=0)
                gt_rel_pose_trans = torch.cat(gt_rel_pose_trans, dim=0)

                # Compute pose translation loss
                pose_trans_loss = self.criterion(
                    pred_rel_pose_trans, gt_rel_pose_trans, factor="pose_trans"
                )
                pose_trans_loss = pose_trans_loss * self.pose_trans_loss_weight
                pose_trans_losses.append(pose_trans_loss)

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                pose_quats_loss = torch.minimum(
                    self.criterion(
                        pred_rel_pose_quats, gt_rel_pose_quats, factor="pose_quats"
                    ),
                    self.criterion(
                        pred_rel_pose_quats, -gt_rel_pose_quats, factor="pose_quats"
                    ),
                )
                pose_quats_loss = pose_quats_loss * self.pose_quats_loss_weight
                pose_quats_losses.append(pose_quats_loss)
            else:
                # Error
                raise ValueError(
                    "compute_absolute_pose_loss and compute_pairwise_relative_pose_loss cannot both be False"
                )

            # Compute ray direction loss
            ray_directions_loss = self.criterion(
                pred_ray_directions, gt_ray_directions, factor="ray_directions"
            )
            ray_directions_loss = ray_directions_loss * self.ray_directions_loss_weight
            ray_directions_losses.append(ray_directions_loss)

            # Compute depth loss
            depth_loss = self.criterion(pred_depth, gt_depth, factor="depth")
            depth_loss = depth_loss * self.depth_loss_weight
            depth_losses.append(depth_loss)

            # Compute camera frame point loss
            cam_pts3d_loss = self.criterion(
                pred_cam_pts3d, gt_cam_pts3d, factor="points"
            )
            cam_pts3d_loss = cam_pts3d_loss * self.cam_frame_points_loss_weight
            cam_pts3d_losses.append(cam_pts3d_loss)

            if self.compute_world_frame_points_loss:
                # Compute point loss
                pts3d_loss = self.criterion(pred_pts3d, gt_pts3d, factor="points")
                pts3d_loss = pts3d_loss * self.world_frame_points_loss_weight
                pts3d_losses.append(pts3d_loss)

            # Handle ambiguous pixels
            if self.ambiguous_loss_value > 0:
                if not self.flatten_across_image_only:
                    depth_losses[i] = torch.where(
                        ambiguous_masks[i][valid_masks[i]],
                        self.ambiguous_loss_value,
                        depth_losses[i],
                    )
                    cam_pts3d_losses[i] = torch.where(
                        ambiguous_masks[i][valid_masks[i]],
                        self.ambiguous_loss_value,
                        cam_pts3d_losses[i],
                    )
                    if self.compute_world_frame_points_loss:
                        pts3d_losses[i] = torch.where(
                            ambiguous_masks[i][valid_masks[i]],
                            self.ambiguous_loss_value,
                            pts3d_losses[i],
                        )
                else:
                    depth_losses[i] = torch.where(
                        ambiguous_masks[i].view(ambiguous_masks[i].shape[0], -1),
                        self.ambiguous_loss_value,
                        depth_losses[i],
                    )
                    cam_pts3d_losses[i] = torch.where(
                        ambiguous_masks[i].view(ambiguous_masks[i].shape[0], -1),
                        self.ambiguous_loss_value,
                        cam_pts3d_losses[i],
                    )
                    if self.compute_world_frame_points_loss:
                        pts3d_losses[i] = torch.where(
                            ambiguous_masks[i].view(ambiguous_masks[i].shape[0], -1),
                            self.ambiguous_loss_value,
                            pts3d_losses[i],
                        )

        # Use helper function to generate loss terms and details
        if self.compute_world_frame_points_loss:
            losses_dict = {
                "pts3d": {
                    "values": pts3d_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
            }
        else:
            losses_dict = {}
        losses_dict.update(
            {
                "cam_pts3d": {
                    "values": cam_pts3d_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                self.depth_type_for_loss: {
                    "values": depth_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                "ray_directions": {
                    "values": ray_directions_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
                "pose_quats": {
                    "values": pose_quats_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
                "pose_trans": {
                    "values": pose_trans_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
                "normal": {
                    "values": normal_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
                "gradient_matching": {
                    "values": gradient_matching_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
            }
        )
        loss_terms, details = get_loss_terms_and_details(
            losses_dict,
            valid_masks,
            type(self).__name__,
            n_views,
            self.flatten_across_image_only,
        )
        losses = Sum(*loss_terms)

        return losses, (details | {})


class FactoredGeometryScaleRegr3D(Criterion, MultiLoss):
    """
    Regression Loss for Factored Geometry & Scale.
    """

    def __init__(
        self,
        criterion,
        norm_predictions=True,
        norm_mode="avg_dis",
        ambiguous_loss_value=0,
        loss_in_log=True,
        flatten_across_image_only=False,
        depth_type_for_loss="depth_along_ray",
        cam_frame_points_loss_weight=1,
        depth_loss_weight=1,
        ray_directions_loss_weight=1,
        pose_quats_loss_weight=1,
        pose_trans_loss_weight=1,
        scale_loss_weight=1,
        scale_ratio_loss_weight=0.0,
        compute_absolute_pose_loss=True,
        compute_pairwise_relative_pose_loss=False,
        convert_predictions_to_view0_frame=False,
        compute_world_frame_points_loss=True,
        world_frame_points_loss_weight=1,
    ):
        """
        Initialize the loss criterion for Factored Geometry (Ray Directions, Depth, Pose), Scale
        and the Collective Geometry i.e. Local Frame Pointmaps & optionally World Frame Pointmaps.
        If world-frame pointmap loss is computed, the pixel-level losses are computed in the following order:
        (1) world points, (2) cam points, (3) depth, (4) ray directions, (5) pose quats, (6) pose trans, (7) scale.
        Else, the pixel-level losses are returned in the following order:
        (1) cam points, (2) depth, (3) ray directions, (4) pose quats, (5) pose trans, (6) scale.
        The predicited scene representation is always normalized w.r.t. the frame of view0.
        Loss is applied between the predicted metric scale and the ground truth metric scale.

        Args:
            criterion (BaseCriterion): The base criterion to use for computing the loss.
            norm_predictions (bool): If True, normalize the predictions before computing the loss.
            norm_mode (str): Normalization mode for the gt and predicted (optional) scene representation. Default: "avg_dis".
            ambiguous_loss_value (float): Value to use for ambiguous pixels in the loss.
                If 0, ambiguous pixels are ignored. Default: 0.
            loss_in_log (bool): If True, apply logarithmic transformation to input before
                computing the loss for depth, pointmaps and scale. Default: True.
            flatten_across_image_only (bool): If True, flatten H x W dimensions only when computing
                the loss. If False, flatten across batch and spatial dimensions. Default: False.
            depth_type_for_loss (str): Type of depth to use for loss computation. Default: "depth_along_ray".
                Options: "depth_along_ray", "depth_z"
            cam_frame_points_loss_weight (float): Weight to use for the camera frame pointmap loss. Default: 1.
            depth_loss_weight (float): Weight to use for the depth loss. Default: 1.
            ray_directions_loss_weight (float): Weight to use for the ray directions loss. Default: 1.
            pose_quats_loss_weight (float): Weight to use for the pose quats loss. Default: 1.
            pose_trans_loss_weight (float): Weight to use for the pose trans loss. Default: 1.
            scale_loss_weight (float): Weight to use for the scale loss. Default: 1.
            compute_absolute_pose_loss (bool): If True, compute the absolute pose loss. Default: True.
            compute_pairwise_relative_pose_loss (bool): If True, the pose loss is computed on the
                exhaustive pairwise relative poses. Default: False.
            convert_predictions_to_view0_frame (bool): If True, convert predictions to view0 frame.
                Use this if the predictions are not already in the view0 frame. Default: False.
            compute_world_frame_points_loss (bool): If True, compute the world frame pointmap loss. Default: True.
            world_frame_points_loss_weight (float): Weight to use for the world frame pointmap loss. Default: 1.
        """
        super().__init__(criterion)
        self.norm_predictions = norm_predictions
        self.norm_mode = norm_mode
        self.ambiguous_loss_value = ambiguous_loss_value
        self.loss_in_log = loss_in_log
        self.flatten_across_image_only = flatten_across_image_only
        self.depth_type_for_loss = depth_type_for_loss
        assert self.depth_type_for_loss in [
            "depth_along_ray",
            "depth_z",
        ], "depth_type_for_loss must be one of ['depth_along_ray', 'depth_z']"
        self.cam_frame_points_loss_weight = cam_frame_points_loss_weight
        self.depth_loss_weight = depth_loss_weight
        self.ray_directions_loss_weight = ray_directions_loss_weight
        self.pose_quats_loss_weight = pose_quats_loss_weight
        self.pose_trans_loss_weight = pose_trans_loss_weight
        self.scale_loss_weight = scale_loss_weight
        self.scale_ratio_loss_weight = scale_ratio_loss_weight
        self.compute_absolute_pose_loss = compute_absolute_pose_loss
        self.compute_pairwise_relative_pose_loss = compute_pairwise_relative_pose_loss
        self.convert_predictions_to_view0_frame = convert_predictions_to_view0_frame
        self.compute_world_frame_points_loss = compute_world_frame_points_loss
        self.world_frame_points_loss_weight = world_frame_points_loss_weight
    

    def get_all_info(self, batch, preds, dist_clip=None):
        """
        Function to get all the information needed to compute the loss.
        Returns all quantities normalized w.r.t. camera of view0.
        """
        n_views = len(batch)

        # Everything is normalized w.r.t. camera of view0
        # Intialize lists to store data for all views
        # Ground truth quantities
        in_camera0 = closed_form_pose_inverse(batch[0]["camera_pose"])
        no_norm_gt_pts = []
        no_norm_gt_pts_cam = []
        no_norm_gt_depth = []
        no_norm_gt_pose_trans = []
        valid_masks = []
        gt_ray_directions = []
        gt_pose_quats = []
        # Predicted quantities
        if self.convert_predictions_to_view0_frame:
            # Get the camera transform to convert quantities to view0 frame
            pred_camera0 = torch.eye(4, device=preds[0]["cam_quats"].device).unsqueeze(
                0
            )
            batch_size = preds[0]["cam_quats"].shape[0]
            pred_camera0 = pred_camera0.repeat(batch_size, 1, 1)
            pred_camera0_rot = quaternion_to_rotation_matrix(
                preds[0]["cam_quats"].clone()
            )
            pred_camera0[..., :3, :3] = pred_camera0_rot
            pred_camera0[..., :3, 3] = preds[0]["cam_trans"].clone()
            pred_in_camera0 = closed_form_pose_inverse(pred_camera0)
        no_norm_pr_pts = []
        no_norm_pr_pts_cam = []
        no_norm_pr_depth = []
        no_norm_pr_pose_trans = []
        pr_ray_directions = []
        pr_pose_quats = []
        metric_pr_pts_to_compute_scale = []

        no_norm_pr_camera_pose_trans = []
        pr_camera_pose_quats = []
        pr_camera_fov_hw = []

        # Get ground truth & prediction info for all views
        for i in range(n_views):
            # Get the ground truth
            no_norm_gt_pts.append(geotrf(in_camera0, batch[i]["pts3d"]))
            valid_masks.append(batch[i]["valid_mask"].clone())
            no_norm_gt_pts_cam.append(batch[i]["pts3d_cam"])
            gt_ray_directions.append(batch[i]["ray_directions_cam"])
            if self.depth_type_for_loss == "depth_along_ray":
                no_norm_gt_depth.append(batch[i]["depth_along_ray"])
            elif self.depth_type_for_loss == "depth_z":
                no_norm_gt_depth.append(batch[i]["pts3d_cam"][..., 2:])
            if i == 0:
                # For view0, initialize identity pose
                gt_pose_quats.append(
                    torch.tensor(
                        [0, 0, 0, 1],
                        dtype=gt_ray_directions[0].dtype,
                        device=gt_ray_directions[0].device,
                    )
                    .unsqueeze(0)
                    .repeat(gt_ray_directions[0].shape[0], 1)
                )
                no_norm_gt_pose_trans.append(
                    torch.tensor(
                        [0, 0, 0],
                        dtype=gt_ray_directions[0].dtype,
                        device=gt_ray_directions[0].device,
                    )
                    .unsqueeze(0)
                    .repeat(gt_ray_directions[0].shape[0], 1)
                )
            else:
                # For other views, transform pose to view0's frame
                gt_pose_quats_world = batch[i]["camera_pose_quats"]
                no_norm_gt_pose_trans_world = batch[i]["camera_pose_trans"]
                gt_pose_quats_in_view0, no_norm_gt_pose_trans_in_view0 = (
                    transform_pose_using_quats_and_trans_2_to_1(
                        batch[0]["camera_pose_quats"],
                        batch[0]["camera_pose_trans"],
                        gt_pose_quats_world,
                        no_norm_gt_pose_trans_world,
                    )
                )
                gt_pose_quats.append(gt_pose_quats_in_view0)
                no_norm_gt_pose_trans.append(no_norm_gt_pose_trans_in_view0)

            # Get the global predictions in view0's frame
            if self.convert_predictions_to_view0_frame:
                # Convert predictions to view0 frame
                pr_pts3d_in_view0 = geotrf(pred_in_camera0, preds[i]["pts3d"])
                pr_pose_quats_in_view0, pr_pose_trans_in_view0 = (
                    transform_pose_using_quats_and_trans_2_to_1(
                        preds[0]["cam_quats"],
                        preds[0]["cam_trans"],
                        preds[i]["cam_quats"],
                        preds[i]["cam_trans"],
                    )
                )
            else:
                # Predictions are already in view0 frame
                pr_pts3d_in_view0 = preds[i]["pts3d"]
                pr_pose_trans_in_view0 = preds[i]["cam_trans"]
                pr_pose_quats_in_view0 = preds[i]["cam_quats"]
            
            
            # Get the Camera Head prediction in view0's frame
            if "camera_fov_hw" in preds[i]:
                pr_camera_fov_hw.append(preds[i]["camera_fov_hw"])
            if "camera_translation" in preds[i] and "camera_quaternion" in preds[i]:
                if self.convert_predictions_to_view0_frame:
                    curr_pr_camera_quaternion_in_view0, curr_pr_camera_translation_in_view0 = (
                        transform_pose_using_quats_and_trans_2_to_1(
                            preds[0]["camera_quaternion"],
                            preds[0]["camera_translation"],
                            preds[i]["camera_quaternion"],
                            preds[i]["camera_translation"],
                        )
                    )
                else:
                    curr_pr_camera_quaternion_in_view0 = preds[i]["camera_quaternion"]
                    curr_pr_camera_translation_in_view0 = preds[i]["camera_translation"]

                if "metric_scaling_factor" in preds[i].keys():
                    curr_no_norm_pr_camera_pose_trans = (
                        curr_pr_camera_translation_in_view0 / preds[i]["metric_scaling_factor"]
                    )
                else:
                    curr_no_norm_pr_camera_pose_trans = curr_pr_camera_translation_in_view0
                no_norm_pr_camera_pose_trans.append(curr_no_norm_pr_camera_pose_trans)
                pr_camera_pose_quats.append(curr_pr_camera_quaternion_in_view0)

            # Get predictions for normalized loss
            if self.depth_type_for_loss == "depth_along_ray":
                curr_view_no_norm_depth = preds[i]["depth_along_ray"]
            elif self.depth_type_for_loss == "depth_z":
                curr_view_no_norm_depth = preds[i]["pts3d_cam"][..., 2:]
            if "metric_scaling_factor" in preds[i].keys():
                # Divide by the predicted metric scaling factor to get the raw predicted points, depth_along_ray, and pose_trans
                # This detaches the predicted metric scaling factor from the geometry based loss
                curr_view_no_norm_pr_pts = pr_pts3d_in_view0 / preds[i][
                    "metric_scaling_factor"
                ].unsqueeze(-1).unsqueeze(-1)
                curr_view_no_norm_pr_pts_cam = preds[i]["pts3d_cam"] / preds[i][
                    "metric_scaling_factor"
                ].unsqueeze(-1).unsqueeze(-1)
                curr_view_no_norm_depth = curr_view_no_norm_depth / preds[i][
                    "metric_scaling_factor"
                ].unsqueeze(-1).unsqueeze(-1)
                curr_view_no_norm_pr_pose_trans = (
                    pr_pose_trans_in_view0 / preds[i]["metric_scaling_factor"]
                )
            else:
                curr_view_no_norm_pr_pts = pr_pts3d_in_view0
                curr_view_no_norm_pr_pts_cam = preds[i]["pts3d_cam"]
                curr_view_no_norm_depth = curr_view_no_norm_depth
                curr_view_no_norm_pr_pose_trans = pr_pose_trans_in_view0

            no_norm_pr_pts.append(curr_view_no_norm_pr_pts)
            no_norm_pr_pts_cam.append(curr_view_no_norm_pr_pts_cam)
            no_norm_pr_depth.append(curr_view_no_norm_depth)
            no_norm_pr_pose_trans.append(curr_view_no_norm_pr_pose_trans)
            pr_ray_directions.append(preds[i]["ray_directions"])
            pr_pose_quats.append(pr_pose_quats_in_view0)

            # Get the predicted metric scale points
            if "metric_scaling_factor" in preds[i].keys():
                # Detach the raw predicted points so that the scale loss is only applied to the scaling factor
                curr_view_metric_pr_pts_to_compute_scale = (
                    curr_view_no_norm_pr_pts.detach()
                    * preds[i]["metric_scaling_factor"].unsqueeze(-1).unsqueeze(-1)
                )
            else:
                curr_view_metric_pr_pts_to_compute_scale = (
                    curr_view_no_norm_pr_pts.clone()
                )
            metric_pr_pts_to_compute_scale.append(
                curr_view_metric_pr_pts_to_compute_scale
            )

        if dist_clip is not None:
            # Points that are too far-away == invalid
            for i in range(n_views):
                dis = no_norm_gt_pts[i].norm(dim=-1)
                valid_masks[i] = valid_masks[i] & (dis <= dist_clip)

        # compute normalization factors for pose translations using the un-normalized pose translations
        gt_pose_trans_stacked = torch.stack(no_norm_gt_pose_trans, dim=1)
        pr_pose_trans_stacked = torch.stack(no_norm_pr_pose_trans, dim=1)
        _, gt_pose_norm_factor = normalize_pose_translations(gt_pose_trans_stacked, return_norm_factor=True)
        _, pr_pose_norm_factor = normalize_pose_translations(pr_pose_trans_stacked, return_norm_factor=True)

        # Initialize normalized tensors
        gt_pts = [torch.zeros_like(pts) for pts in no_norm_gt_pts]
        gt_pts_cam = [torch.zeros_like(pts_cam) for pts_cam in no_norm_gt_pts_cam]
        gt_depth = [torch.zeros_like(depth) for depth in no_norm_gt_depth]
        gt_pose_trans = [torch.zeros_like(trans) for trans in no_norm_gt_pose_trans]

        pr_pts = [torch.zeros_like(pts) for pts in no_norm_pr_pts]
        pr_pts_cam = [torch.zeros_like(pts_cam) for pts_cam in no_norm_pr_pts_cam]
        pr_depth = [torch.zeros_like(depth) for depth in no_norm_pr_depth]
        pr_pose_trans = [torch.zeros_like(trans) for trans in no_norm_pr_pose_trans]

        if len(no_norm_pr_camera_pose_trans) > 0:
            pr_camera_pose_trans = [torch.zeros_like(trans) for trans in no_norm_pr_camera_pose_trans]
        else: 
            pr_camera_pose_trans = []

        # Normalize the predicted points if specified
        # if self.norm_predictions:
        #     pr_normalization_output = normalize_multiple_pointclouds(
        #         no_norm_pr_pts,
        #         valid_masks,
        #         self.norm_mode,
        #         ret_factor=True,
        #     )
        #     pr_pts_norm = pr_normalization_output[:-1]
        #     pr_norm_factor = pr_normalization_output[-1]

        pr_normalization_output = normalize_multiple_pointclouds(
            no_norm_pr_pts,
            valid_masks,
            self.norm_mode,
            ret_factor=True,
        )
        pr_pts_norm = pr_normalization_output[:-1]
        pr_norm_factor = pr_normalization_output[-1]

        # Normalize the ground truth points
        gt_normalization_output = normalize_multiple_pointclouds(
            no_norm_gt_pts, valid_masks, self.norm_mode, ret_factor=True
        )
        gt_pts_norm = gt_normalization_output[:-1]
        gt_norm_factor = gt_normalization_output[-1]

        for i in range(n_views):
            if self.norm_predictions:
                # Assign the normalized predictions
                pr_pts[i] = pr_pts_norm[i]
                pr_pts_cam[i] = no_norm_pr_pts_cam[i] / pr_norm_factor
                pr_depth[i] = no_norm_pr_depth[i] / pr_norm_factor
                pr_pose_trans[i] = no_norm_pr_pose_trans[i] / pr_norm_factor[:, :, 0, 0]

                if len(pr_camera_pose_trans) > 0:
                    pr_camera_pose_trans[i] = no_norm_pr_camera_pose_trans[i] / pr_norm_factor[:, :, 0, 0]
            else:
                pr_pts[i] = no_norm_pr_pts[i]
                pr_pts_cam[i] = no_norm_pr_pts_cam[i]
                pr_depth[i] = no_norm_pr_depth[i]
                pr_pose_trans[i] = no_norm_pr_pose_trans[i]

                if len(pr_camera_pose_trans) > 0:
                    pr_camera_pose_trans[i] = no_norm_pr_camera_pose_trans[i]

            # Assign the normalized ground truth quantities
            gt_pts[i] = gt_pts_norm[i]
            gt_pts_cam[i] = no_norm_gt_pts_cam[i] / gt_norm_factor
            gt_depth[i] = no_norm_gt_depth[i] / gt_norm_factor
            gt_pose_trans[i] = no_norm_gt_pose_trans[i] / gt_norm_factor[:, :, 0, 0]

        # Get the mask indicating ground truth metric scale quantities
        metric_scale_mask = batch[0]["is_metric_scale"]
        valid_gt_norm_factor_mask = (
            gt_norm_factor[:, 0, 0, 0] > 1e-8
        )  # Mask out cases where depth for all views is invalid
        valid_metric_scale_mask = metric_scale_mask & valid_gt_norm_factor_mask

        if valid_metric_scale_mask.any():
            # Compute the scale norm factor using the predicted metric scale points
            metric_pr_normalization_output = normalize_multiple_pointclouds(
                metric_pr_pts_to_compute_scale,
                valid_masks,
                self.norm_mode,
                ret_factor=True,
            )
            pr_metric_norm_factor = metric_pr_normalization_output[-1]

            # Get the valid ground truth and predicted scale norm factors for the metric ground truth quantities
            gt_metric_norm_factor = gt_norm_factor[valid_metric_scale_mask]
            pr_metric_norm_factor = pr_metric_norm_factor[valid_metric_scale_mask]
        else:
            gt_metric_norm_factor = None
            pr_metric_norm_factor = None

        # Get ambiguous masks
        ambiguous_masks = []
        for i in range(n_views):
            ambiguous_masks.append(
                (~batch[i]["non_ambiguous_mask"]) & (~valid_masks[i])
            )

        # Pack into info dicts
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

            if len(pr_camera_pose_trans) > 0:
                pred_info[-1]["camera_pose_trans"] = pr_camera_pose_trans[i]
            if len(pr_camera_pose_trans) > 0:
                pred_info[-1]["camera_pose_quats"] = pr_camera_pose_quats[i]
            if len(pr_camera_pose_trans) > 0:
                pred_info[-1]["camera_fov_hw"] = pr_camera_fov_hw[i]

        return (
            gt_info,
            pred_info,
            valid_masks,
            ambiguous_masks,
            gt_metric_norm_factor,
            pr_metric_norm_factor,
            gt_norm_factor,
            pr_norm_factor,
            gt_pose_norm_factor,
            pr_pose_norm_factor,
        )

    def compute_loss(self, batch, preds, **kw):
        (
            gt_info,
            pred_info,
            valid_masks,
            ambiguous_masks,
            gt_metric_norm_factor,
            pr_metric_norm_factor,
            gt_points_norm_factor,
            pr_points_norm_factor,
            gt_pose_norm_factor,
            pr_pose_norm_factor,
        ) = self.get_all_info(batch, preds, **kw)
        n_views = len(batch)

        # Mask out samples in the batch where the gt depth validity mask is entirely zero
        valid_norm_factor_masks = [
            mask.sum(dim=(1, 2)) > 0 for mask in valid_masks
        ]  # List of (B,)

        if self.ambiguous_loss_value > 0:
            assert self.criterion.reduction == "none", (
                "ambiguous_loss_value should be 0 if no conf loss"
            )
            # Add the ambiguous pixel as "valid" pixels...
            valid_masks = [
                mask | ambig_mask
                for mask, ambig_mask in zip(valid_masks, ambiguous_masks)
            ]

        pose_trans_losses = []
        pose_quats_losses = []
        ray_directions_losses = []
        depth_losses = []
        cam_pts3d_losses = []
        if self.compute_world_frame_points_loss:
            pts3d_losses = []

        for i in range(n_views):
            # Get the predicted dense quantities
            if not self.flatten_across_image_only:
                # Flatten the points across the entire batch with the masks
                pred_ray_directions = pred_info[i]["ray_directions"]
                gt_ray_directions = gt_info[i]["ray_directions"]
                pred_depth = pred_info[i][self.depth_type_for_loss][valid_masks[i]]
                gt_depth = gt_info[i][self.depth_type_for_loss][valid_masks[i]]
                pred_cam_pts3d = pred_info[i]["pts3d_cam"][valid_masks[i]]
                gt_cam_pts3d = gt_info[i]["pts3d_cam"][valid_masks[i]]
                if self.compute_world_frame_points_loss:
                    pred_pts3d = pred_info[i]["pts3d"][valid_masks[i]]
                    gt_pts3d = gt_info[i]["pts3d"][valid_masks[i]]
            else:
                # Flatten the H x W dimensions to H*W
                batch_size, _, _, direction_dim = gt_info[i]["ray_directions"].shape
                gt_ray_directions = gt_info[i]["ray_directions"].view(
                    batch_size, -1, direction_dim
                )
                pred_ray_directions = pred_info[i]["ray_directions"].view(
                    batch_size, -1, direction_dim
                )
                depth_dim = gt_info[i][self.depth_type_for_loss].shape[-1]
                gt_depth = gt_info[i][self.depth_type_for_loss].view(
                    batch_size, -1, depth_dim
                )
                pred_depth = pred_info[i][self.depth_type_for_loss].view(
                    batch_size, -1, depth_dim
                )
                cam_pts_dim = gt_info[i]["pts3d_cam"].shape[-1]
                gt_cam_pts3d = gt_info[i]["pts3d_cam"].view(batch_size, -1, cam_pts_dim)
                pred_cam_pts3d = pred_info[i]["pts3d_cam"].view(
                    batch_size, -1, cam_pts_dim
                )
                if self.compute_world_frame_points_loss:
                    pts_dim = gt_info[i]["pts3d"].shape[-1]
                    gt_pts3d = gt_info[i]["pts3d"].view(batch_size, -1, pts_dim)
                    pred_pts3d = pred_info[i]["pts3d"].view(batch_size, -1, pts_dim)
                valid_masks[i] = valid_masks[i].view(batch_size, -1)

            # Apply loss in log space for depth if specified
            if self.loss_in_log:
                gt_depth = apply_log_to_norm(gt_depth)
                pred_depth = apply_log_to_norm(pred_depth)
                gt_cam_pts3d = apply_log_to_norm(gt_cam_pts3d)
                pred_cam_pts3d = apply_log_to_norm(pred_cam_pts3d)
                if self.compute_world_frame_points_loss:
                    gt_pts3d = apply_log_to_norm(gt_pts3d)
                    pred_pts3d = apply_log_to_norm(pred_pts3d)

            # Compute pose loss
            if (
                self.compute_absolute_pose_loss
                and self.compute_pairwise_relative_pose_loss
            ):
                # Compute the absolute pose loss
                # Get the pose info for the current view
                pred_pose_trans = pred_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                gt_pose_trans = gt_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                pred_pose_quats = pred_info[i]["pose_quats"]
                gt_pose_quats = gt_info[i]["pose_quats"]

                # Compute pose translation loss
                abs_pose_trans_loss = self.criterion(
                    pred_pose_trans, gt_pose_trans, factor="pose_trans"
                )
                abs_pose_trans_loss = abs_pose_trans_loss * self.pose_trans_loss_weight

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                abs_pose_quats_loss = torch.minimum(
                    self.criterion(pred_pose_quats, gt_pose_quats, factor="pose_quats"),
                    self.criterion(
                        pred_pose_quats, -gt_pose_quats, factor="pose_quats"
                    ),
                )
                abs_pose_quats_loss = abs_pose_quats_loss * self.pose_quats_loss_weight

                # Compute the pairwise relative pose loss
                # Get the inverse of current view predicted pose
                pred_inv_curr_view_pose_quats = quaternion_inverse(
                    pred_info[i]["pose_quats"]
                )
                pred_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    pred_inv_curr_view_pose_quats
                )
                pred_inv_curr_view_pose_trans = -1 * ein.einsum(
                    pred_inv_curr_view_pose_rot_mat,
                    pred_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the inverse of the current view GT pose
                gt_inv_curr_view_pose_quats = quaternion_inverse(
                    gt_info[i]["pose_quats"]
                )
                gt_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    gt_inv_curr_view_pose_quats
                )
                gt_inv_curr_view_pose_trans = -1 * ein.einsum(
                    gt_inv_curr_view_pose_rot_mat,
                    gt_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the other N-1 relative poses using the current pose as reference frame
                pred_rel_pose_quats = []
                pred_rel_pose_trans = []
                gt_rel_pose_quats = []
                gt_rel_pose_trans = []
                for ov_idx in range(n_views):
                    if ov_idx == i:
                        continue
                    # Get the relative predicted pose
                    pred_ov_rel_pose_quats = quaternion_multiply(
                        pred_inv_curr_view_pose_quats, pred_info[ov_idx]["pose_quats"]
                    )
                    pred_ov_rel_pose_trans = (
                        ein.einsum(
                            pred_inv_curr_view_pose_rot_mat,
                            pred_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + pred_inv_curr_view_pose_trans
                    )

                    # Get the relative GT pose
                    gt_ov_rel_pose_quats = quaternion_multiply(
                        gt_inv_curr_view_pose_quats, gt_info[ov_idx]["pose_quats"]
                    )
                    gt_ov_rel_pose_trans = (
                        ein.einsum(
                            gt_inv_curr_view_pose_rot_mat,
                            gt_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + gt_inv_curr_view_pose_trans
                    )

                    # Get the valid translations using valid_norm_factor_masks for current view and other view
                    overall_valid_mask_for_trans = (
                        valid_norm_factor_masks[i] & valid_norm_factor_masks[ov_idx]
                    )

                    # Append the relative poses
                    pred_rel_pose_quats.append(pred_ov_rel_pose_quats)
                    pred_rel_pose_trans.append(
                        pred_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )
                    gt_rel_pose_quats.append(gt_ov_rel_pose_quats)
                    gt_rel_pose_trans.append(
                        gt_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )

                # Cat the N-1 relative poses along the batch dimension
                pred_rel_pose_quats = torch.cat(pred_rel_pose_quats, dim=0)
                pred_rel_pose_trans = torch.cat(pred_rel_pose_trans, dim=0)
                gt_rel_pose_quats = torch.cat(gt_rel_pose_quats, dim=0)
                gt_rel_pose_trans = torch.cat(gt_rel_pose_trans, dim=0)

                # Compute pose translation loss
                rel_pose_trans_loss = self.criterion(
                    pred_rel_pose_trans, gt_rel_pose_trans, factor="pose_trans"
                )
                rel_pose_trans_loss = rel_pose_trans_loss * self.pose_trans_loss_weight

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                rel_pose_quats_loss = torch.minimum(
                    self.criterion(
                        pred_rel_pose_quats, gt_rel_pose_quats, factor="pose_quats"
                    ),
                    self.criterion(
                        pred_rel_pose_quats, -gt_rel_pose_quats, factor="pose_quats"
                    ),
                )
                rel_pose_quats_loss = rel_pose_quats_loss * self.pose_quats_loss_weight

                # Concatenate the absolute and relative pose losses together
                pose_trans_loss = torch.cat(
                    [abs_pose_trans_loss, rel_pose_trans_loss], dim=0
                )
                pose_quats_loss = torch.cat(
                    [abs_pose_quats_loss, rel_pose_quats_loss], dim=0
                )
                pose_trans_losses.append(pose_trans_loss)
                pose_quats_losses.append(pose_quats_loss)
            elif self.compute_absolute_pose_loss:
                # Get the pose info for the current view
                pred_pose_trans = pred_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                gt_pose_trans = gt_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                pred_pose_quats = pred_info[i]["pose_quats"]
                gt_pose_quats = gt_info[i]["pose_quats"]

                # Compute pose translation loss
                pose_trans_loss = self.criterion(
                    pred_pose_trans, gt_pose_trans, factor="pose_trans"
                )
                pose_trans_loss = pose_trans_loss * self.pose_trans_loss_weight
                pose_trans_losses.append(pose_trans_loss)

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                pose_quats_loss = torch.minimum(
                    self.criterion(pred_pose_quats, gt_pose_quats, factor="pose_quats"),
                    self.criterion(
                        pred_pose_quats, -gt_pose_quats, factor="pose_quats"
                    ),
                )
                pose_quats_loss = pose_quats_loss * self.pose_quats_loss_weight
                pose_quats_losses.append(pose_quats_loss)
            elif self.compute_pairwise_relative_pose_loss:
                # Get the inverse of current view predicted pose
                pred_inv_curr_view_pose_quats = quaternion_inverse(
                    pred_info[i]["pose_quats"]
                )
                pred_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    pred_inv_curr_view_pose_quats
                )
                pred_inv_curr_view_pose_trans = -1 * ein.einsum(
                    pred_inv_curr_view_pose_rot_mat,
                    pred_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the inverse of the current view GT pose
                gt_inv_curr_view_pose_quats = quaternion_inverse(
                    gt_info[i]["pose_quats"]
                )
                gt_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    gt_inv_curr_view_pose_quats
                )
                gt_inv_curr_view_pose_trans = -1 * ein.einsum(
                    gt_inv_curr_view_pose_rot_mat,
                    gt_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the other N-1 relative poses using the current pose as reference frame
                pred_rel_pose_quats = []
                pred_rel_pose_trans = []
                gt_rel_pose_quats = []
                gt_rel_pose_trans = []
                for ov_idx in range(n_views):
                    if ov_idx == i:
                        continue
                    # Get the relative predicted pose
                    pred_ov_rel_pose_quats = quaternion_multiply(
                        pred_inv_curr_view_pose_quats, pred_info[ov_idx]["pose_quats"]
                    )
                    pred_ov_rel_pose_trans = (
                        ein.einsum(
                            pred_inv_curr_view_pose_rot_mat,
                            pred_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + pred_inv_curr_view_pose_trans
                    )

                    # Get the relative GT pose
                    gt_ov_rel_pose_quats = quaternion_multiply(
                        gt_inv_curr_view_pose_quats, gt_info[ov_idx]["pose_quats"]
                    )
                    gt_ov_rel_pose_trans = (
                        ein.einsum(
                            gt_inv_curr_view_pose_rot_mat,
                            gt_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + gt_inv_curr_view_pose_trans
                    )

                    # Get the valid translations using valid_norm_factor_masks for current view and other view
                    overall_valid_mask_for_trans = (
                        valid_norm_factor_masks[i] & valid_norm_factor_masks[ov_idx]
                    )

                    # Append the relative poses
                    pred_rel_pose_quats.append(pred_ov_rel_pose_quats)
                    pred_rel_pose_trans.append(
                        pred_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )
                    gt_rel_pose_quats.append(gt_ov_rel_pose_quats)
                    gt_rel_pose_trans.append(
                        gt_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )

                # Cat the N-1 relative poses along the batch dimension
                pred_rel_pose_quats = torch.cat(pred_rel_pose_quats, dim=0)
                pred_rel_pose_trans = torch.cat(pred_rel_pose_trans, dim=0)
                gt_rel_pose_quats = torch.cat(gt_rel_pose_quats, dim=0)
                gt_rel_pose_trans = torch.cat(gt_rel_pose_trans, dim=0)

                # Compute pose translation loss
                pose_trans_loss = self.criterion(
                    pred_rel_pose_trans, gt_rel_pose_trans, factor="pose_trans"
                )
                pose_trans_loss = pose_trans_loss * self.pose_trans_loss_weight
                pose_trans_losses.append(pose_trans_loss)

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                pose_quats_loss = torch.minimum(
                    self.criterion(
                        pred_rel_pose_quats, gt_rel_pose_quats, factor="pose_quats"
                    ),
                    self.criterion(
                        pred_rel_pose_quats, -gt_rel_pose_quats, factor="pose_quats"
                    ),
                )
                pose_quats_loss = pose_quats_loss * self.pose_quats_loss_weight
                pose_quats_losses.append(pose_quats_loss)
            else:
                # Error
                raise ValueError(
                    "compute_absolute_pose_loss and compute_pairwise_relative_pose_loss cannot both be False"
                )

            # Compute ray direction loss
            ray_directions_loss = self.criterion(
                pred_ray_directions, gt_ray_directions, factor="ray_directions"
            )
            ray_directions_loss = ray_directions_loss * self.ray_directions_loss_weight
            ray_directions_losses.append(ray_directions_loss)

            # Compute depth loss
            depth_loss = self.criterion(pred_depth, gt_depth, factor="depth")
            depth_loss = depth_loss * self.depth_loss_weight
            depth_losses.append(depth_loss)

            # Compute camera frame point loss
            cam_pts3d_loss = self.criterion(
                pred_cam_pts3d, gt_cam_pts3d, factor="points"
            )
            cam_pts3d_loss = cam_pts3d_loss * self.cam_frame_points_loss_weight
            cam_pts3d_losses.append(cam_pts3d_loss)

            if self.compute_world_frame_points_loss:
                # Compute point loss
                pts3d_loss = self.criterion(pred_pts3d, gt_pts3d, factor="points")
                pts3d_loss = pts3d_loss * self.world_frame_points_loss_weight
                pts3d_losses.append(pts3d_loss)

            # Handle ambiguous pixels
            if self.ambiguous_loss_value > 0:
                if not self.flatten_across_image_only:
                    depth_losses[i] = torch.where(
                        ambiguous_masks[i][valid_masks[i]],
                        self.ambiguous_loss_value,
                        depth_losses[i],
                    )
                    cam_pts3d_losses[i] = torch.where(
                        ambiguous_masks[i][valid_masks[i]],
                        self.ambiguous_loss_value,
                        cam_pts3d_losses[i],
                    )
                    if self.compute_world_frame_points_loss:
                        pts3d_losses[i] = torch.where(
                            ambiguous_masks[i][valid_masks[i]],
                            self.ambiguous_loss_value,
                            pts3d_losses[i],
                        )
                else:
                    depth_losses[i] = torch.where(
                        ambiguous_masks[i].view(ambiguous_masks[i].shape[0], -1),
                        self.ambiguous_loss_value,
                        depth_losses[i],
                    )
                    cam_pts3d_losses[i] = torch.where(
                        ambiguous_masks[i].view(ambiguous_masks[i].shape[0], -1),
                        self.ambiguous_loss_value,
                        cam_pts3d_losses[i],
                    )
                    if self.compute_world_frame_points_loss:
                        pts3d_losses[i] = torch.where(
                            ambiguous_masks[i].view(ambiguous_masks[i].shape[0], -1),
                            self.ambiguous_loss_value,
                            pts3d_losses[i],
                        )

        # Compute the scale loss
        if gt_metric_norm_factor is not None:
            if self.loss_in_log:
                gt_metric_norm_factor = apply_log_to_norm(gt_metric_norm_factor)
                pr_metric_norm_factor = apply_log_to_norm(pr_metric_norm_factor)
            scale_loss = (
                self.criterion(
                    pr_metric_norm_factor, gt_metric_norm_factor, factor="scale"
                )
                * self.scale_loss_weight
            )
        else:
            scale_loss = None
        
        # Compute Scale Consistency Loss if specified
        if self.scale_ratio_loss_weight > 0:
            gt_points_nf = gt_points_norm_factor[:, 0, 0, 0]
            pr_points_nf = pr_points_norm_factor[:, 0, 0, 0]
            gt_pose_nf = gt_pose_norm_factor          # [B]
            pr_pose_nf = pr_pose_norm_factor          # [B]

            pr_points_nf = pr_points_nf.detach()

            valid = (
                (gt_points_nf > 1e-8)
                & (pr_points_nf > 1e-8)
                & (gt_pose_nf > 1e-8)
                & (pr_pose_nf > 1e-8)
            )

            if valid.any():
                pred_ratio = pr_pose_nf[valid] / pr_points_nf[valid]
                gt_ratio = gt_pose_nf[valid] / gt_points_nf[valid]

                pred_ratio = pred_ratio.clamp_min(1e-8)
                gt_ratio = gt_ratio.clamp_min(1e-8)

                if self.loss_in_log:
                    pred_ratio = torch.log(pred_ratio)
                    gt_ratio = torch.log(gt_ratio)

                scale_ratio_loss = (
                    self.criterion(pred_ratio.unsqueeze(-1), gt_ratio.unsqueeze(-1), factor="scale",)
                    * self.scale_ratio_loss_weight
                )
            else:
                scale_ratio_loss = None
        else:
            scale_ratio_loss = None

        # Use helper function to generate loss terms and details
        if self.compute_world_frame_points_loss:
            losses_dict = {
                "pts3d": {
                    "values": pts3d_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
            }
        else:
            losses_dict = {}
        losses_dict.update(
            {
                "cam_pts3d": {
                    "values": cam_pts3d_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                self.depth_type_for_loss: {
                    "values": depth_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                "ray_directions": {
                    "values": ray_directions_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
                "pose_quats": {
                    "values": pose_quats_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
                "pose_trans": {
                    "values": pose_trans_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
                "scale": {
                    "values": scale_loss,
                    "use_mask": False,
                    "is_multi_view": False,
                },
                "scale_ratio": {
                    "values": scale_ratio_loss,
                    "use_mask": False,
                    "is_multi_view": False,
                },
            }
        )
        loss_terms, details = get_loss_terms_and_details(
            losses_dict,
            valid_masks,
            type(self).__name__,
            n_views,
            self.flatten_across_image_only,
        )
        losses = Sum(*loss_terms)

        return losses, (details | {})


class FactoredGeometryScaleRegr3DPlusNormalGMLoss(FactoredGeometryScaleRegr3D):
    """
    Regression, Normals & Gradient Matching Loss for Factored Geometry & Scale.
    """

    def __init__(
        self,
        criterion,
        norm_predictions=True,
        norm_mode="avg_dis",
        ambiguous_loss_value=0,
        loss_in_log=True,
        flatten_across_image_only=False,
        depth_type_for_loss="depth_along_ray",
        cam_frame_points_loss_weight=1,
        depth_loss_weight=1,
        ray_directions_loss_weight=1,
        pose_quats_loss_weight=1,
        pose_trans_loss_weight=1,
        scale_loss_weight=1,
        scale_ratio_loss_weight=0.0,
        compute_absolute_pose_loss=True,
        compute_pairwise_relative_pose_loss=False,
        convert_predictions_to_view0_frame=False,
        compute_world_frame_points_loss=True,
        world_frame_points_loss_weight=1,
        apply_normal_and_gm_loss_to_synthetic_data_only=True,
        normal_loss_weight=1,
        gm_loss_weight=1,
        compute_camera_head_loss=False,
        camera_translation_loss_weight=1.0,
        camera_quaternion_loss_weight=1.0,
        camera_fov_hw_loss_weight=1.0,
    ):
        """
        Initialize the loss criterion for Ray Directions, Depth, Pose, Pointmaps & Scale.
        Additionally computes:
        (1) Normal Loss over the Camera Frame Pointmaps in euclidean coordinates,
        (2) Gradient Matching (GM) Loss over the Depth Z in log space. (MiDAS applied GM loss in disparity space)

        Args:
            criterion (BaseCriterion): The base criterion to use for computing the loss.
            norm_predictions (bool): If True, normalize the predictions before computing the loss.
            norm_mode (str): Normalization mode for the gt and predicted (optional) scene representation. Default: "avg_dis".
            ambiguous_loss_value (float): Value to use for ambiguous pixels in the loss.
                If 0, ambiguous pixels are ignored. Default: 0.
            loss_in_log (bool): If True, apply logarithmic transformation to input before
                computing the loss for depth, pointmaps and scale. Default: True.
            flatten_across_image_only (bool): If True, flatten H x W dimensions only when computing
                the loss. If False, flatten across batch and spatial dimensions. Default: False.
            depth_type_for_loss (str): Type of depth to use for loss computation. Default: "depth_along_ray".
                Options: "depth_along_ray", "depth_z"
            cam_frame_points_loss_weight (float): Weight to use for the camera frame pointmap loss. Default: 1.
            depth_loss_weight (float): Weight to use for the depth loss. Default: 1.
            ray_directions_loss_weight (float): Weight to use for the ray directions loss. Default: 1.
            pose_quats_loss_weight (float): Weight to use for the pose quats loss. Default: 1.
            pose_trans_loss_weight (float): Weight to use for the pose trans loss. Default: 1.
            scale_loss_weight (float): Weight to use for the scale loss. Default: 1.
            compute_pairwise_relative_pose_loss (bool): If True, the pose loss is computed on the
                exhaustive pairwise relative poses. Default: False.
            convert_predictions_to_view0_frame (bool): If True, convert predictions to view0 frame.
                Use this if the predictions are not already in the view0 frame. Default: False.
            compute_absolute_pose_loss (bool): If True, compute the absolute pose loss. Default: True.
            compute_world_frame_points_loss (bool): If True, compute the world frame pointmap loss. Default: True.
            world_frame_points_loss_weight (float): Weight to use for the world frame pointmap loss. Default: 1.
            apply_normal_and_gm_loss_to_synthetic_data_only (bool): If True, apply the normal and gm loss only to synthetic data.
                If False, apply the normal and gm loss to all data. Default: True.
            normal_loss_weight (float): Weight to use for the normal loss. Default: 1.
            gm_loss_weight (float): Weight to use for the gm loss. Default: 1.
        """
        super().__init__(
            criterion=criterion,
            norm_predictions=norm_predictions,
            norm_mode=norm_mode,
            ambiguous_loss_value=ambiguous_loss_value,
            loss_in_log=loss_in_log,
            flatten_across_image_only=flatten_across_image_only,
            depth_type_for_loss=depth_type_for_loss,
            cam_frame_points_loss_weight=cam_frame_points_loss_weight,
            depth_loss_weight=depth_loss_weight,
            ray_directions_loss_weight=ray_directions_loss_weight,
            pose_quats_loss_weight=pose_quats_loss_weight,
            pose_trans_loss_weight=pose_trans_loss_weight,
            scale_loss_weight=scale_loss_weight,
            scale_ratio_loss_weight = scale_ratio_loss_weight,
            compute_absolute_pose_loss=compute_absolute_pose_loss,
            compute_pairwise_relative_pose_loss=compute_pairwise_relative_pose_loss,
            convert_predictions_to_view0_frame=convert_predictions_to_view0_frame,
            compute_world_frame_points_loss=compute_world_frame_points_loss,
            world_frame_points_loss_weight=world_frame_points_loss_weight,
        )
        self.apply_normal_and_gm_loss_to_synthetic_data_only = (
            apply_normal_and_gm_loss_to_synthetic_data_only
        )
        self.normal_loss_weight = normal_loss_weight
        self.gm_loss_weight = gm_loss_weight
        
        self.compute_camera_head_loss = compute_camera_head_loss
        self.camera_translation_loss_weight = camera_translation_loss_weight
        self.camera_quaternion_loss_weight = camera_quaternion_loss_weight
        self.camera_fov_hw_loss_weight = camera_fov_hw_loss_weight

    def compute_loss(self, batch, preds, **kw):
        (
            gt_info,
            pred_info,
            valid_masks,
            ambiguous_masks,
            gt_metric_norm_factor,
            pr_metric_norm_factor,
            gt_points_norm_factor,
            pr_points_norm_factor,
            gt_pose_norm_factor,
            pr_pose_norm_factor,
        ) = self.get_all_info(batch, preds, **kw)
        n_views = len(batch)

        # Mask out samples in the batch where the gt depth validity mask is entirely zero
        valid_norm_factor_masks = [
            mask.sum(dim=(1, 2)) > 0 for mask in valid_masks
        ]  # List of (B,)

        if self.ambiguous_loss_value > 0:
            assert self.criterion.reduction == "none", (
                "ambiguous_loss_value should be 0 if no conf loss"
            )
            # Add the ambiguous pixel as "valid" pixels...
            valid_masks = [
                mask | ambig_mask
                for mask, ambig_mask in zip(valid_masks, ambiguous_masks)
            ]

        normal_losses = []
        gradient_matching_losses = []
        pose_trans_losses = []
        pose_quats_losses = []
        ray_directions_losses = []
        depth_losses = []
        cam_pts3d_losses = []
        if self.compute_world_frame_points_loss:
            pts3d_losses = []
        
        camera_translation_losses = []
        camera_quaternion_losses = []
        camera_fov_hw_losses = []

        for i in range(n_views):
            # Get the camera frame points, log space depth_z & valid masks
            pred_local_pts3d = pred_info[i]["pts3d_cam"]
            pred_depth_z = pred_local_pts3d[..., 2:]
            pred_depth_z = apply_log_to_norm(pred_depth_z)
            gt_local_pts3d = gt_info[i]["pts3d_cam"]
            gt_depth_z = gt_local_pts3d[..., 2:]
            gt_depth_z = apply_log_to_norm(gt_depth_z)
            valid_mask_for_normal_gm_loss = valid_masks[i].clone()

            # Update the validity mask for normal & gm loss based on the synthetic data mask if required
            if self.apply_normal_and_gm_loss_to_synthetic_data_only:
                synthetic_mask = batch[i]["is_synthetic"]  # (B, )
                synthetic_mask = synthetic_mask.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                synthetic_mask = synthetic_mask.expand(
                    -1, pred_depth_z.shape[1], pred_depth_z.shape[2]
                )  # (B, H, W)
                valid_mask_for_normal_gm_loss = (
                    valid_mask_for_normal_gm_loss & synthetic_mask
                )

            # Compute the normal loss
            normal_loss = compute_normal_loss(
                pred_local_pts3d, gt_local_pts3d, valid_mask_for_normal_gm_loss.clone()
            )
            normal_loss = normal_loss * self.normal_loss_weight
            normal_losses.append(normal_loss)

            # Compute the gradient matching loss
            gradient_matching_loss = compute_gradient_matching_loss(
                pred_depth_z, gt_depth_z, valid_mask_for_normal_gm_loss.clone()
            )
            gradient_matching_loss = gradient_matching_loss * self.gm_loss_weight
            gradient_matching_losses.append(gradient_matching_loss)

            # Get the predicted dense quantities
            if not self.flatten_across_image_only:
                # Flatten the points across the entire batch with the masks and compute the metrics
                pred_ray_directions = pred_info[i]["ray_directions"]
                gt_ray_directions = gt_info[i]["ray_directions"]
                pred_depth = pred_info[i][self.depth_type_for_loss][valid_masks[i]]
                gt_depth = gt_info[i][self.depth_type_for_loss][valid_masks[i]]
                pred_cam_pts3d = pred_info[i]["pts3d_cam"][valid_masks[i]]
                gt_cam_pts3d = gt_info[i]["pts3d_cam"][valid_masks[i]]
                if self.compute_world_frame_points_loss:
                    pred_pts3d = pred_info[i]["pts3d"][valid_masks[i]]
                    gt_pts3d = gt_info[i]["pts3d"][valid_masks[i]]
            else:
                # Flatten the H x W dimensions to H*W and compute the metrics
                batch_size, _, _, direction_dim = gt_info[i]["ray_directions"].shape
                gt_ray_directions = gt_info[i]["ray_directions"].view(
                    batch_size, -1, direction_dim
                )
                pred_ray_directions = pred_info[i]["ray_directions"].view(
                    batch_size, -1, direction_dim
                )
                depth_dim = gt_info[i][self.depth_type_for_loss].shape[-1]
                gt_depth = gt_info[i][self.depth_type_for_loss].view(
                    batch_size, -1, depth_dim
                )
                pred_depth = pred_info[i][self.depth_type_for_loss].view(
                    batch_size, -1, depth_dim
                )
                cam_pts_dim = gt_info[i]["pts3d_cam"].shape[-1]
                gt_cam_pts3d = gt_info[i]["pts3d_cam"].view(batch_size, -1, cam_pts_dim)
                pred_cam_pts3d = pred_info[i]["pts3d_cam"].view(
                    batch_size, -1, cam_pts_dim
                )
                if self.compute_world_frame_points_loss:
                    pts_dim = gt_info[i]["pts3d"].shape[-1]
                    gt_pts3d = gt_info[i]["pts3d"].view(batch_size, -1, pts_dim)
                    pred_pts3d = pred_info[i]["pts3d"].view(batch_size, -1, pts_dim)
                valid_masks[i] = valid_masks[i].view(batch_size, -1)

            # Apply loss in log space for depth if specified
            if self.loss_in_log:
                gt_depth = apply_log_to_norm(gt_depth)
                pred_depth = apply_log_to_norm(pred_depth)
                gt_cam_pts3d = apply_log_to_norm(gt_cam_pts3d)
                pred_cam_pts3d = apply_log_to_norm(pred_cam_pts3d)
                if self.compute_world_frame_points_loss:
                    gt_pts3d = apply_log_to_norm(gt_pts3d)
                    pred_pts3d = apply_log_to_norm(pred_pts3d)

            # Compute pose loss
            if (
                self.compute_absolute_pose_loss
                and self.compute_pairwise_relative_pose_loss
            ):
                # Compute the absolute pose loss
                # Get the pose info for the current view
                pred_pose_trans = pred_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                gt_pose_trans = gt_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                pred_pose_quats = pred_info[i]["pose_quats"]
                gt_pose_quats = gt_info[i]["pose_quats"]

                # Compute pose translation loss
                abs_pose_trans_loss = self.criterion(
                    pred_pose_trans, gt_pose_trans, factor="pose_trans"
                )
                abs_pose_trans_loss = abs_pose_trans_loss * self.pose_trans_loss_weight

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                abs_pose_quats_loss = torch.minimum(
                    self.criterion(pred_pose_quats, gt_pose_quats, factor="pose_quats"),
                    self.criterion(
                        pred_pose_quats, -gt_pose_quats, factor="pose_quats"
                    ),
                )
                abs_pose_quats_loss = abs_pose_quats_loss * self.pose_quats_loss_weight

                # Compute the pairwise relative pose loss
                # Get the inverse of current view predicted pose
                pred_inv_curr_view_pose_quats = quaternion_inverse(
                    pred_info[i]["pose_quats"]
                )
                pred_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    pred_inv_curr_view_pose_quats
                )
                pred_inv_curr_view_pose_trans = -1 * ein.einsum(
                    pred_inv_curr_view_pose_rot_mat,
                    pred_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the inverse of the current view GT pose
                gt_inv_curr_view_pose_quats = quaternion_inverse(
                    gt_info[i]["pose_quats"]
                )
                gt_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    gt_inv_curr_view_pose_quats
                )
                gt_inv_curr_view_pose_trans = -1 * ein.einsum(
                    gt_inv_curr_view_pose_rot_mat,
                    gt_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the other N-1 relative poses using the current pose as reference frame
                pred_rel_pose_quats = []
                pred_rel_pose_trans = []
                gt_rel_pose_quats = []
                gt_rel_pose_trans = []
                for ov_idx in range(n_views):
                    if ov_idx == i:
                        continue
                    # Get the relative predicted pose
                    pred_ov_rel_pose_quats = quaternion_multiply(
                        pred_inv_curr_view_pose_quats, pred_info[ov_idx]["pose_quats"]
                    )
                    pred_ov_rel_pose_trans = (
                        ein.einsum(
                            pred_inv_curr_view_pose_rot_mat,
                            pred_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + pred_inv_curr_view_pose_trans
                    )

                    # Get the relative GT pose
                    gt_ov_rel_pose_quats = quaternion_multiply(
                        gt_inv_curr_view_pose_quats, gt_info[ov_idx]["pose_quats"]
                    )
                    gt_ov_rel_pose_trans = (
                        ein.einsum(
                            gt_inv_curr_view_pose_rot_mat,
                            gt_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + gt_inv_curr_view_pose_trans
                    )

                    # Get the valid translations using valid_norm_factor_masks for current view and other view
                    overall_valid_mask_for_trans = (
                        valid_norm_factor_masks[i] & valid_norm_factor_masks[ov_idx]
                    )

                    # Append the relative poses
                    pred_rel_pose_quats.append(pred_ov_rel_pose_quats)
                    pred_rel_pose_trans.append(
                        pred_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )
                    gt_rel_pose_quats.append(gt_ov_rel_pose_quats)
                    gt_rel_pose_trans.append(
                        gt_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )

                # Cat the N-1 relative poses along the batch dimension
                pred_rel_pose_quats = torch.cat(pred_rel_pose_quats, dim=0)
                pred_rel_pose_trans = torch.cat(pred_rel_pose_trans, dim=0)
                gt_rel_pose_quats = torch.cat(gt_rel_pose_quats, dim=0)
                gt_rel_pose_trans = torch.cat(gt_rel_pose_trans, dim=0)

                # Compute pose translation loss
                rel_pose_trans_loss = self.criterion(
                    pred_rel_pose_trans, gt_rel_pose_trans, factor="pose_trans"
                )
                rel_pose_trans_loss = rel_pose_trans_loss * self.pose_trans_loss_weight

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                rel_pose_quats_loss = torch.minimum(
                    self.criterion(
                        pred_rel_pose_quats, gt_rel_pose_quats, factor="pose_quats"
                    ),
                    self.criterion(
                        pred_rel_pose_quats, -gt_rel_pose_quats, factor="pose_quats"
                    ),
                )
                rel_pose_quats_loss = rel_pose_quats_loss * self.pose_quats_loss_weight

                # Concatenate the absolute and relative pose losses together
                pose_trans_loss = torch.cat(
                    [abs_pose_trans_loss, rel_pose_trans_loss], dim=0
                )
                pose_quats_loss = torch.cat(
                    [abs_pose_quats_loss, rel_pose_quats_loss], dim=0
                )
                pose_trans_losses.append(pose_trans_loss)
                pose_quats_losses.append(pose_quats_loss)

            elif self.compute_absolute_pose_loss:
                # Get the pose info for the current view
                pred_pose_trans = pred_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                gt_pose_trans = gt_info[i]["pose_trans"][valid_norm_factor_masks[i]]
                pred_pose_quats = pred_info[i]["pose_quats"]
                gt_pose_quats = gt_info[i]["pose_quats"]

                # Compute pose translation loss
                pose_trans_loss = self.criterion(
                    pred_pose_trans, gt_pose_trans, factor="pose_trans"
                )
                pose_trans_loss = pose_trans_loss * self.pose_trans_loss_weight
                pose_trans_losses.append(pose_trans_loss)

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                pose_quats_loss = torch.minimum(
                    self.criterion(pred_pose_quats, gt_pose_quats, factor="pose_quats"),
                    self.criterion(
                        pred_pose_quats, -gt_pose_quats, factor="pose_quats"
                    ),
                )
                pose_quats_loss = pose_quats_loss * self.pose_quats_loss_weight
                pose_quats_losses.append(pose_quats_loss)

            elif self.compute_pairwise_relative_pose_loss:
                # Get the inverse of current view predicted pose
                pred_inv_curr_view_pose_quats = quaternion_inverse(
                    pred_info[i]["pose_quats"]
                )
                pred_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    pred_inv_curr_view_pose_quats
                )
                pred_inv_curr_view_pose_trans = -1 * ein.einsum(
                    pred_inv_curr_view_pose_rot_mat,
                    pred_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the inverse of the current view GT pose
                gt_inv_curr_view_pose_quats = quaternion_inverse(
                    gt_info[i]["pose_quats"]
                )
                gt_inv_curr_view_pose_rot_mat = quaternion_to_rotation_matrix(
                    gt_inv_curr_view_pose_quats
                )
                gt_inv_curr_view_pose_trans = -1 * ein.einsum(
                    gt_inv_curr_view_pose_rot_mat,
                    gt_info[i]["pose_trans"],
                    "b i j, b j -> b i",
                )

                # Get the other N-1 relative poses using the current pose as reference frame
                pred_rel_pose_quats = []
                pred_rel_pose_trans = []
                gt_rel_pose_quats = []
                gt_rel_pose_trans = []
                for ov_idx in range(n_views):
                    if ov_idx == i:
                        continue
                    # Get the relative predicted pose
                    pred_ov_rel_pose_quats = quaternion_multiply(
                        pred_inv_curr_view_pose_quats, pred_info[ov_idx]["pose_quats"]
                    )
                    pred_ov_rel_pose_trans = (
                        ein.einsum(
                            pred_inv_curr_view_pose_rot_mat,
                            pred_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + pred_inv_curr_view_pose_trans
                    )

                    # Get the relative GT pose
                    gt_ov_rel_pose_quats = quaternion_multiply(
                        gt_inv_curr_view_pose_quats, gt_info[ov_idx]["pose_quats"]
                    )
                    gt_ov_rel_pose_trans = (
                        ein.einsum(
                            gt_inv_curr_view_pose_rot_mat,
                            gt_info[ov_idx]["pose_trans"],
                            "b i j, b j -> b i",
                        )
                        + gt_inv_curr_view_pose_trans
                    )

                    # Get the valid translations using valid_norm_factor_masks for current view and other view
                    overall_valid_mask_for_trans = (
                        valid_norm_factor_masks[i] & valid_norm_factor_masks[ov_idx]
                    )

                    # Append the relative poses
                    pred_rel_pose_quats.append(pred_ov_rel_pose_quats)
                    pred_rel_pose_trans.append(
                        pred_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )
                    gt_rel_pose_quats.append(gt_ov_rel_pose_quats)
                    gt_rel_pose_trans.append(
                        gt_ov_rel_pose_trans[overall_valid_mask_for_trans]
                    )

                # Cat the N-1 relative poses along the batch dimension
                pred_rel_pose_quats = torch.cat(pred_rel_pose_quats, dim=0)
                pred_rel_pose_trans = torch.cat(pred_rel_pose_trans, dim=0)
                gt_rel_pose_quats = torch.cat(gt_rel_pose_quats, dim=0)
                gt_rel_pose_trans = torch.cat(gt_rel_pose_trans, dim=0)

                # Compute pose translation loss
                pose_trans_loss = self.criterion(
                    pred_rel_pose_trans, gt_rel_pose_trans, factor="pose_trans"
                )
                pose_trans_loss = pose_trans_loss * self.pose_trans_loss_weight
                pose_trans_losses.append(pose_trans_loss)

                # Compute pose rotation loss
                # Handle quaternion two-to-one mapping
                pose_quats_loss = torch.minimum(
                    self.criterion(
                        pred_rel_pose_quats, gt_rel_pose_quats, factor="pose_quats"
                    ),
                    self.criterion(
                        pred_rel_pose_quats, -gt_rel_pose_quats, factor="pose_quats"
                    ),
                )
                pose_quats_loss = pose_quats_loss * self.pose_quats_loss_weight
                pose_quats_losses.append(pose_quats_loss)
            else:
                # Error
                raise ValueError(
                    "compute_absolute_pose_loss and compute_pairwise_relative_pose_loss cannot both be False"
                )

            # Compute ray direction loss
            ray_directions_loss = self.criterion(
                pred_ray_directions, gt_ray_directions, factor="ray_directions"
            )
            ray_directions_loss = ray_directions_loss * self.ray_directions_loss_weight
            ray_directions_losses.append(ray_directions_loss)

            # Compute depth loss
            depth_loss = self.criterion(pred_depth, gt_depth, factor="depth")
            depth_loss = depth_loss * self.depth_loss_weight
            depth_losses.append(depth_loss)

            # Compute camera frame point loss
            cam_pts3d_loss = self.criterion(
                pred_cam_pts3d, gt_cam_pts3d, factor="points"
            )
            cam_pts3d_loss = cam_pts3d_loss * self.cam_frame_points_loss_weight
            cam_pts3d_losses.append(cam_pts3d_loss)

            if self.compute_world_frame_points_loss:
                # Compute point loss
                pts3d_loss = self.criterion(pred_pts3d, gt_pts3d, factor="points")
                pts3d_loss = pts3d_loss * self.world_frame_points_loss_weight
                pts3d_losses.append(pts3d_loss)
            
            # Compute optional camera-head losses
            if self.compute_camera_head_loss:
                pred_camera_translation = pred_info[i].get("camera_pose_trans", None)
                pred_camera_quaternion = pred_info[i].get("camera_pose_quats", None)
                pred_camera_fov_hw = pred_info[i].get("camera_fov_hw", None)

                # translation: reuse pose_trans factor
                if pred_camera_translation is not None:
                    if i == 0:
                        gt_camera_translation = torch.zeros_like(pred_camera_translation)
                    else:
                        gt_camera_translation = gt_info[i]["pose_trans"]

                    camera_translation_loss = self.criterion(
                        pred_camera_translation,
                        gt_camera_translation,
                        factor="pose_trans",
                    )
                    camera_translation_loss = (
                        camera_translation_loss * self.camera_translation_loss_weight
                    )
                else:
                    camera_translation_loss = None
                camera_translation_losses.append(camera_translation_loss)

                # quaternion: reuse pose_quats factor + sign-invariant min
                if pred_camera_quaternion is not None:
                    if i == 0:
                        gt_camera_quaternion = torch.zeros_like(pred_camera_quaternion)
                        gt_camera_quaternion[..., -1] = 1.0
                    else:
                        gt_camera_quaternion = gt_info[i]["pose_quats"]

                    camera_quaternion_loss = torch.minimum(
                        self.criterion(
                            pred_camera_quaternion,
                            gt_camera_quaternion,
                            factor="pose_quats",
                        ),
                        self.criterion(
                            pred_camera_quaternion,
                            -gt_camera_quaternion,
                            factor="pose_quats",
                        ),
                    )
                    camera_quaternion_loss = (
                        camera_quaternion_loss * self.camera_quaternion_loss_weight
                    )
                else:
                    camera_quaternion_loss = None
                camera_quaternion_losses.append(camera_quaternion_loss)

                # fov_hw: use self.criterion with dedicated factor
                if pred_camera_fov_hw is not None:
                    image_height = batch[i]["pts3d_cam"].shape[1]
                    image_width = batch[i]["pts3d_cam"].shape[2]

                    gt_camera_fov_hw = _camera_intrinsics_to_fov_hw(
                        batch[i]["camera_intrinsics"],
                        image_height=image_height,
                        image_width=image_width,
                    )

                    camera_fov_hw_loss = self.criterion(
                        pred_camera_fov_hw,
                        gt_camera_fov_hw,
                        factor="fov_hw",
                    )
                    camera_fov_hw_loss = (
                        camera_fov_hw_loss * self.camera_fov_hw_loss_weight
                    )
                else:
                    camera_fov_hw_loss = None
                camera_fov_hw_losses.append(camera_fov_hw_loss)

            # Handle ambiguous pixels
            if self.ambiguous_loss_value > 0:
                if not self.flatten_across_image_only:
                    depth_losses[i] = torch.where(
                        ambiguous_masks[i][valid_masks[i]],
                        self.ambiguous_loss_value,
                        depth_losses[i],
                    )
                    cam_pts3d_losses[i] = torch.where(
                        ambiguous_masks[i][valid_masks[i]],
                        self.ambiguous_loss_value,
                        cam_pts3d_losses[i],
                    )
                    if self.compute_world_frame_points_loss:
                        pts3d_losses[i] = torch.where(
                            ambiguous_masks[i][valid_masks[i]],
                            self.ambiguous_loss_value,
                            pts3d_losses[i],
                        )
                else:
                    depth_losses[i] = torch.where(
                        ambiguous_masks[i].view(ambiguous_masks[i].shape[0], -1),
                        self.ambiguous_loss_value,
                        depth_losses[i],
                    )
                    cam_pts3d_losses[i] = torch.where(
                        ambiguous_masks[i].view(ambiguous_masks[i].shape[0], -1),
                        self.ambiguous_loss_value,
                        cam_pts3d_losses[i],
                    )
                    if self.compute_world_frame_points_loss:
                        pts3d_losses[i] = torch.where(
                            ambiguous_masks[i].view(ambiguous_masks[i].shape[0], -1),
                            self.ambiguous_loss_value,
                            pts3d_losses[i],
                        )

        # Compute the scale loss
        if gt_metric_norm_factor is not None:
            if self.loss_in_log:
                gt_metric_norm_factor = apply_log_to_norm(gt_metric_norm_factor)
                pr_metric_norm_factor = apply_log_to_norm(pr_metric_norm_factor)
            scale_loss = (
                self.criterion(
                    pr_metric_norm_factor, gt_metric_norm_factor, factor="scale"
                )
                * self.scale_loss_weight
            )
        else:
            scale_loss = None
        
        # Compute Scale Consistency Loss if specified
        if self.scale_ratio_loss_weight > 0:
            gt_points_nf = gt_points_norm_factor[:, 0, 0, 0]
            pr_points_nf = pr_points_norm_factor[:, 0, 0, 0]
            gt_pose_nf = gt_pose_norm_factor          # [B]
            pr_pose_nf = pr_pose_norm_factor          # [B]

            pr_points_nf = pr_points_nf.detach()

            valid = (
                (gt_points_nf > 1e-8)
                & (pr_points_nf > 1e-8)
                & (gt_pose_nf > 1e-8)
                & (pr_pose_nf > 1e-8)
            )

            if valid.any():
                pred_ratio = pr_pose_nf[valid] / pr_points_nf[valid]
                gt_ratio = gt_pose_nf[valid] / gt_points_nf[valid]

                pred_ratio = pred_ratio.clamp_min(1e-8)
                gt_ratio = gt_ratio.clamp_min(1e-8)

                if self.loss_in_log:
                    pred_ratio = torch.log(pred_ratio)
                    gt_ratio = torch.log(gt_ratio)

                scale_ratio_loss = (
                    self.criterion(pred_ratio.unsqueeze(-1), gt_ratio.unsqueeze(-1), factor="scale",)
                    * self.scale_ratio_loss_weight
                )
            else:
                scale_ratio_loss = None
        else:
            scale_ratio_loss = None

        # Use helper function to generate loss terms and details
        if self.compute_world_frame_points_loss:
            losses_dict = {
                "pts3d": {
                    "values": pts3d_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
            }
        else:
            losses_dict = {}
        losses_dict.update(
            {
                "cam_pts3d": {
                    "values": cam_pts3d_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                self.depth_type_for_loss: {
                    "values": depth_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                "ray_directions": {
                    "values": ray_directions_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
                "pose_quats": {
                    "values": pose_quats_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
                "pose_trans": {
                    "values": pose_trans_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
                "scale": {
                    "values": scale_loss,
                    "use_mask": False,
                    "is_multi_view": False,
                },
                "normal": {
                    "values": normal_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
                "gradient_matching": {
                    "values": gradient_matching_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
                "scale_ratio": {
                    "values": scale_ratio_loss,
                    "use_mask": False,
                    "is_multi_view": False,
                },
            }
        )

        if self.compute_camera_head_loss:
            if any(v is not None for v in camera_translation_losses):
                losses_dict["camera_translation"] = {
                    "values": [
                        v if v is not None else torch.zeros(
                            (),
                            device=batch[i]["camera_intrinsics"].device,
                            dtype=batch[i]["camera_intrinsics"].dtype,
                        )
                        for i, v in enumerate(camera_translation_losses)
                    ],
                    "use_mask": False,
                    "is_multi_view": True,
                }

            if any(v is not None for v in camera_quaternion_losses):
                losses_dict["camera_quaternion"] = {
                    "values": [
                        v if v is not None else torch.zeros(
                            (),
                            device=batch[i]["camera_intrinsics"].device,
                            dtype=batch[i]["camera_intrinsics"].dtype,
                        )
                        for i, v in enumerate(camera_quaternion_losses)
                    ],
                    "use_mask": False,
                    "is_multi_view": True,
                }

            if any(v is not None for v in camera_fov_hw_losses):
                losses_dict["camera_fov_hw"] = {
                    "values": [
                        v if v is not None else torch.zeros(
                            (),
                            device=batch[i]["camera_intrinsics"].device,
                            dtype=batch[i]["camera_intrinsics"].dtype,
                        )
                        for i, v in enumerate(camera_fov_hw_losses)
                    ],
                    "use_mask": False,
                    "is_multi_view": True,
                }

        loss_terms, details = get_loss_terms_and_details(
            losses_dict,
            valid_masks,
            type(self).__name__,
            n_views,
            self.flatten_across_image_only,
        )
        losses = Sum(*loss_terms)

        return losses, (details | {})


class DisentangledFactoredGeometryScaleRegr3D(Criterion, MultiLoss):
    """
    Disentangled Regression Loss for Factored Geometry & Scale.
    """

    def __init__(
        self,
        criterion,
        norm_predictions=True,
        norm_mode="avg_dis",
        loss_in_log=True,
        flatten_across_image_only=False,
        depth_type_for_loss="depth_along_ray",
        depth_loss_weight=1,
        ray_directions_loss_weight=1,
        pose_quats_loss_weight=1,
        pose_trans_loss_weight=1,
        scale_loss_weight=1,
    ):
        """
        Initialize the disentangled loss criterion for Factored Geometry (Ray Directions, Depth, Pose) & Scale.
        It isolates/disentangles the contribution of each factor to the final task of 3D reconstruction.
        All the losses are in the same space where the loss for each factor is computed by constructing world-frame pointmaps.
        This sidesteps the difficulty of finding a proper weighting.
        For insance, for predicted rays, the GT depth & pose is used to construct the predicted world-frame pointmaps on which the loss is computed.
        Inspired by https://openaccess.thecvf.com/content_ICCV_2019/papers/Simonelli_Disentangling_Monocular_3D_Object_Detection_ICCV_2019_paper.pdf

        The pixel-level losses are computed in the following order:
        (1) depth, (2) ray directions, (3) pose quats, (4) pose trans, (5) scale.
        The predicited scene representation is always normalized w.r.t. the frame of view0.
        Loss is applied between the predicted metric scale and the ground truth metric scale.

        Args:
            criterion (BaseCriterion): The base criterion to use for computing the loss.
            norm_predictions (bool): If True, normalize the predictions before computing the loss.
            norm_mode (str): Normalization mode for the gt and predicted (optional) scene representation. Default: "avg_dis".
            loss_in_log (bool): If True, apply logarithmic transformation to input before
                computing the loss for depth, pointmaps and scale. Default: True.
            flatten_across_image_only (bool): If True, flatten H x W dimensions only when computing
                the loss. If False, flatten across batch and spatial dimensions. Default: False.
            depth_type_for_loss (str): Type of depth to use for loss computation. Default: "depth_along_ray".
                Options: "depth_along_ray", "depth_z"
            depth_loss_weight (float): Weight to use for the depth loss. Default: 1.
            ray_directions_loss_weight (float): Weight to use for the ray directions loss. Default: 1.
            pose_quats_loss_weight (float): Weight to use for the pose quats loss. Default: 1.
            pose_trans_loss_weight (float): Weight to use for the pose trans loss. Default: 1.
            scale_loss_weight (float): Weight to use for the scale loss. Default: 1.
        """
        super().__init__(criterion)
        self.norm_predictions = norm_predictions
        self.norm_mode = norm_mode
        self.loss_in_log = loss_in_log
        self.flatten_across_image_only = flatten_across_image_only
        self.depth_type_for_loss = depth_type_for_loss
        assert self.depth_type_for_loss in [
            "depth_along_ray",
            "depth_z",
        ], "depth_type_for_loss must be one of ['depth_along_ray', 'depth_z']"
        self.depth_loss_weight = depth_loss_weight
        self.ray_directions_loss_weight = ray_directions_loss_weight
        self.pose_quats_loss_weight = pose_quats_loss_weight
        self.pose_trans_loss_weight = pose_trans_loss_weight
        self.scale_loss_weight = scale_loss_weight

    def get_all_info(self, batch, preds, dist_clip=None):
        """
        Function to get all the information needed to compute the loss.
        Returns all quantities normalized w.r.t. camera of view0.
        """
        n_views = len(batch)

        # Everything is normalized w.r.t. camera of view0
        # Intialize lists to store data for all views
        # Ground truth quantities
        in_camera0 = closed_form_pose_inverse(batch[0]["camera_pose"])
        no_norm_gt_pts = []
        no_norm_gt_pts_cam = []
        no_norm_gt_depth = []
        no_norm_gt_pose_trans = []
        valid_masks = []
        gt_ray_directions = []
        gt_pose_quats = []
        # Predicted quantities
        no_norm_pr_pts = []
        no_norm_pr_pts_cam = []
        no_norm_pr_depth = []
        no_norm_pr_pose_trans = []
        pr_ray_directions = []
        pr_pose_quats = []
        metric_pr_pts_to_compute_scale = []

        # Get ground truth & prediction info for all views
        for i in range(n_views):
            # Get the ground truth
            no_norm_gt_pts.append(geotrf(in_camera0, batch[i]["pts3d"]))
            valid_masks.append(batch[i]["valid_mask"].clone())
            no_norm_gt_pts_cam.append(batch[i]["pts3d_cam"])
            gt_ray_directions.append(batch[i]["ray_directions_cam"])
            if self.depth_type_for_loss == "depth_along_ray":
                no_norm_gt_depth.append(batch[i]["depth_along_ray"])
            elif self.depth_type_for_loss == "depth_z":
                no_norm_gt_depth.append(batch[i]["pts3d_cam"][..., 2:])
            if i == 0:
                # For view0, initialize identity pose
                gt_pose_quats.append(
                    torch.tensor(
                        [0, 0, 0, 1],
                        dtype=gt_ray_directions[0].dtype,
                        device=gt_ray_directions[0].device,
                    )
                    .unsqueeze(0)
                    .repeat(gt_ray_directions[0].shape[0], 1)
                )
                no_norm_gt_pose_trans.append(
                    torch.tensor(
                        [0, 0, 0],
                        dtype=gt_ray_directions[0].dtype,
                        device=gt_ray_directions[0].device,
                    )
                    .unsqueeze(0)
                    .repeat(gt_ray_directions[0].shape[0], 1)
                )
            else:
                # For other views, transform pose to view0's frame
                gt_pose_quats_world = batch[i]["camera_pose_quats"]
                no_norm_gt_pose_trans_world = batch[i]["camera_pose_trans"]
                gt_pose_quats_in_view0, no_norm_gt_pose_trans_in_view0 = (
                    transform_pose_using_quats_and_trans_2_to_1(
                        batch[0]["camera_pose_quats"],
                        batch[0]["camera_pose_trans"],
                        gt_pose_quats_world,
                        no_norm_gt_pose_trans_world,
                    )
                )
                gt_pose_quats.append(gt_pose_quats_in_view0)
                no_norm_gt_pose_trans.append(no_norm_gt_pose_trans_in_view0)

            # Get predictions for normalized loss
            if self.depth_type_for_loss == "depth_along_ray":
                curr_view_no_norm_depth = preds[i]["depth_along_ray"]
            elif self.depth_type_for_loss == "depth_z":
                curr_view_no_norm_depth = preds[i]["pts3d_cam"][..., 2:]
            if "metric_scaling_factor" in preds[i].keys():
                # Divide by the predicted metric scaling factor to get the raw predicted points, depth_along_ray, and pose_trans
                # This detaches the predicted metric scaling factor from the geometry based loss
                curr_view_no_norm_pr_pts = preds[i]["pts3d"] / preds[i][
                    "metric_scaling_factor"
                ].unsqueeze(-1).unsqueeze(-1)
                curr_view_no_norm_pr_pts_cam = preds[i]["pts3d_cam"] / preds[i][
                    "metric_scaling_factor"
                ].unsqueeze(-1).unsqueeze(-1)
                curr_view_no_norm_depth = curr_view_no_norm_depth / preds[i][
                    "metric_scaling_factor"
                ].unsqueeze(-1).unsqueeze(-1)
                curr_view_no_norm_pr_pose_trans = (
                    preds[i]["cam_trans"] / preds[i]["metric_scaling_factor"]
                )
            else:
                curr_view_no_norm_pr_pts = preds[i]["pts3d"]
                curr_view_no_norm_pr_pts_cam = preds[i]["pts3d_cam"]
                curr_view_no_norm_depth = curr_view_no_norm_depth
                curr_view_no_norm_pr_pose_trans = preds[i]["cam_trans"]
            no_norm_pr_pts.append(curr_view_no_norm_pr_pts)
            no_norm_pr_pts_cam.append(curr_view_no_norm_pr_pts_cam)
            no_norm_pr_depth.append(curr_view_no_norm_depth)
            no_norm_pr_pose_trans.append(curr_view_no_norm_pr_pose_trans)
            pr_ray_directions.append(preds[i]["ray_directions"])
            pr_pose_quats.append(preds[i]["cam_quats"])

            # Get the predicted metric scale points
            if "metric_scaling_factor" in preds[i].keys():
                # Detach the raw predicted points so that the scale loss is only applied to the scaling factor
                curr_view_metric_pr_pts_to_compute_scale = (
                    curr_view_no_norm_pr_pts.detach()
                    * preds[i]["metric_scaling_factor"].unsqueeze(-1).unsqueeze(-1)
                )
            else:
                curr_view_metric_pr_pts_to_compute_scale = (
                    curr_view_no_norm_pr_pts.clone()
                )
            metric_pr_pts_to_compute_scale.append(
                curr_view_metric_pr_pts_to_compute_scale
            )

        if dist_clip is not None:
            # Points that are too far-away == invalid
            for i in range(n_views):
                dis = no_norm_gt_pts[i].norm(dim=-1)
                valid_masks[i] = valid_masks[i] & (dis <= dist_clip)

        # Initialize normalized tensors
        gt_pts = [torch.zeros_like(pts) for pts in no_norm_gt_pts]
        gt_pts_cam = [torch.zeros_like(pts_cam) for pts_cam in no_norm_gt_pts_cam]
        gt_depth = [torch.zeros_like(depth) for depth in no_norm_gt_depth]
        gt_pose_trans = [torch.zeros_like(trans) for trans in no_norm_gt_pose_trans]

        pr_pts = [torch.zeros_like(pts) for pts in no_norm_pr_pts]
        pr_pts_cam = [torch.zeros_like(pts_cam) for pts_cam in no_norm_pr_pts_cam]
        pr_depth = [torch.zeros_like(depth) for depth in no_norm_pr_depth]
        pr_pose_trans = [torch.zeros_like(trans) for trans in no_norm_pr_pose_trans]

        # Normalize the predicted points if specified
        if self.norm_predictions:
            pr_normalization_output = normalize_multiple_pointclouds(
                no_norm_pr_pts,
                valid_masks,
                self.norm_mode,
                ret_factor=True,
            )
            pr_pts_norm = pr_normalization_output[:-1]
            pr_norm_factor = pr_normalization_output[-1]
        
        # Normalize the ground truth points
        gt_normalization_output = normalize_multiple_pointclouds(
            no_norm_gt_pts, valid_masks, self.norm_mode, ret_factor=True
        )
        gt_pts_norm = gt_normalization_output[:-1]
        gt_norm_factor = gt_normalization_output[-1]

        for i in range(n_views):
            if self.norm_predictions:
                # Assign the normalized predictions
                pr_pts[i] = pr_pts_norm[i]
                pr_pts_cam[i] = no_norm_pr_pts_cam[i] / pr_norm_factor
                pr_depth[i] = no_norm_pr_depth[i] / pr_norm_factor
                pr_pose_trans[i] = no_norm_pr_pose_trans[i] / pr_norm_factor[:, :, 0, 0]
            else:
                pr_pts[i] = no_norm_pr_pts[i]
                pr_pts_cam[i] = no_norm_pr_pts_cam[i]
                pr_depth[i] = no_norm_pr_depth[i]
                pr_pose_trans[i] = no_norm_pr_pose_trans[i]
            # Assign the normalized ground truth quantities
            gt_pts[i] = gt_pts_norm[i]
            gt_pts_cam[i] = no_norm_gt_pts_cam[i] / gt_norm_factor
            gt_depth[i] = no_norm_gt_depth[i] / gt_norm_factor
            gt_pose_trans[i] = no_norm_gt_pose_trans[i] / gt_norm_factor[:, :, 0, 0]

        # Get the mask indicating ground truth metric scale quantities
        metric_scale_mask = batch[0]["is_metric_scale"]
        valid_gt_norm_factor_mask = (
            gt_norm_factor[:, 0, 0, 0] > 1e-8
        )  # Mask out cases where depth for all views is invalid
        valid_metric_scale_mask = metric_scale_mask & valid_gt_norm_factor_mask

        if valid_metric_scale_mask.any():
            # Compute the scale norm factor using the predicted metric scale points
            metric_pr_normalization_output = normalize_multiple_pointclouds(
                metric_pr_pts_to_compute_scale,
                valid_masks,
                self.norm_mode,
                ret_factor=True,
            )
            pr_metric_norm_factor = metric_pr_normalization_output[-1]

            # Get the valid ground truth and predicted scale norm factors for the metric ground truth quantities
            gt_metric_norm_factor = gt_norm_factor[valid_metric_scale_mask]
            pr_metric_norm_factor = pr_metric_norm_factor[valid_metric_scale_mask]
        else:
            gt_metric_norm_factor = None
            pr_metric_norm_factor = None

        # Get ambiguous masks
        ambiguous_masks = []
        for i in range(n_views):
            ambiguous_masks.append(
                (~batch[i]["non_ambiguous_mask"]) & (~valid_masks[i])
            )

        # Pack into info dicts
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

        return (
            gt_info,
            pred_info,
            valid_masks,
            ambiguous_masks,
            gt_metric_norm_factor,
            pr_metric_norm_factor,
        )

    def compute_loss(self, batch, preds, **kw):
        (
            gt_info,
            pred_info,
            valid_masks,
            ambiguous_masks,
            gt_metric_norm_factor,
            pr_metric_norm_factor,
        ) = self.get_all_info(batch, preds, **kw)
        n_views = len(batch)

        pose_trans_losses = []
        pose_quats_losses = []
        ray_directions_losses = []
        depth_losses = []

        for i in range(n_views):
            # Get the GT factored quantities for the current view
            gt_pts3d = gt_info[i]["pts3d"]
            gt_ray_directions = gt_info[i]["ray_directions"]
            gt_depth = gt_info[i][self.depth_type_for_loss]
            gt_pose_trans = gt_info[i]["pose_trans"]
            gt_pose_quats = gt_info[i]["pose_quats"]

            # Get the predicted factored quantities for the current view
            pred_ray_directions = pred_info[i]["ray_directions"]
            pred_depth = pred_info[i][self.depth_type_for_loss]
            pred_pose_trans = pred_info[i]["pose_trans"]
            pred_pose_quats = pred_info[i]["pose_quats"]

            # Get the predicted world-frame pointmaps using the different factors
            if self.depth_type_for_loss == "depth_along_ray":
                pred_ray_directions_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        pred_ray_directions,
                        gt_depth,
                        gt_pose_trans,
                        gt_pose_quats,
                    )
                )
                pred_depth_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        gt_ray_directions,
                        pred_depth,
                        gt_pose_trans,
                        gt_pose_quats,
                    )
                )
                pred_pose_trans_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        gt_ray_directions,
                        gt_depth,
                        pred_pose_trans,
                        gt_pose_quats,
                    )
                )
                pred_pose_quats_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        gt_ray_directions,
                        gt_depth,
                        gt_pose_trans,
                        pred_pose_quats,
                    )
                )
            else:
                raise NotImplementedError

            # Mask out the valid quantities as required
            if not self.flatten_across_image_only:
                # Flatten the points across the entire batch with the masks
                pred_ray_directions_pts3d = pred_ray_directions_pts3d[valid_masks[i]]
                pred_depth_pts3d = pred_depth_pts3d[valid_masks[i]]
                pred_pose_trans_pts3d = pred_pose_trans_pts3d[valid_masks[i]]
                pred_pose_quats_pts3d = pred_pose_quats_pts3d[valid_masks[i]]
                gt_pts3d = gt_pts3d[valid_masks[i]]
            else:
                # Flatten the H x W dimensions to H*W
                batch_size, _, _, pts_dim = gt_pts3d.shape
                pred_ray_directions_pts3d = pred_ray_directions_pts3d.view(
                    batch_size, -1, pts_dim
                )
                pred_depth_pts3d = pred_depth_pts3d.view(batch_size, -1, pts_dim)
                pred_pose_trans_pts3d = pred_pose_trans_pts3d.view(
                    batch_size, -1, pts_dim
                )
                pred_pose_quats_pts3d = pred_pose_quats_pts3d.view(
                    batch_size, -1, pts_dim
                )
                gt_pts3d = gt_pts3d.view(batch_size, -1, pts_dim)
                valid_masks[i] = valid_masks[i].view(batch_size, -1)

            # Apply loss in log space if specified
            if self.loss_in_log:
                gt_pts3d = apply_log_to_norm(gt_pts3d)
                pred_ray_directions_pts3d = apply_log_to_norm(pred_ray_directions_pts3d)
                pred_depth_pts3d = apply_log_to_norm(pred_depth_pts3d)
                pred_pose_trans_pts3d = apply_log_to_norm(pred_pose_trans_pts3d)
                pred_pose_quats_pts3d = apply_log_to_norm(pred_pose_quats_pts3d)

            # Compute pose translation loss
            pose_trans_loss = self.criterion(
                pred_pose_trans_pts3d, gt_pts3d, factor="pose_trans"
            )
            pose_trans_loss = pose_trans_loss * self.pose_trans_loss_weight
            pose_trans_losses.append(pose_trans_loss)

            # Compute pose rotation loss
            pose_quats_loss = self.criterion(
                pred_pose_quats_pts3d, gt_pts3d, factor="pose_quats"
            )
            pose_quats_loss = pose_quats_loss * self.pose_quats_loss_weight
            pose_quats_losses.append(pose_quats_loss)

            # Compute ray direction loss
            ray_directions_loss = self.criterion(
                pred_ray_directions_pts3d, gt_pts3d, factor="ray_directions"
            )
            ray_directions_loss = ray_directions_loss * self.ray_directions_loss_weight
            ray_directions_losses.append(ray_directions_loss)

            # Compute depth loss
            depth_loss = self.criterion(pred_depth_pts3d, gt_pts3d, factor="depth")
            depth_loss = depth_loss * self.depth_loss_weight
            depth_losses.append(depth_loss)

        # Compute the scale loss
        if gt_metric_norm_factor is not None:
            if self.loss_in_log:
                gt_metric_norm_factor = apply_log_to_norm(gt_metric_norm_factor)
                pr_metric_norm_factor = apply_log_to_norm(pr_metric_norm_factor)
            scale_loss = (
                self.criterion(
                    pr_metric_norm_factor, gt_metric_norm_factor, factor="scale"
                )
                * self.scale_loss_weight
            )
        else:
            scale_loss = None

        # Use helper function to generate loss terms and details
        losses_dict = {}
        losses_dict.update(
            {
                self.depth_type_for_loss: {
                    "values": depth_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                "ray_directions": {
                    "values": ray_directions_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                "pose_quats": {
                    "values": pose_quats_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                "pose_trans": {
                    "values": pose_trans_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                "scale": {
                    "values": scale_loss,
                    "use_mask": False,
                    "is_multi_view": False,
                },
            }
        )
        loss_terms, details = get_loss_terms_and_details(
            losses_dict,
            valid_masks,
            type(self).__name__,
            n_views,
            self.flatten_across_image_only,
        )
        losses = Sum(*loss_terms)

        return losses, (details | {})


class DisentangledFactoredGeometryScaleRegr3DPlusNormalGMLoss(
    DisentangledFactoredGeometryScaleRegr3D
):
    """
    Disentangled Regression, Normals & Gradient Matching Loss for Factored Geometry & Scale.
    """

    def __init__(
        self,
        criterion,
        norm_predictions=True,
        norm_mode="avg_dis",
        loss_in_log=True,
        flatten_across_image_only=False,
        depth_type_for_loss="depth_along_ray",
        depth_loss_weight=1,
        ray_directions_loss_weight=1,
        pose_quats_loss_weight=1,
        pose_trans_loss_weight=1,
        scale_loss_weight=1,
        apply_normal_and_gm_loss_to_synthetic_data_only=True,
        normal_loss_weight=1,
        gm_loss_weight=1,
    ):
        """
        Initialize the disentangled loss criterion for Factored Geometry (Ray Directions, Depth, Pose) & Scale.
        See parent class (DisentangledFactoredGeometryScaleRegr3D) for more details.
        Additionally computes:
        (1) Normal Loss over the Camera Frame Pointmaps in euclidean coordinates,
        (2) Gradient Matching (GM) Loss over the Depth Z in log space. (MiDAS applied GM loss in disparity space)

        Args:
            criterion (BaseCriterion): The base criterion to use for computing the loss.
            norm_predictions (bool): If True, normalize the predictions before computing the loss.
            norm_mode (str): Normalization mode for the gt and predicted (optional) scene representation. Default: "avg_dis".
            loss_in_log (bool): If True, apply logarithmic transformation to input before
                computing the loss for depth, pointmaps and scale. Default: True.
            flatten_across_image_only (bool): If True, flatten H x W dimensions only when computing
                the loss. If False, flatten across batch and spatial dimensions. Default: False.
            depth_type_for_loss (str): Type of depth to use for loss computation. Default: "depth_along_ray".
                Options: "depth_along_ray", "depth_z"
            depth_loss_weight (float): Weight to use for the depth loss. Default: 1.
            ray_directions_loss_weight (float): Weight to use for the ray directions loss. Default: 1.
            pose_quats_loss_weight (float): Weight to use for the pose quats loss. Default: 1.
            pose_trans_loss_weight (float): Weight to use for the pose trans loss. Default: 1.
            scale_loss_weight (float): Weight to use for the scale loss. Default: 1.
            apply_normal_and_gm_loss_to_synthetic_data_only (bool): If True, apply the normal and gm loss only to synthetic data.
                If False, apply the normal and gm loss to all data. Default: True.
            normal_loss_weight (float): Weight to use for the normal loss. Default: 1.
            gm_loss_weight (float): Weight to use for the gm loss. Default: 1.
        """
        super().__init__(
            criterion=criterion,
            norm_predictions=norm_predictions,
            norm_mode=norm_mode,
            loss_in_log=loss_in_log,
            flatten_across_image_only=flatten_across_image_only,
            depth_type_for_loss=depth_type_for_loss,
            depth_loss_weight=depth_loss_weight,
            ray_directions_loss_weight=ray_directions_loss_weight,
            pose_quats_loss_weight=pose_quats_loss_weight,
            pose_trans_loss_weight=pose_trans_loss_weight,
            scale_loss_weight=scale_loss_weight,
        )
        self.apply_normal_and_gm_loss_to_synthetic_data_only = (
            apply_normal_and_gm_loss_to_synthetic_data_only
        )
        self.normal_loss_weight = normal_loss_weight
        self.gm_loss_weight = gm_loss_weight

    def compute_loss(self, batch, preds, **kw):
        (
            gt_info,
            pred_info,
            valid_masks,
            ambiguous_masks,
            gt_metric_norm_factor,
            pr_metric_norm_factor,
        ) = self.get_all_info(batch, preds, **kw)
        n_views = len(batch)

        normal_losses = []
        gradient_matching_losses = []
        pose_trans_losses = []
        pose_quats_losses = []
        ray_directions_losses = []
        depth_losses = []

        for i in range(n_views):
            # Get the camera frame points, log space depth_z & valid masks
            pred_local_pts3d = pred_info[i]["pts3d_cam"]
            pred_depth_z = pred_local_pts3d[..., 2:]
            pred_depth_z = apply_log_to_norm(pred_depth_z)
            gt_local_pts3d = gt_info[i]["pts3d_cam"]
            gt_depth_z = gt_local_pts3d[..., 2:]
            gt_depth_z = apply_log_to_norm(gt_depth_z)
            valid_mask_for_normal_gm_loss = valid_masks[i].clone()

            # Update the validity mask for normal & gm loss based on the synthetic data mask if required
            if self.apply_normal_and_gm_loss_to_synthetic_data_only:
                synthetic_mask = batch[i]["is_synthetic"]  # (B, )
                synthetic_mask = synthetic_mask.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                synthetic_mask = synthetic_mask.expand(
                    -1, pred_depth_z.shape[1], pred_depth_z.shape[2]
                )  # (B, H, W)
                valid_mask_for_normal_gm_loss = (
                    valid_mask_for_normal_gm_loss & synthetic_mask
                )

            # Compute the normal loss
            normal_loss = compute_normal_loss(
                pred_local_pts3d, gt_local_pts3d, valid_mask_for_normal_gm_loss.clone()
            )
            normal_loss = normal_loss * self.normal_loss_weight
            normal_losses.append(normal_loss)

            # Compute the gradient matching loss
            gradient_matching_loss = compute_gradient_matching_loss(
                pred_depth_z, gt_depth_z, valid_mask_for_normal_gm_loss.clone()
            )
            gradient_matching_loss = gradient_matching_loss * self.gm_loss_weight
            gradient_matching_losses.append(gradient_matching_loss)

            # Get the GT factored quantities for the current view
            gt_pts3d = gt_info[i]["pts3d"]
            gt_ray_directions = gt_info[i]["ray_directions"]
            gt_depth = gt_info[i][self.depth_type_for_loss]
            gt_pose_trans = gt_info[i]["pose_trans"]
            gt_pose_quats = gt_info[i]["pose_quats"]

            # Get the predicted factored quantities for the current view
            pred_ray_directions = pred_info[i]["ray_directions"]
            pred_depth = pred_info[i][self.depth_type_for_loss]
            pred_pose_trans = pred_info[i]["pose_trans"]
            pred_pose_quats = pred_info[i]["pose_quats"]

            # Get the predicted world-frame pointmaps using the different factors
            if self.depth_type_for_loss == "depth_along_ray":
                pred_ray_directions_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        pred_ray_directions,
                        gt_depth,
                        gt_pose_trans,
                        gt_pose_quats,
                    )
                )
                pred_depth_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        gt_ray_directions,
                        pred_depth,
                        gt_pose_trans,
                        gt_pose_quats,
                    )
                )
                pred_pose_trans_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        gt_ray_directions,
                        gt_depth,
                        pred_pose_trans,
                        gt_pose_quats,
                    )
                )
                pred_pose_quats_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        gt_ray_directions,
                        gt_depth,
                        gt_pose_trans,
                        pred_pose_quats,
                    )
                )
            else:
                raise NotImplementedError

            # Mask out the valid quantities as required
            if not self.flatten_across_image_only:
                # Flatten the points across the entire batch with the masks
                pred_ray_directions_pts3d = pred_ray_directions_pts3d[valid_masks[i]]
                pred_depth_pts3d = pred_depth_pts3d[valid_masks[i]]
                pred_pose_trans_pts3d = pred_pose_trans_pts3d[valid_masks[i]]
                pred_pose_quats_pts3d = pred_pose_quats_pts3d[valid_masks[i]]
                gt_pts3d = gt_pts3d[valid_masks[i]]
            else:
                # Flatten the H x W dimensions to H*W
                batch_size, _, _, pts_dim = gt_pts3d.shape
                pred_ray_directions_pts3d = pred_ray_directions_pts3d.view(
                    batch_size, -1, pts_dim
                )
                pred_depth_pts3d = pred_depth_pts3d.view(batch_size, -1, pts_dim)
                pred_pose_trans_pts3d = pred_pose_trans_pts3d.view(
                    batch_size, -1, pts_dim
                )
                pred_pose_quats_pts3d = pred_pose_quats_pts3d.view(
                    batch_size, -1, pts_dim
                )
                gt_pts3d = gt_pts3d.view(batch_size, -1, pts_dim)
                valid_masks[i] = valid_masks[i].view(batch_size, -1)

            # Apply loss in log space if specified
            if self.loss_in_log:
                gt_pts3d = apply_log_to_norm(gt_pts3d)
                pred_ray_directions_pts3d = apply_log_to_norm(pred_ray_directions_pts3d)
                pred_depth_pts3d = apply_log_to_norm(pred_depth_pts3d)
                pred_pose_trans_pts3d = apply_log_to_norm(pred_pose_trans_pts3d)
                pred_pose_quats_pts3d = apply_log_to_norm(pred_pose_quats_pts3d)

            # Compute pose translation loss
            pose_trans_loss = self.criterion(
                pred_pose_trans_pts3d, gt_pts3d, factor="pose_trans"
            )
            pose_trans_loss = pose_trans_loss * self.pose_trans_loss_weight
            pose_trans_losses.append(pose_trans_loss)

            # Compute pose rotation loss
            pose_quats_loss = self.criterion(
                pred_pose_quats_pts3d, gt_pts3d, factor="pose_quats"
            )
            pose_quats_loss = pose_quats_loss * self.pose_quats_loss_weight
            pose_quats_losses.append(pose_quats_loss)

            # Compute ray direction loss
            ray_directions_loss = self.criterion(
                pred_ray_directions_pts3d, gt_pts3d, factor="ray_directions"
            )
            ray_directions_loss = ray_directions_loss * self.ray_directions_loss_weight
            ray_directions_losses.append(ray_directions_loss)

            # Compute depth loss
            depth_loss = self.criterion(pred_depth_pts3d, gt_pts3d, factor="depth")
            depth_loss = depth_loss * self.depth_loss_weight
            depth_losses.append(depth_loss)

        # Compute the scale loss
        if gt_metric_norm_factor is not None:
            if self.loss_in_log:
                gt_metric_norm_factor = apply_log_to_norm(gt_metric_norm_factor)
                pr_metric_norm_factor = apply_log_to_norm(pr_metric_norm_factor)
            scale_loss = (
                self.criterion(
                    pr_metric_norm_factor, gt_metric_norm_factor, factor="scale"
                )
                * self.scale_loss_weight
            )
        else:
            scale_loss = None

        # Use helper function to generate loss terms and details
        losses_dict = {}
        losses_dict.update(
            {
                self.depth_type_for_loss: {
                    "values": depth_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                "ray_directions": {
                    "values": ray_directions_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                "pose_quats": {
                    "values": pose_quats_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                "pose_trans": {
                    "values": pose_trans_losses,
                    "use_mask": True,
                    "is_multi_view": True,
                },
                "scale": {
                    "values": scale_loss,
                    "use_mask": False,
                    "is_multi_view": False,
                },
                "normal": {
                    "values": normal_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
                "gradient_matching": {
                    "values": gradient_matching_losses,
                    "use_mask": False,
                    "is_multi_view": True,
                },
            }
        )
        loss_terms, details = get_loss_terms_and_details(
            losses_dict,
            valid_masks,
            type(self).__name__,
            n_views,
            self.flatten_across_image_only,
        )
        losses = Sum(*loss_terms)

        return losses, (details | {})

class CrossEntropyLoss(BaseCriterion):
    """Cross Entropy loss for semantic segmentation (supports label smoothing / class weights)"""

    def __init__(
        self,
        reduction="mean",
        ignore_index=0,
        label_smoothing=0.0,
        class_weight=None,
    ):
        super().__init__(reduction=reduction)
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)

        # class_weight can be list/tuple/tensor or None
        if class_weight is None:
            self.register_buffer("_class_weight", torch.tensor([], dtype=torch.float32), persistent=False)
        else:
            cw = torch.as_tensor(class_weight, dtype=torch.float32)
            self.register_buffer("_class_weight", cw, persistent=False)

    @property
    def class_weight(self):
        return None if self._class_weight.numel() == 0 else self._class_weight

    def forward(self, predicted_logits, reference_labels):
        """
        Args:
            predicted_logits: (B, C, H, W)
            reference_labels: (B, H, W), dtype long

        Returns:
            scalar tensor
        """
        return F.cross_entropy(
            predicted_logits,
            reference_labels,
            weight=self.class_weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


class SemanticSegmentationLoss(Criterion, MultiLoss):
    """
    Semantic segmentation loss aligned with MapAnythingSemantic outputs.

    Expected prediction keys (preferred -> fallback):
      - pred["semantic_logits"]  (B, C, H, W)
      - pred["semantic"]         (backward compatibility)

    Expected GT key:
      - gt["instance"] (H, W) or (B, H, W), numpy or torch.Tensor

    Optional GT masks for masking supervision:
      - gt["valid_mask"]          (H, W) / (B, H, W)
      - gt["non_ambiguous_mask"]  (H, W) / (B, H, W)
    """

    def __init__(
        self,
        ignore_index=0,
        label_key="instance",
        logit_key="semantic_logits",
        # masking controls
        apply_valid_mask=True,
        apply_non_ambiguous_mask=True,
        apply_pred_semantic_valid_mask=False,
        # CE / Dice composition
        ce_weight=1.0,
        dice_weight=0.0,
        include_background_in_dice=False,
        # CE options
        label_smoothing=0.0,
        class_weight=None,
        # tiny eps
        eps=1e-6,
        **kwargs,
    ):
        ce_criterion = CrossEntropyLoss(
            reduction="mean",
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            class_weight=class_weight,
        )
        super().__init__(ce_criterion)

        self.ignore_index = int(ignore_index)
        self.label_key = label_key
        self.logit_key = logit_key

        self.apply_valid_mask = bool(apply_valid_mask)
        self.apply_non_ambiguous_mask = bool(apply_non_ambiguous_mask)
        self.apply_pred_semantic_valid_mask = bool(apply_pred_semantic_valid_mask)

        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.include_background_in_dice = bool(include_background_in_dice)

        self.eps = float(eps)

    # -------------------------
    # helpers
    # -------------------------
    @staticmethod
    def _to_tensor(x, device=None, dtype=None):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.as_tensor(x)
        if device is not None:
            t = t.to(device=device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t

    @staticmethod
    def _ensure_bhw(label_t: torch.Tensor) -> torch.Tensor:
        """
        Convert label to (B,H,W).
        Accepts:
          - (H,W)
          - (1,H,W)
          - (B,H,W)
          - (B,1,H,W)
        """
        if label_t.ndim == 2:
            label_t = label_t.unsqueeze(0)
        elif label_t.ndim == 4 and label_t.shape[1] == 1:
            label_t = label_t[:, 0]
        elif label_t.ndim == 4 and label_t.shape[-1] == 1:
            label_t = label_t[..., 0]
        elif label_t.ndim != 3:
            raise ValueError(f"Unsupported label shape for semantic gt: {tuple(label_t.shape)}")
        return label_t

    @staticmethod
    def _ensure_mask_bhw(mask_t: torch.Tensor) -> torch.Tensor:
        """
        Convert mask to (B,H,W) bool.
        """
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)
        elif mask_t.ndim == 4 and mask_t.shape[1] == 1:
            mask_t = mask_t[:, 0]
        elif mask_t.ndim == 4 and mask_t.shape[-1] == 1:
            mask_t = mask_t[..., 0]
        elif mask_t.ndim != 3:
            raise ValueError(f"Unsupported mask shape: {tuple(mask_t.shape)}")
        return mask_t.bool()

    def _resize_label_to_logits(self, labels_bhw: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        _, _, h, w = logits.shape
        if labels_bhw.shape[-2:] == (h, w):
            return labels_bhw
        # nearest for labels
        labels_bhw = F.interpolate(
            labels_bhw.unsqueeze(1).float(),
            size=(h, w),
            mode="nearest",
        ).squeeze(1).long()
        return labels_bhw

    def _resize_mask_to_logits(self, mask_bhw: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        _, _, h, w = logits.shape
        if mask_bhw.shape[-2:] == (h, w):
            return mask_bhw.bool()
        mask_bhw = F.interpolate(
            mask_bhw.unsqueeze(1).float(),
            size=(h, w),
            mode="nearest",
        ).squeeze(1).bool()
        return mask_bhw

    def _compute_soft_dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,C,H,W)
        target: (B,H,W), with ignore_index allowed
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (B,C,H,W)

        valid = target != self.ignore_index  # (B,H,W)
        if valid.sum() == 0:
            return logits.new_zeros(())

        # prepare safe target for one_hot (replace ignore_index with 0 temporarily)
        target_safe = target.clone()
        target_safe[~valid] = 0
        one_hot = F.one_hot(target_safe.long(), num_classes=num_classes)  # (B,H,W,C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B,C,H,W)

        valid_f = valid.unsqueeze(1).float()  # (B,1,H,W)
        probs = probs * valid_f
        one_hot = one_hot * valid_f

        if not self.include_background_in_dice and num_classes > 1:
            probs = probs[:, 1:]
            one_hot = one_hot[:, 1:]

        # If all classes removed (e.g. num_classes=1 and exclude bg), return zero
        if probs.shape[1] == 0:
            return logits.new_zeros(())

        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        denom = (probs + one_hot).sum(dim=(0, 2, 3))

        # only average classes present in GT to avoid unstable empty-class dice
        gt_present = one_hot.sum(dim=(0, 2, 3)) > 0
        if gt_present.sum() == 0:
            return logits.new_zeros(())

        dice_per_class = (2.0 * intersection + self.eps) / (denom + self.eps)
        dice_loss = 1.0 - dice_per_class[gt_present].mean()
        return dice_loss

    def _get_pred_logits(self, pred: dict):
        # aligned with MapAnythingSemantic.forward/infer
        if self.logit_key in pred:
            return pred[self.logit_key]
        if "semantic_logits" in pred:
            return pred["semantic_logits"]
        if "semantic" in pred:  # backward compatibility
            return pred["semantic"]
        return None

    def compute_loss(self, batch, preds, **kw):
        """
        Args:
            batch: list[dict] GT views
            preds: list[dict] predicted views (from MapAnythingSemantic.forward)

        Returns:
            loss, details
        """
        loss_list = []
        details = {}
        self_name = type(self).__name__

        ce_running = []
        dice_running = []
        total_running = []

        # find fallback device for zero tensor if all skipped
        fallback_device = None
        for p in preds:
            if isinstance(p, dict):
                for v in p.values():
                    if isinstance(v, torch.Tensor):
                        fallback_device = v.device
                        break
            if fallback_device is not None:
                break
        if fallback_device is None:
            fallback_device = torch.device("cpu")

        for view_idx, (gt, pred) in enumerate(zip(batch, preds)):
            logits = self._get_pred_logits(pred)
            if logits is None:
                continue
            if logits.ndim != 4:
                raise ValueError(
                    f"Expected semantic logits shape (B,C,H,W), got {tuple(logits.shape)}"
                )

            gt_labels = gt.get(self.label_key, None)
            if gt_labels is None:
                continue

            # GT labels -> (B,H,W), long, same device
            target = self._to_tensor(gt_labels, device=logits.device)
            target = self._ensure_bhw(target).long()
            target = self._resize_label_to_logits(target, logits)

            # Build ignore mask from GT/pred masks
            ignore_mask = torch.zeros_like(target, dtype=torch.bool, device=target.device)

            if self.apply_valid_mask and ("valid_mask" in gt):
                valid_mask = self._to_tensor(gt["valid_mask"], device=logits.device)
                valid_mask = self._ensure_mask_bhw(valid_mask)
                valid_mask = self._resize_mask_to_logits(valid_mask, logits)
                ignore_mask |= (~valid_mask)

            if self.apply_non_ambiguous_mask and ("non_ambiguous_mask" in gt):
                non_amb = self._to_tensor(gt["non_ambiguous_mask"], device=logits.device)
                non_amb = self._ensure_mask_bhw(non_amb)
                non_amb = self._resize_mask_to_logits(non_amb, logits)
                ignore_mask |= (~non_amb)

            if self.apply_pred_semantic_valid_mask and ("semantic_valid_mask" in pred):
                sem_valid = self._to_tensor(pred["semantic_valid_mask"], device=logits.device)
                sem_valid = self._ensure_mask_bhw(sem_valid)
                sem_valid = self._resize_mask_to_logits(sem_valid, logits)
                ignore_mask |= (~sem_valid)

            target = target.clone()
            target[ignore_mask] = self.ignore_index

            # If a view has no valid pixels after masking, skip it
            valid_count = int((target != self.ignore_index).sum().item())
            if valid_count == 0:
                continue

            ce_loss = self.criterion(logits, target)  # scalar
            total_loss = self.ce_weight * ce_loss

            view_details = {
                f"{self_name}_ce_view{view_idx + 1}": float(ce_loss),
            }

            if self.dice_weight > 0:
                dice_loss = self._compute_soft_dice_loss(logits, target)
                total_loss = total_loss + self.dice_weight * dice_loss
                view_details[f"{self_name}_dice_view{view_idx + 1}"] = float(dice_loss)
                dice_running.append(float(dice_loss))

            view_details[f"{self_name}_view{view_idx + 1}"] = float(total_loss)
            view_details[f"{self_name}_valid_pixels_view{view_idx + 1}"] = valid_count

            details.update(view_details)
            ce_running.append(float(ce_loss))
            total_running.append(float(total_loss))

            loss_list.append((total_loss, None, "semantic"))

        if len(loss_list) == 0:
            zero = torch.tensor(0.0, device=fallback_device)
            details[f"{self_name}_avg"] = 0.0
            details[f"{self_name}_ce_avg"] = 0.0
            if self.dice_weight > 0:
                details[f"{self_name}_dice_avg"] = 0.0
            loss_list.append((zero, None, "semantic"))
        else:
            details[f"{self_name}_avg"] = sum(total_running) / len(total_running)
            details[f"{self_name}_ce_avg"] = sum(ce_running) / len(ce_running)
            if self.dice_weight > 0 and len(dice_running) > 0:
                details[f"{self_name}_dice_avg"] = sum(dice_running) / len(dice_running)

        return Sum(*loss_list), (details | {})
    


class AnySplatOfficialRenderLoss(MultiLoss):
    """AnySplat-style render supervision.

    Terms:
        RGB MSE against GT images.
        LPIPS against GT images.
        Rendered z-depth regression against the base predicted z-depth.
    """

    def __init__(
        self,
        mse_weight=1.0,
        lpips_weight=0.05,
        depth_consis_weight=0.1,
        lpips_apply_after_step=0,
        use_valid_mask=True,

        # Depth supervision.
        depth_loss_type="smooth_l1",
        depth_beta=0.05,

        # Depth validity filtering in normalized GS depth space.
        target_depth_min=1.0e-6,
        rendered_depth_min=1.0e-6,
        rendered_depth_max=0.0,  # <=0 disables max-range filtering

        # Outlier rejection based on |rendered_depth - target_depth|.
        depth_residual_max=10.0,      # <=0 disables hard residual filtering
        depth_keep_percent=95.0,      # keep smallest residuals; 100 disables quantile filtering
        min_depth_valid_pixels=32,

        # Ignore pixels that were barely rendered.
        alpha_mask_min=1.0e-4,
    ):
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.lpips_weight = float(lpips_weight)
        self.depth_consis_weight = float(depth_consis_weight)
        self.lpips_apply_after_step = int(lpips_apply_after_step)
        self.use_valid_mask = bool(use_valid_mask)

        self.depth_loss_type = str(depth_loss_type)
        if self.depth_loss_type not in {"mse", "l1", "smooth_l1"}:
            raise ValueError(
                "depth_loss_type must be one of ['mse', 'l1', 'smooth_l1'], "
                f"got {self.depth_loss_type}"
            )
        self.depth_beta = float(depth_beta)

        self.target_depth_min = float(target_depth_min)
        self.rendered_depth_min = float(rendered_depth_min)
        self.rendered_depth_max = float(rendered_depth_max)

        self.depth_residual_max = float(depth_residual_max)
        self.depth_keep_percent = float(depth_keep_percent)
        if not (0.0 < self.depth_keep_percent <= 100.0):
            raise ValueError(
                f"depth_keep_percent must be in (0, 100], got {self.depth_keep_percent}"
            )
        self.min_depth_valid_pixels = int(min_depth_valid_pixels)
        self.alpha_mask_min = float(alpha_mask_min)

        self.schedule_step = 0
        self.lpips = None
        if self.lpips_weight > 0:
            try:
                from lpips import LPIPS
            except ImportError as exc:
                raise ImportError(
                    "AnySplatOfficialRenderLoss with lpips_weight > 0 requires "
                    "the `lpips` package. Install project dependencies or set "
                    "lpips_weight=0."
                ) from exc
            self.lpips = LPIPS(net="vgg")
            self.lpips.eval()
            for param in self.lpips.parameters():
                param.requires_grad_(False)

    def get_name(self):
        return "AnySplatOfficialRenderLoss"

    def set_schedule_step(self, step: int, epoch: float = 0.0) -> None:
        self.schedule_step = int(step)

    @staticmethod
    def _rgb01(img):
        if img.min().detach() < -0.01:
            return (img + 1.0) * 0.5
        return img.clamp(0.0, 1.0)

    @staticmethod
    def _stack_optional_valid_mask(gt, device):
        if not all("valid_mask" in view for view in gt):
            return None
        masks = []
        for view in gt:
            mask = view["valid_mask"].to(device=device, dtype=torch.bool)
            if mask.dim() == 4 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask[:, 0]
            masks.append(mask)
        return torch.stack(masks, dim=1)

    def _filter_depth_outliers(self, rendered_depth, target_depth, depth_valid):
        """Remove coarse rendered-depth outliers from depth supervision.

        Filtering is based on detached residuals, so rejected pixels do not produce
        gradients. The filtering is done per sample/view to avoid one bad view
        dominating the threshold for all views.
        """
        if not bool(depth_valid.any()):
            return depth_valid

        residual = (rendered_depth.detach().float() - target_depth.detach().float()).abs()
        filtered = depth_valid.clone()

        if self.depth_residual_max > 0:
            filtered = filtered & (residual <= self.depth_residual_max)

        if self.depth_keep_percent >= 100.0:
            return filtered

        keep_ratio = self.depth_keep_percent / 100.0
        b, v = filtered.shape[:2]

        for bi in range(b):
            for vi in range(v):
                mask = filtered[bi, vi]
                num_valid = int(mask.sum().item())
                if num_valid < self.min_depth_valid_pixels:
                    continue

                values = residual[bi, vi][mask]
                threshold = torch.quantile(values, keep_ratio).detach()
                filtered[bi, vi] = mask & (residual[bi, vi] <= threshold)

        return filtered

    def _depth_regression_loss(self, rendered_depth, target_depth, depth_valid):
        if not bool(depth_valid.any()):
            return rendered_depth.sum() * 0.0

        pred_values = rendered_depth[depth_valid].float()
        target_values = target_depth[depth_valid].float()

        if self.depth_loss_type == "mse":
            loss = F.mse_loss(pred_values, target_values)
        elif self.depth_loss_type == "l1":
            loss = F.l1_loss(pred_values, target_values)
        else:
            loss = F.smooth_l1_loss(
                pred_values,
                target_values,
                beta=self.depth_beta,
            )

        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    def _masked_mse(self, pred, target, valid_mask):
        delta = (pred - target).square()
        if valid_mask is None:
            return torch.nan_to_num(delta.mean(), nan=0.0, posinf=0.0, neginf=0.0)
        mask = valid_mask[:, :, None].expand_as(delta)
        if bool(mask.any()):
            return torch.nan_to_num(delta[mask].mean(), nan=0.0, posinf=0.0, neginf=0.0)
        return delta.sum() * 0.0

    def _lpips_loss(self, pred, target, valid_mask):
        if self.lpips is None or self.schedule_step < self.lpips_apply_after_step:
            return pred.sum() * 0.0
        pred_lpips = pred
        target_lpips = target
        if valid_mask is not None:
            mask = valid_mask[:, :, None].to(dtype=pred.dtype)
            pred_lpips = pred_lpips * mask
            target_lpips = target_lpips * mask
        b, v = pred_lpips.shape[:2]
        loss = self.lpips(
            pred_lpips.reshape(b * v, *pred_lpips.shape[2:]),
            target_lpips.reshape(b * v, *target_lpips.shape[2:]),
            normalize=True,
        )
        return torch.nan_to_num(loss.mean(), nan=0.0, posinf=0.0, neginf=0.0)

    def compute_loss(self, gt, pred):
        if not isinstance(pred, (list, tuple)) or len(pred) == 0:
            raise ValueError(
                "AnySplatOfficialRenderLoss expects pred to be a non-empty list."
            )
        if "gs_rendered_rgb" not in pred[0]:
            raise KeyError(
                "Missing pred[0]['gs_rendered_rgb']. "
                "Use a model that produces Gaussian-rendered outputs."
            )

        rendered_rgb = pred[0]["gs_rendered_rgb"].float()
        rendered_depth = pred[0].get("gs_rendered_depth", None)
        base_depth = pred[0].get("gs_base_depth", None)
        device = rendered_rgb.device
        target_rgb = torch.stack(
            [self._rgb01(view["img"].to(device)).float() for view in gt],
            dim=1,
        )
        valid_mask = (
            self._stack_optional_valid_mask(gt, device)
            if self.use_valid_mask
            else None
        )

        mse_loss = self._masked_mse(rendered_rgb, target_rgb, valid_mask)
        total = self.mse_weight * mse_loss
        details = {
            "AnySplatOfficialRenderLoss_mse": float(mse_loss.detach()),
        }

        if self.lpips_weight > 0:
            lpips_loss = self._lpips_loss(rendered_rgb, target_rgb, valid_mask)
            total = total + self.lpips_weight * lpips_loss
            details["AnySplatOfficialRenderLoss_lpips"] = float(
                lpips_loss.detach()
            )

        if self.depth_consis_weight > 0:
            if rendered_depth is None:
                raise KeyError("Missing pred[0]['gs_rendered_depth'].")
            if base_depth is None:
                raise KeyError("Missing pred[0]['gs_base_depth'].")

            base_depth = base_depth.to(device=device).float()
            if base_depth.dim() == 5 and base_depth.shape[-1] == 1:
                base_depth = base_depth[..., 0]
            if base_depth.dim() == 3 and rendered_depth.dim() == 4:
                base_depth = base_depth[:, None]
            if base_depth.dim() != rendered_depth.dim():
                raise ValueError(
                    "AnySplatOfficialRenderLoss expects pred[0]['gs_base_depth'] "
                    f"to have {rendered_depth.dim()} dims after squeezing, got "
                    f"shape {tuple(base_depth.shape)}"
                )

            if base_depth.shape != rendered_depth.shape:
                raise ValueError(
                    "AnySplatOfficialRenderLoss depth shape mismatch: "
                    f"rendered_depth={tuple(rendered_depth.shape)}, "
                    f"base_depth={tuple(base_depth.shape)}"
                )

            # Base validity: finite/positive base prediction + finite/positive render.
            depth_valid = (
                torch.isfinite(base_depth)
                & torch.isfinite(rendered_depth)
                & (base_depth > self.target_depth_min)
                & (rendered_depth > self.rendered_depth_min)
            )

            # Optional upper bound for rendered depth, in normalized GS depth space.
            # Use <=0 to disable.
            if self.rendered_depth_max > 0:
                depth_valid = depth_valid & (rendered_depth < self.rendered_depth_max)

            # Optional generic dataset valid mask.
            if valid_mask is not None:
                depth_valid = depth_valid & valid_mask

            # Optional alpha mask: do not supervise pixels not actually covered by splats.
            rendered_alpha = pred[0].get("gs_rendered_alpha", None)
            if rendered_alpha is not None and self.alpha_mask_min > 0:
                depth_valid = depth_valid & (
                    rendered_alpha.detach().float() > self.alpha_mask_min
                )

            depth_valid_before_outlier = depth_valid
            depth_valid = self._filter_depth_outliers(
                rendered_depth=rendered_depth,
                target_depth=base_depth,
                depth_valid=depth_valid,
            )

            depth_loss = self._depth_regression_loss(
                rendered_depth=rendered_depth,
                target_depth=base_depth.detach(),
                depth_valid=depth_valid,
            )

            total = total + self.depth_consis_weight * depth_loss

            pre_count = depth_valid_before_outlier.sum().clamp_min(1)
            post_count = depth_valid.sum()
            rejected_ratio = 1.0 - (post_count.float() / pre_count.float())

            details["AnySplatOfficialRenderLoss_depth_consis"] = float(depth_loss.detach())
            details["AnySplatOfficialRenderLoss_depth_valid_ratio"] = float(
                depth_valid.float().mean().detach()
            )
            details["AnySplatOfficialRenderLoss_depth_rejected_ratio"] = float(
                rejected_ratio.detach()
            )

            if bool(depth_valid.any()):
                details["AnySplatOfficialRenderLoss_rendered_depth_mean"] = float(
                    rendered_depth[depth_valid].detach().float().mean()
                )
                details["AnySplatOfficialRenderLoss_base_depth_mean"] = float(
                    base_depth[depth_valid].detach().float().mean()
                )

        details["AnySplatOfficialRenderLoss_total"] = float(total.detach())
        return total, details


class AnySplatOfficial2DGSRenderLoss(AnySplatOfficialRenderLoss):
    """2DGS render supervision with optional distortion and normal losses.

    This keeps the RGB and rendered-depth/base-depth losses from
    AnySplatOfficialRenderLoss, then adds the regularizers exposed by
    gsplat.rasterization_2dgs.
    """

    def __init__(
        self,
        dist_weight=0.0,
        normal_weight=0.0,
        normal_target_source="render_depth",
        normal_detach_target=False,
        normal_apply_alpha_mask=True,
        gt_depth_key="depthmap",
        normal_eps=1.0e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dist_weight = float(dist_weight)
        self.normal_weight = float(normal_weight)
        self.normal_target_source = str(normal_target_source)
        if self.normal_target_source not in {"render_depth", "gt_depth"}:
            raise ValueError(
                "normal_target_source must be one of "
                "['render_depth', 'gt_depth'], got "
                f"{self.normal_target_source!r}"
            )
        self.normal_detach_target = bool(normal_detach_target)
        self.normal_apply_alpha_mask = bool(normal_apply_alpha_mask)
        self.gt_depth_key = str(gt_depth_key)
        self.normal_eps = float(normal_eps)

    def get_name(self):
        return "AnySplatOfficial2DGSRenderLoss"

    @staticmethod
    def _ensure_bhw_depth(depth, name="depth"):
        if depth.dim() == 3:
            return depth
        if depth.dim() == 4 and depth.shape[-1] == 1:
            return depth[..., 0]
        if depth.dim() == 4 and depth.shape[1] == 1:
            return depth[:, 0]
        raise ValueError(
            f"Cannot convert {name} with shape {tuple(depth.shape)} to [B,H,W]."
        )

    @staticmethod
    def _ensure_bvchw(normals, name="normals"):
        if normals.dim() == 5 and normals.shape[2] == 3:
            return normals
        if normals.dim() == 5 and normals.shape[-1] == 3:
            return normals.permute(0, 1, 4, 2, 3)
        raise ValueError(
            f"Cannot convert {name} with shape {tuple(normals.shape)} "
            "to [B,V,3,H,W]."
        )

    def _stack_gt_depth(self, gt, device):
        depths = []
        for view_idx, view in enumerate(gt):
            if self.gt_depth_key not in view:
                raise KeyError(
                    "AnySplatOfficial2DGSRenderLoss normal_target_source='gt_depth' "
                    f"requires view[{view_idx}]['{self.gt_depth_key}']."
                )
            depth = self._ensure_bhw_depth(
                view[self.gt_depth_key].to(device=device).float(),
                name=f"view[{view_idx}]['{self.gt_depth_key}']",
            )
            depths.append(depth)
        return torch.stack(depths, dim=1)

    def _normal_from_depth(self, depth, intrinsics_norm):
        """Estimate camera-space normals from z-depth.

        Args:
            depth: [B,V,H,W]
            intrinsics_norm: [B,V,3,3], with row 0 divided by W and row 1 by H.

        Returns:
            [B,V,3,H,W] normals, with zero border pixels.
        """
        b, v, h, w = depth.shape
        depth_flat = depth.reshape(b * v, h, w).float()
        intr_flat = intrinsics_norm.reshape(b * v, 3, 3).float()

        fu = intr_flat[:, 0, 0] * float(w)
        fv = intr_flat[:, 1, 1] * float(h)
        cu = intr_flat[:, 0, 2] * float(w)
        cv = intr_flat[:, 1, 2] * float(h)

        u = torch.arange(w, device=depth.device, dtype=depth_flat.dtype)
        vv = torch.arange(h, device=depth.device, dtype=depth_flat.dtype)
        u = u.view(1, 1, w).expand(b * v, h, w)
        vv = vv.view(1, h, 1).expand(b * v, h, w)

        x_cam = (u - cu[:, None, None] + 0.5) * depth_flat / fu[:, None, None]
        y_cam = (vv - cv[:, None, None] + 0.5) * depth_flat / fv[:, None, None]
        cam = torch.stack((x_cam, y_cam, depth_flat), dim=-1)

        normals = torch.zeros_like(cam)
        dx = cam[:, 2:, 1:-1] - cam[:, :-2, 1:-1]
        dy = cam[:, 1:-1, 2:] - cam[:, 1:-1, :-2]
        normals[:, 1:-1, 1:-1] = F.normalize(
            torch.cross(dx, dy, dim=-1),
            dim=-1,
            eps=self.normal_eps,
        )
        return normals.reshape(b, v, h, w, 3).permute(0, 1, 4, 2, 3)

    def _distortion_loss(self, pred0, valid_mask):
        distortion = pred0.get("gs_2dgs_distortion", None)
        if distortion is None:
            raise KeyError(
                "AnySplatOfficial2DGSRenderLoss with dist_weight > 0 requires "
                "pred[0]['gs_2dgs_distortion']. Use gaussian_render_mode='2dgs'."
            )
        distortion = distortion.float()
        if distortion.dim() == 5 and distortion.shape[-1] == 1:
            distortion = distortion[..., 0]
        valid = torch.isfinite(distortion)
        if valid_mask is not None:
            valid = valid & valid_mask
        if not bool(valid.any()):
            return distortion.sum() * 0.0
        return torch.nan_to_num(
            distortion[valid].mean(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _normal_loss(self, gt, pred0, valid_mask):
        render_normals = pred0.get("gs_2dgs_render_normals", None)
        if render_normals is None:
            raise KeyError(
                "AnySplatOfficial2DGSRenderLoss with normal_weight > 0 requires "
                "pred[0]['gs_2dgs_render_normals']. Use gaussian_render_mode='2dgs'."
            )
        render_normals = self._ensure_bvchw(
            render_normals.float(),
            name="pred[0]['gs_2dgs_render_normals']",
        )

        if self.normal_target_source == "render_depth":
            target_normals = pred0.get("gs_2dgs_surf_normals", None)
            if target_normals is None:
                raise KeyError(
                    "normal_target_source='render_depth' requires "
                    "pred[0]['gs_2dgs_surf_normals'] from rasterization_2dgs."
                )
            target_normals = self._ensure_bvchw(
                target_normals.float(),
                name="pred[0]['gs_2dgs_surf_normals']",
            )
        else:
            intrinsics = pred0.get("gs_pred_intrinsics", None)
            if intrinsics is None:
                raise KeyError(
                    "normal_target_source='gt_depth' requires "
                    "pred[0]['gs_pred_intrinsics']."
                )
            c2w = pred0.get("gs_pred_c2w", None)
            if c2w is None:
                raise KeyError(
                    "normal_target_source='gt_depth' requires "
                    "pred[0]['gs_pred_c2w'] to transform depth normals to world space."
                )
            gt_depth = self._stack_gt_depth(gt, device=render_normals.device)
            depth_normal_valid = (
                torch.isfinite(gt_depth)
                & (gt_depth > self.target_depth_min)
            )
            target_normals = self._normal_from_depth(
                depth=gt_depth,
                intrinsics_norm=intrinsics.to(device=render_normals.device),
            )
            target_normals = torch.einsum(
                "bvij,bvjhw->bvihw",
                c2w.to(device=render_normals.device).float()[..., :3, :3],
                target_normals,
            )
            target_normals = torch.where(
                depth_normal_valid[:, :, None],
                target_normals,
                torch.zeros_like(target_normals),
            )

        if self.normal_detach_target:
            target_normals = target_normals.detach()

        normal_valid = (
            torch.isfinite(render_normals).all(dim=2)
            & torch.isfinite(target_normals).all(dim=2)
            & (render_normals.norm(dim=2) > self.normal_eps)
            & (target_normals.norm(dim=2) > self.normal_eps)
        )
        if valid_mask is not None:
            normal_valid = normal_valid & valid_mask

        rendered_alpha = pred0.get("gs_rendered_alpha", None)
        if rendered_alpha is not None:
            alpha = rendered_alpha.detach().float()
            if self.normal_apply_alpha_mask and self.alpha_mask_min > 0:
                normal_valid = normal_valid & (alpha > self.alpha_mask_min)
            if self.normal_target_source == "render_depth":
                target_normals = target_normals * alpha[:, :, None]

        if not bool(normal_valid.any()):
            return render_normals.sum() * 0.0

        dot = (render_normals * target_normals).sum(dim=2)
        normal_error = 1.0 - dot
        return torch.nan_to_num(
            normal_error[normal_valid].mean(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def compute_loss(self, gt, pred):
        total, details = super().compute_loss(gt, pred)
        pred0 = pred[0]
        device = pred0["gs_rendered_rgb"].device
        valid_mask = (
            self._stack_optional_valid_mask(gt, device)
            if self.use_valid_mask
            else None
        )

        if self.dist_weight > 0:
            dist_loss = self._distortion_loss(pred0, valid_mask)
            total = total + self.dist_weight * dist_loss
            details["AnySplatOfficial2DGSRenderLoss_dist"] = float(
                dist_loss.detach()
            )
            details["AnySplatOfficial2DGSRenderLoss_dist_weight"] = float(
                self.dist_weight
            )

        if self.normal_weight > 0:
            normal_loss = self._normal_loss(gt, pred0, valid_mask)
            total = total + self.normal_weight * normal_loss
            details["AnySplatOfficial2DGSRenderLoss_normal"] = float(
                normal_loss.detach()
            )
            details["AnySplatOfficial2DGSRenderLoss_normal_weight"] = float(
                self.normal_weight
            )

        details["AnySplatOfficial2DGSRenderLoss_total"] = float(total.detach())
        return total, details


class AnySplatRenderLoss(MultiLoss):
    """Photometric/depth regularization loss for Gaussian-rendering outputs.

    Required prediction keys are stored in preds[0]:
        gs_rendered_rgb:   [B,V,3,H,W]
        gs_rendered_depth: [B,V,H,W]
        gs_rendered_alpha: [B,V,H,W]
        gs_base_depth:     [B,V,H,W]
    """

    def __init__(
        self,
        rgb_loss_weight=1.0,
        depth_self_loss_weight=0.05,
        alpha_sparsity_weight=0.001,
        alpha_mask_loss_weight=0.0,
        charbonnier_eps=1.0e-3,
        use_depth_self_consistency=True,
    ):
        super().__init__()
        self.rgb_loss_weight = float(rgb_loss_weight)
        self.depth_self_loss_weight = float(depth_self_loss_weight)
        self.alpha_sparsity_weight = float(alpha_sparsity_weight)
        self.alpha_mask_loss_weight = float(alpha_mask_loss_weight)
        self.charbonnier_eps = float(charbonnier_eps)
        self.use_depth_self_consistency = bool(use_depth_self_consistency)

    def get_name(self):
        return "AnySplatRenderLoss"

    @staticmethod
    def _rgb01(img):
        if img.min().detach() < -0.01:
            return (img + 1.0) * 0.5
        return img.clamp(0.0, 1.0)

    def _charbonnier(self, x):
        return torch.sqrt(x * x + self.charbonnier_eps * self.charbonnier_eps)

    def compute_loss(self, gt, pred):
        if not isinstance(pred, (list, tuple)) or len(pred) == 0:
            raise ValueError(
                "AnySplatRenderLoss expects pred to be a non-empty list."
            )
        if "gs_rendered_rgb" not in pred[0]:
            raise KeyError(
                "Missing pred[0]['gs_rendered_rgb']. "
                "Use a model that produces Gaussian-rendered outputs."
            )

        rendered_rgb = pred[0]["gs_rendered_rgb"].float()
        rendered_depth = pred[0].get("gs_rendered_depth", None)
        rendered_alpha = pred[0].get("gs_rendered_alpha", None)
        base_depth = pred[0].get("gs_base_depth", None)

        target_rgb = torch.stack(
            [
                self._rgb01(view["img"].to(rendered_rgb.device)).float()
                for view in gt
            ],
            dim=1,
        )

        if rendered_rgb.shape != target_rgb.shape:
            raise ValueError(
                f"rendered_rgb shape {tuple(rendered_rgb.shape)} != "
                f"target_rgb shape {tuple(target_rgb.shape)}"
            )

        rgb_loss = self._charbonnier(rendered_rgb - target_rgb).mean()
        total = self.rgb_loss_weight * rgb_loss

        details = {
            "AnySplatRenderLoss_rgb": float(rgb_loss.detach()),
        }

        if (
            self.use_depth_self_consistency
            and rendered_depth is not None
            and base_depth is not None
            and self.depth_self_loss_weight > 0.0
        ):
            rd = rendered_depth.float()
            bd = base_depth.float()
            valid = torch.isfinite(rd) & torch.isfinite(bd) & (bd > 1.0e-6)
            if valid.any():
                depth_loss = self._charbonnier(rd[valid] - bd[valid]).mean()
            else:
                depth_loss = rd.sum() * 0.0
            total = total + self.depth_self_loss_weight * depth_loss
            details["AnySplatRenderLoss_depth_self"] = float(
                depth_loss.detach()
            )

        if rendered_alpha is not None and self.alpha_sparsity_weight > 0.0:
            alpha = rendered_alpha.float().clamp(0.0, 1.0)
            alpha_sparse = alpha.mean()
            total = total + self.alpha_sparsity_weight * alpha_sparse
            details["AnySplatRenderLoss_alpha_sparse"] = float(
                alpha_sparse.detach()
            )

        if rendered_alpha is not None and self.alpha_mask_loss_weight > 0.0:
            alpha = rendered_alpha.float().clamp(1.0e-4, 1.0 - 1.0e-4)
            alpha_target = torch.ones_like(alpha)
            alpha_mask_loss = F.binary_cross_entropy(alpha, alpha_target)
            total = total + self.alpha_mask_loss_weight * alpha_mask_loss
            details["AnySplatRenderLoss_alpha_mask"] = float(
                alpha_mask_loss.detach()
            )

        details["AnySplatRenderLoss_total"] = float(total.detach())
        return total, details
