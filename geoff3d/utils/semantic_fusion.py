# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Semantic fusion utilities for projecting 2D semantic predictions to 3D point clouds.
"""

import torch
from typing import List, Dict, Optional, Tuple


def project_semantic_to_3d(
    semantic_outputs: List[Dict],
    pointcloud: torch.Tensor,
    fusion_method: str = "voting",
) -> torch.Tensor:
    """
    Project 2D semantic predictions from multiple views to 3D point cloud.

    Args:
        semantic_outputs: List of dicts, each containing:
            - "semantic": (1, num_classes, H, W) semantic logits
            - "depth_along_ray": (1, 1, H, W) depth values
            - "ray_directions": (1, H, W, 3) ray directions in camera frame
            - "camera_pose_quats": (1, 4) camera pose quaternion (cam2world)
            - "camera_pose_trans": (1, 3) camera translation
            - "metric_scaling_factor": scale factor for depth
        pointcloud: (N, 3) point cloud in world coordinates
        fusion_method: "voting", "average", or "weighted"

    Returns:
        (N,) semantic labels for each point
    """
    num_classes = semantic_outputs[0]["semantic"].shape[1]
    num_points = pointcloud.shape[0]
    device = pointcloud.device

    # Get predictions from all views
    all_view_predictions = []

    for view_output in semantic_outputs:
        # Get semantic logits and depth
        semantic_logits = view_output["semantic"]  # (1, num_classes, H, W)
        depth = view_output["depth_along_ray"] * view_output["metric_scaling_factor"].view(1, 1, 1, 1)
        ray_dirs = view_output["ray_directions"]  # (1, H, W, 3)

        # Get camera pose
        cam_rot = view_output["camera_pose_quats"]  # (1, 4)
        cam_trans = view_output["camera_pose_trans"]  # (1, 3)

        # Convert quaternion to rotation matrix
        R = quat_to_rotmat(cam_rot[0])  # (3, 3)
        t = cam_trans[0]  # (3,)

        # Project 3D points to 2D image plane
        # Transform world points to camera frame
        points_cam = (R @ (pointcloud - t).T).T  # (N, 3)

        # Project to image plane (assuming pinhole camera)
        # Use focal lengths from ray directions
        fx = fy = 1.0 / ray_dirs[0, 0, 0, 0].abs() if ray_dirs.numel() > 0 else 1.0
        cx = cy = 0.0

        # Image coordinates
        u = (points_cam[:, 0] * fx / points_cam[:, 2] + cx).long()
        v = (points_cam[:, 1] * fy / points_cam[:, 2] + cy).long()

        # Get image dimensions
        H, W = depth.shape[2:]

        # Valid points (in front of camera and within image bounds)
        valid_mask = (
            (points_cam[:, 2] > 0) &
            (u >= 0) & (u < W) &
            (v >= 0) & (v < H)
        )

        # Get semantic predictions for valid points
        view_predictions = torch.full(
            (num_points, num_classes),
            float('-inf'),
            device=device
        )

        if valid_mask.sum() > 0:
            valid_u = u[valid_mask]
            valid_v = v[valid_mask]
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            # Get depth values at valid pixel locations
            valid_depth = depth[0, 0, valid_v, valid_u]  # (M,)

            # Get semantic predictions
            valid_semantic = semantic_logits[0, :, valid_v, valid_u].T  # (M, num_classes)

            # Filter by depth consistency (point depth should match rendered depth)
            depth_diff = torch.abs(points_cam[valid_mask, 2] - valid_depth)
            depth_threshold = 0.1  # 10cm tolerance
            depth_consistent = depth_diff < depth_threshold

            # Update predictions for depth-consistent points
            for i, (idx, consistent) in enumerate(zip(valid_indices.tolist(), depth_consistent.tolist())):
                if consistent:
                    view_predictions[idx] = valid_semantic[i]

        all_view_predictions.append(view_predictions)

    # Fuse predictions from all views
    if fusion_method == "voting":
        # Majority voting
        final_predictions = torch.zeros(num_points, dtype=torch.long, device=device)

        for i in range(num_points):
            votes = []
            for view_pred in all_view_predictions:
                if view_pred[i].isfinite().all():
                    votes.append(view_pred[i].argmax().item())

            if votes:
                final_predictions[i] = max(set(votes), key=votes.count)
            else:
                final_predictions[i] = 0  # Default class

    elif fusion_method == "average":
        # Average probabilities
        stacked = torch.stack(all_view_predictions, dim=0)  # (V, N, C)
        # Replace -inf with 0 for averaging
        stacked = torch.where(
            stacked == float('-inf'),
            torch.zeros_like(stacked),
            torch.exp(stacked)  # softmax
        )
        avg_probs = stacked.mean(dim=0)  # (N, C)
        final_predictions = avg_probs.argmax(dim=1)

    elif fusion_method == "weighted":
        # Weighted by confidence (use entropy as confidence measure)
        final_predictions = torch.zeros(num_points, dtype=torch.long, device=device)

        for i in range(num_points):
            weights = []
            class_votes = []

            for view_pred in all_view_predictions:
                if view_pred[i].isfinite().all():
                    probs = torch.softmax(view_pred[i], dim=0)
                    # Weight = 1 - entropy
                    entropy = -(probs * torch.log(probs + 1e-8)).sum()
                    weight = 1.0 - entropy / torch.log(torch.tensor(num_classes))
                    weights.append(weight.item())
                    class_votes.append(view_pred[i].argmax().item())

            if weights:
                # Weighted voting
                class_weights = {}
                for cls, w in zip(class_votes, weights):
                    class_weights[cls] = class_weights.get(cls, 0) + w
                final_predictions[i] = max(class_weights.keys(), key=lambda x: class_weights[x])
            else:
                final_predictions[i] = 0

    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")

    return final_predictions


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.

    Args:
        quat: (4,) quaternion (w, x, y, z) or (x, y, z, w)

    Returns:
        (3, 3) rotation matrix
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    R = torch.zeros(3, 3, device=quat.device, dtype=quat.dtype)

    R[0, 0] = 1 - 2 * (y * y + z * z)
    R[0, 1] = 2 * (x * y - w * z)
    R[0, 2] = 2 * (x * z + w * y)

    R[1, 0] = 2 * (x * y + w * z)
    R[1, 1] = 1 - 2 * (x * x + z * z)
    R[1, 2] = 2 * (y * z - w * x)

    R[2, 0] = 2 * (x * z - w * y)
    R[2, 1] = 2 * (y * z + w * x)
    R[2, 2] = 1 - 2 * (x * x + y * y)

    return R


def compute_point_to_image_mapping(
    pointcloud: torch.Tensor,
    camera_poses: List[torch.Tensor],
    intrinsics: torch.Tensor,
    image_size: Tuple[int, int],
) -> List[torch.Tensor]:
    """
    Compute mapping from 3D points to 2D image coordinates for each view.

    Args:
        pointcloud: (N, 3) point cloud in world coordinates
        camera_poses: List of (4, 4) camera pose matrices (cam2world)
        intrinsics: (3, 3) camera intrinsic matrix
        image_size: (H, W) image size

    Returns:
        List of (N,) tensors containing flattened image indices for each view
        (-1 for points not visible in the view)
    """
    N = pointcloud.shape[0]
    H, W = image_size
    device = pointcloud.device

    fx, fy = intrinsics[0, 0].item(), intrinsics[1, 1].item()
    cx, cy = intrinsics[0, 2].item(), intrinsics[1, 2].item()

    mappings = []

    for pose in camera_poses:
        # Transform points to camera frame
        R = pose[:3, :3]
        t = pose[:3, 3]

        points_cam = (R @ pointcloud.T).T + t  # (N, 3)

        # Project to image plane
        u = (points_cam[:, 0] * fx / points_cam[:, 2] + cx).round().long()
        v = (points_cam[:, 1] * fy / points_cam[:, 2] + cy).round().long()

        # Valid pixels
        valid = (
            (points_cam[:, 2] > 0) &
            (u >= 0) & (u < W) &
            (v >= 0) & (v < H)
        )

        # Flattened indices (-1 for invalid)
        indices = torch.full((N,), -1, dtype=torch.long, device=device)
        indices[valid] = valid.nonzero(as_tuple=True)[0] * W + u[valid] + v[valid] * W

        mappings.append(indices)

    return mappings
