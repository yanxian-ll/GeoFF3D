# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Point cloud rendering utilities for generating multi-view depth/RGB images.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


def render_pointcloud_to_multi_view(
    pointcloud: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    poses: List[torch.Tensor] = None,
    intrinsics: Optional[torch.Tensor] = None,
    image_size: Tuple[int, int] = (512, 512),
    depth_trunc: float = 100.0,
) -> List[Dict]:
    """
    Render point cloud to multiple views.

    Args:
        pointcloud: (N, 3) point cloud in world coordinates
        colors: (N, 3) optional point colors (RGB, 0-1)
        poses: List of (4, 4) camera pose matrices (cam2world)
        intrinsics: (3, 3) or (V, 3, 3) camera intrinsic matrix
        image_size: (H, W) output image size
        depth_trunc: Maximum depth value

    Returns:
        List of view dicts with keys:
            - "depth_along_ray": (B, 1, H, W) depth map
            - "img": (B, 3, H, W) RGB image (if colors provided)
            - "ray_directions_cam": (B, H, W, 3) ray directions
            - "camera_pose_quats": (B, 4) camera pose quaternion
            - "camera_pose_trans": (B, 3) camera translation
    """
    device = pointcloud.device
    H, W = image_size

    # Handle single intrinsic or per-view intrinsics
    if intrinsics is None:
        # Default intrinsics (identity-like)
        fx = fy = max(H, W)
        cx, cy = W / 2, H / 2
        intrinsics = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], device=device)
        has_per_view_intrinsics = False
    elif intrinsics.dim() == 2:
        has_per_view_intrinsics = False
    else:
        has_per_view_intrinsics = True

    if poses is None:
        # Default: single front view
        poses = [torch.eye(4, device=device)]

    num_views = len(poses)
    views = []

    for view_idx in range(num_views):
        pose = poses[view_idx]

        # Get intrinsics for this view
        if has_per_view_intrinsics:
            K = intrinsics[view_idx]
        else:
            K = intrinsics

        # Render depth map
        depth_map = render_depth_map(
            pointcloud=pointcloud,
            pose=pose,
            K=K,
            image_size=image_size,
            depth_trunc=depth_trunc,
        )

        # Render ray directions
        ray_dirs = compute_ray_directions(
            K=K,
            image_size=image_size,
            device=device,
        )

        # Extract camera pose in the format expected by MapAnything
        cam_rot = rotmat_to_quat(pose[:3, :3])
        cam_trans = pose[:3, 3]

        view_dict = {
            "depth_along_ray": depth_map.unsqueeze(0),  # (1, 1, H, W)
            "ray_directions_cam": ray_dirs.unsqueeze(0),  # (1, H, W, 3)
            "camera_pose_quats": cam_rot.unsqueeze(0),  # (1, 4)
            "camera_pose_trans": cam_trans.unsqueeze(0),  # (1, 3)
        }

        # Render RGB if colors available
        if colors is not None:
            rgb_map = render_rgb_map(
                pointcloud=pointcloud,
                colors=colors,
                pose=pose,
                K=K,
                image_size=image_size,
            )
            view_dict["img"] = rgb_map.unsqueeze(0)  # (1, 3, H, W)
        else:
            # Create dummy RGB (all gray)
            rgb_map = torch.full((3, H, W), 0.5, device=device)
            view_dict["img"] = rgb_map.unsqueeze(0)

        views.append(view_dict)

    return views


def render_depth_map(
    pointcloud: torch.Tensor,
    pose: torch.Tensor,
    K: torch.Tensor,
    image_size: Tuple[int, int],
    depth_trunc: float = 100.0,
) -> torch.Tensor:
    """
    Render depth map from point cloud.

    Args:
        pointcloud: (N, 3) point cloud in world coordinates
        pose: (4, 4) camera pose (cam2world)
        K: (3, 3) camera intrinsic matrix
        image_size: (H, W) output image size
        depth_trunc: Maximum depth value

    Returns:
        (H, W) depth map
    """
    device = pointcloud.device
    H, W = image_size

    # Transform points to camera frame
    R = pose[:3, :3]
    t = pose[:3, 3]

    points_cam = (R @ pointcloud.T).T + t  # (N, 3)

    # Filter points in front of camera
    valid = points_cam[:, 2] > 0
    points_cam = points_cam[valid]

    if points_cam.shape[0] == 0:
        return torch.zeros(H, W, device=device)

    # Project to image plane
    x = points_cam[:, 0] / points_cam[:, 2]
    y = points_cam[:, 1] / points_cam[:, 2]

    fx, fy = K[0, 0].item(), K[1, 1].item()
    cx, cy = K[0, 2].item(), K[1, 2].item()

    u = (x * fx + cx).round().long()
    v = (y * fy + cy).round().long()

    # Filter points within image bounds
    valid = (
        (u >= 0) & (u < W) &
        (v >= 0) & (v < H) &
        (points_cam[:, 2] < depth_trunc)
    )

    u = u[valid]
    v = v[valid]
    depths = points_cam[valid, 2]

    # Create depth map (take closest depth for each pixel)
    depth_map = torch.full((H, W), float('inf'), device=device)
    depth_map[v, u] = torch.minimum(depth_map[v, u], depths)

    # Replace inf with 0
    depth_map = torch.where(
        depth_map == float('inf'),
        torch.zeros_like(depth_map),
        depth_map
    )

    return depth_map


def render_rgb_map(
    pointcloud: torch.Tensor,
    colors: torch.Tensor,
    pose: torch.Tensor,
    K: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Render RGB map from colored point cloud.

    Args:
        pointcloud: (N, 3) point cloud in world coordinates
        colors: (N, 3) point colors (RGB, 0-1)
        pose: (4, 4) camera pose (cam2world)
        K: (3, 3) camera intrinsic matrix
        image_size: (H, W) output image size

    Returns:
        (3, H, W) RGB map
    """
    device = pointcloud.device
    H, W = image_size

    # Transform points to camera frame
    R = pose[:3, :3]
    t = pose[:3, 3]

    points_cam = (R @ pointcloud.T).T + t  # (N, 3)

    # Filter points in front of camera
    valid = points_cam[:, 2] > 0
    points_cam = points_cam[valid]
    colors = colors[valid]

    if points_cam.shape[0] == 0:
        return torch.full((3, H, W), 0.5, device=device)

    # Project to image plane
    x = points_cam[:, 0] / points_cam[:, 2]
    y = points_cam[:, 1] / points_cam[:, 2]

    fx, fy = K[0, 0].item(), K[1, 1].item()
    cx, cy = K[0, 2].item(), K[1, 2].item()

    u = (x * fx + cx).round().long()
    v = (y * fy + cy).round().long()

    # Filter points within image bounds
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid]
    v = v[valid]
    colors = colors[valid]

    # Create RGB map
    rgb_map = torch.full((3, H, W), 0.5, device=device)
    rgb_map[:, v, u] = colors.T

    return rgb_map


def compute_ray_directions(
    K: torch.Tensor,
    image_size: Tuple[int, int],
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Compute ray directions for each pixel in the image.

    Args:
        K: (3, 3) camera intrinsic matrix
        image_size: (H, W) image size
        device: torch device

    Returns:
        (H, W, 3) ray directions in camera frame
    """
    H, W = image_size
    fx, fy = K[0, 0].item(), K[1, 1].item()
    cx, cy = K[0, 2].item(), K[1, 2].item()

    # Create pixel grid
    v, u = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    # Normalize to -1, 1
    u = (u.float() - cx) / fx
    v = (v.float() - cy) / fy

    # Ray directions (normalized)
    directions = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # (H, W, 3)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    return directions


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion.

    Args:
        R: (3, 3) rotation matrix

    Returns:
        (4,) quaternion (w, x, y, z)
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return torch.tensor([w, x, y, z], device=R.device, dtype=R.dtype)


def create_default_camera_poses(
    num_views: int,
    radius: float = 2.0,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """
    Create default camera poses in a circular pattern.

    Args:
        num_views: Number of views
        radius: Distance from origin
        device: torch device

    Returns:
        List of (4, 4) camera pose matrices
    """
    poses = []

    for i in range(num_views):
        angle = 2 * np.pi * i / num_views

        # Camera position
        x = radius * np.cos(angle)
        y = 0
        z = radius * np.sin(angle)

        # Look at origin
        position = torch.tensor([x, y, z], dtype=torch.float32, device=device)
        target = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
        up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)

        pose = look_at(position, target, up)
        poses.append(pose)

    return poses


def look_at(
    position: torch.Tensor,
    target: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """
    Create a camera pose matrix using look-at transformation.

    Args:
        position: (3,) camera position
        target: (3,) target point
        up: (3,) up vector

    Returns:
        (4, 4) camera pose matrix (cam2world)
    """
    device = position.device
    dtype = position.dtype

    # Forward, right, up vectors
    forward = (target - position)
    forward = forward / forward.norm()

    right = torch.cross(forward, up)
    right = right / right.norm()

    up = torch.cross(right, forward)

    # Build transformation matrix
    pose = torch.eye(4, device=device, dtype=dtype)
    pose[0, :3] = right
    pose[1, :3] = up
    pose[2, :3] = -forward
    pose[:3, 3] = position

    return pose
