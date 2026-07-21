"""World-frame augmentation for BaseDataset view dictionaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation


class WorldFrameAugmentDataset:
    """Wrap a BaseDataset and transform only fields returned by base_dataset.py."""

    # These are the distance/3D geometry fields that BaseDataset returns in the
    # camera frame. They are not recentered, translated, or world-rotated, but
    # they must follow the sampled scale because scale changes the metric unit.
    CAMERA_SCALE_KEYS = (
        "depthmap",
        "depth_along_ray",
        "prior_depth_along_ray",
        "pts3d_cam",
    )

    # These are the only world-frame geometry fields returned by BaseDataset.
    WORLD_POINT_KEYS = ("pts3d",)
    WORLD_POSE_KEY = "camera_pose"
    WORLD_POSE_QUAT_KEY = "camera_pose_quats"
    WORLD_POSE_TRANS_KEY = "camera_pose_trans"

    # BaseDataset returns these fields, but the world-frame augmentation should
    # not change them. This list is documentation for the full BaseDataset schema
    # handled by this wrapper; it is intentionally not used to mutate the view.
    UNCHANGED_KEYS = (
        "idx",
        "dataset",
        "label",
        "instance",
        "is_metric_scale",
        "is_synthetic",
        "camera_intrinsics",
        "img",
        "true_shape",
        "data_norm_type",
        "valid_mask",
        "ray_directions_cam",
        "non_ambiguous_mask",
        "rng",
    )

    def __init__(
        self,
        dataset,
        enabled: bool = True,
        z_rotation_deg: float = 0.0,
        translation_range: float | list[float] | tuple[float, ...] = 0.0,
        recenter: bool = False,
        recenter_mode: str = "first_camera",
        scale_range: float | list[float] | tuple[float, ...] = (0.9, 1.1),
        rotation_deg: None | float | list[float] | tuple[float, ...] = None,
        x_rotation_deg: None | float = None,
        y_rotation_deg: None | float = None,
    ):
        self.dataset = dataset
        self.enabled = bool(enabled)
        self.rotation_deg = self._parse_rotation_range(
            rotation_deg=rotation_deg,
            x_rotation_deg=x_rotation_deg,
            y_rotation_deg=y_rotation_deg,
            z_rotation_deg=z_rotation_deg,
        )
        self.x_rotation_deg = float(self.rotation_deg[0])
        self.y_rotation_deg = float(self.rotation_deg[1])
        self.z_rotation_deg = float(self.rotation_deg[2])
        self.translation_range = self._parse_translation_range(translation_range)
        self.recenter = bool(recenter)
        self.recenter_mode = str(recenter_mode)
        self.scale_range = self._parse_positive_range(scale_range, "scale_range")

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __repr__(self):
        return (
            f"WorldFrameAugmentDataset(enabled={self.enabled}, "
            f"rotation_deg={self.rotation_deg.tolist()}, "
            f"translation_range={self.translation_range.tolist()}, "
            f"recenter={self.recenter}, recenter_mode={self.recenter_mode}, "
            f"scale_range={self.scale_range.tolist()}, "
            f"dataset={repr(self.dataset)})"
        )

    @staticmethod
    def _parse_rotation_range(
        rotation_deg,
        x_rotation_deg=None,
        y_rotation_deg=None,
        z_rotation_deg=0.0,
    ):
        """Parse random rotation ranges as [x_deg, y_deg, z_deg].

        Backward compatibility:
        - If rotation_deg is None, x/y default to 0 and z uses z_rotation_deg.
        - If rotation_deg is provided, it takes precedence over x/y/z split keys.
        """
        if rotation_deg is None:
            arr = np.array(
                [
                    0.0 if x_rotation_deg is None else x_rotation_deg,
                    0.0 if y_rotation_deg is None else y_rotation_deg,
                    z_rotation_deg,
                ],
                dtype=np.float64,
            )
        elif isinstance(rotation_deg, (int, float)):
            arr = np.array([rotation_deg] * 3, dtype=np.float64)
        else:
            arr = np.asarray(rotation_deg, dtype=np.float64)
            if arr.shape != (3,):
                raise ValueError("rotation_deg must be scalar or [x, y, z]")
        if np.any(arr < 0):
            raise ValueError("rotation_deg values must be non-negative")
        return arr

    @staticmethod
    def _parse_translation_range(translation_range):
        if isinstance(translation_range, (int, float)):
            arr = np.array([translation_range] * 3, dtype=np.float64)
        else:
            arr = np.asarray(translation_range, dtype=np.float64)
            if arr.shape == (2,):
                arr = np.array([arr[0], arr[0], arr[1]], dtype=np.float64)
            if arr.shape != (3,):
                raise ValueError("translation_range must be scalar, [xy, z], or [x, y, z]")
        if np.any(arr < 0):
            raise ValueError("translation_range values must be non-negative")
        return arr

    @staticmethod
    def _parse_positive_range(value, name: str):
        if isinstance(value, (int, float)):
            arr = np.array([value, value], dtype=np.float64)
        else:
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != (2,):
                raise ValueError(f"{name} must be scalar or [min, max]")
        if np.any(arr <= 0):
            raise ValueError(f"{name} values must be positive")
        if arr[0] > arr[1]:
            raise ValueError(f"{name} min must be <= max")
        return arr

    def _rng(self):
        rng = getattr(self.dataset, "_rng", None)
        if rng is not None:
            return rng
        seed = int(torch.initial_seed() % (2**32))
        return np.random.default_rng(seed=seed)

    def _sample_rotation_and_translation(self, dtype=np.float32):
        rng = self._rng()
        if np.any(self.rotation_deg > 0):
            angles = rng.uniform(-self.rotation_deg, self.rotation_deg)
            rot = Rotation.from_euler("xyz", angles, degrees=True).as_matrix().astype(
                dtype, copy=False
            )
        else:
            rot = np.eye(3, dtype=dtype)

        local_trans = rng.uniform(-self.translation_range, self.translation_range).astype(dtype)
        if not np.any(self.translation_range > 0):
            local_trans[...] = 0
        return rot, local_trans

    def _sample_scale(self, dtype=np.float32):
        rng = self._rng()
        if np.allclose(self.scale_range[0], self.scale_range[1]):
            scale = self.scale_range[0]
        else:
            scale = rng.uniform(self.scale_range[0], self.scale_range[1])
        return np.asarray(scale, dtype=dtype)

    def _compute_anchor(self, views, dtype=np.float32):
        if not self.recenter:
            return np.zeros(3, dtype=dtype)

        if self.recenter_mode == "first_camera":
            return views[0][self.WORLD_POSE_KEY][:3, 3].astype(dtype, copy=False)

        if self.recenter_mode == "mean_camera":
            centers = [view[self.WORLD_POSE_KEY][:3, 3] for view in views]
            return np.stack(centers, axis=0).mean(axis=0).astype(dtype)

        if self.recenter_mode == "scene_points":
            pts = []
            for view in views:
                curr_pts = view["pts3d"]
                valid = np.isfinite(curr_pts).all(axis=-1) & view["valid_mask"].astype(bool)
                if valid.any():
                    pts.append(curr_pts[valid].reshape(-1, 3))
            if pts:
                return np.concatenate(pts, axis=0).mean(axis=0).astype(dtype)
            return views[0][self.WORLD_POSE_KEY][:3, 3].astype(dtype, copy=False)

        raise ValueError(
            "recenter_mode must be first_camera, mean_camera, or scene_points; "
            f"got {self.recenter_mode!r}"
        )

    @staticmethod
    def _transform_world_points(points, rot, local_trans, anchor, scale):
        return (((points - anchor + local_trans) * scale) @ rot.T).astype(
            points.dtype, copy=False
        )

    @staticmethod
    def _scale_camera_value(value, scale):
        return (value * scale).astype(value.dtype, copy=False)

    @staticmethod
    def _transform_pose(pose, rot, local_trans, anchor, scale):
        out = pose.copy()
        out[:3, :3] = (rot.astype(pose.dtype, copy=False) @ pose[:3, :3]).astype(
            pose.dtype, copy=False
        )
        out[:3, 3] = (((pose[:3, 3] - anchor + local_trans) * scale) @ rot.T).astype(
            pose.dtype, copy=False
        )
        return out

    def _augment_view_np(self, view, rot, local_trans, anchor, scale):
        view[self.WORLD_POSE_KEY] = self._transform_pose(
            view[self.WORLD_POSE_KEY], rot, local_trans, anchor, scale
        )
        view[self.WORLD_POSE_QUAT_KEY] = (
            Rotation.from_matrix(view[self.WORLD_POSE_KEY][:3, :3])
            .as_quat()
            .astype(view[self.WORLD_POSE_KEY].dtype)
        )
        view[self.WORLD_POSE_TRANS_KEY] = view[self.WORLD_POSE_KEY][:3, 3].astype(
            view[self.WORLD_POSE_KEY].dtype
        )

        for key in self.WORLD_POINT_KEYS:
            view[key] = self._transform_world_points(view[key], rot, local_trans, anchor, scale)

        for key in self.CAMERA_SCALE_KEYS:
            if key in view:
                view[key] = self._scale_camera_value(view[key], scale)

    def _augment_views(self, views):
        if not self.enabled:
            return views
        if (
            not np.any(self.rotation_deg > 0)
            and not np.any(self.translation_range > 0)
            and not self.recenter
            and np.allclose(self.scale_range, np.array([1.0, 1.0]))
        ):
            return views

        dtype = views[0][self.WORLD_POSE_KEY].dtype
        rot, local_trans = self._sample_rotation_and_translation(dtype=dtype)
        anchor = self._compute_anchor(views, dtype=dtype)
        scale = self._sample_scale(dtype=dtype)

        for view in views:
            self._augment_view_np(view, rot, local_trans, anchor=anchor, scale=scale)
        return views

    def __getitem__(self, idx):
        views = self.dataset[idx]
        return self._augment_views(views)
