# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Wrapper for Pi3X
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors

from mapanything.models.external.pi3.models.pi3x import Pi3X
from mapanything.models.external.vggt.utils.rotation import mat_to_quat


def _get_autocast_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16

def _stack_views(views: List[Dict], key: str, device: torch.device) -> torch.Tensor:
    return torch.stack([view[key].to(device) for view in views], dim=1)

def _safe_depth_to_hw(depth: torch.Tensor) -> torch.Tensor:
    if depth.dim() == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    return depth

def _all_have(views: List[Dict], key: str) -> bool:
    return all(key in view for view in views)


def _cfg_float(cfg, key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except Exception:
        return default


class Pi3XWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        geometric_input_config,
        pretrained_model_name_or_path="yyfz233/Pi3X",
        load_pretrained_weights: bool = True,
        use_conditioning: bool = True,
        use_multimodal: bool = True,
        gradient_checkpointing: bool = False,
        torch_hub_force_reload: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.geometric_input_config = geometric_input_config
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.load_pretrained_weights = load_pretrained_weights
        self.use_conditioning = use_conditioning
        self.use_multimodal = use_multimodal
        self.gradient_checkpointing = gradient_checkpointing
        self.torch_hub_force_reload = torch_hub_force_reload
        self.dtype = _get_autocast_dtype()

        if self.load_pretrained_weights:
            if not torch_hub_force_reload:
                print(f"Loading Pi3X from {pretrained_model_name_or_path} ...")
                self.model = Pi3X.from_pretrained(
                    pretrained_model_name_or_path,
                    gradient_checkpointing=gradient_checkpointing,
                )
            else:
                self.model = Pi3X.from_pretrained(
                    "yyfz233/Pi3X",
                    force_download=True,
                    gradient_checkpointing=gradient_checkpointing,
                )
        else:
            self.model = Pi3X(
                use_multimodal=self.use_multimodal,
                gradient_checkpointing=gradient_checkpointing,
            )

        self.dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

    @staticmethod
    def _convert_outputs(results: Dict[str, torch.Tensor], num_views: int):
        res = []
        for i in range(num_views):
            curr_view_extrinsic = results["camera_poses"][:, i, ...]
            curr_view_cam_translations = curr_view_extrinsic[..., :3, 3]
            curr_view_cam_quats = mat_to_quat(curr_view_extrinsic[..., :3, :3])

            curr_view_pts3d_cam = results["local_points"][:, i, ...]
            curr_view_depth_along_ray = torch.norm(
                curr_view_pts3d_cam, dim=-1, keepdim=True
            ).clamp_min(1e-8)
            curr_view_ray_dirs = curr_view_pts3d_cam / curr_view_depth_along_ray
            curr_view_pts3d = results["points"][:, i, ...]
            curr_view_confidence = results["conf"][:, i, ...]

            res.append(
                {
                    "pts3d": curr_view_pts3d,
                    "pts3d_cam": curr_view_pts3d_cam,
                    "ray_directions": curr_view_ray_dirs,
                    "depth_along_ray": curr_view_depth_along_ray,
                    "cam_trans": curr_view_cam_translations,
                    "cam_quats": curr_view_cam_quats,
                    "conf": curr_view_confidence,
                }
            )
        return res

    def forward(self, views: List[Dict]):
        device = views[0]["img"].device
        num_views = len(views)

        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", "Pi3X expects identity normalization."

        images = torch.stack([view["img"] for view in views], dim=1)

        cfg = self.geometric_input_config

        # ---- task probabilities are passed into Pi3X.encode(), not sampled here ----
        overall_prob = _cfg_float(cfg, "overall_prob", 1.0)
        ray_dirs_prob = _cfg_float(cfg, "ray_dirs_prob", 0.0)
        depth_prob = _cfg_float(cfg, "depth_prob", 0.0)
        cam_prob = _cfg_float(cfg, "cam_prob", 0.0)

        # images_only should really mean with_prior=False.
        # Other modes use with_prior=None so Pi3X internally samples masks from probabilities.
        has_nonzero_prior_prob = (
            self.use_conditioning
            and self.use_multimodal
            and overall_prob > 0.0
            and (ray_dirs_prob > 0.0 or depth_prob > 0.0 or cam_prob > 0.0)
        )
        with_prior = None if has_nonzero_prior_prob else False

        depths = None
        intrinsics = None
        poses = None
        pose_mask = None

        if has_nonzero_prior_prob and depth_prob > 0.0 and _all_have(views, "depthmap"):
            depths = torch.stack(
                [_safe_depth_to_hw(view["depthmap"].to(device)) for view in views],
                dim=1,
            )

        if has_nonzero_prior_prob and ray_dirs_prob > 0.0 and _all_have(views, "camera_intrinsics"):
            intrinsics = _stack_views(views, "camera_intrinsics", device=device)

        if has_nonzero_prior_prob and cam_prob > 0.0 and _all_have(views, "camera_pose"):
            poses = _stack_views(views, "camera_pose", device=device)
            if _all_have(views, "camera_pose_mask"):
                pose_mask = _stack_views(views, "camera_pose_mask", device=device).bool()

            # pose prior 在 Pi3X 里要求 rays/intrinsics 一起存在
            if intrinsics is None and _all_have(views, "camera_intrinsics"):
                intrinsics = _stack_views(views, "camera_intrinsics", device=device)

        model_kwargs = {
            "imgs": images,
            "with_prior": with_prior,
            "depths": depths,
            "intrinsics": intrinsics,
            "poses": poses,
            "pose_mask": pose_mask,
            "overall_prob": overall_prob,
            "ray_dirs_prob": ray_dirs_prob,
            "depth_prob": depth_prob,
            "cam_prob": cam_prob,
        }

        # Remove None tensors, but keep scalar probabilities.
        model_kwargs = {
            k: v for k, v in model_kwargs.items()
            if v is not None or k in {
                "with_prior",
                "overall_prob",
                "ray_dirs_prob",
                "depth_prob",
                "cam_prob",
            }
        }

        with torch.autocast("cuda", dtype=self.dtype, enabled=torch.cuda.is_available()):
            results = self.model(**model_kwargs)

        with torch.autocast("cuda", enabled=False):
            return self._convert_outputs(results, num_views)
