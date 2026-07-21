# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for HunyuanWorld-Mirror
"""

import torch
from hunyuanworld_mirror.models.models.worldmirror import WorldMirror

from mapanything.models.external.vggt.utils.geometry import closed_form_inverse_se3
from mapanything.models.external.vggt.utils.rotation import mat_to_quat
from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    convert_z_depth_to_depth_along_ray,
    depthmap_to_camera_frame,
    get_rays_in_camera_frame,
)


class HunyuanWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        torch_hub_force_reload,
        hf_model_name="tencent/HunyuanWorld-Mirror",
        geometric_input_config=None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload
        self.hf_model_name = hf_model_name
        self.geometric_input_config = geometric_input_config or {
            "overall_prob": 0.0,
            "ray_dirs_prob": 0.0,
            "depth_prob": 0.0,
            "cam_prob": 0.0,
        }
        # Load pre-trained weights
        if not torch_hub_force_reload:
            # Initialize the HunyuanWorld-Mirror model from huggingface hub cache
            print(f"Loading {hf_model_name} from huggingface cache ...")
            self.model = WorldMirror.from_pretrained(self.hf_model_name)
        else:
            # Initialize the HunyuanWorld-Mirror model with force download
            self.model = WorldMirror.from_pretrained(
                self.hf_model_name, force_download=True
            )
        
        self.model.enable_gs = False
        self.model.enable_norm = False

        # Get the dtype for HunyuanWorld-Mirror inference
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

    def forward(self, views):
        """
        Forward pass wrapper for HunyuanWorld-Mirror

        Assumption:
        - All the input views have the same image shape.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).

        Returns:
            List[dict]: A list containing the final outputs for all N views.
        """
        # Get input shape of the images, number of views, and batch size per view
        batch_size_per_view, _, height, width = views[0]["img"].shape
        device = views[0]["img"].device
        num_views = len(views)

        # Check the data norm type
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "HunyuanWorld-Mirror expects identity normalization for the input images"
        )

        # Prepare inputs for HunyuanWorld-Mirror
        inputs = {}
        
        # Stack images to create a (B, V, C, H, W) tensor
        img_list = [view["img"] for view in views]
        images = torch.stack(img_list, dim=1)  # (B, V, C, H, W)
        inputs['img'] = images

        # Handle geometric inputs if provided
        cond_flags = [0, 0, 0]
        if torch.rand(1, device=device) < self.geometric_input_config["overall_prob"]:
            # Handle camera poses
            if torch.rand(1, device=device) < self.geometric_input_config["cam_prob"]:
                pose_list = [view["camera_pose"] for view in views]
                camera_poses = torch.stack(pose_list, dim=1)  # (B, V, 4, 4)
                inputs['camera_poses'] = camera_poses
                cond_flags[0] = 1
            
            # Handle depth maps
            if torch.rand(1, device=device) < self.geometric_input_config["depth_prob"]:
                depth_list = [view["depthmap"] for view in views]
                depthmap = torch.stack(depth_list, dim=1)  # (B, V, H, W)
                inputs['depthmap'] = depthmap
                cond_flags[1] = 1
            
            # Handle camera intrinsics (ray directions)
            if torch.rand(1, device=device) < self.geometric_input_config["ray_dirs_prob"]:
                intrinsics_list = [view["camera_intrinsics"].to(device) for view in views]
                camera_intrs = torch.stack(intrinsics_list, dim=1)  # (B, V, 3, 3)
                inputs['camera_intrs'] = camera_intrs
                cond_flags[2] = 1

        # Run inference with autocast for mixed precision
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            predictions = self.model(inputs, cond_flags=cond_flags)

        # Process predictions to match MapAnything output format
        with torch.autocast(device_type='cuda', enabled=False):
            res = []
            for view_idx in range(num_views):
                curr_view_extrinsic = predictions["camera_poses"][:, view_idx, ...]  # (B, 4, 4)  Camera-to-world poses (OpenCV convention)
                curr_view_intrinsic = predictions["camera_intrs"][:, view_idx, ...]  # (B, 3, 3)

                curr_view_depth_z = predictions["depth"][:, view_idx, :, :, 0]  # (B, H, W)
                curr_view_confidence = predictions["depth_conf"][:, view_idx, :, :]  # (B, H, W)

                # Get the camera frame pointmaps
                curr_view_pts3d_cam, _ = depthmap_to_camera_frame(
                    curr_view_depth_z, curr_view_intrinsic
                )

                # Convert the extrinsics to quaternions and translations
                curr_view_cam_translations = curr_view_extrinsic[..., :3, 3]
                curr_view_cam_quats = mat_to_quat(curr_view_extrinsic[..., :3, :3])

                # Convert the z depth to depth along ray
                curr_view_depth_along_ray = convert_z_depth_to_depth_along_ray(
                    curr_view_depth_z, curr_view_intrinsic
                )
                curr_view_depth_along_ray = curr_view_depth_along_ray.unsqueeze(-1)

                # Get the ray directions on the unit sphere in the camera frame
                _, curr_view_ray_dirs = get_rays_in_camera_frame(
                    curr_view_intrinsic, height, width, normalize_to_unit_sphere=True
                )

                # Get the pointmaps
                curr_view_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        curr_view_ray_dirs,
                        curr_view_depth_along_ray,
                        curr_view_cam_translations,
                        curr_view_cam_quats,
                    )
                )

                # Append the outputs to the result list
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

    def train(self, mode=True):
        """Override train method to ensure model stays in eval mode for inference"""
        super().train(False)  # Always keep in eval mode
        return self

    def eval(self):
        """Override eval method"""
        super().eval()
        return self
