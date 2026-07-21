# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MapAnything inference/training wrapper for the original VGGT-Omega."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from mapanything.models.external.vggt_omega.models.vggt_omega import VGGTOmega
from mapanything.models.external.vggt_omega.utils.geometry import (
    closed_form_inverse_se3,
)
from mapanything.models.external.vggt_omega.utils.pose_enc import (
    encoding_to_camera,
)
from mapanything.models.external.vggt_omega.utils.rotation import mat_to_quat
from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    convert_z_depth_to_depth_along_ray,
    depthmap_to_camera_frame,
    get_rays_in_camera_frame,
)


def _is_official_vggt_omega_state_dict(
    state_dict: Mapping[str, object],
) -> bool:
    return any(
        isinstance(key, str)
        and key.startswith(
            (
                "aggregator.",
                "camera_head.",
                "dense_head.",
                "text_alignment_head.",
            )
        )
        for key in state_dict.keys()
    )


def _to_float32_if_floating(value):
    if torch.is_tensor(value) and value.is_floating_point():
        return value.float()
    return value


class VGGTOmegaWrapper(torch.nn.Module):
    """Original VGGT-Omega exposed through the MapAnything output contract."""

    def __init__(
        self,
        name,
        torch_hub_force_reload=False,
        load_pretrained_weights=False,
        pretrained_model_name_or_path=None,
        checkpoint_path=None,
        load_custom_ckpt=False,
        custom_ckpt_path=None,
        strict_checkpoint=True,
        patch_size=16,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=False,
        gradient_checkpointing=False,
        frames_chunk_size=8,
    ):
        super().__init__()
        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload
        self.patch_size = int(patch_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.frames_chunk_size = frames_chunk_size

        if torch_hub_force_reload:
            print("[WARN] torch_hub_force_reload is unused for VGGT-Omega.")

        self.model = VGGTOmega(
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_camera=enable_camera,
            enable_depth=enable_depth,
            enable_alignment=enable_alignment,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        ckpt_path = checkpoint_path
        if load_custom_ckpt:
            ckpt_path = custom_ckpt_path
        elif load_pretrained_weights and ckpt_path is None:
            ckpt_path = pretrained_model_name_or_path

        if load_pretrained_weights and ckpt_path is None:
            raise ValueError(
                "VGGT-Omega load_pretrained_weights=True requires an "
                "official local checkpoint path."
            )
        if ckpt_path is not None:
            self.load_official_checkpoint(
                ckpt_path,
                strict=bool(strict_checkpoint),
            )

    @staticmethod
    def _validate_official_state_dict(
        state_dict,
        checkpoint_path="checkpoint",
    ):
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                "VGGT-Omega expects the official raw state_dict checkpoint, "
                f"got {type(state_dict)!r} from {checkpoint_path}."
            )
        if not _is_official_vggt_omega_state_dict(state_dict):
            raise ValueError(
                "Unsupported VGGT-Omega checkpoint format. Expected official "
                "raw state_dict keys such as 'aggregator.*', "
                "'camera_head.*', or 'dense_head.*'."
            )

    def load_official_checkpoint(self, checkpoint_path, strict=True):
        checkpoint_path = str(Path(checkpoint_path).expanduser())
        print(
            "Loading official VGGT-Omega checkpoint from "
            f"{checkpoint_path} ..."
        )
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self._validate_official_state_dict(
            state_dict,
            checkpoint_path=checkpoint_path,
        )
        print(self.model.load_state_dict(state_dict, strict=strict))
        del state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if _is_official_vggt_omega_state_dict(state_dict):
            try:
                return self.model.load_state_dict(
                    state_dict,
                    strict=strict,
                    assign=assign,
                )
            except TypeError:
                return self.model.load_state_dict(
                    state_dict,
                    strict=strict,
                )
        try:
            return super().load_state_dict(
                state_dict,
                strict=strict,
                assign=assign,
            )
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

    def _autocast_context(self, images):
        if images.device.type == "cuda":
            major = torch.cuda.get_device_capability(images.device)[0]
            dtype = torch.bfloat16 if major >= 8 else torch.float16
            return torch.autocast(device_type="cuda", dtype=dtype)
        return torch.autocast(
            device_type=images.device.type,
            enabled=False,
        )

    def _validate_input_size(self, height, width):
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "VGGT-Omega input image size must be divisible by "
                f"patch_size={self.patch_size}, got H={height}, W={width}."
            )

    def _run_vggt_omega_heads(self, images):
        with self._autocast_context(images):
            aggregated_tokens_list, patch_token_start = (
                self.model.aggregator(images)
            )

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError(
                "Aggregator did not cache the final layer."
            )

        predictions = {
            "camera_and_register_tokens": (
                final_tokens[:, :, :patch_token_start].contiguous()
            )
        }
        with torch.autocast(
            device_type=images.device.type,
            enabled=False,
        ):
            if self.model.camera_head is not None:
                predictions["pose_enc"] = self.model.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )
            if self.model.dense_head is not None:
                depth, depth_conf = self.model.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                    frames_chunk_size=self.frames_chunk_size,
                )
                predictions["depth"] = depth.float()
                predictions["depth_conf"] = depth_conf.float()
            if self.model.text_alignment_head is not None:
                predictions.update(
                    self.model.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )
        return predictions

    def forward(self, views):
        batch_size_per_view, _, height, width = views[0]["img"].shape
        num_views = len(views)
        self._validate_input_size(height, width)

        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "VGGT-Omega expects images in [0, 1] without DINO/VGGT "
            "mean-std normalization."
        )

        images = torch.stack([view["img"] for view in views], dim=1)
        predictions = self._run_vggt_omega_heads(images)

        if "pose_enc" not in predictions:
            raise RuntimeError(
                "VGGT-Omega wrapper requires enable_camera=True."
            )
        if "depth" not in predictions or "depth_conf" not in predictions:
            raise RuntimeError(
                "VGGT-Omega wrapper requires enable_depth=True."
            )

        with torch.autocast(
            device_type=images.device.type,
            enabled=False,
        ):
            pose_enc = predictions["pose_enc"].float()
            extrinsic, intrinsic = encoding_to_camera(
                pose_enc,
                images.shape[-2:],
                build_intrinsics=True,
            )
            extrinsic = extrinsic.float()
            intrinsic = intrinsic.float()
            depth_map = predictions["depth"].float()
            depth_conf = predictions["depth_conf"].float()
            camera_and_register_tokens = predictions.get(
                "camera_and_register_tokens"
            )
            text_alignment_embedding = predictions.get(
                "text_alignment_embedding"
            )
            text_alignment_token = predictions.get(
                "text_alignment_token"
            )

            results = []
            for view_idx in range(num_views):
                curr_extrinsic = closed_form_inverse_se3(
                    extrinsic[:, view_idx]
                ).float()
                curr_intrinsic = intrinsic[:, view_idx].float()
                curr_depth_z = depth_map[:, view_idx].squeeze(-1).float()
                curr_confidence = depth_conf[:, view_idx].float()
                curr_points_cam, _ = depthmap_to_camera_frame(
                    curr_depth_z,
                    curr_intrinsic,
                )
                curr_points_cam = curr_points_cam.float()
                curr_translation = curr_extrinsic[..., :3, 3].float()
                curr_quaternion = mat_to_quat(
                    curr_extrinsic[..., :3, :3]
                ).float()
                curr_depth_ray = convert_z_depth_to_depth_along_ray(
                    curr_depth_z,
                    curr_intrinsic,
                ).unsqueeze(-1).float()
                _, curr_rays = get_rays_in_camera_frame(
                    curr_intrinsic,
                    height,
                    width,
                    normalize_to_unit_sphere=True,
                )
                curr_rays = curr_rays.float()
                curr_points = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        curr_rays,
                        curr_depth_ray,
                        curr_translation,
                        curr_quaternion,
                    )
                ).float()
                curr_pose_enc = pose_enc[:, view_idx].float()
                curr_fov_xy = torch.stack(
                    [
                        curr_pose_enc[..., 8],
                        curr_pose_enc[..., 7],
                    ],
                    dim=-1,
                ).float()

                current = {
                    "pts3d": curr_points,
                    "pts3d_cam": curr_points_cam,
                    "ray_directions": curr_rays,
                    "intrinsics": curr_intrinsic,
                    "depth_along_ray": curr_depth_ray,
                    "cam_trans": curr_translation,
                    "cam_quats": curr_quaternion,
                    "conf": curr_confidence,
                    "pose_enc": curr_pose_enc,
                    "camera_translation": curr_translation,
                    "camera_quaternion": curr_quaternion,
                    "camera_fov_hw": curr_fov_xy,
                    "camera_pose_trans": curr_translation,
                    "camera_pose_quats": curr_quaternion,
                }
                if camera_and_register_tokens is not None:
                    current["camera_and_register_tokens"] = (
                        camera_and_register_tokens[:, view_idx]
                    )
                if text_alignment_embedding is not None:
                    current["text_alignment_embedding"] = (
                        text_alignment_embedding
                    )
                if text_alignment_token is not None:
                    current["text_alignment_token"] = text_alignment_token
                current = {
                    key: _to_float32_if_floating(value)
                    for key, value in current.items()
                }
                results.append(current)

        assert len(results) == num_views
        assert batch_size_per_view == results[0]["pts3d"].shape[0]
        return results
