# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the Apache License, Version 2.0

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray

from mapanything.models.external.vggt.utils.geometry import closed_form_inverse_se3

logger = logging.getLogger(__name__)


class DA3Wrapper(nn.Module):
    """
    Trainable DA3 wrapper.

    """

    def __init__(
        self,
        name,
        torch_hub_force_reload,
        hf_model_name,
        geometric_input_config,
        use_conditioning=True,
        ref_view_strategy="saddle_balanced",
        gradient_checkpointing=False,
        gradient_checkpointing_use_reentrant=False,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.hf_model_name = hf_model_name
        self.geometric_input_config = geometric_input_config
        self.use_conditioning = use_conditioning
        self.ref_view_strategy = ref_view_strategy
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_use_reentrant = gradient_checkpointing_use_reentrant

        if torch_hub_force_reload:
            api = DepthAnything3.from_pretrained(
                self.hf_model_name, force_download=True
            )
        else:
            api = DepthAnything3.from_pretrained(self.hf_model_name)

        # Important:
        # Keep the whole API object so the saved checkpoint namespace becomes
        # `model.model.*`, matching the current inference wrapper.
        self.model = api

        # We will NEVER call self.model(...) in training, because the official
        # API forward is decorated with inference_mode / no_grad. We only use
        # the underlying trainable network at self.model.model.
        core_model = self.model.model

        if hasattr(core_model, "backbone") and core_model.backbone is not None:
            if hasattr(core_model.backbone.pretrained, "set_gradient_checkpointing"):
                core_model.backbone.pretrained.set_gradient_checkpointing(
                    enable=gradient_checkpointing,
                    use_reentrant=gradient_checkpointing_use_reentrant,
                )
            else:
                setattr(
                    core_model.backbone.pretrained,
                    "gradient_checkpointing",
                    gradient_checkpointing,
                )
                setattr(
                    core_model.backbone.pretrained,
                    "gradient_checkpointing_use_reentrant",
                    gradient_checkpointing_use_reentrant,
                )

        if not hasattr(core_model, "cam_dec") or core_model.cam_dec is None:
            raise RuntimeError(
                "DA3 camera decoder is required, but cam_dec is missing in the loaded checkpoint."
            )

    @staticmethod
    def _normalize_extrinsics_batch(ex_t: torch.Tensor | None):
        """
        Batch training version of official API `_normalize_extrinsics`.
        """
        if ex_t is None:
            return None, None

        transform = affine_inverse(ex_t[:, :1])
        ex_t_norm = ex_t @ transform

        c2ws = affine_inverse(ex_t_norm)
        translations = c2ws[..., :3, 3]
        dists = translations.norm(dim=-1)
        pose_scale = torch.median(dists, dim=1).values.clamp(min=1e-1)

        ex_t_norm = ex_t_norm.clone()
        ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / pose_scale[:, None, None]
        return ex_t_norm, pose_scale

    def _build_conditions(self, views, device):
        intrinsics = None
        extrinsics = None

        if not self.use_conditioning:
            return extrinsics, intrinsics

        if torch.rand(1, device=device) >= self.geometric_input_config["overall_prob"]:
            return extrinsics, intrinsics

        has_intrinsics = all("camera_intrinsics" in v for v in views)
        has_pose = all("camera_pose" in v for v in views)

        if has_intrinsics and (
            torch.rand(1, device=device) < self.geometric_input_config["ray_dirs_prob"]
        ):
            intrinsics = torch.stack(
                [v["camera_intrinsics"].to(device) for v in views], dim=1
            )

        if has_pose and (
            torch.rand(1, device=device) < self.geometric_input_config["cam_prob"]
        ):
            poses_w2c = torch.stack(
                [closed_form_inverse_se3(v["camera_pose"].to(device)) for v in views],
                dim=1,
            )
            extrinsics, _ = self._normalize_extrinsics_batch(poses_w2c)

        return extrinsics, intrinsics

    @staticmethod
    def _intrinsics_to_fov(intrinsics, height, width):
        fx = intrinsics[..., 0, 0].clamp_min(1e-6)
        fy = intrinsics[..., 1, 1].clamp_min(1e-6)
        width_t = torch.tensor(width, device=intrinsics.device, dtype=intrinsics.dtype)
        height_t = torch.tensor(height, device=intrinsics.device, dtype=intrinsics.dtype)
        fov_x = 2.0 * torch.atan(width_t / (2.0 * fx))
        fov_y = 2.0 * torch.atan(height_t / (2.0 * fy))
        return torch.stack([fov_x, fov_y], dim=-1)

    @staticmethod
    def _resize_ray_to(ray: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        """
        ray:
            [B,N,H,W,6] or [B,H,W,6]
        return:
            same rank, resized to target_hw
        """
        th, tw = target_hw

        if ray.dim() == 5:
            b, n, h, w, c = ray.shape
            if (h, w) == (th, tw):
                return ray
            ray_nchw = ray.permute(0, 1, 4, 2, 3).reshape(b * n, c, h, w)
            ray_nchw = F.interpolate(
                ray_nchw, size=(th, tw), mode="bilinear", align_corners=False
            )
            return ray_nchw.reshape(b, n, c, th, tw).permute(0, 1, 3, 4, 2)

        if ray.dim() == 4:
            b, h, w, c = ray.shape
            if (h, w) == (th, tw):
                return ray
            ray_nchw = ray.permute(0, 3, 1, 2)
            ray_nchw = F.interpolate(
                ray_nchw, size=(th, tw), mode="bilinear", align_corners=False
            )
            return ray_nchw.permute(0, 2, 3, 1)

        raise ValueError(f"Unexpected ray shape: {tuple(ray.shape)}")

    @staticmethod
    def _to_4x4(mat: torch.Tensor) -> torch.Tensor:
        """
        Convert [B,3,4] / [B,4,4] / [B,N,3,4] / [B,N,4,4] to full homogeneous 4x4.
        """
        if mat.shape[-2:] == (4, 4):
            return mat
        if mat.shape[-2:] != (3, 4):
            raise ValueError(f"Unexpected extrinsic shape: {tuple(mat.shape)}")

        out = torch.zeros(*mat.shape[:-2], 4, 4, device=mat.device, dtype=mat.dtype)
        out[..., :3, :] = mat
        out[..., 3, 3] = 1.0
        return out

    @staticmethod
    def _backproject_depth_z_to_world(
        depth_z: torch.Tensor,
        intrinsics: torch.Tensor,
        c2w: torch.Tensor,
    ) -> torch.Tensor:
        """
        Back-project z-depth to world-space 3D points using intrinsics + c2w.

        depth_z: [B,H,W] or [B,H,W,1]
        intrinsics: [B,3,3]
        c2w: [B,4,4] or [B,3,4]
        return: [B,H,W,3]
        """
        if depth_z.dim() == 4 and depth_z.shape[-1] == 1:
            depth_z = depth_z[..., 0]
        elif depth_z.dim() != 3:
            raise ValueError(f"Unexpected depth_z shape: {tuple(depth_z.shape)}")

        c2w = DA3Wrapper._to_4x4(c2w)

        b, h, w = depth_z.shape
        device = depth_z.device
        dtype = depth_z.dtype

        ys = torch.arange(h, device=device, dtype=dtype)
        xs = torch.arange(w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        xx = xx.unsqueeze(0).expand(b, -1, -1)
        yy = yy.unsqueeze(0).expand(b, -1, -1)

        fx = intrinsics[:, 0, 0].view(b, 1, 1).clamp_min(1e-6)
        fy = intrinsics[:, 1, 1].view(b, 1, 1).clamp_min(1e-6)
        cx = intrinsics[:, 0, 2].view(b, 1, 1)
        cy = intrinsics[:, 1, 2].view(b, 1, 1)

        x_cam = (xx - cx) / fx * depth_z
        y_cam = (yy - cy) / fy * depth_z
        z_cam = depth_z
        pts3d_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)

        rot = c2w[:, :3, :3]
        trans = c2w[:, :3, 3]
        pts3d_world = torch.einsum("bij,bhwj->bhwi", rot, pts3d_cam) + trans[:, None, None, :]
        return pts3d_world

    def forward(self, views):
        _, _, height, width = views[0]["img"].shape
        device = views[0]["img"].device
        num_views = len(views)

        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "dinov2", "DA3 expects DINOv2 normalization"

        images = torch.stack([v["img"] for v in views], dim=1)
        extrinsics, intrinsics = self._build_conditions(views, device=device)

        # Do NOT call self.model(...) here. That would enter the official API
        # forward, which is inference-only. We only use the underlying trainable
        # network at self.model.model.
        core_model = self.model.model

        if extrinsics is not None or intrinsics is not None:
            with torch.autocast(device_type=images.device.type, enabled=False):
                cam_token = core_model.cam_enc(extrinsics, intrinsics, images.shape[-2:])
        else:
            cam_token = None
        
        feats, _ = core_model.backbone(
            images,
            cam_token=cam_token,
            export_feat_layers=[],
            ref_view_strategy=self.ref_view_strategy,
        )

        with torch.autocast(device_type=images.device.type, enabled=False):
            output = core_model._process_depth_head(feats, height, width)
            pred_c2w_cam, pred_intrinsics_cam = pose_encoding_to_extri_intri(
                core_model.cam_dec(feats[-1][1]),
                (height, width),
            )
            pred_c2w_cam = self._to_4x4(pred_c2w_cam)
            pred_extrinsics_cam = affine_inverse(pred_c2w_cam)  # w2c
            pred_fov_cam = self._intrinsics_to_fov(pred_intrinsics_cam, height, width)

            if hasattr(core_model, "_process_mono_sky_estimation"):
                output = core_model._process_mono_sky_estimation(output)

        # DA3 official `output["depth"]` is z-depth in the model camera space,
        # not Euclidean distance along a unit ray.
        pred_depth = output["depth"]
        pred_depth_conf = output.get("depth_conf", None)
        pred_ray = output["ray"]

        depth_h, depth_w = pred_depth.shape[-2], pred_depth.shape[-1]
        pred_ray = self._resize_ray_to(pred_ray, (depth_h, depth_w))  # [B, N, H, W, 6]
        # pred_ray_conf = output.get("ray_conf", None)  # [B, N, H, W]
        # if pred_ray_conf is not None:
        #     pred_ray_conf = F.interpolate(
        #         pred_ray_conf, size=(depth_h, depth_w), mode="nearest"
        #     )  # [B, N, H, W]

        # with torch.autocast(device_type=images.device.type, enabled=False):
        #     pred_c2w_ray, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
        #         output["ray"],
        #         output["ray_conf"],
        #         output["ray"].shape[-3],
        #         output["ray"].shape[-2],
        #         training=True,
        #     )
        #     pred_extrinsics_ray_w2c = affine_inverse(pred_c2w_ray)

        #     pred_intrinsics_ray = torch.eye(3, 3, device=pred_c2w_ray.device, dtype=pred_c2w_ray.dtype)[None, None].repeat(pred_c2w_ray.shape[0], pred_c2w_ray.shape[1], 1, 1).clone()
        #     pred_intrinsics_ray[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * width
        #     pred_intrinsics_ray[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * height
        #     pred_intrinsics_ray[:, :, 0, 2] = pred_principal_points[:, :, 0] * width * 0.5
        #     pred_intrinsics_ray[:, :, 1, 2] = pred_principal_points[:, :, 1] * height * 0.5
        #     pred_fov_ray = self._intrinsics_to_fov(pred_intrinsics_ray, height, width)

        results = []
        for i in range(num_views):
            # DA3 depth is z-depth, while DA3 ray directions are intentionally
            # left unnormalized to preserve projection scale. Therefore:
            #   pts3d = ray_origins + depth_z * ray_directions
            # is the correct reconstruction formula for the raw DA3 outputs.
            depth_z = pred_depth[:, i]
            if depth_z.dim() == 3:
                depth_z = depth_z.unsqueeze(-1)

            ray_map = pred_ray[:, i]
            ray_origins = ray_map[..., 3:]
            ray_directions = ray_map[..., :3]

            # ray_direction_norm = torch.linalg.norm(ray_directions, dim=-1, keepdim=True).clamp_min(1e-8)
            # ray_directions_unit = ray_directions / ray_direction_norm

            # # True Euclidean distance along the unit ray, provided only for
            # # compatibility with downstream code that explicitly expects it.
            # depth_along_ray = depth_z * ray_direction_norm

            # Correct DA3 point reconstruction from raw outputs.
            pts3d = ray_origins + depth_z * ray_directions
            # pts3d_ = pts3d[0].reshape(-1, 3)   # For ViewPLY Debug

            # # Debug-only 3D reconstructions from the two predicted camera solutions.
            # # They are intentionally computed but not returned.
            # pts3d_from_cam_dec = self._backproject_depth_z_to_world(
            #     depth_z,
            #     pred_intrinsics_cam[:, i],
            #     pred_c2w_cam[:, i],
            # )
            # pts3d_from_cam_dec_ = pts3d_from_cam_dec[0].reshape(-1, 3)  # For ViewPLY Debug

            # pts3d_from_ray_calib = self._backproject_depth_z_to_world(
            #     depth_z,
            #     pred_intrinsics_ray[:, i],
            #     pred_c2w_ray[:, i],
            # )
            # pts3d_from_ray_calib_ = pts3d_from_ray_calib[0].reshape(-1, 3)  # For ViewPLY Debug

            out = {
                "depth_z": depth_z,
                "ray_map": ray_map,
                "ray_origins": ray_origins,
                "ray_directions": ray_directions,
                "pts3d": pts3d,
                # Keep legacy keys for downstream compatibility.
                "pred_extrinsics": pred_extrinsics_cam[:, i],
                "pred_intrinsics": pred_intrinsics_cam[:, i],
                "pred_fov": pred_fov_cam[:, i],
                # # Explicitly return both camera solutions.
                # "pred_extrinsics_cam": pred_extrinsics_cam[:, i],
                # "pred_intrinsics_cam": pred_intrinsics_cam[:, i],
                # "pred_c2w_cam": pred_c2w_cam[:, i],
                # "pred_fov_cam": pred_fov_cam[:, i],
                # "pred_extrinsics_ray": pred_extrinsics_ray_w2c[:, i],
                # "pred_intrinsics_ray": pred_intrinsics_ray[:, i],
                # "pred_c2w_ray": pred_c2w_ray[:, i],
                # "pred_fov_ray": pred_fov_ray[:, i],
            }

            if pred_depth_conf is not None:
                out["conf"] = pred_depth_conf[:, i]

            # if pred_ray_conf is not None:
            #     out["ray_conf"] = pred_ray_conf[:, i]

            if "sky" in output:
                out["sky_logits"] = output["sky"][:, i]
            if "obj" in output:
                out["object_logits"] = output["obj"][:, i]

            results.append(out)

        return results
