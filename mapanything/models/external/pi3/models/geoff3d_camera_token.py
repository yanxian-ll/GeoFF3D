from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mapanything.models.external.pi3.models.geoff3d import (
    GeoFF3D,
    homogenize_points,
)


class CameraTokenHead(nn.Module):
    """Predict an OpenCV c2w pose directly from one decoded camera token."""

    def __init__(self, dim=512):
        super().__init__()
        self.more_mlps = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.fc_t = nn.Linear(dim, 3)
        self.fc_rot = nn.Linear(dim, 9)

    def forward(self, token):
        if token.dim() == 3:
            if token.shape[1] != 1:
                raise ValueError(
                    f"CameraTokenHead expects one token, got shape {tuple(token.shape)}"
                )
            token = token[:, 0]
        if token.dim() != 2:
            raise ValueError(
                f"CameraTokenHead expects [B,C] or [B,1,C], got {tuple(token.shape)}"
            )
        feat = self.more_mlps(token)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            out_t = self.fc_t(feat.float())
            out_r = self.fc_rot(feat.float())
            pose = self.convert_pose_to_4x4(token.shape[0], out_r, out_t, token.device)
        return pose

    def convert_pose_to_4x4(self, batch_size, out_r, out_t, device):
        out_r = self.svd_orthogonalize(out_r)
        pose = torch.zeros((batch_size, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.0
        return pose

    @staticmethod
    def svd_orthogonalize(m):
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(
            torch.nn.functional.normalize(m, p=2, dim=-1),
            dim0=-1,
            dim1=-2,
        )
        u, _, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1),
        )
        return r


class GeoFF3DCameraToken(GeoFF3D):
    """GeoFF3D variant with an explicit VGGT-style camera token.

    World translation and world rotation priors are injected only into the
    camera token. The pose branch predicts from that decoded camera token
    directly, while point/conf heads continue to use image patch tokens.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_register_tokens = self.patch_start_idx
        self.patch_start_idx = 1 + self.num_register_tokens
        self.camera_token = nn.Parameter(
            torch.randn(1, 1, 1, self.dec_embed_dim)
        )
        nn.init.normal_(self.camera_token, std=1e-6)
        self.camera_head = CameraTokenHead(dim=512)

    def forward(
        self,
        imgs,
        depths=None,
        depth_mask=None,
        intrinsics=None,
        rays=None,
        poses=None,
        world_translations=None,
        world_translation_mask=None,
        world_rotations=None,
        world_rotation_mask=None,
        with_prior=None,
        overall_prob=1.0,
        ray_dirs_prob=0.0,
        depth_prob=0.0,
        world_translation_prob=None,
        world_rotation_prob=None,
        cam_prob=0.0,
        return_gs_features=False,
        return_scene_normalization=False,
    ):
        imgs = (imgs - self.image_mean) / self.image_std
        batch_size, num_views, _, height, width = imgs.shape
        device = imgs.device
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        p_ray, p_depth, p_t, p_r = self._resolve_probs(
            with_prior,
            overall_prob,
            ray_dirs_prob,
            depth_prob,
            world_translation_prob,
            world_rotation_prob,
            device,
        )

        if world_translations is None and poses is not None:
            world_translations = poses[..., :3, 3]
        if world_rotations is None and poses is not None:
            world_rotations = poses[..., :3, :3]

        center = scale = None
        t_mask = None
        norm_t = None
        depth_scale = None
        degenerate_translation_mask = None

        need_translation_scale = (
            p_depth > 0.0 and self.depth_prior_normalization == "world_translation"
        )
        if self.use_world_translation_prior and p_t > 0.0:
            if (
                world_translations is None
                or world_translations.shape[:2] != (batch_size, num_views)
                or world_translations.shape[-1] != 3
            ):
                raise ValueError("world_translations must have shape [B, N, 3]")
            t_mask = self._sample_mask(
                batch_size,
                num_views,
                device,
                p_t,
                "world_translation_mask",
                self.min_translation_prior_views,
                world_translation_mask,
            )
            norm_t, center, scale = self._norm_t(world_translations.to(device), t_mask)
            depth_scale = scale.view(batch_size, 1, 1, 1).clamp_min(1e-8)
            if self.training and self.force_rotation_prior_for_degenerate_translation:
                degenerate_translation_mask = self._translation_degenerate_mask(
                    world_translations.to(device),
                    t_mask,
                )
        elif need_translation_scale:
            depth_scale = self._translation_scale_for_depth(
                world_translations,
                B=batch_size,
                device=device,
                mask=world_translation_mask,
            )

        output_norm_center = center
        output_norm_scale = scale
        if return_scene_normalization and (center is None or scale is None):
            if (
                world_translations is None
                or world_translations.shape[:2] != (batch_size, num_views)
                or world_translations.shape[-1] != 3
            ):
                raise ValueError(
                    "return_scene_normalization=True requires world_translations "
                    "with shape [B, N, 3]."
                )
            _, center, scale = self._norm_t(
                world_translations.to(device),
                world_translation_mask,
            )

        hidden = self.encoder(
            imgs.reshape(batch_size * num_views, 3, height, width),
            is_training=True,
        )["x_norm_patchtokens"]
        ray_emb = self._ray_emb(
            hidden,
            batch_size,
            num_views,
            height,
            width,
            rays,
            intrinsics,
            self._sample_mask(batch_size, num_views, device, p_ray, "ray_dirs_mask"),
        )
        depth_emb = self._depth_emb(
            hidden,
            batch_size,
            num_views,
            height,
            width,
            depths,
            self._sample_mask(
                batch_size,
                num_views,
                device,
                p_depth,
                "depth_mask",
                mask=depth_mask,
            ),
            depth_scale=depth_scale,
        )
        hidden = hidden + ray_emb + depth_emb
        hidden = hidden.reshape(batch_size, num_views, -1, self.dec_embed_dim)
        camera_token_prior_emb = None

        if self.use_world_translation_prior and p_t > 0.0:
            t_emb = self.world_translation_encoder(norm_t).to(hidden.dtype)
            t_emb = t_emb * t_mask.to(device, hidden.dtype)[:, :, None]
            camera_token_prior_emb = t_emb

        force_rotation_prior = (
            self.training
            and self.force_rotation_prior_for_degenerate_translation
            and degenerate_translation_mask is not None
            and bool(degenerate_translation_mask.any())
        )
        use_rotation_prior_now = self.use_world_rotation_prior and (
            p_r > 0.0 or force_rotation_prior
        )
        if use_rotation_prior_now:
            if (
                world_rotations is None
                or world_rotations.shape[:2] != (batch_size, num_views)
                or world_rotations.shape[-2:] != (3, 3)
            ):
                raise ValueError("world_rotations must have shape [B, N, 3, 3]")
            r_mask = self._sample_mask(
                batch_size,
                num_views,
                device,
                p_r,
                "world_rotation_mask",
                self.min_rotation_prior_views,
                world_rotation_mask,
            )
            r_mask = self._force_one_rotation_for_degenerate_translation(
                r_mask,
                degenerate_translation_mask,
            )
            r6 = self._rot6d(world_rotations.to(device).float())
            r6 = r6 * r_mask.to(device, torch.float32)[:, :, None]
            r_emb = self.world_rotation_encoder(r6).to(hidden.dtype)
            r_emb = r_emb * r_mask.to(device, hidden.dtype)[:, :, None]
            camera_token_prior_emb = (
                r_emb
                if camera_token_prior_emb is None
                else camera_token_prior_emb + r_emb
            )

        hidden, pos, gs_feature_tokens = self.decode(
            hidden,
            num_views,
            height,
            width,
            return_gs_features=return_gs_features,
            camera_token_prior_emb=camera_token_prior_emb,
        )

        outputs = self.forward_head(
            hidden,
            pos,
            batch_size,
            num_views,
            height,
            width,
            patch_h,
            patch_w,
            return_gs_features=return_gs_features,
            gs_feature_tokens=gs_feature_tokens,
        )
        outputs = self._apply_translation_residual_anchor(outputs, norm_t, t_mask)

        if (
            self.use_world_translation_prior
            and self.de_normalize_outputs
            and output_norm_center is not None
            and output_norm_scale is not None
        ):
            outputs = self._denorm(
                outputs,
                output_norm_center.to(device, outputs["points"].dtype),
                output_norm_scale.to(device, outputs["points"].dtype),
                True,
            )
        if center is not None and scale is not None:
            outputs["world_translation_center"] = center
            outputs["world_translation_scale"] = scale
        return outputs

    def forward_head(
        self,
        hidden,
        pos,
        B,
        N,
        H,
        W,
        patch_h,
        patch_w,
        return_gs_features=False,
        gs_feature_tokens=None,
    ):
        ret_point = self.point_decoder(hidden, xpos=pos)
        ret_camera = self.camera_decoder(hidden, xpos=pos)
        ret_conf = self.conf_decoder(hidden, xpos=pos)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            point_feat = ret_point[:, self.patch_start_idx:].float()
            xy, z = self._chunked_conv_head(self.point_head, point_feat, patch_h, patch_w)
            del point_feat
            xy = xy.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
            z = z.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
            xy = torch.nan_to_num(
                xy.float(),
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            ).clamp(min=-1e4, max=1e4)
            z = torch.exp(
                torch.nan_to_num(
                    z.float(),
                    nan=0.0,
                    posinf=15.0,
                    neginf=-15.0,
                ).clamp(min=-15.0, max=15.0)
            )
            local_points = torch.nan_to_num(
                torch.cat([xy * z, z], dim=-1),
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            )
            ray_input = torch.nan_to_num(
                torch.cat([xy, torch.ones_like(z)], dim=-1),
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            )
            rays = torch.nan_to_num(
                F.normalize(ray_input, dim=-1, eps=1e-6),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            camera_token = ret_camera[:, 0].float()
            camera_poses = self.camera_head(camera_token).reshape(B, N, 4, 4)
            conf_feat = ret_conf[:, self.patch_start_idx:].float()
            conf = self._chunked_conv_head(self.conf_head, conf_feat, patch_h, patch_w)[0]
            del conf_feat
            conf = conf.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
            camera_poses = torch.nan_to_num(
                camera_poses,
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            )
            points = torch.einsum(
                "bnij,bnhwj->bnhwi",
                camera_poses,
                homogenize_points(local_points),
            )[..., :3]
            points = torch.nan_to_num(
                points,
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            )
            metric = torch.ones(B, device=hidden.device, dtype=points.dtype)
        outputs = dict(
            points=points,
            local_points=local_points,
            rays=rays,
            conf=conf,
            camera_poses=camera_poses,
            metric=metric,
        )
        if return_gs_features:
            if gs_feature_tokens is None:
                gs_feature_tokens = [
                    ret_point.float().reshape(B, N, ret_point.shape[1], -1)
                ] * 4
            outputs["gs_feature_tokens"] = gs_feature_tokens
            outputs["gs_patch_start_idx"] = self.patch_start_idx
        return outputs

    def decode(
        self,
        hidden,
        N,
        H,
        W,
        return_gs_features=False,
        camera_token_prior_emb=None,
    ):
        B, N, hw, _ = hidden.shape
        hidden = hidden.reshape(B * N, hw, -1)
        camera_token = self.camera_token.repeat(B, N, 1, 1).reshape(
            B * N,
            1,
            self.dec_embed_dim,
        )
        if camera_token_prior_emb is not None:
            camera_token = camera_token + camera_token_prior_emb.reshape(
                B * N,
                1,
                self.dec_embed_dim,
            ).to(device=camera_token.device, dtype=camera_token.dtype)
        reg = self.register_token.repeat(B, N, 1, 1).reshape(
            B * N,
            self.num_register_tokens,
            self.dec_embed_dim,
        )
        hidden = torch.cat([camera_token, reg, hidden], 1)
        hw = hidden.shape[1]
        pos = self.position_getter(
            B * N,
            H // self.patch_size,
            W // self.patch_size,
            hidden.device,
        )
        if self.patch_start_idx > 0:
            pos = torch.cat(
                [
                    torch.zeros(
                        B * N,
                        self.patch_start_idx,
                        2,
                        device=hidden.device,
                        dtype=pos.dtype,
                    ),
                    pos + 1,
                ],
                1,
            )
        temp = None
        gs_features = []
        gs_layer_idx = list(
            getattr(self, "gs_intermediate_layer_idx", [8, 17, 26, 35])
        )
        for i, blk in enumerate(self.decoder):
            if i % 2 == 0:
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)
            do_ckpt = self.gradient_checkpointing and (
                self.checkpoint_strategy == "all"
                or (self.checkpoint_strategy == "global_only" and i % 2 != 0)
            )
            pos = (
                pos.to(device=hidden.device, dtype=torch.long)
                .contiguous()
                .detach()
                .clone()
            )
            if self.training and do_ckpt:
                def run_blk(x, xpos, blk=blk):
                    return blk(x, xpos=xpos)

                hidden = checkpoint(run_blk, hidden, pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)
            if return_gs_features and i in gs_layer_idx:
                gs_features.append(hidden.reshape(B, N, hw, -1).float())
            if i == len(self.decoder) - 2:
                temp = hidden.clone().reshape(B * N, hw, -1)
        if temp is None:
            temp = hidden.clone().reshape(B * N, hw, -1)
        if return_gs_features and len(gs_features) != 4:
            raise ValueError(
                "gs_intermediate_layer_idx must select exactly 4 decoder layers, "
                f"got {gs_layer_idx!r} and collected {len(gs_features)}."
            )
        return (
            torch.cat([temp, hidden.reshape(B * N, hw, -1)], -1),
            pos.reshape(B * N, hw, -1),
            gs_features if return_gs_features else None,
        )
