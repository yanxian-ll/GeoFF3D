from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch.utils.checkpoint import checkpoint

from geoff3d.models.external.dinov2.hub.backbones import dinov2_vitl14_reg
from geoff3d.models.external.dinov2.layers import Mlp, PatchEmbed
from geoff3d.models.external.pi3.layers.attention import FlashAttentionRope
from geoff3d.models.external.pi3.layers.block import BlockRope
from geoff3d.models.external.pi3.layers.camera_head import CameraHead
from geoff3d.models.external.pi3.layers.conv_head import ConvHead
from geoff3d.models.external.pi3.layers.pos_embed import PositionGetter, RoPE2D
from geoff3d.models.external.pi3.layers.transformer_head import TransformerDecoder


def get_pixel(H, W):
    u_a, v_a = np.meshgrid(np.arange(W), np.arange(H))
    return np.stack(
        [u_a.flatten() + 0.5, v_a.flatten() + 0.5, np.ones_like(u_a.flatten())],
        axis=0,
    )


def homogenize_points(points):
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, zero_init=True, layer_norm=False):
        super().__init__()
        layers = []
        if layer_norm:
            layers.append(nn.LayerNorm(in_dim))
        layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class WorldTranslationEncoder(_MLP):
    pass


class WorldRotationEncoder(_MLP):
    pass


class GeoFF3D(nn.Module, PyTorchModelHubMixin):
    """GeoFF3D model with unified ray/depth/world-translation/world-rotation priors.

    Priors are controlled by probabilities from model.task. There is no separate
    use_multimodal switch, no Pi3X pose injection, and no metric branch.
    """

    def __init__(
        self,
        ckpt=None,
        gradient_checkpointing=False,
        checkpoint_strategy="all",
        use_world_translation_prior=True,
        translation_normalization="scale",
        depth_prior_normalization="world_translation",
        de_normalize_outputs=False,
        translation_encoder_hidden_dim=256,
        zero_init_translation_encoder=True,
        translation_encoder_input_layer_norm=False,
        min_translation_prior_views=3,
        use_world_rotation_prior=False,
        rotation_encoder_hidden_dim=256,
        zero_init_rotation_encoder=True,
        rotation_encoder_input_layer_norm=False,
        min_rotation_prior_views=3,
        force_rotation_prior_for_degenerate_translation=True,
        translation_collinearity_threshold=0.05,
        translation_degenerate_baseline_eps=1e-6,
        default_world_translation_prob=1.0,
        default_world_rotation_prob=1.0,
        translation_prior_prob: Optional[float] = None,
        rotation_prior_prob: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_strategy = checkpoint_strategy

        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        if hasattr(self.encoder, "mask_token"):
            del self.encoder.mask_token

        if self.gradient_checkpointing:
            for i in range(len(self.encoder.blocks)):
                self.encoder.blocks[i] = self.wrap_module_with_gradient_checkpointing(self.encoder.blocks[i])

        self.rope = RoPE2D(freq=100)
        self.position_getter = PositionGetter()

        dec_embed_dim = 1024
        dec_num_heads = 16
        mlp_ratio = 4
        dec_depth = 36
        self.decoder = nn.ModuleList(
            [
                BlockRope(
                    dim=dec_embed_dim,
                    num_heads=dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    drop_path=0.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    ffn_layer=Mlp,
                    init_values=0.01,
                    qk_norm=True,
                    attn_class=FlashAttentionRope,
                    rope=self.rope,
                )
                for _ in range(dec_depth)
            ]
        )
        self.dec_embed_dim = dec_embed_dim

        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # Prior encoders are always constructed. Actual usage is controlled only
        # by ray_dirs_prob/depth_prob/world_translation_prob/world_rotation_prob.
        self.depth_encoder = deepcopy(self.encoder)
        del self.depth_encoder.patch_embed
        self.depth_encoder.patch_embed = PatchEmbed(img_size=224, patch_size=14, in_chans=2, embed_dim=1024)
        self.depth_emb = nn.Parameter(torch.zeros(1, 1, 1024))

        self.ray_embed = PatchEmbed(img_size=224, patch_size=14, in_chans=2, embed_dim=1024)
        nn.init.constant_(self.ray_embed.proj.weight, 0)
        nn.init.constant_(self.ray_embed.proj.bias, 0)

        self.point_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
            use_checkpoint=self.gradient_checkpointing,
        )
        self.point_head = ConvHead(
            num_features=4,
            dim_in=dec_embed_dim,
            projects=nn.Identity(),
            dim_out=[2, 1],
            dim_proj=1024,
            dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            res_block_norm="group_norm",
            last_res_blocks=0,
            last_conv_channels=32,
            last_conv_size=1,
            using_uv=True,
        )

        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False,
        )
        self.camera_head = CameraHead(dim=512)

        self.conf_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
            use_checkpoint=self.gradient_checkpointing,
        )
        self.conf_head = ConvHead(
            num_features=4,
            dim_in=dec_embed_dim,
            projects=nn.Identity(),
            dim_out=[1],
            dim_proj=1024,
            dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            res_block_norm="group_norm",
            last_res_blocks=0,
            last_conv_channels=32,
            last_conv_size=1,
            using_uv=True,
        )

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        if depth_prior_normalization not in {"world_translation", "mean_aug"}:
            raise ValueError(
                "depth_prior_normalization must be one of "
                "{'world_translation', 'mean_aug'}, got "
                f"{depth_prior_normalization!r}."
            )
        self.depth_prior_normalization = depth_prior_normalization
        self.use_world_translation_prior = bool(use_world_translation_prior)
        self.translation_normalization = translation_normalization
        self.de_normalize_outputs = bool(de_normalize_outputs)
        self.min_translation_prior_views = int(min_translation_prior_views)
        self.use_world_rotation_prior = bool(use_world_rotation_prior)
        self.min_rotation_prior_views = int(min_rotation_prior_views)
        self.force_rotation_prior_for_degenerate_translation = bool(force_rotation_prior_for_degenerate_translation)
        self.translation_collinearity_threshold = float(translation_collinearity_threshold)
        self.translation_degenerate_baseline_eps = float(translation_degenerate_baseline_eps)
        self.default_world_translation_prob = float(default_world_translation_prob if translation_prior_prob is None else translation_prior_prob)
        self.default_world_rotation_prob = float(default_world_rotation_prob if rotation_prior_prob is None else rotation_prior_prob)
        self.world_translation_encoder = WorldTranslationEncoder(
            3,
            translation_encoder_hidden_dim,
            self.dec_embed_dim,
            zero_init_translation_encoder,
            translation_encoder_input_layer_norm,
        )
        self.world_rotation_encoder = WorldRotationEncoder(
            6,
            rotation_encoder_hidden_dim,
            self.dec_embed_dim,
            zero_init_rotation_encoder,
            rotation_encoder_input_layer_norm,
        )

    def wrap_module_with_gradient_checkpointing(self, module: nn.Module):
        class _CheckpointingWrapper(module.__class__):
            _restore_cls = module.__class__

            def forward(self, *args, **kwargs):
                return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)

        module.__class__ = _CheckpointingWrapper
        return module

    @staticmethod
    def _prob(name, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        return value

    def _resolve_probs(self, with_prior, overall_prob, ray_dirs_prob, depth_prob, world_translation_prob, world_rotation_prob, device):
        if with_prior is False or torch.rand(1, device=device) >= self._prob("overall_prob", overall_prob):
            return 0.0, 0.0, 0.0, 0.0
        return (
            self._prob("ray_dirs_prob", ray_dirs_prob),
            self._prob("depth_prob", depth_prob),
            self._prob("world_translation_prob", self.default_world_translation_prob if world_translation_prob is None else world_translation_prob),
            self._prob("world_rotation_prob", self.default_world_rotation_prob if world_rotation_prob is None else world_rotation_prob),
        )

    def _sample_mask(self, B, N, device, prob, name, min_views=0, mask=None):
        if mask is not None:
            if mask.shape != (B, N):
                raise ValueError(f"{name} must have shape [B, N]")
            return mask.to(device=device, dtype=torch.bool)
        prob = self._prob(name.replace("_mask", "_prob"), prob)
        if prob <= 0.0:
            return torch.zeros(B, N, device=device, dtype=torch.bool)
        if (not self.training) or prob >= 1.0:
            return torch.ones(B, N, device=device, dtype=torch.bool)
        mask = torch.rand(B, N, device=device) <= prob
        min_views = min(N, int(min_views))
        for b in range(B):
            if int(mask[b].sum()) < min_views:
                missing = torch.nonzero(~mask[b], as_tuple=False).flatten()
                need = min(min_views - int(mask[b].sum()), int(missing.numel()))
                if need > 0:
                    perm = torch.randperm(int(missing.numel()), device=device)[:need]
                    mask[b, missing[perm]] = True
        return mask

    @staticmethod
    def _rot6d(R):
        return R[..., :3, :2].transpose(-1, -2).reshape(*R.shape[:-2], 6)

    def _translation_degenerate_mask(self, world_translations, translation_mask):
        """Return per-sample training mask where translation anchors are ambiguous.

        Degenerate cases: effective translation prior count <= 2, nearly zero
        baseline, or valid camera centers are close to a line. Collinearity is
        measured by the second singular value relative to the first singular
        value of centered valid translation centers.
        """
        B, N = world_translations.shape[:2]
        device = world_translations.device
        if translation_mask is None:
            translation_mask = torch.ones(B, N, device=device, dtype=torch.bool)
        else:
            translation_mask = translation_mask.to(device=device, dtype=torch.bool)

        counts = translation_mask.sum(dim=1)
        degenerate = counts <= 2
        if N < 2:
            return torch.ones(B, device=device, dtype=torch.bool)

        valid = translation_mask.to(device=device, dtype=torch.float32).unsqueeze(-1)
        count_f = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        t = world_translations.to(device=device, dtype=torch.float32)
        center = (t * valid).sum(dim=1, keepdim=True) / count_f
        centered = (t - center) * valid
        svals = torch.linalg.svdvals(centered)
        first = svals[:, 0]
        if svals.shape[-1] >= 2:
            second = svals[:, 1]
        else:
            second = torch.zeros_like(first)
        near_zero_baseline = first <= self.translation_degenerate_baseline_eps
        nearly_collinear = second <= (self.translation_collinearity_threshold * first.clamp_min(self.translation_degenerate_baseline_eps))
        degenerate = degenerate | ((counts >= 3) & (near_zero_baseline | nearly_collinear))
        return degenerate

    def _force_one_rotation_for_degenerate_translation(self, rotation_mask, degenerate_translation_mask):
        if (
            (not self.training)
            or degenerate_translation_mask is None
            or not bool(degenerate_translation_mask.any())
        ):
            return rotation_mask
        rotation_mask = rotation_mask.clone()
        needs_rotation = degenerate_translation_mask & (~rotation_mask.any(dim=1))
        if not bool(needs_rotation.any()):
            return rotation_mask
        rows = torch.nonzero(needs_rotation, as_tuple=False).flatten()
        cols = torch.randint(0, rotation_mask.shape[1], (rows.numel(),), device=rotation_mask.device)
        rotation_mask[rows, cols] = True
        return rotation_mask

    def _norm_t(self, t, mask=None, eps=1e-6):
        t = t.float()
        B, N = t.shape[:2]
        valid = torch.ones(B, N, 1, device=t.device, dtype=t.dtype) if mask is None else mask.to(t.device, t.dtype).unsqueeze(-1)
        count = valid.sum(1, keepdim=True).clamp_min(1.0)
        one = torch.ones(B, 1, 1, device=t.device, dtype=t.dtype)
        if self.translation_normalization == "none":
            return t * valid, torch.zeros_like(t[:, :1]), one
        if self.translation_normalization == "scale":
            center = torch.zeros_like(t[:, :1])
            scale = (t.norm(dim=-1, keepdim=True) * valid).sum(1, keepdim=True) / count
            scale = torch.where(scale > eps, scale, one)
            return (t / scale) * valid, center, scale
        if self.translation_normalization == "mean":
            center = (t * valid).sum(1, keepdim=True) / count
        elif self.translation_normalization == "first_view":
            center = t[:, :1]
        else:
            raise ValueError(f"unknown translation_normalization: {self.translation_normalization}")
        d = t - center
        scale = (d.norm(dim=-1, keepdim=True) * valid).sum(1, keepdim=True) / count
        scale = torch.where(scale > eps, scale, one)
        return (d / scale) * valid, center, scale

    def _translation_scale_for_depth(self, world_translations: Optional[torch.Tensor], B: int, device: torch.device, mask=None) -> torch.Tensor:
        if world_translations is None:
            raise ValueError(
                "depth_prior_normalization='world_translation' requires "
                "world_translations with shape [B, N, 3]."
            )
        _, _, scale = self._norm_t(world_translations.to(device), mask=mask)
        return scale.view(B, 1, 1, 1).clamp_min(1e-8)

    def _ray_emb(self, hidden, B, N, H, W, rays, intrinsics, mask):
        if int(mask.sum()) <= 0:
            return torch.zeros_like(hidden)
        device = hidden.device
        if rays is not None:
            rays_device = rays.to(device)
            rays_xy = rays_device[..., :2] / (rays_device[..., 2:3] + 1e-6)
        elif intrinsics is not None:
            pix = torch.from_numpy(get_pixel(H, W).T.reshape(H, W, 3)).to(device).float()[None].repeat(B, 1, 1, 1)
            rays_xy = torch.einsum("bnij,bhwj->bnhwi", torch.inverse(intrinsics.to(device)), pix)[..., :2]
        else:
            return torch.zeros_like(hidden)
        emb = self.ray_embed(rays_xy.reshape(B * N, H, W, 2).permute(0, 3, 1, 2))
        return emb * mask.reshape(B * N, 1, 1).to(device, emb.dtype)

    def _depth_emb(self, hidden, B, N, H, W, depths, mask, depth_scale=None):
        if depths is None or int(mask.sum()) <= 0:
            return torch.zeros_like(hidden)
        depths = depths.to(hidden.device)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            if self.depth_prior_normalization == "mean_aug":
                d, _ = self.normalize_depth(depths, method="mean")
                if self.training:
                    d = d / (0.8 + torch.rand(B, device=hidden.device) * 0.4).view(B, 1, 1, 1)
            elif self.depth_prior_normalization == "world_translation":
                if depth_scale is None:
                    raise ValueError(
                        "depth_prior_normalization='world_translation' requires depth_scale."
                    )
                d = depths.float() / depth_scale.to(device=hidden.device, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown depth_prior_normalization: {self.depth_prior_normalization!r}")
            valid = (depths > 0).float().reshape(B * N, 1, H, W)
            d = d.reshape(B * N, 1, H, W)
        emb = self.depth_encoder(torch.cat([d, valid], 1), is_training=True)["x_norm_patchtokens"] + self.depth_emb
        return emb * mask.reshape(B * N, 1, 1).to(hidden.device, emb.dtype)

    @staticmethod
    def _with_t(poses, t):
        return torch.cat([torch.cat([poses[..., :3, :3], t], -1), poses[..., 3:4, :]], -2)

    @staticmethod
    def _denorm(outputs, center, scale, scale_local_points=False):
        outputs = dict(outputs)
        B = center.shape[0]
        if outputs.get("points") is not None:
            outputs["points"] = outputs["points"] * scale.view(B, 1, 1, 1, 1) + center.view(B, 1, 1, 1, 3)
        if scale_local_points and outputs.get("local_points") is not None:
            outputs["local_points"] = outputs["local_points"] * scale.view(B, 1, 1, 1, 1)
        if outputs.get("camera_poses") is not None:
            outputs["camera_poses"] = GeoFF3D._with_t(
                outputs["camera_poses"],
                outputs["camera_poses"][..., :3, 3:4] * scale.view(B, 1, 1, 1) + center.view(B, 1, 3, 1),
            )
        return outputs

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
        B, N, _, H, W = imgs.shape
        device = imgs.device
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
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

        # Compute world-translation normalization first. The resulting `scale`
        # is the single source of truth for both translation prior embedding and
        # depth-prior normalization.
        need_translation_scale = p_depth > 0.0 and self.depth_prior_normalization == "world_translation"
        if self.use_world_translation_prior and p_t > 0.0:
            if world_translations is None or world_translations.shape[:2] != (B, N) or world_translations.shape[-1] != 3:
                raise ValueError("world_translations must have shape [B, N, 3]")
            t_mask = self._sample_mask(B, N, device, p_t, "world_translation_mask", self.min_translation_prior_views, world_translation_mask)
            norm_t, center, scale = self._norm_t(world_translations.to(device), t_mask)
            depth_scale = scale.view(B, 1, 1, 1).clamp_min(1e-8)
            if self.training and self.force_rotation_prior_for_degenerate_translation:
                degenerate_translation_mask = self._translation_degenerate_mask(world_translations.to(device), t_mask)
        elif need_translation_scale:
            depth_scale = self._translation_scale_for_depth(world_translations, B=B, device=device, mask=world_translation_mask)

        output_norm_center = center
        output_norm_scale = scale
        if return_scene_normalization and (center is None or scale is None):
            if world_translations is None or world_translations.shape[:2] != (B, N) or world_translations.shape[-1] != 3:
                raise ValueError(
                    "return_scene_normalization=True requires world_translations "
                    "with shape [B, N, 3]."
                )
            _, center, scale = self._norm_t(world_translations.to(device), world_translation_mask)

        # encode image and dense priors
        hidden = self.encoder(imgs.reshape(B * N, 3, H, W), is_training=True)["x_norm_patchtokens"]
        ray_emb = self._ray_emb(hidden, B, N, H, W, rays, intrinsics, self._sample_mask(B, N, device, p_ray, "ray_dirs_mask"))
        depth_emb = self._depth_emb(
            hidden,
            B,
            N,
            H,
            W,
            depths,
            self._sample_mask(B, N, device, p_depth, "depth_mask", mask=depth_mask),
            depth_scale=depth_scale,
        )
        hidden = hidden + ray_emb + depth_emb
        hidden = hidden.reshape(B, N, -1, self.dec_embed_dim)

        # encode translation & rotation. Translation uses the same `scale` that
        # was already used for depth_emb when depth_prior_normalization is
        # 'world_translation'.
        if self.use_world_translation_prior and p_t > 0.0:
            t_emb = self.world_translation_encoder(norm_t).to(hidden.dtype) * t_mask.to(device, hidden.dtype)[:, :, None]
            hidden = hidden + t_emb[:, :, None, :]

        force_rotation_prior = (
            self.training
            and self.force_rotation_prior_for_degenerate_translation
            and degenerate_translation_mask is not None
            and bool(degenerate_translation_mask.any())
        )
        use_rotation_prior_now = self.use_world_rotation_prior and (p_r > 0.0 or force_rotation_prior)
        if use_rotation_prior_now:
            if world_rotations is None or world_rotations.shape[:2] != (B, N) or world_rotations.shape[-2:] != (3, 3):
                raise ValueError("world_rotations must have shape [B, N, 3, 3]")
            r_mask = self._sample_mask(B, N, device, p_r, "world_rotation_mask", self.min_rotation_prior_views, world_rotation_mask)
            r_mask = self._force_one_rotation_for_degenerate_translation(r_mask, degenerate_translation_mask)
            r6 = self._rot6d(world_rotations.to(device).float()) * r_mask.to(device, torch.float32)[:, :, None]
            r_emb = self.world_rotation_encoder(r6).to(hidden.dtype) * r_mask.to(device, hidden.dtype)[:, :, None]
            hidden = hidden + r_emb[:, :, None, :]

        # decode
        hidden, pos, gs_feature_tokens = self.decode(
            hidden,
            N,
            H,
            W,
            return_gs_features=return_gs_features,
        )

        # head
        outputs = self.forward_head(
            hidden,
            pos,
            B,
            N,
            H,
            W,
            patch_h,
            patch_w,
            return_gs_features=return_gs_features,
            gs_feature_tokens=gs_feature_tokens,
        )
        # denorm
        if self.use_world_translation_prior and self.de_normalize_outputs and output_norm_center is not None and output_norm_scale is not None:
            outputs = self._denorm(outputs, output_norm_center.to(device, outputs["points"].dtype), output_norm_scale.to(device, outputs["points"].dtype), True)
        if center is not None and scale is not None:
            outputs["world_translation_center"] = center
            outputs["world_translation_scale"] = scale
        return outputs

    def _chunked_conv_head(self, head, feat, patch_h, patch_w, chunk_size=64):
        BN = feat.shape[0]
        if BN <= chunk_size:
            return head(feat, patch_h=patch_h, patch_w=patch_w)
        outputs = [[] for _ in range(len(head.output_block))] if isinstance(head.output_block, nn.ModuleList) else []
        for i in range(0, BN, chunk_size):
            chunk_out = head(feat[i:i + chunk_size], patch_h=patch_h, patch_w=patch_w)
            if isinstance(chunk_out, list):
                for j, out in enumerate(chunk_out):
                    outputs[j].append(out)
            else:
                outputs.append(chunk_out)
        if isinstance(outputs[0], list):
            return [torch.cat(parts, dim=0) for parts in outputs]
        return torch.cat(outputs, dim=0)

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
            xy = torch.nan_to_num(xy.float(), nan=0.0, posinf=1e4, neginf=-1e4).clamp(min=-1e4, max=1e4)
            z = torch.exp(torch.nan_to_num(z.float(), nan=0.0, posinf=15.0, neginf=-15.0).clamp(min=-15.0, max=15.0))
            local_points = torch.nan_to_num(torch.cat([xy * z, z], dim=-1), nan=0.0, posinf=1e6, neginf=-1e6)
            ray_input = torch.nan_to_num(torch.cat([xy, torch.ones_like(z)], dim=-1), nan=0.0, posinf=1e4, neginf=-1e4)
            rays = torch.nan_to_num(F.normalize(ray_input, dim=-1, eps=1e-6), nan=0.0, posinf=1.0, neginf=-1.0)
            camera_poses = self.camera_head(ret_camera[:, self.patch_start_idx:].float(), patch_h, patch_w).reshape(B, N, 4, 4)
            conf_feat = ret_conf[:, self.patch_start_idx:].float()
            conf = self._chunked_conv_head(self.conf_head, conf_feat, patch_h, patch_w)[0]
            del conf_feat
            conf = conf.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
            camera_poses = torch.nan_to_num(camera_poses, nan=0.0, posinf=1e6, neginf=-1e6)
            points = torch.einsum("bnij,bnhwj->bnhwi", camera_poses, homogenize_points(local_points))[..., :3]
            points = torch.nan_to_num(points, nan=0.0, posinf=1e6, neginf=-1e6)
            metric = torch.ones(B, device=hidden.device, dtype=points.dtype)
        outputs = dict(points=points, local_points=local_points, rays=rays, conf=conf, camera_poses=camera_poses, metric=metric)
        if return_gs_features:
            if gs_feature_tokens is None:
                gs_feature_tokens = [
                    ret_point.float().reshape(B, N, ret_point.shape[1], -1)
                ] * 4
            outputs["gs_feature_tokens"] = gs_feature_tokens
            outputs["gs_patch_start_idx"] = self.patch_start_idx
        return outputs

    def normalize_depth(self, depths: torch.Tensor, method: str = "median"):
        if not isinstance(depths, torch.Tensor):
            depths = torch.tensor(depths, dtype=torch.float32)
        if method not in ["median", "mean"]:
            raise ValueError(f"Invalid normalization method: {method!r}. Choose 'median' or 'mean'.")
        B, N, H, W = depths.shape
        valid_depths = torch.where(depths > 0, depths, torch.nan).view(B, -1)
        if method == "median":
            factors, _ = torch.nanmedian(valid_depths, dim=1)
        else:
            factors = torch.nanmean(valid_depths, dim=1)
        factors = torch.nan_to_num(factors, nan=1.0)
        return depths / (factors.view(B, 1, 1, 1) + 1e-8), factors.reshape(-1)

    def decode(self, hidden, N, H, W, return_gs_features=False):
        B, N, hw, _ = hidden.shape
        hidden = hidden.reshape(B * N, hw, -1)
        reg = self.register_token.repeat(B, N, 1, 1).reshape(B * N, *self.register_token.shape[-2:])
        hidden = torch.cat([reg, hidden], 1)
        hw = hidden.shape[1]
        pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, hidden.device)
        if self.patch_start_idx > 0:
            pos = torch.cat([torch.zeros(B * N, self.patch_start_idx, 2, device=hidden.device, dtype=pos.dtype), pos + 1], 1)
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
            do_ckpt = self.gradient_checkpointing and (self.checkpoint_strategy == "all" or (self.checkpoint_strategy == "global_only" and i % 2 != 0))
            pos = pos.to(device=hidden.device, dtype=torch.long).contiguous().detach().clone()
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
