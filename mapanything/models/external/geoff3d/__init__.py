"""Wrapper for GeoFF3D."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from mapanything.models.external.pi3.models.pi3x import Pi3X
from mapanything.models.external.pi3.models.geoff3d import GeoFF3D
from mapanything.models.external.pi3.models.geoff3d_camera_token import (
    GeoFF3DCameraToken,
)
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


def _any_have(views: List[Dict], key: str) -> bool:
    return any(key in view for view in views)


def _stack_optional_masks(views: List[Dict], key: str, device: torch.device) -> Optional[torch.Tensor]:
    if not _all_have(views, key):
        return None
    masks = []
    for view in views:
        mask = view[key]
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, device=device)
        mask = mask.to(device=device, dtype=torch.bool)
        if mask.dim() == 0:
            mask = mask.view(1)
        masks.append(mask)
    return torch.stack(masks, dim=1)


def _cfg_float(cfg, key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except Exception:
        return default


def _cfg_get(cfg, key: str, default):
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except AttributeError:
        return default


def _maybe_sparsify_depths_like_mapanything(
    depths: torch.Tensor,
    sparse_depth_prob: float,
    sparsification_removal_percent: float,
) -> torch.Tensor:
    """Randomly remove valid depth pixels using MapAnything's sparse-depth policy.

    `depths` is expected to be [B, N, H, W]. One Bernoulli sample decides whether
    sparsification is applied for the current forward pass. If enabled, each view
    removes the configured fraction of currently valid depth pixels, matching the
    per-view behavior in MapAnything's `_encode_and_fuse_depths`.
    """
    sparse_depth_prob = float(sparse_depth_prob)
    sparsification_removal_percent = float(sparsification_removal_percent)
    if sparse_depth_prob <= 0.0 or sparsification_removal_percent <= 0.0:
        return depths
    if depths is None:
        return depths
    if depths.dim() != 4:
        raise ValueError(f"depths must have shape [B, N, H, W], got {tuple(depths.shape)}")

    sparse_depth_prob = min(max(sparse_depth_prob, 0.0), 1.0)
    sparsification_removal_percent = min(max(sparsification_removal_percent, 0.0), 1.0)
    if bool((torch.rand(1, device=depths.device) >= sparse_depth_prob).item()):
        return depths

    sparse_depths = depths.clone()
    num_views = sparse_depths.shape[1]
    for view_idx in range(num_views):
        valid_pixel_mask = sparse_depths[:, view_idx] > 0
        num_valid_pixels = int(valid_pixel_mask.sum().item())
        num_to_zero = int(num_valid_pixels * sparsification_removal_percent)
        if num_to_zero <= 0:
            continue

        valid_indices = valid_pixel_mask.nonzero(as_tuple=False)
        indices_to_zero = torch.randperm(
            num_valid_pixels,
            device=sparse_depths.device,
        )[:num_to_zero]
        selected = valid_indices[indices_to_zero]
        sparse_depths[
            selected[:, 0],
            view_idx,
            selected[:, 1],
            selected[:, 2],
        ] = 0

    return sparse_depths


class GeoFF3DWrapper(torch.nn.Module):
    model_cls = GeoFF3D
    model_display_name = "GeoFF3D"

    def __init__(
        self,
        name,
        geometric_input_config,
        pretrained_model_name_or_path="checkpoints/pi3x",
        load_pretrained_weights: bool = True,
        use_conditioning: bool = True,
        gradient_checkpointing: bool = False,
        torch_hub_force_reload: bool = False,
        use_world_translation_prior: bool = True,
        translation_normalization: str = "scale",
        depth_prior_normalization: str = "world_translation",
        de_normalize_outputs: bool = False,
        translation_encoder_hidden_dim: int = 256,
        zero_init_translation_encoder: bool = True,
        translation_encoder_input_layer_norm: bool = False,
        min_translation_prior_views: int = 3,
        world_translation_prior_jitter: Optional[Dict] = None,
        use_world_rotation_prior: bool = False,
        rotation_encoder_hidden_dim: int = 256,
        zero_init_rotation_encoder: bool = True,
        rotation_encoder_input_layer_norm: bool = False,
        min_rotation_prior_views: int = 3,
        force_rotation_prior_for_degenerate_translation: bool = True,
        translation_collinearity_threshold: float = 0.05,
        translation_degenerate_baseline_eps: float = 1e-6,
        translation_prior_prob: float = 1.0,
        rotation_prior_prob: float = 1.0,
        use_translation_residual_anchor: bool = False,
        translation_residual_anchor_delta_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.geometric_input_config = geometric_input_config
        self.use_conditioning = use_conditioning
        self.dtype = _get_autocast_dtype()
        self.use_world_translation_prior = bool(use_world_translation_prior)
        self.use_world_rotation_prior = bool(use_world_rotation_prior)
        self.force_rotation_prior_for_degenerate_translation = bool(
            force_rotation_prior_for_degenerate_translation
        )
        self.default_translation_prior_prob = float(translation_prior_prob)
        self.default_rotation_prior_prob = float(rotation_prior_prob)
        self.translation_normalization = translation_normalization
        self.translation_residual_anchor_delta_scale = float(
            translation_residual_anchor_delta_scale
        )
        self.world_translation_prior_jitter = self._parse_world_translation_prior_jitter(
            world_translation_prior_jitter
        )

        self.model = self.model_cls(
            gradient_checkpointing=gradient_checkpointing,
            use_world_translation_prior=use_world_translation_prior,
            translation_normalization=translation_normalization,
            depth_prior_normalization=depth_prior_normalization,
            de_normalize_outputs=de_normalize_outputs,
            translation_encoder_hidden_dim=translation_encoder_hidden_dim,
            zero_init_translation_encoder=zero_init_translation_encoder,
            translation_encoder_input_layer_norm=translation_encoder_input_layer_norm,
            min_translation_prior_views=min_translation_prior_views,
            use_world_rotation_prior=use_world_rotation_prior,
            rotation_encoder_hidden_dim=rotation_encoder_hidden_dim,
            zero_init_rotation_encoder=zero_init_rotation_encoder,
            rotation_encoder_input_layer_norm=rotation_encoder_input_layer_norm,
            min_rotation_prior_views=min_rotation_prior_views,
            force_rotation_prior_for_degenerate_translation=force_rotation_prior_for_degenerate_translation,
            translation_collinearity_threshold=translation_collinearity_threshold,
            translation_degenerate_baseline_eps=translation_degenerate_baseline_eps,
            default_world_translation_prob=translation_prior_prob,
            default_world_rotation_prob=rotation_prior_prob,
            use_translation_residual_anchor=use_translation_residual_anchor,
            translation_residual_anchor_delta_scale=translation_residual_anchor_delta_scale,
        )

        if load_pretrained_weights:
            if torch_hub_force_reload:
                base_model = Pi3X.from_pretrained("yyfz233/Pi3X", force_download=True)
            else:
                base_model = Pi3X.from_pretrained(pretrained_model_name_or_path)
            incompatible = self.model.load_state_dict(base_model.state_dict(), strict=False)
            print(
                f"Loaded base Pi3X weights into independent {self.model_display_name}. "
                f"Missing keys: {list(incompatible.missing_keys)}; "
                f"Unexpected keys: {list(incompatible.unexpected_keys)}"
            )
            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _parse_world_translation_prior_jitter(cfg: Optional[Dict]) -> Dict:
        parsed = {
            "enabled": bool(_cfg_get(cfg, "enabled", False)),
            "apply_prob": float(_cfg_get(cfg, "apply_prob", 0.5)),
            "std_xy": float(_cfg_get(cfg, "std_xy", 0.05)),
            "std_z": float(_cfg_get(cfg, "std_z", 0.10)),
            "bias_std_xy": float(_cfg_get(cfg, "bias_std_xy", 0.02)),
            "bias_std_z": float(_cfg_get(cfg, "bias_std_z", 0.05)),
            "scale_by_residual_anchor_delta": bool(
                _cfg_get(cfg, "scale_by_residual_anchor_delta", True)
            ),
        }
        if not 0.0 <= parsed["apply_prob"] <= 1.0:
            raise ValueError("world_translation_prior_jitter.apply_prob must be in [0, 1]")
        for key in ("std_xy", "std_z", "bias_std_xy", "bias_std_z"):
            if parsed[key] < 0.0:
                raise ValueError(f"world_translation_prior_jitter.{key} must be non-negative")
        return parsed

    def _translation_jitter_scale(self, translations: torch.Tensor) -> torch.Tensor:
        batch_size = translations.shape[0]
        one = torch.ones(
            batch_size,
            1,
            1,
            device=translations.device,
            dtype=translations.dtype,
        )
        if self.translation_normalization == "none":
            return one
        if self.translation_normalization == "scale":
            centered = translations
        elif self.translation_normalization == "mean":
            centered = translations - translations.mean(dim=1, keepdim=True)
        elif self.translation_normalization == "first_view":
            centered = translations - translations[:, :1]
        else:
            raise ValueError(
                f"unknown translation_normalization: {self.translation_normalization}"
            )
        scale = centered.norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
        return torch.where(scale > 1e-6, scale, one)

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
            res.append(
                {
                    "pts3d": results["points"][:, i, ...],
                    "pts3d_cam": curr_view_pts3d_cam,
                    "ray_directions": curr_view_ray_dirs,
                    "depth_along_ray": curr_view_depth_along_ray,
                    "cam_trans": curr_view_cam_translations,
                    "cam_quats": curr_view_cam_quats,
                    "conf": results["conf"][:, i, ...],
                }
            )
        return res

    def _apply_world_translation_prior_jitter(self, translations: torch.Tensor) -> torch.Tensor:
        cfg = self.world_translation_prior_jitter
        if (not self.training) or (not cfg["enabled"]) or cfg["apply_prob"] <= 0.0:
            return translations
        batch_size, num_views = translations.shape[:2]
        device = translations.device
        work_dtype = torch.float32
        translations_f = translations.to(dtype=work_dtype)
        apply_mask = (
            torch.rand(batch_size, 1, 1, device=device, dtype=work_dtype)
            < cfg["apply_prob"]
        ).to(work_dtype)
        view_std = torch.tensor(
            [cfg["std_xy"], cfg["std_xy"], cfg["std_z"]],
            device=device,
            dtype=work_dtype,
        ).view(1, 1, 3)
        bias_std = torch.tensor(
            [cfg["bias_std_xy"], cfg["bias_std_xy"], cfg["bias_std_z"]],
            device=device,
            dtype=work_dtype,
        ).view(1, 1, 3)
        view_delta = torch.randn(batch_size, num_views, 3, device=device, dtype=work_dtype) * view_std
        scene_delta = torch.randn(batch_size, 1, 3, device=device, dtype=work_dtype) * bias_std
        residual_scale = (
            abs(self.translation_residual_anchor_delta_scale)
            if cfg["scale_by_residual_anchor_delta"]
            else 1.0
        )
        jitter_scale = self._translation_jitter_scale(translations_f)
        delta = (view_delta + scene_delta) * jitter_scale * residual_scale
        return translations + (delta * apply_mask).to(dtype=translations.dtype)

    def forward(self, views: List[Dict]):
        device = views[0]["img"].device
        num_views = len(views)
        assert views[0]["data_norm_type"][0] == "identity"

        images = torch.stack([view["img"] for view in views], dim=1)
        cfg = self.geometric_input_config

        overall_prob = _cfg_float(cfg, "overall_prob", 1.0)
        ray_dirs_prob = _cfg_float(cfg, "ray_dirs_prob", 0.0)
        depth_prob = _cfg_float(cfg, "depth_prob", 0.0)
        sparse_depth_prob = _cfg_float(cfg, "sparse_depth_prob", 0.0)
        sparsification_removal_percent = _cfg_float(
            cfg,
            "sparsification_removal_percent",
            0.0,
        )
        world_translation_prob = _cfg_float(
            cfg,
            "world_translation_prob",
            _cfg_float(cfg, "translation_prior_prob", self.default_translation_prior_prob),
        )
        world_rotation_prob = _cfg_float(
            cfg,
            "world_rotation_prob",
            _cfg_float(cfg, "rotation_prior_prob", self.default_rotation_prior_prob),
        )

        if not self.use_conditioning:
            overall_prob = 0.0
        if not self.use_world_translation_prior:
            world_translation_prob = 0.0
        if not self.use_world_rotation_prior:
            world_rotation_prob = 0.0

        return_scene_normalization = bool(
            getattr(self, "return_gs_features", False)
            and getattr(self, "normalize_scene", False)
        )

        has_any_prior = overall_prob > 0.0 and (
            ray_dirs_prob > 0.0
            or depth_prob > 0.0
            or world_translation_prob > 0.0
            or world_rotation_prob > 0.0
        )
        with_prior = None if has_any_prior else False

        depths = None
        depth_mask = None
        intrinsics = None
        rays = None
        if depth_prob > 0.0 and _all_have(views, "depthmap"):
            depths = torch.stack(
                [_safe_depth_to_hw(view["depthmap"].to(device)) for view in views],
                dim=1,
            )
            depth_mask = _stack_optional_masks(views, "depth_prior_mask", device=device)
            depths = _maybe_sparsify_depths_like_mapanything(
                depths,
                sparse_depth_prob=sparse_depth_prob,
                sparsification_removal_percent=sparsification_removal_percent,
            )
        if ray_dirs_prob > 0.0:
            if _all_have(views, "ray_directions"):
                rays = _stack_views(views, "ray_directions", device=device)
            elif _all_have(views, "camera_intrinsics"):
                intrinsics = _stack_views(views, "camera_intrinsics", device=device)

        world_translations = None
        world_translation_mask = None
        world_rotations = None
        world_rotation_mask = None
        needs_world_pose = (
            world_translation_prob > 0.0
            or world_rotation_prob > 0.0
            or depth_prob > 0.0
            or return_scene_normalization
            or (
                self.training
                and self.use_world_rotation_prior
                and self.force_rotation_prior_for_degenerate_translation
                and world_translation_prob > 0.0
            )
        )
        if needs_world_pose:
            has_explicit_translation = _any_have(views, "world_translation")
            has_explicit_rotation = _any_have(views, "world_rotation")
            has_camera_pose = _all_have(views, "camera_pose")

            if has_explicit_translation and not _all_have(views, "world_translation"):
                raise ValueError("If any view provides world_translation, all views must provide it.")
            if has_explicit_rotation and not _all_have(views, "world_rotation"):
                raise ValueError("If any view provides world_rotation, all views must provide it.")
            if not (has_explicit_translation or has_explicit_rotation or has_camera_pose):
                raise ValueError(
                    "GeoFF3D needs explicit world_translation/world_rotation "
                    "or camera_pose when world translation, world rotation, world-scale "
                    "depth, or degenerate-translation rotation fallback priors can be used."
                )

            if world_translation_prob > 0.0 or depth_prob > 0.0 or return_scene_normalization:
                if has_explicit_translation:
                    world_translations = _stack_views(views, "world_translation", device=device)
                    world_translation_mask = _stack_optional_masks(views, "world_translation_mask", device=device)
                elif has_camera_pose:
                    world_poses = _stack_views(views, "camera_pose", device=device)
                    world_translations = world_poses[..., :3, 3]
                else:
                    raise ValueError("World translation/depth prior requires world_translation or camera_pose.")
                if world_translation_prob > 0.0:
                    world_translations = self._apply_world_translation_prior_jitter(
                        world_translations
                    )
            if self.use_world_rotation_prior and (
                world_rotation_prob > 0.0
                or (
                    self.training
                    and self.force_rotation_prior_for_degenerate_translation
                    and world_translation_prob > 0.0
                )
            ):
                if has_explicit_rotation:
                    world_rotations = _stack_views(views, "world_rotation", device=device)
                    world_rotation_mask = _stack_optional_masks(views, "world_rotation_mask", device=device)
                elif has_camera_pose:
                    world_poses = _stack_views(views, "camera_pose", device=device)
                    world_rotations = world_poses[..., :3, :3]
                else:
                    raise ValueError("World rotation prior requires world_rotation or camera_pose.")

        model_kwargs = {
            "imgs": images,
            "with_prior": with_prior,
            "depths": depths,
            "depth_mask": depth_mask,
            "intrinsics": intrinsics,
            "rays": rays,
            "world_translations": world_translations,
            "world_translation_mask": world_translation_mask,
            "world_rotations": world_rotations,
            "world_rotation_mask": world_rotation_mask,
            "overall_prob": overall_prob,
            "ray_dirs_prob": ray_dirs_prob,
            "depth_prob": depth_prob,
            "world_translation_prob": world_translation_prob,
            "world_rotation_prob": world_rotation_prob,
            "return_gs_features": bool(getattr(self, "return_gs_features", False)),
            "return_scene_normalization": return_scene_normalization,
        }
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        with torch.autocast("cuda", dtype=self.dtype, enabled=torch.cuda.is_available()):
            results = self.model(**model_kwargs)

        with torch.autocast("cuda", enabled=False):
            preds = self._convert_outputs(results, num_views)
            if bool(getattr(self, "return_raw_results", False)):
                preds[0]["_raw_results"] = results
            return preds


class GeoFF3DCameraTokenWrapper(GeoFF3DWrapper):
    model_cls = GeoFF3DCameraToken
    model_display_name = "GeoFF3DCameraToken"
