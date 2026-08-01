"""Wrapper for GeoFF3D."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from geoff3d.models.external.pi3.models.geoff3d import GeoFF3D
from geoff3d.models.external.vggt.utils.rotation import mat_to_quat


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
    num_present = sum(key in view for view in views)
    if num_present == 0:
        return None
    if num_present != len(views):
        raise ValueError(
            f"Inference mask {key!r} must be provided for every view or for none; "
            f"got {num_present}/{len(views)} views."
        )
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


def sparsify_depth(
    depths: torch.Tensor,
    sparse_depth_prob: float,
    sparsification_removal_percent: float,
) -> torch.Tensor:
    """Randomly remove valid depth pixels according to the configured policy.

    `depths` is expected to be [B, N, H, W]. One Bernoulli sample decides whether
    sparsification is applied for the current forward pass. If enabled, each view
    removes the configured fraction of currently valid depth pixels, matching the
    configured fraction independently from each view.
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
        gradient_checkpointing: bool = False,
        torch_hub_force_reload: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.geometric_input_config = geometric_input_config
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.load_pretrained_weights = bool(load_pretrained_weights)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.torch_hub_force_reload = bool(torch_hub_force_reload)
        self.dtype = _get_autocast_dtype()

        # only for Training: jitter world translation priors to improve generalization
        self.world_translation_prior_jitter = self._parse_world_translation_prior_jitter(
            _cfg_get(geometric_input_config, "world_translation_prior_jitter", None)
        )

        if self.load_pretrained_weights:
            if not self.torch_hub_force_reload:
                print(f"Loading {self.model_display_name} base weights from {pretrained_model_name_or_path} ...")
                self.model = self.model_cls.from_pretrained(
                    pretrained_model_name_or_path,
                    gradient_checkpointing=gradient_checkpointing,
                )
            else:
                self.model = self.model_cls.from_pretrained(
                    "yyfz233/Pi3X",
                    force_download=True,
                    gradient_checkpointing=gradient_checkpointing,
                )
        else:
            self.model = self.model_cls(
                gradient_checkpointing=gradient_checkpointing,
            )

    @staticmethod
    def _parse_world_translation_prior_jitter(cfg: Optional[Dict]) -> Dict:
        parsed = {
            "enabled": bool(_cfg_get(cfg, "enabled", False)),
            "apply_prob": float(_cfg_get(cfg, "apply_prob", 0.5)),
            "std_xy": float(_cfg_get(cfg, "std_xy", 0.05)),
            "std_z": float(_cfg_get(cfg, "std_z", 0.10)),
            "bias_std_xy": float(_cfg_get(cfg, "bias_std_xy", 0.02)),
            "bias_std_z": float(_cfg_get(cfg, "bias_std_z", 0.05)),
        }
        if not 0.0 <= parsed["apply_prob"] <= 1.0:
            raise ValueError("world_translation_prior_jitter.apply_prob must be in [0, 1]")
        for key in ("std_xy", "std_z", "bias_std_xy", "bias_std_z"):
            if parsed[key] < 0.0:
                raise ValueError(f"world_translation_prior_jitter.{key} must be non-negative")
        return parsed

    def _translation_jitter_scale(self, translations: torch.Tensor) -> torch.Tensor:
        one = torch.ones(
            translations.shape[0],
            1,
            1,
            device=translations.device,
            dtype=translations.dtype,
        )
        centered = translations - translations.mean(dim=1, keepdim=True)
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
        jitter_scale = self._translation_jitter_scale(translations_f)
        delta = (view_delta + scene_delta) * jitter_scale
        return translations + (delta * apply_mask).to(dtype=translations.dtype)

    def forward(self, views: List[Dict]):
        device = views[0]["img"].device
        num_views = len(views)
        assert views[0]["data_norm_type"][0] == "identity"

        images = torch.stack([view["img"] for view in views], dim=1)
        cfg = self.geometric_input_config

        ray_dirs_prob = _cfg_float(cfg, "ray_dirs_prob", 0.0)
        depth_prob = _cfg_float(cfg, "depth_prob", 0.0)
        sparse_depth_prob = _cfg_float(cfg, "sparse_depth_prob", 0.0)
        sparsification_removal_percent = _cfg_float(cfg, "sparsification_removal_percent", 0.0)
        world_rotation_prob = _cfg_float(cfg, "world_rotation_prob", 0.0)
        world_translation_prob = _cfg_float(cfg, "world_translation_prob", 0.0)

        if world_translation_prob <= 0.0:
            raise ValueError(
                "GeoFF3D requires model.task.world_translation_prob > 0, "
                f"got {world_translation_prob}."
            )

        depths = None
        depth_mask = None
        intrinsics = None
        rays = None
        ray_dirs_mask = None
        if depth_prob > 0.0 and _all_have(views, "depthmap"):
            depths = torch.stack(
                [_safe_depth_to_hw(view["depthmap"].to(device)) for view in views],
                dim=1,
            )
            if not self.training:
                depth_mask = _stack_optional_masks(views, "depth_prior_mask", device=device)
            if self.training:
                depths = sparsify_depth(
                    depths,
                    sparse_depth_prob=sparse_depth_prob,
                    sparsification_removal_percent=sparsification_removal_percent,
                )
        if ray_dirs_prob > 0.0:
            if _all_have(views, "ray_directions"):
                rays = _stack_views(views, "ray_directions", device=device)
            elif _all_have(views, "camera_intrinsics"):
                intrinsics = _stack_views(views, "camera_intrinsics", device=device)
            if not self.training:
                ray_dirs_mask = _stack_optional_masks(views, "ray_dirs_mask", device=device)

        world_translation_mask = None
        world_rotations = None
        world_rotation_mask = None
        has_explicit_translation = _any_have(views, "world_translation")
        has_explicit_rotation = _any_have(views, "world_rotation")
        has_camera_pose = _all_have(views, "camera_pose")

        if has_explicit_translation and not _all_have(views, "world_translation"):
            raise ValueError("If any view provides world_translation, all views must provide it.")
        if has_explicit_rotation and not _all_have(views, "world_rotation"):
            raise ValueError("If any view provides world_rotation, all views must provide it.")

        if has_explicit_translation:
            world_translations = _stack_views(views, "world_translation", device=device)
        elif has_camera_pose:
            world_poses = _stack_views(views, "camera_pose", device=device)
            world_translations = world_poses[..., :3, 3]
        else:
            raise ValueError("GeoFF3D requires world_translation or camera_pose for every view.")
        if not self.training:
            world_translation_mask = _stack_optional_masks(
                views, "world_translation_mask", device=device
            )

        # Training：Apply jitter to world translations
        world_translations = self._apply_world_translation_prior_jitter(world_translations)

        if world_rotation_prob > 0.0:
            if has_explicit_rotation:
                world_rotations = _stack_views(views, "world_rotation", device=device)
            elif has_camera_pose:
                world_poses = _stack_views(views, "camera_pose", device=device)
                world_rotations = world_poses[..., :3, :3]
            else:
                raise ValueError("World rotation prior requires world_rotation or camera_pose.")
            if not self.training:
                world_rotation_mask = _stack_optional_masks(
                    views, "world_rotation_mask", device=device
                )

        model_kwargs = {
            "imgs": images,
            "depths": depths,
            "depth_mask": depth_mask,
            "intrinsics": intrinsics,
            "rays": rays,
            "ray_dirs_mask": ray_dirs_mask,
            "world_translations": world_translations,
            "world_translation_mask": world_translation_mask,
            "world_rotations": world_rotations,
            "world_rotation_mask": world_rotation_mask,
            "ray_dirs_prob": ray_dirs_prob,
            "depth_prob": depth_prob,
            "world_translation_prob": world_translation_prob,
            "world_rotation_prob": world_rotation_prob,
            "return_gs_features": bool(getattr(self, "return_gs_features", False)),
        }
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        with torch.autocast("cuda", dtype=self.dtype, enabled=torch.cuda.is_available()):
            results = self.model(**model_kwargs)

        with torch.autocast("cuda", enabled=False):
            preds = self._convert_outputs(results, num_views)
            if bool(getattr(self, "return_raw_results", False)):
                preds[0]["_raw_results"] = results
            return preds
