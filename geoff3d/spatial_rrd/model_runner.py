# -*- coding: utf-8 -*-
"""Model initialization, prior policies, depth prior cache."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import hydra
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from geoff3d.models import init_model
from geoff3d.utils.geometry import (
    get_rays_in_camera_frame,
    quaternion_to_rotation_matrix,
    recover_pinhole_intrinsics_from_ray_directions,
)
from geoff3d.utils.torch_hub_setup import configure_torch_hub

# ---------------------------------------------------------------------------
# Model families
# ---------------------------------------------------------------------------
NO_PRIOR_MODELS = {
    "vggt",
}

INPUT_PRIOR_MODELS = {
    "geoff3d",
    "pi3x",
}

OUR_WORLD_MODELS = {
    "geoff3d",
}

WORLD_TRANSLATION_PRIOR_MODELS = {
    "geoff3d",
}

WORLD_ROTATION_PRIOR_MODELS = {
    "geoff3d",
}

RAY_PRIOR_MODELS = {
    "geoff3d",
    "pi3x",
}

DEPTH_PRIOR_MODELS = {
    "geoff3d",
    "pi3x",
}


def infer_model_family(model_name: str) -> str:
    name = str(model_name).lower().strip()

    if name in OUR_WORLD_MODELS:
        return "ours"

    if name in INPUT_PRIOR_MODELS:
        return "input_prior"

    if name in NO_PRIOR_MODELS:
        return "no_prior"

    # 默认按无先验模型处理，避免错误喂入 prior 造成不稳定
    return "no_prior"


# ---------------------------------------------------------------------------
# Prior policy resolution
# ---------------------------------------------------------------------------
def _resolve_requested_source(requested: str, default: str) -> str:
    requested = str(requested).lower().strip()
    if requested == "auto":
        return default
    return requested


def _has_all_camera_priors(meta: Dict[str, object]) -> bool:
    n = len(meta.get("stems", []))
    return n > 0 and int(meta.get("num_cam_priors", 0)) == n


def _has_any_camera_priors(meta: Dict[str, object]) -> bool:
    return int(meta.get("num_cam_priors", 0)) > 0


def _has_any_depth_priors(meta: Dict[str, object]) -> bool:
    return int(meta.get("num_depth_priors", 0)) > 0


def resolve_prior_policy(args, model_name: str, meta: Dict[str, object]) -> Dict[str, object]:
    family = str(args.model_family)
    if family == "auto":
        family = infer_model_family(model_name)

    has_pose = _has_all_camera_priors(meta)
    has_cam = _has_any_camera_priors(meta)
    has_depth = _has_any_depth_priors(meta)

    policy = {
        "family": family,

        # 完整 pose prior：给 geoff3d / pi3x 使用
        "pose": "none",

        # split pose prior：给 geoff3d 使用
        "translation": "none",
        "rotation": "none",

        # camera intrinsics / ray prior
        "ray": "none",

        # depth prior
        "depth": "none",

        # runtime bootstrap
        "bootstrap_ray": False,
        "bootstrap_depth": False,
    }

    if family == "no_prior":
        policy.update({
            "pose": "none",
            "translation": "none",
            "rotation": "none",
            "ray": "none",
            "depth": "none",
            "bootstrap_ray": False,
            "bootstrap_depth": False,
        })
        return policy

    if family == "input_prior":
        pose_default = "input" if has_pose else "none"
        ray_default = "input" if has_cam else "pred"
        depth_default = "input" if has_depth else "pred"

        pose = _resolve_requested_source(args.pose_prior, pose_default)
        ray = _resolve_requested_source(args.ray_prior, ray_default)
        depth = _resolve_requested_source(args.depth_prior, depth_default)

        if pose == "input" and not has_pose:
            print("[WARN] pose_prior=input requested but not all camera priors exist; use none.")
            pose = "none"

        if ray == "input" and not has_cam:
            print("[WARN] ray_prior=input requested but camera intrinsics are missing; use pred.")
            ray = "pred"

        if depth == "input" and not has_depth:
            print("[WARN] depth_prior=input requested but depth priors are missing; use pred.")
            depth = "pred"

        policy.update({
            "pose": pose,
            "translation": "none",
            "rotation": "none",
            "ray": ray,
            "depth": depth,
            "bootstrap_ray": ray == "pred",
            "bootstrap_depth": depth == "pred",
        })
        return policy

    if family == "ours":
        translation_default = "input" if has_cam else "none"
        rotation_default = "input" if has_cam else "none"
        ray_default = "input" if has_cam else "pred"
        depth_default = "input" if has_depth else "pred"

        translation = _resolve_requested_source(args.translation_prior, translation_default)
        rotation = _resolve_requested_source(args.rotation_prior, rotation_default)
        ray = _resolve_requested_source(args.ray_prior, ray_default)
        depth = _resolve_requested_source(args.depth_prior, depth_default)

        if translation == "input" and not has_cam:
            translation = "none"
        if rotation == "input" and not has_cam:
            rotation = "none"
        if ray == "input" and not has_cam:
            ray = "pred"
        if depth == "input" and not has_depth:
            depth = "pred"

        policy.update({
            "pose": "none",
            "translation": translation,
            "rotation": rotation,
            "ray": ray,
            "depth": depth,
            "bootstrap_ray": ray == "pred",
            "bootstrap_depth": depth == "pred",
        })
        return policy

    raise ValueError(f"Unknown model family: {family}")


# ---------------------------------------------------------------------------
# View prior filtering
# ---------------------------------------------------------------------------
BASE_VIEW_KEYS = {
    "img",
    "is_metric_scale",
    "is_synthetic",
    "true_shape",
    "data_norm_type",
    "label",
    "instance",
    "idx",
}

POSE_KEYS = {
    "camera_pose",
    "camera_pose_quats",
    "camera_pose_trans",
}

TRANSLATION_KEYS = {
    "world_translation",
    "camera_pose_trans",
}

ROTATION_KEYS = {
    "camera_pose",
    "camera_pose_quats",
}

RAY_KEYS = {
    "camera_intrinsics",
    "ray_directions_cam",
}

DEPTH_KEYS = {
    "depthmap",
    "depth_prior_mask",
    "valid_mask",
    "non_ambiguous_mask",
    "pts3d",
    "pts3d_cam",
    "depth_along_ray",
}


def filter_views_for_prior_policy(
    views: Sequence[Dict[str, object]],
    policy: Dict[str, object],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    family = str(policy.get("family", "no_prior"))

    for view in views:
        next_view = {k: v for k, v in view.items() if k in BASE_VIEW_KEYS}

        if family == "input_prior" and policy.get("pose") == "input":
            for k in POSE_KEYS | {"world_translation"}:
                if k in view:
                    next_view[k] = view[k]

        if family == "ours":
            if policy.get("translation") == "input":
                for k in TRANSLATION_KEYS:
                    if k in view:
                        next_view[k] = view[k]
            if policy.get("rotation") == "input":
                for k in ROTATION_KEYS:
                    if k in view:
                        next_view[k] = view[k]

        if policy.get("ray") == "input":
            for k in RAY_KEYS:
                if k in view:
                    next_view[k] = view[k]

        if policy.get("depth") == "input":
            for k in DEPTH_KEYS:
                if k in view:
                    next_view[k] = view[k]

        out.append(next_view)

    return out


# ---------------------------------------------------------------------------
# Build prior overrides (Hydra)
# ---------------------------------------------------------------------------
def build_prior_overrides(model: str, policy: Dict[str, object]) -> List[str]:
    overrides: List[str] = []
    model = str(model)

    def add_prob(task_key: str, enabled: bool) -> None:
        overrides.append(f"model.task.{task_key}={1.0 if enabled else 0.0}")

    family = str(policy.get("family", "no_prior"))

    # 无先验模型：不要传任何 prior override
    if family == "no_prior":
        return overrides

    # 我们自己的 geoff3d：split pose prior
    if model in WORLD_TRANSLATION_PRIOR_MODELS:
        use_t = policy.get("translation") == "input"
        add_prob("world_translation_prob", use_t)
        overrides.append(
            f"model.model_config.use_world_translation_prior={str(use_t).lower()}"
        )

    if model in WORLD_ROTATION_PRIOR_MODELS:
        use_r = policy.get("rotation") == "input"
        add_prob("world_rotation_prob", use_r)
        overrides.append(
            f"model.model_config.use_world_rotation_prior={str(use_r).lower()}"
        )

    # ray/depth：对支持的模型显式打开或关闭
    if model in OUR_WORLD_MODELS:
        add_prob("ray_dirs_prob", policy.get("ray") == "input")
        add_prob("depth_prob", policy.get("depth") == "input")

    return overrides


# ---------------------------------------------------------------------------
# Runtime prior policy application
# ---------------------------------------------------------------------------
def apply_runtime_prior_policy(model: torch.nn.Module, policy: Dict[str, object]) -> None:
    family = str(policy.get("family", "no_prior"))

    if family == "no_prior":
        set_model_task_prob(model, "ray_dirs_prob", False)
        set_model_task_prob(model, "depth_prob", False)
        set_model_task_prob(model, "world_translation_prob", False)
        set_model_task_prob(model, "world_rotation_prob", False)
        return

    set_model_task_prob(model, "ray_dirs_prob", policy.get("ray") == "input")
    set_model_task_prob(model, "depth_prob", policy.get("depth") == "input")
    set_model_task_prob(model, "world_translation_prob", policy.get("translation") == "input")
    set_model_task_prob(model, "world_rotation_prob", policy.get("rotation") == "input")


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------
WRAPPER_PRETRAINED_PATH_FIELDS = {
    "pi3x": "pretrained_model_name_or_path",
    "vggt": "pretrained_model_name_or_path",
}

WRAPPER_CHECKPOINT_PATH_FIELDS = {}

CHECKPOINT_FILE_SUFFIXES = {
    ".pth",
    ".pt",
    ".ckpt",
    ".bin",
    ".safetensors",
}


def _looks_like_weight_file(checkpoint: str) -> bool:
    suffix = Path(str(checkpoint)).suffix.lower()
    return suffix in CHECKPOINT_FILE_SUFFIXES


def _resolve_checkpoint_reference(checkpoint: str) -> str:
    path = Path(str(checkpoint)).expanduser()
    if path.is_absolute() or path.exists():
        return str(path)
    repo_path = Path(__file__).resolve().parents[2] / path
    if repo_path.exists():
        return str(repo_path.resolve())
    return str(checkpoint)


def _is_wrapper_pretrained_reference(model_name: str, checkpoint: Optional[str]) -> bool:
    if not checkpoint:
        return False
    model = str(model_name).lower().strip()
    checkpoint = _resolve_checkpoint_reference(str(checkpoint))
    path = Path(str(checkpoint)).expanduser()
    if path.exists():
        return path.is_dir()
    return model in WRAPPER_PRETRAINED_PATH_FIELDS and not _looks_like_weight_file(str(checkpoint))


def checkpoint_hydra_overrides(
    model_name: str,
    checkpoint: Optional[str],
) -> Tuple[List[str], Optional[str]]:
    """Route wrapper-style checkpoints through model config instead of torch.load."""
    if not checkpoint:
        return [], None

    model = str(model_name).lower().strip()
    checkpoint = _resolve_checkpoint_reference(str(checkpoint))
    if _is_wrapper_pretrained_reference(model, checkpoint):
        field = WRAPPER_PRETRAINED_PATH_FIELDS.get(model)
        if field is not None:
            return [f"model.model_config.{field}={checkpoint}"], None

    field = WRAPPER_CHECKPOINT_PATH_FIELDS.get(model)
    if field is not None:
        return [f"model.model_config.{field}={checkpoint}"], None

    path = Path(checkpoint).expanduser()
    if path.exists() and path.is_dir():
        field = WRAPPER_PRETRAINED_PATH_FIELDS.get(model)
        if field is None:
            raise IsADirectoryError(
                f"--checkpoint points to a directory, but model {model_name!r} "
                "does not expose a wrapper pretrained path field: "
                f"{checkpoint}"
            )
        return [f"model.model_config.{field}={checkpoint}"], None

    return [], checkpoint


def init_model_from_hydra(
    model_name: str,
    machine: str,
    hydra_overrides: Sequence[str],
    device: torch.device,
):
    GlobalHydra.instance().clear()
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "configs"
    hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir))
    overrides = [f"model={model_name}", f"machine={machine}"] + list(
        hydra_overrides
    )
    cfg = hydra.compose(config_name="train", overrides=overrides)
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    configure_torch_hub(cfg.machine)
    model = init_model(
        cfg.model.model_str,
        cfg.model.model_config,
        torch_hub_force_reload=cfg.model.torch_hub_force_reload,
    )

    pretrained = getattr(cfg.model, "pretrained", None)
    if pretrained:
        print(f"Loading checkpoint from config model.pretrained={pretrained}")
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        print(model.load_state_dict(state, strict=False))
        del ckpt

    model.to(device)
    model.eval()
    return model, cfg


def load_checkpoint(model: torch.nn.Module, checkpoint: Optional[str]) -> None:
    if not checkpoint:
        return
    path = Path(str(checkpoint)).expanduser()
    if path.exists() and path.is_dir():
        print(
            "Skipping checkpoint override because it is a directory already "
            f"handled by the model wrapper: {checkpoint}"
        )
        return
    print(f"Loading checkpoint override: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    print(model.load_state_dict(state, strict=False))
    del ckpt


# ---------------------------------------------------------------------------
# Runtime model config tweaks
# ---------------------------------------------------------------------------
def set_model_task_value(
    model: torch.nn.Module, key: str, value: float
) -> None:
    cfg = getattr(model, "geometric_input_config", None)
    if cfg is None:
        return
    try:
        from omegaconf import open_dict

        with open_dict(cfg):
            cfg[key] = float(value)
        return
    except Exception:
        pass
    try:
        cfg[key] = float(value)
    except Exception:
        try:
            setattr(cfg, key, float(value))
        except Exception:
            print(
                f"[WARN] Could not update "
                f"model.geometric_input_config.{key} at runtime."
            )


def set_model_task_prob(
    model: torch.nn.Module, key: str, enabled: bool
) -> None:
    set_model_task_value(model, key, 1.0 if enabled else 0.0)


def set_pi3x_ray_prior_prob(
    model: torch.nn.Module, enabled: bool
) -> None:
    set_model_task_prob(model, "ray_dirs_prob", enabled)


# ---------------------------------------------------------------------------
# Bootstrap intrinsics from predicted rays
# ---------------------------------------------------------------------------
def recover_average_intrinsics_from_pred_rays(
    preds: Sequence[Dict[str, torch.Tensor]],
) -> Optional[torch.Tensor]:
    intrinsics: List[torch.Tensor] = []
    for pred in preds:
        rays = pred.get("ray_directions", None)
        if rays is None:
            continue
        K = recover_pinhole_intrinsics_from_ray_directions(
            rays.float(),
            use_geometric_calculation=True,
        )
        if K.dim() == 2:
            K = K.unsqueeze(0)
        finite = torch.isfinite(K).flatten(1).all(dim=1)
        if bool(finite.any()):
            intrinsics.append(K[finite])

    if not intrinsics:
        return None

    K_mean = torch.cat(intrinsics, dim=0).mean(dim=0)
    if not bool(torch.isfinite(K_mean).all()):
        return None
    return K_mean


def apply_bootstrap_intrinsics_to_views(
    views: Sequence[Dict[str, object]],
    intrinsics: torch.Tensor,
    device: torch.device,
) -> List[Dict[str, object]]:
    K = intrinsics.to(device=device, dtype=torch.float32).unsqueeze(0)
    out: List[Dict[str, object]] = []

    for view in views:
        next_view = dict(view)
        next_view["camera_intrinsics"] = K.clone()

        true_shape = view.get("true_shape", None)
        if true_shape is not None and torch.is_tensor(true_shape):
            h = int(true_shape.reshape(-1)[0].item())
            w = int(true_shape.reshape(-1)[1].item())
            _, rays = get_rays_in_camera_frame(
                K,
                h,
                w,
                normalize_to_unit_sphere=True,
            )
            next_view["ray_directions_cam"] = rays

        out.append(next_view)

    return out


# ---------------------------------------------------------------------------
# Prediction post-processing
# ---------------------------------------------------------------------------
def torch_to_np(x, dtype=np.float32):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach()
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        x = x.cpu().numpy()
    return np.asarray(x, dtype=dtype)


def pred_pose_to_c2w(
    pred: Dict[str, torch.Tensor],
) -> Optional[np.ndarray]:
    if "cam_trans" not in pred or "cam_quats" not in pred:
        return None
    trans = pred["cam_trans"]
    quat = pred["cam_quats"]
    if trans.ndim == 2:
        trans = trans[0]
    if quat.ndim == 2:
        quat = quat[0]
    R_t = quaternion_to_rotation_matrix(quat.float().unsqueeze(0))[0]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = torch_to_np(R_t)
    T[:3, 3] = torch_to_np(trans.float())
    return T


def collect_pred_outputs(
    preds: List[Dict[str, torch.Tensor]],
    rgbs: Sequence[np.ndarray],
    pred_min_depth: float = 1e-6,
    conf_quantile: float = 0.0,
    stems: Optional[Sequence[str]] = None,
    collect_point_indices: Optional[Sequence[int]] = None,
):
    import cv2

    collect_set = None
    if collect_point_indices is not None:
        collect_set = set(int(i) for i in collect_point_indices)

    pred_points_all: List[np.ndarray] = []
    pred_colors_all: List[np.ndarray] = []
    pred_maps: List[np.ndarray] = []
    pred_valid_masks: List[np.ndarray] = []
    pred_cams: List[Dict[str, object]] = []

    for i, pred in enumerate(preds):
        pred_stem = (
            str(stems[i])
            if stems is not None and i < len(stems)
            else f"pred_{i:03d}"
        )

        T = pred_pose_to_c2w(pred)
        if T is not None and np.isfinite(T).all():
            pred_cams.append(
                {"stem": pred_stem, "pred_index": int(i), "T_c2w": T}
            )

        pts = pred.get("pts3d", None)
        if pts is None:
            pred_maps.append(np.empty((0, 0, 3), np.float32))
            pred_valid_masks.append(np.zeros((0, 0), dtype=bool))
            continue

        pts_np = torch_to_np(
            pts[0] if pts.ndim == 4 else pts, dtype=np.float32
        )
        h, w = pts_np.shape[:2]
        rgb = rgbs[i]
        if rgb.shape[0] != h or rgb.shape[1] != w:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

        finite = np.isfinite(pts_np).all(axis=-1)
        if "pts3d_cam" in pred:
            pts_cam = torch_to_np(pred["pts3d_cam"][0], dtype=np.float32)
            if pts_cam.shape[:2] == finite.shape:
                finite &= np.isfinite(pts_cam).all(axis=-1)
                finite &= pts_cam[..., 2] > float(pred_min_depth)

        if "conf" in pred and conf_quantile > 0:
            conf = torch_to_np(pred["conf"][0], dtype=np.float32)

            # 兼容 [H, W, 1] / [H, W]  (pi3x/geoff3d conf 是 [H,W,1])
            if conf.ndim == 3 and conf.shape[-1] == 1:
                conf = conf[..., 0]
            elif conf.ndim == 3:
                # 多通道 conf：取均值比直接 broadcast 安全
                conf = np.nanmean(conf, axis=-1)

            if conf.shape == finite.shape:
                q = float(conf_quantile)
                if not (0.0 <= q <= 1.0):
                    raise ValueError(f"conf_quantile must be in [0, 1], got {q}")
                good = np.isfinite(conf) & finite
                if good.any():
                    thr = np.quantile(conf[good], q)
                    finite &= conf >= thr
            else:
                print(
                    f"[WARN] confidence shape {conf.shape} does not match "
                    f"point mask shape {finite.shape}; skip confidence filtering."
                )

        pred_maps.append(pts_np)
        pred_valid_masks.append(finite)
        if collect_set is None or int(i) in collect_set:
            pred_points_all.append(pts_np[finite].reshape(-1, 3))
            pred_colors_all.append(rgb[finite].reshape(-1, 3).astype(np.uint8))

    points = (
        np.concatenate(pred_points_all, axis=0)
        if pred_points_all
        else np.empty((0, 3), np.float32)
    )
    colors = (
        np.concatenate(pred_colors_all, axis=0)
        if pred_colors_all
        else np.empty((0, 3), np.uint8)
    )
    return points, colors, pred_maps, pred_valid_masks, pred_cams


# ---------------------------------------------------------------------------
# Depth prior cache for seam reuse
# ---------------------------------------------------------------------------
def pred_depth_prior(
    pred: Dict[str, torch.Tensor],
    valid_mask: Optional[np.ndarray] = None,
) -> Optional[torch.Tensor]:
    pts_cam = pred.get("pts3d_cam", None)
    if pts_cam is None or pts_cam.shape[-1] != 3:
        return None

    depth = pts_cam[..., 2]
    if depth.dim() == 3:
        depth = depth[:1]
    elif depth.dim() == 2:
        depth = depth.unsqueeze(0)
    else:
        return None

    depth = depth.detach().float()
    valid = torch.isfinite(depth) & (depth > 0)

    if valid_mask is not None:
        mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=depth.device)
        if mask.ndim == 2 and valid.ndim == 3:
            mask = mask.unsqueeze(0)
        if mask.shape == valid.shape:
            valid = valid & mask
        else:
            print(
                f"[WARN] depth prior valid_mask shape {tuple(mask.shape)} "
                f"does not match depth shape {tuple(valid.shape)}; ignore mask."
            )

    if not bool(valid.any()):
        return None

    return torch.where(valid, depth, torch.zeros_like(depth))


def build_future_overlap_counter(
    chunks: Sequence[Dict[str, object]],
) -> Counter:
    """
    Count how many times each global frame index will be used as overlap in future chunks.
    """
    counter = Counter()
    for chunk in chunks:
        for idx in chunk.get("overlap_indices", []):
            counter[int(idx)] += 1
    return counter


def build_depth_prior_cache(
    preds: Sequence[Dict[str, torch.Tensor]],
    indices: Sequence[int],
    keep_indices: Optional[Set[int]] = None,
    valid_masks: Optional[Sequence[Optional[np.ndarray]]] = None,
) -> Dict[int, torch.Tensor]:
    cache: Dict[int, torch.Tensor] = {}
    keep_indices = set(int(i) for i in keep_indices) if keep_indices is not None else None

    for local_i, (global_idx, pred) in enumerate(zip(indices, preds)):
        global_idx = int(global_idx)
        if keep_indices is not None and global_idx not in keep_indices:
            continue

        valid_mask = None
        if valid_masks is not None and local_i < len(valid_masks):
            valid_mask = valid_masks[local_i]

        depth = pred_depth_prior(pred, valid_mask=valid_mask)
        if depth is not None:
            cache[global_idx] = depth.detach().cpu()

    return cache


def apply_cached_depth_priors_to_views(
    views: Sequence[Dict[str, object]],
    indices: Sequence[int],
    depth_cache: Dict[int, torch.Tensor],
    device: torch.device,
) -> Tuple[List[Dict[str, object]], int]:
    matched = [int(i) for i in indices if int(i) in depth_cache]
    if not matched:
        return list(views), 0

    template = depth_cache[matched[0]].to(device=device, dtype=torch.float32)
    zero_depth = torch.zeros_like(template)
    zero_mask = torch.zeros((1,), device=device, dtype=torch.bool)
    one_mask = torch.ones((1,), device=device, dtype=torch.bool)

    out: List[Dict[str, object]] = []
    used = 0
    for view, global_idx in zip(views, indices):
        prior = depth_cache.get(int(global_idx))
        next_view = dict(view)
        if prior is None:
            next_view["depthmap"] = zero_depth
            next_view["depth_prior_mask"] = zero_mask
        else:
            next_view["depthmap"] = prior.to(device=device, dtype=torch.float32)
            next_view["depth_prior_mask"] = one_mask
            used += 1
        out.append(next_view)
    return out, used


# ---------------------------------------------------------------------------
# Points from maps helper
# ---------------------------------------------------------------------------
def points_from_maps(
    pred_maps: Sequence[np.ndarray],
    pred_valid_masks: Sequence[np.ndarray],
    rgbs: Sequence[np.ndarray],
    local_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    points_all: List[np.ndarray] = []
    colors_all: List[np.ndarray] = []
    for local_i in local_indices:
        point_map = np.asarray(pred_maps[int(local_i)], dtype=np.float32)
        if point_map.ndim != 3 or point_map.shape[-1] != 3:
            continue
        mask = np.asarray(pred_valid_masks[int(local_i)], dtype=bool)
        if mask.shape != point_map.shape[:2]:
            continue
        mask = mask & np.isfinite(point_map).all(axis=-1)
        if not mask.any():
            continue
        rgb = rgbs[int(local_i)]
        if rgb.shape[:2] != point_map.shape[:2]:
            rgb = cv2.resize(
                rgb,
                (point_map.shape[1], point_map.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        points_all.append(point_map[mask].reshape(-1, 3).astype(np.float32))
        colors_all.append(rgb[mask].reshape(-1, 3).astype(np.uint8))

    points = (
        np.concatenate(points_all, axis=0)
        if points_all
        else np.empty((0, 3), np.float32)
    )
    colors = (
        np.concatenate(colors_all, axis=0)
        if colors_all
        else np.empty((0, 3), np.uint8)
    )
    return points, colors
