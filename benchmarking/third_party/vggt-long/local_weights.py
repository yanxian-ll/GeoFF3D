"""Local weight path helpers for running VGGT-Long inside map-anything.

The upstream VGGT-Long config expects weights under ``third_party/vggt-long/weights``.
This project keeps shared checkpoints under the repository root instead:

  checkpoints/
    vggt/model.pt
    torch_cache/hub/checkpoints/dino_salad.ckpt
    torch_cache/hub/checkpoints/dinov2_vitb14_pretrain.pth

Environment variables can override the defaults:
  VGGT_MODEL_PATH, DINO_SALAD_CKPT, DINOV2_CKPT, ORBVOC_PATH,
  VGGT_LONG_TORCH_HUB_DIR, VGGT_LONG_OFFLINE
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


_VGGT_LONG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _VGGT_LONG_DIR.parent.parent.parent

_DEFAULT_TORCH_HUB_DIR = _PROJECT_ROOT / "checkpoints" / "torch_cache" / "hub"
_DEFAULT_VGGT_MODEL_PATH = _PROJECT_ROOT / "checkpoints" / "vggt" / "model.pt"
_DEFAULT_SALAD_CKPT = _DEFAULT_TORCH_HUB_DIR / "checkpoints" / "dino_salad.ckpt"
_DEFAULT_DINOV2_CKPT = _DEFAULT_TORCH_HUB_DIR / "checkpoints" / "dinov2_vitb14_pretrain.pth"
_DEFAULT_ORBVOC_PATH = _PROJECT_ROOT / "checkpoints" / "torch_cache" / "hub" / "checkpoints" / "ORBvoc.txt"

VGGT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _expand_path(path: Any) -> Optional[Path]:
    if path in (None, ""):
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _resolve_existing_path(*candidates: Any) -> Optional[Path]:
    for candidate in candidates:
        path = _expand_path(candidate)
        if path and path.is_file():
            return path.resolve()
    return None


def configure_vggt_long_runtime(torch_hub_dir: Any = None) -> None:
    hub_dir = _expand_path(
        torch_hub_dir
        or os.environ.get("VGGT_LONG_TORCH_HUB_DIR")
        or os.environ.get("VGGT_SLAM_TORCH_HUB_DIR")
        or _DEFAULT_TORCH_HUB_DIR
    )
    if hub_dir:
        hub_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_dir))


def resolve_vggt_model_path(config_path: Any = None) -> Optional[Path]:
    return _resolve_existing_path(
        os.environ.get("VGGT_MODEL_PATH"),
        config_path,
        _DEFAULT_VGGT_MODEL_PATH,
    )


def resolve_salad_ckpt_path(config_path: Any = None) -> Optional[Path]:
    return _resolve_existing_path(
        os.environ.get("DINO_SALAD_CKPT"),
        config_path,
        _DEFAULT_SALAD_CKPT,
        Path(torch.hub.get_dir()) / "checkpoints" / "dino_salad.ckpt",
    )


def resolve_dinov2_ckpt_path(config_path: Any = None) -> Optional[Path]:
    return _resolve_existing_path(
        os.environ.get("DINOV2_CKPT"),
        config_path,
        _DEFAULT_DINOV2_CKPT,
        Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vitb14_pretrain.pth",
    )


def resolve_orbvoc_path(config_path: Any = None) -> Optional[Path]:
    return _resolve_existing_path(
        os.environ.get("ORBVOC_PATH"),
        config_path,
        _DEFAULT_ORBVOC_PATH,
    )


def apply_local_weight_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate VGGT-Long config so it follows this repository's checkpoint layout."""
    configure_vggt_long_runtime()

    weights = config.setdefault("Weights", {})
    resolvers = {
        "VGGT": resolve_vggt_model_path,
        "SALAD": resolve_salad_ckpt_path,
        "DNIO": resolve_dinov2_ckpt_path,
        "DBoW": resolve_orbvoc_path,
    }

    for key, resolver in resolvers.items():
        resolved = resolver(weights.get(key))
        if resolved is not None:
            weights[key] = str(resolved)
            print(f"[VGGT-Long] Using local {key} checkpoint: {resolved}")

    return config


def load_vggt_weights(model: torch.nn.Module, model_path: Any = None) -> torch.nn.Module:
    offline = _env_flag("VGGT_LONG_OFFLINE", _env_flag("VGGT_SLAM_OFFLINE", False))
    ckpt_path = resolve_vggt_model_path(model_path)

    if ckpt_path:
        print(f"[VGGT-Long] Loading VGGT-1B weights from local path: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        # MapAnything fine-tuning checkpoints contain only the trained wrapper
        # submodule. Initialize untouched VGGT heads from the original weights,
        # then overlay the fine-tuned ``model.*`` parameters.
        wrapped_finetune = isinstance(state, dict) and any(
            k.startswith("model.") for k in state
        )
        if wrapped_finetune:
            base_state = torch.load(str(_DEFAULT_VGGT_MODEL_PATH), map_location="cpu")
            model.load_state_dict(base_state, strict=False)
            state = {
                k.replace("model.", "", 1): v
                for k, v in state.items()
                if k.startswith("model.")
            }
        incompatible = model.load_state_dict(state, strict=False)
        print(f"[VGGT-Long] Checkpoint load result: {incompatible}")
        return model

    if offline:
        raise FileNotFoundError(
            "VGGT local model.pt not found. Set VGGT_MODEL_PATH or place it at "
            f"{_DEFAULT_VGGT_MODEL_PATH}."
        )

    print(f"[VGGT-Long] Local VGGT weights not found. Downloading from: {VGGT_MODEL_URL}")
    model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_MODEL_URL), strict=False)
    return model
