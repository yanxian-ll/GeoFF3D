"""
Local weight loading utilities for VGGT-SLAM.

Allows loading VGGT-1B and SALAD (DINOv2) weights from local paths,
avoiding automatic downloads from HuggingFace / torch.hub.

Typical local directory layout under the project root:
  checkpoints/
    vggt/
      model.pt                # VGGT-1B weights
    torch_cache/
      hub/                    # torch.hub cache dir
        checkpoints/
          dino_salad.ckpt     # SALAD checkpoint (if available locally)
        facebookresearch_dinov2_main/  # local DINOv2 repo clone

Environment variables (all optional):
  VGGT_SLAM_TORCH_HUB_DIR   – override torch.hub directory
  VGGT_SLAM_LOCAL_DINO_REPO – override local DINOv2 repo path
  VGGT_SLAM_LOCAL_SALAD_REPO – override local SALAD repo path
  VGGT_MODEL_PATH            – override VGGT-1B model.pt path
  DINO_SALAD_CKPT            – override SALAD dino_salad.ckpt path
  VGGT_SLAM_OFFLINE          – if "1", forbid online fallback
"""

import os
from pathlib import Path

import torch

# ── Find project root relative to this file ──────────────────────────
_LOCAL_WEIGHTS_DIR = Path(__file__).resolve().parent
_VGGT_SLAM_DIR = _LOCAL_WEIGHTS_DIR.parent  # third_party/vggt-slam
_PROJECT_ROOT = _VGGT_SLAM_DIR.parent.parent.parent

# Default local paths (relative to project root)
_DEFAULT_TORCH_HUB_DIR = _PROJECT_ROOT / "checkpoints" / "torch_cache" / "hub"
_DEFAULT_LOCAL_DINO_REPO = _DEFAULT_TORCH_HUB_DIR / "facebookresearch_dinov2_main"
_DEFAULT_VGGT_MODEL_PATH = _PROJECT_ROOT / "checkpoints" / "vggt" / "model.pt"
_DEFAULT_SALAD_CKPT = _DEFAULT_TORCH_HUB_DIR / "checkpoints" / "dino_salad.ckpt"

VGGT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

_ORIGINAL_TORCH_HUB_LOAD = torch.hub.load
_ALREADY_PATCHED = False


# ── Helpers ──────────────────────────────────────────────────────────


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _expand_path(path):
    if not path:
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


# ── Main configuration ───────────────────────────────────────────────


def configure_vggt_slam_runtime(
    torch_hub_dir=None,
    local_dino_repo=None,
    local_salad_repo=None,
    offline=None,
):
    """
    Configure torch.hub for VGGT-SLAM offline/local execution.

    Call this BEFORE creating Solver/VGGT/ImageRetrieval instances.

    Parameters:
      torch_hub_dir    – path to local torch.hub directory
      local_dino_repo  – path to local facebookresearch/dinov2 repo
      local_salad_repo – path to local serizba/salad repo
      offline          – if True, do not allow online hub fallback
    """
    global _ALREADY_PATCHED

    torch_hub_dir = (
        torch_hub_dir
        or os.environ.get("VGGT_SLAM_TORCH_HUB_DIR")
        or str(_DEFAULT_TORCH_HUB_DIR)
    )
    local_dino_repo = (
        local_dino_repo
        or os.environ.get("VGGT_SLAM_LOCAL_DINO_REPO")
        or str(_DEFAULT_LOCAL_DINO_REPO)
    )
    local_salad_repo = local_salad_repo or os.environ.get(
        "VGGT_SLAM_LOCAL_SALAD_REPO"
    )
    offline = (
        _env_flag("VGGT_SLAM_OFFLINE", False) if offline is None else offline
    )

    hub_dir = _expand_path(torch_hub_dir)
    if hub_dir:
        hub_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_dir))

    if _ALREADY_PATCHED:
        return

    def offline_torch_hub_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2" and local_dino_repo:
            dino_path = _expand_path(local_dino_repo)
            if dino_path and dino_path.is_dir():
                print(
                    "[VGGT-SLAM] Redirect DINOv2 torch.hub.load to local repo: "
                    f"{dino_path}"
                )
                repo_or_dir = str(dino_path)
                kwargs["source"] = "local"

        elif repo_or_dir == "serizba/salad" and local_salad_repo:
            salad_path = _expand_path(local_salad_repo)
            if salad_path and salad_path.is_dir():
                print(
                    "[VGGT-SLAM] Redirect SALAD torch.hub.load to local repo: "
                    f"{salad_path}"
                )
                repo_or_dir = str(salad_path)
                kwargs["source"] = "local"

        elif offline and repo_or_dir in {
            "facebookresearch/dinov2",
            "serizba/salad",
        }:
            name_map = {
                "facebookresearch/dinov2": "DINOv2",
                "serizba/salad": "SALAD",
            }
            raise RuntimeError(
                f"VGGT-SLAM offline mode is enabled, but no local repo is "
                f"configured for {name_map.get(repo_or_dir, repo_or_dir)}."
            )

        return _ORIGINAL_TORCH_HUB_LOAD(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = offline_torch_hub_load
    _ALREADY_PATCHED = True


# ── VGGT-1B weights ──────────────────────────────────────────────────


def resolve_vggt_model_path(cli_path=None):
    """
    Resolve local VGGT-1B model.pt path.

    Priority:
      1. CLI path
      2. VGGT_MODEL_PATH env var
      3. Default: checkpoints/vggt/model.pt
    """
    candidates = [
        cli_path,
        os.environ.get("VGGT_MODEL_PATH"),
        str(_DEFAULT_VGGT_MODEL_PATH),
    ]

    for candidate in candidates:
        path = _expand_path(candidate)
        if path and path.is_file():
            return path

    return None


def load_vggt_weights(model, model_path=None, offline=None):
    """
    Load VGGT weights from local path first.
    Fallback to HuggingFace URL only when offline mode is disabled.
    """
    offline = _env_flag("VGGT_SLAM_OFFLINE", False) if offline is None else offline

    ckpt_path = resolve_vggt_model_path(model_path)
    if ckpt_path:
        print(f"[VGGT-SLAM] Loading VGGT-1B weights from local path: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location="cpu")

        # Handle checkpoint wrappers (state_dict / model key)
        if isinstance(state, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break

        # Strip "module." prefix if present (from DataParallel checkpoints)
        if isinstance(state, dict) and any(
            k.startswith("module.") for k in state.keys()
        ):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        # MapAnything fine-tuning checkpoints contain only the trained wrapper
        # submodule. Initialize untouched VGGT heads from the original weights,
        # then overlay the fine-tuned ``model.*`` parameters.
        wrapped_finetune = isinstance(state, dict) and any(
            k.startswith("model.") for k in state
        )
        if wrapped_finetune:
            base_state = torch.load(str(_DEFAULT_VGGT_MODEL_PATH), map_location="cpu")
            model.load_state_dict(base_state, strict=True)
            state = {
                k.replace("model.", "", 1): v
                for k, v in state.items()
                if k.startswith("model.")
            }

        incompatible = model.load_state_dict(state, strict=False)
        print(f"[VGGT-SLAM] Checkpoint load result: {incompatible}")
        return model

    if offline:
        raise FileNotFoundError(
            "VGGT local model.pt not found. "
            "Please set VGGT_MODEL_PATH or ensure checkpoints/vggt/model.pt exists."
        )

    print(
        "[VGGT-SLAM] Local VGGT weights not found. "
        f"Downloading from: {VGGT_MODEL_URL}"
    )
    model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_MODEL_URL))
    return model


# ── SALAD (DINOv2) checkpoint ────────────────────────────────────────


def resolve_salad_ckpt_path(cli_path=None):
    """
    Resolve local dino_salad.ckpt path.

    Priority:
      1. CLI path
      2. DINO_SALAD_CKPT env var
      3. {torch.hub.get_dir()}/checkpoints/dino_salad.ckpt
      4. Default: checkpoints/torch_cache/hub/checkpoints/dino_salad.ckpt
    """
    candidates = [
        cli_path,
        os.environ.get("DINO_SALAD_CKPT"),
        str(_DEFAULT_SALAD_CKPT),
        Path(torch.hub.get_dir()) / "checkpoints" / "dino_salad.ckpt",
    ]

    for candidate in candidates:
        path = _expand_path(candidate)
        if path and path.is_file():
            return path

    return None


def ensure_salad_ckpt(ckpt_path=None, offline=None):
    """
    Return local SALAD checkpoint path.
    If missing and offline=False, fall back to original torch.hub download
    behaviour.
    """
    # Patch torch.hub.load so serizba/salad redirects to local repo if configured
    configure_vggt_slam_runtime()

    offline = _env_flag("VGGT_SLAM_OFFLINE", False) if offline is None else offline

    resolved = resolve_salad_ckpt_path(ckpt_path)
    if resolved:
        print(f"[VGGT-SLAM] Loading SALAD checkpoint from local path: {resolved}")
        return resolved

    if offline:
        raise FileNotFoundError(
            "dino_salad.ckpt not found. "
            "Please set DINO_SALAD_CKPT or place it at "
            "checkpoints/torch_cache/hub/checkpoints/dino_salad.ckpt"
        )

    print(
        "[VGGT-SLAM] Local dino_salad.ckpt not found. "
        "Falling back to torch.hub download."
    )
    _ORIGINAL_TORCH_HUB_LOAD("serizba/salad", "dinov2_salad")

    resolved = Path(torch.hub.get_dir()) / "checkpoints" / "dino_salad.ckpt"
    if not resolved.is_file():
        raise FileNotFoundError(
            f"SALAD checkpoint still not found after torch.hub.load: {resolved}"
        )

    return resolved
