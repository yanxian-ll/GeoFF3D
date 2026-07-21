import os
from pathlib import Path
import torch

_ORIGINAL_TORCH_HUB_LOAD = torch.hub.load
_ALREADY_PATCHED = False


def _resolve_repo_relative(path):
    if not path:
        return path
    path = Path(os.path.expanduser(str(path)))
    if path.is_absolute():
        return str(path)
    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root / path).resolve())


def configure_torch_hub(machine_cfg):
    global _ALREADY_PATCHED

    if getattr(machine_cfg, "torch_hub_disable_download", False):
        os.environ["TORCH_HUB_DISABLE_DOWNLOAD"] = "1"

    hub_dir = getattr(machine_cfg, "torch_hub_dir", None)
    if hub_dir:
        hub_dir = _resolve_repo_relative(hub_dir)
        os.environ["TORCH_HOME"] = str(os.path.dirname(hub_dir))
        torch.hub.set_dir(hub_dir)

    local_dino_repo = getattr(machine_cfg, "local_dino_repo", None)
    if local_dino_repo:
        local_dino_repo = _resolve_repo_relative(local_dino_repo)
    if local_dino_repo and not _ALREADY_PATCHED:

        def offline_torch_hub_load(repo_or_dir, model, *args, **kwargs):
            if repo_or_dir == "facebookresearch/dinov2":
                print("Redirecting DINOv2 torch.hub.load to local repo")
                repo_or_dir = local_dino_repo
                kwargs["source"] = "local"
            return _ORIGINAL_TORCH_HUB_LOAD(repo_or_dir, model, *args, **kwargs)

        torch.hub.load = offline_torch_hub_load
        _ALREADY_PATCHED = True
        
