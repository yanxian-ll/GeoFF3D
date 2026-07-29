import os
from pathlib import Path

import torch


def _resolve_repo_relative(path):
    if not path:
        return path
    path = Path(os.path.expanduser(str(path)))
    if path.is_absolute():
        return str(path)
    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root / path).resolve())


def configure_torch_hub(machine_cfg):
    hub_dir = getattr(machine_cfg, "torch_hub_dir", None)
    if hub_dir:
        hub_dir = _resolve_repo_relative(hub_dir)
        os.environ["TORCH_HOME"] = str(os.path.dirname(hub_dir))
        torch.hub.set_dir(hub_dir)
