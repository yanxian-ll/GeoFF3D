"""
Local VGGT source path utility for VGGT-SLAM 2.0.

Ensures the correct VGGT source tree (MIT-SPARK/VGGT_SPARK) is on sys.path
before importing, so different VGGT-SLAM versions can coexist without conflict.
"""

import os
import sys
from pathlib import Path


def _expand_path(path):
    if not path:
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def setup_local_vggt_path(vggt_repo=None, default_relative_path=None):
    """
    Put a local VGGT source repo at the beginning of sys.path.

    Priority:
      1. vggt_repo argument
      2. VGGT_REPO environment variable
      3. default_relative_path relative to the current VGGT-SLAM root

    Returns the resolved Path.
    """
    this_file = Path(__file__).resolve()
    vggt_slam_root = this_file.parents[1]  # third_party/vggt-slam2.0

    candidates = [
        vggt_repo,
        os.environ.get("VGGT_REPO"),
    ]

    if default_relative_path:
        candidates.append(vggt_slam_root / default_relative_path)

    for candidate in candidates:
        path = _expand_path(candidate)
        if path and path.exists() and (path / "vggt").is_dir():
            path_str = str(path.resolve())
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            print(f"[VGGT-SLAM2] Using local VGGT source: {path_str}")
            return path

    raise FileNotFoundError(
        "Cannot find local VGGT source repo. "
        "Please set --vggt_repo or VGGT_REPO. "
        "The repo path must contain a `vggt/` Python package directory."
    )


def build_vggt_from_local_path(vggt_repo=None, default_relative_path=None):
    """Set up local VGGT path and return a fresh VGGT() model instance."""
    setup_local_vggt_path(
        vggt_repo=vggt_repo,
        default_relative_path=default_relative_path,
    )

    from vggt.models.vggt import VGGT

    return VGGT()