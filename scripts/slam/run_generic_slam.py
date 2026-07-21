#!/usr/bin/env python
"""Run the Generic Feed-forward SLAM dummy pipeline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import numpy as np

from slam.backend.no_opt_backend import NoOptBackend
from slam.core.config import load_config
from slam.core.data_types import SlamFrame
from slam.core.generic_slam import GenericFeedForwardSLAM
from slam.core.registry import build_adapter
from slam.frontend.chunk_manager import ChunkManager
from slam.mapping.pointcloud_map import PointCloudMap


def build_dummy_frames(num_frames=10):
    return [
        SlamFrame(
            frame_id=i,
            timestamp=float(i),
            image=np.zeros((2, 2, 3), dtype=float),
            world_translation=np.array([float(i), 0.0, 0.0]),
        )
        for i in range(num_frames)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    adapter = build_adapter(cfg["model"]["name"], {k: v for k, v in cfg["model"].items() if k != "name"})
    frontend = cfg["frontend"]
    slam = GenericFeedForwardSLAM(
        adapter=adapter,
        chunk_manager=ChunkManager(
            chunk_size=frontend["chunk_size"],
            overlap=frontend.get("overlap", 0),
            min_chunk_size=frontend.get("min_chunk_size", 1),
        ),
        backend=NoOptBackend(),
        mapping=PointCloudMap(),
    )
    result = slam.run(build_dummy_frames(cfg.get("input", {}).get("num_frames", 10)))
    print(f"trajectory_frames: {len(result['trajectory'])}")
    print(f"map_summary: {result['map_summary']}")
    print(f"backend_diagnostics: {result['backend_diagnostics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
