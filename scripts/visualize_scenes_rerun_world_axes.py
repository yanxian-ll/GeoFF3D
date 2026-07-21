#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-scene Rerun exporter with world axes enabled by default.

This is a convenience entrypoint around scripts/visualize_scenes_rerun.py. It
supports scenes that have images/cams but no depth: in that case it logs camera
axes, centers, trajectory/frustums, optional RGB images, and the world-axis
marker, but skips point-cloud generation.

Example:
    python scripts/visualize_scenes_rerun_world_axes.py \
      --scene /opt/data/private/dataset/data/NPU_Dronemap/gopro-npu-kfs \
      --save_rrd experiments/debug/gopro-npu-kfs.rrd \
      --show_frustum --show_traj --log_selected_images 8
"""

from __future__ import annotations

import sys

import visualize_scenes_rerun as scene_rerun


def main() -> None:
    if "--show_world_axes" not in sys.argv:
        sys.argv.append("--show_world_axes")
    scene_rerun.main()


if __name__ == "__main__":
    main()
