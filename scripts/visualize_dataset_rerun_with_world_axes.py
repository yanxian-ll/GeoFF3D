#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset Rerun exporter with world axes enabled by default.

This is a convenience entrypoint around scripts/visualize_dataset_rerun.py. It
exports one .rrd per scene and supports scenes that have images/cams but no
depth: in that case it logs camera axes, centers, trajectory/frustums, optional
RGB images, and the world-axis marker, but skips point-cloud generation.

Example:
    python scripts/visualize_dataset_rerun_with_world_axes.py \
      --dataset_root /path/to/dataset \
      --output_dir experiments/dataset_viz/no_depth_ok \
      --show_frustum --show_traj --log_selected_images 8
"""

from __future__ import annotations

import sys

import visualize_dataset_rerun as dataset_rerun


def main() -> None:
    if "--show_world_axes" not in sys.argv:
        sys.argv.append("--show_world_axes")
    dataset_rerun.main()


if __name__ == "__main__":
    main()
