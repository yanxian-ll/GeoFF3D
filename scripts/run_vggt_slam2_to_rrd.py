#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run VGGT-SLAM 2.0 on a MapAnything-format scene and save a Rerun .rrd."""

from __future__ import annotations

import sys

from run_vggt_slam_to_rrd import main


if __name__ == "__main__":
    raise SystemExit(main(["--method", "vggt-slam2.0", *sys.argv[1:]]))
