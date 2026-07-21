#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate reconstruction after aligning prediction to GT.

This script keeps evaluation as a separate shell step. It reads:
  eval/pred_cameras.npz
  eval/pred_points.ply

Then it independently aligns:
  1. predicted poses to GT poses by Sim3 on matched camera centers
  2. fused predicted point cloud to GT RGB-D point cloud by trimmed NN ICP

If no GT point cloud can be built, point-cloud metrics are marked invalid and
pose metrics are still computed when GT poses exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import torch

from geoff3d.spatial_rrd.metrics import (
    compute_aligned_metrics,
    load_eval_outputs,
    parse_float_list,
    parse_int_list,
)
from geoff3d.spatial_rrd.scene_io import build_views_from_scene


def restrict_meta_to_stems(meta: Dict[str, object], stems: Sequence[str]) -> Dict[str, object]:
    available = set(str(s) for s in meta.get("stems", []))
    keep = [str(s) for s in stems if str(s) in available]
    if not keep:
        keep = [str(s) for s in meta.get("stems", [])]

    out = dict(meta)
    out["stems"] = keep
    for key in ("image_paths", "depth_paths", "cam_paths", "cams"):
        value = meta.get(key, {})
        if isinstance(value, dict):
            out[key] = {stem: value[stem] for stem in keep if stem in value}

    cams = out.get("cams", {})
    depths = out.get("depth_paths", {})
    out["num_cam_priors"] = int(
        sum(1 for stem in keep if isinstance(cams, dict) and stem in cams)
    )
    out["num_depth_priors"] = int(
        sum(1 for stem in keep if isinstance(depths, dict) and stem in depths)
    )
    return out


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None)

    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--cams_dir", default="cams")
    parser.add_argument("--depth_dir", default="depth")
    parser.add_argument("--frame_glob", default="*")
    parser.add_argument("--num_views", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_side", type=int, default=518)
    parser.add_argument("--size_multiple", type=int, default=14)

    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--depth_min", type=float, default=1e-6)
    parser.add_argument("--depth_max", type=float, default=1e6)

    parser.add_argument("--thresholds", default="0.5,1.0,2.0,5.0")
    parser.add_argument("--rpe_steps", default="1,5,10")
    parser.add_argument("--max_gt_points", type=int, default=300000)
    parser.add_argument("--max_pred_points_eval", type=int, default=300000)
    parser.add_argument("--max_gt_points_eval", type=int, default=300000)
    parser.add_argument("--max_align_points", type=int, default=100000)
    parser.add_argument("--icp_iterations", type=int, default=8)
    parser.add_argument("--icp_trim_quantile", type=float, default=0.7)
    parser.add_argument("--gt_io_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else eval_dir / "metrics.json"
    )
    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else eval_dir / "metrics_summary.csv"
    )

    pred = load_eval_outputs(eval_dir)
    _views, meta = build_views_from_scene(
        scene_dir=scene_dir,
        images_dir=args.images_dir,
        cams_dir=args.cams_dir,
        depth_dir=args.depth_dir,
        frame_glob=args.frame_glob,
        num_views=args.num_views,
        start=args.start,
        stride=args.stride,
        max_side=args.max_side,
        size_multiple=args.size_multiple,
        depth_scale=args.depth_scale,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        device=torch.device("cpu"),
        show_progress=not bool(args.no_progress),
    )
    meta = restrict_meta_to_stems(meta, pred["stems"])

    metrics = compute_aligned_metrics(
        eval_dir=eval_dir,
        meta=meta,
        thresholds=parse_float_list(args.thresholds),
        rpe_steps=parse_int_list(args.rpe_steps),
        max_pred_points_eval=int(args.max_pred_points_eval),
        max_gt_points_eval=int(args.max_gt_points_eval),
        max_gt_points=int(args.max_gt_points),
        max_align_points=int(args.max_align_points),
        icp_iterations=int(args.icp_iterations),
        icp_trim_quantile=float(args.icp_trim_quantile),
        seed=int(args.seed),
        gt_io_workers=int(args.gt_io_workers),
        output_json=output_json,
        output_csv=output_csv,
    )

    pose = metrics.get("pose", {})
    point_cloud = metrics.get("point_cloud", {})
    print(f"Saved aligned metrics JSON: {output_json}")
    print(f"Saved aligned metrics CSV:  {output_csv}")
    print(
        "[METRICS] "
        f"pose_valid={bool(pose.get('valid', False))}, "
        f"pose_matches={int(pose.get('num_matches', 0))}, "
        f"points_valid={bool(point_cloud.get('valid', False))}, "
        f"gt_points={int(metrics.get('gt', {}).get('num_points', 0))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
