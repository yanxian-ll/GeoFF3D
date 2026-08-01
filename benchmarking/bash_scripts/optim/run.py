#!/usr/bin/env python3
"""Run one optimization-based reconstruction method over a scene list."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

BENCHMARK_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))

from common.runtime import aligned_evaluation_command, load_scenes, prediction_and_metrics_complete, run_command


METHODS = {"vggt_long", "vggt_slam2.0", "vggt_slam_sim3", "vggt_slam_sl4"}


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=sorted(METHODS))
    parser.add_argument("--scene-list", type=Path, default=Path(__file__).with_name("vggt_slam_scenes.yaml"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "benchmarking" / "optim",
    )
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def common_args(scene_dir: Path, output_rrd: Path, device: str, stride: int) -> list[str]:
    return [
        "--scene_dir", str(scene_dir), "--output_rrd", str(output_rrd),
        "--device", f"cuda:{device}", "--seed", "0", "--stride", str(stride),
        "--max_side", "518", "--size_multiple", "14",
        "--max_gt_points", "800000", "--max_points_per_view", "500000",
        "--voxel_downsample", "0.01",
    ]


def method_command(
    method: str,
    scene_dir: Path,
    output_rrd: Path,
    device: str,
    stride: int,
    config: dict[str, object],
) -> list[str]:
    shared = common_args(scene_dir, output_rrd, device, stride)
    checkpoint = str(config.get("checkpoint", ""))
    if method == "vggt_long":
        command = [
            sys.executable, str(BENCHMARK_DIR / "adapters" / "vggt_optim.py"),
            "--method", "vggt-long", *shared,
            "--vggt_long_config", env("VGGT_LONG_CONFIG", str(config.get("config", BENCHMARK_DIR / "third_party/vggt-long/configs/ours.yaml"))),
            "--max_pred_points", "500000", "--scene_io_workers", "4",
        ]
        if checkpoint:
            command += ["--vggt_model_path", checkpoint]
        return command
    if method == "vggt_slam2.0":
        command = [
            env("VGGT_SLAM2_PYTHON", sys.executable),
            str(BENCHMARK_DIR / "third_party/vggt-slam2.0/run_scene_to_rrd.py"),
            "--method", "vggt-slam2.0", *shared,
            "--submap_size", env("SUBMAP_SIZE", "16"),
            "--overlapping_window_size", env("OVERLAPPING_WINDOW_SIZE", "2"),
            "--global_point_stride", env("GLOBAL_POINT_STRIDE", "1"),
        ]
        if checkpoint:
            command += ["--vggt_model_path", checkpoint]
        return command
    backend_method = "vggt-slam-sim3" if method == "vggt_slam_sim3" else "vggt-slam-sl4"
    command = [
        env("VGGT_SLAM_PYTHON", sys.executable),
        str(BENCHMARK_DIR / "third_party/vggt-slam/run_scene_to_rrd.py"),
        "--method", backend_method, *shared,
        "--submap_size", env("SUBMAP_SIZE", "16"),
        "--overlapping_window_size", env("OVERLAPPING_WINDOW_SIZE", "2"),
        "--global_point_stride", env("GLOBAL_POINT_STRIDE", "1"),
    ]
    if method == "vggt_slam_sim3":
        command.append("--use_sim3")
    if checkpoint:
        command += ["--vggt_model_path", checkpoint]
    return command


def main() -> int:
    args = parse_args()
    raw_config = yaml.safe_load(args.scene_list.read_text(encoding="utf-8")) or {}
    methods_config = raw_config.get("methods", {})
    method_config = methods_config.get(args.method, {}) if isinstance(methods_config, dict) else {}
    failures = []
    for scene in load_scenes(args.scene_list):
        output_rrd = args.output_root / args.method / scene.dataset / f"{scene.name}.rrd"
        if prediction_and_metrics_complete(output_rrd) and not args.overwrite:
            print(f"[SKIP] {scene.dataset}/{scene.name}")
            continue
        if not scene.path.is_dir():
            print(f"[WARN] Missing scene: {scene.path}")
            failures.append(scene.name)
            continue
        output_rrd.parent.mkdir(parents=True, exist_ok=True)
        stride = int(scene.params.get("stride", 1))
        code = run_command(
            method_command(
                args.method,
                scene.path.resolve(),
                output_rrd.resolve(),
                args.cuda_device,
                stride,
                method_config,
            ),
            dry_run=args.dry_run,
        )
        if code == 0:
            code = run_command(
                aligned_evaluation_command(BENCHMARK_DIR, scene.path.resolve(), output_rrd.resolve(), stride),
                dry_run=args.dry_run,
            )
        if code != 0:
            failures.append(f"{scene.dataset}/{scene.name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
