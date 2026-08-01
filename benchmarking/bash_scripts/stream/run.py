#!/usr/bin/env python3
"""Run one streaming method over a UAV scene list."""

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


METHODS = {"lingbot-map", "stream3r", "streamvggt", "ttt3r"}


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def configured(config: dict[str, object], key: str, env_name: str, default: str = "") -> str:
    return env(env_name, str(config.get(key, default)))


def method_options(method: str, config: dict[str, object]) -> list[str]:
    if method == "lingbot-map":
        options = [
            "--python", env("STREAM_PYTHON", configured(config, "python", "LINGBOT_PYTHON", sys.executable)),
            "--stream_mode", env("LINGBOT_STREAM_MODE", "streaming"),
            "--keyframe_interval", env("LINGBOT_KEYFRAME_INTERVAL", "auto"),
        ]
        model = env("STREAM_MODEL_PATH", configured(config, "checkpoint", "LINGBOT_MODEL_PATH"))
        if model:
            options += ["--model_path", model]
        options.append("--use_sdpa" if env("USE_SDPA", "1").lower() in {"1", "true", "yes"} else "--no-use_sdpa")
        return options
    if method == "stream3r":
        return [
            "--python", env("STREAM_PYTHON", configured(config, "python", "STREAM3R_PYTHON", "/opt/conda/envs/stream3r/bin/python")),
            "--model_path", env("STREAM_MODEL_PATH", configured(config, "checkpoint", "STREAM3R_MODEL_PATH", "yslan/STream3R")),
            "--stream_mode", env("STREAM_MODE", env("STREAM3R_MODE", "causal")),
        ]
    if method == "streamvggt":
        options = ["--python", env("STREAM_PYTHON", configured(config, "python", "STREAMVGGT_PYTHON", "/opt/conda/envs/streamvggt/bin/python"))]
        model = env("STREAM_MODEL_PATH", configured(config, "checkpoint", "STREAMVGGT_MODEL_PATH"))
        return options + (["--model_path", model] if model else [])
    return [
        "--python", env("STREAM_PYTHON", configured(config, "python", "TTT3R_PYTHON", "/opt/conda/envs/ttt3r/bin/python")),
        "--model_path", env("STREAM_MODEL_PATH", configured(config, "checkpoint", "TTT3R_MODEL_PATH", "checkpoints/ttt3r/cut3r_512_dpt_4_64.pth")),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=sorted(METHODS))
    parser.add_argument("--scene-list", type=Path, default=Path(__file__).with_name("stream_scenes.yaml"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "benchmarking" / "stream",
    )
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def evaluate(scene_dir: Path, output_rrd: Path, stride: int, dry_run: bool) -> int:
    return run_command(
        aligned_evaluation_command(BENCHMARK_DIR, scene_dir, output_rrd, stride),
        dry_run=dry_run,
    )


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
        stride = int(scene.params.get("stride", 1))
        output_rrd.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(BENCHMARK_DIR / "adapters" / "streaming.py"),
            "--method", args.method, "--scene_dir", str(scene.path.resolve()),
            "--output_rrd", str(output_rrd.resolve()), "--device", f"cuda:{args.cuda_device}",
            "--stride", str(stride), "--max_side", "518", "--size_multiple", "14",
            "--scene_io_workers", "4", "--max_pred_points", "800000",
            "--max_gt_points", "800000", "--max_points_per_view", "500000",
            "--voxel_downsample", "0.01", *method_options(args.method, method_config),
        ]
        code = run_command(command, dry_run=args.dry_run)
        if code == 0:
            code = evaluate(scene.path.resolve(), output_rrd.resolve(), stride, args.dry_run)
        if code != 0:
            failures.append(f"{scene.dataset}/{scene.name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
