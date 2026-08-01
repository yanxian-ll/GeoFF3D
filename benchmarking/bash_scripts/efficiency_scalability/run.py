#!/usr/bin/env python3
"""Run one UAV reconstruction method for scalability measurement."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


BENCHMARK_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))

from common.runtime import hydra_value, load_scenes, run_command


METHODS = {"geoff3d", "lingbot-map", "vggt_slam2.0"}


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=sorted(METHODS))
    parser.add_argument("--scene-list", type=Path, default=Path(__file__).with_name("test_scenes.yaml"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "benchmarking" / "efficiency_scalability",
    )
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--checkpoint", default=env("CHECKPOINT"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Extra Hydra overrides for GeoFF3D")
    return parser.parse_intermixed_args()


def geoff3d_command(args: argparse.Namespace, scene: Path, output: Path, params: dict[str, object], config: dict[str, object]) -> list[str]:
    checkpoint = args.checkpoint or str(config.get("checkpoint", PROJECT_ROOT / "checkpoints/geoff3d/checkpoint-best.pth"))
    fixed = {
        "model": "geoff3d",
        "scene_dir": scene,
        "checkpoint": checkpoint,
        "output_path": output,
        "device": f"cuda:{args.cuda_device}",
        "footprint_estimation": "sequential",
        "align": env("ALIGN", "scale_yaw_translation"),
        "translation_prior": "input",
        "rotation_prior": env("ROTATION_PRIOR", "input"),
        "ray_prior": env("RAY_PRIOR", "pred"),
        "depth_prior": env("DEPTH_PRIOR", "pred"),
        "max_chunk_size": env("MAX_CHUNK_SIZE", "30"),
        "min_chunk_size": env("MIN_CHUNK_SIZE", "8"),
        "post_chunk_align": env("POST_CHUNK_ALIGN", "true"),
        "post_chunk_align_mode": env("POST_CHUNK_ALIGN_MODE", "yaw_translation"),
    }
    fixed.update(params)
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_slrf.py"),
        *(f"{key}={hydra_value(value)}" for key, value in fixed.items()),
        *args.overrides,
    ]


def lingbot_command(args: argparse.Namespace, scene: Path, output: Path, params: dict[str, object], config: dict[str, object]) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARK_DIR / "adapters" / "streaming.py"),
        "--method", "lingbot-map",
        "--scene_dir", str(scene),
        "--output_rrd", str(output / "result.rrd"),
        "--output_dir", str(output),
        "--device", f"cuda:{args.cuda_device}",
        "--num_views", str(int(params.get("num_views", 0))),
        "--stride", str(int(params.get("stride", 1))),
        "--max_side", "518",
        "--size_multiple", "14",
        "--max_pred_points", "800000",
        "--max_gt_points", "800000",
        "--max_points_per_view", "500000",
        "--voxel_downsample", "0.01",
        "--python", env("LINGBOT_PYTHON", str(config.get("python", sys.executable))),
        "--stream_mode", env("LINGBOT_STREAM_MODE", "streaming"),
        "--keyframe_interval", env("LINGBOT_KEYFRAME_INTERVAL", "auto"),
    ]
    model_path = env("LINGBOT_MODEL_PATH", str(config.get("checkpoint", "")))
    if model_path:
        command += ["--model_path", model_path]
    command.append("--use_sdpa" if env("USE_SDPA", "1").lower() in {"1", "true", "yes"} else "--no-use_sdpa")
    return command


def vggt_slam2_command(args: argparse.Namespace, scene: Path, output: Path, params: dict[str, object], config: dict[str, object]) -> list[str]:
    command = [
        env("VGGT_SLAM2_PYTHON", str(config.get("python", "/opt/conda/envs/mapanything/bin/python"))),
        str(BENCHMARK_DIR / "third_party" / "vggt-slam2.0" / "run_scene_to_rrd.py"),
        "--method", "vggt-slam2.0",
        "--scene_dir", str(scene),
        "--output_rrd", str(output / "result.rrd"),
        "--output_dir", str(output),
        "--device", f"cuda:{args.cuda_device}",
        "--num_views", str(int(params.get("num_views", 0))),
        "--stride", str(int(params.get("stride", 1))),
        "--max_side", "518",
        "--size_multiple", "14",
        "--submap_size", env("SUBMAP_SIZE", "16"),
        "--overlapping_window_size", env("OVERLAPPING_WINDOW_SIZE", "2"),
        "--global_point_stride", "1",
        "--max_gt_points", "800000",
        "--max_points_per_view", "500000",
        "--voxel_downsample", "0.01",
    ]
    checkpoint = str(config.get("checkpoint", ""))
    if checkpoint:
        command += ["--vggt_model_path", checkpoint]
    if env("DISABLE_KEYFRAME_SELECTION", "0").lower() in {"1", "true", "yes"}:
        command.append("--disable_keyframe_selection")
    return command


def main() -> int:
    args = parse_args()
    raw_config = yaml.safe_load(args.scene_list.read_text(encoding="utf-8")) or {}
    methods_config = raw_config.get("methods", {})
    method_config = methods_config.get(args.method, {}) if isinstance(methods_config, dict) else {}
    failures: list[str] = []
    output_method = "pi3x_world_translation" if args.method == "geoff3d" else args.method
    for item in load_scenes(args.scene_list):
        output = args.output_root / output_method / item.dataset / item.name
        timing = output / "processing_time.json"
        if timing.is_file() and not args.overwrite:
            print(f"[SKIP] {item.dataset}/{item.name}")
            continue
        if not item.path.is_dir():
            print(f"[WARN] Missing scene: {item.path}")
            failures.append(f"{item.dataset}/{item.name}")
            continue
        scene = item.path.resolve()
        output = output.resolve()
        if args.method == "geoff3d":
            command = geoff3d_command(args, scene, output, item.params, method_config)
        elif args.method == "lingbot-map":
            command = lingbot_command(args, scene, output, item.params, method_config)
        else:
            command = vggt_slam2_command(args, scene, output, item.params, method_config)
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=True)
        code = run_command(command, dry_run=args.dry_run)
        if code != 0:
            failures.append(f"{item.dataset}/{item.name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
