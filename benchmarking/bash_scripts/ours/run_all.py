#!/usr/bin/env python3
"""Run all selected SLRF methods over a benchmark scene YAML."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import yaml


METHODS = ("geoff3d", "pi3x", "vggt", "vggt_omega")
CHECKPOINT_ENV = {
    "geoff3d": "GEOFF3D_CHECKPOINT",
    "pi3x": "PI3X_CHECKPOINT",
    "vggt": "VGGT_CHECKPOINT",
    "vggt_omega": "VGGT_OMEGA_CHECKPOINT",
}


def flatten_params(value: object) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if not isinstance(value, dict):
        return out
    for key, item in value.items():
        if isinstance(item, dict):
            out.update(flatten_params(item))
        else:
            out[str(key).lower().replace("-", "_")] = item
    return out


def hydra_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def iter_scenes(config: Dict[str, object]) -> Iterator[Tuple[str, str, Path, Dict[str, object]]]:
    for dataset in config.get("datasets", []):
        if not isinstance(dataset, dict) or not bool(dataset.get("enabled", True)):
            continue
        dataset_name = str(dataset.get("name", "default"))
        dataset_root = Path(str(dataset.get("root", "."))).expanduser()
        dataset_params = flatten_params(dataset.get("params", {}))
        for entry in dataset.get("scenes", []):
            if isinstance(entry, str):
                scene_name = entry
                scene_path = dataset_root / entry
                scene_params: Dict[str, object] = {}
            elif isinstance(entry, dict):
                scene_name = str(entry.get("name") or entry.get("path"))
                raw_path = Path(str(entry.get("path", scene_name))).expanduser()
                scene_path = raw_path if raw_path.is_absolute() else dataset_root / raw_path
                scene_params = flatten_params(entry.get("params", {}))
            else:
                continue
            yield dataset_name, scene_name, scene_path, {**dataset_params, **scene_params}


def parse_args() -> argparse.Namespace:
    benchmark_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Methods to run (default: all methods).",
    )
    parser.add_argument(
        "--scene-list", type=Path, default=Path(__file__).with_name("default_scenes.yaml")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=benchmark_dir.parent / "experiments" / "benchmarking" / "slrf",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Result subdirectory name (defaults to the method name).",
    )
    parser.add_argument(
        "--checkpoint-profile",
        default="default",
        help="Checkpoint profile selected from methods.<method>.checkpoints.",
    )
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides")
    return parser.parse_intermixed_args()


def main() -> int:
    args = parse_args()
    if args.output_name is not None and len(args.methods) != 1:
        raise ValueError("--output-name can only be used when exactly one method is selected.")

    benchmark_dir = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(args.scene_list.read_text(encoding="utf-8"))
    env = dict(os.environ)
    env["CUDA_DEVICE"] = str(args.cuda_device)
    method_configs = config.get("methods", {})
    for method, variable in CHECKPOINT_ENV.items():
        method_config = method_configs.get(method, {}) if isinstance(method_configs, dict) else {}
        checkpoint = None
        if isinstance(method_config, dict):
            profiles = method_config.get("checkpoints", {})
            if isinstance(profiles, dict) and args.checkpoint_profile in profiles:
                checkpoint = profiles[args.checkpoint_profile]
            elif args.checkpoint_profile == "default":
                checkpoint = method_config.get("checkpoint")
            elif method in args.methods:
                raise ValueError(
                    f"Checkpoint profile {args.checkpoint_profile!r} is not configured "
                    f"for method {method!r}."
                )
        if checkpoint and variable not in env and "CHECKPOINT" not in env:
            env[variable] = str(checkpoint)

    failures: List[str] = []
    scenes = list(iter_scenes(config))
    for method in args.methods:
        launcher = benchmark_dir / "bash_scripts" / "ours" / f"{method}.sh"
        print(f"\n[METHOD] {method}")
        for dataset, scene_name, scene_dir, params in scenes:
            output_dir = args.output_root / (args.output_name or method) / dataset / scene_name
            item_name = f"{method}/{dataset}/{scene_name}"
            complete = (
                (output_dir / "result.rrd").is_file()
                and (output_dir / "eval" / "metrics.json").is_file()
            )
            if complete and not args.overwrite:
                print(f"[SKIP] {item_name}")
                continue
            if not scene_dir.is_dir():
                print(f"[WARN] Missing scene: {scene_dir}")
                failures.append(f"{item_name}: missing scene")
                continue

            scene_overrides = [f"{key}={hydra_value(value)}" for key, value in params.items()]
            command = [
                str(launcher),
                str(scene_dir.resolve()),
                str(output_dir.resolve()),
                *scene_overrides,
                *args.overrides,
            ]
            print("[RUN] " + " ".join(shlex.quote(part) for part in command))
            if args.dry_run:
                continue
            result = subprocess.run(command, env=env, check=False)
            if result.returncode != 0:
                failures.append(f"{item_name}: exit {result.returncode}")

    if failures:
        print("[ERROR] Failed scenes:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
