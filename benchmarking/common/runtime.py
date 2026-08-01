"""Shared scene-list and process helpers for UAV-SLAM benchmarks."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class Scene:
    dataset: str
    name: str
    path: Path
    params: Dict[str, object]


def flatten_params(value: object) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if not isinstance(value, Mapping):
        return out
    for key, item in value.items():
        if isinstance(item, Mapping):
            out.update(flatten_params(item))
        else:
            out[str(key).lower().replace("-", "_")] = item
    return out


def load_scenes(path: Path) -> Iterator[Scene]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for dataset in config.get("datasets", []):
        if not isinstance(dataset, Mapping) or not as_bool(dataset.get("enabled", True)):
            continue
        dataset_name = str(dataset.get("name", "default"))
        dataset_root = Path(str(dataset.get("root", "."))).expanduser()
        dataset_params = flatten_params(dataset.get("params", {}))
        for entry in dataset.get("scenes", []):
            if isinstance(entry, str):
                name, raw_path, scene_params = entry, Path(entry), {}
            elif isinstance(entry, Mapping):
                name = str(entry.get("name") or entry.get("path"))
                raw_path = Path(str(entry.get("path", name))).expanduser()
                scene_params = flatten_params(entry.get("params", {}))
            else:
                continue
            scene_path = raw_path if raw_path.is_absolute() else dataset_root / raw_path
            yield Scene(dataset_name, name, scene_path, {**dataset_params, **scene_params})


def as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def hydra_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def print_command(command: Sequence[str]) -> None:
    print("[CMD] " + " ".join(shlex.quote(str(part)) for part in command), flush=True)


def run_command(
    command: Sequence[str], *, dry_run: bool = False, env: Mapping[str, str] | None = None
) -> int:
    print_command(command)
    if dry_run:
        return 0
    result = subprocess.run(
        [str(part) for part in command],
        env={**os.environ, **dict(env or {})},
        check=False,
    )
    return int(result.returncode)


def prediction_and_metrics_complete(output_rrd: Path) -> bool:
    result_dir = output_rrd.with_suffix("")
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in (
            output_rrd,
            output_rrd.with_suffix(".json"),
            result_dir / "eval" / "pred_cameras.npz",
            result_dir / "eval" / "pred_points.ply",
            result_dir / "eval" / "metrics.json",
            result_dir / "eval" / "metrics_summary.csv",
        )
    )


def aligned_evaluation_command(
    benchmark_dir: Path, scene_dir: Path, output_rrd: Path, stride: int
) -> list[str]:
    eval_dir = output_rrd.with_suffix("") / "eval"
    return [
        os.environ.get("BENCHMARK_PYTHON", "python3"),
        str(benchmark_dir / "common" / "evaluate.py"),
        "--scene_dir", str(scene_dir), "--eval_dir", str(eval_dir),
        "--output_json", str(eval_dir / "metrics.json"),
        "--output_csv", str(eval_dir / "metrics_summary.csv"),
        "--stride", str(stride), "--max_side", "518", "--size_multiple", "14",
        "--thresholds", "0.5,1.0,2.0,5.0", "--rpe_steps", "1,5,10",
        "--max_gt_points", "300000", "--max_pred_points_eval", "300000",
        "--max_gt_points_eval", "300000", "--max_align_points", "100000",
        "--icp_iterations", "8", "--icp_trim_quantile", "0.7",
        "--gt_io_workers", "4",
    ]
