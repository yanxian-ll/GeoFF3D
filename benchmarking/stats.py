#!/usr/bin/env python3
"""Generate Markdown tables from the current UAV-SLAM benchmark outputs.

Examples:
    python benchmarking/stats.py
    python benchmarking/stats.py ate_rmse acc_mean fscore_1.0
    python benchmarking/stats.py --groups slrf,stream,optim
    python benchmarking/stats.py --groups ablation --methods 03_full,04_full_stage1
    python benchmarking/stats.py --list-methods
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


GROUPS = ("slrf", "stream", "optim", "ablation", "efficiency_scalability")
DEFAULT_GROUPS = ("slrf", "stream", "optim")
DEFAULT_METRICS = ("acc_mean", "comp_mean", "fscore_1.0")

GROUP_DISPLAY = {
    "slrf": "SLRF",
    "stream": "Streaming",
    "optim": "Optimization",
    "ablation": "Ablation",
    "efficiency_scalability": "Efficiency and Scalability",
}

METHOD_DISPLAY = {
    "geoff3d": "GeoFF3D + SLRF (Ours)",
    "pi3x": "Pi3X + SLRF",
    "vggt": "VGGT + SLRF",
    "vggt_omega": "VGGT-Omega + SLRF",
    "lingbot-map": "LingBot-Map",
    "stream3r": "STream3R",
    "streamvggt": "StreamVGGT",
    "ttt3r": "TTT3R",
    "vggt_long": "VGGT-Long",
    "vggt_slam2.0": "VGGT-SLAM 2.0",
    "vggt_slam_sim3": "VGGT-SLAM Sim(3)",
    "vggt_slam_sl4": "VGGT-SLAM SL(4)",
    "pi3x_world_translation": "Pi3X + World Translation",
}

DATASET_DISPLAY = {
    "usegeo": "UseGeo",
    "uavff3d_real": "UAVFF3D-Real",
    "uavscenes": "UAVScenes",
    "npu_dronemap": "NPU DroneMap",
}

# short name -> (metrics.json path, table label)
METRICS: dict[str, tuple[tuple[str, ...], str]] = {
    "ate_rmse": (("pose", "ate", "ate_rmse"), "ATE RMSE ↓"),
    "ate_mean": (("pose", "ate", "ate_mean"), "ATE Mean ↓"),
    "ate_median": (("pose", "ate", "ate_median"), "ATE Median ↓"),
    "ate_p90": (("pose", "ate", "ate_p90"), "ATE P90 ↓"),
    "ate_p95": (("pose", "ate", "ate_p95"), "ATE P95 ↓"),
    "rot_deg_rmse": (("pose", "rotation", "rot_deg_rmse"), "Rotation RMSE (deg) ↓"),
    "rot_deg_mean": (("pose", "rotation", "rot_deg_mean"), "Rotation Mean (deg) ↓"),
    "rot_deg_median": (("pose", "rotation", "rot_deg_median"), "Rotation Median (deg) ↓"),
    "rot_deg_p90": (("pose", "rotation", "rot_deg_p90"), "Rotation P90 (deg) ↓"),
    "rpe_k1_trans_rmse": (("pose", "rpe", "k1", "translation", "rpe_trans_rmse"), "RPE k1 Translation RMSE ↓"),
    "rpe_k1_rot_rmse": (("pose", "rpe", "k1", "rotation", "rpe_rot_deg_rmse"), "RPE k1 Rotation RMSE (deg) ↓"),
    "rpe_k5_trans_rmse": (("pose", "rpe", "k5", "translation", "rpe_trans_rmse"), "RPE k5 Translation RMSE ↓"),
    "rpe_k5_rot_rmse": (("pose", "rpe", "k5", "rotation", "rpe_rot_deg_rmse"), "RPE k5 Rotation RMSE (deg) ↓"),
    "rpe_k10_trans_rmse": (("pose", "rpe", "k10", "translation", "rpe_trans_rmse"), "RPE k10 Translation RMSE ↓"),
    "rpe_k10_rot_rmse": (("pose", "rpe", "k10", "rotation", "rpe_rot_deg_rmse"), "RPE k10 Rotation RMSE (deg) ↓"),
    "acc_mean": (("point_cloud", "accuracy_pred_to_gt", "acc_mean"), "Accuracy Mean ↓"),
    "acc_rmse": (("point_cloud", "accuracy_pred_to_gt", "acc_rmse"), "Accuracy RMSE ↓"),
    "acc_median": (("point_cloud", "accuracy_pred_to_gt", "acc_median"), "Accuracy Median ↓"),
    "acc_p90": (("point_cloud", "accuracy_pred_to_gt", "acc_p90"), "Accuracy P90 ↓"),
    "comp_mean": (("point_cloud", "completeness_gt_to_pred", "comp_mean"), "Completeness Mean ↓"),
    "comp_rmse": (("point_cloud", "completeness_gt_to_pred", "comp_rmse"), "Completeness RMSE ↓"),
    "comp_median": (("point_cloud", "completeness_gt_to_pred", "comp_median"), "Completeness Median ↓"),
    "comp_p90": (("point_cloud", "completeness_gt_to_pred", "comp_p90"), "Completeness P90 ↓"),
    "chamfer_l1": (("point_cloud", "chamfer_l1"), "Chamfer L1 ↓"),
    "chamfer_l2": (("point_cloud", "chamfer_l2"), "Chamfer L2 ↓"),
    "fscore_0.5": (("point_cloud", "fscore", "0.5", "fscore"), "F-score@0.5 ↑"),
    "fscore_1.0": (("point_cloud", "fscore", "1.0", "fscore"), "F-score@1.0 ↑"),
    "fscore_2.0": (("point_cloud", "fscore", "2.0", "fscore"), "F-score@2.0 ↑"),
    "fscore_5.0": (("point_cloud", "fscore", "5.0", "fscore"), "F-score@5.0 ↑"),
    "precision_1.0": (("point_cloud", "fscore", "1.0", "precision"), "Precision@1.0 ↑"),
    "recall_1.0": (("point_cloud", "fscore", "1.0", "recall"), "Recall@1.0 ↑"),
}


@dataclass(frozen=True)
class Method:
    group: str
    name: str
    root: Path

    @property
    def key(self) -> str:
        return f"{self.group}/{self.name}"

    @property
    def display(self) -> str:
        return METHOD_DISPLAY.get(self.name, self.name)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        nargs="*",
        default=list(DEFAULT_METRICS),
        help="Metric short names or dotted JSON paths.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root / "experiments" / "benchmarking",
        help="Benchmark output root.",
    )
    parser.add_argument(
        "--groups",
        default=",".join(DEFAULT_GROUPS),
        help=f"Comma-separated groups. Choices: {', '.join(GROUPS)}.",
    )
    parser.add_argument(
        "--methods", "-m",
        default=None,
        help="Comma-separated method names or group/method keys.",
    )
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--list-metrics", action="store_true")
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    return parser.parse_args(argv)


def selected_groups(raw: str) -> list[str]:
    groups = list(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    unknown = [group for group in groups if group not in GROUPS]
    if unknown:
        raise ValueError(f"Unknown benchmark groups: {', '.join(unknown)}")
    return groups


def discover_methods(root: Path, groups: list[str]) -> list[Method]:
    methods: list[Method] = []
    for group in groups:
        group_root = root / group
        if not group_root.is_dir():
            continue
        for path in sorted(group_root.iterdir()):
            if path.is_dir() and not path.name.startswith("."):
                methods.append(Method(group, path.name, path))
    return methods


def filter_methods(methods: list[Method], requested: Optional[str]) -> list[Method]:
    if not requested:
        return methods
    specs = [item.strip() for item in requested.split(",") if item.strip()]
    selected: list[Method] = []
    for spec in specs:
        matches = [method for method in methods if method.key == spec]
        if "/" not in spec:
            matches = [method for method in methods if method.name == spec]
        if not matches:
            raise ValueError(f"Method not found under selected groups: {spec}")
        if len(matches) > 1:
            choices = ", ".join(method.key for method in matches)
            raise ValueError(f"Ambiguous method {spec!r}; use one of: {choices}")
        if matches[0] not in selected:
            selected.append(matches[0])
    return selected


def resolve_metric(raw: str) -> tuple[tuple[str, ...], str]:
    if raw in METRICS:
        return METRICS[raw]
    path = tuple(part for part in raw.split(".") if part)
    if not path:
        raise ValueError(f"Invalid metric: {raw!r}")
    return path, raw


def load_json(path: Path, quiet: bool) -> Optional[dict[str, Any]]:
    if not path.is_file():
        if not quiet:
            print(f"[WARN] Missing: {path}", file=sys.stderr)
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError) as error:
        if not quiet:
            print(f"[WARN] Cannot read {path}: {error}", file=sys.stderr)
        return None


def nested_number(data: Optional[dict[str, Any]], path: tuple[str, ...]) -> Optional[float]:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def discover_dataset_scenes(methods: list[Method]) -> dict[str, list[str]]:
    discovered: dict[str, set[str]] = {}
    for method in methods:
        if not method.root.is_dir():
            continue
        for metrics_path in method.root.glob("*/*/eval/metrics.json"):
            relative = metrics_path.relative_to(method.root)
            dataset, scene = relative.parts[0], relative.parts[1]
            discovered.setdefault(dataset, set()).add(scene)
    return {dataset: sorted(scenes) for dataset, scenes in sorted(discovered.items())}


def format_number(value: Optional[float], precision: int) -> str:
    return "-" if value is None else f"{value:.{precision}f}"


def build_table(
    dataset: str,
    scenes: list[str],
    methods: list[Method],
    metric_path: tuple[str, ...],
    metric_label: str,
    precision: int,
    quiet: bool,
) -> str:
    lines = [f"### {metric_label}", ""]
    header = ["Method", *scenes, "Average"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for method in methods:
        values = [
            nested_number(
                load_json(method.root / dataset / scene / "eval" / "metrics.json", quiet),
                metric_path,
            )
            for scene in scenes
        ]
        valid = [value for value in values if value is not None]
        average = sum(valid) / len(valid) if valid else None
        label = f"{method.display} ({GROUP_DISPLAY[method.group]})"
        cells = [label, *(format_number(value, precision) for value in values), format_number(average, precision)]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_metrics:
        for name, (_path, label) in METRICS.items():
            print(f"{name:<24} {label}")
        return 0

    try:
        groups = selected_groups(args.groups)
        methods = filter_methods(discover_methods(args.root, groups), args.methods)
        metric_specs = [(raw, *resolve_metric(raw)) for raw in (args.metrics or DEFAULT_METRICS)]
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.list_methods:
        for method in methods:
            print(f"{method.key:<40} {method.display}")
        return 0
    if not methods:
        print(f"error: no method directories found under {args.root}", file=sys.stderr)
        return 1

    datasets = discover_dataset_scenes(methods)
    if not datasets:
        print("error: no eval/metrics.json files found for the selected methods", file=sys.stderr)
        return 1

    parts = ["# UAV-SLAM Benchmark Results", ""]
    for dataset, scenes in datasets.items():
        parts.extend([f"## {DATASET_DISPLAY.get(dataset, dataset)}", ""])
        for _raw, path, label in metric_specs:
            parts.append(build_table(dataset, scenes, methods, path, label, args.precision, args.quiet))
    result = "\n".join(parts).rstrip() + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
        print(f"Saved: {args.output}")
    else:
        print(result, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
