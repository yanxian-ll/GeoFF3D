#!/usr/bin/env python3
"""Macro-average scene-level seam errors for all ablation variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXPERIMENTS = (
    "00_temporal_base",
    "01_base",
    "02_yaw_before_propagation",
    "02_propagation_before_yaw",
    "03_full",
    "04_full_stage1",
    "05_full_stronger_noise",
    "06_full_translation_only",
    "07_full_chunk20",
    "07_full_chunk30",
    "08_full_chunk40",
    "09_colmap_dense",
)

DISPLAY_NAMES = {
    "00_temporal_base": "Temporal base (Sequential + Sim(3))",
    "01_base": "Base (Tree + Sim(3))",
    "02_yaw_before_propagation": "+ Yaw alignment",
    "02_propagation_before_yaw": "+ Prior propagation",
    "03_full": "Full (Yaw + propagation)",
    "04_full_stage1": "Full, Stage-1 weights",
    "05_full_stronger_noise": "Full, stronger pose noise",
    "06_full_translation_only": "Full, translation prior only",
    "07_full_chunk20": "Full, chunk size 20",
    "07_full_chunk30": "Full, chunk size 30",
    "08_full_chunk40": "Full, chunk size 40",
    "09_colmap_dense": "COLMAP dense",
}

SEAM_OPTIONAL_EXPERIMENTS = {"09_colmap_dense"}

# The chunk-20 launcher previously used METHOD_NAME=07_full_chunk15 by mistake.
# Keep completed historical runs usable while all new runs use the corrected
# 07_full_chunk20 output directory.
LEGACY_EXPERIMENT_DIRS = {
    "07_full_chunk20": ("07_full_chunk15",),
    # Chunk size 30 is the full-method default. The dedicated launcher was
    # removed to avoid duplicate computation, so fresh runs reuse 03_full.
    "07_full_chunk30": ("03_full",),
}

STAGE_COMPARISON = (
    ("04_full_stage1", "Stage 1"),
    ("03_full", "Stage 2"),
)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_mean(rows: list[dict], key: str) -> float:
    values = [float(row.get(key, np.nan)) for row in rows]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def _finite_max(rows: list[dict], key: str) -> float:
    values = [float(row.get(key, np.nan)) for row in rows]
    values = [value for value in values if np.isfinite(value)]
    return float(np.max(values)) if values else float("nan")


def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{number:.{digits}f}" if np.isfinite(number) else "--"


def _write_markdown(root: Path, rows: list[dict], experiments: list[str]) -> None:
    lines = [
        "# UAV-SLAM Ablation Results",
        "",
        "All values are macro-averaged over scenes. Lower is better for all error metrics.",
        "",
        "## Reconstruction and seam consistency",
        "",
        "| Experiment | Chamfer-L1 ↓ (m) | ATE RMSE ↓ (m) | Seam Error ↓ (m) | Vertical Seam ↓ (m) | Valid edges |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for experiment in experiments:
        selected = [row for row in rows if row["experiment"] == experiment]
        valid_edges = sum(int(row["num_valid_edges"]) for row in selected)
        total_edges = sum(int(row["num_adjacency_edges"]) for row in selected)
        edge_summary = f"{valid_edges}/{total_edges}" if selected else "--"
        lines.append(
            "| {name} | {chamfer} | {ate} | {seam} | {seam_z} | {edges} |".format(
                name=DISPLAY_NAMES.get(experiment, experiment),
                chamfer=_fmt(_finite_mean(selected, "chamfer_l1")),
                ate=_fmt(_finite_mean(selected, "ate_rmse")),
                seam=_fmt(_finite_mean(selected, "seam_error")),
                seam_z=_fmt(_finite_mean(selected, "seam_error_z")),
                edges=edge_summary,
            )
        )

    available = set(experiments)
    if all(experiment in available for experiment, _label in STAGE_COMPARISON):
        lines.extend(
            [
                "",
                "## Training stage comparison",
                "",
                "The inference pipeline is identical; only the training checkpoint changes.",
                "",
                "| Weights | Checkpoint experiment | Chamfer-L1 ↓ (m) | ATE RMSE ↓ (m) | Seam Error ↓ (m) | Vertical Seam ↓ (m) |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for experiment, label in STAGE_COMPARISON:
            selected = [row for row in rows if row["experiment"] == experiment]
            lines.append(
                "| {label} | {experiment} | {chamfer} | {ate} | {seam} | {seam_z} |".format(
                    label=label,
                    experiment=experiment,
                    chamfer=_fmt(_finite_mean(selected, "chamfer_l1")),
                    ate=_fmt(_finite_mean(selected, "ate_rmse")),
                    seam=_fmt(_finite_mean(selected, "seam_error")),
                    seam_z=_fmt(_finite_mean(selected, "seam_error_z")),
                )
            )

    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            "Runtime is `processing_time_seconds` averaged over scenes; image/chunk "
            "counts are also averaged, while peak GPU memory is the maximum over scenes.",
            "",
            "| Experiment | Images | Chunks | Runtime (s) | Peak GPU memory (GiB) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for experiment in experiments:
        selected = [row for row in rows if row["experiment"] == experiment]
        lines.append(
            "| {name} | {images} | {chunks} | {runtime} | {memory} |".format(
                name=DISPLAY_NAMES.get(experiment, experiment),
                images=_fmt(_finite_mean(selected, "num_frames"), 0),
                chunks=_fmt(_finite_mean(selected, "num_chunks"), 1),
                runtime=_fmt(
                    _finite_mean(selected, "processing_time_seconds"), 2
                ),
                memory=_fmt(_finite_max(selected, "peak_gpu_memory_gib"), 2),
            )
        )

    lines.extend(
        [
            "",
            "## Per-scene results",
            "",
            "| Experiment | Dataset | Scene | Chamfer-L1 (m) | ATE RMSE (m) | Seam Error (m) | Vertical Seam (m) | Runtime (s) | Peak GPU memory (GiB) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {experiment} | {dataset} | {scene} | {chamfer} | {ate} | {seam} | {seam_z} | {runtime} | {memory} |".format(
                experiment=DISPLAY_NAMES.get(row["experiment"], row["experiment"]),
                dataset=row["dataset"],
                scene=row["scene"],
                chamfer=_fmt(row["chamfer_l1"]),
                ate=_fmt(row["ate_rmse"]),
                seam=_fmt(row["seam_error"]),
                seam_z=_fmt(row["seam_error_z"]),
                runtime=_fmt(row["processing_time_seconds"], 2),
                memory=_fmt(row["peak_gpu_memory_gib"], 2),
            )
        )
    lines.append("")
    (root / "ablation_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS,
        default=list(EXPERIMENTS),
        help="Experiment directories to summarize (default: all).",
    )
    args = parser.parse_args()
    root = args.results_root.expanduser().resolve()
    rows = []
    summary = {}
    invalid_results = []
    for experiment in args.experiments:
        scene_values = []
        scene_z_values = []
        experiment_root = root / experiment
        sidecar_paths = sorted(experiment_root.glob("*/*/result.json"))
        sidecar_paths += sorted(experiment_root.glob("*/*.json"))
        if not sidecar_paths:
            for legacy_name in LEGACY_EXPERIMENT_DIRS.get(experiment, ()):
                legacy_root = root / legacy_name
                legacy_sidecars = sorted(legacy_root.glob("*/*/result.json"))
                legacy_sidecars += sorted(legacy_root.glob("*/*.json"))
                if legacy_sidecars:
                    experiment_root = legacy_root
                    sidecar_paths = legacy_sidecars
                    print(
                        f"[INFO] {experiment}: using legacy results directory "
                        f"{legacy_root}"
                    )
                    break
        for sidecar_path in sidecar_paths:
            data = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar_path.name == "result.json":
                result_dir = sidecar_path.parent
                dataset_name = sidecar_path.parent.parent.name
                scene_name = sidecar_path.parent.name
            else:
                result_dir = sidecar_path.with_suffix("")
                dataset_name = sidecar_path.parent.name
                scene_name = sidecar_path.stem
            metrics = _read_json(result_dir / "eval" / "metrics.json")
            timing = _read_json(result_dir / "processing_time.json")
            seam = data.get("grid", {}).get("seam_error", {})
            value = float(seam.get("seam_error", np.nan))
            value_z = float(seam.get("seam_error_z", np.nan))
            if experiment not in SEAM_OPTIONAL_EXPERIMENTS and (
                not np.isfinite(value)
                or not np.isfinite(value_z)
                or int(seam.get("num_valid_edges", 0)) <= 0
            ):
                invalid_results.append(
                    f"{sidecar_path}: missing/non-finite seam metrics or no valid edges"
                )
            rows.append(
                {
                    "experiment": experiment,
                    "dataset": dataset_name,
                    "scene": scene_name,
                    "seam_error": value,
                    "seam_error_z": value_z,
                    "num_valid_edges": int(seam.get("num_valid_edges", 0)),
                    "num_adjacency_edges": int(seam.get("num_adjacency_edges", 0)),
                    "chamfer_l1": float(
                        metrics.get("point_cloud", {}).get("chamfer_l1", np.nan)
                    ),
                    "ate_rmse": float(
                        metrics.get("pose", {}).get("ate", {}).get("ate_rmse", np.nan)
                    ),
                    "num_frames": float(timing.get("num_frames", np.nan)),
                    "num_chunks": float(timing.get("num_chunks", np.nan)),
                    "processing_time_seconds": float(
                        timing.get("processing_time_seconds", np.nan)
                    ),
                    "peak_gpu_memory_gib": float(
                        timing.get("peak_gpu_memory_allocated_mib", np.nan)
                    ) / 1024.0,
                }
            )
            if np.isfinite(value):
                scene_values.append(value)
            if np.isfinite(value_z):
                scene_z_values.append(value_z)
        summary[experiment] = {
            "aggregation": "macro_average_over_scenes",
            "num_valid_scenes": len(scene_values),
            "seam_error": float(np.mean(scene_values)) if scene_values else float("nan"),
            "seam_error_z": float(np.mean(scene_z_values)) if scene_z_values else float("nan"),
        }
        if not scene_values and experiment not in SEAM_OPTIONAL_EXPERIMENTS:
            invalid_results.append(
                f"{experiment_root}: no valid scene seam results"
            )

    if invalid_results:
        print("[WARN] Incomplete results are shown as --:")
        for item in invalid_results:
            print(f"  - {item}")

    root.mkdir(parents=True, exist_ok=True)
    (root / "seam_error_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (root / "seam_error_per_scene.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "experiment", "dataset", "scene", "seam_error", "seam_error_z",
            "num_valid_edges", "num_adjacency_edges", "processing_time_seconds",
            "peak_gpu_memory_gib",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [{field: row[field] for field in fields} for row in rows]
        )
    _write_markdown(root, rows, list(args.experiments))
    print(f"Saved summaries and Markdown report under: {root}")


if __name__ == "__main__":
    main()
