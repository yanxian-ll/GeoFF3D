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
)

DISPLAY_NAMES = {
    "00_temporal_base": "Temporal base (Sequential + Sim(3))",
    "01_base": "Base (Tree + Sim(3))",
    "02_yaw_before_propagation": "+ Yaw alignment",
    "02_propagation_before_yaw": "+ Prior propagation",
    "03_full": "Full (Yaw + propagation)",
    "04_full_stage1": "Full, Stage-1 weights",
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


def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.{digits}f}" if np.isfinite(number) else "N/A"


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
        lines.append(
            "| {name} | {chamfer} | {ate} | {seam} | {seam_z} | {valid}/{total} |".format(
                name=DISPLAY_NAMES.get(experiment, experiment),
                chamfer=_fmt(_finite_mean(selected, "chamfer_l1")),
                ate=_fmt(_finite_mean(selected, "ate_rmse")),
                seam=_fmt(_finite_mean(selected, "seam_error")),
                seam_z=_fmt(_finite_mean(selected, "seam_error_z")),
                valid=valid_edges,
                total=total_edges,
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
            "| Experiment | Images | Chunks | Runtime (s) | Chunking (s) | Inference (s) | Post-align (s) | Peak GPU memory (GiB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for experiment in experiments:
        selected = [row for row in rows if row["experiment"] == experiment]
        lines.append(
            "| {name} | {images} | {chunks} | {runtime} | {chunking} | {inference} | {post} | {memory} |".format(
                name=DISPLAY_NAMES.get(experiment, experiment),
                images=_fmt(_finite_mean(selected, "num_frames"), 0),
                chunks=_fmt(_finite_mean(selected, "num_chunks"), 1),
                runtime=_fmt(_finite_mean(selected, "runtime_seconds"), 2),
                chunking=_fmt(_finite_mean(selected, "chunking_seconds"), 2),
                inference=_fmt(_finite_mean(selected, "inference_seconds"), 2),
                post=_fmt(_finite_mean(selected, "post_alignment_seconds"), 2),
                memory=_fmt(_finite_mean(selected, "peak_gpu_memory_gib"), 2),
            )
        )

    lines.extend(
        [
            "",
            "## Per-scene results",
            "",
            "| Experiment | Dataset | Scene | Chamfer-L1 (m) | ATE RMSE (m) | Seam Error (m) | Vertical Seam (m) |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {experiment} | {dataset} | {scene} | {chamfer} | {ate} | {seam} | {seam_z} |".format(
                experiment=DISPLAY_NAMES.get(row["experiment"], row["experiment"]),
                dataset=row["dataset"],
                scene=row["scene"],
                chamfer=_fmt(row["chamfer_l1"]),
                ate=_fmt(row["ate_rmse"]),
                seam=_fmt(row["seam_error"]),
                seam_z=_fmt(row["seam_error_z"]),
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
        for sidecar_path in sorted((root / experiment).glob("*/*.json")):
            data = json.loads(sidecar_path.read_text(encoding="utf-8"))
            result_dir = sidecar_path.with_suffix("")
            metrics = _read_json(result_dir / "eval" / "metrics.json")
            timing = _read_json(result_dir / "processing_time.json")
            seam = data.get("grid", {}).get("seam_error", {})
            value = float(seam.get("seam_error", np.nan))
            value_z = float(seam.get("seam_error_z", np.nan))
            if (
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
                    "dataset": sidecar_path.parent.name,
                    "scene": sidecar_path.stem,
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
                    "runtime_seconds": float(
                        timing.get("processing_time_seconds", np.nan)
                    ),
                    "chunking_seconds": float(timing.get("chunking_seconds", np.nan)),
                    "inference_seconds": float(
                        timing.get("model_prediction_seconds", np.nan)
                    ),
                    "post_alignment_seconds": float(
                        timing.get("post_alignment_seconds", np.nan)
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
        if not scene_values:
            invalid_results.append(f"{root / experiment}: no valid scene seam results")

    if invalid_results:
        details = "\n".join(f"  - {item}" for item in invalid_results)
        raise RuntimeError(
            "Cannot summarize incomplete Seam Error results. Re-run the affected "
            f"ablation scenes:\n{details}"
        )

    root.mkdir(parents=True, exist_ok=True)
    (root / "seam_error_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (root / "seam_error_per_scene.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "experiment", "dataset", "scene", "seam_error", "seam_error_z",
            "num_valid_edges", "num_adjacency_edges",
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
