#!/usr/bin/env python3
"""Plot horizontal-noise and missing-prior robustness side by side."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = {
    "horizontal": ([0, 0.5, 1, 2, 5], r"Horizontal noise $\sigma_{xy}$ (m)"),
    # Keep vertical-noise results in the CSV even though its paper panel is
    # intentionally disabled below.
    "vertical": ([0, 0.2, 0.5, 1], r"Vertical noise $\sigma_z$ (m)"),
    "missing": ([16, 8, 4, 3, 2], "Retained translation priors"),
}
# Only these conditions are rendered; all CONDITIONS are still summarized.
PLOT_CONDITIONS = ("horizontal", "missing")
METHODS = (
    ("prior_yaw", "GA-Sim"),
    ("prior_pose", "Sim(3)"),
    # ("pi3x", "Pi3X-TR + Sim(3)"),
)
METRIC_ALIASES = {
    "chamfer-l1": ("abs_fused_pc_chamfer_l1", "Chamfer-L1 (m)", "chamfer_l1"),
    "chamfer_l1": ("abs_fused_pc_chamfer_l1", "Chamfer-L1 (m)", "chamfer_l1"),
    "ate": ("abs_pose_ate", "ATE (m)", "ate"),
}


def value_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def resolve_metric(name: str) -> tuple[str, str, str]:
    """Resolve a friendly metric alias or accept an exact result JSON key."""
    normalized = name.strip().lower()
    if normalized in METRIC_ALIASES:
        return METRIC_ALIASES[normalized]
    key = name.strip()
    if not key:
        raise ValueError("Metric names cannot be empty")
    slug = "".join(char if char.isalnum() else "_" for char in key).strip("_")
    return key, key, slug


def collect_scene_values(run_dir: Path, metric_key: str) -> list[float]:
    values = []
    for path in sorted(run_dir.glob("*_per_scene_results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for metrics in data.values():
            samples = np.asarray(metrics.get(metric_key, []), dtype=float)
            if np.isfinite(samples).any():
                values.append(float(np.nanmean(samples)))
    return values


def equivalent_control_run_dirs(root: Path, alignment: str) -> list[Path]:
    """Return zero-perturbation runs, which have identical model inputs."""
    candidates = (
        root / "horizontal" / "0",
        root / "vertical" / "0",
        root / "missing" / "16",
    )
    result = []
    result_name = "pi3x_prior_pose" if alignment == "pi3x" else alignment
    for candidate in candidates:
        result.extend(sorted(candidate.glob(f"seed_*/{result_name}")))
    return result


def load_rows(root: Path, metric_key: str) -> list[dict[str, object]]:
    rows = []
    for condition, (levels, _xlabel) in CONDITIONS.items():
        for level in levels:
            for alignment, method in METHODS:
                pooled = []
                if alignment == "pi3x":
                    run_dirs = sorted(
                        (root / condition / value_tag(level)).glob(
                            "seed_*/pi3x_prior_pose"
                        )
                    )
                else:
                    run_dirs = sorted(
                        (root / condition / value_tag(level)).glob(
                            f"seed_*/{alignment}"
                        )
                    )
                for run_dir in run_dirs:
                    pooled.extend(collect_scene_values(run_dir, metric_key))
                is_control = (
                    (condition in {"horizontal", "vertical"} and float(level) == 0.0)
                    or (condition == "missing" and int(level) == 16)
                )
                if not pooled and is_control:
                    for run_dir in equivalent_control_run_dirs(root, alignment):
                        pooled.extend(collect_scene_values(run_dir, metric_key))
                unavailable_sim3 = (
                    condition == "missing"
                    and int(level) < 3
                    and alignment in {"prior_pose", "pi3x"}
                )
                if not pooled and unavailable_sim3:
                    rows.append(
                        {
                            "condition": condition,
                            "level": float(level),
                            "alignment": alignment,
                            "method": method,
                            "mean": float("nan"),
                            "std": float("nan"),
                            "ci95": float("nan"),
                            "num_scene_trials": 0,
                        }
                    )
                    continue
                if not pooled:
                    raise FileNotFoundError(
                        f"No finite {metric_key} results for "
                        f"{condition}={level}, {alignment}"
                    )
                pooled_array = np.asarray(pooled, dtype=float)
                std = float(np.std(pooled_array, ddof=1)) if len(pooled) > 1 else 0.0
                ci95 = 1.96 * std / np.sqrt(len(pooled))
                rows.append(
                    {
                        "condition": condition,
                        "level": float(level),
                        "alignment": alignment,
                        "method": method,
                        "mean": float(np.mean(pooled_array)),
                        "std": std,
                        "ci95": float(ci95),
                        "num_scene_trials": int(len(pooled)),
                    }
                )
    return rows


def plot(
    rows: list[dict[str, object]],
    output: Path,
    ylabel: str,
    font_scale: float = 1.0,
    padding_ratio: float = 0.02,
    show_legend: bool = True,
) -> None:
    font_scale = max(0.1, float(font_scale))
    padding_ratio = max(0.0, float(padding_ratio))
    style_context = {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.sf": "Times New Roman",
            "mathtext.cal": "Times New Roman:italic",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 18.0 * font_scale,
            "axes.titlesize": 18.0 * font_scale,
            "axes.titleweight": "bold",
            "axes.labelsize": 18.0 * font_scale,
            "xtick.labelsize": 18.0 * font_scale,
            "ytick.labelsize": 18.0 * font_scale,
            "legend.fontsize": 18.0 * font_scale,
        }
    with plt.rc_context(
        {
            **style_context,
        }
    ):
        # CVPR full-width figure with the two retained robustness panels in a
        # compact horizontal layout. The vertical-noise panel is disabled in
        # PLOT_CONDITIONS rather than deleting its experiment/statistics.
        fig, axes = plt.subplots(
            1, 2, figsize=(6.4, 2.7), sharey=False
        )
        styles = {
            "prior_yaw": dict(color="#E67E22", linestyle="--", marker="s"),
            "prior_pose": dict(color="#8E63B0", linestyle="-.", marker="D"),
            "pi3x": dict(color="#579D68", linestyle=":", marker="^"),
        }
        panel_names = (
            "(a) Horizontal noise",
            "(b) Missing priors",
        )
        plotted_conditions = [
            (condition, CONDITIONS[condition]) for condition in PLOT_CONDITIONS
        ]
        for axis, (condition, (levels, xlabel)), title in zip(
            axes, plotted_conditions, panel_names
        ):
            if condition == "missing":
            # Retained-prior counts are discrete experimental settings. Use a
            # categorical axis so every marker lies exactly on a labeled tick
            # and the low-count settings do not crowd together.
                x = np.arange(len(levels), dtype=float)
                axis.set_xticks(x, [str(int(level)) for level in levels])
            else:
                x = np.asarray(levels, dtype=float)
                axis.set_xticks(x, [f"{level:g}" for level in levels])
            for alignment, method in METHODS:
                selected = [
                    row for row in rows
                    if row["condition"] == condition and row["alignment"] == alignment
                ]
                mean = np.asarray([row["mean"] for row in selected], dtype=float)
                ci95 = np.asarray([row["ci95"] for row in selected], dtype=float)
                style = styles[alignment]
                axis.plot(
                    x, mean, label=method,
                    linewidth=1.6, markersize=4.2 * font_scale, **style,
                )
                axis.fill_between(
                    x, mean - ci95, mean + ci95,
                    color=style["color"], alpha=0.18,
                )
            if condition == "missing":
            # Full Sim(3) is under-constrained with only two correspondences.
                axis.text(
                    x[-1], 0.96, "×\nN/A",
                    transform=axis.get_xaxis_transform(),
                    ha="center", va="top", color="#666666",
                    fontsize=18.0 * font_scale, fontweight="bold",
                )
            axis.set_title(title, loc="center", pad=4.0 * font_scale)
            axis.set_xlabel(xlabel)
            axis.margins(x=padding_ratio, y=padding_ratio)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
        # Use figure coordinates explicitly so the shared label remains
        # vertically centered after tight-bbox export.
        fig.text(
            0.018, 0.515, ylabel,
            ha="center", va="center", rotation="vertical",
            fontsize=18.0 * font_scale,
        )
        if show_legend:
            handles, labels = axes[0].get_legend_handles_labels()
            # Repeat the compact legend in both panels so each subplot remains
            # self-contained when cropped or referenced independently.
            for axis in axes:
                axis.legend(
                    handles, labels,
                    loc="upper left", ncol=1, frameon=True,
                    bbox_to_anchor=(0.02, 0.98),
                    borderaxespad=0.0,
                    borderpad=0.2,
                    framealpha=0.82,
                    facecolor="white",
                    edgecolor="none",
                    handletextpad=0.4,
                    labelspacing=0.12,
                )
        fig.subplots_adjust(
            left=0.10, right=0.99, bottom=0.20, top=0.88, wspace=0.27
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        fig.savefig(output.with_suffix(".png"), dpi=300, **save_kwargs)
        fig.savefig(output.with_suffix(".pdf"), **save_kwargs)
        fig.savefig(output.with_suffix(".svg"), **save_kwargs)
        plt.close(fig)
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--padding-ratio", type=float, default=0.02)
    parser.add_argument(
        "--show-legend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the shared legend inside the figure canvas (default: enabled).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["chamfer-l1"],
        help=(
            "Metrics to plot, one figure per metric. Friendly aliases: "
            "chamfer-l1, ate. Exact per-scene JSON metric keys are also accepted."
        ),
    )
    args = parser.parse_args()
    root = args.results_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    metric_names = [item for group in args.metrics for item in group.split(",") if item]
    seen = set()
    for metric_name in metric_names:
        metric_key, ylabel, slug = resolve_metric(metric_name)
        if metric_key in seen:
            continue
        seen.add(metric_key)
        metric_output = output.with_name(f"{output.name}_{slug}")
        rows = load_rows(root, metric_key)
        plot(
            rows,
            metric_output,
            ylabel,
            font_scale=args.font_scale,
            padding_ratio=args.padding_ratio,
            show_legend=args.show_legend,
        )
        print(f"Saved {metric_key} prior robustness figure: {metric_output}")


if __name__ == "__main__":
    main()
