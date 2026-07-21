#!/usr/bin/env python3
"""Plot runtime and memory scalability for interval5_AMtown01."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np


IMAGE_COUNTS = (100, 250, 500, 1000, 2000)
SCENE_PREFIX = "interval5_AMtown01_n"
METHODS = (
    ("GeoFF3D", "geoff3d"),
    ("VGGT-SLAM 2.0", "vggt_slam2.0"),
    ("LingBot-Map", "lingbot-map"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root containing one result directory per method.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path without extension; .png, .pdf, .svg, and .csv are written.",
    )
    parser.add_argument("--font-scale", type=float, default=1.2)
    parser.add_argument("--padding-ratio", type=float, default=0.02)
    parser.add_argument(
        "--show-legend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show legends inside the subplot axes (default: enabled).",
    )
    return parser.parse_args()


def load_rows(results_root: Path) -> list[dict[str, object]]:
    rows = []
    missing = []
    for label, method_dir in METHODS:
        for count in IMAGE_COUNTS:
            path = (
                results_root
                / method_dir
                / "uavscenes"
                / f"{SCENE_PREFIX}{count}"
                / "processing_time.json"
            )
            if not path.is_file():
                missing.append(path)
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            total = data.get("processing_time_seconds", data.get("total_seconds"))
            memory = data.get("peak_gpu_memory_allocated_mib")
            if total is None or memory is None:
                raise KeyError(
                    f"{path} lacks total runtime or peak GPU memory. Re-run with the updated code."
                )
            rows.append(
                {
                    "method": label,
                    "method_dir": method_dir,
                    "num_images": int(count),
                    "total_runtime_seconds": float(total),
                    "peak_gpu_memory_mib": float(memory),
                    "footprint_preprocessing_seconds": float(data.get("chunking_seconds", np.nan)),
                    "model_inference_seconds": float(data.get("model_prediction_seconds", np.nan)),
                    "hierarchical_alignment_seconds": float(data.get("post_alignment_seconds", np.nan)),
                }
            )
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing scalability results:\n{formatted}")
    return rows


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(
    rows: list[dict[str, object]],
    output: Path,
    font_scale: float = 1.2,
    padding_ratio: float = 0.02,
    show_legend: bool = True,
) -> None:
    font_scale = max(0.1, float(font_scale))
    padding_ratio = max(0.0, float(padding_ratio))
    labels = [f"{value // 1000}k" if value >= 1000 else str(value) for value in IMAGE_COUNTS]
    x = np.arange(len(IMAGE_COUNTS))
    colors = {
        "chunking": "#4C78A8",
        "inference": "#F58518",
        "alignment": "#54A24B",
    }

    # Paper-oriented sizing: the less panoramic aspect ratio and deliberately
    # large source fonts remain readable after scaling to a two-column width.
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
        # Defaults evaluate to at least 18 pt at font_scale=1.2.
        "font.size": 15.0 * font_scale,
        "axes.titlesize": 16.0 * font_scale,
        "axes.titleweight": "bold",
        "axes.labelsize": 15.0 * font_scale,
        "xtick.labelsize": 15.0 * font_scale,
        "ytick.labelsize": 15.0 * font_scale,
        "legend.fontsize": 15.0 * font_scale,
    }
    with plt.rc_context(style_context):
        # Panels (a) and (c) share the same runtime unit and are overlaid in
        # the left axis: bars show our time breakdown and curves show total
        # runtime for every method. Panel (b) retains the memory comparison.
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.4))

        ours_rows = [row for row in rows if row["method"] == "GeoFF3D"]
        chunking = np.asarray([row["footprint_preprocessing_seconds"] for row in ours_rows], dtype=float)
        inference = np.asarray([row["model_inference_seconds"] for row in ours_rows], dtype=float)
        alignment = np.asarray([row["hierarchical_alignment_seconds"] for row in ours_rows], dtype=float)

        # Draw breakdown bars first so the total-runtime curves remain visible
        # on top. All quantities use minutes on the same y axis.
        axes[0].bar(
            x, chunking / 60.0, width=0.62, color=colors["chunking"], alpha=0.72,
            label="Chunking", zorder=1,
        )
        axes[0].bar(
            x, inference / 60.0, width=0.62, bottom=chunking / 60.0,
            color=colors["inference"], alpha=0.72,
            label="Inference", zorder=1,
        )
        axes[0].bar(
            x, alignment / 60.0, width=0.62,
            bottom=(chunking + inference) / 60.0,
            color=colors["alignment"], alpha=0.72,
            label="Alignment", zorder=1,
        )

        method_colors = ("#3B5BA5", "#E45756", "#72B7B2")
        method_handles: dict[str, object] = {}
        for (method_label, _), color in zip(METHODS, method_colors):
            method_rows = [row for row in rows if row["method"] == method_label]
            total = np.asarray([row["total_runtime_seconds"] for row in method_rows], dtype=float)
            memory_gib = np.asarray([row["peak_gpu_memory_mib"] for row in method_rows], dtype=float) / 1024.0
            runtime_line, = axes[0].plot(
                x, total / 60.0, marker="o", markersize=6.0 * font_scale, linewidth=2.5,
                label=method_label, color=color, zorder=5,
            )
            method_handles[method_label] = runtime_line
            axes[1].plot(
                x, memory_gib, marker="o", markersize=6.0 * font_scale, linewidth=2.5,
                label=method_label, color=color, zorder=5,
            )

        axes[0].set_title("(a) Runtime breakdown (m)")
        if show_legend:
            handles, legend_labels = axes[0].get_legend_handles_labels()
            handle_by_label = dict(zip(legend_labels, handles))
            component_order = ("Chunking", "Inference", "Alignment")
            runtime_legend = axes[0].legend(
                [handle_by_label[label] for label in component_order],
                list(component_order),
                loc="upper left", ncol=1, frameon=True,
                bbox_to_anchor=(0.02, 0.98), borderaxespad=0.0,
                borderpad=0.25, labelspacing=0.18, handletextpad=0.45,
                framealpha=0.82, facecolor="white", edgecolor="none",
            )
            # Curves use zorder=5 and remain visible above the legend.
            runtime_legend.set_zorder(2)

            method_order = ("VGGT-SLAM 2.0", "LingBot-Map", "GeoFF3D")
            fig.legend(
                [method_handles[label] for label in method_order],
                list(method_order),
                loc="upper center", ncol=3, frameon=False,
                bbox_to_anchor=(0.535, 0.995),
                columnspacing=0.8, handletextpad=0.4,
            )

        axes[1].set_title("(b) Peak GPU memory (G)")
        axes[1].set_yscale("log", base=2)
        memory_ticks = [2, 4, 8, 16, 32]
        axes[1].yaxis.set_major_locator(FixedLocator(memory_ticks))
        axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
        axes[1].set_ylim(2.5, 30.0)
        axes[1].minorticks_off()
        for axis in axes:
            axis.set_xlabel("Number of images")
            axis.set_xticks(x, labels)
            axis.margins(x=padding_ratio, y=padding_ratio)
            axis.tick_params(axis="both", width=1.1, length=4)
            axis.grid(axis="y", alpha=0.25, linewidth=0.8)
            axis.set_axisbelow(True)

        fig.subplots_adjust(
            left=0.08, right=0.99, bottom=0.18, top=0.79, wspace=0.12
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        fig.savefig(output.with_suffix(".png"), dpi=300, **save_kwargs)
        fig.savefig(output.with_suffix(".pdf"), **save_kwargs)
        fig.savefig(output.with_suffix(".svg"), **save_kwargs)
        plt.close(fig)
    save_csv(rows, output.with_suffix(".csv"))


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results_root.expanduser().resolve())
    plot(
        rows,
        args.output.expanduser().resolve(),
        font_scale=args.font_scale,
        padding_ratio=args.padding_ratio,
        show_legend=args.show_legend,
    )
    print(f"Saved scalability figure and CSV: {args.output}")


if __name__ == "__main__":
    main()
