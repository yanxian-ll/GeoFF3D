#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Figure 5.2: Relative improvement of different fine-tuning data over MA-Pretrained.

This script reuses the loading / parsing utilities from
scripts/viz/table52_53_54_domain_adaptation.py. It reads
per_dataset_results.json under dense_*_view/<method>/, averages each metric over
selected view settings, then computes relative improvement (%) w.r.t.
MA-Pretrained:

    improvement(%) = (baseline - compared) / baseline * 100

Since all selected metrics are lower-is-better, positive values mean the
fine-tuned setting improves over MA-Pretrained, while negative values mean it is
worse.

Default figure layout:
  - 2 x 2 subplots: RelDepth / Chamfer / Ray / Pose ATE
  - x-axis groups: UseGeo / Enrich / Urban / A3D-Real / Avg.
  - bars: MA-FT-Public / MA-FT-A3D-Syn / MA-FT-A3D-Full

Example:
  python scripts/viz/figure52_relative_improvement_domain_adaptation.py \
    --benchmarking experiments/mapanything/benchmarking \
    --views 8,16,24,32 \
    --output experiments/mapanything/benchmarking/figures/figure52
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Make sibling import work when this script is placed under scripts/viz/.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from table52_53_54_domain_adaptation import (  # type: ignore
    DEFAULT_DATASETS,
    DEFAULT_SETTINGS,
    DatasetSpec,
    MetricSpec,
    SettingSpec,
    TableSpec,
    collect_table_values,
    dedupe_warnings,
    ensure_json_filename,
    is_finite_number,
    load_setting_jsons,
    mean_finite,
    parse_dataset_specs,
    parse_setting_specs,
    parse_views,
    safe_stem,
)


PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def build_metric_specs(args: argparse.Namespace) -> "OrderedDict[str, MetricSpec]":
    return OrderedDict(
        [
            (
                "reldepth",
                MetricSpec(
                    alias="reldepth",
                    key=args.depth_key,
                    display="RelDepth",
                    precision=int(args.precision_depth),
                ),
            ),
            (
                "chamfer",
                MetricSpec(
                    alias="chamfer",
                    key=args.chamfer_key,
                    display="Chamfer",
                    precision=int(args.precision_chamfer),
                ),
            ),
            (
                "ray",
                MetricSpec(
                    alias="ray",
                    key=args.ray_key,
                    display="Ray",
                    precision=int(args.precision_ray),
                ),
            ),
            (
                "pose_ate",
                MetricSpec(
                    alias="pose_ate",
                    key=args.pose_ate_key,
                    display="Pose ATE",
                    precision=int(args.precision_pose_ate),
                ),
            ),
        ]
    )


def make_collection_table(metrics: "OrderedDict[str, MetricSpec]") -> TableSpec:
    return TableSpec(
        table_id="5.2-5.4-all-metrics",
        caption="All domain-adaptation metrics for Figure 5.2.",
        latex_label="fig:relative_improvement_domain_adaptation",
        stem="figure52_relative_improvement_domain_adaptation",
        metrics=metrics,
        layout="dataset_metric",
    )


def ordered_compare_settings(
    settings: "OrderedDict[str, SettingSpec]",
    baseline_label: str,
    compare_labels_spec: str | None,
) -> list[str]:
    if compare_labels_spec and str(compare_labels_spec).strip():
        labels = [x.strip() for x in str(compare_labels_spec).split(",") if x.strip()]
    else:
        labels = [k for k in settings.keys() if k != baseline_label]
    for label in labels:
        if label not in settings:
            raise ValueError(f"Unknown compare setting: {label!r}. Available: {list(settings.keys())}")
    if baseline_label not in settings:
        raise ValueError(f"Baseline setting {baseline_label!r} not found in settings: {list(settings.keys())}")
    if not labels:
        raise ValueError("No compare settings selected.")
    return labels


def compute_relative_improvement(baseline: float, compared: float) -> float:
    if not is_finite_number(baseline) or not is_finite_number(compared):
        return float("nan")
    baseline = float(baseline)
    compared = float(compared)
    if math.isclose(baseline, 0.0, abs_tol=1e-12):
        return float("nan")
    return (baseline - compared) / baseline * 100.0


def compute_improvement_values(
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
    baseline_label: str,
    compare_labels: list[str],
    include_avg_group: bool,
) -> tuple[dict, list[dict]]:
    """
    Returns:
      improvement_values[setting][dataset_or_avg][metric_alias] = rel improvement (%)
      details rows
    """
    out: dict = OrderedDict()
    rows: list[dict] = []

    for setting in compare_labels:
        out[setting] = OrderedDict()
        for dataset_label in datasets.keys():
            out[setting][dataset_label] = OrderedDict()
            for metric_alias in metrics.keys():
                base_v = values.get(baseline_label, {}).get(dataset_label, {}).get(metric_alias, float("nan"))
                cur_v = values.get(setting, {}).get(dataset_label, {}).get(metric_alias, float("nan"))
                rel = compute_relative_improvement(base_v, cur_v)
                out[setting][dataset_label][metric_alias] = rel
                rows.append(
                    {
                        "setting": setting,
                        "baseline_setting": baseline_label,
                        "dataset": dataset_label,
                        "metric_alias": metric_alias,
                        "metric_display": metrics[metric_alias].display,
                        "baseline_value": base_v,
                        "compared_value": cur_v,
                        "relative_improvement_pct": rel,
                    }
                )

        if include_avg_group:
            out[setting]["Avg."] = OrderedDict()
            for metric_alias in metrics.keys():
                vals = [
                    out[setting][dataset_label].get(metric_alias, float("nan"))
                    for dataset_label in datasets.keys()
                ]
                avg_rel = mean_finite(vals)
                out[setting]["Avg."][metric_alias] = avg_rel
                rows.append(
                    {
                        "setting": setting,
                        "baseline_setting": baseline_label,
                        "dataset": "Avg.",
                        "metric_alias": metric_alias,
                        "metric_display": metrics[metric_alias].display,
                        "baseline_value": float("nan"),
                        "compared_value": float("nan"),
                        "relative_improvement_pct": avg_rel,
                    }
                )

    return out, rows


def write_improvement_summary_csv(
    path: Path,
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
    x_groups: list[str],
    metrics: "OrderedDict[str, MetricSpec]",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["setting"]
    for group in x_groups:
        for metric_alias in metrics.keys():
            fieldnames.append(f"{safe_stem(group)}_{metric_alias}")

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for setting in values.keys():
            row = {"setting": setting}
            for group in x_groups:
                for metric_alias in metrics.keys():
                    row[f"{safe_stem(group)}_{metric_alias}"] = values[setting].get(group, {}).get(metric_alias, float("nan"))
            writer.writerow(row)


def write_details_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    # Details rows are mixed from two sources:
    #   1) collect_table_values(): benchmark aggregation details
    #   2) compute_improvement_values(): relative-improvement details
    # Their keys are different, so we must use the union of all keys instead
    # of only rows[0].keys(). Otherwise csv.DictWriter raises:
    # ValueError: dict contains fields not in fieldnames.
    preferred_order = [
        "table",
        "setting",
        "baseline_setting",
        "dataset",
        "metric_alias",
        "metric_display",
        "metric_key",
        "baseline_value",
        "compared_value",
        "relative_improvement_pct",
        "value_mean_over_views",
        "n_views_used",
        "views_used",
        "dataset_keys_used",
        "json_paths_used",
        "setting_subdir_candidates",
        "setting_subdirs_used",
    ]

    fieldnames: list[str] = []
    seen: set[str] = set()
    for key in preferred_order:
        if any(key in row for row in rows):
            fieldnames.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rr = dict(row)
            for k, v in list(rr.items()):
                if isinstance(v, float) and not math.isfinite(v):
                    rr[k] = ""
            writer.writerow(rr)


def nice_metric_title(metric_alias: str, metric: MetricSpec, panel_letter: str | None = None) -> str:
    title_map = {
        "reldepth": "Relative Depth Error",
        "chamfer": "Chamfer-L1",
        "ray": "Ray Error",
        "pose_ate": "Pose ATE",
    }
    core = title_map.get(metric_alias, metric.display)
    if panel_letter:
        return f"({panel_letter}) {core}"
    return core


def plot_figure52(
    improvement_values: Mapping[str, Mapping[str, Mapping[str, float]]],
    metrics: "OrderedDict[str, MetricSpec]",
    compare_labels: list[str],
    x_groups: list[str],
    title: str | None,
    ylabel: str,
    legend_ncol: int,
    annotate: bool,
    rotate_xticks: float,
    ncols: int,
    figsize_scale: float,
) -> plt.Figure:
    metric_items = list(metrics.items())
    n_metrics = len(metric_items)
    ncols = max(1, min(int(ncols), n_metrics))
    nrows = int(math.ceil(n_metrics / ncols))
    fig_w = figsize_scale * 4.8 * ncols
    fig_h = figsize_scale * 3.8 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = list(axes.flat)

    x = np.arange(len(x_groups), dtype=float)
    n_bars = len(compare_labels)
    total_width = 0.82
    bar_width = total_width / max(1, n_bars)
    offsets = (np.arange(n_bars, dtype=float) - (n_bars - 1) / 2.0) * bar_width

    y_min_global = float("inf")
    y_max_global = float("-inf")
    panel_idx = 0

    for ax, (metric_alias, metric) in zip(axes_flat, metric_items):
        for i, setting in enumerate(compare_labels):
            ys = [improvement_values.get(setting, {}).get(group, {}).get(metric_alias, float("nan")) for group in x_groups]
            ys_plot = [float(y) if is_finite_number(y) else float("nan") for y in ys]
            xpos = x + offsets[i]
            bars = ax.bar(xpos, ys_plot, width=bar_width * 0.92, label=setting)

            finite_vals = [float(y) for y in ys_plot if is_finite_number(y)]
            if finite_vals:
                y_min_global = min(y_min_global, min(finite_vals))
                y_max_global = max(y_max_global, max(finite_vals))

            if annotate:
                for rect, y in zip(bars, ys_plot):
                    if not is_finite_number(y):
                        continue
                    height = float(y)
                    va = "bottom" if height >= 0 else "top"
                    offset = 1.0 if height >= 0 else -1.0
                    ax.text(
                        rect.get_x() + rect.get_width() / 2.0,
                        height + offset,
                        f"{height:.1f}",
                        ha="center",
                        va=va,
                        fontsize=8,
                        rotation=0,
                    )

        panel_letter = PANEL_LETTERS[panel_idx] if panel_idx < len(PANEL_LETTERS) else None
        panel_idx += 1
        ax.set_title(nice_metric_title(metric_alias, metric, panel_letter), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(x_groups, rotation=rotate_xticks)
        ax.set_ylabel(ylabel)
        ax.axhline(0.0, linewidth=1.0)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

    for ax in axes_flat[n_metrics:]:
        ax.axis("off")

    finite_extrema = [v for v in [y_min_global, y_max_global] if math.isfinite(v)]
    if finite_extrema:
        y_min = y_min_global
        y_max = y_max_global
        if math.isclose(y_min, y_max, rel_tol=1e-12, abs_tol=1e-12):
            y_min -= 1.0
            y_max += 1.0
        span = y_max - y_min
        lower = y_min - max(2.0, span * 0.12)
        upper = y_max + max(2.0, span * 0.18)
        for ax in axes_flat[:n_metrics]:
            ax.set_ylim(lower, upper)

    handles, labels = axes_flat[0].get_legend_handles_labels() if axes_flat else ([], [])
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, legend_ncol), frameon=False, bbox_to_anchor=(0.5, 1.01))

    if title:
        fig.suptitle(title, y=1.04, fontsize=13)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot Figure 5.2: Relative improvement of different fine-tuning data over MA-Pretrained.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmarking", type=Path, default=Path("experiments/mapanything/benchmarking"))
    p.add_argument("--output", type=Path, default=None, help="Output directory. Default: <benchmarking>/figures/figure52")
    p.add_argument("--views", type=str, default="8,16,24,32", help="Comma-separated views or 'all'.")
    p.add_argument(
        "--settings",
        nargs="*",
        default=None,
        help="Rows as Label=subdir1|subdir2. Defaults to MA-Pretrained/Public/A3D-Syn/A3D-Full.",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Datasets as Label=alias1|alias2. Defaults to UseGeo/Enrich/Urban/A3D-Real.",
    )
    p.add_argument("--baseline", type=str, default="MA-Pretrained", help="Baseline setting used in relative improvement computation.")
    p.add_argument("--compare-settings", type=str, default=None, help="Comma-separated settings to compare against baseline. Default: all non-baseline settings in row order.")
    p.add_argument("--json-name", type=str, default="per_dataset_results.json")
    p.add_argument("--depth-key", type=str, default="abs_depth_rel_scale_aligned")
    p.add_argument("--chamfer-key", type=str, default="abs_fused_pc_chamfer_l1")
    p.add_argument("--ray-key", type=str, default="ray_dir_mean_angle_deg")
    p.add_argument("--pose-ate-key", type=str, default="abs_pose_ate")
    p.add_argument("--agg", choices=["mean", "median", "first", "second"], default="mean", help="How to reduce list-valued JSON entries.")
    p.add_argument("--precision-depth", type=int, default=3)
    p.add_argument("--precision-chamfer", type=int, default=3)
    p.add_argument("--precision-ray", type=int, default=2)
    p.add_argument("--precision-pose-ate", type=int, default=3)
    p.add_argument("--figure-stem", type=str, default="figure52_relative_improvement_domain_adaptation")
    p.add_argument("--title", type=str, default="Figure 5.2. Relative improvement of different fine-tuning data over MA-Pretrained", help="Figure title. Use empty string to disable.")
    p.add_argument("--ylabel", type=str, default="Relative Improvement (%)")
    p.add_argument("--ncols", type=int, default=2, help="Number of subplot columns.")
    p.add_argument("--figsize-scale", type=float, default=1.0)
    p.add_argument("--legend-ncol", type=int, default=3)
    p.add_argument("--rotate-xticks", type=float, default=0.0)
    p.add_argument("--annotate", action="store_true", help="Annotate each bar with its value.")
    p.add_argument("--no-avg-group", action="store_true", help="Do not append an 'Avg.' x-axis group averaged over datasets.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--save-svg", action="store_true")
    p.add_argument("--strict", action="store_true", help="Raise on missing files/datasets/metrics instead of warning.")
    p.add_argument("--quiet", action="store_true", help="Suppress warning printout.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    root = Path(args.benchmarking)
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmarking root not found: {root}")

    output_dir = Path(args.output) if args.output is not None else (root / "figures" / "figure52")
    output_dir.mkdir(parents=True, exist_ok=True)

    views = parse_views(args.views, root)
    if not views:
        raise ValueError(f"No views selected/found under {root}")

    settings = parse_setting_specs(args.settings)
    datasets = parse_dataset_specs(args.datasets)
    compare_labels = ordered_compare_settings(settings, baseline_label=args.baseline, compare_labels_spec=args.compare_settings)
    json_name = ensure_json_filename(args.json_name)
    metrics = build_metric_specs(args)
    collect_spec = make_collection_table(metrics)

    loaded_jsons, load_warnings = load_setting_jsons(
        root=root,
        views=views,
        settings=settings,
        json_name=json_name,
        strict=bool(args.strict),
    )

    values, collect_details, collect_warnings = collect_table_values(
        loaded_jsons=loaded_jsons,
        settings=settings,
        datasets=datasets,
        table=collect_spec,
        agg=args.agg,
        strict=bool(args.strict),
    )

    improvement_values, improvement_rows = compute_improvement_values(
        values=values,
        datasets=datasets,
        metrics=metrics,
        baseline_label=args.baseline,
        compare_labels=compare_labels,
        include_avg_group=not bool(args.no_avg_group),
    )

    x_groups = list(datasets.keys()) + ([] if args.no_avg_group else ["Avg."])
    title = args.title if str(args.title).strip() else None
    fig = plot_figure52(
        improvement_values=improvement_values,
        metrics=metrics,
        compare_labels=compare_labels,
        x_groups=x_groups,
        title=title,
        ylabel=args.ylabel,
        legend_ncol=int(args.legend_ncol),
        annotate=bool(args.annotate),
        rotate_xticks=float(args.rotate_xticks),
        ncols=int(args.ncols),
        figsize_scale=float(args.figsize_scale),
    )

    stem = safe_stem(args.figure_stem)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    svg_path = output_dir / f"{stem}.svg"
    csv_path = output_dir / f"{stem}.csv"
    details_csv_path = output_dir / f"{stem}_details.csv"
    metadata_path = output_dir / f"{stem}_metadata.json"

    fig.savefig(png_path, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    if args.save_svg:
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    write_improvement_summary_csv(csv_path, improvement_values, x_groups=x_groups, metrics=metrics)

    all_details = []
    all_details.extend(collect_details)
    for row in improvement_rows:
        all_details.append({"table": "figure5.2", **row})
    write_details_csv(details_csv_path, all_details)

    all_warnings = dedupe_warnings(load_warnings + collect_warnings)
    metadata = {
        "figure_id": "5.2",
        "title": title,
        "benchmarking_root": str(root),
        "output_dir": str(output_dir),
        "views": views,
        "baseline": args.baseline,
        "compare_settings": compare_labels,
        "settings": {k: list(v.subdirs) for k, v in settings.items()},
        "datasets": {k: list(v.aliases) for k, v in datasets.items()},
        "metrics": {k: {"key": v.key, "display": v.display} for k, v in metrics.items()},
        "json_name": json_name,
        "agg": args.agg,
        "x_groups": x_groups,
        "relative_improvement_formula": "(baseline - compared) / baseline * 100",
        "outputs": {
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
            "figure_svg": str(svg_path) if args.save_svg else None,
            "summary_csv": str(csv_path),
            "details_csv": str(details_csv_path),
        },
        "warnings": all_warnings,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote Figure 5.2 PNG:      {png_path}")
    print(f"Wrote Figure 5.2 PDF:      {pdf_path}")
    if args.save_svg:
        print(f"Wrote Figure 5.2 SVG:      {svg_path}")
    print(f"Wrote Figure 5.2 CSV:      {csv_path}")
    print(f"Wrote Figure 5.2 details:  {details_csv_path}")
    print(f"Wrote Figure 5.2 metadata: {metadata_path}")

    if all_warnings and not args.quiet:
        print("\nWarnings:")
        for w in all_warnings[:80]:
            print(f"  - {w}")
        if len(all_warnings) > 80:
            print(f"  ... {len(all_warnings) - 80} more warnings. See {metadata_path}")


if __name__ == "__main__":
    main()


"""
python scripts/viz/figure52_relative_improvement_domain_adaptation.py \
  --benchmarking experiments/mapanything/benchmarking \
  --views 8,16,24,32 \
  --output experiments/mapanything/benchmarking/figures/figure52
"""
