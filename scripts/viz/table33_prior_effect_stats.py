#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Table 3.3-style prior-effect statistics from dense_n_view benchmark outputs.

Table 3.3:
  Effect of camera and pose priors in zero-shot UAV evaluation.
  Values indicate relative change compared with RGB-only input.

This script is intentionally built on top of scripts/viz/table32_zero_shot_stats.py.
Put this file next to table32_zero_shot_stats.py, e.g.:

  scripts/viz/table33_prior_effect_stats.py

Expected benchmark layout:

  <benchmarking_root>/
    dense_8_view/hunyuan/per_dataset_results.json        # RGB-only baseline
    dense_8_view/hunyuan_csfm/per_dataset_results.json   # C prior
    dense_8_view/hunyuan_psfm/per_dataset_results.json   # P prior
    dense_8_view/hunyuan_mvs/per_dataset_results.json    # CP prior
    dense_8_view/da3/per_dataset_results.json            # RGB-only baseline
    dense_8_view/da3_mvs/per_dataset_results.json        # CP prior only
    ...

Default behavior:
  - read views 8,16,24,32;
  - aggregate selected views for each model/prior/dataset/metric;
  - average selected datasets into one value per model/prior/metric;
  - compute relative change against the corresponding RGB-only baseline:

        relative_delta_percent = 100 * (prior_value - rgb_value) / abs(rgb_value)

    For lower-is-better error metrics, negative values mean improvement.

Examples:

  # Default Table 3.3 over 8/16/24/32 views
  python scripts/viz/table33_prior_effect_stats.py \
    --benchmarking experiments/mapanything/benchmarking \
    --output experiments/mapanything/benchmarking/tables/table33

  # Use all discovered dense_*_view folders
  python scripts/viz/table33_prior_effect_stats.py \
    --benchmarking experiments/mapanything/benchmarking \
    --views all

  # Only report selected metrics
  python scripts/viz/table33_prior_effect_stats.py \
    --metrics chamfer,ray,pose_ate,depth_absrel

  # Only report selected models. DA3 still has CP only by default.
  python scripts/viz/table33_prior_effect_stats.py \
    --models Hunyuan,MapAnything,DA3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

# Make sure importing a sibling script works when this file is copied elsewhere
# or executed as python scripts/viz/table33_prior_effect_stats.py.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from table32_zero_shot_stats import (  # type: ignore
        DatasetSpec,
        MethodSpec,
        MetricSpec,
        DEFAULT_DATASETS,
        DEFAULT_VIEWS,
        METRIC_ALIASES,
        aggregate_views,
        collect_one_view,
        is_finite_number,
        latex_escape,
        mean_finite,
        parse_datasets,
        parse_metrics,
        parse_views,
        safe_stem,
        split_csv_like,
        union_dataset_specs,
    )
except Exception as exc:  # pragma: no cover - user-facing error path
    raise ImportError(
        "Failed to import table32_zero_shot_stats.py. "
        "Please place table33_prior_effect_stats.py next to scripts/viz/table32_zero_shot_stats.py."
    ) from exc


# -----------------------------------------------------------------------------
# Prior group specs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorGroupSpec:
    """One RGB-only baseline and its prior-enabled variants."""

    model: str
    rgb_subdir: str
    prior_subdirs: "OrderedDict[str, str]"


DEFAULT_PRIOR_GROUPS: "OrderedDict[str, PriorGroupSpec]" = OrderedDict(
    [
        (
            "Hunyuan",
            PriorGroupSpec(
                model="Hunyuan",
                rgb_subdir="hunyuan",
                prior_subdirs=OrderedDict(
                    [
                        ("C", "hunyuan_csfm"),
                        ("P", "hunyuan_psfm"),
                        ("CP", "hunyuan_mvs"),
                    ]
                ),
            ),
        ),
        (
            "MapAnything",
            PriorGroupSpec(
                model="MapAnything",
                rgb_subdir="mapa",
                prior_subdirs=OrderedDict(
                    [
                        ("C", "mapa_csfm"),
                        ("P", "mapa_psfm"),
                        ("CP", "mapa_mvs"),
                    ]
                ),
            ),
        ),
        (
            "Pi3X",
            PriorGroupSpec(
                model="Pi3X",
                rgb_subdir="pi3x",
                prior_subdirs=OrderedDict(
                    [
                        ("C", "pi3x_csfm"),
                        ("P", "pi3x_psfm"),
                        ("CP", "pi3x_mvs"),
                    ]
                ),
            ),
        ),
        # Important: DA3 has only CP prior in the current benchmark scripts.
        # Do not create DA3-C or DA3-P rows.
        (
            "DA3",
            PriorGroupSpec(
                model="DA3",
                rgb_subdir="da3",
                prior_subdirs=OrderedDict(
                    [
                        ("CP", "da3_mvs"),
                    ]
                ),
            ),
        ),

        (
            "UAV-MapAnything",
            PriorGroupSpec(
                model="UAV-MapAnything",
                rgb_subdir="uav_mapa_aug_images_only|uav_mapa_aug_images_only_1|uav_mapa|mapa-ft-a3dsyn",
                prior_subdirs=OrderedDict(
                    [
                        ("C", "uav_mapa_aug_csfm|uav_mapa_aug_csfm_1|uav_mapa_csfm"),
                        ("P", "uav_mapa_aug_psfm|uav_mapa_aug_psfm_1|uav_mapa_psfm"),
                        ("CP", "uav_mapa_aug_mvs|uav_mapa_aug_mvs_1"),
                    ]
                ),
            ),
        ),
        (
            "UAV-Pi3X",
            PriorGroupSpec(
                model="UAV-Pi3X",
                rgb_subdir="uav_pi3x_aug_images_only|pi3x_aug_images_only|uav_pi3x",
                prior_subdirs=OrderedDict(
                    [
                        ("C", "uav_pi3x_aug_csfm|pi3x_aug_csfm|uav_pi3x_csfm"),
                        ("P", "uav_pi3x_aug_psfm|pi3x_aug_psfm|uav_pi3x_psfm"),
                        ("CP", "uav_pi3x_aug_mvs|pi3x_aug_mvs|uav_pi3x_mvs"),
                    ]
                ),
            ),
        ),
    ]
)

DEFAULT_TABLE33_METRICS = "chamfer,ray,pose_ate,depth_absrel"


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def finite_or_nan(x: object) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def relative_delta(prior_value: float, rgb_value: float, mode: str) -> float:
    """Compute prior-vs-RGB change.

    mode:
      percent: 100 * (prior - rgb) / abs(rgb)
      fraction: (prior - rgb) / abs(rgb)
      raw: prior - rgb
      ratio: prior / rgb
    """
    p = finite_or_nan(prior_value)
    b = finite_or_nan(rgb_value)
    if not math.isfinite(p) or not math.isfinite(b):
        return float("nan")
    if mode == "raw":
        return p - b
    if mode == "ratio":
        if abs(b) <= 1e-12:
            return float("nan")
        return p / b
    if abs(b) <= 1e-12:
        return float("nan")
    frac = (p - b) / abs(b)
    if mode == "fraction":
        return frac
    if mode == "percent":
        return 100.0 * frac
    raise ValueError(f"Unsupported change mode: {mode}")


def improvement_value(delta: float, metric: MetricSpec, change_mode: str) -> float:
    """Convert a change value to an improvement score for best-row selection.

    For percent/fraction/raw deltas, lower-is-better metrics improve when delta is
    negative, while higher-is-better metrics improve when delta is positive.
    For ratio, lower-is-better metrics improve when ratio < 1, while
    higher-is-better metrics improve when ratio > 1.
    """
    if not is_finite_number(delta):
        return float("nan")
    d = float(delta)
    if change_mode == "ratio":
        return (1.0 - d) if not metric.higher_is_better else (d - 1.0)
    return (-d) if not metric.higher_is_better else d


def format_change_value(v: float, precision: int, missing: str, mode: str, percent_sign: bool) -> str:
    if not is_finite_number(v):
        return missing
    suffix = "%" if mode == "percent" and percent_sign else ""
    return f"{float(v):.{precision}f}{suffix}"


def format_change_cell(
    v: float,
    precision: int,
    missing: str,
    mode: str,
    percent_sign: bool,
    is_best: bool,
    style: Literal["plain", "md", "tex"],
) -> str:
    s = format_change_value(v, precision, missing, mode, percent_sign)
    if s == missing or not is_best:
        return s
    if style == "md":
        return f"**{s}**"
    if style == "tex":
        return r"\textbf{" + s + "}"
    return s


def parse_models(spec: str | None) -> "OrderedDict[str, PriorGroupSpec]":
    if not spec:
        return DEFAULT_PRIOR_GROUPS.copy()

    out: "OrderedDict[str, PriorGroupSpec]" = OrderedDict()
    for item in split_csv_like(spec):
        if item not in DEFAULT_PRIOR_GROUPS:
            raise ValueError(
                f"Unknown model: {item!r}. Known models: {', '.join(DEFAULT_PRIOR_GROUPS.keys())}. "
                "For non-default directory names, use --group-json."
            )
        out[item] = DEFAULT_PRIOR_GROUPS[item]
    return out


def parse_group_json(path: str | None) -> "OrderedDict[str, PriorGroupSpec] | None":
    """Load custom model/prior directory mapping.

    JSON format:
      {
        "Hunyuan": {"rgb": "hunyuan", "priors": {"C": "hunyuan_csfm", "P": "hunyuan_psfm", "CP": "hunyuan_mvs"}},
        "DA3": {"rgb": "da3", "priors": {"CP": "da3_mvs"}}
      }
    """
    if not path:
        return None
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("--group-json must point to a JSON object")
    out: "OrderedDict[str, PriorGroupSpec]" = OrderedDict()
    for model, spec in obj.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Bad group spec for {model!r}: expected object")
        rgb = spec.get("rgb") or spec.get("baseline") or spec.get("rgb_subdir")
        priors = spec.get("priors") or spec.get("prior_subdirs")
        if not isinstance(rgb, str) or not rgb:
            raise ValueError(f"Bad group spec for {model!r}: missing non-empty 'rgb'")
        if not isinstance(priors, dict) or not priors:
            raise ValueError(f"Bad group spec for {model!r}: missing non-empty 'priors' object")
        prior_od: "OrderedDict[str, str]" = OrderedDict()
        # Keep a stable and paper-friendly order when possible.
        for p in ["C", "P", "CP"]:
            if p in priors:
                prior_od[p] = str(priors[p])
        for p, d in priors.items():
            if p not in prior_od:
                prior_od[str(p)] = str(d)
        out[str(model)] = PriorGroupSpec(str(model), str(rgb), prior_od)
    return out


def make_collection_methods(groups: "OrderedDict[str, PriorGroupSpec]") -> "OrderedDict[str, MethodSpec]":
    """Build unique MethodSpec labels for table32 collection functions."""
    methods: "OrderedDict[str, MethodSpec]" = OrderedDict()
    for model, group in groups.items():
        rgb_label = f"{model}__RGB"
        methods[rgb_label] = MethodSpec(rgb_label, group.rgb_subdir)
        for prior, subdir in group.prior_subdirs.items():
            label = f"{model}__{prior}"
            methods[label] = MethodSpec(label, subdir)
    return methods


def best_prior_cells(table_rows: list[dict], metrics: list[MetricSpec], change_mode: str) -> dict[tuple[int, str], bool]:
    """Return best prior effect per metric over all rows."""
    out: dict[tuple[int, str], bool] = {}
    for metric in metrics:
        scored: list[tuple[int, float]] = []
        for idx, row in enumerate(table_rows):
            d = row.get("deltas", {}).get(metric.alias, float("nan"))
            score = improvement_value(d, metric, change_mode)
            if is_finite_number(score):
                scored.append((idx, float(score)))
        if not scored:
            continue
        best_score = max(s for _, s in scored)
        eps = max(abs(best_score) * 1e-12, 1e-12)
        for idx, score in scored:
            if abs(score - best_score) <= eps:
                out[(idx, metric.alias)] = True
    return out


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------


def compute_table33_rows(
    aggregate_data: dict,
    groups: "OrderedDict[str, PriorGroupSpec]",
    metrics: list[MetricSpec],
    dataset_labels: list[str],
    include_dataset_breakdown: bool,
    change_mode: str,
) -> tuple[list[dict], list[dict]]:
    """Compute table rows and long rows.

    table_rows: one row per model/prior; metric columns contain deltas for the
    selected-dataset average.

    long_rows: one row per model/prior/metric/dataset plus Avg., including RGB
    and prior absolute values and their relative change.
    """
    table_rows: list[dict] = []
    long_rows: list[dict] = []

    for model, group in groups.items():
        rgb_label = f"{model}__RGB"
        for prior in group.prior_subdirs.keys():
            prior_label = f"{model}__{prior}"
            row = {
                "model": model,
                "prior": prior,
                "rgb_method": rgb_label,
                "prior_method": prior_label,
                "deltas": OrderedDict(),
                "rgb_values": OrderedDict(),
                "prior_values": OrderedDict(),
            }
            for metric in metrics:
                metric_rows = aggregate_data.get(metric.alias, {})
                rgb_obj = metric_rows.get(rgb_label, {})
                prior_obj = metric_rows.get(prior_label, {})

                rgb_avg = finite_or_nan(rgb_obj.get("avg", float("nan")))
                prior_avg = finite_or_nan(prior_obj.get("avg", float("nan")))
                delta_avg = relative_delta(prior_avg, rgb_avg, change_mode)
                row["rgb_values"][metric.alias] = rgb_avg
                row["prior_values"][metric.alias] = prior_avg
                row["deltas"][metric.alias] = delta_avg

                long_rows.append(
                    {
                        "model": model,
                        "prior": prior,
                        "metric_alias": metric.alias,
                        "metric_key": metric.key,
                        "metric_display": metric.display,
                        "dataset": "Avg.",
                        "rgb_method": rgb_label,
                        "prior_method": prior_label,
                        "rgb_value": rgb_avg,
                        "prior_value": prior_avg,
                        "change": delta_avg,
                        "change_mode": change_mode,
                        "n_views_rgb": rgb_obj.get("avg_n_views", ""),
                        "n_views_prior": prior_obj.get("avg_n_views", ""),
                    }
                )

                if include_dataset_breakdown:
                    rgb_values = rgb_obj.get("values", {})
                    prior_values = prior_obj.get("values", {})
                    rgb_n = rgb_obj.get("n_views", {})
                    prior_n = prior_obj.get("n_views", {})
                    for dataset in dataset_labels:
                        rv = finite_or_nan(rgb_values.get(dataset, float("nan")))
                        pv = finite_or_nan(prior_values.get(dataset, float("nan")))
                        dv = relative_delta(pv, rv, change_mode)
                        long_rows.append(
                            {
                                "model": model,
                                "prior": prior,
                                "metric_alias": metric.alias,
                                "metric_key": metric.key,
                                "metric_display": metric.display,
                                "dataset": dataset,
                                "rgb_method": rgb_label,
                                "prior_method": prior_label,
                                "rgb_value": rv,
                                "prior_value": pv,
                                "change": dv,
                                "change_mode": change_mode,
                                "n_views_rgb": rgb_n.get(dataset, ""),
                                "n_views_prior": prior_n.get(dataset, ""),
                            }
                        )

            table_rows.append(row)

    return table_rows, long_rows


# -----------------------------------------------------------------------------
# Writing outputs
# -----------------------------------------------------------------------------


def metric_column_name(metric: MetricSpec, change_mode: str) -> str:
    if change_mode == "percent":
        suffix = " Δ (%)"
    elif change_mode == "fraction":
        suffix = " Δ"
    elif change_mode == "raw":
        suffix = " Δ"
    elif change_mode == "ratio":
        suffix = " Ratio"
    else:
        suffix = " Δ"
    return f"{metric.display}{suffix}"


def write_table_csv(
    path: Path,
    table_rows: list[dict],
    metrics: list[MetricSpec],
    precision: int,
    missing: str,
    change_mode: str,
    percent_sign: bool,
) -> None:
    cols = ["Model", "Prior"] + [metric_column_name(m, change_mode) for m in metrics]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in table_rows:
            vals = [
                format_change_value(row["deltas"].get(m.alias, float("nan")), precision, missing, change_mode, percent_sign)
                for m in metrics
            ]
            writer.writerow([row["model"], row["prior"]] + vals)


def write_table_markdown(
    path: Path,
    table_rows: list[dict],
    metrics: list[MetricSpec],
    precision: int,
    missing: str,
    change_mode: str,
    percent_sign: bool,
    bold_best: bool,
    title_suffix: str,
) -> None:
    cols = ["Model", "Prior"] + [metric_column_name(m, change_mode) for m in metrics]
    best = best_prior_cells(table_rows, metrics, change_mode) if bold_best else {}
    lines: list[str] = []
    lines.append(f"# Table 3.3 ({title_suffix})")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---", "---"] + ["---:"] * len(metrics)) + " |")
    for idx, row in enumerate(table_rows):
        cells = [str(row["model"]), str(row["prior"])]
        for metric in metrics:
            cells.append(
                format_change_cell(
                    row["deltas"].get(metric.alias, float("nan")),
                    precision,
                    missing,
                    change_mode,
                    percent_sign,
                    is_best=best.get((idx, metric.alias), False),
                    style="md",
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if change_mode == "percent":
        lines.append("Values are `100 * (prior - RGB) / abs(RGB)`.")
    elif change_mode == "fraction":
        lines.append("Values are `(prior - RGB) / abs(RGB)`.")
    elif change_mode == "raw":
        lines.append("Values are `prior - RGB`.")
    elif change_mode == "ratio":
        lines.append("Values are `prior / RGB`.")
    lower_metrics = [m.display for m in metrics if not m.higher_is_better]
    higher_metrics = [m.display for m in metrics if m.higher_is_better]
    if lower_metrics:
        lines.append("For lower-is-better metrics, negative changes indicate improvement: " + ", ".join(lower_metrics) + ".")
    if higher_metrics:
        lines.append("For higher-is-better metrics, positive changes indicate improvement: " + ", ".join(higher_metrics) + ".")
    lines.append("DA3 includes only CP prior because no separate C/P prior runs are defined in the current benchmark setup.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_latex(
    path: Path,
    table_rows: list[dict],
    metrics: list[MetricSpec],
    precision: int,
    missing: str,
    change_mode: str,
    percent_sign: bool,
    bold_best: bool,
    title_suffix: str,
    label_suffix: str,
) -> None:
    cols = ["Model", "Prior"] + [metric_column_name(m, change_mode) for m in metrics]
    best = best_prior_cells(table_rows, metrics, change_mode) if bold_best else {}
    col_spec = "ll" + "r" * len(metrics)
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Effect of camera and pose priors in zero-shot UAV evaluation "
        + f"({latex_escape(title_suffix)}). Values indicate relative change compared with RGB-only input."
        + r"}"
    )
    lines.append(r"\label{tab:table33_prior_effect_" + safe_stem(label_suffix) + r"}")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(latex_escape(c) for c in cols) + r" \\")
    lines.append(r"\midrule")
    prev_model = None
    for idx, row in enumerate(table_rows):
        model = str(row["model"])
        model_cell = latex_escape(model) if model != prev_model else ""
        prev_model = model
        cells = [model_cell, latex_escape(str(row["prior"]))]
        for metric in metrics:
            cells.append(
                format_change_cell(
                    row["deltas"].get(metric.alias, float("nan")),
                    precision,
                    missing,
                    change_mode,
                    percent_sign,
                    is_best=best.get((idx, metric.alias), False),
                    style="tex",
                )
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_long_csv(path: Path, long_rows: list[dict]) -> None:
    fieldnames = [
        "model",
        "prior",
        "metric_alias",
        "metric_key",
        "metric_display",
        "dataset",
        "rgb_method",
        "prior_method",
        "rgb_value",
        "prior_value",
        "change",
        "change_mode",
        "n_views_rgb",
        "n_views_prior",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in long_rows:
            rr = dict(row)
            for k in ["rgb_value", "prior_value", "change"]:
                v = rr.get(k)
                rr[k] = "" if not is_finite_number(v) else f"{float(v):.12g}"
            writer.writerow(rr)


def write_absolute_debug_csv(
    path: Path,
    aggregate_data: dict,
    groups: "OrderedDict[str, PriorGroupSpec]",
    metrics: list[MetricSpec],
    dataset_labels: list[str],
) -> None:
    """Save the aggregated absolute values used before relative normalization."""
    fieldnames = ["model", "prior", "method_label", "metric_alias", "metric_key", "dataset", "value", "n_views"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, group in groups.items():
            items = [("RGB", f"{model}__RGB")] + [(p, f"{model}__{p}") for p in group.prior_subdirs.keys()]
            for prior, method_label in items:
                for metric in metrics:
                    robj = aggregate_data.get(metric.alias, {}).get(method_label, {})
                    for dataset in dataset_labels:
                        v = robj.get("values", {}).get(dataset, float("nan"))
                        writer.writerow(
                            {
                                "model": model,
                                "prior": prior,
                                "method_label": method_label,
                                "metric_alias": metric.alias,
                                "metric_key": metric.key,
                                "dataset": dataset,
                                "value": "" if not is_finite_number(v) else f"{float(v):.12g}",
                                "n_views": robj.get("n_views", {}).get(dataset, ""),
                            }
                        )
                    avg = robj.get("avg", float("nan"))
                    writer.writerow(
                        {
                            "model": model,
                            "prior": prior,
                            "method_label": method_label,
                            "metric_alias": metric.alias,
                            "metric_key": metric.key,
                            "dataset": "Avg.",
                            "value": "" if not is_finite_number(avg) else f"{float(avg):.12g}",
                            "n_views": robj.get("avg_n_views", ""),
                        }
                    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Table 3.3 prior-effect statistics from current dense_n_view benchmark JSON outputs."
    )
    parser.add_argument(
        "--benchmarking",
        type=str,
        default="experiments/mapanything/benchmarking",
        help="Root directory containing dense_*_view folders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/mapanything/benchmarking/tables/table33",
        help="Output directory for csv/md/tex tables.",
    )
    parser.add_argument(
        "--views",
        type=str,
        default=DEFAULT_VIEWS,
        help="Comma-separated view counts to aggregate, e.g. '8,16,24,32' or '24'. Use 'all' to scan dense_*_view dirs.",
    )
    parser.add_argument(
        "--view-stat",
        choices=["mean", "median", "std", "min", "max", "n"],
        default="mean",
        help="Statistic used to aggregate selected view counts before computing prior-vs-RGB change. Default: mean.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names from defaults: Hunyuan,MapAnything,Pi3X,DA3. DA3 has CP only.",
    )
    parser.add_argument(
        "--group-json",
        type=str,
        default=None,
        help="Optional JSON file defining custom RGB/prior subdirectories. Overrides --models when set.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated dataset labels or Label=alias1|alias2 specs. "
            "Default: UseGeo,Enrich-Aerial,UrbanScene3D,A3D-Real. Use 'auto' to include all discovered dataset keys."
        ),
    )
    parser.add_argument(
        "--auto-datasets-if-empty",
        action="store_true",
        default=True,
        help="If selected dataset specs match no JSON keys for a view, fall back to all discovered datasets for that view.",
    )
    parser.add_argument("--no-auto-datasets-if-empty", dest="auto_datasets_if_empty", action="store_false")
    parser.add_argument(
        "--metrics",
        type=str,
        default=DEFAULT_TABLE33_METRICS,
        help="Comma-separated metric aliases or exact current JSON keys. Default: chamfer,ray,pose_ate,depth_absrel.",
    )
    parser.add_argument(
        "--avg",
        choices=["selected", "json", "none"],
        default="selected",
        help="How to reduce datasets before computing Table 3.3. selected = mean over displayed datasets; json = JSON['Average']; none = NaN.",
    )
    parser.add_argument(
        "--change-mode",
        choices=["percent", "fraction", "raw", "ratio"],
        default="percent",
        help="How to express prior-vs-RGB change. Default: percent = 100*(prior-RGB)/abs(RGB).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=1,
        help="Decimal precision for relative changes. Default: 1, e.g. -12.3%%.",
    )
    parser.add_argument("--missing", type=str, default="--", help="String used for missing values.")
    parser.add_argument("--no-percent-sign", dest="percent_sign", action="store_false", help="Do not append %% in table cells.")
    parser.set_defaults(percent_sign=True)
    parser.add_argument("--bold-best", action="store_true", default=True, help="Bold largest relative improvement per metric in md/tex.")
    parser.add_argument("--no-bold-best", dest="bold_best", action="store_false")
    parser.add_argument(
        "--include-dataset-breakdown",
        action="store_true",
        default=True,
        help="Include per-dataset prior-vs-RGB rows in the long CSV.",
    )
    parser.add_argument("--no-dataset-breakdown", dest="include_dataset_breakdown", action="store_false")
    parser.add_argument("--out-prefix", type=str, default="table33", help="Output filename prefix.")
    args = parser.parse_args()

    root = Path(args.benchmarking)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        raise FileNotFoundError(f"Benchmarking root not found: {root}")

    groups = parse_group_json(args.group_json) or parse_models(args.models)
    methods = make_collection_methods(groups)
    datasets_arg = parse_datasets(args.datasets)
    metrics = parse_metrics(args.metrics)
    views = parse_views(args.views, root)
    if not views:
        raise RuntimeError(f"No view counts selected. Check --views and dense_*_view folders under {root}")

    metadata: dict = {
        "benchmarking_root": str(root),
        "output": str(out_dir),
        "views": views,
        "view_stat": args.view_stat,
        "avg": args.avg,
        "change_mode": args.change_mode,
        "groups": {
            model: {"rgb": g.rgb_subdir, "priors": dict(g.prior_subdirs)} for model, g in groups.items()
        },
        "metrics": {m.alias: {"display": m.display, "key": m.key, "higher_is_better": m.higher_is_better} for m in metrics},
        "warnings": [],
        "outputs": [],
    }

    per_view_data: dict[int, dict] = OrderedDict()
    per_view_dataset_specs: dict[int, list[DatasetSpec]] = OrderedDict()

    for view in views:
        data, dataset_specs, warnings = collect_one_view(
            root=root,
            view=view,
            methods=methods,
            datasets=datasets_arg,
            metrics=metrics,
            avg_mode=args.avg,
            auto_datasets_if_empty=args.auto_datasets_if_empty,
        )
        for w in warnings:
            print(f"[WARN] {w}")
        metadata["warnings"].extend(warnings)
        per_view_data[view] = data
        per_view_dataset_specs[view] = dataset_specs

    if datasets_arg == "auto":
        aggregate_dataset_specs = union_dataset_specs(per_view_dataset_specs.values())
    else:
        aggregate_dataset_specs = union_dataset_specs([datasets_arg] + list(per_view_dataset_specs.values()))
    dataset_labels = [d.label for d in aggregate_dataset_specs]

    aggregate_data = aggregate_views(
        per_view_data=per_view_data,
        methods=methods,
        dataset_specs=aggregate_dataset_specs,
        metrics=metrics,
        view_stat=args.view_stat,
        avg_mode=args.avg,
    )

    table_rows, long_rows = compute_table33_rows(
        aggregate_data=aggregate_data,
        groups=groups,
        metrics=metrics,
        dataset_labels=dataset_labels,
        include_dataset_breakdown=args.include_dataset_breakdown,
        change_mode=args.change_mode,
    )

    view_suffix = "views_" + "_".join(str(v) for v in views)
    title_suffix = f"{args.view_stat} over views " + "/".join(str(v) for v in views)
    stem = f"{args.out_prefix}_prior_effect_{view_suffix}"

    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    tex_path = out_dir / f"{stem}.tex"
    long_csv = out_dir / f"{args.out_prefix}_prior_effect_long.csv"
    abs_csv = out_dir / f"{args.out_prefix}_absolute_values_used.csv"
    meta_path = out_dir / f"{args.out_prefix}_metadata.json"

    write_table_csv(
        csv_path,
        table_rows,
        metrics,
        args.precision,
        args.missing,
        args.change_mode,
        args.percent_sign,
    )
    write_table_markdown(
        md_path,
        table_rows,
        metrics,
        args.precision,
        args.missing,
        args.change_mode,
        args.percent_sign,
        args.bold_best,
        title_suffix,
    )
    write_table_latex(
        tex_path,
        table_rows,
        metrics,
        args.precision,
        args.missing,
        args.change_mode,
        args.percent_sign,
        args.bold_best,
        title_suffix,
        view_suffix,
    )
    write_long_csv(long_csv, long_rows)
    write_absolute_debug_csv(abs_csv, aggregate_data, groups, metrics, dataset_labels)

    metadata["outputs"].extend(str(p) for p in [csv_path, md_path, tex_path, long_csv, abs_csv])
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata["outputs"].append(str(meta_path))

    print(f"[OK] Wrote Table 3.3 CSV: {csv_path}")
    print(f"[OK] Wrote Table 3.3 Markdown: {md_path}")
    print(f"[OK] Wrote Table 3.3 LaTeX: {tex_path}")
    print(f"[OK] Wrote long-format CSV: {long_csv}")
    print(f"[OK] Wrote absolute-value debug CSV: {abs_csv}")
    print(f"[OK] Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()

"""

python scripts/viz/table33_prior_effect_stats.py \
  --benchmarking experiments/mapanything/benchmarking \
  --views 8,16,24,32 \
  --metrics chamfer,ray,pose_ate,depth_absrel \
  --output experiments/mapanything/benchmarking/tables/table33

"""