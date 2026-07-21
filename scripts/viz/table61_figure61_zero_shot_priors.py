#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Table 6.1 and plot Figure 6.1 for zero-shot prior sensitivity.

Table 6.1:
  Success rate of zero-shot priors across UAV benchmarks.

Figure 6.1:
  Zero-shot prior sensitivity of pretrained models on UAV benchmarks.

The script reads per_dataset_results.json under:
  <benchmarking>/dense_{view}_view/<method_subdir>/per_dataset_results.json

For each pretrained model, the RGB-only setting is used as the baseline, and
C / P / CP prior variants are compared with:

  delta = (Error_prior - Error_RGB) / Error_RGB * 100

All metrics are lower-is-better, so:
  - delta < 0 means the prior improves over RGB-only
  - delta > 0 means the prior degrades performance

Default prior groups follow the same naming convention as the Chapter 3/6
prior-aware zero-shot benchmark scripts:
  Hunyuan:      hunyuan, hunyuan_csfm, hunyuan_psfm, hunyuan_mvs
  MapAnything:  mapa, mapa_csfm, mapa_psfm, mapa_mvs
  Pi3X:         pi3x, pi3x_csfm, pi3x_psfm, pi3x_mvs
  DA3:          da3, da3_mvs

Example:
  python scripts/viz/table61_figure61_zero_shot_priors.py \
    --benchmarking experiments/mapanything/benchmarking \
    --views 8,16,24,32 \
    --output experiments/mapanything/benchmarking/figures_tables/table61_figure61
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


# Reuse parser / collection utilities from the existing Chapter 5 table script.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from table52_53_54_domain_adaptation import (  # type: ignore
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
    parse_views,
    safe_stem,
    tex_escape,
)


# -----------------------------------------------------------------------------
# Specs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorGroupSpec:
    model: str
    rgb_subdirs: tuple[str, ...]
    prior_subdirs: "OrderedDict[str, tuple[str, ...]]"


DEFAULT_PRIOR_GROUPS: "OrderedDict[str, PriorGroupSpec]" = OrderedDict(
    [
        (
            "Hunyuan",
            PriorGroupSpec(
                model="Hunyuan",
                rgb_subdirs=("hunyuan",),
                prior_subdirs=OrderedDict(
                    [
                        ("C", ("hunyuan_csfm",)),
                        ("P", ("hunyuan_psfm",)),
                        ("CP", ("hunyuan_mvs",)),
                    ]
                ),
            ),
        ),
        (
            "MapAnything",
            PriorGroupSpec(
                model="MapAnything",
                rgb_subdirs=("mapa", "mapa_24v", "uav_mapa", "mapanything"),
                prior_subdirs=OrderedDict(
                    [
                        ("C", ("mapa_csfm",)),
                        ("P", ("mapa_psfm",)),
                        ("CP", ("mapa_mvs",)),
                    ]
                ),
            ),
        ),
        (
            "Pi3X",
            PriorGroupSpec(
                model="Pi3X",
                rgb_subdirs=("pi3x",),
                prior_subdirs=OrderedDict(
                    [
                        ("C", ("pi3x_csfm",)),
                        ("P", ("pi3x_psfm",)),
                        ("CP", ("pi3x_mvs",)),
                    ]
                ),
            ),
        ),
        (
            "DA3",
            PriorGroupSpec(
                model="DA3",
                rgb_subdirs=("da3",),
                prior_subdirs=OrderedDict(
                    [
                        ("CP", ("da3_mvs",)),
                    ]
                ),
            ),
        ),
    ]
)

DEFAULT_DATASET_SPECS = [
    "UseGeo=UseGeoWAI|UseGeo|usegeo",
    "Enrich=ENRICHWAI|Enrich-Aerial|EnrichAerial|Enrich|enrich_aerial",
    "Urban=UrbanScene3DWAI|UrbanScene3D|US3D|us3d|Urban",
    "A3D-Real=A3DRealWAI|A3D-Real|A3DReal|A3D_Real|a3d_real",
]

METRIC_ORDER = ["reldepth", "ray", "chamfer", "pose_ate"]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def split_csv_like(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def flat_label(model: str, variant: str) -> str:
    return f"{model}::{variant}"


def row_label(model: str, prior: str) -> str:
    return f"{model}-{prior}"


def parse_models(spec: str | None) -> list[str]:
    if spec is None or not str(spec).strip() or str(spec).strip().lower() in {"all", "auto"}:
        return list(DEFAULT_PRIOR_GROUPS.keys())
    out = split_csv_like(spec)
    for name in out:
        if name not in DEFAULT_PRIOR_GROUPS:
            raise ValueError(f"Unknown model {name!r}. Available: {list(DEFAULT_PRIOR_GROUPS.keys())}")
    return out


def parse_prior_group_json(path: str | None) -> "OrderedDict[str, PriorGroupSpec] | None":
    """
    Optional JSON override format:
    {
      "MapAnything": {
        "rgb": "mapa|mapa_24v",
        "priors": {"C": "mapa_csfm", "P": "mapa_psfm", "CP": "mapa_mvs"}
      }
    }
    """
    if not path:
        return None
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("--prior-group-json must point to a JSON object")

    groups: "OrderedDict[str, PriorGroupSpec]" = OrderedDict()
    for model, spec in obj.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Bad prior group spec for {model!r}: expected object")
        rgb = spec.get("rgb") or spec.get("baseline") or spec.get("rgb_subdirs")
        priors = spec.get("priors") or spec.get("prior_subdirs")
        if isinstance(rgb, str):
            rgb_subdirs = tuple(x.strip() for x in rgb.split("|") if x.strip())
        elif isinstance(rgb, list):
            rgb_subdirs = tuple(str(x).strip() for x in rgb if str(x).strip())
        else:
            rgb_subdirs = ()
        if not rgb_subdirs:
            raise ValueError(f"Bad prior group spec for {model!r}: missing non-empty rgb")
        if not isinstance(priors, dict) or not priors:
            raise ValueError(f"Bad prior group spec for {model!r}: missing priors object")
        prior_od: "OrderedDict[str, tuple[str, ...]]" = OrderedDict()
        for prior_name in ["C", "P", "CP"]:
            if prior_name not in priors:
                continue
            val = priors[prior_name]
            if isinstance(val, str):
                subdirs = tuple(x.strip() for x in val.split("|") if x.strip())
            elif isinstance(val, list):
                subdirs = tuple(str(x).strip() for x in val if str(x).strip())
            else:
                subdirs = ()
            if subdirs:
                prior_od[prior_name] = subdirs
        for prior_name, val in priors.items():
            if prior_name in prior_od:
                continue
            if isinstance(val, str):
                subdirs = tuple(x.strip() for x in val.split("|") if x.strip())
            elif isinstance(val, list):
                subdirs = tuple(str(x).strip() for x in val if str(x).strip())
            else:
                subdirs = ()
            if subdirs:
                prior_od[str(prior_name)] = subdirs
        groups[str(model)] = PriorGroupSpec(model=str(model), rgb_subdirs=rgb_subdirs, prior_subdirs=prior_od)
    return groups


def select_prior_groups(args: argparse.Namespace) -> "OrderedDict[str, PriorGroupSpec]":
    groups = parse_prior_group_json(args.prior_group_json) or DEFAULT_PRIOR_GROUPS
    selected_models = parse_models(args.models)
    selected: "OrderedDict[str, PriorGroupSpec]" = OrderedDict()
    for model in selected_models:
        if model not in groups:
            raise ValueError(f"Model {model!r} is not found in prior groups. Available: {list(groups.keys())}")
        selected[model] = groups[model]
    return selected


def flatten_prior_settings(prior_groups: "OrderedDict[str, PriorGroupSpec]") -> "OrderedDict[str, SettingSpec]":
    settings: "OrderedDict[str, SettingSpec]" = OrderedDict()
    for model, group in prior_groups.items():
        settings[flat_label(model, "RGB-only")] = SettingSpec(label=flat_label(model, "RGB-only"), subdirs=group.rgb_subdirs)
        for prior, subdirs in group.prior_subdirs.items():
            settings[flat_label(model, prior)] = SettingSpec(label=flat_label(model, prior), subdirs=subdirs)
    return settings


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
                "ray",
                MetricSpec(
                    alias="ray",
                    key=args.ray_key,
                    display="Ray Error",
                    precision=int(args.precision_ray),
                ),
            ),
            (
                "chamfer",
                MetricSpec(
                    alias="chamfer",
                    key=args.chamfer_key,
                    display="Chamfer-L1",
                    precision=int(args.precision_chamfer),
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
        table_id="6.1-figure6.1",
        caption="Zero-shot prior sensitivity metrics.",
        latex_label="tab:zero_shot_prior_sensitivity_collection",
        stem="table61_figure61_zero_shot_priors_collection",
        metrics=metrics,
        layout="dataset_metric",
    )


def relative_delta_percent(rgb_value: float, prior_value: float) -> float:
    if not is_finite_number(rgb_value) or not is_finite_number(prior_value):
        return float("nan")
    rgb = float(rgb_value)
    prior = float(prior_value)
    if math.isclose(rgb, 0.0, abs_tol=1e-12):
        return float("nan")
    return (prior - rgb) / rgb * 100.0


def fmt_float(value: float, precision: int, missing: str = "") -> str:
    if not is_finite_number(value):
        return missing
    return f"{float(value):.{precision}f}"


def fmt_success(count: int, total: int, mode: str, precision: int) -> str:
    if total <= 0:
        return ""
    pct = count / total * 100.0
    if mode == "count":
        return f"{count}/{total}"
    if mode == "both":
        return f"{count}/{total} ({pct:.{precision}f}%)"
    return f"{pct:.{precision}f}%"


# -----------------------------------------------------------------------------
# Delta and success computation
# -----------------------------------------------------------------------------


def compute_delta_details(
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
    prior_groups: "OrderedDict[str, PriorGroupSpec]",
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
) -> tuple[dict, list[dict]]:
    """
    Returns:
      deltas[model][prior][dataset][metric_alias] = delta percent
      details: long-form raw RGB / prior values and deltas
    """
    deltas: dict = OrderedDict()
    details: list[dict] = []

    for model, group in prior_groups.items():
        deltas[model] = OrderedDict()
        rgb_label = flat_label(model, "RGB-only")
        for prior in group.prior_subdirs.keys():
            prior_label = flat_label(model, prior)
            deltas[model][prior] = OrderedDict()
            for dataset_label in datasets.keys():
                deltas[model][prior][dataset_label] = OrderedDict()
                for metric_alias, metric in metrics.items():
                    rgb_v = values.get(rgb_label, {}).get(dataset_label, {}).get(metric_alias, float("nan"))
                    prior_v = values.get(prior_label, {}).get(dataset_label, {}).get(metric_alias, float("nan"))
                    delta = relative_delta_percent(rgb_v, prior_v)
                    deltas[model][prior][dataset_label][metric_alias] = delta
                    details.append(
                        {
                            "model": model,
                            "prior": prior,
                            "row_label": row_label(model, prior),
                            "rgb_setting": rgb_label,
                            "prior_setting": prior_label,
                            "dataset": dataset_label,
                            "metric_alias": metric_alias,
                            "metric_display": metric.display,
                            "metric_key": metric.key,
                            "rgb_value": rgb_v,
                            "prior_value": prior_v,
                            "delta_percent": delta,
                            "improved": bool(is_finite_number(delta) and float(delta) < 0.0),
                        }
                    )
    return deltas, details


def compute_heatmap_rows(
    deltas: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    prior_groups: "OrderedDict[str, PriorGroupSpec]",
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
) -> tuple[list[str], np.ndarray, list[dict]]:
    rows: list[str] = []
    matrix_rows: list[list[float]] = []
    summary_rows: list[dict] = []

    for model, group in prior_groups.items():
        for prior in group.prior_subdirs.keys():
            rlabel = row_label(model, prior)
            rows.append(rlabel)
            metric_means: list[float] = []
            for metric_alias in metrics.keys():
                vals = [
                    deltas.get(model, {}).get(prior, {}).get(dataset_label, {}).get(metric_alias, float("nan"))
                    for dataset_label in datasets.keys()
                ]
                metric_means.append(mean_finite(vals))
            avg_delta = mean_finite(metric_means)
            matrix_rows.append(metric_means + [avg_delta])
            summary_rows.append(
                {
                    "model": model,
                    "prior": prior,
                    "row_label": rlabel,
                    **{f"{metric_alias}_delta_mean_percent": metric_means[i] for i, metric_alias in enumerate(metrics.keys())},
                    "avg_delta_percent": avg_delta,
                }
            )

    if not matrix_rows:
        return rows, np.zeros((0, len(metrics) + 1), dtype=float), summary_rows
    return rows, np.array(matrix_rows, dtype=float), summary_rows


def compute_success_table(
    deltas: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    prior_groups: "OrderedDict[str, PriorGroupSpec]",
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
    improve_eps: float,
) -> list[dict]:
    rows: list[dict] = []
    n_datasets = len(datasets)
    for model, group in prior_groups.items():
        for prior in group.prior_subdirs.keys():
            row: dict[str, Any] = {
                "model": model,
                "prior": prior,
                "row_label": row_label(model, prior),
            }
            overall_count = 0
            overall_total = 0
            for metric_alias in metrics.keys():
                count = 0
                total = 0
                for dataset_label in datasets.keys():
                    delta = deltas.get(model, {}).get(prior, {}).get(dataset_label, {}).get(metric_alias, float("nan"))
                    if not is_finite_number(delta):
                        continue
                    total += 1
                    overall_total += 1
                    # Lower-is-better metrics: prior improves if delta < 0.
                    if float(delta) < -float(improve_eps):
                        count += 1
                        overall_count += 1
                row[f"{metric_alias}_success_count"] = count
                row[f"{metric_alias}_success_total"] = total if total > 0 else n_datasets
                row[f"{metric_alias}_success_rate"] = (count / total) if total > 0 else float("nan")
            row["overall_success_count"] = overall_count
            row["overall_success_total"] = overall_total
            row["overall_success_rate"] = (overall_count / overall_total) if overall_total > 0 else float("nan")
            rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# Writers
# -----------------------------------------------------------------------------


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rr = dict(row)
            for k, v in list(rr.items()):
                if isinstance(v, float) and not math.isfinite(v):
                    rr[k] = ""
            writer.writerow(rr)


def build_table61_markdown(
    success_rows: list[dict],
    metrics: "OrderedDict[str, MetricSpec]",
    success_format: str,
    precision: int,
    include_caption: bool,
) -> str:
    headers = ["Model", "Prior"] + [f"{m.display} Improve ↑" for m in metrics.values()] + ["Overall Success ↑"]
    aligns = [":---", ":---"] + ["---:" for _ in headers[2:]]
    lines: list[str] = []
    if include_caption:
        lines.append("**Table 6.1. Success rate of zero-shot priors across UAV benchmarks.**")
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(aligns) + " |")
    for row in success_rows:
        out = [str(row["model"]), str(row["prior"])]
        for metric_alias in metrics.keys():
            out.append(
                fmt_success(
                    int(row.get(f"{metric_alias}_success_count", 0)),
                    int(row.get(f"{metric_alias}_success_total", 0)),
                    mode=success_format,
                    precision=precision,
                )
            )
        out.append(
            fmt_success(
                int(row.get("overall_success_count", 0)),
                int(row.get("overall_success_total", 0)),
                mode=success_format,
                precision=precision,
            )
        )
        lines.append("| " + " | ".join(out) + " |")
    return "\n".join(lines)


def build_table61_latex(
    success_rows: list[dict],
    metrics: "OrderedDict[str, MetricSpec]",
    success_format: str,
    precision: int,
) -> str:
    headers = ["Model", "Prior"] + [f"{m.display} Improve $\\uparrow$" for m in metrics.values()] + ["Overall Success $\\uparrow$"]
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Success rate of zero-shot priors across UAV benchmarks.}")
    lines.append("\\label{tab:zero_shot_prior_success_rate}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{ll" + "r" * (len(headers) - 2) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(tex_escape(h) if "$" not in h else h for h in headers) + " " + r"\\")
    lines.append("\\midrule")
    for row in success_rows:
        out = [tex_escape(str(row["model"])), tex_escape(str(row["prior"]))]
        for metric_alias in metrics.keys():
            out.append(
                fmt_success(
                    int(row.get(f"{metric_alias}_success_count", 0)),
                    int(row.get(f"{metric_alias}_success_total", 0)),
                    mode=success_format,
                    precision=precision,
                ).replace("%", "\\%")
            )
        out.append(
            fmt_success(
                int(row.get("overall_success_count", 0)),
                int(row.get("overall_success_total", 0)),
                mode=success_format,
                precision=precision,
            ).replace("%", "\\%")
        )
        lines.append(" & ".join(out) + " " + r"\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Figure 6.1 heatmap
# -----------------------------------------------------------------------------


def metric_column_labels(metrics: "OrderedDict[str, MetricSpec]") -> list[str]:
    labels = []
    for metric_alias, metric in metrics.items():
        if metric_alias == "reldepth":
            labels.append("RelDepth Δ")
        elif metric_alias == "ray":
            labels.append("Ray Error Δ")
        elif metric_alias == "chamfer":
            labels.append("Chamfer-L1 Δ")
        elif metric_alias == "pose_ate":
            labels.append("Pose ATE Δ")
        else:
            labels.append(f"{metric.display} Δ")
    labels.append("Avg. Δ")
    return labels


def plot_figure61_heatmap(
    row_labels: list[str],
    matrix: np.ndarray,
    col_labels: list[str],
    title: str | None,
    vlim: float | None,
    annotate_precision: int,
    figsize_scale: float,
) -> plt.Figure:
    nrows, ncols = matrix.shape if matrix.size else (0, len(col_labels))
    fig_w = figsize_scale * max(7.2, 1.35 * max(1, ncols) + 2.2)
    fig_h = figsize_scale * max(4.8, 0.42 * max(1, nrows) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    finite_vals = matrix[np.isfinite(matrix)] if matrix.size else np.array([])
    if finite_vals.size == 0:
        vmax = 1.0
    elif vlim is not None and vlim > 0:
        vmax = float(vlim)
    else:
        vmax = float(np.nanpercentile(np.abs(finite_vals), 95))
        vmax = max(5.0, vmax)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Metric relative change averaged over UAV benchmarks")
    ax.set_ylabel("Pretrained model + prior")

    if title:
        ax.set_title(title, fontsize=12, pad=14)

    # Draw subtle cell boundaries.
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(nrows):
        for j in range(ncols):
            val = matrix[i, j]
            if not np.isfinite(val):
                text = ""
            else:
                text = f"{val:+.{annotate_precision}f}%"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label("Δ = (Error_prior - Error_RGB) / Error_RGB (%)\nnegative = improvement")

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build Table 6.1 and Figure 6.1 for zero-shot prior sensitivity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmarking", type=Path, default=Path("experiments/mapanything/benchmarking"))
    p.add_argument("--output", type=Path, default=None, help="Output directory. Default: <benchmarking>/figures_tables/table61_figure61")
    p.add_argument("--views", type=str, default="8,16,24,32", help="Comma-separated views or 'all'.")
    p.add_argument("--models", type=str, default="all", help="Models to include: all or comma-separated subset, e.g. Hunyuan,MapAnything,Pi3X,DA3.")
    p.add_argument("--prior-group-json", type=str, default=None, help="Optional JSON overriding model->RGB/prior subdir mapping.")
    p.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASET_SPECS,
        help="Datasets as Label=alias1|alias2. Defaults to UseGeo/Enrich/Urban/A3D-Real.",
    )
    p.add_argument("--json-name", type=str, default="per_dataset_results.json")
    p.add_argument("--depth-key", type=str, default="abs_depth_rel_scale_aligned")
    p.add_argument("--ray-key", type=str, default="ray_dir_mean_angle_deg")
    p.add_argument("--chamfer-key", type=str, default="abs_fused_pc_chamfer_l1")
    p.add_argument("--pose-ate-key", type=str, default="abs_pose_ate")
    p.add_argument("--agg", choices=["mean", "median", "first", "second"], default="mean", help="How to reduce list-valued JSON entries.")
    p.add_argument("--improve-eps", type=float, default=0.0, help="Require delta < -eps to count as improvement.")
    p.add_argument("--success-format", choices=["percent", "count", "both"], default="percent")
    p.add_argument("--success-precision", type=int, default=1)
    p.add_argument("--precision-depth", type=int, default=3)
    p.add_argument("--precision-ray", type=int, default=2)
    p.add_argument("--precision-chamfer", type=int, default=3)
    p.add_argument("--precision-pose-ate", type=int, default=3)
    p.add_argument("--heatmap-vlim", type=float, default=None, help="Symmetric heatmap range. Example: 50 means [-50, 50].")
    p.add_argument("--annotate-precision", type=int, default=1)
    p.add_argument("--figsize-scale", type=float, default=1.0)
    p.add_argument("--title", type=str, default="Figure 6.1. Zero-shot prior sensitivity of pretrained models on UAV benchmarks", help="Figure title. Use empty string to disable.")
    p.add_argument("--table-stem", type=str, default="table61_zero_shot_prior_success_rate")
    p.add_argument("--figure-stem", type=str, default="figure61_zero_shot_prior_sensitivity")
    p.add_argument("--no-caption-markdown", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--save-svg", action="store_true")
    p.add_argument("--strict", action="store_true", help="Raise on missing files/datasets/metrics instead of warning.")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    root = Path(args.benchmarking)
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmarking root not found: {root}")

    output_dir = Path(args.output) if args.output is not None else root / "figures_tables" / "table61_figure61"
    output_dir.mkdir(parents=True, exist_ok=True)

    views = parse_views(args.views, root)
    if not views:
        raise ValueError(f"No views selected/found under {root}")

    prior_groups = select_prior_groups(args)
    flat_settings = flatten_prior_settings(prior_groups)
    datasets = parse_dataset_specs(args.datasets)
    metrics = build_metric_specs(args)
    table_for_collection = make_collection_table(metrics)
    json_name = ensure_json_filename(args.json_name)

    loaded_jsons, load_warnings = load_setting_jsons(
        root=root,
        views=views,
        settings=flat_settings,
        json_name=json_name,
        strict=bool(args.strict),
    )

    values, collect_details, collect_warnings = collect_table_values(
        loaded_jsons=loaded_jsons,
        settings=flat_settings,
        datasets=datasets,
        table=table_for_collection,
        agg=args.agg,
        strict=bool(args.strict),
    )

    deltas, delta_details = compute_delta_details(
        values=values,
        prior_groups=prior_groups,
        datasets=datasets,
        metrics=metrics,
    )
    row_labels, heatmap_matrix, heatmap_rows = compute_heatmap_rows(
        deltas=deltas,
        prior_groups=prior_groups,
        datasets=datasets,
        metrics=metrics,
    )
    success_rows = compute_success_table(
        deltas=deltas,
        prior_groups=prior_groups,
        datasets=datasets,
        metrics=metrics,
        improve_eps=float(args.improve_eps),
    )

    # Table 6.1 outputs.
    table_stem = safe_stem(args.table_stem)
    md_path = output_dir / f"{table_stem}.md"
    tex_path = output_dir / f"{table_stem}.tex"
    table_csv_path = output_dir / f"{table_stem}.csv"
    table_details_path = output_dir / f"{table_stem}_details.csv"
    table_metadata_path = output_dir / f"{table_stem}_metadata.json"

    md = build_table61_markdown(
        success_rows=success_rows,
        metrics=metrics,
        success_format=args.success_format,
        precision=int(args.success_precision),
        include_caption=not bool(args.no_caption_markdown),
    )
    latex = build_table61_latex(
        success_rows=success_rows,
        metrics=metrics,
        success_format=args.success_format,
        precision=int(args.success_precision),
    )
    md_path.write_text(md + "\n", encoding="utf-8")
    tex_path.write_text(latex + "\n", encoding="utf-8")
    write_rows_csv(table_csv_path, success_rows)

    table_detail_rows: list[dict] = []
    for row in success_rows:
        table_detail_rows.append({"kind": "success_summary", **row})
    for row in delta_details:
        table_detail_rows.append({"kind": "delta_detail", **row})
    write_rows_csv(table_details_path, table_detail_rows)

    # Figure 6.1 outputs.
    figure_stem = safe_stem(args.figure_stem)
    fig_png_path = output_dir / f"{figure_stem}.png"
    fig_pdf_path = output_dir / f"{figure_stem}.pdf"
    fig_svg_path = output_dir / f"{figure_stem}.svg"
    heatmap_csv_path = output_dir / f"{figure_stem}_heatmap_values.csv"
    figure_details_path = output_dir / f"{figure_stem}_details.csv"
    figure_metadata_path = output_dir / f"{figure_stem}_metadata.json"

    fig_title = args.title if str(args.title).strip() else None
    fig = plot_figure61_heatmap(
        row_labels=row_labels,
        matrix=heatmap_matrix,
        col_labels=metric_column_labels(metrics),
        title=fig_title,
        vlim=args.heatmap_vlim,
        annotate_precision=int(args.annotate_precision),
        figsize_scale=float(args.figsize_scale),
    )
    fig.savefig(fig_png_path, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(fig_pdf_path, bbox_inches="tight")
    if args.save_svg:
        fig.savefig(fig_svg_path, bbox_inches="tight")
    plt.close(fig)

    write_rows_csv(heatmap_csv_path, heatmap_rows)
    figure_detail_rows: list[dict] = []
    for row in heatmap_rows:
        figure_detail_rows.append({"kind": "heatmap_summary", **row})
    for row in delta_details:
        figure_detail_rows.append({"kind": "delta_detail", **row})
    write_rows_csv(figure_details_path, figure_detail_rows)

    all_warnings = dedupe_warnings(load_warnings + collect_warnings)

    common_metadata = {
        "benchmarking_root": str(root),
        "output_dir": str(output_dir),
        "views": views,
        "prior_groups": {
            model: {
                "rgb": list(group.rgb_subdirs),
                "priors": {p: list(s) for p, s in group.prior_subdirs.items()},
            }
            for model, group in prior_groups.items()
        },
        "datasets": {k: list(v.aliases) for k, v in datasets.items()},
        "metrics": {k: {"key": v.key, "display": v.display} for k, v in metrics.items()},
        "json_name": json_name,
        "agg": args.agg,
        "delta_formula": "(Error_prior - Error_RGB) / Error_RGB * 100",
        "success_rule": f"delta < -{float(args.improve_eps)}",
        "warnings": all_warnings,
    }

    table_metadata = {
        "table_id": "6.1",
        "caption": "Success rate of zero-shot priors across UAV benchmarks.",
        **common_metadata,
        "outputs": {
            "markdown": str(md_path),
            "latex": str(tex_path),
            "csv": str(table_csv_path),
            "details_csv": str(table_details_path),
        },
    }
    table_metadata_path.write_text(json.dumps(table_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_metadata = {
        "figure_id": "6.1",
        "caption": "Zero-shot prior sensitivity of pretrained models on UAV benchmarks.",
        **common_metadata,
        "outputs": {
            "figure_png": str(fig_png_path),
            "figure_pdf": str(fig_pdf_path),
            "figure_svg": str(fig_svg_path) if args.save_svg else None,
            "heatmap_values_csv": str(heatmap_csv_path),
            "details_csv": str(figure_details_path),
        },
    }
    figure_metadata_path.write_text(json.dumps(figure_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    combined_metadata_path = output_dir / "table61_figure61_zero_shot_priors_metadata.json"
    combined_metadata = {
        **common_metadata,
        "outputs": {
            "table61": {
                "markdown": str(md_path),
                "latex": str(tex_path),
                "csv": str(table_csv_path),
                "details_csv": str(table_details_path),
                "metadata": str(table_metadata_path),
            },
            "figure61": {
                "figure_png": str(fig_png_path),
                "figure_pdf": str(fig_pdf_path),
                "figure_svg": str(fig_svg_path) if args.save_svg else None,
                "heatmap_values_csv": str(heatmap_csv_path),
                "details_csv": str(figure_details_path),
                "metadata": str(figure_metadata_path),
            },
        },
    }
    combined_metadata_path.write_text(json.dumps(combined_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(md)
    print(f"\nWrote Table 6.1 Markdown: {md_path}")
    print(f"Wrote Table 6.1 LaTeX:    {tex_path}")
    print(f"Wrote Table 6.1 CSV:      {table_csv_path}")
    print(f"Wrote Table 6.1 details:  {table_details_path}")
    print(f"Wrote Figure 6.1 PNG:     {fig_png_path}")
    print(f"Wrote Figure 6.1 PDF:     {fig_pdf_path}")
    if args.save_svg:
        print(f"Wrote Figure 6.1 SVG:     {fig_svg_path}")
    print(f"Wrote Figure 6.1 values:  {heatmap_csv_path}")
    print(f"Wrote combined metadata:  {combined_metadata_path}")

    if all_warnings and not args.quiet:
        print("\nWarnings:")
        for w in all_warnings[:80]:
            print(f"  - {w}")
        if len(all_warnings) > 80:
            print(f"  ... {len(all_warnings) - 80} more warnings. See {combined_metadata_path}")


if __name__ == "__main__":
    main()

"""
python scripts/viz/table61_figure61_zero_shot_priors.py \
  --benchmarking experiments/mapanything/benchmarking \
  --views 8,16,24,32 \
  --output experiments/mapanything/benchmarking/figures_tables/table61_figure61
"""