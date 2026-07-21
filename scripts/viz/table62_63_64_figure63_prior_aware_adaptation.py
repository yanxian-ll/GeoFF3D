#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Table 6.2, Table 6.3, Table 6.4, and plot Figure 6.3.

This script evaluates camera / pose priors after UAV-domain adaptation, using the
same per-dataset benchmark JSON layout as previous Chapter 5/6 scripts:

  <benchmarking>/dense_{view}_view/<method_subdir>/per_dataset_results.json

It reuses helper functions from scripts/viz/table52_53_54_domain_adaptation.py.

Table 6.2:
  Prior-aware evaluation after UAV-domain adaptation.
  Rows: RGB / C / P / CP for MA-FT-A3D.
  Columns: average RelDepth, Ray Error, Chamfer-L1, Pose ATE over selected datasets,
           plus Avg. Rank.

Table 6.3:
  Dataset-wise prior-aware results after UAV-domain adaptation.
  Rows: RGB / C / P / CP.
  Columns: UseGeo / Enrich / Urban / A3D-Real / A3D-FA / Avg.
  Default cells are per-dataset Avg. Rank across all selected metrics.

Figure 6.3:
  Relative improvement of priors over RGB-only after UAV-domain adaptation.
  Improvement(%) = (Error_RGB - Error_prior) / Error_RGB * 100.

Table 6.4:
  Prior reliability before and after UAV-domain adaptation.
  Rows: MapAnything Zero-shot / MA-FT-A3D.
  Columns: C/P/CP success rates and overall success.
  Success = prior input outperforms RGB-only.

Default directory assumptions:
  Zero-shot MapAnything:
    RGB: mapa | mapa_24v | uav_mapa | mapanything
    C:   mapa_csfm
    P:   mapa_psfm
    CP:  mapa_mvs

  MA-FT-A3D:
    RGB: mapa-ft-a3dfull | mapa_ft_a3dfull | mapa-ft-a3d | mapa_ft_a3d | mapa-ft-a3dsyn | mapa_ft_a3dsyn
    C:   mapa-ft-a3dfull_csfm | mapa-ft-a3dfull-csfm | ...
    P:   mapa-ft-a3dfull_psfm | mapa-ft-a3dfull-psfm | ...
    CP:  mapa-ft-a3dfull_mvs  | mapa-ft-a3dfull-mvs  | ...

Example:
  python scripts/viz/table62_63_64_figure63_prior_aware_adaptation.py \
    --benchmarking experiments/mapanything/benchmarking \
    --views 8,16,24,32 \
    --output experiments/mapanything/benchmarking/figures_tables/table62_63_64_figure63
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
class TrainingPriorSpec:
    model: str
    training: str
    input_subdirs: "OrderedDict[str, tuple[str, ...]]"


INPUT_ORDER = ["RGB", "C", "P", "CP"]
PRIOR_ORDER = ["C", "P", "CP"]
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"

DEFAULT_DATASET_SPECS = [
    "UseGeo=UseGeoWAI|UseGeo|usegeo",
    "Enrich=ENRICHWAI|Enrich-Aerial|EnrichAerial|Enrich|enrich_aerial",
    "Urban=UrbanScene3DWAI|UrbanScene3D|US3D|us3d|Urban",
    "A3D-Real=A3DRealWAI|A3D-Real|A3DReal|A3D_Real|a3d_real",
    "A3D-FA=A3DSynLargeFAWAI|A3D-FA|A3DFA|a3dfa|a3d_fa",
]

DEFAULT_ZERO_SHOT_SPEC = TrainingPriorSpec(
    model="MapAnything",
    training="Zero-shot",
    input_subdirs=OrderedDict(
        [
            ("RGB", ("mapa", "mapa_24v", "uav_mapa", "mapanything")),
            ("C", ("mapa_csfm",)),
            ("P", ("mapa_psfm",)),
            ("CP", ("mapa_mvs",)),
        ]
    ),
)

DEFAULT_ADAPTED_SPEC = TrainingPriorSpec(
    model="MapAnything",
    training="MA-FT-A3D",
    input_subdirs=OrderedDict(
        [
            (
                "RGB",
                (
                    "uav_mapa",
                ),
            ),
            (
                "C",
                (
                    "uav_mapa_csfm",
                ),
            ),
            (
                "P",
                (
                    "uav_mapa_psfm",
                ),
            ),
            (
                "CP",
                (
                    "uav_mapa_mvs",
                ),
            ),
        ]
    ),
)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def split_csv_like(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def setting_label(training: str, input_name: str) -> str:
    return f"{training}::{input_name}"


def fmt_float(value: float, precision: int, missing: str = "") -> str:
    if not is_finite_number(value):
        return missing
    return f"{float(value):.{precision}f}"


def fmt_percent(value01: float, precision: int, missing: str = "") -> str:
    if not is_finite_number(value01):
        return missing
    return f"{float(value01) * 100.0:.{precision}f}%"


def fmt_success(count: int, total: int, mode: str, precision: int) -> str:
    if total <= 0:
        return ""
    pct = count / total
    if mode == "count":
        return f"{count}/{total}"
    if mode == "both":
        return f"{count}/{total} ({pct * 100.0:.{precision}f}%)"
    return f"{pct * 100.0:.{precision}f}%"


def average_tie_ranks(label_to_value: Mapping[str, float]) -> dict[str, float]:
    finite = [(label, float(v)) for label, v in label_to_value.items() if is_finite_number(v)]
    finite.sort(key=lambda x: x[1])  # lower is better
    ranks: dict[str, float] = {label: float("nan") for label in label_to_value.keys()}
    i = 0
    while i < len(finite):
        j = i + 1
        while j < len(finite) and math.isclose(finite[j][1], finite[i][1], rel_tol=1e-12, abs_tol=1e-12):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[finite[k][0]] = avg_rank
        i = j
    return ranks


def relative_improvement_percent(rgb_value: float, prior_value: float) -> float:
    """Positive means prior improves lower-is-better error over RGB."""
    if not is_finite_number(rgb_value) or not is_finite_number(prior_value):
        return float("nan")
    rgb = float(rgb_value)
    prior = float(prior_value)
    if math.isclose(rgb, 0.0, abs_tol=1e-12):
        return float("nan")
    return (rgb - prior) / rgb * 100.0


def metric_precision(metric_alias: str, metric: MetricSpec) -> int:
    return int(metric.precision)


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
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


def maybe_bold_md(s: str, bold: bool) -> str:
    return f"**{s}**" if bold and s != "" else s


def maybe_bold_tex(s: str, bold: bool, missing: str = "") -> str:
    return f"\\textbf{{{s}}}" if bold and s != missing else s


# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------


def parse_subdir_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(x.strip() for x in value.split("|") if x.strip())
    if isinstance(value, list):
        return tuple(str(x).strip() for x in value if str(x).strip())
    return ()


def parse_training_prior_json(path: str | None) -> tuple[TrainingPriorSpec, TrainingPriorSpec] | None:
    """
    Optional JSON format:
    {
      "zero_shot": {
        "model": "MapAnything",
        "training": "Zero-shot",
        "inputs": {"RGB": "mapa|mapa_24v", "C": "mapa_csfm", "P": "mapa_psfm", "CP": "mapa_mvs"}
      },
      "adapted": {
        "model": "MapAnything",
        "training": "MA-FT-A3D",
        "inputs": {"RGB": "mapa-ft-a3dfull", "C": "mapa-ft-a3dfull_csfm", ...}
      }
    }
    """
    if not path:
        return None
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("--prior-settings-json must point to a JSON object")

    def parse_one(key: str, default: TrainingPriorSpec) -> TrainingPriorSpec:
        spec = obj.get(key, None)
        if spec is None:
            return default
        if not isinstance(spec, dict):
            raise ValueError(f"Bad {key!r} spec: expected object")
        model = str(spec.get("model", default.model))
        training = str(spec.get("training", default.training))
        inputs = spec.get("inputs") or spec.get("input_subdirs")
        if not isinstance(inputs, dict):
            raise ValueError(f"Bad {key!r} spec: missing inputs object")
        od: "OrderedDict[str, tuple[str, ...]]" = OrderedDict()
        for input_name in INPUT_ORDER:
            if input_name not in inputs:
                continue
            subdirs = parse_subdir_list(inputs[input_name])
            if subdirs:
                od[input_name] = subdirs
        for input_name, value in inputs.items():
            if input_name in od:
                continue
            subdirs = parse_subdir_list(value)
            if subdirs:
                od[str(input_name)] = subdirs
        if "RGB" not in od:
            raise ValueError(f"Bad {key!r} spec: inputs must include RGB")
        return TrainingPriorSpec(model=model, training=training, input_subdirs=od)

    return parse_one("zero_shot", DEFAULT_ZERO_SHOT_SPEC), parse_one("adapted", DEFAULT_ADAPTED_SPEC)


def select_training_specs(args: argparse.Namespace) -> tuple[TrainingPriorSpec, TrainingPriorSpec]:
    parsed = parse_training_prior_json(args.prior_settings_json)
    if parsed is not None:
        return parsed
    return DEFAULT_ZERO_SHOT_SPEC, DEFAULT_ADAPTED_SPEC


def flatten_training_settings(specs: Iterable[TrainingPriorSpec]) -> "OrderedDict[str, SettingSpec]":
    settings: "OrderedDict[str, SettingSpec]" = OrderedDict()
    for spec in specs:
        for input_name, subdirs in spec.input_subdirs.items():
            label = setting_label(spec.training, input_name)
            settings[label] = SettingSpec(label=label, subdirs=subdirs)
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
        table_id="6.2-6.4-figure6.3",
        caption="Prior-aware adaptation metrics.",
        latex_label="tab:prior_aware_adaptation_collection",
        stem="table62_63_64_figure63_collection",
        metrics=metrics,
        layout="dataset_metric",
    )


# -----------------------------------------------------------------------------
# Core computations
# -----------------------------------------------------------------------------


def compute_input_metric_averages(
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
    training: str,
    inputs: list[str],
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = OrderedDict()
    for input_name in inputs:
        slabel = setting_label(training, input_name)
        out[input_name] = OrderedDict()
        for metric_alias in metrics.keys():
            vals = [
                values.get(slabel, {}).get(dataset_label, {}).get(metric_alias, float("nan"))
                for dataset_label in datasets.keys()
            ]
            out[input_name][metric_alias] = mean_finite(vals)
    return out


def compute_input_avg_ranks(
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
    training: str,
    inputs: list[str],
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
) -> dict[str, float]:
    rank_lists: dict[str, list[float]] = {x: [] for x in inputs}
    for dataset_label in datasets.keys():
        for metric_alias in metrics.keys():
            label_to_value = {
                input_name: values.get(setting_label(training, input_name), {}).get(dataset_label, {}).get(metric_alias, float("nan"))
                for input_name in inputs
            }
            ranks = average_tie_ranks(label_to_value)
            for input_name in inputs:
                r = ranks.get(input_name, float("nan"))
                if is_finite_number(r):
                    rank_lists[input_name].append(float(r))
    return {input_name: mean_finite(rs) for input_name, rs in rank_lists.items()}


def compute_datasetwise_scores(
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
    training: str,
    inputs: list[str],
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
    mode: str,
    table63_metric: str,
) -> tuple[dict[str, dict[str, float]], dict[str, float], str]:
    """
    Returns dataset_scores[input][dataset] and avg_scores[input].
    mode=rank: average rank over metrics for each dataset.
    mode=metric: show one selected metric value per dataset.
    mode=raw: average raw metric values per dataset. Not recommended because units differ.
    """
    out: dict[str, dict[str, float]] = OrderedDict((input_name, OrderedDict()) for input_name in inputs)

    if mode == "rank":
        for dataset_label in datasets.keys():
            rank_lists: dict[str, list[float]] = {x: [] for x in inputs}
            for metric_alias in metrics.keys():
                label_to_value = {
                    input_name: values.get(setting_label(training, input_name), {}).get(dataset_label, {}).get(metric_alias, float("nan"))
                    for input_name in inputs
                }
                ranks = average_tie_ranks(label_to_value)
                for input_name in inputs:
                    r = ranks.get(input_name, float("nan"))
                    if is_finite_number(r):
                        rank_lists[input_name].append(float(r))
            for input_name in inputs:
                out[input_name][dataset_label] = mean_finite(rank_lists[input_name])
        label = "Avg. Rank"

    elif mode == "metric":
        if table63_metric not in metrics:
            raise ValueError(f"--table63-metric must be one of {list(metrics.keys())}, got {table63_metric!r}")
        for input_name in inputs:
            slabel = setting_label(training, input_name)
            for dataset_label in datasets.keys():
                out[input_name][dataset_label] = values.get(slabel, {}).get(dataset_label, {}).get(table63_metric, float("nan"))
        label = metrics[table63_metric].display

    elif mode == "raw":
        for input_name in inputs:
            slabel = setting_label(training, input_name)
            for dataset_label in datasets.keys():
                vals = [
                    values.get(slabel, {}).get(dataset_label, {}).get(metric_alias, float("nan"))
                    for metric_alias in metrics.keys()
                ]
                out[input_name][dataset_label] = mean_finite(vals)
        label = "Raw Mean"

    else:
        raise ValueError(f"Unknown Table 6.3 mode: {mode}")

    avg_scores = {input_name: mean_finite(out[input_name].values()) for input_name in inputs}
    return out, avg_scores, label


def compute_improvement_values(
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
    training: str,
    priors: list[str],
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
) -> tuple[dict, list[dict], list[dict]]:
    """
    Returns:
      improvements[prior][metric_alias] = average improvement percent over datasets
      details long-form rows
      summary rows for CSV
    """
    improvements: dict = OrderedDict()
    details: list[dict] = []
    summary_rows: list[dict] = []
    rgb_label = setting_label(training, "RGB")

    for prior in priors:
        prior_label = setting_label(training, prior)
        improvements[prior] = OrderedDict()
        summary_row: dict[str, Any] = {"prior": prior}
        for metric_alias, metric in metrics.items():
            vals: list[float] = []
            for dataset_label in datasets.keys():
                rgb_v = values.get(rgb_label, {}).get(dataset_label, {}).get(metric_alias, float("nan"))
                prior_v = values.get(prior_label, {}).get(dataset_label, {}).get(metric_alias, float("nan"))
                imp = relative_improvement_percent(rgb_v, prior_v)
                if is_finite_number(imp):
                    vals.append(float(imp))
                details.append(
                    {
                        "training": training,
                        "prior": prior,
                        "dataset": dataset_label,
                        "metric_alias": metric_alias,
                        "metric_display": metric.display,
                        "metric_key": metric.key,
                        "rgb_value": rgb_v,
                        "prior_value": prior_v,
                        "relative_improvement_percent": imp,
                    }
                )
            avg_imp = mean_finite(vals)
            improvements[prior][metric_alias] = avg_imp
            summary_row[f"{metric_alias}_improvement_percent"] = avg_imp
        summary_row["avg_improvement_percent"] = mean_finite(improvements[prior].values())
        summary_rows.append(summary_row)
    return improvements, details, summary_rows


def compute_training_success_rows(
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
    specs: list[TrainingPriorSpec],
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
    improve_eps: float,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    details: list[dict] = []
    for spec in specs:
        rgb_label = setting_label(spec.training, "RGB")
        row: dict[str, Any] = {
            "model": spec.model,
            "training": spec.training,
        }
        overall_count = 0
        overall_total = 0
        for prior in PRIOR_ORDER:
            if prior not in spec.input_subdirs:
                row[f"{prior}_success_count"] = 0
                row[f"{prior}_success_total"] = 0
                row[f"{prior}_success_rate"] = float("nan")
                continue
            prior_label = setting_label(spec.training, prior)
            count = 0
            total = 0
            for dataset_label in datasets.keys():
                for metric_alias, metric in metrics.items():
                    rgb_v = values.get(rgb_label, {}).get(dataset_label, {}).get(metric_alias, float("nan"))
                    prior_v = values.get(prior_label, {}).get(dataset_label, {}).get(metric_alias, float("nan"))
                    imp = relative_improvement_percent(rgb_v, prior_v)
                    if not is_finite_number(imp):
                        continue
                    total += 1
                    overall_total += 1
                    improved = float(imp) > float(improve_eps)
                    if improved:
                        count += 1
                        overall_count += 1
                    details.append(
                        {
                            "model": spec.model,
                            "training": spec.training,
                            "prior": prior,
                            "dataset": dataset_label,
                            "metric_alias": metric_alias,
                            "metric_display": metric.display,
                            "metric_key": metric.key,
                            "rgb_value": rgb_v,
                            "prior_value": prior_v,
                            "relative_improvement_percent": imp,
                            "improved": improved,
                        }
                    )
            row[f"{prior}_success_count"] = count
            row[f"{prior}_success_total"] = total
            row[f"{prior}_success_rate"] = (count / total) if total > 0 else float("nan")
        row["overall_success_count"] = overall_count
        row["overall_success_total"] = overall_total
        row["overall_success_rate"] = (overall_count / overall_total) if overall_total > 0 else float("nan")
        rows.append(row)
    return rows, details


# -----------------------------------------------------------------------------
# Markdown / LaTeX builders
# -----------------------------------------------------------------------------


def best_inputs_for_metrics(metric_values: Mapping[str, Mapping[str, float]], avg_ranks: Mapping[str, float], inputs: list[str], metrics: "OrderedDict[str, MetricSpec]") -> tuple[dict[str, set[str]], set[str]]:
    best_metric: dict[str, set[str]] = {}
    for metric_alias in metrics.keys():
        vals = {input_name: metric_values[input_name].get(metric_alias, float("nan")) for input_name in inputs}
        finite = [v for v in vals.values() if is_finite_number(v)]
        if not finite:
            best_metric[metric_alias] = set()
            continue
        best_v = min(finite)
        best_metric[metric_alias] = {
            input_name for input_name, v in vals.items()
            if is_finite_number(v) and math.isclose(float(v), best_v, rel_tol=1e-12, abs_tol=1e-12)
        }

    finite_avg = [v for v in avg_ranks.values() if is_finite_number(v)]
    if not finite_avg:
        best_avg = set()
    else:
        best_v = min(finite_avg)
        best_avg = {
            input_name for input_name, v in avg_ranks.items()
            if is_finite_number(v) and math.isclose(float(v), best_v, rel_tol=1e-12, abs_tol=1e-12)
        }
    return best_metric, best_avg


def build_table62_markdown(
    training: str,
    inputs: list[str],
    metric_values: Mapping[str, Mapping[str, float]],
    avg_ranks: Mapping[str, float],
    metrics: "OrderedDict[str, MetricSpec]",
    precision_avg: int,
    missing: str,
    bold_best: bool,
    include_caption: bool,
) -> str:
    best_metric, best_avg = best_inputs_for_metrics(metric_values, avg_ranks, inputs, metrics)
    headers = ["Setting", "Input"] + [f"{m.display} ↓" for m in metrics.values()] + ["Avg. Rank ↓"]
    aligns = [":---", ":---"] + ["---:" for _ in headers[2:]]
    lines: list[str] = []
    if include_caption:
        lines.append("**Table 6.2. Prior-aware evaluation after UAV-domain adaptation.**")
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(aligns) + " |")
    for input_name in inputs:
        row = [training, input_name]
        for metric_alias, metric in metrics.items():
            cell = fmt_float(metric_values[input_name].get(metric_alias, float("nan")), metric_precision(metric_alias, metric), missing)
            cell = maybe_bold_md(cell, bold_best and input_name in best_metric[metric_alias])
            row.append(cell)
        avg_cell = fmt_float(avg_ranks.get(input_name, float("nan")), precision_avg, missing)
        avg_cell = maybe_bold_md(avg_cell, bold_best and input_name in best_avg)
        row.append(avg_cell)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_table62_latex(
    training: str,
    inputs: list[str],
    metric_values: Mapping[str, Mapping[str, float]],
    avg_ranks: Mapping[str, float],
    metrics: "OrderedDict[str, MetricSpec]",
    precision_avg: int,
    missing: str,
    bold_best: bool,
) -> str:
    best_metric, best_avg = best_inputs_for_metrics(metric_values, avg_ranks, inputs, metrics)
    headers = ["Setting", "Input"] + [f"{m.display} $\\downarrow$" for m in metrics.values()] + ["Avg. Rank $\\downarrow$"]
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Prior-aware evaluation after UAV-domain adaptation.}")
    lines.append("\\label{tab:prior_aware_after_uav_adaptation}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{ll" + "r" * (len(headers) - 2) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(tex_escape(h) if "$" not in h else h for h in headers) + " " + r"\\")
    lines.append("\\midrule")
    for input_name in inputs:
        row = [tex_escape(training), tex_escape(input_name)]
        for metric_alias, metric in metrics.items():
            cell = fmt_float(metric_values[input_name].get(metric_alias, float("nan")), metric_precision(metric_alias, metric), missing)
            cell = maybe_bold_tex(cell, bold_best and input_name in best_metric[metric_alias], missing)
            row.append(cell)
        avg_cell = fmt_float(avg_ranks.get(input_name, float("nan")), precision_avg, missing)
        avg_cell = maybe_bold_tex(avg_cell, bold_best and input_name in best_avg, missing)
        row.append(avg_cell)
        lines.append(" & ".join(row) + " " + r"\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_table63_markdown(
    inputs: list[str],
    dataset_scores: Mapping[str, Mapping[str, float]],
    avg_scores: Mapping[str, float],
    datasets: "OrderedDict[str, DatasetSpec]",
    score_label: str,
    precision: int,
    missing: str,
    bold_best: bool,
    include_caption: bool,
) -> str:
    headers = ["Input"] + list(datasets.keys()) + ["Avg."]
    aligns = [":---"] + ["---:" for _ in headers[1:]]
    # lower is better for rank / raw / metric values.
    best_by_dataset: dict[str, set[str]] = {}
    for dataset_label in datasets.keys():
        vals = {inp: dataset_scores[inp].get(dataset_label, float("nan")) for inp in inputs}
        finite = [v for v in vals.values() if is_finite_number(v)]
        best_by_dataset[dataset_label] = set()
        if finite:
            best_v = min(finite)
            best_by_dataset[dataset_label] = {inp for inp, v in vals.items() if is_finite_number(v) and math.isclose(float(v), best_v, rel_tol=1e-12, abs_tol=1e-12)}
    finite_avg = [v for v in avg_scores.values() if is_finite_number(v)]
    best_avg = set()
    if finite_avg:
        best_v = min(finite_avg)
        best_avg = {inp for inp, v in avg_scores.items() if is_finite_number(v) and math.isclose(float(v), best_v, rel_tol=1e-12, abs_tol=1e-12)}

    lines: list[str] = []
    if include_caption:
        lines.append(f"**Table 6.3. Dataset-wise prior-aware results after UAV-domain adaptation. ({score_label})**")
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(aligns) + " |")
    for input_name in inputs:
        row = [input_name]
        for dataset_label in datasets.keys():
            cell = fmt_float(dataset_scores[input_name].get(dataset_label, float("nan")), precision, missing)
            cell = maybe_bold_md(cell, bold_best and input_name in best_by_dataset[dataset_label])
            row.append(cell)
        avg_cell = fmt_float(avg_scores.get(input_name, float("nan")), precision, missing)
        avg_cell = maybe_bold_md(avg_cell, bold_best and input_name in best_avg)
        row.append(avg_cell)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_table63_latex(
    inputs: list[str],
    dataset_scores: Mapping[str, Mapping[str, float]],
    avg_scores: Mapping[str, float],
    datasets: "OrderedDict[str, DatasetSpec]",
    score_label: str,
    precision: int,
    missing: str,
    bold_best: bool,
) -> str:
    headers = ["Input"] + list(datasets.keys()) + ["Avg."]
    best_by_dataset: dict[str, set[str]] = {}
    for dataset_label in datasets.keys():
        vals = {inp: dataset_scores[inp].get(dataset_label, float("nan")) for inp in inputs}
        finite = [v for v in vals.values() if is_finite_number(v)]
        best_by_dataset[dataset_label] = set()
        if finite:
            best_v = min(finite)
            best_by_dataset[dataset_label] = {inp for inp, v in vals.items() if is_finite_number(v) and math.isclose(float(v), best_v, rel_tol=1e-12, abs_tol=1e-12)}
    finite_avg = [v for v in avg_scores.values() if is_finite_number(v)]
    best_avg = set()
    if finite_avg:
        best_v = min(finite_avg)
        best_avg = {inp for inp, v in avg_scores.items() if is_finite_number(v) and math.isclose(float(v), best_v, rel_tol=1e-12, abs_tol=1e-12)}

    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Dataset-wise prior-aware results after UAV-domain adaptation ({tex_escape(score_label)}).}}")
    lines.append("\\label{tab:dataset_wise_prior_aware_after_uav_adaptation}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(tex_escape(h) for h in headers) + " " + r"\\")
    lines.append("\\midrule")
    for input_name in inputs:
        row = [tex_escape(input_name)]
        for dataset_label in datasets.keys():
            cell = fmt_float(dataset_scores[input_name].get(dataset_label, float("nan")), precision, missing)
            cell = maybe_bold_tex(cell, bold_best and input_name in best_by_dataset[dataset_label], missing)
            row.append(cell)
        avg_cell = fmt_float(avg_scores.get(input_name, float("nan")), precision, missing)
        avg_cell = maybe_bold_tex(avg_cell, bold_best and input_name in best_avg, missing)
        row.append(avg_cell)
        lines.append(" & ".join(row) + " " + r"\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_table64_markdown(
    rows: list[dict],
    success_format: str,
    precision: int,
    include_caption: bool,
) -> str:
    headers = ["Model", "Training", "C Success ↑", "P Success ↑", "CP Success ↑", "Overall Success ↑"]
    aligns = [":---", ":---"] + ["---:" for _ in headers[2:]]
    lines: list[str] = []
    if include_caption:
        lines.append("**Table 6.4. Prior reliability before and after UAV-domain adaptation.**")
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(aligns) + " |")
    for row in rows:
        out = [str(row.get("model", "")), str(row.get("training", ""))]
        for prior in PRIOR_ORDER:
            out.append(fmt_success(int(row.get(f"{prior}_success_count", 0)), int(row.get(f"{prior}_success_total", 0)), success_format, precision))
        out.append(fmt_success(int(row.get("overall_success_count", 0)), int(row.get("overall_success_total", 0)), success_format, precision))
        lines.append("| " + " | ".join(out) + " |")
    return "\n".join(lines)


def build_table64_latex(rows: list[dict], success_format: str, precision: int) -> str:
    headers = ["Model", "Training", "C Success $\\uparrow$", "P Success $\\uparrow$", "CP Success $\\uparrow$", "Overall Success $\\uparrow$"]
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Prior reliability before and after UAV-domain adaptation.}")
    lines.append("\\label{tab:prior_reliability_before_after_uav_adaptation}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("\\toprule")
    lines.append(" & ".join(tex_escape(h) if "$" not in h else h for h in headers) + " " + r"\\")
    lines.append("\\midrule")
    for row in rows:
        out = [tex_escape(str(row.get("model", ""))), tex_escape(str(row.get("training", "")))]
        for prior in PRIOR_ORDER:
            out.append(fmt_success(int(row.get(f"{prior}_success_count", 0)), int(row.get(f"{prior}_success_total", 0)), success_format, precision).replace("%", "\\%"))
        out.append(fmt_success(int(row.get("overall_success_count", 0)), int(row.get("overall_success_total", 0)), success_format, precision).replace("%", "\\%"))
        lines.append(" & ".join(out) + " " + r"\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Figure 6.3
# -----------------------------------------------------------------------------


def metric_panel_title(metric_alias: str, metric: MetricSpec, panel_idx: int) -> str:
    names = {
        "reldepth": "RelDepth improvement",
        "ray": "Ray Error improvement",
        "chamfer": "Chamfer-L1 improvement",
        "pose_ate": "Pose ATE improvement",
    }
    letter = PANEL_LETTERS[panel_idx] if panel_idx < len(PANEL_LETTERS) else "?"
    return f"({letter}) {names.get(metric_alias, metric.display + ' improvement')}"


def plot_figure63(
    improvements: Mapping[str, Mapping[str, float]],
    priors: list[str],
    metrics: "OrderedDict[str, MetricSpec]",
    title: str | None,
    ylabel: str,
    ncols: int,
    figsize_scale: float,
    annotate: bool,
) -> plt.Figure:
    metric_items = list(metrics.items())
    n_metrics = len(metric_items)
    ncols = max(1, min(int(ncols), n_metrics))
    nrows = int(math.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_scale * 4.5 * ncols, figsize_scale * 3.5 * nrows), squeeze=False)
    axes_flat = list(axes.flat)
    x = np.arange(len(priors))

    y_min, y_max = float("inf"), float("-inf")
    for panel_idx, (ax, (metric_alias, metric)) in enumerate(zip(axes_flat, metric_items)):
        ys = [improvements.get(prior, {}).get(metric_alias, float("nan")) for prior in priors]
        ys_plot = [float(v) if is_finite_number(v) else float("nan") for v in ys]
        bars = ax.bar(x, ys_plot, width=0.65)
        finite = [v for v in ys_plot if is_finite_number(v)]
        if finite:
            y_min = min(y_min, min(finite))
            y_max = max(y_max, max(finite))
        if annotate:
            for rect, y in zip(bars, ys_plot):
                if not is_finite_number(y):
                    continue
                va = "bottom" if float(y) >= 0 else "top"
                offset = 1.0 if float(y) >= 0 else -1.0
                ax.text(rect.get_x() + rect.get_width() / 2, float(y) + offset, f"{float(y):.1f}", ha="center", va=va, fontsize=8)
        ax.set_title(metric_panel_title(metric_alias, metric, panel_idx), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(priors)
        ax.set_ylabel(ylabel)
        ax.axhline(0.0, linewidth=1.0)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

    for ax in axes_flat[n_metrics:]:
        ax.axis("off")

    if math.isfinite(y_min) and math.isfinite(y_max):
        if math.isclose(y_min, y_max, rel_tol=1e-12, abs_tol=1e-12):
            y_min -= 1.0
            y_max += 1.0
        span = y_max - y_min
        lower = y_min - max(2.0, span * 0.15)
        upper = y_max + max(2.0, span * 0.18)
        for ax in axes_flat[:n_metrics]:
            ax.set_ylim(lower, upper)

    if title:
        fig.suptitle(title, y=1.04, fontsize=13)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Summary CSV writers
# -----------------------------------------------------------------------------


def write_table62_csv(path: Path, inputs: list[str], metric_values: Mapping[str, Mapping[str, float]], avg_ranks: Mapping[str, float], metrics: "OrderedDict[str, MetricSpec]", training: str) -> None:
    rows: list[dict] = []
    for input_name in inputs:
        row = {"setting": training, "input": input_name}
        for metric_alias in metrics.keys():
            row[metric_alias] = metric_values[input_name].get(metric_alias, float("nan"))
        row["avg_rank"] = avg_ranks.get(input_name, float("nan"))
        rows.append(row)
    write_rows_csv(path, rows)


def write_table63_csv(path: Path, inputs: list[str], dataset_scores: Mapping[str, Mapping[str, float]], avg_scores: Mapping[str, float], datasets: "OrderedDict[str, DatasetSpec]", score_label: str) -> None:
    rows: list[dict] = []
    for input_name in inputs:
        row = {"input": input_name, "score_type": score_label}
        for dataset_label in datasets.keys():
            row[safe_stem(dataset_label)] = dataset_scores[input_name].get(dataset_label, float("nan"))
        row["avg"] = avg_scores.get(input_name, float("nan"))
        rows.append(row)
    write_rows_csv(path, rows)


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build Table 6.2, Table 6.3, Table 6.4, and plot Figure 6.3 for prior-aware UAV-domain adaptation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmarking", type=Path, default=Path("experiments/mapanything/benchmarking"))
    p.add_argument("--output", type=Path, default=None, help="Output directory. Default: <benchmarking>/figures_tables/table62_63_64_figure63")
    p.add_argument("--views", type=str, default="8,16,24,32", help="Comma-separated views or all.")
    p.add_argument("--datasets", nargs="*", default=DEFAULT_DATASET_SPECS, help="Datasets as Label=alias1|alias2.")
    p.add_argument("--prior-settings-json", type=str, default=None, help="Optional JSON overriding zero-shot/adapted RGB/C/P/CP subdir mappings.")
    p.add_argument("--json-name", type=str, default="per_dataset_results.json")
    p.add_argument("--depth-key", type=str, default="abs_depth_rel_scale_aligned")
    p.add_argument("--chamfer-key", type=str, default="abs_fused_pc_chamfer_l1")
    p.add_argument("--ray-key", type=str, default="ray_dir_mean_angle_deg")
    p.add_argument("--pose-ate-key", type=str, default="abs_pose_ate")
    p.add_argument("--agg", choices=["mean", "median", "first", "second"], default="mean", help="How to reduce list-valued JSON entries.")
    p.add_argument("--table63-mode", choices=["rank", "metric", "raw"], default="rank", help="Table 6.3 cell meaning. rank=dataset-wise average rank across metrics; metric=selected metric value; raw=mean raw metrics.")
    p.add_argument("--table63-metric", type=str, default="chamfer", help="Metric alias used when --table63-mode metric.")
    p.add_argument("--improve-eps", type=float, default=0.0, help="Minimum improvement percentage required to count as success in Table 6.4.")
    p.add_argument("--success-format", choices=["percent", "count", "both"], default="percent")
    p.add_argument("--precision-depth", type=int, default=3)
    p.add_argument("--precision-chamfer", type=int, default=3)
    p.add_argument("--precision-ray", type=int, default=2)
    p.add_argument("--precision-pose-ate", type=int, default=3)
    p.add_argument("--precision-rank", type=int, default=2)
    p.add_argument("--precision-success", type=int, default=1)
    p.add_argument("--missing", type=str, default="")
    p.add_argument("--table62-stem", type=str, default="table62_prior_aware_after_uav_adaptation")
    p.add_argument("--table63-stem", type=str, default="table63_dataset_wise_prior_aware_after_uav_adaptation")
    p.add_argument("--table64-stem", type=str, default="table64_prior_reliability_before_after_uav_adaptation")
    p.add_argument("--figure63-stem", type=str, default="figure63_prior_improvement_after_uav_adaptation")
    p.add_argument("--figure-title", type=str, default="Figure 6.3. Relative improvement of priors over RGB-only after UAV-domain adaptation", help="Use empty string to disable.")
    p.add_argument("--figure-ylabel", type=str, default="Relative Improvement (%)")
    p.add_argument("--figure-ncols", type=int, default=2)
    p.add_argument("--figsize-scale", type=float, default=1.0)
    p.add_argument("--annotate", action="store_true", help="Annotate Figure 6.3 bars.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--save-svg", action="store_true")
    p.add_argument("--no-caption-markdown", action="store_true")
    p.add_argument("--no-bold-best", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    root = Path(args.benchmarking)
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmarking root not found: {root}")

    output_dir = Path(args.output) if args.output is not None else root / "figures_tables" / "table62_63_64_figure63"
    output_dir.mkdir(parents=True, exist_ok=True)

    views = parse_views(args.views, root)
    if not views:
        raise ValueError(f"No views selected/found under {root}")

    zero_spec, adapted_spec = select_training_specs(args)
    settings = flatten_training_settings([zero_spec, adapted_spec])
    datasets = parse_dataset_specs(args.datasets)
    metrics = build_metric_specs(args)
    collection_table = make_collection_table(metrics)
    json_name = ensure_json_filename(args.json_name)

    loaded_jsons, load_warnings = load_setting_jsons(
        root=root,
        views=views,
        settings=settings,
        json_name=json_name,
        strict=bool(args.strict),
    )
    values, collection_details, collection_warnings = collect_table_values(
        loaded_jsons=loaded_jsons,
        settings=settings,
        datasets=datasets,
        table=collection_table,
        agg=args.agg,
        strict=bool(args.strict),
    )

    adapted_inputs = [inp for inp in INPUT_ORDER if inp in adapted_spec.input_subdirs]
    adapted_priors = [inp for inp in PRIOR_ORDER if inp in adapted_spec.input_subdirs]

    # Table 6.2
    table62_values = compute_input_metric_averages(values, adapted_spec.training, adapted_inputs, datasets, metrics)
    table62_avg_ranks = compute_input_avg_ranks(values, adapted_spec.training, adapted_inputs, datasets, metrics)
    table62_md = build_table62_markdown(
        training=adapted_spec.training,
        inputs=adapted_inputs,
        metric_values=table62_values,
        avg_ranks=table62_avg_ranks,
        metrics=metrics,
        precision_avg=int(args.precision_rank),
        missing=args.missing,
        bold_best=not bool(args.no_bold_best),
        include_caption=not bool(args.no_caption_markdown),
    )
    table62_tex = build_table62_latex(
        training=adapted_spec.training,
        inputs=adapted_inputs,
        metric_values=table62_values,
        avg_ranks=table62_avg_ranks,
        metrics=metrics,
        precision_avg=int(args.precision_rank),
        missing=args.missing,
        bold_best=not bool(args.no_bold_best),
    )

    # Table 6.3
    table63_scores, table63_avg, table63_score_label = compute_datasetwise_scores(
        values=values,
        training=adapted_spec.training,
        inputs=adapted_inputs,
        datasets=datasets,
        metrics=metrics,
        mode=args.table63_mode,
        table63_metric=args.table63_metric,
    )
    table63_precision = int(args.precision_rank) if args.table63_mode == "rank" else 3
    if args.table63_mode == "metric" and args.table63_metric in metrics:
        table63_precision = int(metrics[args.table63_metric].precision)
    table63_md = build_table63_markdown(
        inputs=adapted_inputs,
        dataset_scores=table63_scores,
        avg_scores=table63_avg,
        datasets=datasets,
        score_label=table63_score_label,
        precision=table63_precision,
        missing=args.missing,
        bold_best=not bool(args.no_bold_best),
        include_caption=not bool(args.no_caption_markdown),
    )
    table63_tex = build_table63_latex(
        inputs=adapted_inputs,
        dataset_scores=table63_scores,
        avg_scores=table63_avg,
        datasets=datasets,
        score_label=table63_score_label,
        precision=table63_precision,
        missing=args.missing,
        bold_best=not bool(args.no_bold_best),
    )

    # Figure 6.3
    improvements, improvement_details, improvement_summary = compute_improvement_values(
        values=values,
        training=adapted_spec.training,
        priors=adapted_priors,
        datasets=datasets,
        metrics=metrics,
    )
    fig_title = args.figure_title if str(args.figure_title).strip() else None
    fig = plot_figure63(
        improvements=improvements,
        priors=adapted_priors,
        metrics=metrics,
        title=fig_title,
        ylabel=args.figure_ylabel,
        ncols=int(args.figure_ncols),
        figsize_scale=float(args.figsize_scale),
        annotate=bool(args.annotate),
    )

    # Table 6.4
    table64_rows, table64_details = compute_training_success_rows(
        values=values,
        specs=[zero_spec, adapted_spec],
        datasets=datasets,
        metrics=metrics,
        improve_eps=float(args.improve_eps),
    )
    table64_md = build_table64_markdown(table64_rows, args.success_format, int(args.precision_success), include_caption=not bool(args.no_caption_markdown))
    table64_tex = build_table64_latex(table64_rows, args.success_format, int(args.precision_success))

    # Paths
    table62_stem = safe_stem(args.table62_stem)
    table63_stem = safe_stem(args.table63_stem)
    table64_stem = safe_stem(args.table64_stem)
    figure63_stem = safe_stem(args.figure63_stem)

    paths = {
        "table62_md": output_dir / f"{table62_stem}.md",
        "table62_tex": output_dir / f"{table62_stem}.tex",
        "table62_csv": output_dir / f"{table62_stem}.csv",
        "table63_md": output_dir / f"{table63_stem}.md",
        "table63_tex": output_dir / f"{table63_stem}.tex",
        "table63_csv": output_dir / f"{table63_stem}.csv",
        "table64_md": output_dir / f"{table64_stem}.md",
        "table64_tex": output_dir / f"{table64_stem}.tex",
        "table64_csv": output_dir / f"{table64_stem}.csv",
        "figure63_png": output_dir / f"{figure63_stem}.png",
        "figure63_pdf": output_dir / f"{figure63_stem}.pdf",
        "figure63_svg": output_dir / f"{figure63_stem}.svg",
        "figure63_csv": output_dir / f"{figure63_stem}_improvement_values.csv",
        "details_csv": output_dir / "table62_63_64_figure63_details.csv",
        "metadata": output_dir / "table62_63_64_figure63_metadata.json",
    }

    # Write outputs
    paths["table62_md"].write_text(table62_md + "\n", encoding="utf-8")
    paths["table62_tex"].write_text(table62_tex + "\n", encoding="utf-8")
    write_table62_csv(paths["table62_csv"], adapted_inputs, table62_values, table62_avg_ranks, metrics, adapted_spec.training)

    paths["table63_md"].write_text(table63_md + "\n", encoding="utf-8")
    paths["table63_tex"].write_text(table63_tex + "\n", encoding="utf-8")
    write_table63_csv(paths["table63_csv"], adapted_inputs, table63_scores, table63_avg, datasets, table63_score_label)

    paths["table64_md"].write_text(table64_md + "\n", encoding="utf-8")
    paths["table64_tex"].write_text(table64_tex + "\n", encoding="utf-8")
    write_rows_csv(paths["table64_csv"], table64_rows)

    fig.savefig(paths["figure63_png"], dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(paths["figure63_pdf"], bbox_inches="tight")
    if args.save_svg:
        fig.savefig(paths["figure63_svg"], bbox_inches="tight")
    plt.close(fig)
    write_rows_csv(paths["figure63_csv"], improvement_summary)

    all_details: list[dict] = []
    for row in collection_details:
        all_details.append({"source": "benchmark_collection", **row})
    for row in improvement_details:
        all_details.append({"source": "figure6.3_improvement", **row})
    for row in table64_details:
        all_details.append({"source": "table6.4_success", **row})
    write_rows_csv(paths["details_csv"], all_details)

    all_warnings = dedupe_warnings(load_warnings + collection_warnings)
    metadata = {
        "benchmarking_root": str(root),
        "output_dir": str(output_dir),
        "views": views,
        "zero_shot_spec": {
            "model": zero_spec.model,
            "training": zero_spec.training,
            "inputs": {k: list(v) for k, v in zero_spec.input_subdirs.items()},
        },
        "adapted_spec": {
            "model": adapted_spec.model,
            "training": adapted_spec.training,
            "inputs": {k: list(v) for k, v in adapted_spec.input_subdirs.items()},
        },
        "datasets": {k: list(v.aliases) for k, v in datasets.items()},
        "metrics": {k: {"key": v.key, "display": v.display, "precision": v.precision} for k, v in metrics.items()},
        "json_name": json_name,
        "agg": args.agg,
        "table63_mode": args.table63_mode,
        "table63_metric": args.table63_metric,
        "improve_eps_percent": args.improve_eps,
        "success_format": args.success_format,
        "figure63_formula": "Improvement(%) = (Error_RGB - Error_prior) / Error_RGB * 100",
        "outputs": {k: str(v) for k, v in paths.items() if k != "figure63_svg" or args.save_svg},
        "warnings": all_warnings,
    }
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(table62_md)
    print()
    print(table63_md)
    print()
    print(table64_md)
    print()
    print(f"Wrote Table 6.2 Markdown: {paths['table62_md']}")
    print(f"Wrote Table 6.3 Markdown: {paths['table63_md']}")
    print(f"Wrote Table 6.4 Markdown: {paths['table64_md']}")
    print(f"Wrote Figure 6.3 PNG:      {paths['figure63_png']}")
    print(f"Wrote Figure 6.3 PDF:      {paths['figure63_pdf']}")
    if args.save_svg:
        print(f"Wrote Figure 6.3 SVG:      {paths['figure63_svg']}")
    print(f"Wrote details CSV:         {paths['details_csv']}")
    print(f"Wrote metadata:            {paths['metadata']}")

    if all_warnings and not args.quiet:
        print("\nWarnings:")
        for w in all_warnings[:100]:
            print(f"  - {w}")
        if len(all_warnings) > 100:
            print(f"  ... {len(all_warnings) - 100} more warnings. See {paths['metadata']}")


if __name__ == "__main__":
    main()

"""
python scripts/viz/table62_63_64_figure63_prior_aware_adaptation.py \
  --benchmarking experiments/mapanything/benchmarking \
  --views 8,16,24,32 \
  --output experiments/mapanything/benchmarking/figures_tables/table62_63_64_figure63
"""