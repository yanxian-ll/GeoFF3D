#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Table 5.2, Table 5.3, and Table 5.4 for Chapter 5 domain-adaptation ablations.

Table 5.2:
  Dense geometry generalization after domain adaptation.
  Metrics: RelDepth, Chamfer-L1

Table 5.3:
  Camera geometry generalization after domain adaptation.
  Metrics: Ray Error, Pose ATE

Table 5.4:
  Average performance over UAV benchmarks.
  Metrics: average RelDepth, Chamfer-L1, Ray Error, and Pose ATE over all selected datasets

Expected benchmark layout:
  <benchmarking>/
    dense_8_view/<method_subdir>/per_dataset_results.json
    dense_16_view/<method_subdir>/per_dataset_results.json
    dense_24_view/<method_subdir>/per_dataset_results.json
    dense_32_view/<method_subdir>/per_dataset_results.json

The script averages each metric over selected view settings first, then formats
paper-ready Markdown / LaTeX / CSV tables.

Example:
  python scripts/viz/table52_53_54_domain_adaptation.py \
    --benchmarking experiments/mapanything/benchmarking \
    --views 8,16,24,32 \
    --output experiments/mapanything/benchmarking/tables/table52_53_54
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


# -----------------------------------------------------------------------------
# Specs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SettingSpec:
    label: str
    subdirs: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class MetricSpec:
    alias: str
    key: str
    display: str
    precision: int


@dataclass(frozen=True)
class TableSpec:
    table_id: str
    caption: str
    latex_label: str
    stem: str
    metrics: "OrderedDict[str, MetricSpec]"
    # dataset_metric: columns are Dataset x Metric; metric_average: columns are Metric averaged over datasets.
    layout: str = "dataset_metric"


DEFAULT_SETTINGS: "OrderedDict[str, SettingSpec]" = OrderedDict(
    [
        # Keep several candidates for the zero-shot MapAnything directory because
        # different scripts / experiments may use slightly different names.
        ("MA-Pretrained", SettingSpec("MA-Pretrained", ("mapa_24v", "mapa", "mapanything"))),
        ("MA-FT-Public", SettingSpec("MA-FT-Public", ("mapa-ft-public",))),
        ("MA-FT-A3D-Syn", SettingSpec("MA-FT-A3D-Syn", ("mapa-ft-a3dsyn",))),
        ("MA-FT-A3D-Full", SettingSpec("MA-FT-A3D-Full", ("mapa-ft-a3dfull", "uav_mapa",))),
    ]
)

DEFAULT_DATASETS: "OrderedDict[str, DatasetSpec]" = OrderedDict(
    [
        ("UseGeo", DatasetSpec("UseGeo", ("UseGeoWAI", "UseGeo", "usegeo"))),
        (
            "Enrich",
            DatasetSpec(
                "Enrich",
                ("EnrichAerialWAI", "Enrich-Aerial", "EnrichAerial", "Enrich", "enrich_aerial"),
            ),
        ),
        (
            "Urban",
            DatasetSpec("Urban", ("UrbanScene3DWAI", "UrbanScene3D", "US3D", "us3d", "Urban")),
        ),
        (
            "A3D-Real",
            DatasetSpec("A3D-Real", ("A3DRealWAI", "A3D-Real", "A3DReal", "A3D_Real", "a3d_real")),
        ),
    ]
)


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def split_csv_like(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def safe_stem(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")
    return out or "output"


def is_finite_number(x: object) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)


def mean_finite(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if is_finite_number(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def aggregate_value(value: Any, mode: str = "mean") -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        vals = [float(v) for v in value if is_finite_number(v)]
        if not vals:
            return float("nan")
        if mode == "mean":
            return float(sum(vals) / len(vals))
        if mode == "median":
            return float(statistics.median(vals))
        if mode == "first":
            return float(vals[0])
        if mode == "second":
            return float(vals[1] if len(vals) > 1 else vals[0])
    return float("nan")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return obj


def ensure_json_filename(name: str) -> str:
    return name if name.endswith(".json") else f"{name}.json"


def tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


# -----------------------------------------------------------------------------
# CLI spec parsing
# -----------------------------------------------------------------------------


def parse_setting_specs(items: list[str] | None) -> "OrderedDict[str, SettingSpec]":
    if not items:
        return DEFAULT_SETTINGS.copy()
    out: "OrderedDict[str, SettingSpec]" = OrderedDict()
    for item in items:
        if "=" in item:
            label, rhs = item.split("=", 1)
            label = label.strip()
            subdirs = tuple(x.strip() for x in rhs.split("|") if x.strip())
        else:
            label = item.strip()
            subdirs = (label,)
        if not label or not subdirs:
            raise ValueError(f"Bad setting spec: {item!r}. Expected Label=subdir1|subdir2")
        out[label] = SettingSpec(label=label, subdirs=subdirs)
    return out


def parse_dataset_specs(items: list[str] | None) -> "OrderedDict[str, DatasetSpec]":
    if not items:
        return DEFAULT_DATASETS.copy()
    out: "OrderedDict[str, DatasetSpec]" = OrderedDict()
    for item in items:
        if "=" in item:
            label, rhs = item.split("=", 1)
            label = label.strip()
            aliases = tuple(x.strip() for x in rhs.split("|") if x.strip())
        else:
            label = item.strip()
            aliases = (label,)
        if not label or not aliases:
            raise ValueError(f"Bad dataset spec: {item!r}. Expected Label=alias1|alias2")
        out[label] = DatasetSpec(label=label, aliases=aliases)
    return out


def discover_views(root: Path) -> list[int]:
    views: list[int] = []
    for p in root.glob("dense_*_view"):
        if not p.is_dir():
            continue
        m = re.fullmatch(r"dense_(\d+)_view", p.name)
        if m:
            views.append(int(m.group(1)))
    return sorted(set(views))


def parse_views(spec: str | None, root: Path) -> list[int]:
    if spec is None or not str(spec).strip():
        return [8, 16, 24, 32]
    if str(spec).strip().lower() in {"all", "auto"}:
        return discover_views(root)
    return sorted(set(int(x) for x in split_csv_like(spec)))


def parse_tables(spec: str | None) -> list[str]:
    if spec is None or not str(spec).strip():
        return ["5.2", "5.3", "5.4"]
    raw = str(spec).strip().lower().replace(" ", "")
    if raw in {"all", "both", "52,53,54", "5.2,5.3,5.4"}:
        return ["5.2", "5.3", "5.4"]
    if raw in {"52,53", "5.2,5.3"}:
        return ["5.2", "5.3"]
    out: list[str] = []
    for item in split_csv_like(raw):
        item = item.replace("table", "").strip()
        if item in {"52", "5.2"}:
            out.append("5.2")
        elif item in {"53", "5.3"}:
            out.append("5.3")
        elif item in {"54", "5.4"}:
            out.append("5.4")
        else:
            raise ValueError(f"Unknown table spec {item!r}; use all, 5.2, 5.3, 5.4, or comma-separated selections.")
    return list(OrderedDict.fromkeys(out))


def make_table_specs(args: argparse.Namespace) -> "OrderedDict[str, TableSpec]":
    dense_metrics: "OrderedDict[str, MetricSpec]" = OrderedDict(
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
        ]
    )
    camera_metrics: "OrderedDict[str, MetricSpec]" = OrderedDict(
        [
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

    average_metrics: "OrderedDict[str, MetricSpec]" = OrderedDict()
    average_metrics.update(dense_metrics)
    average_metrics.update(camera_metrics)

    all_specs: "OrderedDict[str, TableSpec]" = OrderedDict(
        [
            (
                "5.2",
                TableSpec(
                    table_id="5.2",
                    caption="Dense geometry generalization after domain adaptation.",
                    latex_label="tab:dense_geometry_generalization_domain_adaptation",
                    stem=args.table52_stem,
                    metrics=dense_metrics,
                    layout="dataset_metric",
                ),
            ),
            (
                "5.3",
                TableSpec(
                    table_id="5.3",
                    caption="Camera geometry generalization after domain adaptation.",
                    latex_label="tab:camera_geometry_generalization_domain_adaptation",
                    stem=args.table53_stem,
                    metrics=camera_metrics,
                    layout="dataset_metric",
                ),
            ),
            (
                "5.4",
                TableSpec(
                    table_id="5.4",
                    caption="Average performance over UAV benchmarks.",
                    latex_label="tab:average_performance_uav_benchmarks",
                    stem=args.table54_stem,
                    metrics=average_metrics,
                    layout="metric_average",
                ),
            ),
        ]
    )

    selected = parse_tables(args.tables)
    return OrderedDict((k, all_specs[k]) for k in selected)


# -----------------------------------------------------------------------------
# JSON lookup
# -----------------------------------------------------------------------------


def find_setting_json(
    root: Path,
    view: int,
    setting: SettingSpec,
    json_name: str,
    fuzzy: bool = True,
) -> tuple[Path | None, str | None]:
    view_dir = root / f"dense_{view}_view"
    if not view_dir.is_dir():
        return None, None

    for subdir in setting.subdirs:
        candidate = view_dir / subdir / json_name
        if candidate.is_file():
            return candidate, subdir

    if fuzzy:
        wanted = {norm_name(x) for x in setting.subdirs}
        for method_dir in sorted(view_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            if norm_name(method_dir.name) in wanted and (method_dir / json_name).is_file():
                return method_dir / json_name, method_dir.name

    return None, None


def dataset_match_score(raw_key: str, dataset: DatasetSpec) -> int:
    """Return a higher score for a better match; 0 means no match."""
    raw_norm = norm_name(raw_key)
    aliases = [dataset.label, *dataset.aliases]
    best = 0
    for alias in aliases:
        alias_norm = norm_name(alias)
        if not alias_norm:
            continue
        if raw_norm == alias_norm:
            best = max(best, 1000 + len(alias_norm))
        elif alias_norm in raw_norm:
            best = max(best, 500 + len(alias_norm))
        elif raw_norm in alias_norm:
            best = max(best, 100 + len(raw_norm))
    return best


def find_dataset_obj(json_obj: Mapping[str, Any], dataset: DatasetSpec) -> tuple[Mapping[str, Any] | None, str | None]:
    scored: list[tuple[int, str, Any]] = []
    for raw_key, raw_val in json_obj.items():
        if not isinstance(raw_val, dict):
            continue
        score = dataset_match_score(str(raw_key), dataset)
        if score > 0:
            scored.append((score, str(raw_key), raw_val))
    if not scored:
        return None, None
    scored.sort(key=lambda x: (-x[0], len(x[1]), x[1]))
    return scored[0][2], scored[0][1]


def load_setting_jsons(
    root: Path,
    views: list[int],
    settings: "OrderedDict[str, SettingSpec]",
    json_name: str,
    strict: bool,
) -> tuple[dict[str, dict[int, tuple[Path, str, dict[str, Any]]]], list[str]]:
    """
    Load each setting/view JSON once and share it across Table 5.2, Table 5.3, and Table 5.4.
    """
    loaded: dict[str, dict[int, tuple[Path, str, dict[str, Any]]]] = OrderedDict()
    warnings: list[str] = []

    for setting_label, setting in settings.items():
        loaded[setting_label] = OrderedDict()
        for view in views:
            json_path, used_subdir = find_setting_json(root, view, setting, json_name=json_name)
            if json_path is None or used_subdir is None:
                msg = (
                    f"Missing {json_name} for setting={setting_label!r}, view={view}. "
                    f"Tried subdirs={setting.subdirs} under {root / f'dense_{view}_view'}"
                )
                if strict:
                    raise FileNotFoundError(msg)
                warnings.append(msg)
                continue
            try:
                loaded[setting_label][view] = (json_path, used_subdir, load_json(json_path))
            except Exception as exc:
                msg = f"Failed to read {json_path}: {exc}"
                if strict:
                    raise
                warnings.append(msg)

    return loaded, warnings


# -----------------------------------------------------------------------------
# Collection / ranking
# -----------------------------------------------------------------------------


def collect_table_values(
    loaded_jsons: Mapping[str, Mapping[int, tuple[Path, str, dict[str, Any]]]],
    settings: "OrderedDict[str, SettingSpec]",
    datasets: "OrderedDict[str, DatasetSpec]",
    table: TableSpec,
    agg: str,
    strict: bool,
) -> tuple[dict, list[dict], list[str]]:
    """
    Returns:
      values[setting][dataset][metric_alias] = mean over selected views
      details: long-form rows for auditing
      warnings: missing dataset / metric warnings
    """
    values: dict = OrderedDict()
    details: list[dict] = []
    warnings: list[str] = []

    for setting_label, setting in settings.items():
        values[setting_label] = OrderedDict()
        json_by_view = loaded_jsons.get(setting_label, {})

        for dataset_label, dataset in datasets.items():
            values[setting_label][dataset_label] = OrderedDict()
            for metric_alias, metric in table.metrics.items():
                collected: list[float] = []
                used_views: list[int] = []
                used_dataset_keys: list[str] = []
                used_json_paths: list[str] = []
                used_subdirs: list[str] = []

                for view, (json_path, used_subdir, json_obj) in json_by_view.items():
                    dataset_obj, raw_dataset_key = find_dataset_obj(json_obj, dataset)
                    if dataset_obj is None or raw_dataset_key is None:
                        msg = (
                            f"Dataset {dataset_label!r} not found for table={table.table_id}, "
                            f"setting={setting_label!r}, view={view}, json={json_path}. "
                            f"Available keys={list(json_obj.keys())}"
                        )
                        if strict:
                            raise KeyError(msg)
                        warnings.append(msg)
                        continue

                    raw_value = dataset_obj.get(metric.key, None)
                    value = aggregate_value(raw_value, mode=agg)
                    if is_finite_number(value):
                        collected.append(float(value))
                        used_views.append(view)
                        used_dataset_keys.append(raw_dataset_key)
                        used_json_paths.append(str(json_path))
                        used_subdirs.append(used_subdir)
                    else:
                        msg = (
                            f"Metric {metric.key!r} is missing/non-finite for table={table.table_id}, "
                            f"setting={setting_label!r}, dataset={raw_dataset_key!r}, "
                            f"view={view}, json={json_path}"
                        )
                        if strict:
                            raise KeyError(msg)
                        warnings.append(msg)

                mean_value = mean_finite(collected)
                values[setting_label][dataset_label][metric_alias] = mean_value
                details.append(
                    {
                        "table": table.table_id,
                        "table_caption": table.caption,
                        "setting": setting_label,
                        "setting_subdir_candidates": "|".join(setting.subdirs),
                        "setting_subdirs_used": "|".join(OrderedDict.fromkeys(used_subdirs).keys()),
                        "dataset": dataset_label,
                        "metric_alias": metric_alias,
                        "metric_display": metric.display,
                        "metric_key": metric.key,
                        "value_mean_over_views": mean_value,
                        "n_views_used": len(used_views),
                        "views_used": ",".join(str(v) for v in used_views),
                        "dataset_keys_used": "|".join(OrderedDict.fromkeys(used_dataset_keys).keys()),
                        "json_paths_used": "|".join(OrderedDict.fromkeys(used_json_paths).keys()),
                    }
                )

    return values, details, warnings


def average_tie_ranks(label_to_value: Mapping[str, float]) -> dict[str, float]:
    finite = [(label, float(v)) for label, v in label_to_value.items() if is_finite_number(v)]
    finite.sort(key=lambda x: x[1])  # lower is better
    ranks: dict[str, float] = {label: float("nan") for label in label_to_value.keys()}
    i = 0
    while i < len(finite):
        j = i + 1
        while j < len(finite) and math.isclose(finite[j][1], finite[i][1], rel_tol=1e-12, abs_tol=1e-12):
            j += 1
        # Ranks are 1-based; average rank for ties.
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[finite[k][0]] = avg_rank
        i = j
    return ranks


def compute_avg_column(
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
    mode: str,
) -> dict[str, float]:
    settings = list(values.keys())
    if mode == "raw":
        out: dict[str, float] = {}
        for setting in settings:
            vals = []
            for dataset_label in datasets.keys():
                for metric_alias in metrics.keys():
                    vals.append(values[setting][dataset_label][metric_alias])
            out[setting] = mean_finite(vals)
        return out

    if mode == "rank":
        rank_lists: dict[str, list[float]] = {s: [] for s in settings}
        for dataset_label in datasets.keys():
            for metric_alias in metrics.keys():
                label_to_value = {
                    setting: values[setting][dataset_label][metric_alias]
                    for setting in settings
                }
                ranks = average_tie_ranks(label_to_value)
                for setting in settings:
                    if is_finite_number(ranks.get(setting, float("nan"))):
                        rank_lists[setting].append(float(ranks[setting]))
        return {setting: mean_finite(rs) for setting, rs in rank_lists.items()}

    raise ValueError(f"Unknown avg mode: {mode}")


def best_masks(
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
    avg_values: Mapping[str, float],
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
) -> tuple[dict, set[str]]:
    best_cells: dict = {}
    for dataset_label in datasets.keys():
        best_cells[dataset_label] = {}
        for metric_alias in metrics.keys():
            vals = {
                setting: values[setting][dataset_label][metric_alias]
                for setting in values.keys()
            }
            finite_vals = [v for v in vals.values() if is_finite_number(v)]
            if not finite_vals:
                best_cells[dataset_label][metric_alias] = set()
                continue
            best_v = min(finite_vals)
            best_cells[dataset_label][metric_alias] = {
                setting for setting, v in vals.items()
                if is_finite_number(v) and math.isclose(float(v), best_v, rel_tol=1e-12, abs_tol=1e-12)
            }

    finite_avg = [v for v in avg_values.values() if is_finite_number(v)]
    if not finite_avg:
        best_avg = set()
    else:
        best_v = min(finite_avg)
        best_avg = {
            setting for setting, v in avg_values.items()
            if is_finite_number(v) and math.isclose(float(v), best_v, rel_tol=1e-12, abs_tol=1e-12)
        }
    return best_cells, best_avg


# -----------------------------------------------------------------------------
# Formatting / writers
# -----------------------------------------------------------------------------


def fmt_number(value: float, precision: int, missing: str) -> str:
    if not is_finite_number(value):
        return missing
    return f"{float(value):.{precision}f}"


def maybe_bold_md(s: str, bold: bool) -> str:
    if not bold or s == "":
        return s
    return f"**{s}**"


def maybe_bold_tex(s: str, bold: bool, missing: str) -> str:
    return f"\\textbf{{{s}}}" if bold and s != missing else s


def metric_header(dataset_label: str, metric: MetricSpec) -> str:
    return f"{dataset_label} {metric.display} ↓"


def avg_header(avg_mode: str) -> str:
    return "Avg. Rank ↓" if avg_mode == "rank" else "Avg. ↓"


def build_markdown_table(
    table: TableSpec,
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
    avg_values: Mapping[str, float],
    datasets: "OrderedDict[str, DatasetSpec]",
    avg_mode: str,
    precision_avg: int,
    missing: str,
    bold_best: bool,
    include_caption: bool,
) -> str:
    best_cells, best_avg = best_masks(values, avg_values, datasets, table.metrics)

    headers = ["Setting"]
    for dataset_label in datasets.keys():
        for metric in table.metrics.values():
            headers.append(metric_header(dataset_label, metric))
    headers.append(avg_header(avg_mode))

    aligns = [":---"] + ["---:" for _ in headers[1:]]
    rows = []
    if include_caption:
        rows.append(f"**Table {table.table_id}. {table.caption}**")
        rows.append("")
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(aligns) + " |")

    for setting in values.keys():
        row = [setting]
        for dataset_label in datasets.keys():
            for metric_alias, metric in table.metrics.items():
                val = values[setting][dataset_label][metric_alias]
                cell = fmt_number(val, metric.precision, missing)
                cell = maybe_bold_md(cell, bold_best and setting in best_cells[dataset_label][metric_alias])
                row.append(cell)
        avg_cell = fmt_number(avg_values.get(setting, float("nan")), precision_avg, missing)
        avg_cell = maybe_bold_md(avg_cell, bold_best and setting in best_avg)
        row.append(avg_cell)
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def build_latex_table(
    table: TableSpec,
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
    avg_values: Mapping[str, float],
    datasets: "OrderedDict[str, DatasetSpec]",
    avg_mode: str,
    precision_avg: int,
    missing: str,
    bold_best: bool,
) -> str:
    best_cells, best_avg = best_masks(values, avg_values, datasets, table.metrics)
    avg_header_tex = "Avg. Rank $\\downarrow$" if avg_mode == "rank" else "Avg. $\\downarrow$"

    n_numeric = len(datasets) * len(table.metrics) + 1
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{tex_escape(table.caption)}}}")
    lines.append(f"\\label{{{tex_escape(table.latex_label)}}}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{l" + "r" * n_numeric + "}")
    lines.append("\\toprule")
    header = ["Setting"]
    for dataset_label in datasets.keys():
        for metric in table.metrics.values():
            header.append(f"{dataset_label} {metric.display} $\\downarrow$")
    header.append(avg_header_tex)
    lines.append(" & ".join(tex_escape(x) if "$" not in x else x for x in header) + " " + r"\\")
    lines.append("\\midrule")
    for setting in values.keys():
        row = [tex_escape(setting)]
        for dataset_label in datasets.keys():
            for metric_alias, metric in table.metrics.items():
                val = values[setting][dataset_label][metric_alias]
                cell = fmt_number(val, metric.precision, missing)
                cell = maybe_bold_tex(cell, bold_best and setting in best_cells[dataset_label][metric_alias], missing)
                row.append(cell)
        avg_cell = fmt_number(avg_values.get(setting, float("nan")), precision_avg, missing)
        avg_cell = maybe_bold_tex(avg_cell, bold_best and setting in best_avg, missing)
        row.append(avg_cell)
        lines.append(" & ".join(row) + " " + r"\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def write_summary_csv(
    path: Path,
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
    avg_values: Mapping[str, float],
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
    avg_mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    avg_field = "avg_rank" if avg_mode == "rank" else "avg_raw"
    fieldnames = ["setting"]
    for dataset_label in datasets.keys():
        for metric_alias in metrics.keys():
            fieldnames.append(f"{safe_stem(dataset_label)}_{metric_alias}")
    fieldnames.append(avg_field)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for setting in values.keys():
            row = {"setting": setting}
            for dataset_label in datasets.keys():
                for metric_alias in metrics.keys():
                    row[f"{safe_stem(dataset_label)}_{metric_alias}"] = values[setting][dataset_label][metric_alias]
            row[avg_field] = avg_values.get(setting, float("nan"))
            writer.writerow(row)




# -----------------------------------------------------------------------------
# Table 5.4: metric averages over UAV benchmarks
# -----------------------------------------------------------------------------


def compute_metric_average_values(
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
) -> dict[str, dict[str, float]]:
    """Average each metric over selected datasets for Table 5.4."""
    out: dict[str, dict[str, float]] = OrderedDict()
    for setting in values.keys():
        out[setting] = OrderedDict()
        for metric_alias in metrics.keys():
            vals = [
                values[setting][dataset_label][metric_alias]
                for dataset_label in datasets.keys()
                if dataset_label in values[setting]
                and metric_alias in values[setting][dataset_label]
            ]
            out[setting][metric_alias] = mean_finite(vals)
    return out


def compute_metric_average_avg_column(
    metric_values: Mapping[str, Mapping[str, float]],
    metrics: "OrderedDict[str, MetricSpec]",
    mode: str,
) -> dict[str, float]:
    settings = list(metric_values.keys())
    if mode == "raw":
        return {
            setting: mean_finite(metric_values[setting].get(metric_alias, float("nan")) for metric_alias in metrics.keys())
            for setting in settings
        }

    if mode == "rank":
        rank_lists: dict[str, list[float]] = {s: [] for s in settings}
        for metric_alias in metrics.keys():
            label_to_value = {
                setting: metric_values[setting].get(metric_alias, float("nan"))
                for setting in settings
            }
            ranks = average_tie_ranks(label_to_value)
            for setting in settings:
                if is_finite_number(ranks.get(setting, float("nan"))):
                    rank_lists[setting].append(float(ranks[setting]))
        return {setting: mean_finite(rs) for setting, rs in rank_lists.items()}

    raise ValueError(f"Unknown avg mode: {mode}")


def best_metric_average_masks(
    metric_values: Mapping[str, Mapping[str, float]],
    avg_values: Mapping[str, float],
    metrics: "OrderedDict[str, MetricSpec]",
) -> tuple[dict[str, set[str]], set[str]]:
    best_cells: dict[str, set[str]] = {}
    for metric_alias in metrics.keys():
        vals = {
            setting: metric_values[setting].get(metric_alias, float("nan"))
            for setting in metric_values.keys()
        }
        finite_vals = [v for v in vals.values() if is_finite_number(v)]
        if not finite_vals:
            best_cells[metric_alias] = set()
            continue
        best_v = min(finite_vals)
        best_cells[metric_alias] = {
            setting for setting, v in vals.items()
            if is_finite_number(v) and math.isclose(float(v), best_v, rel_tol=1e-12, abs_tol=1e-12)
        }

    finite_avg = [v for v in avg_values.values() if is_finite_number(v)]
    if not finite_avg:
        best_avg = set()
    else:
        best_v = min(finite_avg)
        best_avg = {
            setting for setting, v in avg_values.items()
            if is_finite_number(v) and math.isclose(float(v), best_v, rel_tol=1e-12, abs_tol=1e-12)
        }
    return best_cells, best_avg


def build_metric_average_markdown_table(
    table: TableSpec,
    metric_values: Mapping[str, Mapping[str, float]],
    avg_values: Mapping[str, float],
    avg_mode: str,
    precision_avg: int,
    missing: str,
    bold_best: bool,
    include_caption: bool,
) -> str:
    best_cells, best_avg = best_metric_average_masks(metric_values, avg_values, table.metrics)

    headers = ["Setting"] + [f"Avg. {m.display} ↓" for m in table.metrics.values()] + [avg_header(avg_mode)]
    aligns = [":---"] + ["---:" for _ in headers[1:]]
    rows: list[str] = []
    if include_caption:
        rows.append(f"**Table {table.table_id}. {table.caption}**")
        rows.append("")
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(aligns) + " |")

    for setting in metric_values.keys():
        row = [setting]
        for metric_alias, metric in table.metrics.items():
            val = metric_values[setting].get(metric_alias, float("nan"))
            cell = fmt_number(val, metric.precision, missing)
            cell = maybe_bold_md(cell, bold_best and setting in best_cells[metric_alias])
            row.append(cell)
        avg_cell = fmt_number(avg_values.get(setting, float("nan")), precision_avg, missing)
        avg_cell = maybe_bold_md(avg_cell, bold_best and setting in best_avg)
        row.append(avg_cell)
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def build_metric_average_latex_table(
    table: TableSpec,
    metric_values: Mapping[str, Mapping[str, float]],
    avg_values: Mapping[str, float],
    avg_mode: str,
    precision_avg: int,
    missing: str,
    bold_best: bool,
) -> str:
    best_cells, best_avg = best_metric_average_masks(metric_values, avg_values, table.metrics)
    avg_header_tex = "Avg. Rank $\\downarrow$" if avg_mode == "rank" else "Avg. $\\downarrow$"

    n_numeric = len(table.metrics) + 1
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{tex_escape(table.caption)}}}")
    lines.append(f"\\label{{{tex_escape(table.latex_label)}}}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{l" + "r" * n_numeric + "}")
    lines.append("\\toprule")
    header = ["Setting"] + [f"Avg. {m.display} $\\downarrow$" for m in table.metrics.values()] + [avg_header_tex]
    lines.append(" & ".join(tex_escape(x) if "$" not in x else x for x in header) + " " + r"\\")
    lines.append("\\midrule")
    for setting in metric_values.keys():
        row = [tex_escape(setting)]
        for metric_alias, metric in table.metrics.items():
            val = metric_values[setting].get(metric_alias, float("nan"))
            cell = fmt_number(val, metric.precision, missing)
            cell = maybe_bold_tex(cell, bold_best and setting in best_cells[metric_alias], missing)
            row.append(cell)
        avg_cell = fmt_number(avg_values.get(setting, float("nan")), precision_avg, missing)
        avg_cell = maybe_bold_tex(avg_cell, bold_best and setting in best_avg, missing)
        row.append(avg_cell)
        lines.append(" & ".join(row) + " " + r"\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def write_metric_average_summary_csv(
    path: Path,
    metric_values: Mapping[str, Mapping[str, float]],
    avg_values: Mapping[str, float],
    metrics: "OrderedDict[str, MetricSpec]",
    avg_mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    avg_field = "avg_rank" if avg_mode == "rank" else "avg_raw"
    fieldnames = ["setting"] + [f"avg_{metric_alias}" for metric_alias in metrics.keys()] + [avg_field]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for setting in metric_values.keys():
            row = {"setting": setting}
            for metric_alias in metrics.keys():
                row[f"avg_{metric_alias}"] = metric_values[setting].get(metric_alias, float("nan"))
            row[avg_field] = avg_values.get(setting, float("nan"))
            writer.writerow(row)


def write_details_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            rr = dict(row)
            for k, v in list(rr.items()):
                if isinstance(v, float) and not math.isfinite(v):
                    rr[k] = ""
            writer.writerow(rr)


def dedupe_warnings(warnings: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build Table 5.2, Table 5.3, and Table 5.4 domain-adaptation generalization tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmarking", type=Path, default=Path("experiments/mapanything/benchmarking"))
    p.add_argument("--output", type=Path, default=None, help="Output directory. Default: <benchmarking>/tables/table52_53_54")
    p.add_argument("--views", type=str, default="8,16,24,32", help="Comma-separated views or 'all'.")
    p.add_argument("--tables", type=str, default="all", help="Which tables to build: all, 5.2, 5.3, 5.4, or comma-separated selections.")
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
    p.add_argument("--json-name", type=str, default="per_dataset_results.json")
    p.add_argument("--depth-key", type=str, default="abs_depth_rel_scale_aligned")
    p.add_argument("--chamfer-key", type=str, default="abs_fused_pc_chamfer_l1")
    p.add_argument("--ray-key", type=str, default="ray_dir_mean_angle_deg")
    p.add_argument("--pose-ate-key", type=str, default="abs_pose_ate")
    p.add_argument("--agg", choices=["mean", "median", "first", "second"], default="mean", help="How to reduce list-valued JSON entries.")
    p.add_argument(
        "--avg-mode",
        choices=["rank", "raw"],
        default="rank",
        help="Final Avg column. rank avoids mixing metrics with different units; raw averages displayed numbers directly.",
    )
    p.add_argument("--precision-depth", type=int, default=3)
    p.add_argument("--precision-chamfer", type=int, default=3)
    p.add_argument("--precision-ray", type=int, default=2)
    p.add_argument("--precision-pose-ate", type=int, default=3)
    p.add_argument("--precision-avg", type=int, default=2)
    p.add_argument("--missing", type=str, default="")
    p.add_argument("--table52-stem", type=str, default="table52_dense_geometry_domain_adaptation")
    p.add_argument("--table53-stem", type=str, default="table53_camera_geometry_domain_adaptation")
    p.add_argument("--table54-stem", type=str, default="table54_average_uav_benchmarks")
    p.add_argument("--no-caption-markdown", action="store_true", help="Do not write the bold caption line above Markdown tables.")
    p.add_argument("--no-bold-best", action="store_true", help="Do not bold best values in Markdown/LaTeX.")
    p.add_argument("--strict", action="store_true", help="Raise on missing files/datasets/metrics instead of warning.")
    p.add_argument("--quiet", action="store_true", help="Suppress warning printout.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    root = Path(args.benchmarking)
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmarking root not found: {root}")

    output_dir = Path(args.output) if args.output is not None else root / "tables" / "table52_53_54"
    output_dir.mkdir(parents=True, exist_ok=True)

    views = parse_views(args.views, root)
    if not views:
        raise ValueError(f"No views selected/found under {root}")

    settings = parse_setting_specs(args.settings)
    datasets = parse_dataset_specs(args.datasets)
    tables = make_table_specs(args)
    json_name = ensure_json_filename(args.json_name)

    loaded_jsons, load_warnings = load_setting_jsons(
        root=root,
        views=views,
        settings=settings,
        json_name=json_name,
        strict=bool(args.strict),
    )

    all_details: list[dict] = []
    all_warnings: list[str] = list(load_warnings)
    outputs: dict[str, dict[str, str]] = {}

    for table_id, table in tables.items():
        values, details, warnings = collect_table_values(
            loaded_jsons=loaded_jsons,
            settings=settings,
            datasets=datasets,
            table=table,
            agg=args.agg,
            strict=bool(args.strict),
        )
        all_details.extend(details)
        all_warnings.extend(warnings)

        if table.layout == "metric_average":
            table_values_for_csv = compute_metric_average_values(
                values,
                datasets=datasets,
                metrics=table.metrics,
            )
            avg_values = compute_metric_average_avg_column(
                table_values_for_csv,
                metrics=table.metrics,
                mode=args.avg_mode,
            )
            md = build_metric_average_markdown_table(
                table=table,
                metric_values=table_values_for_csv,
                avg_values=avg_values,
                avg_mode=args.avg_mode,
                precision_avg=int(args.precision_avg),
                missing=args.missing,
                bold_best=not bool(args.no_bold_best),
                include_caption=not bool(args.no_caption_markdown),
            )
            latex = build_metric_average_latex_table(
                table=table,
                metric_values=table_values_for_csv,
                avg_values=avg_values,
                avg_mode=args.avg_mode,
                precision_avg=int(args.precision_avg),
                missing=args.missing,
                bold_best=not bool(args.no_bold_best),
            )
        else:
            table_values_for_csv = values
            avg_values = compute_avg_column(
                values,
                datasets=datasets,
                metrics=table.metrics,
                mode=args.avg_mode,
            )
            md = build_markdown_table(
                table=table,
                values=values,
                avg_values=avg_values,
                datasets=datasets,
                avg_mode=args.avg_mode,
                precision_avg=int(args.precision_avg),
                missing=args.missing,
                bold_best=not bool(args.no_bold_best),
                include_caption=not bool(args.no_caption_markdown),
            )
            latex = build_latex_table(
                table=table,
                values=values,
                avg_values=avg_values,
                datasets=datasets,
                avg_mode=args.avg_mode,
                precision_avg=int(args.precision_avg),
                missing=args.missing,
                bold_best=not bool(args.no_bold_best),
            )

        stem = safe_stem(table.stem)
        md_path = output_dir / f"{stem}.md"
        tex_path = output_dir / f"{stem}.tex"
        csv_path = output_dir / f"{stem}.csv"
        details_csv_path = output_dir / f"{stem}_details.csv"
        metadata_path = output_dir / f"{stem}_metadata.json"

        md_path.write_text(md + "\n", encoding="utf-8")
        tex_path.write_text(latex + "\n", encoding="utf-8")
        if table.layout == "metric_average":
            write_metric_average_summary_csv(csv_path, table_values_for_csv, avg_values, table.metrics, avg_mode=args.avg_mode)
        else:
            write_summary_csv(csv_path, table_values_for_csv, avg_values, datasets, table.metrics, avg_mode=args.avg_mode)
        write_details_csv(details_csv_path, details)

        table_metadata = {
            "table_id": table.table_id,
            "caption": table.caption,
            "benchmarking_root": str(root),
            "views": views,
            "settings": {k: list(v.subdirs) for k, v in settings.items()},
            "datasets": {k: list(v.aliases) for k, v in datasets.items()},
            "metrics": {k: {"key": v.key, "display": v.display} for k, v in table.metrics.items()},
            "layout": table.layout,
            "json_name": json_name,
            "agg": args.agg,
            "avg_mode": args.avg_mode,
            "outputs": {
                "markdown": str(md_path),
                "latex": str(tex_path),
                "csv": str(csv_path),
                "details_csv": str(details_csv_path),
            },
            "warnings": dedupe_warnings(load_warnings + warnings),
        }
        metadata_path.write_text(json.dumps(table_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        outputs[table_id] = {
            "markdown": str(md_path),
            "latex": str(tex_path),
            "csv": str(csv_path),
            "details_csv": str(details_csv_path),
            "metadata": str(metadata_path),
        }

        print(md)
        print(f"\nWrote Table {table_id} Markdown: {md_path}")
        print(f"Wrote Table {table_id} LaTeX:    {tex_path}")
        print(f"Wrote Table {table_id} CSV:      {csv_path}")
        print(f"Wrote Table {table_id} details:  {details_csv_path}")
        print(f"Wrote Table {table_id} metadata: {metadata_path}\n")

    combined_details_path = output_dir / "table52_53_54_domain_adaptation_details.csv"
    combined_metadata_path = output_dir / "table52_53_54_domain_adaptation_metadata.json"
    write_details_csv(combined_details_path, all_details)

    combined_metadata = {
        "benchmarking_root": str(root),
        "output_dir": str(output_dir),
        "views": views,
        "tables": list(tables.keys()),
        "settings": {k: list(v.subdirs) for k, v in settings.items()},
        "datasets": {k: list(v.aliases) for k, v in datasets.items()},
        "json_name": json_name,
        "agg": args.agg,
        "avg_mode": args.avg_mode,
        "outputs": outputs,
        "combined_details_csv": str(combined_details_path),
        "warnings": dedupe_warnings(all_warnings),
    }
    combined_metadata_path.write_text(json.dumps(combined_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote combined details:  {combined_details_path}")
    print(f"Wrote combined metadata: {combined_metadata_path}")

    uniq_warnings = dedupe_warnings(all_warnings)
    if uniq_warnings and not args.quiet:
        print("\nWarnings:")
        for w in uniq_warnings[:80]:
            print(f"  - {w}")
        if len(uniq_warnings) > 80:
            print(f"  ... {len(uniq_warnings) - 80} more warnings. See {combined_metadata_path}")


if __name__ == "__main__":
    main()

"""
python scripts/viz/table52_53_54_domain_adaptation.py \
  --benchmarking experiments/mapanything/benchmarking \
  --views 8,16,24,32 \
  --output experiments/mapanything/benchmarking/tables/table52_53_54
"""