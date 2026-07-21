#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Figure 6.6: cross-dataset FOV analysis on all UAV test sets.

This script collects per-scene benchmark results from all selected UAV test
sets, assigns each scene a fixed hFOV using a built-in mapping, pools the
scene/view values across datasets by hFOV, and plots metric-vs-hFOV curves.

Default use case (Chapter 6): compare MA-FT-A3D prior-aware inputs RGB/C/P/CP
on all UAV test datasets (UseGeo / Enrich / Urban / A3D-Real).

Expected benchmark layout:
  <benchmarking>/
    dense_8_view/<method_subdir>/<dataset>_per_scene_results.json
    dense_16_view/<method_subdir>/<dataset>_per_scene_results.json
    dense_24_view/<method_subdir>/<dataset>_per_scene_results.json
    dense_32_view/<method_subdir>/<dataset>_per_scene_results.json

Example:
  python scripts/viz/figure66_cross_dataset_fov_analysis.py \
    --benchmarking experiments/mapanything/benchmarking \
    --views 8,16,24,32 \
    --output experiments/mapanything/benchmarking/figures/figure66
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# -----------------------------------------------------------------------------
# Specs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class MetricSpec:
    alias: str
    display: str
    key: str
    ylabel: str
    default_precision: int = 3


@dataclass(frozen=True)
class SettingSpec:
    label: str
    subdirs: tuple[str, ...]


@dataclass
class SceneMetricValue:
    view: int
    setting: str
    method_subdir: str
    dataset_label: str
    dataset_key: str
    scene: str
    hfov: float
    metric_alias: str
    metric_key: str
    value: float
    n_values_in_scene: int


DEFAULT_DATASETS: "OrderedDict[str, DatasetSpec]" = OrderedDict(
    [
        (
            "A3D-Real",
            DatasetSpec("A3D-Real", ("A3DRealWAI", "A3D-Real", "A3DReal", "A3D_Real", "a3d_real")),
        ),
        (
            "Enrich",
            DatasetSpec("Enrich", ("ENRICHWAI", "Enrich-Aerial", "EnrichAerial", "Enrich", "enrich_aerial")),
        ),
        (
            "Urban",
            DatasetSpec("Urban", ("UrbanScene3DWAI", "UrbanScene3D", "US3D", "us3d", "Urban")),
        ),
        (
            "UseGeo",
            DatasetSpec("UseGeo", ("UseGeoWAI", "UseGeo", "usegeo")),
        ),
    ]
)

DEFAULT_SETTINGS: "OrderedDict[str, SettingSpec]" = OrderedDict(
    [
        (
            "RGB",
            SettingSpec(
                "RGB",
                (
                    "uav_mapa",
                ),
            ),
        ),
        (
            "C",
            SettingSpec(
                "C",
                (
                    "uav_mapa_csfm",
                ),
            ),
        ),
        (
            "P",
            SettingSpec(
                "P",
                (
                    "uav_mapa_psfm",
                ),
            ),
        ),
        (
            "CP",
            SettingSpec(
                "CP",
                (
                    "uav_mapa_mvs",
                ),
            ),
        ),
    ]
)

METRIC_ALIASES: "OrderedDict[str, MetricSpec]" = OrderedDict(
    [
        (
            "ray",
            MetricSpec(
                alias="ray",
                display="Ray Error",
                key="ray_dir_mean_angle_deg",
                ylabel="Ray Error (deg)",
                default_precision=2,
            ),
        ),
        (
            "depth_absrel",
            MetricSpec(
                alias="depth_absrel",
                display="Relative Depth Error",
                key="abs_depth_rel_scale_aligned",
                ylabel="Relative Depth Error",
                default_precision=3,
            ),
        ),
        (
            "pose_ate",
            MetricSpec(
                alias="pose_ate",
                display="Pose ATE",
                key="abs_pose_ate",
                ylabel="Pose ATE",
                default_precision=3,
            ),
        ),
        (
            "chamfer",
            MetricSpec(
                alias="chamfer",
                display="Chamfer-L1",
                key="abs_fused_pc_chamfer_l1",
                ylabel="Chamfer-L1",
                default_precision=3,
            ),
        ),
    ]
)

DEFAULT_METRICS = "ray,depth_absrel,pose_ate,chamfer"
DEFAULT_VIEWS = "8,16,24,32"

SETTING_COLORS = {
    "RGB": "#4d4d4d",
    "C": "#1f77b4",
    "P": "#ff7f0e",
    "CP": "#d62728",
}
SETTING_MARKERS = {
    "RGB": "o",
    "C": "s",
    "P": "^",
    "CP": "D",
}
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Built-in scene -> hfov mapping provided by the user.
DEFAULT_SCENE_HFOV_MAP_RAW = {
    "A3D-Real": {
        "nanfang_part0_ndir": 36,
        "nanfang_part0_oblique": 50,
        "nanfang_part1_ndir": 36,
        "nanfang_part1_oblique": 50,
        "xiaoxiang_part0_ndir": 36,
        "xiaoxiang_part0_oblique": 50,
        "xiaoxiang_part1_ndir": 36,
        "xiaoxiang_part1_oblique": 50,
        "xiaoxiang_part2_ndir": 36,
        "xiaoxiang_part2_oblique": 50,
        "xiaoxiang_part3_ndir": 36,
        "xiaoxiang_part3_oblique": 50,
        "yanghaitang_part0_ndir": 36,
        "yanghaitang_part0_oblique": 50,
        "yanghaitang_part1_ndir": 36,
        "yanghaitang_part1_oblique": 50,
    },
    "Enrich": {
        "aerial_ndiir2": 54,
        "aerial_ndir2": 54,
        "aerial_ndir": 54,
        "aerial_oblique": 29,
    },
    "Urban": {
        "artsci": 37,
        "artsci_ndir": 37,
        "artsci_oblique": 37,
        "polytech": 37,
        "polytech_ndir": 37,
        "polytech_oblique": 37,
    },
    "UseGeo": {
        "dataset1": 81,
        "dataset2": 81,
        "dataset3": 81,
    },
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def split_csv_like(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def safe_stem(s: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s)).strip("_")
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


def std_finite(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if is_finite_number(v)]
    if not vals:
        return float("nan")
    return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0


def discover_views(root: Path) -> list[int]:
    views: list[int] = []
    for p in root.glob("dense_*_view"):
        if p.is_dir():
            m = re.match(r"dense_(\d+)_view$", p.name)
            if m:
                views.append(int(m.group(1)))
    return sorted(set(views))


def parse_views(spec: str | None, root: Path) -> list[int]:
    if not spec or spec.strip().lower() in {"all", "auto"}:
        return discover_views(root)
    return sorted(set(int(x) for x in split_csv_like(spec)))


def parse_metrics(spec: str | None) -> list[MetricSpec]:
    items = split_csv_like(spec or DEFAULT_METRICS)
    out: list[MetricSpec] = []
    for item in items:
        if item in METRIC_ALIASES:
            out.append(METRIC_ALIASES[item])
        else:
            out.append(MetricSpec(alias=item, display=item, key=item, ylabel=item, default_precision=3))
    if not out:
        raise ValueError("No metrics selected")
    return out


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


def dataset_alias_norms(dataset: DatasetSpec) -> list[str]:
    out = [norm_name(dataset.label)]
    out.extend(norm_name(a) for a in dataset.aliases)
    return [x for x in OrderedDict.fromkeys(out) if x]


def strip_per_scene_suffix(path: Path) -> str:
    suffix = "_per_scene_results.json"
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    return path.stem


def match_dataset_stem(stem: str, dataset: DatasetSpec) -> bool:
    ns = norm_name(stem)
    aliases = dataset_alias_norms(dataset)
    if ns in aliases:
        return True
    return any(a and (a in ns or ns in a) for a in aliases)


def find_per_scene_json(method_dir: Path, dataset: DatasetSpec) -> tuple[Path | None, str | None, list[str]]:
    warnings: list[str] = []
    if not method_dir.is_dir():
        warnings.append(f"Missing method directory: {method_dir}")
        return None, None, warnings
    candidates = sorted(method_dir.glob("*_per_scene_results.json"))
    if not candidates:
        warnings.append(f"No *_per_scene_results.json found in {method_dir}")
        return None, None, warnings
    matched = [p for p in candidates if match_dataset_stem(strip_per_scene_suffix(p), dataset)]
    if len(matched) == 1:
        p = matched[0]
        return p, strip_per_scene_suffix(p), warnings
    if len(matched) > 1:
        aliases = set(dataset_alias_norms(dataset))
        exact = [p for p in matched if norm_name(strip_per_scene_suffix(p)) in aliases]
        p = exact[0] if exact else matched[0]
        warnings.append(f"Multiple per-scene JSONs match dataset {dataset.label!r} in {method_dir}; using {p.name}")
        return p, strip_per_scene_suffix(p), warnings
    if len(candidates) == 1:
        p = candidates[0]
        warnings.append(
            f"Dataset aliases {dataset.aliases!r} did not match {p.name}; using the only per-scene JSON in {method_dir}."
        )
        return p, strip_per_scene_suffix(p), warnings
    warnings.append(
        f"Could not find per-scene JSON for dataset={dataset.label!r} in {method_dir}. Available: {', '.join(p.name for p in candidates)}"
    )
    return None, None, warnings


def load_json_dict(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def aggregate_scene_metric(metric_values: object, stat: str) -> tuple[float, int]:
    if isinstance(metric_values, list):
        vals = [float(v) for v in metric_values if is_finite_number(v)]
    else:
        vals = [float(metric_values)] if is_finite_number(metric_values) else []
    if not vals:
        return float("nan"), 0
    if stat == "mean":
        return float(sum(vals) / len(vals)), len(vals)
    if stat == "median":
        return float(statistics.median(vals)), len(vals)
    if stat == "min":
        return float(min(vals)), len(vals)
    if stat == "max":
        return float(max(vals)), len(vals)
    raise ValueError(f"Unsupported --scene-stat: {stat}")


def build_scene_hfov_map(raw_map: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for dataset_label, scene_map in raw_map.items():
        dd: dict[str, float] = {}
        for scene, hfov in scene_map.items():
            dd[scene] = float(hfov)
            dd[norm_name(scene)] = float(hfov)
        out[dataset_label] = dd
        out[norm_name(dataset_label)] = dd
    return out


def lookup_scene_hfov(dataset_label: str, scene: str, scene_hfov_map: dict[str, dict[str, float]]) -> float | None:
    dataset_keys = [dataset_label, norm_name(dataset_label)]
    scene_norm = norm_name(scene)
    dataset_map: dict[str, float] | None = None
    for dk in dataset_keys:
        if dk in scene_hfov_map:
            dataset_map = scene_hfov_map[dk]
            break
    if not dataset_map:
        return None
    if scene in dataset_map:
        return dataset_map[scene]
    if scene_norm in dataset_map:
        return dataset_map[scene_norm]
    for key, hfov in dataset_map.items():
        if len(str(key)) < 5:
            continue
        kk = norm_name(key)
        if kk and (kk in scene_norm or scene_norm in kk):
            return hfov
    return None


# -----------------------------------------------------------------------------
# Data collection / aggregation
# -----------------------------------------------------------------------------


def collect_scene_values(
    root: Path,
    views: list[int],
    settings: "OrderedDict[str, SettingSpec]",
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: list[MetricSpec],
    scene_hfov_map: dict[str, dict[str, float]],
    scene_stat: str,
    strict: bool,
) -> tuple[list[SceneMetricValue], list[str]]:
    scene_values: list[SceneMetricValue] = []
    warnings: list[str] = []

    for view in views:
        view_dir = root / f"dense_{view}_view"
        if not view_dir.is_dir():
            msg = f"Missing view directory: {view_dir}"
            if strict:
                raise FileNotFoundError(msg)
            warnings.append(msg)
            continue

        for setting_label, setting in settings.items():
            chosen_subdir: str | None = None
            for subdir in setting.subdirs:
                method_dir = view_dir / subdir
                if method_dir.is_dir():
                    chosen_subdir = subdir
                    break
            if chosen_subdir is None:
                msg = f"Could not locate method directory for setting={setting_label!r} under {view_dir}. Tried {setting.subdirs}"
                if strict:
                    raise FileNotFoundError(msg)
                warnings.append(msg)
                continue

            method_dir = view_dir / chosen_subdir
            for dataset_label, dataset in datasets.items():
                json_path, dataset_key, ws = find_per_scene_json(method_dir, dataset)
                warnings.extend(ws)
                if json_path is None or dataset_key is None:
                    continue
                try:
                    obj = load_json_dict(json_path)
                except Exception as exc:
                    msg = f"Failed to read {json_path}: {exc}"
                    if strict:
                        raise
                    warnings.append(msg)
                    continue

                for scene, metric_obj in obj.items():
                    if not isinstance(metric_obj, dict):
                        warnings.append(f"Bad scene object in {json_path}: scene={scene!r}")
                        continue
                    hfov = lookup_scene_hfov(dataset_label, str(scene), scene_hfov_map)
                    if hfov is None:
                        warnings.append(
                            f"Could not map scene to hFOV: dataset={dataset_label}, scene={scene!r}, json={json_path}"
                        )
                        continue
                    for metric in metrics:
                        val, n = aggregate_scene_metric(metric_obj.get(metric.key, []), scene_stat)
                        if not is_finite_number(val):
                            continue
                        scene_values.append(
                            SceneMetricValue(
                                view=view,
                                setting=setting_label,
                                method_subdir=chosen_subdir,
                                dataset_label=dataset_label,
                                dataset_key=dataset_key,
                                scene=str(scene),
                                hfov=float(hfov),
                                metric_alias=metric.alias,
                                metric_key=metric.key,
                                value=float(val),
                                n_values_in_scene=n,
                            )
                        )
    return scene_values, warnings


def aggregate_hfov_means(scene_values: list[SceneMetricValue], min_count: int = 1) -> tuple[dict, list[dict]]:
    values: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    scene_sets: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    dataset_sets: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    view_sets: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    for x in scene_values:
        values[x.setting][x.metric_alias][x.hfov].append(x.value)
        scene_sets[x.setting][x.metric_alias][x.hfov].add((x.dataset_label, x.scene))
        dataset_sets[x.setting][x.metric_alias][x.hfov].add(x.dataset_label)
        view_sets[x.setting][x.metric_alias][x.hfov].add(x.view)

    agg: dict = OrderedDict()
    rows: list[dict] = []
    for setting_label in values.keys():
        agg[setting_label] = OrderedDict()
        for metric_alias in values[setting_label].keys():
            agg[setting_label][metric_alias] = OrderedDict()
            for hfov in sorted(values[setting_label][metric_alias].keys()):
                vals = [v for v in values[setting_label][metric_alias][hfov] if is_finite_number(v)]
                if len(vals) < min_count:
                    continue
                item = {
                    "mean": mean_finite(vals),
                    "std": std_finite(vals),
                    "n_values": len(vals),
                    "n_scenes": len(scene_sets[setting_label][metric_alias][hfov]),
                    "n_datasets": len(dataset_sets[setting_label][metric_alias][hfov]),
                    "datasets": sorted(dataset_sets[setting_label][metric_alias][hfov]),
                    "n_views": len(view_sets[setting_label][metric_alias][hfov]),
                    "views": sorted(view_sets[setting_label][metric_alias][hfov]),
                }
                agg[setting_label][metric_alias][hfov] = item
                rows.append(
                    {
                        "setting": setting_label,
                        "metric_alias": metric_alias,
                        "hfov": hfov,
                        "mean": item["mean"],
                        "std": item["std"],
                        "n_values": item["n_values"],
                        "n_scenes": item["n_scenes"],
                        "n_datasets": item["n_datasets"],
                        "datasets": ",".join(item["datasets"]),
                        "n_views": item["n_views"],
                        "views": ",".join(str(v) for v in item["views"]),
                    }
                )
    rows.sort(key=lambda r: (str(r["setting"]), str(r["metric_alias"]), float(r["hfov"])))
    return agg, rows


# -----------------------------------------------------------------------------
# Writers / plotting
# -----------------------------------------------------------------------------


def write_scene_values_csv(path: Path, rows: list[SceneMetricValue]) -> None:
    fields = [
        "view",
        "setting",
        "method_subdir",
        "dataset_label",
        "dataset_key",
        "scene",
        "hfov",
        "metric_alias",
        "metric_key",
        "value",
        "n_values_in_scene",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for x in rows:
            writer.writerow(
                {
                    "view": x.view,
                    "setting": x.setting,
                    "method_subdir": x.method_subdir,
                    "dataset_label": x.dataset_label,
                    "dataset_key": x.dataset_key,
                    "scene": x.scene,
                    "hfov": x.hfov,
                    "metric_alias": x.metric_alias,
                    "metric_key": x.metric_key,
                    "value": x.value,
                    "n_values_in_scene": x.n_values_in_scene,
                }
            )


def write_dict_rows_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            rr = dict(row)
            for k, v in list(rr.items()):
                if isinstance(v, float) and not math.isfinite(v):
                    rr[k] = ""
            writer.writerow(rr)


def apply_metric_specific_yscale(ax, metric_alias: str) -> None:
    configs = {
        "ray": {
            "linthresh": 2.0,
            "linscale": 1.6,
            "ticks": [0, 0.25, 0.5, 1, 1.5, 2, 5, 10, 20, 50, 100],
        },
        "pose_ate": {
            "linthresh": 20.0,
            "linscale": 1.6,
            "ticks": [0, 2, 5, 10, 15, 20, 50, 100, 200, 500, 1000],
        },
    }
    if metric_alias not in configs:
        return
    cfg = configs[metric_alias]
    ymin, ymax = ax.get_ylim()
    if ymax <= 0:
        return
    all_ticks = cfg["ticks"]
    ticks = [t for t in all_ticks if t <= ymax * 1.05]
    larger_ticks = [t for t in all_ticks if t > ymax * 1.05]
    if larger_ticks:
        ticks.append(larger_ticks[0])
    if len(ticks) < 2:
        ticks = all_ticks[:4]
    ax.set_yscale("symlog", linthresh=cfg["linthresh"], linscale=cfg["linscale"], base=10)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_ylim(bottom=0, top=max(ticks[-1], ymax))


def plot_figure66(
    hfov_data: dict,
    settings: "OrderedDict[str, SettingSpec]",
    metrics: list[MetricSpec],
    title: str | None,
    show_std: bool,
    marker_size: float,
    linewidth: float,
    legend_ncol: int,
    log_y: bool,
    adaptive_y: bool,
) -> plt.Figure:
    n_metrics = len(metrics)
    ncols = 2 if n_metrics > 1 else 1
    nrows = int(math.ceil(n_metrics / ncols))
    figsize = (5.0 * max(1, ncols), 3.9 * max(1, nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = list(axes.flat)

    panel_idx = 0
    for metric in metrics:
        ax = axes_flat[panel_idx]
        for setting_label in settings.keys():
            bins = hfov_data.get(setting_label, {}).get(metric.alias, {})
            if not bins:
                continue
            xs = sorted(float(h) for h in bins.keys())
            ys = [float(bins[h]["mean"]) for h in xs]
            yerr = [float(bins[h]["std"]) if is_finite_number(bins[h].get("std")) else 0.0 for h in xs]
            color = SETTING_COLORS.get(setting_label, None)
            marker = SETTING_MARKERS.get(setting_label, "o")
            if show_std:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerr,
                    marker=marker,
                    color=color,
                    linewidth=linewidth,
                    markersize=marker_size,
                    capsize=2.5,
                    label=setting_label,
                )
            else:
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    color=color,
                    linewidth=linewidth,
                    markersize=marker_size,
                    label=setting_label,
                )
        letter = PANEL_LETTERS[panel_idx] if panel_idx < len(PANEL_LETTERS) else "?"
        ax.set_title(f"({letter}) {metric.display} vs. hFOV", fontsize=11)
        ax.set_xlabel("hFOV (deg)")
        ax.set_ylabel(metric.ylabel)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        if log_y:
            ax.set_yscale("log")
        elif adaptive_y:
            apply_metric_specific_yscale(ax, metric.alias)
        ticks = sorted(
            {
                float(h)
                for setting_label in settings.keys()
                for h in hfov_data.get(setting_label, {}).get(metric.alias, {}).keys()
                if is_finite_number(h)
            }
        )
        if ticks:
            ax.set_xticks(ticks)
            if len(ticks) == 1:
                ax.set_xlim(ticks[0] - 4.0, ticks[0] + 4.0)
            else:
                ax.set_xlim(min(ticks) - 3.0, max(ticks) + 3.0)
        panel_idx += 1

    for ax in axes_flat[panel_idx:]:
        ax.axis("off")

    handles: list[Line2D] = []
    labels: list[str] = []
    for setting_label in settings.keys():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=SETTING_MARKERS.get(setting_label, "o"),
                color=SETTING_COLORS.get(setting_label, None),
                linestyle="-",
                linewidth=linewidth,
                markersize=marker_size,
            )
        )
        labels.append(setting_label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, legend_ncol), frameon=False, bbox_to_anchor=(0.5, 1.02))
    if title:
        fig.suptitle(title, y=1.05, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot Figure 6.6: metric-vs-hFOV curves pooled across all UAV test datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmarking", type=Path, default=Path("experiments/mapanything/benchmarking"), help="Benchmarking root containing dense_*_view folders.")
    p.add_argument("--output", type=Path, default=None, help="Output directory. Default: <benchmarking>/figures/figure66")
    p.add_argument("--views", type=str, default=DEFAULT_VIEWS, help="Comma-separated views, e.g. 8,16,24,32, or all.")
    p.add_argument("--settings", nargs="*", default=None, help="Curves as Label=subdir1|subdir2. Defaults to RGB/C/P/CP for MA-FT-A3D.")
    p.add_argument("--datasets", nargs="*", default=None, help="Datasets as Label=alias1|alias2. Defaults to A3D-Real/Enrich/Urban/UseGeo.")
    p.add_argument("--metrics", type=str, default=DEFAULT_METRICS, help="Metrics to plot.")
    p.add_argument("--scene-stat", choices=["mean", "median", "min", "max"], default="mean", help="How to aggregate each scene's metric list before pooling across hFOV/views.")
    p.add_argument("--min-count", type=int, default=1, help="Minimum number of values required for a pooled hFOV point.")
    p.add_argument("--show-std", action="store_true", help="Plot error bars using std across pooled scene/view values.")
    p.add_argument("--log-y", action="store_true", help="Use log scale for all y axes.")
    p.add_argument("--no-adaptive-y", action="store_true", help="Disable metric-specific compressed y-axis for Ray / Pose ATE.")
    p.add_argument("--dpi", type=int, default=300, help="PNG DPI.")
    p.add_argument("--save-svg", action="store_true", help="Also save SVG.")
    p.add_argument("--title", type=str, default="Figure 6.6. Cross-dataset FOV analysis on all UAV test sets", help="Figure title. Use empty string to disable.")
    p.add_argument("--figure-stem", type=str, default="figure66_cross_dataset_fov_analysis", help="Output figure stem.")
    p.add_argument("--marker-size", type=float, default=5.0, help="Marker size.")
    p.add_argument("--linewidth", type=float, default=1.8, help="Line width.")
    p.add_argument("--legend-ncol", type=int, default=4, help="Number of legend columns.")
    p.add_argument("--strict", action="store_true", help="Raise on missing files/scenes instead of warning.")
    p.add_argument("--quiet", action="store_true", help="Suppress warning printout.")
    return p


def dedupe_warnings(warnings: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    root = Path(args.benchmarking)
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmarking root not found: {root}")

    output_dir = Path(args.output) if args.output is not None else (root / "figures" / "figure66")
    output_dir.mkdir(parents=True, exist_ok=True)

    views = parse_views(args.views, root)
    if not views:
        raise ValueError(f"No views selected/found under {root}")

    settings = parse_setting_specs(args.settings)
    datasets = parse_dataset_specs(args.datasets)
    metrics = parse_metrics(args.metrics)
    scene_hfov_map = build_scene_hfov_map(DEFAULT_SCENE_HFOV_MAP_RAW)

    scene_values, warnings = collect_scene_values(
        root=root,
        views=views,
        settings=settings,
        datasets=datasets,
        metrics=metrics,
        scene_hfov_map=scene_hfov_map,
        scene_stat=args.scene_stat,
        strict=bool(args.strict),
    )
    if not scene_values:
        raise RuntimeError(
            "No scene values collected. Check --benchmarking, settings subdirs, dataset aliases, scene names, and metric keys."
        )

    hfov_data, hfov_rows = aggregate_hfov_means(scene_values, min_count=int(args.min_count))

    title = args.title if str(args.title).strip() else None
    fig = plot_figure66(
        hfov_data=hfov_data,
        settings=settings,
        metrics=metrics,
        title=title,
        show_std=bool(args.show_std),
        marker_size=float(args.marker_size),
        linewidth=float(args.linewidth),
        legend_ncol=int(args.legend_ncol),
        log_y=bool(args.log_y),
        adaptive_y=not bool(args.no_adaptive_y),
    )

    view_suffix = "all" if str(args.views).strip().lower() in {"all", "auto"} else "_".join(str(v) for v in views)
    setting_suffix = "_".join(safe_stem(k) for k in settings.keys())
    metric_suffix = "_".join(safe_stem(m.alias) for m in metrics)
    stem = safe_stem(f"{args.figure_stem}_{setting_suffix}_{metric_suffix}_views_{view_suffix}")

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    svg_path = output_dir / f"{stem}.svg"
    hfov_csv_path = output_dir / f"{stem}_hfov_means.csv"
    scene_csv_path = output_dir / f"{stem}_scene_values.csv"
    metadata_path = output_dir / f"{stem}_metadata.json"

    fig.savefig(png_path, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    if args.save_svg:
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    write_dict_rows_csv(
        hfov_csv_path,
        hfov_rows,
        fieldnames=[
            "setting",
            "metric_alias",
            "hfov",
            "mean",
            "std",
            "n_values",
            "n_scenes",
            "n_datasets",
            "datasets",
            "n_views",
            "views",
        ],
    )
    write_scene_values_csv(scene_csv_path, scene_values)

    metadata = {
        "figure_id": "6.6",
        "title": title,
        "benchmarking_root": str(root),
        "output_dir": str(output_dir),
        "views": views,
        "settings": {k: list(v.subdirs) for k, v in settings.items()},
        "datasets": {k: list(v.aliases) for k, v in datasets.items()},
        "metrics": [{"alias": m.alias, "display": m.display, "key": m.key, "ylabel": m.ylabel} for m in metrics],
        "scene_stat": args.scene_stat,
        "min_count": args.min_count,
        "scene_hfov_map": DEFAULT_SCENE_HFOV_MAP_RAW,
        "n_scene_metric_values": len(scene_values),
        "n_hfov_rows": len(hfov_rows),
        "outputs": {
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
            "figure_svg": str(svg_path) if args.save_svg else None,
            "hfov_means_csv": str(hfov_csv_path),
            "scene_values_csv": str(scene_csv_path),
        },
        "warnings": dedupe_warnings(warnings),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote Figure 6.6 PNG: {png_path}")
    print(f"Wrote Figure 6.6 PDF: {pdf_path}")
    if args.save_svg:
        print(f"Wrote Figure 6.6 SVG: {svg_path}")
    print(f"Wrote pooled hFOV means: {hfov_csv_path}")
    print(f"Wrote scene-level values: {scene_csv_path}")
    print(f"Wrote metadata: {metadata_path}")

    uniq_warnings = dedupe_warnings(warnings)
    if uniq_warnings and not args.quiet:
        print("\nWarnings:")
        for w in uniq_warnings[:80]:
            print(f"  - {w}")
        if len(uniq_warnings) > 80:
            print(f"  ... {len(uniq_warnings) - 80} more warnings. See {metadata_path}")


if __name__ == "__main__":
    main()

"""
python scripts/viz/figure66_cross_dataset_fov_analysis.py \
  --benchmarking experiments/mapanything/benchmarking \
  --views 8,16,24,32 \
  --output experiments/mapanything/benchmarking/figures/figure66
"""