#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build oblique-vs-nadir forward reconstruction tables from per-scene benchmark outputs.

The script reads *_per_scene_results.json under dense_*_view/<method>/, assigns
scenes to oblique / ndir according to a built-in cross-dataset split map, and
reports metric differences between the two view types.

Default outputs:
  1) Overall compact table averaged over datasets:
       Setting x {Nadir, Oblique, Gap%}
  2) Dataset-wise long table:
       Dataset x Setting x {Nadir, Oblique, Gap%}
  3) Long-form CSV details for auditing scene/view contributions.

Gap% is defined as:
    (Oblique - Nadir) / Nadir * 100

All default metrics are lower-is-better. Therefore positive Gap% means the
oblique split is worse than the nadir split, while negative Gap% means the
oblique split is better.

Expected benchmark layout:
  <benchmarking>/
    dense_8_view/<method_subdir>/<dataset>_per_scene_results.json
    dense_16_view/<method_subdir>/<dataset>_per_scene_results.json
    dense_24_view/<method_subdir>/<dataset>_per_scene_results.json
    dense_32_view/<method_subdir>/<dataset>_per_scene_results.json

Example:
  python scripts/viz/table_oblique_ndir_forward_reconstruction.py \
    --benchmarking experiments/mapanything/benchmarking \
    --views 8,16,24,32 \
    --output experiments/mapanything/benchmarking/tables/oblique_ndir
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


# -----------------------------------------------------------------------------
# Specs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class SettingSpec:
    label: str
    subdirs: tuple[str, ...]


@dataclass(frozen=True)
class MetricSpec:
    alias: str
    display: str
    key: str
    precision: int = 3


@dataclass
class SceneMetricValue:
    view: int
    setting: str
    method_subdir: str
    dataset_label: str
    dataset_key: str
    split: str
    scene: str
    metric_alias: str
    metric_display: str
    metric_key: str
    value: float
    n_values_in_scene: int


DEFAULT_DATASETS: "OrderedDict[str, DatasetSpec]" = OrderedDict(
    [
        ("A3D-Real", DatasetSpec("A3D-Real", ("A3DRealWAI", "A3D-Real", "A3DReal", "A3D_Real", "a3d_real"))),
        ("Enrich", DatasetSpec("Enrich", ("ENRICHWAI", "Enrich-Aerial", "EnrichAerial", "Enrich", "enrich_aerial"))),
        ("Urban", DatasetSpec("Urban", ("UrbanScene3DWAI", "UrbanScene3D", "US3D", "us3d", "Urban"))),
        ("UseGeo", DatasetSpec("UseGeo", ("UseGeoWAI", "UseGeo", "usegeo"))),
    ]
)

# Common feed-forward reconstruction method directories. Missing methods are
# skipped with warnings unless --strict is enabled. Override with --settings.
DEFAULT_SETTINGS: "OrderedDict[str, SettingSpec]" = OrderedDict(
    [
        ("MapAnything", SettingSpec("MapAnything", ("mapa", "mapa_24v", "uav_mapa", "mapanything"))),
        ("Pi3X", SettingSpec("Pi3X", ("pi3x",))),
        ("DA3", SettingSpec("DA3", ("da3",))),
        ("Hunyuan", SettingSpec("Hunyuan", ("hunyuan",))),
        ("VGGT", SettingSpec("VGGT", ("vggt", ))),
        ("Pi3", SettingSpec("Pi3", ("pi3", ))),
    ]
)

METRIC_ALIASES: "OrderedDict[str, MetricSpec]" = OrderedDict(
    [
        ("reldepth", MetricSpec("reldepth", "RelDepth", "abs_depth_rel_scale_aligned", 3)),
        ("ray", MetricSpec("ray", "Ray Error", "ray_dir_mean_angle_deg", 2)),
        ("chamfer", MetricSpec("chamfer", "Chamfer-L1", "abs_fused_pc_chamfer_l1", 3)),
        ("pose_ate", MetricSpec("pose_ate", "Pose ATE", "abs_pose_ate", 3)),
    ]
)

DEFAULT_METRICS = "reldepth,ray,chamfer,pose_ate"
DEFAULT_VIEWS = "8,16,24,32"
SPLIT_ORDER = ("ndir", "oblique", "gap_pct")
SPLIT_DISPLAY = {"ndir": "Nadir", "oblique": "Oblique", "gap_pct": "Gap%"}

# User-provided scene split mapping. The typo aerial_ndiir2 is kept and an alias
# aerial_ndir2 is added for robustness.
DEFAULT_SCENE_SPLIT_MAP_RAW: dict[str, dict[str, set[str]]] = {
    "A3D-Real": {
        "oblique": {
            "nanfang_part0_oblique",
            "nanfang_part1_oblique",
            "yanghaitang_part0_oblique",
            "yanghaitang_part1_oblique",
            "xiaoxiang_part0_oblique",
            "xiaoxiang_part1_oblique",
            "xiaoxiang_part2_oblique",
            "xiaoxiang_part3_oblique",
        },
        "ndir": {
            "nanfang_part0_ndir",
            "nanfang_part1_ndir",
            "yanghaitang_part0_ndir",
            "yanghaitang_part1_ndir",
            "xiaoxiang_part0_ndir",
            "xiaoxiang_part1_ndir",
            "xiaoxiang_part2_ndir",
            "xiaoxiang_part3_ndir",
        },
    },
    "Enrich": {
        "oblique": {"aerial_oblique"},
        "ndir": {"aerial_ndiir2", "aerial_ndir2", "aerial_ndir"},
    },
    "Urban": {
        "oblique": {"artsci_oblique", "polytech_oblique"},
        "ndir": {"artsci_ndir", "polytech_ndir"},
    },
    "UseGeo": {
        "oblique": set(),
        "ndir": {"dataset1", "dataset2", "dataset3"},
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


def std_finite(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if is_finite_number(v)]
    if not vals:
        return float("nan")
    return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0


def fmt_number(value: float, precision: int, missing: str = "") -> str:
    if not is_finite_number(value):
        return missing
    return f"{float(value):.{precision}f}"


def fmt_gap(value: float, precision: int = 1, missing: str = "") -> str:
    if not is_finite_number(value):
        return missing
    return f"{float(value):.{precision}f}%"


def discover_views(root: Path) -> list[int]:
    views: list[int] = []
    for p in root.glob("dense_*_view"):
        if p.is_dir():
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
        out[label] = SettingSpec(label, subdirs)
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
        out[label] = DatasetSpec(label, aliases)
    return out


def parse_metrics(spec: str | None) -> "OrderedDict[str, MetricSpec]":
    out: "OrderedDict[str, MetricSpec]" = OrderedDict()
    for item in split_csv_like(spec or DEFAULT_METRICS):
        if item in METRIC_ALIASES:
            out[item] = METRIC_ALIASES[item]
        else:
            out[item] = MetricSpec(item, item, item, 3)
    if not out:
        raise ValueError("No metrics selected")
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
        # warnings.append(f"Dataset aliases {dataset.aliases!r} did not match {p.name}; using the only per-scene JSON in {method_dir}.")
        return p, strip_per_scene_suffix(p), warnings
    warnings.append(f"Could not find per-scene JSON for dataset={dataset.label!r} in {method_dir}. Available: {', '.join(p.name for p in candidates)}")
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


def build_scene_split_map(raw_map: dict[str, dict[str, set[str]]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for dataset_label, split_map in raw_map.items():
        dd: dict[str, str] = {}
        for split, scenes in split_map.items():
            for scene in scenes:
                dd[scene] = split
                dd[norm_name(scene)] = split
        out[dataset_label] = dd
        out[norm_name(dataset_label)] = dd
    return out


def lookup_scene_split(dataset_label: str, scene: str, scene_split_map: dict[str, dict[str, str]]) -> str | None:
    dataset_map = scene_split_map.get(dataset_label) or scene_split_map.get(norm_name(dataset_label))
    if not dataset_map:
        return None
    scene_norm = norm_name(scene)
    if scene in dataset_map:
        return dataset_map[scene]
    if scene_norm in dataset_map:
        return dataset_map[scene_norm]
    for key, split in dataset_map.items():
        kk = norm_name(key)
        if len(kk) < 4:
            continue
        if kk in scene_norm or scene_norm in kk:
            return split
    return None


def relative_gap_pct(ndir_value: float, oblique_value: float) -> float:
    if not is_finite_number(ndir_value) or not is_finite_number(oblique_value):
        return float("nan")
    ndir_value = float(ndir_value)
    if math.isclose(ndir_value, 0.0, abs_tol=1e-12):
        return float("nan")
    return (float(oblique_value) - ndir_value) / ndir_value * 100.0


def dedupe_warnings(warnings: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


# -----------------------------------------------------------------------------
# Collection / aggregation
# -----------------------------------------------------------------------------


def collect_scene_values(
    root: Path,
    views: list[int],
    settings: "OrderedDict[str, SettingSpec]",
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
    scene_split_map: dict[str, dict[str, str]],
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
            method_dir = None
            used_subdir = None
            for subdir in setting.subdirs:
                candidate = view_dir / subdir
                if candidate.is_dir():
                    method_dir = candidate
                    used_subdir = subdir
                    break
            if method_dir is None or used_subdir is None:
                msg = f"Could not locate method directory for setting={setting_label!r}, view={view}. Tried {setting.subdirs} under {view_dir}"
                if strict:
                    raise FileNotFoundError(msg)
                warnings.append(msg)
                continue

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
                    split = lookup_scene_split(dataset_label, str(scene), scene_split_map)
                    if split is None:
                        # warnings.append(f"Scene not assigned to oblique/ndir split: dataset={dataset_label}, scene={scene!r}, json={json_path}")
                        continue
                    for metric_alias, metric in metrics.items():
                        value, n = aggregate_scene_metric(metric_obj.get(metric.key, []), scene_stat)
                        if not is_finite_number(value):
                            continue
                        scene_values.append(
                            SceneMetricValue(
                                view=view,
                                setting=setting_label,
                                method_subdir=used_subdir,
                                dataset_label=dataset_label,
                                dataset_key=dataset_key,
                                split=split,
                                scene=str(scene),
                                metric_alias=metric_alias,
                                metric_display=metric.display,
                                metric_key=metric.key,
                                value=float(value),
                                n_values_in_scene=n,
                            )
                        )
    return scene_values, warnings


def aggregate_dataset_split_values(scene_values: list[SceneMetricValue]) -> tuple[dict, list[dict]]:
    buckets: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    scene_sets: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
    view_sets: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
    subdirs: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))

    for x in scene_values:
        buckets[x.setting][x.dataset_label][x.split][x.metric_alias].append(x.value)
        scene_sets[x.setting][x.dataset_label][x.split][x.metric_alias].add(x.scene)
        view_sets[x.setting][x.dataset_label][x.split][x.metric_alias].add(x.view)
        subdirs[x.setting][x.dataset_label][x.split][x.metric_alias].add(x.method_subdir)

    values: dict = OrderedDict()
    rows: list[dict] = []
    for setting in sorted(buckets.keys()):
        values[setting] = OrderedDict()
        for dataset in sorted(buckets[setting].keys()):
            values[setting][dataset] = OrderedDict()
            for split in ["ndir", "oblique"]:
                values[setting][dataset][split] = OrderedDict()
                metric_map = buckets[setting][dataset].get(split, {})
                for metric_alias in sorted(metric_map.keys()):
                    vals = metric_map[metric_alias]
                    item = {
                        "mean": mean_finite(vals),
                        "std": std_finite(vals),
                        "n_values": len([v for v in vals if is_finite_number(v)]),
                        "n_scenes": len(scene_sets[setting][dataset][split][metric_alias]),
                        "n_views": len(view_sets[setting][dataset][split][metric_alias]),
                        "views": sorted(view_sets[setting][dataset][split][metric_alias]),
                        "method_subdirs": sorted(subdirs[setting][dataset][split][metric_alias]),
                    }
                    values[setting][dataset][split][metric_alias] = item
                    rows.append(
                        {
                            "setting": setting,
                            "dataset": dataset,
                            "split": split,
                            "metric_alias": metric_alias,
                            "mean": item["mean"],
                            "std": item["std"],
                            "n_values": item["n_values"],
                            "n_scenes": item["n_scenes"],
                            "n_views": item["n_views"],
                            "views": ",".join(str(v) for v in item["views"]),
                            "method_subdirs": "|".join(item["method_subdirs"]),
                        }
                    )
    return values, rows


def add_average_dataset(
    values: dict,
    datasets: "OrderedDict[str, DatasetSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
    average_label: str = "Average",
) -> dict:
    """Average each split/metric over datasets, giving every dataset equal weight."""
    out = values
    for setting in list(out.keys()):
        out[setting][average_label] = OrderedDict()
        for split in ["ndir", "oblique"]:
            out[setting][average_label][split] = OrderedDict()
            for metric_alias in metrics.keys():
                vals = []
                stds = []
                n_values = 0
                n_scenes = 0
                n_views_set: set[int] = set()
                used_subdirs: set[str] = set()
                for dataset_label in datasets.keys():
                    item = out[setting].get(dataset_label, {}).get(split, {}).get(metric_alias)
                    if item is None:
                        continue
                    vals.append(item.get("mean", float("nan")))
                    stds.append(item.get("std", float("nan")))
                    n_values += int(item.get("n_values", 0) or 0)
                    n_scenes += int(item.get("n_scenes", 0) or 0)
                    n_views_set.update(int(v) for v in item.get("views", []) if str(v).isdigit() or isinstance(v, int))
                    used_subdirs.update(str(s) for s in item.get("method_subdirs", []))
                out[setting][average_label][split][metric_alias] = {
                    "mean": mean_finite(vals),
                    "std": mean_finite(stds),
                    "n_values": n_values,
                    "n_scenes": n_scenes,
                    "n_views": len(n_views_set),
                    "views": sorted(n_views_set),
                    "method_subdirs": sorted(used_subdirs),
                }
    return out


def compute_gap_rows(
    values: dict,
    dataset_labels: list[str],
    metrics: "OrderedDict[str, MetricSpec]",
) -> list[dict]:
    rows: list[dict] = []
    for setting in values.keys():
        for dataset in dataset_labels:
            for metric_alias in metrics.keys():
                ndir = values[setting].get(dataset, {}).get("ndir", {}).get(metric_alias, {}).get("mean", float("nan"))
                oblique = values[setting].get(dataset, {}).get("oblique", {}).get(metric_alias, {}).get("mean", float("nan"))
                rows.append(
                    {
                        "setting": setting,
                        "dataset": dataset,
                        "metric_alias": metric_alias,
                        "ndir_mean": ndir,
                        "oblique_mean": oblique,
                        "oblique_minus_ndir": float(oblique) - float(ndir) if is_finite_number(oblique) and is_finite_number(ndir) else float("nan"),
                        "gap_pct": relative_gap_pct(ndir, oblique),
                    }
                )
    return rows


# -----------------------------------------------------------------------------
# Table builders / writers
# -----------------------------------------------------------------------------


def best_by_metric_split(values: dict, dataset: str, split: str, metric_alias: str) -> set[str]:
    label_to_value = {
        setting: values[setting].get(dataset, {}).get(split, {}).get(metric_alias, {}).get("mean", float("nan"))
        for setting in values.keys()
    }
    finite = [v for v in label_to_value.values() if is_finite_number(v)]
    if not finite:
        return set()
    best = min(finite)
    return {s for s, v in label_to_value.items() if is_finite_number(v) and math.isclose(float(v), best, rel_tol=1e-12, abs_tol=1e-12)}


def maybe_bold(s: str, bold: bool) -> str:
    return f"**{s}**" if bold and s else s


def build_overall_markdown(
    values: dict,
    metrics: "OrderedDict[str, MetricSpec]",
    average_label: str,
    missing: str,
    bold_best: bool,
    include_caption: bool,
) -> str:
    headers = ["Setting", "Split"] + [f"{m.display} ↓" for m in metrics.values()]
    aligns = [":---", ":---"] + ["---:" for _ in metrics]
    lines: list[str] = []
    if include_caption:
        lines.extend([
            "**Oblique-vs-nadir forward reconstruction difference averaged over all test datasets.**",
            "",
        ])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(aligns) + " |")
    for setting in values.keys():
        for split in SPLIT_ORDER:
            row = [setting if split == "ndir" else "", SPLIT_DISPLAY[split]]
            for metric_alias, metric in metrics.items():
                if split == "gap_pct":
                    ndir = values[setting].get(average_label, {}).get("ndir", {}).get(metric_alias, {}).get("mean", float("nan"))
                    oblique = values[setting].get(average_label, {}).get("oblique", {}).get(metric_alias, {}).get("mean", float("nan"))
                    cell = fmt_gap(relative_gap_pct(ndir, oblique), 1, missing)
                else:
                    val = values[setting].get(average_label, {}).get(split, {}).get(metric_alias, {}).get("mean", float("nan"))
                    cell = fmt_number(val, metric.precision, missing)
                    best = setting in best_by_metric_split(values, average_label, split, metric_alias)
                    cell = maybe_bold(cell, bold_best and best)
                row.append(cell)
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_dataset_markdown(
    values: dict,
    dataset_labels: list[str],
    metrics: "OrderedDict[str, MetricSpec]",
    missing: str,
    bold_best: bool,
    include_caption: bool,
) -> str:
    headers = ["Dataset", "Setting", "Split"] + [f"{m.display} ↓" for m in metrics.values()]
    aligns = [":---", ":---", ":---"] + ["---:" for _ in metrics]
    lines: list[str] = []
    if include_caption:
        lines.extend([
            "**Dataset-wise oblique-vs-nadir forward reconstruction difference.**",
            "",
        ])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(aligns) + " |")
    for dataset in dataset_labels:
        first_dataset_row = True
        for setting in values.keys():
            for split in SPLIT_ORDER:
                row = [dataset if first_dataset_row else "", setting if split == "ndir" else "", SPLIT_DISPLAY[split]]
                first_dataset_row = False
                for metric_alias, metric in metrics.items():
                    if split == "gap_pct":
                        ndir = values[setting].get(dataset, {}).get("ndir", {}).get(metric_alias, {}).get("mean", float("nan"))
                        oblique = values[setting].get(dataset, {}).get("oblique", {}).get(metric_alias, {}).get("mean", float("nan"))
                        cell = fmt_gap(relative_gap_pct(ndir, oblique), 1, missing)
                    else:
                        val = values[setting].get(dataset, {}).get(split, {}).get(metric_alias, {}).get("mean", float("nan"))
                        cell = fmt_number(val, metric.precision, missing)
                        best = setting in best_by_metric_split(values, dataset, split, metric_alias)
                        cell = maybe_bold(cell, bold_best and best)
                    row.append(cell)
                lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def build_overall_latex(values: dict, metrics: "OrderedDict[str, MetricSpec]", average_label: str, missing: str) -> str:
    ncols = 2 + len(metrics)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Oblique-vs-nadir forward reconstruction difference averaged over all test datasets.}",
        "\\label{tab:oblique_nadir_forward_reconstruction}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{ll" + "r" * len(metrics) + "}",
        "\\toprule",
    ]
    header = ["Setting", "Split"] + [f"{m.display} $\\downarrow$" for m in metrics.values()]
    lines.append(" & ".join(tex_escape(h) if "$" not in h else h for h in header) + " " + r"\\")
    lines.append("\\midrule")
    for setting in values.keys():
        for split in SPLIT_ORDER:
            row = [tex_escape(setting) if split == "ndir" else "", tex_escape(SPLIT_DISPLAY[split])]
            for metric_alias, metric in metrics.items():
                if split == "gap_pct":
                    ndir = values[setting].get(average_label, {}).get("ndir", {}).get(metric_alias, {}).get("mean", float("nan"))
                    oblique = values[setting].get(average_label, {}).get("oblique", {}).get(metric_alias, {}).get("mean", float("nan"))
                    cell = fmt_gap(relative_gap_pct(ndir, oblique), 1, missing)
                else:
                    val = values[setting].get(average_label, {}).get(split, {}).get(metric_alias, {}).get("mean", float("nan"))
                    cell = fmt_number(val, metric.precision, missing)
                row.append(cell)
            lines.append(" & ".join(row) + " " + r"\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(lines)


def write_summary_csv(path: Path, values: dict, dataset_labels: list[str], metrics: "OrderedDict[str, MetricSpec]") -> None:
    fields = ["dataset", "setting", "split"]
    for metric_alias in metrics.keys():
        fields.append(metric_alias)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset in dataset_labels:
            for setting in values.keys():
                for split in SPLIT_ORDER:
                    row = {"dataset": dataset, "setting": setting, "split": split}
                    for metric_alias in metrics.keys():
                        if split == "gap_pct":
                            ndir = values[setting].get(dataset, {}).get("ndir", {}).get(metric_alias, {}).get("mean", float("nan"))
                            oblique = values[setting].get(dataset, {}).get("oblique", {}).get(metric_alias, {}).get("mean", float("nan"))
                            row[metric_alias] = relative_gap_pct(ndir, oblique)
                        else:
                            row[metric_alias] = values[setting].get(dataset, {}).get(split, {}).get(metric_alias, {}).get("mean", float("nan"))
                    writer.writerow(row)


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rr = dict(row)
            for k, v in list(rr.items()):
                if isinstance(v, float) and not math.isfinite(v):
                    rr[k] = ""
            writer.writerow(rr)


def write_scene_values_csv(path: Path, rows: list[SceneMetricValue]) -> None:
    fields = [
        "view",
        "setting",
        "method_subdir",
        "dataset_label",
        "dataset_key",
        "split",
        "scene",
        "metric_alias",
        "metric_display",
        "metric_key",
        "value",
        "n_values_in_scene",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for x in rows:
            writer.writerow({k: getattr(x, k) for k in fields})


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build oblique-vs-nadir forward reconstruction tables from per-scene benchmark outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmarking", type=Path, default=Path("experiments/mapanything/benchmarking"))
    p.add_argument("--output", type=Path, default=None, help="Output directory. Default: <benchmarking>/tables/oblique_ndir")
    p.add_argument("--views", type=str, default=DEFAULT_VIEWS, help="Comma-separated views or 'all'.")
    p.add_argument("--settings", nargs="*", default=None, help="Rows as Label=subdir1|subdir2. Defaults to common feed-forward methods.")
    p.add_argument("--datasets", nargs="*", default=None, help="Datasets as Label=alias1|alias2. Defaults to A3D-Real/Enrich/Urban/UseGeo.")
    p.add_argument("--metrics", type=str, default=DEFAULT_METRICS, help="Metrics to include.")
    p.add_argument("--scene-stat", choices=["mean", "median", "min", "max"], default="mean", help="How to aggregate each scene's metric list before pooling.")
    p.add_argument("--missing", type=str, default="")
    p.add_argument("--table-stem", type=str, default="table_oblique_ndir_forward_reconstruction")
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
    output_dir = Path(args.output) if args.output is not None else root / "tables" / "oblique_ndir"
    output_dir.mkdir(parents=True, exist_ok=True)

    views = parse_views(args.views, root)
    if not views:
        raise ValueError(f"No views selected/found under {root}")
    settings = parse_setting_specs(args.settings)
    datasets = parse_dataset_specs(args.datasets)
    metrics = parse_metrics(args.metrics)
    scene_split_map = build_scene_split_map(DEFAULT_SCENE_SPLIT_MAP_RAW)

    scene_values, warnings = collect_scene_values(
        root=root,
        views=views,
        settings=settings,
        datasets=datasets,
        metrics=metrics,
        scene_split_map=scene_split_map,
        scene_stat=args.scene_stat,
        strict=bool(args.strict),
    )
    if not scene_values:
        raise RuntimeError(
            "No scene values collected. Check --benchmarking, --settings, --datasets, scene split names, and metric keys."
        )

    values, dataset_split_rows = aggregate_dataset_split_values(scene_values)
    values = add_average_dataset(values, datasets=datasets, metrics=metrics, average_label="Average")
    dataset_labels = list(datasets.keys()) + ["Average"]
    gap_rows = compute_gap_rows(values, dataset_labels=dataset_labels, metrics=metrics)

    stem = safe_stem(args.table_stem)
    overall_md_path = output_dir / f"{stem}_overall.md"
    dataset_md_path = output_dir / f"{stem}_by_dataset.md"
    tex_path = output_dir / f"{stem}_overall.tex"
    summary_csv_path = output_dir / f"{stem}_summary.csv"
    split_details_csv_path = output_dir / f"{stem}_split_details.csv"
    gap_csv_path = output_dir / f"{stem}_gap_details.csv"
    scene_csv_path = output_dir / f"{stem}_scene_values.csv"
    metadata_path = output_dir / f"{stem}_metadata.json"

    overall_md = build_overall_markdown(
        values,
        metrics=metrics,
        average_label="Average",
        missing=args.missing,
        bold_best=not bool(args.no_bold_best),
        include_caption=not bool(args.no_caption_markdown),
    )
    dataset_md = build_dataset_markdown(
        values,
        dataset_labels=dataset_labels,
        metrics=metrics,
        missing=args.missing,
        bold_best=not bool(args.no_bold_best),
        include_caption=not bool(args.no_caption_markdown),
    )
    tex = build_overall_latex(values, metrics=metrics, average_label="Average", missing=args.missing)

    overall_md_path.write_text(overall_md + "\n", encoding="utf-8")
    dataset_md_path.write_text(dataset_md + "\n", encoding="utf-8")
    tex_path.write_text(tex + "\n", encoding="utf-8")
    write_summary_csv(summary_csv_path, values, dataset_labels=dataset_labels, metrics=metrics)
    write_rows_csv(split_details_csv_path, dataset_split_rows)
    write_rows_csv(gap_csv_path, gap_rows)
    write_scene_values_csv(scene_csv_path, scene_values)

    metadata = {
        "benchmarking_root": str(root),
        "output_dir": str(output_dir),
        "views": views,
        "settings": {k: list(v.subdirs) for k, v in settings.items()},
        "datasets": {k: list(v.aliases) for k, v in datasets.items()},
        "metrics": {k: {"display": v.display, "key": v.key, "precision": v.precision} for k, v in metrics.items()},
        "scene_stat": args.scene_stat,
        "split_order": list(SPLIT_ORDER),
        "gap_definition": "(Oblique - Nadir) / Nadir * 100; positive means oblique is worse for lower-is-better metrics.",
        "scene_split_map": {d: {s: sorted(v) for s, v in sm.items()} for d, sm in DEFAULT_SCENE_SPLIT_MAP_RAW.items()},
        "n_scene_metric_values": len(scene_values),
        "outputs": {
            "overall_markdown": str(overall_md_path),
            "dataset_markdown": str(dataset_md_path),
            "overall_latex": str(tex_path),
            "summary_csv": str(summary_csv_path),
            "split_details_csv": str(split_details_csv_path),
            "gap_details_csv": str(gap_csv_path),
            "scene_values_csv": str(scene_csv_path),
        },
        "warnings": dedupe_warnings(warnings),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(overall_md)
    print(f"\nWrote overall Markdown: {overall_md_path}")
    print(f"Wrote dataset Markdown: {dataset_md_path}")
    print(f"Wrote overall LaTeX:    {tex_path}")
    print(f"Wrote summary CSV:      {summary_csv_path}")
    print(f"Wrote split details:    {split_details_csv_path}")
    print(f"Wrote gap details:      {gap_csv_path}")
    print(f"Wrote scene values:     {scene_csv_path}")
    print(f"Wrote metadata:         {metadata_path}")

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
python scripts/viz/table_oblique_ndir_forward_reconstruction.py \
  --benchmarking experiments/mapanything/benchmarking \
  --views 8,16,24,32 \
  --output experiments/mapanything/benchmarking/tables/oblique_ndir
"""