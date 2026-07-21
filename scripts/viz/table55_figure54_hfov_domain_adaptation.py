#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Table 5.5 and plot Figure 5.4 for Chapter 5 hFOV-controlled domain-adaptation analysis.

Table 5.5:
  hFOV sensitivity after domain adaptation on A3D-FA.

Figure 5.4:
  hFOV-controlled evaluation after UAV-domain adaptation.

Expected benchmark layout:
  <benchmarking>/
    dense_8_view/<method_subdir>/<A3D-FA dataset>_per_scene_results.json
    dense_16_view/<method_subdir>/<A3D-FA dataset>_per_scene_results.json
    dense_24_view/<method_subdir>/<A3D-FA dataset>_per_scene_results.json
    dense_32_view/<method_subdir>/<A3D-FA dataset>_per_scene_results.json

The script reuses the common settings/view/format helpers from
scripts/viz/table52_53_54_domain_adaptation.py, but reads per-scene benchmark
JSONs because hFOV sensitivity must be computed over hFOV bins.

Sensitivity definition:
  Sensitivity = (max error across hFOV bins - min error across hFOV bins) / mean error

Example:
  python scripts/viz/table55_figure54_hfov_domain_adaptation.py \
    --benchmarking experiments/mapanything/benchmarking \
    --views 8,16,24,32 \
    --output experiments/mapanything/benchmarking/figures_tables/table55_figure54
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# Make sibling import work when this script is placed under scripts/viz/.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from table52_53_54_domain_adaptation import (  # type: ignore
    DatasetSpec,
    MetricSpec,
    SettingSpec,
    dedupe_warnings,
    is_finite_number,
    mean_finite,
    parse_setting_specs,
    parse_views,
    safe_stem,
    tex_escape,
)


# -----------------------------------------------------------------------------
# Specs
# -----------------------------------------------------------------------------


@dataclass
class SceneMetricValue:
    view: int
    setting: str
    method_subdir: str
    dataset_key: str
    scene: str
    hfov: float | None
    hfov_bin: str
    metric_alias: str
    metric_key: str
    value: float
    n_values_in_scene: int


DEFAULT_A3D_FA_DATASET = DatasetSpec(
    "A3D-FA",
    (
        "A3DSynLargeFAWAI",
        "a3dsynlargefawai",
        "a3d_syn_large_fawai",
        "a3d-fa",
        "a3d_fa",
        "a3dfa",
        "A3DFA",
        "A3D-FA",
    ),
)

# Built-in A3D-FA / A3DSynLargeFAWAI scene-id to hFOV mapping.
DEFAULT_A3D_FA_HFOV_SCENE_IDS: dict[int, tuple[str, ...]] = {
    25: (
        "e1b883efa2b8768cfab20347",
        "a73bdd58a0e011e8e415e625",
        "20eed7076da120a7d398df66",
        "e63c154a77fb5ca738875320",
    ),
    35: (
        "71040e8faffc08ba7082b029",
        "768416ab0299c27d86bf292b",
        "9d64efeb3ecfd03c26161c18",
        "667452d4325d8916a88db95a",
    ),
    45: (
        "fa73d296a111a7e3e973f237",
        "7392aec7502366689224419c",
        "c11ff72f5113f4f111d89dbd",
        "6b2399f2b4821c795dcf57ea",
    ),
    55: (
        "23dcba4dffe0c6bf0f59042e",
        "f74808af2ee1b0430f5e1cb2",
        "19ec8ddb25b71a5be6976b93",
        "51e2d0bb51027f115b78f914",
    ),
    65: (
        "179e2063c562a60e3308d99a",
        "9ef090e43e5036b438022bac",
        "e3a928e88f9643a03c8a1adc",
        "64bdf3e8e7a1f57668f00bf8",
    ),
    75: (
        "647ac219f9bf5eb6154d0f2b",
        "5e1ed8ee3c7f5951664de02c",
        "b276a6c7098388ff1bbcded5",
        "1c79ba1167b39cddf16b9c38",
    ),
    85: (
        "3b6bb1e3910ef5b714da4f28",
        "a4d4c5752f7a802fd1871d09",
        "2158192f4299118faf68f4fc",
        "078dbc15de74d692b0b30787",
    ),
    95: (
        "1cb8e5e8baf385a3cb2dcf9a",
        "f6302ec5904cd552d3fda600",
        "9777b95bc27e62b2674937ff",
        "652584ec9dff985cecebcf3a",
    ),
}

DEFAULT_HFOV_PATTERNS = [
    r"(?i)(?:^|[^a-z0-9])h[_-]?fov[_=:-]?([0-9]+(?:\.[0-9]+)?)",
    r"(?i)(?:^|[^a-z0-9])hfov([0-9]+(?:\.[0-9]+)?)",
    r"(?i)(?:^|[^a-z0-9])fov[_=:-]?([0-9]+(?:\.[0-9]+)?)",
    r"(?i)(?:^|[^a-z0-9])fov([0-9]+(?:\.[0-9]+)?)",
    r"(?i)([0-9]+(?:\.[0-9]+)?)[_-]?(?:deg|degree|degrees)(?:$|[^a-z0-9])",
]
LOOSE_HFOV_PATTERN = r"(?:^|[_\-/])([0-9]{2,3}(?:\.[0-9]+)?)(?:[_\-/]|$)"

PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"

DEFAULT_SETTING_COLORS = {
    "MA-Pretrained": "#4d4d4d",
    "MA-FT-Public": "#1f77b4",
    "MA-FT-A3D-Syn": "#ff7f0e",
    "MA-FT-A3D-Full": "#d62728",
}
DEFAULT_SETTING_MARKERS = {
    "MA-Pretrained": "o",
    "MA-FT-Public": "s",
    "MA-FT-A3D-Syn": "^",
    "MA-FT-A3D-Full": "D",
}


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def split_csv_like(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def std_finite(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if is_finite_number(v)]
    if not vals:
        return float("nan")
    return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0


def aggregate_value(value: Any, mode: str = "mean") -> tuple[float, int]:
    if value is None:
        return float("nan"), 0
    if isinstance(value, (int, float)):
        v = float(value)
        return (v, 1) if math.isfinite(v) else (float("nan"), 0)
    if isinstance(value, (list, tuple)):
        vals = [float(v) for v in value if is_finite_number(v)]
        if not vals:
            return float("nan"), 0
        if mode == "mean":
            return float(sum(vals) / len(vals)), len(vals)
        if mode == "median":
            return float(statistics.median(vals)), len(vals)
        if mode == "first":
            return float(vals[0]), len(vals)
        if mode == "second":
            return float(vals[1] if len(vals) > 1 else vals[0]), len(vals)
    return float("nan"), 0


def load_json_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return obj


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


# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------


def parse_dataset_spec(spec: str | None) -> DatasetSpec:
    if spec is None or not str(spec).strip():
        return DEFAULT_A3D_FA_DATASET
    if "=" in spec:
        label, rhs = spec.split("=", 1)
        label = label.strip()
        aliases = tuple(x.strip() for x in rhs.split("|") if x.strip())
    else:
        label = spec.strip()
        aliases = (label,)
    if not label or not aliases:
        raise ValueError(f"Bad dataset spec: {spec!r}. Expected Label=alias1|alias2")
    return DatasetSpec(label=label, aliases=aliases)


def build_metric_specs(args: argparse.Namespace) -> "OrderedDict[str, MetricSpec]":
    return OrderedDict(
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
                "depth",
                MetricSpec(
                    alias="depth",
                    key=args.depth_key,
                    display="Depth",
                    precision=int(args.precision_depth),
                ),
            ),
            (
                "pose",
                MetricSpec(
                    alias="pose",
                    key=args.pose_ate_key,
                    display="Pose",
                    precision=int(args.precision_pose_ate),
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


# -----------------------------------------------------------------------------
# hFOV mapping / parsing
# -----------------------------------------------------------------------------


def default_a3d_fa_hfov_map() -> dict[str, float]:
    out: dict[str, float] = {}
    for hfov, scene_ids in DEFAULT_A3D_FA_HFOV_SCENE_IDS.items():
        for scene_id in scene_ids:
            out[str(scene_id)] = float(hfov)
            out[norm_name(scene_id)] = float(hfov)
    return out


def load_scene_hfov_mapping(
    path: Path | None,
    use_default_a3d_fa_map: bool = True,
) -> tuple[dict[str, float], dict[str, str]]:
    hfov_map: dict[str, float] = default_a3d_fa_hfov_map() if use_default_a3d_fa_map else {}
    bin_map: dict[str, str] = {}
    if path is None:
        return hfov_map, bin_map
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mapping file not found: {path}")

    def add_scene(scene: str, hfov: object = None, bin_label: object = None) -> None:
        if not scene:
            return
        keys = {scene, norm_name(scene)}
        if hfov is not None and str(hfov).strip() != "":
            try:
                hv = float(hfov)
            except Exception:
                hv = float("nan")
            if math.isfinite(hv):
                for k in keys:
                    hfov_map[k] = hv
        if bin_label is not None and str(bin_label).strip() != "":
            bl = str(bin_label).strip()
            for k in keys:
                bin_map[k] = bl

    if path.suffix.lower() == ".json":
        obj = load_json_dict(path)
        for scene, val in obj.items():
            if isinstance(val, dict):
                add_scene(str(scene), val.get("hfov", None), val.get("bin", None))
            else:
                add_scene(str(scene), val, None)
        return hfov_map, bin_map

    text = path.read_text(encoding="utf-8")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    delimiter = "\t" if "\t" in first_line and "," not in first_line else ","
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Mapping file has no header: {path}")
        fields = {norm_name(x): x for x in reader.fieldnames}
        scene_col = fields.get("scene") or fields.get("scenename") or fields.get("label")
        hfov_col = fields.get("hfov") or fields.get("hfieldofview") or fields.get("fov")
        bin_col = fields.get("bin") or fields.get("hfovbin") or fields.get("group")
        if scene_col is None:
            raise ValueError(f"Mapping file must have a scene column: {path}")
        if hfov_col is None and bin_col is None:
            raise ValueError(f"Mapping file must have hfov or bin column: {path}")
        for row in reader:
            add_scene(
                row.get(scene_col, ""),
                row.get(hfov_col, None) if hfov_col else None,
                row.get(bin_col, None) if bin_col else None,
            )
    return hfov_map, bin_map


def lookup_hfov_from_mapping(scene: str, hfov_map: Mapping[str, float]) -> float | None:
    if scene in hfov_map:
        return float(hfov_map[scene])
    ns = norm_name(scene)
    if ns in hfov_map:
        return float(hfov_map[ns])
    scene_lower = str(scene).lower()
    for key, value in hfov_map.items():
        if not key or len(str(key)) < 8:
            continue
        k = str(key).lower()
        if k in scene_lower or k in ns:
            return float(value)
    return None


def parse_hfov_from_scene(scene: str, patterns: list[str], hfov_map: Mapping[str, float]) -> float | None:
    mapped_hfov = lookup_hfov_from_mapping(scene, hfov_map)
    if mapped_hfov is not None:
        return mapped_hfov
    for pattern in patterns:
        m = re.search(pattern, scene)
        if not m:
            continue
        raw = m.group(1) if m.groups() else m.group(0)
        try:
            v = float(raw)
        except Exception:
            continue
        if 1.0 <= v <= 179.0:
            return v
    return None


def make_hfov_bin(
    scene: str,
    hfov: float | None,
    bin_map: Mapping[str, str],
    bin_width: float,
    round_hfov: int,
) -> tuple[str, float | None]:
    if scene in bin_map:
        return bin_map[scene], None
    ns = norm_name(scene)
    if ns in bin_map:
        return bin_map[ns], None
    if hfov is None or not math.isfinite(float(hfov)):
        return "unknown", None
    hv = float(hfov)
    if bin_width and bin_width > 0:
        low = math.floor(hv / bin_width) * bin_width
        high = low + bin_width
        center = (low + high) / 2.0
        return f"{low:g}-{high:g}", center
    hv_round = round(hv, int(round_hfov))
    return f"{hv_round:g}", float(hv_round)


def bin_sort_key(label: str) -> float:
    m = re.search(r"-?[0-9]+(?:\.[0-9]+)?", str(label))
    if m:
        return float(m.group(0))
    return float("inf")


# -----------------------------------------------------------------------------
# Per-scene JSON lookup and collection
# -----------------------------------------------------------------------------


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
        f"Could not find per-scene JSON for dataset={dataset.label!r} in {method_dir}. "
        f"Available: {', '.join(p.name for p in candidates)}"
    )
    return None, None, warnings


def find_setting_per_scene_json(
    root: Path,
    view: int,
    setting: SettingSpec,
    dataset: DatasetSpec,
    fuzzy: bool = True,
) -> tuple[Path | None, str | None, str | None, list[str]]:
    view_dir = root / f"dense_{view}_view"
    warnings: list[str] = []
    if not view_dir.is_dir():
        warnings.append(f"Missing view directory: {view_dir}")
        return None, None, None, warnings

    for subdir in setting.subdirs:
        json_path, dataset_key, ws = find_per_scene_json(view_dir / subdir, dataset)
        # Suppress warnings for failed candidates until all candidates fail.
        if json_path is not None and dataset_key is not None:
            warnings.extend(ws)
            return json_path, subdir, dataset_key, warnings

    if fuzzy:
        wanted = {norm_name(x) for x in setting.subdirs}
        for method_dir in sorted(view_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            if norm_name(method_dir.name) not in wanted:
                continue
            json_path, dataset_key, ws = find_per_scene_json(method_dir, dataset)
            if json_path is not None and dataset_key is not None:
                warnings.extend(ws)
                return json_path, method_dir.name, dataset_key, warnings

    warnings.append(
        f"Missing per-scene JSON for setting={setting.label!r}, view={view}. "
        f"Tried subdirs={setting.subdirs} under {view_dir}"
    )
    return None, None, None, warnings


def collect_scene_values(
    root: Path,
    views: list[int],
    settings: "OrderedDict[str, SettingSpec]",
    dataset: DatasetSpec,
    metrics: "OrderedDict[str, MetricSpec]",
    hfov_patterns: list[str],
    hfov_map: Mapping[str, float],
    bin_map: Mapping[str, str],
    bin_width: float,
    round_hfov: int,
    agg: str,
    skip_missing_hfov: bool,
    strict: bool,
) -> tuple[list[SceneMetricValue], list[str]]:
    scene_values: list[SceneMetricValue] = []
    warnings: list[str] = []
    missing_hfov_scenes: set[str] = set()

    for view in views:
        for setting_label, setting in settings.items():
            json_path, used_subdir, dataset_key, ws = find_setting_per_scene_json(
                root=root,
                view=view,
                setting=setting,
                dataset=dataset,
            )
            warnings.extend(ws)
            if json_path is None or used_subdir is None or dataset_key is None:
                if strict:
                    raise FileNotFoundError(ws[-1] if ws else f"Missing result for {setting_label}, view={view}")
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
                hfov = parse_hfov_from_scene(str(scene), hfov_patterns, hfov_map)
                hfov_bin, _ = make_hfov_bin(str(scene), hfov, bin_map, bin_width, round_hfov)
                if hfov_bin == "unknown" and skip_missing_hfov:
                    missing_hfov_scenes.add(str(scene))
                    continue

                for metric_alias, metric in metrics.items():
                    val, n = aggregate_value(metric_obj.get(metric.key, None), mode=agg)
                    if not is_finite_number(val):
                        msg = (
                            f"Metric {metric.key!r} is missing/non-finite for setting={setting_label!r}, "
                            f"view={view}, scene={scene!r}, json={json_path}"
                        )
                        if strict:
                            raise KeyError(msg)
                        warnings.append(msg)
                        continue
                    scene_values.append(
                        SceneMetricValue(
                            view=view,
                            setting=setting_label,
                            method_subdir=used_subdir,
                            dataset_key=dataset_key,
                            scene=str(scene),
                            hfov=hfov,
                            hfov_bin=hfov_bin,
                            metric_alias=metric_alias,
                            metric_key=metric.key,
                            value=float(val),
                            n_values_in_scene=n,
                        )
                    )

    if missing_hfov_scenes:
        sample = ", ".join(sorted(missing_hfov_scenes)[:20])
        suffix = "..." if len(missing_hfov_scenes) > 20 else ""
        warnings.append(
            f"Could not parse hFOV for {len(missing_hfov_scenes)} scene(s). Sample: {sample}{suffix}. "
            "Use --scene-hfov-csv or --hfov-regex if needed."
        )
    return scene_values, warnings


def aggregate_hfov_bin_stats(
    scene_values: list[SceneMetricValue],
    min_bin_count: int = 1,
) -> tuple[dict, list[dict]]:
    raw_values: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    raw_scenes: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    raw_views: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    raw_hfovs: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for x in scene_values:
        raw_values[x.setting][x.metric_alias][x.hfov_bin].append(x.value)
        raw_scenes[x.setting][x.metric_alias][x.hfov_bin].add(x.scene)
        raw_views[x.setting][x.metric_alias][x.hfov_bin].add(x.view)
        if x.hfov is not None and math.isfinite(float(x.hfov)):
            raw_hfovs[x.setting][x.metric_alias][x.hfov_bin].append(float(x.hfov))

    bin_stats: dict = OrderedDict()
    rows: list[dict] = []
    for setting in raw_values.keys():
        bin_stats[setting] = OrderedDict()
        for metric_alias in raw_values[setting].keys():
            bin_stats[setting][metric_alias] = OrderedDict()
            for bin_label in sorted(raw_values[setting][metric_alias].keys(), key=bin_sort_key):
                vals = [v for v in raw_values[setting][metric_alias][bin_label] if is_finite_number(v)]
                if len(vals) < min_bin_count:
                    continue
                hfov_vals = raw_hfovs[setting][metric_alias][bin_label]
                sort_key = mean_finite(hfov_vals) if hfov_vals else bin_sort_key(bin_label)
                item = {
                    "mean": mean_finite(vals),
                    "std": std_finite(vals),
                    "n_values": len(vals),
                    "n_scenes": len(raw_scenes[setting][metric_alias][bin_label]),
                    "n_views": len(raw_views[setting][metric_alias][bin_label]),
                    "views": sorted(raw_views[setting][metric_alias][bin_label]),
                    "hfov_sort_key": sort_key,
                }
                bin_stats[setting][metric_alias][bin_label] = item
                rows.append(
                    {
                        "setting": setting,
                        "metric_alias": metric_alias,
                        "hfov_bin": bin_label,
                        "hfov_sort_key": sort_key,
                        "bin_mean": item["mean"],
                        "bin_std": item["std"],
                        "n_values": item["n_values"],
                        "n_scenes": item["n_scenes"],
                        "n_views": item["n_views"],
                        "views": ",".join(str(v) for v in item["views"]),
                    }
                )
    rows.sort(key=lambda r: (str(r["setting"]), str(r["metric_alias"]), float(r["hfov_sort_key"])))
    return bin_stats, rows


# -----------------------------------------------------------------------------
# Table 5.5
# -----------------------------------------------------------------------------


def compute_sensitivity(bin_means: Iterable[float]) -> float:
    vals = [float(v) for v in bin_means if is_finite_number(v)]
    if not vals:
        return float("nan")
    mean_v = mean_finite(vals)
    if not is_finite_number(mean_v) or math.isclose(float(mean_v), 0.0, abs_tol=1e-12):
        return float("nan")
    return float((max(vals) - min(vals)) / mean_v)


def compute_table55_values(
    bin_stats: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    scene_values: list[SceneMetricValue],
    settings: "OrderedDict[str, SettingSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
    mean_mode: str,
) -> tuple[dict, list[dict]]:
    """
    table_values[setting][metric_alias] = {mean, sensitivity, n_bins}
    mean_mode:
      - bin: mean over hFOV-bin means, matching sensitivity denominator.
      - sample: mean over all scene/view samples.
    """
    sample_values: dict = defaultdict(lambda: defaultdict(list))
    for x in scene_values:
        sample_values[x.setting][x.metric_alias].append(x.value)

    table_values: dict = OrderedDict()
    rows: list[dict] = []
    for setting in settings.keys():
        table_values[setting] = OrderedDict()
        for metric_alias in metrics.keys():
            bins = bin_stats.get(setting, {}).get(metric_alias, {})
            bin_means = [float(item.get("mean", float("nan"))) for item in bins.values()]
            if mean_mode == "sample":
                metric_mean = mean_finite(sample_values.get(setting, {}).get(metric_alias, []))
            else:
                metric_mean = mean_finite(bin_means)
            sens = compute_sensitivity(bin_means)
            table_values[setting][metric_alias] = {
                "mean": metric_mean,
                "sensitivity": sens,
                "n_bins": len([v for v in bin_means if is_finite_number(v)]),
            }
            rows.append(
                {
                    "setting": setting,
                    "metric_alias": metric_alias,
                    "metric_display": metrics[metric_alias].display,
                    "metric_key": metrics[metric_alias].key,
                    "mean_mode": mean_mode,
                    "metric_mean": metric_mean,
                    "sensitivity": sens,
                    "n_bins": table_values[setting][metric_alias]["n_bins"],
                }
            )
    return table_values, rows


def best_table55_masks(table_values: Mapping[str, Mapping[str, Mapping[str, float]]], metrics: "OrderedDict[str, MetricSpec]") -> dict:
    masks: dict = OrderedDict()
    for metric_alias in metrics.keys():
        masks[metric_alias] = {"mean": set(), "sensitivity": set()}
        for field in ["mean", "sensitivity"]:
            vals = {
                setting: table_values.get(setting, {}).get(metric_alias, {}).get(field, float("nan"))
                for setting in table_values.keys()
            }
            finite_vals = [v for v in vals.values() if is_finite_number(v)]
            if not finite_vals:
                continue
            best_v = min(finite_vals)
            masks[metric_alias][field] = {
                setting for setting, v in vals.items()
                if is_finite_number(v) and math.isclose(float(v), best_v, rel_tol=1e-12, abs_tol=1e-12)
            }
    return masks


def table55_metric_prefix(metric_alias: str) -> str:
    return {
        "ray": "Ray",
        "depth": "Depth",
        "pose": "Pose",
        "chamfer": "Chamfer",
    }.get(metric_alias, metric_alias)


def build_table55_markdown(
    table_values: Mapping[str, Mapping[str, Mapping[str, float]]],
    metrics: "OrderedDict[str, MetricSpec]",
    precision_sens: int,
    missing: str,
    bold_best: bool,
    include_caption: bool,
) -> str:
    masks = best_table55_masks(table_values, metrics)
    headers = ["Setting"]
    for metric_alias in metrics.keys():
        prefix = table55_metric_prefix(metric_alias)
        headers.extend([f"{prefix} Mean ↓", f"{prefix} Sens. ↓"])
    aligns = [":---"] + ["---:" for _ in headers[1:]]

    rows: list[str] = []
    if include_caption:
        rows.append("**Table 5.5. hFOV sensitivity after domain adaptation on A3D-FA.**")
        rows.append("")
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(aligns) + " |")

    for setting in table_values.keys():
        row = [setting]
        for metric_alias, metric in metrics.items():
            mean_v = table_values[setting][metric_alias].get("mean", float("nan"))
            sens_v = table_values[setting][metric_alias].get("sensitivity", float("nan"))
            mean_cell = fmt_number(mean_v, metric.precision, missing)
            sens_cell = fmt_number(sens_v, precision_sens, missing)
            mean_cell = maybe_bold_md(mean_cell, bold_best and setting in masks[metric_alias]["mean"])
            sens_cell = maybe_bold_md(sens_cell, bold_best and setting in masks[metric_alias]["sensitivity"])
            row.extend([mean_cell, sens_cell])
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def build_table55_latex(
    table_values: Mapping[str, Mapping[str, Mapping[str, float]]],
    metrics: "OrderedDict[str, MetricSpec]",
    precision_sens: int,
    missing: str,
    bold_best: bool,
) -> str:
    masks = best_table55_masks(table_values, metrics)
    n_numeric = len(metrics) * 2
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{hFOV sensitivity after domain adaptation on A3D-FA.}")
    lines.append("\\label{tab:hfov_sensitivity_domain_adaptation}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{l" + "r" * n_numeric + "}")
    lines.append("\\toprule")
    header = ["Setting"]
    for metric_alias in metrics.keys():
        prefix = table55_metric_prefix(metric_alias)
        header.extend([f"{prefix} Mean $\\downarrow$", f"{prefix} Sens. $\\downarrow$"])
    lines.append(" & ".join(tex_escape(x) if "$" not in x else x for x in header) + " " + r"\\")
    lines.append("\\midrule")
    for setting in table_values.keys():
        row = [tex_escape(setting)]
        for metric_alias, metric in metrics.items():
            mean_v = table_values[setting][metric_alias].get("mean", float("nan"))
            sens_v = table_values[setting][metric_alias].get("sensitivity", float("nan"))
            mean_cell = fmt_number(mean_v, metric.precision, missing)
            sens_cell = fmt_number(sens_v, precision_sens, missing)
            mean_cell = maybe_bold_tex(mean_cell, bold_best and setting in masks[metric_alias]["mean"], missing)
            sens_cell = maybe_bold_tex(sens_cell, bold_best and setting in masks[metric_alias]["sensitivity"], missing)
            row.extend([mean_cell, sens_cell])
        lines.append(" & ".join(row) + " " + r"\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def write_table55_csv(path: Path, table_values: Mapping[str, Mapping[str, Mapping[str, float]]], metrics: "OrderedDict[str, MetricSpec]") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["setting"]
    for metric_alias in metrics.keys():
        fieldnames.extend([f"{metric_alias}_mean", f"{metric_alias}_sensitivity", f"{metric_alias}_n_bins"])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for setting in table_values.keys():
            row = {"setting": setting}
            for metric_alias in metrics.keys():
                row[f"{metric_alias}_mean"] = table_values[setting][metric_alias].get("mean", float("nan"))
                row[f"{metric_alias}_sensitivity"] = table_values[setting][metric_alias].get("sensitivity", float("nan"))
                row[f"{metric_alias}_n_bins"] = table_values[setting][metric_alias].get("n_bins", 0)
            writer.writerow(row)


# -----------------------------------------------------------------------------
# Figure 5.4
# -----------------------------------------------------------------------------


def metric_title(metric_alias: str, metric: MetricSpec, panel_letter: str | None = None) -> str:
    name = {
        "ray": "Ray Error vs. hFOV",
        "depth": "Relative Depth Error vs. hFOV",
        "pose": "Pose ATE vs. hFOV",
        "chamfer": "Chamfer-L1 vs. hFOV",
    }.get(metric_alias, f"{metric.display} vs. hFOV")
    if panel_letter:
        return f"({panel_letter}) {name}"
    return name


def metric_ylabel(metric_alias: str, metric: MetricSpec) -> str:
    return {
        "ray": "Ray Error (deg)",
        "depth": "Relative Depth Error",
        "pose": "Pose ATE",
        "chamfer": "Chamfer-L1",
    }.get(metric_alias, metric.display)


def apply_metric_specific_yscale(ax: plt.Axes, metric_alias: str) -> None:
    """Use symlog compression for metrics whose high outliers hide low-range differences."""
    configs = {
        "ray": {
            "linthresh": 2.0,
            "linscale": 1.6,
            "ticks": [0, 0.25, 0.5, 1, 1.5, 2, 5, 10, 20, 50, 100],
        },
        "pose": {
            "linthresh": 20.0,
            "linscale": 1.6,
            "ticks": [0, 2, 5, 10, 15, 20, 50, 100, 200, 500, 1000],
        },
    }
    if metric_alias not in configs:
        return
    ymin, ymax = ax.get_ylim()
    if ymax <= 0:
        return
    cfg = configs[metric_alias]
    ticks = [t for t in cfg["ticks"] if t <= ymax * 1.05]
    larger_ticks = [t for t in cfg["ticks"] if t > ymax * 1.05]
    if larger_ticks:
        ticks.append(larger_ticks[0])
    if len(ticks) < 2:
        ticks = cfg["ticks"][:4]
    ax.set_yscale("symlog", linthresh=cfg["linthresh"], linscale=cfg["linscale"], base=10)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_ylim(bottom=0, top=max(float(ticks[-1]), float(ymax)))


def collect_xticks_from_bin_stats(bin_stats: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]]) -> list[float]:
    ticks: set[float] = set()
    for setting_data in bin_stats.values():
        for metric_data in setting_data.values():
            for bin_label, item in metric_data.items():
                x = item.get("hfov_sort_key", bin_sort_key(bin_label))
                if is_finite_number(x):
                    ticks.add(float(x))
    return sorted(ticks)


def plot_figure54(
    bin_stats: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    settings: "OrderedDict[str, SettingSpec]",
    metrics: "OrderedDict[str, MetricSpec]",
    title: str | None,
    show_std: bool,
    marker_size: float,
    linewidth: float,
    legend_ncol: int,
    log_y: bool,
    adaptive_y: bool,
    ncols: int,
    figsize_scale: float,
) -> plt.Figure:
    metric_items = list(metrics.items())
    n_metrics = len(metric_items)
    ncols = max(1, min(int(ncols), n_metrics))
    nrows = int(math.ceil(n_metrics / ncols))
    fig_w = 5.2 * ncols * figsize_scale
    fig_h = 3.8 * nrows * figsize_scale
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = list(axes.flat)

    panel_idx = 0
    for ax, (metric_alias, metric) in zip(axes_flat, metric_items):
        for setting_label in settings.keys():
            bins = bin_stats.get(setting_label, {}).get(metric_alias, {})
            if not bins:
                continue
            xs: list[float] = []
            ys: list[float] = []
            yerr: list[float] = []
            for bin_label, item in bins.items():
                x = item.get("hfov_sort_key", bin_sort_key(bin_label))
                y = item.get("mean", float("nan"))
                if not is_finite_number(x) or not is_finite_number(y):
                    continue
                xs.append(float(x))
                ys.append(float(y))
                yerr.append(float(item.get("std", 0.0)) if is_finite_number(item.get("std", 0.0)) else 0.0)
            if not xs:
                continue
            color = DEFAULT_SETTING_COLORS.get(setting_label, None)
            marker = DEFAULT_SETTING_MARKERS.get(setting_label, "o")
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

        panel_letter = PANEL_LETTERS[panel_idx] if panel_idx < len(PANEL_LETTERS) else None
        panel_idx += 1
        ax.set_title(metric_title(metric_alias, metric, panel_letter), fontsize=11)
        ax.set_xlabel("hFOV (deg)")
        ax.set_ylabel(metric_ylabel(metric_alias, metric))
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ticks = collect_xticks_from_bin_stats(bin_stats)
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xlim(min(ticks) - 3.0, max(ticks) + 3.0)
        if log_y:
            ax.set_yscale("log")
        elif adaptive_y:
            apply_metric_specific_yscale(ax, metric_alias)

    for ax in axes_flat[n_metrics:]:
        ax.axis("off")

    handles: list[Line2D] = []
    labels: list[str] = []
    for setting_label in settings.keys():
        if setting_label not in bin_stats:
            continue
        handles.append(
            Line2D(
                [0],
                [0],
                marker=DEFAULT_SETTING_MARKERS.get(setting_label, "o"),
                color=DEFAULT_SETTING_COLORS.get(setting_label, None),
                linestyle="-",
                linewidth=linewidth,
                markersize=marker_size,
            )
        )
        labels.append(setting_label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, legend_ncol), frameon=False, bbox_to_anchor=(0.5, 1.02))
    if title:
        fig.suptitle(title, y=1.06, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


# -----------------------------------------------------------------------------
# Writers
# -----------------------------------------------------------------------------


def write_scene_values_csv(path: Path, rows: list[SceneMetricValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "view",
        "setting",
        "method_subdir",
        "dataset_key",
        "scene",
        "hfov",
        "hfov_bin",
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
                    "dataset_key": x.dataset_key,
                    "scene": x.scene,
                    "hfov": "" if x.hfov is None else x.hfov,
                    "hfov_bin": x.hfov_bin,
                    "metric_alias": x.metric_alias,
                    "metric_key": x.metric_key,
                    "value": x.value,
                    "n_values_in_scene": x.n_values_in_scene,
                }
            )


def write_dict_rows_csv(path: Path, rows: list[dict]) -> None:
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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build Table 5.5 and Figure 5.4 hFOV-controlled domain-adaptation analysis on A3D-FA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmarking", type=Path, default=Path("experiments/mapanything/benchmarking"))
    p.add_argument("--output", type=Path, default=None, help="Output directory. Default: <benchmarking>/figures_tables/table55_figure54")
    p.add_argument("--views", type=str, default="8,16,24,32", help="Comma-separated views or 'all'.")
    p.add_argument(
        "--settings",
        nargs="*",
        default=None,
        help="Rows/curves as Label=subdir1|subdir2. Defaults to MA-Pretrained/Public/A3D-Syn/A3D-Full.",
    )
    p.add_argument("--dataset", type=str, default=None, help="A3D-FA dataset spec, e.g. A3D-FA=A3DSynLargeFAWAI|A3D-FA.")
    p.add_argument("--scene-hfov-csv", type=Path, default=None, help="Optional CSV/TSV/JSON mapping from scene to hfov or bin.")
    p.add_argument("--no-default-a3d-fa-hfov-map", action="store_true", help="Disable the built-in A3D-FA scene-id to hFOV mapping.")
    p.add_argument("--hfov-regex", action="append", default=None, help="Additional regex with one numeric capturing group for hFOV parsing. Can be passed multiple times.")
    p.add_argument("--allow-loose-hfov-token", action="store_true", help="Also parse standalone 2-3 digit tokens in scene names as hFOV.")
    p.add_argument("--hfov-bin-width", type=float, default=0.0, help="If >0, group hFOV values into bins of this width; otherwise use exact hFOV values.")
    p.add_argument("--round-hfov", type=int, default=3, help="Decimals for exact hFOV bin labels when --hfov-bin-width <= 0.")
    p.add_argument("--min-bin-count", type=int, default=1, help="Minimum number of scene/view values required for an hFOV bin.")
    p.add_argument("--keep-unknown-hfov", action="store_true", help="Keep scenes whose hFOV cannot be parsed in an 'unknown' bin. Default skips them.")
    p.add_argument("--agg", choices=["mean", "median", "first", "second"], default="mean", help="How to reduce list-valued JSON entries per scene.")
    p.add_argument("--table-mean-mode", choices=["bin", "sample"], default="bin", help="Table 5.5 Mean values: average hFOV-bin means, or all scene/view samples.")
    p.add_argument("--depth-key", type=str, default="abs_depth_rel_scale_aligned")
    p.add_argument("--chamfer-key", type=str, default="abs_fused_pc_chamfer_l1")
    p.add_argument("--ray-key", type=str, default="ray_dir_mean_angle_deg")
    p.add_argument("--pose-ate-key", type=str, default="abs_pose_ate")
    p.add_argument("--precision-depth", type=int, default=3)
    p.add_argument("--precision-chamfer", type=int, default=3)
    p.add_argument("--precision-ray", type=int, default=2)
    p.add_argument("--precision-pose-ate", type=int, default=3)
    p.add_argument("--precision-sens", type=int, default=3)
    p.add_argument("--missing", type=str, default="")
    p.add_argument("--no-caption-markdown", action="store_true")
    p.add_argument("--no-bold-best", action="store_true")
    p.add_argument("--table-stem", type=str, default="table55_hfov_sensitivity_domain_adaptation")
    p.add_argument("--figure-stem", type=str, default="figure54_hfov_controlled_domain_adaptation")
    p.add_argument("--title", type=str, default="Figure 5.4. hFOV-controlled evaluation after UAV-domain adaptation", help="Figure title. Use empty string to disable.")
    p.add_argument("--ncols", type=int, default=2)
    p.add_argument("--figsize-scale", type=float, default=1.0)
    p.add_argument("--show-std", action="store_true", help="Plot std error bars across scene/view values in each hFOV bin.")
    p.add_argument("--marker-size", type=float, default=5.0)
    p.add_argument("--linewidth", type=float, default=1.8)
    p.add_argument("--legend-ncol", type=int, default=4)
    p.add_argument("--log-y", action="store_true", help="Use log scale for all y axes.")
    p.add_argument("--no-adaptive-y", action="store_true", help="Disable symlog compression for Ray/Pose y axes.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--save-svg", action="store_true")
    p.add_argument("--strict", action="store_true", help="Raise on missing files/scenes/metrics instead of warning.")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    root = Path(args.benchmarking)
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmarking root not found: {root}")

    output_dir = Path(args.output) if args.output is not None else root / "figures_tables" / "table55_figure54"
    output_dir.mkdir(parents=True, exist_ok=True)

    views = parse_views(args.views, root)
    if not views:
        raise ValueError(f"No views selected/found under {root}")

    settings = parse_setting_specs(args.settings)
    dataset = parse_dataset_spec(args.dataset)
    metrics = build_metric_specs(args)

    hfov_map, bin_map = load_scene_hfov_mapping(
        args.scene_hfov_csv,
        use_default_a3d_fa_map=not bool(args.no_default_a3d_fa_hfov_map),
    )
    hfov_patterns = list(args.hfov_regex or []) + list(DEFAULT_HFOV_PATTERNS)
    if args.allow_loose_hfov_token:
        hfov_patterns.append(LOOSE_HFOV_PATTERN)

    scene_values, warnings = collect_scene_values(
        root=root,
        views=views,
        settings=settings,
        dataset=dataset,
        metrics=metrics,
        hfov_patterns=hfov_patterns,
        hfov_map=hfov_map,
        bin_map=bin_map,
        bin_width=float(args.hfov_bin_width),
        round_hfov=int(args.round_hfov),
        agg=args.agg,
        skip_missing_hfov=not bool(args.keep_unknown_hfov),
        strict=bool(args.strict),
    )
    if not scene_values:
        raise RuntimeError(
            "No scene values collected. Check --benchmarking, setting subdirs, dataset aliases, and metric keys."
        )

    bin_stats, bin_rows = aggregate_hfov_bin_stats(scene_values, min_bin_count=int(args.min_bin_count))
    table_values, table_rows = compute_table55_values(
        bin_stats=bin_stats,
        scene_values=scene_values,
        settings=settings,
        metrics=metrics,
        mean_mode=args.table_mean_mode,
    )

    # Write Table 5.5
    table_stem = safe_stem(args.table_stem)
    md_path = output_dir / f"{table_stem}.md"
    tex_path = output_dir / f"{table_stem}.tex"
    table_csv_path = output_dir / f"{table_stem}.csv"
    table_details_path = output_dir / f"{table_stem}_details.csv"
    table_metadata_path = output_dir / f"{table_stem}_metadata.json"

    md = build_table55_markdown(
        table_values=table_values,
        metrics=metrics,
        precision_sens=int(args.precision_sens),
        missing=args.missing,
        bold_best=not bool(args.no_bold_best),
        include_caption=not bool(args.no_caption_markdown),
    )
    latex = build_table55_latex(
        table_values=table_values,
        metrics=metrics,
        precision_sens=int(args.precision_sens),
        missing=args.missing,
        bold_best=not bool(args.no_bold_best),
    )
    md_path.write_text(md + "\n", encoding="utf-8")
    tex_path.write_text(latex + "\n", encoding="utf-8")
    write_table55_csv(table_csv_path, table_values, metrics)
    write_dict_rows_csv(table_details_path, table_rows)

    table_metadata = {
        "table_id": "5.5",
        "caption": "hFOV sensitivity after domain adaptation on A3D-FA.",
        "benchmarking_root": str(root),
        "views": views,
        "settings": {k: list(v.subdirs) for k, v in settings.items()},
        "dataset": {"label": dataset.label, "aliases": list(dataset.aliases)},
        "metrics": {k: {"key": v.key, "display": v.display} for k, v in metrics.items()},
        "agg": args.agg,
        "table_mean_mode": args.table_mean_mode,
        "sensitivity_formula": "(max error across hFOV bins - min error across hFOV bins) / mean error",
        "outputs": {
            "markdown": str(md_path),
            "latex": str(tex_path),
            "csv": str(table_csv_path),
            "details_csv": str(table_details_path),
        },
        "warnings": dedupe_warnings(warnings),
    }
    table_metadata_path.write_text(json.dumps(table_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plot Figure 5.4
    title = args.title if str(args.title).strip() else None
    fig = plot_figure54(
        bin_stats=bin_stats,
        settings=settings,
        metrics=metrics,
        title=title,
        show_std=bool(args.show_std),
        marker_size=float(args.marker_size),
        linewidth=float(args.linewidth),
        legend_ncol=int(args.legend_ncol),
        log_y=bool(args.log_y),
        adaptive_y=not bool(args.no_adaptive_y),
        ncols=int(args.ncols),
        figsize_scale=float(args.figsize_scale),
    )

    figure_stem = safe_stem(args.figure_stem)
    png_path = output_dir / f"{figure_stem}.png"
    pdf_path = output_dir / f"{figure_stem}.pdf"
    svg_path = output_dir / f"{figure_stem}.svg"
    bin_csv_path = output_dir / f"{figure_stem}_hfov_bin_means.csv"
    scene_csv_path = output_dir / f"{figure_stem}_scene_values.csv"
    figure_metadata_path = output_dir / f"{figure_stem}_metadata.json"

    fig.savefig(png_path, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    if args.save_svg:
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    write_dict_rows_csv(bin_csv_path, bin_rows)
    write_scene_values_csv(scene_csv_path, scene_values)

    figure_metadata = {
        "figure_id": "5.4",
        "title": title,
        "benchmarking_root": str(root),
        "views": views,
        "settings": {k: list(v.subdirs) for k, v in settings.items()},
        "dataset": {"label": dataset.label, "aliases": list(dataset.aliases)},
        "metrics": {k: {"key": v.key, "display": v.display} for k, v in metrics.items()},
        "agg": args.agg,
        "hfov_bin_width": args.hfov_bin_width,
        "round_hfov": args.round_hfov,
        "min_bin_count": args.min_bin_count,
        "use_default_a3d_fa_hfov_map": not bool(args.no_default_a3d_fa_hfov_map),
        "outputs": {
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
            "figure_svg": str(svg_path) if args.save_svg else None,
            "hfov_bin_means_csv": str(bin_csv_path),
            "scene_values_csv": str(scene_csv_path),
        },
        "warnings": dedupe_warnings(warnings),
    }
    figure_metadata_path.write_text(json.dumps(figure_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    combined_metadata_path = output_dir / "table55_figure54_hfov_domain_adaptation_metadata.json"
    combined_metadata = {
        "benchmarking_root": str(root),
        "output_dir": str(output_dir),
        "views": views,
        "settings": {k: list(v.subdirs) for k, v in settings.items()},
        "dataset": {"label": dataset.label, "aliases": list(dataset.aliases)},
        "metrics": {k: {"key": v.key, "display": v.display} for k, v in metrics.items()},
        "outputs": {
            "table55": {
                "markdown": str(md_path),
                "latex": str(tex_path),
                "csv": str(table_csv_path),
                "details_csv": str(table_details_path),
                "metadata": str(table_metadata_path),
            },
            "figure54": {
                "figure_png": str(png_path),
                "figure_pdf": str(pdf_path),
                "figure_svg": str(svg_path) if args.save_svg else None,
                "hfov_bin_means_csv": str(bin_csv_path),
                "scene_values_csv": str(scene_csv_path),
                "metadata": str(figure_metadata_path),
            },
        },
        "warnings": dedupe_warnings(warnings),
    }
    combined_metadata_path.write_text(json.dumps(combined_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(md)
    print(f"\nWrote Table 5.5 Markdown: {md_path}")
    print(f"Wrote Table 5.5 LaTeX:    {tex_path}")
    print(f"Wrote Table 5.5 CSV:      {table_csv_path}")
    print(f"Wrote Table 5.5 details:  {table_details_path}")
    print(f"Wrote Table 5.5 metadata: {table_metadata_path}")
    print(f"Wrote Figure 5.4 PNG:     {png_path}")
    print(f"Wrote Figure 5.4 PDF:     {pdf_path}")
    if args.save_svg:
        print(f"Wrote Figure 5.4 SVG:     {svg_path}")
    print(f"Wrote hFOV bin means:     {bin_csv_path}")
    print(f"Wrote scene-level values: {scene_csv_path}")
    print(f"Wrote Figure metadata:    {figure_metadata_path}")
    print(f"Wrote combined metadata:  {combined_metadata_path}")

    uniq_warnings = dedupe_warnings(warnings)
    if uniq_warnings and not args.quiet:
        print("\nWarnings:")
        for w in uniq_warnings[:80]:
            print(f"  - {w}")
        if len(uniq_warnings) > 80:
            print(f"  ... {len(uniq_warnings) - 80} more warnings. See {combined_metadata_path}")


if __name__ == "__main__":
    main()

"""
python scripts/viz/table55_figure54_hfov_domain_adaptation.py \
  --benchmarking experiments/mapanything/benchmarking \
  --views 8,16,24,32 \
  --output experiments/mapanything/benchmarking/figures_tables/table55_figure54
"""
