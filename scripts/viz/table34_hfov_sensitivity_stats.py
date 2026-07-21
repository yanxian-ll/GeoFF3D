#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Table 3.4-style hFOV sensitivity statistics on A3D-FA / A3DSynLargeFAWAI.

Table 3.4:
  hFOV sensitivity of zero-shot models on A3D-FA.

Definition used by default:

  hFOV Sensitivity = (max error across hFOV bins - min error across hFOV bins)
                     / mean error across hFOV bins

This script reads per-scene benchmark outputs because hFOV sensitivity cannot be
computed from per_dataset_results.json alone.

Expected benchmark layout:

  <benchmarking_root>/
    dense_8_view/<method>/<A3DSynLargeFAWAI>_per_scene_results.json
    dense_16_view/<method>/<A3DSynLargeFAWAI>_per_scene_results.json
    dense_24_view/<method>/<A3DSynLargeFAWAI>_per_scene_results.json
    dense_32_view/<method>/<A3DSynLargeFAWAI>_per_scene_results.json

Each per-scene JSON is expected to have:

  {
    "scene_hfov_45_xxx": {
      "ray_dir_mean_angle_deg": [1.2, 1.3, ...],
      "abs_depth_rel_scale_aligned": [...],
      "abs_pose_ate": [...],
      "abs_fused_pc_chamfer_l1": [...],
      ...
    },
    ...
  }

Default behavior:
  - read views 8,16,24,32;
  - use methods VGGT, Pi3, MapAnything, DA3;
  - use A3DSynLargeFAWAI / A3D-FA dataset;
  - use the built-in A3D-FA scene-id -> hFOV mapping first;
  - otherwise parse hFOV from scene names using patterns such as hfov_45, hfov45, fov-60;
  - average scene metric lists, then average values within each hFOV bin;
  - compute normalized range sensitivity for each method and metric.

Examples:

  # Default Table 3.4 over 8/16/24/32 views
  python scripts/viz/table34_hfov_sensitivity_stats.py \
    --benchmarking experiments/mapanything/benchmarking \
    --output experiments/mapanything/benchmarking/tables/table34

  # Use every dense_*_view directory found under the benchmarking root
  python scripts/viz/table34_hfov_sensitivity_stats.py \
    --benchmarking experiments/mapanything/benchmarking \
    --views all

  # If hFOV cannot be parsed from scene names, provide a mapping CSV
  python scripts/viz/table34_hfov_sensitivity_stats.py \
    --scene-hfov-csv data/a3d_fa_scene_hfov.csv

  # Mapping CSV supports either: scene,hfov  or  scene,bin

  # Use a custom dataset file stem / alias
  python scripts/viz/table34_hfov_sensitivity_stats.py \
    --dataset "A3D-FA=A3DSynLargeFAWAI|a3d_fa|a3dfa"
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
from typing import Iterable, Literal

# Make sibling import work when executed as python scripts/viz/table34_...py.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from table32_zero_shot_stats import (  # type: ignore
        DatasetSpec,
        MethodSpec,
        MetricSpec,
        METRIC_ALIASES,
        DEFAULT_VIEWS,
        is_finite_number,
        latex_escape,
        mean_finite,
        parse_methods,
        parse_metrics,
        parse_views,
        safe_stem,
        split_csv_like,
    )
except Exception as exc:  # pragma: no cover - user-facing error path
    raise ImportError(
        "Failed to import table32_zero_shot_stats.py. "
        "Please place table34_hfov_sensitivity_stats.py next to "
        "scripts/viz/table32_zero_shot_stats.py."
    ) from exc


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------


DEFAULT_TABLE34_METHODS = "VGGT,Pi3,MapAnything,DA3"
DEFAULT_TABLE34_METRICS = "ray,depth_absrel,pose_ate,chamfer"

# A3D-FA is the paper-facing name, while benchmark config/file names may use the
# dataset class/config key A3DSynLargeFAWAI.
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

TABLE34_COLUMN_LABELS = {
    "ray": "Ray Error Sens. ↑",
    "depth_absrel": "Depth Sens. ↑",
    "rel_depth_abs": "Depth Sens. ↑",
    "pose_ate": "Pose Sens. ↑",
    "chamfer": "Chamfer Sens. ↑",
}

# Conservative default hFOV patterns. They avoid parsing arbitrary scene IDs.
# Use --allow-loose-hfov-token if your scene names are only like scene_45_xxx.
DEFAULT_HFOV_PATTERNS = [
    r"(?i)(?:^|[^a-z0-9])h[_-]?fov[_=:-]?([0-9]+(?:\.[0-9]+)?)",
    r"(?i)(?:^|[^a-z0-9])hfov([0-9]+(?:\.[0-9]+)?)",
    r"(?i)(?:^|[^a-z0-9])fov[_=:-]?([0-9]+(?:\.[0-9]+)?)",
    r"(?i)(?:^|[^a-z0-9])fov([0-9]+(?:\.[0-9]+)?)",
    r"(?i)([0-9]+(?:\.[0-9]+)?)[_-]?(?:deg|degree|degrees)(?:$|[^a-z0-9])",
]
LOOSE_HFOV_PATTERN = r"(?:^|[_\-/])([0-9]{2,3}(?:\.[0-9]+)?)(?:[_\-/]|$)"

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


def default_a3d_fa_hfov_map() -> dict[str, float]:
    """Return scene-id -> hFOV mapping for A3D-FA.

    Both exact ids and normalized ids are included. The lookup code also supports
    scene names that contain these ids as substrings, which is useful when
    per-scene JSON keys include prefixes or relative paths.
    """
    out: dict[str, float] = {}
    for hfov, scene_ids in DEFAULT_A3D_FA_HFOV_SCENE_IDS.items():
        for scene_id in scene_ids:
            out[scene_id] = float(hfov)
            out[norm_name(scene_id)] = float(hfov)
    return out


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HfovBin:
    key: float | str
    label: str
    low: float | None = None
    high: float | None = None
    center: float | None = None


@dataclass
class SceneMetricValue:
    view: int
    method: str
    method_subdir: str
    dataset_key: str
    scene: str
    hfov: float | None
    hfov_bin: str
    metric_alias: str
    metric_key: str
    value: float
    n_values_in_scene: int


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


def format_float(v: float, precision: int, missing: str, percent: bool = False) -> str:
    if not is_finite_number(v):
        return missing
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{float(v) * scale:.{precision}f}{suffix}"


def std_finite(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if is_finite_number(v)]
    if len(vals) <= 1:
        return 0.0 if vals else float("nan")
    return float(statistics.pstdev(vals))


def parse_dataset_spec(spec: str | None) -> DatasetSpec:
    if not spec:
        return DEFAULT_A3D_FA_DATASET
    if "=" in spec:
        label, rhs = spec.split("=", 1)
        label = label.strip()
        aliases = tuple(x.strip() for x in rhs.split("|") if x.strip())
        if not label or not aliases:
            raise ValueError(f"Bad dataset spec: {spec!r}. Expected Label=alias1|alias2")
        return DatasetSpec(label, aliases)
    return DatasetSpec(spec.strip(), (spec.strip(),))


def dataset_alias_norms(dataset: DatasetSpec) -> list[str]:
    out = [norm_name(dataset.label)]
    out.extend(norm_name(a) for a in dataset.aliases)
    return [x for x in OrderedDict.fromkeys(out) if x]


def strip_per_scene_suffix(path: Path) -> str:
    name = path.name
    suffix = "_per_scene_results.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def match_dataset_stem(stem: str, dataset: DatasetSpec) -> bool:
    ns = norm_name(stem)
    aliases = dataset_alias_norms(dataset)
    if ns in aliases:
        return True
    return any(a and (a in ns or ns in a) for a in aliases)


def find_per_scene_json(method_dir: Path, dataset: DatasetSpec) -> tuple[Path | None, str | None, list[str]]:
    """Find <dataset>_per_scene_results.json inside a method directory."""
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
        # Prefer exact normalized stem match over contains match.
        aliases = set(dataset_alias_norms(dataset))
        exact = [p for p in matched if norm_name(strip_per_scene_suffix(p)) in aliases]
        p = exact[0] if exact else matched[0]
        warnings.append(
            f"Multiple per-scene JSONs match dataset {dataset.label!r} in {method_dir}; using {p.name}"
        )
        return p, strip_per_scene_suffix(p), warnings

    # Helpful fallback: if only one per-scene JSON exists, use it but warn.
    if len(candidates) == 1:
        p = candidates[0]
        warnings.append(
            f"Dataset aliases {dataset.aliases!r} did not match {p.name}; "
            f"using the only per-scene JSON in {method_dir}."
        )
        return p, strip_per_scene_suffix(p), warnings

    warnings.append(
        f"Could not find per-scene JSON for dataset={dataset.label!r} in {method_dir}. "
        f"Available: {', '.join(p.name for p in candidates)}"
    )
    return None, None, warnings


def load_json_dict(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def aggregate_scene_metric(metric_values: object, stat: str) -> tuple[float, int]:
    if isinstance(metric_values, list):
        vals = [finite_or_nan(v) for v in metric_values if is_finite_number(v)]
    else:
        vals = [finite_or_nan(metric_values)] if is_finite_number(metric_values) else []
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


# -----------------------------------------------------------------------------
# hFOV parsing / binning
# -----------------------------------------------------------------------------


def load_scene_hfov_mapping(path: Path | None, use_default_a3d_fa_map: bool = True) -> tuple[dict[str, float], dict[str, str]]:
    """Load optional scene->hFOV or scene->bin mapping from CSV/TSV/JSON.

    CSV/TSV accepted columns:
      scene,hfov
      scene,bin
      scene,hfov,bin

    JSON accepted formats:
      {"scene_name": 45.0, ...}
      {"scene_name": {"hfov": 45.0, "bin": "45"}, ...}
    """
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
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object in mapping file: {path}")
        for scene, val in obj.items():
            if isinstance(val, dict):
                add_scene(str(scene), val.get("hfov", None), val.get("bin", None))
            else:
                add_scene(str(scene), val, None)
        return hfov_map, bin_map

    # CSV/TSV. Use DictReader and sniff delimiter lightly.
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
            add_scene(row.get(scene_col, ""), row.get(hfov_col, None) if hfov_col else None, row.get(bin_col, None) if bin_col else None)
    return hfov_map, bin_map


def lookup_hfov_from_mapping(scene: str, hfov_map: dict[str, float]) -> float | None:
    """Lookup hFOV by exact scene key, normalized key, or contained scene id."""
    if scene in hfov_map:
        return hfov_map[scene]
    ns = norm_name(scene)
    if ns in hfov_map:
        return hfov_map[ns]

    # A3D-FA benchmark scene keys are sometimes full paths or prefixed names that
    # contain the 24-char scene id. Support that without requiring a CSV file.
    scene_lower = str(scene).lower()
    for key, value in hfov_map.items():
        if not key or len(key) < 8:
            continue
        k = str(key).lower()
        if k in scene_lower or k in ns:
            return value
    return None


def parse_hfov_from_scene(
    scene: str,
    patterns: list[str],
    hfov_map: dict[str, float],
) -> float | None:
    mapped_hfov = lookup_hfov_from_mapping(scene, hfov_map)
    if mapped_hfov is not None:
        return mapped_hfov
    for pattern in patterns:
        m = re.search(pattern, scene)
        if not m:
            continue
        # Use the first capturing group if present; otherwise full match.
        raw = m.group(1) if m.groups() else m.group(0)
        try:
            v = float(raw)
        except Exception:
            continue
        # hFOV should be within a physically plausible camera range.
        if 1.0 <= v <= 179.0:
            return v
    return None


def make_hfov_bin(
    scene: str,
    hfov: float | None,
    bin_map: dict[str, str],
    bin_width: float,
    round_hfov: int,
) -> HfovBin | None:
    if scene in bin_map:
        return HfovBin(key=bin_map[scene], label=bin_map[scene])
    ns = norm_name(scene)
    if ns in bin_map:
        return HfovBin(key=bin_map[ns], label=bin_map[ns])
    if hfov is None or not math.isfinite(float(hfov)):
        return None
    hv = float(hfov)
    if bin_width and bin_width > 0:
        low = math.floor(hv / bin_width) * bin_width
        high = low + bin_width
        label = f"{low:g}-{high:g}"
        return HfovBin(key=low, label=label, low=low, high=high, center=(low + high) / 2.0)
    hv_round = round(hv, int(round_hfov))
    label = f"{hv_round:g}"
    return HfovBin(key=hv_round, label=label, center=hv_round)


def sort_bin_labels(bin_stats: dict[str, dict]) -> list[str]:
    def key_fn(label: str):
        obj = bin_stats.get(label, {})
        if is_finite_number(obj.get("sort_key", float("nan"))):
            return (0, float(obj["sort_key"]))
        # Try first number in label.
        m = re.search(r"-?[0-9]+(?:\.[0-9]+)?", str(label))
        if m:
            return (0, float(m.group(0)))
        return (1, str(label))

    return sorted(bin_stats.keys(), key=key_fn)


# -----------------------------------------------------------------------------
# Collecting values
# -----------------------------------------------------------------------------


def collect_scene_values(
    root: Path,
    views: list[int],
    methods: "OrderedDict[str, MethodSpec]",
    dataset: DatasetSpec,
    metrics: list[MetricSpec],
    hfov_patterns: list[str],
    hfov_map: dict[str, float],
    bin_map: dict[str, str],
    bin_width: float,
    round_hfov: int,
    scene_stat: str,
    skip_missing_hfov: bool,
) -> tuple[list[SceneMetricValue], list[str]]:
    scene_values: list[SceneMetricValue] = []
    warnings: list[str] = []
    missing_hfov_scenes: set[str] = set()

    for view in views:
        view_dir = root / f"dense_{view}_view"
        if not view_dir.is_dir():
            warnings.append(f"Missing view directory: {view_dir}")
            continue
        for method_label, method in methods.items():
            method_dir = view_dir / method.subdir
            json_path, dataset_key, ws = find_per_scene_json(method_dir, dataset)
            warnings.extend(ws)
            if json_path is None or dataset_key is None:
                continue
            try:
                obj = load_json_dict(json_path)
            except Exception as exc:
                warnings.append(f"Failed to read {json_path}: {exc}")
                continue

            for scene, metric_obj in obj.items():
                if not isinstance(metric_obj, dict):
                    warnings.append(f"Bad scene object in {json_path}: scene={scene!r}")
                    continue
                hfov = parse_hfov_from_scene(str(scene), hfov_patterns, hfov_map)
                hfov_bin = make_hfov_bin(str(scene), hfov, bin_map, bin_width, round_hfov)
                if hfov_bin is None:
                    missing_hfov_scenes.add(str(scene))
                    if skip_missing_hfov:
                        continue
                    hfov_bin = HfovBin(key="unknown", label="unknown")

                for metric in metrics:
                    val, n = aggregate_scene_metric(metric_obj.get(metric.key, []), scene_stat)
                    if not is_finite_number(val):
                        continue
                    scene_values.append(
                        SceneMetricValue(
                            view=view,
                            method=method_label,
                            method_subdir=method.subdir,
                            dataset_key=dataset_key,
                            scene=str(scene),
                            hfov=hfov,
                            hfov_bin=hfov_bin.label,
                            metric_alias=metric.alias,
                            metric_key=metric.key,
                            value=float(val),
                            n_values_in_scene=n,
                        )
                    )

    if missing_hfov_scenes:
        sample = ", ".join(sorted(missing_hfov_scenes)[:20])
        suffix = "..." if len(missing_hfov_scenes) > 20 else ""
        warnings.append(
            f"Could not parse hFOV for {len(missing_hfov_scenes)} scene(s). "
            f"Sample: {sample}{suffix}. Use --scene-hfov-csv or --hfov-regex if needed."
        )
    return scene_values, warnings


def aggregate_bin_means(
    scene_values: list[SceneMetricValue],
    methods: "OrderedDict[str, MethodSpec]",
    metrics: list[MetricSpec],
    min_bin_count: int,
) -> tuple[dict, list[dict]]:
    """Aggregate scene/view values into hFOV-bin means.

    Return:
      bin_data[method][metric_alias][bin_label] = {
        mean, std, n, views, scenes, sort_key
      }
      bin_rows: flat rows for CSV output.
    """
    values: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    scenes: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    views: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    hfovs: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for x in scene_values:
        values[x.method][x.metric_alias][x.hfov_bin].append(x.value)
        scenes[x.method][x.metric_alias][x.hfov_bin].add(x.scene)
        views[x.method][x.metric_alias][x.hfov_bin].add(x.view)
        if x.hfov is not None and math.isfinite(float(x.hfov)):
            hfovs[x.method][x.metric_alias][x.hfov_bin].append(float(x.hfov))

    bin_data: dict = OrderedDict()
    bin_rows: list[dict] = []

    for method_label in methods.keys():
        bin_data[method_label] = OrderedDict()
        for metric in metrics:
            metric_bins: dict = OrderedDict()
            raw_bins = values.get(method_label, {}).get(metric.alias, {})
            for bin_label, vals in raw_bins.items():
                n = len([v for v in vals if is_finite_number(v)])
                if n < min_bin_count:
                    continue
                mean_v = mean_finite(vals)
                std_v = std_finite(vals)
                hfov_vals = hfovs[method_label][metric.alias][bin_label]
                sort_key = mean_finite(hfov_vals) if hfov_vals else float("nan")
                metric_bins[bin_label] = {
                    "mean": mean_v,
                    "std": std_v,
                    "n": n,
                    "n_scenes": len(scenes[method_label][metric.alias][bin_label]),
                    "n_views": len(views[method_label][metric.alias][bin_label]),
                    "views": sorted(views[method_label][metric.alias][bin_label]),
                    "sort_key": sort_key,
                    "metric_key": metric.key,
                }

            # Reorder bins numerically where possible.
            ordered_bins = OrderedDict((b, metric_bins[b]) for b in sort_bin_labels(metric_bins))
            bin_data[method_label][metric.alias] = ordered_bins

            for bin_label, obj in ordered_bins.items():
                bin_rows.append(
                    {
                        "method": method_label,
                        "metric_alias": metric.alias,
                        "metric_key": metric.key,
                        "metric_display": metric.display,
                        "hfov_bin": bin_label,
                        "hfov_sort_key": obj["sort_key"],
                        "bin_mean": obj["mean"],
                        "bin_std": obj["std"],
                        "n_values": obj["n"],
                        "n_scenes": obj["n_scenes"],
                        "n_views": obj["n_views"],
                        "views": ",".join(str(v) for v in obj["views"]),
                    }
                )
    return bin_data, bin_rows


# -----------------------------------------------------------------------------
# Sensitivity computation
# -----------------------------------------------------------------------------


def compute_sensitivity_from_bin_means(values: list[float], mode: str) -> float:
    vals = [float(v) for v in values if is_finite_number(v)]
    if len(vals) < 2:
        return float("nan")
    mean_v = float(sum(vals) / len(vals))
    if mode == "normalized_range":
        if abs(mean_v) <= 1e-12:
            return float("nan")
        return float((max(vals) - min(vals)) / abs(mean_v))
    if mode == "raw_range":
        return float(max(vals) - min(vals))
    if mode == "coefficient_variation":
        if abs(mean_v) <= 1e-12:
            return float("nan")
        return float(statistics.pstdev(vals) / abs(mean_v)) if len(vals) > 1 else 0.0
    raise ValueError(f"Unsupported --sensitivity-mode: {mode}")


def build_sensitivity_rows(
    bin_data: dict,
    methods: "OrderedDict[str, MethodSpec]",
    metrics: list[MetricSpec],
    sensitivity_mode: str,
) -> tuple[list[dict], list[dict]]:
    table_rows: list[dict] = []
    long_rows: list[dict] = []

    for method_label in methods.keys():
        row = {"method": method_label, "sensitivities": OrderedDict(), "n_bins": OrderedDict()}
        for metric in metrics:
            bins = bin_data.get(method_label, {}).get(metric.alias, {})
            bin_means = [obj.get("mean", float("nan")) for obj in bins.values()]
            sens = compute_sensitivity_from_bin_means(bin_means, sensitivity_mode)
            row["sensitivities"][metric.alias] = sens
            row["n_bins"][metric.alias] = len([v for v in bin_means if is_finite_number(v)])

            valid_items = [(label, obj) for label, obj in bins.items() if is_finite_number(obj.get("mean"))]
            min_bin = max_bin = ""
            min_value = max_value = mean_value = float("nan")
            if valid_items:
                min_bin, min_obj = min(valid_items, key=lambda kv: kv[1]["mean"])
                max_bin, max_obj = max(valid_items, key=lambda kv: kv[1]["mean"])
                min_value = float(min_obj["mean"])
                max_value = float(max_obj["mean"])
                mean_value = mean_finite([obj["mean"] for _, obj in valid_items])

            long_rows.append(
                {
                    "method": method_label,
                    "metric_alias": metric.alias,
                    "metric_key": metric.key,
                    "metric_display": metric.display,
                    "sensitivity_mode": sensitivity_mode,
                    "sensitivity": sens,
                    "n_bins": row["n_bins"][metric.alias],
                    "min_bin": min_bin,
                    "min_bin_mean": min_value,
                    "max_bin": max_bin,
                    "max_bin_mean": max_value,
                    "mean_bin_mean": mean_value,
                }
            )
        table_rows.append(row)

    return table_rows, long_rows


def best_sensitivity_cells(table_rows: list[dict], metrics: list[MetricSpec]) -> dict[tuple[int, str], bool]:
    """For sensitivity, smaller is better: less hFOV-sensitive."""
    out: dict[tuple[int, str], bool] = {}
    for metric in metrics:
        vals: list[tuple[float, int]] = []
        for i, row in enumerate(table_rows):
            v = row["sensitivities"].get(metric.alias, float("nan"))
            if is_finite_number(v):
                vals.append((float(v), i))
        if not vals:
            continue
        best = min(v for v, _ in vals)
        for v, i in vals:
            if abs(v - best) <= 1e-12:
                out[(i, metric.alias)] = True
    return out


def metric_col_label(metric: MetricSpec) -> str:
    return TABLE34_COLUMN_LABELS.get(metric.alias, f"{metric.display} Sens. ↑")


# -----------------------------------------------------------------------------
# Writers
# -----------------------------------------------------------------------------


def write_table_csv(
    path: Path,
    table_rows: list[dict],
    metrics: list[MetricSpec],
    precision: int,
    missing: str,
    as_percent: bool,
    include_n_bins: bool,
) -> None:
    cols = ["Model"] + [metric_col_label(m) for m in metrics]
    if include_n_bins:
        cols += [f"{metric_col_label(m)} #bins" for m in metrics]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in table_rows:
            vals = [
                format_float(row["sensitivities"].get(m.alias, float("nan")), precision, missing, as_percent)
                for m in metrics
            ]
            if include_n_bins:
                vals += [row["n_bins"].get(m.alias, "") for m in metrics]
            writer.writerow([row["method"]] + vals)


def write_table_markdown(
    path: Path,
    table_rows: list[dict],
    metrics: list[MetricSpec],
    precision: int,
    missing: str,
    as_percent: bool,
    bold_best: bool,
    title_suffix: str,
    sensitivity_mode: str,
    dataset_label: str,
) -> None:
    cols = ["Model"] + [metric_col_label(m) for m in metrics]
    best = best_sensitivity_cells(table_rows, metrics) if bold_best else {}
    lines: list[str] = []
    lines.append(f"# Table 3.4 ({title_suffix})")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * len(metrics)) + " |")
    for i, row in enumerate(table_rows):
        cells = [str(row["method"])]
        for metric in metrics:
            s = format_float(row["sensitivities"].get(metric.alias, float("nan")), precision, missing, as_percent)
            if bold_best and best.get((i, metric.alias), False) and s != missing:
                s = f"**{s}**"
            cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if sensitivity_mode == "normalized_range":
        lines.append("Sensitivity = `(max error across hFOV bins - min error across hFOV bins) / mean error across hFOV bins`.")
    elif sensitivity_mode == "raw_range":
        lines.append("Sensitivity = `max error across hFOV bins - min error across hFOV bins`.")
    elif sensitivity_mode == "coefficient_variation":
        lines.append("Sensitivity = `std(error across hFOV bins) / mean error across hFOV bins`.")
    lines.append(f"Dataset: `{dataset_label}`. Smaller sensitivity means the model is less affected by hFOV changes.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_latex(
    path: Path,
    table_rows: list[dict],
    metrics: list[MetricSpec],
    precision: int,
    missing: str,
    as_percent: bool,
    bold_best: bool,
    title_suffix: str,
    label_suffix: str,
    sensitivity_mode: str,
    dataset_label: str,
) -> None:
    cols = ["Model"] + [metric_col_label(m) for m in metrics]
    best = best_sensitivity_cells(table_rows, metrics) if bold_best else {}
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{hFOV sensitivity of zero-shot models on "
        + latex_escape(dataset_label)
        + f" ({latex_escape(title_suffix)}). "
        + r"Sensitivity is computed across hFOV bins; lower values indicate less sensitivity.}"
    )
    lines.append(r"\label{tab:table34_hfov_sensitivity_" + safe_stem(label_suffix) + r"}")
    lines.append(r"\begin{tabular}{l" + "r" * len(metrics) + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(latex_escape(c) for c in cols) + r" \\")
    lines.append(r"\midrule")
    for i, row in enumerate(table_rows):
        cells = [latex_escape(str(row["method"]))]
        for metric in metrics:
            s = format_float(row["sensitivities"].get(metric.alias, float("nan")), precision, missing, as_percent)
            if bold_best and best.get((i, metric.alias), False) and s != missing:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scene_values_csv(path: Path, rows: list[SceneMetricValue]) -> None:
    fields = [
        "view",
        "method",
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
                    "method": x.method,
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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create Table 3.4 hFOV sensitivity statistics from dense_n_view per-scene benchmark outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmarking", type=Path, default=Path("experiments/mapanything/benchmarking"), help="Benchmarking root containing dense_*_view folders.")
    p.add_argument("--output", type=Path, default=None, help="Output directory. Default: <benchmarking>/tables/table34")
    p.add_argument("--views", type=str, default=DEFAULT_VIEWS, help="Comma-separated views, e.g. 8,16,24,32, or all.")
    p.add_argument("--methods", type=str, default=DEFAULT_TABLE34_METHODS, help="Methods to include. Uses table32 method specs; supports Label=subdir.")
    p.add_argument("--metrics", type=str, default=DEFAULT_TABLE34_METRICS, help="Metrics to include. Uses current benchmark keys via table32 aliases.")
    p.add_argument("--dataset", type=str, default=None, help="Dataset spec, e.g. A3D-FA=A3DSynLargeFAWAI|a3d_fa|a3dfa.")
    p.add_argument("--scene-hfov-csv", type=Path, default=None, help="Optional CSV/TSV/JSON mapping from scene to hfov or bin.")
    p.add_argument("--no-default-a3d-fa-hfov-map", action="store_true", help="Disable the built-in A3D-FA scene-id to hFOV mapping.")
    p.add_argument("--hfov-regex", action="append", default=None, help="Additional regex with one numeric capturing group for hFOV parsing. Can be passed multiple times.")
    p.add_argument("--allow-loose-hfov-token", action="store_true", help="Also parse standalone 2-3 digit tokens in scene names as hFOV. Use only if scene names are unambiguous.")
    p.add_argument("--hfov-bin-width", type=float, default=0.0, help="If >0, group numeric hFOV values into bins of this width; otherwise use exact hFOV values.")
    p.add_argument("--round-hfov", type=int, default=3, help="Decimals for exact hFOV bin labels when --hfov-bin-width <= 0.")
    p.add_argument("--scene-stat", choices=["mean", "median", "min", "max"], default="mean", help="How to aggregate the per-set metric list for each scene.")
    p.add_argument("--min-bin-count", type=int, default=1, help="Minimum number of scene/view values required for an hFOV bin.")
    p.add_argument("--sensitivity-mode", choices=["normalized_range", "raw_range", "coefficient_variation"], default="normalized_range", help="Sensitivity definition.")
    p.add_argument("--as-percent", action="store_true", help="Display sensitivity values as percentages by multiplying by 100.")
    p.add_argument("--precision", type=int, default=3, help="Number formatting precision.")
    p.add_argument("--missing", type=str, default="-", help="Missing value string in tables.")
    p.add_argument("--no-bold-best", action="store_true", help="Do not bold the least sensitive model for each metric in md/tex.")
    p.add_argument("--include-n-bins", action="store_true", help="Include number of valid hFOV bins in the main CSV table.")
    p.add_argument("--keep-unknown-hfov", action="store_true", help="Keep scenes whose hFOV cannot be parsed in an 'unknown' bin. Default skips them.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    root = args.benchmarking
    output_dir = args.output or (root / "tables" / "table34")
    output_dir.mkdir(parents=True, exist_ok=True)

    views = parse_views(args.views, root)
    if not views:
        raise ValueError(f"No views selected/found under {root}")

    methods = parse_methods(args.methods)
    metrics = parse_metrics(args.metrics)
    dataset = parse_dataset_spec(args.dataset)

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
        methods=methods,
        dataset=dataset,
        metrics=metrics,
        hfov_patterns=hfov_patterns,
        hfov_map=hfov_map,
        bin_map=bin_map,
        bin_width=float(args.hfov_bin_width),
        round_hfov=int(args.round_hfov),
        scene_stat=args.scene_stat,
        skip_missing_hfov=not bool(args.keep_unknown_hfov),
    )

    bin_data, bin_rows = aggregate_bin_means(
        scene_values=scene_values,
        methods=methods,
        metrics=metrics,
        min_bin_count=int(args.min_bin_count),
    )
    table_rows, sensitivity_long_rows = build_sensitivity_rows(
        bin_data=bin_data,
        methods=methods,
        metrics=metrics,
        sensitivity_mode=args.sensitivity_mode,
    )

    view_suffix = "all" if str(args.views).strip().lower() in {"all", "auto"} else "_".join(str(v) for v in views)
    metric_suffix = "_".join(m.alias for m in metrics)
    suffix = f"views_{view_suffix}"
    label_suffix = f"{suffix}_{safe_stem(metric_suffix)}"

    csv_path = output_dir / f"table34_hfov_sensitivity_{suffix}.csv"
    md_path = output_dir / f"table34_hfov_sensitivity_{suffix}.md"
    tex_path = output_dir / f"table34_hfov_sensitivity_{suffix}.tex"
    scene_csv_path = output_dir / f"table34_scene_values_{suffix}.csv"
    bin_csv_path = output_dir / f"table34_hfov_bin_means_{suffix}.csv"
    long_csv_path = output_dir / f"table34_hfov_sensitivity_long_{suffix}.csv"
    metadata_path = output_dir / "table34_metadata.json"

    title_suffix = f"{dataset.label}, views={','.join(str(v) for v in views)}"

    write_table_csv(
        csv_path,
        table_rows,
        metrics,
        precision=args.precision,
        missing=args.missing,
        as_percent=bool(args.as_percent),
        include_n_bins=bool(args.include_n_bins),
    )
    write_table_markdown(
        md_path,
        table_rows,
        metrics,
        precision=args.precision,
        missing=args.missing,
        as_percent=bool(args.as_percent),
        bold_best=not bool(args.no_bold_best),
        title_suffix=title_suffix,
        sensitivity_mode=args.sensitivity_mode,
        dataset_label=dataset.label,
    )
    write_table_latex(
        tex_path,
        table_rows,
        metrics,
        precision=args.precision,
        missing=args.missing,
        as_percent=bool(args.as_percent),
        bold_best=not bool(args.no_bold_best),
        title_suffix=title_suffix,
        label_suffix=label_suffix,
        sensitivity_mode=args.sensitivity_mode,
        dataset_label=dataset.label,
    )

    write_scene_values_csv(scene_csv_path, scene_values)
    write_dict_rows_csv(
        bin_csv_path,
        bin_rows,
        fieldnames=[
            "method",
            "metric_alias",
            "metric_key",
            "metric_display",
            "hfov_bin",
            "hfov_sort_key",
            "bin_mean",
            "bin_std",
            "n_values",
            "n_scenes",
            "n_views",
            "views",
        ],
    )
    write_dict_rows_csv(
        long_csv_path,
        sensitivity_long_rows,
        fieldnames=[
            "method",
            "metric_alias",
            "metric_key",
            "metric_display",
            "sensitivity_mode",
            "sensitivity",
            "n_bins",
            "min_bin",
            "min_bin_mean",
            "max_bin",
            "max_bin_mean",
            "mean_bin_mean",
        ],
    )

    metadata = {
        "benchmarking_root": str(root),
        "output_dir": str(output_dir),
        "views": views,
        "methods": {k: v.subdir for k, v in methods.items()},
        "dataset": {"label": dataset.label, "aliases": list(dataset.aliases)},
        "metrics": [
            {
                "alias": m.alias,
                "display": m.display,
                "key": m.key,
                "higher_is_better": m.higher_is_better,
            }
            for m in metrics
        ],
        "sensitivity_mode": args.sensitivity_mode,
        "sensitivity_formula": {
            "normalized_range": "(max bin mean - min bin mean) / abs(mean bin mean)",
            "raw_range": "max bin mean - min bin mean",
            "coefficient_variation": "std(bin means) / abs(mean bin mean)",
        }[args.sensitivity_mode],
        "scene_stat": args.scene_stat,
        "hfov_bin_width": args.hfov_bin_width,
        "round_hfov": args.round_hfov,
        "min_bin_count": args.min_bin_count,
        "use_default_a3d_fa_hfov_map": not bool(args.no_default_a3d_fa_hfov_map),
        "n_hfov_mapping_keys": len(hfov_map),
        "as_percent": bool(args.as_percent),
        "n_scene_metric_values": len(scene_values),
        "outputs": {
            "table_csv": str(csv_path),
            "table_markdown": str(md_path),
            "table_latex": str(tex_path),
            "scene_values_csv": str(scene_csv_path),
            "hfov_bin_means_csv": str(bin_csv_path),
            "sensitivity_long_csv": str(long_csv_path),
        },
        "warnings": warnings,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote Table 3.4 CSV: {csv_path}")
    print(f"Wrote Table 3.4 Markdown: {md_path}")
    print(f"Wrote Table 3.4 LaTeX: {tex_path}")
    print(f"Wrote hFOV bin means: {bin_csv_path}")
    print(f"Wrote scene-level long CSV: {scene_csv_path}")
    if warnings:
        print("\nWarnings:")
        for w in warnings[:50]:
            print(f"  - {w}")
        if len(warnings) > 50:
            print(f"  ... {len(warnings) - 50} more warnings. See {metadata_path}")


if __name__ == "__main__":
    main()

"""

python scripts/viz/table34_hfov_sensitivity_stats.py \
  --benchmarking experiments/mapanything/benchmarking \
  --views 8,16,24,32 \
  --output experiments/mapanything/benchmarking/tables/table34

"""
