#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Table 3.2-style statistics from dense_n_view benchmark outputs.

This version targets the current `benchmarking/dense_n_view/benchmark.py` JSON
format only. It does NOT keep compatibility aliases for old metric keys.

Expected layout:

  <benchmarking_root>/
    dense_8_view/<method>/per_dataset_results.json
    dense_16_view/<method>/per_dataset_results.json
    dense_24_view/<method>/per_dataset_results.json
    dense_32_view/<method>/per_dataset_results.json

Each `per_dataset_results.json` is expected to have:

  {
    "<dataset_name>": {
      "abs_fused_pc_chamfer_l1": 1.23,
      "abs_pose_ate": 0.45,
      ...
    },
    "Average": {...}
  }

Default behavior:
  - read views 8,16,24,32;
  - average the selected views for each method / dataset / metric;
  - write one table per metric.

Examples:

  # Table 3.2: mean over 8/16/24/32 views
  python scripts/viz/table32_zero_shot_stats.py \
    --benchmarking experiments/mapanything/benchmarking \
    --views 8,16,24,32 \
    --metrics chamfer,ray,pose_ate,depth_absrel \
    --output experiments/mapanything/benchmarking/tables/table32

  # Scan every dense_*_view directory and average all available view counts
  python scripts/viz/table32_zero_shot_stats.py \
    --benchmarking experiments/mapanything/benchmarking \
    --views all \
    --metrics chamfer,ray,pose_ate,pose_rot,depth_absrel

  # Also save the old-style per-view tables for checking
  python scripts/viz/table32_zero_shot_stats.py \
    --benchmarking experiments/mapanything/benchmarking \
    --views 8,16,24,32 \
    --view-output both

Custom method / dataset specs:
  --methods "MyModel=my_subdir,VGGT,MapAnything=mapa"
  --datasets "UseGeo=usegeo,US3D=urbanscene3d,A3D-Real=a3dreal"
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
from typing import Iterable, Literal


# -----------------------------------------------------------------------------
# Specs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    label: str
    subdir: str


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class MetricSpec:
    alias: str
    display: str
    key: str
    higher_is_better: bool
    default_precision: int = 3


# Current method directory names used by bash_scripts/benchmark/uav_dense_n_view.
DEFAULT_METHODS: "OrderedDict[str, MethodSpec]" = OrderedDict(
    [
        ("VGGT", MethodSpec("VGGT", "vggt")),
        ("Pi3", MethodSpec("Pi3", "pi3")),
        ("Hunyuan", MethodSpec("Hunyuan", "hunyuan")),
        ("MapAnything", MethodSpec("MapAnything", "mapa")),
        ("Pi3X", MethodSpec("Pi3X", "pi3x")),
        ("DA3", MethodSpec("DA3", "da3")),
    ]
)

# Prior-enabled / ablation variants. These are not included by default, but can be
# selected with --methods Hunyuan-C,Hunyuan-P,Hunyuan-CP,...
EXTRA_METHODS: "OrderedDict[str, MethodSpec]" = OrderedDict(
    [
        ("MapAnything-C", MethodSpec("MapAnything-C", "mapa_csfm")),
        ("MapAnything-P", MethodSpec("MapAnything-P", "mapa_psfm")),
        ("MapAnything-CP", MethodSpec("MapAnything-CP", "mapa_mvs")),
        ("Hunyuan-C", MethodSpec("Hunyuan-C", "hunyuan_csfm")),
        ("Hunyuan-P", MethodSpec("Hunyuan-P", "hunyuan_psfm")),
        ("Hunyuan-CP", MethodSpec("Hunyuan-CP", "hunyuan_mvs")),
        ("Pi3X-C", MethodSpec("Pi3X-C", "pi3x_csfm")),
        ("Pi3X-P", MethodSpec("Pi3X-P", "pi3x_psfm")),
        ("Pi3X-CP", MethodSpec("Pi3X-CP", "pi3x_mvs")),
        ("DA3-CP", MethodSpec("DA3-CP", "da3_mvs")),
        
        # Fine-tuned MapAnything / Pi3X variants.
        # The "|" syntax is supported by method_subdirs(); the first existing
        # directory with per_dataset_results.json will be used.
        ("MapAnything-FT-A3DSyn", MethodSpec("MapAnything-FT-A3DSyn", "mapa-ft-a3dsyn")),
        ("MapAnything-FT-Public", MethodSpec("MapAnything-FT-Public", "mapa-ft-public")),

        ("UAV-MapAnything", MethodSpec("UAV-MapAnything", "uav_mapa_aug_images_only|uav_mapa_aug_images_only_1|uav_mapa|mapa-ft-a3dsyn")),
        ("UAV-MapAnything-C", MethodSpec("UAV-MapAnything-C", "uav_mapa_aug_csfm|uav_mapa_aug_csfm_1|uav_mapa_csfm")),
        ("UAV-MapAnything-P", MethodSpec("UAV-MapAnything-P", "uav_mapa_aug_psfm|uav_mapa_aug_psfm_1|uav_mapa_psfm")),
        ("UAV-MapAnything-CP", MethodSpec("UAV-MapAnything-CP", "uav_mapa_aug_mvs|uav_mapa_aug_mvs_1")),

        ("UAV-Pi3X", MethodSpec("UAV-Pi3X", "uav_pi3x_aug_images_only|pi3x_aug_images_only|uav_pi3x")),
        ("UAV-Pi3X-C", MethodSpec("UAV-Pi3X-C", "uav_pi3x_aug_csfm|pi3x_aug_csfm|uav_pi3x_csfm")),
        ("UAV-Pi3X-P", MethodSpec("UAV-Pi3X-P", "uav_pi3x_aug_psfm|pi3x_aug_psfm|uav_pi3x_psfm")),
        ("UAV-Pi3X-CP", MethodSpec("UAV-Pi3X-CP", "uav_pi3x_aug_mvs|pi3x_aug_mvs|uav_pi3x_mvs")),
    ]
)

# Default paper-table datasets. Matching is still flexible because Hydra dataset
# names are often short keys, while paper table headers are formatted labels.
DEFAULT_DATASETS: list[DatasetSpec] = [
    DatasetSpec("UseGeo", ("usegeo",)),
    DatasetSpec("Enrich-Aerial", ("enrich", "enrich_aerial", "enrichaerial")),
    DatasetSpec("UrbanScene3D", ("urbanscene3d", "urban_scene_3d", "us3d")),
    DatasetSpec("A3D-Real", ("a3dreal", "a3d_real")),
]

# Current metric keys written by benchmarking/dense_n_view/benchmark.py.
# Aliases are convenience names only; every alias maps to exactly one current key.
METRIC_ALIASES: dict[str, MetricSpec] = {
    # Dense reconstruction / point cloud
    "chamfer": MetricSpec("chamfer", "Chamfer-L1", "abs_fused_pc_chamfer_l1", False, 3),
    "precision": MetricSpec("precision", "Precision", "abs_fused_pc_precision", True, 3),
    "recall": MetricSpec("recall", "Recall", "abs_fused_pc_recall", True, 3),
    "f1": MetricSpec("f1", "F1", "abs_fused_pc_f1", True, 3),
    # Camera / pose
    "ray": MetricSpec("ray", "Ray Error (deg)", "ray_dir_mean_angle_deg", False, 2),
    "pose_ate": MetricSpec("pose_ate", "Pose ATE", "abs_pose_ate", False, 3),
    "pose_auc": MetricSpec("pose_auc", "Pose AUC@5deg", "abs_pose_auc_5deg", True, 3),
    "pose_rot": MetricSpec("pose_rot", "Rot. MAE (deg)", "abs_pose_rot_mae_deg", False, 2),
    # Depth
    "depth_absrel": MetricSpec("depth_absrel", "Depth AbsRel", "abs_depth_rel_scale_aligned", False, 3),
    "depth_mae": MetricSpec("depth_mae", "Depth MAE", "abs_depth_mae_scale_aligned", False, 3),
    "depth_rmse": MetricSpec("depth_rmse", "Depth RMSE", "abs_depth_rmse_scale_aligned", False, 3),
    "depth_delta1": MetricSpec("depth_delta1", "Depth δ1", "abs_depth_delta1_scale_aligned", True, 3),
    # Pointmap
    "pointmap_mae": MetricSpec("pointmap_mae", "Pointmap MAE", "abs_pointmap_mae", False, 3),
    "pointmap_rmse": MetricSpec("pointmap_rmse", "Pointmap RMSE", "abs_pointmap_rmse", False, 3),
    "rel_pointmap_abs": MetricSpec("rel_pointmap_abs", "Rel. Pointmap Abs", "rel_pointmap_abs", False, 3),
    "rel_pointmap_delta": MetricSpec("rel_pointmap_delta", "Rel. Pointmap δ1.03", "rel_pointmap_delta_1p03", True, 3),
    # Relative metrics kept in the current benchmark output
    "rel_pose_ate": MetricSpec("rel_pose_ate", "Rel. Pose ATE", "rel_pose_ate", False, 3),
    "rel_pose_auc": MetricSpec("rel_pose_auc", "Rel. Pose AUC@5deg", "rel_pose_auc_5deg", True, 3),
    "rel_depth_abs": MetricSpec("rel_depth_abs", "Rel. Depth Abs", "rel_depth_abs", False, 3),
    "rel_depth_delta": MetricSpec("rel_depth_delta", "Rel. Depth δ1.03", "rel_depth_delta_1p03", True, 3),
    # Alignment diagnostics
    "sim3_scale": MetricSpec("sim3_scale", "Sim3 Scale", "sim3_scale", False, 3),
    "sim3_valid": MetricSpec("sim3_valid", "Sim3 Valid Rate", "sim3_valid", True, 3),
    "sim3_num_corr": MetricSpec("sim3_num_corr", "Sim3 #Corr", "sim3_num_corr", True, 1),
    "sim3_median_residual": MetricSpec("sim3_median_residual", "Sim3 Median Residual", "sim3_median_residual", False, 3),
    "sim3_inlier_ratio": MetricSpec("sim3_inlier_ratio", "Sim3 Inlier Ratio", "sim3_inlier_ratio", True, 3),
    "sim3_fail": MetricSpec("sim3_fail", "Sim3 Failure Rate", "sim3_failure_rate", False, 3),
}

DEFAULT_METRICS = "chamfer,ray,pose_ate,depth_absrel"
DEFAULT_VIEWS = "8,16,24,32"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def method_subdirs(method: MethodSpec) -> tuple[str, ...]:
    """Return candidate subdirs for a method.

    Supports:
      MethodSpec(..., "dir")
      MethodSpec(..., "dir1|dir2")
      MethodSpec(..., ("dir1", "dir2"))
    """
    raw = method.subdir
    if isinstance(raw, (tuple, list)):
        parts: list[str] = []
        for x in raw:
            parts.extend(str(x).split("|"))
    else:
        parts = str(raw).split("|")
    return tuple(x.strip() for x in parts if x.strip())


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def split_csv_like(s: str | None) -> list[str]:
    if s is None:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def safe_stem(s: str) -> str:
    s = str(s).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_\-.]+", "", s)
    return s[:180] if len(s) > 180 else s


def is_finite_number(x: object) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)


def to_float(x: object) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def mean_finite(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if is_finite_number(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def aggregate_values(values: Iterable[float], stat: str) -> float:
    vals = [float(v) for v in values if is_finite_number(v)]
    if not vals:
        return float("nan")
    if stat == "mean":
        return float(sum(vals) / len(vals))
    if stat == "median":
        return float(statistics.median(vals))
    if stat == "std":
        return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    if stat == "min":
        return float(min(vals))
    if stat == "max":
        return float(max(vals))
    if stat == "n":
        return float(len(vals))
    raise ValueError(f"Unsupported stat: {stat}")


def latex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(s))


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------


def all_known_methods() -> "OrderedDict[str, MethodSpec]":
    out: "OrderedDict[str, MethodSpec]" = OrderedDict()
    out.update(DEFAULT_METHODS)
    out.update(EXTRA_METHODS)
    return out


def parse_methods(spec: str | None) -> "OrderedDict[str, MethodSpec]":
    known = all_known_methods()
    if not spec:
        return OrderedDict((k, DEFAULT_METHODS[k]) for k in DEFAULT_METHODS)

    out: "OrderedDict[str, MethodSpec]" = OrderedDict()
    for item in split_csv_like(spec):
        if "=" in item:
            label, subdir = item.split("=", 1)
            label = label.strip()
            subdir = subdir.strip()
            if not label or not subdir:
                raise ValueError(f"Bad method spec: {item!r}. Expected Label=subdir1|subdir2")
            out[label] = MethodSpec(label, subdir)
        else:
            if item not in known:
                raise ValueError(
                    f"Unknown method label: {item!r}. Known labels: {', '.join(known.keys())}. "
                    "Use Label=subdir to add a custom method."
                )
            out[item] = known[item]
    return out


def parse_datasets(spec: str | None) -> list[DatasetSpec] | Literal["auto"]:
    if not spec:
        return DEFAULT_DATASETS
    if spec.strip().lower() in {"auto", "all"}:
        return "auto"

    known = {d.label: d for d in DEFAULT_DATASETS}
    out: list[DatasetSpec] = []
    for item in split_csv_like(spec):
        if "=" in item:
            label, rhs = item.split("=", 1)
            label = label.strip()
            aliases = tuple(x.strip() for x in rhs.split("|") if x.strip())
            if not label or not aliases:
                raise ValueError(f"Bad dataset spec: {item!r}. Expected Label=alias1|alias2")
            out.append(DatasetSpec(label, aliases))
        else:
            if item in known:
                out.append(known[item])
            else:
                out.append(DatasetSpec(item, (item,)))
    return out


def parse_metrics(spec: str | None) -> list[MetricSpec]:
    items = split_csv_like(spec or DEFAULT_METRICS)
    out: list[MetricSpec] = []
    for item in items:
        if item in METRIC_ALIASES:
            out.append(METRIC_ALIASES[item])
        else:
            # Treat unknown input as an exact current JSON key.
            out.append(MetricSpec(item, item, item, higher_is_better=False, default_precision=3))
    return out


def discover_views(root: Path) -> list[int]:
    views: list[int] = []
    if not root.is_dir():
        return views
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"dense_(\d+)_view", p.name)
        if m:
            views.append(int(m.group(1)))
    return sorted(set(views))


def parse_views(spec: str | None, root: Path) -> list[int]:
    if not spec:
        spec = DEFAULT_VIEWS
    if spec.strip().lower() in {"all", "auto"}:
        return discover_views(root)
    out: list[int] = []
    for x in split_csv_like(spec):
        out.append(int(x))
    return sorted(set(out))


# -----------------------------------------------------------------------------
# IO / matching
# -----------------------------------------------------------------------------


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def match_key(keys: Iterable[str], aliases: Iterable[str], label: str | None = None) -> str | None:
    keys_list = [str(k) for k in keys]
    if not keys_list:
        return None

    alias_norms = [norm_name(a) for a in aliases]
    if label:
        alias_norms.append(norm_name(label))
    alias_norms = [a for a in alias_norms if a]

    # Exact normalized match.
    for k in keys_list:
        if norm_name(k) in alias_norms:
            return k

    # Alias contained in key, e.g. alias=usegeo, key=UseGeoDataset.
    candidates: list[tuple[int, str]] = []
    for k in keys_list:
        nk = norm_name(k)
        for a in alias_norms:
            if a and a in nk:
                candidates.append((len(a), k))
                break
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    return None


def auto_dataset_specs_from_jsons(method_jsons: Iterable[dict]) -> list[DatasetSpec]:
    keys: list[str] = []
    seen: set[str] = set()
    for obj in method_jsons:
        for k in obj.keys():
            if k == "Average":
                continue
            nk = norm_name(k)
            if nk in seen:
                continue
            seen.add(nk)
            keys.append(str(k))
    return [DatasetSpec(k, (k,)) for k in sorted(keys)]


def read_method_jsons(
    root: Path,
    view: int,
    methods: "OrderedDict[str, MethodSpec]",
) -> tuple[dict[str, dict], dict[str, str | None], list[str]]:
    dense_view_dir = root / f"dense_{view}_view"
    warnings: list[str] = []
    method_jsons: dict[str, dict] = {}
    method_dirs: dict[str, str | None] = {}

    if not dense_view_dir.is_dir():
        warnings.append(f"Missing view directory: {dense_view_dir}")
        for m_label in methods:
            method_dirs[m_label] = None
        return method_jsons, method_dirs, warnings

    for m_label, m_spec in methods.items():
        candidates = method_subdirs(m_spec)
        method_dirs[m_label] = None
        searched: list[str] = []
        loaded = False

        for subdir in candidates:
            mdir = dense_view_dir / subdir
            jf = mdir / "per_dataset_results.json"
            searched.append(str(jf))

            if not jf.is_file():
                if method_dirs[m_label] is None and mdir.is_dir():
                    method_dirs[m_label] = str(mdir)
                continue

            method_dirs[m_label] = str(mdir)
            try:
                obj = load_json(jf)
            except Exception as e:
                warnings.append(f"Failed to read {jf}: {e}")
                continue

            if not isinstance(obj, dict):
                warnings.append(f"Bad JSON object in {jf}")
                continue

            method_jsons[m_label] = obj
            loaded = True
            break

        if not loaded:
            warnings.append(
                f"Missing per_dataset_results.json: method={m_label}, view={view}, "
                f"searched={searched}"
            )

    return method_jsons, method_dirs, warnings


def collect_one_view(
    root: Path,
    view: int,
    methods: "OrderedDict[str, MethodSpec]",
    datasets: list[DatasetSpec] | Literal["auto"],
    metrics: list[MetricSpec],
    avg_mode: str,
    auto_datasets_if_empty: bool,
) -> tuple[dict, list[DatasetSpec], list[str]]:
    """Collect values from one dense_<view>_view directory.

    Return:
      data[metric_alias][method_label] = {
        "values": {dataset_label: float},
        "avg": float,
        "method_dir": str | None,
        "matched_dataset_keys": {dataset_label: json_key | None},
        "metric_key": current_metric_key,
      }
    """
    method_jsons, method_dirs, warnings = read_method_jsons(root, view, methods)

    if datasets == "auto":
        dataset_specs = auto_dataset_specs_from_jsons(method_jsons.values())
    else:
        dataset_specs = datasets

    data = _collect_from_loaded_jsons(method_jsons, method_dirs, methods, dataset_specs, metrics, avg_mode)

    if auto_datasets_if_empty and datasets != "auto":
        any_match = False
        for metric in metrics:
            for robj in data.get(metric.alias, {}).values():
                if any(k is not None for k in robj.get("matched_dataset_keys", {}).values()):
                    any_match = True
                    break
            if any_match:
                break
        if not any_match and method_jsons:
            msg = (
                f"No selected dataset names matched JSON keys for view={view}; "
                "falling back to --datasets auto for this view."
            )
            warnings.append(msg)
            dataset_specs = auto_dataset_specs_from_jsons(method_jsons.values())
            data = _collect_from_loaded_jsons(method_jsons, method_dirs, methods, dataset_specs, metrics, avg_mode)

    return data, dataset_specs, warnings


def _collect_from_loaded_jsons(
    method_jsons: dict[str, dict],
    method_dirs: dict[str, str | None],
    methods: "OrderedDict[str, MethodSpec]",
    dataset_specs: list[DatasetSpec],
    metrics: list[MetricSpec],
    avg_mode: str,
) -> dict:
    data: dict = OrderedDict()
    for metric in metrics:
        data[metric.alias] = OrderedDict()
        for m_label in methods.keys():
            obj = method_jsons.get(m_label)
            row_values: "OrderedDict[str, float]" = OrderedDict()
            matched_dataset_keys: "OrderedDict[str, str | None]" = OrderedDict()

            if obj is None:
                for d in dataset_specs:
                    row_values[d.label] = float("nan")
                    matched_dataset_keys[d.label] = None
                avg = float("nan")
            else:
                for d in dataset_specs:
                    json_dataset_key = match_key(obj.keys(), d.aliases, d.label)
                    matched_dataset_keys[d.label] = json_dataset_key
                    if json_dataset_key is None or not isinstance(obj.get(json_dataset_key), dict):
                        row_values[d.label] = float("nan")
                        continue
                    row_values[d.label] = to_float(obj[json_dataset_key].get(metric.key))

                if avg_mode == "none":
                    avg = float("nan")
                elif avg_mode == "json" and isinstance(obj.get("Average"), dict):
                    avg = to_float(obj["Average"].get(metric.key))
                else:
                    avg = mean_finite(row_values.values())

            data[metric.alias][m_label] = {
                "values": row_values,
                "avg": avg,
                "method_dir": method_dirs.get(m_label),
                "matched_dataset_keys": matched_dataset_keys,
                "metric_key": metric.key,
            }
    return data


# -----------------------------------------------------------------------------
# View aggregation
# -----------------------------------------------------------------------------


def union_dataset_specs(view_dataset_specs: Iterable[list[DatasetSpec]]) -> list[DatasetSpec]:
    out: list[DatasetSpec] = []
    seen: set[str] = set()
    for specs in view_dataset_specs:
        for d in specs:
            nd = norm_name(d.label)
            if nd in seen:
                continue
            seen.add(nd)
            out.append(d)
    return out


def aggregate_views(
    per_view_data: dict[int, dict],
    methods: "OrderedDict[str, MethodSpec]",
    dataset_specs: list[DatasetSpec],
    metrics: list[MetricSpec],
    view_stat: str,
    avg_mode: str,
) -> dict:
    agg: dict = OrderedDict()
    for metric in metrics:
        agg[metric.alias] = OrderedDict()
        for m_label in methods.keys():
            row_values: "OrderedDict[str, float]" = OrderedDict()
            row_counts: "OrderedDict[str, int]" = OrderedDict()
            matched_dataset_keys: "OrderedDict[str, str | None]" = OrderedDict()

            for d in dataset_specs:
                vals: list[float] = []
                matched_keys: list[str] = []
                for view, data in sorted(per_view_data.items()):
                    robj = data.get(metric.alias, {}).get(m_label)
                    if not robj:
                        continue
                    v = robj.get("values", {}).get(d.label, float("nan"))
                    if is_finite_number(v):
                        vals.append(float(v))
                    mk = robj.get("matched_dataset_keys", {}).get(d.label)
                    if mk is not None and mk not in matched_keys:
                        matched_keys.append(mk)
                row_values[d.label] = aggregate_values(vals, view_stat)
                row_counts[d.label] = len(vals)
                matched_dataset_keys[d.label] = "|".join(matched_keys) if matched_keys else None

            if avg_mode == "none":
                avg = float("nan")
                avg_n = 0
            elif avg_mode == "json":
                avg_vals: list[float] = []
                for view, data in sorted(per_view_data.items()):
                    robj = data.get(metric.alias, {}).get(m_label)
                    if robj and is_finite_number(robj.get("avg", float("nan"))):
                        avg_vals.append(float(robj["avg"]))
                avg = aggregate_values(avg_vals, view_stat)
                avg_n = len(avg_vals)
            else:
                avg = mean_finite(row_values.values())
                avg_n = sum(1 for v in row_values.values() if is_finite_number(v))

            # Keep one existing method dir as provenance.
            method_dir = None
            for view, data in sorted(per_view_data.items()):
                robj = data.get(metric.alias, {}).get(m_label)
                if robj and robj.get("method_dir"):
                    method_dir = robj.get("method_dir")
                    break

            agg[metric.alias][m_label] = {
                "values": row_values,
                "avg": avg,
                "n_views": row_counts,
                "avg_n_views": avg_n,
                "method_dir": method_dir,
                "matched_dataset_keys": matched_dataset_keys,
                "metric_key": metric.key,
            }
    return agg


# -----------------------------------------------------------------------------
# Formatting / writing
# -----------------------------------------------------------------------------


def best_method_labels(rows: dict, dataset_labels: list[str], metric: MetricSpec, include_avg: bool) -> dict[str, set[str]]:
    cols = list(dataset_labels)
    if include_avg:
        cols.append("Avg.")

    out: dict[str, set[str]] = {c: set() for c in cols}
    for c in cols:
        pairs: list[tuple[str, float]] = []
        for m_label, robj in rows.items():
            v = robj.get("avg", float("nan")) if c == "Avg." else robj.get("values", {}).get(c, float("nan"))
            if is_finite_number(v):
                pairs.append((m_label, float(v)))
        if not pairs:
            continue
        target = max(v for _, v in pairs) if metric.higher_is_better else min(v for _, v in pairs)
        eps = max(abs(target) * 1e-12, 1e-12)
        out[c] = {m for m, v in pairs if abs(v - target) <= eps}
    return out


def format_value(v: float, precision: int, missing: str) -> str:
    if not is_finite_number(v):
        return missing
    return f"{float(v):.{precision}f}"


def format_cell(
    v: float,
    precision: int,
    missing: str,
    is_best: bool,
    style: Literal["plain", "md", "tex"],
) -> str:
    s = format_value(v, precision, missing)
    if s == missing or not is_best:
        return s
    if style == "md":
        return f"**{s}**"
    if style == "tex":
        return r"\textbf{" + s + "}"
    return s


def write_csv_table(path: Path, rows: dict, dataset_labels: list[str], include_avg: bool, precision: int, missing: str) -> None:
    columns = ["Model"] + dataset_labels + (["Avg."] if include_avg else [])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for m_label, robj in rows.items():
            vals = [format_value(robj["values"].get(d, float("nan")), precision, missing) for d in dataset_labels]
            if include_avg:
                vals.append(format_value(robj.get("avg", float("nan")), precision, missing))
            writer.writerow([m_label] + vals)


def write_markdown_table(
    path: Path,
    rows: dict,
    dataset_labels: list[str],
    metric: MetricSpec,
    title_suffix: str,
    include_avg: bool,
    precision: int,
    missing: str,
    bold_best: bool,
) -> None:
    columns = ["Model"] + dataset_labels + (["Avg."] if include_avg else [])
    best = best_method_labels(rows, dataset_labels, metric, include_avg) if bold_best else {}
    lines: list[str] = []
    lines.append(f"# Table 3.2 ({title_suffix}): {metric.display}")
    lines.append("")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |")
    for m_label, robj in rows.items():
        cells = [m_label]
        for d in dataset_labels:
            cells.append(
                format_cell(
                    robj["values"].get(d, float("nan")),
                    precision,
                    missing,
                    is_best=m_label in best.get(d, set()),
                    style="md",
                )
            )
        if include_avg:
            cells.append(
                format_cell(
                    robj.get("avg", float("nan")),
                    precision,
                    missing,
                    is_best=m_label in best.get("Avg.", set()),
                    style="md",
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"Metric key: `{metric.key}`. Best values are {'max' if metric.higher_is_better else 'min'}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(
    path: Path,
    rows: dict,
    dataset_labels: list[str],
    metric: MetricSpec,
    title_suffix: str,
    label_suffix: str,
    include_avg: bool,
    precision: int,
    missing: str,
    bold_best: bool,
) -> None:
    columns = ["Model"] + dataset_labels + (["Avg."] if include_avg else [])
    best = best_method_labels(rows, dataset_labels, metric, include_avg) if bold_best else {}
    col_spec = "l" + "r" * (len(columns) - 1)
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Zero-shot UAV benchmark results using " + latex_escape(metric.display) + f" ({latex_escape(title_suffix)})." + r"}")
    lines.append(r"\label{tab:table32_" + safe_stem(metric.alias) + "_" + safe_stem(label_suffix) + r"}")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(latex_escape(c) for c in columns) + r" \\")
    lines.append(r"\midrule")
    for m_label, robj in rows.items():
        cells = [latex_escape(m_label)]
        for d in dataset_labels:
            cells.append(
                format_cell(
                    robj["values"].get(d, float("nan")),
                    precision,
                    missing,
                    is_best=m_label in best.get(d, set()),
                    style="tex",
                )
            )
        if include_avg:
            cells.append(
                format_cell(
                    robj.get("avg", float("nan")),
                    precision,
                    missing,
                    is_best=m_label in best.get("Avg.", set()),
                    style="tex",
                )
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_long_rows(
    long_rows: list[dict],
    table_kind: str,
    view_group: str,
    view: int | str,
    metric: MetricSpec,
    rows: dict,
    dataset_labels: list[str],
    include_avg: bool,
) -> None:
    for m_label, robj in rows.items():
        for d in dataset_labels:
            long_rows.append(
                {
                    "table_kind": table_kind,
                    "view_group": view_group,
                    "view": view,
                    "metric_alias": metric.alias,
                    "metric_key": metric.key,
                    "metric_display": metric.display,
                    "method": m_label,
                    "method_dir": robj.get("method_dir"),
                    "dataset": d,
                    "dataset_key": robj.get("matched_dataset_keys", {}).get(d),
                    "n_views": robj.get("n_views", {}).get(d, 1 if isinstance(view, int) else ""),
                    "value": robj.get("values", {}).get(d, float("nan")),
                }
            )
        if include_avg:
            long_rows.append(
                {
                    "table_kind": table_kind,
                    "view_group": view_group,
                    "view": view,
                    "metric_alias": metric.alias,
                    "metric_key": metric.key,
                    "metric_display": metric.display,
                    "method": m_label,
                    "method_dir": robj.get("method_dir"),
                    "dataset": "Avg.",
                    "dataset_key": "Average" if table_kind == "per_view" else "computed_selected_mean",
                    "n_views": robj.get("avg_n_views", 1 if isinstance(view, int) else ""),
                    "value": robj.get("avg", float("nan")),
                }
            )


def write_long_csv(path: Path, long_rows: list[dict]) -> None:
    fieldnames = [
        "table_kind",
        "view_group",
        "view",
        "metric_alias",
        "metric_key",
        "metric_display",
        "method",
        "method_dir",
        "dataset",
        "dataset_key",
        "n_views",
        "value",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in long_rows:
            rr = dict(r)
            v = rr.get("value")
            rr["value"] = "" if not is_finite_number(v) else f"{float(v):.12g}"
            writer.writerow(rr)


def write_tables_for_metric(
    out_dir: Path,
    out_prefix: str,
    stem_suffix: str,
    title_suffix: str,
    label_suffix: str,
    metric: MetricSpec,
    rows: dict,
    dataset_labels: list[str],
    include_avg: bool,
    precision: int,
    missing: str,
    bold_best: bool,
) -> list[Path]:
    stem = f"{out_prefix}_{safe_stem(metric.alias)}_{stem_suffix}"
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    tex_path = out_dir / f"{stem}.tex"
    write_csv_table(csv_path, rows, dataset_labels, include_avg, precision, missing)
    write_markdown_table(md_path, rows, dataset_labels, metric, title_suffix, include_avg, precision, missing, bold_best)
    write_latex_table(tex_path, rows, dataset_labels, metric, title_suffix, label_suffix, include_avg, precision, missing, bold_best)
    return [csv_path, md_path, tex_path]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Table 3.2-style statistics from current dense_n_view benchmark JSON outputs."
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
        default="experiments/mapanything/benchmarking/tables/table32",
        help="Output directory for csv/md/tex tables.",
    )
    parser.add_argument(
        "--views",
        type=str,
        default=DEFAULT_VIEWS,
        help=(
            "Comma-separated view counts to aggregate, e.g. '8,16,24,32' or '24'. "
            "Use 'all' to scan all dense_*_view dirs and average all discovered view counts."
        ),
    )
    parser.add_argument(
        "--view-stat",
        choices=["mean", "median", "std", "min", "max", "n"],
        default="mean",
        help="Statistic used to aggregate selected view counts. Default: mean.",
    )
    parser.add_argument(
        "--view-output",
        choices=["aggregate", "per_view", "both"],
        default="aggregate",
        help="aggregate = one table averaged over selected views; per_view = one table per view; both = write both.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated method labels or custom Label=subdir specs. Default: VGGT,Pi3,Hunyuan,MapAnything,Pi3X,DA3.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated dataset labels or Label=alias1|alias2 specs. "
            "Default: UseGeo,Enrich-Aerial,UrbanScene3D,A3D-Real. "
            "Use 'auto' to include all discovered dataset keys."
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
        default=DEFAULT_METRICS,
        help="Comma-separated metric aliases or exact current JSON keys. Useful aliases: " + ", ".join(METRIC_ALIASES.keys()),
    )
    parser.add_argument(
        "--avg",
        choices=["selected", "json", "none"],
        default="selected",
        help=(
            "How to fill Avg. column. selected = mean over displayed dataset columns; "
            "json = average JSON['Average'] over views; none = omit values."
        ),
    )
    parser.add_argument("--no-avg", dest="include_avg", action="store_false", help="Do not write Avg. column.")
    parser.set_defaults(include_avg=True)
    parser.add_argument("--precision", type=int, default=None, help="Override decimal precision for all metrics.")
    parser.add_argument("--missing", type=str, default="--", help="String used for missing values.")
    parser.add_argument("--bold-best", action="store_true", default=True, help="Bold best value per column in md/tex.")
    parser.add_argument("--no-bold-best", dest="bold_best", action="store_false")
    parser.add_argument("--out-prefix", type=str, default="table32", help="Output filename prefix.")
    args = parser.parse_args()

    root = Path(args.benchmarking)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        raise FileNotFoundError(f"Benchmarking root not found: {root}")

    methods = parse_methods(args.methods)
    datasets_arg = parse_datasets(args.datasets)
    metrics = parse_metrics(args.metrics)
    views = parse_views(args.views, root)
    if not views:
        raise RuntimeError(f"No view counts selected. Check --views and dense_*_view folders under {root}")

    long_rows: list[dict] = []
    metadata: dict = {
        "benchmarking_root": str(root),
        "output": str(out_dir),
        "views": views,
        "view_stat": args.view_stat,
        "view_output": args.view_output,
        "avg": args.avg,
        "methods": {k: v.subdir for k, v in methods.items()},
        "metrics": {m.alias: {"display": m.display, "key": m.key} for m in metrics},
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

    # If datasets are fixed, use those table labels. If auto fallback happened for
    # some views, use the union so the aggregate table remains well-defined.
    if datasets_arg == "auto":
        aggregate_dataset_specs = union_dataset_specs(per_view_dataset_specs.values())
    else:
        # Prefer the requested datasets, but if a view fell back to auto because
        # nothing matched, include its discovered keys as well.
        aggregate_dataset_specs = union_dataset_specs([datasets_arg] + list(per_view_dataset_specs.values()))

    dataset_labels = [d.label for d in aggregate_dataset_specs]
    view_suffix = "views_" + "_".join(str(v) for v in views)
    view_title = f"{args.view_stat} over views " + "/".join(str(v) for v in views)

    if args.view_output in {"aggregate", "both"}:
        aggregate_data = aggregate_views(
            per_view_data=per_view_data,
            methods=methods,
            dataset_specs=aggregate_dataset_specs,
            metrics=metrics,
            view_stat=args.view_stat,
            avg_mode=args.avg,
        )
        for metric in metrics:
            precision = args.precision if args.precision is not None else metric.default_precision
            paths = write_tables_for_metric(
                out_dir=out_dir,
                out_prefix=args.out_prefix,
                stem_suffix=view_suffix,
                title_suffix=view_title,
                label_suffix=view_suffix,
                metric=metric,
                rows=aggregate_data[metric.alias],
                dataset_labels=dataset_labels,
                include_avg=args.include_avg,
                precision=precision,
                missing=args.missing,
                bold_best=args.bold_best,
            )
            append_long_rows(
                long_rows,
                table_kind="aggregate",
                view_group=view_suffix,
                view="aggregate",
                metric=metric,
                rows=aggregate_data[metric.alias],
                dataset_labels=dataset_labels,
                include_avg=args.include_avg,
            )
            metadata["outputs"].extend(str(p) for p in paths)
            print(f"[OK] {metric.alias} @ {view_title} -> {', '.join(p.name for p in paths)}")

    if args.view_output in {"per_view", "both"}:
        for view, data in per_view_data.items():
            ds_specs = per_view_dataset_specs[view]
            ds_labels = [d.label for d in ds_specs]
            for metric in metrics:
                precision = args.precision if args.precision is not None else metric.default_precision
                paths = write_tables_for_metric(
                    out_dir=out_dir,
                    out_prefix=args.out_prefix,
                    stem_suffix=f"{view}v",
                    title_suffix=f"{view} views",
                    label_suffix=f"{view}v",
                    metric=metric,
                    rows=data[metric.alias],
                    dataset_labels=ds_labels,
                    include_avg=args.include_avg,
                    precision=precision,
                    missing=args.missing,
                    bold_best=args.bold_best,
                )
                append_long_rows(
                    long_rows,
                    table_kind="per_view",
                    view_group=f"{view}v",
                    view=view,
                    metric=metric,
                    rows=data[metric.alias],
                    dataset_labels=ds_labels,
                    include_avg=args.include_avg,
                )
                metadata["outputs"].extend(str(p) for p in paths)
                print(f"[OK] {metric.alias} @ {view} views -> {', '.join(p.name for p in paths)}")

    long_csv = out_dir / f"{args.out_prefix}_long.csv"
    write_long_csv(long_csv, long_rows)
    metadata["outputs"].append(str(long_csv))

    meta_path = out_dir / f"{args.out_prefix}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata["outputs"].append(str(meta_path))

    print(f"[OK] Wrote long-format CSV: {long_csv}")
    print(f"[OK] Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()

"""

python scripts/viz/table32_zero_shot_stats.py \
  --benchmarking experiments/mapanything/benchmarking \
  --output experiments/mapanything/benchmarking/tables/table32 \
  --views 8,16,24,32

"""
