#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Figure 3.5: Effect of camera and pose priors under different hFOV bins.

This script follows the same data path as table34_hfov_sensitivity_stats.py and
figure34_hfov_controlled_diagnosis.py:
  per-scene benchmark JSON -> scene-id to hFOV mapping -> hFOV-bin means.

Figure 3.5 focuses on prior-capable models and compares prior settings under
hFOV bins on A3D-FA / A3DSynLargeFAWAI.

Default behavior:
  - dataset: A3D-FA / A3DSynLargeFAWAI
  - views: 8,16,24,32, averaged together per hFOV bin
  - models: MapAnything, Pi3X
  - prior variants per model: RGB-only, C, P, CP
  - metrics: Ray Error, Pose ATE

Important note:
  - DA3 only has RGB-only and CP in the current benchmark setup.
    It does NOT have separate C and P variants.

Expected benchmark layout:
  <benchmarking_root>/
    dense_8_view/mapa/A3DSynLargeFAWAI_per_scene_results.json
    dense_8_view/mapa_csfm/A3DSynLargeFAWAI_per_scene_results.json
    dense_8_view/mapa_psfm/A3DSynLargeFAWAI_per_scene_results.json
    dense_8_view/mapa_mvs/A3DSynLargeFAWAI_per_scene_results.json
    dense_8_view/pi3x/A3DSynLargeFAWAI_per_scene_results.json
    dense_8_view/pi3x_csfm/A3DSynLargeFAWAI_per_scene_results.json
    ...

Example:
  python scripts/viz/figure35_prior_effect_hfov_bins.py \
    --benchmarking experiments/mapanything/benchmarking \
    --views 8,16,24,32 \
    --models MapAnything,Pi3X \
    --output experiments/mapanything/benchmarking/figures/figure35
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
class PriorGroupSpec:
    model: str
    rgb_subdir: str
    prior_subdirs: "OrderedDict[str, str]"


@dataclass
class SceneMetricValue:
    view: int
    model: str
    prior: str
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

DEFAULT_PRIOR_GROUPS: "OrderedDict[str, PriorGroupSpec]" = OrderedDict(
    [
        (
            "Hunyuan",
            PriorGroupSpec(
                model="Hunyuan",
                rgb_subdir="hunyuan",
                prior_subdirs=OrderedDict([
                    ("C", "hunyuan_csfm"),
                    ("P", "hunyuan_psfm"),
                    ("CP", "hunyuan_mvs"),
                ]),
            ),
        ),
        (
            "MapAnything",
            PriorGroupSpec(
                model="MapAnything",
                rgb_subdir="mapa",
                prior_subdirs=OrderedDict([
                    ("C", "mapa_csfm"),
                    ("P", "mapa_psfm"),
                    ("CP", "mapa_mvs"),
                ]),
            ),
        ),
        (
            "Pi3X",
            PriorGroupSpec(
                model="Pi3X",
                rgb_subdir="pi3x",
                prior_subdirs=OrderedDict([
                    ("C", "pi3x_csfm"),
                    ("P", "pi3x_psfm"),
                    ("CP", "pi3x_mvs"),
                ]),
            ),
        ),
        (
            "DA3",
            PriorGroupSpec(
                model="DA3",
                rgb_subdir="da3",
                prior_subdirs=OrderedDict([
                    ("CP", "da3_mvs"),
                ]),
            ),
        ),
        (
            "UAV-MapAnything",
            PriorGroupSpec(
                model="UAV-MapAnything",
                rgb_subdir="uav_mapa_aug_images_only|uav_mapa_aug_images_only_1|uav_mapa|mapa-ft-a3dsyn",
                prior_subdirs=OrderedDict([
                    ("C", "uav_mapa_aug_csfm|uav_mapa_aug_csfm_1|uav_mapa_csfm"),
                    ("P", "uav_mapa_aug_psfm|uav_mapa_aug_psfm_1|uav_mapa_psfm"),
                    ("CP", "uav_mapa_aug_mvs|uav_mapa_aug_mvs_1"),
                ]),
            ),
        ),
        (
            "UAV-Pi3X",
            PriorGroupSpec(
                model="UAV-Pi3X",
                rgb_subdir="uav_pi3x_aug_images_only|pi3x_aug_images_only|uav_pi3x",
                prior_subdirs=OrderedDict([
                    ("C", "uav_pi3x_aug_csfm|pi3x_aug_csfm|uav_pi3x_csfm"),
                    ("P", "uav_pi3x_aug_psfm|pi3x_aug_psfm|uav_pi3x_psfm"),
                    ("CP", "uav_pi3x_aug_mvs|pi3x_aug_mvs|uav_pi3x_mvs"),
                ]),
            ),
        ),
    ]
)

DEFAULT_VIEWS = "8,16,24,32"
DEFAULT_MODELS = "MapAnything,Pi3"
DEFAULT_METRICS = "ray,pose_ate"

PRIOR_DISPLAY_ORDER = ["RGB-only", "C", "P", "CP"]
PRIOR_COLORS = {
    "RGB-only": "#4d4d4d",
    "C": "#1f77b4",
    "P": "#ff7f0e",
    "CP": "#d62728",
}
PRIOR_MARKERS = {
    "RGB-only": "o",
    "C": "s",
    "P": "^",
    "CP": "D",
}

# Conservative hFOV name parsing.
DEFAULT_HFOV_PATTERNS = [
    r"(?i)(?:^|[^a-z0-9])h[_-]?fov[_=:-]?([0-9]+(?:\.[0-9]+)?)",
    r"(?i)(?:^|[^a-z0-9])hfov([0-9]+(?:\.[0-9]+)?)",
    r"(?i)(?:^|[^a-z0-9])fov[_=:-]?([0-9]+(?:\.[0-9]+)?)",
    r"(?i)(?:^|[^a-z0-9])fov([0-9]+(?:\.[0-9]+)?)",
    r"(?i)([0-9]+(?:\.[0-9]+)?)[_-]?(?:deg|degree|degrees)(?:$|[^a-z0-9])",
]
LOOSE_HFOV_PATTERN = r"(?:^|[_\-/])([0-9]{2,3}(?:\.[0-9]+)?)(?:[_\-/]|$)"

PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"

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


def parse_dataset_spec(spec: str | None) -> DatasetSpec:
    if not spec:
        return DEFAULT_A3D_FA_DATASET
    if "=" in spec:
        label, rhs = spec.split("=", 1)
        aliases = tuple(x.strip() for x in rhs.split("|") if x.strip())
        if not label.strip() or not aliases:
            raise ValueError(f"Bad dataset spec: {spec!r}. Expected Label=alias1|alias2")
        return DatasetSpec(label.strip(), aliases)
    return DatasetSpec(spec.strip(), (spec.strip(),))


def parse_models(spec: str | None) -> "OrderedDict[str, PriorGroupSpec]":
    if not spec:
        return OrderedDict((k, DEFAULT_PRIOR_GROUPS[k]) for k in split_csv_like(DEFAULT_MODELS))
    out: "OrderedDict[str, PriorGroupSpec]" = OrderedDict()
    for item in split_csv_like(spec):
        if item not in DEFAULT_PRIOR_GROUPS:
            raise ValueError(
                f"Unknown model: {item!r}. Known models: {', '.join(DEFAULT_PRIOR_GROUPS.keys())}."
            )
        out[item] = DEFAULT_PRIOR_GROUPS[item]
    if len(out) == 0:
        raise ValueError("No models selected")
    if len(out) > 2:
        raise ValueError("Figure 3.5 is intended for one or two prior-capable models. Please select at most 2 with --models.")
    return out


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


def parse_group_json(path: str | None) -> "OrderedDict[str, PriorGroupSpec] | None":
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
        for key in ["C", "P", "CP"]:
            if key in priors and isinstance(priors[key], str) and priors[key]:
                prior_od[key] = priors[key]
        for key, val in priors.items():
            if key not in prior_od and isinstance(val, str) and val:
                prior_od[str(key)] = val
        out[str(model)] = PriorGroupSpec(model=str(model), rgb_subdir=rgb, prior_subdirs=prior_od)
    return out

def split_subdir_candidates(subdir_spec: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(subdir_spec).split("|") if x.strip())


def find_method_dir(view_dir: Path, subdir_spec: str) -> tuple[Path | None, str | None]:
    for subdir in split_subdir_candidates(subdir_spec):
        p = view_dir / subdir
        if p.is_dir():
            return p, subdir
    return None, None

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


# -----------------------------------------------------------------------------
# hFOV mapping / binning
# -----------------------------------------------------------------------------


def default_a3d_fa_hfov_map() -> dict[str, float]:
    out: dict[str, float] = {}
    for hfov, scene_ids in DEFAULT_A3D_FA_HFOV_SCENE_IDS.items():
        for scene_id in scene_ids:
            out[scene_id] = float(hfov)
            out[norm_name(scene_id)] = float(hfov)
    return out


def load_scene_hfov_mapping(path: Path | None, use_default_a3d_fa_map: bool = True) -> tuple[dict[str, float], dict[str, str]]:
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
    if scene in hfov_map:
        return hfov_map[scene]
    ns = norm_name(scene)
    if ns in hfov_map:
        return hfov_map[ns]
    scene_lower = str(scene).lower()
    for key, value in hfov_map.items():
        if not key or len(str(key)) < 8:
            continue
        k = str(key).lower()
        if k in scene_lower or k in ns:
            return value
    return None


def parse_hfov_from_scene(scene: str, patterns: list[str], hfov_map: dict[str, float]) -> float | None:
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


def make_hfov_bin(scene: str, hfov: float | None, bin_map: dict[str, str], bin_width: float, round_hfov: int) -> tuple[str, float | None]:
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
# Data collection and aggregation
# -----------------------------------------------------------------------------


def collect_scene_values(
    root: Path,
    views: list[int],
    prior_groups: "OrderedDict[str, PriorGroupSpec]",
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

        for model_label, group in prior_groups.items():
            variant_subdirs: "OrderedDict[str, str]" = OrderedDict()
            variant_subdirs["RGB-only"] = group.rgb_subdir
            for prior_name, subdir in group.prior_subdirs.items():
                variant_subdirs[prior_name] = subdir

            for prior_label, subdir in variant_subdirs.items():
                # method_dir = view_dir / subdir
                # json_path, dataset_key, ws = find_per_scene_json(method_dir, dataset)
                
                method_dir, actual_subdir = find_method_dir(view_dir, subdir)
                if method_dir is None or actual_subdir is None:
                    warnings.append(
                        f"Missing method directory for {model_label}/{prior_label} under {view_dir}; "
                        f"searched={split_subdir_candidates(subdir)}"
                    )
                    continue

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
                    hfov_bin, _ = make_hfov_bin(str(scene), hfov, bin_map, bin_width, round_hfov)
                    if hfov_bin == "unknown" and skip_missing_hfov:
                        missing_hfov_scenes.add(str(scene))
                        continue
                    for metric in metrics:
                        val, n = aggregate_scene_metric(metric_obj.get(metric.key, []), scene_stat)
                        if not is_finite_number(val):
                            continue
                        scene_values.append(
                            SceneMetricValue(
                                view=view,
                                model=model_label,
                                prior=prior_label,
                                dataset_key=dataset_key,
                                scene=str(scene),
                                hfov=hfov,
                                hfov_bin=hfov_bin,
                                metric_alias=metric.alias,
                                metric_key=metric.key,
                                value=float(val),
                                n_values_in_scene=n,
                                method_subdir=actual_subdir,
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


def aggregate_hfov_bin_means(scene_values: list[SceneMetricValue], min_bin_count: int = 1) -> tuple[dict, list[dict]]:
    values: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    scenes: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
    views: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
    hfovs: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for x in scene_values:
        values[x.model][x.prior][x.metric_alias][x.hfov_bin].append(x.value)
        scenes[x.model][x.prior][x.metric_alias][x.hfov_bin].add(x.scene)
        views[x.model][x.prior][x.metric_alias][x.hfov_bin].add(x.view)
        if x.hfov is not None and math.isfinite(float(x.hfov)):
            hfovs[x.model][x.prior][x.metric_alias][x.hfov_bin].append(float(x.hfov))

    bin_data: dict = OrderedDict()
    rows: list[dict] = []
    for model_label in values.keys():
        bin_data[model_label] = OrderedDict()
        for prior_label in values[model_label].keys():
            bin_data[model_label][prior_label] = OrderedDict()
            for metric_alias in values[model_label][prior_label].keys():
                metric_bins = values[model_label][prior_label][metric_alias]
                ordered = OrderedDict()
                for bin_label in sorted(metric_bins.keys(), key=bin_sort_key):
                    vals = [v for v in metric_bins[bin_label] if is_finite_number(v)]
                    if len(vals) < min_bin_count:
                        continue
                    hfov_vals = hfovs[model_label][prior_label][metric_alias][bin_label]
                    sort_key = mean_finite(hfov_vals) if hfov_vals else bin_sort_key(bin_label)
                    item = {
                        "mean": mean_finite(vals),
                        "std": std_finite(vals),
                        "n_values": len(vals),
                        "n_scenes": len(scenes[model_label][prior_label][metric_alias][bin_label]),
                        "n_views": len(views[model_label][prior_label][metric_alias][bin_label]),
                        "views": sorted(views[model_label][prior_label][metric_alias][bin_label]),
                        "hfov_sort_key": sort_key,
                    }
                    ordered[bin_label] = item
                    rows.append(
                        {
                            "model": model_label,
                            "prior": prior_label,
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
                bin_data[model_label][prior_label][metric_alias] = ordered
    rows.sort(
        key=lambda r: (
            str(r["model"]),
            PRIOR_DISPLAY_ORDER.index(r["prior"]) if r["prior"] in PRIOR_DISPLAY_ORDER else 999,
            str(r["metric_alias"]),
            float(r["hfov_sort_key"]) if is_finite_number(r["hfov_sort_key"]) else 1e9,
        )
    )
    return bin_data, rows


# -----------------------------------------------------------------------------
# Writers and plotting
# -----------------------------------------------------------------------------


def write_scene_values_csv(path: Path, rows: list[SceneMetricValue]) -> None:
    fields = [
        "view",
        "model",
        "prior",
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
                    "model": x.model,
                    "prior": x.prior,
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


def plot_figure35(
    bin_data: dict,
    prior_groups: "OrderedDict[str, PriorGroupSpec]",
    metrics: list[MetricSpec],
    title: str | None,
    show_std: bool,
    marker_size: float,
    linewidth: float,
    legend_ncol: int,
    log_y: bool,
) -> plt.Figure:
    models = list(prior_groups.keys())
    nrows = len(models)
    ncols = len(metrics)
    figsize = (5.1 * max(1, ncols), 3.8 * max(1, nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = list(axes.flat)

    panel_idx = 0
    for r, model_label in enumerate(models):
        group = prior_groups[model_label]
        available_priors = ["RGB-only"] + [p for p in group.prior_subdirs.keys()]
        ordered_priors = [p for p in PRIOR_DISPLAY_ORDER if p in available_priors] + [p for p in available_priors if p not in PRIOR_DISPLAY_ORDER]
        for c, metric in enumerate(metrics):
            ax = axes[r][c]
            for prior_label in ordered_priors:
                bins = bin_data.get(model_label, {}).get(prior_label, {}).get(metric.alias, {})
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
                color = PRIOR_COLORS.get(prior_label, None)
                marker = PRIOR_MARKERS.get(prior_label, "o")
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
                        label=prior_label,
                    )
                else:
                    ax.plot(
                        xs,
                        ys,
                        marker=marker,
                        color=color,
                        linewidth=linewidth,
                        markersize=marker_size,
                        label=prior_label,
                    )

            letter = PANEL_LETTERS[panel_idx] if panel_idx < len(PANEL_LETTERS) else "?"
            panel_idx += 1
            ax.set_title(f"({letter}) {model_label}: {metric.display} vs. hFOV", fontsize=11)
            ax.set_xlabel("hFOV (deg)")
            ax.set_ylabel(metric.ylabel)
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
            if log_y:
                ax.set_yscale("log")
            ticks = sorted(
                {
                    float(item.get("hfov_sort_key", bin_sort_key(bin_label)))
                    for prior_label in ordered_priors
                    for bin_label, item in bin_data.get(model_label, {}).get(prior_label, {}).get(metric.alias, {}).items()
                    if is_finite_number(item.get("hfov_sort_key", bin_sort_key(bin_label)))
                }
            )
            if ticks:
                ax.set_xticks(ticks)
                ax.set_xlim(min(ticks) - 3.0, max(ticks) + 3.0)

    for ax in axes_flat[panel_idx:]:
        ax.axis("off")

    # Global legend uses the union of priors over the selected models, respecting order.
    union_priors: list[str] = []
    for p in PRIOR_DISPLAY_ORDER:
        for group in prior_groups.values():
            avail = ["RGB-only"] + list(group.prior_subdirs.keys())
            if p in avail and p not in union_priors:
                union_priors.append(p)
    handles: list[Line2D] = []
    labels: list[str] = []
    for prior_label in union_priors:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=PRIOR_MARKERS.get(prior_label, "o"),
                color=PRIOR_COLORS.get(prior_label, None),
                linestyle="-",
                linewidth=linewidth,
                markersize=marker_size,
            )
        )
        labels.append(prior_label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, legend_ncol), frameon=False, bbox_to_anchor=(0.5, 1.02))
    if title:
        fig.suptitle(title, y=1.06, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot Figure 3.5 prior effects under different hFOV bins from A3D-FA per-scene benchmark outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmarking", type=Path, default=Path("experiments/mapanything/benchmarking"), help="Benchmarking root containing dense_*_view folders.")
    p.add_argument("--output", type=Path, default=None, help="Output directory. Default: <benchmarking>/figures/figure35")
    p.add_argument("--views", type=str, default=DEFAULT_VIEWS, help="Comma-separated views, e.g. 8,16,24,32, or all.")
    p.add_argument("--models", type=str, default=DEFAULT_MODELS, help="Prior-capable models to include. Supports at most two. E.g. MapAnything,Pi3X or MapAnything or DA3.")
    p.add_argument("--metrics", type=str, default=DEFAULT_METRICS, help="Metrics to plot. Default: ray,pose_ate")
    p.add_argument("--dataset", type=str, default=None, help="Dataset spec, e.g. A3D-FA=A3DSynLargeFAWAI|a3d_fa|a3dfa.")
    p.add_argument("--group-json", type=str, default=None, help="Optional JSON overriding model->subdir mapping. Same format as table33_prior_effect_stats.py.")
    p.add_argument("--scene-hfov-csv", type=Path, default=None, help="Optional CSV/TSV/JSON mapping from scene to hfov or bin.")
    p.add_argument("--no-default-a3d-fa-hfov-map", action="store_true", help="Disable the built-in A3D-FA scene-id to hFOV mapping.")
    p.add_argument("--hfov-regex", action="append", default=None, help="Additional regex with one numeric capturing group for hFOV parsing. Can be passed multiple times.")
    p.add_argument("--allow-loose-hfov-token", action="store_true", help="Also parse standalone 2-3 digit tokens in scene names as hFOV.")
    p.add_argument("--hfov-bin-width", type=float, default=0.0, help="If >0, group hFOV values into bins of this width; otherwise use exact hFOV values.")
    p.add_argument("--round-hfov", type=int, default=3, help="Decimals for exact hFOV bin labels when --hfov-bin-width <= 0.")
    p.add_argument("--scene-stat", choices=["mean", "median", "min", "max"], default="mean", help="How to aggregate each scene's metric list before pooling across hFOV/views.")
    p.add_argument("--min-bin-count", type=int, default=1, help="Minimum number of scene/view values required for an hFOV bin.")
    p.add_argument("--keep-unknown-hfov", action="store_true", help="Keep scenes whose hFOV cannot be parsed in an 'unknown' bin. Default skips them.")
    p.add_argument("--show-std", action="store_true", help="Plot error bars using std across scene/view values in each hFOV bin.")
    p.add_argument("--log-y", action="store_true", help="Use log scale for all y axes.")
    p.add_argument("--dpi", type=int, default=300, help="PNG DPI.")
    p.add_argument("--save-svg", action="store_true", help="Also save SVG.")
    p.add_argument("--title", type=str, default="Figure 3.5. Effect of camera and pose priors under different hFOV bins", help="Figure title. Use empty string to disable.")
    p.add_argument("--figure-stem", type=str, default="figure35_prior_effect_hfov_bins", help="Output figure stem.")
    p.add_argument("--marker-size", type=float, default=5.0, help="Marker size.")
    p.add_argument("--linewidth", type=float, default=1.8, help="Line width.")
    p.add_argument("--legend-ncol", type=int, default=4, help="Number of legend columns.")
    p.add_argument("--quiet", action="store_true", help="Suppress warning printout.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    root = Path(args.benchmarking)
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmarking root not found: {root}")

    output_dir = Path(args.output) if args.output is not None else (root / "figures" / "figure35")
    output_dir.mkdir(parents=True, exist_ok=True)

    views = parse_views(args.views, root)
    if not views:
        raise ValueError(f"No views selected/found under {root}")

    prior_groups = parse_group_json(args.group_json) if args.group_json else None
    if prior_groups is None:
        prior_groups = parse_models(args.models)
    else:
        selected = split_csv_like(args.models)
        if selected:
            tmp: "OrderedDict[str, PriorGroupSpec]" = OrderedDict()
            for name in selected:
                if name not in prior_groups:
                    raise ValueError(f"Model {name!r} not found in --group-json")
                tmp[name] = prior_groups[name]
            prior_groups = tmp
        if len(prior_groups) > 2:
            raise ValueError("Figure 3.5 is intended for one or two prior-capable models. Please select at most 2.")

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
        prior_groups=prior_groups,
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
    if not scene_values:
        raise RuntimeError(
            "No scene values collected. Check --benchmarking, model/prior subdirs, dataset aliases, and metric keys."
        )

    bin_data, bin_rows = aggregate_hfov_bin_means(scene_values, min_bin_count=int(args.min_bin_count))

    title = args.title if str(args.title).strip() else None
    fig = plot_figure35(
        bin_data=bin_data,
        prior_groups=prior_groups,
        metrics=metrics,
        title=title,
        show_std=bool(args.show_std),
        marker_size=float(args.marker_size),
        linewidth=float(args.linewidth),
        legend_ncol=int(args.legend_ncol),
        log_y=bool(args.log_y),
    )

    view_suffix = "all" if str(args.views).strip().lower() in {"all", "auto"} else "_".join(str(v) for v in views)
    model_suffix = "_".join(safe_stem(k) for k in prior_groups.keys())
    metric_suffix = "_".join(safe_stem(m.alias) for m in metrics)
    stem = safe_stem(f"{args.figure_stem}_{model_suffix}_{metric_suffix}_views_{view_suffix}")

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    svg_path = output_dir / f"{stem}.svg"
    bin_csv_path = output_dir / f"{stem}_hfov_bin_means.csv"
    scene_csv_path = output_dir / f"{stem}_scene_values.csv"
    metadata_path = output_dir / f"{stem}_metadata.json"

    fig.savefig(png_path, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    if args.save_svg:
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    write_dict_rows_csv(
        bin_csv_path,
        bin_rows,
        fieldnames=[
            "model",
            "prior",
            "metric_alias",
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
    write_scene_values_csv(scene_csv_path, scene_values)

    metadata = {
        "benchmarking_root": str(root),
        "output_dir": str(output_dir),
        "views": views,
        "models": {
            k: {
                "rgb": v.rgb_subdir,
                "priors": dict(v.prior_subdirs),
            }
            for k, v in prior_groups.items()
        },
        "dataset": {"label": dataset.label, "aliases": list(dataset.aliases)},
        "metrics": [
            {"alias": m.alias, "display": m.display, "key": m.key, "ylabel": m.ylabel}
            for m in metrics
        ],
        "scene_stat": args.scene_stat,
        "hfov_bin_width": args.hfov_bin_width,
        "round_hfov": args.round_hfov,
        "min_bin_count": args.min_bin_count,
        "use_default_a3d_fa_hfov_map": not bool(args.no_default_a3d_fa_hfov_map),
        "n_scene_metric_values": len(scene_values),
        "n_bin_rows": len(bin_rows),
        "outputs": {
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
            "figure_svg": str(svg_path) if args.save_svg else None,
            "hfov_bin_means_csv": str(bin_csv_path),
            "scene_values_csv": str(scene_csv_path),
        },
        "warnings": warnings,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote Figure 3.5 PNG: {png_path}")
    print(f"Wrote Figure 3.5 PDF: {pdf_path}")
    if args.save_svg:
        print(f"Wrote Figure 3.5 SVG: {svg_path}")
    print(f"Wrote hFOV bin means: {bin_csv_path}")
    print(f"Wrote scene-level values: {scene_csv_path}")
    print(f"Wrote metadata: {metadata_path}")

    if warnings and not args.quiet:
        print("\nWarnings:")
        seen: set[str] = set()
        uniq: list[str] = []
        for w in warnings:
            if w not in seen:
                seen.add(w)
                uniq.append(w)
        for w in uniq[:50]:
            print(f"  - {w}")
        if len(uniq) > 50:
            print(f"  ... {len(uniq) - 50} more warnings. See {metadata_path}")


if __name__ == "__main__":
    main()

"""
python scripts/viz/figure35_prior_effect_hfov_bins.py \
  --benchmarking experiments/mapanything/benchmarking \
  --views 8,16,24,32 \
  --models  "MapAnything,Pi3X"  --log-y \
  --output experiments/mapanything/benchmarking/figures/figure35

python scripts/viz/figure35_prior_effect_hfov_bins.py \
  --benchmarking experiments/mapanything/benchmarking \
  --views 8,16,24,32 \
  --models  "MapAnything,Hunyuan"  --log-y \
  --output experiments/mapanything/benchmarking/figures/figure35

python scripts/viz/figure35_prior_effect_hfov_bins.py \
  --benchmarking experiments/mapanything/benchmarking \
  --views 8,16,24,32 \
  --models  "MapAnything,DA3"  --log-y \
  --output experiments/mapanything/benchmarking/figures/figure35

"""
