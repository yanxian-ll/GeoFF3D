#!/usr/bin/env python3
"""
Collect and tabulate benchmark metrics across methods and datasets.

The script prints two tables:

1. Original averaged table:
   Read each method's per_dataset_results.json and average available values.

2. Common-success averaged table:
   Read each method's <dataset>_per_scene_results.json, keep only the
   sample-level records (scene, sample_idx) where every compared method has
   valid/successful results, then average on this shared sample subset.
   This avoids unfair comparisons caused by different methods silently dropping
   different failed samples.

Usage:
  python scripts/evaluation/collect_metrics.py \
    pi3 pi3x_cp geoff3d_cp

By default, each method is summarized independently with no common-success
intersection filtering.
"""

import argparse
import json
import math
import os
import sys
from collections import OrderedDict

# ---------------------------------------------------------------------------
METHOD_SHORT = {
    "pi3": "Pi3",
    "pi3_ft": "Pi3",
    "pi3x_ft": "(a) Pi3X",
    "pi3x_p": "(a) Pi3X",
    "pi3x_p_ft": "Pi3X-TR",
    "pi3x_cp": "Pi3X-CTR",
    "pi3x_cp_ft": "Pi3X-CTR",
    "geoff3d_t": "Ours-T",
    "geoff3d_p": "(e) Full",
    "geoff3d_p_woworld": "(b) w/o world",
    "geoff3d_p_wogravity": "(c) w/o gravity",
    "geoff3d_t_yaw": "Ours-T",
    "geoff3d_p_yaw": "(d) Full",
    "geoff3d_cp": "Ours-CTR",
    "geoff3d_p_woalign": "(f) Full",
    "geoff3d_t_woalign": "Ours-T*",
    "geoff3d_cp_woalign": "Ours-CTR*",
    "da3": "DA3",
    "da3_cp": "DA3-CTR",
    "mapa": "MAPA",
    "mapa_ft": "MAPA",
    "mapa_t_ft": "MAPA-T",
    "mapa_p": "MAPA-TR",
    "mapa_p_ft": "MAPA-TR",
    "mapa_cp": "MAPA-CTR",
    "vggt": "VGGT",
    "vggt_ft": "VGGT",
}

METHOD_ALIGNMENT = {
    "pi3x_p": "Sim(3)",
    "vggt_ft": "pose",
    "mapa_ft": "pose",
    "mapa_t_ft": "pose",
    "mapa_p_ft": "pose",
    "pi3_ft": "pose",
    "pi3x_ft": "Sim(3)",
    "pi3x_p_ft": "pose",
    "geoff3d_t": "pose",
    "geoff3d_p": "Sim(3)",
    "geoff3d_p_woworld": "GA-Sim",
    "geoff3d_p_wogravity": "GA-Sim",
    "geoff3d_t_yaw": "yaw",
    "geoff3d_p_yaw": "GA-Sim",
    "geoff3d_t_woalign": "none",
    "geoff3d_p_woalign": "None",
}

DATASET_SHORT = {
    "UseGeoWAI": "UseGeo",
    "UAVFF3D": "UAVFF3D",
    "A3DRealWAI": "UAVFF3D",
}

# File-name aliases: when searching per-scene JSONs, also try these names.
_DATASET_FILE_ALIASES = {
    "A3DRealWAI": ["A3DRealWAI", "UAVFF3D"],
    "UAVFF3D": ["UAVFF3D", "A3DRealWAI"],
}

METRIC_SHORT = {
    "camera_scale_gt": "GT Scale (m)",
    "camera_scale_pred": "Pred Scale (m)",
    "camera_scale_ratio": "Scale Ratio",
    "camera_scale_rel_error": "Scale Rel Err",
    "camera_scale_log_error": "Scale Log Err",
    "abs_pose_ate": "ATE (m)",
    "abs_pose_auc_5deg": "AUC@5°",
    "abs_pose_rot_mae_deg": "Rot MAE (°)",
    "abs_fused_pc_chamfer_l1": "Chamfer (m)",
    "abs_fused_pc_precision": "Prec",
    "abs_fused_pc_recall": "Rec",
    "abs_fused_pc_f1": "F1",
    "abs_pointmap_mae": "PM MAE (m)",
    "abs_pointmap_rmse": "PM RMSE (m)",
    "abs_depth_mae_scale_aligned": "AbsRel",
    "abs_depth_rmse_scale_aligned": "RMSE",
    "abs_depth_rel_scale_aligned": "Rel",
    "abs_depth_delta1_scale_aligned": "δ<1.25",
    "rel_pointmap_abs": "Rel PM Abs",
    "rel_pointmap_delta_1p03": "Rel PM δ<1.03",
    "rel_pose_ate": "Rel ATE",
    "rel_pose_auc_5deg": "Rel AUC@5°",
    "rel_depth_abs": "Rel Depth Abs",
    "rel_depth_delta_1p03": "Rel Depth δ<1.03",
    "ray_dir_mean_angle_deg": "Ray Err (°)",
    "sim3_valid": "Success Rate (%)",
    "eval_valid": "Eval Valid",
    "eval_failure_rate": "Eval Fail",
}

BOOKKEEPING_METRICS = {
    "sim3_valid",
    "eval_valid",
    "failure_rate",
    "eval_failure_rate",
    "total_count",
    "failure_count",
}


def find_num_view_dirs(base_dir):
    if not os.path.isdir(base_dir):
        return []
    dirs = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and name.startswith("uav_dense_") and name.endswith("_view"):
            parts = name.replace("uav_dense_", "").replace("_view", "")
            try:
                num = int(parts)
            except ValueError:
                continue
            dirs.append((num, full))
    dirs.sort(key=lambda x: x[0])
    return dirs


def auto_discover_methods(num_view_dirs):
    methods = set()
    for _, num_view_dir in num_view_dirs:
        if not os.path.isdir(num_view_dir):
            continue
        for name in os.listdir(num_view_dir):
            full = os.path.join(num_view_dir, name)
            if os.path.isdir(full) and os.path.isfile(os.path.join(full, "per_dataset_results.json")):
                methods.add(name)
    return sorted(methods)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_method_metrics(num_view_dir, method_name):
    json_path = os.path.join(num_view_dir, method_name, "per_dataset_results.json")
    if not os.path.isfile(json_path):
        return None
    return load_json(json_path)


def load_per_scene_metrics(num_view_dir, method_name, dataset_key):
    """
    Load <dataset>_per_scene_results.json for one method.
    Tries exact match first, then tries file-name aliases (UAVFF3D ↔ A3DRealWAI).
    """
    method_dir = os.path.join(num_view_dir, method_name)
    if not os.path.isdir(method_dir):
        return None, None

    aliases = _DATASET_FILE_ALIASES.get(dataset_key, [dataset_key])
    suffix = "_per_scene_results.json"

    for alias in aliases:
        exact_path = os.path.join(method_dir, f"{alias}_per_scene_results.json")
        if os.path.isfile(exact_path):
            return load_json(exact_path), exact_path

    # fallback: substring match in directory listing
    for name in os.listdir(method_dir):
        if name.endswith(suffix):
            for alias in aliases:
                if alias in name:
                    return load_json(os.path.join(method_dir, name)), os.path.join(method_dir, name)

    return None, None


def extract_dataset_metrics(data, metric_keys, dataset_key):
    """
    From loaded per_dataset_results.json, extract metric values for a given
    dataset (e.g. "UseGeoWAI" or "A3DRealWAI"). Also tries aliases.
    """
    aliases = _DATASET_FILE_ALIASES.get(dataset_key, [dataset_key])
    for key in data:
        for alias in aliases:
            if alias in key:
                row = data[key]
                out = {}
                for mk in metric_keys:
                    out[mk] = row.get(mk, float("nan"))
                return out
    return None


def to_float(v):
    if v is None:
        return float("nan")
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def get_sample_value(scene_row, metric_key, sample_idx):
    """
    per_scene_results format:
        scene_row[metric_key] = [sample_0, sample_1, ...]

    Return NaN if the metric/sample does not exist or is not finite-convertible.
    """
    if not isinstance(scene_row, dict):
        return float("nan")

    values = scene_row.get(metric_key, None)
    if values is None:
        return float("nan")

    if isinstance(values, (list, tuple)):
        if sample_idx < 0 or sample_idx >= len(values):
            return float("nan")
        return to_float(values[sample_idx])

    # Legacy scalar fallback. Treat scalar as a single-sample scene.
    if sample_idx == 0:
        return to_float(values)
    return float("nan")


def get_scene_sample_count(scene_row):
    """
    Infer how many random samples are stored under one scene.
    Normally all metric lists have the same length. Use max length to avoid
    under-counting when some metric is missing/shorter due to older results.
    """
    if not isinstance(scene_row, dict):
        return 0

    n = 0
    for values in scene_row.values():
        if isinstance(values, (list, tuple)):
            n = max(n, len(values))
        elif values is not None:
            n = max(n, 1)
    return n


def sample_is_success(scene_row, sample_idx, metric_keys, valid_keys, min_valid=0.5):
    """
    Decide whether a single random sample is valid.

    Rules:
    1. If sim3_valid/eval_valid exists, this sample must be >= min_valid.
    2. Requested non-bookkeeping metrics must be finite at this sample index.
    """
    if not isinstance(scene_row, dict):
        return False

    for vk in valid_keys:
        if vk in scene_row:
            vv = get_sample_value(scene_row, vk, sample_idx)
            if not math.isfinite(vv) or vv < min_valid:
                return False

    required_metrics = [mk for mk in metric_keys if mk not in BOOKKEEPING_METRICS]
    for mk in required_metrics:
        vv = get_sample_value(scene_row, mk, sample_idx)
        if not math.isfinite(vv):
            return False

    return True


def format_value(v, metric_key=None):
    if v is None:
        return "-"
    if isinstance(v, float) and not math.isfinite(v):
        return "-"
    if isinstance(v, (int, float)):
        fv = float(v)
        if metric_key in BOOKKEEPING_METRICS:
            fv = fv * 100.0
            precision = 1
        else:
            precision = 2
        return f"{fv:.{precision}f}"
    return str(v)


def average_per_dataset_results(args, num_view_dirs, metric_keys, dataset_keys):
    results = OrderedDict()
    for method in args.methods:
        results[method] = {}
        for ds in dataset_keys:
            results[method][ds] = {mk: [] for mk in metric_keys}

        for _, num_view_dir in num_view_dirs:
            data = load_method_metrics(num_view_dir, method)
            if data is None:
                continue
            for ds in dataset_keys:
                ds_metrics = extract_dataset_metrics(data, metric_keys, ds)
                if ds_metrics is None:
                    continue
                for mk in metric_keys:
                    results[method][ds][mk].append(ds_metrics[mk])

    averaged = OrderedDict()
    for method in args.methods:
        averaged[method] = {}
        for ds in dataset_keys:
            averaged[method][ds] = {}
            for mk in metric_keys:
                vals = [to_float(v) for v in results[method][ds][mk]]
                vals = [v for v in vals if math.isfinite(v)]
                averaged[method][ds][mk] = sum(vals) / len(vals) if vals else float("nan")
    return averaged


def average_common_success_results(args, num_view_dirs, metric_keys, dataset_keys):
    """
    Fair comparison on common successful sample-level records.

    Unit of comparison:
        (num_views, dataset, scene_name, sample_idx)

    Not just scene_name, because each scene has multiple random sampled sets.
    """
    valid_keys = [x.strip() for x in args.common_valid_keys.split(",") if x.strip()]

    results = OrderedDict()
    for method in args.methods:
        results[method] = {}
        for ds in dataset_keys:
            results[method][ds] = {mk: [] for mk in metric_keys}

    common_sample_counts = OrderedDict()
    missing_per_scene = []

    for num_views, num_view_dir in num_view_dirs:
        common_sample_counts[num_views] = OrderedDict()

        for ds in dataset_keys:
            per_method_scene = OrderedDict()
            missing_methods = []

            for method in args.methods:
                data, _ = load_per_scene_metrics(num_view_dir, method, ds)
                if data is None:
                    missing_methods.append(method)
                else:
                    per_method_scene[method] = data

            if missing_methods:
                missing_per_scene.append({
                    "num_views": num_views,
                    "dataset": ds,
                    "missing_methods": missing_methods,
                })
                common_samples = []
            else:
                # First require the same scene to exist in every method.
                scene_sets = [set(data.keys()) for data in per_method_scene.values()]
                candidate_scenes = set.intersection(*scene_sets) if scene_sets else set()

                common_samples = []

                for scene in sorted(candidate_scenes):
                    # For a scene, each method may have N random samples.
                    # Use min count so sample_idx is present in all methods.
                    sample_counts = [
                        get_scene_sample_count(data.get(scene, {}))
                        for data in per_method_scene.values()
                    ]
                    max_common_samples = min(sample_counts) if sample_counts else 0

                    for sample_idx in range(max_common_samples):
                        ok = True

                        for method, data in per_method_scene.items():
                            if not sample_is_success(
                                data.get(scene, {}),
                                sample_idx=sample_idx,
                                metric_keys=metric_keys,
                                valid_keys=valid_keys,
                                min_valid=args.common_min_valid,
                            ):
                                ok = False
                                break

                        if ok:
                            common_samples.append((scene, sample_idx))

                # Accumulate metric values on exactly the same common samples.
                for method, data in per_method_scene.items():
                    for scene, sample_idx in common_samples:
                        scene_row = data.get(scene, {})
                        for mk in metric_keys:
                            vv = get_sample_value(scene_row, mk, sample_idx)
                            if math.isfinite(vv):
                                results[method][ds][mk].append(vv)

            common_sample_counts[num_views][ds] = len(common_samples)

    averaged = OrderedDict()
    for method in args.methods:
        averaged[method] = {}
        for ds in dataset_keys:
            averaged[method][ds] = {}
            for mk in metric_keys:
                vals = [to_float(v) for v in results[method][ds][mk]]
                vals = [v for v in vals if math.isfinite(v)]
                averaged[method][ds][mk] = sum(vals) / len(vals) if vals else float("nan")

    meta = {
        "common_sample_counts": common_sample_counts,
        "missing_per_scene": missing_per_scene,
        "valid_keys": valid_keys,
        "min_valid": args.common_min_valid,
    }
    return averaged, meta


def build_cvpr_table(averaged, methods, dataset_keys, metric_keys):
    method_short = OrderedDict()
    for m in methods:
        method_short[m] = METHOD_SHORT.get(m, m)

    dataset_short = [DATASET_SHORT.get(ds, ds) for ds in dataset_keys]
    metric_short = [METRIC_SHORT.get(mk, mk) for mk in metric_keys]

    header_top = ["Method", "Alignment"]
    header_bot = ["", ""]
    for ds_name in dataset_short:
        for mk_name in metric_short:
            header_top.append(ds_name)
            header_bot.append(f"{mk_name}↓")

    col_widths = []
    col_widths.append(max(len("Method"), max(len(v) for v in method_short.values()) + 1))
    col_widths.append(
        max(
            len("Alignment"),
            max(len(METHOD_ALIGNMENT.get(m, "Unknown")) for m in methods),
        ) + 1
    )
    for i in range(2, len(header_top)):
        w = max(len(header_top[i]), len(header_bot[i]))
        for method in methods:
            ds_idx = (i - 2) // len(metric_keys)
            mk_idx = (i - 2) % len(metric_keys)
            mk = metric_keys[mk_idx]
            v = format_value(averaged[method][dataset_keys[ds_idx]][mk], metric_key=mk)
            w = max(w, len(v))
        col_widths.append(w + 1)

    line_top = "  ".join(h.ljust(w) for h, w in zip(header_top, col_widths))
    line_bot = "  ".join(h.ljust(w) for h, w in zip(header_bot, col_widths))
    line_sep = "-" * len(line_top)

    text_lines = [line_top, line_bot, line_sep]
    for method in methods:
        row = [method_short[method], METHOD_ALIGNMENT.get(method, "Unknown")]
        for ds in dataset_keys:
            for mk in metric_keys:
                row.append(format_value(averaged[method][ds][mk], metric_key=mk))
        text_lines.append("  ".join(str(r).ljust(w) for r, w in zip(row, col_widths)))

    md_lines = []
    md_lines.append("| " + " | ".join(header_top) + " |")
    md_lines.append("| " + " | ".join(header_bot) + " |")
    md_lines.append("| " + " | ".join("---" for _ in header_top) + " |")
    for method in methods:
        md_row = [method_short[method], METHOD_ALIGNMENT.get(method, "Unknown")]
        for ds in dataset_keys:
            for mk in metric_keys:
                md_row.append(format_value(averaged[method][ds][mk], metric_key=mk))
        md_lines.append("| " + " | ".join(str(r) for r in md_row) + " |")

    return "\n".join(text_lines), "\n".join(md_lines)


def build_common_count_markdown(common_meta):
    lines = ["| Num views | Dataset | Common successful samples |", "| --- | --- | ---: |"]
    counts = common_meta["common_sample_counts"]
    for num_views, ds_counts in counts.items():
        for ds, count in ds_counts.items():
            lines.append(f"| {num_views} | {DATASET_SHORT.get(ds, ds)} | {count} |")
    return "\n".join(lines)


def print_common_count_summary(common_meta):
    print()
    print("Common-success sample counts:")
    counts = common_meta["common_sample_counts"]
    for num_views, ds_counts in counts.items():
        for ds, count in ds_counts.items():
            print(f"  - {num_views}-view / {ds}: {count}")

    missing = common_meta.get("missing_per_scene", [])
    if missing:
        print()
        print("WARNING: Missing per-scene JSON for some method/dataset/num_view combinations:")
        for item in missing:
            methods = ", ".join(item["missing_methods"])
            print(f"  - {item['num_views']}-view / {item['dataset']}: {methods}")


def main():
    parser = argparse.ArgumentParser(description="Collect benchmark metrics")
    parser.add_argument(
        "methods", nargs="*",
        default=[
            "pi3x_ft",
            "geoff3d_p_woworld",
            "geoff3d_p_wogravity",
            "geoff3d_p_yaw",
            "geoff3d_p",
            "geoff3d_p_woalign",
        ],
        help="Method names (sub-directory names under each num_view dir)",
    )
    parser.add_argument("--base-dir", default="experiments/pose_align/benchmarking")
    parser.add_argument("--num-views", default=None)
    parser.add_argument(
        "--metrics",
        default="abs_pose_ate,abs_fused_pc_chamfer_l1",
        help="Comma-separated metric keys. Validity keys such as sim3_valid/eval_valid are controlled by --common-valid-keys.",
    )
    parser.add_argument(
        "--datasets", default="UseGeoWAI,A3DRealWAI",
        help="Comma-separated dataset substrings to match in JSON keys (UseGeoWAI, A3DRealWAI, or UAVFF3D)",
    )
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--output-md", default="./experiments/pose_align/benchmarking/benchmark_results.md")
    parser.add_argument(
        "--common-success-table", action="store_true",
        help="Optionally generate the legacy common-success intersection table",
    )
    parser.add_argument(
        "--no-common-success-table", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--common-valid-keys",
        default="sim3_valid,eval_valid",
        help=(
            "Sample-level validity metric keys. If a key exists in one method's per-scene JSON, "
            "the value at the same sample_idx must be >= --common-min-valid. "
            "Use sim3_valid for aligned benchmark.py and eval_valid for absolute-world benchmark_absolute_world.py."
        ),
    )
    parser.add_argument("--common-min-valid", type=float, default=0.5)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "../../.."))
    base_dir = os.path.join(repo_root, args.base_dir)
    if not os.path.isdir(base_dir):
        base_dir = os.path.abspath(args.base_dir)

    metric_keys = [m.strip() for m in args.metrics.split(",") if m.strip()]
    dataset_keys = [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_num_view_dirs = find_num_view_dirs(base_dir)
    if not all_num_view_dirs:
        print(f"ERROR: No uav_dense_*_view/ dirs found under {base_dir}", file=sys.stderr)
        sys.exit(1)

    if args.num_views is not None:
        selected_nums = set(int(n.strip()) for n in args.num_views.split(",") if n.strip())
        num_view_dirs = [(n, d) for n, d in all_num_view_dirs if n in selected_nums]
        if not num_view_dirs:
            print(f"ERROR: None of requested num_views {sorted(selected_nums)} found. "
                  f"Available: {sorted(n for n, _ in all_num_view_dirs)}", file=sys.stderr)
            sys.exit(1)
    else:
        num_view_dirs = all_num_view_dirs

    print(f"Base dir:       {base_dir}")
    print(f"Num views:      {sorted(n for n, _ in num_view_dirs)}")
    print(f"Methods:        {args.methods}")
    print(f"Metrics:        {metric_keys}")
    print(f"Datasets:       {dataset_keys}")
    print()

    # ---- Original per-dataset table ----
    averaged = average_per_dataset_results(args, num_view_dirs, metric_keys, dataset_keys)
    text_table, md_table = build_cvpr_table(averaged, args.methods, dataset_keys, metric_keys)

    print("Original average table")
    print(text_table)
    print()
    print("(↓ = lower is better)")

    md_sections = [
        "# UAV Dense N-View Benchmark Results", "",
        f"- **Num views**: {sorted(n for n, _ in num_view_dirs)}",
        f"- **Metrics**: {metric_keys}",
        f"- **Datasets**: {dataset_keys}", "",
        "## Original average table", "",
        "Each method's per_dataset_results.json is averaged independently.", "",
        md_table, "",
        "*(↓ = lower is better)*",
    ]

    common_averaged = None
    common_meta = None

    if args.common_success_table and not args.no_common_success_table:
        common_averaged, common_meta = average_common_success_results(
            args, num_view_dirs, metric_keys, dataset_keys,
        )
        # For bookkeeping metrics (sim3_valid, eval_valid, etc.), use the
        # original per_dataset_results values instead of recomputing from samples.
        for mk in metric_keys:
            if mk in BOOKKEEPING_METRICS:
                for method in args.methods:
                    for ds in dataset_keys:
                        common_averaged[method][ds][mk] = averaged[method][ds][mk]

        common_text_table, common_md_table = build_cvpr_table(
            common_averaged, args.methods, dataset_keys, metric_keys,
        )

        print()
        print("Common-success average table")
        print(common_text_table)
        print()
        print("(↓ = lower is better)")
        print("Only sample-level records (scene, sample_idx) successful for ALL methods are used.")
        print_common_count_summary(common_meta)

        md_sections.extend([
            "", "## Common-success average table", "",
            "Only sample-level records `(scene, sample_idx)` that are present and successful for **all** compared methods are averaged.", "",
            f"- **Validity keys**: {common_meta['valid_keys']}",
            f"- **Minimum validity value**: {common_meta['min_valid']}", "",
            common_md_table, "",
            "*(↓ = lower is better)*", "",
            "### Common successful sample counts", "",
            build_common_count_markdown(common_meta),
        ])

    if args.print_json:
        json_out = OrderedDict()
        json_out["original_average"] = averaged
        if common_averaged is not None:
            json_out["common_success_average"] = common_averaged
            json_out["common_success_meta"] = common_meta
        print()
        print(json.dumps(json_out, indent=2, ensure_ascii=False))

    if args.output_md:
        out_path = args.output_md
        if not os.path.isabs(out_path):
            out_path = os.path.join(repo_root, out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        md_output = "\n".join(md_sections) + "\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_output)
        print()
        print(f"Markdown tables saved to: {out_path}")


if __name__ == "__main__":
    main()
