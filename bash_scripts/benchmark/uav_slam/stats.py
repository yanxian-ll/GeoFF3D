#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计 UAV-SLAM benchmark 结果，按数据集生成 markdown 表格。

用法:
    python stats.py [--methods NAME,...] [--spatial-root DIR] [--optim-root DIR]
                    [--output FILE] [-q] METRIC [METRIC ...]
    python stats.py --paper-large-scale [--output FILE] [-q] [METRIC ...]

    如果不指定 --methods，则自动扫描 spatial_root 和 optim_root 下所有方法目录。

    每个 METRIC 可以用简写名（如 ate_rmse）或完整 JSON 路径（如 pose.ate.ate_rmse）。
    使用 --list-metrics 查看所有可用指标。

示例:
    python stats.py -m pi3x,geoff3d,vggt_long ate_rmse acc_mean
    python stats.py -m "pi3x:Pi3X,vggt_long:VGGT-Long" ate_rmse rot_deg_rmse fscore_1.0
    python stats.py --paper-large-scale
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 数据集配置（内置，便于手动调整）
# ---------------------------------------------------------------------------
# 每个数据集包含:
#   display: 表格中显示的名称
#   dir_name: 在 outputs 目录下的文件夹名
#   scenes: 场景列表（按顺序）
DATASETS: Dict[str, Dict[str, Any]] = {
    "usegeo": {
        "display": "UseGeo",
        "dir_name": "usegeo",
        "scenes": [
            "dataset1",
            "dataset2",
            "dataset3",
        ],
    },
    "uavff3d": {
        "display": "UAVFF3D",
        "dir_name": "uavff3d_real",
        "scenes": [
            "nanfang_ndir2",
            "yanghaitang_ndir2",
            "xiaoxiang_ndir2",
            "nanfang_part0_ndir",
            "xiaoxiang_part3_ndir",
            "yanghaitang_part0_ndir",
        ],
    },
    "uavscenes": {
        "display": "UAVScenes",
        "dir_name": "uavscenes",
        "scenes": [
            "interval5_AMtown01",
            "interval5_AMtown02",
            "interval5_AMvalley01",
            "interval5_AMvalley02",
            "interval5_HKairport01",
            "interval5_HKairport02",
            "interval5_HKisland01",
            "interval5_HKisland02",
        ],
    },
}

# ---------------------------------------------------------------------------
# 指标简写 → JSON 路径映射
# ---------------------------------------------------------------------------
# 格式: 简写名 -> (JSON 路径元组, 显示名)
# 同时也支持直接用点分隔的 JSON 路径，如 pose.ate.ate_rmse
METRIC_REGISTRY: Dict[str, Tuple[Tuple[str, ...], str]] = {
    # ── ATE ──
    "ate_rmse": (("pose", "ate", "ate_rmse"), "ATE RMSE ↓"),
    "ate_mean": (("pose", "ate", "ate_mean"), "ATE Mean ↓"),
    "ate_median": (("pose", "ate", "ate_median"), "ATE Median ↓"),
    "ate_p90": (("pose", "ate", "ate_p90"), "ATE P90 ↓"),
    "ate_p95": (("pose", "ate", "ate_p95"), "ATE P95 ↓"),
    # ── 旋转误差 ──
    "rot_deg_rmse": (("pose", "rotation", "rot_deg_rmse"), "Rot RMSE(°) ↓"),
    "rot_deg_mean": (("pose", "rotation", "rot_deg_mean"), "Rot Mean(°) ↓"),
    "rot_deg_median": (("pose", "rotation", "rot_deg_median"), "Rot Median(°) ↓"),
    "rot_deg_p90": (("pose", "rotation", "rot_deg_p90"), "Rot P90(°) ↓"),
    # ── RPE k=1 ──
    "rpe_k1_trans_rmse": (("pose", "rpe", "k1", "translation", "rpe_trans_rmse"), "RPE k1 Trans RMSE ↓"),
    "rpe_k1_trans_mean": (("pose", "rpe", "k1", "translation", "rpe_trans_mean"), "RPE k1 Trans Mean ↓"),
    "rpe_k1_trans_median": (("pose", "rpe", "k1", "translation", "rpe_trans_median"), "RPE k1 Trans Med ↓"),
    "rpe_k1_rot_rmse": (("pose", "rpe", "k1", "rotation", "rpe_rot_deg_rmse"), "RPE k1 Rot RMSE(°) ↓"),
    "rpe_k1_rot_mean": (("pose", "rpe", "k1", "rotation", "rpe_rot_deg_mean"), "RPE k1 Rot Mean(°) ↓"),
    # ── RPE k=5 ──
    "rpe_k5_trans_rmse": (("pose", "rpe", "k5", "translation", "rpe_trans_rmse"), "RPE k5 Trans RMSE ↓"),
    "rpe_k5_trans_mean": (("pose", "rpe", "k5", "translation", "rpe_trans_mean"), "RPE k5 Trans Mean ↓"),
    "rpe_k5_trans_median": (("pose", "rpe", "k5", "translation", "rpe_trans_median"), "RPE k5 Trans Med ↓"),
    "rpe_k5_rot_rmse": (("pose", "rpe", "k5", "rotation", "rpe_rot_deg_rmse"), "RPE k5 Rot RMSE(°) ↓"),
    "rpe_k5_rot_mean": (("pose", "rpe", "k5", "rotation", "rpe_rot_deg_mean"), "RPE k5 Rot Mean(°) ↓"),
    # ── RPE k=10 ──
    "rpe_k10_trans_rmse": (("pose", "rpe", "k10", "translation", "rpe_trans_rmse"), "RPE k10 Trans RMSE ↓"),
    "rpe_k10_trans_mean": (("pose", "rpe", "k10", "translation", "rpe_trans_mean"), "RPE k10 Trans Mean ↓"),
    "rpe_k10_trans_median": (("pose", "rpe", "k10", "translation", "rpe_trans_median"), "RPE k10 Trans Med ↓"),
    "rpe_k10_rot_rmse": (("pose", "rpe", "k10", "rotation", "rpe_rot_deg_rmse"), "RPE k10 Rot RMSE(°) ↓"),
    "rpe_k10_rot_mean": (("pose", "rpe", "k10", "rotation", "rpe_rot_deg_mean"), "RPE k10 Rot Mean(°) ↓"),
    # ── Accuracy (Pred→GT) ──
    "acc_mean": (("point_cloud", "accuracy_pred_to_gt", "acc_mean"), "Acc Mean ↓"),
    "acc_rmse": (("point_cloud", "accuracy_pred_to_gt", "acc_rmse"), "Acc RMSE ↓"),
    "acc_median": (("point_cloud", "accuracy_pred_to_gt", "acc_median"), "Acc Median ↓"),
    "acc_p90": (("point_cloud", "accuracy_pred_to_gt", "acc_p90"), "Acc P90 ↓"),
    "acc_p95": (("point_cloud", "accuracy_pred_to_gt", "acc_p95"), "Acc P95 ↓"),
    # ── Completeness (GT→Pred) ──
    "comp_mean": (("point_cloud", "completeness_gt_to_pred", "comp_mean"), "Comp Mean ↓"),
    "comp_rmse": (("point_cloud", "completeness_gt_to_pred", "comp_rmse"), "Comp RMSE ↓"),
    "comp_median": (("point_cloud", "completeness_gt_to_pred", "comp_median"), "Comp Median ↓"),
    "comp_p90": (("point_cloud", "completeness_gt_to_pred", "comp_p90"), "Comp P90 ↓"),
    "comp_p95": (("point_cloud", "completeness_gt_to_pred", "comp_p95"), "Comp P95 ↓"),
    # ── Chamfer ──
    "chamfer_l1": (("point_cloud", "chamfer_l1"), "Chamfer L1 ↓"),
    "chamfer_l2": (("point_cloud", "chamfer_l2"), "Chamfer L2 ↓"),
    # ── F-score ──
    "fscore_0.5": (("point_cloud", "fscore", "0.5", "fscore"), "F-score@0.5 ↑"),
    "fscore_1.0": (("point_cloud", "fscore", "1.0", "fscore"), "F-score@1.0 ↑"),
    "fscore_2.0": (("point_cloud", "fscore", "2.0", "fscore"), "F-score@2.0 ↑"),
    "fscore_5.0": (("point_cloud", "fscore", "5.0", "fscore"), "F-score@5.0 ↑"),
    "precision_1.0": (("point_cloud", "fscore", "1.0", "precision"), "Precision@1.0 ↑"),
    "recall_1.0": (("point_cloud", "fscore", "1.0", "recall"), "Recall@1.0 ↑"),
    "outlier_1.0": (("point_cloud", "fscore", "1.0", "outlier_ratio"), "Outlier@1.0 ↓"),
    # ── 对齐参数 ──
    "pose_align_scale": (("pose", "alignment", "scale"), "Pose Align Scale"),
    "points_align_scale": (("point_cloud", "alignment", "scale"), "PC Align Scale"),
    "pose_matches": (("pose", "num_matches"), "Pose Matches"),
    "pose_valid": (("pose", "valid"), "Pose Valid"),
    "points_valid": (("point_cloud", "valid"), "PC Valid"),
}


# ---------------------------------------------------------------------------
# 论文 4.2 Large-Scale Reconstruction 表格配置
# ---------------------------------------------------------------------------
# 不修改 DATASETS：这里单独定义论文表格里的场景简称和方法顺序。
@dataclass(frozen=True)
class PaperScene:
    display: str
    dataset_dir: str
    scene: str


@dataclass(frozen=True)
class PaperMethod:
    display: str
    group: str
    method: str
    kind: str


PAPER_TABLE5_SCENES: List[PaperScene] = [
    PaperScene("YHT-1", "uavff3d_real", "yanghaitang_ndir2"),
    PaperScene("YHT-2", "uavff3d_real", "yanghaitang_part0_ndir"),
    PaperScene("NF-1", "uavff3d_real", "nanfang_ndir2"),
    PaperScene("NF-2", "uavff3d_real", "nanfang_part0_ndir"),
    PaperScene("XX-1", "uavff3d_real", "xiaoxiang_ndir2"),
    PaperScene("XX-2", "uavff3d_real", "xiaoxiang_part3_ndir"),
    PaperScene("D1", "usegeo", "dataset1"),
    PaperScene("D2", "usegeo", "dataset2"),
    PaperScene("D3", "usegeo", "dataset3"),
]

PAPER_TABLE6_SCENES: List[PaperScene] = [
    PaperScene("Town01", "uavscenes", "interval5_AMtown01"),
    PaperScene("Town02", "uavscenes", "interval5_AMtown02"),
    PaperScene("Valley01", "uavscenes", "interval5_AMvalley01"),
    PaperScene("Valley02", "uavscenes", "interval5_AMvalley02"),
    PaperScene("Airport01", "uavscenes", "interval5_HKairport01"),
    PaperScene("Airport02", "uavscenes", "interval5_HKairport02"),
    PaperScene("Island01", "uavscenes", "interval5_HKisland01"),
    PaperScene("Island02", "uavscenes", "interval5_HKisland02"),
]

PAPER_TABLE5_METHODS: List[PaperMethod] = [
    PaperMethod("VGGT + chunking", "", "vggt", "spatial"),
    PaperMethod("VGGT-FT + chunking", "", "vggt_ft", "spatial"),
    PaperMethod("Pi3X + chunking", "", "pi3x", "spatial"),
    PaperMethod(
        "Pi3X-FT + chunking (GNSS noise)",
        "",
        "pi3x_ft_gnss_perturb",
        "spatial",
    ),
    PaperMethod("Pi3X-FT + chunking", "", "pi3x_ft", "spatial"),
    PaperMethod("Ours", "", "geoff3d", "spatial"),
    PaperMethod(
        "Ours (GNSS noise)",
        "",
        "geoff3d_gnss_perturb",
        "spatial",
    ),
]

PAPER_TABLE6_METHODS: List[PaperMethod] = [
    PaperMethod("DROID-SLAM", "optim", "droid_slam", "optim"),
    PaperMethod("VGGT-SLAM (Sim3)", "optim", "vggt_slam_sim3", "optim"),
    PaperMethod("VGGT-SLAM-FT (Sim3)", "optim", "vggt_slam_sim3_ft", "optim"),
    PaperMethod("VGGT-SLAM (SL4)", "optim", "vggt_slam_sl4", "optim"),
    PaperMethod("VGGT-SLAM-FT (SL4)", "optim", "vggt_slam_sl4_ft", "optim"),
    PaperMethod("VGGT-SLAM 2.0", "optim", "vggt_slam2.0", "optim"),
    PaperMethod("VGGT-SLAM 2.0-FT", "optim", "vggt_slam2.0_ft", "optim"),
    PaperMethod("VGGT-Long", "optim", "vggt_long", "optim"),
    PaperMethod("VGGT-Long-FT", "optim", "vggt_long_ft", "optim"),
    PaperMethod("CUT3R", "online", "cut3r", "stream"),
    PaperMethod("StreamVGGT", "online", "streamvggt", "stream"),
    PaperMethod("InfiniteVGGT", "online", "infinitevggt", "stream"),
    PaperMethod("Stream3R", "online", "stream3r", "stream"),
    PaperMethod("TTT3R", "online", "ttt3r", "stream"),
    PaperMethod("Wint3R", "online", "wint3r", "stream"),
    PaperMethod("LingBot-Map", "online", "lingbot-map", "stream"),
    PaperMethod("VGGT + Ours", "offline", "vggt", "spatial"),
    PaperMethod("VGGT-FT + Ours", "offline", "vggt_ft", "spatial"),
    PaperMethod("Pi3X + Ours", "offline", "pi3x", "spatial"),
    PaperMethod(
        "Pi3X-FT + Ours (GNSS noise)",
        "offline",
        "pi3x_ft_gnss_perturb",
        "spatial",
    ),
    PaperMethod("Pi3X-FT + Ours", "offline", "pi3x_ft", "spatial"),
    PaperMethod("Ours", "offline", "geoff3d", "spatial"),
    PaperMethod(
        "Ours (GNSS noise)",
        "offline",
        "geoff3d_gnss_perturb",
        "spatial",
    ),
]

PAPER_DEFAULT_METRICS = ("acc_mean", "comp_mean", "fscore_5.0")

# Streaming methods currently exposed by bash_scripts/benchmark/uav_slam/stream.
# They are listed explicitly so methods that have not run yet still appear as
# empty rows instead of disappearing during output-directory discovery.
STREAM_METHODS: List[PaperMethod] = [
    PaperMethod("LingBot-Map", "online", "lingbot-map", "stream"),
    PaperMethod("StreamVGGT", "online", "streamvggt", "stream"),
    PaperMethod("Stream3R", "online", "stream3r", "stream"),
    PaperMethod("TTT3R", "online", "ttt3r", "stream"),
]

METHOD_DISPLAY_NAMES: Dict[str, str] = {
    method.method: method.display
    for method in [
        *PAPER_TABLE5_METHODS,
        *PAPER_TABLE6_METHODS,
        *STREAM_METHODS,
    ]
}


def resolve_metric(raw: str) -> Tuple[Tuple[str, ...], str]:
    """将指标简写或点分隔路径解析为 (JSON 路径元组, 显示名)。"""
    if raw in METRIC_REGISTRY:
        return METRIC_REGISTRY[raw]
    # 尝试解析为点分隔的 JSON 路径
    parts = tuple(raw.strip().split("."))
    if len(parts) >= 1:
        return parts, raw
    raise ValueError(f"未知指标: {raw!r}，使用 --list-metrics 查看所有可用指标")


def nested_get(obj: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    """从嵌套字典中按路径取值。"""
    cur: Any = obj
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def safe_float(value: Any) -> Optional[float]:
    """安全转换为 float，失败返回 None。"""
    if value is None:
        return None
    try:
        v = float(value)
        if v != v:  # NaN
            return None
        return v
    except (TypeError, ValueError):
        return None


def load_metrics(metrics_path: Path) -> Optional[Dict[str, Any]]:
    """加载 metrics.json，失败返回 None。"""
    if not metrics_path.is_file():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def discover_methods(
    spatial_root: Path,
    optim_root: Path,
    stream_root: Path,
) -> List[Tuple[str, Path]]:
    """自动发现所有方法目录。

    返回 [(方法名, 完整路径), ...]，按方法名排序。
    """
    methods: Dict[str, Path] = {}
    for root in (spatial_root, optim_root, stream_root):
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                if entry.name not in methods:
                    methods[entry.name] = entry

    for method in STREAM_METHODS:
        methods.setdefault(method.method, stream_root / method.method)

    return sorted(methods.items(), key=lambda x: x[0])


def display_name_for_method(name: str) -> str:
    """返回方法展示名，未知方法保留原名。"""
    return METHOD_DISPLAY_NAMES.get(name, name)


def parse_method_spec(raw: str) -> Tuple[str, str]:
    """解析方法规格: name 或 name:DisplayName。"""
    if ":" in raw:
        name, display = raw.split(":", 1)
        return name.strip(), display.strip()
    return raw.strip(), raw.strip()


def format_value(val: Optional[float], precision: int = 4) -> str:
    """格式化数值为表格字符串。"""
    if val is None:
        return "-"
    return f"{val:.{precision}f}"


def metric_value_for_scene(
    method_root: Path,
    scene: PaperScene,
    metric_path: Tuple[str, ...],
    quiet: bool,
) -> Optional[float]:
    """读取论文表格中单个方法/场景/指标的值。"""
    metrics_path = method_root / scene.dataset_dir / scene.scene / "eval" / "metrics.json"
    metrics = load_metrics(metrics_path)
    if metrics is None:
        if not quiet:
            print(f"[WARN] 缺少: {metrics_path}", file=sys.stderr)
        return None
    return safe_float(nested_get(metrics, metric_path))


def build_paper_metric_table(
    title: str,
    scenes: List[PaperScene],
    methods: List[PaperMethod],
    root_by_kind: Dict[str, Path],
    metric_path: Tuple[str, ...],
    metric_display: str,
    quiet: bool,
    include_type: bool,
) -> str:
    """构建论文 4.2 中某个指标的一张 markdown 表格。"""
    header = ["Method"]
    if include_type:
        header.append("Type")
    header.extend(scene.display for scene in scenes)
    header.append("Avg")

    lines = [f"### {title}: {metric_display}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for method in methods:
        method_root = root_by_kind[method.kind] / method.method
        values = [
            metric_value_for_scene(
                method_root,
                scene,
                metric_path,
                quiet=quiet,
            )
            for scene in scenes
        ]
        valid_values = [v for v in values if v is not None]
        avg = sum(valid_values) / len(valid_values) if valid_values else None

        cells = [method.display]
        if include_type:
            cells.append(method.group)
        cells.extend(format_value(v) if v is not None else "" for v in values)
        cells.append(format_value(avg) if avg is not None else "")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def build_paper_large_scale_tables(
    metric_specs: List[Tuple[str, Tuple[str, ...], str]],
    root_by_kind: Dict[str, Path],
    quiet: bool,
) -> str:
    """生成论文 4.2 Large-Scale Reconstruction 的 Table 5/6 结果。"""
    output_parts = [
        "# Large-Scale Reconstruction Tables",
        "",
        "## Table 5. Aerial Mapping Blocks",
        "",
    ]

    for _raw_metric, metric_path, metric_display in metric_specs:
        output_parts.append(
            build_paper_metric_table(
                title="Table 5",
                scenes=PAPER_TABLE5_SCENES,
                methods=PAPER_TABLE5_METHODS,
                root_by_kind=root_by_kind,
                metric_path=metric_path,
                metric_display=metric_display,
                quiet=quiet,
                include_type=False,
            )
        )

    output_parts.extend(["## Table 6. UAVScenes", ""])
    for _raw_metric, metric_path, metric_display in metric_specs:
        output_parts.append(
            build_paper_metric_table(
                title="Table 6",
                scenes=PAPER_TABLE6_SCENES,
                methods=PAPER_TABLE6_METHODS,
                root_by_kind=root_by_kind,
                metric_path=metric_path,
                metric_display=metric_display,
                quiet=quiet,
                include_type=True,
            )
        )

    return "\n".join(output_parts)


def build_table(
    dataset_key: str,
    dataset_info: Dict[str, Any],
    methods: List[Tuple[str, str, Path]],
    metric_raw: str,
    metric_path: Tuple[str, ...],
    metric_display: str,
    quiet: bool,
) -> str:
    """为一个数据集的一个指标构建 markdown 表格。"""
    display_name = dataset_info["display"]
    dir_name = dataset_info["dir_name"]
    scenes = dataset_info["scenes"]

    # 表头
    header = ["Method"] + scenes + ["平均"]
    col_widths = [max(len(h), 6) for h in header]

    # 收集每行数据
    rows: List[Tuple[str, List[Optional[float]], Optional[float]]] = []
    for method_name, method_display, method_root in methods:
        scene_values: List[Optional[float]] = []
        for scene in scenes:
            metrics_path = method_root / dir_name / scene / "eval" / "metrics.json"
            metrics = load_metrics(metrics_path)
            if metrics is None:
                if not quiet:
                    print(f"[WARN] 缺少: {metrics_path}", file=sys.stderr)
                scene_values.append(None)
            else:
                val = nested_get(metrics, metric_path)
                scene_values.append(safe_float(val))
        # 计算平均（仅对有值的场景）
        valid_vals = [v for v in scene_values if v is not None]
        avg = sum(valid_vals) / len(valid_vals) if valid_vals else None
        rows.append((method_display, scene_values, avg))

    # 格式化
    lines: List[str] = []
    # 子标题: ### metric_name
    lines.append(f"### {metric_display}")
    lines.append("")

    # 表头行
    header_str = "| " + " | ".join(header) + " |"
    lines.append(header_str)

    # 分隔行
    sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    lines.append(sep)

    # 数据行
    for method_display, scene_values, avg in rows:
        cells = [method_display]
        for v in scene_values:
            cells.append(format_value(v))
        cells.append(format_value(avg))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def list_metrics():
    """打印所有可用指标。"""
    print("可用指标简写（也可用点分隔 JSON 路径）:")
    print()
    # 按类别分组
    categories = [
        ("── ATE ──", [k for k in METRIC_REGISTRY if k.startswith("ate_")]),
        ("── Rotation ──", [k for k in METRIC_REGISTRY if k.startswith("rot_deg_")]),
        ("── RPE k=1 ──", [k for k in METRIC_REGISTRY if k.startswith("rpe_k1_")]),
        ("── RPE k=5 ──", [k for k in METRIC_REGISTRY if k.startswith("rpe_k5_")]),
        ("── RPE k=10 ──", [k for k in METRIC_REGISTRY if k.startswith("rpe_k10_")]),
        ("── Accuracy ──", [k for k in METRIC_REGISTRY if k.startswith("acc_")]),
        ("── Completeness ──", [k for k in METRIC_REGISTRY if k.startswith("comp_")]),
        ("── Chamfer ──", [k for k in METRIC_REGISTRY if k.startswith("chamfer_")]),
        ("── F-score ──", [k for k in METRIC_REGISTRY if k.startswith("fscore_") or k.startswith("precision_") or k.startswith("recall_") or k.startswith("outlier_")]),
        ("── Alignment ──", [k for k in METRIC_REGISTRY if "align" in k or k in ("pose_matches", "pose_valid", "points_valid")]),
    ]
    seen: set = set()
    for cat_name, keys in categories:
        keys = [k for k in keys if k not in seen]
        if not keys:
            continue
        print(f"  {cat_name}")
        for k in sorted(keys):
            _, display = METRIC_REGISTRY[k]
            print(f"    {k:<28s}  {display}")
            seen.add(k)
        print()
    print("  也可直接使用点分隔路径，如: pose.ate.ate_rmse")
    print("  或: point_cloud.fscore.1.0.fscore")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent  # map-anything 根目录
    default_spatial = repo_root / "outputs" / "spatial"
    default_optim = repo_root / "outputs" / "optim"
    default_all_in_one = repo_root / "outputs" / "all_in_one"
    default_stream = repo_root / "outputs" / "stream"

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--methods", "-m",
        type=str,
        default=None,
        help="方法列表（逗号分隔），格式: name 或 name:DisplayName。不指定则自动扫描所有方法。",
    )
    parser.add_argument(
        "--spatial-root",
        type=Path,
        default=default_spatial,
        help=f"Spatial 方法输出根目录（默认: {default_spatial}）",
    )
    parser.add_argument(
        "--optim-root",
        type=Path,
        default=default_optim,
        help=f"Optim 方法输出根目录（默认: {default_optim}）",
    )
    parser.add_argument(
        "--all-in-one-root",
        type=Path,
        default=default_all_in_one,
        help=f"All-in-one 方法输出根目录（默认: {default_all_in_one}）",
    )
    parser.add_argument(
        "--stream-root",
        type=Path,
        default=default_stream,
        help=f"Streaming 方法输出根目录（默认: {default_stream}）",
    )
    parser.add_argument(
        "--paper-large-scale",
        action="store_true",
        help=(
            "生成论文 4.2 Large-Scale Reconstruction 的 Table 5/6。"
            "未跑完的方法或场景留空；不传 METRIC 时默认输出 acc_mean、comp_mean、fscore_1.0。"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="输出到文件（默认输出到 stdout）",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="列出所有可用指标简写后退出",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="抑制缺少指标文件的警告",
    )
    parser.add_argument(
        "metrics",
        nargs="*",
        help="要统计的指标（简写或点分隔路径），至少一个",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.list_metrics:
        list_metrics()
        return 0

    if not args.paper_large_scale and not args.metrics and not args.methods:
        args.paper_large_scale = True

    if args.paper_large_scale and not args.metrics:
        args.metrics = list(PAPER_DEFAULT_METRICS)

    if args.paper_large_scale:
        args.quiet = True
        if args.output is None:
            args.output = Path(__file__).resolve().parent / "large_scale_reconstruction_tables.md"

    if not args.metrics:
        print("错误: 请至少指定一个指标。使用 --list-metrics 查看可用指标。", file=sys.stderr)
        print("示例: python stats.py ate_rmse acc_mean fscore_1.0", file=sys.stderr)
        print("或: python stats.py --paper-large-scale", file=sys.stderr)
        return 1

    # 解析指标
    metric_specs: List[Tuple[str, Tuple[str, ...], str]] = []
    for raw_metric in args.metrics:
        try:
            path, display = resolve_metric(raw_metric)
            metric_specs.append((raw_metric, path, display))
        except ValueError as e:
            print(f"错误: {e}", file=sys.stderr)
            return 1

    if args.paper_large_scale:
        root_by_kind = {
            "all_in_one": args.all_in_one_root,
            "spatial": args.spatial_root,
            "optim": args.optim_root,
            "stream": args.stream_root,
        }
        result = build_paper_large_scale_tables(
            metric_specs=metric_specs,
            root_by_kind=root_by_kind,
            quiet=args.quiet,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result, encoding="utf-8")
            print(f"已保存到: {args.output}", file=sys.stderr)
        else:
            print(result)
        return 0

    # 解析方法
    if args.methods:
        method_list: List[Tuple[str, str, Path]] = []
        seen_names: set = set()
        for raw in args.methods.split(","):
            raw = raw.strip()
            if not raw:
                continue
            name, display = parse_method_spec(raw)
            if display == name:
                display = display_name_for_method(name)
            if name in seen_names:
                continue
            seen_names.add(name)
            # 查找方法目录
            found = None
            for root in (args.spatial_root, args.optim_root, args.stream_root):
                candidate = root / name
                if candidate.is_dir():
                    found = candidate
                    break
            if found is None:
                if name in {method.method for method in STREAM_METHODS}:
                    found = args.stream_root / name
                else:
                    print(
                        f"错误: 未找到方法目录: {name} "
                        f"(在 {args.spatial_root}、{args.optim_root} 或 {args.stream_root})",
                        file=sys.stderr,
                    )
                    return 1
            method_list.append((name, display, found))
    else:
        # 自动发现所有方法
        discovered = discover_methods(args.spatial_root, args.optim_root, args.stream_root)
        if not discovered:
            print("错误: 未发现任何方法目录。请用 --methods 手动指定。", file=sys.stderr)
            return 1
        method_list = [(name, display_name_for_method(name), path) for name, path in discovered]

    if not method_list:
        print("错误: 未指定任何方法。", file=sys.stderr)
        return 1

    # 生成表格
    output_parts: List[str] = []

    # 总体标题
    output_parts.append("# UAV-SLAM Benchmark 统计")
    output_parts.append("")

    for dataset_key, dataset_info in DATASETS.items():
        output_parts.append(f"## {dataset_info['display']}")
        output_parts.append("")

        for raw_metric, metric_path, metric_display in metric_specs:
            table = build_table(
                dataset_key=dataset_key,
                dataset_info=dataset_info,
                methods=method_list,
                metric_raw=raw_metric,
                metric_path=metric_path,
                metric_display=metric_display,
                quiet=args.quiet,
            )
            output_parts.append(table)

    result = "\n".join(output_parts)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
        print(f"已保存到: {args.output}", file=sys.stderr)
    else:
        print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
