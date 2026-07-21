#!/usr/bin/env python3
"""
根据评估指标替换满足条件的预测深度图为GT深度。

用法示例（数据集模式）：
    python scripts/replace_depth_by_metric.py \
        --dataset_root /path/to/dataset \
        --gt_subdir depth \
        --pred_subdir depth_complete \
        --output_subdir depth_complete_viz \
        --dry-run   # 先预览，不实际替换

可选参数：
    --ext .exr .png              # 深度图可能的扩展名（默认 ['.exr','.png']）
    --backup_dir ./backup         # 备份被替换的原始预测文件到此目录
    --yes                         # 自动确认替换，无需交互
    --dry-run                     # 只统计不替换
"""

import os
import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict
from glob import glob
import sys
from tqdm import tqdm

# ---------- 辅助函数（从原评估脚本借鉴） ----------
def get_file_list_by_stem(directory, extensions, recursive=False):
    """
    返回字典 {stem: 文件路径}，如果有多个同stem文件，取第一个并警告。
    """
    directory = Path(directory)
    if not directory.exists():
        return {}

    files = defaultdict(list)
    pattern = "**/*" if recursive else "*"
    for ext in extensions:
        for p in directory.glob(f"{pattern}{ext}"):
            if p.is_file():
                files[p.stem].append(p)

    result = {}
    for stem, paths in files.items():
        if len(paths) > 1:
            print(f"Warning: Multiple files with stem '{stem}' in {directory}: {[str(p) for p in paths]}. Using the first one.")
        result[stem] = str(paths[0])
    return result

# ---------- 主逻辑 ----------
def main():
    parser = argparse.ArgumentParser(description="Replace predicted depth with GT depth based on evaluation metrics.")
    parser.add_argument("--dataset_root", default="../../dataset/data/a3dscenes", help="Root directory containing scene subfolders.")
    parser.add_argument("--gt_subdir", default="depth", help="Subdirectory name for GT depth inside each scene (default: depth).")
    parser.add_argument("--pred_subdir", default="depth_complete", help="Subdirectory name for predicted depth inside each scene (default: depth_complete).")
    parser.add_argument("--output_subdir", default="depth_complete_viz", help="Subdirectory where evaluation JSON files are stored (default: depth_complete_viz).")
    parser.add_argument("--ext", nargs="+", default=[".exr", ".png"], help="Possible depth file extensions (default: .exr .png).")
    parser.add_argument("--backup_dir", help="If set, backup original predicted files to this directory before replacing.")
    parser.add_argument("--yes", action="store_true", help="Automatically answer yes to confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Only count files that would be replaced, without actually doing it.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        sys.exit(f"Error: dataset_root '{dataset_root}' does not exist.")

    # 收集所有场景（直接子目录）
    scenes = [p for p in dataset_root.iterdir() if p.is_dir()]
    if not scenes:
        print("No scene subfolders found.")
        return

    # 准备扩展名（确保以点开头）
    exts = [e if e.startswith('.') else '.' + e for e in args.ext]

    # 统计信息
    total_json = 0
    eligible_count = 0
    eligible_files = []  # 存储 (scene, stem, pred_path, gt_path)

    # 遍历每个场景
    for scene in tqdm(sorted(scenes)):
        scene_name = scene.name
        if scene_name in ["csu2026", "nanfang", "yanghaitang"]:
            continue

        # 评估结果所在目录
        eval_dir = scene / args.output_subdir
        if not eval_dir.exists():
            continue

        # 获取该场景下所有 JSON 文件
        json_files = list(eval_dir.glob("*.json"))
        if not json_files:
            continue

        # 预先获取该场景下 GT 和预测文件的映射（按 stem）
        gt_dir = scene / args.gt_subdir
        pred_dir = scene / args.pred_subdir
        if not gt_dir.exists() or not pred_dir.exists():
            print(f"Warning: Scene {scene_name} missing gt or pred directory, skipping.")
            continue

        gt_map = get_file_list_by_stem(gt_dir, exts)
        pred_map = get_file_list_by_stem(pred_dir, exts)

        for json_path in json_files:
            stem = json_path.stem  # 不含扩展名
            if stem not in gt_map or stem not in pred_map:
                # 可能 stem 不匹配，跳过
                continue

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                continue

            total_json += 1

            # 提取指标
            inliers = data.get("inliers", {})
            # 确保键存在（注意 JSON 中键是字符串）
            inlier_2 = inliers.get("2.0")
            inlier_5 = inliers.get("5.0")
            if inlier_2 is None or inlier_5 is None:
                print(f"Warning: Missing required inlier thresholds in {json_path}")
                continue

            # 判断条件：2.0 > 70% 且 5.0 > 90%
            if inlier_2 > 0.7 and inlier_5 > 0.9:
                eligible_count += 1
                eligible_files.append({
                    "scene": scene_name,
                    "stem": stem,
                    "gt_path": gt_map[stem],
                    "pred_path": pred_map[stem]
                })

    # 输出统计
    print("\n===== 替换预览 =====")
    print(f"总评估样本数: {total_json}")
    print(f"满足条件（inlier@2>70% 且 inlier@5>90%）的样本数: {eligible_count}")
    if args.dry_run:
        # print("DRY RUN: 以下文件将被替换（但未实际执行）:")
        # for f in eligible_files:
        #     print(f"  {f['scene']}/{f['stem']} -> {f['pred_path']}")
        # print("未执行任何替换。")
        return

    if eligible_count == 0:
        print("没有需要替换的文件。")
        return

    # 确认替换
    if not args.yes:
        print(f"即将用 GT 深度替换以上 {eligible_count} 个预测文件。")
        resp = input("是否继续？ [y/N] ").strip().lower()
        if resp != 'y' and resp != 'yes':
            print("操作取消。")
            return

    # 执行替换
    replaced_count = 0
    for info in tqdm(eligible_files):
        gt_path = info["gt_path"]
        pred_path = info["pred_path"]

        # 如果需要备份
        if args.backup_dir:
            backup_dir = Path(args.backup_dir) / info["scene"]
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / Path(pred_path).name
            shutil.copy2(pred_path, backup_path)
            print(f"备份 {pred_path} -> {backup_path}")

        shutil.copy2(gt_path, pred_path)
        replaced_count += 1

    print(f"\n替换完成，共替换 {replaced_count} 个文件。")

if __name__ == "__main__":
    main()
