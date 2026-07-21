#!/usr/bin/env python3
"""
评估深度图（单张、批量目录或数据集多场景）。

数据集模式（遍历dataset_root下每个场景子文件夹）：
    python scripts/evaluate_depth.py \
        --dataset_root /path/to/dataset \
        --gt_subdir depth --pred_subdir depth_da3 --rgb_subdir images \
        --output_subdir eval_results  # 结果保存在 /path/to/dataset/eval_results/ 下

批量模式（处理单个目录下所有配对图像）：
    python scripts/evaluate_depth.py \
        --gt_dir /path/to/gt_depth/ \
        --pred_dir /path/to/pred_depth/ \
        --rgb_dir /path/to/rgb/ \
        --output_dir ./eval_results

单张模式：
    python scripts/evaluate_depth.py \
        --gt /path/to/gt.exr --pred /path/to/pred.exr [--mask /path/to/mask.png] [--rgb /path/to/rgb.jpg] \
        --output_dir ./eval_result

可选参数：
    --align {none,scale,affine}       # 对齐模式（默认affine）
    --thresholds 0.5 1.0 2.0 5.0      # 误差阈值列表
    --save_aligned                     # 是否保存对齐后的pred深度（仅当align非none时有效）
    --ext .exr .png                    # 允许的文件扩展名（可多次指定，默认['.exr','.png','.jpg']）
    --recursive                         # 是否递归子目录（批量模式时有效）
    --scene_recursive                   # 数据集模式下是否递归查找场景子文件夹
"""

import os

# OPENCV_IO_ENABLE_OPENEXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ----------------------------------------------------------------------
# 以下函数从原代码中复制，未做修改
# ----------------------------------------------------------------------
def align_depth_least_square_np(gt_arr: np.ndarray, pred_arr: np.ndarray, valid_mask_arr: np.ndarray):
    m = valid_mask_arr.astype(bool)
    if m.sum() < 10:
        return pred_arr, 1.0, 0.0

    x = pred_arr[m].reshape(-1).astype(np.float64)
    y = gt_arr[m].reshape(-1).astype(np.float64)

    A = np.stack([x, np.ones_like(x)], axis=1)
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    pred_aligned = (a * pred_arr + b).astype(np.float32)
    return pred_aligned, a, b


def compute_depth_abs_metrics_np(gt_z, pr_z, mask, thresholds=(0.5, 1.0, 2.0, 5.0)):
    m = mask.astype(bool)
    finite = np.isfinite(gt_z) & np.isfinite(pr_z)
    positive = gt_z > 0
    v = m & finite & positive

    if v.sum() == 0:
        return np.nan, np.nan, {float(t): np.nan for t in thresholds}

    e = np.abs(pr_z - gt_z)[v]
    mae = float(np.mean(e))
    rmse = float(np.sqrt(np.mean(e * e)))
    inliers = {float(t): float(np.mean(e <= float(t))) for t in thresholds}
    return mae, rmse, inliers


def save_depth_vis(
    save_path: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    err: np.ndarray,
    mask: np.ndarray,
    title_prefix: str = "",
    cmap_depth: str = "viridis",
    cmap_err: str = "magma",
    thr_list=(0.5, 1.0, 2.0, 5.0),
):
    """
    简化版可视化：只显示 GT、Pred、绝对误差以及各个阈值下的误差 clip 图。
    """
    assert pred.shape == err.shape == mask.shape
    mask = mask.astype(bool)

    if mask.sum() > 0:
        valid_depth = np.concatenate([gt[mask].reshape(-1), pred[mask].reshape(-1)])
        vmin = float(np.nanpercentile(valid_depth, 1)) - 10
        vmax = float(np.nanpercentile(valid_depth, 99)) + 10
    else:
        vmin, vmax = 0.0, 1.0
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1.0

    gt_show = gt.astype(np.float32, copy=True)
    pred_show = pred.astype(np.float32, copy=True)
    err_show = err.astype(np.float32, copy=True)

    gt_show[~mask] = np.nan
    err_show[~mask] = np.nan

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    # 子图布局：第一行 RGB, GT, Pred, 绝对误差；第二行各阈值 clip 误差
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), dpi=200, constrained_layout=True)
    cbar_kw = dict(orientation="vertical", fraction=0.04, pad=0.02, shrink=0.95)

    # 第一行
    ax_rgb = axes[0, 0]
    im_rgb = ax_rgb.imshow(rgb)
    ax_rgb.set_title(f"{title_prefix}RGB")
    ax_rgb.axis("off")

    ax_gt = axes[0, 1]
    im_gt = ax_gt.imshow(gt_show, vmin=vmin, vmax=vmax, cmap=cmap_depth)
    ax_gt.set_title(f"{title_prefix}GT depth")
    ax_gt.axis("off")
    fig.colorbar(im_gt, ax=ax_gt, **cbar_kw).set_label("Depth")

    ax_pred = axes[0, 2]
    im_pred = ax_pred.imshow(pred_show, vmin=vmin, vmax=vmax, cmap=cmap_depth)
    ax_pred.set_title(f"{title_prefix}Pred depth")
    ax_pred.axis("off")
    fig.colorbar(im_pred, ax=ax_pred, **cbar_kw).set_label("Depth")

    ax_err = axes[0, 3]
    im_err = ax_err.imshow(err_show, vmin=0, vmax=max(thr_list) if thr_list else 5.0, cmap=cmap_err)
    ax_err.set_title(f"{title_prefix}Abs error")
    ax_err.axis("off")
    fig.colorbar(im_err, ax=ax_err, **cbar_kw).set_label("Error")

    # 第二行：各阈值 clip 误差
    total_valid = int(mask.sum())
    for j, thr in enumerate(thr_list):
        ax = axes[1, j]
        thr = float(thr)
        err_clip = np.clip(err.astype(np.float32, copy=False), 0.0, thr)
        err_clip_show = err_clip.copy()
        err_clip_show[~mask] = np.nan

        if total_valid > 0:
            within = ((err <= thr) & mask).sum()
            pct = 100.0 * float(within) / float(total_valid)
        else:
            pct = 0.0

        im_thr = ax.imshow(err_clip_show, vmin=0.0, vmax=thr, cmap=cmap_err)
        ax.set_title(rf"$|e|\leq{thr:g}$ clip  ({pct:.1f}%)")
        ax.axis("off")
        fig.colorbar(im_thr, ax=ax, **cbar_kw).set_label("Abs error (clipped)")

    if title_prefix:
        fig.suptitle(title_prefix.strip(), fontsize=12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_exr(path: str, depth: np.ndarray):
    """depth: (H,W) float32"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    depth = depth.astype(np.float32)
    # OpenCV EXR 写入需要启用 OpenEXR
    # export OPENCV_IO_ENABLE_OPENEXR=1
    ok = cv2.imwrite(path, depth)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for EXR: {path}. Try setting OPENCV_IO_ENABLE_OPENEXR=1")


def load_depth(path: str) -> np.ndarray:
    """读取深度图，支持 .exr 和常见的 .png/.tiff (16位或8位)"""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth image: {path}")
    # 如果是多通道，取第一个通道
    if depth.ndim == 3:
        depth = depth[..., 0]
    # 转换为 float32，如果是 uint16 则除以比例（通常 65535 对应最大深度，但需用户知晓）
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 65535.0  # 假设最大深度为1（归一化），实际可能需要用户提供尺度
        print("Warning: uint16 depth loaded, assuming normalized depth (0-1). If metric, please convert manually.")
    elif depth.dtype == np.uint8:
        depth = depth.astype(np.float32) / 255.0
    else:
        depth = depth.astype(np.float32)
    return depth


def load_mask(path: str, shape=None) -> np.ndarray:
    """读取掩码，二值化 (任何非零值视为有效)"""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    mask = (mask > 0).astype(bool)
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def get_file_list(directory, extensions, recursive=False):
    """获取目录下所有指定扩展名的文件，返回 {stem: 完整路径} 映射（如果有多个同名不同扩展名，取第一个并警告）"""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    pattern = "**/*" if recursive else "*"
    files = defaultdict(list)
    for ext in extensions:
        for p in directory.glob(f"{pattern}{ext}"):
            if p.is_file():
                files[p.stem].append(p)
    
    # 检查同名多个文件的情况
    result = {}
    for stem, paths in files.items():
        if len(paths) > 1:
            print(f"Warning: Multiple files with stem '{stem}': {[str(p) for p in paths]}. Using the first one.")
        result[stem] = str(paths[0])
    return result


def process_pair(gt_path, pred_path, mask_path, rgb_path, output_dir, args):
    """处理一对图像，保存结果到 output_dir 下"""
    stem = Path(gt_path).stem

    # 加载深度图
    gt = load_depth(gt_path)
    pred = load_depth(pred_path)

    # 确保尺寸一致
    if gt.shape != pred.shape:
        print(f"Warning: shapes differ. GT: {gt.shape}, Pred: {pred.shape}. Resizing pred to GT.")
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 加载掩码
    if mask_path and os.path.exists(mask_path):
        mask = load_mask(mask_path, shape=gt.shape)
    else:
        mask = (gt > 0) & np.isfinite(gt)

    # 对齐（如果需要）
    align_info = {"mode": args.align}
    pred_aligned = pred.copy()
    if args.align != "none":
        if args.align == "affine":
            pred_aligned, a, b = align_depth_least_square_np(gt, pred, mask)
            align_info.update({"a": a, "b": b})
        else:  # scale
            if mask.sum() < 10:
                s = 1.0
            else:
                ratio = gt[mask] / np.clip(pred[mask], 1e-8, None)
                ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
                s = float(np.median(ratio)) if ratio.size > 0 else 1.0
            pred_aligned = (pred * s).astype(np.float32)
            align_info.update({"s": s})

    # 计算指标
    mae, rmse, inliers = compute_depth_abs_metrics_np(gt, pred_aligned, mask, thresholds=args.thresholds)

    # 计算深度范围
    valid_depth = np.concatenate([gt[mask].reshape(-1), pred[mask].reshape(-1)])
    vmin = float(np.nanpercentile(valid_depth, 1)) - 2
    vmax = float(np.nanpercentile(valid_depth, 99)) + 2

    # 保存指标
    metrics = {
        "stem": stem,
        "mae": mae,
        "rmse": rmse,
        "inliers": {str(k): v for k, v in inliers.items()},
        "align": align_info,
        "thresholds": args.thresholds,
        "min_depth": vmin,
        "max_depth": vmax,
        "diff_depth": vmax - vmin,
    }
    with open(output_dir / f"{stem}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # # 保存对齐后的深度（如果要求）
    # if args.save_aligned and args.align != "none":
    #     aligned_path = output_dir / f"{stem}_aligned.exr"
    #     save_exr(str(aligned_path), pred_aligned)

    # # 生成可视化图（如果提供了rgb路径）
    # if rgb_path and os.path.exists(rgb_path):
    #     rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    #     rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    #     if rgb.shape[:2] != gt.shape[:2]:
    #         rgb = cv2.resize(rgb, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    #     error = np.abs(pred_aligned - gt).astype(np.float32)
    #     vis_path = output_dir / f"{stem}_vis.png"
    #     save_depth_vis(
    #         save_path=str(vis_path),
    #         rgb=rgb,
    #         gt=gt,
    #         pred=pred_aligned,
    #         err=error,
    #         mask=mask,
    #         title_prefix=f"{stem}  ",
    #         thr_list=args.thresholds,
    #     )
    return metrics


def process_directory(gt_dir, pred_dir, rgb_dir, mask_dir, output_dir, args):
    """
    处理一个目录下的所有图像对。
    返回该目录下所有pair的指标列表。
    """
    # 获取文件列表
    gt_files = get_file_list(gt_dir, args.ext, args.recursive)
    pred_files = get_file_list(pred_dir, args.ext, args.recursive)
    rgb_files = get_file_list(rgb_dir, args.ext, args.recursive)

    # 如果有 mask_dir，也获取 mask 文件列表
    mask_files = {}
    if mask_dir:
        mask_files = get_file_list(mask_dir, args.ext, args.recursive)

    # 找出共同 stem
    common_stems = set(gt_files.keys()) & set(pred_files.keys()) & set(rgb_files.keys())
    if not common_stems:
        print(f"No matching stems found in directories:\n  GT: {gt_dir}\n  Pred: {pred_dir}\n  RGB: {rgb_dir}")
        return []

    print(f"Found {len(common_stems)} common stems in {gt_dir.parent.name if gt_dir.parent else gt_dir}")

    all_metrics = []
    for stem in tqdm(sorted(common_stems), desc=f"Processing {gt_dir.parent.name if gt_dir.parent else 'directory'}"):
        gt_path = gt_files[stem]
        pred_path = pred_files[stem]
        rgb_path = rgb_files[stem]
        mask_path = mask_files.get(stem, None) if mask_files else None
        try:
            metrics = process_pair(gt_path, pred_path, mask_path, rgb_path, output_dir, args)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing {stem}: {e}")
            continue

    # 计算该目录的平均指标
    if all_metrics:
        avg_metrics = {
            "num_pairs": len(all_metrics),
            "mae_mean": float(np.nanmean([m["mae"] for m in all_metrics])),
            "rmse_mean": float(np.nanmean([m["rmse"] for m in all_metrics])),
            "inlier_mean": {}
        }
        for t in args.thresholds:
            vals = [m["inliers"][str(t)] for m in all_metrics if not np.isnan(m["inliers"][str(t)])]
            avg_metrics["inlier_mean"][str(t)] = float(np.nanmean(vals)) if vals else np.nan
        with open(output_dir / "summary.json", "w") as f:
            json.dump(avg_metrics, f, indent=2)
        print(f"Summary saved to {output_dir / 'summary.json'}")
    return all_metrics


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate depth maps (single, batch, or multi-scene dataset).")
    
    # 数据集模式参数
    parser.add_argument("--dataset_root", default="../../dataset/data/a3dscenes",
                        help="Root directory containing scene subfolders.")
    parser.add_argument("--gt_subdir", default="depth", help="Subdirectory name for GT depth inside each scene (default: depth).")
    parser.add_argument("--pred_subdir", default="depth_complete", help="Subdirectory name for predicted depth inside each scene (default: depth_da3).")
    parser.add_argument("--rgb_subdir", default="images", help="Subdirectory name for RGB images inside each scene (default: images).")
    parser.add_argument("--mask_subdir", help="Optional subdirectory name for masks inside each scene.")
    parser.add_argument("--scene_recursive", action="store_true", help="Recursively search for scene subfolders under dataset_root.")
    parser.add_argument("--output_subdir", default="depth_complete_viz", help="Subdirectory under dataset_root to save outputs (dataset mode only).")

    # 批量模式参数
    parser.add_argument("--gt_dir", help="Directory containing ground truth depth images (batch mode).")
    parser.add_argument("--pred_dir", help="Directory containing predicted depth images (batch mode).")
    parser.add_argument("--rgb_dir", help="Directory containing RGB images (batch mode).")
    parser.add_argument("--mask_dir", help="Optional directory containing mask images (batch mode).")

    # 单张模式参数
    parser.add_argument("--gt", help="Path to single ground truth depth image.")
    parser.add_argument("--pred", help="Path to single predicted depth image.")
    parser.add_argument("--mask", help="Optional path to single mask image.")
    parser.add_argument("--rgb", help="Path to RGB image for visualization (optional in single mode).")

    # 公共参数
    parser.add_argument("--output_dir", default="./eval_results", help="Directory to save outputs (batch/single mode). Ignored in dataset mode.")
    parser.add_argument("--align", choices=["none", "scale", "affine"], default="affine",
                        help="Alignment mode for pred before evaluation (default: affine).")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0],
                        help="Error thresholds for inlier percentages (default: 0.5 1.0 2.0 5.0).")
    parser.add_argument("--save_aligned", action="store_true", help="Save aligned prediction as EXR (only if align != none).")
    parser.add_argument("--ext", action="append", default=[], help="File extensions to consider (e.g., .exr .png). Can be used multiple times.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search subdirectories in batch mode (for images inside a scene).")

    args = parser.parse_args()

    # 设置默认扩展名
    if not args.ext:
        args.ext = ['.exr', '.png', '.jpg']
    else:
        # 确保每个扩展名以点开头
        args.ext = [e if e.startswith('.') else '.' + e for e in args.ext]

    # 确定运行模式
    dataset_mode = args.dataset_root is not None
    batch_mode = args.gt_dir is not None and args.pred_dir is not None and args.rgb_dir is not None
    single_mode = args.gt is not None and args.pred is not None

    if not dataset_mode and not batch_mode and not single_mode:
        parser.error("Either provide --dataset_root, or --gt_dir/--pred_dir/--rgb_dir (batch mode), or --gt/--pred (single mode).")

    # 根据模式设置输出基础目录
    if dataset_mode:
        # 数据集模式：结果保存在 dataset_root / output_subdir 下
        output_base = Path(args.dataset_root)
    else:
        # 批量/单张模式：使用 output_dir
        output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # -------------------- 数据集模式 --------------------
    if dataset_mode:
        print("Dataset mode: processing all scenes under", args.dataset_root)
        dataset_root = Path(args.dataset_root)
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

        # 获取场景列表
        if args.scene_recursive:
            # 递归查找所有子目录（任何深度）
            scenes = [p for p in dataset_root.rglob("*") if p.is_dir()]
        else:
            # 仅直接子目录
            scenes = [p for p in dataset_root.iterdir() if p.is_dir()]

        if not scenes:
            print("No scene subfolders found.")
            return

        # 过滤掉那些缺少必要子目录的场景（给出警告但继续）
        valid_scenes = []
        for scene in scenes:
            gt_dir = scene / args.gt_subdir
            pred_dir = scene / args.pred_subdir
            rgb_dir = scene / args.rgb_subdir
            if not gt_dir.exists() or not pred_dir.exists() or not rgb_dir.exists():
                print(f"Warning: Scene {scene.name} missing one of required subdirs (gt, pred, rgb). Skipping.")
                continue
            valid_scenes.append(scene)

        if not valid_scenes:
            print("No valid scenes found.")
            return

        print(f"Found {len(valid_scenes)} valid scenes.")

        all_scenes_metrics = {}  # scene_name -> avg_metrics (from summary.json)

        for scene in tqdm(valid_scenes, desc="Processing scenes"):
            scene_name = scene.name
            gt_dir = scene / args.gt_subdir
            pred_dir = scene / args.pred_subdir
            rgb_dir = scene / args.rgb_subdir
            mask_dir = scene / args.mask_subdir if args.mask_subdir else None

            # 场景输出目录
            if args.output_subdir:
                scene_out_dir = output_base / scene_name / args.output_subdir
            else:
                scene_out_dir = output_base / scene_name
            scene_out_dir.mkdir(parents=True, exist_ok=True)

            print(f"scene output dir: {scene_out_dir}")

            # 处理该场景
            metrics_list = process_directory(gt_dir, pred_dir, rgb_dir, mask_dir, scene_out_dir, args)

            # 收集场景平均指标
            if metrics_list:
                # 读取刚刚生成的summary.json
                summary_path = scene_out_dir / "summary.json"
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        scene_summary = json.load(f)
                    all_scenes_metrics[scene_name] = scene_summary

        # 生成全局汇总
        if all_scenes_metrics:
            global_summary = {
                "num_scenes": len(all_scenes_metrics),
                "scenes": all_scenes_metrics,
                "overall": {
                    "mae_mean": float(np.nanmean([s["mae_mean"] for s in all_scenes_metrics.values()])),
                    "rmse_mean": float(np.nanmean([s["rmse_mean"] for s in all_scenes_metrics.values()])),
                }
            }
            # 计算整体平均 inlier
            overall_inlier = {}
            for t in args.thresholds:
                vals = [s["inlier_mean"][str(t)] for s in all_scenes_metrics.values() if str(t) in s["inlier_mean"]]
                overall_inlier[str(t)] = float(np.nanmean(vals)) if vals else np.nan
            global_summary["overall"]["inlier_mean"] = overall_inlier

            with open(output_base / "all_scenes_summary.json", "w") as f:
                json.dump(global_summary, f, indent=2)
            print(f"Global summary saved to {output_base / 'all_scenes_summary.json'}")
            print(json.dumps(global_summary["overall"], indent=2))

    # -------------------- 批量模式 --------------------
    elif batch_mode:
        print("Batch mode: processing all matching images in directories.")
        process_directory(args.gt_dir, args.pred_dir, args.rgb_dir, args.mask_dir, output_base, args)

    # -------------------- 单张模式 --------------------
    elif single_mode:
        print("Single mode: processing one pair.")
        gt_path = args.gt
        pred_path = args.pred
        mask_path = args.mask
        rgb_path = args.rgb

        stem = Path(gt_path).stem
        # 加载深度
        gt = load_depth(gt_path)
        pred = load_depth(pred_path)
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        # 掩码
        if mask_path and os.path.exists(mask_path):
            mask = load_mask(mask_path, shape=gt.shape)
        else:
            mask = (gt > 0) & np.isfinite(gt)
        # 对齐
        align_info = {"mode": args.align}
        pred_aligned = pred.copy()
        if args.align != "none":
            if args.align == "affine":
                pred_aligned, a, b = align_depth_least_square_np(gt, pred, mask)
                align_info.update({"a": a, "b": b})
            else:
                if mask.sum() < 10:
                    s = 1.0
                else:
                    ratio = gt[mask] / np.clip(pred[mask], 1e-8, None)
                    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
                    s = float(np.median(ratio)) if ratio.size > 0 else 1.0
                pred_aligned = (pred * s).astype(np.float32)
                align_info.update({"s": s})
        # 计算指标
        mae, rmse, inliers = compute_depth_abs_metrics_np(gt, pred_aligned, mask, thresholds=args.thresholds)
        metrics = {
            "stem": stem,
            "mae": mae,
            "rmse": rmse,
            "inliers": {str(k): v for k, v in inliers.items()},
            "align": align_info,
            "thresholds": args.thresholds,
        }
        # 保存json
        with open(output_base / f"{stem}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        # 如果提供了rgb且文件存在，生成可视化
        if rgb_path and os.path.exists(rgb_path):
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != gt.shape[:2]:
                rgb = cv2.resize(rgb, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            error = np.abs(pred_aligned - gt).astype(np.float32)
            vis_path = output_base / f"{stem}_vis.png"
            save_depth_vis(
                save_path=str(vis_path),
                rgb=rgb,
                gt=gt,
                pred=pred_aligned,
                err=error,
                mask=mask,
                title_prefix=f"{stem}  ",
                thr_list=args.thresholds,
            )
        print("Done.")

if __name__ == "__main__":
    main()
