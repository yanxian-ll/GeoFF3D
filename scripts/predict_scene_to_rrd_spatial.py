#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatial chunk reconstruction pipeline for multi-view UAV scenes.

This is the single main entry point for spatial RRD inference. It:
  1. Reads images, camera priors, and optional depth from a scene directory.
  2. Partitions frames into footprint-tree spatial chunks.
  3. Initializes a model with automatic prior policy resolution.
  4. Runs each chunk, reusing predicted depth as seam priors.
  5. Applies per-chunk alignment.
  6. Aggregates core point clouds and outputs overall RRD, chunk artifacts,
     and eval ply/npz/json.

Examples:
    # Minimal invocation with auto policies:
    python scripts/predict_scene_to_rrd_spatial.py \\
      --scene_dir /path/to/scene \\
      --model geoff3d \\
      --checkpoint experiments/.../checkpoint-last.pth \\
      --output_rrd outputs/scene_spatial.rrd

    # No-prior model with only translation alignment:
    python scripts/predict_scene_to_rrd_spatial.py \\
      --scene_dir /path/to/scene \\
      --model vggt \\
      --output_rrd outputs/scene_spatial.rrd
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None

from geoff3d.spatial_rrd.scene_io import (
    build_views_from_scene,
    load_chunk_views_from_scene,
    resolve_device,
)
from geoff3d.spatial_rrd.chunk_cache import (
    AsyncChunkCacheWriter,
    chunk_cache_dir_for_output,
    load_chunk_cache,
    strip_array_payload,
    update_chunk_cache,
    write_chunk_record_manifest,
)
from geoff3d.spatial_rrd.chunking import (
    CHUNK_ORDER_STRATEGIES,
    SPATIAL_PARTITIONS,
    build_spatial_chunks,
    order_spatial_chunks,
)
from geoff3d.spatial_rrd.model_runner import (
    collect_pred_outputs,
    init_model_from_hydra,
    load_checkpoint,
    checkpoint_hydra_overrides,
    build_prior_overrides,
    set_model_task_value,
    set_model_task_prob,
    set_pi3x_ray_prior_prob,
    recover_average_intrinsics_from_pred_rays,
    apply_bootstrap_intrinsics_to_views,
    build_depth_prior_cache,
    build_future_overlap_counter,
    apply_cached_depth_priors_to_views,
    points_from_maps,
    resolve_prior_policy,
    filter_views_for_prior_policy,
    apply_runtime_prior_policy,
)
from geoff3d.spatial_rrd.geometry_align import (
    normalize_align_mode,
    maybe_recenter_anchor_from_meta,
    restore_predictions_from_recenter,
    apply_chunk_pose_alignment,
    estimate_chunk_pose_alignment,
    resolve_align_mode,
    resolve_recenter_mode,
    POSE_TRANSLATION_ALIGN_MODES,
)
from geoff3d.spatial_rrd.rrd_writer import (
    dedupe_cameras_by_stem,
    save_spatial_rrd,
    input_pose_centers_by_stem,
)
from geoff3d.spatial_rrd.multiview_consistency import (
    intrinsics_from_views,
    apply_mvsnet_style_multiview_filter,
)
from geoff3d.spatial_rrd.pose_perturb import perturb_scene_camera_poses
from geoff3d.spatial_rrd.gsplat_bundle import (
    build_gsplat_optimization_bundles,
    collect_gsplat_bundle_inputs,
)
from geoff3d.spatial_rrd.gsplat_refine import optimize_and_save_gsplat_bundle
from geoff3d.spatial_rrd.orthodom import (
    render_orthodom_from_fused_points,
    render_orthodom_from_gsplat_summary,
)
from geoff3d.spatial_rrd.chunk_post_align import (
    apply_deferred_chunk_post_alignment,
    compute_adjacent_chunk_seam_error,
)
from geoff3d.spatial_rrd.chunk_artifacts import make_chunk_color_lookup
from geoff3d.spatial_rrd.chunk_transform import (
    compose_record_similarity,
    get_transformed_cameras,
)


def apply_final_global_pose_alignment(
    chunk_records: List[Dict[str, object]],
    reference_centers_by_stem: Dict[str, np.ndarray],
    mode: str,
    seed: int,
) -> Dict[str, object]:
    """Restore the final fused reconstruction to the input-pose world gauge."""
    pred_cams = dedupe_cameras_by_stem(
        [
            cam
            for record in chunk_records
            for cam in get_transformed_cameras(record)
        ]
    )
    target_stems = [str(cam.get("stem", "")) for cam in pred_cams]
    align_mode = mode if mode in POSE_TRANSLATION_ALIGN_MODES else "pose_sim3"
    align_meta = estimate_chunk_pose_alignment(
        mode=align_mode,
        chunk_id=-1,
        reference_cams_by_stem=reference_centers_by_stem,
        raw_pred_cams=pred_cams,
        target_stems=target_stems,
        seed=int(seed) + 970003,
    )
    align_meta = dict(align_meta)
    align_meta["stage"] = "final_global_pose_alignment"
    align_meta["requested_mode"] = str(mode)
    if not bool(align_meta.get("valid", False)):
        return align_meta

    scale = float(align_meta["scale"])
    R = np.asarray(align_meta["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(align_meta["t"], dtype=np.float64).reshape(3)
    for record in chunk_records:
        compose_record_similarity(record, scale, R, t)
    align_meta["num_chunks_transformed"] = int(len(chunk_records))
    return align_meta


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
class _TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(self.streams and getattr(self.streams[0], "isatty", lambda: False)())


class RunLogger:
    def __init__(self, output_rrd: Path, log_file: Optional[str], enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.path: Optional[Path] = None
        self._fh = None
        self._old_stdout = None
        self._old_stderr = None
        if not self.enabled:
            return
        if log_file:
            self.path = Path(log_file).expanduser().resolve()
        else:
            self.path = Path(output_rrd).expanduser().resolve().with_suffix("") / "logs" / "pipeline.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def install(self) -> None:
        if not self.enabled or self.path is None:
            return
        self._fh = self.path.open("w", encoding="utf-8", buffering=1)
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = _TeeStream(sys.stdout, self._fh)  # type: ignore[assignment]
        sys.stderr = _TeeStream(sys.stderr, self._fh)  # type: ignore[assignment]
        print(f"[LOG] Saving full run log to: {self.path}")

    def close(self) -> None:
        if self._fh is None:
            return
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
        if self._old_stderr is not None:
            sys.stderr = self._old_stderr
        self._fh.close()
        self._fh = None

    def detail(self, message: str) -> None:
        if self._fh is None:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._fh.write(f"[{stamp}] {message}\n")
        self._fh.flush()


def stage(name: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"\n[STAGE] {name}{suffix}")


def iter_progress(
    values: Iterable,
    *,
    desc: str,
    total: Optional[int] = None,
    enabled: bool = True,
):
    if enabled and tqdm is not None:
        return tqdm(
            values,
            desc=desc,
            total=total,
            unit="chunk",
            dynamic_ncols=True,
            file=sys.__stderr__,
        )
    return values


def _pose_filter_reason(meta: Dict[str, object], stem: str) -> Optional[str]:
    cams = meta.get("cams", {})
    cam = cams.get(stem) if isinstance(cams, dict) else None
    if cam is None:
        return "missing_camera_prior"
    T = np.asarray(cam.get("T_c2w", None), dtype=np.float64)
    if T.shape != (4, 4):
        return f"invalid_T_c2w_shape:{T.shape}"
    if not np.isfinite(T).all():
        return "non_finite_T_c2w"
    return None


def filter_views_meta_to_valid_poses(
    views: Sequence[Dict[str, object]],
    meta: Dict[str, object],
    *,
    output_rrd: Path,
    run_logger: RunLogger,
) -> Tuple[List[Dict[str, object]], Dict[str, object], List[Dict[str, object]]]:
    ignored: List[Dict[str, object]] = []
    keep_stems: List[str] = []
    for stem in meta.get("stems", []):
        stem = str(stem)
        reason = _pose_filter_reason(meta, stem)
        if reason is None:
            keep_stems.append(stem)
        else:
            ignored.append(
                {
                    "stem": stem,
                    "reason": reason,
                    "image_path": str(meta.get("image_paths", {}).get(stem, "")),
                    "cam_path": str(meta.get("cam_paths", {}).get(stem, "")),
                    "depth_path": str(meta.get("depth_paths", {}).get(stem, "")),
                }
            )

    if not keep_stems:
        preview = ", ".join(str(item["stem"]) for item in ignored[:8])
        raise RuntimeError(
            "No selected frames have valid camera poses after filtering. "
            f"First ignored frames: {preview}"
        )

    keep_set = set(keep_stems)
    filtered_views = [
        view for view in views if str(view.get("stem", "")) in keep_set
    ]
    filtered_meta = dict(meta)
    filtered_meta["stems"] = keep_stems

    for key in ("image_paths", "depth_paths", "cam_paths", "cams"):
        value = meta.get(key, {})
        if isinstance(value, dict):
            filtered_meta[key] = {
                stem: value[stem] for stem in keep_stems if stem in value
            }

    cams = filtered_meta.get("cams", {})
    depths = filtered_meta.get("depth_paths", {})
    filtered_meta["num_cam_priors"] = int(
        sum(1 for stem in keep_stems if isinstance(cams, dict) and stem in cams)
    )
    filtered_meta["num_depth_priors"] = int(
        sum(1 for stem in keep_stems if isinstance(depths, dict) and stem in depths)
    )
    filtered_meta["ignored_no_pose_frames"] = ignored
    filtered_meta["num_ignored_no_pose_frames"] = int(len(ignored))

    if ignored:
        log_dir = (
            run_logger.path.parent
            if run_logger.path is not None
            else Path(output_rrd).expanduser().resolve().with_suffix("") / "logs"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        json_path = log_dir / "ignored_no_pose_frames.json"
        txt_path = log_dir / "ignored_no_pose_frames.txt"
        json_path.write_text(
            json.dumps(ignored, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        txt_path.write_text(
            "\n".join(
                f"{item['stem']}\t{item['reason']}\t{item.get('image_path', '')}\t{item.get('cam_path', '')}"
                for item in ignored
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            "[WARN] Ignored frames without valid poses: "
            f"{len(ignored)}/{len(meta.get('stems', []))}; "
            f"kept={len(keep_stems)}. See {json_path}"
        )
        for item in ignored:
            run_logger.detail(
                "[ignored_no_pose] "
                f"stem={item['stem']}, reason={item['reason']}, "
                f"image={item.get('image_path', '')}, cam={item.get('cam_path', '')}"
            )

    return filtered_views, filtered_meta, ignored


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Spatial chunk RRD reconstruction for multi-view UAV scenes.",
    )

    # ╔══════════════════════════════════════════════════════════════════
    # ║  必填参数  (Required)
    # ╚══════════════════════════════════════════════════════════════════
    g = p.add_argument_group("必填参数")
    g.add_argument(
        "--scene_dir", required=True,
        help="场景目录，包含 images/ cams/ depth/ 子目录",
    )
    g.add_argument(
        "--output_rrd", required=True,
        help="输出 .rrd 文件路径 （同时生成 .json sidecar 和 eval/ 目录）",
    )
    g.add_argument(
        "--model", required=True,
        help="模型 Hydra 配置名。支持 geoff3d、pi3x、vggt。",
    )

    # ╔══════════════════════════════════════════════════════════════════
    # ║  模型与设备  (Model & Device)
    # ╚══════════════════════════════════════════════════════════════════
    g = p.add_argument_group("模型与设备")
    g.add_argument(
        "--checkpoint", default=None,
        help="可选权重覆盖路径 （模型初始化后加载）",
    )
    g.add_argument(
        "--device", default="auto",
        help="auto / cpu / cuda / cuda:0 / 数字",
    )
    g.add_argument(
        "--machine", default="aws",
        help="Hydra machine 配置名（默认 aws，使用本地 torch hub 缓存，禁止在线下载）",
    )
    g.add_argument(
        "--hydra_override", action="append", default=[],
        help="额外 Hydra override，可重复使用",
    )

    # ╔══════════════════════════════════════════════════════════════════
    # ║  场景读取 —— 帧选择与尺寸  (Scene IO)
    # ╚══════════════════════════════════════════════════════════════════
    g = p.add_argument_group("场景读取 —— 帧选择与尺寸")
    g.add_argument(
        "--images_dir", default="images",
        help="场景子目录：RGB 图像 （默认 images）",
    )
    g.add_argument(
        "--cams_dir", default="cams",
        help="场景子目录：相机外参 (cams/*.txt)（默认 cams）",
    )
    g.add_argument(
        "--depth_dir", default="depth",
        help="场景子目录：深度图 （默认 depth）",
    )
    g.add_argument(
        "--frame_glob", default="*",
        help="帧 glob 过滤，如 'DJI_*' （默认全部）",
    )
    g.add_argument(
        "--num_views", type=int, default=0,
        help="最多选取帧数；<=0 表示全选",
    )
    g.add_argument(
        "--start", type=int, default=0,
        help="从排序后第 start 帧开始取",
    )
    g.add_argument(
        "--stride", type=int, default=1,
        help="帧采样步长",
    )
    g.add_argument(
        "--max_side", type=int, default=518,
        help="缩放后最大边长",
    )
    g.add_argument(
        "--size_multiple", type=int, default=14,
        help="缩放后尺寸对齐倍数；保留兼容，推荐用 --patch_size",
    )
    g.add_argument(
        "--patch_size", type=int, default=None,
        help="模型 patch size，用于缩放后尺寸对齐；未设置时使用 --size_multiple",
    )
    g.add_argument(
        "--norm_type", default="identity",
        help="输入图像归一化类型：identity / dinov2 / dust3r 等",
    )
    g.add_argument(
        "--scene_io_workers",
        type=int,
        default=0,
        help="每个 chunk 内并行加载/resize RGB-D 的线程数；<=1 表示串行",
    )
    g.add_argument(
        "--footprint_workers",
        type=int,
        default=0,
        help="build_spatial_chunks 需要读取 depth footprint 时使用的进程数；<=1 表示串行",
    )

    # ╔══════════════════════════════════════════════════════════════════
    # ║  深度与点云过滤  (Depth & Point Filtering)
    # ╚══════════════════════════════════════════════════════════════════
    g = p.add_argument_group("深度与点云过滤")
    g.add_argument(
        "--depth_scale", type=float, default=1.0,
        help="深度值缩放因子",
    )
    g.add_argument(
        "--depth_min", type=float, default=1e-6,
        help="最小有效深度",
    )
    g.add_argument(
        "--depth_max", type=float, default=1e6,
        help="最大有效深度",
    )
    g.add_argument(
        "--pred_min_depth", type=float, default=1e-6,
        help="预测点云最小深度过滤 （基于 pts3d_cam z）",
    )
    g.add_argument(
        "--conf_quantile", type=float, default=0.25,
        help="预测深度置信度分位过滤阈值；默认 0.25 表示去掉最低 25%% confidence",
    )
    g.add_argument(
        "--depth_conf_filter",
        action="store_true",
        default=True,
        help="在每个 chunk 推理后按 confidence percentile 过滤预测 depth，并将过滤 depth 置 0（默认开启）",
    )
    g.add_argument(
        "--no_depth_conf_filter",
        action="store_false",
        dest="depth_conf_filter",
        help="关闭 chunk 内预测 depth confidence percentile 过滤",
    )
    g.add_argument(
        "--debug_depth_conf_filter",
        action="store_true",
        help="保存每个 chunk/view 的 confidence、过滤 mask、过滤前后 depth 和 RGB 调试图",
    )
    g.add_argument(
        "--mv_consistency",
        action="store_true",
        help="启用 MVSNet-style 多视角几何一致性过滤（所有chunk预测结束后统一执行）",
    )
    g.add_argument(
        "--mv_min_support",
        type=int,
        default=1,
        help="每个点至少需要多少个邻近视角支持；0 表示禁用",
    )
    g.add_argument(
        "--mv_max_neighbors",
        type=int,
        default=4,
        help="每个 source view 最多检查多少个最近邻 target view",
    )
    g.add_argument(
        "--mv_conf_threshold",
        type=float,
        default=0.0,
        help="几何一致性 confidence 阈值；confidence=support/evidence",
    )
    g.add_argument(
        "--mv_depth_abs_tol",
        type=float,
        default=0.05,
        help="MVSNet-style 深度一致性绝对阈值",
    )
    g.add_argument(
        "--mv_depth_rel_tol",
        type=float,
        default=0.02,
        help="MVSNet-style 深度一致性相对阈值",
    )
    g.add_argument(
        "--mv_point_abs_tol",
        type=float,
        default=0.25,
        help="world pointmap 3D 一致性绝对距离阈值，单位同点云坐标",
    )
    g.add_argument(
        "--mv_point_rel_tol",
        type=float,
        default=0.02,
        help="world pointmap 3D 一致性相对距离阈值，按目标深度缩放",
    )
    g.add_argument(
        "--mv_no_point_check",
        action="store_true",
        help="只做 MVSNet-style depth consistency，不做 world point 3D 距离检查",
    )

    # ╔══════════════════════════════════════════════════════════════════
    # ║  3DGS refinement  (Optional, default off)
    # ╚══════════════════════════════════════════════════════════════════
    g = p.add_argument_group("3DGS 优化（默认关闭）")
    g.add_argument(
        "--gsplat_refine",
        action="store_true",
        help="启用 gsplat 3DGS bundle 优化；默认关闭，在所有chunk预测后统一执行",
    )
    g.add_argument(
        "--gsplat_steps",
        type=int,
        default=3000,
        help="每个 3DGS bundle 的优化步数；200~500 仅适合 smoke test，默认 3000",
    )
    g.add_argument(
        "--gsplat_max_gaussians",
        type=int,
        default=120000,
        help="每个 3DGS bundle 最多采样多少个 Gaussian；<=0 表示使用全部",
    )
    g.add_argument(
        "--gsplat_batch_views",
        type=int,
        default=2,
        help="每步随机优化多少个视角；<=0 表示使用全部",
    )
    g.add_argument(
        "--gsplat_render_scale",
        type=float,
        default=0.5,
        help="优化渲染分辨率比例；0.5 表示半分辨率",
    )
    g.add_argument(
        "--gsplat_no_pose_opt",
        action="store_true",
        help="禁用相机外参同步优化，仅优化 Gaussians",
    )
    g.add_argument(
        "--gsplat_single_max_images",
        type=int,
        default=80,
        help="场景图像数<=此值时用单个 scene 级 3DGS；<=0 永远单个",
    )
    g.add_argument(
        "--gsplat_max_images_per_bundle",
        type=int,
        default=80,
        help="大场景分 bundle 后每个 3DGS bundle 最多渲染图像数",
    )
    g.add_argument(
        "--gsplat_min_core_images_per_bundle",
        type=int,
        default=24,
        help="每个 3DGS bundle 尽量包含的最少 core 图像数",
    )
    g.add_argument(
        "--gsplat_log_every",
        type=int,
        default=50,
        help="gsplat 优化每隔多少 step 打印/记录一次 loss；<=0 只记录最后一步",
    )
    g.add_argument(
        "--gsplat_no_tqdm",
        action="store_true",
        help="禁用 gsplat tqdm 进度条，只使用普通 print log",
    )
    g.add_argument(
        "--gsplat_save_rendered_views",
        action="store_true",
        help="gsplat 优化结束后保存部分 render RGB/depth/target/error PNG 到 bundle 目录",
    )
    g.add_argument(
        "--gsplat_render_output_max_views",
        type=int,
        default=12,
        help="每个 gsplat bundle 最多保存多少个渲染预览视角；<=0 表示全部保存",
    )
    g.add_argument(
        "--gsplat_render_output_stride",
        type=int,
        default=1,
        help="保存渲染预览时的视角步长",
    )

    # Official-style gsplat trainer parameters.
    g.add_argument(
        "--gsplat_strategy",
        default="default",
        choices=["default", "mcmc", "none"],
        help="gsplat 官方 densification/pruning 策略：default / mcmc / none",
    )
    g.add_argument(
        "--gsplat_sh_degree",
        type=int,
        default=3,
        help="3DGS spherical harmonics degree；0 表示只用 DC 颜色",
    )
    g.add_argument(
        "--gsplat_sh_degree_interval",
        type=int,
        default=1000,
        help="每隔多少 step 提升一次 SH degree",
    )
    g.add_argument("--gsplat_init_opacity", type=float, default=0.1)
    g.add_argument("--gsplat_init_scale", type=float, default=1.0)
    g.add_argument("--gsplat_ssim_lambda", type=float, default=0.2)

    g.add_argument("--gsplat_means_lr", type=float, default=1.6e-4)
    g.add_argument("--gsplat_scales_lr", type=float, default=5e-3)
    g.add_argument("--gsplat_opacities_lr", type=float, default=5e-2)
    g.add_argument("--gsplat_quats_lr", type=float, default=1e-3)
    g.add_argument("--gsplat_sh0_lr", type=float, default=2.5e-3)
    g.add_argument("--gsplat_shN_lr", type=float, default=2.5e-3 / 20.0)

    g.add_argument("--gsplat_pose_lr", type=float, default=1e-5)
    g.add_argument("--gsplat_pose_reg", type=float, default=1e-6)
    g.add_argument("--gsplat_opacity_reg", type=float, default=0.0)
    g.add_argument("--gsplat_scale_reg", type=float, default=0.0)

    g.add_argument("--gsplat_refine_start_iter", type=int, default=500)
    g.add_argument("--gsplat_refine_stop_iter", type=int, default=15000)
    g.add_argument("--gsplat_refine_every", type=int, default=100)
    g.add_argument("--gsplat_reset_every", type=int, default=3000)
    g.add_argument("--gsplat_prune_opa", type=float, default=0.005)
    g.add_argument("--gsplat_grow_grad2d", type=float, default=0.0002)
    g.add_argument("--gsplat_grow_scale3d", type=float, default=0.01)
    g.add_argument("--gsplat_grow_scale2d", type=float, default=0.05)
    g.add_argument("--gsplat_prune_scale3d", type=float, default=0.1)
    g.add_argument("--gsplat_prune_scale2d", type=float, default=0.15)
    g.add_argument(
        "--gsplat_absgrad",
        action="store_true",
        help="使用 AbsGS 风格 absolute gradient；开启后 grow_grad2d 通常要调大",
    )
    g.add_argument(
        "--gsplat_strategy_quiet",
        action="store_true",
        help="关闭 strategy verbose log",
    )

    g.add_argument(
        "--gsplat_packed",
        action="store_true",
        help="gsplat packed rasterization，省显存但可能慢一些",
    )
    g.add_argument(
        "--gsplat_sparse_grad",
        action="store_true",
        help="使用 sparse gradient；需要 packed=True",
    )
    g.add_argument(
        "--gsplat_visible_adam",
        action="store_true",
        help="使用 gsplat SelectiveAdam / visible Adam",
    )
    g.add_argument(
        "--gsplat_antialiased",
        action="store_true",
        help="使用 antialiased rasterization",
    )
    g.add_argument(
        "--gsplat_random_bkgd",
        action="store_true",
        help="训练时使用随机背景，减少透明度作弊",
    )

    # ╔══════════════════════════════════════════════════════════════════
    # ║  DOM rendering  (Optional, default off)
    # ╚══════════════════════════════════════════════════════════════════
    g = p.add_argument_group("DOM 渲染（默认关闭）")
    g.add_argument(
        "--render_dom",
        action="store_true",
        help="启用基于预测点图的 gsplat top-down DOM 渲染",
    )
    g.add_argument(
        "--dom_output_dir",
        default=None,
        help="DOM 输出目录；默认 output_rrd 去后缀后的 orthodom/ 目录",
    )
    g.add_argument(
        "--dom_gsd",
        type=float,
        default=0.0,
        help="DOM GSD，单位同世界坐标；<=0 时从预测点图相邻像素距离自动估计",
    )
    g.add_argument(
        "--dom_axes",
        default="xy",
        choices=["xy", "xz", "yz"],
        help="DOM 水平平面坐标轴，默认 xy",
    )
    g.add_argument(
        "--dom_up_axis",
        default="z",
        choices=["x", "y", "z"],
        help="DOM 高程轴，默认 z；不能和 dom_axes 重复",
    )
    g.add_argument(
        "--dom_source",
        default="core",
        choices=["core", "all"],
        help="DOM 使用的点图来源：core 只用每个 chunk 的 core 图像；all 使用 chunk 内全部图像",
    )
    g.add_argument(
        "--dom_tile_px",
        type=int,
        default=1024,
        help="DOM 分块渲染 tile 像素大小",
    )
    g.add_argument(
        "--dom_margin_px",
        type=int,
        default=32,
        help="每个 DOM tile 额外选取多少像素宽度的世界边界作为 splat margin",
    )
    g.add_argument(
        "--dom_max_gaussians_per_tile",
        type=int,
        default=0,
        help="每个 DOM tile 最多使用多少个 Gaussian；<=0 表示 DOM 渲染阶段不采样、不丢 Gaussian",
    )
    g.add_argument(
        "--dom_splat_scale",
        type=float,
        default=2.0,
        help="DOM splat scale；fused-point 模式为点间距倍率，optimized 模式为 optimized scale 倍率",
    )
    g.add_argument(
        "--dom_dsm_smooth_radius_px",
        type=int,
        default=2,
        help="fused-point DSM 的 NaN-aware 平滑半径，单位像素；0 表示关闭",
    )
    g.add_argument(
        "--dom_dsm_smooth_sigma",
        type=float,
        default=0.0,
        help="fused-point DSM 平滑高斯 sigma；<=0 使用盒式平均",
    )
    g.add_argument(
        "--dom_dsm_smooth_iterations",
        type=int,
        default=1,
        help="fused-point DSM 平滑迭代次数",
    )
    g.add_argument(
        "--dom_dsm_smooth_min_weight",
        type=float,
        default=0.05,
        help="fused-point DSM 平滑后保留像素所需的最小有效邻域权重",
    )
    g.add_argument(
        "--dom_save_contours",
        action="store_true",
        help="保存 DSM 高程图上叠加等高线和高度标注的 PNG/SVG；默认关闭",
    )
    g.add_argument(
        "--dom_opacity",
        type=float,
        default=0.95,
        help="DOM Gaussian opacity",
    )
    g.add_argument(
        "--dom_gsd_stride",
        type=int,
        default=8,
        help="自动估计 GSD 时采样相邻点的步长",
    )
    g.add_argument(
        "--dom_bounds_quantile_min",
        type=float,
        default=0.5,
        help="DOM bbox 最小分位数，过滤飞点",
    )
    g.add_argument(
        "--dom_bounds_quantile_max",
        type=float,
        default=99.5,
        help="DOM bbox 最大分位数，过滤飞点",
    )
    g.add_argument(
        "--dom_padding_m",
        type=float,
        default=0.0,
        help="DOM bbox 额外 padding，<=0 时自动使用 4*GSD",
    )
    g.add_argument(
        "--dom_max_pixels",
        type=int,
        default=160000000,
        help="DOM 最大像素数，防止 GSD 过小导致超大图",
    )
    g.add_argument(
        "--dom_allow_large",
        action="store_true",
        help="允许输出超过 dom_max_pixels 的 DOM",
    )
    g.add_argument(
        "--dom_rasterize_mode",
        default="classic",
        choices=["classic", "antialiased"],
        help="gsplat rasterize_mode",
    )
    g.add_argument(
        "--dom_no_save_tiles",
        action="store_true",
        help="不保存中间 DOM tile PNG，只保存最终拼接结果",
    )
    g.add_argument(
        "--dom_epsg",
        type=int,
        default=0,
        help="可选 EPSG；>0 时尝试保存 GeoTIFF",
    )

    # ╔══════════════════════════════════════════════════════════════════
    # ║  空间分块  (Spatial Chunking)
    # ╚══════════════════════════════════════════════════════════════════
    g = p.add_argument_group("空间分块")
    g.add_argument(
        "--spatial_partition",
        default="footprint_tree",
        choices=sorted(SPATIAL_PARTITIONS),
        help="空间分块方式：默认 footprint_tree；footprint_grid 用于固定格网消融",
    )
    g.add_argument(
        "--pose_grid_size",
        type=float,
        default=0.0,
        help="固定格网边长；<=0 根据 max_chunk_size 自动估计",
    )
    g.add_argument(
        "--pose_grid_neighbor_radius",
        type=int,
        default=1,
        help="固定格网选择 overlap 候选时使用的邻接格半径",
    )
    g.add_argument(
        "--max_chunk_size", type=int, default=32,
        help="每个 chunk 最多包含的图像数",
    )
    g.add_argument(
        "--min_chunk_size", type=int, default=8,
        help="跳过低于此值的 core cell",
    )
    g.add_argument(
        "--max_chunks", type=int, default=0,
        help="最多 chunk 数量限制；<=0 不限制",
    )
    g.add_argument(
        "--temporal_overlap_ratio",
        type=float,
        default=0.25,
        help="temporal 分块中相邻 chunk 的图像重叠比例；默认 0.25",
    )
    g.add_argument(
        "--chunk_order",
        default="spatial_center_bfs",
        choices=sorted(CHUNK_ORDER_STRATEGIES),
        help="chunk 执行顺序：spatial_sort 保留原空间排序；默认 spatial_center_bfs",
    )
    g.add_argument(
        "--chunk_footprint_point_size",
        type=float,
        default=12.0,
        help="分块 footprint 可视化中每个 chunk core 点的 marker 面积；<=0 使用自动大小",
    )
    g.add_argument(
        "--chunk_footprint_bg_point_size",
        type=float,
        default=3.0,
        help="分块 footprint 可视化中灰色背景 footprint 点的 marker 面积；<=0 使用自动大小",
    )
    g.add_argument(
        "--chunk_footprint_alpha",
        type=float,
        default=0.78,
        help="分块 footprint 可视化中彩色 core 点透明度",
    )
    g.add_argument(
        "--chunk_footprint_bg_alpha",
        type=float,
        default=0.22,
        help="分块 footprint 可视化中灰色背景点透明度",
    )
    g.add_argument(
        "--chunk_footprint_label_size",
        type=float,
        default=11.0,
        help="分块 footprint 图中 chunk id 与 X/Y 坐标轴标题的共同基础字号；最终字号还会乘 font_scale，<=0 不绘制 chunk id",
    )
    g.add_argument(
        "--chunk_footprint_font_scale",
        type=float,
        default=1.35,
        help="分块 footprint 可视化全局字体缩放比例，作用于坐标轴、刻度、图例和 chunk 标签",
    )
    g.add_argument(
        "--chunk_footprint_padding_ratio",
        type=float,
        default=0.02,
        help="分块 footprint 图在实际数据范围外保留的 X/Y 留白比例；0 表示不额外留白",
    )
    g.add_argument(
        "--chunk_footprint_legend_cols",
        type=int,
        default=0,
        help="分块 footprint 图 legend 列数；<=0 根据 chunk 数自动拆列",
    )
    g.add_argument(
        "--chunk_footprint_show_legend",
        action="store_true",
        help="在分块 footprint 图内部显示 legend；默认关闭",
    )
    g.add_argument(
        "--chunk_footprint_legend_max_rows",
        type=int,
        default=16,
        help="legend 自动拆列时每列最多多少行",
    )

    # ╔══════════════════════════════════════════════════════════════════
    # ║  对齐模式  (Alignment)
    # ╚══════════════════════════════════════════════════════════════════
    g = p.add_argument_group("对齐模式")
    g.add_argument(
        "--align", default="auto",
        help="auto / none / scale / pose_translation / pose_scale / pose_scale_yaw_translation / pose_sim3",
    )

    # ╔══════════════════════════════════════════════════════════════════
    # ║  Chunk 后对齐  (Deferred chunk post-alignment)
    # ╚══════════════════════════════════════════════════════════════════
    g = p.add_argument_group("Chunk 后对齐")
    g.add_argument(
        "--post_chunk_align",
        action="store_true",
        help="启用所有 chunk 推理完成后的层次化 seam-geometry 后对齐；默认关闭",
    )
    g.add_argument(
        "--post_chunk_align_mode",
        default="rigid",
        choices=["translation", "yaw_translation", "rigid", "sim3"],
        help="后对齐变换类型：translation / yaw_translation / rigid / sim3。yaw_translation 只估计绕 Z 轴旋转和平移，适合 Z-up UAV 场景。默认 rigid",
    )
    g.add_argument(
        "--post_chunk_align_min_corr",
        type=int,
        default=512,
        help="估计 chunk 间后对齐变换所需的最少几何对应点数量",
    )
    g.add_argument(
        "--post_chunk_align_max_corr_per_view",
        type=int,
        default=4096,
        help="每个共同 seam view 最多采样多少个像素级对应点",
    )
    g.add_argument(
        "--post_chunk_align_max_corr",
        type=int,
        default=50000,
        help="每条 chunk/node 对齐边最多使用多少个对应点",
    )
    g.add_argument(
        "--post_chunk_align_no_spatial_balance",
        action="store_true",
        help="关闭后对齐对应点的空间网格均衡采样；默认开启",
    )
    g.add_argument(
        "--post_chunk_align_spatial_grid_size",
        type=int,
        default=16,
        help="后对齐空间均衡采样的 XY 网格边长；例如 16 表示 16x16",
    )
    g.add_argument(
        "--post_chunk_align_spatial_min_corr_per_cell",
        type=int,
        default=64,
        help="自动每格采样上限的下限，防止网格过细时对应点过少",
    )
    g.add_argument(
        "--post_chunk_align_spatial_max_corr_per_cell",
        type=int,
        default=0,
        help="每个空间网格最多保留多少对应点；<=0 根据 max_corr/grid_size 自动推断",
    )
    g.add_argument(
        "--post_chunk_align_workers",
        type=int,
        default=2,
        help="后对齐同一层不同 parent group 的并行线程数；1 表示串行",
    )
    g.add_argument(
        "--post_chunk_align_nn_fallback",
        action="store_true",
        help="当共同 view 对应点不足时，使用点云最近邻作为 fallback",
    )
    g.add_argument(
        "--post_chunk_align_nn_points",
        type=int,
        default=20000,
        help="NN fallback 中每个 node 最多采样多少点",
    )
    g.add_argument(
        "--post_chunk_align_nn_quantile",
        type=float,
        default=0.7,
        help="NN fallback 保留最近邻距离较小的分位比例",
    )
    g.add_argument(
        "--compute_seam_error",
        action="store_true",
        help="计算相邻 chunk overlap 的双向 median NN seam error；默认关闭",
    )
    g.add_argument(
        "--seam_error_max_points_per_edge",
        type=int,
        default=20000,
        help="计算每条相邻 chunk 边的 seam error 时，每侧最多采样的点数",
    )

    # ╔══════════════════════════════════════════════════════════════════
    # ║  Prior 策略  (Prior Policy)
    # ╚══════════════════════════════════════════════════════════════════
    g2 = p.add_argument_group("Prior 策略")

    g2.add_argument(
        "--model_family",
        default="auto",
        choices=["auto", "no_prior", "input_prior", "ours"],
        help="模型类别。auto 会根据 --model 自动判断。",
    )

    g2.add_argument(
        "--pose_prior",
        default="auto",
        choices=["auto", "input", "none"],
        help="完整 pose 先验，仅用于 geoff3d / pi3x 这类支持输入先验的模型。",
    )

    g2.add_argument(
        "--translation_prior",
        default="auto",
        choices=["auto", "input", "none"],
        help="平移先验，主要用于 geoff3d。",
    )

    g2.add_argument(
        "--rotation_prior",
        default="auto",
        choices=["auto", "input", "none"],
        help="旋转先验，主要用于 geoff3d。",
    )

    g2.add_argument(
        "--ray_prior",
        default="auto",
        choices=["auto", "input", "pred", "none"],
        help="内参/ray 先验：input 使用真实内参，pred 使用 chunk0 预测恢复的内参。",
    )

    g2.add_argument(
        "--depth_prior",
        default="auto",
        choices=["auto", "input", "pred", "none"],
        help="深度先验：input 使用真实 depth，pred 使用前面 chunk 预测的 seam depth。",
    )

    # ╔══════════════════════════════════════════════════════════════════
    # ║  Pose 扰动  (GNSS/GPS prior simulation)
    # ╚══════════════════════════════════════════════════════════════════
    g = p.add_argument_group("Pose 扰动")
    g.add_argument(
        "--pose_perturb",
        action="store_true",
        help="在读入 scene manifest 后立即扰动输入 pose 先验，用于模拟 GNSS/GPS 误差；默认关闭",
    )
    g.add_argument(
        "--pose_perturb_xy_std",
        type=float,
        default=0.0,
        help="每帧水平平移扰动标准差，单位米；0 表示无水平扰动",
    )
    g.add_argument(
        "--pose_perturb_z_std",
        type=float,
        default=0.0,
        help="每帧高程平移扰动标准差，单位米；0 表示无高程扰动",
    )
    g.add_argument(
        "--pose_perturb_yaw_std_deg",
        type=float,
        default=0.0,
        help="每帧 yaw 扰动标准差，单位度；0 表示无旋转扰动",
    )
    g.add_argument(
        "--pose_perturb_xy_max",
        type=float,
        default=0.0,
        help="水平平移扰动最大范数，单位米；<=0 不截断",
    )
    g.add_argument(
        "--pose_perturb_z_max",
        type=float,
        default=0.0,
        help="高程平移扰动最大绝对值，单位米；<=0 不截断",
    )
    g.add_argument(
        "--pose_perturb_yaw_max_deg",
        type=float,
        default=0.0,
        help="yaw 扰动最大绝对值，单位度；<=0 不截断",
    )
    g.add_argument(
        "--pose_perturb_seed_offset",
        type=int,
        default=930001,
        help="pose 扰动随机种子偏移；最终 seed = --seed + 该偏移",
    )

    # ╔══════════════════════════════════════════════════════════════════
    # ║  坐标平移  (Recenter)
    # ╚══════════════════════════════════════════════════════════════════
    g = p.add_argument_group("坐标平移")
    g.add_argument(
        "--recenter", default="auto",
        choices=["auto", "none", "mean_camera", "first_camera"],
        help="auto 会根据模型类别决定是否 recenter",
    )

    # ╔══════════════════════════════════════════════════════════════════
    # ║  输出控制  (Output)
    # ╚══════════════════════════════════════════════════════════════════
    g = p.add_argument_group("输出控制")
    g.add_argument(
        "--max_points_per_view", type=int, default=250000,
        help="每张图像最多保留的点云数量（GT 和预测共用）",
    )
    g.add_argument(
        "--voxel_downsample", type=float, default=0.05,
        help="输出点云体素下采样分辨率 （米），0 禁用",
    )
    g.add_argument(
        "--point_downsample",
        "--final_point_downsample",
        dest="point_downsample",
        action="store_true",
        default=True,
        help="保存最终 pred/gt 点云前执行全局 voxel 下采样（默认开启）",
    )
    g.add_argument(
        "--no_point_downsample",
        "--no_final_point_downsample",
        dest="point_downsample",
        action="store_false",
        help="保存最终 pred/gt 点云前不执行全局 voxel 下采样",
    )
    g.add_argument(
        "--xy_fill_unmasked",
        action="store_true",
        default=False,
        help="先用 confidence mask 融合主点云，再用 unmasked 点在 XY 未覆盖网格中补洞",
    )
    g.add_argument(
        "--no_xy_fill_unmasked",
        action="store_false",
        dest="xy_fill_unmasked",
        help="关闭 unmasked XY 补洞",
    )
    g.add_argument(
        "--xy_fill_grid_size",
        type=float,
        default=0.0,
        help="XY 补洞网格大小；<=0 自动使用 max(5*voxel_downsample, 0.05)",
    )
    g.add_argument(
        "--xy_fill_max_points_per_chunk",
        type=int,
        default=50000,
        help="每个 chunk 最多取多少 unmasked 候选点用于 XY 补洞；<=0 不限制",
    )
    g.add_argument(
        "--log_chunks", action="store_true", default=True,
        help="保存 chunk artifacts：footprint 总览图和每 chunk PLY （默认开启）",
    )
    g.add_argument(
        "--no_log_chunks", action="store_false", dest="log_chunks",
        help="禁止保存 chunk artifacts",
    )
    g.add_argument(
        "--chunk_cache_workers",
        type=int,
        default=1,
        help="每 chunk 预测结果落盘的后台线程数；0 表示同步写入",
    )
    g.add_argument(
        "--chunk_cache_max_pending",
        type=int,
        default=0,
        help="最多允许多少个 chunk cache 写任务挂起；<=0 时等于 workers，防止内存堆积",
    )
    g.add_argument(
        "--keep_chunk_cache",
        action="store_true",
        help="运行结束后保留逐 chunk/逐视图预测 NPZ 与关系 manifest，便于复算 Seam Error",
    )
    g.add_argument(
        "--log_images", action="store_true",
        help="在 RRD 中记录输入图像",
    )
    g.add_argument(
        "--log_file",
        default=None,
        help="完整运行日志路径；默认 output_rrd 去后缀目录下 logs/pipeline.log",
    )
    g.add_argument(
        "--no_file_log",
        action="store_true",
        help="不保存 stdout/stderr 运行日志文件",
    )
    g.add_argument(
        "--chunk_tqdm",
        action="store_true",
        default=True,
        help="chunk 预测阶段使用 tqdm 进度条（默认开启）",
    )
    g.add_argument(
        "--no_chunk_tqdm",
        action="store_false",
        dest="chunk_tqdm",
        help="禁用 chunk tqdm，使用普通阶段日志",
    )
    g.add_argument(
        "--seed", type=int, default=0,
        help="随机种子 （采样 & 对齐）",
    )

    # ╔══════════════════════════════════════════════════════════════════
    # ║  Rerun 可视化  (Visualization)
    # ╚══════════════════════════════════════════════════════════════════
    g = p.add_argument_group("Rerun 可视化")
    g.add_argument(
        "--view_coordinates", default="RDF",
        help="Rerun 坐标系，如 RDF / RIGHT_HAND_Z_UP",
    )
    g.add_argument(
        "--background", type=int, nargs=3, default=[255, 255, 255],
        help="Rerun 背景色 RGB",
    )
    g.add_argument(
        "--hide_grid", action="store_true",
        help="隐藏 Rerun 网格",
    )
    g.add_argument(
        "--show_world_axes", action="store_true", default=True,
        help="显示世界坐标轴标记 （默认开启）",
    )
    g.add_argument(
        "--no_world_axes", action="store_false", dest="show_world_axes",
        help="隐藏世界坐标轴标记",
    )
    g.add_argument(
        "--world_axes_origin", default="scene_center",
        choices=["zero", "scene_center"],
        help="世界坐标轴原点位置",
    )
    g.add_argument(
        "--world_up_axis", default="z",
        help="世界坐标系 up 轴",
    )
    g.add_argument(
        "--world_axis_size", type=float, default=0.0,
        help="世界坐标轴长度；<=0 自动推断",
    )
    g.add_argument(
        "--world_axis_size_ratio", type=float, default=0.12,
        help="自动推断时轴长占场景对角线的比例",
    )
    g.add_argument(
        "--world_axis_min_size", type=float, default=0.1,
        help="世界坐标轴最小长度",
    )
    g.add_argument(
        "--world_axis_up_offset_ratio", type=float, default=1.2,
        help="世界坐标轴原点向上偏移比例",
    )
    g.add_argument(
        "--world_axis_radius", type=float, default=0.0,
        help="世界坐标轴线条半径",
    )
    g.add_argument(
        "--point_radius", type=float, default=0.0,
        help="点云半径",
    )
    g.add_argument(
        "--camera_axis_size", type=float, default=0.0,
        help="相机轴长度；<=0 自动推断",
    )
    g.add_argument(
        "--camera_axis_radius", type=float, default=0.0,
        help="相机轴线条半径",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Deferred post-processing
# ---------------------------------------------------------------------------
def _pred_tensor_hw_map(
    tensor: Optional[torch.Tensor],
    *,
    reduce_channels: bool = False,
) -> Optional[torch.Tensor]:
    if tensor is None or not torch.is_tensor(tensor):
        return None
    value = tensor
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    elif value.ndim == 3 and reduce_channels:
        value = value.float().mean(dim=-1)
    if value.ndim != 2:
        return None
    return value


def _zero_pred_depth_at_mask(pred: Dict[str, torch.Tensor], drop_mask: torch.Tensor) -> None:
    drop_mask = drop_mask.to(dtype=torch.bool)
    for key in ("pts3d_cam", "pts3d"):
        tensor = pred.get(key)
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if tensor.ndim == 4 and tensor.shape[0] == 1 and tensor.shape[1:3] == drop_mask.shape:
            tensor = tensor.clone()
            tensor[0][drop_mask] = 0
            pred[key] = tensor
        elif tensor.ndim == 3 and tensor.shape[:2] == drop_mask.shape:
            tensor = tensor.clone()
            tensor[drop_mask] = 0
            pred[key] = tensor

    for key in ("depth_along_ray", "depth", "depthmap"):
        tensor = pred.get(key)
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if tensor.ndim == 4 and tensor.shape[0] == 1 and tensor.shape[1:3] == drop_mask.shape:
            tensor = tensor.clone()
            if tensor.shape[-1] == 1:
                tensor[0, ..., 0][drop_mask] = 0
            else:
                tensor[0][drop_mask] = 0
            pred[key] = tensor
        elif tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[1:] == drop_mask.shape:
            tensor = tensor.clone()
            tensor[0][drop_mask] = 0
            pred[key] = tensor
        elif tensor.ndim == 2 and tensor.shape == drop_mask.shape:
            tensor = tensor.clone()
            tensor[drop_mask] = 0
            pred[key] = tensor


def _tensor_to_np_float(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach()
    if value.dtype in (torch.float16, torch.bfloat16):
        value = value.float()
    return value.cpu().numpy().astype(np.float32, copy=False)


def _tensor_bool_to_np(mask: torch.Tensor) -> np.ndarray:
    return mask.detach().to(dtype=torch.bool).cpu().numpy()


def apply_optional_keep_masks_to_valid_masks(
    pred_valid_masks: Sequence[np.ndarray],
    keep_masks: Optional[Sequence[Optional[np.ndarray]]],
) -> List[np.ndarray]:
    out: List[np.ndarray] = []

    for i, valid in enumerate(pred_valid_masks):
        mask = np.asarray(valid, dtype=bool).copy()

        if keep_masks is not None and i < len(keep_masks):
            keep = keep_masks[i]
            if keep is not None:
                keep = np.asarray(keep, dtype=bool)
                if keep.shape == mask.shape:
                    mask &= keep
                else:
                    print(
                        f"[WARN] confidence keep mask shape {keep.shape} "
                        f"does not match valid mask shape {mask.shape}; skip for view {i}"
                    )

        out.append(mask)

    return out


def _colorize_debug_scalar(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    cmap_name: str = "turbo",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    import matplotlib

    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if vmin is None or vmax is None:
        if valid.any():
            selected = values[valid]
            lo, hi = np.percentile(selected, [2.0, 98.0])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = float(np.nanmin(selected))
                hi = float(np.nanmax(selected))
            if hi <= lo:
                hi = lo + 1.0
            vmin = float(lo)
            vmax = float(hi)
        else:
            vmin = 0.0
            vmax = 1.0
    norm = np.clip((values - float(vmin)) / max(float(vmax) - float(vmin), 1e-8), 0.0, 1.0)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb, float(vmin), float(vmax)


def _save_depth_conf_filter_debug_view(
    *,
    path: Path,
    rgb: np.ndarray,
    conf: np.ndarray,
    depth_before: np.ndarray,
    depth_after: np.ndarray,
    drop_mask: np.ndarray,
    threshold: float,
    quantile: float,
    chunk_id: int,
    stem: str,
) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = conf.shape
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)

    conf_valid = np.isfinite(conf)
    depth_valid = np.isfinite(depth_before) & (depth_before > 0)
    _, conf_vmin, conf_vmax = _colorize_debug_scalar(conf, conf_valid, cmap_name="viridis")
    _, depth_vmin, depth_vmax = _colorize_debug_scalar(depth_before, depth_valid, cmap_name="turbo")
    mask_vis = np.zeros((height, width, 3), dtype=np.uint8)
    mask_vis[..., 0] = np.asarray(drop_mask, dtype=np.uint8) * 255

    panels = [
        ("RGB", rgb, None, None, None),
        ("confidence", None, conf, conf_valid, (conf_vmin, conf_vmax)),
        ("drop mask", mask_vis, None, None, None),
        ("depth before", None, depth_before, depth_valid, (depth_vmin, depth_vmax)),
        (
            "depth after",
            None,
            depth_after,
            np.isfinite(depth_after) & (depth_after > 0),
            (depth_vmin, depth_vmax),
        ),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.0), dpi=150)
    for ax, (title, image, scalar, valid, value_range) in zip(axes, panels):
        if scalar is None:
            ax.imshow(image)
        else:
            cmap_name = "viridis" if title == "confidence" else "turbo"
            cmap = matplotlib.colormaps.get_cmap(cmap_name).copy()
            cmap.set_bad("black")
            masked = np.ma.masked_where(~valid, scalar)
            im = ax.imshow(masked, cmap=cmap, vmin=value_range[0], vmax=value_range[1])
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"chunk {chunk_id:03d} {stem} | conf q={quantile:.3f}, threshold={threshold:.6g}"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def apply_depth_confidence_filter_to_preds(
    *,
    preds: Sequence[Dict[str, torch.Tensor]],
    rgbs: Sequence[np.ndarray],
    stems: Sequence[str],
    chunk_id: int,
    conf_quantile: float,
    pred_min_depth: float,
    debug_dir: Optional[Path] = None,
) -> Tuple[Dict[str, object], List[Optional[np.ndarray]]]:
    q = float(conf_quantile)
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"conf_quantile must be in [0, 1], got {q}")

    keep_masks: List[Optional[np.ndarray]] = [None for _ in preds]

    if q <= 0.0:
        return {
            "enabled": False,
            "quantile": q,
            "num_views": int(len(preds)),
            "note": "confidence filtering disabled; original predicted depth/points are preserved",
        }, keep_masks

    per_view: List[Dict[str, object]] = []
    total_valid = 0
    total_dropped = 0
    total_missing_conf = 0

    for local_i, pred in enumerate(preds):
        stem = str(stems[local_i]) if local_i < len(stems) else f"view_{local_i:03d}"

        conf_t = _pred_tensor_hw_map(pred.get("conf"), reduce_channels=True)
        pts_cam_t = pred.get("pts3d_cam")

        if conf_t is None or pts_cam_t is None or not torch.is_tensor(pts_cam_t):
            total_missing_conf += 1
            per_view.append(
                {
                    "local_index": int(local_i),
                    "stem": stem,
                    "valid": False,
                    "reason": "missing_conf_or_pts3d_cam",
                }
            )
            continue

        pts_cam = pts_cam_t[0] if pts_cam_t.ndim == 4 and pts_cam_t.shape[0] == 1 else pts_cam_t

        if pts_cam.ndim != 3 or pts_cam.shape[-1] != 3 or pts_cam.shape[:2] != conf_t.shape:
            total_missing_conf += 1
            per_view.append(
                {
                    "local_index": int(local_i),
                    "stem": stem,
                    "valid": False,
                    "reason": "shape_mismatch",
                    "conf_shape": list(conf_t.shape),
                    "pts3d_cam_shape": list(pts_cam.shape),
                }
            )
            continue

        depth_before_t = pts_cam[..., 2].detach().float()
        conf = conf_t.detach().float()

        valid = (
            torch.isfinite(conf)
            & torch.isfinite(pts_cam).all(dim=-1)
            & torch.isfinite(depth_before_t)
            & (depth_before_t > float(pred_min_depth))
        )

        num_valid = int(valid.sum().item())
        if num_valid <= 0:
            per_view.append(
                {
                    "local_index": int(local_i),
                    "stem": stem,
                    "valid": False,
                    "reason": "no_valid_depth",
                }
            )
            continue

        threshold = torch.quantile(conf[valid], q)
        keep_mask = valid & (conf >= threshold)
        drop_mask = valid & ~keep_mask

        keep_masks[local_i] = _tensor_bool_to_np(keep_mask)

        num_drop = int(drop_mask.sum().item())
        total_valid += num_valid
        total_dropped += num_drop

        per_view.append(
            {
                "local_index": int(local_i),
                "stem": stem,
                "valid": True,
                "threshold": float(threshold.detach().cpu().item()),
                "num_valid": num_valid,
                "num_dropped": num_drop,
                "drop_fraction": float(num_drop / max(1, num_valid)),
            }
        )

        if debug_dir is not None:
            depth_before_np = _tensor_to_np_float(depth_before_t)
            keep_mask_np = keep_masks[local_i]
            depth_after = np.where(keep_mask_np, depth_before_np, 0.0).astype(np.float32)

            debug_path = debug_dir / f"chunk_{chunk_id:03d}" / f"{local_i:02d}_{stem}_depth_conf_filter.png"
            _save_depth_conf_filter_debug_view(
                path=debug_path,
                rgb=np.asarray(rgbs[local_i], dtype=np.uint8),
                conf=_tensor_to_np_float(conf),
                depth_before=depth_before_np,
                depth_after=depth_after,
                drop_mask=_tensor_bool_to_np(drop_mask),
                threshold=float(threshold.detach().cpu().item()),
                quantile=q,
                chunk_id=int(chunk_id),
                stem=stem,
            )

    return {
        "enabled": True,
        "quantile": q,
        "num_views": int(len(preds)),
        "num_views_missing_conf": int(total_missing_conf),
        "num_valid_pixels": int(total_valid),
        "num_dropped_pixels": int(total_dropped),
        "drop_fraction": float(total_dropped / max(1, total_valid)),
        "per_view": per_view,
        "note": "confidence filtering is stored as masks; original predicted depth/points are preserved",
    }, keep_masks


def apply_deferred_multiview_filter_to_chunk_records(
    chunk_records: List[Dict[str, object]],
    args: argparse.Namespace,
    device: torch.device,
    run_logger: Optional[RunLogger] = None,
) -> None:
    """Apply multi-view consistency after all chunks have been predicted."""
    if not args.mv_consistency:
        return

    print(
        "[INFO] Applying deferred multi-view consistency filtering "
        "after all chunk predictions."
    )

    scene_before = 0
    scene_after = 0

    mv_iter = iter_progress(
        chunk_records,
        desc="MV consistency",
        total=len(chunk_records),
        enabled=bool(getattr(args, "chunk_tqdm", True)),
    )
    for record in mv_iter:
        chunk_id = int(record["chunk_id"])
        cached = load_chunk_cache(
            record,
            keys=[
                "_pred_maps",
                "_pred_valid_masks",
                "_chunk_intrinsics",
                "_chunk_point_local_indices",
                "_core_local_indices",
                "rgbs",
            ],
        )

        pred_maps = cached.get("_pred_maps", None)
        pred_valid_masks = cached.get("_pred_valid_masks", None)
        chunk_intrinsics = cached.get("_chunk_intrinsics", None)
        chunk_point_local_indices = cached.get("_chunk_point_local_indices", None)
        core_local_indices = cached.get("_core_local_indices", None)
        rgbs = cached.get("rgbs", None)

        if (
            pred_maps is None
            or pred_valid_masks is None
            or chunk_intrinsics is None
            or chunk_point_local_indices is None
            or core_local_indices is None
            or rgbs is None
        ):
            record["mv_consistency_meta"] = {
                "enabled": False,
                "reason": "missing deferred mv cached fields",
            }
            continue

        filtered_masks, mv_meta = apply_mvsnet_style_multiview_filter(
            pred_maps=pred_maps,
            pred_valid_masks=pred_valid_masks,
            pred_cams=record["pred_cams"],
            intrinsics=chunk_intrinsics,
            device=device,
            enabled=True,
            min_support=int(args.mv_min_support),
            max_neighbors=int(args.mv_max_neighbors),
            conf_threshold=float(args.mv_conf_threshold),
            depth_abs_tol=float(args.mv_depth_abs_tol),
            depth_rel_tol=float(args.mv_depth_rel_tol),
            point_abs_tol=float(args.mv_point_abs_tol),
            point_rel_tol=float(args.mv_point_rel_tol),
            min_depth=float(args.pred_min_depth),
            use_point_check=not bool(args.mv_no_point_check),
        )

        chunk_points, chunk_colors = points_from_maps(
            pred_maps=pred_maps,
            pred_valid_masks=filtered_masks,
            rgbs=rgbs,
            local_indices=chunk_point_local_indices,
        )
        core_points, core_colors = points_from_maps(
            pred_maps=pred_maps,
            pred_valid_masks=filtered_masks,
            rgbs=rgbs,
            local_indices=core_local_indices,
        )

        update_chunk_cache(
            record,
            {
                "chunk_pred_points": chunk_points,
                "chunk_pred_colors": chunk_colors,
                "core_pred_points": core_points,
                "core_pred_colors": core_colors,
                "_pred_valid_masks": filtered_masks,
            },
        )
        record["mv_consistency_meta"] = mv_meta
        record["num_chunk_pred_points_filtered"] = int(chunk_points.shape[0])
        record["num_core_pred_points"] = int(core_points.shape[0])

        before = int(mv_meta.get("total_before", 0))
        after = int(mv_meta.get("total_after", 0))
        scene_before += before
        scene_after += after

        if hasattr(mv_iter, "set_postfix"):
            mv_iter.set_postfix(
                chunk=f"{chunk_id:03d}",
                keep=f"{float(mv_meta.get('keep_ratio', 1.0)):.3f}",
                after=after,
            )
        if run_logger is not None:
            run_logger.detail(
                f"[chunk {chunk_id:03d}] deferred mv consistency: "
                f"before={before}, after={after}, dropped={before - after}, "
                f"keep_ratio={float(mv_meta.get('keep_ratio', 1.0)):.3f}, "
                f"support>={args.mv_min_support}, neighbors<={args.mv_max_neighbors}, "
                f"conf>={args.mv_conf_threshold}"
            )

    print(
        "[INFO] Deferred multi-view filtering summary: "
        f"before={scene_before}, after={scene_after}, "
        f"dropped={scene_before - scene_after}, "
        f"keep_ratio={scene_after / max(scene_before, 1):.3f}"
    )


def run_deferred_dom_rendering(
    chunk_records: List[Dict[str, object]],
    args: argparse.Namespace,
    device: torch.device,
    gsplat_summary: Optional[Dict[str, object]] = None,
) -> Optional[Dict[str, object]]:
    """Render a full DOM after all chunks and optional MV filtering.

    When --gsplat_refine is enabled and optimized bundles exist, DOM is rendered
    from the optimized 3DGS gaussians.npz. Otherwise falls back to raw pred_maps.
    """
    if not bool(args.render_dom):
        return None

    output_rrd_path = Path(args.output_rrd).expanduser().resolve()
    if args.dom_output_dir:
        output_dir = Path(args.dom_output_dir).expanduser().resolve()
    else:
        output_dir = output_rrd_path.with_suffix("") / "orthodom"
    fused_points_path = output_rrd_path.with_suffix("") / "eval" / "pred_points.ply"

    use_optimized_gsplat = (
        bool(args.gsplat_refine)
        and gsplat_summary is not None
        and bool(gsplat_summary.get("enabled", False))
        and len(gsplat_summary.get("bundles", [])) > 0
    )

    if use_optimized_gsplat:
        print(
            "[INFO] Rendering DOM from optimized gsplat bundles: "
            f"num_bundles={len(gsplat_summary.get('bundles', []))}"
        )

        dom_meta = render_orthodom_from_gsplat_summary(
            gsplat_summary=gsplat_summary,
            fallback_chunk_records=chunk_records,
            fallback_fused_points_path=fused_points_path,
            output_dir=output_dir,
            device=device,
            dom_gsd=float(args.dom_gsd),
            dom_axes=str(args.dom_axes),
            dom_up_axis=str(args.dom_up_axis),
            dom_tile_px=int(args.dom_tile_px),
            dom_margin_px=int(args.dom_margin_px),
            dom_max_gaussians_per_tile=int(args.dom_max_gaussians_per_tile),
            dom_splat_scale=float(args.dom_splat_scale),
            dom_dsm_smooth_radius_px=int(args.dom_dsm_smooth_radius_px),
            dom_dsm_smooth_sigma=float(args.dom_dsm_smooth_sigma),
            dom_dsm_smooth_iterations=int(args.dom_dsm_smooth_iterations),
            dom_dsm_smooth_min_weight=float(args.dom_dsm_smooth_min_weight),
            dom_save_contours=bool(args.dom_save_contours),
            dom_opacity=float(args.dom_opacity),
            dom_gsd_stride=int(args.dom_gsd_stride),
            dom_bounds_quantile_min=float(args.dom_bounds_quantile_min),
            dom_bounds_quantile_max=float(args.dom_bounds_quantile_max),
            dom_padding_m=float(args.dom_padding_m),
            dom_max_pixels=int(args.dom_max_pixels),
            dom_allow_large=bool(args.dom_allow_large),
            dom_rasterize_mode=str(args.dom_rasterize_mode),
            dom_save_tiles=not bool(args.dom_no_save_tiles),
            dom_epsg=int(args.dom_epsg) if int(args.dom_epsg) > 0 else None,
            seed=int(args.seed) + 820001,
        )
    else:
        print(
            "[INFO] Rendering DOM/DSM from fused final point cloud "
            "(gsplat refinement disabled or no optimized bundles)."
        )

        dom_meta = render_orthodom_from_fused_points(
            fused_points_path=fused_points_path,
            output_dir=output_dir,
            dom_gsd=float(args.dom_gsd),
            dom_axes=str(args.dom_axes),
            dom_up_axis=str(args.dom_up_axis),
            dom_splat_scale=float(args.dom_splat_scale),
            dom_dsm_smooth_radius_px=int(args.dom_dsm_smooth_radius_px),
            dom_dsm_smooth_sigma=float(args.dom_dsm_smooth_sigma),
            dom_dsm_smooth_iterations=int(args.dom_dsm_smooth_iterations),
            dom_dsm_smooth_min_weight=float(args.dom_dsm_smooth_min_weight),
            dom_save_contours=bool(args.dom_save_contours),
            dom_tile_px=int(args.dom_tile_px),
            dom_bounds_quantile_min=float(args.dom_bounds_quantile_min),
            dom_bounds_quantile_max=float(args.dom_bounds_quantile_max),
            dom_padding_m=float(args.dom_padding_m),
            dom_max_pixels=int(args.dom_max_pixels),
            dom_allow_large=bool(args.dom_allow_large),
            dom_save_tiles=not bool(args.dom_no_save_tiles),
            dom_epsg=int(args.dom_epsg) if int(args.dom_epsg) > 0 else None,
        )

    # Attach lightweight DOM meta to all records for sidecar visibility.
    lightweight = {
        "enabled": bool(dom_meta.get("enabled", False)),
        "output_dir": dom_meta.get("output_dir", str(output_dir)),
        "rgb_path": dom_meta.get("rgb_path", None),
        "alpha_path": dom_meta.get("alpha_path", None),
        "dsm_npy_path": dom_meta.get("dsm_npy_path", None),
        "dsm_elevation_png_path": dom_meta.get("dsm_elevation_png_path", None),
        "dsm_elevation_svg_path": dom_meta.get("dsm_elevation_svg_path", None),
        "dsm_contour_png_path": dom_meta.get("dsm_contour_png_path", None),
        "dsm_contour_svg_path": dom_meta.get("dsm_contour_svg_path", None),
        "geotiff_path": dom_meta.get("geotiff_path", None),
        "gsd": dom_meta.get("gsd", None),
        "image_size": dom_meta.get("image_size", None),
    }
    for record in chunk_records:
        record["dom_meta"] = lightweight

    return dom_meta


def run_deferred_gsplat_bundle_refinement(
    chunk_records: List[Dict[str, object]],
    args: argparse.Namespace,
    device: torch.device,
) -> Optional[Dict[str, object]]:
    """Run 3DGS optimization after all feed-forward chunks are predicted.

    RRD/eval outputs remain pre-gsplat. Optimized 3DGS is saved separately.
    Returns the gsplat summary dict for downstream DOM rendering.
    """
    if not args.gsplat_refine:
        return None

    output_rrd_path = Path(args.output_rrd).expanduser().resolve()
    output_dir = output_rrd_path.with_suffix("") / "gsplat"
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = build_gsplat_optimization_bundles(
        chunk_records=chunk_records,
        single_max_images=int(args.gsplat_single_max_images),
        max_images_per_bundle=int(args.gsplat_max_images_per_bundle),
        min_core_images_per_bundle=int(args.gsplat_min_core_images_per_bundle),
    )

    print(
        "[INFO] Deferred gsplat refinement: "
        f"num_bundles={len(bundles)}, output_dir={output_dir}"
    )

    all_meta: List[Dict[str, object]] = []

    for bundle in bundles:
        bundle_name = str(bundle["name"])
        bundle_id = int(bundle["bundle_id"])

        inputs = collect_gsplat_bundle_inputs(
            chunk_records=chunk_records,
            bundle=bundle,
        )

        num_init = len(inputs["init_pred_maps"])
        num_render = len(inputs["render_rgbs"])

        if num_init == 0 or num_render == 0:
            gs_meta: Dict[str, object] = {
                "enabled": False,
                "reason": "empty init or render inputs",
                **dict(bundle),
            }
            all_meta.append(gs_meta)
            print(
                f"[gsplat:{bundle_name}] skip: "
                f"init_views={num_init}, render_views={num_render}"
            )
            continue

        gs_meta = optimize_and_save_gsplat_bundle(
            init_pred_maps=inputs["init_pred_maps"],
            init_pred_valid_masks=inputs["init_pred_valid_masks"],
            init_rgbs=inputs["init_rgbs"],
            render_views_raw=inputs["render_views_raw"],
            render_rgbs=inputs["render_rgbs"],
            render_cams=inputs["render_cams"],
            output_dir=output_dir,
            bundle_name=bundle_name,
            bundle_meta={
                **dict(bundle),
                "init_global_indices": inputs["init_global_indices"],
                "render_global_indices": inputs["render_global_indices"],
            },
            device=device,
            steps=int(args.gsplat_steps),
            max_gaussians=int(args.gsplat_max_gaussians),
            batch_views=int(args.gsplat_batch_views),
            render_scale=float(args.gsplat_render_scale),
            seed=int(args.seed) + 910001 + bundle_id,

            # official-style splat params
            sh_degree=int(args.gsplat_sh_degree),
            sh_degree_interval=int(args.gsplat_sh_degree_interval),
            init_opacity=float(args.gsplat_init_opacity),
            init_scale=float(args.gsplat_init_scale),
            ssim_lambda=float(args.gsplat_ssim_lambda),
            means_lr=float(args.gsplat_means_lr),
            scales_lr=float(args.gsplat_scales_lr),
            opacities_lr=float(args.gsplat_opacities_lr),
            quats_lr=float(args.gsplat_quats_lr),
            sh0_lr=float(args.gsplat_sh0_lr),
            shN_lr=float(args.gsplat_shN_lr),

            # strategy
            strategy_name=str(args.gsplat_strategy),
            refine_start_iter=int(args.gsplat_refine_start_iter),
            refine_stop_iter=int(args.gsplat_refine_stop_iter),
            refine_every=int(args.gsplat_refine_every),
            reset_every=int(args.gsplat_reset_every),
            prune_opa=float(args.gsplat_prune_opa),
            grow_grad2d=float(args.gsplat_grow_grad2d),
            grow_scale3d=float(args.gsplat_grow_scale3d),
            grow_scale2d=float(args.gsplat_grow_scale2d),
            prune_scale3d=float(args.gsplat_prune_scale3d),
            prune_scale2d=float(args.gsplat_prune_scale2d),
            absgrad=bool(args.gsplat_absgrad),
            strategy_verbose=not bool(args.gsplat_strategy_quiet),

            # optimization behavior
            optimize_means=True,
            optimize_pose=not bool(args.gsplat_no_pose_opt),
            pose_lr=float(args.gsplat_pose_lr),
            pose_reg=float(args.gsplat_pose_reg),
            opacity_reg=float(args.gsplat_opacity_reg),
            scale_reg=float(args.gsplat_scale_reg),

            # rasterization / optimizer mode
            packed=bool(args.gsplat_packed),
            sparse_grad=bool(args.gsplat_sparse_grad),
            visible_adam=bool(args.gsplat_visible_adam),
            antialiased=bool(args.gsplat_antialiased),
            random_bkgd=bool(args.gsplat_random_bkgd),

            # logging/output
            log_every=int(args.gsplat_log_every),
            use_tqdm=not bool(args.gsplat_no_tqdm),
            save_rendered_views=bool(args.gsplat_save_rendered_views),
            render_output_max_views=int(args.gsplat_render_output_max_views),
            render_output_stride=int(args.gsplat_render_output_stride),
        )
        all_meta.append(gs_meta)

        print(
            f"[gsplat:{bundle_name}] saved: "
            f"gaussians={int(gs_meta.get('num_gaussians', 0))}, "
            f"init_views={int(gs_meta.get('num_init_core_views', 0))}, "
            f"render_views={int(gs_meta.get('num_render_views', 0))}, "
            f"loss={float(gs_meta.get('final_loss', float('nan'))):.6g}, "
            f"ply={gs_meta.get('ply_path', '<none>')}"
        )

    summary = {
        "enabled": True,
        "output_dir": str(output_dir),
        "num_bundles": int(len(all_meta)),
        "single_max_images": int(args.gsplat_single_max_images),
        "max_images_per_bundle": int(args.gsplat_max_images_per_bundle),
        "min_core_images_per_bundle": int(args.gsplat_min_core_images_per_bundle),
        "bundles": all_meta,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Saved gsplat summary: {summary_path}")

    # Attach lightweight meta back to chunk records for sidecar.
    bundle_by_record: Dict[int, List[Dict[str, object]]] = {}
    for meta in all_meta:
        for rid in meta.get("record_ids", []):
            bundle_by_record.setdefault(int(rid), []).append(
                {
                    "bundle_name": meta.get("bundle_name", meta.get("name", "")),
                    "enabled": bool(meta.get("enabled", False)),
                    "ply_path": meta.get("ply_path", None),
                    "num_gaussians": int(meta.get("num_gaussians", 0)),
                    "num_render_views": int(meta.get("num_render_views", 0)),
                }
            )

    for rid, record in enumerate(chunk_records):
        record["gsplat_meta"] = {
            "enabled": bool(args.gsplat_refine),
            "deferred": True,
            "bundles": bundle_by_record.get(int(rid), []),
        }

    return summary


def save_spatial_chunk_core_footprint_xy_visualization(
    meta: Dict[str, object],
    grid_meta: Dict[str, object],
    chunks: Sequence[Dict[str, object]],
    output_rrd: Path,
    point_size: float = 12.0,
    bg_point_size: float = 3.0,
    point_alpha: float = 0.78,
    bg_alpha: float = 0.22,
    label_size: float = 11.0,
    font_scale: float = 1.35,
    padding_ratio: float = 0.02,
    show_legend: bool = False,
    legend_cols: int = 0,
    legend_max_rows: int = 16,
) -> Optional[Path]:
    """
    Save a paper-style chunk visualization on the footprint plane.

    Points are footprint centers, not camera centers. Gray background points
    show all selected footprints; colored points show core footprints per chunk.
    Seam/overlap points are not drawn to avoid clutter.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Failed to import matplotlib for chunk visualization: {exc}")
        return None

    stems = list(meta.get("stems", []))
    footprint_centers = grid_meta.get("footprint_centers", None)

    if footprint_centers is None:
        print(
            "[WARN] Chunk footprint visualization skipped: "
            "grid_meta['footprint_centers'] is missing. "
            "Please store footprint_centers in build_footprint_tree_chunks()."
        )
        return None

    centers = np.asarray(footprint_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        print(
            "[WARN] Chunk footprint visualization skipped: "
            f"invalid footprint_centers shape={centers.shape}, expected [N, 2]."
        )
        return None

    if len(stems) > 0 and centers.shape[0] != len(stems):
        print(
            "[WARN] Chunk footprint visualization: "
            f"num footprint centers={centers.shape[0]} != num stems={len(stems)}. "
            "Will still draw valid indices."
        )

    finite_all = np.isfinite(centers).all(axis=1)
    if not bool(finite_all.any()):
        print("[WARN] Chunk footprint visualization skipped: no finite footprint centers.")
        return None

    chunk_core_xy: List[Tuple[int, np.ndarray]] = []
    for fallback_chunk_id, chunk in enumerate(chunks):
        chunk_id = int(chunk.get("chunk_id", fallback_chunk_id))
        core_indices = np.asarray(chunk.get("core_indices", []), dtype=np.int64)
        if core_indices.size == 0:
            continue

        valid = (
            (core_indices >= 0)
            & (core_indices < centers.shape[0])
        )
        core_indices = core_indices[valid]
        if core_indices.size == 0:
            continue

        xy = centers[core_indices]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if xy.shape[0] == 0:
            continue

        chunk_core_xy.append((chunk_id, xy))

    if not chunk_core_xy:
        print("[WARN] Chunk footprint visualization skipped: no valid chunk core footprint centers.")
        return None

    output_dir = Path(output_rrd).expanduser().resolve().with_suffix("")
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_path = vis_dir / "chunk_core_footprint_xy.svg"
    png_path = vis_dir / "chunk_core_footprint_xy.png"

    all_xy = centers[finite_all]
    n_chunks = len(chunk_core_xy)
    n_points = int(all_xy.shape[0])

    def _auto_marker_size(value: float, default: float, min_size: float) -> float:
        if float(value) > 0:
            return float(value)
        if n_points <= 0:
            return float(default)
        return float(max(min_size, min(default, 1500.0 / max(np.sqrt(n_points), 1.0))))

    bg_marker_size = _auto_marker_size(bg_point_size, 3.0, 0.6)
    core_marker_size = _auto_marker_size(point_size, 12.0, 2.0)
    point_alpha = float(np.clip(point_alpha, 0.0, 1.0))
    bg_alpha = float(np.clip(bg_alpha, 0.0, 1.0))
    font_scale = max(0.1, float(font_scale))
    padding_ratio = max(0.0, float(padding_ratio))
    label_size = float(label_size) * font_scale
    axes_label_size = (
        label_size if label_size > 0 else 13.0 * font_scale
    )
    legend_max_rows = max(1, int(legend_max_rows))
    if int(legend_cols) > 0:
        legend_ncol = int(legend_cols)
    else:
        legend_ncol = int(max(1, min(4, np.ceil(n_chunks / legend_max_rows))))
    legend_rows = int(np.ceil(n_chunks / max(legend_ncol, 1)))

    split_lines = grid_meta.get("footprint_split_lines", [])
    split_lines_frame = str(grid_meta.get("footprint_split_lines_frame", "original"))
    flight_frame = grid_meta.get("footprint_flight_frame", None)

    def _flight_to_plot_xy(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        shape = pts.shape
        pts = pts.reshape(-1, 2)

        if split_lines_frame != "flight_aligned" or not isinstance(flight_frame, dict):
            return pts.reshape(shape)

        try:
            origin = np.asarray(flight_frame["origin"], dtype=np.float64).reshape(2)
            rot_to_flight = np.asarray(
                flight_frame["rot_to_flight"],
                dtype=np.float64,
            ).reshape(2, 2)
        except Exception:
            return pts.reshape(shape)

        # Inverse of:
        #   p_local = (p_world - origin) @ rot_to_flight.T
        # is:
        #   p_world = p_local @ rot_to_flight + origin
        out = pts @ rot_to_flight + origin[None, :]
        return out.reshape(shape)

    # Use actual footprint centers for the plot bounds. Including the rotated
    # root-region corners creates large empty margins that contain neither
    # observations nor chunk cores. Split lines outside these compact bounds
    # are intentionally clipped by the axes.
    bounds_xy = all_xy

    # Keep SVG text editable and explicitly tagged as Times New Roman. Raster
    # output requires Times New Roman to be installed in the runtime environment.
    try:
        from matplotlib import font_manager

        font_manager.findfont("Times New Roman", fallback_to_default=False)
    except Exception:
        print(
            "[WARN] Times New Roman is not installed. SVG text will retain the "
            "Times New Roman font-family declaration, but raster PNG rendering "
            "may use a fallback."
        )

    with plt.rc_context(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.sf": "Times New Roman",
            "mathtext.cal": "Times New Roman:italic",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11.0 * font_scale,
            "axes.titlesize": 14.0 * font_scale,
            "axes.labelsize": axes_label_size,
            "xtick.labelsize": 11.0 * font_scale,
            "ytick.labelsize": 11.0 * font_scale,
            "legend.fontsize": (8.5 if n_chunks > 20 else 9.5) * font_scale,
            "axes.linewidth": 0.8,
        }
    ):
        # An optional legend is placed inside the axes and therefore does not
        # require an extra empty strip beside the plot.
        fig_w = 7.2
        fig_h = 5.2 + 0.10 * max(0, min(legend_rows, 18) - 12)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

        # Draw adaptive-tree split lines first.
        if isinstance(split_lines, list) and len(split_lines) > 0:
            for item in split_lines:
                if not isinstance(item, dict):
                    continue

                try:
                    axis = int(item.get("axis", -1))
                    threshold = float(item.get("threshold", np.nan))
                    rmin = np.asarray(
                        item.get("region_min", None),
                        dtype=np.float64,
                    ).reshape(-1)
                    rmax = np.asarray(
                        item.get("region_max", None),
                        dtype=np.float64,
                    ).reshape(-1)
                    depth = int(item.get("depth", 0))
                except Exception:
                    continue

                if (
                    axis not in (0, 1)
                    or rmin.shape[0] < 2
                    or rmax.shape[0] < 2
                    or not np.isfinite(threshold)
                    or not np.isfinite(rmin[:2]).all()
                    or not np.isfinite(rmax[:2]).all()
                ):
                    continue

                # Root split is darker/thicker; deeper splits are lighter/thinner.
                alpha = max(0.18, 0.85 - 0.07 * float(depth))
                linewidth = max(0.45, 1.8 - 0.12 * float(depth))

                if axis == 0:
                    # Local vertical split line: local_x = threshold.
                    # In original XY this becomes a line perpendicular to the
                    # main flight direction.
                    line_local = np.asarray(
                        [
                            [threshold, float(rmin[1])],
                            [threshold, float(rmax[1])],
                        ],
                        dtype=np.float64,
                    )
                else:
                    # Local horizontal split line: local_y = threshold.
                    line_local = np.asarray(
                        [
                            [float(rmin[0]), threshold],
                            [float(rmax[0]), threshold],
                        ],
                        dtype=np.float64,
                    )

                line_xy = _flight_to_plot_xy(line_local)

                ax.plot(
                    line_xy[:, 0],
                    line_xy[:, 1],
                    color="0.18",
                    alpha=alpha,
                    linewidth=linewidth,
                    zorder=1.5,
                )

        # All footprint centers as an unlabeled background layer.
        ax.scatter(
            all_xy[:, 0],
            all_xy[:, 1],
            s=bg_marker_size,
            c="0.78",
            alpha=bg_alpha,
            linewidths=0,
            zorder=1,
        )

        rgba_by_chunk_id, _ = make_chunk_color_lookup(chunks)

        legend_handles = []
        for chunk_id, xy in chunk_core_xy:
            color = rgba_by_chunk_id.get(chunk_id, (0.1, 0.3, 0.9, 1.0))
            handle = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=core_marker_size,
                c=[color],
                alpha=point_alpha,
                edgecolors="white",
                linewidths=max(0.0, min(0.25, core_marker_size / 80.0)),
                zorder=2,
                label=f"Chunk {chunk_id}",
            )
            legend_handles.append(handle)

            # Write chunk id at the chunk core footprint centroid.
            if label_size > 0:
                center = np.mean(xy, axis=0)
                # Preserve the chunk hue while darkening the text for stronger
                # contrast against its semi-transparent point markers.
                label_color = tuple(
                    float(np.clip(channel, 0.0, 1.0)) * 0.65
                    for channel in color[:3]
                )
                txt = ax.text(
                    center[0],
                    center[1],
                    str(chunk_id),
                    fontsize=label_size,
                    ha="center",
                    va="center",
                    color=label_color,
                    bbox={
                        "boxstyle": "round,pad=0.06",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.62,
                    },
                    zorder=3,
                )

        xmin, ymin = np.min(bounds_xy, axis=0)
        xmax, ymax = np.max(bounds_xy, axis=0)
        dx = max(float(xmax - xmin), 1e-6)
        dy = max(float(ymax - ymin), 1e-6)
        pad_x = dx * padding_ratio
        pad_y = dy * padding_ratio

        axes_name = str(grid_meta.get("axes", "xy"))
        x_name = axes_name[0] if len(axes_name) >= 1 else "x"
        y_name = axes_name[1] if len(axes_name) >= 2 else "y"

        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Ground {x_name.upper()} (m)")
        ax.set_ylabel(f"Ground {y_name.upper()} (m)")
        ax.tick_params(direction="out", length=3.5, width=0.8)
        ax.grid(True, linewidth=0.35, alpha=0.25, color="0.65")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show_legend and legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="best",
                frameon=True,
                facecolor="white",
                framealpha=0.82,
                edgecolor="none",
                ncol=legend_ncol,
                handletextpad=0.4,
                columnspacing=0.8,
                borderaxespad=0.55,
                labelspacing=0.32 if n_chunks > 20 else 0.42,
                markerscale=1.15,
            )

        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, format="svg")
        fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(fig)

    print(
        "[INFO] Saved chunk core footprint XY visualization: "
        f"{save_path} and {png_path}"
    )
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    run_logger = RunLogger(
        Path(args.output_rrd),
        log_file=args.log_file,
        enabled=not bool(args.no_file_log),
    )
    run_logger.install()
    stage("Setup", f"scene={args.scene_dir}, output={args.output_rrd}, model={args.model}")

    # Early validation: sparse_grad requires packed.
    if args.gsplat_sparse_grad and not args.gsplat_packed:
        raise ValueError("--gsplat_sparse_grad requires --gsplat_packed")

    device = resolve_device(args.device)
    stage_timings: Dict[str, float] = {
        "chunking": 0.0,
        "model_prediction": 0.0,
        "post_alignment": 0.0,
    }
    timing_path = (
        Path(args.output_rrd).expanduser().resolve().with_suffix("")
        / "processing_time.json"
    )

    model_name = str(args.model)

    # ------------------------------------------------------------------
    # 1. Resolve input preprocessing config
    # ------------------------------------------------------------------
    stage("Input Preparation", "resolving input preprocessing and loading lightweight scene manifest")
    if args.patch_size is None:
        args.patch_size = int(args.size_multiple)
    else:
        args.size_multiple = int(args.patch_size)
    norm_type = str(args.norm_type)
    print(
        "[INFO] Input preprocessing: "
        f"norm_type={norm_type}, patch_size={int(args.patch_size)}, "
        f"max_side={int(args.max_side)}"
    )

    # ------------------------------------------------------------------
    # 2. Load scene
    # ------------------------------------------------------------------
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    run_logger.detail(
        f"Loading scene manifest: scene_dir={scene_dir}, images_dir={args.images_dir}, "
        f"cams_dir={args.cams_dir}, depth_dir={args.depth_dir}, num_views={args.num_views}, "
        f"start={args.start}, stride={args.stride}, max_side={args.max_side}"
    )
    views, meta = build_views_from_scene(
        scene_dir=scene_dir,
        images_dir=args.images_dir,
        cams_dir=args.cams_dir,
        depth_dir=args.depth_dir,
        frame_glob=args.frame_glob,
        num_views=args.num_views,
        start=args.start,
        stride=args.stride,
        max_side=args.max_side,
        size_multiple=args.patch_size,
        depth_scale=args.depth_scale,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        device=device,
        show_progress=bool(args.chunk_tqdm),
    )
    pose_perturb_meta = perturb_scene_camera_poses(
        meta,
        enabled=bool(args.pose_perturb),
        seed=int(args.seed) + int(args.pose_perturb_seed_offset),
        xy_std=float(args.pose_perturb_xy_std),
        z_std=float(args.pose_perturb_z_std),
        yaw_std_deg=float(args.pose_perturb_yaw_std_deg),
        xy_max=float(args.pose_perturb_xy_max),
        z_max=float(args.pose_perturb_z_max),
        yaw_max_deg=float(args.pose_perturb_yaw_max_deg),
    )
    if bool(args.pose_perturb):
        print(
            "[INFO] Pose perturbation enabled at input stage: "
            f"scope={pose_perturb_meta.get('scope')}, "
            f"perturbed={pose_perturb_meta.get('num_perturbed_cameras', 0)}/"
            f"{pose_perturb_meta.get('num_cameras', 0)}, "
            f"xy_std={float(args.pose_perturb_xy_std):.6g}m, "
            f"z_std={float(args.pose_perturb_z_std):.6g}m, "
            f"yaw_std={float(args.pose_perturb_yaw_std_deg):.6g}deg"
        )
        run_logger.detail(
            "Pose perturbation config: "
            + json.dumps(pose_perturb_meta, ensure_ascii=False)
        )
    views, meta, ignored_no_pose = filter_views_meta_to_valid_poses(
        views,
        meta,
        output_rrd=Path(args.output_rrd),
        run_logger=run_logger,
    )
    if ignored_no_pose:
        run_logger.detail(
            f"Pose filtering kept {len(views)} frames and ignored "
            f"{len(ignored_no_pose)} frames without valid poses."
        )

    # ------------------------------------------------------------------
    # 3. Resolve prior policy
    # ------------------------------------------------------------------
    stage("Policy Resolution", "resolving priors, alignment, and recenter mode")
    prior_policy = resolve_prior_policy(args, model_name, meta)

    args.align = resolve_align_mode(
        args.align,
        model=model_name,
        meta=meta,
        prior_policy=prior_policy,
    )

    args.recenter = resolve_recenter_mode(
        args.recenter,
        model=model_name,
        meta=meta,
        prior_policy=prior_policy,
    )

    print(
        "[POLICY] "
        f"family={prior_policy['family']}, "
        f"align={args.align}, recenter={args.recenter}, "
        f"pose={prior_policy['pose']}, "
        f"translation={prior_policy['translation']}, "
        f"rotation={prior_policy['rotation']}, "
        f"ray={prior_policy['ray']}, "
        f"depth={prior_policy['depth']}, "
        f"bootstrap_ray={prior_policy['bootstrap_ray']}, "
        f"bootstrap_depth={prior_policy['bootstrap_depth']}"
    )

    # ------------------------------------------------------------------
    # 4. Optional recenter input views
    # ------------------------------------------------------------------
    stage("Recenter", f"mode={args.recenter}")
    recenter_state = maybe_recenter_anchor_from_meta(meta, args.recenter)
    if recenter_state is not None:
        print(
            f"[INFO] Input views recentered by {np.linalg.norm(recenter_state):.3f} units "
            f"(mode={args.recenter})."
        )

    # ------------------------------------------------------------------
    # 5. Spatial chunking
    # ------------------------------------------------------------------
    stage("Spatial Chunking", f"max_chunk_size={args.max_chunk_size}, order={args.chunk_order}")
    chunking_time_start = time.perf_counter()
    chunks, grid_meta = build_spatial_chunks(
        meta=meta,
        spatial_partition=args.spatial_partition,
        max_chunk_size=args.max_chunk_size,
        min_chunk_size=args.min_chunk_size,
        max_chunks=args.max_chunks,
        pose_grid_size=args.pose_grid_size,
        pose_grid_neighbor_radius=args.pose_grid_neighbor_radius,
        footprint_workers=args.footprint_workers,
        temporal_overlap_ratio=args.temporal_overlap_ratio,
    )
    if not chunks:
        raise RuntimeError(
            "No spatial chunks generated. Check "
            "--max_chunk_size/--min_chunk_size/--num_views."
        )

    chunks, chunk_order_meta = order_spatial_chunks(
        chunks,
        meta=meta,
        strategy=args.chunk_order,
    )
    stage_timings["chunking"] = float(time.perf_counter() - chunking_time_start)
    grid_meta["chunk_order"] = chunk_order_meta

    print(
        f"Spatial chunks: frames={len(views)}, chunks={len(chunks)}, "
        f"partition={grid_meta.get('partition', 'footprint_tree')}, "
        f"axes={grid_meta['axes']}, "
        f"core_target={grid_meta['auto_core_target_size']}, "
        f"dropped_seam={grid_meta['total_dropped_seam_images']}, "
        f"order={chunk_order_meta.get('strategy')}, "
        f"start_source_chunk={chunk_order_meta.get('start_source_chunk_id', 0)}, "
        f"adj_edges={chunk_order_meta.get('num_adjacency_edges', 0)}"
    )
    if "footprint" in grid_meta:
        counts = grid_meta["footprint"].get(
            "footprint_source_counts", {}
        )
        print(
            "[INFO] footprint sources: "
            f"depth={int(counts.get('depth', 0))}, "
            f"lookat={int(counts.get('lookat', 0))}, "
            f"center={int(counts.get('center', 0))}"
        )

    save_spatial_chunk_core_footprint_xy_visualization(
        meta=meta,
        grid_meta=grid_meta,
        chunks=chunks,
        output_rrd=Path(args.output_rrd),
        point_size=float(args.chunk_footprint_point_size),
        bg_point_size=float(args.chunk_footprint_bg_point_size),
        point_alpha=float(args.chunk_footprint_alpha),
        bg_alpha=float(args.chunk_footprint_bg_alpha),
        label_size=float(args.chunk_footprint_label_size),
        font_scale=float(args.chunk_footprint_font_scale),
        padding_ratio=float(args.chunk_footprint_padding_ratio),
        show_legend=bool(args.chunk_footprint_show_legend),
        legend_cols=int(args.chunk_footprint_legend_cols),
        legend_max_rows=int(args.chunk_footprint_legend_max_rows),
    )

    # ------------------------------------------------------------------
    # 6. Init model
    # ------------------------------------------------------------------
    stage("Model Initialization", f"model={model_name}")
    checkpoint_overrides, checkpoint_override = checkpoint_hydra_overrides(
        model_name,
        args.checkpoint,
    )
    hydra_overrides = (
        list(args.hydra_override)
        + checkpoint_overrides
        + build_prior_overrides(model_name, prior_policy)
    )
    if checkpoint_overrides:
        run_logger.detail(
            "Routing checkpoint through Hydra model config: "
            + ", ".join(checkpoint_overrides)
        )
    model, _ = init_model_from_hydra(
        model_name=model_name,
        machine=args.machine,
        hydra_overrides=hydra_overrides,
        device=device,
    )
    load_checkpoint(model, checkpoint_override)
    model.eval()

    apply_runtime_prior_policy(model, prior_policy)
    run_logger.detail(
        f"Using image norm_type={norm_type!r}, patch_size={int(args.patch_size)} "
        "for chunk tensor loading."
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    peak_gpu_memory: Dict[str, object] = {
        "enabled": bool(device.type == "cuda"),
        "device": str(device),
        "peak_allocated_bytes": 0,
        "peak_allocated_mib": 0.0,
        "peak_reserved_bytes": 0,
        "peak_reserved_mib": 0.0,
    }

    def update_peak_gpu_memory() -> None:
        if device.type != "cuda":
            return
        torch.cuda.synchronize(device)
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
        peak_gpu_memory.update(
            {
                "peak_allocated_bytes": allocated,
                "peak_allocated_mib": float(allocated / (1024 ** 2)),
                "peak_reserved_bytes": reserved,
                "peak_reserved_mib": float(reserved / (1024 ** 2)),
            }
        )

    def build_processing_time_meta(
        post_align_summary: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        post_align_summary = post_align_summary or {}
        total = float(sum(stage_timings.values()))
        return {
            "schema": "processing_time_v3",
            "method": model_name,
            "processing_time_seconds": float(total),
            "processing_time_ms": float(total * 1000.0),
            "chunking_seconds": float(stage_timings["chunking"]),
            "chunking_ms": float(stage_timings["chunking"] * 1000.0),
            "model_prediction_seconds": float(stage_timings["model_prediction"]),
            "model_prediction_ms": float(stage_timings["model_prediction"] * 1000.0),
            "post_alignment_seconds": float(stage_timings["post_alignment"]),
            "post_alignment_ms": float(stage_timings["post_alignment"] * 1000.0),
            "stage_times_seconds": dict(stage_timings),
            "stage_times_ms": {k: float(v * 1000.0) for k, v in stage_timings.items()},
            "timing_scope": ["chunking", "model_prediction", "post_alignment"],
            "excluded": [
                "scene_data_loading",
                "model_initialization_and_weight_loading",
                "chunk_data_loading_and_preprocessing",
                "prediction_postprocessing",
                "cache_io",
                "multiview_consistency",
                "seam_error_evaluation",
                "rrd_and_eval_saving",
                "gsplat_optimization",
                "dom_generation",
            ],
            "peak_gpu_memory": dict(peak_gpu_memory),
            "peak_gpu_memory_allocated_bytes": int(peak_gpu_memory["peak_allocated_bytes"]),
            "peak_gpu_memory_allocated_mib": float(peak_gpu_memory["peak_allocated_mib"]),
            "peak_gpu_memory_reserved_bytes": int(peak_gpu_memory["peak_reserved_bytes"]),
            "peak_gpu_memory_reserved_mib": float(peak_gpu_memory["peak_reserved_mib"]),
            "num_frames": int(len(views)),
            "num_chunks": int(len(chunk_records)),
            "mv_consistency": bool(args.mv_consistency),
            "post_chunk_align": bool(args.post_chunk_align),
            "post_chunk_align_mode": str(args.post_chunk_align_mode),
            "post_chunk_align_num_valid_edges": int(
                post_align_summary.get("num_valid_edges", 0)
            ),
            "gsplat_refine_included": bool(args.gsplat_refine),
            "dom_rendering_included": bool(args.render_dom),
        }

    def write_processing_time_meta(meta_obj: Dict[str, object]) -> None:
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(
            json.dumps(meta_obj, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # 7. Per-chunk bootstrap state
    # ------------------------------------------------------------------
    stage("Chunk Prediction Setup", "initializing bootstrap priors and chunk cache")
    align_mode = args.align
    input_pose_cams_by_stem_map = input_pose_centers_by_stem(meta)
    if (
        align_mode in POSE_TRANSLATION_ALIGN_MODES
        and not input_pose_cams_by_stem_map
    ):
        print(
            "[WARN] --align pose_* selected, but no input pose translations found; "
            "pose alignment will fall back to raw chunk coordinates."
        )

    bootstrap_ray = bool(prior_policy["bootstrap_ray"])
    bootstrap_depth = bool(prior_policy["bootstrap_depth"])

    bootstrapped_intrinsics: Optional[torch.Tensor] = None
    if bootstrap_ray:
        set_pi3x_ray_prior_prob(model, enabled=False)
        print(
            "[INFO] bootstrap_ray: chunk 0 runs without ray/intrinsics prior; "
            "later chunks use average intrinsics recovered from chunk 0 rays."
        )
    if bootstrap_depth:
        set_model_task_prob(model, "depth_prob", enabled=False)
        set_model_task_value(model, "sparse_depth_prob", 0.0)
        print(
            "[INFO] bootstrap_depth: external depth prior disabled; "
            "later chunks reuse predicted seam depth from earlier chunks."
        )

    depth_prior_cache: Dict[int, torch.Tensor] = {}
    future_overlap_counter = build_future_overlap_counter(chunks)
    chunk_records: List[Dict[str, object]] = []
    chunk_cache_dir = chunk_cache_dir_for_output(Path(args.output_rrd))
    chunk_cache_writer = AsyncChunkCacheWriter(
        chunk_cache_dir,
        max_workers=args.chunk_cache_workers,
        max_pending=args.chunk_cache_max_pending,
    )
    print(
        "[INFO] chunk cache writer: "
        f"workers={chunk_cache_writer.max_workers}, "
        f"max_pending={chunk_cache_writer.max_pending}, "
        f"dir={chunk_cache_dir}"
    )
    depth_conf_debug_dir = (
        Path(args.output_rrd).expanduser().resolve().with_suffix("")
        / "debug"
        / "depth_conf_filter"
        if bool(args.debug_depth_conf_filter)
        else None
    )

    # ------------------------------------------------------------------
    # 8. Run chunks (feed-forward only, and torch.no_grad() for model inference)
    # ------------------------------------------------------------------
    stage("Chunk Prediction", f"chunks={len(chunks)}")
    chunk_iter = iter_progress(
        chunks,
        desc="Predict chunks",
        total=len(chunks),
        enabled=bool(args.chunk_tqdm),
    )
    for chunk in chunk_iter:
        chunk_id = int(chunk["chunk_id"])
        indices = [int(i) for i in chunk["indices"]]
        core_indices = [int(i) for i in chunk["core_indices"]]
        overlap_indices = [int(i) for i in chunk["overlap_indices"]]

        chunk_stems = [meta["stems"][i] for i in indices]
        chunk_views_raw, chunk_rgbs = load_chunk_views_from_scene(
            lightweight_views=views,
            meta=meta,
            indices=indices,
            prior_policy=prior_policy,
            device=device,
            recenter_anchor=recenter_state,
            num_workers=args.scene_io_workers,
            norm_type=norm_type,
        )
        chunk_views = filter_views_for_prior_policy(chunk_views_raw, prior_policy)

        if bootstrap_ray and chunk_id > 0:
            if bootstrapped_intrinsics is None:
                set_pi3x_ray_prior_prob(model, enabled=False)
                run_logger.detail(
                    f"[chunk {chunk_id:03d}] bootstrap ray prior unavailable; "
                    "continuing without ray/intrinsics prior."
                )
            else:
                set_pi3x_ray_prior_prob(model, enabled=True)
                chunk_views = apply_bootstrap_intrinsics_to_views(
                    chunk_views,
                    intrinsics=bootstrapped_intrinsics,
                    device=device,
                )

        num_depth_priors_used = 0
        if bootstrap_depth:
            chunk_views, num_depth_priors_used = (
                apply_cached_depth_priors_to_views(
                    chunk_views,
                    indices=indices,
                    depth_cache=depth_prior_cache,
                    device=device,
                )
            )
            set_model_task_prob(
                model, "depth_prob", enabled=num_depth_priors_used > 0
            )
            if num_depth_priors_used > 0:
                set_model_task_value(model, "sparse_depth_prob", 0.0)

            for idx in overlap_indices:
                idx = int(idx)
                if idx in future_overlap_counter:
                    future_overlap_counter[idx] -= 1
                    if future_overlap_counter[idx] <= 0:
                        future_overlap_counter.pop(idx, None)
                        depth_prior_cache.pop(idx, None)

        run_logger.detail(
            f"[chunk {chunk_id:03d}] cell={chunk['cell_key']}, "
            f"core={len(core_indices)}, overlap={len(overlap_indices)}, "
            f"total={len(indices)}, depth_priors={num_depth_priors_used}, "
            f"stems={chunk_stems[0]}..{chunk_stems[-1]}"
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_prediction_start = time.perf_counter()
        with torch.no_grad():
            preds = model(chunk_views)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        stage_timings["model_prediction"] += float(
            time.perf_counter() - model_prediction_start
        )

        depth_conf_filter_meta: Dict[str, object] = {
            "enabled": False,
            "quantile": float(args.conf_quantile),
        }
        depth_conf_keep_masks: Optional[List[Optional[np.ndarray]]] = None

        if bool(args.depth_conf_filter):
            depth_conf_filter_meta, depth_conf_keep_masks = apply_depth_confidence_filter_to_preds(
                preds=preds,
                rgbs=chunk_rgbs,
                stems=chunk_stems,
                chunk_id=chunk_id,
                conf_quantile=float(args.conf_quantile),
                pred_min_depth=float(args.pred_min_depth),
                debug_dir=depth_conf_debug_dir,
            )
            if depth_conf_filter_meta.get("enabled"):
                run_logger.detail(
                    f"[chunk {chunk_id:03d}] depth confidence filter: "
                    f"q={float(depth_conf_filter_meta.get('quantile', 0.0)):.3f}, "
                    f"dropped={int(depth_conf_filter_meta.get('num_dropped_pixels', 0))}/"
                    f"{int(depth_conf_filter_meta.get('num_valid_pixels', 0))} "
                    f"({float(depth_conf_filter_meta.get('drop_fraction', 0.0)):.3f}), "
                    f"missing_conf_views={int(depth_conf_filter_meta.get('num_views_missing_conf', 0))}"
                )

        if bootstrap_ray and chunk_id == 0:
            bootstrapped_intrinsics = (
                recover_average_intrinsics_from_pred_rays(preds)
            )
            if bootstrapped_intrinsics is None:
                print("[WARN] Failed to recover bootstrap intrinsics from chunk 0.")
            else:
                K_np = bootstrapped_intrinsics.detach().cpu().numpy()
                print(
                    "[INFO] Recovered bootstrap intrinsics from chunk 0: "
                    f"fx={K_np[0, 0]:.3f}, fy={K_np[1, 1]:.3f}, "
                    f"cx={K_np[0, 2]:.3f}, cy={K_np[1, 2]:.3f}"
                )

        if bootstrap_depth:
            keep_core = {
                int(i)
                for i in core_indices
                if future_overlap_counter.get(int(i), 0) > 0
            }
            keep_seam = {
                int(i)
                for i in overlap_indices
                if future_overlap_counter.get(int(i), 0) > 0
            }

            core_depth_cache = build_depth_prior_cache(
                preds,
                indices,
                keep_indices=keep_core,
                valid_masks=depth_conf_keep_masks,
            )
            seam_depth_cache = build_depth_prior_cache(
                preds,
                indices,
                keep_indices=keep_seam,
                valid_masks=depth_conf_keep_masks,
            )

            core_inserted = 0
            core_replaced = 0
            for global_idx, depth in core_depth_cache.items():
                global_idx = int(global_idx)
                if global_idx in depth_prior_cache:
                    core_replaced += 1
                else:
                    core_inserted += 1
                depth_prior_cache[global_idx] = depth

            seam_inserted = 0
            seam_skipped_existing = 0
            for global_idx, depth in seam_depth_cache.items():
                global_idx = int(global_idx)
                if global_idx in depth_prior_cache:
                    seam_skipped_existing += 1
                    continue
                depth_prior_cache[global_idx] = depth
                seam_inserted += 1

            if core_depth_cache or seam_depth_cache:
                run_logger.detail(
                    f"[chunk {chunk_id:03d}] cached predicted depth priors: "
                    f"core_inserted={core_inserted}, "
                    f"core_replaced={core_replaced}, "
                    f"seam_inserted={seam_inserted}, "
                    f"seam_skipped_existing={seam_skipped_existing}, "
                    f"cache_size={len(depth_prior_cache)}"
                )

        collect_point_indices = None
        if not args.log_chunks:
            collect_point_indices = list(chunk["core_local_indices"])

        raw_points_unmasked, raw_colors_unmasked, pred_maps, pred_valid_masks_unmasked, raw_cams = (
            collect_pred_outputs(
                preds=preds,
                rgbs=chunk_rgbs,
                pred_min_depth=args.pred_min_depth,
                conf_quantile=0.0,
                stems=chunk_stems,
                collect_point_indices=collect_point_indices,
            )
        )

        pred_valid_masks_masked = apply_optional_keep_masks_to_valid_masks(
            pred_valid_masks_unmasked,
            depth_conf_keep_masks,
        )

        if collect_point_indices is not None:
            point_local_indices_for_chunk = list(collect_point_indices)
        else:
            point_local_indices_for_chunk = list(range(len(pred_maps)))

        raw_points, raw_colors = points_from_maps(
            pred_maps=pred_maps,
            pred_valid_masks=pred_valid_masks_masked,
            rgbs=chunk_rgbs,
            local_indices=point_local_indices_for_chunk,
        )

        if recenter_state is not None:
            (
                raw_points,
                raw_colors,
                pred_maps,
                raw_cams,
            ) = restore_predictions_from_recenter(
                raw_points, raw_colors, pred_maps, raw_cams, recenter_state
            )

        if align_mode in POSE_TRANSLATION_ALIGN_MODES:
            (
                pred_points,
                pred_colors,
                pred_maps_aligned,
                pred_cams,
                align_meta,
            ) = apply_chunk_pose_alignment(
                mode=align_mode,
                chunk_id=chunk_id,
                reference_cams_by_stem=input_pose_cams_by_stem_map,
                raw_pred_points=raw_points,
                raw_pred_colors=raw_colors,
                pred_maps=pred_maps,
                raw_pred_cams=raw_cams,
                target_stems=chunk_stems,
                seed=int(args.seed) + 70001 + int(chunk_id),
            )
            run_logger.detail(
                f"[chunk {chunk_id:03d}] pose alignment: "
                f"mode={align_meta['mode']}, valid={align_meta['valid']}, "
                f"num_corr={align_meta['num_corr']}, "
                f"scale={float(align_meta['scale']):.6g}, "
                f"median_residual={float(align_meta.get('median_residual', float('nan'))):.6g}"
            )
        else:
            pred_points = raw_points
            pred_colors = raw_colors
            pred_maps_aligned = pred_maps
            pred_cams = raw_cams
            align_meta = {
                "mode": align_mode,
                "valid": True,
                "source": "none",
                "num_corr": 0,
                "num_scale_pairs": 0,
                "matched_camera_stems": [],
                "scale": 1.0,
                "yaw_degrees": 0.0,
                "yaw_valid": False,
                "R": np.eye(3, dtype=np.float32).tolist(),
                "t": np.zeros(3, dtype=np.float32).tolist(),
                "median_residual": float("nan"),
                "note": "no alignment applied",
                "chunk_id": int(chunk_id),
            }

        mv_consistency_meta: Dict[str, object] = {
            "enabled": False,
            "deferred": bool(args.mv_consistency),
        }
        gsplat_meta: Dict[str, object] = {
            "enabled": False,
            "deferred": bool(args.gsplat_refine),
        }

        if collect_point_indices is not None:
            chunk_point_local_indices = list(collect_point_indices)
        else:
            chunk_point_local_indices = list(range(len(pred_maps_aligned)))

        core_points, core_colors = points_from_maps(
            pred_maps=pred_maps_aligned,
            pred_valid_masks=pred_valid_masks_masked,
            rgbs=chunk_rgbs,
            local_indices=chunk["core_local_indices"],
        )

        record: Dict[str, object] = {
            "chunk_id": chunk_id,
            "partition": str(chunk.get("partition", grid_meta.get("partition", "footprint_tree"))),
            "cell_key": tuple(chunk["cell_key"]),
            "alignment_topology": str(chunk.get("alignment_topology", "tree_path")),
            "align_parent_id": chunk.get("align_parent_id", None),
            "align_level": int(chunk.get("align_level", 0)),
            "adjacent_chunk_ids": [
                int(value) for value in chunk.get("adjacent_chunk_ids", [])
            ],
            "indices": indices,
            "core_indices": core_indices,
            "overlap_indices": overlap_indices,
            "stems": chunk_stems,
            "core_stems": [meta["stems"][i] for i in core_indices],
            "overlap_stems": [meta["stems"][i] for i in overlap_indices],
            "num_seam_candidates": int(chunk["num_seam_candidates"]),
            "num_dropped_seam_images": int(chunk["num_dropped_seam_images"]),
            "num_depth_priors_used": int(num_depth_priors_used),
            "pred_cams": pred_cams,
            "align_meta": align_meta,
            "depth_conf_filter_meta": depth_conf_filter_meta,
            "mv_consistency_meta": mv_consistency_meta,
            "gsplat_meta": gsplat_meta,
            "num_chunk_pred_points_raw": int(raw_points.shape[0]),
            "num_chunk_pred_points_filtered": int(pred_points.shape[0]),
            "num_chunk_pred_points_refined": int(pred_points.shape[0]),
            "num_core_pred_points": int(core_points.shape[0]),
        }

        chunk_cache_writer.submit(
            record,
            {
                "rgbs": chunk_rgbs,
                "chunk_pred_points": pred_points,
                "chunk_pred_colors": pred_colors,
                "core_pred_points": core_points,
                "core_pred_colors": core_colors,
                "_pred_maps": pred_maps_aligned,
                "_pred_valid_masks": pred_valid_masks_masked,
                "_pred_valid_masks_unmasked": pred_valid_masks_unmasked,
                "_chunk_intrinsics": intrinsics_from_views(chunk_views_raw),
                "_chunk_point_local_indices": chunk_point_local_indices,
                "_core_local_indices": list(chunk["core_local_indices"]),
            },
        )
        strip_array_payload(record)

        chunk_records.append(record)

        run_logger.detail(
            f"[chunk {chunk_id:03d}] prediction: "
            f"raw_points={raw_points.shape[0]}, "
            f"filtered_points={pred_points.shape[0]}, "
            f"core_points={core_points.shape[0]}, "
            f"cameras={len(pred_cams)}, "
            f"align={align_meta['mode']}, valid={align_meta['valid']}, "
            f"gsplat={bool(gsplat_meta.get('deferred', False))}"
        )
        if hasattr(chunk_iter, "set_postfix"):
            chunk_iter.set_postfix(
                chunk=f"{chunk_id:03d}",
                prior=num_depth_priors_used,
                core=int(core_points.shape[0]),
                cache=len(depth_prior_cache),
                align=str(align_meta.get("mode", align_mode)),
            )
        del (
            chunk_views_raw,
            chunk_views,
            preds,
            raw_points,
            raw_colors,
            pred_maps,
            pred_valid_masks_unmasked,
            pred_valid_masks_masked,
            pred_maps_aligned,
            pred_points,
            pred_colors,
            core_points,
            core_colors,
            chunk_rgbs,
        )
    # ------------------------------------------------------------------
    # 9. Cleanup model (no longer needed for deferred stages)
    # ------------------------------------------------------------------
    stage("Chunk Cache Flush", "waiting for async cache writes")
    chunk_cache_writer.wait()
    write_chunk_record_manifest(
        chunk_records,
        chunk_cache_dir / "manifest.json",
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[INFO] Model deleted, GPU cache cleared.")

    # ------------------------------------------------------------------
    # 10. Deferred MV consistency (runs before save_spatial_rrd so eval
    #     reflects post-filter results)
    # ------------------------------------------------------------------
    stage("Deferred MV Consistency", "filtering cached per-chunk predictions")
    apply_deferred_multiview_filter_to_chunk_records(
        chunk_records=chunk_records,
        args=args,
        device=device,
        run_logger=run_logger,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 10.5 Deferred chunk post-alignment
    # ------------------------------------------------------------------
    stage("Deferred Chunk Alignment", f"enabled={bool(args.post_chunk_align)}")
    post_align_time_start = time.perf_counter()
    post_chunk_align_summary = apply_deferred_chunk_post_alignment(
        chunk_records=chunk_records,
        args=args,
        show_progress=bool(args.chunk_tqdm),
    )
    if "seam_error" in post_chunk_align_summary:
        grid_meta["seam_error"] = post_chunk_align_summary["seam_error"]
    stage_timings["post_alignment"] = float(time.perf_counter() - post_align_time_start)
    if bool(args.compute_seam_error):
        stage("Seam Error", "evaluating adjacent chunk overlap consistency")
        seam_error_start = time.perf_counter()
        post_chunk_align_summary["seam_error"] = compute_adjacent_chunk_seam_error(
            chunk_records,
            max_points_per_edge=int(args.seam_error_max_points_per_edge),
            seed=int(args.seed) + 990017,
        )
        seam = post_chunk_align_summary["seam_error"]
        seam["evaluation_seconds"] = float(time.perf_counter() - seam_error_start)
        # Persist explicitly computed seam metrics in the RRD sidecar.  The
        # earlier assignment only covers a seam result optionally returned by
        # the post-alignment stage itself; compute_seam_error populates it
        # afterwards.
        grid_meta["seam_error"] = seam
        print(
            "[INFO] Seam error: "
            f"edges={seam['num_valid_edges']}/{seam['num_adjacency_edges']}, "
            f"E_seam={float(seam['seam_error']):.6g}, "
            f"E_seam_z={float(seam['seam_error_z']):.6g}, "
            f"time={float(seam['evaluation_seconds']):.3f}s"
        )

    # Post-chunk alignment deliberately has no fixed world gauge. Restore the
    # whole reconstruction to the input camera-pose frame before any final
    # point/camera aggregation so RRD and eval outputs share the GT frame.
    stage("Final Global Pose Alignment", f"mode={args.align}")
    final_global_align_meta = apply_final_global_pose_alignment(
        chunk_records=chunk_records,
        reference_centers_by_stem=input_pose_cams_by_stem_map,
        mode=str(args.align),
        seed=int(args.seed),
    )
    post_chunk_align_summary["final_global_pose_alignment"] = (
        final_global_align_meta
    )
    if bool(final_global_align_meta.get("valid", False)):
        print(
            "[INFO] Final global pose alignment: "
            f"mode={final_global_align_meta['mode']}, "
            f"matches={int(final_global_align_meta['num_corr'])}, "
            f"scale={float(final_global_align_meta['scale']):.6g}, "
            f"yaw={float(final_global_align_meta.get('yaw_degrees', float('nan'))):.6g}deg, "
            f"median_residual={float(final_global_align_meta.get('median_residual', float('nan'))):.6g}"
        )
    else:
        print(
            "[WARN] Final global pose alignment failed; keeping the "
            f"post-chunk world gauge. Reason: {final_global_align_meta.get('note', 'unknown')}"
        )
    update_peak_gpu_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stage("Timing", "writing processing_time.json")
    processing_time_meta = build_processing_time_meta(
        post_align_summary=post_chunk_align_summary,
    )
    print(
        "Processing time so far: "
        f"total={processing_time_meta['processing_time_seconds']:.6f}s, "
        f"chunking={processing_time_meta['chunking_seconds']:.6f}s, "
        f"model_prediction={processing_time_meta['model_prediction_seconds']:.6f}s, "
        f"post_alignment={processing_time_meta['post_alignment_seconds']:.6f}s, "
        f"peak_gpu_allocated={processing_time_meta['peak_gpu_memory_allocated_mib']:.2f}MiB"
    )
    write_processing_time_meta(processing_time_meta)
    print(f"Saved processing timing: {timing_path}")

    # ------------------------------------------------------------------
    # 11. Save RRD/eval (after MV consistency filtering)
    # ------------------------------------------------------------------
    stage("Save RRD and Eval", f"output={args.output_rrd}")
    save_spatial_rrd(
        output_rrd=Path(args.output_rrd),
        scene_dir=str(scene_dir),
        model_name=model_name,
        checkpoint=args.checkpoint,
        meta=meta,
        grid_meta=grid_meta,
        chunk_records=chunk_records,
        align=args.align,
        view_coordinates=args.view_coordinates,
        background=args.background,
        hide_grid=args.hide_grid,
        point_radius=args.point_radius,
        camera_axis_size=args.camera_axis_size,
        camera_axis_radius=args.camera_axis_radius,
        max_points_per_view=args.max_points_per_view,
        voxel_size=args.voxel_downsample,
        point_downsample=bool(args.point_downsample),
        seed=args.seed,
        log_images=args.log_images,
        show_world_axes=args.show_world_axes,
        world_axes_origin=args.world_axes_origin,
        world_up_axis=args.world_up_axis,
        world_axis_size=args.world_axis_size,
        world_axis_size_ratio=args.world_axis_size_ratio,
        world_axis_min_size=args.world_axis_min_size,
        world_axis_up_offset_ratio=args.world_axis_up_offset_ratio,
        world_axis_radius=args.world_axis_radius,
        prior_policy=prior_policy,
        recenter_anchor=recenter_state,
        log_chunks_rrd=args.log_chunks,
        compress_eval=False,
        processing_time=processing_time_meta,
        gt_io_workers=args.scene_io_workers,
        xy_fill_unmasked=bool(args.xy_fill_unmasked),
        xy_fill_grid_size=float(args.xy_fill_grid_size),
        xy_fill_max_points_per_chunk=int(args.xy_fill_max_points_per_chunk),
    )

    # ------------------------------------------------------------------
    # 12. Deferred 3DGS bundle optimization
    # ------------------------------------------------------------------
    stage("Deferred 3DGS", f"enabled={bool(args.gsplat_refine)}")
    gsplat_summary = run_deferred_gsplat_bundle_refinement(
        chunk_records=chunk_records,
        args=args,
        device=device,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 13. Deferred DOM rendering (after 3DGS, may use optimized gaussians)
    # ------------------------------------------------------------------
    stage("Deferred DOM", f"enabled={bool(args.render_dom)}")
    dom_meta = run_deferred_dom_rendering(
        chunk_records=chunk_records,
        args=args,
        device=device,
        gsplat_summary=gsplat_summary,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    processing_time_meta = build_processing_time_meta(
        post_align_summary=post_chunk_align_summary,
    )
    write_processing_time_meta(processing_time_meta)

    # Rewrite after post-alignment so the manifest contains final adjacency
    # and lazy similarity transforms, not only the pre-alignment chunk lists.
    if bool(args.keep_chunk_cache):
        write_chunk_record_manifest(
            chunk_records,
            chunk_cache_dir / "manifest.json",
        )
        print(f"[INFO] Kept reproducible chunk cache: {chunk_cache_dir}")

    chunk_cache_deleted = False
    if chunk_cache_dir.exists() and not bool(args.keep_chunk_cache):
        shutil.rmtree(chunk_cache_dir)
        chunk_cache_deleted = True
        print(f"[INFO] Deleted chunk cache: {chunk_cache_dir}")

    sidecar_path = Path(args.output_rrd).expanduser().resolve().with_suffix(".json")
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["processing_time"] = processing_time_meta
            sidecar["chunk_cache_deleted"] = bool(chunk_cache_deleted)
            sidecar["chunk_cache_kept"] = bool(
                args.keep_chunk_cache and chunk_cache_dir.exists()
            )
            sidecar_path.write_text(
                json.dumps(sidecar, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[WARN] Failed to update sidecar processing_time: {exc}")
    print(
        "Processing time summary: "
        f"chunking={processing_time_meta['chunking_seconds']:.6f}s, "
        f"model_prediction={processing_time_meta['model_prediction_seconds']:.6f}s, "
        f"post_alignment={processing_time_meta['post_alignment_seconds']:.6f}s, "
        f"total={processing_time_meta['processing_time_seconds']:.6f}s, "
        f"peak_gpu_allocated={processing_time_meta['peak_gpu_memory_allocated_mib']:.2f}MiB"
    )
    print(f"Saved final processing timing: {timing_path}")

    stage("Done", "pipeline finished")
    if run_logger.path is not None:
        print(f"[LOG] Full run log saved: {run_logger.path}")
    run_logger.close()


if __name__ == "__main__":
    main()
