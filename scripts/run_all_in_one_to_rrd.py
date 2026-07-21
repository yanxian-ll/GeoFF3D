#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a MapAnything external model on all selected images in one forward pass."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_vggt_slam_to_rrd import (
    align_prediction_to_gt_pose_sim3,
    json_safe,
)
from spatial_rrd.model_runner import (
    checkpoint_hydra_overrides,
    collect_pred_outputs,
    init_model_from_hydra,
    load_checkpoint,
)
from spatial_rrd.rrd_writer import (
    estimate_axis_size,
    gt_cameras_for_stems,
    log_camera_axes,
    log_input_images,
    log_points,
    log_view_coordinates,
    log_world_axes_marker,
    rr_disconnect_compat,
    rr_init_save_compat,
    rr_set_time_compat,
    sanitize_name,
    save_final_eval_outputs,
    send_blueprint,
)
from spatial_rrd.scene_io import (
    build_views_from_scene,
    load_chunk_views_from_scene,
    load_gt_points_from_meta,
    sample_points_and_colors,
    voxel_downsample,
)


def resolve_device(device_arg: str) -> torch.device:
    value = str(device_arg)
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    return torch.device(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hydra model config name, e.g. vggt, pi3, fast3r.")
    parser.add_argument("--checkpoint", default=None, help="Optional wrapper checkpoint/path override.")
    parser.add_argument(
        "--machine",
        default="aws",
        help="Hydra machine config. Defaults to aws to use local torch hub cache and disable online downloads.",
    )
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--output_rrd", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--cams_dir", default="cams")
    parser.add_argument("--depth_dir", default="depth")
    parser.add_argument("--frame_glob", default="*")
    parser.add_argument("--num_views", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_side", type=int, default=518)
    parser.add_argument("--size_multiple", type=int, default=14)
    parser.add_argument("--norm_type", default="identity")
    parser.add_argument("--copy_images", action="store_true")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--depth_min", type=float, default=1e-6)
    parser.add_argument("--depth_max", type=float, default=1e6)
    parser.add_argument("--pred_min_depth", type=float, default=1e-6)
    parser.add_argument("--conf_quantile", type=float, default=0.0)
    parser.add_argument("--max_points_per_view", type=int, default=250000)
    parser.add_argument("--voxel_downsample", type=float, default=0.05)
    parser.add_argument(
        "--point_downsample",
        "--final_point_downsample",
        dest="point_downsample",
        action="store_true",
        default=True,
        help="保存最终 pred/gt 点云前执行全局 voxel 下采样（默认开启）",
    )
    parser.add_argument(
        "--no_point_downsample",
        "--no_final_point_downsample",
        dest="point_downsample",
        action="store_false",
        help="保存最终 pred/gt 点云前不执行全局 voxel 下采样",
    )
    parser.add_argument("--max_pred_points", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max_gt_points", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene_io_workers", type=int, default=0)
    parser.add_argument("--log_images", action="store_true")
    parser.add_argument("--view_coordinates", default="RDF")
    parser.add_argument("--background", type=int, nargs=3, default=(255, 255, 255))
    parser.add_argument("--hide_grid", action="store_true")
    parser.add_argument("--point_radius", type=float, default=0.0)
    parser.add_argument("--camera_axis_size", type=float, default=0.0)
    parser.add_argument("--camera_axis_radius", type=float, default=0.0)
    parser.add_argument("--show_world_axes", action="store_true", default=True)
    parser.add_argument("--no_world_axes", action="store_false", dest="show_world_axes")
    parser.add_argument("--world_axes_origin", choices=("scene_center", "zero"), default="scene_center")
    parser.add_argument("--world_axis_size", type=float, default=0.0)
    parser.add_argument("--world_axis_size_ratio", type=float, default=0.12)
    parser.add_argument("--world_axis_min_size", type=float, default=0.1)
    parser.add_argument("--world_up_axis", default="z")
    parser.add_argument("--world_axis_up_offset_ratio", type=float, default=1.2)
    parser.add_argument("--world_axis_radius", type=float, default=0.0)
    parser.add_argument(
        "hydra_overrides",
        nargs="*",
        help="Extra Hydra overrides, e.g. model.model_config.dtype=float32",
    )
    return parser.parse_args(argv)


def downsample_final_points(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    max_points: int,
    voxel_size: float,
    point_downsample: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    effective_max_points = int(max_points) if bool(point_downsample) else 0
    effective_voxel_size = float(voxel_size) if bool(point_downsample) else 0.0
    points, colors = sample_points_and_colors(
        points,
        colors,
        max_points=effective_max_points,
        seed=int(seed),
    )
    if bool(point_downsample):
        points, colors = voxel_downsample(points, colors, effective_voxel_size)
    return points, colors


def write_spatial_format_rrd(
    *,
    args: argparse.Namespace,
    output_rrd: Path,
    model_name: str,
    stems: Sequence[str],
    rgbs: Sequence[np.ndarray],
    pred_cams: Sequence[Dict[str, object]],
    pred_points: np.ndarray,
    pred_colors: np.ndarray,
    gt_cams: Sequence[Dict[str, object]],
    gt_points: np.ndarray,
    gt_colors: np.ndarray,
) -> None:
    output_rrd = Path(output_rrd).expanduser().resolve()
    output_rrd.parent.mkdir(parents=True, exist_ok=True)
    scene_name = sanitize_name(Path(args.scene_dir).resolve().name)
    recording_id = f"spatial_overall_{scene_name}_{sanitize_name(model_name)}_pose_sim3"

    rr_init_save_compat("predict_scene_to_rrd_spatial_overall", recording_id, output_rrd)
    rr_set_time_compat("frame", 0)
    log_view_coordinates(str(args.view_coordinates))
    send_blueprint(background=tuple(args.background), hide_grid=bool(args.hide_grid))

    pred_root = "pred_spatial_aligned"
    log_points("world/gt/points", gt_points, gt_colors, float(args.point_radius))
    log_points(f"world/{pred_root}/points", pred_points, pred_colors, float(args.point_radius))

    axis_size = estimate_axis_size(
        [pred_points, gt_points],
        float(args.camera_axis_size),
    )
    log_camera_axes(
        "world/cameras/gt/axes",
        gt_cams,
        axis_size,
        float(args.camera_axis_radius),
        ((255, 0, 0), (0, 220, 0), (40, 80, 255)),
    )
    log_camera_axes(
        f"world/cameras/{pred_root}/axes",
        pred_cams,
        axis_size,
        float(args.camera_axis_radius),
        ((255, 0, 255), (255, 180, 0), (0, 220, 255)),
    )

    if bool(args.log_images):
        log_input_images(rgbs, stems)

    if bool(args.show_world_axes):
        bbox_points = gt_points if gt_points.shape[0] > 0 else pred_points
        log_world_axes_marker(
            bbox_points,
            origin_mode=str(args.world_axes_origin),
            axis_size=float(args.world_axis_size),
            axis_size_ratio=float(args.world_axis_size_ratio),
            min_axis_size=float(args.world_axis_min_size),
            up_axis=str(args.world_up_axis),
            up_offset_ratio=float(args.world_axis_up_offset_ratio),
            radius=float(args.world_axis_radius),
        )

    rr_disconnect_compat()
    print(f"Saved overall Rerun recording: {output_rrd}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    output_rrd = Path(args.output_rrd).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else output_rrd.with_suffix("")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_overrides, checkpoint_to_load = checkpoint_hydra_overrides(args.model, args.checkpoint)
    hydra_overrides = list(ckpt_overrides) + list(args.hydra_overrides)
    model, cfg = init_model_from_hydra(
        model_name=str(args.model),
        machine=str(args.machine),
        hydra_overrides=hydra_overrides,
        device=device,
    )
    load_checkpoint(model, checkpoint_to_load)
    model.eval()

    lightweight_views, meta = build_views_from_scene(
        scene_dir=Path(args.scene_dir),
        images_dir=str(args.images_dir),
        cams_dir=str(args.cams_dir),
        depth_dir=str(args.depth_dir),
        frame_glob=str(args.frame_glob),
        num_views=int(args.num_views),
        start=int(args.start),
        stride=int(args.stride),
        max_side=int(args.max_side),
        size_multiple=int(args.size_multiple),
        depth_scale=float(args.depth_scale),
        depth_min=float(args.depth_min),
        depth_max=float(args.depth_max),
        device=device,
        show_progress=True,
    )
    stems = [str(s) for s in meta.get("stems", [v["stem"] for v in lightweight_views])]
    indices = list(range(len(lightweight_views)))
    views, rgbs = load_chunk_views_from_scene(
        lightweight_views=lightweight_views,
        meta=meta,
        indices=indices,
        prior_policy={"ray": "none", "depth": "none"},
        device=device,
        recenter_anchor=None,
        num_workers=int(args.scene_io_workers),
        norm_type=str(args.norm_type),
    )

    print(f"[all-in-one] running model={args.model} on {len(views)} views.")
    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model(views)
    processing_time = {"processing_time_seconds": float(time.perf_counter() - t0)}

    points, colors, _pred_maps, _pred_valid_masks, pred_cams = collect_pred_outputs(
        preds,
        rgbs=rgbs,
        pred_min_depth=float(args.pred_min_depth),
        conf_quantile=float(args.conf_quantile),
        stems=stems,
    )
    gt_cams = gt_cameras_for_stems(meta, stems)
    effective_max_points = int(args.max_points_per_view) if bool(args.point_downsample) else 0
    effective_voxel_size = float(args.voxel_downsample) if bool(args.point_downsample) else 0.0
    pred_cap = int(args.max_pred_points) if args.max_pred_points is not None else effective_max_points
    gt_cap = int(args.max_gt_points) if args.max_gt_points is not None else effective_max_points
    if args.max_pred_points is not None or args.max_gt_points is not None:
        print(
            "[WARN] --max_pred_points/--max_gt_points are legacy all-in-one options; "
            "prefer spatial-compatible --max_points_per_view."
        )
    gt_points, gt_colors = load_gt_points_from_meta(
        meta,
        max_points=gt_cap,
        seed=int(args.seed),
        num_workers=int(args.scene_io_workers),
    )

    points, pred_cams, align_meta = align_prediction_to_gt_pose_sim3(
        pred_points=points,
        pred_cams=pred_cams,
        gt_cams=gt_cams,
    )
    points, colors = downsample_final_points(
        points,
        colors,
        max_points=pred_cap,
        voxel_size=float(args.voxel_downsample),
        point_downsample=bool(args.point_downsample),
        seed=int(args.seed) + 17,
    )
    if bool(args.point_downsample):
        gt_points, gt_colors = voxel_downsample(gt_points, gt_colors, float(args.voxel_downsample))

    eval_meta = {
        "schema": "final_eval_v1",
        "script": "scripts/run_all_in_one_to_rrd.py",
        "scene_dir": str(Path(args.scene_dir).expanduser().resolve()),
        "method": str(args.model),
        "method_display": str(args.model),
        "pose_convention": "T_c2w",
        "points_coordinate": "same_as_pred_cameras",
        "processing_time": processing_time,
        "model": {
            "name": str(args.model),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "hydra_overrides": hydra_overrides,
            "norm_type": str(args.norm_type),
            "max_side": int(args.max_side),
            "size_multiple": int(args.size_multiple),
        },
        "post_align": {
            "enabled": True,
            "type": "pose_sim3",
            "target": "gt_pose",
            **align_meta,
            "valid": bool(align_meta.get("valid", False)),
        },
        "aggregation": {
            "points": "all_pred_points",
            "cameras": "all_pred_cameras",
            "num_chunks": 1,
            "max_points_per_view": int(args.max_points_per_view),
            "effective_max_points_per_view": int(effective_max_points),
            "voxel_downsample": float(args.voxel_downsample),
            "effective_voxel_downsample": float(effective_voxel_size),
            "point_downsample": bool(args.point_downsample),
        },
    }
    save_final_eval_outputs(
        eval_dir=output_dir / "eval",
        pred_cams=pred_cams,
        gt_cams=gt_cams,
        pred_points=points,
        pred_colors=colors,
        gt_points=gt_points,
        gt_colors=gt_colors,
        meta=eval_meta,
    )

    sidecar = output_rrd.with_suffix(".json")
    sidecar_payload = {
        "method": str(args.model),
        "scene_dir": Path(args.scene_dir).expanduser().resolve(),
        "output_rrd": output_rrd,
        "output_dir": output_dir,
        "stems": stems,
        "target_size": {
            "height": int(meta["target_h"]),
            "width": int(meta["target_w"]),
        },
        "num_poses": len(pred_cams),
        "num_pred_points_logged": int(points.shape[0]),
        "num_gt_cameras": int(len(gt_cams)),
        "num_gt_points_logged": int(gt_points.shape[0]),
        "gt": {
            "num_cam_matches": int(len(gt_cams)),
            "num_depth_pointmaps_used": int(meta.get("num_depth_priors", 0)),
        },
        "alignment": align_meta,
        "chunking": {
            "max_chunk_size": int(len(stems)),
            "min_chunk_size": int(len(stems)),
            "auto_core_target_size": int(len(stems)),
            "num_chunks": 1,
            "max_points_per_view": int(args.max_points_per_view),
            "effective_max_points_per_view": int(effective_max_points),
            "voxel_downsample": float(args.voxel_downsample),
            "effective_voxel_downsample": float(effective_voxel_size),
            "point_downsample": bool(args.point_downsample),
            "note": "All-in-one method: all selected images are processed in one forward pass.",
        },
        "processing_time": processing_time,
        "model": eval_meta["model"],
    }
    sidecar.write_text(json.dumps(json_safe(sidecar_payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved sidecar metadata: {sidecar}")

    write_spatial_format_rrd(
        args=args,
        output_rrd=output_rrd,
        model_name=str(args.model),
        stems=stems,
        rgbs=rgbs,
        pred_cams=pred_cams,
        pred_points=points,
        pred_colors=colors,
        gt_cams=gt_cams,
        gt_points=gt_points,
        gt_colors=gt_colors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
