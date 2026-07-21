#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run an external streaming reconstruction method and save RRD/eval outputs.

Example:

    python scripts/run_streaming_to_rrd.py \
      --method lingbot-map \
      --scene_dir /path/to/scene \
      --output_rrd outputs/stream/lingbot-map/scene.rrd \
      --model_path /path/to/lingbot-map.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from omegaconf import OmegaConf


def parse_args(argv: Optional[Sequence[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, help="Streaming method name, e.g. lingbot-map.")
    parser.add_argument("--scene_dir", required=True, help="Scene folder containing images/.")
    parser.add_argument("--output_rrd", required=True, help="Output .rrd path.")
    parser.add_argument("--output_dir", default=None, help="Directory for prepared images and logs.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model_path", default=None, help="Streaming model checkpoint/path override.")
    parser.add_argument(
        "--stream_mode",
        default=None,
        choices=["streaming", "windowed", "causal", "window", "full"],
    )
    parser.add_argument("--keyframe_interval", default=None)
    parser.add_argument("--reset_interval", type=int, default=None)
    parser.add_argument("--model_update_type", default=None)
    parser.add_argument("--use_sdpa", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--cams_dir", default="cams")
    parser.add_argument("--depth_dir", default="depth")
    parser.add_argument("--frame_glob", default="*")
    parser.add_argument("--num_views", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_side", type=int, default=0)
    parser.add_argument("--size_multiple", type=int, default=1)
    parser.add_argument("--copy_images", action="store_true")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--depth_min", type=float, default=1e-6)
    parser.add_argument("--depth_max", type=float, default=1e6)
    parser.add_argument("--scene_io_workers", type=int, default=0)

    parser.add_argument("--max_pred_points", type=int, default=800000)
    parser.add_argument("--max_gt_points", type=int, default=800000)
    parser.add_argument("--max_points_per_view", type=int, default=500000)
    parser.add_argument("--voxel_downsample", type=float, default=0.01)
    parser.add_argument("--no_point_downsample", action="store_false", dest="point_downsample")
    parser.set_defaults(point_downsample=True)

    parser.add_argument("--point_radius", type=float, default=0.0)
    parser.add_argument("--view_coordinates", default="RDF")
    parser.add_argument("--background", type=int, nargs=3, default=[255, 255, 255])
    parser.add_argument("--hide_grid", action="store_true")
    parser.add_argument("--log_images", action="store_true")
    parser.add_argument("--camera_axis_size", type=float, default=0.0)
    parser.add_argument("--camera_axis_radius", type=float, default=0.0)
    parser.add_argument("--show_world_axes", action="store_true", default=True)
    parser.add_argument("--no_world_axes", action="store_false", dest="show_world_axes")
    parser.add_argument("--world_axis_size", type=float, default=0.0)
    parser.add_argument("--world_axis_radius", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep_intermediate", action="store_true")
    return parser.parse_known_args(argv)


def load_model_config(method: str) -> Dict[str, object]:
    config_name = str(method).replace("-", "_")
    config_path = Path(__file__).resolve().parents[1] / "configs" / "model" / f"{config_name}.yaml"
    if not config_path.exists():
        return {}
    cfg = OmegaConf.load(config_path)
    values = OmegaConf.to_container(cfg.get("model_config", {}), resolve=True)
    return dict(values or {})


def load_streaming_points(
    point_cloud_path: Optional[Path],
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from run_vggt_slam_to_rrd import load_point_cloud_file, sample_points

    if point_cloud_path is None or not point_cloud_path.exists():
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            [],
        )
    points, colors = load_point_cloud_file(point_cloud_path)
    points, colors = sample_points(points, colors, max_points=max_points, seed=seed)
    return points, colors, [str(point_cloud_path)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, passthrough = parse_args(argv)
    from run_vggt_slam_to_rrd import (
        align_prediction_to_gt_pose_sim3,
        downsample_final_points,
        json_safe,
        load_gt_artifacts,
        load_json_file,
        materialize_images,
        read_pose_log,
        save_final_eval_outputs,
        select_images,
        write_rrd,
    )

    output_rrd = Path(args.output_rrd).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else output_rrd.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    stems, image_paths = select_images(args)
    prepared_dir = output_dir / "selected_images"
    prepared_images, prepared_metadata = materialize_images(
        image_paths=image_paths,
        stems=stems,
        out_dir=prepared_dir,
        max_side=int(args.max_side),
        size_multiple=int(args.size_multiple),
        copy_images=bool(args.copy_images),
    )

    from mapanything.models import model_factory

    model_kwargs = load_model_config(args.method)
    model_kwargs.update(
        {
            "name": args.method,
            "torch_hub_force_reload": False,
            "max_points": int(args.max_pred_points),
            "seed": int(args.seed),
        }
    )
    if args.model_path is not None:
        model_kwargs["checkpoint"] = args.model_path
        model_kwargs["model_path"] = args.model_path
    if args.device is not None:
        model_kwargs["device"] = str(args.device)
    else:
        model_kwargs.setdefault("device", "auto")
    if args.stream_mode is not None:
        model_kwargs["mode"] = args.stream_mode
    if args.use_sdpa is not None:
        model_kwargs["use_sdpa"] = bool(args.use_sdpa)
    if int(args.max_side) > 0 and "image_size" not in model_kwargs:
        model_kwargs["image_size"] = int(args.max_side)
    if args.keyframe_interval is not None:
        model_kwargs["keyframe_interval"] = args.keyframe_interval
    if args.reset_interval is not None:
        model_kwargs["reset_interval"] = int(args.reset_interval)
    if args.model_update_type is not None:
        model_kwargs["model_update_type"] = str(args.model_update_type)

    method = model_factory(args.method, **model_kwargs)
    print(
        f"Selected {len(prepared_images)} images for {method.display_name}; "
        f"prepared input: {prepared_dir}"
    )

    pose_log_path = output_dir / "camera_poses.txt"
    point_cloud_path = output_dir / "pred_points.ply"
    timing_path = output_dir / "processing_time.json"
    for stale in (pose_log_path, point_cloud_path, timing_path):
        if stale.exists():
            stale.unlink()

    result = method.run(
        scene_dir=Path(args.scene_dir).expanduser().resolve(),
        image_dir=prepared_dir,
        output_dir=output_dir,
        pose_log_path=pose_log_path,
        point_cloud_path=point_cloud_path,
        timing_path=timing_path,
        device=str(model_kwargs.get("device", "auto")),
        python=str(args.python),
        max_points=int(args.max_pred_points),
        extra_args=passthrough,
    )

    processing_time_meta = load_json_file(timing_path)
    if int(result["return_code"]) != 0:
        sidecar = output_rrd.with_suffix(".json")
        payload: Dict[str, object] = {
            "method": method.name,
            "method_display": method.display_name,
            "scene_dir": str(Path(args.scene_dir).expanduser().resolve()),
            "output_rrd": str(output_rrd),
            "output_dir": str(output_dir),
            "prepared_dir": str(prepared_dir),
            "return_code": int(result["return_code"]),
            "command": result["command"],
            "stdout_log_path": str(result["stdout_log_path"]),
            "error": "External streaming process failed; skip post-processing.",
        }
        sidecar.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[ERROR] {method.display_name} exited with code {result['return_code']}. See {result['stdout_log_path']}")
        return int(result["return_code"])

    cams = read_pose_log(pose_log_path, prepared_metadata=prepared_metadata)
    points, colors, point_artifacts = load_streaming_points(
        result["point_cloud_path"] or point_cloud_path,
        max_points=int(args.max_pred_points),
        seed=int(args.seed),
    )
    gt_cams, gt_points, gt_colors, gt_meta = load_gt_artifacts(
        args=args,
        stems=stems,
        image_paths=image_paths,
    )

    points, cams, align_meta = align_prediction_to_gt_pose_sim3(
        pred_points=points,
        pred_cams=cams,
        gt_cams=gt_cams,
    )

    points, colors = downsample_final_points(
        points,
        colors,
        enabled=bool(args.point_downsample),
        max_points=int(args.max_points_per_view),
        voxel_size=float(args.voxel_downsample),
        seed=int(args.seed) + 17,
        label="pred",
    )
    gt_points, gt_colors = downsample_final_points(
        gt_points,
        gt_colors,
        enabled=bool(args.point_downsample),
        max_points=int(args.max_points_per_view),
        voxel_size=float(args.voxel_downsample),
        seed=int(args.seed) + 23,
        label="gt",
    )

    save_final_eval_outputs(
        eval_dir=output_dir / "eval",
        pred_cams=cams,
        gt_cams=gt_cams,
        pred_points=points,
        pred_colors=colors,
        gt_points=gt_points,
        gt_colors=gt_colors,
        meta={
            "schema": "final_eval_v1",
            "script": "scripts/run_streaming_to_rrd.py",
            "scene_dir": str(Path(args.scene_dir).expanduser().resolve()),
            "method": method.name,
            "method_display": method.display_name,
            "pose_convention": "T_c2w",
            "points_coordinate": "same_as_pred_cameras",
            "processing_time": processing_time_meta,
            "post_align": {
                "enabled": True,
                "type": "pose_sim3",
                "target": "gt_pose",
                "valid": bool(align_meta.get("valid", False)),
                "scale": float(align_meta.get("scale", 1.0)),
                "R": align_meta.get("R", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                "t": align_meta.get("t", [0, 0, 0]),
                "median_camera_residual": float(align_meta.get("median_residual", float("nan"))),
            },
        },
    )

    sidecar = output_rrd.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            json_safe(
                {
                    "method": method.name,
                    "method_display": method.display_name,
                    "scene_dir": Path(args.scene_dir).expanduser().resolve(),
                    "output_rrd": output_rrd,
                    "output_dir": output_dir,
                    "prepared_dir": prepared_dir,
                    "pose_log_path": pose_log_path,
                    "point_artifacts": point_artifacts,
                    "staged_outputs": result["staged_outputs"],
                    "stdout_log_path": result["stdout_log_path"],
                    "processing_time": processing_time_meta,
                    "return_code": int(result["return_code"]),
                    "command": result["command"],
                    "stems": stems,
                    "prepared_images": prepared_metadata,
                    "num_poses": len(cams),
                    "num_pred_points_logged": int(points.shape[0]),
                    "num_gt_cameras": int(len(gt_cams)),
                    "num_gt_points_logged": int(gt_points.shape[0]),
                    "gt": gt_meta,
                    "alignment": align_meta,
                    "passthrough_args": passthrough,
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved sidecar metadata: {sidecar}")

    write_rrd(
        args=args,
        method=method.name,
        output_rrd=output_rrd,
        prepared_metadata=prepared_metadata,
        prepared_images=prepared_images,
        cams=cams,
        points=points,
        colors=colors,
        gt_cams=gt_cams,
        gt_points=gt_points,
        gt_colors=gt_colors,
    )

    if not args.keep_intermediate:
        try:
            shutil.rmtree(prepared_dir)
            print(f"[CLEANUP] removed directory: {prepared_dir}")
        except Exception as exc:
            print(f"[CLEANUP][WARN] failed to remove {prepared_dir}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
