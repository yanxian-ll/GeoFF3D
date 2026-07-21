#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Fast3R on all selected images in one forward pass and save benchmark outputs."""

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
    load_gt_artifacts,
    materialize_images,
    sample_points,
    save_final_eval_outputs,
    select_images,
    write_rrd,
)


def add_fast3r_paths(repo_root: Path) -> None:
    path = repo_root / "third_party/fast3r"
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def resolve_device(device_arg: str) -> torch.device:
    value = str(device_arg)
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    return torch.device(value)


def resolve_dtype(dtype_arg: str):
    key = str(dtype_arg).strip().lower()
    if key in {"float32", "fp32", "32"}:
        return "32"
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float16", "fp16", "16"}:
        return "16-mixed"
    raise ValueError(f"Unsupported --dtype {dtype_arg}")


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _view_rgb_u8(view: Dict[str, object]) -> np.ndarray:
    img = _as_numpy(view["img"])
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    rgb = np.clip((img * 0.5 + 0.5) * 255.0, 0.0, 255.0)
    return np.round(rgb).astype(np.uint8)


def predictions_to_dense_points(
    output_dict: Dict[str, object],
    *,
    min_conf_percentile: float,
    min_conf_value: float,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    preds = list(output_dict["preds"])
    views = list(output_dict["views"])
    point_parts: List[np.ndarray] = []
    color_parts: List[np.ndarray] = []
    per_view_meta: List[Dict[str, object]] = []

    for i, (pred, view) in enumerate(zip(preds, views)):
        pts = _as_numpy(pred["pts3d_in_other_view"])
        conf = _as_numpy(pred["conf"])
        if pts.ndim == 4:
            pts = pts[0]
        if conf.ndim == 3:
            conf = conf[0]
        pts = np.asarray(pts, dtype=np.float32)
        conf = np.asarray(conf, dtype=np.float32)
        rgb = _view_rgb_u8(view)
        if rgb.shape[:2] != pts.shape[:2]:
            raise RuntimeError(
                f"Fast3R RGB/point shape mismatch for view {i}: rgb={rgb.shape[:2]} pts={pts.shape[:2]}"
            )

        finite = np.isfinite(pts).all(axis=-1) & np.isfinite(conf)
        if finite.any() and float(min_conf_percentile) > 0:
            thr = float(np.nanpercentile(conf[finite], float(min_conf_percentile)))
        else:
            thr = float(min_conf_value)
        thr = max(thr, float(min_conf_value))
        mask = finite & (conf >= thr)

        num_kept = int(mask.sum())
        per_view_meta.append(
            {
                "view_index": int(i),
                "num_valid_points": int(finite.sum()),
                "num_kept_points": num_kept,
                "conf_threshold": float(thr),
                "max_conf": float(np.nanmax(conf[finite])) if finite.any() else float("nan"),
            }
        )
        if num_kept <= 0:
            continue
        point_parts.append(pts[mask].reshape(-1, 3).astype(np.float32))
        color_parts.append(rgb[mask].reshape(-1, 3).astype(np.uint8))

    if not point_parts:
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.uint8)
    else:
        points = np.concatenate(point_parts, axis=0)
        colors = np.concatenate(color_parts, axis=0)
    points, colors = sample_points(points, colors, int(max_points), int(seed))
    meta = {
        "min_conf_percentile": float(min_conf_percentile),
        "min_conf_value": float(min_conf_value),
        "num_views_with_points": int(len(point_parts)),
        "per_view": per_view_meta,
    }
    return points, colors, meta


def predictions_to_cameras(
    output_dict: Dict[str, object],
    stems: Sequence[str],
    *,
    niter_pnp: int,
    focal_length_estimation_method: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

    poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict["preds"],
        niter_PnP=int(niter_pnp),
        focal_length_estimation_method=str(focal_length_estimation_method),
    )
    poses = poses_c2w_batch[0] if poses_c2w_batch else []
    focals = estimated_focals[0] if estimated_focals else []
    cams: List[Dict[str, object]] = []
    for i, pose in enumerate(poses):
        stem = str(stems[i]) if i < len(stems) else f"view_{i:06d}"
        T = np.asarray(pose, dtype=np.float32)
        if T.shape == (4, 4) and np.isfinite(T).all():
            cam: Dict[str, object] = {"frame_id": float(i), "stem": stem, "T_c2w": T}
            if i < len(focals) and focals[i] is not None:
                cam["focal"] = float(focals[i])
            cams.append(cam)
    meta = {
        "niter_pnp": int(niter_pnp),
        "focal_length_estimation_method": str(focal_length_estimation_method),
        "estimated_focals": [None if f is None else float(f) for f in focals],
    }
    return cams, meta


def run_fast3r(
    *,
    image_paths: Sequence[Path],
    model_path: str,
    device: torch.device,
    image_size: int,
    dtype,
    profiling: bool,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    from fast3r.dust3r.inference_multiview import inference
    from fast3r.dust3r.utils.image import load_images
    from fast3r.utils.checkpoint_utils import load_model

    print(f"[Fast3R] loading model: {model_path}")
    model, lit_module = load_model(str(model_path), device=device, is_lightning_checkpoint=False)
    model.eval()
    lit_module.eval()

    images = load_images([str(p) for p in image_paths], size=int(image_size), verbose=True)
    if len(images) < 1:
        raise RuntimeError("Fast3R requires at least one image.")

    print(f"[Fast3R] one-pass inference on {len(images)} images.")
    result = inference(
        images,
        model,
        device,
        dtype=dtype,
        verbose=True,
        profiling=bool(profiling),
    )
    if bool(profiling):
        output_dict, profiling_info = result
    else:
        output_dict, profiling_info = result, None
    meta = {
        "model_path": str(model_path),
        "image_size": int(image_size),
        "dtype": str(dtype).replace("torch.", ""),
        "profiling": profiling_info,
    }
    return output_dict, meta


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--max_side", type=int, default=512)
    parser.add_argument("--size_multiple", type=int, default=16)
    parser.add_argument("--copy_images", action="store_true")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--depth_min", type=float, default=1e-6)
    parser.add_argument("--depth_max", type=float, default=1e6)
    parser.add_argument("--model_path", default="checkpoints/Fast3R_ViT_Large_512")
    parser.add_argument("--dtype", default="float32", choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument("--min_conf_percentile", type=float, default=10.0)
    parser.add_argument("--min_conf_value", type=float, default=1.0)
    parser.add_argument("--niter_pnp", type=int, default=100)
    parser.add_argument("--focal_length_estimation_method", default="first_view_from_global_head")
    parser.add_argument("--max_pred_points", type=int, default=2000000)
    parser.add_argument("--max_gt_points", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profiling", action="store_true")
    parser.add_argument("--log_images", action="store_true")
    parser.add_argument("--view_coordinates", default="RDF")
    parser.add_argument("--background", type=int, nargs=3, default=(255, 255, 255))
    parser.add_argument("--hide_grid", action="store_true")
    parser.add_argument("--point_radius", type=float, default=0.0)
    parser.add_argument("--camera_axis_size", type=float, default=0.0)
    parser.add_argument("--camera_axis_radius", type=float, default=0.0)
    parser.add_argument("--show_world_axes", action="store_true", default=True)
    parser.add_argument("--no_world_axes", action="store_false", dest="show_world_axes")
    return parser.parse_known_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, passthrough = parse_args(argv)
    if passthrough:
        print(f"[WARN] Ignoring passthrough args for Fast3R: {passthrough}")

    repo_root = Path(__file__).resolve().parents[1]
    add_fast3r_paths(repo_root)

    output_rrd = Path(args.output_rrd).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else output_rrd.with_suffix("")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stems, image_paths = select_images(args)
    prepared_dir = output_dir / "source" / "images"
    prepared_images, prepared_metadata = materialize_images(
        image_paths=image_paths,
        stems=stems,
        out_dir=prepared_dir,
        max_side=int(args.max_side),
        size_multiple=int(args.size_multiple),
        copy_images=bool(args.copy_images),
    )

    model_path = str(args.model_path)
    local_model_path = Path(model_path)
    if (local_model_path.is_absolute() or "/" in model_path or model_path.startswith(".")) and local_model_path.exists():
        model_path = str(local_model_path.resolve())
    elif (repo_root / model_path).exists():
        model_path = str((repo_root / model_path).resolve())

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    t0 = time.perf_counter()
    output_dict, fast3r_meta = run_fast3r(
        image_paths=prepared_images,
        model_path=model_path,
        device=device,
        image_size=int(args.max_side) if int(args.max_side) > 0 else 512,
        dtype=dtype,
        profiling=bool(args.profiling),
    )
    processing_time = {"processing_time_seconds": float(time.perf_counter() - t0)}

    points, colors, point_meta = predictions_to_dense_points(
        output_dict,
        min_conf_percentile=float(args.min_conf_percentile),
        min_conf_value=float(args.min_conf_value),
        max_points=int(args.max_pred_points),
        seed=int(args.seed),
    )
    cams, camera_meta = predictions_to_cameras(
        output_dict,
        stems,
        niter_pnp=int(args.niter_pnp),
        focal_length_estimation_method=str(args.focal_length_estimation_method),
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
            "script": "scripts/run_fast3r_to_rrd.py",
            "scene_dir": str(Path(args.scene_dir).expanduser().resolve()),
            "method": "fast3r",
            "method_display": "Fast3R",
            "pose_convention": "T_c2w",
            "points_coordinate": "same_as_pred_cameras",
            "processing_time": processing_time,
            "fast3r": fast3r_meta,
            "point_filter": point_meta,
            "camera_estimation": camera_meta,
            "post_align": {
                "enabled": True,
                "type": "pose_sim3",
                "target": "gt_pose",
                **align_meta,
                "valid": bool(align_meta.get("valid", False)),
            },
        },
    )

    sidecar = output_rrd.with_suffix(".json")
    payload = {
        "method": "fast3r",
        "method_display": "Fast3R",
        "scene_dir": Path(args.scene_dir).expanduser().resolve(),
        "output_rrd": output_rrd,
        "output_dir": output_dir,
        "model_path": model_path,
        "stems": stems,
        "prepared_images": prepared_metadata,
        "num_poses": len(cams),
        "num_pred_points_logged": int(points.shape[0]),
        "num_gt_cameras": int(len(gt_cams)),
        "num_gt_points_logged": int(gt_points.shape[0]),
        "gt": gt_meta,
        "alignment": align_meta,
        "fast3r": fast3r_meta,
        "point_filter": point_meta,
        "camera_estimation": camera_meta,
        "processing_time": processing_time,
    }
    sidecar.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved sidecar metadata: {sidecar}")

    write_rrd(
        args=args,
        method="fast3r",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
