#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run DROID-SLAM on a MapAnything scene and save benchmark RRD/eval outputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
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
    collect_stem_to_path,
    json_safe,
    load_gt_artifacts,
    materialize_images,
    parse_cam_txt,
    sample_points,
    save_final_eval_outputs,
    select_images,
    write_rrd,
)


def resolve_device_index(device: str) -> int:
    value = str(device).strip().lower()
    if value == "auto":
        return 0
    if value.isdigit():
        return int(value)
    if value.startswith("cuda:"):
        return int(value.split(":", 1)[1])
    return 0


def resolve_repo_path(repo_root: Path, path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def make_calib_file(
    *,
    scene_dir: Path,
    cams_dir: str,
    stems: Sequence[str],
    prepared_metadata: Sequence[Dict[str, object]],
    output_dir: Path,
) -> Path:
    cam_paths = collect_stem_to_path(scene_dir / cams_dir, {".txt"})
    first_cam = None
    first_meta = None
    for stem, meta in zip(stems, prepared_metadata):
        cam_path = cam_paths.get(str(stem))
        if cam_path is None:
            continue
        first_cam = parse_cam_txt(cam_path)
        first_meta = meta
        break
    if first_cam is None or first_meta is None:
        raise RuntimeError(
            "DROID-SLAM requires camera intrinsics. No matching GT camera txt was found "
            f"under {scene_dir / cams_dir} for the selected frames."
        )

    K = np.asarray(first_cam["K"], dtype=np.float64).copy()
    source_size = dict(first_meta.get("source_size", {}))
    prepared_size = dict(first_meta.get("prepared_size", {}))
    src_w = int(first_cam.get("width") or source_size.get("width") or prepared_size.get("width"))
    src_h = int(first_cam.get("height") or source_size.get("height") or prepared_size.get("height"))
    dst_w = int(prepared_size.get("width") or src_w)
    dst_h = int(prepared_size.get("height") or src_h)
    if src_w <= 0 or src_h <= 0:
        raise RuntimeError(f"Invalid camera/image size for DROID-SLAM calibration: {first_cam.get('path')}")

    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy

    calib_path = output_dir / "droid_calib.txt"
    calib_path.write_text(
        f"{K[0, 0]:.12g} {K[1, 1]:.12g} {K[0, 2]:.12g} {K[1, 2]:.12g}\n",
        encoding="utf-8",
    )
    return calib_path


def quaternion_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def droid_pose_w2c_to_c2w(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    R_w2c = quaternion_xyzw_to_matrix(pose[3:7])
    t_w2c = pose[:3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_w2c.T.astype(np.float32)
    T[:3, 3] = (-R_w2c.T @ t_w2c).astype(np.float32)
    return T


def depth_to_world_points(depth: np.ndarray, intrinsics: np.ndarray, T_c2w: np.ndarray) -> np.ndarray:
    h, w = depth.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64)
    fx, fy, cx, cy = [float(x) for x in intrinsics[:4]]
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)
    R = np.asarray(T_c2w, dtype=np.float64)[:3, :3]
    t = np.asarray(T_c2w, dtype=np.float64)[:3, 3]
    return (np.einsum("ij,hwj->hwi", R, pts_cam) + t[None, None, :]).astype(np.float32)


def load_droid_reconstruction(
    reconstruction_path: Path,
    prepared_metadata: Sequence[Dict[str, object]],
    *,
    max_pred_points: int,
    min_disp_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], np.ndarray, np.ndarray]:
    blob = torch.load(str(reconstruction_path), map_location="cpu")
    poses = np.asarray(blob["poses"], dtype=np.float32)
    disps = np.asarray(blob["disps"], dtype=np.float32)
    images = np.asarray(blob["images"], dtype=np.uint8)
    intrinsics = np.asarray(blob["intrinsics"], dtype=np.float32)
    tstamps = np.asarray(blob["tstamps"], dtype=np.float32)

    cams: List[Dict[str, object]] = []
    point_parts: List[np.ndarray] = []
    color_parts: List[np.ndarray] = []
    per_frame_cap = max(1, int(np.ceil(float(max_pred_points) * 1.5 / float(max(1, len(poses))))))

    for i, pose in enumerate(poses):
        frame_index = int(round(float(tstamps[i]))) if i < len(tstamps) else i
        if frame_index < 0 or frame_index >= len(prepared_metadata):
            frame_index = i
        stem = str(prepared_metadata[frame_index]["stem"]) if frame_index < len(prepared_metadata) else str(i)
        T_c2w = droid_pose_w2c_to_c2w(pose)
        if np.isfinite(T_c2w).all():
            cams.append({"frame_id": float(frame_index), "stem": stem, "T_c2w": T_c2w})

        if i >= len(disps) or i >= len(images) or i >= len(intrinsics):
            continue
        disp = np.asarray(disps[i], dtype=np.float32)
        mean_disp = float(np.nanmean(disp[np.isfinite(disp)])) if np.isfinite(disp).any() else 0.0
        mask = np.isfinite(disp) & (disp > max(1e-6, float(min_disp_ratio) * mean_disp))
        if not bool(mask.any()):
            continue
        depth = np.zeros_like(disp, dtype=np.float32)
        depth[mask] = 1.0 / disp[mask]
        pts = depth_to_world_points(depth, intrinsics[i], T_c2w)
        img = np.asarray(images[i], dtype=np.uint8).transpose(1, 2, 0)
        colors = img[..., ::-1]
        if colors.shape[:2] != disp.shape[:2]:
            colors = colors[: disp.shape[0], : disp.shape[1], :]

        valid = mask & np.isfinite(pts).all(axis=-1)
        if not bool(valid.any()):
            continue
        pts_i = pts[valid].reshape(-1, 3).astype(np.float32)
        col_i = colors[valid].reshape(-1, 3).astype(np.uint8)
        pts_i, col_i = sample_points(pts_i, col_i, per_frame_cap, seed + 8191 * (i + 1))
        point_parts.append(pts_i)
        color_parts.append(col_i)

    if point_parts:
        points = np.concatenate(point_parts, axis=0)
        colors = np.concatenate(color_parts, axis=0)
        points, colors = sample_points(points, colors, int(max_pred_points), int(seed))
    else:
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.uint8)
    return cams, points, colors


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--output_rrd", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--droid_root", default="third_party/DROID-SLAM")
    parser.add_argument("--weights", default="checkpoints/droid/droid.pth")
    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--cams_dir", default="cams")
    parser.add_argument("--depth_dir", default="depth")
    parser.add_argument("--frame_glob", default="*")
    parser.add_argument("--num_views", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_side", type=int, default=512)
    parser.add_argument("--size_multiple", type=int, default=8)
    parser.add_argument("--copy_images", action="store_true")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--depth_min", type=float, default=1e-6)
    parser.add_argument("--depth_max", type=float, default=1e6)
    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--keyframe_thresh", type=float, default=4.0)
    parser.add_argument("--frontend_thresh", type=float, default=16.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--max_pred_points", type=int, default=800000)
    parser.add_argument("--max_gt_points", type=int, default=800000)
    parser.add_argument("--min_disp_ratio", type=float, default=0.25)
    parser.add_argument("--scene_io_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_images", action="store_true")
    parser.add_argument("--view_coordinates", default="RDF")
    parser.add_argument("--background", type=int, nargs=3, default=(255, 255, 255))
    parser.add_argument("--hide_grid", action="store_true")
    parser.add_argument("--point_radius", type=float, default=0.0)
    parser.add_argument("--camera_axis_size", type=float, default=0.0)
    parser.add_argument("--camera_axis_radius", type=float, default=0.0)
    parser.add_argument("--show_world_axes", action="store_true", default=True)
    parser.add_argument("--no_world_axes", action="store_false", dest="show_world_axes")
    parser.add_argument("--world_axis_size", type=float, default=0.0)
    parser.add_argument("--world_axis_radius", type=float, default=0.0)
    return parser.parse_known_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, passthrough = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    droid_root = resolve_repo_path(repo_root, args.droid_root)
    weights = resolve_repo_path(repo_root, args.weights)
    if not (droid_root / "demo.py").exists():
        raise FileNotFoundError(f"Missing DROID-SLAM demo.py under {droid_root}")
    if not weights.exists():
        raise FileNotFoundError(f"Missing DROID-SLAM checkpoint: {weights}")

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
    calib_path = make_calib_file(
        scene_dir=Path(args.scene_dir).expanduser().resolve(),
        cams_dir=str(args.cams_dir),
        stems=stems,
        prepared_metadata=prepared_metadata,
        output_dir=output_dir,
    )

    reconstruction_path = output_dir / "droid_reconstruction.pth"
    device_index = resolve_device_index(args.device)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_index)
    pythonpath = [
        str(droid_root),
        str(droid_root / "droid_slam"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join([p for p in pythonpath if p])
    cmd = [
        str(args.python),
        str(droid_root / "demo.py"),
        "--imagedir",
        str(prepared_dir),
        "--calib",
        str(calib_path),
        "--stride",
        "1",
        "--weights",
        str(weights),
        "--buffer",
        str(args.buffer),
        "--warmup",
        str(args.warmup),
        "--filter_thresh",
        str(args.filter_thresh),
        "--keyframe_thresh",
        str(args.keyframe_thresh),
        "--frontend_thresh",
        str(args.frontend_thresh),
        "--frontend_window",
        str(args.frontend_window),
        "--frontend_radius",
        str(args.frontend_radius),
        "--frontend_nms",
        str(args.frontend_nms),
        "--backend_thresh",
        str(args.backend_thresh),
        "--backend_radius",
        str(args.backend_radius),
        "--backend_nms",
        str(args.backend_nms),
        "--disable_vis",
        "--reconstruction_path",
        str(reconstruction_path),
    ]
    if bool(args.asynchronous):
        cmd.append("--asynchronous")
    if bool(args.upsample):
        cmd.append("--upsample")
    cmd.extend(passthrough)

    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(droid_root), env=env, check=True)
    processing_time = {"processing_time_seconds": float(time.perf_counter() - t0)}
    if not reconstruction_path.exists():
        raise RuntimeError(f"DROID-SLAM did not write reconstruction: {reconstruction_path}")

    cams, points, colors = load_droid_reconstruction(
        reconstruction_path,
        prepared_metadata,
        max_pred_points=int(args.max_pred_points),
        min_disp_ratio=float(args.min_disp_ratio),
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
            "script": "scripts/run_droid_slam_to_rrd.py",
            "scene_dir": str(Path(args.scene_dir).expanduser().resolve()),
            "method": "droid-slam",
            "method_display": "DROID-SLAM",
            "pose_convention": "T_c2w",
            "points_coordinate": "same_as_pred_cameras",
            "processing_time": processing_time,
            "droid_slam": {
                "droid_root": str(droid_root),
                "weights": str(weights),
                "buffer": int(args.buffer),
                "warmup": int(args.warmup),
                "asynchronous": bool(args.asynchronous),
                "upsample": bool(args.upsample),
                "calib": str(calib_path),
                "reconstruction": str(reconstruction_path),
            },
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
        "method": "droid-slam",
        "method_display": "DROID-SLAM",
        "scene_dir": Path(args.scene_dir).expanduser().resolve(),
        "output_rrd": output_rrd,
        "output_dir": output_dir,
        "droid_root": droid_root,
        "weights": weights,
        "stems": stems,
        "prepared_images": prepared_metadata,
        "num_poses": len(cams),
        "num_pred_points_logged": int(points.shape[0]),
        "num_gt_cameras": int(len(gt_cams)),
        "num_gt_points_logged": int(gt_points.shape[0]),
        "gt": gt_meta,
        "alignment": align_meta,
        "processing_time": processing_time,
    }
    sidecar.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved sidecar metadata: {sidecar}")

    write_rrd(
        args=args,
        method="droid-slam",
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
