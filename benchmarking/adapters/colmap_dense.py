#!/usr/bin/env python3
"""Export a COLMAP dense reconstruction to the UAV-SLAM eval/RRD format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (BENCHMARK_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.colmap_io import read_model
from geoff3d.slrf.geometry_align import (
    apply_similarity_to_cameras,
    apply_similarity_to_points,
    estimate_similarity_umeyama,
)
from geoff3d.slrf.rrd_writer import (
    estimate_axis_size,
    load_point_cloud_ply,
    log_camera_axes,
    log_camera_labels,
    log_points,
    log_view_coordinates,
    rr_disconnect_compat,
    rr_init_save_compat,
    rr_set_time_compat,
    save_point_cloud_ply,
    send_blueprint,
)
from geoff3d.slrf.scene_io import build_views_from_scene, sample_points_and_colors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--sparse-model", type=Path, required=True)
    parser.add_argument("--fused-ply", type=Path, required=True)
    parser.add_argument("--output-rrd", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-side", type=int, default=518)
    parser.add_argument("--size-multiple", type=int, default=14)
    parser.add_argument("--max-rrd-points", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_colmap_cameras(sparse_model: Path):
    extension = ".bin" if (sparse_model / "images.bin").is_file() else ".txt"
    _cameras, images, _points = read_model(str(sparse_model), extension)
    pred_cams = []
    seen_stems = set()
    for image in sorted(images.values(), key=lambda item: str(item.name)):
        stem = Path(str(image.name)).stem
        if not stem or stem in seen_stems:
            continue
        R_w2c = np.asarray(image.qvec2rotmat(), dtype=np.float64)
        t_w2c = np.asarray(image.tvec, dtype=np.float64).reshape(3)
        T_c2w = np.eye(4, dtype=np.float64)
        T_c2w[:3, :3] = R_w2c.T
        T_c2w[:3, 3] = -R_w2c.T @ t_w2c
        pred_cams.append({"stem": stem, "T_c2w": T_c2w})
        seen_stems.add(stem)
    return pred_cams, extension


def align_to_gt(pred_cams, points: np.ndarray, scene_meta: dict):
    gt_by_stem = {
        str(stem): np.asarray(camera["T_c2w"], dtype=np.float64)
        for stem, camera in scene_meta.get("cams", {}).items()
        if "T_c2w" in camera
    }
    matched_pred = []
    matched_gt = []
    for camera in pred_cams:
        gt_pose = gt_by_stem.get(str(camera["stem"]))
        if gt_pose is None or gt_pose.shape != (4, 4):
            continue
        matched_pred.append(np.asarray(camera["T_c2w"])[:3, 3])
        matched_gt.append(gt_pose[:3, 3])

    if len(matched_pred) < 3:
        return pred_cams, points, {
            "valid": False,
            "num_matches": len(matched_pred),
            "scale": 1.0,
            "R": np.eye(3),
            "t": np.zeros(3),
            "note": "Need at least three matched cameras for COLMAP-to-GT Sim3.",
        }

    scale, rotation, translation, valid, note = estimate_similarity_umeyama(
        np.asarray(matched_pred),
        np.asarray(matched_gt),
        estimate_scale=True,
    )
    alignment = {
        "valid": bool(valid),
        "num_matches": len(matched_pred),
        "scale": float(scale),
        "R": rotation,
        "t": translation,
        "note": str(note),
    }
    if not valid:
        return pred_cams, points, alignment
    return (
        apply_similarity_to_cameras(
            pred_cams, scale, rotation, translation
        ),
        apply_similarity_to_points(points, scale, rotation, translation),
        alignment,
    )


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    sparse_model = args.sparse_model.expanduser().resolve()
    fused_ply = args.fused_ply.expanduser().resolve()
    output_rrd = args.output_rrd.expanduser().resolve()
    result_dir = output_rrd.with_suffix("")
    eval_dir = result_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    pred_cams, model_extension = load_colmap_cameras(sparse_model)
    if not pred_cams:
        raise RuntimeError(f"No registered COLMAP cameras found in {sparse_model}")
    points, colors = load_point_cloud_ply(fused_ply)
    if points.shape[0] == 0:
        raise RuntimeError(f"No finite points found in {fused_ply}")

    _views, scene_meta = build_views_from_scene(
        scene_dir=scene_dir,
        stride=max(1, int(args.stride)),
        max_image_size=int(args.max_side),
        patch_size=int(args.size_multiple),
        device=torch.device("cpu"),
        show_progress=False,
    )
    pred_cams, points, alignment = align_to_gt(pred_cams, points, scene_meta)

    stems = np.asarray([camera["stem"] for camera in pred_cams], dtype=str)
    poses = np.stack(
        [np.asarray(camera["T_c2w"], dtype=np.float32) for camera in pred_cams]
    )
    np.savez(
        eval_dir / "pred_cameras.npz",
        stems=stems,
        T_c2w=poses,
        valid=np.ones(len(pred_cams), dtype=bool),
    )

    eval_ply = eval_dir / "pred_points.ply"
    if eval_ply.is_symlink() or eval_ply.exists():
        eval_ply.unlink()
    save_point_cloud_ply(eval_ply, points=points, colors=colors)
    meta = {
        "schema": "colmap_dense_eval_v1",
        "script": "adapters/colmap_dense.py",
        "scene_dir": str(scene_dir),
        "method": "colmap_dense",
        "pose_convention": "T_c2w",
        "points_coordinate": "same_as_pred_cameras",
        "num_cameras": len(pred_cams),
        "num_valid_cameras": len(pred_cams),
        "num_points": int(points.shape[0]),
        "pred_cameras_path": "pred_cameras.npz",
        "pred_points_path": "pred_points.ply",
        "colmap_model_extension": model_extension,
        "colmap_to_gt_alignment": alignment,
    }
    (eval_dir / "meta.json").write_text(
        json.dumps(json_safe(meta), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rrd_points, rrd_colors = sample_points_and_colors(
        points,
        colors,
        max_points=max(0, int(args.max_rrd_points)),
        seed=int(args.seed),
    )
    output_rrd.parent.mkdir(parents=True, exist_ok=True)
    rr_init_save_compat(
        "colmap_dense_uav_slam",
        f"colmap_dense_{scene_dir.name}",
        output_rrd,
    )
    rr_set_time_compat("frame", 0)
    log_view_coordinates("RDF")
    send_blueprint(background=(255, 255, 255), hide_grid=False)
    log_points("world/colmap_dense/points", rrd_points, rrd_colors, 0.0)
    axis_size = estimate_axis_size([rrd_points], 0.0)
    log_camera_axes(
        "world/cameras/colmap/axes",
        pred_cams,
        axis_size,
        0.0,
        ((255, 0, 255), (255, 180, 0), (0, 220, 255)),
    )
    log_camera_labels(
        "world/cameras/colmap/labels",
        pred_cams,
        (255, 180, 0),
    )
    rr_disconnect_compat()

    sidecar = output_rrd.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            json_safe(
                {
                    "scene_dir": str(scene_dir),
                    "model": "colmap_dense",
                    "output_rrd": str(output_rrd),
                    "stems": stems.tolist(),
                    "grid": {
                        "seam_error": {
                            "enabled": False,
                            "reason": "COLMAP produces one global reconstruction.",
                        }
                    },
                    "chunking": {"num_chunks": 1},
                    "alignment": alignment,
                    "num_pred_cameras": len(pred_cams),
                    "num_pred_points": int(points.shape[0]),
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved COLMAP eval outputs: {eval_dir}")
    print(f"Saved COLMAP RRD: {output_rrd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
