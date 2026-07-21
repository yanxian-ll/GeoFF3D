#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate final UAV SLAM/reconstruction outputs.

Expected eval_dir format:
    eval/
      pred_cameras.npz   # stems, T_c2w, valid
      pred_points.ply    # x, y, z, red, green, blue
      meta.json          # optional

This script intentionally does NOT apply any extra alignment.
VGGT-SLAM outputs should already be aligned by scripts/run_vggt_slam_to_rrd.py.
Our spatial-chunk methods should already be in the intended evaluation frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_vggt_slam_to_rrd as scene_io
from spatial_rrd.rrd_writer import load_point_cloud_ply


def json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def parse_float_list(text: str) -> List[float]:
    vals = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    return vals


def parse_int_list(text: str) -> List[int]:
    vals = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    return vals


def load_eval_outputs(eval_dir: Path) -> Dict[str, object]:
    cam_path = eval_dir / "pred_cameras.npz"
    ply_path = eval_dir / "pred_points.ply"
    npz_path = eval_dir / "pred_points.npz"
    meta_path = eval_dir / "meta.json"

    if not cam_path.exists():
        raise FileNotFoundError(f"Missing {cam_path}")
    if not ply_path.exists() and not npz_path.exists():
        raise FileNotFoundError(f"Missing {ply_path}")

    with np.load(cam_path, allow_pickle=True) as data:
        stems = [str(x) for x in np.asarray(data["stems"]).tolist()]
        T_c2w = np.asarray(data["T_c2w"], dtype=np.float64)
        valid = np.asarray(data["valid"], dtype=bool)

    if ply_path.exists():
        points, colors = load_point_cloud_ply(ply_path)
    else:
        with np.load(npz_path, allow_pickle=True) as data:
            points = np.asarray(data["points"], dtype=np.float32).reshape(-1, 3)
            colors = (
                np.asarray(data["colors"], dtype=np.uint8).reshape(-1, 3)
                if "colors" in data
                else np.empty((0, 3), np.uint8)
            )

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return {
        "stems": stems,
        "T_c2w": T_c2w,
        "valid": valid,
        "points": points,
        "colors": colors,
        "meta": meta,
    }


def load_gt_for_eval(
    scene_dir: Path,
    pred_stems: Sequence[str],
    images_dir: str,
    cams_dir: str,
    depth_dir: str,
    depth_scale: float,
    depth_min: float,
    depth_max: float,
    max_gt_points: int,
    seed: int,
):
    images = scene_io.collect_stem_to_path(scene_dir / images_dir, scene_io.IMAGE_EXTS)

    stems = [s for s in pred_stems if s and s in images]
    if not stems:
        stems = sorted(images.keys())

    image_paths = [images[s] for s in stems]

    args = argparse.Namespace(
        scene_dir=str(scene_dir),
        cams_dir=cams_dir,
        depth_dir=depth_dir,
        depth_scale=depth_scale,
        depth_min=depth_min,
        depth_max=depth_max,
        max_gt_points=max_gt_points,
        seed=seed,
    )

    gt_cams, gt_points, gt_colors, gt_meta = scene_io.load_gt_artifacts(
        args=args,
        stems=stems,
        image_paths=image_paths,
    )
    return gt_cams, gt_points, gt_colors, gt_meta


def rotation_angle_deg(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = (float(np.trace(R)) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def summarize(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_rmse": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(values * values))),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def match_cameras(
    pred_stems: Sequence[str],
    pred_T: np.ndarray,
    pred_valid: np.ndarray,
    gt_cams: Sequence[Dict[str, object]],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    gt_by_stem = {
        str(cam["stem"]): np.asarray(cam["T_c2w"], dtype=np.float64)
        for cam in gt_cams
        if "stem" in cam and "T_c2w" in cam
    }

    matched_stems: List[str] = []
    pred_list: List[np.ndarray] = []
    gt_list: List[np.ndarray] = []

    for i, stem in enumerate(pred_stems):
        if i >= len(pred_valid) or not bool(pred_valid[i]):
            continue
        if stem not in gt_by_stem:
            continue

        T_pred = np.asarray(pred_T[i], dtype=np.float64)
        T_gt = gt_by_stem[stem]
        if T_pred.shape != (4, 4) or T_gt.shape != (4, 4):
            continue
        if not np.isfinite(T_pred).all() or not np.isfinite(T_gt).all():
            continue

        matched_stems.append(stem)
        pred_list.append(T_pred)
        gt_list.append(T_gt)

    if not pred_list:
        return matched_stems, np.empty((0, 4, 4), np.float64), np.empty((0, 4, 4), np.float64)

    return matched_stems, np.stack(pred_list, axis=0), np.stack(gt_list, axis=0)


def evaluate_pose(
    pred_stems: Sequence[str],
    pred_T: np.ndarray,
    pred_valid: np.ndarray,
    gt_cams: Sequence[Dict[str, object]],
    rpe_steps: Sequence[int],
) -> Dict[str, object]:
    matched_stems, T_pred, T_gt = match_cameras(pred_stems, pred_T, pred_valid, gt_cams)

    result: Dict[str, object] = {
        "valid": bool(T_pred.shape[0] > 0),
        "num_pred_cameras": int(len(pred_stems)),
        "num_gt_cameras": int(len(gt_cams)),
        "num_matches": int(T_pred.shape[0]),
        "matched_stems": matched_stems,
    }

    if T_pred.shape[0] == 0:
        result["note"] = "No matched valid cameras."
        return result

    center_pred = T_pred[:, :3, 3]
    center_gt = T_gt[:, :3, 3]
    ate = np.linalg.norm(center_pred - center_gt, axis=1)

    rot_err = []
    for i in range(T_pred.shape[0]):
        R_err = T_gt[i, :3, :3].T @ T_pred[i, :3, :3]
        rot_err.append(rotation_angle_deg(R_err))
    rot_err = np.asarray(rot_err, dtype=np.float64)

    result["ate"] = summarize(ate, "ate")
    result["rotation"] = summarize(rot_err, "rot_deg")

    rpe: Dict[str, object] = {}
    for step in rpe_steps:
        step = int(step)
        if step <= 0 or T_pred.shape[0] <= step:
            continue

        trans_err = []
        rot_step_err = []

        for i in range(0, T_pred.shape[0] - step):
            j = i + step
            rel_gt = np.linalg.inv(T_gt[i]) @ T_gt[j]
            rel_pred = np.linalg.inv(T_pred[i]) @ T_pred[j]
            E = np.linalg.inv(rel_gt) @ rel_pred

            trans_err.append(float(np.linalg.norm(E[:3, 3])))
            rot_step_err.append(rotation_angle_deg(E[:3, :3]))

        rpe[f"k{step}"] = {
            "num_pairs": int(len(trans_err)),
            "translation": summarize(np.asarray(trans_err), "rpe_trans"),
            "rotation": summarize(np.asarray(rot_step_err), "rpe_rot_deg"),
        }

    result["rpe"] = rpe
    return result


def sample_points_for_eval(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    max_points: int,
    seed: int,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if colors is None or colors.shape[0] != points.shape[0]:
        colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    points, _ = scene_io.sample_points(points, colors, max_points=max_points, seed=seed)
    return points


def nn_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float32).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float32).reshape(-1, 3)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(dst)
        try:
            dists, _idx = tree.query(src, k=1, workers=-1)
        except TypeError:
            dists, _idx = tree.query(src, k=1)
        return np.asarray(dists, dtype=np.float32)
    except Exception as e:
        # Fallback only for small inputs. Large brute-force NN is too slow.
        max_pairs = 50_000_000
        if src.shape[0] * dst.shape[0] > max_pairs:
            raise RuntimeError(
                "scipy.spatial.cKDTree is required for large point-cloud metrics. "
                "Install scipy or lower --max_pred_points_eval/--max_gt_points."
            ) from e

        out = np.empty((src.shape[0],), dtype=np.float32)
        chunk = 4096
        dst64 = dst.astype(np.float64)
        for start in range(0, src.shape[0], chunk):
            part = src[start : start + chunk].astype(np.float64)
            diff = part[:, None, :] - dst64[None, :, :]
            d2 = np.sum(diff * diff, axis=-1)
            out[start : start + chunk] = np.sqrt(np.min(d2, axis=1)).astype(np.float32)
        return out


def evaluate_geometry(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    thresholds: Sequence[float],
    max_pred_points_eval: int,
    max_gt_points_eval: int,
    seed: int,
) -> Dict[str, object]:
    pred_points = sample_points_for_eval(pred_points, None, max_pred_points_eval, seed + 101)
    gt_points = sample_points_for_eval(gt_points, None, max_gt_points_eval, seed + 202)

    result: Dict[str, object] = {
        "valid": bool(pred_points.shape[0] > 0 and gt_points.shape[0] > 0),
        "num_pred_points": int(pred_points.shape[0]),
        "num_gt_points": int(gt_points.shape[0]),
        "thresholds": [float(t) for t in thresholds],
    }

    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        result["note"] = "Missing pred or GT points."
        return result

    d_pred_to_gt = nn_distances(pred_points, gt_points)
    d_gt_to_pred = nn_distances(gt_points, pred_points)

    result["accuracy_pred_to_gt"] = summarize(d_pred_to_gt, "acc")
    result["completeness_gt_to_pred"] = summarize(d_gt_to_pred, "comp")
    result["chamfer_l1"] = float(np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred))
    result["chamfer_l2"] = float(np.mean(d_pred_to_gt * d_pred_to_gt) + np.mean(d_gt_to_pred * d_gt_to_pred))

    fscores: Dict[str, object] = {}
    for t in thresholds:
        t = float(t)
        precision = float(np.mean(d_pred_to_gt <= t))
        recall = float(np.mean(d_gt_to_pred <= t))
        fscore = 0.0 if precision + recall <= 0 else float(2.0 * precision * recall / (precision + recall))
        fscores[str(t)] = {
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
            "outlier_ratio": float(1.0 - precision),
        }

    result["fscore"] = fscores
    return result


def nested_get(d: Dict[str, object], path: Sequence[str], default=float("nan")):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def write_summary_csv(metrics: Dict[str, object], output_csv: Path) -> None:
    pose = metrics.get("pose", {})
    geom = metrics.get("geometry", {})
    meta = metrics.get("eval_meta", {})

    row = {
        "scene": metrics.get("scene_name", ""),
        "eval_dir": metrics.get("eval_dir", ""),
        "method": meta.get("method", ""),
        "checkpoint": meta.get("checkpoint", ""),
        "post_align_enabled": nested_get(meta, ["post_align", "enabled"], ""),
        "post_align_type": nested_get(meta, ["post_align", "type"], ""),
        "pose_num_matches": nested_get(pose, ["num_matches"], 0),
        "ate_rmse": nested_get(pose, ["ate", "ate_rmse"]),
        "ate_median": nested_get(pose, ["ate", "ate_median"]),
        "ate_p90": nested_get(pose, ["ate", "ate_p90"]),
        "rot_deg_rmse": nested_get(pose, ["rotation", "rot_deg_rmse"]),
        "rot_deg_median": nested_get(pose, ["rotation", "rot_deg_median"]),
        "rot_deg_p90": nested_get(pose, ["rotation", "rot_deg_p90"]),
        "rpe_k1_trans_rmse": nested_get(pose, ["rpe", "k1", "translation", "rpe_trans_rmse"]),
        "rpe_k1_rot_rmse": nested_get(pose, ["rpe", "k1", "rotation", "rpe_rot_deg_rmse"]),
        "num_pred_points": nested_get(geom, ["num_pred_points"], 0),
        "num_gt_points": nested_get(geom, ["num_gt_points"], 0),
        "acc_mean": nested_get(geom, ["accuracy_pred_to_gt", "acc_mean"]),
        "acc_median": nested_get(geom, ["accuracy_pred_to_gt", "acc_median"]),
        "acc_p90": nested_get(geom, ["accuracy_pred_to_gt", "acc_p90"]),
        "comp_mean": nested_get(geom, ["completeness_gt_to_pred", "comp_mean"]),
        "comp_median": nested_get(geom, ["completeness_gt_to_pred", "comp_median"]),
        "comp_p90": nested_get(geom, ["completeness_gt_to_pred", "comp_p90"]),
        "chamfer_l1": nested_get(geom, ["chamfer_l1"]),
        "chamfer_l2": nested_get(geom, ["chamfer_l2"]),
    }

    fscore = geom.get("fscore", {}) if isinstance(geom, dict) else {}
    for key, val in fscore.items():
        if not isinstance(val, dict):
            continue
        suffix = key.replace(".", "p")
        row[f"precision@{suffix}"] = val.get("precision", float("nan"))
        row[f"recall@{suffix}"] = val.get("recall", float("nan"))
        row[f"fscore@{suffix}"] = val.get("fscore", float("nan"))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(json_safe(row))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None)

    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--cams_dir", default="cams")
    parser.add_argument("--depth_dir", default="depth")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--depth_min", type=float, default=1e-6)
    parser.add_argument("--depth_max", type=float, default=1e6)

    parser.add_argument("--thresholds", default="0.1,0.2,0.5,1.0")
    parser.add_argument("--rpe_steps", default="1,5,10")

    parser.add_argument("--max_gt_points", type=int, default=300000)
    parser.add_argument("--max_pred_points_eval", type=int, default=300000)
    parser.add_argument("--max_gt_points_eval", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--skip_pose", action="store_true")
    parser.add_argument("--skip_geometry", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    scene_dir = Path(args.scene_dir).expanduser().resolve()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else eval_dir / "metrics.json"
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else eval_dir / "metrics_summary.csv"

    thresholds = parse_float_list(args.thresholds)
    rpe_steps = parse_int_list(args.rpe_steps)

    pred = load_eval_outputs(eval_dir)

    pred_stems = pred["stems"]
    pred_T = pred["T_c2w"]
    pred_valid = pred["valid"]
    pred_points = pred["points"]
    eval_meta = pred.get("meta", {})

    gt_cams, gt_points, gt_colors, gt_meta = load_gt_for_eval(
        scene_dir=scene_dir,
        pred_stems=pred_stems,
        images_dir=args.images_dir,
        cams_dir=args.cams_dir,
        depth_dir=args.depth_dir,
        depth_scale=float(args.depth_scale),
        depth_min=float(args.depth_min),
        depth_max=float(args.depth_max),
        max_gt_points=int(args.max_gt_points),
        seed=int(args.seed),
    )

    metrics: Dict[str, object] = {
        "schema": "final_eval_metrics_v1",
        "scene_dir": str(scene_dir),
        "scene_name": scene_dir.name,
        "eval_dir": str(eval_dir),
        "eval_meta": eval_meta,
        "gt_meta": gt_meta,
        "settings": {
            "thresholds": thresholds,
            "rpe_steps": rpe_steps,
            "max_gt_points": int(args.max_gt_points),
            "max_pred_points_eval": int(args.max_pred_points_eval),
            "max_gt_points_eval": int(args.max_gt_points_eval),
            "skip_pose": bool(args.skip_pose),
            "skip_geometry": bool(args.skip_geometry),
        },
    }

    if not args.skip_pose:
        metrics["pose"] = evaluate_pose(
            pred_stems=pred_stems,
            pred_T=pred_T,
            pred_valid=pred_valid,
            gt_cams=gt_cams,
            rpe_steps=rpe_steps,
        )

    if not args.skip_geometry:
        metrics["geometry"] = evaluate_geometry(
            pred_points=pred_points,
            gt_points=gt_points,
            thresholds=thresholds,
            max_pred_points_eval=int(args.max_pred_points_eval),
            max_gt_points_eval=int(args.max_gt_points_eval),
            seed=int(args.seed),
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(json_safe(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_summary_csv(metrics, output_csv)

    print(f"Saved metrics JSON: {output_json}")
    print(f"Saved metrics CSV:  {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
