# -*- coding: utf-8 -*-
"""Aligned pose and fused point-cloud metrics for spatial reconstruction."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from geoff3d.slrf.geometry_align import (
    apply_similarity_to_cameras,
    apply_similarity_to_points,
    estimate_similarity_umeyama,
)
from geoff3d.slrf.rrd_writer import load_point_cloud_ply, save_point_cloud_ply
from geoff3d.slrf.scene_io import load_gt_points_from_meta, sample_points_and_colors


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
    out: List[float] = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


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


def rotation_angle_deg(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = (float(np.trace(R)) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def load_eval_outputs(eval_dir: Path) -> Dict[str, object]:
    eval_dir = Path(eval_dir)
    with np.load(eval_dir / "pred_cameras.npz", allow_pickle=True) as data:
        stems = [str(x) for x in np.asarray(data["stems"]).tolist()]
        T_c2w = np.asarray(data["T_c2w"], dtype=np.float64)
        valid = np.asarray(data["valid"], dtype=bool)
    ply_path = eval_dir / "pred_points.ply"
    npz_path = eval_dir / "pred_points.npz"
    if ply_path.exists():
        points, colors = load_point_cloud_ply(ply_path)
    else:
        with np.load(npz_path, allow_pickle=True) as data:
            points = np.asarray(data["points"], dtype=np.float32).reshape(-1, 3)
            colors = (
                np.asarray(data["colors"], dtype=np.uint8).reshape(-1, 3)
                if "colors" in data
                else np.empty((0, 3), dtype=np.uint8)
            )
    meta = {}
    meta_path = eval_dir / "meta.json"
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


def gt_cameras_for_stems(
    meta: Dict[str, object],
    stems: Sequence[str],
) -> List[Dict[str, object]]:
    cams = meta.get("gt_cams", meta.get("cams", {}))
    out: List[Dict[str, object]] = []
    if not isinstance(cams, dict):
        return out
    for stem in stems:
        cam = cams.get(str(stem))
        if cam is None:
            continue
        T = np.asarray(cam.get("T_c2w"), dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            continue
        out.append({"stem": str(stem), "T_c2w": T})
    return out


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
    stems: List[str] = []
    pred_list: List[np.ndarray] = []
    gt_list: List[np.ndarray] = []
    for i, stem in enumerate(pred_stems):
        if i >= len(pred_valid) or not bool(pred_valid[i]):
            continue
        if stem not in gt_by_stem:
            continue
        T_pred = np.asarray(pred_T[i], dtype=np.float64)
        T_gt = gt_by_stem[stem]
        if T_pred.shape != (4, 4) or not np.isfinite(T_pred).all():
            continue
        stems.append(str(stem))
        pred_list.append(T_pred)
        gt_list.append(T_gt)
    if not pred_list:
        return stems, np.empty((0, 4, 4), np.float64), np.empty((0, 4, 4), np.float64)
    return stems, np.stack(pred_list, axis=0), np.stack(gt_list, axis=0)


def _transform_meta(
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
    valid: bool,
    note: str,
    num_corr: int,
) -> Dict[str, object]:
    return {
        "valid": bool(valid),
        "num_corr": int(num_corr),
        "scale": float(scale),
        "R": np.asarray(R, dtype=np.float32),
        "t": np.asarray(t, dtype=np.float32),
        "note": str(note),
    }


def evaluate_pose_aligned(
    pred_stems: Sequence[str],
    pred_T: np.ndarray,
    pred_valid: np.ndarray,
    gt_cams: Sequence[Dict[str, object]],
    rpe_steps: Sequence[int],
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    matched_stems, T_pred, T_gt = match_cameras(pred_stems, pred_T, pred_valid, gt_cams)
    result: Dict[str, object] = {
        "valid": bool(T_pred.shape[0] >= 3),
        "num_pred_cameras": int(len(pred_stems)),
        "num_gt_cameras": int(len(gt_cams)),
        "num_matches": int(T_pred.shape[0]),
        "matched_stems": matched_stems,
    }
    if T_pred.shape[0] < 3:
        result["note"] = "Need at least 3 matched cameras for pose alignment."
        return result

    pred_centers = T_pred[:, :3, 3]
    gt_centers = T_gt[:, :3, 3]
    scale, R, t, valid, note = estimate_similarity_umeyama(
        pred_centers,
        gt_centers,
        estimate_scale=True,
    )
    result["alignment"] = _transform_meta(scale, R, t, valid, note, T_pred.shape[0])
    if not valid:
        result["valid"] = False
        return result

    pred_cams = [
        {"stem": stem, "T_c2w": T}
        for stem, T in zip(matched_stems, T_pred)
    ]
    aligned_cams = apply_similarity_to_cameras(pred_cams, scale, R, t)
    T_aligned = np.stack(
        [np.asarray(cam["T_c2w"], dtype=np.float64) for cam in aligned_cams],
        axis=0,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_dir / "aligned_pred_cameras.npz",
            stems=np.asarray(matched_stems, dtype=str),
            T_c2w=T_aligned.astype(np.float32),
            gt_T_c2w=T_gt.astype(np.float32),
            alignment_scale=np.asarray(scale, dtype=np.float32),
            alignment_R=np.asarray(R, dtype=np.float32),
            alignment_t=np.asarray(t, dtype=np.float32),
        )

    center_err = np.linalg.norm(T_aligned[:, :3, 3] - gt_centers, axis=1)
    rot_err = []
    for i in range(T_aligned.shape[0]):
        R_err = T_gt[i, :3, :3].T @ T_aligned[i, :3, :3]
        rot_err.append(rotation_angle_deg(R_err))
    result["ate"] = summarize(center_err, "ate")
    result["rotation"] = summarize(np.asarray(rot_err), "rot_deg")

    rpe: Dict[str, object] = {}
    for step in rpe_steps:
        step = int(step)
        if step <= 0 or T_aligned.shape[0] <= step:
            continue
        trans_err: List[float] = []
        rot_step_err: List[float] = []
        for i in range(0, T_aligned.shape[0] - step):
            j = i + step
            rel_gt = np.linalg.inv(T_gt[i]) @ T_gt[j]
            rel_pred = np.linalg.inv(T_aligned[i]) @ T_aligned[j]
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


def nn_distances(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float32).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float32).reshape(-1, 3)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(dst)
        try:
            dists, idx = tree.query(src, k=1, workers=-1)
        except TypeError:
            dists, idx = tree.query(src, k=1)
        return np.asarray(dists, dtype=np.float32), np.asarray(idx, dtype=np.int64)
    except Exception as exc:
        max_pairs = 50_000_000
        if src.shape[0] * dst.shape[0] > max_pairs:
            raise RuntimeError(
                "scipy.spatial.cKDTree is required for large point-cloud metrics."
            ) from exc
        out_d = np.empty((src.shape[0],), dtype=np.float32)
        out_i = np.empty((src.shape[0],), dtype=np.int64)
        dst64 = dst.astype(np.float64)
        block = 4096
        for start in range(0, src.shape[0], block):
            part = src[start : start + block].astype(np.float64)
            d2 = np.sum((part[:, None, :] - dst64[None, :, :]) ** 2, axis=-1)
            idx = np.argmin(d2, axis=1)
            out_i[start : start + idx.shape[0]] = idx
            out_d[start : start + idx.shape[0]] = np.sqrt(d2[np.arange(idx.shape[0]), idx])
        return out_d, out_i


def _finite_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def align_point_cloud_icp(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    max_align_points: int,
    iterations: int,
    trim_quantile: float,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    pred = sample_points_and_colors(
        _finite_points(pred_points),
        np.zeros((_finite_points(pred_points).shape[0], 3), dtype=np.uint8),
        max_points=int(max_align_points),
        seed=int(seed) + 11,
    )[0]
    gt = sample_points_and_colors(
        _finite_points(gt_points),
        np.zeros((_finite_points(gt_points).shape[0], 3), dtype=np.uint8),
        max_points=int(max_align_points),
        seed=int(seed) + 22,
    )[0]
    if pred.shape[0] < 3 or gt.shape[0] < 3:
        return _finite_points(pred_points), _transform_meta(
            1.0, np.eye(3), np.zeros(3), False, "not enough points for ICP", 0
        )

    aligned_sample = pred.copy()
    total_scale = 1.0
    total_R = np.eye(3, dtype=np.float64)
    total_t = np.zeros(3, dtype=np.float64)
    last_corr = 0
    last_median = float("nan")
    trim = float(max(0.05, min(1.0, trim_quantile)))

    for _iter in range(max(1, int(iterations))):
        dists, nn = nn_distances(aligned_sample, gt)
        if dists.size == 0:
            break
        threshold = float(np.quantile(dists[np.isfinite(dists)], trim))
        keep = np.isfinite(dists) & (dists <= threshold)
        if int(np.count_nonzero(keep)) < 3:
            break
        src = aligned_sample[keep]
        dst = gt[nn[keep]]
        scale, R, t, valid, _note = estimate_similarity_umeyama(
            src,
            dst,
            estimate_scale=True,
        )
        if not valid:
            break
        aligned_sample = apply_similarity_to_points(aligned_sample, scale, R, t)
        total_t = np.asarray(R, dtype=np.float64) @ (float(scale) * total_t) + t
        total_R = np.asarray(R, dtype=np.float64) @ total_R
        total_scale = float(scale) * float(total_scale)
        last_corr = int(np.count_nonzero(keep))
        last_median = float(np.median(dists[keep]))

    aligned_full = apply_similarity_to_points(
        _finite_points(pred_points),
        total_scale,
        total_R,
        total_t,
    )
    return aligned_full, {
        **_transform_meta(
            total_scale,
            total_R,
            total_t,
            last_corr >= 3,
            "trimmed nearest-neighbor ICP Sim3 alignment",
            last_corr,
        ),
        "iterations": int(iterations),
        "trim_quantile": float(trim),
        "last_median_nn_distance": float(last_median),
        "num_alignment_pred_points": int(pred.shape[0]),
        "num_alignment_gt_points": int(gt.shape[0]),
    }


def evaluate_point_cloud_aligned(
    pred_points: np.ndarray,
    pred_colors: np.ndarray,
    gt_points: np.ndarray,
    thresholds: Sequence[float],
    max_pred_points_eval: int,
    max_gt_points_eval: int,
    max_align_points: int,
    icp_iterations: int,
    icp_trim_quantile: float,
    seed: int,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    pred_points = _finite_points(pred_points)
    gt_points = _finite_points(gt_points)
    result: Dict[str, object] = {
        "valid": bool(pred_points.shape[0] > 0 and gt_points.shape[0] > 0),
        "num_pred_points_total": int(pred_points.shape[0]),
        "num_gt_points_total": int(gt_points.shape[0]),
        "thresholds": [float(t) for t in thresholds],
    }
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        result["valid"] = False
        result["note"] = "Missing predicted points or GT point cloud; skip point metrics."
        return result

    aligned_points, align_meta = align_point_cloud_icp(
        pred_points,
        gt_points,
        max_align_points=max_align_points,
        iterations=icp_iterations,
        trim_quantile=icp_trim_quantile,
        seed=seed,
    )
    result["alignment"] = align_meta
    if not bool(align_meta.get("valid", False)):
        result["valid"] = False
        result["note"] = "Point cloud alignment failed; skip point distance metrics."
        return result

    eval_pred = sample_points_and_colors(
        aligned_points,
        np.zeros((aligned_points.shape[0], 3), dtype=np.uint8),
        max_points=int(max_pred_points_eval),
        seed=int(seed) + 101,
    )[0]
    eval_gt = sample_points_and_colors(
        gt_points,
        np.zeros((gt_points.shape[0], 3), dtype=np.uint8),
        max_points=int(max_gt_points_eval),
        seed=int(seed) + 202,
    )[0]

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        colors = np.asarray(pred_colors, dtype=np.uint8).reshape(-1, 3)
        if colors.shape[0] != aligned_points.shape[0]:
            colors = np.full((aligned_points.shape[0], 3), 220, dtype=np.uint8)
        save_point_cloud_ply(
            output_dir / "aligned_pred_points.ply",
            points=aligned_points.astype(np.float32),
            colors=colors.astype(np.uint8),
        )

    d_pred_to_gt, _ = nn_distances(eval_pred, eval_gt)
    d_gt_to_pred, _ = nn_distances(eval_gt, eval_pred)
    result.update(
        {
            "num_pred_points_eval": int(eval_pred.shape[0]),
            "num_gt_points_eval": int(eval_gt.shape[0]),
            "accuracy_pred_to_gt": summarize(d_pred_to_gt, "acc"),
            "completeness_gt_to_pred": summarize(d_gt_to_pred, "comp"),
            "chamfer_l1": float(np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred)),
            "chamfer_l2": float(
                np.mean(d_pred_to_gt * d_pred_to_gt)
                + np.mean(d_gt_to_pred * d_gt_to_pred)
            ),
        }
    )

    fscores: Dict[str, object] = {}
    for threshold in thresholds:
        t = float(threshold)
        precision = float(np.mean(d_pred_to_gt <= t))
        recall = float(np.mean(d_gt_to_pred <= t))
        fscore = 0.0 if precision + recall <= 0 else float(
            2.0 * precision * recall / (precision + recall)
        )
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


def write_metrics_csv(metrics: Dict[str, object], path: Path) -> None:
    pose = metrics.get("pose", {})
    points = metrics.get("point_cloud", {})
    row = {
        "scene": metrics.get("scene_name", ""),
        "pose_valid": nested_get(pose, ["valid"], False),
        "pose_matches": nested_get(pose, ["num_matches"], 0),
        "pose_align_scale": nested_get(pose, ["alignment", "scale"]),
        "ate_rmse": nested_get(pose, ["ate", "ate_rmse"]),
        "ate_median": nested_get(pose, ["ate", "ate_median"]),
        "ate_p90": nested_get(pose, ["ate", "ate_p90"]),
        "rot_deg_rmse": nested_get(pose, ["rotation", "rot_deg_rmse"]),
        "rot_deg_median": nested_get(pose, ["rotation", "rot_deg_median"]),
        "rpe_k1_trans_rmse": nested_get(pose, ["rpe", "k1", "translation", "rpe_trans_rmse"]),
        "rpe_k1_rot_rmse": nested_get(pose, ["rpe", "k1", "rotation", "rpe_rot_deg_rmse"]),
        "points_valid": nested_get(points, ["valid"], False),
        "points_align_scale": nested_get(points, ["alignment", "scale"]),
        "num_pred_points_eval": nested_get(points, ["num_pred_points_eval"], 0),
        "num_gt_points_eval": nested_get(points, ["num_gt_points_eval"], 0),
        "acc_mean": nested_get(points, ["accuracy_pred_to_gt", "acc_mean"]),
        "acc_median": nested_get(points, ["accuracy_pred_to_gt", "acc_median"]),
        "comp_mean": nested_get(points, ["completeness_gt_to_pred", "comp_mean"]),
        "comp_median": nested_get(points, ["completeness_gt_to_pred", "comp_median"]),
        "chamfer_l1": nested_get(points, ["chamfer_l1"]),
        "chamfer_l2": nested_get(points, ["chamfer_l2"]),
    }
    fscores = points.get("fscore", {}) if isinstance(points, dict) else {}
    for key, val in fscores.items():
        if not isinstance(val, dict):
            continue
        suffix = str(key).replace(".", "p")
        row[f"precision@{suffix}"] = val.get("precision", float("nan"))
        row[f"recall@{suffix}"] = val.get("recall", float("nan"))
        row[f"fscore@{suffix}"] = val.get("fscore", float("nan"))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(json_safe(row))


def compute_aligned_metrics(
    eval_dir: Path,
    meta: Dict[str, object],
    *,
    thresholds: Sequence[float],
    rpe_steps: Sequence[int],
    max_pred_points_eval: int,
    max_gt_points_eval: int,
    max_gt_points: int,
    max_align_points: int,
    icp_iterations: int,
    icp_trim_quantile: float,
    seed: int,
    gt_io_workers: int = 0,
    output_json: Optional[Path] = None,
    output_csv: Optional[Path] = None,
) -> Dict[str, object]:
    eval_dir = Path(eval_dir).expanduser().resolve()
    output_json = Path(output_json) if output_json is not None else eval_dir / "aligned_metrics.json"
    output_csv = Path(output_csv) if output_csv is not None else eval_dir / "aligned_metrics_summary.csv"
    aligned_dir = eval_dir / "aligned_metrics"
    pred = load_eval_outputs(eval_dir)

    pred_stems = pred["stems"]
    gt_cams = gt_cameras_for_stems(meta, pred_stems)
    gt_points, _gt_colors = load_gt_points_from_meta(
        meta,
        max_points=int(max_gt_points),
        seed=int(seed),
        num_workers=int(gt_io_workers),
    )

    metrics: Dict[str, object] = {
        "schema": "aligned_spatial_metrics_v1",
        "eval_dir": str(eval_dir),
        "scene_dir": str(meta.get("scene_dir", "")),
        "scene_name": Path(str(meta.get("scene_dir", ""))).name,
        "settings": {
            "thresholds": [float(t) for t in thresholds],
            "rpe_steps": [int(k) for k in rpe_steps],
            "max_pred_points_eval": int(max_pred_points_eval),
            "max_gt_points_eval": int(max_gt_points_eval),
            "max_gt_points": int(max_gt_points),
            "max_align_points": int(max_align_points),
            "icp_iterations": int(icp_iterations),
            "icp_trim_quantile": float(icp_trim_quantile),
        },
        "gt": {
            "num_cameras": int(len(gt_cams)),
            "num_points": int(np.asarray(gt_points).reshape(-1, 3).shape[0])
            if np.asarray(gt_points).size
            else 0,
        },
    }
    metrics["pose"] = evaluate_pose_aligned(
        pred_stems=pred_stems,
        pred_T=pred["T_c2w"],
        pred_valid=pred["valid"],
        gt_cams=gt_cams,
        rpe_steps=rpe_steps,
        output_dir=aligned_dir,
    )
    metrics["point_cloud"] = evaluate_point_cloud_aligned(
        pred_points=pred["points"],
        pred_colors=pred["colors"],
        gt_points=gt_points,
        thresholds=thresholds,
        max_pred_points_eval=int(max_pred_points_eval),
        max_gt_points_eval=int(max_gt_points_eval),
        max_align_points=int(max_align_points),
        icp_iterations=int(icp_iterations),
        icp_trim_quantile=float(icp_trim_quantile),
        seed=int(seed),
        output_dir=aligned_dir,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(json_safe(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_metrics_csv(metrics, output_csv)
    return metrics
