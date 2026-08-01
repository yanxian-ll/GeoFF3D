# -*- coding: utf-8 -*-
"""Camera pose perturbation for GNSS/GPS prior simulation."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def _clip_xy(delta_xy: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0.0:
        return delta_xy
    norm = float(np.linalg.norm(delta_xy))
    if norm <= max_norm or norm <= 1e-12:
        return delta_xy
    return delta_xy * (float(max_norm) / norm)


def _clip_abs(value: float, max_abs: float) -> float:
    if max_abs <= 0.0:
        return float(value)
    return float(np.clip(value, -float(max_abs), float(max_abs)))


def yaw_rotation_matrix(degrees: float) -> np.ndarray:
    rad = math.radians(float(degrees))
    c = math.cos(rad)
    s = math.sin(rad)
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def sample_frame_pose_noise(
    *,
    frame_index: int,
    seed: int,
    xy_std: float,
    z_std: float,
    yaw_std_deg: float,
    xy_max: float = 0.0,
    z_max: float = 0.0,
    yaw_max_deg: float = 0.0,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(int(seed) + 1000003 * (int(frame_index) + 1))
    xy = (
        rng.normal(0.0, float(xy_std), size=2).astype(np.float64)
        if float(xy_std) > 0.0
        else np.zeros(2, dtype=np.float64)
    )
    xy = _clip_xy(xy, float(xy_max))
    z = (
        float(rng.normal(0.0, float(z_std)))
        if float(z_std) > 0.0
        else 0.0
    )
    z = _clip_abs(z, float(z_max))
    yaw_deg = (
        float(rng.normal(0.0, float(yaw_std_deg)))
        if float(yaw_std_deg) > 0.0
        else 0.0
    )
    yaw_deg = _clip_abs(yaw_deg, float(yaw_max_deg))
    return np.asarray([xy[0], xy[1], z], dtype=np.float64), yaw_deg


def apply_pose_noise_to_matrix(
    T_c2w: np.ndarray,
    translation: np.ndarray,
    yaw_deg: float,
) -> np.ndarray:
    T = np.asarray(T_c2w, dtype=np.float64)
    R_noise = yaw_rotation_matrix(float(yaw_deg))
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = (R_noise @ T[:3, :3]).astype(np.float32)
    out[:3, 3] = (R_noise @ T[:3, 3] + np.asarray(translation, dtype=np.float64)).astype(
        np.float32
    )
    return out


def _clone_camera_map(cams: object) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    if not isinstance(cams, dict):
        return out
    for stem, cam in cams.items():
        if not isinstance(cam, dict):
            continue
        next_cam: Dict[str, object] = dict(cam)
        for key in ("K", "T_w2c", "T_c2w"):
            value = next_cam.get(key)
            if value is not None:
                next_cam[key] = np.asarray(value).copy()
        out[str(stem)] = next_cam
    return out


def perturb_scene_camera_poses(
    meta: Dict[str, object],
    *,
    enabled: bool,
    seed: int,
    xy_std: float,
    z_std: float,
    yaw_std_deg: float,
    xy_max: float = 0.0,
    z_max: float = 0.0,
    yaw_max_deg: float = 0.0,
) -> Dict[str, object]:
    """Perturb meta['cams'] immediately after scene manifest loading.

    The original cameras are preserved in meta['gt_cams'] for GT point loading
    and metrics. Downstream reconstruction code continues to read meta['cams'],
    so chunking, priors, chunk alignment, post-alignment, DOM, and RRD output all
    use the same noisy GNSS-like camera priors.
    """
    cams = meta.get("cams", {})
    summary: Dict[str, object] = {
        "enabled": bool(enabled),
        "scope": "frame",
        "seed": int(seed),
        "xy_std": float(xy_std),
        "z_std": float(z_std),
        "yaw_std_deg": float(yaw_std_deg),
        "xy_max": float(xy_max),
        "z_max": float(z_max),
        "yaw_max_deg": float(yaw_max_deg),
        "num_cameras": int(len(cams)) if isinstance(cams, dict) else 0,
        "num_perturbed_cameras": 0,
        "samples": [],
    }
    meta["pose_perturb"] = summary

    if not enabled or not isinstance(cams, dict):
        return summary

    if "gt_cams" not in meta:
        meta["gt_cams"] = _clone_camera_map(cams)

    stems = [str(s) for s in meta.get("stems", [])]
    for frame_index, stem in enumerate(stems):
        cam = cams.get(stem)
        if not isinstance(cam, dict):
            continue
        T = np.asarray(cam.get("T_c2w", None), dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            continue
        translation, yaw_deg = sample_frame_pose_noise(
            frame_index=frame_index,
            seed=int(seed),
            xy_std=float(xy_std),
            z_std=float(z_std),
            yaw_std_deg=float(yaw_std_deg),
            xy_max=float(xy_max),
            z_max=float(z_max),
            yaw_max_deg=float(yaw_max_deg),
        )
        T_perturbed = apply_pose_noise_to_matrix(T, translation, yaw_deg)
        next_cam = dict(cam)
        next_cam["T_c2w"] = T_perturbed.astype(np.float64)
        next_cam["T_w2c"] = np.linalg.inv(next_cam["T_c2w"])
        cams[stem] = next_cam
        summary["num_perturbed_cameras"] = int(summary["num_perturbed_cameras"]) + 1
        if len(summary["samples"]) < 20:
            summary["samples"].append(
                {
                    "stem": stem,
                    "translation": translation.astype(float).tolist(),
                    "yaw_deg": float(yaw_deg),
                }
            )

    return summary
