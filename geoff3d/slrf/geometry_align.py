# -*- coding: utf-8 -*-
"""Alignment, recenter, and pose-translation alignment for spatial chunks."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from geoff3d.slrf.scene_io import (
    estimate_scale_from_random_baselines,
    sample_alignment_correspondences,
)

# ---------------------------------------------------------------------------
# Align mode registry
# ---------------------------------------------------------------------------
ALIGN_MODES = {
    "none",
    "scale",
    "translation",
    "scale_translation",
    "scale_yaw_translation",
    "yaw_translation",
    "sim3",
}

def _restore_points_inplace(
    points: np.ndarray,
    anchor: np.ndarray,
) -> np.ndarray:
    if np.linalg.norm(anchor) < 1e-9:
        return np.asarray(points, dtype=np.float32)
    return (np.asarray(points, dtype=np.float64) + anchor).astype(np.float32)


def _restore_maps_inplace(
    point_maps: Sequence[np.ndarray],
    anchor: np.ndarray,
) -> List[np.ndarray]:
    if np.linalg.norm(anchor) < 1e-9:
        return list(point_maps)
    out: List[np.ndarray] = []
    for m in point_maps:
        shape = m.shape
        if m.size == 0:
            out.append(m)
            continue
        out.append(
            _restore_points_inplace(
                m.reshape(-1, 3), anchor
            ).reshape(shape)
        )
    return out


def _restore_cameras_inplace(
    cams: Sequence[Dict[str, object]],
    anchor: np.ndarray,
) -> List[Dict[str, object]]:
    if np.linalg.norm(anchor) < 1e-9:
        return list(cams)
    anchor = np.asarray(anchor, dtype=np.float64)
    out: List[Dict[str, object]] = []
    for cam in cams:
        T = np.asarray(cam["T_c2w"], dtype=np.float64)
        T_out = np.eye(4, dtype=np.float32)
        T_out[:3, :3] = T[:3, :3].astype(np.float32)
        T_out[:3, 3] = (T[:3, 3] + anchor).astype(np.float32)
        out.append({**cam, "T_c2w": T_out})
    return out


def recenter_anchor_from_meta(meta: Dict[str, object]) -> np.ndarray:
    """Return the mean input-camera center used to recenter every scene."""
    centers: List[np.ndarray] = []
    cams = meta.get("cams", {})
    for stem in meta.get("stems", []):
        cam = cams.get(stem) if isinstance(cams, dict) else None
        if cam is None:
            continue
        T = np.asarray(cam.get("T_c2w", None), dtype=np.float64)
        if T.shape == (4, 4) and np.isfinite(T[:3, 3]).all():
            centers.append(T[:3, 3].astype(np.float64))

    if not centers:
        raise ValueError("Cannot recenter scene: no valid input camera centers found.")

    anchor = np.stack(centers, axis=0).mean(axis=0)
    if not np.isfinite(anchor).all():
        raise ValueError("Cannot recenter scene: mean camera center is not finite.")
    return np.asarray(anchor, dtype=np.float64)


def restore_predictions_from_recenter(
    points: np.ndarray,
    colors: np.ndarray,
    pred_maps: Sequence[np.ndarray],
    pred_cams: Sequence[Dict[str, object]],
    anchor: Optional[np.ndarray],
) -> Tuple[
    np.ndarray, np.ndarray, List[np.ndarray], List[Dict[str, object]]
]:
    if anchor is None:
        return points, colors, list(pred_maps), list(pred_cams)
    anchor = np.asarray(anchor, dtype=np.float64)
    return (
        _restore_points_inplace(points, anchor),
        colors,
        _restore_maps_inplace(pred_maps, anchor),
        _restore_cameras_inplace(pred_cams, anchor),
    )


# ---------------------------------------------------------------------------
# Camera centers helpers
# ---------------------------------------------------------------------------
def camera_centers_by_stem(
    cams: Sequence[Dict[str, object]],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for cam in cams:
        stem = str(cam.get("stem", ""))
        if not stem:
            continue
        T = np.asarray(cam.get("T_c2w"), dtype=np.float64)
        if T.shape != (4, 4):
            continue
        center = T[:3, 3]
        if np.isfinite(center).all():
            out[stem] = center.astype(np.float64)
    return out


def matched_pose_translation_correspondences(
    reference_cams_by_stem: Dict[str, np.ndarray],
    current_cams: Sequence[Dict[str, object]],
    target_stems: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    current_by_stem = camera_centers_by_stem(current_cams)
    ref_corr: List[np.ndarray] = []
    cur_corr: List[np.ndarray] = []
    matched_stems: List[str] = []
    for stem in target_stems:
        stem = str(stem)
        ref_center = reference_cams_by_stem.get(stem)
        cur_center = current_by_stem.get(stem)
        if ref_center is None or cur_center is None:
            continue
        ref_corr.append(ref_center)
        cur_corr.append(cur_center)
        matched_stems.append(stem)

    if not ref_corr:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            [],
        )
    return (
        np.asarray(ref_corr, dtype=np.float32).reshape(-1, 3),
        np.asarray(cur_corr, dtype=np.float32).reshape(-1, 3),
        matched_stems,
    )


# ---------------------------------------------------------------------------
# Geometry transforms
# ---------------------------------------------------------------------------
def yaw_rotation_from_xy_correspondences(
    src_xy: np.ndarray, dst_xy: np.ndarray
) -> Tuple[float, bool, str]:
    src = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
    if src.shape[0] < 2 or dst.shape[0] != src.shape[0]:
        return 0.0, False, "not enough correspondences to estimate yaw"

    src_centered = src - np.mean(src, axis=0, keepdims=True)
    dst_centered = dst - np.mean(dst, axis=0, keepdims=True)
    if (
        np.linalg.norm(src_centered) < 1e-8
        or np.linalg.norm(dst_centered) < 1e-8
    ):
        return (
            0.0,
            False,
            "not enough non-zero XY baselines to estimate yaw",
        )

    cross = float(
        np.sum(
            src_centered[:, 0] * dst_centered[:, 1]
            - src_centered[:, 1] * dst_centered[:, 0]
        )
    )
    dot = float(
        np.sum(
            src_centered[:, 0] * dst_centered[:, 0]
            + src_centered[:, 1] * dst_centered[:, 1]
        )
    )
    yaw = float(np.arctan2(cross, dot))
    if not np.isfinite(yaw):
        return 0.0, False, "yaw solve produced non-finite angle"
    return yaw, True, "yaw estimated from XY pose baselines"


def rotation_matrix_z(yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_scale_yaw_translation_from_correspondences(
    src: np.ndarray,
    dst: np.ndarray,
    seed: int,
) -> Tuple[float, np.ndarray, np.ndarray, bool, Dict[str, object]]:
    """Estimate the spatial-pipeline scale + Z-yaw + translation transform."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[finite]
    dst = dst[finite]
    if src.shape[0] == 0:
        return 1.0, np.eye(3), np.zeros(3), False, {
            "num_corr": int(src.shape[0]),
            "num_scale_pairs": 0,
            "yaw_degrees": 0.0,
            "yaw_valid": False,
            "note": "not enough correspondences",
        }
    if src.shape[0] == 1:
        # With one correspondence, scale and yaw are unobservable, but a 3D
        # translation is still fully determined.
        t = dst[0] - src[0]
        valid = bool(np.isfinite(t).all())
        return 1.0, np.eye(3, dtype=np.float64), t, valid, {
            "num_corr": 1,
            "num_scale_pairs": 0,
            "yaw_degrees": 0.0,
            "yaw_valid": False,
            "note": "single correspondence: translation-only fallback",
        }

    scale, num_scale_pairs, scale_valid = estimate_scale_from_random_baselines(
        pr_corr=src, gt_corr=dst, seed=seed
    )
    if not scale_valid:
        # Preserve the spatial pipeline's established fallback: yaw and
        # translation are still estimated with unit scale.
        scale = 1.0

    scaled_src = float(scale) * src
    yaw, yaw_valid, yaw_note = yaw_rotation_from_xy_correspondences(
        src_xy=scaled_src[:, :2], dst_xy=dst[:, :2]
    )
    R = rotation_matrix_z(yaw)
    transformed = scaled_src @ R.T
    t = np.median(dst - transformed, axis=0)
    valid = bool(
        np.isfinite(scale)
        and scale > 1e-12
        and np.isfinite(R).all()
        and np.isfinite(t).all()
    )
    return float(scale), R, t, valid, {
        "num_corr": int(src.shape[0]),
        "num_scale_pairs": int(num_scale_pairs),
        "yaw_degrees": float(np.degrees(yaw)),
        "yaw_valid": bool(yaw_valid),
        "note": (
            yaw_note if scale_valid else f"scale fallback to 1.0; {yaw_note}"
        ),
    }


def estimate_similarity_umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    estimate_scale: bool = True,
    eps: float = 1e-12,
) -> Tuple[float, np.ndarray, np.ndarray, bool, str]:
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] < 3 or dst.shape != src.shape:
        return (
            1.0,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            False,
            "not enough 3D correspondences to estimate full similarity",
        )

    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[finite]
    dst = dst[finite]
    if src.shape[0] < 3:
        return (
            1.0,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            False,
            "not enough finite 3D correspondences to estimate full similarity",
        )

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst
    src_var = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if src_var <= eps:
        return (
            1.0,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            False,
            "source pose translations are degenerate",
        )

    cov = (dst_centered.T @ src_centered) / float(src.shape[0])
    try:
        U, singular_values, Vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return (
            1.0,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            False,
            "SVD failed",
        )

    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if estimate_scale:
        scale = float(np.sum(singular_values * np.diag(S)) / src_var)
    else:
        scale = 1.0
    t = mu_dst - scale * (R @ mu_src)

    if (
        not np.isfinite(scale)
        or scale <= eps
        or not np.isfinite(R).all()
        or not np.isfinite(t).all()
    ):
        return (
            1.0,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            False,
            "full similarity solve produced non-finite transform",
        )
    return (
        scale,
        R,
        t,
        True,
        "scale+rotation+translation estimated from 3D pose correspondences",
    )


def apply_similarity_to_points(
    points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts
    out = (float(scale) * pts.astype(np.float64)) @ np.asarray(
        R, dtype=np.float64
    ).T
    out += np.asarray(t, dtype=np.float64)[None, :]
    return out.astype(np.float32)


def apply_similarity_to_point_maps(
    point_maps: Sequence[np.ndarray],
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> List[np.ndarray]:
    out = []
    for point_map in point_maps:
        shape = point_map.shape
        if point_map.size == 0:
            out.append(point_map)
            continue
        out.append(
            apply_similarity_to_points(
                point_map.reshape(-1, 3), scale, R, t
            ).reshape(shape)
        )
    return out


def apply_similarity_to_cameras(
    cams: Sequence[Dict[str, object]],
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> List[Dict[str, object]]:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    aligned = []
    for cam in cams:
        T = np.asarray(cam["T_c2w"], dtype=np.float64)
        T_al = np.eye(4, dtype=np.float32)
        T_al[:3, :3] = (R @ T[:3, :3]).astype(np.float32)
        T_al[:3, 3] = (R @ (float(scale) * T[:3, 3]) + t).astype(np.float32)
        aligned.append({**cam, "T_c2w": T_al})
    return aligned


def identity_pose_alignment_meta(
    mode: str, chunk_id: int, note: str, valid: bool
) -> Dict[str, object]:
    return {
        "mode": mode,
        "valid": bool(valid),
        "source": "pose_translations",
        "num_corr": 0,
        "num_scale_pairs": 0,
        "matched_camera_stems": [],
        "scale": 1.0,
        "yaw_degrees": 0.0,
        "yaw_valid": False,
        "R": np.eye(3, dtype=np.float32).tolist(),
        "t": np.zeros(3, dtype=np.float32).tolist(),
        "median_residual": float("nan"),
        "note": note,
        "chunk_id": int(chunk_id),
    }


def estimate_chunk_pose_alignment(
    mode: str,
    chunk_id: int,
    reference_cams_by_stem: Dict[str, np.ndarray],
    raw_pred_cams: Sequence[Dict[str, object]],
    target_stems: Sequence[str],
    seed: int,
) -> Dict[str, object]:
    if mode == "none":
        return identity_pose_alignment_meta(
            mode,
            chunk_id,
            "alignment disabled",
            valid=True,
        )

    ref_corr, cur_corr, matched_stems = (
        matched_pose_translation_correspondences(
            reference_cams_by_stem=reference_cams_by_stem,
            current_cams=raw_pred_cams,
            target_stems=target_stems,
        )
    )
    if ref_corr.shape[0] == 0:
        return identity_pose_alignment_meta(
            mode,
            chunk_id,
            "no predicted pose translations matched input pose translations "
            "for this chunk; using raw chunk coordinates",
            valid=False,
        )

    if mode == "translation":
        t = np.median(
            ref_corr.astype(np.float64) - cur_corr.astype(np.float64),
            axis=0,
        )
        R = np.eye(3, dtype=np.float64)
        scale = 1.0
        yaw = 0.0
        yaw_valid = False

        transformed_corr = apply_similarity_to_points(
            cur_corr, scale, R, t
        )
        residual = np.linalg.norm(
            transformed_corr.astype(np.float64)
            - ref_corr.astype(np.float64),
            axis=1,
        )
        median_residual = (
            float(np.median(residual)) if residual.size else float("nan")
        )

        return {
            "mode": mode,
            "valid": True,
            "source": "pose_translations",
            "num_corr": int(ref_corr.shape[0]),
            "num_scale_pairs": 0,
            "matched_camera_stems": matched_stems,
            "scale": 1.0,
            "yaw_degrees": 0.0,
            "yaw_valid": False,
            "R": R.astype(np.float32).tolist(),
            "t": t.astype(np.float32).tolist(),
            "median_residual": median_residual,
            "note": "translation estimated from matched pose centers; no scale or rotation applied",
            "chunk_id": int(chunk_id),
        }

    scale, num_scale_pairs, scale_valid = (
        estimate_scale_from_random_baselines(
            pr_corr=cur_corr,
            gt_corr=ref_corr,
            seed=seed,
        )
    )
    scale_note = (
        "scale estimated from matched pose-translation baselines"
    )
    if not scale_valid:
        scale = 1.0
        scale_note = (
            "scale fallback to 1.0; not enough non-zero pose-translation baselines"
        )

    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    yaw = 0.0
    yaw_valid = False
    yaw_note = "yaw not requested"
    if mode == "sim3":
        scale, R, t, sim3_valid, sim3_note = estimate_similarity_umeyama(
            src=cur_corr,
            dst=ref_corr,
            estimate_scale=True,
        )
        if not sim3_valid:
            return identity_pose_alignment_meta(
                mode,
                chunk_id,
                f"{sim3_note}; using raw chunk coordinates",
                valid=False,
            )
        scale_note = sim3_note
        yaw_note = "full 3D rotation estimated"
    elif mode == "scale_yaw_translation":
        scale, R, t, constrained_valid, constrained_meta = (
            estimate_scale_yaw_translation_from_correspondences(
                src=cur_corr,
                dst=ref_corr,
                seed=seed,
            )
        )
        if not constrained_valid:
            return identity_pose_alignment_meta(
                mode,
                chunk_id,
                f"{constrained_meta['note']}; using raw chunk coordinates",
                valid=False,
            )
        yaw = np.radians(float(constrained_meta["yaw_degrees"]))
        yaw_valid = bool(constrained_meta["yaw_valid"])
        yaw_note = str(constrained_meta["note"])
        num_scale_pairs = int(constrained_meta["num_scale_pairs"])
    elif mode in {"scale_translation", "yaw_translation"}:
        if mode == "yaw_translation":
            scale = 1.0
            num_scale_pairs = 0
            yaw, yaw_valid, yaw_note = yaw_rotation_from_xy_correspondences(
                src_xy=cur_corr[:, :2],
                dst_xy=ref_corr[:, :2],
            )
            R = rotation_matrix_z(yaw)
            scale_note = "scale fixed to 1.0"
        else:
            yaw_note = "yaw not requested"
        transformed = (float(scale) * cur_corr.astype(np.float64)) @ R.T
        t = np.median(ref_corr.astype(np.float64) - transformed, axis=0)

    if (
        not np.isfinite(scale)
        or scale <= 1e-12
        or not np.isfinite(R).all()
        or not np.isfinite(t).all()
    ):
        return identity_pose_alignment_meta(
            mode,
            chunk_id,
            "pose alignment solve produced non-finite transform; "
            "using raw chunk coordinates",
            valid=False,
        )

    transformed_corr = apply_similarity_to_points(
        cur_corr, scale, R, t
    )
    residual = np.linalg.norm(
        transformed_corr.astype(np.float64)
        - ref_corr.astype(np.float64),
        axis=1,
    )
    median_residual = (
        float(np.median(residual)) if residual.size else float("nan")
    )
    note = f"{scale_note}; {yaw_note}"
    if mode == "scale":
        note += "; no rotation or translation applied"
    elif mode == "sim3":
        note += "; full 3D rotation and translation applied"
    elif mode == "scale_translation":
        note += "; translation estimated after scale"
    elif mode == "yaw_translation":
        note += "; translation estimated after yaw"
    else:
        note += "; translation estimated after scale+yaw"

    return {
        "mode": mode,
        "valid": True,
        "source": "pose_translations",
        "num_corr": int(ref_corr.shape[0]),
        "num_scale_pairs": int(num_scale_pairs),
        "matched_camera_stems": matched_stems,
        "scale": float(scale),
        "yaw_degrees": float(np.degrees(yaw)),
        "yaw_valid": bool(yaw_valid),
        "R": R.astype(np.float32).tolist(),
        "t": t.astype(np.float32).tolist(),
        "median_residual": median_residual,
        "note": note,
        "chunk_id": int(chunk_id),
    }


def apply_chunk_pose_alignment(
    mode: str,
    chunk_id: int,
    reference_cams_by_stem: Dict[str, np.ndarray],
    raw_pred_points: np.ndarray,
    raw_pred_colors: np.ndarray,
    pred_maps: Sequence[np.ndarray],
    raw_pred_cams: Sequence[Dict[str, object]],
    target_stems: Sequence[str],
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[np.ndarray],
    List[Dict[str, object]],
    Dict[str, object],
]:
    align_meta = estimate_chunk_pose_alignment(
        mode=mode,
        chunk_id=chunk_id,
        reference_cams_by_stem=reference_cams_by_stem,
        raw_pred_cams=raw_pred_cams,
        target_stems=target_stems,
        seed=seed,
    )
    scale = float(align_meta["scale"])
    R = np.asarray(align_meta["R"], dtype=np.float64)
    t = np.asarray(align_meta["t"], dtype=np.float64)
    pred_points_aligned = apply_similarity_to_points(
        raw_pred_points, scale, R, t
    )
    pred_maps_aligned = apply_similarity_to_point_maps(
        pred_maps, scale, R, t
    )
    pred_cams_aligned = apply_similarity_to_cameras(
        raw_pred_cams, scale, R, t
    )
    return (
        pred_points_aligned,
        raw_pred_colors,
        pred_maps_aligned,
        pred_cams_aligned,
        align_meta,
    )
