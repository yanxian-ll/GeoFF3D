# -*- coding: utf-8 -*-
"""Optional feature-track bundle adjustment for spatial reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from geoff3d.spatial_rrd.chunk_cache import get_cached_sequence
from geoff3d.spatial_rrd.chunk_transform import (
    get_transformed_cached_point_maps,
    get_transformed_cameras,
)


def _collect_core_views(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    views: List[Dict[str, object]] = []
    seen = set()
    for record in records:
        maps = get_transformed_cached_point_maps(record)
        masks = get_cached_sequence(record, "_pred_valid_masks")
        Ks = get_cached_sequence(record, "_chunk_intrinsics")
        rgbs = get_cached_sequence(record, "rgbs")
        core = [int(v) for v in get_cached_sequence(record, "_core_local_indices")]
        cams = {int(c.get("pred_index", -1)): c for c in get_transformed_cameras(record)}
        stems = list(record.get("stems", []))
        for i in core:
            if not (0 <= i < len(maps) and i < len(masks) and i < len(Ks) and i < len(rgbs) and i in cams):
                continue
            stem = str(stems[i]) if i < len(stems) else str(cams[i].get("stem", ""))
            if not stem or stem in seen:
                continue
            seen.add(stem)
            views.append({
                "stem": stem,
                "global_index": int(cams[i].get("global_index", len(views))),
                "points": np.asarray(maps[i], np.float32),
                "mask": np.asarray(masks[i], bool),
                "K": np.asarray(Ks[i], np.float64),
                "rgb": np.asarray(rgbs[i]),
                "T_c2w": np.asarray(cams[i]["T_c2w"], np.float64),
            })
    views.sort(key=lambda view: int(view["global_index"]))
    return views


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)


def _project(point: np.ndarray, c2w: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, float]:
    w2c = np.linalg.inv(c2w)
    cam = w2c[:3, :3] @ point + w2c[:3, 3]
    if cam[2] <= 1e-8:
        return np.zeros(2), float(cam[2])
    uv = K @ cam
    return uv[:2] / uv[2], float(cam[2])


def run_bundle_adjustment(
    chunk_records: Sequence[Dict[str, object]],
    output_dir: Path,
    *,
    max_keypoints: int = 2048,
    pair_window: int = 2,
    ratio_test: float = 0.8,
    max_reproj_error: float = 8.0,
    refine_intrinsics: bool = False,
) -> Dict[str, object]:
    """Build SIFT tracks, run pycolmap BA, and update final camera poses."""
    try:
        import pycolmap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Bundle adjustment requires pycolmap. Install with: pip install pycolmap"
        ) from exc

    views = _collect_core_views(chunk_records)
    if len(views) < 2:
        raise RuntimeError("Bundle adjustment requires at least two valid core views")

    sift = cv2.SIFT_create(nfeatures=max(128, int(max_keypoints)))
    features = [sift.detectAndCompute(_to_gray(v["rgb"]), None) for v in views]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    tracks: List[Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]] = []

    for i in range(len(views)):
        kp_i, des_i = features[i]
        if des_i is None:
            continue
        for j in range(i + 1, min(len(views), i + 1 + max(1, int(pair_window)))):
            kp_j, des_j = features[j]
            if des_j is None:
                continue
            for pair in matcher.knnMatch(des_i, des_j, k=2):
                if len(pair) != 2 or pair[0].distance >= float(ratio_test) * pair[1].distance:
                    continue
                uv_i = np.asarray(kp_i[pair[0].queryIdx].pt, np.float64)
                uv_j = np.asarray(kp_j[pair[0].trainIdx].pt, np.float64)
                x = int(round(uv_i[0])); y = int(round(uv_i[1]))
                pmap = views[i]["points"]; mask = views[i]["mask"]
                if not (0 <= y < pmap.shape[0] and 0 <= x < pmap.shape[1] and mask[y, x]):
                    continue
                point = np.asarray(pmap[y, x], np.float64)
                if not np.isfinite(point).all():
                    continue
                pred_i, zi = _project(point, views[i]["T_c2w"], views[i]["K"])
                pred_j, zj = _project(point, views[j]["T_c2w"], views[j]["K"])
                if zi <= 0 or zj <= 0:
                    continue
                if np.linalg.norm(pred_i - uv_i) > float(max_reproj_error):
                    continue
                if np.linalg.norm(pred_j - uv_j) > float(max_reproj_error):
                    continue
                tracks.append((point, [(i, uv_i), (j, uv_j)]))

    if len(tracks) < 64:
        raise RuntimeError(f"Bundle adjustment found too few valid tracks: {len(tracks)}")

    reconstruction = pycolmap.Reconstruction()
    image_observations: List[List[Tuple[np.ndarray, int]]] = [[] for _ in views]
    for point, observations in tracks:
        point_id = reconstruction.add_point3D(point, pycolmap.Track(), np.zeros(3, np.uint8))
        for image_index, uv in observations:
            point2d_index = len(image_observations[image_index])
            image_observations[image_index].append((uv, int(point_id)))
            reconstruction.points3D[point_id].track.add_element(image_index + 1, point2d_index)

    for i, view in enumerate(views):
        K = view["K"]
        h, w = view["points"].shape[:2]
        camera = pycolmap.Camera(
            model="PINHOLE", width=w, height=h,
            params=np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]),
            camera_id=i + 1,
        )
        reconstruction.add_camera(camera)
        w2c = np.linalg.inv(view["T_c2w"])
        image_kwargs = {
            "name": f"{view['stem']}.jpg",
            "camera_id": i + 1,
            "cam_from_world": pycolmap.Rigid3d(
                pycolmap.Rotation3d(w2c[:3, :3]), w2c[:3, 3]
            ),
        }
        try:
            image = pycolmap.Image(image_id=i + 1, **image_kwargs)
        except (AttributeError, TypeError):  # pycolmap <= 3.x
            image = pycolmap.Image(id=i + 1, **image_kwargs)
        image.points2D = pycolmap.ListPoint2D([
            pycolmap.Point2D(uv, point_id)
            for uv, point_id in image_observations[i]
        ])
        reconstruction.add_image(image)

    options = pycolmap.BundleAdjustmentOptions()
    for name in ("refine_focal_length", "refine_principal_point", "refine_extra_params"):
        if hasattr(options, name):
            setattr(options, name, bool(refine_intrinsics) if name != "refine_extra_params" else False)
    pycolmap.bundle_adjustment(reconstruction, options)

    refined_by_stem: Dict[str, np.ndarray] = {}
    for i, view in enumerate(views):
        matrix = np.asarray(reconstruction.images[i + 1].cam_from_world.matrix(), np.float64)
        w2c = np.eye(4, dtype=np.float64); w2c[:3, :4] = matrix
        refined_by_stem[str(view["stem"])] = np.linalg.inv(w2c)

    for record in chunk_records:
        refined_cameras = []
        for camera in get_transformed_cameras(record):
            updated = dict(camera)
            stem = str(updated.get("stem", ""))
            if stem in refined_by_stem:
                updated["T_c2w"] = refined_by_stem[stem].astype(np.float32)
            refined_cameras.append(updated)
        record["ba_refined_cameras"] = refined_cameras

    output_dir = Path(output_dir).expanduser().resolve()
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(sparse_dir))
    summary = {
        "enabled": True,
        "method": "sift_tracks_pycolmap",
        "num_images": len(views),
        "num_tracks": len(tracks),
        "max_reproj_error": float(max_reproj_error),
        "refine_intrinsics": bool(refine_intrinsics),
        "sparse_dir": str(sparse_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
