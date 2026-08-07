#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export image-name lists for GeoFF3D large-scene spatial chunks.

This utility reuses the same prior-footprint spatial chunking path as
``scripts/run_slrf.py`` without loading a reconstruction checkpoint or running
model inference. The input scene is expected to contain matching ``images/``,
``cams/``, and metric ``depth/`` files.

Frames whose depth maps cannot provide enough valid footprint points are
skipped before spatial chunking.

Example:
    python scripts/export_chunks.py input_scene_dir 32 --output-dir ./output
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

from geoff3d.slrf.chunking import (
    build_spatial_chunks,
    infer_spatial_axes,
    order_spatial_chunks,
    spatial_axis_indices,
)
from geoff3d.slrf.footprint_estimation import (
    FOOTPRINT_MIN_POINTS,
    FOOTPRINT_QUANTILE_MAX,
    FOOTPRINT_QUANTILE_MIN,
    FOOTPRINT_SAMPLE_STRIDE,
    _prior_footprint_worker,
)
from geoff3d.slrf.scene_io import (
    DEPTH_MAX_METERS,
    DEPTH_MIN_METERS,
    build_views_from_scene,
    read_depth,
    read_rgb,
    resize_rgb_depth_K,
    sanitize_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Partition a scene with GeoFF3D's footprint-tree algorithm and "
            "save every chunk's image file names."
        )
    )
    parser.add_argument(
        "scene_dir",
        type=Path,
        help="Scene directory containing images/, cams/, and depth/.",
    )
    parser.add_argument(
        "max_images_per_chunk",
        type=int,
        help="Maximum number of images in each chunk, including overlap images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to <scene_dir>/chunk_image_lists."
        ),
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=0,
        help="Maximum number of selected input images. 0 keeps all images.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the naturally sorted image sequence.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Select every N-th image after applying --start.",
    )
    parser.add_argument(
        "--min-images-per-chunk",
        type=int,
        default=1,
        help=(
            "Minimum core-image count for keeping a chunk. Defaults to 1. "
            "Use 8 to match configs/slrf.yaml."
        ),
    )
    parser.add_argument(
        "--footprint-workers",
        type=int,
        default=0,
        help="Worker processes for prior footprint estimation. Defaults to 0.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=518,
        help="Depth/image resize limit used for footprint estimation.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=14,
        help="Resize multiple used by the scene manifest loader.",
    )
    parser.add_argument(
        "--save-chunk-montages",
        action="store_true",
        help=(
            "Save all images of every chunk as one fixed-resolution JPEG "
            "contact sheet."
        ),
    )
    parser.add_argument(
        "--montage-width",
        type=int,
        default=1600,
        help="Fixed chunk montage width in pixels. Defaults to 1600.",
    )
    parser.add_argument(
        "--montage-height",
        type=int,
        default=900,
        help="Fixed chunk montage height in pixels. Defaults to 900.",
    )
    parser.add_argument(
        "--save-chunk-rrd",
        action="store_true",
        help=(
            "Save one RRD per chunk with input camera poses and a sampled "
            "metric-depth point cloud."
        ),
    )
    parser.add_argument(
        "--rrd-depth-stride",
        type=int,
        default=8,
        help=(
            "Pixel stride used when back-projecting depth for RRD point clouds. "
            "Defaults to 8."
        ),
    )
    parser.add_argument(
        "--rrd-max-points",
        type=int,
        default=500000,
        help="Maximum number of depth points saved in each chunk RRD.",
    )
    parser.add_argument(
        "--rrd-point-radius",
        type=float,
        default=0.0,
        help="Rerun point radius. 0 lets the viewer choose automatically.",
    )
    parser.add_argument(
        "--rrd-camera-axis-size",
        type=float,
        default=0.0,
        help=(
            "Camera-axis size in world units. 0 estimates it from the chunk "
            "point-cloud extent."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable scene-loading progress bars.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_images_per_chunk <= 0:
        raise ValueError("max_images_per_chunk must be positive.")
    if args.num_views < 0:
        raise ValueError("num_views cannot be negative.")
    if args.start < 0:
        raise ValueError("start cannot be negative.")
    if args.stride <= 0:
        raise ValueError("stride must be positive.")
    if args.min_images_per_chunk <= 0:
        raise ValueError("min_images_per_chunk must be positive.")
    if args.min_images_per_chunk > args.max_images_per_chunk:
        raise ValueError(
            "min_images_per_chunk cannot exceed max_images_per_chunk."
        )
    if args.footprint_workers < 0:
        raise ValueError("footprint_workers cannot be negative.")
    if args.max_image_size <= 0:
        raise ValueError("max_image_size must be positive.")
    if args.patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if args.montage_width <= 0 or args.montage_height <= 0:
        raise ValueError("montage width and height must be positive.")
    if args.rrd_depth_stride <= 0:
        raise ValueError("rrd_depth_stride must be positive.")
    if args.rrd_max_points <= 0:
        raise ValueError("rrd_max_points must be positive.")
    if args.rrd_point_radius < 0:
        raise ValueError("rrd_point_radius cannot be negative.")
    if args.rrd_camera_axis_size < 0:
        raise ValueError("rrd_camera_axis_size cannot be negative.")


def require_matching_priors(meta: Dict[str, object]) -> None:
    stems = [str(stem) for stem in meta.get("stems", [])]
    cams = meta.get("cams", {})
    depth_paths = meta.get("depth_paths", {})

    missing_cams = [
        stem for stem in stems if not isinstance(cams, dict) or stem not in cams
    ]
    missing_depths = [
        stem
        for stem in stems
        if not isinstance(depth_paths, dict) or stem not in depth_paths
    ]
    if missing_cams or missing_depths:
        raise RuntimeError(
            "Prior footprint chunking requires matching cams/*.txt and "
            "depth/*.exr for every image. "
            f"missing_cams={missing_cams[:8]}, "
            f"missing_depths={missing_depths[:8]}."
        )


def names_for_indices(
    indices: Iterable[int],
    stems: Sequence[str],
    image_paths: Dict[str, str],
) -> List[str]:
    names: List[str] = []
    for index in indices:
        stem = str(stems[int(index)])
        path = image_paths.get(stem)
        names.append(Path(path).name if path else stem)
    return names


def filter_meta_by_indices(
    meta: Dict[str, object],
    indices: Sequence[int],
) -> Dict[str, object]:
    filtered = dict(meta)
    original_stems = [str(stem) for stem in meta.get("stems", [])]
    selected_stems = [original_stems[int(index)] for index in indices]
    selected_set = set(selected_stems)
    filtered["stems"] = selected_stems

    for key in ("image_paths", "depth_paths", "cam_paths", "cams"):
        value = meta.get(key)
        if isinstance(value, dict):
            filtered[key] = {
                str(stem): item
                for stem, item in value.items()
                if str(stem) in selected_set
            }

    filtered["num_cam_priors"] = int(
        sum(
            1
            for stem in selected_stems
            if stem in filtered.get("cams", {})
        )
    )
    filtered["num_depth_priors"] = int(
        sum(
            1
            for stem in selected_stems
            if stem in filtered.get("depth_paths", {})
        )
    )
    return filtered


def estimate_footprints_and_skip_invalid(
    *,
    meta: Dict[str, object],
    axis_indices: Tuple[int, ...],
    workers: int,
) -> Tuple[Dict[str, object], Dict[str, object], List[Dict[str, str]]]:
    stems = [str(stem) for stem in meta["stems"]]
    cams = meta.get("cams", {})
    depth_paths = meta.get("depth_paths", {})
    jobs: List[Tuple[object, ...]] = []

    for index, stem in enumerate(stems):
        cam = cams.get(stem) if isinstance(cams, dict) else None
        depth_path = (
            depth_paths.get(stem)
            if isinstance(depth_paths, dict)
            else None
        )
        if cam is None or not depth_path:
            raise ValueError(
                f"Cannot estimate prior footprint for {stem}: "
                "missing camera or depth."
            )
        jobs.append(
            (
                index,
                str(depth_path),
                np.asarray(cam["K"]),
                np.asarray(cam["T_c2w"]),
                cam.get("width"),
                cam.get("height"),
                int(meta["target_h"]),
                int(meta["target_w"]),
                axis_indices,
                FOOTPRINT_SAMPLE_STRIDE,
                FOOTPRINT_MIN_POINTS,
                FOOTPRINT_QUANTILE_MIN,
                FOOTPRINT_QUANTILE_MAX,
            )
        )

    if int(workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            results = list(pool.map(_prior_footprint_worker, jobs))
    else:
        results = [_prior_footprint_worker(job) for job in jobs]

    valid_indices: List[int] = []
    centers: List[np.ndarray] = []
    bbox_mins: List[np.ndarray] = []
    bbox_maxs: List[np.ndarray] = []
    skipped: List[Dict[str, str]] = []

    for index, center, bbox_min, bbox_max, status in results:
        stem = stems[int(index)]
        if (
            status != "ok"
            or center is None
            or bbox_min is None
            or bbox_max is None
        ):
            skipped.append({"stem": stem, "reason": str(status)})
            continue

        valid_indices.append(int(index))
        centers.append(np.asarray(center, dtype=np.float64))
        bbox_mins.append(np.asarray(bbox_min, dtype=np.float64))
        bbox_maxs.append(np.asarray(bbox_max, dtype=np.float64))

    if not valid_indices:
        raise RuntimeError(
            "Prior footprint estimation did not produce any valid frames."
        )

    filtered_meta = filter_meta_by_indices(meta, valid_indices)
    estimated = {
        "centers": np.stack(centers, axis=0),
        "bbox_mins": np.stack(bbox_mins, axis=0),
        "bbox_maxs": np.stack(bbox_maxs, axis=0),
        "meta": {
            "estimation": "prior",
            "coordinate_axes": list(axis_indices),
            "source_counts": {"prior": len(valid_indices)},
            "sources": ["prior"] * len(valid_indices),
            "sample_stride": FOOTPRINT_SAMPLE_STRIDE,
            "min_points": FOOTPRINT_MIN_POINTS,
            "quantile_min": FOOTPRINT_QUANTILE_MIN,
            "quantile_max": FOOTPRINT_QUANTILE_MAX,
            "workers": int(workers),
            "num_input_frames": len(stems),
            "num_valid_frames": len(valid_indices),
            "num_skipped_frames": len(skipped),
        },
    }
    filtered_meta["estimated_footprints"] = estimated
    return filtered_meta, estimated, skipped


def remove_stale_chunk_lists(output_dir: Path) -> None:
    for path in output_dir.glob("chunk_*.txt"):
        if path.is_file():
            path.unlink()


def remove_stale_chunk_artifacts(folder: Path, suffix: str) -> None:
    if not folder.is_dir():
        return
    for path in folder.glob(f"chunk_*{suffix}"):
        if path.is_file():
            path.unlink()


def montage_grid_shape(num_images: int, width: int, height: int) -> Tuple[int, int]:
    if num_images <= 0:
        return 1, 1
    aspect = float(width) / float(height)
    cols = max(1, int(math.ceil(math.sqrt(float(num_images) * aspect))))
    rows = max(1, int(math.ceil(float(num_images) / float(cols))))
    return rows, cols


def save_chunk_montage(
    *,
    output_path: Path,
    image_paths: Sequence[Path],
    image_names: Sequence[str],
    width: int,
    height: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (int(width), int(height)), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    rows, cols = montage_grid_shape(len(image_paths), int(width), int(height))
    cell_w = max(1, int(width) // cols)
    cell_h = max(1, int(height) // rows)
    padding = max(2, min(cell_w, cell_h) // 40)
    label_h = max(16, min(28, cell_h // 7))

    for index, (image_path, image_name) in enumerate(
        zip(image_paths, image_names)
    ):
        row = index // cols
        col = index % cols
        x0 = col * cell_w
        y0 = row * cell_h
        x1 = min(int(width), x0 + cell_w)
        y1 = min(int(height), y0 + cell_h)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(190, 190, 190))

        image_box_w = max(1, x1 - x0 - 2 * padding)
        image_box_h = max(1, y1 - y0 - label_h - 2 * padding)
        try:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                thumb = ImageOps.contain(
                    image,
                    (image_box_w, image_box_h),
                    method=Image.Resampling.LANCZOS,
                )
        except Exception as exc:
            draw.text(
                (x0 + padding, y0 + padding),
                f"read failed: {exc}",
                fill=(170, 0, 0),
            )
            continue

        paste_x = x0 + padding + (image_box_w - thumb.width) // 2
        paste_y = y0 + padding + (image_box_h - thumb.height) // 2
        canvas.paste(thumb, (paste_x, paste_y))

        label_top = y1 - label_h
        draw.rectangle(
            [x0 + 1, label_top, x1 - 1, y1 - 1],
            fill=(32, 32, 32),
        )
        max_chars = max(8, (x1 - x0 - 2 * padding) // 7)
        label = str(image_name)
        if len(label) > max_chars:
            label = "..." + label[-max(1, max_chars - 3) :]
        draw.text(
            (x0 + padding, label_top + max(1, (label_h - 11) // 2)),
            label,
            fill=(255, 255, 255),
        )

    canvas.save(output_path, format="JPEG", quality=88, optimize=True)


def sample_depth_points_for_frame(
    *,
    stem: str,
    meta: Dict[str, object],
    stride: int,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    cams = meta.get("cams", {})
    image_paths = meta.get("image_paths", {})
    depth_paths = meta.get("depth_paths", {})
    cam = cams.get(stem) if isinstance(cams, dict) else None
    image_path = image_paths.get(stem) if isinstance(image_paths, dict) else None
    depth_path = depth_paths.get(stem) if isinstance(depth_paths, dict) else None
    if cam is None or not image_path or not depth_path:
        raise ValueError(f"Missing RGB, camera, or depth input for {stem}.")

    rgb = read_rgb(Path(str(image_path)))
    depth = read_depth(Path(str(depth_path)))
    depth_h, depth_w = depth.shape[:2]
    rgb, depth, K = resize_rgb_depth_K(
        rgb=rgb,
        depth=depth,
        K=np.asarray(cam["K"], dtype=np.float64),
        cam_width=cam.get("width"),
        cam_height=cam.get("height"),
        target_h=depth_h,
        target_w=depth_w,
    )

    ys = np.arange(0, depth_h, int(stride), dtype=np.int64)
    xs = np.arange(0, depth_w, int(stride), dtype=np.int64)
    u, v = np.meshgrid(xs.astype(np.float64), ys.astype(np.float64))
    z = depth[np.ix_(ys, xs)].astype(np.float64)
    valid = (
        np.isfinite(z)
        & (z > DEPTH_MIN_METERS)
        & (z < DEPTH_MAX_METERS)
    )
    if not valid.any():
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            {
                "stem": stem,
                "num_sampled_pixels": int(z.size),
                "num_valid_points": 0,
            },
        )

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if abs(fx) < 1e-12 or abs(fy) < 1e-12:
        raise ValueError(f"Invalid camera intrinsics for {stem}.")

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_camera = np.stack([x, y, z], axis=-1)[valid].reshape(-1, 3)
    T_c2w = np.asarray(cam["T_c2w"], dtype=np.float64)
    points_world = (
        points_camera @ T_c2w[:3, :3].T + T_c2w[:3, 3][None, :]
    )
    colors = rgb[np.ix_(ys, xs)][valid].reshape(-1, 3).astype(np.uint8)

    finite = np.isfinite(points_world).all(axis=1)
    points_world = points_world[finite]
    colors = colors[finite]
    num_valid_before_sampling = int(points_world.shape[0])

    if points_world.shape[0] > int(max_points):
        rng = np.random.default_rng(int(seed))
        selection = rng.choice(
            points_world.shape[0], size=int(max_points), replace=False
        )
        points_world = points_world[selection]
        colors = colors[selection]

    return (
        points_world.astype(np.float32),
        colors.astype(np.uint8),
        {
            "stem": stem,
            "num_sampled_pixels": int(z.size),
            "num_valid_points": num_valid_before_sampling,
            "num_saved_points": int(points_world.shape[0]),
            "depth_height": int(depth_h),
            "depth_width": int(depth_w),
            "K": K.astype(float).tolist(),
        },
    )


def build_chunk_input_point_cloud(
    *,
    stems: Sequence[str],
    meta: Dict[str, object],
    depth_stride: int,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]], List[str]]:
    if not stems:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            [],
            [],
        )

    per_frame_cap = max(1, int(math.ceil(float(max_points) / len(stems))))
    points_all: List[np.ndarray] = []
    colors_all: List[np.ndarray] = []
    frame_meta: List[Dict[str, object]] = []
    failed_frames: List[str] = []

    for frame_index, stem in enumerate(stems):
        try:
            points, colors, detail = sample_depth_points_for_frame(
                stem=str(stem),
                meta=meta,
                stride=int(depth_stride),
                max_points=per_frame_cap,
                seed=int(seed) + 1009 * (frame_index + 1),
            )
        except Exception as exc:
            failed_frames.append(f"{stem}: {exc}")
            continue
        frame_meta.append(detail)
        if points.shape[0] > 0:
            points_all.append(points)
            colors_all.append(colors)

    if not points_all:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            frame_meta,
            failed_frames,
        )

    points = np.concatenate(points_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)
    if points.shape[0] > int(max_points):
        rng = np.random.default_rng(int(seed) + 7919)
        selection = rng.choice(points.shape[0], size=int(max_points), replace=False)
        points = points[selection]
        colors = colors[selection]
    return points, colors, frame_meta, failed_frames


def camera_records_for_stems(
    meta: Dict[str, object], stems: Sequence[str]
) -> List[Dict[str, object]]:
    cams = meta.get("cams", {})
    records: List[Dict[str, object]] = []
    for stem in stems:
        cam = cams.get(stem) if isinstance(cams, dict) else None
        if cam is None:
            continue
        T_c2w = np.asarray(cam.get("T_c2w"), dtype=np.float64)
        K = np.asarray(cam.get("K"), dtype=np.float64)
        if T_c2w.shape != (4, 4) or K.shape != (3, 3):
            continue
        if not np.isfinite(T_c2w).all() or not np.isfinite(K).all():
            continue
        records.append(
            {
                "stem": str(stem),
                "T_c2w": T_c2w.astype(np.float32),
                "K": K.astype(np.float64),
                "width": cam.get("width"),
                "height": cam.get("height"),
            }
        )
    return records


def save_chunk_rrd(
    *,
    output_path: Path,
    scene_dir: Path,
    chunk_id: int,
    stems: Sequence[str],
    meta: Dict[str, object],
    depth_stride: int,
    max_points: int,
    point_radius: float,
    camera_axis_size: float,
) -> Dict[str, object]:
    import rerun as rr

    from geoff3d.slrf.rrd_writer import (
        estimate_axis_size,
        log_camera_axes,
        log_camera_labels,
        log_points,
        log_view_coordinates,
        rr_disconnect_compat,
        rr_init_save_compat,
        send_blueprint,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    points, colors, frame_meta, failed_frames = build_chunk_input_point_cloud(
        stems=stems,
        meta=meta,
        depth_stride=int(depth_stride),
        max_points=int(max_points),
        seed=930001 + int(chunk_id),
    )
    camera_records = camera_records_for_stems(meta, stems)
    axis_size = estimate_axis_size(
        [points], explicit=float(camera_axis_size)
    )

    scene_name = sanitize_name(scene_dir.resolve().name)
    recording_id = f"{scene_name}_input_chunk_{int(chunk_id):04d}"
    rr_init_save_compat(
        app_id="geoff3d_chunk_input_export",
        recording_id=recording_id,
        save_rrd=output_path,
    )
    try:
        log_view_coordinates("RIGHT_HAND_Z_UP")
        send_blueprint(background=(255, 255, 255), hide_grid=False)
        log_points(
            "world/input_depth_points",
            points=points,
            colors=colors,
            radius=float(point_radius),
        )
        log_camera_axes(
            "world/input_cameras/axes",
            cams=camera_records,
            axis_size=float(axis_size),
            radius=0.0,
            colors_xyz=(
                (255, 0, 0),
                (0, 220, 0),
                (40, 80, 255),
            ),
        )
        log_camera_labels(
            "world/input_cameras/labels",
            cams=camera_records,
            color=(20, 20, 20),
        )

        for camera in tqdm(camera_records, desc="Logging camera transforms"):
            stem = str(camera["stem"])
            T_c2w = np.asarray(camera["T_c2w"], dtype=np.float64)
            entity = f"world/input_cameras/{sanitize_name(stem)}"
            rr.log(
                entity,
                rr.Transform3D(
                    translation=T_c2w[:3, 3],
                    mat3x3=T_c2w[:3, :3],
                ),
                static=True,
            )
            width = camera.get("width")
            height = camera.get("height")
            if width is not None and height is not None:
                rr.log(
                    entity,
                    rr.Pinhole(
                        image_from_camera=np.asarray(camera["K"], dtype=np.float64),
                        resolution=[int(width), int(height)],
                        camera_xyz=rr.ViewCoordinates.RDF,
                        image_plane_distance=float(axis_size),
                    ),
                    static=True,
                )

        metadata = {
            "chunk_id": int(chunk_id),
            "scene_dir": str(scene_dir),
            "stems": [str(stem) for stem in stems],
            "num_cameras": int(len(camera_records)),
            "num_points": int(points.shape[0]),
            "depth_stride": int(depth_stride),
            "max_points": int(max_points),
            "camera_axis_size": float(axis_size),
            "failed_depth_frames": failed_frames,
            "frames": frame_meta,
        }
        rr.log(
            "world/chunk_metadata",
            rr.TextDocument(
                json.dumps(metadata, ensure_ascii=False, indent=2)
            ),
            static=True,
        )
    finally:
        rr_disconnect_compat()

    return {
        "path": str(output_path),
        "num_points": int(points.shape[0]),
        "num_cameras": int(len(camera_records)),
        "failed_depth_frames": failed_frames,
    }


def relative_output_path(path: Optional[Path], output_dir: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def export_chunk_lists(
    *,
    output_dir: Path,
    chunks: Sequence[Dict[str, object]],
    meta: Dict[str, object],
    grid_meta: Dict[str, object],
    chunk_order_meta: Dict[str, object],
    scene_dir: Path,
    max_images_per_chunk: int,
    min_images_per_chunk: int,
    selected_input_count: int,
    skipped_frames: Sequence[Dict[str, str]],
    original_image_paths: Dict[str, str],
    save_montages: bool,
    montage_width: int,
    montage_height: int,
    save_rrd: bool,
    rrd_depth_stride: int,
    rrd_max_points: int,
    rrd_point_radius: float,
    rrd_camera_axis_size: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_chunk_lists(output_dir)

    montage_dir = output_dir / "chunk_montages"
    rrd_dir = output_dir / "chunk_rrd"
    if save_montages:
        montage_dir.mkdir(parents=True, exist_ok=True)
        remove_stale_chunk_artifacts(montage_dir, ".jpg")
    if save_rrd:
        rrd_dir.mkdir(parents=True, exist_ok=True)
        remove_stale_chunk_artifacts(rrd_dir, ".rrd")

    stems = [str(stem) for stem in meta["stems"]]
    image_paths = {
        str(stem): str(path)
        for stem, path in dict(meta.get("image_paths", {})).items()
    }

    chunk_records: List[Dict[str, object]] = []
    covered_indices = set()
    covered_core_indices = set()

    for chunk in tqdm(chunks, desc="Processing chunks"):
        chunk_id = int(chunk["chunk_id"])
        indices = [int(index) for index in chunk.get("indices", [])]
        core_indices = [int(index) for index in chunk.get("core_indices", [])]
        overlap_indices = [
            int(index) for index in chunk.get("overlap_indices", [])
        ]
        chunk_stems = [stems[index] for index in indices]
        image_names = names_for_indices(indices, stems, image_paths)
        core_image_names = names_for_indices(core_indices, stems, image_paths)
        overlap_image_names = names_for_indices(
            overlap_indices, stems, image_paths
        )

        covered_indices.update(indices)
        covered_core_indices.update(core_indices)

        list_path = output_dir / f"chunk_{chunk_id:04d}.txt"
        list_path.write_text(
            "".join(f"{name}\n" for name in image_names),
            encoding="utf-8",
        )

        montage_path: Optional[Path] = None
        montage_error: Optional[str] = None
        if save_montages:
            montage_path = montage_dir / f"chunk_{chunk_id:04d}.jpg"
            try:
                save_chunk_montage(
                    output_path=montage_path,
                    image_paths=[Path(image_paths[stem]) for stem in chunk_stems],
                    image_names=image_names,
                    width=int(montage_width),
                    height=int(montage_height),
                )
            except Exception as exc:
                montage_error = str(exc)
                montage_path = None
                print(
                    f"[WARN] Failed to save montage for chunk {chunk_id}: {exc}"
                )

        rrd_path: Optional[Path] = None
        rrd_detail: Optional[Dict[str, object]] = None
        rrd_error: Optional[str] = None
        if save_rrd:
            rrd_path = rrd_dir / f"chunk_{chunk_id:04d}.rrd"
            try:
                rrd_detail = save_chunk_rrd(
                    output_path=rrd_path,
                    scene_dir=scene_dir,
                    chunk_id=chunk_id,
                    stems=chunk_stems,
                    meta=meta,
                    depth_stride=int(rrd_depth_stride),
                    max_points=int(rrd_max_points),
                    point_radius=float(rrd_point_radius),
                    camera_axis_size=float(rrd_camera_axis_size),
                )
            except Exception as exc:
                rrd_error = str(exc)
                rrd_path = None
                print(
                    f"[WARN] Failed to save RRD for chunk {chunk_id}: {exc}"
                )

        record: Dict[str, object] = {
            "chunk_id": chunk_id,
            "source_chunk_id": int(
                chunk.get("source_chunk_id", chunk_id)
            ),
            "cell_key": [int(value) for value in chunk.get("cell_key", ())],
            "adjacent_chunk_ids": [
                int(value)
                for value in chunk.get("adjacent_chunk_ids", [])
            ],
            "image_count": len(image_names),
            "core_image_count": len(core_image_names),
            "overlap_image_count": len(overlap_image_names),
            "images": image_names,
            "core_images": core_image_names,
            "overlap_images": overlap_image_names,
            "list_file": list_path.name,
            "montage_file": relative_output_path(montage_path, output_dir),
            "rrd_file": relative_output_path(rrd_path, output_dir),
        }
        if montage_error is not None:
            record["montage_error"] = montage_error
        if rrd_detail is not None:
            record["rrd_num_points"] = int(rrd_detail["num_points"])
            record["rrd_num_cameras"] = int(rrd_detail["num_cameras"])
            record["rrd_failed_depth_frames"] = list(
                rrd_detail.get("failed_depth_frames", [])
            )
        if rrd_error is not None:
            record["rrd_error"] = rrd_error
        chunk_records.append(record)

    all_indices = set(range(len(stems)))
    unassigned_indices = sorted(all_indices - covered_indices)
    unassigned_core_indices = sorted(all_indices - covered_core_indices)
    skipped_records = []
    for item in tqdm(skipped_frames, desc="Processing skipped frames"):
        stem = str(item["stem"])
        path = original_image_paths.get(stem)
        skipped_records.append(
            {
                "stem": stem,
                "image": Path(path).name if path else stem,
                "reason": str(item["reason"]),
            }
        )

    manifest = {
        "scene_dir": str(scene_dir),
        "output_dir": str(output_dir),
        "partition": str(grid_meta.get("partition", "footprint_tree")),
        "axes": str(grid_meta.get("axes", "xy")),
        "chunk_order": str(chunk_order_meta.get("strategy", "")),
        "max_images_per_chunk": int(max_images_per_chunk),
        "min_images_per_chunk": int(min_images_per_chunk),
        "auto_core_target_size": int(
            grid_meta.get("auto_core_target_size", 0)
        ),
        "num_selected_input_images": int(selected_input_count),
        "num_valid_input_images": len(stems),
        "num_skipped_invalid_depth_images": len(skipped_records),
        "num_chunks": len(chunks),
        "num_unique_images_in_chunks": len(covered_indices),
        "num_unique_core_images": len(covered_core_indices),
        "total_dropped_seam_images": int(
            grid_meta.get("total_dropped_seam_images", 0)
        ),
        "save_chunk_montages": bool(save_montages),
        "montage_resolution": [int(montage_width), int(montage_height)],
        "save_chunk_rrd": bool(save_rrd),
        "rrd_depth_stride": int(rrd_depth_stride),
        "rrd_max_points_per_chunk": int(rrd_max_points),
        "all_valid_images": names_for_indices(
            range(len(stems)), stems, image_paths
        ),
        "skipped_invalid_depth_images": skipped_records,
        "unassigned_images": names_for_indices(
            unassigned_indices, stems, image_paths
        ),
        "unassigned_core_images": names_for_indices(
            unassigned_core_indices, stems, image_paths
        ),
        "chunks": chunk_records,
    }

    manifest_path = output_dir / "chunk_image_names.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    skipped_path = output_dir / "skipped_invalid_depth_images.txt"
    skipped_path.write_text(
        "".join(
            f"{record['image']}\t{record['reason']}\n"
            for record in skipped_records
        ),
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    args = parse_args()
    validate_args(args)

    scene_dir = args.scene_dir.expanduser().resolve()
    if not scene_dir.is_dir():
        raise RuntimeError(f"Scene directory does not exist: {scene_dir}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else scene_dir / "chunk_image_lists"
    )

    _, meta = build_views_from_scene(
        scene_dir=scene_dir,
        images_dir="images",
        cams_dir="cams",
        depth_dir="depth",
        num_views=int(args.num_views),
        start=int(args.start),
        stride=int(args.stride),
        max_image_size=int(args.max_image_size),
        patch_size=int(args.patch_size),
        show_progress=not bool(args.quiet),
    )
    require_matching_priors(meta)

    selected_input_count = len(meta["stems"])
    original_image_paths = {
        str(stem): str(path)
        for stem, path in dict(meta.get("image_paths", {})).items()
    }
    axes = infer_spatial_axes(meta)
    meta, _, skipped_frames = estimate_footprints_and_skip_invalid(
        meta=meta,
        axis_indices=spatial_axis_indices(axes),
        workers=int(args.footprint_workers),
    )
    if skipped_frames:
        print(
            "[WARN] Skipped frames with invalid footprint depth: "
            f"{len(skipped_frames)}/{selected_input_count}. "
            f"First skipped={skipped_frames[:8]}"
        )

    chunks, grid_meta = build_spatial_chunks(
        meta=meta,
        spatial_partition="footprint_tree",
        axes=axes,
        max_chunk_size=int(args.max_images_per_chunk),
        min_chunk_size=int(args.min_images_per_chunk),
        max_chunks=0,
        footprint_source="prior",
        footprint_workers=int(args.footprint_workers),
    )
    if not chunks:
        raise RuntimeError(
            "No chunks were generated. Reduce --min-images-per-chunk or "
            "check the scene priors."
        )

    chunks, chunk_order_meta = order_spatial_chunks(
        chunks,
        meta=meta,
        strategy="spatial_center_bfs",
    )

    manifest_path = export_chunk_lists(
        output_dir=output_dir,
        chunks=chunks,
        meta=meta,
        grid_meta=grid_meta,
        chunk_order_meta=chunk_order_meta,
        scene_dir=scene_dir,
        max_images_per_chunk=int(args.max_images_per_chunk),
        min_images_per_chunk=int(args.min_images_per_chunk),
        selected_input_count=selected_input_count,
        skipped_frames=skipped_frames,
        original_image_paths=original_image_paths,
        save_montages=bool(args.save_chunk_montages),
        montage_width=int(args.montage_width),
        montage_height=int(args.montage_height),
        save_rrd=bool(args.save_chunk_rrd),
        rrd_depth_stride=int(args.rrd_depth_stride),
        rrd_max_points=int(args.rrd_max_points),
        rrd_point_radius=float(args.rrd_point_radius),
        rrd_camera_axis_size=float(args.rrd_camera_axis_size),
    )

    chunk_sizes = [len(chunk.get("indices", [])) for chunk in chunks]
    print(
        "Exported chunk image lists: "
        f"selected_images={selected_input_count}, "
        f"valid_images={len(meta['stems'])}, "
        f"skipped_images={len(skipped_frames)}, "
        f"chunks={len(chunks)}, axes={axes}, "
        f"chunk_size_min={min(chunk_sizes)}, "
        f"chunk_size_max={max(chunk_sizes)}, output={output_dir}"
    )
    if args.save_chunk_montages:
        print(
            "Chunk montages: "
            f"{output_dir / 'chunk_montages'} "
            f"({args.montage_width}x{args.montage_height})"
        )
    if args.save_chunk_rrd:
        print(f"Chunk RRD files: {output_dir / 'chunk_rrd'}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
