# -*- coding: utf-8 -*-
"""Bounded TSDF mesh extraction from cached GeoFF3D predictions."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm

from geoff3d.spatial_rrd.chunk_cache import get_cached_sequence
from geoff3d.spatial_rrd.chunk_transform import (
    get_transformed_cached_point_maps,
    get_transformed_cameras,
)


def _camera_by_local_index(record: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    cameras: Dict[int, Dict[str, object]] = {}
    for camera in get_transformed_cameras(record):
        index = int(camera.get("pred_index", -1))
        if index >= 0:
            cameras[index] = camera
    return cameras


def _post_process_mesh(mesh, keep_clusters: int, min_triangles: int):
    """Keep the largest connected components and discard small floaters."""
    if len(mesh.triangles) == 0:
        return copy.deepcopy(mesh)
    triangle_clusters, cluster_counts, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters, dtype=np.int64)
    cluster_counts = np.asarray(cluster_counts, dtype=np.int64)
    if cluster_counts.size == 0:
        return copy.deepcopy(mesh)

    keep_count = max(1, min(int(keep_clusters), int(cluster_counts.size)))
    largest = np.argsort(cluster_counts)[-keep_count:]
    retained = largest[cluster_counts[largest] >= max(1, int(min_triangles))]
    if retained.size == 0:
        retained = largest[-1:]

    result = copy.deepcopy(mesh)
    result.remove_triangles_by_mask(~np.isin(triangle_clusters, retained))
    result.remove_unreferenced_vertices()
    result.remove_degenerate_triangles()
    return result


def export_tsdf_mesh(
    chunk_records: Sequence[Dict[str, object]],
    output_dir: Path,
    *,
    voxel_size: float,
    sdf_trunc: float,
    depth_trunc: float,
    min_depth: float = 1e-6,
    pixel_stride: int = 2,
    keep_clusters: int = 50,
    min_triangles: int = 50,
) -> Dict[str, object]:
    """Fuse core-view RGB-D predictions and export raw/cleaned PLY meshes."""
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "TSDF mesh export requires Open3D. Install with: pip install open3d"
        ) from exc

    voxel_size = float(voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError(f"tsdf voxel_size must be positive, got {voxel_size}")
    sdf_trunc = float(sdf_trunc) if float(sdf_trunc) > 0 else 5.0 * voxel_size
    depth_trunc = float(depth_trunc)
    if not np.isfinite(depth_trunc) or depth_trunc <= 0:
        raise ValueError(f"tsdf depth_trunc must be positive, got {depth_trunc}")
    pixel_stride = max(1, int(pixel_stride))

    integrated_stems = set()
    skipped = 0
    valid_pixels = 0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    total_core_views = sum(
        len(record.get("core_indices", [])) for record in chunk_records
    )
    progress = tqdm(
        total=total_core_views,
        desc="TSDF integrate",
        unit="view",
        dynamic_ncols=True,
    )
    for chunk_number, record in enumerate(chunk_records, start=1):
        chunk_id = int(record.get("chunk_id", chunk_number - 1))
        progress.set_postfix_str(
            f"chunk={chunk_id} ({chunk_number}/{len(chunk_records)}), fused={len(integrated_stems)}"
        )
        chunk_integrated = 0
        point_maps = get_transformed_cached_point_maps(record)
        # This is the final mask produced by depth-confidence filtering and
        # optional input keep masks.  Invalid depth never reaches Open3D.
        valid_masks = get_cached_sequence(record, "_pred_valid_masks")
        intrinsics = get_cached_sequence(record, "_chunk_intrinsics")
        rgbs = get_cached_sequence(record, "rgbs")
        core_indices = [int(v) for v in get_cached_sequence(record, "_core_local_indices")]
        cameras = _camera_by_local_index(record)
        stems = list(record.get("stems", []))

        for index in core_indices:
            progress.update(1)
            if not (
                0 <= index < len(point_maps)
                and index < len(valid_masks)
                and index < len(intrinsics)
                and index < len(rgbs)
                and index in cameras
            ):
                skipped += 1
                continue
            stem = str(stems[index]) if index < len(stems) else f"{record['chunk_id']}:{index}"
            if stem in integrated_stems:
                continue

            points = np.asarray(point_maps[index], dtype=np.float32)
            mask = np.asarray(valid_masks[index], dtype=bool)
            intrinsic = np.asarray(intrinsics[index], dtype=np.float64)
            color = np.asarray(rgbs[index])
            c2w = np.asarray(cameras[index].get("T_c2w"), dtype=np.float64)
            if (
                points.ndim != 3
                or points.shape[-1] != 3
                or mask.shape != points.shape[:2]
                or intrinsic.shape != (3, 3)
                or c2w.shape != (4, 4)
            ):
                skipped += 1
                continue

            w2c = np.linalg.inv(c2w)
            depth = (
                points @ w2c[:3, :3].T + w2c[:3, 3]
            )[..., 2].astype(np.float32)
            valid = (
                mask
                & np.isfinite(depth)
                & (depth >= float(min_depth))
                & (depth <= depth_trunc)
            )
            depth[~valid] = 0.0
            if not np.any(valid):
                skipped += 1
                continue

            if color.ndim == 3 and color.shape[0] == 3 and color.shape[-1] != 3:
                color = np.moveaxis(color, 0, -1)
            if color.shape[:2] != depth.shape or color.ndim != 3 or color.shape[-1] < 3:
                skipped += 1
                continue
            color = color[..., :3]
            if np.issubdtype(color.dtype, np.floating):
                scale = 255.0 if float(np.nanmax(color)) <= 1.0 else 1.0
                color = np.clip(color * scale, 0, 255).astype(np.uint8)
            else:
                color = np.clip(color, 0, 255).astype(np.uint8)

            if pixel_stride > 1:
                depth = depth[::pixel_stride, ::pixel_stride]
                color = color[::pixel_stride, ::pixel_stride]
                intrinsic = intrinsic.copy()
                intrinsic[0, 0] /= pixel_stride
                intrinsic[1, 1] /= pixel_stride
                intrinsic[0, 2] /= pixel_stride
                intrinsic[1, 2] /= pixel_stride

            height, width = depth.shape
            pinhole = o3d.camera.PinholeCameraIntrinsic(
                width,
                height,
                float(intrinsic[0, 0]),
                float(intrinsic[1, 1]),
                float(intrinsic[0, 2]),
                float(intrinsic[1, 2]),
            )
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color)),
                o3d.geometry.Image(np.ascontiguousarray(depth)),
                depth_scale=1.0,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, pinhole, w2c)
            integrated_stems.add(stem)
            chunk_integrated += 1
            valid_pixels += int(valid.sum())

        progress.set_postfix_str(
            f"chunk={chunk_id} complete, fused={len(integrated_stems)}"
        )
        del point_maps, valid_masks, intrinsics, rgbs
    progress.close()

    if not integrated_stems:
        raise RuntimeError("TSDF integration found no valid RGB-D camera views")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "tsdf_mesh.ply"
    post_path = output_dir / "tsdf_mesh_post.ply"

    print("[INFO] TSDF mesh extraction: extracting the global volume")
    raw_mesh = volume.extract_triangle_mesh()
    raw_mesh.compute_vertex_normals()
    print(f"[INFO] Writing raw TSDF mesh: {raw_path}")
    if not o3d.io.write_triangle_mesh(str(raw_path), raw_mesh):
        raise RuntimeError(f"Failed to write TSDF mesh: {raw_path}")

    post_mesh = _post_process_mesh(raw_mesh, keep_clusters, min_triangles)
    post_mesh.compute_vertex_normals()
    print(f"[INFO] Writing cleaned TSDF mesh: {post_path}")
    if not o3d.io.write_triangle_mesh(str(post_path), post_mesh):
        raise RuntimeError(f"Failed to write post-processed TSDF mesh: {post_path}")

    summary: Dict[str, object] = {
        "enabled": True,
        "method": "open3d_global_scalable_tsdf",
        "depth_source": "predicted_world_points_masked_by_filtered_depth_confidence",
        "voxel_size": voxel_size,
        "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc,
        "min_depth": float(min_depth),
        "pixel_stride": int(pixel_stride),
        "num_integrated_views": len(integrated_stems),
        "num_skipped_views": int(skipped),
        "num_valid_pixels": int(valid_pixels),
        "raw_vertices": len(raw_mesh.vertices),
        "raw_triangles": len(raw_mesh.triangles),
        "post_vertices": len(post_mesh.vertices),
        "post_triangles": len(post_mesh.triangles),
        "raw_mesh": str(raw_path),
        "post_mesh": str(post_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
