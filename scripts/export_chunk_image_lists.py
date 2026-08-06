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
    python scripts/export_chunk_image_lists.py /path/to/scene 32
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

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
from geoff3d.slrf.scene_io import build_views_from_scene


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
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_chunk_lists(output_dir)

    stems = [str(stem) for stem in meta["stems"]]
    image_paths = {
        str(stem): str(path)
        for stem, path in dict(meta.get("image_paths", {})).items()
    }

    chunk_records: List[Dict[str, object]] = []
    covered_indices = set()
    covered_core_indices = set()

    for chunk in chunks:
        chunk_id = int(chunk["chunk_id"])
        indices = [int(index) for index in chunk.get("indices", [])]
        core_indices = [int(index) for index in chunk.get("core_indices", [])]
        overlap_indices = [
            int(index) for index in chunk.get("overlap_indices", [])
        ]
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

        chunk_records.append(
            {
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
            }
        )

    all_indices = set(range(len(stems)))
    unassigned_indices = sorted(all_indices - covered_indices)
    unassigned_core_indices = sorted(all_indices - covered_core_indices)
    skipped_records = []
    for item in skipped_frames:
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
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
