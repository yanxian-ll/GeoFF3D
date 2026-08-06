#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export image-name lists for GeoFF3D large-scene spatial chunks.

This utility reuses the same prior-footprint spatial chunking path as
``scripts/run_slrf.py`` without loading a reconstruction checkpoint or running
model inference. The input scene is expected to contain matching ``images/``,
``cams/``, and metric ``depth/`` files.

Example:
    python scripts/export_chunk_image_lists.py /path/to/scene 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from geoff3d.slrf.chunking import (
    build_spatial_chunks,
    infer_spatial_axes,
    order_spatial_chunks,
    spatial_axis_indices,
)
from geoff3d.slrf.footprint_estimation import estimate_footprints_from_prior
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
        "num_input_images": len(stems),
        "num_chunks": len(chunks),
        "num_unique_images_in_chunks": len(covered_indices),
        "num_unique_core_images": len(covered_core_indices),
        "total_dropped_seam_images": int(
            grid_meta.get("total_dropped_seam_images", 0)
        ),
        "all_images": names_for_indices(range(len(stems)), stems, image_paths),
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
        num_views=0,
        start=0,
        stride=1,
        max_image_size=int(args.max_image_size),
        patch_size=int(args.patch_size),
        show_progress=not bool(args.quiet),
    )
    require_matching_priors(meta)

    axes = infer_spatial_axes(meta)
    meta["estimated_footprints"] = estimate_footprints_from_prior(
        meta=meta,
        axis_indices=spatial_axis_indices(axes),
        workers=int(args.footprint_workers),
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
    )

    chunk_sizes = [len(chunk.get("indices", [])) for chunk in chunks]
    print(
        "Exported chunk image lists: "
        f"images={len(meta['stems'])}, chunks={len(chunks)}, axes={axes}, "
        f"chunk_size_min={min(chunk_sizes)}, "
        f"chunk_size_max={max(chunk_sizes)}, output={output_dir}"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
