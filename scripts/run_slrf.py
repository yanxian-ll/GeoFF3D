#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatial chunk reconstruction pipeline for multi-view UAV scenes.

This is the single main entry point for spatial RRD inference. It:
  1. Reads images, camera priors, and optional depth from a scene directory.
  2. Partitions frames into footprint-tree spatial chunks.
  3. Initializes a model with automatic prior policy resolution.
  4. Runs each chunk, reusing predicted depth as seam priors.
  5. Applies per-chunk alignment.
  6. Aggregates core point clouds and outputs overall RRD, chunk artifacts,
     and eval ply/npz/json.

Examples:
    # Hydra-style invocation with auto policies:
    python scripts/run_slrf.py \\
      model=geoff3d \\
      scene_dir=/path/to/scene \\
      checkpoint=experiments/.../checkpoint-last.pth \\
      output_path=outputs/scene_spatial

    # Override reconstruction config directly:
    python scripts/run_slrf.py \\
      model=vggt \\
      scene_dir=/path/to/scene \\
      checkpoint=checkpoints/vggt.pth \\
      output_path=outputs/scene_spatial \\
      max_chunk_size=24
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from geoff3d.slrf.scene_io import (
    build_views_from_scene,
    intrinsics_from_views,
    load_chunk_views_from_scene,
    resolve_device,
)
from geoff3d.slrf.chunk_cache import (
    AsyncChunkCacheWriter,
    chunk_cache_dir_for_output,
    strip_array_payload,
    write_chunk_record_manifest,
)
from geoff3d.slrf.chunking import (
    build_spatial_chunks,
    order_spatial_chunks,
    infer_spatial_axes,
    spatial_axis_indices,
)
from geoff3d.slrf.footprint_estimation import (
    estimate_footprints_from_prior,
    estimate_footprints_sequentially as estimate_sequential_footprints,
)
from geoff3d.slrf.depth_confidence import (
    apply_optional_keep_masks_to_valid_masks,
    apply_depth_confidence_filter_to_preds,
)
from geoff3d.slrf.pipeline_runtime import (
    RunLogger,
    stage,
    iter_progress,
    filter_views_meta_to_valid_poses,
)
from geoff3d.slrf.pipeline_postprocess import (
    apply_final_global_pose_alignment,
    run_deferred_dom_rendering,
    run_deferred_gsplat_bundle_refinement,
)
from geoff3d.slrf.chunk_visualization import (
    save_spatial_chunk_core_footprint_xy_visualization,
)
from geoff3d.slrf.model_runner import (
    collect_pred_outputs,
    init_model_from_hydra,
    load_checkpoint,
    checkpoint_hydra_overrides,
    build_prior_overrides,
    set_model_task_value,
    set_model_task_prob,
    set_pi3x_ray_prior_prob,
    recover_average_intrinsics_from_pred_rays,
    apply_bootstrap_intrinsics_to_views,
    build_depth_prior_cache,
    build_future_overlap_counter,
    apply_cached_depth_priors_to_views,
    points_from_maps,
    resolve_prior_policy,
    filter_views_for_prior_policy,
    apply_runtime_prior_policy,
)
from geoff3d.slrf.geometry_align import (
    recenter_anchor_from_meta,
    restore_predictions_from_recenter,
    apply_chunk_pose_alignment,
    ALIGN_MODES,
)
from geoff3d.slrf.rrd_writer import save_spatial_rrd, input_pose_centers_by_stem
from geoff3d.slrf.pose_perturb import perturb_scene_camera_poses
from geoff3d.slrf.chunk_post_align import (
    apply_deferred_chunk_post_alignment,
    compute_adjacent_chunk_seam_error,
)
from geoff3d.slrf.tsdf_mesh import export_tsdf_mesh
from geoff3d.slrf.bundle_adjustment import run_bundle_adjustment


@hydra.main(version_base=None, config_path="../configs", config_name="slrf")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    values = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("SLRF Hydra config must resolve to a mapping.")
    
    model_cfg = cfg.model
    values.pop("model", None)
    values.pop("machine", None)
    values["model"] = str(model_cfg.model_str)
    values["data_norm_type"] = str(model_cfg.data_norm_type)
    values["patch_size"] = int(model_cfg.patch_size)
    args = SimpleNamespace(**values)

    model_name = str(args.model).lower().strip()
    align_mode = str(args.align).lower().strip()
    if model_name == "geoff3d":
        if align_mode not in ALIGN_MODES:
            raise ValueError(
                f"GeoFF3D does not support align={args.align!r}. "
                f"Supported modes: {sorted(ALIGN_MODES)}"
            )
    elif align_mode != "sim3":
        raise ValueError(
            f"Model {model_name!r} only supports align=sim3, "
            f"got {args.align!r}."
        )
    args.align = align_mode
    
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Checkpoint file not found: {checkpoint_path}")
    args.checkpoint = str(checkpoint_path)

    output_dir = Path(args.output_path).expanduser().resolve()
    if output_dir.suffix.lower() == ".rrd":
        raise ValueError(
            "output_path must be an output directory, not an .rrd file. "
            "The recording is written automatically to output_path/result.rrd."
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(f"output_path must be a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_path = str(output_dir)

    scene_dir = Path(args.scene_dir).expanduser().resolve()
    images_path = scene_dir / "images"
    cams_path = scene_dir / "cams"
    depth_path = scene_dir / "depth"

    if not images_path.is_dir():
        raise RuntimeError(f"Missing required input directory: {images_path}")
    supported_image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not any(
        path.is_file() and path.suffix.lower() in supported_image_suffixes
        for path in images_path.iterdir()
    ):
        raise RuntimeError(
            f"No RGB images found under: {images_path}. "
            "Supported formats: jpg, jpeg, png, bmp, tif, tiff."
        )
    if not cams_path.is_dir():
        raise RuntimeError(f"Missing required input directory: {cams_path}")
    if not any(cams_path.glob("*.txt")):
        raise RuntimeError(f"No camera files found; expected cams/*.txt under: {cams_path}")

    if args.footprint_estimation not in {"prior", "sequential"}:
        raise ValueError(
            "footprint_estimation must be 'prior' or 'sequential', "
            f"got {args.footprint_estimation!r}."
        )

    stage("Footprint Estimation", f"method={args.footprint_estimation}")
    if args.spatial_partition == "temporal":
        footprint_axes = "xy"
        footprint_source = "none"
    elif args.footprint_estimation == "prior":
        if not depth_path.is_dir():
            raise RuntimeError(
                "footprint_estimation=prior requires the depth directory: "
                f"{depth_path}"
            )
        if not any(depth_path.glob("*.exr")):
            raise RuntimeError(
                "footprint_estimation=prior requires metric depth/*.exr files under: "
                f"{depth_path}"
            )

    if args.depth_prior == "input":
        if not depth_path.is_dir():
            raise RuntimeError(
                f"depth_prior=input requires the depth directory: {depth_path}"
            )
        if not any(depth_path.glob("*.exr")):
            raise RuntimeError(
                f"depth_prior=input requires metric depth/*.exr files under: {depth_path}"
            )

    run_logger = RunLogger(
        Path(args.output_path),
        log_file=None,
        enabled=True,
    )
    run_logger.install()
    stage("Setup", f"scene={args.scene_dir}, output={args.output_path}, model={args.model}")

    device = resolve_device(args.device)
    stage_timings: Dict[str, float] = {
        "chunking": 0.0,
        "model_prediction": 0.0,
        "post_alignment": 0.0,
    }
    timing_path = output_dir / "processing_time.json"

    # ------------------------------------------------------------------
    # 1. Resolve input preprocessing config
    # ------------------------------------------------------------------
    stage("Input Preparation", "resolving input preprocessing and loading lightweight scene manifest")
    norm_type = args.data_norm_type
    print(
        "[INFO] Input preprocessing: "
        f"norm_type={norm_type}, patch_size={int(args.patch_size)}, "
        f"max_image_size={int(args.max_image_size)}"
    )

    # ------------------------------------------------------------------
    # 2. Load scene
    # ------------------------------------------------------------------
    run_logger.detail(
        f"Loading scene manifest: scene_dir={scene_dir}, images_dir=images, "
        f"cams_dir=cams, depth_dir=depth, num_views={args.num_views}, "
        f"start={args.start}, stride={args.stride}, max_image_size={args.max_image_size}"
    )
    views, meta = build_views_from_scene(
        scene_dir=scene_dir,
        images_dir="images",
        cams_dir="cams",
        depth_dir="depth",
        num_views=args.num_views,
        start=args.start,
        stride=args.stride,
        max_image_size=args.max_image_size,
        patch_size=args.patch_size,
        device=device,
        show_progress=True,
    )
    if args.footprint_estimation == "prior":
        selected_stems = [str(stem) for stem in meta["stems"]]
        cams = meta.get("cams", {})
        depth_paths = meta.get("depth_paths", {})
        missing_cams = [stem for stem in selected_stems if stem not in cams]
        missing_depths = [stem for stem in selected_stems if stem not in depth_paths]
        if missing_cams or missing_depths:
            raise RuntimeError(
                "footprint_estimation=prior requires matching cams/*.txt and "
                "depth/*.exr for every selected image. "
                f"missing_cams={missing_cams[:8]}, "
                f"missing_depths={missing_depths[:8]}."
            )

    # 模拟扰相机位姿
    pose_perturb_meta = perturb_scene_camera_poses(
        meta,
        enabled=bool(args.pose_perturb),
        seed=int(args.pose_perturb_seed_offset),
        xy_std=float(args.pose_perturb_xy_std),
        z_std=float(args.pose_perturb_z_std),
        yaw_std_deg=float(args.pose_perturb_yaw_std_deg),
        xy_max=float(args.pose_perturb_xy_max),
        z_max=float(args.pose_perturb_z_max),
        yaw_max_deg=float(args.pose_perturb_yaw_max_deg),
    )
    if bool(args.pose_perturb):
        print(
            "[INFO] Pose perturbation enabled at input stage: "
            f"scope={pose_perturb_meta.get('scope')}, "
            f"perturbed={pose_perturb_meta.get('num_perturbed_cameras', 0)}/"
            f"{pose_perturb_meta.get('num_cameras', 0)}, "
            f"xy_std={float(args.pose_perturb_xy_std):.6g}m, "
            f"z_std={float(args.pose_perturb_z_std):.6g}m, "
            f"yaw_std={float(args.pose_perturb_yaw_std_deg):.6g}deg"
        )
        run_logger.detail(
            "Pose perturbation config: "
            + json.dumps(pose_perturb_meta, ensure_ascii=False)
        )
        
    views, meta, ignored_no_pose = filter_views_meta_to_valid_poses(
        views,
        meta,
        output_path=Path(args.output_path),
        run_logger=run_logger,
    )
    if ignored_no_pose:
        run_logger.detail(
            f"Pose filtering kept {len(views)} frames and ignored "
            f"{len(ignored_no_pose)} frames without valid poses."
        )

    # ------------------------------------------------------------------
    # 3. Resolve prior policy
    # ------------------------------------------------------------------
    stage("Policy Resolution", "resolving priors and alignment mode")
    prior_policy = resolve_prior_policy(args, model_name, meta)

    print(
        "[POLICY] "
        f"model={prior_policy['model']}, "
        f"align={args.align}, "
        f"pose={prior_policy['pose']}, "
        f"translation={prior_policy['translation']}, "
        f"rotation={prior_policy['rotation']}, "
        f"ray={prior_policy['ray']}, "
        f"depth={prior_policy['depth']}, "
        f"bootstrap_ray={prior_policy['bootstrap_ray']}, "
        f"bootstrap_depth={prior_policy['bootstrap_depth']}"
    )

    # ------------------------------------------------------------------
    # 4. Recenter input views around the mean camera center
    # ------------------------------------------------------------------
    stage("Recenter", "centering on the mean input-camera position")
    recenter_state = recenter_anchor_from_meta(meta)
    print(
        f"[INFO] Input views recentered by {np.linalg.norm(recenter_state):.3f} units."
    )

    if args.footprint_estimation == "prior":
        footprint_axes = infer_spatial_axes(meta)
        meta["estimated_footprints"] = estimate_footprints_from_prior(
            meta=meta,
            axis_indices=spatial_axis_indices(footprint_axes),
            workers=int(args.footprint_workers),
        )
        footprint_source = "prior"
    elif args.footprint_estimation == "sequential":
        footprint_axes = "xy"
        checkpoint_overrides, checkpoint_override = checkpoint_hydra_overrides(
            model_name,
            args.checkpoint,
        )
        footprint_model, _ = init_model_from_hydra(
            model_name=model_name,
            machine="aws",
            hydra_overrides=(
                ["model.model_config.load_pretrained_weights=false"]
                + checkpoint_overrides
                + build_prior_overrides(model_name, prior_policy)
            ),
            device=device,
        )
        load_checkpoint(footprint_model, checkpoint_override)
        footprint_model.eval()
        apply_runtime_prior_policy(footprint_model, prior_policy)
        meta["estimated_footprints"] = estimate_sequential_footprints(
            model=footprint_model,
            model_name=model_name,
            views=views,
            meta=meta,
            prior_policy=prior_policy,
            device=device,
            recenter_state=recenter_state,
            norm_type=norm_type,
            args=args,
        )
        footprint_source = "sequential"
        del footprint_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Spatial chunking
    # ------------------------------------------------------------------
    chunk_order = (
        "sequential"
        if args.spatial_partition == "temporal"
        else args.chunk_order
    )
    stage(
        "Spatial Chunking",
        f"max_chunk_size={args.max_chunk_size}, order={chunk_order}",
    )
    chunking_time_start = time.perf_counter()
    chunks, grid_meta = build_spatial_chunks(
        meta=meta,
        spatial_partition=args.spatial_partition,
        axes=footprint_axes,
        max_chunk_size=args.max_chunk_size,
        min_chunk_size=args.min_chunk_size,
        max_chunks=args.max_chunks,
        footprint_source=footprint_source,
        footprint_workers=args.footprint_workers,
        temporal_overlap_ratio=args.temporal_overlap_ratio,
    )
    if not chunks:
        raise RuntimeError(
            "No spatial chunks generated. Check "
            "--max_chunk_size/--min_chunk_size/--num_views."
        )

    chunks, chunk_order_meta = order_spatial_chunks(
        chunks,
        meta=meta,
        strategy=chunk_order,
    )
    stage_timings["chunking"] = float(time.perf_counter() - chunking_time_start)
    grid_meta["chunk_order"] = chunk_order_meta

    print(
        f"Spatial chunks: frames={len(views)}, chunks={len(chunks)}, "
        f"partition={grid_meta.get('partition', 'footprint_tree')}, "
        f"axes={grid_meta['axes']}, "
        f"core_target={grid_meta['auto_core_target_size']}, "
        f"dropped_seam={grid_meta['total_dropped_seam_images']}, "
        f"order={chunk_order_meta.get('strategy')}, "
        f"start_source_chunk={chunk_order_meta.get('start_source_chunk_id', 0)}, "
        f"adj_edges={chunk_order_meta.get('num_adjacency_edges', 0)}"
    )
    if "footprint" in grid_meta:
        counts = grid_meta["footprint"].get(
            "footprint_source_counts", {}
        )
        print(
            "[INFO] footprint sources: "
            f"depth={int(counts.get('depth', 0))}, "
            f"lookat={int(counts.get('lookat', 0))}, "
            f"center={int(counts.get('center', 0))}"
            f", predicted={int(counts.get('predicted', 0))}"
        )

    save_spatial_chunk_core_footprint_xy_visualization(
        meta=meta,
        grid_meta=grid_meta,
        chunks=chunks,
        output_path=Path(args.output_path),
        point_size=12.0,
        bg_point_size=3.0,
        point_alpha=0.78,
        bg_alpha=0.22,
        label_size=11.0,
        font_scale=1.35,
        padding_ratio=0.02,
        show_legend=False,
        legend_cols=0,
        legend_max_rows=16,
    )

    # ------------------------------------------------------------------
    # 6. Init model
    # ------------------------------------------------------------------
    stage("Model Initialization", f"model={model_name}")
    checkpoint_overrides, checkpoint_override = checkpoint_hydra_overrides(
        model_name,
        args.checkpoint,
    )
    hydra_overrides = (
        ["model.model_config.load_pretrained_weights=false"]
        + checkpoint_overrides
        + build_prior_overrides(model_name, prior_policy)
    )
    if checkpoint_overrides:
        run_logger.detail(
            "Routing checkpoint through Hydra model config: "
            + ", ".join(checkpoint_overrides)
        )
    model, _ = init_model_from_hydra(
        model_name=model_name,
        machine="aws",
        hydra_overrides=hydra_overrides,
        device=device,
    )
    load_checkpoint(model, checkpoint_override)
    model.eval()

    apply_runtime_prior_policy(model, prior_policy)
    run_logger.detail(
        f"Using image norm_type={norm_type!r}, patch_size={int(args.patch_size)} "
        "for chunk tensor loading."
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    peak_gpu_memory: Dict[str, object] = {
        "enabled": bool(device.type == "cuda"),
        "device": str(device),
        "peak_allocated_bytes": 0,
        "peak_allocated_mib": 0.0,
        "peak_reserved_bytes": 0,
        "peak_reserved_mib": 0.0,
    }

    def update_peak_gpu_memory() -> None:
        if device.type != "cuda":
            return
        torch.cuda.synchronize(device)
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
        peak_gpu_memory.update(
            {
                "peak_allocated_bytes": allocated,
                "peak_allocated_mib": float(allocated / (1024 ** 2)),
                "peak_reserved_bytes": reserved,
                "peak_reserved_mib": float(reserved / (1024 ** 2)),
            }
        )

    def build_processing_time_meta(
        post_align_summary: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        post_align_summary = post_align_summary or {}
        total = float(sum(stage_timings.values()))
        return {
            "schema": "processing_time_v3",
            "method": model_name,
            "processing_time_seconds": float(total),
            "processing_time_ms": float(total * 1000.0),
            "chunking_seconds": float(stage_timings["chunking"]),
            "chunking_ms": float(stage_timings["chunking"] * 1000.0),
            "model_prediction_seconds": float(stage_timings["model_prediction"]),
            "model_prediction_ms": float(stage_timings["model_prediction"] * 1000.0),
            "post_alignment_seconds": float(stage_timings["post_alignment"]),
            "post_alignment_ms": float(stage_timings["post_alignment"] * 1000.0),
            "stage_times_seconds": dict(stage_timings),
            "stage_times_ms": {k: float(v * 1000.0) for k, v in stage_timings.items()},
            "timing_scope": ["chunking", "model_prediction", "post_alignment"],
            "excluded": [
                "scene_data_loading",
                "model_initialization_and_weight_loading",
                "chunk_data_loading_and_preprocessing",
                "prediction_postprocessing",
                "cache_io",
                "seam_error_evaluation",
                "rrd_and_eval_saving",
                "bundle_adjustment",
                "tsdf_mesh_export",
                "gsplat_optimization",
                "dom_generation",
            ],
            "peak_gpu_memory": dict(peak_gpu_memory),
            "peak_gpu_memory_allocated_bytes": int(peak_gpu_memory["peak_allocated_bytes"]),
            "peak_gpu_memory_allocated_mib": float(peak_gpu_memory["peak_allocated_mib"]),
            "peak_gpu_memory_reserved_bytes": int(peak_gpu_memory["peak_reserved_bytes"]),
            "peak_gpu_memory_reserved_mib": float(peak_gpu_memory["peak_reserved_mib"]),
            "num_frames": int(len(views)),
            "num_chunks": int(len(chunk_records)),
            "post_chunk_align": bool(args.post_chunk_align),
            "post_chunk_align_mode": str(args.post_chunk_align_mode),
            "post_chunk_align_num_valid_edges": int(
                post_align_summary.get("num_valid_edges", 0)
            ),
            "gsplat_refine_included": bool(args.gsplat_refine),
            "dom_rendering_included": bool(args.render_dom),
        }

    def write_processing_time_meta(meta_obj: Dict[str, object]) -> None:
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(
            json.dumps(meta_obj, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # 7. Per-chunk bootstrap state
    # ------------------------------------------------------------------
    stage("Chunk Prediction Setup", "initializing bootstrap priors and chunk cache")
    align_mode = args.align
    input_pose_cams_by_stem_map = input_pose_centers_by_stem(meta)

    bootstrap_ray = bool(prior_policy["bootstrap_ray"])
    bootstrap_depth = bool(prior_policy["bootstrap_depth"])

    bootstrapped_intrinsics: Optional[torch.Tensor] = None
    if bootstrap_ray:
        set_pi3x_ray_prior_prob(model, enabled=False)
        print(
            "[INFO] bootstrap_ray: chunk 0 runs without ray/intrinsics prior; "
            "later chunks use average intrinsics recovered from chunk 0 rays."
        )
    if bootstrap_depth:
        set_model_task_prob(model, "depth_prob", enabled=False)
        set_model_task_value(model, "sparse_depth_prob", 0.0)
        print(
            "[INFO] bootstrap_depth: external depth prior disabled; "
            "later chunks reuse predicted seam depth from earlier chunks."
        )

    depth_prior_cache: Dict[int, torch.Tensor] = {}
    future_overlap_counter = build_future_overlap_counter(chunks)
    chunk_records: List[Dict[str, object]] = []
    chunk_cache_dir = chunk_cache_dir_for_output(Path(args.output_path))
    chunk_cache_writer = AsyncChunkCacheWriter(
        chunk_cache_dir,
        max_workers=1,
        max_pending=0,
    )
    print(
        "[INFO] chunk cache writer: "
        f"workers={chunk_cache_writer.max_workers}, "
        f"max_pending={chunk_cache_writer.max_pending}, "
        f"dir={chunk_cache_dir}"
    )
    depth_conf_debug_dir = (
        output_dir
        / "debug"
        / "depth_conf_filter"
        if bool(args.debug_depth_conf_filter)
        else None
    )

    # ------------------------------------------------------------------
    # 8. Run chunks (feed-forward only, and torch.no_grad() for model inference)
    # ------------------------------------------------------------------
    stage("Chunk Prediction", f"chunks={len(chunks)}")
    chunk_iter = iter_progress(
        chunks,
        desc="Predict chunks",
        total=len(chunks),
        enabled=True,
    )
    for chunk in chunk_iter:
        chunk_id = int(chunk["chunk_id"])
        indices = [int(i) for i in chunk["indices"]]
        core_indices = [int(i) for i in chunk["core_indices"]]
        overlap_indices = [int(i) for i in chunk["overlap_indices"]]

        chunk_stems = [meta["stems"][i] for i in indices]
        chunk_views_raw, chunk_rgbs = load_chunk_views_from_scene(
            lightweight_views=views,
            meta=meta,
            indices=indices,
            prior_policy=prior_policy,
            device=device,
            recenter_anchor=recenter_state,
            num_workers=args.scene_io_workers,
            norm_type=norm_type,
        )
        chunk_views = filter_views_for_prior_policy(chunk_views_raw, prior_policy)

        if bootstrap_ray and chunk_id > 0:
            if bootstrapped_intrinsics is None:
                set_pi3x_ray_prior_prob(model, enabled=False)
                run_logger.detail(
                    f"[chunk {chunk_id:03d}] bootstrap ray prior unavailable; "
                    "continuing without ray/intrinsics prior."
                )
            else:
                set_pi3x_ray_prior_prob(model, enabled=True)
                chunk_views = apply_bootstrap_intrinsics_to_views(
                    chunk_views,
                    intrinsics=bootstrapped_intrinsics,
                    device=device,
                )

        num_depth_priors_used = 0
        if bootstrap_depth:
            chunk_views, num_depth_priors_used = (
                apply_cached_depth_priors_to_views(
                    chunk_views,
                    indices=indices,
                    depth_cache=depth_prior_cache,
                    device=device,
                )
            )
            set_model_task_prob(
                model, "depth_prob", enabled=num_depth_priors_used > 0
            )
            if num_depth_priors_used > 0:
                set_model_task_value(model, "sparse_depth_prob", 0.0)

            for idx in overlap_indices:
                idx = int(idx)
                if idx in future_overlap_counter:
                    future_overlap_counter[idx] -= 1
                    if future_overlap_counter[idx] <= 0:
                        future_overlap_counter.pop(idx, None)
                        depth_prior_cache.pop(idx, None)

        run_logger.detail(
            f"[chunk {chunk_id:03d}] cell={chunk['cell_key']}, "
            f"core={len(core_indices)}, overlap={len(overlap_indices)}, "
            f"total={len(indices)}, depth_priors={num_depth_priors_used}, "
            f"stems={chunk_stems[0]}..{chunk_stems[-1]}"
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_prediction_start = time.perf_counter()
        with torch.no_grad():
            preds = model(chunk_views)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        stage_timings["model_prediction"] += float(
            time.perf_counter() - model_prediction_start
        )

        depth_conf_filter_meta: Dict[str, object] = {
            "enabled": False,
            "quantile": float(args.conf_quantile),
        }
        depth_conf_keep_masks: Optional[List[Optional[np.ndarray]]] = None

        if bool(args.depth_conf_filter):
            depth_conf_filter_meta, depth_conf_keep_masks = apply_depth_confidence_filter_to_preds(
                preds=preds,
                rgbs=chunk_rgbs,
                stems=chunk_stems,
                chunk_id=chunk_id,
                conf_quantile=float(args.conf_quantile),
                pred_min_depth=float(args.pred_min_depth),
                debug_dir=depth_conf_debug_dir,
            )
            if depth_conf_filter_meta.get("enabled"):
                run_logger.detail(
                    f"[chunk {chunk_id:03d}] depth confidence filter: "
                    f"q={float(depth_conf_filter_meta.get('quantile', 0.0)):.3f}, "
                    f"dropped={int(depth_conf_filter_meta.get('num_dropped_pixels', 0))}/"
                    f"{int(depth_conf_filter_meta.get('num_valid_pixels', 0))} "
                    f"({float(depth_conf_filter_meta.get('drop_fraction', 0.0)):.3f}), "
                    f"missing_conf_views={int(depth_conf_filter_meta.get('num_views_missing_conf', 0))}"
                )

        if bootstrap_ray and chunk_id == 0:
            bootstrapped_intrinsics = (
                recover_average_intrinsics_from_pred_rays(preds)
            )
            if bootstrapped_intrinsics is None:
                print("[WARN] Failed to recover bootstrap intrinsics from chunk 0.")
            else:
                K_np = bootstrapped_intrinsics.detach().cpu().numpy()
                print(
                    "[INFO] Recovered bootstrap intrinsics from chunk 0: "
                    f"fx={K_np[0, 0]:.3f}, fy={K_np[1, 1]:.3f}, "
                    f"cx={K_np[0, 2]:.3f}, cy={K_np[1, 2]:.3f}"
                )

        if bootstrap_depth:
            keep_core = {
                int(i)
                for i in core_indices
                if future_overlap_counter.get(int(i), 0) > 0
            }
            keep_seam = {
                int(i)
                for i in overlap_indices
                if future_overlap_counter.get(int(i), 0) > 0
            }

            core_depth_cache = build_depth_prior_cache(
                preds,
                indices,
                keep_indices=keep_core,
                valid_masks=depth_conf_keep_masks,
            )
            seam_depth_cache = build_depth_prior_cache(
                preds,
                indices,
                keep_indices=keep_seam,
                valid_masks=depth_conf_keep_masks,
            )

            core_inserted = 0
            core_replaced = 0
            for global_idx, depth in core_depth_cache.items():
                global_idx = int(global_idx)
                if global_idx in depth_prior_cache:
                    core_replaced += 1
                else:
                    core_inserted += 1
                depth_prior_cache[global_idx] = depth

            seam_inserted = 0
            seam_skipped_existing = 0
            for global_idx, depth in seam_depth_cache.items():
                global_idx = int(global_idx)
                if global_idx in depth_prior_cache:
                    seam_skipped_existing += 1
                    continue
                depth_prior_cache[global_idx] = depth
                seam_inserted += 1

            if core_depth_cache or seam_depth_cache:
                run_logger.detail(
                    f"[chunk {chunk_id:03d}] cached predicted depth priors: "
                    f"core_inserted={core_inserted}, "
                    f"core_replaced={core_replaced}, "
                    f"seam_inserted={seam_inserted}, "
                    f"seam_skipped_existing={seam_skipped_existing}, "
                    f"cache_size={len(depth_prior_cache)}"
                )

        collect_point_indices = None

        raw_points_unmasked, raw_colors_unmasked, pred_maps, pred_valid_masks_unmasked, raw_cams = (
            collect_pred_outputs(
                preds=preds,
                rgbs=chunk_rgbs,
                pred_min_depth=args.pred_min_depth,
                conf_quantile=0.0,
                stems=chunk_stems,
                collect_point_indices=collect_point_indices,
            )
        )

        pred_valid_masks_masked = apply_optional_keep_masks_to_valid_masks(
            pred_valid_masks_unmasked,
            depth_conf_keep_masks,
        )

        if collect_point_indices is not None:
            point_local_indices_for_chunk = list(collect_point_indices)
        else:
            point_local_indices_for_chunk = list(range(len(pred_maps)))

        raw_points, raw_colors = points_from_maps(
            pred_maps=pred_maps,
            pred_valid_masks=pred_valid_masks_masked,
            rgbs=chunk_rgbs,
            local_indices=point_local_indices_for_chunk,
        )

        if recenter_state is not None:
            (
                raw_points,
                raw_colors,
                pred_maps,
                raw_cams,
            ) = restore_predictions_from_recenter(
                raw_points, raw_colors, pred_maps, raw_cams, recenter_state
            )

        if align_mode != "none":
            (
                pred_points,
                pred_colors,
                pred_maps_aligned,
                pred_cams,
                align_meta,
            ) = apply_chunk_pose_alignment(
                mode=align_mode,
                chunk_id=chunk_id,
                reference_cams_by_stem=input_pose_cams_by_stem_map,
                raw_pred_points=raw_points,
                raw_pred_colors=raw_colors,
                pred_maps=pred_maps,
                raw_pred_cams=raw_cams,
                target_stems=chunk_stems,
                seed=70001 + int(chunk_id),
            )
            run_logger.detail(
                f"[chunk {chunk_id:03d}] pose alignment: "
                f"mode={align_meta['mode']}, valid={align_meta['valid']}, "
                f"num_corr={align_meta['num_corr']}, "
                f"scale={float(align_meta['scale']):.6g}, "
                f"median_residual={float(align_meta.get('median_residual', float('nan'))):.6g}"
            )
        else:
            pred_points = raw_points
            pred_colors = raw_colors
            pred_maps_aligned = pred_maps
            pred_cams = raw_cams
            align_meta = {
                "mode": align_mode,
                "valid": True,
                "source": "none",
                "num_corr": 0,
                "num_scale_pairs": 0,
                "matched_camera_stems": [],
                "scale": 1.0,
                "yaw_degrees": 0.0,
                "yaw_valid": False,
                "R": np.eye(3, dtype=np.float32).tolist(),
                "t": np.zeros(3, dtype=np.float32).tolist(),
                "median_residual": float("nan"),
                "note": "no alignment applied",
                "chunk_id": int(chunk_id),
            }

        gsplat_meta: Dict[str, object] = {
            "enabled": False,
            "deferred": bool(args.gsplat_refine),
        }

        if collect_point_indices is not None:
            chunk_point_local_indices = list(collect_point_indices)
        else:
            chunk_point_local_indices = list(range(len(pred_maps_aligned)))

        core_points, core_colors = points_from_maps(
            pred_maps=pred_maps_aligned,
            pred_valid_masks=pred_valid_masks_masked,
            rgbs=chunk_rgbs,
            local_indices=chunk["core_local_indices"],
        )

        record: Dict[str, object] = {
            "chunk_id": chunk_id,
            "partition": str(chunk.get("partition", grid_meta.get("partition", "footprint_tree"))),
            "cell_key": tuple(chunk["cell_key"]),
            "alignment_topology": str(chunk.get("alignment_topology", "tree_path")),
            "align_parent_id": chunk.get("align_parent_id", None),
            "align_level": int(chunk.get("align_level", 0)),
            "adjacent_chunk_ids": [
                int(value) for value in chunk.get("adjacent_chunk_ids", [])
            ],
            "indices": indices,
            "core_indices": core_indices,
            "overlap_indices": overlap_indices,
            "stems": chunk_stems,
            "core_stems": [meta["stems"][i] for i in core_indices],
            "overlap_stems": [meta["stems"][i] for i in overlap_indices],
            "num_seam_candidates": int(chunk["num_seam_candidates"]),
            "num_dropped_seam_images": int(chunk["num_dropped_seam_images"]),
            "num_depth_priors_used": int(num_depth_priors_used),
            "pred_cams": pred_cams,
            "align_meta": align_meta,
            "depth_conf_filter_meta": depth_conf_filter_meta,
            "gsplat_meta": gsplat_meta,
            "num_chunk_pred_points_raw": int(raw_points.shape[0]),
            "num_chunk_pred_points_filtered": int(pred_points.shape[0]),
            "num_chunk_pred_points_refined": int(pred_points.shape[0]),
            "num_core_pred_points": int(core_points.shape[0]),
        }

        chunk_cache_writer.submit(
            record,
            {
                "rgbs": chunk_rgbs,
                "chunk_pred_points": pred_points,
                "chunk_pred_colors": pred_colors,
                "core_pred_points": core_points,
                "core_pred_colors": core_colors,
                "_pred_maps": pred_maps_aligned,
                "_pred_valid_masks": pred_valid_masks_masked,
                "_pred_valid_masks_unmasked": pred_valid_masks_unmasked,
                "_chunk_intrinsics": intrinsics_from_views(chunk_views_raw),
                "_chunk_point_local_indices": chunk_point_local_indices,
                "_core_local_indices": list(chunk["core_local_indices"]),
            },
        )
        strip_array_payload(record)

        chunk_records.append(record)

        run_logger.detail(
            f"[chunk {chunk_id:03d}] prediction: "
            f"raw_points={raw_points.shape[0]}, "
            f"filtered_points={pred_points.shape[0]}, "
            f"core_points={core_points.shape[0]}, "
            f"cameras={len(pred_cams)}, "
            f"align={align_meta['mode']}, valid={align_meta['valid']}, "
            f"gsplat={bool(gsplat_meta.get('deferred', False))}"
        )
        if hasattr(chunk_iter, "set_postfix"):
            chunk_iter.set_postfix(
                chunk=f"{chunk_id:03d}",
                prior=num_depth_priors_used,
                core=int(core_points.shape[0]),
                cache=len(depth_prior_cache),
                align=str(align_meta.get("mode", align_mode)),
            )
        del (
            chunk_views_raw,
            chunk_views,
            preds,
            raw_points,
            raw_colors,
            pred_maps,
            pred_valid_masks_unmasked,
            pred_valid_masks_masked,
            pred_maps_aligned,
            pred_points,
            pred_colors,
            core_points,
            core_colors,
            chunk_rgbs,
        )
        
    # ------------------------------------------------------------------
    # 9. Cleanup model (no longer needed for deferred stages)
    # ------------------------------------------------------------------
    stage("Chunk Cache Flush", "waiting for async cache writes")
    chunk_cache_writer.wait()
    write_chunk_record_manifest(
        chunk_records,
        chunk_cache_dir / "manifest.json",
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[INFO] Model deleted, GPU cache cleared.")

    # ------------------------------------------------------------------
    # 10. Deferred chunk post-alignment
    # ------------------------------------------------------------------
    stage("Deferred Chunk Alignment", f"enabled={bool(args.post_chunk_align)}")
    post_align_time_start = time.perf_counter()
    post_chunk_align_summary = apply_deferred_chunk_post_alignment(
        chunk_records=chunk_records,
        args=args,
        show_progress=True,
    )
    if "seam_error" in post_chunk_align_summary:
        grid_meta["seam_error"] = post_chunk_align_summary["seam_error"]
    stage_timings["post_alignment"] = float(time.perf_counter() - post_align_time_start)
    if bool(args.compute_seam_error):
        stage("Seam Error", "evaluating adjacent chunk overlap consistency")
        seam_error_start = time.perf_counter()
        post_chunk_align_summary["seam_error"] = compute_adjacent_chunk_seam_error(
            chunk_records,
            max_points_per_edge=int(args.seam_error_max_points_per_edge),
            seed=990017,
        )
        seam = post_chunk_align_summary["seam_error"]
        seam["evaluation_seconds"] = float(time.perf_counter() - seam_error_start)
        # Persist explicitly computed seam metrics in the RRD sidecar.  The
        # earlier assignment only covers a seam result optionally returned by
        # the post-alignment stage itself; compute_seam_error populates it
        # afterwards.
        grid_meta["seam_error"] = seam
        print(
            "[INFO] Seam error: "
            f"edges={seam['num_valid_edges']}/{seam['num_adjacency_edges']}, "
            f"E_seam={float(seam['seam_error']):.6g}, "
            f"E_seam_z={float(seam['seam_error_z']):.6g}, "
            f"time={float(seam['evaluation_seconds']):.3f}s"
        )

    # Post-chunk alignment deliberately has no fixed world gauge. Restore the
    # whole reconstruction to the input camera-pose frame before any final
    # point/camera aggregation so RRD and eval outputs share the GT frame.
    stage("Final Global Pose Alignment", f"mode={args.align}")
    final_global_align_meta = apply_final_global_pose_alignment(
        chunk_records=chunk_records,
        reference_centers_by_stem=input_pose_cams_by_stem_map,
        mode=str(args.align),
        seed=0,
    )
    post_chunk_align_summary["final_global_pose_alignment"] = (
        final_global_align_meta
    )
    if bool(final_global_align_meta.get("valid", False)):
        print(
            "[INFO] Final global pose alignment: "
            f"mode={final_global_align_meta['mode']}, "
            f"matches={int(final_global_align_meta['num_corr'])}, "
            f"scale={float(final_global_align_meta['scale']):.6g}, "
            f"yaw={float(final_global_align_meta.get('yaw_degrees', float('nan'))):.6g}deg, "
            f"median_residual={float(final_global_align_meta.get('median_residual', float('nan'))):.6g}"
        )
    else:
        print(
            "[WARN] Final global pose alignment failed; keeping the "
            f"post-chunk world gauge. Reason: {final_global_align_meta.get('note', 'unknown')}"
        )
    update_peak_gpu_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stage("Bundle Adjustment", f"enabled={bool(args.bundle_adjustment)}")
    if bool(args.bundle_adjustment):
        ba_output_dir = output_dir / "bundle_adjustment"
        ba_summary = run_bundle_adjustment(
            chunk_records,
            ba_output_dir,
            max_keypoints=2048,
            pair_window=2,
            ratio_test=0.8,
            max_reproj_error=8.0,
            refine_intrinsics=False,
        )
        post_chunk_align_summary["bundle_adjustment"] = ba_summary
        print(
            "[INFO] Bundle adjustment complete: "
            f"images={ba_summary['num_images']}, tracks={ba_summary['num_tracks']}, "
            f"output={ba_summary['sparse_dir']}"
        )

    stage("Timing", "writing processing_time.json")
    processing_time_meta = build_processing_time_meta(
        post_align_summary=post_chunk_align_summary,
    )
    print(
        "Processing time so far: "
        f"total={processing_time_meta['processing_time_seconds']:.6f}s, "
        f"chunking={processing_time_meta['chunking_seconds']:.6f}s, "
        f"model_prediction={processing_time_meta['model_prediction_seconds']:.6f}s, "
        f"post_alignment={processing_time_meta['post_alignment_seconds']:.6f}s, "
        f"peak_gpu_allocated={processing_time_meta['peak_gpu_memory_allocated_mib']:.2f}MiB"
    )
    write_processing_time_meta(processing_time_meta)
    print(f"Saved processing timing: {timing_path}")

    # ------------------------------------------------------------------
    # 11. Save RRD/eval
    # ------------------------------------------------------------------
    stage("Save RRD and Eval", f"output={args.output_path}")
    save_spatial_rrd(
        output_path=Path(args.output_path),
        scene_dir=str(scene_dir),
        model_name=model_name,
        checkpoint=args.checkpoint,
        meta=meta,
        grid_meta=grid_meta,
        chunk_records=chunk_records,
        align=args.align,
        view_coordinates="RDF",
        background=[255, 255, 255],
        hide_grid=False,
        point_radius=0.0,
        camera_axis_size=0.0,
        camera_axis_radius=0.0,
        max_points_per_view=args.max_points_per_view,
        voxel_size=args.voxel_downsample,
        point_downsample=bool(args.point_downsample),
        seed=0,
        log_images=False,
        show_world_axes=True,
        world_axes_origin="scene_center",
        world_up_axis="z",
        world_axis_size=0.0,
        world_axis_size_ratio=0.12,
        world_axis_min_size=0.1,
        world_axis_up_offset_ratio=1.2,
        world_axis_radius=0.0,
        prior_policy=prior_policy,
        recenter_anchor=recenter_state,
        log_chunks_rrd=True,
        compress_eval=False,
        processing_time=processing_time_meta,
        gt_io_workers=args.scene_io_workers,
        xy_fill_unmasked=bool(args.xy_fill_unmasked),
        xy_fill_grid_size=float(args.xy_fill_grid_size),
        xy_fill_max_points_per_chunk=int(args.xy_fill_max_points_per_chunk),
    )

    # ------------------------------------------------------------------
    # 12. Optional bounded TSDF mesh extraction
    # ------------------------------------------------------------------
    stage("TSDF Mesh", f"enabled={bool(args.export_tsdf_mesh)}")
    if bool(args.export_tsdf_mesh):
        tsdf_output_dir = output_dir / "mesh"
        tsdf_summary = export_tsdf_mesh(
            chunk_records,
            tsdf_output_dir,
            voxel_size=float(args.tsdf_voxel_size),
            sdf_trunc=-1.0,
            depth_trunc=1000.0,
            min_depth=1.0e-6,
            pixel_stride=2,
            keep_clusters=50,
            min_triangles=50,
        )
        print(
            "[INFO] TSDF mesh saved: "
            f"views={tsdf_summary['num_integrated_views']}, "
            f"vertices={tsdf_summary['post_vertices']}, "
            f"triangles={tsdf_summary['post_triangles']}, "
            f"path={tsdf_summary['post_mesh']}"
        )

    # ------------------------------------------------------------------
    # 13. Deferred 3DGS bundle optimization
    # ------------------------------------------------------------------
    stage("Deferred 3DGS", f"enabled={bool(args.gsplat_refine)}")
    gsplat_summary = run_deferred_gsplat_bundle_refinement(
        chunk_records=chunk_records,
        args=args,
        device=device,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 14. Deferred DOM rendering (after 3DGS, may use optimized gaussians)
    # ------------------------------------------------------------------
    stage("Deferred DOM", f"enabled={bool(args.render_dom)}")
    dom_meta = run_deferred_dom_rendering(
        chunk_records=chunk_records,
        args=args,
        device=device,
        gsplat_summary=gsplat_summary,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    processing_time_meta = build_processing_time_meta(
        post_align_summary=post_chunk_align_summary,
    )
    write_processing_time_meta(processing_time_meta)

    chunk_cache_deleted = False
    if chunk_cache_dir.exists():
        shutil.rmtree(chunk_cache_dir)
        chunk_cache_deleted = True
        print(f"[INFO] Deleted chunk cache: {chunk_cache_dir}")

    sidecar_path = output_dir / "result.json"
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["processing_time"] = processing_time_meta
            sidecar["chunk_cache_deleted"] = bool(chunk_cache_deleted)
            sidecar["chunk_cache_kept"] = False
            sidecar_path.write_text(
                json.dumps(sidecar, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[WARN] Failed to update sidecar processing_time: {exc}")
    print(
        "Processing time summary: "
        f"chunking={processing_time_meta['chunking_seconds']:.6f}s, "
        f"model_prediction={processing_time_meta['model_prediction_seconds']:.6f}s, "
        f"post_alignment={processing_time_meta['post_alignment_seconds']:.6f}s, "
        f"total={processing_time_meta['processing_time_seconds']:.6f}s, "
        f"peak_gpu_allocated={processing_time_meta['peak_gpu_memory_allocated_mib']:.2f}MiB"
    )
    print(f"Saved final processing timing: {timing_path}")

    stage("Done", "pipeline finished")
    if run_logger.path is not None:
        print(f"[LOG] Full run log saved: {run_logger.path}")
    run_logger.close()


if __name__ == "__main__":
    main()
