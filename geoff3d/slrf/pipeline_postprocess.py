# -*- coding: utf-8 -*-
"""Final alignment, 3DGS refinement, and orthophoto post-processing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch

from geoff3d.slrf.geometry_align import estimate_chunk_pose_alignment
from geoff3d.slrf.rrd_writer import dedupe_cameras_by_stem
from geoff3d.slrf.chunk_transform import (
    compose_record_similarity,
    get_transformed_cameras,
)
from geoff3d.slrf.gsplat_bundle import (
    build_gsplat_optimization_bundles,
    collect_gsplat_bundle_inputs,
)
from geoff3d.slrf.gsplat_refine import optimize_and_save_gsplat_bundle
from geoff3d.slrf.orthodom import (
    render_orthodom_from_fused_points,
    render_orthodom_from_gsplat_summary,
)

def apply_final_global_pose_alignment(
    chunk_records: List[Dict[str, object]],
    reference_centers_by_stem: Dict[str, np.ndarray],
    mode: str,
    seed: int,
) -> Dict[str, object]:
    """Restore the final fused reconstruction to the input-pose world gauge."""
    pred_cams = dedupe_cameras_by_stem(
        [
            cam
            for record in chunk_records
            for cam in get_transformed_cameras(record)
        ]
    )
    target_stems = [str(cam.get("stem", "")) for cam in pred_cams]
    align_meta = estimate_chunk_pose_alignment(
        mode=mode,
        chunk_id=-1,
        reference_cams_by_stem=reference_centers_by_stem,
        raw_pred_cams=pred_cams,
        target_stems=target_stems,
        seed=int(seed) + 970003,
    )
    align_meta = dict(align_meta)
    align_meta["stage"] = "final_global_pose_alignment"
    align_meta["requested_mode"] = str(mode)
    if not bool(align_meta.get("valid", False)):
        return align_meta

    scale = float(align_meta["scale"])
    R = np.asarray(align_meta["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(align_meta["t"], dtype=np.float64).reshape(3)
    for record in chunk_records:
        compose_record_similarity(record, scale, R, t)
    align_meta["num_chunks_transformed"] = int(len(chunk_records))
    return align_meta


# ---------------------------------------------------------------------------
# Deferred post-processing
# ---------------------------------------------------------------------------
def run_deferred_dom_rendering(
    chunk_records: List[Dict[str, object]],
    args: SimpleNamespace,
    device: torch.device,
    gsplat_summary: Optional[Dict[str, object]] = None,
) -> Optional[Dict[str, object]]:
    """Render a full DOM after all chunks.

    When --gsplat_refine is enabled and optimized bundles exist, DOM is rendered
    from the optimized 3DGS gaussians.npz. Otherwise falls back to raw pred_maps.
    """
    if not bool(args.render_dom):
        return None

    resolved_output_path = Path(args.output_path).expanduser().resolve()
    output_dir = resolved_output_path / "orthodom"
    fused_points_path = resolved_output_path / "eval" / "pred_points.ply"

    use_optimized_gsplat = (
        bool(args.gsplat_refine)
        and gsplat_summary is not None
        and bool(gsplat_summary.get("enabled", False))
        and len(gsplat_summary.get("bundles", [])) > 0
    )

    if use_optimized_gsplat:
        print(
            "[INFO] Rendering DOM from optimized gsplat bundles: "
            f"num_bundles={len(gsplat_summary.get('bundles', []))}"
        )

        dom_meta = render_orthodom_from_gsplat_summary(
            gsplat_summary=gsplat_summary,
            fallback_chunk_records=chunk_records,
            fallback_fused_points_path=fused_points_path,
            output_dir=output_dir,
            device=device,
            dom_gsd=float(args.dom_gsd),
            dom_axes="xy",
            dom_up_axis="z",
            dom_tile_px=1024,
            dom_margin_px=32,
            dom_max_gaussians_per_tile=0,
            dom_splat_scale=2.0,
            dom_dsm_smooth_radius_px=2,
            dom_dsm_smooth_sigma=0.0,
            dom_dsm_smooth_iterations=1,
            dom_dsm_smooth_min_weight=0.05,
            dom_save_contours=False,
            dom_opacity=0.95,
            dom_gsd_stride=8,
            dom_bounds_quantile_min=0.5,
            dom_bounds_quantile_max=99.5,
            dom_padding_m=0.0,
            dom_max_pixels=160000000,
            dom_allow_large=False,
            dom_rasterize_mode="classic",
            dom_save_tiles=True,
            dom_epsg=int(args.dom_epsg) if int(args.dom_epsg) > 0 else None,
            seed=820001,
        )
    else:
        print(
            "[INFO] Rendering DOM/DSM from fused final point cloud "
            "(gsplat refinement disabled or no optimized bundles)."
        )

        dom_meta = render_orthodom_from_fused_points(
            fused_points_path=fused_points_path,
            output_dir=output_dir,
            dom_gsd=float(args.dom_gsd),
            dom_axes="xy",
            dom_up_axis="z",
            dom_splat_scale=2.0,
            dom_dsm_smooth_radius_px=2,
            dom_dsm_smooth_sigma=0.0,
            dom_dsm_smooth_iterations=1,
            dom_dsm_smooth_min_weight=0.05,
            dom_save_contours=False,
            dom_tile_px=1024,
            dom_bounds_quantile_min=0.5,
            dom_bounds_quantile_max=99.5,
            dom_padding_m=0.0,
            dom_max_pixels=160000000,
            dom_allow_large=False,
            dom_save_tiles=True,
            dom_epsg=int(args.dom_epsg) if int(args.dom_epsg) > 0 else None,
        )

    # Attach lightweight DOM meta to all records for sidecar visibility.
    lightweight = {
        "enabled": bool(dom_meta.get("enabled", False)),
        "output_dir": dom_meta.get("output_dir", str(output_dir)),
        "rgb_path": dom_meta.get("rgb_path", None),
        "alpha_path": dom_meta.get("alpha_path", None),
        "dsm_npy_path": dom_meta.get("dsm_npy_path", None),
        "dsm_elevation_png_path": dom_meta.get("dsm_elevation_png_path", None),
        "dsm_elevation_svg_path": dom_meta.get("dsm_elevation_svg_path", None),
        "dsm_contour_png_path": dom_meta.get("dsm_contour_png_path", None),
        "dsm_contour_svg_path": dom_meta.get("dsm_contour_svg_path", None),
        "geotiff_path": dom_meta.get("geotiff_path", None),
        "gsd": dom_meta.get("gsd", None),
        "image_size": dom_meta.get("image_size", None),
    }
    for record in chunk_records:
        record["dom_meta"] = lightweight

    return dom_meta


def run_deferred_gsplat_bundle_refinement(
    chunk_records: List[Dict[str, object]],
    args: SimpleNamespace,
    device: torch.device,
) -> Optional[Dict[str, object]]:
    """Run 3DGS optimization after all feed-forward chunks are predicted.

    RRD/eval outputs remain pre-gsplat. Optimized 3DGS is saved separately.
    Returns the gsplat summary dict for downstream DOM rendering.
    """
    if not args.gsplat_refine:
        return None

    resolved_output_path = Path(args.output_path).expanduser().resolve()
    output_dir = resolved_output_path / "gsplat"
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = build_gsplat_optimization_bundles(
        chunk_records=chunk_records,
        single_max_images=80,
        max_images_per_bundle=80,
        min_core_images_per_bundle=26,
    )

    print(
        "[INFO] Deferred gsplat refinement: "
        f"num_bundles={len(bundles)}, output_dir={output_dir}"
    )

    all_meta: List[Dict[str, object]] = []

    for bundle in bundles:
        bundle_name = str(bundle["name"])
        bundle_id = int(bundle["bundle_id"])

        inputs = collect_gsplat_bundle_inputs(
            chunk_records=chunk_records,
            bundle=bundle,
        )

        num_init = len(inputs["init_pred_maps"])
        num_render = len(inputs["render_rgbs"])

        if num_init == 0 or num_render == 0:
            gs_meta: Dict[str, object] = {
                "enabled": False,
                "reason": "empty init or render inputs",
                **dict(bundle),
            }
            all_meta.append(gs_meta)
            print(
                f"[gsplat:{bundle_name}] skip: "
                f"init_views={num_init}, render_views={num_render}"
            )
            continue

        gs_meta = optimize_and_save_gsplat_bundle(
            init_pred_maps=inputs["init_pred_maps"],
            init_pred_valid_masks=inputs["init_pred_valid_masks"],
            init_rgbs=inputs["init_rgbs"],
            render_views_raw=inputs["render_views_raw"],
            render_rgbs=inputs["render_rgbs"],
            render_cams=inputs["render_cams"],
            output_dir=output_dir,
            bundle_name=bundle_name,
            bundle_meta={
                **dict(bundle),
                "init_global_indices": inputs["init_global_indices"],
                "render_global_indices": inputs["render_global_indices"],
            },
            device=device,
            steps=int(args.gsplat_steps),
            max_gaussians=int(args.gsplat_max_gaussians),
            batch_views=1,
            render_scale=0.5,
            seed=910001 + bundle_id,

            # official-style splat params
            sh_degree=3,
            sh_degree_interval=1000,
            init_opacity=0.1,
            init_scale=1.0,
            ssim_lambda=0.2,
            means_lr=1.6e-4,
            scales_lr=5e-3,
            opacities_lr=5e-2,
            quats_lr=1e-3,
            sh0_lr=2.5e-3,
            shN_lr=1.25e-4,

            # strategy
            strategy_name="default",
            refine_start_iter=500,
            refine_stop_iter=max(500, int(args.gsplat_steps) - 500),
            refine_every=100,
            reset_every=3000,
            prune_opa=0.005,
            grow_grad2d=0.0002,
            grow_scale3d=0.01,
            grow_scale2d=0.05,
            prune_scale3d=0.1,
            prune_scale2d=0.15,
            absgrad=False,
            strategy_verbose=False,

            # optimization behavior
            optimize_means=True,
            optimize_pose=False,
            pose_lr=1e-5,
            pose_reg=0.0,
            opacity_reg=0.0,
            scale_reg=0.0,

            # rasterization / optimizer mode
            packed=False,
            sparse_grad=False,
            visible_adam=False,
            antialiased=True,
            random_bkgd=False,

            # logging/output
            log_every=100,
            use_tqdm=True,
            save_rendered_views=False,
            render_output_max_views=12,
            render_output_stride=1,
        )
        all_meta.append(gs_meta)

        print(
            f"[gsplat:{bundle_name}] saved: "
            f"gaussians={int(gs_meta.get('num_gaussians', 0))}, "
            f"init_views={int(gs_meta.get('num_init_core_views', 0))}, "
            f"render_views={int(gs_meta.get('num_render_views', 0))}, "
            f"loss={float(gs_meta.get('final_loss', float('nan'))):.6g}, "
            f"ply={gs_meta.get('ply_path', '<none>')}"
        )

    summary = {
        "enabled": True,
        "output_dir": str(output_dir),
        "num_bundles": int(len(all_meta)),
        "bundle_images": 80,
        "bundles": all_meta,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Saved gsplat summary: {summary_path}")

    # Attach lightweight meta back to chunk records for sidecar.
    bundle_by_record: Dict[int, List[Dict[str, object]]] = {}
    for meta in all_meta:
        for rid in meta.get("record_ids", []):
            bundle_by_record.setdefault(int(rid), []).append(
                {
                    "bundle_name": meta.get("bundle_name", meta.get("name", "")),
                    "enabled": bool(meta.get("enabled", False)),
                    "ply_path": meta.get("ply_path", None),
                    "num_gaussians": int(meta.get("num_gaussians", 0)),
                    "num_render_views": int(meta.get("num_render_views", 0)),
                }
            )

    for rid, record in enumerate(chunk_records):
        record["gsplat_meta"] = {
            "enabled": bool(args.gsplat_refine),
            "deferred": True,
            "bundles": bundle_by_record.get(int(rid), []),
        }

    return summary

