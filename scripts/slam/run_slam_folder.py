#!/usr/bin/env python
"""Run Generic Feed-forward SLAM on a folder of images."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from slam.backend.no_opt_backend import NoOptBackend
from slam.backend.se3_pose_graph import SE3PoseGraphBackend
from slam.backend.sensor_fusion_graph import SensorFusionGraphBackend
from slam.backend.sim3_pose_graph import Sim3PoseGraphBackend
from slam.backend.sl4_submap_graph import SL4SubmapGraphBackend
from slam.core.config import load_config
from slam.core.generic_slam import GenericFeedForwardSLAM
from slam.core.registry import build_adapter
from slam.frontend.chunk_manager import ChunkManager
from slam.frontend.overlap_manager import OverlapManager
from slam.geometry.coordinate_normalizer import CoordinateNormalizer
from slam.io.export_outputs import save_slam_outputs
from slam.io.folder_dataset import load_folder_frames
from slam.loop.factor_adapter import LoopFactorAdapter
from slam.loop.geometric_verifier import PnPRansacVerifier, PointCloudRegistrationVerifier, PoseDistanceVerifier
from slam.loop.loop_manager import GenericLoopClosureManager
from slam.loop.model_pair_verifier import ModelPairReprojectionVerifier
from slam.loop.retrieval import TemporalDistanceLoopRetrieval
from slam.mapping.pointcloud_map import PointCloudMap


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--priors", default=None)
    parser.add_argument("--cams_dir", default=None)
    parser.add_argument("--depth_path", default=None)
    parser.add_argument("--mask_path", default=None)
    parser.add_argument("--model_checkpoint", default=None)
    parser.add_argument("--enable_loop_closure", action="store_true")
    parser.add_argument(
        "--loop_verifier",
        choices=["pose_distance", "model_pair_reprojection", "pointcloud_registration", "pnp_ransac"],
        default=None,
    )
    parser.add_argument("--loop_distance_threshold", type=float, default=None)
    parser.add_argument("--loop_min_temporal_gap", type=int, default=None)
    parser.add_argument("--loop_max_translation_error", type=float, default=None)
    parser.add_argument("--loop_max_rmse", type=float, default=None)
    parser.add_argument("--loop_max_correspondence_distance", type=float, default=None)
    parser.add_argument("--loop_min_inliers", type=int, default=None)
    parser.add_argument("--loop_min_inlier_ratio", type=float, default=None)
    parser.add_argument("--loop_transform_type", choices=["se3", "sim3"], default=None)
    parser.add_argument("--loop_max_reprojection_error", type=float, default=None)
    parser.add_argument("--loop_min_correspondences", type=int, default=None)
    parser.add_argument("--loop_ransac_iterations", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_npz", default=None)
    parser.add_argument("--output_rrd", default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--resize", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--min_chunk_size", type=int, default=None)
    parser.add_argument("--max_rrd_points", type=int, default=None)
    parser.add_argument("--rrd_seed", type=int, default=None)
    parser.add_argument("--max_colmap_points", type=int, default=None)
    parser.add_argument("--colmap_seed", type=int, default=None)
    parser.add_argument("--log_images", action="store_true")
    parser.add_argument(
        "--backend_type",
        default=None,
        choices=["no_opt", "se3_pose_graph", "sim3_pose_graph", "sensor_fusion_graph", "sl4_submap_graph"],
    )
    args = parser.parse_args(argv)

    # Load config once, then apply CLI overrides so every later stage reads the
    # same resolved settings.
    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)

    input_cfg = cfg.get("input", {})
    image_dir = input_cfg.get("image_dir")
    if image_dir is None:
        raise ValueError("Provide --image_dir or input.image_dir in the config")

    # Load images plus optional camera/depth/mask priors into SlamFrame objects.
    frames = load_folder_frames(
        image_dir=image_dir,
        priors_path=input_cfg.get("priors"),
        max_frames=input_cfg.get("max_frames") or input_cfg.get("num_frames"),
        stride=input_cfg.get("stride", 1),
        resize=input_cfg.get("resize"),
        cams_dir=input_cfg.get("cams_dir"),
        depth_path=input_cfg.get("depth_path"),
        mask_path=input_cfg.get("mask_path"),
    )

    # Build the model adapter through the registry; model-specific logic stays
    # inside the adapter layer.
    adapter = build_adapter(cfg["model"]["name"], {k: v for k, v in cfg["model"].items() if k != "name"})
    frontend = cfg["frontend"]
    backend_cfg = cfg.get("backend", {})
    alignment_cfg = cfg.get("alignment", {})
    loop_cfg = cfg.get("loop_closure", {})

    # Wire frontend chunking, backend optimization, and mapping into the generic
    # feed-forward SLAM pipeline.
    overlap_manager = OverlapManager()
    slam = GenericFeedForwardSLAM(
        adapter=adapter,
        chunk_manager=ChunkManager(
            chunk_size=frontend["chunk_size"],
            overlap=frontend.get("overlap", 1),
            min_chunk_size=frontend.get("min_chunk_size", 4),
        ),
        normalizer=_build_normalizer(alignment_cfg, overlap_manager),
        backend=_build_backend(backend_cfg.get("type", "no_opt")),
        mapping=PointCloudMap(),
        overlap_manager=overlap_manager,
        loop_manager=_build_loop_manager(loop_cfg, backend_cfg.get("type", "no_opt")),
    )

    # Run reconstruction, backend optimization, and map integration.
    result = slam.run(frames)
    print(f"input_frames: {len(frames)}")
    print(f"trajectory_frames: {len(result['trajectory'])}")
    print(f"map_summary: {result['map_summary']}")
    print(f"backend_diagnostics: {result['backend_diagnostics']}")

    # Export debug/evaluation artifacts requested by the resolved config.
    export_cfg = cfg.get("export", {})
    exported = save_slam_outputs(
        output_dir=export_cfg.get("output_dir"),
        frames=frames,
        result=result,
        output_npz=export_cfg.get("output_npz"),
        output_rrd=export_cfg.get("output_rrd"),
        max_rrd_points=export_cfg.get("max_rrd_points", 800000),
        rrd_seed=export_cfg.get("rrd_seed", 0),
        log_images=export_cfg.get("log_images", False),
        max_colmap_points=export_cfg.get("max_colmap_points", 200000),
        colmap_seed=export_cfg.get("colmap_seed", 0),
    )
    for name, path in sorted(exported.items()):
        print(f"saved_{name}: {path}")
    return 0


def _apply_cli_overrides(cfg, args):
    input_cfg = cfg.setdefault("input", {})
    frontend_cfg = cfg.setdefault("frontend", {})
    backend_cfg = cfg.setdefault("backend", {})
    export_cfg = cfg.setdefault("export", {})

    _set_if_not_none(input_cfg, "image_dir", args.image_dir)
    _set_if_not_none(input_cfg, "priors", args.priors)
    _set_if_not_none(input_cfg, "cams_dir", args.cams_dir)
    _set_if_not_none(input_cfg, "depth_path", args.depth_path)
    _set_if_not_none(input_cfg, "mask_path", args.mask_path)
    _set_if_not_none(input_cfg, "max_frames", args.max_frames)
    _set_if_not_none(input_cfg, "resize", args.resize)
    _apply_model_checkpoint_override(cfg, args.model_checkpoint)

    _set_if_not_none(frontend_cfg, "chunk_size", args.chunk_size)
    _set_if_not_none(frontend_cfg, "overlap", args.overlap)
    _set_if_not_none(frontend_cfg, "min_chunk_size", args.min_chunk_size)

    _set_if_not_none(backend_cfg, "type", args.backend_type)
    loop_cfg = cfg.setdefault("loop_closure", {})
    if args.enable_loop_closure:
        loop_cfg["enabled"] = True
    _set_if_not_none(loop_cfg, "verifier", args.loop_verifier)
    _set_if_not_none(loop_cfg, "distance_threshold", args.loop_distance_threshold)
    _set_if_not_none(loop_cfg, "min_temporal_gap", args.loop_min_temporal_gap)
    _set_if_not_none(loop_cfg, "max_translation_error", args.loop_max_translation_error)
    _set_if_not_none(loop_cfg, "max_rmse", args.loop_max_rmse)
    _set_if_not_none(loop_cfg, "max_correspondence_distance", args.loop_max_correspondence_distance)
    _set_if_not_none(loop_cfg, "min_inliers", args.loop_min_inliers)
    _set_if_not_none(loop_cfg, "min_inlier_ratio", args.loop_min_inlier_ratio)
    _set_if_not_none(loop_cfg, "transform_type", args.loop_transform_type)
    _set_if_not_none(loop_cfg, "max_reprojection_error", args.loop_max_reprojection_error)
    _set_if_not_none(loop_cfg, "min_correspondences", args.loop_min_correspondences)
    _set_if_not_none(loop_cfg, "ransac_iterations", args.loop_ransac_iterations)

    _set_if_not_none(export_cfg, "output_dir", args.output_dir)
    _set_if_not_none(export_cfg, "output_npz", args.output_npz)
    _set_if_not_none(export_cfg, "output_rrd", args.output_rrd)
    _set_if_not_none(export_cfg, "max_rrd_points", args.max_rrd_points)
    _set_if_not_none(export_cfg, "rrd_seed", args.rrd_seed)
    _set_if_not_none(export_cfg, "max_colmap_points", args.max_colmap_points)
    _set_if_not_none(export_cfg, "colmap_seed", args.colmap_seed)
    if args.log_images:
        export_cfg["log_images"] = True


def _set_if_not_none(mapping, key, value):
    if value is not None:
        mapping[key] = value


def _apply_model_checkpoint_override(cfg, checkpoint):
    if checkpoint is None:
        return
    model_cfg = cfg.setdefault("model", {})
    model_name = model_cfg.get("model_name") or model_cfg.get("name")
    overrides = list(model_cfg.get("hydra_overrides") or [])
    checkpoint_path = Path(checkpoint)
    pretrained_location = checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path

    if model_name == "mapanything_v1" or model_cfg.get("name") == "mapanything":
        overrides = _replace_or_append_override(overrides, "model.pretrained", str(checkpoint_path))
    elif model_name == "vggt_omega" or model_cfg.get("name") == "vggt_omega":
        overrides = _replace_or_append_override(
            overrides,
            "model.model_config.checkpoint_path",
            str(checkpoint_path),
        )
    else:
        overrides = _replace_or_append_override(
            overrides,
            "model.model_config.pretrained_model_name_or_path",
            str(pretrained_location),
        )
    model_cfg["hydra_overrides"] = overrides


def _replace_or_append_override(overrides, key, value):
    prefix = f"{key}="
    new_override = f"{key}={value}"
    return [new_override if str(item).startswith(prefix) else item for item in overrides] + (
        [] if any(str(item).startswith(prefix) for item in overrides) else [new_override]
    )


def _build_normalizer(alignment_cfg, overlap_manager):
    return CoordinateNormalizer(
        overlap_manager=overlap_manager,
        alignment_mode=alignment_cfg.get("mode", "auto"),
        use_world_translation=alignment_cfg.get("use_world_translation", True),
        use_world_rotation=alignment_cfg.get("use_world_rotation", False),
        prefer_overlap_points=alignment_cfg.get("prefer_overlap_points", True),
        refine_world_with_overlap=alignment_cfg.get("refine_world_with_overlap", False),
        min_overlap_points=alignment_cfg.get("min_overlap_points", 100),
        max_overlap_points=alignment_cfg.get("max_overlap_points", 20000),
    )


def _build_backend(backend_type):
    if backend_type == "no_opt":
        return NoOptBackend()
    if backend_type == "se3_pose_graph":
        return SE3PoseGraphBackend()
    if backend_type == "sim3_pose_graph":
        return Sim3PoseGraphBackend()
    if backend_type == "sensor_fusion_graph":
        return SensorFusionGraphBackend()
    if backend_type == "sl4_submap_graph":
        return SL4SubmapGraphBackend()
    raise ValueError(f"Unknown backend type: {backend_type}")


def _build_loop_manager(loop_cfg, backend_type):
    if not loop_cfg.get("enabled", False):
        return None
    retrieval = TemporalDistanceLoopRetrieval(
        distance_threshold=loop_cfg.get("distance_threshold", 1.0),
        min_temporal_gap=loop_cfg.get("min_temporal_gap", 20),
        max_candidates_per_query=loop_cfg.get("max_candidates_per_query", 1),
    )
    verifier_type = loop_cfg.get("verifier", "pose_distance")
    if verifier_type == "pose_distance":
        verifier = PoseDistanceVerifier(
            max_translation_error=loop_cfg.get("max_translation_error", 1.0),
            max_rotation_angle_deg=loop_cfg.get("max_rotation_angle_deg", 180.0),
            min_score=loop_cfg.get("min_score", 0.0),
            noise_sigma=loop_cfg.get("noise_sigma", 0.05),
        )
    elif verifier_type == "model_pair_reprojection":
        verifier = ModelPairReprojectionVerifier(
            max_point_error=loop_cfg.get("max_point_error", 0.10),
            min_inlier_ratio=loop_cfg.get("min_inlier_ratio", 0.20),
            min_valid_projections=loop_cfg.get("min_valid_projections", 16),
            min_valid_projection_ratio=loop_cfg.get("min_valid_projection_ratio", 0.05),
            max_median_point_error=loop_cfg.get("max_median_point_error"),
            noise_sigma=loop_cfg.get("noise_sigma", 0.05),
        )
    elif verifier_type == "pointcloud_registration":
        verifier = PointCloudRegistrationVerifier(
            max_rmse=loop_cfg.get("max_rmse", 0.10),
            max_correspondence_distance=loop_cfg.get("max_correspondence_distance", 0.10),
            min_inliers=loop_cfg.get("min_inliers", 16),
            min_inlier_ratio=loop_cfg.get("min_inlier_ratio", 0.20),
            transform_type=loop_cfg.get("transform_type", "se3"),
            max_points=loop_cfg.get("max_points", 5000),
            use_open3d=loop_cfg.get("use_open3d", True),
            noise_sigma=loop_cfg.get("noise_sigma", 0.05),
            pair_chunk_id=loop_cfg.get("pair_chunk_id", -2),
        )
    elif verifier_type == "pnp_ransac":
        verifier = PnPRansacVerifier(
            max_reprojection_error=loop_cfg.get("max_reprojection_error", 3.0),
            min_inliers=loop_cfg.get("min_inliers", 12),
            min_inlier_ratio=loop_cfg.get("min_inlier_ratio", 0.30),
            min_correspondences=loop_cfg.get("min_correspondences", 6),
            ransac_iterations=loop_cfg.get("ransac_iterations", 128),
            confidence=loop_cfg.get("confidence", 0.999),
            use_opencv=loop_cfg.get("pnp_use_opencv", True),
            random_seed=loop_cfg.get("random_seed", 0),
            noise_sigma=loop_cfg.get("noise_sigma", 0.05),
        )
    else:
        raise ValueError(f"Unknown loop verifier: {verifier_type}")
    return GenericLoopClosureManager(
        retrieval=retrieval,
        verifier=verifier,
        factor_adapter=LoopFactorAdapter(backend_type, noise_sigma=loop_cfg.get("noise_sigma", 0.05)),
        max_iterations=loop_cfg.get("max_iterations", 1),
    )


if __name__ == "__main__":
    raise SystemExit(main())
