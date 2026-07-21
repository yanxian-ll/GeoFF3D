#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Headless Rerun exporter for benchmark bundle json + fused ply point clouds.

Use case:
- Run on a remote server without desktop/GUI.
- Save visualization data to a .rrd file.
- Copy the .rrd file back to your local machine and open it with the Rerun viewer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rerun as rr

try:
    import open3d as o3d
except Exception as e:
    raise ImportError(
        "This script requires open3d to read .ply point clouds. Please install open3d."
    ) from e


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_np(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64)


def validate_c2w(mat: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mat is None:
        return None
    if mat.shape != (4, 4):
        return None
    return mat


def infer_hw_from_intrinsics(
    K: Optional[np.ndarray], default_w: int, default_h: int
) -> Tuple[int, int]:
    if K is None or K.shape != (3, 3):
        return default_h, default_w

    cx = float(K[0, 2])
    cy = float(K[1, 2])

    w = int(round(max(default_w, 2.0 * cx))) if cx > 1e-6 else default_w
    h = int(round(max(default_h, 2.0 * cy))) if cy > 1e-6 else default_h
    return h, w


# -----------------------------------------------------------------------------
# PLY helpers
# -----------------------------------------------------------------------------
def load_ply_xyzrgb(path: str | Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise RuntimeError(f"Failed to read point cloud or empty file: {path}")

    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = None
    if pcd.has_colors():
        cols_f = np.asarray(pcd.colors, dtype=np.float32)
        cols = np.clip(cols_f * 255.0, 0, 255).astype(np.uint8)
    return pts, cols


def collect_point_cloud_paths(
    bundle: dict,
    pc_mode: str,
    custom_paths: Optional[Sequence[str]],
) -> List[Tuple[str, str]]:
    fused = bundle.get("fused_outputs", {})
    items: List[Tuple[str, str]] = []

    if pc_mode == "gt":
        path = fused.get("gt_ply", None)
        if path:
            items.append(("gt", path))
    elif pc_mode == "pred":
        path = fused.get("pred_ply", None)
        if path:
            items.append(("pred", path))
    elif pc_mode == "both":
        gt_path = fused.get("gt_ply", None)
        pred_path = fused.get("pred_ply", None)
        if gt_path:
            items.append(("gt", gt_path))
        if pred_path:
            items.append(("pred", pred_path))
    elif pc_mode == "custom":
        if not custom_paths:
            raise ValueError("--pc_mode custom requires --point_clouds")
        for i, p in enumerate(custom_paths):
            items.append((f"custom_{i}", p))
    elif pc_mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported pc_mode: {pc_mode}")

    return items


# -----------------------------------------------------------------------------
# Camera helpers
# -----------------------------------------------------------------------------
def iter_cameras(
    bundle: dict, which: str
) -> Iterable[Tuple[str, int, str, Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Yields:
      (cam_type, view_idx, instance, c2w, intrinsics)
    cam_type in {"gt", "pred"}
    """
    views = bundle.get("views", [])
    for view in views:
        view_idx = int(view.get("view_idx", -1))
        instance = str(view.get("instance", ""))

        if which in ("gt", "both"):
            gt_cam = view.get("gt_cam", {})
            c2w = validate_c2w(as_np(gt_cam.get("c2w", None)))
            intr = as_np(gt_cam.get("intrinsics", None))
            yield "gt", view_idx, instance, c2w, intr

        if which in ("pred", "both"):
            pred_cam = view.get("pred_cam", {})
            c2w = validate_c2w(as_np(pred_cam.get("c2w", None)))
            intr = as_np(pred_cam.get("intrinsics", None))
            yield "pred", view_idx, instance, c2w, intr


def camera_vis_style(cam_type: str) -> dict:
    if cam_type == "gt":
        return {
            "color": np.array([60, 200, 120], dtype=np.uint8),   # green
            "line_width": 0.0035,
            "point_radius": 0.030,
            "track_radius": 0.006,
            "axis_length": 0.20,
            "name": "GT",
        }
    elif cam_type == "pred":
        return {
            "color": np.array([255, 120, 40], dtype=np.uint8),   # orange
            "line_width": 0.0075,
            "point_radius": 0.045,
            "track_radius": 0.010,
            "axis_length": 0.28,
            "name": "PRED",
        }
    else:
        return {
            "color": np.array([180, 180, 180], dtype=np.uint8),
            "line_width": 0.004,
            "point_radius": 0.030,
            "track_radius": 0.006,
            "axis_length": 0.20,
            "name": cam_type.upper(),
        }
    
# -----------------------------------------------------------------------------
# Rerun logging
# -----------------------------------------------------------------------------
def log_point_cloud_to_rerun(
    entity_path: str,
    pts: np.ndarray,
    colors: Optional[np.ndarray],
) -> None:
    if colors is None:
        rr.log(entity_path, rr.Points3D(positions=pts))
    else:
        rr.log(entity_path, rr.Points3D(positions=pts, colors=colors))


def log_transform3d_compat(entity_base: str, c2w: np.ndarray, axis_length: float) -> None:
    """
    Compatibility wrapper:
    - Newer rerun may support axis_length
    - Older rerun may not
    """
    try:
        rr.log(
            entity_base,
            rr.Transform3D(
                translation=c2w[:3, 3],
                mat3x3=c2w[:3, :3],
                axis_length=axis_length,
            ),
        )
    except TypeError:
        rr.log(
            entity_base,
            rr.Transform3D(
                translation=c2w[:3, 3],
                mat3x3=c2w[:3, :3],
            ),
        )


def make_pinhole_compat(
    intrinsics: np.ndarray,
    h: int,
    w: int,
    cam_type: str,
):
    """
    Compatibility wrapper for rr.Pinhole across rerun versions.

    Important:
    - Do NOT manually set image_plane_distance.
    - This keeps the frustum size close to the original source behavior.
    """
    style = camera_vis_style(cam_type)

    base_kwargs = dict(
        image_from_camera=intrinsics,
        height=h,
        width=w,
        camera_xyz=rr.ViewCoordinates.RDF,
    )

    # newer rerun: may support color + line_width
    try:
        return rr.Pinhole(
            **base_kwargs,
            color=style["color"],
            line_width=style["line_width"],
        )
    except TypeError:
        pass

    # some versions may support color only
    try:
        return rr.Pinhole(
            **base_kwargs,
            color=style["color"],
        )
    except TypeError:
        pass

    # oldest compatible behavior = closest to your original source code
    return rr.Pinhole(**base_kwargs)


def log_camera_to_rerun(
    entity_base: str,
    c2w: np.ndarray,
    intrinsics: Optional[np.ndarray],
    default_w: int,
    default_h: int,
    cam_type: str,
) -> None:
    style = camera_vis_style(cam_type)

    log_transform3d_compat(
        entity_base=entity_base,
        c2w=c2w,
        axis_length=style["axis_length"],
    )

    if intrinsics is not None and intrinsics.shape == (3, 3):
        h, w = infer_hw_from_intrinsics(
            intrinsics,
            default_w=default_w,
            default_h=default_h,
        )
        rr.log(
            f"{entity_base}/pinhole",
            make_pinhole_compat(
                intrinsics=intrinsics,
                h=h,
                w=w,
                cam_type=cam_type,
            ),
        )


def log_camera_track_and_centers(
    cam_type: str,
    centers: Sequence[np.ndarray],
    labels: Sequence[str],
) -> None:
    if len(centers) == 0:
        return

    style = camera_vis_style(cam_type)
    centers_np = np.asarray(centers, dtype=np.float32)
    colors_np = np.repeat(style["color"][None, :], len(centers_np), axis=0)

    rr.log(
        f"world/{cam_type}/camera_centers",
        rr.Points3D(
            positions=centers_np,
            colors=colors_np,
            radii=style["point_radius"],
            labels=list(labels),
        ),
    )

    if len(centers_np) >= 2:
        rr.log(
            f"world/{cam_type}/camera_track",
            rr.LineStrips3D(
                strips=[centers_np],
                colors=[style["color"]],
                radii=style["track_radius"],
                labels=[f"{style['name']} trajectory"],
            ),
        )


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export benchmark bundle json visualization into a headless RRD file"
    )
    parser.add_argument("bundle_json", type=str, help="Path to *_bundle.json")
    parser.add_argument(
        "--pc_mode",
        type=str,
        default="both",
        choices=["gt", "pred", "both", "custom", "none"],
        help="Which point clouds to visualize",
    )
    parser.add_argument(
        "--point_clouds",
        type=str,
        nargs="*",
        default=None,
        help="Custom point clouds when --pc_mode custom",
    )
    parser.add_argument(
        "--cams",
        type=str,
        default="both",
        choices=["gt", "pred", "both", "none"],
        help="Which camera poses to visualize",
    )
    parser.add_argument(
        "--default_width",
        type=int,
        default=640,
        help="Fallback image width for camera frustums when intrinsics cannot infer image size",
    )
    parser.add_argument(
        "--default_height",
        type=int,
        default=480,
        help="Fallback image height for camera frustums when intrinsics cannot infer image size",
    )
    parser.add_argument(
        "--time_sequence",
        type=int,
        default=0,
        help="Rerun stable_time sequence value",
    )
    parser.add_argument(
        "--save_rrd",
        type=str,
        default=None,
        help="Output .rrd file path",
    )
    parser.add_argument(
        "--app_id",
        type=str,
        default="visualize_benchmark_ply_cams",
        help="Rerun application ID",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    bundle = load_json(args.bundle_json)

    if args.save_rrd is None:
        args.save_rrd = str(Path(args.bundle_json).with_suffix(".rrd"))

    save_path = Path(args.save_rrd)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init(args.app_id, spawn=False)
    rr.save(str(save_path))

    # old/new rerun timeline compatibility
    try:
        rr.set_time("stable_time", sequence=args.time_sequence)
    except AttributeError:
        rr.set_time_sequence("stable_time", args.time_sequence)

    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # ------------------------------------------------------------------
    # Point clouds
    # ------------------------------------------------------------------
    pc_items = collect_point_cloud_paths(bundle, args.pc_mode, args.point_clouds)
    if not pc_items and args.cams == "none":
        raise RuntimeError("Nothing to visualize: both point clouds and cameras are disabled.")

    for label, path in pc_items:
        pts, colors = load_ply_xyzrgb(path)
        if label == "gt":
            entity = "world/gt/points"
        elif label == "pred":
            entity = "world/pred/points"
        else:
            entity = f"world/{label}/points"

        log_point_cloud_to_rerun(entity, pts, colors)
        print(f"[Info] Logged point cloud [{label}] from: {path}")

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------
    if args.cams != "none":
        cam_centers = {"gt": [], "pred": []}
        cam_labels = {"gt": [], "pred": []}

        for cam_type, view_idx, instance, c2w, intrinsics in iter_cameras(bundle, args.cams):
            if c2w is None:
                print(f"[Warn] Skip {cam_type} camera for view {view_idx}: invalid c2w")
                continue

            base = f"world/{cam_type}/cameras/view_{view_idx:03d}"
            log_camera_to_rerun(
                entity_base=base,
                c2w=c2w,
                intrinsics=intrinsics,
                default_w=args.default_width,
                default_h=args.default_height,
                cam_type=cam_type,
            )

            center = c2w[:3, 3].astype(np.float32)
            cam_centers[cam_type].append(center)
            cam_labels[cam_type].append(f"{cam_type}_{view_idx:03d}")

            print(f"[Info] Logged {cam_type} camera view {view_idx}: {instance}")

        for cam_type in ("gt", "pred"):
            log_camera_track_and_centers(
                cam_type=cam_type,
                centers=cam_centers[cam_type],
                labels=cam_labels[cam_type],
            )

    print(f"[Done] Saved RRD to: {save_path}")
    print("[Next] Copy this .rrd file to your local machine and open it with:")
    print(f"       rerun {save_path}")


if __name__ == "__main__":
    main()



"""

python3 scripts/visualize_benchmark_ply_cams.py \
    "experiments/mapanything/benchmarking_focal_ambiguilty/da3/ 64 @ A3DSynLargeFAWAI_fused_ply/e1b883efa2b8768cfab20347/e1b883efa2b8768cfab20347_set000_bundle.json" \
    --save_rrd "experiments/mapanything/benchmarking_focal_ambiguilty/da3/rrd/e1b883efa2b8768cfab20347.rrd"

python3 scripts/visualize_benchmark_ply_cams.py \
    "experiments/mapanything/benchmarking_focal_ambiguilty/da3/ 64 @ A3DSynLargeFAWAI_fused_ply/71040e8faffc08ba7082b029/71040e8faffc08ba7082b029_set000_bundle.json" \
    --save_rrd "experiments/mapanything/benchmarking_focal_ambiguilty/da3/rrd/71040e8faffc08ba7082b029.rrd"


python3 scripts/visualize_benchmark_ply_cams.py \
    "experiments/mapanything/benchmarking_focal_ambiguilty/da3/ 64 @ A3DSynLargeFAWAI_fused_ply/fa73d296a111a7e3e973f237/fa73d296a111a7e3e973f237_set000_bundle.json" \
    --save_rrd "experiments/mapanything/benchmarking_focal_ambiguilty/da3/rrd/fa73d296a111a7e3e973f237.rrd"

python3 scripts/visualize_benchmark_ply_cams.py \
    "experiments/mapanything/benchmarking_focal_ambiguilty/da3/ 64 @ A3DSynLargeFAWAI_fused_ply/23dcba4dffe0c6bf0f59042e/23dcba4dffe0c6bf0f59042e_set000_bundle.json" \
    --save_rrd "experiments/mapanything/benchmarking_focal_ambiguilty/da3/rrd/23dcba4dffe0c6bf0f59042e.rrd"

python3 scripts/visualize_benchmark_ply_cams.py \
    "experiments/mapanything/benchmarking_focal_ambiguilty/da3/ 64 @ A3DSynLargeFAWAI_fused_ply/179e2063c562a60e3308d99a/179e2063c562a60e3308d99a_set000_bundle.json" \
    --save_rrd "experiments/mapanything/benchmarking_focal_ambiguilty/da3/rrd/179e2063c562a60e3308d99a.rrd"

python3 scripts/visualize_benchmark_ply_cams.py \
    "experiments/mapanything/benchmarking_focal_ambiguilty/da3/ 64 @ A3DSynLargeFAWAI_fused_ply/647ac219f9bf5eb6154d0f2b/647ac219f9bf5eb6154d0f2b_set000_bundle.json" \
    --save_rrd "experiments/mapanything/benchmarking_focal_ambiguilty/da3/rrd/647ac219f9bf5eb6154d0f2b.rrd"

python3 scripts/visualize_benchmark_ply_cams.py \
    "experiments/mapanything/benchmarking_focal_ambiguilty/da3/ 64 @ A3DSynLargeFAWAI_fused_ply/3b6bb1e3910ef5b714da4f28/3b6bb1e3910ef5b714da4f28_set000_bundle.json" \
    --save_rrd "experiments/mapanything/benchmarking_focal_ambiguilty/da3/rrd/3b6bb1e3910ef5b714da4f28.rrd"

python3 scripts/visualize_benchmark_ply_cams.py \
    "experiments/mapanything/benchmarking_focal_ambiguilty/da3/ 64 @ A3DSynLargeFAWAI_fused_ply/1cb8e5e8baf385a3cb2dcf9a/1cb8e5e8baf385a3cb2dcf9a_set000_bundle.json" \
    --save_rrd "experiments/mapanything/benchmarking_focal_ambiguilty/da3/rrd/1cb8e5e8baf385a3cb2dcf9a.rrd"

"""