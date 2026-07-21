#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recursively convert all *bundle.json files under a root directory to .rrd files.

Usage:
    python3 convert_all_bundle_json_to_rrd.py /path/to/root
    python3 convert_all_bundle_json_to_rrd.py /path/to/root --overwrite
    python3 convert_all_bundle_json_to_rrd.py /path/to/root --pc_mode both --cams both
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


def resolve_data_path(path_str: Optional[str], bundle_json_path: Path) -> Optional[Path]:
    """
    Resolve absolute/relative data path recorded inside bundle json.
    Prefer:
      1) absolute path as-is
      2) relative to bundle json directory
    """
    if not path_str:
        return None

    p = Path(path_str)
    if p.is_absolute():
        return p

    p2 = (bundle_json_path.parent / p).resolve()
    return p2


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
    bundle_json_path: Path,
    pc_mode: str,
    custom_paths: Optional[Sequence[str]],
) -> List[Tuple[str, Path]]:
    fused = bundle.get("fused_outputs", {})
    items: List[Tuple[str, Path]] = []

    if pc_mode == "gt":
        path = resolve_data_path(fused.get("gt_ply", None), bundle_json_path)
        if path:
            items.append(("gt", path))
    elif pc_mode == "pred":
        path = resolve_data_path(fused.get("pred_ply", None), bundle_json_path)
        if path:
            items.append(("pred", path))
    elif pc_mode == "both":
        gt_path = resolve_data_path(fused.get("gt_ply", None), bundle_json_path)
        pred_path = resolve_data_path(fused.get("pred_ply", None), bundle_json_path)
        if gt_path:
            items.append(("gt", gt_path))
        if pred_path:
            items.append(("pred", pred_path))
    elif pc_mode == "custom":
        if not custom_paths:
            raise ValueError("--pc_mode custom requires --point_clouds")
        for i, p in enumerate(custom_paths):
            items.append((f"custom_{i}", Path(p).resolve()))
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
    style = camera_vis_style(cam_type)

    base_kwargs = dict(
        image_from_camera=intrinsics,
        height=h,
        width=w,
        camera_xyz=rr.ViewCoordinates.RDF,
    )

    try:
        return rr.Pinhole(
            **base_kwargs,
            color=style["color"],
            line_width=style["line_width"],
        )
    except TypeError:
        pass

    try:
        return rr.Pinhole(
            **base_kwargs,
            color=style["color"],
        )
    except TypeError:
        pass

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
# Core conversion
# -----------------------------------------------------------------------------
def save_bundle_json_to_rrd(
    bundle_json_path: Path,
    save_rrd_path: Path,
    pc_mode: str = "both",
    cams: str = "both",
    default_width: int = 640,
    default_height: int = 480,
    time_sequence: int = 0,
    app_id: str = "visualize_benchmark_ply_cams",
    point_clouds: Optional[Sequence[str]] = None,
) -> None:
    bundle = load_json(bundle_json_path)

    save_rrd_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init(app_id, spawn=False)
    rr.save(str(save_rrd_path))

    try:
        rr.set_time("stable_time", sequence=time_sequence)
    except AttributeError:
        rr.set_time_sequence("stable_time", time_sequence)

    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # ------------------------------------------------------------------
    # Point clouds
    # ------------------------------------------------------------------
    pc_items = collect_point_cloud_paths(
        bundle=bundle,
        bundle_json_path=bundle_json_path,
        pc_mode=pc_mode,
        custom_paths=point_clouds,
    )
    if not pc_items and cams == "none":
        raise RuntimeError("Nothing to visualize: both point clouds and cameras are disabled.")

    for label, path in pc_items:
        if not path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {path}")

        pts, colors = load_ply_xyzrgb(path)
        if label == "gt":
            entity = "world/gt/points"
        elif label == "pred":
            entity = "world/pred/points"
        else:
            entity = f"world/{label}/points"

        log_point_cloud_to_rerun(entity, pts, colors)

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------
    if cams != "none":
        cam_centers = {"gt": [], "pred": []}
        cam_labels = {"gt": [], "pred": []}

        for cam_type, view_idx, instance, c2w, intrinsics in iter_cameras(bundle, cams):
            if c2w is None:
                print(f"[Warn] Skip {cam_type} camera for view {view_idx}: invalid c2w")
                continue

            base = f"world/{cam_type}/cameras/view_{view_idx:03d}"
            log_camera_to_rerun(
                entity_base=base,
                c2w=c2w,
                intrinsics=intrinsics,
                default_w=default_width,
                default_h=default_height,
                cam_type=cam_type,
            )

            center = c2w[:3, 3].astype(np.float32)
            cam_centers[cam_type].append(center)
            cam_labels[cam_type].append(f"{cam_type}_{view_idx:03d}")

        for cam_type in ("gt", "pred"):
            log_camera_track_and_centers(
                cam_type=cam_type,
                centers=cam_centers[cam_type],
                labels=cam_labels[cam_type],
            )


# -----------------------------------------------------------------------------
# Batch traversal
# -----------------------------------------------------------------------------
def find_all_bundle_jsons(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*bundle.json"):
        if p.is_file():
            files.append(p)
    return sorted(files)


def convert_all(
    root: Path,
    overwrite: bool,
    pc_mode: str,
    cams: str,
    default_width: int,
    default_height: int,
    app_id: str,
) -> None:
    bundle_jsons = find_all_bundle_jsons(root)

    print(f"[Info] Root: {root}")
    print(f"[Info] Found {len(bundle_jsons)} bundle json files")

    if len(bundle_jsons) == 0:
        return

    ok = 0
    skipped = 0
    failed = 0

    for i, bundle_json in enumerate(bundle_jsons, start=1):
        rrd_path = bundle_json.with_suffix(".rrd")

        if rrd_path.exists() and not overwrite:
            print(f"[{i}/{len(bundle_jsons)}] Skip existing: {rrd_path}")
            skipped += 1
            continue

        try:
            save_bundle_json_to_rrd(
                bundle_json_path=bundle_json,
                save_rrd_path=rrd_path,
                pc_mode=pc_mode,
                cams=cams,
                default_width=default_width,
                default_height=default_height,
                time_sequence=0,
                app_id=app_id,
            )
            print(f"[{i}/{len(bundle_jsons)}] OK: {bundle_json} -> {rrd_path}")
            ok += 1
        except Exception as e:
            print(f"[{i}/{len(bundle_jsons)}] FAIL: {bundle_json}")
            print(f"    Reason: {e}")
            failed += 1

    print("\n[Summary]")
    print(f"  Success: {ok}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed : {failed}")


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively convert all *bundle.json files under a root directory to .rrd"
    )
    parser.add_argument(
        "--root",
        default="experiments/mapanything/benchmarking/dense_16_view/mapa_24v/ 30 @ UseGeoWAI_fused_ply",
        type=str,
        help="Root directory to recursively search for *bundle.json files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .rrd files",
    )
    parser.add_argument(
        "--pc_mode",
        type=str,
        default="both",
        choices=["gt", "pred", "both", "none"],
        help="Which point clouds to visualize",
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
        help="Fallback image width for camera frustums",
    )
    parser.add_argument(
        "--default_height",
        type=int,
        default=480,
        help="Fallback image height for camera frustums",
    )
    parser.add_argument(
        "--app_id",
        type=str,
        default="visualize_benchmark_ply_cams",
        help="Rerun application ID",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    convert_all(
        root=root,
        overwrite=args.overwrite,
        pc_mode=args.pc_mode,
        cams=args.cams,
        default_width=args.default_width,
        default_height=args.default_height,
        app_id=args.app_id,
    )


if __name__ == "__main__":
    main()
    