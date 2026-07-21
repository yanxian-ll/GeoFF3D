#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Gaussian SLAM systems on a MapAnything scene and save a Rerun .rrd.

Supported methods:
  - on-the-fly-nvs
  - artdeco
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_vggt_slam_to_rrd import (
    IMAGE_EXTS,
    align_prediction_to_gt_pose_sim3,
    build_env as build_base_env,
    collect_stem_to_path,
    json_safe,
    load_gt_artifacts,
    load_json_file,
    load_point_cloud_file,
    materialize_images,
    sample_points,
    sanitize_name,
    save_final_eval_outputs,
    select_images,
    write_rrd,
)


METHODS = {
    "on-the-fly-nvs": {
        "repo": "third_party/on-the-fly-nvs",
        "script": "train.py",
        "display": "On-the-Fly NVS",
    },
    "artdeco": {
        "repo": "third_party/ARTDECO",
        "script": "run_system.py",
        "display": "ARTDECO",
    },
}

SH_C0 = 0.28209479177387814


def normalize_method(method: str) -> str:
    key = str(method).strip().lower().replace("_", "-")
    aliases = {
        "otf-nvs": "on-the-fly-nvs",
        "onthefly-nvs": "on-the-fly-nvs",
        "on-the-fly": "on-the-fly-nvs",
        "on-the-fly-nvs": "on-the-fly-nvs",
        "artdeco": "artdeco",
    }
    key = aliases.get(key, key)
    if key not in METHODS:
        raise ValueError(f"Unknown --method {method}. Use on-the-fly-nvs or artdeco.")
    return key


def resolve_method_root(repo_root: Path, method: str) -> Path:
    return (repo_root / str(METHODS[method]["repo"])).resolve()


def build_env(args: argparse.Namespace, method_root: Path) -> Dict[str, str]:
    env = build_base_env(args, method_root=method_root)
    repo_root = Path(__file__).resolve().parents[1]
    extra_parts = [str(repo_root)]
    if args.method == "artdeco":
        extra_parts.extend(
            [
                str(method_root / "VSLAM" / "thirdparty" / "mast3r"),
                str(method_root / "VSLAM" / "thirdparty" / "mast3r" / "dust3r"),
                str(method_root / "VSLAM" / "thirdparty" / "mast3r" / "asmk"),
                str(method_root / "VSLAM" / "thirdparty" / "Pi3"),
            ]
        )
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([p for p in extra_parts + [existing] if p])
    return env


def run_external(cmd: Sequence[str], cwd: Path, env: Dict[str, str], log_path: Path) -> int:
    print("Running:", " ".join(str(x) for x in cmd))
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
        return proc.wait()


def build_external_command(
    args: argparse.Namespace,
    method: str,
    source_dir: Path,
    external_dir: Path,
    timing_path: Path,
    passthrough: Sequence[str],
) -> List[str]:
    if method == "on-the-fly-nvs":
        cmd = [
            args.python,
            "train.py",
            "-s",
            str(source_dir),
            "-i",
            "images",
            "-m",
            str(external_dir),
            "--viewer_mode",
            "none",
            "--timing_path",
            str(timing_path),
        ]
    else:
        cmd = [
            args.python,
            "run_system.py",
            "-s",
            str(source_dir),
            "-i",
            "images",
            "-m",
            str(external_dir),
            "--viewer_mode",
            "none",
            "--timing_path",
            str(timing_path),
        ]
        if args.artdeco_config:
            cmd.extend(["--config", args.artdeco_config])
        if args.artdeco_checkpoint:
            cmd.extend(["--checkpoint_path", args.artdeco_checkpoint])
    cmd.extend(passthrough)
    return cmd


def prepared_name_to_original_stem(prepared_metadata: Sequence[Dict[str, object]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for meta in prepared_metadata:
        prepared = Path(str(meta["prepared"]))
        out[prepared.name] = str(meta["stem"])
        out[prepared.stem] = str(meta["stem"])
    return out


def stem_from_keyframe_name(name: object, name_map: Dict[str, str]) -> str:
    base = Path(str(name)).name
    if base in name_map:
        return name_map[base]
    stem = Path(base).stem
    if stem in name_map:
        return name_map[stem]
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


def load_cameras_from_metadata(metadata_path: Path, prepared_metadata: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    metadata = load_json_file(metadata_path)
    name_map = prepared_name_to_original_stem(prepared_metadata)
    cams: List[Dict[str, object]] = []
    for index, keyframe in enumerate(metadata.get("keyframes", []) or []):
        info = keyframe.get("info", {}) if isinstance(keyframe, dict) else {}
        name = info.get("name", f"frame_{index:06d}") if isinstance(info, dict) else f"frame_{index:06d}"
        Rt = np.asarray(keyframe.get("Rt"), dtype=np.float32)
        if Rt.shape != (4, 4) or not np.isfinite(Rt).all():
            continue
        cams.append(
            {
                "frame_id": float(index),
                "stem": stem_from_keyframe_name(name, name_map),
                "T_c2w": np.linalg.inv(Rt).astype(np.float32),
                "f": keyframe.get("f"),
            }
        )
    return cams


def load_gaussian_ply_with_sh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData
    except Exception:
        return load_point_cloud_file(path)

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    pts = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)
    names = set(vertex.data.dtype.names or ())
    if {"red", "green", "blue"}.issubset(names):
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=-1).astype(np.uint8)
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1).astype(np.float32)
        colors = np.clip(sh * SH_C0 + 0.5, 0.0, 1.0)
        colors = np.round(colors * 255.0).astype(np.uint8)
    else:
        colors = np.full((pts.shape[0], 3), 220, dtype=np.uint8)
    return pts, colors


def load_gaussian_point_clouds(
    method: str,
    external_dir: Path,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    candidates: List[Path] = []
    if method == "artdeco":
        candidates.extend(
            [
                external_dir / "point_clouds" / "xyz_rgb.ply",
                external_dir / "colmap" / "points3D.ply",
                external_dir / "point_clouds" / "gs.ply",
            ]
        )
    else:
        candidates.extend(sorted((external_dir / "point_clouds").glob("anchor_*.ply")))

    pts_parts: List[np.ndarray] = []
    col_parts: List[np.ndarray] = []
    used: List[str] = []
    for cloud_path in candidates:
        if not cloud_path.exists():
            continue
        pts, cols = load_gaussian_ply_with_sh(cloud_path)
        if pts.shape[0] == 0:
            continue
        pts_parts.append(pts)
        col_parts.append(cols)
        used.append(str(cloud_path))
        if method == "artdeco":
            break

    if not pts_parts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), used

    points = np.concatenate(pts_parts, axis=0)
    colors = np.concatenate(col_parts, axis=0)
    points, colors = sample_points(points, colors, max_points=max_points, seed=seed)
    return points, colors, used


def read_processing_time(timing_path: Path, metadata_path: Path) -> Dict[str, object]:
    timing = load_json_file(timing_path)
    if timing:
        return timing
    metadata = load_json_file(metadata_path)
    if "time" in metadata:
        seconds = float(metadata["time"])
        frames = int(metadata.get("num_frames", 0) or 0)
        return {
            "processing_time_sec": seconds,
            "num_frames": frames,
            "fps": float(metadata.get("FPS")) if metadata.get("FPS") is not None else None,
            "source": "metadata.json/time",
            "excludes": ["weight_loading", "module_initialization", "final_result_saving"],
        }
    return {}


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, help="on-the-fly-nvs or artdeco")
    parser.add_argument("--scene_dir", required=True, help="Scene folder containing an images/ directory.")
    parser.add_argument("--output_rrd", required=True, help="Output .rrd path.")
    parser.add_argument("--output_dir", default=None, help="Directory for prepared input and external outputs.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run the third-party method.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or numeric CUDA index.")

    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--cams_dir", default="cams")
    parser.add_argument("--depth_dir", default="depth")
    parser.add_argument("--frame_glob", default="*")
    parser.add_argument("--num_views", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_side", type=int, default=0)
    parser.add_argument("--size_multiple", type=int, default=1)
    parser.add_argument("--copy_images", action="store_true")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--depth_min", type=float, default=1e-6)
    parser.add_argument("--depth_max", type=float, default=1e6)

    parser.add_argument("--artdeco_config", default=None, help="ARTDECO config path, relative to third_party/ARTDECO by default.")
    parser.add_argument("--artdeco_checkpoint", default=None, help="Optional ARTDECO SHARP checkpoint override.")

    parser.add_argument("--max_pred_points", type=int, default=800000)
    parser.add_argument("--max_gt_points", type=int, default=800000)
    parser.add_argument("--point_radius", type=float, default=0.0)
    parser.add_argument("--view_coordinates", default="RDF")
    parser.add_argument("--background", type=int, nargs=3, default=[255, 255, 255])
    parser.add_argument("--hide_grid", action="store_true")
    parser.add_argument("--log_images", action="store_true")
    parser.add_argument("--camera_axis_size", type=float, default=0.0)
    parser.add_argument("--camera_axis_radius", type=float, default=0.0)
    parser.add_argument("--show_world_axes", action="store_true", default=True)
    parser.add_argument("--no_world_axes", action="store_false", dest="show_world_axes")
    parser.add_argument("--world_axis_size", type=float, default=0.0)
    parser.add_argument("--world_axis_radius", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_known_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, passthrough = parse_args(argv)
    method = normalize_method(args.method)
    args.method = method

    repo_root = Path(__file__).resolve().parents[1]
    method_root = resolve_method_root(repo_root, method)
    method_script = method_root / str(METHODS[method]["script"])
    if not method_script.exists():
        raise RuntimeError(f"Missing checkout for {method}: {method_root}. Expected script: {method_script.name}")

    output_rrd = Path(args.output_rrd).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else output_rrd.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    stems, image_paths = select_images(args)
    source_dir = output_dir / "source"
    prepared_dir = source_dir / "images"
    if source_dir.exists():
        shutil.rmtree(source_dir)
    prepared_images, prepared_metadata = materialize_images(
        image_paths=image_paths,
        stems=stems,
        out_dir=prepared_dir,
        max_side=int(args.max_side),
        size_multiple=int(args.size_multiple),
        copy_images=bool(args.copy_images),
    )
    print(f"Selected {len(prepared_images)} images for {METHODS[method]['display']}; prepared input: {prepared_dir}")

    external_dir = output_dir / "external"
    if external_dir.exists():
        shutil.rmtree(external_dir)
    external_dir.mkdir(parents=True, exist_ok=True)
    stdout_log_path = output_dir / "gaussian_slam_stdout.log"
    timing_path = output_dir / "processing_time.json"
    if timing_path.exists():
        timing_path.unlink()

    cmd = build_external_command(
        args=args,
        method=method,
        source_dir=source_dir,
        external_dir=external_dir,
        timing_path=timing_path,
        passthrough=passthrough,
    )
    env = build_env(args, method_root=method_root)
    return_code = run_external(cmd, cwd=method_root, env=env, log_path=stdout_log_path)

    metadata_path = external_dir / "metadata.json"
    processing_time_meta = read_processing_time(timing_path, metadata_path)
    cams = load_cameras_from_metadata(metadata_path, prepared_metadata)
    points, colors, point_artifacts = load_gaussian_point_clouds(
        method=method,
        external_dir=external_dir,
        max_points=int(args.max_pred_points),
        seed=int(args.seed),
    )
    gt_cams, gt_points, gt_colors, gt_meta = load_gt_artifacts(args=args, stems=stems, image_paths=image_paths)

    points, cams, align_meta = align_prediction_to_gt_pose_sim3(
        pred_points=points,
        pred_cams=cams,
        gt_cams=gt_cams,
    )

    save_final_eval_outputs(
        eval_dir=output_dir / "eval",
        pred_cams=cams,
        gt_cams=gt_cams,
        pred_points=points,
        pred_colors=colors,
        gt_points=gt_points,
        gt_colors=gt_colors,
        meta={
            "schema": "final_eval_v1",
            "script": "scripts/run_gaussian_slam_to_rrd.py",
            "scene_dir": str(Path(args.scene_dir).expanduser().resolve()),
            "method": method,
            "method_display": METHODS[method]["display"],
            "pose_convention": "T_c2w",
            "points_coordinate": "same_as_pred_cameras",
            "processing_time": processing_time_meta,
            "post_align": {
                "enabled": True,
                "type": "pose_sim3",
                "target": "gt_pose",
                **align_meta,
                "valid": bool(align_meta.get("valid", False)),
                "median_camera_residual": float(align_meta.get("median_residual", float("nan"))),
            },
        },
    )

    sidecar = output_rrd.with_suffix(".json")
    payload = {
        "method": method,
        "method_display": METHODS[method]["display"],
        "method_root": method_root,
        "scene_dir": Path(args.scene_dir).expanduser().resolve(),
        "images_dir": args.images_dir,
        "cams_dir": args.cams_dir,
        "depth_dir": args.depth_dir,
        "output_rrd": output_rrd,
        "output_dir": output_dir,
        "source_dir": source_dir,
        "external_dir": external_dir,
        "metadata_path": metadata_path,
        "stdout_log_path": stdout_log_path,
        "timing_path": timing_path,
        "processing_time": processing_time_meta,
        "point_artifacts": point_artifacts,
        "return_code": int(return_code),
        "command": cmd,
        "stems": stems,
        "prepared_images": prepared_metadata,
        "num_poses": len(cams),
        "num_pred_points_logged": int(points.shape[0]),
        "num_gt_cameras": int(len(gt_cams)),
        "num_gt_points_logged": int(gt_points.shape[0]),
        "gt": gt_meta,
        "alignment": align_meta,
        "passthrough_args": list(passthrough),
    }
    sidecar.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved sidecar metadata: {sidecar}")

    if return_code != 0:
        print(f"[ERROR] {METHODS[method]['display']} exited with code {return_code}. See {stdout_log_path}")
        return int(return_code)

    write_rrd(
        args=args,
        method=method,
        output_rrd=output_rrd,
        prepared_metadata=prepared_metadata,
        prepared_images=prepared_images,
        cams=cams,
        points=points,
        colors=colors,
        gt_cams=gt_cams,
        gt_points=gt_points,
        gt_colors=gt_colors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
