#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run VGGT-SLAM 1.0/2.0 or VGGT-Long on a MapAnything scene and save a Rerun .rrd.

The scene input follows the benchmark scene-directory convention:

    scene_dir/
      images/  *.jpg | *.jpeg | *.png | *.bmp | *.tif | *.tiff

Example:

    python benchmarking/adapters/vggt_optim.py \
      --method vggt-slam \
      --scene_dir /opt/data/private/dataset/data/usegeo/dataset1 \
      --output_rrd outputs/vggt-slam/usegeo-dataset1.rrd \
      --num_views 0 \
      --stride 1 \
      --max_side 518 \
      --submap_size 16 \
      --max_loops 1

    python benchmarking/adapters/vggt_optim.py \
      --method vggt-slam2.0 \
      --scene_dir /opt/data/private/dataset/data/uavscenes/interval5_AMtown01 \
      --output_rrd outputs/vggt-slam2.0/uavscenes-AMtown01.rrd \
      --num_views 0 \
      --stride 1 \
      --max_side 518 \
      --submap_size 16 \
      --max_loops 1
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geoff3d.slrf.scene_io import voxel_downsample

cv2 = None
rr = None
rrb = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEPTH_EXTS = {".exr", ".npy", ".png", ".tif", ".tiff"}
CAM_EXTS = {".txt"}
METHODS = {
    "vggt-slam": {
        "repo": "third_party/vggt-slam",
        "display": "VGGT-SLAM 1.0",
    },
    "vggt-slam1.0": {
        "repo": "third_party/vggt-slam",
        "display": "VGGT-SLAM 1.0",
    },
    "vggt-slam2.0": {
        "repo": "third_party/vggt-slam2.0",
        "script": "main.py",
        "display": "VGGT-SLAM 2.0",
    },
    "vggt-long": {
        "repo": "third_party/vggt-long",
        "alt_repos": ("third_party/VGGT-Long",),
        "script": "vggt_long.py",
        "display": "VGGT-Long",
    },
}


def require_cv2():
    global cv2
    if cv2 is None:
        import cv2 as cv2_mod

        cv2 = cv2_mod
    return cv2


def require_rerun():
    global rr, rrb
    if rr is None:
        import rerun as rr_mod

        rr = rr_mod
        try:
            import rerun.blueprint as rrb_mod
        except Exception:
            rrb_mod = None
        rrb = rrb_mod
    return rr, rrb


def sanitize_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[\\/:*?\"<>|\s]+", "_", name)
    return name or "scene"


def json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def save_point_cloud_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if colors.shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(
                f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def load_json_file(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] failed to read JSON {path}: {e}")
        return {}


def collect_stem_to_path(folder: Path, exts: Iterable[str]) -> Dict[str, Path]:
    exts = {e.lower() for e in exts}
    if not folder.exists():
        return {}
    return {
        p.stem: p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    }


def _float_tokens(line: str) -> Optional[List[float]]:
    try:
        return [float(x) for x in line.replace(",", " ").split()]
    except ValueError:
        return None


def _find_line(lines: Sequence[str], prefixes: Sequence[str]) -> int:
    prefixes = tuple(p.lower() for p in prefixes)
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(prefixes):
            return i
    return -1


def _read_numeric_rows(lines: Sequence[str], start: int, n_rows: int, n_cols: int, path: Path) -> np.ndarray:
    rows: List[List[float]] = []
    for j in range(start, len(lines)):
        vals = _float_tokens(lines[j])
        if vals is None or len(vals) < n_cols:
            continue
        rows.append(vals[:n_cols])
        if len(rows) == n_rows:
            break
    if len(rows) != n_rows:
        raise ValueError(f"Cannot read {n_rows}x{n_cols} numeric matrix from {path}")
    return np.asarray(rows, dtype=np.float64)


def parse_cam_txt(cam_path: Path) -> Dict[str, object]:
    lines = [ln.strip() for ln in cam_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    idx_ext = _find_line(lines, ["extrinsic"])
    idx_int = _find_line(lines, ["intrinsic"])
    if idx_ext < 0 or idx_int < 0:
        raise ValueError(f"Invalid camera txt, missing extrinsic/intrinsic: {cam_path}")

    T_w2c = _read_numeric_rows(lines, idx_ext + 1, 4, 4, cam_path)
    K = _read_numeric_rows(lines, idx_int + 1, 3, 3, cam_path)

    height: Optional[int] = None
    width: Optional[int] = None
    fov: Optional[float] = None
    idx_hwf = -1
    for i, line in enumerate(lines):
        tokens = line.lower().replace(":", " ").split()
        if "h" in tokens and "w" in tokens and ("fov" in tokens or "hfov" in tokens):
            idx_hwf = i
            break
    if idx_hwf >= 0:
        vals = None
        for j in range(idx_hwf + 1, len(lines)):
            vals = _float_tokens(lines[j])
            if vals is not None and len(vals) >= 2:
                break
        if vals is not None and len(vals) >= 2:
            height = int(round(vals[0]))
            width = int(round(vals[1]))
            if len(vals) >= 3:
                fov = float(vals[2])

    return {
        "stem": cam_path.stem,
        "path": str(cam_path),
        "K": K,
        "T_w2c": T_w2c,
        "T_c2w": np.linalg.inv(T_w2c),
        "height": height,
        "width": width,
        "fov": fov,
    }


def read_rgb(path: Path) -> np.ndarray:
    cv2_mod = require_cv2()
    img = cv2_mod.imread(str(path), cv2_mod.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2_mod.cvtColor(img, cv2_mod.COLOR_BGR2RGB)


def read_depth(path: Path, depth_scale: float) -> np.ndarray:
    cv2_mod = require_cv2()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(str(path))
    elif suffix == ".exr":
        depth = cv2_mod.imread(str(path), cv2_mod.IMREAD_UNCHANGED)
        if depth is None:
            from spatial_rrd.scene_io import read_exr_depth

            depth = read_exr_depth(path)
    else:
        depth = cv2_mod.imread(str(path), cv2_mod.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Cannot read depth: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if float(depth_scale) != 1.0:
        depth = depth / float(depth_scale)
    return depth


def scale_K_to_depth(
    K: np.ndarray,
    cam_width: Optional[int],
    cam_height: Optional[int],
    rgb_shape: Tuple[int, int],
    depth_shape: Tuple[int, int],
) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64).copy()
    rgb_h, rgb_w = int(rgb_shape[0]), int(rgb_shape[1])
    depth_h, depth_w = int(depth_shape[0]), int(depth_shape[1])
    if cam_width is None or cam_height is None:
        cam_width, cam_height = rgb_w, rgb_h
    if int(cam_width) != depth_w or int(cam_height) != depth_h:
        sx = float(depth_w) / float(cam_width)
        sy = float(depth_h) / float(cam_height)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
    return K


def depth_to_world_points_numpy(depth: np.ndarray, K: np.ndarray, T_c2w: np.ndarray) -> np.ndarray:
    h, w = depth.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)
    R = np.asarray(T_c2w, dtype=np.float64)[:3, :3]
    t = np.asarray(T_c2w, dtype=np.float64)[:3, 3]
    pts_world = np.einsum("ij,hwj->hwi", R, pts_cam) + t[None, None, :]
    return pts_world.astype(np.float32)


def contiguous_select(items: Sequence[str], max_count: int) -> List[str]:
    items = list(items)
    if max_count <= 0 or len(items) <= max_count:
        return items
    return items[: int(max_count)]


def select_images(args: argparse.Namespace) -> Tuple[List[str], List[Path]]:
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    images_dir = scene_dir / args.images_dir
    images = collect_stem_to_path(images_dir, IMAGE_EXTS)
    if not images:
        raise RuntimeError(f"No images found under {images_dir}")

    stems = sorted(images)
    if args.frame_glob and args.frame_glob != "*":
        stems = [s for s in stems if fnmatch.fnmatch(s, args.frame_glob)]
    if args.start > 0:
        stems = stems[int(args.start) :]
    if args.stride > 1:
        stems = stems[:: int(args.stride)]
    stems = contiguous_select(stems, int(args.num_views))
    if not stems:
        raise RuntimeError(f"No frames selected under {images_dir}")
    return stems, [images[s] for s in stems]


def round_down_to_multiple(x: int, multiple: int) -> int:
    if multiple <= 1:
        return int(x)
    return max(int(multiple), int(x) // int(multiple) * int(multiple))


def target_hw(height: int, width: int, max_side: int, size_multiple: int) -> Tuple[int, int]:
    h, w = int(height), int(width)
    if max_side > 0 and max(h, w) > int(max_side):
        scale = float(max_side) / float(max(h, w))
        h = max(1, int(round(h * scale)))
        w = max(1, int(round(w * scale)))
    h = round_down_to_multiple(h, size_multiple)
    w = round_down_to_multiple(w, size_multiple)
    return h, w


def materialize_images(
    image_paths: Sequence[Path],
    stems: Sequence[str],
    out_dir: Path,
    max_side: int,
    size_multiple: int,
    copy_images: bool,
) -> Tuple[List[Path], List[Dict[str, object]]]:
    cv2_mod = require_cv2()
    marker = out_dir / ".mapanything_vggt_slam_input"
    if out_dir.exists():
        if not marker.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing input folder without marker: {out_dir}. "
                "Choose a fresh --output_dir or remove it manually."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text("generated by adapters/vggt_optim.py\n", encoding="utf-8")

    materialized: List[Path] = []
    metadata: List[Dict[str, object]] = []
    for i, (src, stem) in enumerate(zip(image_paths, stems)):
        dst = out_dir / f"{i:06d}__{sanitize_name(stem)}{src.suffix.lower()}"
        img = cv2_mod.imread(str(src), cv2_mod.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image: {src}")
        src_h, src_w = img.shape[:2]
        dst_h, dst_w = target_hw(src_h, src_w, max_side=max_side, size_multiple=size_multiple)
        resized = (dst_h, dst_w) != (src_h, src_w)

        if resized or copy_images:
            if resized:
                img = cv2_mod.resize(img, (dst_w, dst_h), interpolation=cv2_mod.INTER_AREA)
            if not cv2_mod.imwrite(str(dst), img):
                raise RuntimeError(f"Failed to write prepared image: {dst}")
        else:
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        materialized.append(dst)
        metadata.append(
            {
                "index": i,
                "stem": stem,
                "source": src,
                "prepared": dst,
                "source_size": {"height": src_h, "width": src_w},
                "prepared_size": {"height": dst_h, "width": dst_w},
                "resized": resized,
            }
        )
    return materialized, metadata


def normalize_method(method: str) -> str:
    key = str(method).strip().lower()
    if key in {
        "vggt-slam1",
        "vggt_slam",
        "vggt_slam1.0",
        "vggt-slam-1.0",
        "vggt-slam-sim3",
        "vggt_slam_sim3",
        "vggt-slam-sl4",
        "vggt_slam_sl4",
    }:
        key = "vggt-slam"
    if key in {"vggt_slam2", "vggt_slam2.0", "vggt-slam-2.0", "vggt-slam2"}:
        key = "vggt-slam2.0"
    if key in {"vggt_long", "vggtlong", "vggt-long"}:
        key = "vggt-long"
    if key not in METHODS:
        raise ValueError(
            f"Unknown --method {method}. Use vggt-slam, vggt-slam-sim3, "
            "vggt-slam-sl4, vggt-slam2.0, or vggt-long."
        )
    return key


def normalize_method_variant(method: str) -> str:
    key = str(method).strip().lower().replace("_", "-")
    if key == "vggt-slam-sim3":
        return "sim3"
    if key == "vggt-slam-sl4":
        return "sl4"
    return "default"


def resolve_method_root(repo_root: Path, method: str) -> Path:
    info = METHODS[method]
    candidates = [repo_root / str(info["repo"])]
    candidates.extend(repo_root / str(p) for p in info.get("alt_repos", ()))
    script_name = str(info.get("script", "main.py"))
    for candidate in candidates:
        if (candidate / script_name).exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_existing_cli_path(path: str) -> str:
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    if p.exists():
        return str(p.resolve())
    return str(p)


def build_external_command(
    args: argparse.Namespace,
    method: str,
    method_root: Path,
    image_folder: Path,
    pose_log_path: Path,
    timing_path: Path,
    passthrough: Sequence[str],
) -> List[str]:
    if method == "vggt-long":
        cmd = [
            args.python,
            str(METHODS[method].get("script", "vggt_long.py")),
            "--image_dir",
            str(image_folder),
            "--timing_path",
            str(timing_path),
        ]
        if args.vggt_long_config:
            cmd.extend(["--config", resolve_existing_cli_path(args.vggt_long_config)])
        cmd.extend(passthrough)
        return cmd

    cmd = [
        args.python,
        "main.py",
        "--image_folder",
        str(image_folder),
        "--log_results",
        "--log_path",
        str(pose_log_path),
        "--timing_path",
        str(timing_path),
        "--submap_size",
        str(args.submap_size),
        "--overlapping_window_size",
        str(args.overlapping_window_size),
        "--max_loops",
        str(args.max_loops),
        "--min_disparity",
        str(args.min_disparity),
        "--conf_threshold",
        str(args.conf_threshold),
    ]
    if args.vis_map:
        cmd.append("--vis_map")
    if args.vis_flow:
        cmd.append("--vis_flow")
    if args.skip_dense_log:
        cmd.append("--skip_dense_log")

    if method == "vggt-slam2.0":
        cmd.extend(["--lc_thres", str(args.lc_thres)])
        if args.vis_voxel_size is not None:
            cmd.extend(["--vis_voxel_size", str(args.vis_voxel_size)])
        if args.run_os:
            cmd.append("--run_os")
    else:
        cmd.extend(["--downsample_factor", str(args.downsample_factor)])
        cmd.extend(["--vis_stride", str(args.vis_stride)])
        cmd.extend(["--vis_point_size", str(args.vis_point_size)])
        if args.use_sim3:
            cmd.append("--use_sim3")
        if args.use_point_map:
            cmd.append("--use_point_map")
        if args.plot_focal_lengths:
            cmd.append("--plot_focal_lengths")

    # VGGT-SLAM 1.0: headless + sampled global point cloud export for wrapper consumption.
    if method == "vggt-slam":
        if args.headless:
            cmd.append("--headless")

        # Save one global cloud for wrapper consumption, not framewise *.npz logs.
        cmd.append("--log_global_points")
        cmd.extend([
            "--global_points_path",
            str(pose_log_path.with_name(pose_log_path.stem + "_points.ply")),
        ])
        cmd.extend([
            "--max_global_points",
            str(args.max_points_per_view),
        ])
        cmd.extend([
            "--global_voxel_downsample",
            str(args.voxel_downsample),
        ])
        cmd.extend([
            "--global_point_stride",
            str(args.global_point_stride),
        ])
        cmd.append("--skip_dense_log")

    cmd.extend(passthrough)
    return cmd


def build_env(args: argparse.Namespace, method_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    if args.vggt_model_path:
        env["VGGT_MODEL_PATH"] = resolve_existing_cli_path(args.vggt_model_path)
    repo_root = Path(__file__).resolve().parents[1]
    pythonpath_parts = [str(method_root)]
    local_salad_candidates = [
        method_root / "salad",
        method_root / "third_party" / "salad",
    ]
    for salad_root in local_salad_candidates:
        if (salad_root / "salad").is_dir():
            pythonpath_parts.append(str(salad_root.resolve()))
    pythonpath_parts.append(str(repo_root))
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    device = str(args.device).lower()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif device.isdigit():
        env["CUDA_VISIBLE_DEVICES"] = device
    elif device.startswith("cuda:") and device.split(":", 1)[1].isdigit():
        env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
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


def _vggt_long_exp_parent(method_root: Path, image_folder: Path) -> Path:
    return method_root / "exps" / str(image_folder).replace("/", "_")


def _collect_vggt_long_exps(method_root: Path, image_folder: Path) -> List[Path]:
    candidates: List[Path] = []
    parent = _vggt_long_exp_parent(method_root, image_folder)
    if parent.exists():
        candidates.extend(p for p in parent.iterdir() if (p / "camera_poses.txt").exists())

    if not candidates:
        exps_dir = method_root / "exps"
        if exps_dir.exists():
            candidates.extend(p.parent for p in exps_dir.glob("**/camera_poses.txt"))

    return candidates


def _find_latest_vggt_long_exp(
    method_root: Path,
    image_folder: Path,
    exclude: Optional[Sequence[Path]] = None,
) -> Optional[Path]:
    candidates = _collect_vggt_long_exps(method_root, image_folder)
    if exclude:
        excluded = {p.resolve() for p in exclude}
        candidates = [p for p in candidates if p.resolve() not in excluded]

    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "camera_poses.txt").stat().st_mtime)


def stage_vggt_long_outputs(
    method_root: Path,
    image_folder: Path,
    output_dir: Path,
    require_outputs: bool,
    exclude_exp_dirs: Optional[Sequence[Path]] = None,
) -> Dict[str, object]:
    exp_dir = _find_latest_vggt_long_exp(method_root, image_folder, exclude=exclude_exp_dirs)
    if exp_dir is None:
        msg = f"Cannot find VGGT-Long output under {method_root / 'exps'}"
        if require_outputs:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")
        return {"source_exp_dir": None, "staged": False}

    staged: Dict[str, object] = {
        "source_exp_dir": str(exp_dir),
        "staged": True,
    }

    for name in ("camera_poses.txt", "intrinsic.txt"):
        src = exp_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            staged[name] = str(output_dir / name)
        elif require_outputs and name == "camera_poses.txt":
            raise RuntimeError(f"Missing VGGT-Long output file: {src}")

    src_pcd_dir = exp_dir / "pcd"
    dst_pcd_dir = output_dir / "pcd"
    if (src_pcd_dir / "combined_pcd.ply").exists():
        dst_pcd_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_pcd_dir / "combined_pcd.ply", dst_pcd_dir / "combined_pcd.ply")
        staged["combined_pcd"] = str(dst_pcd_dir / "combined_pcd.ply")
    elif require_outputs:
        print(f"[WARN] Missing VGGT-Long combined point cloud: {src_pcd_dir / 'combined_pcd.ply'}")

    return staged


def quat_xyzw_to_rotmat(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.asarray(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def read_pose_log(path: Path, prepared_metadata: Optional[Sequence[Dict[str, object]]] = None) -> List[Dict[str, object]]:
    cams: List[Dict[str, object]] = []
    frame_to_stem: Dict[int, str] = {}
    if prepared_metadata is not None:
        for meta in prepared_metadata:
            frame_to_stem[int(meta["index"])] = str(meta["stem"])
    if not path.exists():
        return cams
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) not in {8, 12, 13, 16}:
            print(f"[WARN] skip malformed pose line {line_no}: {line}")
            continue
        vals = [float(x) for x in parts]
        if len(vals) == 16:
            frame_id = float(len(cams))
            T = np.asarray(vals, dtype=np.float32).reshape(4, 4)
        elif len(vals) == 8:
            frame_id = vals[0]
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = quat_xyzw_to_rotmat(vals[4:8])
            T[:3, 3] = np.asarray(vals[1:4], dtype=np.float32)
        else:
            frame_id = vals[0] if len(vals) == 13 else float(len(cams))
            pose_vals = vals[1:] if len(vals) == 13 else vals
            T = np.eye(4, dtype=np.float32)
            T[:3, :] = np.asarray(pose_vals, dtype=np.float32).reshape(3, 4)
        frame_idx = int(round(frame_id))
        stem = frame_to_stem.get(frame_idx, f"frame_{frame_idx:06d}")
        cams.append({"frame_id": frame_id, "stem": stem, "T_c2w": T})
    cams.sort(key=lambda c: float(c["frame_id"]))
    return cams


def sample_points(points: np.ndarray, colors: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if colors.shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    return points, colors


def downsample_final_points(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    enabled: bool,
    max_points: int,
    voxel_size: float,
    seed: int,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match the final output downsampling policy used by spatial RRD export."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if colors.shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]

    n0 = int(points.shape[0])
    if not enabled:
        print(f"[POINT] {label}: keep all finite points: {n0:,}")
        return points, colors

    points, colors = sample_points(
        points,
        colors,
        max_points=int(max_points),
        seed=int(seed),
    )

    if float(voxel_size) > 0:
        points, colors = voxel_downsample(points, colors, float(voxel_size))

    print(
        f"[POINT] {label}: {n0:,} -> {points.shape[0]:,} "
        f"(max_points={int(max_points)}, voxel={float(voxel_size)})"
    )
    return points.astype(np.float32), colors.astype(np.uint8)


def load_gt_artifacts(
    args: argparse.Namespace,
    stems: Sequence[str],
    image_paths: Sequence[Path],
) -> Tuple[List[Dict[str, object]], np.ndarray, np.ndarray, Dict[str, object]]:
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    cams_dir = scene_dir / args.cams_dir
    depth_dir = scene_dir / args.depth_dir
    cam_paths = collect_stem_to_path(cams_dir, CAM_EXTS)
    depth_paths = collect_stem_to_path(depth_dir, DEPTH_EXTS)

    gt_cams: List[Dict[str, object]] = []
    point_parts: List[np.ndarray] = []
    color_parts: List[np.ndarray] = []
    num_depth_used = 0
    num_cam_used = 0
    cv2_mod = require_cv2()
    cam_cache: Dict[str, Dict[str, object]] = {}

    for stem in stems:
        cam_path = cam_paths.get(stem)
        if cam_path is None:
            continue
        try:
            cam = parse_cam_txt(cam_path)
            cam_cache[str(stem)] = cam
            T_c2w = np.asarray(cam["T_c2w"], dtype=np.float32)
            gt_cams.append({"stem": str(stem), "T_c2w": T_c2w})
            num_cam_used += 1
        except Exception as e:
            print(f"[WARN] failed to parse GT camera for {stem}: {e}")

    depth_eligible = [
        (stem, image_path)
        for stem, image_path in zip(stems, image_paths)
        if stem in cam_paths and stem in depth_paths
    ]

    if int(args.max_gt_points) > 0 and len(depth_eligible) > 0:
        per_frame_cap = max(
            1,
            int(np.ceil(float(args.max_gt_points) * 1.5 / float(len(depth_eligible)))),
        )
    else:
        per_frame_cap = 0

    print(
        f"[GT] loading sampled GT point cloud: "
        f"cameras={num_cam_used}/{len(stems)}, "
        f"depth_eligible={len(depth_eligible)}/{len(stems)}, "
        f"max_gt_points={int(args.max_gt_points)}, "
        f"per_frame_cap={int(per_frame_cap)}, "
        f"workers={max(0, int(getattr(args, 'scene_io_workers', 0)))}"
    )

    def load_one_depth(item: Tuple[int, str, Path]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        local_i, stem, image_path = item
        stem_str = str(stem)
        cam = cam_cache.get(stem_str)
        if cam is None:
            cam_path = cam_paths.get(stem)
            try:
                cam = parse_cam_txt(cam_path) if cam_path is not None else None
            except Exception as e:
                return local_i, None, None, f"failed to parse GT camera for {stem}: {e}"

        depth_path = depth_paths.get(stem)
        if cam is None or depth_path is None:
            return local_i, None, None, None

        try:
            rgb = read_rgb(image_path)
            depth = read_depth(depth_path, depth_scale=float(args.depth_scale))
            K = scale_K_to_depth(
                np.asarray(cam["K"], dtype=np.float64),
                cam_width=cam.get("width"),
                cam_height=cam.get("height"),
                rgb_shape=rgb.shape[:2],
                depth_shape=depth.shape[:2],
            )

            if rgb.shape[:2] != depth.shape[:2]:
                rgb = cv2_mod.resize(
                    rgb,
                    (depth.shape[1], depth.shape[0]),
                    interpolation=cv2_mod.INTER_AREA,
                )

            valid = (
                np.isfinite(depth)
                & (depth > float(args.depth_min))
                & (depth < float(args.depth_max))
            )

            if not bool(valid.any()):
                return local_i, None, None, None

            pts_world = depth_to_world_points_numpy(
                depth,
                K,
                np.asarray(cam["T_c2w"], dtype=np.float64),
            )
            valid = valid & np.isfinite(pts_world).all(axis=-1)

            if not bool(valid.any()):
                return local_i, None, None, None

            pts = pts_world[valid].reshape(-1, 3).astype(np.float32)
            cols = rgb[valid].reshape(-1, 3).astype(np.uint8)

            # Critical: sample per frame before storing.
            pts, cols = sample_points(
                pts,
                cols,
                max_points=int(per_frame_cap),
                seed=int(args.seed) + 104729 * (local_i + 1),
            )

            return local_i, pts, cols, None

        except Exception as e:
            return local_i, None, None, f"failed to load GT depth/points for {stem}: {e}"

    work_items = [
        (local_i, str(stem), image_path)
        for local_i, (stem, image_path) in enumerate(depth_eligible)
    ]
    workers = max(0, int(getattr(args, "scene_io_workers", 0)))
    if workers > 1 and len(work_items) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            loaded_iter = executor.map(load_one_depth, work_items)
            loaded = list(loaded_iter)
    else:
        loaded = [load_one_depth(item) for item in work_items]

    for _local_i, pts, cols, warning in sorted(loaded, key=lambda item: item[0]):
        if warning:
            print(f"[WARN] {warning}")
        if pts is not None and cols is not None and pts.shape[0] > 0:
            point_parts.append(pts)
            color_parts.append(cols)
            num_depth_used += 1

    if point_parts:
        gt_points = np.concatenate(point_parts, axis=0)
        gt_colors = np.concatenate(color_parts, axis=0)
    else:
        gt_points = np.empty((0, 3), dtype=np.float32)
        gt_colors = np.empty((0, 3), dtype=np.uint8)

    gt_points, gt_colors = sample_points(
        gt_points,
        gt_colors,
        int(args.max_gt_points),
        int(args.seed) + 911,
    )

    meta = {
        "cams_dir": str(cams_dir),
        "depth_dir": str(depth_dir),
        "num_cam_matches": int(num_cam_used),
        "num_depth_pointmaps_used": int(num_depth_used),
        "max_gt_points": int(args.max_gt_points),
        "per_frame_cap": int(per_frame_cap),
    }

    print(
        f"[GT] sampled GT point cloud: "
        f"frames={num_depth_used}, points={gt_points.shape[0]:,}"
    )

    return gt_cams, gt_points, gt_colors, meta


def load_point_cloud_with_open3d(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors)
    if cols.size == 0:
        cols = np.full((pts.shape[0], 3), 220, dtype=np.uint8)
    else:
        if cols.max(initial=0.0) <= 1.0:
            cols = np.clip(cols * 255.0, 0, 255)
        cols = cols.astype(np.uint8)
    return pts, cols


def load_ascii_pcd(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    data_start = -1
    fields: List[str] = []
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if low.startswith("fields"):
            fields = low.split()[1:]
        if low.startswith("data"):
            if "ascii" not in low:
                raise RuntimeError(f"PCD is not ASCII and open3d could not read it: {path}")
            data_start = i + 1
            break
    if data_start < 0:
        raise RuntimeError(f"Invalid PCD header: {path}")
    pts = []
    cols = []
    ix = fields.index("x") if "x" in fields else 0
    iy = fields.index("y") if "y" in fields else 1
    iz = fields.index("z") if "z" in fields else 2
    irgb = fields.index("rgb") if "rgb" in fields else -1
    for line in lines[data_start:]:
        if not line.strip():
            continue
        vals = line.split()
        pts.append([float(vals[ix]), float(vals[iy]), float(vals[iz])])
        if irgb >= 0:
            rgb_int = int(float(vals[irgb]))
            cols.append([(rgb_int >> 16) & 255, (rgb_int >> 8) & 255, rgb_int & 255])
        else:
            cols.append([220, 220, 220])
    return np.asarray(pts, dtype=np.float32), np.asarray(cols, dtype=np.uint8)


def load_ply_fallback(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        first = f.readline().decode("utf-8", errors="ignore").strip()
        if first != "ply":
            raise RuntimeError(f"Invalid PLY header: {path}")

        vertex_count = 0
        fmt = "ascii"
        props: List[str] = []
        in_vertex = False
        while True:
            raw = f.readline()
            if not raw:
                raise RuntimeError(f"Invalid PLY header, missing end_header: {path}")
            line = raw.decode("utf-8", errors="ignore").strip()
            low = line.lower()
            if low.startswith("format"):
                fmt = low.split()[1]
            elif low.startswith("element"):
                tokens = low.split()
                in_vertex = len(tokens) >= 3 and tokens[1] == "vertex"
                if in_vertex:
                    vertex_count = int(tokens[2])
            elif in_vertex and low.startswith("property"):
                tokens = low.split()
                if tokens:
                    props.append(tokens[-1])
            elif low == "end_header":
                break

        def prop_idx(names: Sequence[str], default: int = -1) -> int:
            for name in names:
                if name in props:
                    return props.index(name)
            return default

        ix = prop_idx(("x",), 0)
        iy = prop_idx(("y",), 1)
        iz = prop_idx(("z",), 2)
        ir = prop_idx(("red", "r"), -1)
        ig = prop_idx(("green", "g"), -1)
        ib = prop_idx(("blue", "b"), -1)

        pts: List[List[float]] = []
        cols: List[List[int]] = []
        if fmt == "ascii":
            for _ in range(vertex_count):
                vals = f.readline().decode("utf-8", errors="ignore").split()
                if len(vals) <= max(ix, iy, iz):
                    continue
                pts.append([float(vals[ix]), float(vals[iy]), float(vals[iz])])
                if min(ir, ig, ib) >= 0 and len(vals) > max(ir, ig, ib):
                    cols.append([int(float(vals[ir])), int(float(vals[ig])), int(float(vals[ib]))])
                else:
                    cols.append([220, 220, 220])
            return np.asarray(pts, dtype=np.float32), np.asarray(cols, dtype=np.uint8)

        if fmt != "binary_little_endian":
            raise RuntimeError(f"Unsupported PLY format without open3d: {fmt}")

        # VGGT-Long writes common xyz/rgb vertices. Keep the fallback strict so
        # unexpected binary PLY layouts fail loudly instead of being misread.
        expected = ["x", "y", "z", "red", "green", "blue"]
        if props[:6] != expected:
            raise RuntimeError(f"Unsupported binary PLY vertex layout without open3d: {props}")
        import struct

        for _ in range(vertex_count):
            data = f.read(15)
            if len(data) != 15:
                break
            x, y, z, r, g, b = struct.unpack("<3f3B", data)
            pts.append([x, y, z])
            cols.append([r, g, b])
        return np.asarray(pts, dtype=np.float32), np.asarray(cols, dtype=np.uint8)


def load_point_cloud_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        return load_point_cloud_with_open3d(path)
    except Exception as e:
        print(f"[WARN] open3d failed to read {path}: {e}; trying lightweight parser")
    if path.suffix.lower() == ".pcd":
        return load_ascii_pcd(path)
    if path.suffix.lower() == ".ply":
        return load_ply_fallback(path)
    raise RuntimeError(f"Unsupported point cloud format: {path}")


def load_point_cloud_artifacts(
    method: str,
    pose_log_path: Path,
    prepared_images: Sequence[Path],
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    candidates: List[str] = []
    if method == "vggt-long":
        for cloud_path in (
            pose_log_path.parent / "pcd" / "combined_pcd.ply",
            pose_log_path.parent / "combined_pcd.ply",
        ):
            if cloud_path.exists():
                candidates.append(str(cloud_path))
                pts, cols = load_point_cloud_file(cloud_path)
                pts, cols = sample_points(pts, cols, max_points=max_points, seed=seed)
                return pts, cols, candidates

    for ext in (".ply", ".pcd"):
        cloud_path = pose_log_path.with_name(pose_log_path.stem + f"_points{ext}")
        if cloud_path.exists():
            candidates.append(str(cloud_path))
            pts, cols = load_point_cloud_file(cloud_path)
            # Do not do final policy downsampling here. Keep it centralized after pose alignment.
            pts, cols = sample_points(pts, cols, max_points=max_points, seed=seed)
            return pts, cols, candidates

    logs_dir = pose_log_path.with_name(pose_log_path.stem + "_logs")
    if method != "vggt-slam2.0" and logs_dir.exists():
        cv2_mod = require_cv2()
        pts_parts: List[np.ndarray] = []
        col_parts: List[np.ndarray] = []
        for npz_path in sorted(logs_dir.glob("*.npz")):
            candidates.append(str(npz_path))
            with np.load(npz_path) as data:
                points = np.asarray(data["pointcloud"], dtype=np.float32)
                mask = np.asarray(data["mask"]).astype(bool)
            if points.ndim != 3 or points.shape[-1] != 3:
                continue
            if mask.shape != points.shape[:2]:
                mask = np.isfinite(points).all(axis=-1)
            keep = mask & np.isfinite(points).all(axis=-1)
            pts_parts.append(points[keep].reshape(-1, 3))

            frame_match = re.search(r"\d+(?:\.\d+)?", npz_path.stem)
            frame_idx = int(float(frame_match.group())) if frame_match else -1
            if 0 <= frame_idx < len(prepared_images):
                img = cv2_mod.imread(str(prepared_images[frame_idx]), cv2_mod.IMREAD_COLOR)
                if img is not None:
                    img = cv2_mod.cvtColor(img, cv2_mod.COLOR_BGR2RGB)
                    if img.shape[:2] != points.shape[:2]:
                        img = cv2_mod.resize(
                            img,
                            (points.shape[1], points.shape[0]),
                            interpolation=cv2_mod.INTER_AREA,
                        )
                    col_parts.append(img[keep].reshape(-1, 3).astype(np.uint8))
                    continue
            col_parts.append(np.full((int(keep.sum()), 3), 220, dtype=np.uint8))

        if pts_parts:
            pts = np.concatenate(pts_parts, axis=0)
            cols = np.concatenate(col_parts, axis=0)
            pts, cols = sample_points(pts, cols, max_points=max_points, seed=seed)
            return pts, cols, candidates

    return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), candidates


# ---------------------------------------------------------------------------
# Final pred -> GT pose_sim3 alignment (always applied, no CLI switch)
# ---------------------------------------------------------------------------


def _camera_centers_by_stem(cams: Sequence[Dict[str, object]]) -> Dict[str, np.ndarray]:
    """Extract camera centers keyed by stem."""
    out: Dict[str, np.ndarray] = {}
    for cam in cams:
        stem = str(cam.get("stem", ""))
        if not stem:
            continue
        T = np.asarray(cam.get("T_c2w"), dtype=np.float64)
        if T.shape != (4, 4):
            continue
        center = T[:3, 3]
        if np.isfinite(center).all():
            out[stem] = center.astype(np.float64)
    return out


def _matched_pose_center_correspondences(
    gt_cams: Sequence[Dict[str, object]],
    pred_cams: Sequence[Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build matched GT/pred camera-center correspondences by stem."""
    gt_by_stem = _camera_centers_by_stem(gt_cams)
    pred_by_stem = _camera_centers_by_stem(pred_cams)

    gt_corr: List[np.ndarray] = []
    pred_corr: List[np.ndarray] = []
    matched_stems: List[str] = []

    for stem in sorted(set(gt_by_stem.keys()).intersection(pred_by_stem.keys())):
        gt_corr.append(gt_by_stem[stem])
        pred_corr.append(pred_by_stem[stem])
        matched_stems.append(stem)

    if not gt_corr:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            [],
        )

    return (
        np.asarray(gt_corr, dtype=np.float32).reshape(-1, 3),
        np.asarray(pred_corr, dtype=np.float32).reshape(-1, 3),
        matched_stems,
    )


def _estimate_similarity_umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    estimate_scale: bool = True,
    eps: float = 1e-12,
) -> Tuple[float, np.ndarray, np.ndarray, bool, str]:
    """Estimate Sim3 from src to dst:  X_dst = scale * R @ X_src + t."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)

    if src.shape[0] < 3 or dst.shape != src.shape:
        return (
            1.0,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            False,
            "not enough 3D pose correspondences to estimate pose_sim3",
        )

    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[finite]
    dst = dst[finite]

    if src.shape[0] < 3:
        return (
            1.0,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            False,
            "not enough finite 3D pose correspondences to estimate pose_sim3",
        )

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)

    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    src_var = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if src_var <= eps:
        return (
            1.0,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            False,
            "predicted pose centers are degenerate",
        )

    cov = (dst_centered.T @ src_centered) / float(src.shape[0])

    try:
        U, singular_values, Vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return (
            1.0,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            False,
            "SVD failed while estimating pose_sim3",
        )

    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt

    if estimate_scale:
        scale = float(np.sum(singular_values * np.diag(S)) / src_var)
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)

    if (
        not np.isfinite(scale)
        or scale <= eps
        or not np.isfinite(R).all()
        or not np.isfinite(t).all()
    ):
        return (
            1.0,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            False,
            "pose_sim3 solve produced non-finite transform",
        )

    return scale, R, t, True, "pose_sim3 estimated from matched camera centers"


def _apply_similarity_to_points(
    points: np.ndarray,
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts

    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    out = (float(scale) * pts.astype(np.float64)) @ R.T
    out += t[None, :]
    return out.astype(np.float32)


def _apply_similarity_to_cameras(
    cams: Sequence[Dict[str, object]],
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> List[Dict[str, object]]:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    aligned: List[Dict[str, object]] = []
    for cam in cams:
        T = np.asarray(cam["T_c2w"], dtype=np.float64)
        T_aligned = np.eye(4, dtype=np.float32)

        # Apply pred -> GT Sim3 to camera orientation and camera center.
        T_aligned[:3, :3] = (R @ T[:3, :3]).astype(np.float32)
        T_aligned[:3, 3] = (R @ (float(scale) * T[:3, 3]) + t).astype(np.float32)

        aligned.append({**cam, "T_c2w": T_aligned})

    return aligned


def align_prediction_to_gt_pose_sim3(
    pred_points: np.ndarray,
    pred_cams: Sequence[Dict[str, object]],
    gt_cams: Sequence[Dict[str, object]],
) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
    """Fixed final pred->GT pose_sim3 alignment for visualization/evaluation.

    This function intentionally has no CLI mode switch. It is always pose_sim3.
    GT stays unchanged. Prediction point cloud and prediction cameras are
    transformed into the GT pose coordinate system.
    """
    gt_corr, pred_corr, matched_stems = _matched_pose_center_correspondences(
        gt_cams=gt_cams,
        pred_cams=pred_cams,
    )

    align_meta: Dict[str, object] = {
        "mode": "pose_sim3",
        "valid": False,
        "source": "pose_translations",
        "num_corr": int(gt_corr.shape[0]),
        "matched_camera_stems": matched_stems,
        "scale": 1.0,
        "R": np.eye(3, dtype=np.float32).tolist(),
        "t": np.zeros(3, dtype=np.float32).tolist(),
        "median_residual": float("nan"),
        "note": "no alignment applied",
    }

    if gt_corr.shape[0] < 3:
        align_meta["note"] = (
            f"not enough matched camera centers for pose_sim3: "
            f"{gt_corr.shape[0]} < 3; using raw prediction"
        )
        print(f"[WARN] {align_meta['note']}")
        return pred_points, list(pred_cams), align_meta

    scale, R, t, valid, note = _estimate_similarity_umeyama(
        src=pred_corr,
        dst=gt_corr,
        estimate_scale=True,
    )

    if not valid:
        align_meta["note"] = f"{note}; using raw prediction"
        print(f"[WARN] {align_meta['note']}")
        return pred_points, list(pred_cams), align_meta

    pred_points_aligned = _apply_similarity_to_points(pred_points, scale, R, t)
    pred_cams_aligned = _apply_similarity_to_cameras(pred_cams, scale, R, t)

    pred_corr_aligned = _apply_similarity_to_points(pred_corr, scale, R, t)
    residual = np.linalg.norm(
        pred_corr_aligned.astype(np.float64) - gt_corr.astype(np.float64),
        axis=1,
    )
    median_residual = float(np.median(residual)) if residual.size else float("nan")

    align_meta.update(
        {
            "valid": True,
            "num_corr": int(gt_corr.shape[0]),
            "matched_camera_stems": matched_stems,
            "scale": float(scale),
            "R": R.astype(np.float32).tolist(),
            "t": t.astype(np.float32).tolist(),
            "median_residual": median_residual,
            "note": (
                "fixed final pred->GT pose_sim3 alignment applied to predicted "
                "point cloud and predicted cameras"
            ),
        }
    )

    print(
        "Final pose_sim3 alignment: "
        f"valid=True, num_corr={align_meta['num_corr']}, "
        f"scale={align_meta['scale']:.6g}, "
        f"median_residual={median_residual:.6g}"
    )

    return pred_points_aligned, pred_cams_aligned, align_meta


# ---------------------------------------------------------------------------
# Final eval output saving (standardized format)
# ---------------------------------------------------------------------------


def save_final_eval_outputs(
    eval_dir: Path,
    pred_cams: Sequence[Dict[str, object]],
    gt_cams: Sequence[Dict[str, object]],
    pred_points: np.ndarray,
    pred_colors: np.ndarray,
    gt_points: np.ndarray,
    gt_colors: np.ndarray,
    meta: Dict[str, object],
) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)

    stems: List[str] = []
    Ts: List[np.ndarray] = []
    valid: List[bool] = []

    for cam in pred_cams:
        stem = str(cam.get("stem", ""))
        T = np.asarray(cam.get("T_c2w"), dtype=np.float32)
        ok = T.shape == (4, 4) and np.isfinite(T).all()

        stems.append(stem)
        Ts.append(T if ok else np.full((4, 4), np.nan, dtype=np.float32))
        valid.append(bool(ok))

    np.savez_compressed(
        eval_dir / "pred_cameras.npz",
        stems=np.asarray(stems, dtype=str),
        T_c2w=np.stack(Ts, axis=0).astype(np.float32) if Ts else np.empty((0, 4, 4), dtype=np.float32),
        valid=np.asarray(valid, dtype=bool),
    )

    gt_stems: List[str] = []
    gt_Ts: List[np.ndarray] = []
    gt_valid: List[bool] = []
    for cam in gt_cams:
        stem = str(cam.get("stem", ""))
        T = np.asarray(cam.get("T_c2w"), dtype=np.float32)
        ok = T.shape == (4, 4) and np.isfinite(T).all()

        gt_stems.append(stem)
        gt_Ts.append(T if ok else np.full((4, 4), np.nan, dtype=np.float32))
        gt_valid.append(bool(ok))

    np.savez_compressed(
        eval_dir / "gt_cameras.npz",
        stems=np.asarray(gt_stems, dtype=str),
        T_c2w=np.stack(gt_Ts, axis=0).astype(np.float32)
        if gt_Ts
        else np.empty((0, 4, 4), dtype=np.float32),
        valid=np.asarray(gt_valid, dtype=bool),
    )

    points = np.asarray(pred_points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(pred_colors, dtype=np.uint8).reshape(-1, 3)
    if colors.shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]

    save_point_cloud_ply(
        eval_dir / "pred_points.ply",
        points=points.astype(np.float32),
        colors=colors.astype(np.uint8),
    )
    save_point_cloud_ply(
        eval_dir / "gt_points.ply",
        points=gt_points,
        colors=gt_colors,
    )

    meta = dict(meta)
    meta["num_cameras"] = int(len(stems))
    meta["num_valid_cameras"] = int(np.asarray(valid, dtype=bool).sum())
    meta["num_gt_cameras"] = int(len(gt_stems))
    meta["num_valid_gt_cameras"] = int(np.asarray(gt_valid, dtype=bool).sum())
    meta["num_points"] = int(points.shape[0])
    meta["num_gt_points"] = int(
        np.asarray(gt_points, dtype=np.float32).reshape(-1, 3).shape[0]
    )
    meta["pred_points_path"] = "pred_points.ply"
    meta["gt_points_path"] = "gt_points.ply"
    meta["pred_cameras_path"] = "pred_cameras.npz"
    meta["gt_cameras_path"] = "gt_cameras.npz"

    (eval_dir / "meta.json").write_text(
        json.dumps(json_safe(meta), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved final eval outputs: {eval_dir}")


def rr_set_time(name: str, sequence: int) -> None:
    rr_mod, _ = require_rerun()
    try:
        rr_mod.set_time(name, sequence=sequence)
    except AttributeError:
        rr_mod.set_time_sequence(name, sequence)


def rr_init_save(app_id: str, recording_id: str, save_rrd: Path) -> None:
    rr_mod, _ = require_rerun()
    try:
        rr_mod.init(app_id, recording_id=recording_id, spawn=False)
    except TypeError:
        rr_mod.init(app_id, spawn=False)
    rr_mod.save(str(save_rrd))


def rr_disconnect() -> None:
    rr_mod, _ = require_rerun()
    disconnect_fn = getattr(rr_mod, "disconnect", None)
    shutdown_fn = getattr(rr_mod, "shutdown", None)
    try:
        if callable(disconnect_fn):
            disconnect_fn()
        elif callable(shutdown_fn):
            shutdown_fn()
    except Exception:
        pass


def log_view_coordinates(mode: str) -> None:
    rr_mod, _ = require_rerun()
    obj = getattr(rr_mod.ViewCoordinates, str(mode), None)
    if obj is None:
        obj = getattr(rr_mod.ViewCoordinates, "RDF")
    rr_mod.log("world", obj() if callable(obj) else obj, static=True)


def send_blueprint(background=(255, 255, 255), hide_grid: bool = False) -> None:
    rr_mod, rrb_mod = require_rerun()
    if rrb_mod is None:
        return
    try:
        line_grid = rrb_mod.LineGrid3D(visible=not hide_grid)
        blueprint = rrb_mod.Blueprint(
            rrb_mod.Spatial3DView(origin="/world", name="VGGT-SLAM", background=list(background), line_grid=line_grid),
            collapse_panels=True,
        )
        rr_mod.send_blueprint(blueprint)
    except Exception as e:
        print(f"[WARN] failed to send Rerun blueprint: {e}")


def camera_axis_strips(cams: Sequence[Dict[str, object]], axis_size: float, color_xyz: Optional[Sequence[Sequence[int]]] = None):
    strips: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    if color_xyz is None:
        color_xyz = ((255, 0, 0), (0, 220, 0), (40, 80, 255))
    color_xyz_np = [np.asarray(c, dtype=np.uint8) for c in color_xyz]
    for cam in cams:
        T = np.asarray(cam["T_c2w"], dtype=np.float64)
        R = T[:3, :3]
        o = T[:3, 3]
        for axis in range(3):
            strips.append(np.stack([o, o + R[:, axis] * axis_size], axis=0).astype(np.float32))
            colors.append(color_xyz_np[axis])
    return strips, colors


def estimate_axis_size(point_arrays: Sequence[np.ndarray], cams: Sequence[Dict[str, object]], explicit: float) -> float:
    if explicit > 0:
        return float(explicit)
    valid_parts = []
    for pts in point_arrays:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        finite = np.isfinite(pts).all(axis=1)
        if finite.any():
            valid_parts.append(pts[finite])
    if not valid_parts and cams:
        valid_parts.append(np.asarray([np.asarray(c["T_c2w"])[:3, 3] for c in cams], dtype=np.float32))
    if not valid_parts:
        return 0.1
    pts = np.concatenate(valid_parts, axis=0)
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    if not np.isfinite(diag) or diag <= 0:
        diag = 1.0
    return max(diag * 0.03, 1e-4)


def log_world_axes(points: np.ndarray, cams: Sequence[Dict[str, object]], axis_size: float, radius: float) -> None:
    rr_mod, _ = require_rerun()
    size = estimate_axis_size([points], cams, axis_size)
    origin = np.zeros(3, dtype=np.float32)
    strips = [
        np.stack([origin, origin + np.asarray([size, 0, 0], dtype=np.float32)]),
        np.stack([origin, origin + np.asarray([0, size, 0], dtype=np.float32)]),
        np.stack([origin, origin + np.asarray([0, 0, size], dtype=np.float32)]),
    ]
    kwargs = {
        "strips": strips,
        "colors": [
            np.asarray([255, 0, 0], dtype=np.uint8),
            np.asarray([0, 220, 0], dtype=np.uint8),
            np.asarray([40, 80, 255], dtype=np.uint8),
        ],
    }
    if radius > 0:
        kwargs["radii"] = float(radius)
    rr_mod.log("world/world_axes", rr_mod.LineStrips3D(**kwargs))


def write_rrd(
    args: argparse.Namespace,
    method: str,
    output_rrd: Path,
    prepared_metadata: Sequence[Dict[str, object]],
    prepared_images: Sequence[Path],
    cams: Sequence[Dict[str, object]],
    points: np.ndarray,
    colors: np.ndarray,
    gt_cams: Sequence[Dict[str, object]],
    gt_points: np.ndarray,
    gt_colors: np.ndarray,
) -> None:
    cv2_mod = require_cv2()
    rr_mod, _ = require_rerun()
    output_rrd.parent.mkdir(parents=True, exist_ok=True)
    scene_name = sanitize_name(Path(args.scene_dir).resolve().name)
    recording_id = f"{method}_{scene_name}"
    rr_init_save("run_vggt_slam_to_rrd", recording_id, output_rrd)
    rr_set_time("frame", 0)
    log_view_coordinates(args.view_coordinates)
    send_blueprint(background=tuple(args.background), hide_grid=args.hide_grid)

    if gt_points.shape[0] > 0:
        kwargs = {"positions": gt_points.astype(np.float32), "colors": gt_colors.astype(np.uint8)}
        if args.point_radius > 0:
            kwargs["radii"] = float(args.point_radius)
        rr_mod.log("world/gt/points", rr_mod.Points3D(**kwargs))

    if points.shape[0] > 0:
        kwargs = {"positions": points.astype(np.float32), "colors": colors.astype(np.uint8)}
        if args.point_radius > 0:
            kwargs["radii"] = float(args.point_radius)
        rr_mod.log("world/pred/points", rr_mod.Points3D(**kwargs))

    axis_size = estimate_axis_size([points, gt_points], list(cams) + list(gt_cams), args.camera_axis_size)
    if gt_cams:
        gt_centers = np.asarray([np.asarray(c["T_c2w"])[:3, 3] for c in gt_cams], dtype=np.float32)
        gt_labels = [str(c["stem"]) for c in gt_cams]
        try:
            rr_mod.log("world/cameras/gt/centers", rr_mod.Points3D(positions=gt_centers, labels=gt_labels, radii=0.0))
        except TypeError:
            rr_mod.log("world/cameras/gt/centers", rr_mod.Points3D(positions=gt_centers, labels=gt_labels))
        gt_strips, gt_strip_colors = camera_axis_strips(
            gt_cams,
            axis_size=axis_size,
            color_xyz=((255, 0, 0), (0, 220, 0), (40, 80, 255)),
        )
        kwargs = {"strips": gt_strips, "colors": gt_strip_colors}
        if args.camera_axis_radius > 0:
            kwargs["radii"] = float(args.camera_axis_radius)
        rr_mod.log("world/cameras/gt/axes", rr_mod.LineStrips3D(**kwargs))
        rr_mod.log(
            "world/cameras/gt/trajectory",
            rr_mod.LineStrips3D(strips=[gt_centers], colors=[np.asarray([40, 120, 40], dtype=np.uint8)]),
        )

    if cams:
        centers = np.asarray([np.asarray(c["T_c2w"])[:3, 3] for c in cams], dtype=np.float32)
        labels = [str(c["stem"]) for c in cams]
        try:
            rr_mod.log("world/cameras/pred/centers", rr_mod.Points3D(positions=centers, labels=labels, radii=0.0))
        except TypeError:
            rr_mod.log("world/cameras/pred/centers", rr_mod.Points3D(positions=centers, labels=labels))

        strips, strip_colors = camera_axis_strips(
            cams,
            axis_size=axis_size,
            color_xyz=((255, 0, 255), (255, 180, 0), (0, 220, 255)),
        )
        kwargs = {"strips": strips, "colors": strip_colors}
        if args.camera_axis_radius > 0:
            kwargs["radii"] = float(args.camera_axis_radius)
        rr_mod.log("world/cameras/pred/axes", rr_mod.LineStrips3D(**kwargs))
        rr_mod.log(
            "world/cameras/pred/trajectory",
            rr_mod.LineStrips3D(strips=[centers], colors=[np.asarray([0, 0, 0], dtype=np.uint8)]),
        )

    if args.log_images:
        for meta, image_path in zip(prepared_metadata, prepared_images):
            img = cv2_mod.imread(str(image_path), cv2_mod.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2_mod.cvtColor(img, cv2_mod.COLOR_BGR2RGB)
            rr_mod.log(
                f"inputs/view_{int(meta['index']):06d}_{sanitize_name(str(meta['stem']))}/rgb",
                rr_mod.Image(img),
            )

    if args.show_world_axes:
        world_points = gt_points if gt_points.shape[0] > 0 else points
        log_world_axes(world_points, list(cams) + list(gt_cams), args.world_axis_size, args.world_axis_radius)

    rr_disconnect()
    print(f"Saved Rerun recording: {output_rrd}")


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        default=None,
        help=(
            "vggt-slam (version1.0), vggt-slam-sim3, vggt-slam-sl4, "
            "vggt-slam2.0 (main), or vggt-long"
        ),
    )
    parser.add_argument("--scene_dir", required=True, help="Scene folder containing an images/ directory.")
    parser.add_argument("--output_rrd", required=True, help="Output .rrd path.")
    parser.add_argument("--output_dir", default=None, help="Directory for prepared images and VGGT-SLAM logs.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run third_party VGGT-SLAM.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or numeric CUDA index.")

    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--cams_dir", default="cams")
    parser.add_argument("--depth_dir", default="depth")
    parser.add_argument("--frame_glob", default="*")
    parser.add_argument("--num_views", type=int, default=0, help="Select at most this many views after filtering; <=0 means all.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_side", type=int, default=0, help="Resize prepared input frames so max(H,W)<=max_side; <=0 keeps source size.")
    parser.add_argument("--size_multiple", type=int, default=1, help="Round prepared image size down to this multiple after --max_side.")
    parser.add_argument("--copy_images", action="store_true", help="Copy prepared images even when no resizing is needed.")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--depth_min", type=float, default=1e-6)
    parser.add_argument("--depth_max", type=float, default=1e6)
    parser.add_argument(
        "--scene_io_workers",
        type=int,
        default=0,
        help="Number of worker threads for loading GT RGB/depth/camera artifacts; <=1 is serial.",
    )

    parser.add_argument("--submap_size", type=int, default=16)
    parser.add_argument("--overlapping_window_size", type=int, default=1)
    parser.add_argument("--max_loops", type=int, default=1)
    parser.add_argument("--min_disparity", type=float, default=50.0)
    parser.add_argument("--conf_threshold", type=float, default=25.0)
    parser.add_argument("--vis_map", action="store_true")
    parser.add_argument("--vis_flow", action="store_true")
    parser.add_argument("--skip_dense_log", action="store_true")

    parser.add_argument("--use_sim3", action="store_true", help="VGGT-SLAM 1.0 only.")
    parser.add_argument("--plot_focal_lengths", action="store_true", help="VGGT-SLAM 1.0 only.")
    parser.add_argument("--downsample_factor", type=int, default=1, help="VGGT-SLAM 1.0 only.")
    parser.add_argument("--use_point_map", action="store_true", help="VGGT-SLAM 1.0 only.")
    parser.add_argument("--vis_stride", type=int, default=1, help="VGGT-SLAM 1.0 only.")
    parser.add_argument("--vis_point_size", type=float, default=0.003, help="VGGT-SLAM 1.0 only.")

    parser.add_argument("--lc_thres", type=float, default=0.95, help="VGGT-SLAM 2.0 only.")
    parser.add_argument("--vis_voxel_size", type=float, default=None, help="VGGT-SLAM 2.0 only.")
    parser.add_argument("--run_os", action="store_true", help="VGGT-SLAM 2.0 open-set semantic mode.")
    parser.add_argument("--vggt_long_config", default=None, help="VGGT-Long config yaml. Defaults to its configs/base_config.yaml.")
    parser.add_argument(
        "--vggt_model_path",
        default=None,
        help="VGGT checkpoint used by VGGT-SLAM or VGGT-Long; also exported as VGGT_MODEL_PATH.",
    )

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

    # Final output point downsampling policy (matching predict_scene_to_rrd_spatial.py).
    parser.add_argument(
        "--max_points_per_view",
        type=int,
        default=500000,
        help="Final output point cap, matching predict_scene_to_rrd_spatial.py.",
    )
    parser.add_argument(
        "--voxel_downsample",
        type=float,
        default=0.01,
        help="Final output voxel size, matching predict_scene_to_rrd_spatial.py.",
    )
    parser.add_argument(
        "--no_point_downsample",
        action="store_false",
        dest="point_downsample",
        help="Disable final output point sampling and voxel downsampling.",
    )
    parser.set_defaults(point_downsample=True)

    parser.add_argument(
        "--no_headless",
        action="store_false",
        dest="headless",
        help="Allow VGGT-SLAM to start Viser visualization.",
    )
    parser.set_defaults(headless=True)

    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep selected_images and temporary VGGT-SLAM point/pose logs.",
    )
    parser.add_argument(
        "--global_point_stride",
        type=int,
        default=1,
        help="Stride passed to VGGT-SLAM global point export before sampling.",
    )
    return parser.parse_known_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, passthrough = parse_args(argv)
    if args.method is None:
        raise ValueError(
            "--method is required. Use vggt-slam, vggt-slam-sim3, "
            "vggt-slam-sl4, vggt-slam2.0, or vggt-long."
        )
    requested_method = str(args.method)
    method = normalize_method(args.method)
    method_variant = normalize_method_variant(args.method)
    if method == "vggt-slam" and method_variant == "sim3":
        args.use_sim3 = True
    elif method == "vggt-slam" and method_variant == "sl4":
        args.use_sim3 = False
    repo_root = Path(__file__).resolve().parents[1]
    method_root = resolve_method_root(repo_root, method)
    method_script = str(METHODS[method].get("script", "main.py"))
    if not (method_root / method_script).exists():
        raise RuntimeError(
            f"Missing checkout for {method}: {method_root}. "
            f"Expected script: {method_script}"
        )

    output_rrd = Path(args.output_rrd).expanduser().resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = output_rrd.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    stems, image_paths = select_images(args)
    prepared_dir = output_dir / "selected_images"
    prepared_images, prepared_metadata = materialize_images(
        image_paths=image_paths,
        stems=stems,
        out_dir=prepared_dir,
        max_side=int(args.max_side),
        size_multiple=int(args.size_multiple),
        copy_images=bool(args.copy_images),
    )
    print(
        f"Selected {len(prepared_images)} images for {METHODS[method]['display']}; "
        f"prepared input: {prepared_dir}"
    )

    pose_log_path = output_dir / ("camera_poses.txt" if method == "vggt-long" else "poses.txt")
    stdout_log_path = output_dir / "vggt_slam_stdout.log"
    timing_path = output_dir / "processing_time.json"
    if timing_path.exists():
        timing_path.unlink()
    vggt_long_existing_exps: List[Path] = []
    if method == "vggt-long":
        for stale_path in (
            output_dir / "camera_poses.txt",
            output_dir / "intrinsic.txt",
            output_dir / "pcd" / "combined_pcd.ply",
        ):
            if stale_path.exists():
                stale_path.unlink()
        vggt_long_existing_exps = _collect_vggt_long_exps(method_root, prepared_dir)
    cmd = build_external_command(
        args=args,
        method=method,
        method_root=method_root,
        image_folder=prepared_dir,
        pose_log_path=pose_log_path,
        timing_path=timing_path,
        passthrough=passthrough,
    )
    env = build_env(args, method_root=method_root)
    return_code = run_external(cmd, cwd=method_root, env=env, log_path=stdout_log_path)
    external_outputs: Dict[str, object] = {}
    if method == "vggt-long":
        external_outputs = stage_vggt_long_outputs(
            method_root=method_root,
            image_folder=prepared_dir,
            output_dir=output_dir,
            require_outputs=(return_code == 0),
            exclude_exp_dirs=vggt_long_existing_exps,
        )

    processing_time_meta = load_json_file(timing_path)

    if return_code != 0:
        sidecar = output_rrd.with_suffix(".json")
        payload = {
            "method": method,
            "requested_method": requested_method,
            "method_variant": method_variant,
            "method_display": METHODS[method]["display"],
            "scene_dir": str(Path(args.scene_dir).expanduser().resolve()),
            "output_rrd": str(output_rrd),
            "output_dir": str(output_dir),
            "prepared_dir": str(prepared_dir),
            "pose_log_path": str(pose_log_path),
            "stdout_log_path": str(stdout_log_path),
            "timing_path": str(timing_path),
            "processing_time": processing_time_meta,
            "return_code": int(return_code),
            "command": cmd,
            "error": "External VGGT-SLAM process failed; skip heavy post-processing.",
        }
        sidecar.write_text(
            json.dumps(json_safe(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[ERROR] VGGT-SLAM exited with code {return_code}. See {stdout_log_path}")
        print(f"Saved failure sidecar metadata: {sidecar}")
        return int(return_code)

    cams = read_pose_log(pose_log_path, prepared_metadata=prepared_metadata)
    points, colors, point_artifacts = load_point_cloud_artifacts(
        method=method,
        pose_log_path=pose_log_path,
        prepared_images=prepared_images,
        max_points=int(args.max_pred_points),
        seed=int(args.seed),
    )
    gt_cams, gt_points, gt_colors, gt_meta = load_gt_artifacts(
        args=args,
        stems=stems,
        image_paths=image_paths,
    )

    # VGGT-SLAM: final pred -> GT pose_sim3 alignment for visualization/evaluation.
    points, cams, align_meta = align_prediction_to_gt_pose_sim3(
        pred_points=points,
        pred_cams=cams,
        gt_cams=gt_cams,
    )

    # Match pi3x_world_translation-style final point policy.
    points, colors = downsample_final_points(
        points,
        colors,
        enabled=bool(args.point_downsample),
        max_points=int(args.max_points_per_view),
        voxel_size=float(args.voxel_downsample),
        seed=int(args.seed) + 17,
        label="pred",
    )

    gt_points, gt_colors = downsample_final_points(
        gt_points,
        gt_colors,
        enabled=bool(args.point_downsample),
        max_points=int(args.max_points_per_view),
        voxel_size=float(args.voxel_downsample),
        seed=int(args.seed) + 23,
        label="gt",
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
            "script": "adapters/vggt_optim.py",
            "scene_dir": str(Path(args.scene_dir).expanduser().resolve()),
            "method": method,
            "requested_method": requested_method,
            "method_variant": method_variant,
            "method_display": METHODS[method]["display"],
            "pose_convention": "T_c2w",
            "points_coordinate": "same_as_pred_cameras",
            "processing_time": processing_time_meta,
            "post_align": {
                "enabled": True,
                "type": "pose_sim3",
                "target": "gt_pose",
                **{k: v for k, v in align_meta.items() if not isinstance(v, np.ndarray)},
                "scale": float(align_meta.get("scale", 1.0)),
                "R": np.asarray(align_meta.get("R", np.eye(3))).tolist() if isinstance(align_meta.get("R"), np.ndarray) else align_meta.get("R", [[1,0,0],[0,1,0],[0,0,1]]),
                "t": np.asarray(align_meta.get("t", np.zeros(3))).tolist() if isinstance(align_meta.get("t"), np.ndarray) else align_meta.get("t", [0,0,0]),
                "valid": bool(align_meta.get("valid", False)),
                "median_camera_residual": float(align_meta.get("median_residual", float("nan"))),
            },
        },
    )

    sidecar = output_rrd.with_suffix(".json")
    payload = {
        "method": method,
        "requested_method": requested_method,
        "method_variant": method_variant,
        "method_display": METHODS[method]["display"],
        "method_root": method_root,
        "scene_dir": Path(args.scene_dir).expanduser().resolve(),
        "images_dir": args.images_dir,
        "cams_dir": args.cams_dir,
        "depth_dir": args.depth_dir,
        "output_rrd": output_rrd,
        "output_dir": output_dir,
        "prepared_dir": prepared_dir,
        "pose_log_path": pose_log_path,
        "stdout_log_path": stdout_log_path,
        "timing_path": timing_path,
        "processing_time": processing_time_meta,
        "external_outputs": external_outputs,
        "point_artifacts": point_artifacts,
        "return_code": int(return_code),
        "command": cmd,
        "use_sim3": bool(args.use_sim3),
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
        print(f"[ERROR] VGGT-SLAM exited with code {return_code}. See {stdout_log_path}")
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

    if not args.keep_intermediate:
        # Remove heavy or temporary VGGT-SLAM artifacts. Final outputs remain:
        #   - output_rrd
        #   - output_dir/eval/pred_points.ply
        #   - output_dir/eval/gt_points.ply
        #   - small eval camera/meta files for metrics
        for tmp_path in (
            prepared_dir,
            pose_log_path.with_name(pose_log_path.stem + "_points.ply"),
            pose_log_path.with_name(pose_log_path.stem + "_points.pcd"),
            pose_log_path.with_name(pose_log_path.stem + "_logs"),
        ):
            tmp_path = Path(tmp_path)
            try:
                if tmp_path.is_dir():
                    shutil.rmtree(tmp_path)
                    print(f"[CLEANUP] removed directory: {tmp_path}")
                elif tmp_path.exists():
                    tmp_path.unlink()
                    print(f"[CLEANUP] removed file: {tmp_path}")
            except Exception as exc:
                print(f"[CLEANUP][WARN] failed to remove {tmp_path}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
