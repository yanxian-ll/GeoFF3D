# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEPTH_EXTS = {".exr", ".npy", ".png", ".tif", ".tiff"}
CAM_EXTS = {".txt"}


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


def sanitize_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[\\/:*?\"<>|\s]+", "_", name)
    return name or "scene"


def collect_stem_to_path(folder: Path, exts: Iterable[str]) -> Dict[str, Path]:
    exts = {e.lower() for e in exts}
    if not folder.exists():
        return {}
    return {
        p.stem: p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    }


def _float_tokens(line: str):
    try:
        return [float(x) for x in line.replace(",", " ").split()]
    except ValueError:
        return None


def _find_line(lines: Sequence[str], prefixes: Sequence[str]) -> int:
    prefixes = tuple(p.lower().rstrip(":") for p in prefixes)
    for i, line in enumerate(lines):
        low = line.strip().lower().rstrip(":")
        if any(low.startswith(p) for p in prefixes):
            return i
    return -1


def _read_numeric_rows(lines, start, n_rows, n_cols, path):
    rows = []
    for j in range(start, len(lines)):
        vals = _float_tokens(lines[j])
        if vals is None or len(vals) < n_cols:
            continue
        rows.append(vals[:n_cols])
        if len(rows) == n_rows:
            break
    if len(rows) != n_rows:
        raise ValueError(f"Cannot read {n_rows}x{n_cols} matrix from {path}")
    return np.asarray(rows, dtype=np.float64)


def parse_cam_txt(cam_path: Path) -> Dict[str, object]:
    lines = [ln.strip() for ln in cam_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    idx_ext = _find_line(lines, ["extrinsic"])
    idx_int = _find_line(lines, ["intrinsic"])
    if idx_ext < 0 or idx_int < 0:
        raise ValueError(f"Invalid camera txt: {cam_path}")

    T_w2c = _read_numeric_rows(lines, idx_ext + 1, 4, 4, cam_path)
    K = _read_numeric_rows(lines, idx_int + 1, 3, 3, cam_path)

    height = None
    width = None
    for i, line in enumerate(lines):
        tokens = line.lower().replace(":", " ").split()
        if "h" in tokens and "w" in tokens and ("fov" in tokens or "hfov" in tokens):
            for j in range(i + 1, len(lines)):
                vals = _float_tokens(lines[j])
                if vals is not None and len(vals) >= 2:
                    height = int(round(vals[0]))
                    width = int(round(vals[1]))
                    break
            break

    return {
        "stem": cam_path.stem,
        "K": K,
        "T_w2c": T_w2c,
        "T_c2w": np.linalg.inv(T_w2c),
        "height": height,
        "width": width,
    }


def read_rgb(path: Path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_depth(path: Path, depth_scale: float = 1.0) -> np.ndarray:
    import cv2
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(str(path))
    elif suffix == ".exr":
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            from spatial_rrd.scene_io import read_exr_depth
            depth = read_exr_depth(path)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Cannot read depth: {path}")

    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if float(depth_scale) != 1.0:
        depth = depth / float(depth_scale)
    return depth


def _round_down_to_multiple(x: int, m: int) -> int:
    if int(m) <= 1:
        return int(x)
    return max(int(m), int(x) // int(m) * int(m))


def compute_target_hw(height: int, width: int, max_side: int, size_multiple: int):
    h, w = int(height), int(width)
    if int(max_side) > 0 and max(h, w) > int(max_side):
        scale = float(max_side) / float(max(h, w))
        h = max(1, int(round(h * scale)))
        w = max(1, int(round(w * scale)))
    h = _round_down_to_multiple(h, int(size_multiple))
    w = _round_down_to_multiple(w, int(size_multiple))
    return h, w


def resize_rgb_depth_K(rgb, depth, K, cam_width, cam_height, target_h, target_w):
    """Same GT resize rule as scripts/spatial_rrd/scene_io.py."""
    import cv2
    K = np.asarray(K, dtype=np.float64).copy()
    depth_h, depth_w = depth.shape[:2]

    if cam_width is None or cam_height is None:
        cam_height, cam_width = rgb.shape[:2]

    if int(cam_width) != int(depth_w) or int(cam_height) != int(depth_h):
        sx = float(depth_w) / float(cam_width)
        sy = float(depth_h) / float(cam_height)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    if rgb.shape[:2] != depth.shape[:2]:
        rgb = cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_AREA)

    if int(depth_h) != int(target_h) or int(depth_w) != int(target_w):
        sx = float(target_w) / float(depth_w)
        sy = float(target_h) / float(depth_h)
        rgb = cv2.resize(rgb, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (int(target_w), int(target_h)), interpolation=cv2.INTER_NEAREST)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    return rgb.astype(np.uint8), depth.astype(np.float32), K


def depth_pixels_to_world_points_numpy(depth, K, T_c2w, rows, cols):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols = np.asarray(cols, dtype=np.int64).reshape(-1)

    z = depth[rows, cols].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)

    R = np.asarray(T_c2w, dtype=np.float64)[:3, :3]
    t = np.asarray(T_c2w, dtype=np.float64)[:3, 3]
    pts_world = pts_cam @ R.T + t[None, :]
    return pts_world.astype(np.float32)


def sample_points(points, colors, max_points: int, seed: int):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if colors.shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]

    if int(max_points) > 0 and points.shape[0] > int(max_points):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(points.shape[0], size=int(max_points), replace=False)
        points = points[idx]
        colors = colors[idx]
    return points.astype(np.float32), colors.astype(np.uint8)


def voxel_downsample_numpy(points, colors, voxel_size: float):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if float(voxel_size) <= 0 or points.shape[0] == 0:
        return points, colors

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    if points.shape[0] == 0:
        return points, colors

    vox = np.floor(points.astype(np.float64) / float(voxel_size)).astype(np.int64)
    _, idx = np.unique(vox, axis=0, return_index=True)
    return points[idx].astype(np.float32), colors[idx].astype(np.uint8)


def prepared_stem_from_path(path: Path):
    stem = Path(path).stem
    m = re.match(r"^(\d+?)__(.+)$", stem)
    if m:
        return int(m.group(1)), m.group(2)
    m = re.search(r"\d+(?:\.\d+)?", stem)
    return int(float(m.group())) if m else -1, stem


def build_frame_to_stem(image_names: Sequence[str]):
    mapping = {}
    for p in image_names:
        idx, stem = prepared_stem_from_path(Path(p))
        if idx >= 0:
            mapping[idx] = stem
    return mapping


def collect_pred_cameras_from_solver(solver, image_names):
    frame_to_stem = build_frame_to_stem(image_names)
    cams = []
    for submap in solver.map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        poses = submap.get_all_poses_world(solver.graph, give_camera_mat=False)
        frame_ids = submap.get_frame_ids()
        for frame_id, T in zip(frame_ids, poses):
            frame_idx = int(round(float(frame_id)))
            cams.append(
                {
                    "frame_id": float(frame_id),
                    "stem": frame_to_stem.get(frame_idx, f"frame_{frame_idx:06d}"),
                    "T_c2w": np.asarray(T, dtype=np.float32).reshape(4, 4),
                }
            )
    cams.sort(key=lambda c: float(c.get("frame_id", 0.0)))
    return cams


def load_gt_artifacts(
    scene_dir: Path,
    selected_stems: Sequence[str],
    images_dir: str,
    cams_dir: str,
    depth_dir: str,
    depth_scale: float,
    depth_min: float,
    depth_max: float,
    max_gt_points: int,
    voxel_size: float,
    seed: int,
    target_h: Optional[int],
    target_w: Optional[int],
    max_side: int,
    size_multiple: int,
):
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    scene_dir = Path(scene_dir).expanduser().resolve()
    img_paths = collect_stem_to_path(scene_dir / images_dir, IMAGE_EXTS)
    cam_paths = collect_stem_to_path(scene_dir / cams_dir, CAM_EXTS)
    depth_paths = collect_stem_to_path(scene_dir / depth_dir, DEPTH_EXTS)

    # ------------------------------------------------------------------
    # GT cameras and GT depth point cloud must be treated separately.
    # A scene may have camera GT but no depth GT. We still need to save
    # GT poses for pose-Sim3 alignment and visualization.
    # ------------------------------------------------------------------
    cam_eligible = [s for s in selected_stems if s in cam_paths]
    depth_eligible = [
        s for s in selected_stems
        if s in img_paths and s in cam_paths and s in depth_paths
    ]

    if int(max_gt_points) > 0 and depth_eligible:
        per_frame_cap = max(1, int(np.ceil(float(max_gt_points) * 1.5 / float(len(depth_eligible)))))
    else:
        per_frame_cap = 0

    print(
        f"[VGGT-SLAM2][GT] loading GT artifacts: "
        f"cams={len(cam_eligible)}/{len(selected_stems)}, "
        f"rgbd={len(depth_eligible)}/{len(selected_stems)}, "
        f"max_gt_points={int(max_gt_points)}, per_frame_cap={int(per_frame_cap)}"
    )

    gt_cams = []
    gt_cam_by_stem = {}
    point_parts = []
    color_parts = []
    num_cam_used = 0
    num_depth_used = 0

    # 1) Always load GT cameras when cams exist, even if depth does not.
    for stem in cam_eligible:
        try:
            cam = parse_cam_txt(cam_paths[stem])
            gt_cams.append({"stem": stem, "T_c2w": np.asarray(cam["T_c2w"], dtype=np.float32)})
            gt_cam_by_stem[stem] = cam
            num_cam_used += 1
        except Exception as exc:
            print(f"[WARN] failed to parse GT camera for {stem}: {exc}")

    # 2) Load GT depth point cloud only for frames that have depth.
    #    Missing depth should not remove GT cameras.
    depth_iter = tqdm(depth_eligible, desc="[VGGT-SLAM2][GT]", dynamic_ncols=True) if tqdm and depth_eligible else depth_eligible

    for i, stem in enumerate(depth_iter):
        cam = gt_cam_by_stem.get(stem)
        if cam is None:
            continue

        try:
            rgb = read_rgb(img_paths[stem])
            depth = read_depth(depth_paths[stem], depth_scale=float(depth_scale))

            if target_h is None or target_w is None:
                gt_h, gt_w = compute_target_hw(depth.shape[0], depth.shape[1], max_side, size_multiple)
            else:
                gt_h, gt_w = int(target_h), int(target_w)

            rgb, depth, K = resize_rgb_depth_K(
                rgb=rgb,
                depth=depth,
                K=np.asarray(cam["K"], dtype=np.float64),
                cam_width=cam.get("width"),
                cam_height=cam.get("height"),
                target_h=gt_h,
                target_w=gt_w,
            )

            valid = (
                np.isfinite(depth)
                & (depth > float(depth_min))
                & (depth < float(depth_max))
            )
            if not bool(valid.any()):
                continue

            rows, cols_px = np.nonzero(valid)
            if rows.shape[0] <= 0:
                continue

            if int(per_frame_cap) > 0 and rows.shape[0] > int(per_frame_cap):
                rng = np.random.default_rng(int(seed) + 104729 * (i + 1))
                sel = rng.choice(rows.shape[0], size=int(per_frame_cap), replace=False)
                rows = rows[sel]
                cols_px = cols_px[sel]

            pts = depth_pixels_to_world_points_numpy(
                depth,
                K,
                np.asarray(cam["T_c2w"], dtype=np.float64),
                rows,
                cols_px,
            )
            cols = rgb[rows, cols_px].reshape(-1, 3).astype(np.uint8)

            finite = np.isfinite(pts).all(axis=1)
            pts = pts[finite].astype(np.float32)
            cols = cols[finite].astype(np.uint8)

            if pts.shape[0] > 0:
                point_parts.append(pts)
                color_parts.append(cols)
                num_depth_used += 1

        except Exception as exc:
            print(f"[WARN] failed to load GT for {stem}: {exc}")

    if point_parts:
        gt_points = np.concatenate(point_parts, axis=0)
        gt_colors = np.concatenate(color_parts, axis=0)
    else:
        gt_points = np.empty((0, 3), dtype=np.float32)
        gt_colors = np.empty((0, 3), dtype=np.uint8)

    gt_points, gt_colors = sample_points(gt_points, gt_colors, int(max_gt_points), int(seed) + 911)
    if float(voxel_size) > 0:
        gt_points, gt_colors = voxel_downsample_numpy(gt_points, gt_colors, float(voxel_size))

    meta = {
        "scene_dir": str(scene_dir),
        "images_dir": images_dir,
        "cams_dir": cams_dir,
        "depth_dir": depth_dir,
        "num_gt_cameras": int(num_cam_used),
        "num_depth_pointmaps_used": int(num_depth_used),
        "num_selected_stems": int(len(selected_stems)),
        "num_cam_eligible": int(len(cam_eligible)),
        "num_depth_eligible": int(len(depth_eligible)),
        "has_gt_pose": bool(num_cam_used > 0),
        "has_gt_depth": bool(num_depth_used > 0),
        "max_gt_points": int(max_gt_points),
        "per_frame_cap": int(per_frame_cap),
        "target_h": int(target_h) if target_h is not None else None,
        "target_w": int(target_w) if target_w is not None else None,
        "max_side": int(max_side),
        "size_multiple": int(size_multiple),
        "gt_resize_policy": "spatial_compatible_resize_rgb_depth_K_then_sample_pixels",
    }
    print(
        f"[VGGT-SLAM2][GT] loaded GT artifacts: "
        f"cameras={len(gt_cams)}, "
        f"depth_frames={num_depth_used}, "
        f"points={gt_points.shape[0]:,}"
    )
    return gt_cams, gt_points, gt_colors, meta


def _camera_centers_by_stem(cams: Sequence[Dict[str, object]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for cam in cams:
        stem = str(cam.get("stem", ""))
        T = np.asarray(cam.get("T_c2w"), dtype=np.float64)
        if stem and T.shape == (4, 4) and np.isfinite(T).all():
            out[stem] = T[:3, 3].astype(np.float64)
    return out


def _matched_pose_center_correspondences(gt_cams, pred_cams):
    gt_by = _camera_centers_by_stem(gt_cams)
    pred_by = _camera_centers_by_stem(pred_cams)
    gt_corr: List[np.ndarray] = []
    pred_corr: List[np.ndarray] = []
    stems: List[str] = []
    for stem in sorted(set(gt_by).intersection(pred_by)):
        gt_corr.append(gt_by[stem])
        pred_corr.append(pred_by[stem])
        stems.append(stem)
    if not gt_corr:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), []
    return np.asarray(gt_corr, dtype=np.float32), np.asarray(pred_corr, dtype=np.float32), stems


def _estimate_similarity_umeyama(src, dst, estimate_scale=True, eps=1e-12):
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] < 3 or dst.shape != src.shape:
        return 1.0, np.eye(3), np.zeros(3), False, "not enough correspondences"
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[finite]
    dst = dst[finite]
    if src.shape[0] < 3:
        return 1.0, np.eye(3), np.zeros(3), False, "not enough finite correspondences"

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    src_var = float(np.mean(np.sum(src_c * src_c, axis=1)))
    if src_var <= eps:
        return 1.0, np.eye(3), np.zeros(3), False, "degenerate source variance"

    cov = (dst_c.T @ src_c) / float(src.shape[0])
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if estimate_scale:
        scale = float(np.sum(D * np.diag(S)) / src_var)
    else:
        scale = 1.0
    if not np.isfinite(scale) or scale <= eps:
        return 1.0, np.eye(3), np.zeros(3), False, "invalid scale"
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t, True, "ok"


def _apply_similarity_to_points(points, scale, R, t):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    out = (float(scale) * pts.astype(np.float64)) @ R.T
    out += t[None, :]
    return out.astype(np.float32)


def _apply_similarity_to_cameras(cams, scale, R, t):
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    aligned = []
    for cam in cams:
        T = np.asarray(cam["T_c2w"], dtype=np.float64)
        T_aligned = np.eye(4, dtype=np.float32)
        T_aligned[:3, :3] = (R @ T[:3, :3]).astype(np.float32)
        T_aligned[:3, 3] = (R @ (float(scale) * T[:3, 3]) + t).astype(np.float32)
        aligned.append({**cam, "T_c2w": T_aligned})
    return aligned


def align_prediction_to_gt_pose_sim3(pred_points, pred_cams, gt_cams):
    gt_corr, pred_corr, matched_stems = _matched_pose_center_correspondences(gt_cams, pred_cams)
    meta = {
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
        meta["note"] = f"not enough matched camera centers for pose_sim3: {gt_corr.shape[0]} < 3"
        print(f"[WARN] {meta['note']}")
        return pred_points, list(pred_cams), meta

    scale, R, t, valid, note = _estimate_similarity_umeyama(pred_corr, gt_corr, estimate_scale=True)
    if not valid:
        meta["note"] = f"{note}; using raw prediction"
        print(f"[WARN] {meta['note']}")
        return pred_points, list(pred_cams), meta

    pred_points_aligned = _apply_similarity_to_points(pred_points, scale, R, t)
    pred_cams_aligned = _apply_similarity_to_cameras(pred_cams, scale, R, t)
    pred_corr_aligned = _apply_similarity_to_points(pred_corr, scale, R, t)
    residual = np.linalg.norm(pred_corr_aligned.astype(np.float64) - gt_corr.astype(np.float64), axis=1)
    median_residual = float(np.median(residual)) if residual.size else float("nan")
    meta.update(
        {
            "valid": True,
            "num_corr": int(gt_corr.shape[0]),
            "matched_camera_stems": matched_stems,
            "scale": float(scale),
            "R": R.astype(np.float32).tolist(),
            "t": t.astype(np.float32).tolist(),
            "median_residual": median_residual,
            "note": "source-internal pred->GT pose_sim3 alignment applied",
        }
    )
    print(
        "[VGGT-SLAM2][ALIGN] pose_sim3: "
        f"valid=True, num_corr={meta['num_corr']}, scale={meta['scale']:.6g}, "
        f"median_residual={median_residual:.6g}"
    )
    return pred_points_aligned, pred_cams_aligned, meta


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

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices = np.empty(points.shape[0], dtype=dtype)
    if points.shape[0] > 0:
        vertices["x"] = points[:, 0]
        vertices["y"] = points[:, 1]
        vertices["z"] = points[:, 2]
        vertices["red"] = colors[:, 0]
        vertices["green"] = colors[:, 1]
        vertices["blue"] = colors[:, 2]
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)


def _save_camera_npz(path: Path, cams: Sequence[Dict[str, object]]) -> None:
    stems: List[str] = []
    Ts: List[np.ndarray] = []
    valid: List[bool] = []
    for cam in cams:
        stem = str(cam.get("stem", ""))
        T = np.asarray(cam.get("T_c2w"), dtype=np.float32)
        ok = T.shape == (4, 4) and np.isfinite(T).all()
        stems.append(stem)
        Ts.append(T if ok else np.full((4, 4), np.nan, dtype=np.float32))
        valid.append(bool(ok))
    np.savez_compressed(
        path,
        stems=np.asarray(stems, dtype=str),
        T_c2w=np.stack(Ts, axis=0).astype(np.float32) if Ts else np.empty((0, 4, 4), dtype=np.float32),
        valid=np.asarray(valid, dtype=bool),
    )


def save_eval_outputs(eval_dir: Path, pred_cams, gt_cams, pred_points, pred_colors, gt_points, gt_colors, meta):
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    _save_camera_npz(eval_dir / "pred_cameras.npz", pred_cams)
    _save_camera_npz(eval_dir / "gt_cameras.npz", gt_cams)
    save_point_cloud_ply(eval_dir / "pred_points.ply", pred_points, pred_colors)
    save_point_cloud_ply(eval_dir / "gt_points.ply", gt_points, gt_colors)

    meta = dict(meta)
    meta["num_cameras"] = int(len(pred_cams))
    meta["num_gt_cameras"] = int(len(gt_cams))
    meta["num_points"] = int(np.asarray(pred_points).reshape(-1, 3).shape[0])
    meta["num_gt_points"] = int(np.asarray(gt_points).reshape(-1, 3).shape[0])
    meta["pred_points_path"] = "pred_points.ply"
    meta["gt_points_path"] = "gt_points.ply"
    meta["pred_cameras_path"] = "pred_cameras.npz"
    meta["gt_cameras_path"] = "gt_cameras.npz"
    (eval_dir / "meta.json").write_text(json.dumps(json_safe(meta), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[VGGT-SLAM2][EXPORT] saved eval outputs: {eval_dir}")


def camera_axis_strips(cams: Sequence[Dict[str, object]], axis_size: float):
    strips: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    color_xyz = [np.asarray(c, dtype=np.uint8) for c in ((255, 0, 0), (0, 220, 0), (40, 80, 255))]
    for cam in cams:
        T = np.asarray(cam["T_c2w"], dtype=np.float64)
        R = T[:3, :3]
        o = T[:3, 3]
        for axis in range(3):
            strips.append(np.stack([o, o + R[:, axis] * axis_size], axis=0).astype(np.float32))
            colors.append(color_xyz[axis])
    return strips, colors


def estimate_axis_size(point_arrays: Sequence[np.ndarray], cams: Sequence[Dict[str, object]], explicit: float = 0.0) -> float:
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


def write_rrd(output_rrd: Path, pred_cams, pred_points, pred_colors, gt_cams, gt_points, gt_colors, app_id="vggt_slam2_source_export"):
    import rerun as rr

    output_rrd = Path(output_rrd)
    output_rrd.parent.mkdir(parents=True, exist_ok=True)
    scene_name = sanitize_name(output_rrd.stem)
    try:
        rr.init(app_id, recording_id=scene_name, spawn=False)
    except TypeError:
        rr.init(app_id, spawn=False)
    rr.save(str(output_rrd))
    try:
        rr.set_time("frame", sequence=0)
    except AttributeError:
        rr.set_time_sequence("frame", 0)

    try:
        rr.log("world", rr.ViewCoordinates.RDF(), static=True)
    except Exception:
        pass

    if np.asarray(gt_points).reshape(-1, 3).shape[0] > 0:
        rr.log("world/gt/points", rr.Points3D(positions=np.asarray(gt_points, dtype=np.float32), colors=np.asarray(gt_colors, dtype=np.uint8)))
    if np.asarray(pred_points).reshape(-1, 3).shape[0] > 0:
        rr.log("world/pred/points", rr.Points3D(positions=np.asarray(pred_points, dtype=np.float32), colors=np.asarray(pred_colors, dtype=np.uint8)))

    axis_size = estimate_axis_size([pred_points, gt_points], list(pred_cams) + list(gt_cams))
    for group, cams in (("gt", gt_cams), ("pred", pred_cams)):
        if not cams:
            continue
        centers = np.asarray([np.asarray(c["T_c2w"])[:3, 3] for c in cams], dtype=np.float32)
        labels = [str(c.get("stem", "")) for c in cams]
        try:
            rr.log(f"world/cameras/{group}/centers", rr.Points3D(positions=centers, labels=labels, radii=0.0))
        except TypeError:
            rr.log(f"world/cameras/{group}/centers", rr.Points3D(positions=centers, labels=labels))
        strips, colors = camera_axis_strips(cams, axis_size=axis_size)
        rr.log(f"world/cameras/{group}/axes", rr.LineStrips3D(strips=strips, colors=colors))
        rr.log(f"world/cameras/{group}/trajectory", rr.LineStrips3D(strips=[centers]))

    disconnect = getattr(rr, "disconnect", None)
    shutdown = getattr(rr, "shutdown", None)
    try:
        if callable(disconnect):
            disconnect()
        elif callable(shutdown):
            shutdown()
    except Exception:
        pass
    print(f"[VGGT-SLAM2][EXPORT] saved Rerun recording: {output_rrd}")


def export_solver_outputs(
    solver,
    image_names: Sequence[str],
    output_rrd: Path,
    eval_dir: Path,
    scene_dir: Path,
    images_dir: str = "images",
    cams_dir: str = "cams",
    depth_dir: str = "depth",
    depth_scale: float = 1.0,
    depth_min: float = 1e-6,
    depth_max: float = 1e6,
    max_pred_points: int = 500000,
    max_gt_points: int = 800000,
    voxel_size: float = 0.01,
    point_stride: int = 1,
    seed: int = 0,
    processing_time: Optional[Dict[str, object]] = None,
    method: str = "vggt-slam2.0",
    method_variant: str = "default",
    gt_target_h: Optional[int] = None,
    gt_target_w: Optional[int] = None,
    max_side: int = 518,
    size_multiple: int = 14,
) -> Dict[str, object]:
    pred_cams = collect_pred_cameras_from_solver(solver, image_names)
    selected_stems = [prepared_stem_from_path(Path(p))[1] for p in image_names]

    pred_points, pred_colors = solver.map.collect_sampled_points(
        solver.graph,
        max_points=int(max_pred_points),
        voxel_size=float(voxel_size),
        seed=int(seed),
        point_stride=int(point_stride),
        skip_loop_closure_submaps=True,
    )

    gt_cams, gt_points, gt_colors, gt_meta = load_gt_artifacts(
        scene_dir=Path(scene_dir),
        selected_stems=selected_stems,
        images_dir=images_dir,
        cams_dir=cams_dir,
        depth_dir=depth_dir,
        depth_scale=float(depth_scale),
        depth_min=float(depth_min),
        depth_max=float(depth_max),
        max_gt_points=int(max_gt_points),
        voxel_size=float(voxel_size),
        seed=int(seed),
        target_h=gt_target_h,
        target_w=gt_target_w,
        max_side=int(max_side),
        size_multiple=int(size_multiple),
    )

    pred_points, pred_cams, align_meta = align_prediction_to_gt_pose_sim3(
        pred_points,
        pred_cams,
        gt_cams,
    )

    pred_points, pred_colors = sample_points(pred_points, pred_colors, int(max_pred_points), int(seed) + 17)
    if float(voxel_size) > 0:
        pred_points, pred_colors = voxel_downsample_numpy(pred_points, pred_colors, float(voxel_size))

    meta = {
        "schema": "final_eval_v1",
        "script": "third_party/vggt-slam2.0/run_scene_to_rrd.py",
        "method": method,
        "method_variant": method_variant,
        "scene_dir": str(Path(scene_dir).expanduser().resolve()),
        "pose_convention": "T_c2w",
        "points_coordinate": "gt_pose_coordinate_after_pose_sim3",
        "processing_time": processing_time or {},
        "gt": gt_meta,
        "post_align": align_meta,
        "selected_stems": selected_stems,
        "max_pred_points": int(max_pred_points),
        "max_gt_points": int(max_gt_points),
        "voxel_size": float(voxel_size),
        "point_stride": int(point_stride),
    }

    save_eval_outputs(
        Path(eval_dir),
        pred_cams,
        gt_cams,
        pred_points,
        pred_colors,
        gt_points,
        gt_colors,
        meta=meta,
    )
    write_rrd(
        Path(output_rrd),
        pred_cams,
        pred_points,
        pred_colors,
        gt_cams,
        gt_points,
        gt_colors,
        app_id="vggt_slam2_source_export",
    )
    return meta