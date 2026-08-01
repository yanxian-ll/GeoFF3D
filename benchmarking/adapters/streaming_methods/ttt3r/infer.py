#!/usr/bin/env python3
"""Run TTT3R inference and write MapAnything streaming artifacts."""

from __future__ import annotations

import argparse
from copy import deepcopy
import glob
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image, ImageOps


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pose_log_path", required=True)
    parser.add_argument("--point_cloud_path", required=True)
    parser.add_argument("--timing_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--model_update_type", default="ttt3r")
    parser.add_argument("--reset_interval", type=int, default=1000000)
    parser.add_argument("--max_points", type=int, default=800000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def image_paths(image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(Path(p) for p in glob.glob(str(image_dir / f"*{ext}")))
        paths.extend(Path(p) for p in glob.glob(str(image_dir / f"*{ext.upper()}")))
    return sorted(set(paths))


def _square_image_tensor(path: Path, size: int) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    original_size = img.size
    w, h = img.size
    scale = float(size) / float(min(w, h))
    resized = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    rw, rh = resized.size
    left = max((rw - size) // 2, 0)
    top = max((rh - size) // 2, 0)
    cropped = resized.crop((left, top, left + size, top + size))
    arr = np.asarray(cropped, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0), original_size, cropped.size


def prepare_square_input(paths: Sequence[Path], size: int, reset_interval: int) -> list[dict]:
    print(f">> Loading a list of {len(paths)} square TTT3R images")
    views: list[dict] = []
    reset_interval = max(int(reset_interval), 1)
    for i, path in enumerate(paths):
        img_tensor, original_size, cropped_size = _square_image_tensor(path, size)
        h, w = img_tensor.shape[-2:]
        print(
            f" - adding {path} with resolution "
            f"{original_size[0]}x{original_size[1]} --> {cropped_size[0]}x{cropped_size[1]}"
        )
        view = {
            "img": img_tensor,
            "ray_map": torch.full((1, 6, h, w), torch.nan),
            "true_shape": torch.from_numpy(np.int32([[h, w]])),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0),
        }
        views.append(view)
        if (i + 1) % reset_interval == 0:
            overlap_view = deepcopy(view)
            overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
            views.append(overlap_view)
    print(f" (Found {len(paths)} images)")
    return views


def write_pose_log(path: Path, poses: Sequence[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pose in poses:
            T = np.asarray(pose, dtype=np.float32).reshape(4, 4)
            f.write(" ".join(f"{float(v):.9g}" for v in T.reshape(-1)) + "\n")


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if colors.shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def depth_to_world(depth: np.ndarray, K: np.ndarray, T_c2w: np.ndarray) -> np.ndarray:
    h, w = depth.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u.astype(np.float64) - cx) * z / max(fx, 1e-8)
    y = (v.astype(np.float64) - cy) * z / max(fy, 1e-8)
    pts_cam = np.stack([x, y, z], axis=-1)
    R = np.asarray(T_c2w, dtype=np.float64)[:3, :3]
    t = np.asarray(T_c2w, dtype=np.float64)[:3, 3]
    return (np.einsum("ij,hwj->hwi", R, pts_cam) + t[None, None, :]).astype(np.float32)


def sample_points(points: np.ndarray, colors: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    return points, colors


def write_ttt3r_outputs(outputs: dict, outdir: Path, revisit: int = 1, use_pose: bool = True) -> None:
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod

    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)
    outputs["pred"] = [pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    pr_poses = [pose_encoding_to_camera(pred["camera_pose"].clone()).cpu() for pred in outputs["pred"]]
    if reset_mask.any():
        pr_poses_cat = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses_cat.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses_cat, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = list(torch.einsum("bij,bjk->bik", shifted_bases, pr_poses_cat).unsqueeze(1).unbind(0))

    if use_pose:
        pts3ds_other = [geotrf(pose, pself.unsqueeze(0)) for pose, pself in zip(pr_poses, pts3ds_self)]

    b, h, w, _ = pts3ds_self.shape
    pp = torch.tensor([w // 2, h // 2], device=pts3ds_self.device).float().repeat(b, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = torch.cat(
        [0.5 * (output["img"].permute(0, 2, 3, 1).cpu() + 1.0) for output in outputs["views"]]
    )
    cam2world = torch.cat(pr_poses)
    intrinsics = torch.eye(3).unsqueeze(0).repeat(cam2world.shape[0], 1, 1)
    intrinsics[:, 0, 0] = focal.detach().cpu()
    intrinsics[:, 1, 1] = focal.detach().cpu()
    intrinsics[:, 0, 2] = pp[:, 0].cpu()
    intrinsics[:, 1, 2] = pp[:, 1].cpu()

    depth_dir = outdir / "depth"
    conf_dir = outdir / "conf"
    color_dir = outdir / "color"
    camera_dir = outdir / "camera"
    for folder in (depth_dir, conf_dir, color_dir, camera_dir):
        if folder.exists():
            for child in folder.iterdir():
                child.unlink()
        folder.mkdir(parents=True, exist_ok=True)

    depths = pts3ds_self[..., 2]
    conf = torch.cat(conf_self)
    _ = pts3ds_other
    for frame_id in range(len(pts3ds_self)):
        np.save(depth_dir / f"{frame_id:06d}.npy", depths[frame_id].cpu().numpy())
        np.save(conf_dir / f"{frame_id:06d}.npy", conf[frame_id].cpu().numpy())
        iio.imwrite(color_dir / f"{frame_id:06d}.png", (colors[frame_id].cpu().numpy() * 255).astype(np.uint8))
        np.savez(
            camera_dir / f"{frame_id:06d}.npz",
            pose=cam2world[frame_id].cpu().numpy(),
            intrinsics=intrinsics[frame_id].cpu().numpy(),
        )


def convert_outputs(raw_dir: Path, pose_log_path: Path, point_cloud_path: Path, max_points: int, seed: int) -> tuple[int, int]:
    camera_dir = raw_dir / "camera"
    depth_dir = raw_dir / "depth"
    color_dir = raw_dir / "color"
    poses: list[np.ndarray] = []
    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []

    for cam_path in sorted(camera_dir.glob("*.npz")):
        stem = cam_path.stem
        with np.load(cam_path) as data:
            T_c2w = np.asarray(data["pose"], dtype=np.float32).reshape(4, 4)
            K = np.asarray(data["intrinsics"], dtype=np.float32).reshape(3, 3)
        poses.append(T_c2w)
        depth_path = depth_dir / f"{stem}.npy"
        color_path = color_dir / f"{stem}.png"
        if not depth_path.exists() or not color_path.exists():
            continue
        depth = np.asarray(np.load(depth_path), dtype=np.float32)
        color = np.asarray(iio.imread(color_path), dtype=np.uint8)
        if color.ndim == 2:
            color = np.repeat(color[..., None], 3, axis=-1)
        if color.shape[:2] != depth.shape[:2]:
            color = np.full(depth.shape[:2] + (3,), 220, dtype=np.uint8)
        pts = depth_to_world(depth, K, T_c2w)
        valid = np.isfinite(depth) & (depth > 1e-6) & np.isfinite(pts).all(axis=-1)
        if valid.any():
            point_parts.append(pts[valid].reshape(-1, 3).astype(np.float32))
            color_parts.append(color[valid].reshape(-1, 3).astype(np.uint8))

    write_pose_log(pose_log_path, poses)
    if point_parts:
        points = np.concatenate(point_parts, axis=0)
        colors = np.concatenate(color_parts, axis=0)
        points, colors = sample_points(points, colors, max_points=max_points, seed=seed)
    else:
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.uint8)
    save_ply(point_cloud_path, points, colors)
    return len(poses), int(points.shape[0])


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    sys.path.insert(0, str(repo_root))

    from add_ckpt_path import add_path_to_dust3r
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    paths = image_paths(Path(args.image_dir))
    if not paths:
        raise RuntimeError(f"No images found under {args.image_dir}")

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "ttt3r_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    add_path_to_dust3r(args.checkpoint)
    views = prepare_square_input(paths, size=int(args.image_size), reset_interval=int(args.reset_interval))
    model = ARCroco3DStereo.from_pretrained(args.checkpoint).to(device)
    model.config.model_update_type = args.model_update_type
    model.eval()
    with torch.no_grad():
        outputs, _state_args = inference_recurrent_lighter(views, model, device)
    write_ttt3r_outputs(outputs, raw_dir, revisit=1, use_pose=True)
    num_poses, num_points = convert_outputs(
        raw_dir,
        Path(args.pose_log_path),
        Path(args.point_cloud_path),
        max_points=args.max_points,
        seed=args.seed,
    )
    Path(args.timing_path).write_text(
        json.dumps(
            {
                "total_seconds": float(time.time() - t0),
                "num_frames": int(len(paths)),
                "num_poses": int(num_poses),
                "num_points": int(num_points),
                "checkpoint": str(args.checkpoint),
                "image_size": int(args.image_size),
                "model_update_type": str(args.model_update_type),
                "reset_interval": int(args.reset_interval),
                "max_points": int(args.max_points),
                "square_input": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
