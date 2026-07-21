#!/usr/bin/env python3
"""Run STream3R inference and write MapAnything streaming artifacts."""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pose_log_path", required=True)
    parser.add_argument("--point_cloud_path", required=True)
    parser.add_argument("--timing_path", required=True)
    parser.add_argument("--checkpoint", default="yslan/STream3R")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="causal", choices=["causal", "window", "full"])
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max_points", type=int, default=800000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def image_paths(image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(Path(p) for p in glob.glob(str(image_dir / f"*{ext}")))
        paths.extend(Path(p) for p in glob.glob(str(image_dir / f"*{ext.upper()}")))
    return sorted(set(paths))


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


def sample_points(points: np.ndarray, colors: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    return points, colors


def prepare_points(predictions: dict, images: torch.Tensor, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pts = predictions.get("world_points_from_depth", predictions.get("world_points"))
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().float().cpu().numpy()
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 5:
        pts = pts.squeeze(0)

    image_np = images.detach().float().cpu().numpy()
    if image_np.ndim == 5:
        image_np = image_np.squeeze(0)
    colors = np.transpose(image_np, (0, 2, 3, 1))
    colors = (colors * 255.0).clip(0, 255).astype(np.uint8)

    if colors.shape[:3] != pts.shape[:3]:
        colors = np.full(pts.shape[:3] + (3,), 220, dtype=np.uint8)

    conf = predictions.get("world_points_conf", predictions.get("depth_conf"))
    if isinstance(conf, torch.Tensor):
        conf = conf.detach().float().cpu().numpy()
    if conf is not None:
        conf = np.asarray(conf)
        if conf.ndim == 4 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        if conf.ndim == 4:
            conf = conf.squeeze(0)
        valid = np.isfinite(pts).all(axis=-1) & np.isfinite(conf)
    else:
        valid = np.isfinite(pts).all(axis=-1)

    points = pts[valid].reshape(-1, 3).astype(np.float32)
    cols = colors[valid].reshape(-1, 3).astype(np.uint8)
    return sample_points(points, cols, max_points=max_points, seed=seed)


def poses_from_extrinsic(extrinsic: object) -> list[np.ndarray]:
    if isinstance(extrinsic, torch.Tensor):
        extrinsic = extrinsic.detach().float().cpu().numpy()
    ext = np.asarray(extrinsic, dtype=np.float32)
    if ext.ndim == 4:
        ext = ext.squeeze(0)
    poses: list[np.ndarray] = []
    for i in range(ext.shape[0]):
        T = np.eye(4, dtype=np.float32)
        T[:3, :4] = ext[i, :3, :4]
        poses.append(T)
    return poses


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    sys.path.insert(0, str(repo_root))

    from stream3r.models.components.utils.geometry import unproject_depth_map_to_point_map
    from stream3r.models.components.utils.load_fn import load_and_preprocess_images
    from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri
    from stream3r.models.stream3r import STream3R
    from stream3r.stream_session import StreamSession

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    paths = image_paths(Path(args.image_dir))
    if not paths:
        raise RuntimeError(f"No images found under {args.image_dir}")

    t0 = time.time()
    model = STream3R.from_pretrained(args.checkpoint).to(device).eval()
    images = load_and_preprocess_images([str(p) for p in paths]).to(device)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        if args.streaming:
            mode = "causal" if args.mode == "full" else args.mode
            session = StreamSession(model, mode=mode)
            predictions = {}
            for i in range(images.shape[0]):
                predictions = session.forward_stream(images[i : i + 1])
            session.clear()
        else:
            predictions = model(images, mode=args.mode)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"], extrinsic, intrinsic
    )

    poses = poses_from_extrinsic(extrinsic)
    points, colors = prepare_points(predictions, images, max_points=args.max_points, seed=args.seed)
    write_pose_log(Path(args.pose_log_path), poses)
    save_ply(Path(args.point_cloud_path), points, colors)
    Path(args.timing_path).write_text(
        json.dumps(
            {
                "total_seconds": float(time.time() - t0),
                "num_frames": int(len(paths)),
                "mode": args.mode,
                "streaming": bool(args.streaming),
                "checkpoint": args.checkpoint,
                "max_points": int(args.max_points),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
