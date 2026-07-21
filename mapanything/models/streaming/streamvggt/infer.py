#!/usr/bin/env python3
"""Run StreamVGGT inference and write MapAnything streaming artifacts."""

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
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--hf_repo", default="lch01/StreamVGGT")
    parser.add_argument("--hf_filename", default="checkpoints.pth")
    parser.add_argument("--device", default="cuda")
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


def prepare_points(world_points: torch.Tensor, conf: torch.Tensor, images: torch.Tensor, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pts = world_points.detach().float().cpu().numpy()
    cnf = conf.detach().float().cpu().numpy()
    img = images.detach().float().cpu().numpy()
    if pts.ndim == 5:
        pts = pts.squeeze(0)
    if cnf.ndim == 4 and cnf.shape[-1] == 1:
        cnf = cnf[..., 0]
    if cnf.ndim == 4:
        cnf = cnf.squeeze(0)
    if img.ndim == 5:
        img = img.squeeze(0)
    colors = np.transpose(img, (0, 2, 3, 1))
    colors = (colors * 255.0).clip(0, 255).astype(np.uint8)
    if colors.shape[:3] != pts.shape[:3]:
        colors = np.full(pts.shape[:3] + (3,), 220, dtype=np.uint8)
    valid = np.isfinite(pts).all(axis=-1) & np.isfinite(cnf)
    points = pts[valid].reshape(-1, 3).astype(np.float32)
    cols = colors[valid].reshape(-1, 3).astype(np.uint8)
    return sample_points(points, cols, max_points=max_points, seed=seed)


def resolve_checkpoint(args: argparse.Namespace, repo_root: Path) -> str:
    if args.checkpoint:
        ckpt = Path(args.checkpoint).expanduser()
        if not ckpt.is_absolute():
            ckpt = Path.cwd() / ckpt
        return str(ckpt)
    local = repo_root / "ckpt" / args.hf_filename
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=args.hf_repo, filename=args.hf_filename, revision="main")


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

    from streamvggt.models.streamvggt import StreamVGGT
    from streamvggt.utils.load_fn import load_and_preprocess_images
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    paths = image_paths(Path(args.image_dir))
    if not paths:
        raise RuntimeError(f"No images found under {args.image_dir}")

    t0 = time.time()
    ckpt_path = resolve_checkpoint(args, repo_root)
    model = StreamVGGT()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    del ckpt
    model = model.to(device).eval()

    images = load_and_preprocess_images([str(p) for p in paths]).to(device)
    frames = [{"img": images[i].unsqueeze(0)} for i in range(images.shape[0])]
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype, enabled=device.type == "cuda"):
        output = model.inference(frames)

    world_points = torch.stack([res["pts3d_in_other_view"].squeeze(0) for res in output.ress], dim=0)
    conf = torch.stack([res["conf"].squeeze(0) for res in output.ress], dim=0)
    pose_enc = torch.stack([res["camera_pose"].squeeze(0) for res in output.ress], dim=0)
    extrinsic, _intrinsic = pose_encoding_to_extri_intri(pose_enc.unsqueeze(0), images.shape[-2:])
    extrinsic = extrinsic.squeeze(0)

    poses = poses_from_extrinsic(extrinsic)
    points, colors = prepare_points(world_points, conf, images, max_points=args.max_points, seed=args.seed)
    write_pose_log(Path(args.pose_log_path), poses)
    save_ply(Path(args.point_cloud_path), points, colors)
    Path(args.timing_path).write_text(
        json.dumps(
            {
                "total_seconds": float(time.time() - t0),
                "num_frames": int(len(paths)),
                "checkpoint": str(ckpt_path),
                "max_points": int(args.max_points),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
