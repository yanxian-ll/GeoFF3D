#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize geoff3d predictions for a small scene chunk.

Example:
    python scripts/visualize_geoff3d.py \
        --scene_dir /opt/data/private/dataset/data/usegeo/dataset1 \
        --checkpoint experiments/dom/uav_training/pi3x_zup_translation_yaw_aug_8v_6d_16ipg_2g/checkpoint-last.pth \
        --out_dir outputs/geoff3d_viz/dataset1 \
        --num_views 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from spatial_rrd.model_runner import (
    apply_runtime_prior_policy,
    build_prior_overrides,
    checkpoint_hydra_overrides,
    collect_pred_outputs,
    filter_views_for_prior_policy,
    init_model_from_hydra,
    load_checkpoint,
    resolve_prior_policy,
)
from spatial_rrd.rrd_writer import save_point_cloud_ply
from spatial_rrd.scene_io import build_views_from_scene, load_chunk_views_from_scene


def resolve_device(device_arg: str) -> torch.device:
    value = str(device_arg).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    return torch.device(device_arg)


def robust_range(values: np.ndarray, valid: np.ndarray, q_low: float = 2.0, q_high: float = 98.0) -> tuple[float, float]:
    valid = valid & np.isfinite(values)
    if not bool(valid.any()):
        return 0.0, 1.0
    selected = values[valid]
    lo, hi = np.percentile(selected, [q_low, q_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(selected))
        hi = float(np.nanmax(selected))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def colorize_scalar(
    values: np.ndarray,
    valid: np.ndarray | None = None,
    cmap_name: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(values)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if vmin is None or vmax is None:
        vmin, vmax = robust_range(values, valid)
    norm = np.clip((values - float(vmin)) / max(float(vmax) - float(vmin), 1e-8), 0.0, 1.0)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def tensor_to_numpy(x: torch.Tensor | None) -> np.ndarray | None:
    if x is None:
        return None
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    return x.detach().cpu().numpy()


def pred_depth_along_ray(pred: dict) -> np.ndarray | None:
    depth = pred.get("depth_along_ray")
    if depth is not None:
        arr = tensor_to_numpy(depth)
        if arr is not None:
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            return np.asarray(arr, dtype=np.float32)

    pts_cam = pred.get("pts3d_cam")
    if pts_cam is None:
        return None
    arr = tensor_to_numpy(pts_cam)
    if arr is None:
        return None
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return np.linalg.norm(arr.astype(np.float32), axis=-1)
    return None


def pred_confidence(pred: dict) -> np.ndarray | None:
    conf = pred.get("conf")
    if conf is None:
        return None
    arr = tensor_to_numpy(conf)
    if arr is None:
        return None
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3:
        arr = np.nanmean(arr, axis=-1)
    return np.asarray(arr, dtype=np.float32)


def resize_rgb_like(rgb: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if rgb.shape[:2] == (h, w):
        return rgb
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)


def save_view_overview(
    out_path: Path,
    rgb: np.ndarray,
    depth: np.ndarray | None,
    conf: np.ndarray | None,
    title: str,
) -> None:
    panels: list[tuple[str, np.ndarray]] = []
    if depth is not None:
        rgb = resize_rgb_like(rgb, depth.shape[:2])
    elif conf is not None:
        rgb = resize_rgb_like(rgb, conf.shape[:2])
    panels.append(("RGB", rgb))

    if depth is not None:
        valid = np.isfinite(depth) & (depth > 0)
        panels.append(("pred depth along ray", colorize_scalar(depth, valid, "turbo")))
    if conf is not None:
        panels.append(("confidence", colorize_scalar(conf, np.isfinite(conf), "viridis")))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.0), dpi=160)
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_contact_sheet(paths: Sequence[Path], out_path: Path, cols: int = 4) -> None:
    paths = [Path(p) for p in paths if Path(p).exists()]
    if not paths:
        return
    images = [Image.open(p).convert("RGB") for p in paths]
    thumb_w = min(520, max(img.width for img in images))
    thumbs = []
    for img in images:
        scale = thumb_w / float(img.width)
        thumb_h = max(1, int(round(img.height * scale)))
        thumbs.append(img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS))
    rows = int(math.ceil(len(thumbs) / max(1, cols)))
    cell_h = max(img.height for img in thumbs)
    canvas = Image.new("RGB", (thumb_w * cols, cell_h * rows), (255, 255, 255))
    for i, img in enumerate(thumbs):
        x = (i % cols) * thumb_w
        y = (i // cols) * cell_h
        canvas.paste(img, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def sample_points(points: np.ndarray, colors: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(points.shape[0], size=int(max_points), replace=False)
    return points[idx], colors[idx]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize geoff3d wrapper predictions on a small scene chunk."
    )
    parser.add_argument("--scene_dir", required=True, help="Scene directory with images/cams/depth subfolders.")
    parser.add_argument("--out_dir", default="outputs/geoff3d_viz", help="Output directory.")
    parser.add_argument("--checkpoint", default=None, help="Fine-tuned checkpoint path or wrapper pretrained path.")
    parser.add_argument("--model", default="geoff3d", choices=("geoff3d", "geoff3d_camera_token"))
    parser.add_argument("--machine", default="aws", help="Hydra machine config.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hydra_override", action="append", default=[], help="Extra Hydra override, can be repeated.")

    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--cams_dir", default="cams")
    parser.add_argument("--depth_dir", default="depth")
    parser.add_argument("--frame_glob", default="*")
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_side", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--depth_min", type=float, default=1e-6)
    parser.add_argument("--depth_max", type=float, default=1e6)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--norm_type", default="identity")

    parser.add_argument("--translation_prior", choices=("auto", "input", "none"), default="input")
    parser.add_argument("--rotation_prior", choices=("auto", "input", "none"), default="input")
    parser.add_argument("--ray_prior", choices=("auto", "input", "pred", "none"), default="input")
    parser.add_argument("--depth_prior", choices=("auto", "input", "pred", "none"), default="input")
    parser.add_argument("--model_family", default="ours", choices=("auto", "ours", "input_prior", "no_prior"))

    parser.add_argument("--conf_quantile", type=float, default=0.0, help="Filter PLY points below this confidence quantile.")
    parser.add_argument("--max_ply_points", type=int, default=500000, help="Random cap for merged PLY; <=0 keeps all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overview_cols", type=int, default=2)
    return parser


def main() -> int:
    args = get_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print(f"[INFO] building scene manifest: {args.scene_dir}")
    views_light, meta = build_views_from_scene(
        scene_dir=Path(args.scene_dir),
        images_dir=args.images_dir,
        cams_dir=args.cams_dir,
        depth_dir=args.depth_dir,
        frame_glob=args.frame_glob,
        num_views=args.num_views,
        start=args.start,
        stride=args.stride,
        max_side=args.max_side,
        size_multiple=args.patch_size,
        depth_scale=args.depth_scale,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        device=torch.device("cpu"),
        show_progress=True,
    )
    if not views_light:
        raise RuntimeError("No frames selected.")

    prior_policy = resolve_prior_policy(args, args.model, meta)
    print(
        "[POLICY] "
        f"translation={prior_policy['translation']}, rotation={prior_policy['rotation']}, "
        f"ray={prior_policy['ray']}, depth={prior_policy['depth']}"
    )

    indices = list(range(len(views_light)))
    chunk_views, rgbs = load_chunk_views_from_scene(
        lightweight_views=views_light,
        meta=meta,
        indices=indices,
        prior_policy=prior_policy,
        device=device,
        recenter_anchor=None,
        num_workers=args.num_workers,
        norm_type=args.norm_type,
    )
    chunk_views = filter_views_for_prior_policy(chunk_views, prior_policy)

    checkpoint_overrides, checkpoint_override = checkpoint_hydra_overrides(args.model, args.checkpoint)
    hydra_overrides = list(args.hydra_override) + checkpoint_overrides + build_prior_overrides(args.model, prior_policy)

    print(f"[INFO] loading model={args.model} on {device}")
    model, cfg = init_model_from_hydra(
        model_name=args.model,
        machine=args.machine,
        hydra_overrides=hydra_overrides,
        device=device,
    )
    load_checkpoint(model, checkpoint_override)
    apply_runtime_prior_policy(model, prior_policy)
    model.eval()

    print(f"[INFO] running forward for {len(chunk_views)} views")
    with torch.inference_mode():
        preds = model(chunk_views)

    stems = [str(v.get("stem", f"view_{i:03d}")) for i, v in enumerate(views_light)]
    points, colors, pred_maps, pred_valid_masks, pred_cams = collect_pred_outputs(
        preds,
        rgbs,
        pred_min_depth=float(args.depth_min),
        conf_quantile=float(args.conf_quantile),
        stems=stems,
    )
    points, colors = sample_points(points, colors, int(args.max_ply_points), int(args.seed))
    ply_path = out_dir / "pred_points.ply"
    save_point_cloud_ply(ply_path, points, colors)

    view_dir = out_dir / "views"
    overview_paths: list[Path] = []
    for i, (stem, pred, rgb) in enumerate(zip(stems, preds, rgbs)):
        depth = pred_depth_along_ray(pred)
        conf = pred_confidence(pred)
        view_path = view_dir / f"{i:03d}_{stem}_overview.png"
        save_view_overview(view_path, rgb, depth, conf, title=f"{i:03d} {stem}")
        overview_paths.append(view_path)

        if depth is not None:
            Image.fromarray(colorize_scalar(depth, np.isfinite(depth) & (depth > 0), "turbo")).save(
                view_dir / f"{i:03d}_{stem}_pred_depth.png"
            )
        if conf is not None:
            Image.fromarray(colorize_scalar(conf, np.isfinite(conf), "viridis")).save(
                view_dir / f"{i:03d}_{stem}_conf.png"
            )

    save_contact_sheet(overview_paths, out_dir / "overview.png", cols=max(1, int(args.overview_cols)))

    metadata = {
        "scene_dir": str(Path(args.scene_dir).expanduser().resolve()),
        "model": args.model,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "num_views": len(stems),
        "stems": stems,
        "target_hw": [int(meta["target_h"]), int(meta["target_w"])],
        "prior_policy": prior_policy,
        "hydra_overrides": hydra_overrides,
        "pred_points_ply": str(ply_path),
        "num_pred_points_saved": int(points.shape[0]),
        "num_pred_cameras": int(len(pred_cams)),
        "config_model_str": str(cfg.model.model_str),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[DONE] saved overview: {out_dir / 'overview.png'}")
    print(f"[DONE] saved point cloud: {ply_path} ({points.shape[0]} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
