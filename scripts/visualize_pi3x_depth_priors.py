#!/usr/bin/env python3
"""Visualize Pi3X depth/ray priors and predicted confidence for one view.

Example:
    python scripts/visualize_pi3x_depth_priors.py \
        --rgb /opt/data/private/dataset/data/UAVFF3D-Syn-S/043c8723b7a0bee88c93cdbe/images/000002.png \
        --depth_exr /opt/data/private/dataset/data/UAVFF3D-Syn-S/043c8723b7a0bee88c93cdbe/depth/000002.exr \
        --camera_txt /opt/data/private/dataset/data/UAVFF3D-Syn-S/043c8723b7a0bee88c93cdbe/cams/000002.txt \
        --out_dir outputs/pi3x_prior_viz \
        --checkpoint checkpoints/pi3x
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from mapanything.models.external.pi3.models.pi3x import Pi3X


def parse_camera_txt(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int], float]:
    """Parse the camera txt format shown in the user prompt.

    Returns:
        w2c: [4, 4] OpenCV world-to-camera matrix.
        K: [3, 3] pixel intrinsics.
        image_hw: (height, width).
        hfov: horizontal field of view in degrees.
    """
    text = Path(path).read_text().splitlines()
    clean = [line.strip() for line in text if line.strip()]

    def find_line(prefix: str) -> int:
        for idx, line in enumerate(clean):
            if line.lower().startswith(prefix):
                return idx
        raise ValueError(f"Could not find section {prefix!r} in {path}")

    ext_idx = find_line("extrinsic")
    int_idx = find_line("intrinsic")
    hw_idx = find_line("h w hfov")

    w2c = np.array(
        [[float(x) for x in clean[ext_idx + 1 + r].split()] for r in range(4)],
        dtype=np.float32,
    )
    K = np.array(
        [[float(x) for x in clean[int_idx + 1 + r].split()] for r in range(3)],
        dtype=np.float32,
    )
    h, w, hfov = [float(x) for x in clean[hw_idx + 1].split()[:3]]
    return w2c, K, (int(round(h)), int(round(w))), float(hfov)


def read_rgb(path: str | Path) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(image)


def read_depth_exr(path: str | Path, channel: int) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to read EXR depth: {path}")
    if depth.ndim == 3:
        if channel < 0 or channel >= depth.shape[2]:
            raise ValueError(
                f"--depth_channel must be in [0, {depth.shape[2] - 1}], got {channel}"
            )
        depth = depth[..., channel]
    depth = depth.astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 0.0] = 0.0
    return depth


def target_size_from_pixel_limit(
    width: int,
    height: int,
    pixel_limit: int,
    patch_size: int,
) -> tuple[int, int]:
    if pixel_limit <= 0:
        target_w = (width // patch_size) * patch_size
        target_h = (height // patch_size) * patch_size
        return max(patch_size, target_w), max(patch_size, target_h)

    scale = math.sqrt(float(pixel_limit) / float(width * height))
    scale = min(scale, 1.0)
    target_w = width * scale
    target_h = height * scale
    k = max(1, round(target_w / patch_size))
    m = max(1, round(target_h / patch_size))
    while (k * patch_size) * (m * patch_size) > pixel_limit:
        if k / max(m, 1) > target_w / max(target_h, 1e-6):
            k = max(1, k - 1)
        else:
            m = max(1, m - 1)
    return k * patch_size, m * patch_size


def parse_size(size: str | None) -> tuple[int, int] | None:
    if size is None:
        return None
    if "x" not in size.lower():
        raise ValueError("--target_size must be formatted as WIDTHxHEIGHT")
    w_str, h_str = size.lower().split("x", 1)
    width = int(w_str)
    height = int(h_str)
    if width <= 0 or height <= 0:
        raise ValueError("--target_size values must be positive")
    return width, height


def resize_inputs(
    rgb: np.ndarray,
    depth_z: np.ndarray,
    K: np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src_h, src_w = rgb.shape[:2]
    target_w, target_h = target_size
    if depth_z.shape[:2] != (src_h, src_w):
        raise ValueError(
            "RGB and depth resolution must match before resizing, got "
            f"rgb={(src_h, src_w)} depth={depth_z.shape[:2]}"
        )

    if (src_w, src_h) == (target_w, target_h):
        return rgb, depth_z, K.copy()

    rgb_resized = np.asarray(
        Image.fromarray(rgb).resize((target_w, target_h), Image.Resampling.LANCZOS)
    )
    depth_resized = cv2.resize(
        depth_z,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)

    K_resized = K.copy().astype(np.float32)
    sx = target_w / float(src_w)
    sy = target_h / float(src_h)
    K_resized[0, 0] *= sx
    K_resized[0, 2] *= sx
    K_resized[1, 1] *= sy
    K_resized[1, 2] *= sy
    return rgb_resized, depth_resized, K_resized


def scale_intrinsics_between_sizes(
    K: np.ndarray,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> np.ndarray:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    if (src_h, src_w) == (dst_h, dst_w):
        return K.copy()
    scaled = K.copy().astype(np.float32)
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def compute_rays(K: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (u + 0.5 - K[0, 2]) / K[0, 0]
    y = (v + 0.5 - K[1, 2]) / K[1, 1]
    rays_unnorm = np.stack([x, y, np.ones_like(x)], axis=-1)
    ray_norm = np.linalg.norm(rays_unnorm, axis=-1, keepdims=True).clip(min=1e-8)
    rays_unit = rays_unnorm / ray_norm
    return rays_unit.astype(np.float32), ray_norm[..., 0].astype(np.float32)


def z_depth_to_along_ray(depth_z: np.ndarray, ray_norm: np.ndarray) -> np.ndarray:
    along = depth_z.astype(np.float32) * ray_norm.astype(np.float32)
    along[(depth_z <= 0) | (~np.isfinite(along))] = 0.0
    return along


def robust_range(values: np.ndarray, valid: np.ndarray, q_low: float = 2.0, q_high: float = 98.0) -> tuple[float, float]:
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
) -> tuple[np.ndarray, tuple[float, float]]:
    if valid is None:
        valid = np.isfinite(values)
    valid = valid & np.isfinite(values)
    if vmin is None or vmax is None:
        vmin, vmax = robust_range(values, valid)
    norm = np.clip((values.astype(np.float32) - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb, (float(vmin), float(vmax))


def ray_direction_to_rgb(rays_unit: np.ndarray) -> np.ndarray:
    rgb = np.clip((rays_unit + 1.0) * 0.5, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_scalar_with_colorbar(
    path: Path,
    values: np.ndarray,
    valid: np.ndarray,
    title: str,
    cmap_name: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[float, float]:
    _, (vmin, vmax) = colorize_scalar(values, valid, cmap_name=cmap_name, vmin=vmin, vmax=vmax)
    masked = np.ma.masked_where(~valid, values)
    cmap = matplotlib.colormaps.get_cmap(cmap_name).copy()
    cmap.set_bad("black")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return float(vmin), float(vmax)


def save_overview(
    path: Path,
    rgb: np.ndarray,
    depth_z: np.ndarray,
    rays_rgb: np.ndarray,
    depth_along_ray: np.ndarray,
    conf_map: np.ndarray,
    depth_range: tuple[float, float],
    ray_depth_range: tuple[float, float],
    conf_range: tuple[float, float],
) -> None:
    valid_depth = depth_z > 0
    depth_vis, _ = colorize_scalar(depth_z, valid_depth, vmin=depth_range[0], vmax=depth_range[1])
    along_vis, _ = colorize_scalar(
        depth_along_ray,
        valid_depth,
        vmin=ray_depth_range[0],
        vmax=ray_depth_range[1],
    )
    conf_vis, _ = colorize_scalar(
        conf_map,
        np.isfinite(conf_map),
        cmap_name="viridis",
        vmin=conf_range[0],
        vmax=conf_range[1],
    )

    panels = [
        ("RGB", rgb, None, None, None),
        ("z-depth", depth_vis, depth_z, valid_depth, depth_range),
        ("ray direction", rays_rgb, None, None, None),
        ("GT depth along ray", along_vis, depth_along_ray, valid_depth, ray_depth_range),
        ("Pi3X confidence", conf_vis, conf_map, np.isfinite(conf_map), conf_range),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), dpi=160)
    for ax, (title, image, scalar, valid, value_range) in zip(axes, panels):
        if scalar is None:
            ax.imshow(image)
        else:
            masked = np.ma.masked_where(~valid, scalar)
            cmap_name = "viridis" if "confidence" in title.lower() else "turbo"
            cmap = matplotlib.colormaps.get_cmap(cmap_name).copy()
            cmap.set_bad("black")
            im = ax.imshow(masked, cmap=cmap, vmin=value_range[0], vmax=value_range[1])
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(path)
    plt.close(fig)


def confidence_transform(conf: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return conf
    if transform == "sigmoid":
        return 1.0 / (1.0 + np.exp(-conf))
    if transform == "exp":
        return np.exp(np.clip(conf, -20.0, 20.0))
    if transform == "expp1":
        return 1.0 + np.exp(np.clip(conf, -20.0, 20.0))
    raise ValueError(f"Unknown confidence transform: {transform}")


@torch.inference_mode()
def run_pi3x_confidence(
    rgb: np.ndarray,
    depth_z: np.ndarray,
    K: np.ndarray,
    checkpoint: str,
    device: str,
    conf_transform_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = Pi3X.from_pretrained(checkpoint).to(device).eval()
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    image = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    image = image.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    depth = torch.from_numpy(depth_z.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    intrinsics = torch.from_numpy(K.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    amp_dtype = torch.bfloat16
    if device.startswith("cuda") and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    with torch.autocast("cuda", dtype=amp_dtype, enabled=device.startswith("cuda")):
        pred = model(
            imgs=image,
            depths=depth,
            intrinsics=intrinsics,
            with_prior=True,
            overall_prob=1.0,
            ray_dirs_prob=1.0,
            depth_prob=1.0,
            cam_prob=0.0,
        )

    raw_conf = pred["conf"][0, 0, ..., 0].detach().float().cpu().numpy()
    return raw_conf, confidence_transform(raw_conf, conf_transform_name)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one RGB + z-depth EXR + camera txt sample for Pi3X: z-depth, "
            "ray direction, GT depth-along-ray, and predicted confidence."
        )
    )
    parser.add_argument("--rgb", required=True, help="Path to RGB image.")
    parser.add_argument("--depth_exr", required=True, help="Path to z-depth EXR.")
    parser.add_argument("--camera_txt", required=True, help="Path to camera txt.")
    parser.add_argument("--out_dir", default="outputs/pi3x_depth_prior_viz", help="Output directory.")
    parser.add_argument("--checkpoint", default="yyfz233/Pi3X", help="Pi3X checkpoint or HF repo id.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--depth_channel", type=int, default=0, help="EXR channel index when depth EXR is multi-channel.")
    parser.add_argument(
        "--pixel_limit",
        type=int,
        default=255000,
        help="Resize to at most this many pixels, aligned to patch size. Use 0 to keep nearest lower 14-multiple.",
    )
    parser.add_argument(
        "--target_size",
        default=None,
        help="Optional explicit model/visualization size formatted as WIDTHxHEIGHT. Values are rounded down to 14-multiple.",
    )
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0, help="Seed for Pi3X prior-path sampling.")
    parser.add_argument(
        "--conf_transform",
        choices=("none", "sigmoid", "exp", "expp1"),
        default="none",
        help="Transform applied only for confidence visualization; raw logits are saved separately.",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w2c, K, cam_hw, hfov = parse_camera_txt(args.camera_txt)
    rgb = read_rgb(args.rgb)
    depth_z = read_depth_exr(args.depth_exr, args.depth_channel)

    if tuple(rgb.shape[:2]) != tuple(cam_hw):
        print(
            "Warning: RGB resolution does not match camera txt h/w: "
            f"rgb={rgb.shape[:2]} camera={cam_hw}. Scaling intrinsics to RGB resolution first."
        )
        K = scale_intrinsics_between_sizes(K, cam_hw, tuple(rgb.shape[:2]))
    if tuple(depth_z.shape[:2]) != tuple(rgb.shape[:2]):
        raise ValueError(
            f"RGB and depth resolution must match, got rgb={rgb.shape[:2]} depth={depth_z.shape[:2]}"
        )

    explicit_size = parse_size(args.target_size)
    if explicit_size is None:
        target_w, target_h = target_size_from_pixel_limit(
            rgb.shape[1],
            rgb.shape[0],
            args.pixel_limit,
            args.patch_size,
        )
    else:
        target_w = max(args.patch_size, (explicit_size[0] // args.patch_size) * args.patch_size)
        target_h = max(args.patch_size, (explicit_size[1] // args.patch_size) * args.patch_size)

    rgb, depth_z, K = resize_inputs(rgb, depth_z, K, (target_w, target_h))
    height, width = depth_z.shape
    rays_unit, ray_norm = compute_rays(K, height, width)
    depth_along_ray = z_depth_to_along_ray(depth_z, ray_norm)

    raw_conf, conf_map = run_pi3x_confidence(
        rgb=rgb,
        depth_z=depth_z,
        K=K,
        checkpoint=args.checkpoint,
        device=args.device,
        conf_transform_name=args.conf_transform,
        seed=args.seed,
    )

    valid_depth = depth_z > 0
    depth_range = save_scalar_with_colorbar(
        out_dir / "z_depth.png",
        depth_z,
        valid_depth,
        "z-depth",
    )
    ray_depth_range = save_scalar_with_colorbar(
        out_dir / "gt_depth_along_ray.png",
        depth_along_ray,
        valid_depth,
        "GT depth along ray",
    )
    conf_range = save_scalar_with_colorbar(
        out_dir / "pi3x_confidence.png",
        conf_map,
        np.isfinite(conf_map),
        f"Pi3X confidence ({args.conf_transform})",
        cmap_name="viridis",
    )

    rays_rgb = ray_direction_to_rgb(rays_unit)
    save_image(out_dir / "rgb_resized.png", rgb)
    save_image(out_dir / "ray_direction_rgb.png", rays_rgb)
    save_overview(
        out_dir / "overview.png",
        rgb,
        depth_z,
        rays_rgb,
        depth_along_ray,
        conf_map,
        depth_range,
        ray_depth_range,
        conf_range,
    )

    np.save(out_dir / "z_depth.npy", depth_z)
    np.save(out_dir / "ray_directions_unit.npy", rays_unit)
    np.save(out_dir / "gt_depth_along_ray.npy", depth_along_ray)
    np.save(out_dir / "pi3x_confidence_raw.npy", raw_conf)
    np.save(out_dir / "pi3x_confidence_visualized.npy", conf_map)
    np.save(out_dir / "intrinsics_resized.npy", K)
    np.save(out_dir / "w2c_opencv.npy", w2c)

    metadata = [
        f"rgb: {args.rgb}",
        f"depth_exr: {args.depth_exr}",
        f"camera_txt: {args.camera_txt}",
        f"checkpoint: {args.checkpoint}",
        f"device: {args.device}",
        f"input_camera_hw: {cam_hw[0]} {cam_hw[1]}",
        f"output_hw: {height} {width}",
        f"hfov: {hfov}",
        f"conf_transform: {args.conf_transform}",
        f"seed: {args.seed}",
        f"z_depth_range: {depth_range[0]} {depth_range[1]}",
        f"depth_along_ray_range: {ray_depth_range[0]} {ray_depth_range[1]}",
        f"confidence_range: {conf_range[0]} {conf_range[1]}",
    ]
    (out_dir / "metadata.txt").write_text("\n".join(metadata) + "\n")

    print(f"Saved visualization to {out_dir}")
    print(f"Overview: {out_dir / 'overview.png'}")


if __name__ == "__main__":
    main()
