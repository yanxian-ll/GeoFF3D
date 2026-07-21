"""LingBot-Map streaming reconstruction wrapper."""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _resolve_keyframe_interval(value, num_frames: int, threshold: int) -> int:
    if value is None or value == 0 or (isinstance(value, str) and value.lower() == "auto"):
        return 1 if num_frames <= threshold else (num_frames + threshold - 1) // threshold
    return int(value)


def _write_pose_log(path: Path, poses: Sequence[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pose in poses:
            T = np.asarray(pose, dtype=np.float32).reshape(4, 4)
            f.write(" ".join(f"{float(v):.9g}" for v in T.reshape(-1)) + "\n")


def _save_point_cloud_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
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


def _depth_to_world_points(depth: np.ndarray, K: np.ndarray, T_c2w: np.ndarray) -> np.ndarray:
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


class LingBotMapWrapper(torch.nn.Module):
    """Thin wrapper around the upstream ``lingbot_map`` streaming model."""

    def __init__(
        self,
        name: str = "lingbot-map",
        torch_hub_force_reload: bool = False,
        checkpoint: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "cuda",
        mode: str = "streaming",
        use_amp: bool = True,
        use_sdpa: bool = False,
        image_size: int = 518,
        patch_size: int = 14,
        enable_3d_rope: bool = True,
        num_scale_frames: int = 8,
        max_frame_num: int = 1024,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        window_size: int = 64,
        overlap_size: Optional[int] = None,
        overlap_keyframes: Optional[int] = None,
        keyframe_interval: object = "auto",
        auto_keyframe_threshold: int = 320,
        conf_threshold: float = 0.0,
        max_points: int = 800000,
        seed: int = 0,
        **_: object,
    ) -> None:
        super().__init__()
        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload
        self.checkpoint = checkpoint or model_path
        self.device_name = device
        self.mode = mode
        self.use_amp = use_amp
        self.use_sdpa = use_sdpa
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.enable_3d_rope = bool(enable_3d_rope)
        self.num_scale_frames = int(num_scale_frames)
        self.max_frame_num = int(max_frame_num)
        self.kv_cache_sliding_window = int(kv_cache_sliding_window)
        self.kv_cache_scale_frames = int(kv_cache_scale_frames)
        self.window_size = int(window_size)
        self.overlap_size = overlap_size
        self.overlap_keyframes = overlap_keyframes
        self.keyframe_interval = keyframe_interval
        self.auto_keyframe_threshold = int(auto_keyframe_threshold)
        self.conf_threshold = float(conf_threshold)
        self.max_points = int(max_points)
        self.seed = int(seed)

        self.model = None

    @property
    def display_name(self) -> str:
        return "LingBot-Map"

    def _device(self) -> torch.device:
        if self.device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_name if torch.cuda.is_available() or self.device_name == "cpu" else "cpu")

    def _load_model(self) -> torch.nn.Module:
        if self.model is not None:
            return self.model
        if self.mode == "windowed":
            from lingbot_map.models.gct_stream_window import GCTStream
        else:
            from lingbot_map.models.gct_stream import GCTStream

        device = self._device()
        model = GCTStream(
            img_size=self.image_size,
            patch_size=self.patch_size,
            enable_3d_rope=self.enable_3d_rope,
            max_frame_num=self.max_frame_num,
            kv_cache_sliding_window=self.kv_cache_sliding_window,
            kv_cache_scale_frames=self.kv_cache_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=self.use_sdpa,
        )
        if self.checkpoint:
            print(f"Loading LingBot-Map checkpoint: {self.checkpoint}")
            ckpt = torch.load(self.checkpoint, map_location=device, weights_only=False)
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        self.model = model.to(device).eval()
        return self.model

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        model = self._load_model()
        device = self._device()
        images = images.to(device)
        dtype = (
            torch.bfloat16
            if self.use_amp and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
            if self.use_amp and torch.cuda.is_available()
            else torch.float32
        )
        keyframe_interval = _resolve_keyframe_interval(
            self.keyframe_interval,
            int(images.shape[0]),
            self.auto_keyframe_threshold,
        )
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available() and self.use_amp):
            if self.mode == "streaming":
                return model.inference_streaming(
                    images,
                    num_scale_frames=self.num_scale_frames,
                    keyframe_interval=keyframe_interval,
                    output_device=torch.device("cpu"),
                )
            return model.inference_windowed(
                images,
                window_size=self.window_size,
                overlap_size=self.overlap_size,
                overlap_keyframes=self.overlap_keyframes,
                num_scale_frames=self.num_scale_frames,
                keyframe_interval=keyframe_interval,
                output_device=torch.device("cpu"),
            )

    def _load_images(self, image_dir: Path) -> Tuple[torch.Tensor, List[Path]]:
        from lingbot_map.utils.load_fn import load_and_preprocess_images

        paths: List[Path] = []
        for ext in IMAGE_EXTS:
            paths.extend(Path(p) for p in glob.glob(str(image_dir / f"*{ext}")))
        paths = sorted(paths)
        if not paths:
            raise RuntimeError(f"No images found under {image_dir}")
        images = load_and_preprocess_images(
            [str(p) for p in paths],
            mode="crop",
            image_size=self.image_size,
            patch_size=self.patch_size,
        )
        return images, paths

    def _convert_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        images: torch.Tensor,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        extrinsic = extrinsic.float().cpu().numpy().squeeze(0)
        intrinsic = intrinsic.float().cpu().numpy().squeeze(0)
        depth = predictions["depth"].float().cpu().numpy().squeeze(0)
        image_np = images.float().cpu().numpy()
        if image_np.ndim == 5:
            image_np = image_np.squeeze(0)

        poses: List[np.ndarray] = []
        depths: List[np.ndarray] = []
        Ks: List[np.ndarray] = []
        rgbs: List[np.ndarray] = []
        for i in range(extrinsic.shape[0]):
            T = np.eye(4, dtype=np.float32)
            T[:3, :] = extrinsic[i].astype(np.float32)
            poses.append(T)
            K = intrinsic[i].astype(np.float32)
            Ks.append(K)
            d = depth[i]
            if d.ndim == 3 and d.shape[-1] == 1:
                d = d[..., 0]
            depths.append(d.astype(np.float32))
            rgb = image_np[i].transpose(1, 2, 0)
            rgbs.append((rgb * 255).clip(0, 255).astype(np.uint8))
        return poses, depths, Ks, rgbs

    def _make_point_cloud(
        self,
        poses: Sequence[np.ndarray],
        depths: Sequence[np.ndarray],
        intrinsics: Sequence[np.ndarray],
        rgbs: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        point_parts: List[np.ndarray] = []
        color_parts: List[np.ndarray] = []
        for T, depth, K, rgb in zip(poses, depths, intrinsics, rgbs):
            valid = np.isfinite(depth) & (depth > 1e-6)
            if not valid.any():
                continue
            pts = _depth_to_world_points(depth, K, T)
            valid = valid & np.isfinite(pts).all(axis=-1)
            if not valid.any():
                continue
            point_parts.append(pts[valid].reshape(-1, 3).astype(np.float32))
            color_parts.append(rgb[valid].reshape(-1, 3).astype(np.uint8))
        if not point_parts:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        points = np.concatenate(point_parts, axis=0)
        colors = np.concatenate(color_parts, axis=0)
        if self.max_points > 0 and points.shape[0] > self.max_points:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(points.shape[0], size=self.max_points, replace=False)
            points = points[idx]
            colors = colors[idx]
        return points, colors

    def run(
        self,
        *,
        scene_dir: str | Path,
        image_dir: str | Path,
        output_dir: str | Path,
        pose_log_path: str | Path,
        point_cloud_path: str | Path,
        timing_path: str | Path,
        device: Optional[str] = None,
        python: Optional[str] = None,
        max_points: Optional[int] = None,
        extra_args: Sequence[str] = (),
    ) -> Dict[str, object]:
        del scene_dir, python, extra_args
        if device is not None:
            self.device_name = str(device)
        if max_points is not None:
            self.max_points = int(max_points)

        image_dir = Path(image_dir).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        pose_log_path = Path(pose_log_path).expanduser().resolve()
        point_cloud_path = Path(point_cloud_path).expanduser().resolve()
        timing_path = Path(timing_path).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        cuda_device = torch.device(self.device_name)
        if cuda_device.type == "cuda":
            torch.cuda.synchronize(cuda_device)
            torch.cuda.reset_peak_memory_stats(cuda_device)

        t0 = time.time()
        images, paths = self._load_images(image_dir)
        predictions = self.forward(images)
        poses, depths, Ks, rgbs = self._convert_predictions(predictions, images)
        points, colors = self._make_point_cloud(poses, depths, Ks, rgbs)
        _write_pose_log(pose_log_path, poses)
        _save_point_cloud_ply(point_cloud_path, points, colors)
        if cuda_device.type == "cuda":
            torch.cuda.synchronize(cuda_device)
            peak_allocated = int(torch.cuda.max_memory_allocated(cuda_device))
            peak_reserved = int(torch.cuda.max_memory_reserved(cuda_device))
        else:
            peak_allocated = 0
            peak_reserved = 0
        total_seconds = float(time.time() - t0)
        timing = {
            "schema": "processing_time_v2",
            "processing_time_seconds": total_seconds,
            "processing_time_ms": float(total_seconds * 1000.0),
            "total_seconds": total_seconds,
            "num_frames": int(len(paths)),
            "mode": self.mode,
            "image_size": int(self.image_size),
            "max_points": int(self.max_points),
            "peak_gpu_memory_allocated_bytes": peak_allocated,
            "peak_gpu_memory_allocated_mib": float(peak_allocated / (1024 ** 2)),
            "peak_gpu_memory_reserved_bytes": peak_reserved,
            "peak_gpu_memory_reserved_mib": float(peak_reserved / (1024 ** 2)),
        }
        timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")

        return {
            "return_code": 0,
            "command": ["python-api", "lingbot_map"],
            "cwd": None,
            "stdout_log_path": None,
            "pose_log_path": pose_log_path,
            "point_cloud_path": point_cloud_path,
            "timing_path": timing_path,
            "staged_outputs": {
                "pose_source": str(pose_log_path),
                "point_cloud_path": str(point_cloud_path),
            },
        }

    def run_scene(self, **kwargs: object) -> Dict[str, object]:
        return self.run(**kwargs)


__all__ = ["LingBotMapWrapper"]
