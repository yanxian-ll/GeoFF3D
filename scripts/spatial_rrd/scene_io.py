# -*- coding: utf-8 -*-
"""Scene IO: read images, cameras, depth, and build MapAnything-style view dicts."""

from __future__ import annotations

import fnmatch
import math
import os
import re
import struct
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision.transforms as tvf

from mapanything.utils.geometry import get_rays_in_camera_frame
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEPTH_EXTS = {".exr", ".npy", ".png", ".tif", ".tiff"}
CAM_EXTS = {".txt"}


def iter_progress(
    values: Iterable,
    *,
    desc: str,
    total: Optional[int] = None,
    unit: str = "it",
    enabled: bool = True,
):
    if enabled and tqdm is not None:
        return tqdm(
            values,
            desc=desc,
            total=total,
            unit=unit,
            dynamic_ncols=True,
            file=sys.__stderr__,
        )
    return values


# ---------------------------------------------------------------------------
# File collection / camera parsing
# ---------------------------------------------------------------------------
def collect_stem_to_path(folder: Path, exts: Iterable[str]) -> Dict[str, Path]:
    exts = {e.lower() for e in exts}
    if not folder.exists():
        return {}
    return {
        p.stem: p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    }


def sanitize_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[\\/:*?\"<>|\s]+", "_", name)
    return name or "scene"


def contiguous_select(items: Sequence[str], max_count: int) -> List[str]:
    items = list(items)
    if max_count <= 0 or len(items) <= max_count:
        return items
    return items[: int(max_count)]


def _float_tokens(line: str) -> Optional[List[float]]:
    try:
        vals = [float(x) for x in line.replace(",", " ").split()]
        return vals if vals else None
    except ValueError:
        return None


def _find_line(lines: Sequence[str], prefixes: Sequence[str]) -> int:
    prefixes = tuple(p.lower().rstrip(":") for p in prefixes)
    for i, line in enumerate(lines):
        l = line.strip().lower().rstrip(":")
        if any(l.startswith(p) for p in prefixes):
            return i
    return -1


def _read_numeric_rows(
    lines: Sequence[str], start: int, n_rows: int, n_cols: int, path: Path
) -> np.ndarray:
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
    with open(cam_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

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
    for i, ln in enumerate(lines):
        tokens = ln.lower().replace(":", " ").split()
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


# ---------------------------------------------------------------------------
# RGB-D IO and geometry
# ---------------------------------------------------------------------------
def read_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def read_image_size(path: Path) -> Tuple[int, int]:
    """Return (height, width) without decoding the full RGB image when possible."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            width, height = img.size
        return int(height), int(width)
    except Exception:
        img = read_rgb(path)
        return int(img.shape[0]), int(img.shape[1])


def _exr_cstr(data: bytes, pos: int) -> Tuple[str, int]:
    end = data.index(b"\x00", pos)
    return data[pos:end].decode("ascii"), end + 1


def _exr_zip_reconstruct(payload: bytes) -> bytes:
    raw = bytearray(zlib.decompress(payload))
    for i in range(1, len(raw)):
        raw[i] = (raw[i] + raw[i - 1] - 128) & 0xFF

    out = bytearray(len(raw))
    t1 = 0
    t2 = (len(raw) + 1) // 2
    s = 0
    while s < len(raw):
        out[s] = raw[t1]
        s += 1
        t1 += 1
        if s >= len(raw):
            break
        out[s] = raw[t2]
        s += 1
        t2 += 1
    return bytes(out)


def read_exr_depth(path: Path) -> np.ndarray:
    """Read simple scanline EXR depth files without OpenCV/OpenEXR support.

    This intentionally supports the UAV depth-map case: scanline EXR with
    UINT/HALF/FLOAT channels and NONE/ZIPS/ZIP compression. It is not a full
    general-purpose EXR implementation.
    """
    try:
        import OpenEXR

        exr = OpenEXR.File(str(path))
        channels = exr.channels()
        for name in ("Y", "Z", "R", "depth", "Depth"):
            if name in channels:
                return np.asarray(channels[name].pixels, dtype=np.float32)
        first = next(iter(channels.values()))
        return np.asarray(first.pixels, dtype=np.float32)
    except Exception:
        pass

    data = Path(path).read_bytes()
    if len(data) < 16 or data[:4] != b"v/1\x01":
        raise ValueError(f"Not an OpenEXR file: {path}")

    pos = 8
    attrs: Dict[str, Tuple[str, bytes]] = {}
    while True:
        name, pos = _exr_cstr(data, pos)
        if name == "":
            break
        attr_type, pos = _exr_cstr(data, pos)
        size = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        attrs[name] = (attr_type, data[pos : pos + size])
        pos += size

    compression = 0
    if "compression" in attrs:
        compression = int(attrs["compression"][1][0])

    if "dataWindow" not in attrs:
        raise ValueError(f"EXR missing dataWindow: {path}")
    xmin, ymin, xmax, ymax = struct.unpack("<iiii", attrs["dataWindow"][1])
    width = int(xmax - xmin + 1)
    height = int(ymax - ymin + 1)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid EXR dataWindow for {path}: {(xmin, ymin, xmax, ymax)}")

    if "channels" not in attrs:
        raise ValueError(f"EXR missing channels: {path}")
    ch_data = attrs["channels"][1]
    ch_pos = 0
    channels: List[Tuple[str, int, int]] = []
    while ch_pos < len(ch_data):
        name, ch_pos = _exr_cstr(ch_data, ch_pos)
        if name == "":
            break
        pixel_type = struct.unpack_from("<i", ch_data, ch_pos)[0]
        ch_pos += 4
        ch_pos += 4  # pLinear + reserved
        x_sampling = struct.unpack_from("<i", ch_data, ch_pos)[0]
        ch_pos += 4
        y_sampling = struct.unpack_from("<i", ch_data, ch_pos)[0]
        ch_pos += 4
        if int(x_sampling) == 1 and int(y_sampling) == 1:
            channels.append((name, int(pixel_type), ch_pos))

    if not channels:
        raise ValueError(f"EXR has no supported full-resolution channels: {path}")

    preferred_names = {"Y", "Z", "R", "depth", "Depth"}
    selected_i = 0
    for i, (name, _pixel_type, _unused) in enumerate(channels):
        if name in preferred_names:
            selected_i = i
            break

    pixel_sizes = {0: 4, 1: 2, 2: 4}
    if any(pixel_type not in pixel_sizes for _name, pixel_type, _unused in channels):
        raise ValueError(f"EXR has unsupported channel pixel type: {path}")
    bytes_per_line = sum(pixel_sizes[pixel_type] * width for _name, pixel_type, _unused in channels)
    selected_name, selected_type, _unused = channels[selected_i]
    selected_size = pixel_sizes[selected_type]
    selected_offset = sum(
        pixel_sizes[pixel_type] * width
        for _name, pixel_type, _unused in channels[:selected_i]
    )

    if compression == 0:
        lines_per_chunk = 1
    elif compression == 2:
        lines_per_chunk = 1
    elif compression == 3:
        lines_per_chunk = 16
    else:
        raise ValueError(
            f"Unsupported EXR compression={compression} for {path}; "
            f"channel={selected_name}"
        )

    num_chunks = int(math.ceil(float(height) / float(lines_per_chunk)))
    offset_table_pos = pos
    chunk_offsets = [
        struct.unpack_from("<Q", data, offset_table_pos + 8 * i)[0]
        for i in range(num_chunks)
    ]

    depth = np.empty((height, width), dtype=np.float32)
    for chunk_offset in chunk_offsets:
        chunk_pos = int(chunk_offset)
        y = struct.unpack_from("<i", data, chunk_pos)[0]
        packed_size = struct.unpack_from("<I", data, chunk_pos + 4)[0]
        packed = data[chunk_pos + 8 : chunk_pos + 8 + int(packed_size)]
        if compression == 0:
            unpacked = packed
        else:
            unpacked = _exr_zip_reconstruct(packed)

        y0 = int(y - ymin)
        chunk_lines = min(lines_per_chunk, height - y0)
        expected_size = int(bytes_per_line * chunk_lines)
        if len(unpacked) < expected_size:
            raise ValueError(
                f"Short EXR chunk in {path}: got {len(unpacked)}, expected {expected_size}"
            )

        for line_i in range(chunk_lines):
            line_start = line_i * bytes_per_line + selected_offset
            line_bytes = unpacked[line_start : line_start + selected_size * width]
            if selected_type == 1:
                row = np.frombuffer(line_bytes, dtype="<f2").astype(np.float32)
            elif selected_type == 2:
                row = np.frombuffer(line_bytes, dtype="<f4").astype(np.float32)
            else:
                row = np.frombuffer(line_bytes, dtype="<u4").astype(np.float32)
            depth[y0 + line_i] = row.reshape(width)

    return depth


def read_depth(path: Path, depth_scale: float = 1.0) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(str(path))
    elif suffix == ".exr":
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            depth = read_exr_depth(path)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(
                f"Cannot read depth: {path}. For EXR, check "
                "OPENCV_IO_ENABLE_OPENEXR/OpenCV build."
            )
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if depth_scale != 1.0:
        depth = depth / float(depth_scale)
    return depth


def _round_down_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return int(x)
    return max(m, int(x) // m * m)


def compute_target_hw(
    depth_h: int, depth_w: int, max_side: int, multiple: int
) -> Tuple[int, int]:
    h, w = int(depth_h), int(depth_w)
    if max_side > 0 and max(h, w) > max_side:
        scale = float(max_side) / float(max(h, w))
        h = max(1, int(round(h * scale)))
        w = max(1, int(round(w * scale)))
    h = _round_down_to_multiple(h, multiple)
    w = _round_down_to_multiple(w, multiple)
    return h, w


def resize_rgb_depth_K(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    cam_width: Optional[int],
    cam_height: Optional[int],
    target_h: int,
    target_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = K.astype(np.float64).copy()
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

    if rgb.shape[0] != depth_h or rgb.shape[1] != depth_w:
        rgb = cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_AREA)

    if int(depth_h) != int(target_h) or int(depth_w) != int(target_w):
        sx = float(target_w) / float(depth_w)
        sy = float(target_h) / float(depth_h)
        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    return rgb, depth.astype(np.float32), K


def resize_rgb_to_target(
    rgb: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    if rgb.shape[0] != int(target_h) or rgb.shape[1] != int(target_w):
        rgb = cv2.resize(
            rgb, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA
        )
    return rgb


def resize_depth_to_target(
    depth: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    if depth.shape[0] != int(target_h) or depth.shape[1] != int(target_w):
        depth = cv2.resize(
            depth, (int(target_w), int(target_h)), interpolation=cv2.INTER_NEAREST
        )
    return depth.astype(np.float32)


def scale_K_to_target(
    K: np.ndarray,
    cam_width: Optional[int],
    cam_height: Optional[int],
    source_h: int,
    source_w: int,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    K = K.astype(np.float64).copy()

    if cam_width is None or cam_height is None:
        cam_height, cam_width = int(source_h), int(source_w)

    if int(cam_width) != int(source_w) or int(cam_height) != int(source_h):
        sx = float(source_w) / float(cam_width)
        sy = float(source_h) / float(cam_height)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    if int(source_h) != int(target_h) or int(source_w) != int(target_w):
        sx = float(target_w) / float(source_w)
        sy = float(target_h) / float(source_h)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    return K


def depth_to_world_points_numpy(
    depth: np.ndarray, K: np.ndarray, T_c2w: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)
    R = T_c2w[:3, :3]
    t = T_c2w[:3, 3]
    pts_world = np.einsum("ij,hwj->hwi", R, pts_cam) + t[None, None, :]
    return pts_cam.astype(np.float32), pts_world.astype(np.float32)


def build_image_norm_transform(norm_type: str):
    if norm_type in IMAGE_NORMALIZATION_DICT.keys():
        img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
        return tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    if norm_type == "identity":
        return tvf.ToTensor()
    raise ValueError(
        f"Unknown image normalization type: {norm_type}. "
        f"Available options: identity or {list(IMAGE_NORMALIZATION_DICT.keys())}"
    )


def np_to_torch_img(
    rgb: np.ndarray,
    device: torch.device,
    norm_type: str = "identity",
    img_norm=None,
) -> torch.Tensor:
    if img_norm is None:
        img_norm = build_image_norm_transform(norm_type)
    img = PIL.Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    return img_norm(img).unsqueeze(0).to(device)


def move_view_to_device(view: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in view.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def numpy_quat_xyzw_from_rotmat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    if q[3] < 0:
        q = -q
    return q


# ---------------------------------------------------------------------------
# Build views from scene directory
# ---------------------------------------------------------------------------
def build_views_from_scene(
    scene_dir: Path,
    images_dir: str = "images",
    cams_dir: str = "cams",
    depth_dir: str = "depth",
    frame_glob: str = "*",
    num_views: int = 0,
    start: int = 0,
    stride: int = 1,
    max_side: int = 518,
    size_multiple: int = 14,
    depth_scale: float = 1.0,
    depth_min: float = 1e-6,
    depth_max: float = 1e6,
    device: torch.device = torch.device("cpu"),
    show_progress: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Build a lightweight scene manifest.

    This intentionally does not decode RGB images, depth maps, dense rays, or
    dense point maps. Heavy per-frame tensors are created later by
    load_chunk_views_from_scene for the current chunk only.
    """
    scene_dir = Path(scene_dir).expanduser().resolve()
    images_dir_path = scene_dir / images_dir
    cams_dir_path = scene_dir / cams_dir
    depth_dir_path = scene_dir / depth_dir

    images = collect_stem_to_path(images_dir_path, IMAGE_EXTS)
    depths = collect_stem_to_path(depth_dir_path, DEPTH_EXTS)
    cam_paths = collect_stem_to_path(cams_dir_path, CAM_EXTS)

    if not images:
        raise RuntimeError(f"No images found under {images_dir_path}.")

    selected_stems = sorted(images)
    if frame_glob and frame_glob != "*":
        selected_stems = [
            s for s in selected_stems if fnmatch.fnmatch(s, frame_glob)
        ]
    if start > 0:
        selected_stems = selected_stems[int(start):]
    if stride > 1:
        selected_stems = selected_stems[:: int(stride)]
    selected_stems = contiguous_select(selected_stems, int(num_views))

    if len(selected_stems) == 0:
        raise RuntimeError(f"No frames selected under {images_dir_path}.")

    has_cams_dir = cams_dir_path.exists()
    has_depth_dir = depth_dir_path.exists()
    num_cam_matches = sum(1 for stem in selected_stems if stem in cam_paths)
    num_depth_matches = sum(1 for stem in selected_stems if stem in depths)
    print(
        f"Selected {len(selected_stems)} image frames. "
        f"Camera priors: {'enabled' if num_cam_matches else 'disabled'} "
        f"({num_cam_matches}/{len(selected_stems)} matched; dir_exists={has_cams_dir}). "
        f"Depth priors: {'enabled' if num_depth_matches else 'disabled'} "
        f"({num_depth_matches}/{len(selected_stems)} matched; dir_exists={has_depth_dir})."
    )

    cams: Dict[str, Dict[str, object]] = {}
    cam_stems = [stem for stem in selected_stems if stem in cam_paths]
    for stem in iter_progress(
        cam_stems,
        desc="Load cameras",
        total=len(cam_stems),
        unit="cam",
        enabled=bool(show_progress) and len(cam_stems) > 0,
    ):
        try:
            cams[stem] = parse_cam_txt(cam_paths[stem])
        except Exception as e:
            print(
                f"[WARN] failed to parse camera prior for {stem}: {e}; "
                "skipping camera prior for this frame"
            )

    first_stem = selected_stems[0]
    first_cam = cams.get(first_stem)
    if first_cam is not None and first_cam.get("height") and first_cam.get("width"):
        ref_h, ref_w = int(first_cam["height"]), int(first_cam["width"])
    else:
        ref_h, ref_w = read_image_size(images[first_stem])

    target_h, target_w = compute_target_hw(
        depth_h=ref_h,
        depth_w=ref_w,
        max_side=max_side,
        multiple=size_multiple,
    )

    views: List[Dict[str, object]] = []
    resized_stems: List[str] = []

    manifest_iter = iter_progress(
        selected_stems,
        desc="Build manifest",
        total=len(selected_stems),
        unit="frame",
        enabled=bool(show_progress) and len(selected_stems) > 0,
    )
    for i, stem in enumerate(manifest_iter):
        cam = cams.get(stem)

        if cam is not None:
            if cam.get("width") is None or cam.get("height") is None:
                try:
                    cam["height"], cam["width"] = read_image_size(images[stem])
                except Exception as e:
                    print(
                        f"[WARN] failed to read image size for {stem}: {e}; "
                        "camera K scaling will use target size as source size"
                    )
                    cam["height"], cam["width"] = target_h, target_w

        view: Dict[str, object] = {
            "stem": stem,
            "image_path": str(images[stem]),
            "depth_path": str(depths[stem]) if stem in depths else None,
            "cam_path": str(cam_paths[stem]) if stem in cam_paths else None,
            "label": [scene_dir.name],
            "instance": [stem],
            "idx": [f"{scene_dir.name}/{stem}"],
        }

        views.append(view)
        resized_stems.append(stem)
        if hasattr(manifest_iter, "set_postfix"):
            manifest_iter.set_postfix(
                stem=stem,
                cam=int(cam is not None),
                depth=int(stem in depths),
            )

    image_paths = {stem: str(images[stem]) for stem in resized_stems}
    depth_paths = {stem: str(depths[stem]) for stem in resized_stems if stem in depths}
    cam_path_map = {stem: str(cam_paths[stem]) for stem in resized_stems if stem in cam_paths}

    meta = {
        "scene_dir": str(scene_dir),
        "images_dir": str(images_dir),
        "cams_dir": str(cams_dir),
        "depth_dir": str(depth_dir),
        "stems": resized_stems,
        "target_h": target_h,
        "target_w": target_w,
        "image_paths": image_paths,
        "depth_paths": depth_paths,
        "cam_paths": cam_path_map,
        "cams": cams,
        "depth_scale": float(depth_scale),
        "depth_min": float(depth_min),
        "depth_max": float(depth_max),
        "num_cam_priors": int(sum(1 for stem in resized_stems if stem in cams)),
        "num_depth_priors": int(
            sum(1 for stem in resized_stems if stem in depths)
        ),
        "num_gt_rgbd_priors": 0,
    }
    return views, meta


def _load_chunk_frame_cpu(args: Tuple[Dict[str, object], Dict[str, object], int, int, float, float, float, bool, bool]) -> Dict[str, object]:
    view_ref, cam, target_h, target_w, depth_scale, depth_min, depth_max, need_depth, need_rays = args
    stem = str(view_ref["stem"])
    rgb_raw = read_rgb(Path(str(view_ref["image_path"])))

    K: Optional[np.ndarray] = None
    T_c2w: Optional[np.ndarray] = None
    if cam:
        if cam.get("width") is None or cam.get("height") is None:
            cam = dict(cam)
            cam["height"], cam["width"] = rgb_raw.shape[:2]
        K = scale_K_to_target(
            K=np.asarray(cam["K"], dtype=np.float64),
            cam_width=cam.get("width"),
            cam_height=cam.get("height"),
            source_h=rgb_raw.shape[0],
            source_w=rgb_raw.shape[1],
            target_h=target_h,
            target_w=target_w,
        )
        T_c2w = np.asarray(cam["T_c2w"], dtype=np.float64)

    rgb = resize_rgb_to_target(rgb_raw, target_h=target_h, target_w=target_w)
    depth: Optional[np.ndarray] = None
    if need_depth and view_ref.get("depth_path"):
        depth_raw = read_depth(Path(str(view_ref["depth_path"])), depth_scale=depth_scale)
        if K is not None and cam:
            K = scale_K_to_target(
                K=np.asarray(cam["K"], dtype=np.float64),
                cam_width=cam.get("width"),
                cam_height=cam.get("height"),
                source_h=depth_raw.shape[0],
                source_w=depth_raw.shape[1],
                target_h=target_h,
                target_w=target_w,
            )
        depth = resize_depth_to_target(depth_raw, target_h=target_h, target_w=target_w)

    return {
        "stem": stem,
        "rgb": rgb,
        "depth": depth,
        "K": K,
        "T_c2w": T_c2w,
        "need_rays": bool(need_rays),
        "depth_min": float(depth_min),
        "depth_max": float(depth_max),
    }


def _chunk_loader_need_depth(prior_policy: Dict[str, object]) -> bool:
    return str(prior_policy.get("depth", "none")) == "input"


def _chunk_loader_need_rays(prior_policy: Dict[str, object]) -> bool:
    return str(prior_policy.get("ray", "none")) == "input"


def load_chunk_views_from_scene(
    lightweight_views: Sequence[Dict[str, object]],
    meta: Dict[str, object],
    indices: Sequence[int],
    prior_policy: Dict[str, object],
    device: torch.device,
    recenter_anchor: Optional[np.ndarray] = None,
    num_workers: int = 0,
    norm_type: str = "identity",
) -> Tuple[List[Dict[str, object]], List[np.ndarray]]:
    """Decode and tensorize only the frames needed by one chunk."""
    target_h = int(meta["target_h"])
    target_w = int(meta["target_w"])
    cams = meta.get("cams", {})
    need_depth = _chunk_loader_need_depth(prior_policy)
    need_rays = _chunk_loader_need_rays(prior_policy)

    jobs = []
    for idx in indices:
        ref = lightweight_views[int(idx)]
        stem = str(ref["stem"])
        jobs.append(
            (
                ref,
                cams.get(stem, {}),
                target_h,
                target_w,
                float(meta.get("depth_scale", 1.0)),
                float(meta.get("depth_min", 1e-6)),
                float(meta.get("depth_max", 1e6)),
                need_depth,
                need_rays,
            )
        )

    if int(num_workers) > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=int(num_workers)) as pool:
            frames = list(pool.map(_load_chunk_frame_cpu, jobs))
    else:
        frames = [_load_chunk_frame_cpu(job) for job in jobs]

    anchor = None
    if recenter_anchor is not None:
        anchor = np.asarray(recenter_anchor, dtype=np.float64).reshape(3)

    views: List[Dict[str, object]] = []
    rgbs: List[np.ndarray] = []
    scene_name = Path(str(meta["scene_dir"])).name
    img_norm = build_image_norm_transform(norm_type)

    for frame in frames:
        stem = str(frame["stem"])
        rgb = np.asarray(frame["rgb"], dtype=np.uint8)
        rgbs.append(rgb)

        view: Dict[str, object] = {
            "img": np_to_torch_img(
                rgb,
                device=device,
                norm_type=norm_type,
                img_norm=img_norm,
            ),
            "is_metric_scale": torch.ones((1,), dtype=torch.bool, device=device),
            "is_synthetic": torch.zeros((1,), dtype=torch.bool, device=device),
            "true_shape": torch.tensor([[target_h, target_w]], dtype=torch.int64, device=device),
            "data_norm_type": [norm_type],
            "label": [scene_name],
            "instance": [stem],
            "idx": [f"{scene_name}/{stem}"],
        }

        K = frame.get("K")
        T_c2w = frame.get("T_c2w")
        if K is not None and T_c2w is not None:
            T_adj = np.asarray(T_c2w, dtype=np.float64).copy()
            if anchor is not None:
                T_adj[:3, 3] -= anchor
            K_t = torch.from_numpy(np.asarray(K, dtype=np.float32)).unsqueeze(0).to(device)
            pose_t = torch.from_numpy(T_adj.astype(np.float32)).unsqueeze(0).to(device)
            pose_quat = numpy_quat_xyzw_from_rotmat(T_adj[:3, :3])
            pose_quat_t = torch.from_numpy(pose_quat).unsqueeze(0).to(device)
            pose_trans_t = pose_t[..., :3, 3]

            view["camera_intrinsics"] = K_t
            view["camera_pose"] = pose_t
            view["camera_pose_quats"] = pose_quat_t
            view["camera_pose_trans"] = pose_trans_t
            view["world_translation"] = pose_trans_t

            if bool(frame.get("need_rays", False)):
                _, ray_dirs_t = get_rays_in_camera_frame(
                    K_t,
                    target_h,
                    target_w,
                    normalize_to_unit_sphere=True,
                )
                view["ray_directions_cam"] = ray_dirs_t

        depth = frame.get("depth")
        if depth is not None:
            depth_np = np.asarray(depth, dtype=np.float32)
            valid_np = (
                np.isfinite(depth_np)
                & (depth_np > float(frame["depth_min"]))
                & (depth_np < float(frame["depth_max"]))
            )
            depth_t = torch.from_numpy(depth_np).unsqueeze(0).to(device)
            valid_t = torch.from_numpy(valid_np).unsqueeze(0).to(device)
            view["depthmap"] = depth_t
            view["valid_mask"] = valid_t
            view["non_ambiguous_mask"] = valid_t.clone()

            if K is not None and T_c2w is not None:
                T_for_points = np.asarray(T_c2w, dtype=np.float64).copy()
                if anchor is not None:
                    T_for_points[:3, 3] -= anchor
                pts_cam_np, pts_world_np = depth_to_world_points_numpy(
                    depth_np,
                    np.asarray(K, dtype=np.float64),
                    T_for_points,
                )
                depth_along_ray_np = np.linalg.norm(
                    pts_cam_np, axis=-1, keepdims=True
                ).astype(np.float32)
                valid_np = (
                    valid_np
                    & np.isfinite(pts_world_np).all(axis=-1)
                    & np.isfinite(pts_cam_np).all(axis=-1)
                )
                valid_t = torch.from_numpy(valid_np).unsqueeze(0).to(device)
                view["pts3d"] = torch.from_numpy(pts_world_np).unsqueeze(0).to(device)
                view["pts3d_cam"] = torch.from_numpy(pts_cam_np).unsqueeze(0).to(device)
                view["depth_along_ray"] = torch.from_numpy(depth_along_ray_np).unsqueeze(0).to(device)
                view["valid_mask"] = valid_t
                view["non_ambiguous_mask"] = valid_t.clone()

        views.append(view)

    return views, rgbs


def _load_gt_points_worker(args: Tuple[object, ...]) -> Tuple[np.ndarray, np.ndarray, str]:
    (
        stem,
        image_path,
        depth_path,
        cam,
        target_h,
        target_w,
        depth_scale,
        depth_min,
        depth_max,
        max_points,
        seed,
    ) = args
    try:
        if not depth_path or not cam:
            return (
                np.empty((0, 3), np.float32),
                np.empty((0, 3), np.uint8),
                "missing_depth_or_cam",
            )

        rgb_raw = read_rgb(Path(str(image_path)))
        depth_raw = read_depth(Path(str(depth_path)), depth_scale=float(depth_scale))
        rgb, depth, K = resize_rgb_depth_K(
            rgb=rgb_raw,
            depth=depth_raw,
            K=np.asarray(cam["K"], dtype=np.float64),
            cam_width=cam.get("width"),
            cam_height=cam.get("height"),
            target_h=int(target_h),
            target_w=int(target_w),
        )
        T_c2w = np.asarray(cam["T_c2w"], dtype=np.float64)
        pts_cam_np, pts_world_np = depth_to_world_points_numpy(depth, K, T_c2w)
        valid = (
            np.isfinite(depth)
            & (depth > float(depth_min))
            & (depth < float(depth_max))
            & np.isfinite(pts_cam_np).all(axis=-1)
            & np.isfinite(pts_world_np).all(axis=-1)
        )
        if not bool(valid.any()):
            return (
                np.empty((0, 3), np.float32),
                np.empty((0, 3), np.uint8),
                "no_valid_depth",
            )
        points = pts_world_np[valid].reshape(-1, 3).astype(np.float32)
        colors = rgb[valid].reshape(-1, 3).astype(np.uint8)
        points, colors = sample_points_and_colors(
            points,
            colors,
            max_points=int(max_points),
            seed=int(seed),
        )
        return points, colors, "ok"
    except Exception as exc:
        return (
            np.empty((0, 3), np.float32),
            np.empty((0, 3), np.uint8),
            f"{stem}: {exc}",
        )


def load_gt_points_from_meta(
    meta: Dict[str, object],
    max_points: int,
    seed: int,
    num_workers: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load GT RGB-D priors at save time and convert them to a sampled point cloud."""
    existing_points = meta.get("gt_points", None)
    existing_colors = meta.get("gt_colors", None)
    if existing_points is not None and existing_colors is not None:
        return sample_points_and_colors(
            np.asarray(existing_points, dtype=np.float32),
            np.asarray(existing_colors, dtype=np.uint8),
            max_points=int(max_points),
            seed=int(seed),
        )

    stems = list(meta.get("stems", []))
    cams = meta.get("gt_cams", meta.get("cams", {}))
    image_paths = meta.get("image_paths", {})
    depth_paths = meta.get("depth_paths", {})
    if not stems or not isinstance(cams, dict) or not isinstance(depth_paths, dict):
        cams = cams if isinstance(cams, dict) else {}
        image_paths = image_paths if isinstance(image_paths, dict) else {}
        depth_paths = depth_paths if isinstance(depth_paths, dict) else {}

    scene_dir_text = str(meta.get("scene_dir", "") or "")
    scene_dir = Path(scene_dir_text).expanduser().resolve() if scene_dir_text else None
    if stems and scene_dir is not None and scene_dir.exists():
        images_dir = str(meta.get("images_dir", "images"))
        cams_dir = str(meta.get("cams_dir", "cams"))
        depth_dir = str(meta.get("depth_dir", "depth"))
        fallback_images = collect_stem_to_path(scene_dir / images_dir, IMAGE_EXTS)
        fallback_depths = collect_stem_to_path(scene_dir / depth_dir, DEPTH_EXTS)
        fallback_cam_paths = collect_stem_to_path(scene_dir / cams_dir, CAM_EXTS)

        if len(image_paths) < len(stems):
            image_paths = {
                **{str(k): str(v) for k, v in fallback_images.items()},
                **{str(k): str(v) for k, v in image_paths.items()},
            }
        if len(depth_paths) < len(stems):
            depth_paths = {
                **{str(k): str(v) for k, v in fallback_depths.items()},
                **{str(k): str(v) for k, v in depth_paths.items()},
            }
        if len(cams) < len(stems):
            merged_cams = dict(cams)
            for stem, cam_path in fallback_cam_paths.items():
                if stem in merged_cams:
                    continue
                try:
                    merged_cams[stem] = parse_cam_txt(cam_path)
                except Exception as exc:
                    print(f"[GT][WARN] failed to parse fallback camera {cam_path}: {exc}")
            cams = merged_cams

    eligible = [
        stem
        for stem in stems
        if stem in cams and stem in depth_paths and stem in image_paths
    ]
    if not eligible:
        print(
            "[GT][WARN] no RGB-D frames eligible for GT point cloud: "
            f"stems={len(stems)}, images={len(image_paths)}, "
            f"depths={len(depth_paths)}, cams={len(cams)}"
        )
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)

    if int(max_points) > 0:
        per_frame_cap = max(1, int(math.ceil(float(max_points) * 1.5 / len(eligible))))
    else:
        per_frame_cap = 0

    jobs: List[Tuple[object, ...]] = []
    for local_i, stem in enumerate(eligible):
        jobs.append(
            (
                stem,
                image_paths[stem],
                depth_paths[stem],
                cams[stem],
                int(meta["target_h"]),
                int(meta["target_w"]),
                float(meta.get("depth_scale", 1.0)),
                float(meta.get("depth_min", 1e-6)),
                float(meta.get("depth_max", 1e6)),
                int(per_frame_cap),
                int(seed) + 104729 * (local_i + 1),
            )
        )

    if int(num_workers) > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=int(num_workers)) as pool:
            results = list(pool.map(_load_gt_points_worker, jobs))
    else:
        results = [_load_gt_points_worker(job) for job in jobs]

    points_all: List[np.ndarray] = []
    colors_all: List[np.ndarray] = []
    failed = 0
    failure_counts: Dict[str, int] = {}
    for points, colors, status in results:
        if status == "ok" and points.shape[0] > 0:
            points_all.append(points)
            colors_all.append(colors)
        else:
            failed += 1
            failure_counts[str(status)] = failure_counts.get(str(status), 0) + 1

    if failed:
        top_failures = ", ".join(
            f"{key}={count}"
            for key, count in list(failure_counts.items())[:5]
        )
        print(
            f"[GT] skipped/failed RGB-D point loading for "
            f"{failed}/{len(results)} frames. {top_failures}"
        )

    if not points_all:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)

    points = np.concatenate(points_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)
    points, colors = sample_points_and_colors(
        points,
        colors,
        max_points=int(max_points),
        seed=int(seed),
    )
    print(
        f"[GT] loaded RGB-D point cloud at save time: "
        f"frames={len(results) - failed}/{len(results)}, points={points.shape[0]}"
    )
    return points, colors


# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------
def sample_points_and_colors(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    return points, colors


# ---------------------------------------------------------------------------
# Scale alignment helpers
# ---------------------------------------------------------------------------
def sample_alignment_correspondences(
    gt_maps: Sequence[np.ndarray],
    pred_maps: Sequence[np.ndarray],
    gt_valid_masks: Sequence[np.ndarray],
    pred_valid_masks: Sequence[np.ndarray],
    max_samples_per_view: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    gt_corr_all: List[np.ndarray] = []
    pr_corr_all: List[np.ndarray] = []

    for view_idx, (gt, pr, gt_valid, pr_valid) in enumerate(
        zip(gt_maps, pred_maps, gt_valid_masks, pred_valid_masks)
    ):
        if (
            gt.shape[:2] != pr.shape[:2]
            or gt.ndim != 3
            or pr.ndim != 3
        ):
            print(
                f"[WARN] skip scale alignment view {view_idx}: "
                f"shape mismatch gt={gt.shape}, pred={pr.shape}"
            )
            continue

        valid = (
            gt_valid.astype(bool)
            & pr_valid.astype(bool)
            & np.isfinite(gt).all(axis=-1)
            & np.isfinite(pr).all(axis=-1)
        )
        v, u = np.nonzero(valid)
        if v.size == 0:
            continue
        if max_samples_per_view > 0 and v.size > max_samples_per_view:
            sel = rng.choice(
                v.size, size=max_samples_per_view, replace=False
            )
            v = v[sel]
            u = u[sel]
        gt_corr_all.append(gt[v, u].reshape(-1, 3).astype(np.float32))
        pr_corr_all.append(pr[v, u].reshape(-1, 3).astype(np.float32))

    if not gt_corr_all:
        return (
            np.empty((0, 3), np.float32),
            np.empty((0, 3), np.float32),
        )
    return (
        np.concatenate(gt_corr_all, axis=0),
        np.concatenate(pr_corr_all, axis=0),
    )


def estimate_scale_from_random_baselines(
    pr_corr: np.ndarray,
    gt_corr: np.ndarray,
    seed: int,
    max_pairs: int = 20000,
) -> Tuple[float, int, bool]:
    pr = np.asarray(pr_corr, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt_corr, dtype=np.float64).reshape(-1, 3)
    n = min(pr.shape[0], gt.shape[0])
    if n < 2:
        return 1.0, 0, False

    rng = np.random.default_rng(seed)
    if n == 2:
        i = np.asarray([0], dtype=np.int64)
        j = np.asarray([1], dtype=np.int64)
    else:
        num_pairs = min(int(max_pairs), n * (n - 1) // 2)
        i = rng.integers(0, n, size=num_pairs, endpoint=False)
        j = rng.integers(0, n, size=num_pairs, endpoint=False)
        keep = i != j
        i = i[keep]
        j = j[keep]

    if i.size == 0:
        return 1.0, 0, False

    d_pr = np.linalg.norm(pr[i] - pr[j], axis=1)
    d_gt = np.linalg.norm(gt[i] - gt[j], axis=1)
    valid = (
        np.isfinite(d_pr)
        & np.isfinite(d_gt)
        & (d_pr > 1e-8)
        & (d_gt > 0)
    )
    if not valid.any():
        return 1.0, 0, False

    ratios = d_gt[valid] / d_pr[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios > 1e-12)]
    if ratios.size == 0:
        return 1.0, 0, False
    return float(np.median(ratios)), int(ratios.size), True


# ---------------------------------------------------------------------------
# Voxel downsampling
# ---------------------------------------------------------------------------
def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Voxel-grid downsample a point cloud (keep one point per occupied voxel)."""
    if voxel_size <= 0 or points.shape[0] == 0:
        return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.uint8)

    n_before = int(points.shape[0])

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    n = min(pts.shape[0], cols.shape[0])
    pts = pts[:n]
    cols = cols[:n]

    voxel_coords = np.floor(pts / float(voxel_size)).astype(np.int64)
    _, idx = np.unique(voxel_coords, axis=0, return_index=True)

    n_after = int(idx.size)
    if n_before != n_after:
        print(
            f"[VOXEL] {n_before:,} -> {n_after:,} points "
            f"(voxel_size={voxel_size:.3f}, ratio={n_after / max(n_before, 1):.1%})"
        )

    return pts[idx].astype(np.float32), cols[idx].astype(np.uint8)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device_arg == "cuda":
        device_arg = "cuda:0"
    elif device_arg.isdigit():
        device_arg = f"cuda:{int(device_arg)}"

    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False."
            )
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print(f"Using device {device}: {torch.cuda.get_device_name(device.index)}")
    else:
        print(f"Using device {device}")
    return device
