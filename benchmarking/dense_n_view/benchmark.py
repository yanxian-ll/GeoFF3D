# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Benchmark dense multi-view metric reconstruction.

Key changes compared with the earlier version:
1. Estimate a robust global Sim(3) per multi-view set in `get_all_info_for_metric_computation`.
   The Sim(3) is solved from dense point correspondences sampled from all views plus camera-center
   correspondences, and is used to align predicted world geometry / poses to GT.
2. System-level absolute metrics (pose absolute, merged point cloud absolute, global pointmap absolute)
   are computed after global alignment; pose_points uses independent pose/geometry alignments.
3. Per-view depth absolute metrics are evaluated with sequence-level scale-only alignment, which better
   reflects single-frame depth quality without introducing an extra per-view affine or per-view Sim(3).
4. Removed the old fragmented logic where point/depth/fused-PC used different alignments.
5. Optionally align geometry and poses independently (`alignment_mode=pose_points`).
6. Optionally report a separately aligned Pose ATE for protocol comparison.
"""

import json
import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch

try:
    import rerun as rr
    try:
        import rerun.blueprint as rrb
    except Exception:
        rrb = None
except Exception:
    rr = None
    rrb = None

import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from mapanything.datasets import get_test_data_loader, get_test_many_ar_data_loader
from mapanything.models import init_model
from mapanything.utils.geometry import (
    geotrf,
    inv,
    normalize_multiple_pointclouds,
    quaternion_to_rotation_matrix,
    transform_pose_using_quats_and_trans_2_to_1,
    recover_pinhole_intrinsics_from_ray_directions
)
from mapanything.utils.metrics import (
    camera_center_scale_metrics,
    compute_set_metrics,
    sim3_from_correspondences_robust,
    umeyama_sim3_torch,
)
from mapanything.utils.misc import StreamToLogger
from mapanything.utils.torch_hub_setup import configure_torch_hub

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from spatial_rrd.geometry_align import (  # noqa: E402
    estimate_scale_yaw_translation_from_correspondences,
)


log = logging.getLogger(__name__)


def _cfg_get(cfg, key, default=None):
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return getattr(cfg, key, default)


def resolve_benchmark_device(args):
    """
    Resolve benchmark device from hydra config.

    Supported:
      device=auto
      device=cpu
      device=cuda
      device=cuda:0
      device=cuda:1
      cuda_device=1
    """
    device_cfg = _cfg_get(args, "device", "auto")
    cuda_device_cfg = _cfg_get(args, "cuda_device", None)

    if device_cfg is None:
        device_cfg = "auto"

    device_str = str(device_cfg).strip().lower()

    # cuda_device has higher priority, because it is easier to pass by Hydra CLI.
    if cuda_device_cfg is not None and str(cuda_device_cfg).lower() not in {"", "none", "null"}:
        device_str = f"cuda:{int(cuda_device_cfg)}"
    elif device_str in {"auto", ""}:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device_str.isdigit():
        device_str = f"cuda:{int(device_str)}"
    elif device_str == "cuda":
        device_str = "cuda:0"

    device = torch.device(device_str)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({device_str}), but torch.cuda.is_available() is False.")

        if device.index is None:
            device = torch.device("cuda:0")

        n_cuda = torch.cuda.device_count()
        if device.index < 0 or device.index >= n_cuda:
            raise ValueError(
                f"Invalid cuda device index {device.index}. "
                f"Visible CUDA device count is {n_cuda}. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}"
            )

        # Important: make cuda:N the current CUDA device.
        # This avoids hidden cuda:0 allocations from code that uses device='cuda'.
        torch.cuda.set_device(device)
        print(
            f"[benchmark] Using device: {device} | "
            f"name={torch.cuda.get_device_name(device.index)} | "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}"
        )
    else:
        print(f"[benchmark] Using device: {device}")

    return device

# ============================================================
# IO helpers
# ============================================================

def write_ply_xyzrgb(path, points, colors_u8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors_u8 = np.asarray(colors_u8, dtype=np.uint8)

    assert points.shape[0] == colors_u8.shape[0]
    n = points.shape[0]

    vertex = np.empty(
        n,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors_u8[:, 0]
    vertex["green"] = colors_u8[:, 1]
    vertex["blue"] = colors_u8[:, 2]

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header\n",
        ]
    ).encode("ascii")

    with open(path, "wb") as f:
        f.write(header)
        vertex.tofile(f)


# ============================================================
# Serialization helpers
# ============================================================

def _to_python(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _batched_get(x, batch_idx):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x[batch_idx]
    if isinstance(x, (list, tuple)):
        return x[batch_idx]
    return x


def save_repro_bundle_json(
    json_path,
    benchmark_dataset_name,
    scene,
    set_idx,
    batch_views,
    batch_idx,
    gt_info_abs,
    pr_info_abs_aligned,
    scale_factors,
    metrics,
    gt_ply_path,
    pred_ply_path,
    rrd_path=None,
):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    n_views = len(batch_views)
    views = []

    for i in range(n_views):
        instance = _batched_get(batch_views[i].get("instance", None), batch_idx)
        instance = str(instance) if instance is not None else None

        gt_intr = _batched_get(gt_info_abs["intrinsics"][i], batch_idx) if "intrinsics" in gt_info_abs else None
        gt_pose = _batched_get(gt_info_abs["poses"][i], batch_idx) if "poses" in gt_info_abs else None

        pred_intr = _batched_get(pr_info_abs_aligned["intrinsics"][i], batch_idx) if "intrinsics" in pr_info_abs_aligned else None
        pred_pose = _batched_get(pr_info_abs_aligned["poses"][i], batch_idx) if "poses" in pr_info_abs_aligned else None

        views.append(
            {
                "view_idx": i,
                "instance": instance,
                "gt_cam": {
                    "intrinsics": _to_python(gt_intr),
                    "c2w": _to_python(gt_pose),
                },
                "pred_cam": {
                    "intrinsics": _to_python(pred_intr),
                    "c2w": _to_python(pred_pose),
                },
            }
        )

    scale_meta = {}
    if scale_factors is not None:
        for key, value in scale_factors.items():
            try:
                scale_meta[key] = _to_python(_batched_get(value, batch_idx))
            except Exception:
                scale_meta[key] = _to_python(value)

    payload = {
        "benchmark_dataset": benchmark_dataset_name,
        "scene": str(scene),
        "set_index": int(set_idx),
        "instances": [v["instance"] for v in views],
        "views": views,
        "metrics": {
            k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else _to_python(v)
            for k, v in metrics.items()
        },
        "scale_factors": scale_meta,
        "fused_outputs": {
            "gt_ply": os.path.abspath(gt_ply_path) if gt_ply_path else None,
            "pred_ply": os.path.abspath(pred_ply_path) if pred_ply_path else None,
            "rrd": os.path.abspath(rrd_path) if rrd_path else None,
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ============================================================
# Rerun (.rrd) helpers
# ============================================================

def _as_np(x, dtype=np.float64):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _validate_c2w(mat):
    mat = _as_np(mat, dtype=np.float64)
    if mat is None or mat.shape != (4, 4):
        return None
    if not np.isfinite(mat).all():
        return None
    return mat


def _sanitize_rrd_name(x):
    x = str(x).replace("/", "_").replace("\\", "_")
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in x)


def rr_set_time_compat(name: str, sequence: int) -> None:
    try:
        rr.set_time(name, sequence=sequence)
    except AttributeError:
        rr.set_time_sequence(name, sequence)


def rr_disconnect_compat() -> None:
    disconnect_fn = getattr(rr, "disconnect", None)
    shutdown_fn = getattr(rr, "shutdown", None)
    try:
        if callable(disconnect_fn):
            disconnect_fn()
        elif callable(shutdown_fn):
            shutdown_fn()
    except Exception:
        pass


def rr_init_save_compat(app_id: str, recording_id: str, save_rrd) -> None:
    """Initialize a fresh recording and attach a file sink."""
    try:
        rr.init(app_id, recording_id=recording_id, spawn=False)
    except TypeError:
        rr.init(app_id, spawn=False)
    rr.save(str(save_rrd))


def send_clean_screenshot_blueprint(
    background=(255, 255, 255),
    hide_grid: bool = False,
    collapse_panels: bool = True,
) -> None:
    if rrb is None:
        return

    try:
        line_grid = rrb.LineGrid3D(visible=not hide_grid)
        blueprint = rrb.Blueprint(
            rrb.Spatial3DView(
                origin="/world",
                name="Benchmark Scene",
                background=list(background),
                line_grid=line_grid,
            ),
            collapse_panels=bool(collapse_panels),
        )
        rr.send_blueprint(blueprint)
    except Exception as e:
        print(f"[WARN] failed to send Rerun blueprint: {e}")


def log_point_cloud_to_rerun(entity_path: str, pts, colors=None, point_radius: float = 0.0):
    pts = _as_np(pts, dtype=np.float32)
    if pts is None or pts.size == 0:
        return

    kwargs = dict(positions=pts)

    if colors is not None:
        colors = _as_np(colors, dtype=np.uint8)
        if colors is not None and colors.shape[0] == pts.shape[0]:
            kwargs["colors"] = colors

    if point_radius is not None and point_radius > 0:
        kwargs["radii"] = float(point_radius)

    rr.log(entity_path, rr.Points3D(**kwargs))


def estimate_axis_size_from_points(point_arrays, axis_size: float = 0.0):
    """
    Estimate a visually reasonable camera-axis length from GT/Pred point clouds.
    If axis_size > 0, use it directly.
    """
    if axis_size is not None and axis_size > 0:
        return float(axis_size)

    valid_pts = []
    for pts in point_arrays:
        pts = _as_np(pts, dtype=np.float32)
        if pts is None or pts.size == 0:
            continue

        pts = pts.reshape(-1, 3)
        finite = np.isfinite(pts).all(axis=1)
        if finite.any():
            valid_pts.append(pts[finite])

    if not valid_pts:
        return 0.1

    pts = np.concatenate(valid_pts, axis=0)
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    if not np.isfinite(diag) or diag <= 0:
        diag = 1.0

    return max(diag * 0.03, 1e-4)


def collect_cameras_for_rrd(info_abs, batch_views, batch_idx, prefix: str):
    """
    Collect camera c2w matrices for one sample in a multi-view batch.
    """
    cams = []
    n_views = len(batch_views)

    for view_idx in range(n_views):
        pose = _batched_get(info_abs["poses"][view_idx], batch_idx) if "poses" in info_abs else None
        intr = _batched_get(info_abs["intrinsics"][view_idx], batch_idx) if "intrinsics" in info_abs else None

        pose = _validate_c2w(pose)
        intr = _as_np(intr, dtype=np.float64)

        if pose is None:
            continue

        instance = _batched_get(batch_views[view_idx].get("instance", None), batch_idx)
        instance = str(instance) if instance is not None else ""

        stem = f"{prefix}_{view_idx:03d}"
        if instance:
            stem = f"{stem}_{instance}"

        cams.append(
            {
                "stem": stem,
                "T_c2w": pose,
                "K": intr,
            }
        )

    return cams


def make_camera_axes_strips(
    cams_ordered,
    axis_size: float,
    axis_colors=None,
):
    """
    Draw three camera local axes for each camera.

    GT / Pred should pass different axis_colors, while x/y/z stay distinguishable.
    axis_colors: ((x_color), (y_color), (z_color))
    """
    strips = []
    colors = []

    if axis_colors is None:
        axis_colors = (
            (255, 0, 0),      # x
            (0, 220, 0),      # y
            (40, 80, 255),    # z
        )

    color_x = np.asarray(axis_colors[0], dtype=np.uint8)
    color_y = np.asarray(axis_colors[1], dtype=np.uint8)
    color_z = np.asarray(axis_colors[2], dtype=np.uint8)

    for cam in cams_ordered:
        T_c2w = np.asarray(cam["T_c2w"], dtype=np.float64)
        R = T_c2w[:3, :3]
        o = T_c2w[:3, 3]

        x_end = o + R[:, 0] * axis_size
        y_end = o + R[:, 1] * axis_size
        z_end = o + R[:, 2] * axis_size

        strips.extend(
            [
                np.stack([o, x_end], axis=0).astype(np.float32),
                np.stack([o, y_end], axis=0).astype(np.float32),
                np.stack([o, z_end], axis=0).astype(np.float32),
            ]
        )
        colors.extend([color_x, color_y, color_z])

    return strips, colors


def log_camera_axes(
    entity_path: str,
    cams_ordered,
    axis_size: float,
    radius: float = 0.0,
    axis_colors=None,
):
    strips, colors = make_camera_axes_strips(
        cams_ordered,
        axis_size=axis_size,
        axis_colors=axis_colors,
    )
    if not strips:
        return

    kwargs = dict(
        strips=strips,
        colors=colors,
    )

    if radius is not None and radius > 0:
        kwargs["radii"] = float(radius)

    rr.log(entity_path, rr.LineStrips3D(**kwargs))


def log_camera_center_labels(
    entity_path: str,
    cams_ordered,
    label_color=(255, 255, 255),
):
    """
    Log only camera labels at camera centers.

    No center point and no trajectory are intentionally logged.
    Points3D is used only because Rerun labels are attached to points.
    """
    if len(cams_ordered) == 0:
        return

    centers = np.asarray(
        [np.asarray(c["T_c2w"], dtype=np.float64)[:3, 3] for c in cams_ordered],
        dtype=np.float32,
    )
    labels = [str(c.get("stem", f"cam_{i:04d}")) for i, c in enumerate(cams_ordered)]

    colors = np.repeat(
        np.array([label_color], dtype=np.uint8),
        len(centers),
        axis=0,
    )

    kwargs = dict(
        positions=centers,
        colors=colors,
        labels=labels,
    )

    # radii=0.0 tries to hide the anchor point and keep only the label.
    # Some rerun versions may not accept radii=0.0, so keep a fallback.
    try:
        rr.log(entity_path, rr.Points3D(**kwargs, radii=0.0))
    except TypeError:
        rr.log(entity_path, rr.Points3D(**kwargs))


def save_repro_bundle_rrd(
    rrd_path,
    benchmark_dataset_name,
    scene,
    set_idx,
    batch_views,
    batch_idx,
    gt_info_abs,
    pr_info_abs_aligned,
    gt_points,
    gt_colors,
    pred_points,
    pred_colors,
    default_width=640,
    default_height=480,
    background=(255, 255, 255),
    hide_grid=False,
    collapse_panels=True,
    point_radius=0.0,
    axis_size=0.0,
    axis_radius=0.0,
    show_center_labels=True,
):
    """
    Save a screenshot-friendly Rerun bundle.

    Kept entities:
      world/gt/points
      world/pred/points
      world/gt/cameras/center_labels
      world/pred/cameras/center_labels
      world/gt/cameras/axes
      world/pred/cameras/axes

    Removed intentionally:
      centers
      trajectory
      frustums
    """
    if rr is None:
        raise ImportError("Saving .rrd requires rerun. Please install rerun-sdk first.")

    rrd_path = Path(rrd_path)
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    safe_scene = _sanitize_rrd_name(scene)
    recording_id = f"{benchmark_dataset_name}_{safe_scene}_set{int(set_idx):03d}"

    rr_init_save_compat(
        app_id="benchmark_dense_n_view",
        recording_id=recording_id,
        save_rrd=rrd_path,
    )

    rr_set_time_compat("stable_time", int(set_idx))

    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    send_clean_screenshot_blueprint(
        background=background,
        hide_grid=hide_grid,
        collapse_panels=collapse_panels,
    )

    # Point clouds.
    log_point_cloud_to_rerun(
        "world/gt/points",
        gt_points,
        gt_colors,
        point_radius=point_radius,
    )
    log_point_cloud_to_rerun(
        "world/pred/points",
        pred_points,
        pred_colors,
        point_radius=point_radius,
    )

    # Cameras.
    gt_cams = collect_cameras_for_rrd(
        gt_info_abs,
        batch_views,
        batch_idx,
        prefix="gt",
    )
    pred_cams = collect_cameras_for_rrd(
        pr_info_abs_aligned,
        batch_views,
        batch_idx,
        prefix="pred",
    )

    gt_label_color = (60, 200, 120)       # green: GT labels
    pred_label_color = (255, 120, 40)     # orange: Pred labels

    gt_axis_colors = (
        (255, 0, 0),       # GT x: red
        (0, 220, 0),       # GT y: green
        (40, 80, 255),     # GT z: blue
    )
    pred_axis_colors = (
        (255, 0, 255),     # Pred x: magenta
        (255, 180, 0),     # Pred y: orange
        (0, 220, 255),     # Pred z: cyan
    )

    cam_axis_size = estimate_axis_size_from_points(
        [gt_points, pred_points],
        axis_size=axis_size,
    )

    if gt_cams:
        if show_center_labels:
            log_camera_center_labels(
                "world/gt/cameras/center_labels",
                gt_cams,
                label_color=gt_label_color,
            )

        log_camera_axes(
            "world/gt/cameras/axes",
            gt_cams,
            axis_size=cam_axis_size,
            radius=axis_radius,
            axis_colors=gt_axis_colors,
        )

    if pred_cams:
        if show_center_labels:
            log_camera_center_labels(
                "world/pred/cameras/center_labels",
                pred_cams,
                label_color=pred_label_color,
            )

        log_camera_axes(
            "world/pred/cameras/axes",
            pred_cams,
            axis_size=cam_axis_size,
            radius=axis_radius,
            axis_colors=pred_axis_colors,
        )

    rr_disconnect_compat()

# ============================================================
# Sim(3) solving helpers
# ============================================================
@torch.no_grad()
def _robust_high_quantile(values: torch.Tensor, q: float = 0.995):
    if values.numel() == 0:
        return None
    q = min(max(float(q), 0.5), 1.0)
    try:
        return torch.quantile(values, q)
    except Exception:
        k = max(int(round((values.numel() - 1) * q)), 0)
        return values.flatten().kthvalue(k + 1).values

@torch.no_grad()
def _robust_low_quantile(values: torch.Tensor, q: float = 0.005):
    if values.numel() == 0:
        return None
    q = min(max(float(q), 0.0), 0.5)
    try:
        return torch.quantile(values, q)
    except Exception:
        k = max(int(round((values.numel() - 1) * q)), 0)
        return values.flatten().kthvalue(k + 1).values


@torch.no_grad()
def _local_median_mad_mask_2d(
    x2d: torch.Tensor,
    valid2d: torch.Tensor,
    kernel_size: int = 3,
    mad_scale: float = 6.0,
    min_neighbors: int = 4,
):
    """
    对单张 2D 图做局部 median + MAD 过滤。
    x2d: (H,W)
    valid2d: (H,W) bool
    return: refined_valid2d
    """
    assert x2d.ndim == 2 and valid2d.ndim == 2
    H, W = x2d.shape
    pad = kernel_size // 2

    x = x2d.clone()
    v = valid2d.clone().bool()

    # unfold
    x_pad = torch.nn.functional.pad(x[None, None], (pad, pad, pad, pad), mode="replicate")
    v_pad = torch.nn.functional.pad(v.float()[None, None], (pad, pad, pad, pad), mode="replicate")

    x_unf = torch.nn.functional.unfold(x_pad, kernel_size=kernel_size)   # (1, K, H*W)
    v_unf = torch.nn.functional.unfold(v_pad, kernel_size=kernel_size)   # (1, K, H*W)
    K = x_unf.shape[1]

    x_unf = x_unf.view(K, H * W).t()   # (H*W, K)
    v_unf = v_unf.view(K, H * W).t() > 0.5

    center_valid = v.reshape(-1)
    x_center = x.reshape(-1)

    inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    x_masked = torch.where(v_unf, x_unf, inf)

    x_sorted, _ = torch.sort(x_masked, dim=1)
    n_valid = v_unf.sum(dim=1)

    med = torch.empty((H * W,), device=x.device, dtype=x.dtype)
    med.fill_(float("nan"))

    good = n_valid >= min_neighbors
    if good.any():
        mid_idx = torch.clamp((n_valid[good] - 1) // 2, min=0)
        med_good = x_sorted[good, :]
        med[good] = med_good.gather(1, mid_idx[:, None]).squeeze(1)

    # MAD
    abs_dev = torch.abs(x_unf - med[:, None])
    abs_dev = torch.where(v_unf & torch.isfinite(med[:, None]), abs_dev, inf)

    abs_dev_sorted, _ = torch.sort(abs_dev, dim=1)
    mad = torch.empty((H * W,), device=x.device, dtype=x.dtype)
    mad.fill_(float("nan"))
    if good.any():
        mad_good = abs_dev_sorted[good, :]
        mad[good] = mad_good.gather(1, mid_idx[:, None]).squeeze(1)

    thr = mad_scale * 1.4826 * torch.clamp(mad, min=1e-6)
    ok = center_valid & good & torch.isfinite(med) & torch.isfinite(thr) & (torch.abs(x_center - med) <= thr)

    return ok.view(H, W)

@torch.no_grad()
def _mask_outliers_per_batch(
    pts_world: torch.Tensor,
    pts_cam: torch.Tensor,
    base_mask: torch.Tensor,
    world_low_q: float = 0.002,
    world_high_q: float = 0.995,
    depth_low_q: float = 0.01,
    depth_high_q: float = 0.995,
    min_depth: float = 1e-6,
    local_kernel: int = 3,
    local_mad_scale_depth: float = 5.0,
    local_mad_scale_range: float = 5.0,
    global_mad_scale_depth: float = 6.0,
    global_mad_scale_range: float = 6.0,
):
    """
    更稳健的离群点过滤：
    1) 去掉非有限值 / 非正深度
    2) 深度和空间距离做双边分位裁剪（尤其裁掉异常近点）
    3) 对 z-depth 和 point-range 做局部 median+MAD 过滤
    4) 再做一次全局 MAD 过滤
    """
    mask = base_mask.clone().bool()
    B = pts_world.shape[0]

    for b in range(B):
        m = mask[b]
        if m.sum() == 0:
            continue

        pw = pts_world[b]           # (H,W,3)
        pc = pts_cam[b]             # (H,W,3)

        finite = torch.isfinite(pw).all(dim=-1) & torch.isfinite(pc).all(dim=-1)
        z = pc[..., 2]
        r_world = torch.linalg.norm(pw, dim=-1)

        m = m & finite & torch.isfinite(z) & torch.isfinite(r_world) & (z > min_depth)
        if m.sum() < 16:
            mask[b] = m
            continue

        # ---------- 1) 双边分位裁剪 ----------
        z_valid = z[m]
        r_valid = r_world[m]

        z_lo = _robust_low_quantile(z_valid, q=depth_low_q)
        z_hi = _robust_high_quantile(z_valid, q=depth_high_q)
        r_lo = _robust_low_quantile(r_valid, q=world_low_q)
        r_hi = _robust_high_quantile(r_valid, q=world_high_q)

        if z_lo is not None and torch.isfinite(z_lo):
            m = m & (z >= z_lo)
        if z_hi is not None and torch.isfinite(z_hi):
            m = m & (z <= z_hi)
        if r_lo is not None and torch.isfinite(r_lo):
            m = m & (r_world >= r_lo)
        if r_hi is not None and torch.isfinite(r_hi):
            m = m & (r_world <= r_hi)

        if m.sum() < 16:
            mask[b] = m
            continue

        # ---------- 2) 局部一致性过滤 ----------
        # 对 log-depth / log-range 做局部检测更稳
        log_z = torch.zeros_like(z)
        log_r = torch.zeros_like(r_world)
        log_z[m] = torch.log(torch.clamp(z[m], min=1e-6))
        log_r[m] = torch.log(torch.clamp(r_world[m], min=1e-6))

        m_local_z = _local_median_mad_mask_2d(
            log_z, m, kernel_size=local_kernel, mad_scale=local_mad_scale_depth, min_neighbors=4
        )
        m_local_r = _local_median_mad_mask_2d(
            log_r, m, kernel_size=local_kernel, mad_scale=local_mad_scale_range, min_neighbors=4
        )

        m = m & m_local_z & m_local_r
        if m.sum() < 16:
            mask[b] = m
            continue

        # ---------- 3) 全局 MAD 补充过滤 ----------
        z_valid = log_z[m]
        r_valid = log_r[m]

        z_med = torch.median(z_valid)
        z_mad = torch.median(torch.abs(z_valid - z_med))
        z_thr = global_mad_scale_depth * 1.4826 * torch.clamp(z_mad, min=1e-6)

        r_med = torch.median(r_valid)
        r_mad = torch.median(torch.abs(r_valid - r_med))
        r_thr = global_mad_scale_range * 1.4826 * torch.clamp(r_mad, min=1e-6)

        m = m & (torch.abs(log_z - z_med) <= z_thr) & (torch.abs(log_r - r_med) <= r_thr)

        mask[b] = m

    return mask


@torch.no_grad()
def _sample_valid_corr_from_maps(gt_map, pr_map, valid_mask, max_samples=4096):
    """
    Sample correspondences from valid pixels of a single view.

    Args:
        gt_map: (H, W, 3)
        pr_map: (H, W, 3)
        valid_mask: (H, W)
    Returns:
        pr_corr: (N, 3) or None
        gt_corr: (N, 3) or None
    """
    idx = torch.nonzero(valid_mask, as_tuple=False)
    if idx.shape[0] == 0:
        return None, None

    if idx.shape[0] > max_samples:
        sel = torch.randperm(idx.shape[0], device=idx.device)[:max_samples]
        idx = idx[sel]

    u = idx[:, 0]
    v = idx[:, 1]
    gt_corr = gt_map[u, v]
    pr_corr = pr_map[u, v]

    good = torch.isfinite(gt_corr).all(dim=1) & torch.isfinite(pr_corr).all(dim=1)
    if good.sum() == 0:
        return None, None

    return pr_corr[good], gt_corr[good]


@torch.no_grad()
def _estimate_scene_scale_from_corr(gt_corr: torch.Tensor):
    if gt_corr is None or gt_corr.shape[0] == 0:
        return 1.0
    s = torch.median(torch.linalg.norm(gt_corr, dim=1))
    if not torch.isfinite(s) or s <= 1e-8:
        s = torch.tensor(1.0, device=gt_corr.device, dtype=gt_corr.dtype)
    return float(s.item())

@torch.no_grad()
def solve_batch_global_sim3(
    gt_pts_list,
    pr_pts_list,
    valid_masks,
    gt_pose_trans_list,
    pr_pose_trans_list,
    max_samples_per_view=4096,
    cam_center_repeat=64,
    trim_ratio=0.9,
    iters=3,
    min_corr=10,
    min_inlier_ratio=0.2,
    max_median_residual_ratio=0.5,
    max_scale_ratio=2.0,
):
    """
    Solve one robust pred->gt Sim(3) for each sample in the batch.

    Alignment form:
        X_gt = s * R * X_pred + t

    Returns:
        sim3_scale: (B,)
        sim3_rot: (B,3,3)
        sim3_trans: (B,3)
        sim3_valid: (B,) bool
        sim3_num_corr: (B,) long
        sim3_median_residual: (B,)
        sim3_inlier_ratio: (B,)
    """
    device = gt_pts_list[0].device
    dtype = gt_pts_list[0].dtype
    batch_size = gt_pts_list[0].shape[0]
    n_views = len(gt_pts_list)

    sim3_scale = torch.ones(batch_size, device=device, dtype=dtype)
    sim3_rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    sim3_trans = torch.zeros(batch_size, 3, device=device, dtype=dtype)

    sim3_valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
    sim3_num_corr = torch.zeros(batch_size, device=device, dtype=torch.long)
    sim3_median_residual = torch.full((batch_size,), float("nan"), device=device, dtype=dtype)
    sim3_inlier_ratio = torch.full((batch_size,), float("nan"), device=device, dtype=dtype)

    for b in range(batch_size):
        pr_corr_all = []
        gt_corr_all = []

        for v in range(n_views):
            pr_corr, gt_corr = _sample_valid_corr_from_maps(
                gt_pts_list[v][b],
                pr_pts_list[v][b],
                valid_masks[v][b],
                max_samples=max_samples_per_view,
            )
            if pr_corr is not None:
                pr_corr_all.append(pr_corr)
                gt_corr_all.append(gt_corr)

            # pr_c = pr_pose_trans_list[v][b].reshape(1, 3)
            # gt_c = gt_pose_trans_list[v][b].reshape(1, 3)
            # if torch.isfinite(pr_c).all() and torch.isfinite(gt_c).all():
            #     pr_corr_all.append(pr_c.repeat(cam_center_repeat, 1))
            #     gt_corr_all.append(gt_c.repeat(cam_center_repeat, 1))

        if len(pr_corr_all) == 0:
            continue

        pr_corr_all = torch.cat(pr_corr_all, dim=0)
        gt_corr_all = torch.cat(gt_corr_all, dim=0)

        sim3_num_corr[b] = pr_corr_all.shape[0]
        if pr_corr_all.shape[0] < min_corr:
            continue

        s, R, t = sim3_from_correspondences_robust(
            pr_corr_all,
            gt_corr_all,
            trim_ratio=trim_ratio,
            iters=iters,
        )

        if not (torch.isfinite(s) and torch.isfinite(R).all() and torch.isfinite(t).all()):
            continue

        pr_al = (s * (R @ pr_corr_all.t())).t() + t[None, :]
        resid = torch.linalg.norm(pr_al - gt_corr_all, dim=1)

        if resid.numel() == 0 or not torch.isfinite(resid).any():
            continue

        med_resid = torch.median(resid)
        scene_scale = _estimate_scene_scale_from_corr(gt_corr_all)
        inlier_thr = max(0.1 * scene_scale, 1e-3)
        inlier_ratio = (resid < inlier_thr).float().mean()

        sim3_median_residual[b] = med_resid
        sim3_inlier_ratio[b] = inlier_ratio

        gt_scale = torch.median(torch.linalg.norm(gt_corr_all, dim=1))
        pr_al_scale = torch.median(torch.linalg.norm(pr_al, dim=1))

        scale_ratio = torch.tensor(float("nan"), device=device, dtype=dtype)
        if torch.isfinite(gt_scale) and torch.isfinite(pr_al_scale) and gt_scale > 1e-8 and pr_al_scale > 1e-8:
            ratio1 = pr_al_scale / gt_scale
            ratio2 = gt_scale / pr_al_scale
            scale_ratio = torch.maximum(ratio1, ratio2)

        valid = (
            torch.isfinite(s)
            and (pr_corr_all.shape[0] >= min_corr)
            and torch.isfinite(med_resid)
            and (float(med_resid.item()) <= max_median_residual_ratio * scene_scale)
            and torch.isfinite(inlier_ratio)
            and (float(inlier_ratio.item()) >= min_inlier_ratio)
            and torch.isfinite(scale_ratio)
            and (float(scale_ratio.item()) <= max_scale_ratio)
        )

        if valid:
            sim3_scale[b] = s
            sim3_rot[b] = R
            sim3_trans[b] = t
            sim3_valid[b] = True

    return (
        sim3_scale,
        sim3_rot,
        sim3_trans,
        sim3_valid,
        sim3_num_corr,
        sim3_median_residual,
        sim3_inlier_ratio,
    )


@torch.no_grad()
def solve_batch_pose_sim3(
    gt_pose_trans_list,
    pr_pose_trans_list,
    trim_ratio=1.0,
    iters=1,
    min_corr=3,
    min_inlier_ratio=0.2,
    max_median_residual_ratio=0.5,
    max_scale_ratio=5.0,
):
    """
    Solve one pred->gt Sim(3) from matched camera centers.

    Alignment form:
        C_gt = s * R * C_pred + t

    Args:
        gt_pose_trans_list: list of (B,3), GT camera centers/translations in view0 frame
        pr_pose_trans_list: list of (B,3), predicted camera centers/translations in pred view0 frame

    Returns:
        sim3_scale: (B,)
        sim3_rot: (B,3,3)
        sim3_trans: (B,3)
        sim3_valid: (B,) bool
        sim3_num_corr: (B,) long
        sim3_median_residual: (B,)
        sim3_inlier_ratio: (B,)
    """
    device = gt_pose_trans_list[0].device
    dtype = gt_pose_trans_list[0].dtype
    batch_size = gt_pose_trans_list[0].shape[0]
    n_views = len(gt_pose_trans_list)

    sim3_scale = torch.ones(batch_size, device=device, dtype=dtype)
    sim3_rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    sim3_trans = torch.zeros(batch_size, 3, device=device, dtype=dtype)

    sim3_valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
    sim3_num_corr = torch.zeros(batch_size, device=device, dtype=torch.long)
    sim3_median_residual = torch.full((batch_size,), float("nan"), device=device, dtype=dtype)
    sim3_inlier_ratio = torch.full((batch_size,), float("nan"), device=device, dtype=dtype)

    for b in range(batch_size):
        pr_corr_all = []
        gt_corr_all = []

        for v in range(n_views):
            pr_c = pr_pose_trans_list[v][b].reshape(1, 3)
            gt_c = gt_pose_trans_list[v][b].reshape(1, 3)

            good = (
                torch.isfinite(pr_c).all()
                and torch.isfinite(gt_c).all()
            )
            if good:
                pr_corr_all.append(pr_c)
                gt_corr_all.append(gt_c)

        if len(pr_corr_all) == 0:
            continue

        pr_corr_all = torch.cat(pr_corr_all, dim=0)
        gt_corr_all = torch.cat(gt_corr_all, dim=0)

        sim3_num_corr[b] = pr_corr_all.shape[0]
        if pr_corr_all.shape[0] < min_corr:
            continue

        s, R, t = sim3_from_correspondences_robust(
            pr_corr_all,
            gt_corr_all,
            trim_ratio=trim_ratio,
            iters=iters,
        )

        if not (torch.isfinite(s) and torch.isfinite(R).all() and torch.isfinite(t).all()):
            continue

        pr_al = (s * (R @ pr_corr_all.t())).t() + t[None, :]
        resid = torch.linalg.norm(pr_al - gt_corr_all, dim=1)

        if resid.numel() == 0 or not torch.isfinite(resid).any():
            continue

        med_resid = torch.median(resid)

        gt_centered = gt_corr_all - gt_corr_all.mean(dim=0, keepdim=True)
        scene_scale = torch.median(torch.linalg.norm(gt_centered, dim=1))
        if not torch.isfinite(scene_scale) or scene_scale <= 1e-8:
            scene_scale = torch.median(torch.linalg.norm(gt_corr_all, dim=1))
        if not torch.isfinite(scene_scale) or scene_scale <= 1e-8:
            scene_scale = torch.tensor(1.0, device=device, dtype=dtype)

        inlier_thr = max(0.1 * float(scene_scale.item()), 1e-3)
        inlier_ratio = (resid < inlier_thr).float().mean()

        sim3_median_residual[b] = med_resid
        sim3_inlier_ratio[b] = inlier_ratio

        gt_scale = torch.median(torch.linalg.norm(gt_centered, dim=1))
        pr_al_centered = pr_al - pr_al.mean(dim=0, keepdim=True)
        pr_al_scale = torch.median(torch.linalg.norm(pr_al_centered, dim=1))

        scale_ratio = torch.tensor(float("nan"), device=device, dtype=dtype)
        if torch.isfinite(gt_scale) and torch.isfinite(pr_al_scale) and gt_scale > 1e-8 and pr_al_scale > 1e-8:
            ratio1 = pr_al_scale / gt_scale
            ratio2 = gt_scale / pr_al_scale
            scale_ratio = torch.maximum(ratio1, ratio2)

        valid = (
            torch.isfinite(s)
            and (pr_corr_all.shape[0] >= min_corr)
            and torch.isfinite(med_resid)
            and (float(med_resid.item()) <= max_median_residual_ratio * float(scene_scale.item()))
            and torch.isfinite(inlier_ratio)
            and (float(inlier_ratio.item()) >= min_inlier_ratio)
            and torch.isfinite(scale_ratio)
            and (float(scale_ratio.item()) <= max_scale_ratio)
        )

        if valid:
            sim3_scale[b] = s
            sim3_rot[b] = R
            sim3_trans[b] = t
            sim3_valid[b] = True

    return (
        sim3_scale,
        sim3_rot,
        sim3_trans,
        sim3_valid,
        sim3_num_corr,
        sim3_median_residual,
        sim3_inlier_ratio,
    )


@torch.no_grad()
def solve_batch_scale_yaw_translation(
    gt_corr_batches,
    pr_corr_batches,
    device,
    dtype,
    min_corr=2,
):
    """Batch wrapper around the spatial pipeline's constrained solver."""
    batch_size = len(gt_corr_batches)
    scales = torch.ones(batch_size, device=device, dtype=dtype)
    rotations = torch.eye(3, device=device, dtype=dtype).repeat(batch_size, 1, 1)
    translations = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
    num_corr = torch.zeros(batch_size, device=device, dtype=torch.long)
    median_residual = torch.full((batch_size,), float("nan"), device=device, dtype=dtype)
    inlier_ratio = torch.full((batch_size,), float("nan"), device=device, dtype=dtype)

    for b, (gt_corr, pr_corr) in enumerate(zip(gt_corr_batches, pr_corr_batches)):
        if gt_corr is None or pr_corr is None:
            continue
        finite = torch.isfinite(gt_corr).all(dim=-1) & torch.isfinite(pr_corr).all(dim=-1)
        gt_corr = gt_corr[finite]
        pr_corr = pr_corr[finite]
        num_corr[b] = gt_corr.shape[0]
        if gt_corr.shape[0] < min_corr:
            continue

        scale, rotation, translation, solved, _meta = (
            estimate_scale_yaw_translation_from_correspondences(
                src=pr_corr.detach().cpu().numpy(),
                dst=gt_corr.detach().cpu().numpy(),
                seed=b,
            )
        )
        if not solved:
            continue
        s = torch.as_tensor(scale, device=device, dtype=dtype)
        R = torch.as_tensor(rotation, device=device, dtype=dtype)
        t = torch.as_tensor(translation, device=device, dtype=dtype)
        pr_aligned = s * torch.einsum("ij,nj->ni", R, pr_corr) + t
        residual = torch.linalg.norm(pr_aligned - gt_corr, dim=-1)
        scene_scale = _estimate_scene_scale_from_corr(gt_corr)
        threshold = max(0.1 * scene_scale, 1e-3)

        scales[b], rotations[b], translations[b] = s, R, t
        valid[b] = True
        median_residual[b] = torch.median(residual)
        inlier_ratio[b] = (residual < threshold).float().mean()

    return (
        scales,
        rotations,
        translations,
        valid,
        num_corr,
        median_residual,
        inlier_ratio,
    )


@torch.no_grad()
def solve_batch_sim3_from_correspondence_batches(
    gt_corr_batches, pr_corr_batches, device, dtype, min_corr=3
):
    """Solve full Sim(3) using only the explicitly supplied correspondences."""
    batch_size = len(gt_corr_batches)
    scales = torch.ones(batch_size, device=device, dtype=dtype)
    rotations = torch.eye(3, device=device, dtype=dtype).repeat(batch_size, 1, 1)
    translations = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
    num_corr = torch.zeros(batch_size, device=device, dtype=torch.long)
    median_residual = torch.full((batch_size,), float("nan"), device=device, dtype=dtype)
    inlier_ratio = torch.full((batch_size,), float("nan"), device=device, dtype=dtype)
    for b, (gt_corr, pr_corr) in enumerate(zip(gt_corr_batches, pr_corr_batches)):
        finite = torch.isfinite(gt_corr).all(dim=-1) & torch.isfinite(pr_corr).all(dim=-1)
        gt_corr, pr_corr = gt_corr[finite], pr_corr[finite]
        num_corr[b] = gt_corr.shape[0]
        if gt_corr.shape[0] < min_corr:
            continue
        s, R, t = umeyama_sim3_torch(pr_corr, gt_corr)
        if not (torch.isfinite(s) and torch.isfinite(R).all() and torch.isfinite(t).all()):
            continue
        pr_aligned = s * torch.einsum("ij,nj->ni", R, pr_corr) + t
        residual = torch.linalg.norm(pr_aligned - gt_corr, dim=-1)
        scene_scale = _estimate_scene_scale_from_corr(gt_corr)
        threshold = max(0.1 * scene_scale, 1e-3)
        scales[b], rotations[b], translations[b] = s, R, t
        valid[b] = True
        median_residual[b] = torch.median(residual)
        inlier_ratio[b] = (residual < threshold).float().mean()
    return scales, rotations, translations, valid, num_corr, median_residual, inlier_ratio


@torch.no_grad()
def solve_batch_points_yaw_similarity(gt_pts_list, pr_pts_list, valid_masks, max_samples_per_view=4096):
    batch_size = gt_pts_list[0].shape[0]
    gt_batches, pr_batches = [], []
    for b in range(batch_size):
        gt_parts, pr_parts = [], []
        for v in range(len(gt_pts_list)):
            pr_corr, gt_corr = _sample_valid_corr_from_maps(
                gt_pts_list[v][b], pr_pts_list[v][b], valid_masks[v][b],
                max_samples=max_samples_per_view,
            )
            if pr_corr is not None:
                gt_parts.append(gt_corr)
                pr_parts.append(pr_corr)
        gt_batches.append(torch.cat(gt_parts) if gt_parts else None)
        pr_batches.append(torch.cat(pr_parts) if pr_parts else None)
    return solve_batch_scale_yaw_translation(
        gt_batches, pr_batches, gt_pts_list[0].device, gt_pts_list[0].dtype, min_corr=10
    )


@torch.no_grad()
def solve_batch_pose_yaw_similarity(gt_pose_trans_list, pr_pose_trans_list):
    gt_stacked = torch.stack(gt_pose_trans_list, dim=1)
    pr_stacked = torch.stack(pr_pose_trans_list, dim=1)
    return solve_batch_scale_yaw_translation(
        list(gt_stacked), list(pr_stacked), gt_stacked.device, gt_stacked.dtype, min_corr=2
    )


# ============================================================
# Main info assembly for metric computation
# ============================================================

def get_all_info_for_metric_computation(batch, preds, norm_mode="avg_dis", alignment_mode="points"):
    """
    Returns:
      gt_info,
      pr_info,
      valid_masks,
      gt_info_abs,
      pr_info_abs_aligned,
      scale_factors

    Notes:
      - Relative metrics are computed from normalized pointmaps / normalized poses.
      - Absolute metrics are computed after global Sim(3) alignment.
      - pose_points estimates independent Sim(3)s for geometry and camera poses.
      - pose_points_yaw independently estimates scale + Z-yaw + translation.
      - Absolute pose metrics must not perform any additional alignment afterwards.
      - Per-view depth absolute metrics are later computed with sequence-level scale-only alignment.
    """
    # Normalize alignment_mode
    alignment_mode = str(alignment_mode).lower().strip()
    if alignment_mode in {"point", "dense", "dense_points"}:
        alignment_mode = "points"
    elif alignment_mode in {"camera", "camera_center", "camera_centers", "pose_sim3"}:
        alignment_mode = "pose"
    elif alignment_mode in {"points_pose", "separate", "separate_pose_points"}:
        alignment_mode = "pose_points"
    elif alignment_mode in {
        "pose_points1", "pose_points_yaw", "yaw", "yaw_translation",
        "scale_yaw_translation",
    }:
        alignment_mode = "pose_points_yaw"

    if alignment_mode not in {"points", "pose", "pose_points", "pose_points_yaw"}:
        raise ValueError(
            f"Unsupported alignment_mode={alignment_mode}. "
            "Expected points, pose, pose_points, or pose_points_yaw. "
            "Use benchmarking/dense_n_view/benchmark_absolute_world.py for alignment_mode=none."
        )

    n_views = len(batch)
    batch_size = batch[0]["camera_pose"].shape[0]
    device = preds[0]["cam_quats"].device
    dtype = preds[0]["cam_quats"].dtype

    # --------------------------------------------------------
    # 1) Collect GT / Pred absolute quantities in view0 world
    # --------------------------------------------------------
    in_camera0 = inv(batch[0]["camera_pose"])

    no_norm_gt_pts = []
    no_norm_gt_pts3d_cam = []
    no_norm_gt_pose_trans = []
    gt_ray_directions = []
    gt_intrinsics = []
    gt_pose_rot = []
    valid_masks = []

    pred_camera0 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    pred_camera0_rot = quaternion_to_rotation_matrix(preds[0]["cam_quats"].clone())
    pred_camera0[..., :3, :3] = pred_camera0_rot
    pred_camera0[..., :3, 3] = preds[0]["cam_trans"].clone()
    pred_in_camera0 = inv(pred_camera0)

    no_norm_pr_pts = []
    no_norm_pr_pts3d_cam = []
    no_norm_pr_pose_trans = []
    pr_ray_directions = []
    pr_intrinsics = []
    pr_pose_rot = []

    for i in range(n_views):
        # ---------------- GT ----------------
        gt_pts_i = geotrf(in_camera0, batch[i]["pts3d"])
        no_norm_gt_pts.append(gt_pts_i)
        no_norm_gt_pts3d_cam.append(batch[i]["pts3d_cam"])
        valid_masks.append(batch[i]["valid_mask"].clone())
        gt_ray_directions.append(batch[i]["ray_directions_cam"])
        gt_intrinsics.append(batch[i]["camera_intrinsics"])

        if i == 0:
            gt_rot_i = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)  # 10,3,3
            gt_trans_i = torch.zeros(batch_size, 3, device=device, dtype=dtype)  # 10,3
        else:
            gt_pose_quats_in_view0, gt_trans_i = transform_pose_using_quats_and_trans_2_to_1(
                batch[0]["camera_pose_quats"],
                batch[0]["camera_pose_trans"],
                batch[i]["camera_pose_quats"],
                batch[i]["camera_pose_trans"],
            )
            gt_rot_i = quaternion_to_rotation_matrix(gt_pose_quats_in_view0)

        gt_pose_rot.append(gt_rot_i)
        no_norm_gt_pose_trans.append(gt_trans_i)

        # ---------------- Pred ----------------
        pr_pose_quats_in_view0, pr_trans_i = transform_pose_using_quats_and_trans_2_to_1(
            preds[0]["cam_quats"], preds[0]["cam_trans"], preds[i]["cam_quats"], preds[i]["cam_trans"]
        )
        pr_rot_i = quaternion_to_rotation_matrix(pr_pose_quats_in_view0)
        pr_pts_i = geotrf(pred_in_camera0, preds[i]["pts3d"])

        if "metric_scaling_factor" in preds[i]:
            sf = preds[i]["metric_scaling_factor"].unsqueeze(-1).unsqueeze(-1)
            pr_pts_i = pr_pts_i / sf
            pr_pts3d_cam_i = preds[i]["pts3d_cam"] / sf
            pr_trans_i = pr_trans_i / preds[i]["metric_scaling_factor"]
        else:
            pr_pts3d_cam_i = preds[i]["pts3d_cam"]
        
        if "intrinsics" in preds[i]:
            pr_intr_i = preds[i]['intrinsics']
        else:
            pr_intr_i = recover_pinhole_intrinsics_from_ray_directions(preds[i]["ray_directions"])

        no_norm_pr_pts.append(pr_pts_i)
        no_norm_pr_pts3d_cam.append(pr_pts3d_cam_i)
        no_norm_pr_pose_trans.append(pr_trans_i)
        pr_ray_directions.append(preds[i]["ray_directions"])
        pr_intrinsics.append(pr_intr_i)
        pr_pose_rot.append(pr_rot_i)

    # --------------------------------------------------------
    # 2) Pre-filter obvious outliers before solving global Sim(3)
    # --------------------------------------------------------
    valid_masks = [
        valid_masks[i]
        & _mask_outliers_per_batch(no_norm_gt_pts[i], no_norm_gt_pts3d_cam[i], valid_masks[i])
        & _mask_outliers_per_batch(no_norm_pr_pts[i], no_norm_pr_pts3d_cam[i], valid_masks[i])
        for i in range(n_views)
    ]

    # --------------------------------------------------------
    # 3) Solve one global pred->gt Sim(3)
    # --------------------------------------------------------
    points_sim3 = None
    pose_sim3 = None

    if alignment_mode in {"points", "pose_points", "pose_points_yaw"}:
        if alignment_mode == "pose_points_yaw":
            points_sim3 = solve_batch_points_yaw_similarity(
                no_norm_gt_pts, no_norm_pr_pts, valid_masks, max_samples_per_view=4096
            )
        else:
            points_sim3 = solve_batch_global_sim3(
                gt_pts_list=no_norm_gt_pts,
                pr_pts_list=no_norm_pr_pts,
                valid_masks=valid_masks,
                gt_pose_trans_list=no_norm_gt_pose_trans,
                pr_pose_trans_list=no_norm_pr_pose_trans,
                max_samples_per_view=4096,
            )

    if alignment_mode in {"pose", "pose_points", "pose_points_yaw"}:
        if alignment_mode == "pose_points_yaw":
            pose_sim3 = solve_batch_pose_yaw_similarity(
                no_norm_gt_pose_trans, no_norm_pr_pose_trans
            )
        else:
            pose_sim3 = solve_batch_pose_sim3(
                gt_pose_trans_list=no_norm_gt_pose_trans,
                pr_pose_trans_list=no_norm_pr_pose_trans,
                trim_ratio=1.0,
                iters=1,
                min_corr=3,
            )

    # Single-alignment modes retain their original behavior. In pose_points,
    # geometry/depth and camera poses deliberately use different transforms.
    geometry_sim3 = points_sim3 if points_sim3 is not None else pose_sim3
    camera_sim3 = pose_sim3 if pose_sim3 is not None else points_sim3
    (
        geometry_scale,
        geometry_rot,
        geometry_trans,
        geometry_valid,
        geometry_num_corr,
        geometry_median_residual,
        geometry_inlier_ratio,
    ) = geometry_sim3
    (
        pose_scale,
        pose_rot,
        pose_trans,
        pose_valid,
        pose_num_corr,
        pose_median_residual,
        pose_inlier_ratio,
    ) = camera_sim3

    # --------------------------------------------------------
    # 3) Align pred absolute quantities to GT with this Sim(3)
    # --------------------------------------------------------
    aligned_pr_pts = []
    aligned_pr_pts3d_cam = []
    aligned_pr_pose_trans = []
    aligned_pr_pose_rot = []

    for i in range(n_views):
        pr_pts_i = no_norm_pr_pts[i]  # (B,H,W,3)
        s_map = geometry_scale[:, None, None, None]
        t_map = geometry_trans[:, None, None, :]

        pr_pts_i_al = s_map * torch.einsum("bij,bhwj->bhwi", geometry_rot, pr_pts_i) + t_map
        aligned_pr_pts.append(pr_pts_i_al)

        # camera-frame depth points: only scale is physically meaningful here
        pr_pts3d_cam_i_al = no_norm_pr_pts3d_cam[i] * s_map
        aligned_pr_pts3d_cam.append(pr_pts3d_cam_i_al)

        pr_rot_i_al = torch.einsum("bij,bjk->bik", pose_rot, pr_pose_rot[i])
        pr_trans_i_al = pose_scale[:, None] * torch.einsum(
            "bij,bj->bi", pose_rot, no_norm_pr_pose_trans[i]
        ) + pose_trans

        aligned_pr_pose_rot.append(pr_rot_i_al)
        aligned_pr_pose_trans.append(pr_trans_i_al)

    # --------------------------------------------------------
    # 5) Normalize GT and aligned pred separately for relative metrics
    # --------------------------------------------------------
    gt_norm_out = normalize_multiple_pointclouds(no_norm_gt_pts, valid_masks, norm_mode, ret_factor=True)
    gt_pts_norm = gt_norm_out[:-1]
    gt_norm_factor = gt_norm_out[-1]

    pr_norm_out = normalize_multiple_pointclouds(aligned_pr_pts, valid_masks, norm_mode, ret_factor=True)
    pr_pts_norm = pr_norm_out[:-1]
    pr_norm_factor = pr_norm_out[-1]

    gt_pts = []
    gt_pts3d_cam = []
    gt_pose_trans = []
    pr_pts = []
    pr_pts3d_cam = []
    pr_pose_trans = []

    for i in range(n_views):
        gt_pts.append(gt_pts_norm[i].cpu())
        gt_pts3d_cam.append((no_norm_gt_pts3d_cam[i] / gt_norm_factor).cpu())
        gt_pose_trans.append((no_norm_gt_pose_trans[i] / gt_norm_factor[:, :, 0, 0]).cpu())

        pr_pts.append(pr_pts_norm[i].cpu())
        pr_pts3d_cam.append((aligned_pr_pts3d_cam[i] / pr_norm_factor).cpu())
        pose_norm_factor = (
            gt_norm_factor
            if alignment_mode in {"pose_points", "pose_points_yaw"}
            else pr_norm_factor
        )
        pr_pose_trans.append((aligned_pr_pose_trans[i] / pose_norm_factor[:, :, 0, 0]).cpu())

        valid_masks[i] = valid_masks[i].cpu()

    # --------------------------------------------------------
    # 5) Build pose matrices
    # --------------------------------------------------------
    gt_poses = []
    pr_poses = []
    gt_poses_abs = []
    pr_poses_abs = []
    pr_poses_abs_unaligned = []

    for i in range(n_views):
        gt_pose_curr = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        gt_pose_curr[..., :3, :3] = gt_pose_rot[i]
        gt_pose_curr[..., :3, 3] = gt_pose_trans[i].to(device)
        gt_poses.append(gt_pose_curr.cpu())

        pr_pose_curr = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        pr_pose_curr[..., :3, :3] = aligned_pr_pose_rot[i]
        pr_pose_curr[..., :3, 3] = pr_pose_trans[i].to(device)
        pr_poses.append(pr_pose_curr.cpu())

        gt_pose_abs_curr = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        gt_pose_abs_curr[..., :3, :3] = gt_pose_rot[i]
        gt_pose_abs_curr[..., :3, 3] = no_norm_gt_pose_trans[i]
        gt_poses_abs.append(gt_pose_abs_curr.cpu())

        pr_pose_abs_curr = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        pr_pose_abs_curr[..., :3, :3] = aligned_pr_pose_rot[i]
        pr_pose_abs_curr[..., :3, 3] = aligned_pr_pose_trans[i]
        pr_poses_abs.append(pr_pose_abs_curr.cpu())

        # Raw predicted camera trajectory before the dense-geometry Sim(3) alignment.
        # This is used only by the optional separately aligned Pose ATE metric.
        pr_pose_abs_unaligned_curr = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        pr_pose_abs_unaligned_curr[..., :3, :3] = pr_pose_rot[i]
        pr_pose_abs_unaligned_curr[..., :3, 3] = no_norm_pr_pose_trans[i]
        pr_poses_abs_unaligned.append(pr_pose_abs_unaligned_curr.cpu())

    gt_z_depths = [x[..., 2:].cpu() for x in gt_pts3d_cam]
    pr_z_depths = [x[..., 2:].cpu() for x in pr_pts3d_cam]
    gt_z_depths_abs = [x[..., 2:].cpu() for x in no_norm_gt_pts3d_cam]
    pr_z_depths_abs = [x[..., 2:].cpu() for x in aligned_pr_pts3d_cam]

    gt_info = {
        "ray_directions": gt_ray_directions,
        "intrinsics": gt_intrinsics,
        "z_depths": gt_z_depths,
        "poses": gt_poses,
        "pts3d": gt_pts,
        "metric_scale": gt_norm_factor[:, 0, 0, 0].cpu(),
    }

    pr_info = {
        "ray_directions": pr_ray_directions,
        "intrinsics": pr_intrinsics,
        "z_depths": pr_z_depths,
        "poses": pr_poses,
        "pts3d": pr_pts,
        "metric_scale": pr_norm_factor[:, 0, 0, 0].cpu(),
    }

    gt_info_abs = {
        "ray_directions": gt_ray_directions,
        "intrinsics": gt_intrinsics,
        "poses": gt_poses_abs,
        "pts3d": [x.cpu() for x in no_norm_gt_pts],
        "z_depths": gt_z_depths_abs,
    }

    pr_info_abs_aligned = {
        "ray_directions": pr_ray_directions,
        "intrinsics": pr_intrinsics,
        "poses": pr_poses_abs,
        "poses_unaligned": pr_poses_abs_unaligned,
        "pts3d": [x.cpu() for x in aligned_pr_pts],
        "z_depths": pr_z_depths_abs,
    }

    scale_factors = {
        "alignment_mode": alignment_mode,
        # Legacy keys describe geometry alignment. This is unchanged for the
        # existing points/pose modes and keeps downstream consumers compatible.
        "pr_to_gt_scale": geometry_scale.detach().cpu(),
        "sim3_rot": geometry_rot.detach().cpu(),
        "sim3_trans": geometry_trans.detach().cpu(),
        "sim3_valid": geometry_valid.detach().cpu(),
        "sim3_num_corr": geometry_num_corr.detach().cpu(),
        "sim3_median_residual": geometry_median_residual.detach().cpu(),
        "sim3_inlier_ratio": geometry_inlier_ratio.detach().cpu(),
        "points_sim3_scale": geometry_scale.detach().cpu(),
        "points_sim3_rot": geometry_rot.detach().cpu(),
        "points_sim3_trans": geometry_trans.detach().cpu(),
        "points_sim3_valid": geometry_valid.detach().cpu(),
        "points_sim3_num_corr": geometry_num_corr.detach().cpu(),
        "points_sim3_median_residual": geometry_median_residual.detach().cpu(),
        "points_sim3_inlier_ratio": geometry_inlier_ratio.detach().cpu(),
        "pose_sim3_scale": pose_scale.detach().cpu(),
        "pose_sim3_rot": pose_rot.detach().cpu(),
        "pose_sim3_trans": pose_trans.detach().cpu(),
        "pose_sim3_valid": pose_valid.detach().cpu(),
        "pose_sim3_num_corr": pose_num_corr.detach().cpu(),
        "pose_sim3_median_residual": pose_median_residual.detach().cpu(),
        "pose_sim3_inlier_ratio": pose_inlier_ratio.detach().cpu(),
        "gt_norm_factor": gt_norm_factor.detach().cpu(),
        "pr_norm_factor": pr_norm_factor.detach().cpu(),
    }
    return gt_info, pr_info, valid_masks, gt_info_abs, pr_info_abs_aligned, scale_factors


# ============================================================
# Dataset / benchmark entry
# ============================================================

# def build_dataset(dataset, batch_size, num_workers):
#     print("Building data loader for dataset: ", dataset)
#     loader = get_test_data_loader(
#         dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_mem=True,
#         shuffle=False,
#         drop_last=False,
#     )
#     print("Dataset length: ", len(loader))
#     return loader

def build_dataset(dataset, batch_size, num_workers):
    print("Building data loader for dataset: ", dataset)

    # Support variable aspect ratio and/or variable num_views during benchmark.
    # BaseDataset._getitem_fn() requires tuple indices when self.num_views is a list,
    # otherwise it asserts self.num_views is int.
    use_many_feature_loader = False

    if hasattr(dataset, "num_views") and isinstance(dataset.num_views, (list, tuple)):
        use_many_feature_loader = True

    if hasattr(dataset, "_resolutions") and len(dataset._resolutions) > 1:
        use_many_feature_loader = True

    if use_many_feature_loader:
        print(
            f"[benchmark] Using get_test_many_ar_data_loader "
            f"(num_views={dataset.num_views}, num_resolutions={len(dataset._resolutions)})"
        )
        loader = get_test_many_ar_data_loader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_mem=True,
            drop_last=False,
        )
    else:
        print("[benchmark] Using get_test_data_loader")
        loader = get_test_data_loader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_mem=True,
            shuffle=False,
            drop_last=False,
        )

    print("Dataset length: ", len(loader))
    return loader


@torch.no_grad()
def benchmark(args):
    print("Output Directory: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = resolve_benchmark_device(args)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    amp_enabled = bool(args.amp) and device.type == "cuda"

    if amp_enabled:
        if args.amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif args.amp_dtype == "bf16":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    compute_separate_align_pose_ate = bool(_cfg_get(args, "compute_separate_align_pose_ate", False))
    if compute_separate_align_pose_ate and not bool(_cfg_get(args, "compute_abs_metrics", False)):
        print(
            "[benchmark] compute_separate_align_pose_ate=True but "
            "compute_abs_metrics=False; the separately aligned Pose ATE will not be computed."
        )

    dataset_cfgs = [
        dataset for dataset in args.dataset.test_dataset.split("+") if "(" in dataset
    ]
    dataset_name_to_cfg = {
        dataset.split("(")[0]: dataset for dataset in dataset_cfgs
    }

    per_dataset_results = {}
    datasets_to_run = {}
    requested_alignment = str(_cfg_get(args, "alignment_mode", "points")).lower().strip()
    protocol_path = os.path.join(args.output_dir, "alignment_protocol.json")
    cached_alignment = None
    if os.path.isfile(protocol_path):
        try:
            with open(protocol_path, "r", encoding="utf-8") as f:
                cached_alignment = str(json.load(f).get("alignment_mode", ""))
        except Exception:
            cached_alignment = None

    for benchmark_dataset_name, dataset_cfg in dataset_name_to_cfg.items():
        per_scene_json_path = os.path.join(
            args.output_dir, f"{benchmark_dataset_name}_per_scene_results.json"
        )
        avg_json_path = os.path.join(
            args.output_dir, f"{benchmark_dataset_name}_avg_across_all_scenes.json"
        )

        if (
            os.path.exists(per_scene_json_path)
            and os.path.exists(avg_json_path)
            and cached_alignment == requested_alignment
        ):
            with open(avg_json_path, "r", encoding="utf-8") as f:
                cached_avg = json.load(f)

            missing_requested_metrics = "camera_scale_rel_error" not in cached_avg
            if compute_separate_align_pose_ate:
                missing_requested_metrics = missing_requested_metrics or (
                    "abs_pose_ate_separate_aligned" not in cached_avg
                )

            if missing_requested_metrics:
                print(
                    f"Rerun dataset {benchmark_dataset_name}: cache exists but does not contain "
                    "the currently required scale/pose metrics."
                )
                datasets_to_run[benchmark_dataset_name] = dataset_cfg
            else:
                print(
                    f"Skip dataset {benchmark_dataset_name}: found existing result jsons, "
                    f"will reuse cached metrics from {avg_json_path}"
                )
                per_dataset_results[benchmark_dataset_name] = cached_avg
        else:
            datasets_to_run[benchmark_dataset_name] = dataset_cfg
            if os.path.exists(per_scene_json_path) or os.path.exists(avg_json_path):
                print(
                    f"Cache for dataset {benchmark_dataset_name} is incomplete, rerun this dataset. "
                    f"per_scene_exists={os.path.exists(per_scene_json_path)}, "
                    f"avg_exists={os.path.exists(avg_json_path)}"
                )

    model = None
    if len(datasets_to_run) > 0:
        model = init_model(args.model.model_str, args.model.model_config, torch_hub_force_reload=False)
        model.to(device)
        model.eval()
        
        if args.model.pretrained:
            ckpt = torch.load(args.model.pretrained, map_location="cpu", weights_only=False)
            print(model.load_state_dict(ckpt["model"], strict=False))
            del ckpt
        model.to(device)
        model.eval()

    for benchmark_dataset_name, dataset_cfg in datasets_to_run.items():
        data_loader = build_dataset(dataset_cfg, args.batch_size, args.dataset.num_workers)

        print("Benchmarking dataset: ", benchmark_dataset_name)
        data_loader.dataset.set_epoch(0)

        max_sets_per_scene = getattr(args, "save_n_fused_ply", 1)
        per_scene_saved = {scene: 0 for scene in data_loader.dataset.dataset.scenes}

        fused_out_dir = os.path.join(args.output_dir, f"{benchmark_dataset_name}_fused_ply")
        os.makedirs(fused_out_dir, exist_ok=True)

        per_scene_results = {}
        for dataset_scene in data_loader.dataset.dataset.scenes:
            per_scene_results[dataset_scene] = {
                # relative metrics
                "rel_pointmap_abs": [],
                "rel_pointmap_delta_1p03": [],

                "rel_pose_ate": [],
                "rel_pose_auc_5deg": [],

                "rel_depth_abs": [],
                "rel_depth_delta_1p03": [],

                "ray_dir_mean_angle_deg": [],

                # Scale from raw corresponding camera centers (no alignment).
                "camera_scale_gt": [],
                "camera_scale_pred": [],
                "camera_scale_ratio": [],
                "camera_scale_rel_error": [],
                "camera_scale_log_error": [],
                
                # absolute metrics
                "abs_pointmap_mae": [],
                "abs_pointmap_rmse": [],

                "abs_pose_ate": [],
                "abs_pose_auc_5deg": [],
                "abs_pose_rot_mae_deg": [],
                
                "abs_depth_mae_scale_aligned": [],
                "abs_depth_rmse_scale_aligned": [],
                "abs_depth_rel_scale_aligned": [],
                "abs_depth_delta1_scale_aligned": [],

                "abs_fused_pc_chamfer_l1": [],
                "abs_fused_pc_precision": [],
                "abs_fused_pc_recall": [],
                "abs_fused_pc_f1": [],

                # alignment diagnostics
                "sim3_scale": [],
                "sim3_valid": [],
                "sim3_num_corr": [],
                "sim3_median_residual": [],
                "sim3_inlier_ratio": [],
                "points_sim3_scale": [],
                "points_sim3_valid": [],
                "points_sim3_num_corr": [],
                "points_sim3_median_residual": [],
                "points_sim3_inlier_ratio": [],
                "pose_sim3_scale": [],
                "pose_sim3_valid": [],
                "pose_sim3_num_corr": [],
                "pose_sim3_median_residual": [],
                "pose_sim3_inlier_ratio": [],
            }
            if compute_separate_align_pose_ate:
                per_scene_results[dataset_scene]["abs_pose_ate_separate_aligned"] = []

        for batch in tqdm(data_loader):
            for view in batch:
                view["idx"] = view["idx"][2:]

            ignore_keys = {
                "depthmap",
                "dataset",
                "label",
                "instance",
                "idx",
                "true_shape",
                "rng",
                "data_norm_type",
            }
            for view in batch:
                for name in view.keys():
                    if name in ignore_keys:
                        continue
                    view[name] = view[name].to(device, non_blocking=True)

            # with torch.autocast("cuda", enabled=bool(args.amp), dtype=amp_dtype):
            with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                preds = model(batch)

            alignment_mode = _cfg_get(args, "alignment_mode", "points")

            gt_info, pr_info, valid_masks, gt_info_abs, pr_info_abs_aligned, scale_factors = (
                get_all_info_for_metric_computation(
                    batch,
                    preds,
                    alignment_mode=alignment_mode,
                )
            )

            batch_size = batch[0]["img"].shape[0]
            compute_abs = bool(args.get("compute_abs_metrics", False))
            for batch_idx in range(batch_size):
                scene = batch[0]["label"][batch_idx]
                k = per_scene_saved[scene]
                need_save_fused = (max_sets_per_scene > 0) and (k < max_sets_per_scene)

                metrics, fused_debug = compute_set_metrics(
                    batch_views=batch,
                    batch_idx=batch_idx,
                    gt_info=gt_info,
                    pr_info=pr_info,
                    valid_masks=valid_masks,
                    gt_info_abs=gt_info_abs,
                    pr_info_abs=pr_info_abs_aligned,
                    scale_factors=scale_factors,
                    device=device,
                    voxel=0.25,
                    icp_iters=0,
                    trim_ratio=0.8,
                    return_fused_debug=need_save_fused,
                    compute_abs_metrics=compute_abs,
                    compute_separate_align_pose_ate=compute_separate_align_pose_ate,
                )
                raw_gt_centers = torch.stack(
                    [view["camera_pose"][batch_idx, :3, 3] for view in batch]
                )
                raw_pred_centers = torch.stack(
                    [pred["cam_trans"][batch_idx] for pred in preds]
                )
                metrics.update(
                    camera_center_scale_metrics(raw_gt_centers, raw_pred_centers)
                )

                if fused_debug is not None and k < max_sets_per_scene:
                    safe_scene = str(scene).replace("/", "_")
                    base = f"{safe_scene}_set{str(k).zfill(3)}"
                    score_for_name = fused_debug.chamfer_l1 if np.isfinite(fused_debug.chamfer_l1) else -1.0
                    gt_path = os.path.join(
                        fused_out_dir, safe_scene, base + f"_GT_{score_for_name:.3f}.ply"
                    )
                    pr_path = os.path.join(
                        fused_out_dir, safe_scene, base + f"_Pred_{score_for_name:.3f}.ply"
                    )
                    meta_path = os.path.join(fused_out_dir, safe_scene, base + "_bundle.json")
                    rrd_path = os.path.join(fused_out_dir, safe_scene, base + "_bundle.rrd")

                    write_ply_xyzrgb(gt_path, fused_debug.gt_ds, fused_debug.gt_colors_ds)
                    write_ply_xyzrgb(pr_path, fused_debug.pr_ds, fused_debug.pr_colors_ds)

                    save_repro_bundle_rrd(
                        rrd_path=rrd_path,
                        benchmark_dataset_name=benchmark_dataset_name,
                        scene=scene,
                        set_idx=k,
                        batch_views=batch,
                        batch_idx=batch_idx,
                        gt_info_abs=gt_info_abs,
                        pr_info_abs_aligned=pr_info_abs_aligned,
                        gt_points=fused_debug.gt_ds,
                        gt_colors=fused_debug.gt_colors_ds,
                        pred_points=fused_debug.pr_ds,
                        pred_colors=fused_debug.pr_colors_ds,

                        background=tuple(_cfg_get(args, "rrd_background", [255, 255, 255])),
                        hide_grid=bool(_cfg_get(args, "rrd_hide_grid", True)),
                        collapse_panels=bool(_cfg_get(args, "rrd_collapse_panels", True)),
                        show_center_labels=bool(_cfg_get(args, "rrd_show_center_labels", True)),
                        point_radius=float(_cfg_get(args, "rrd_point_radius", 0.0)),
                        axis_size=float(_cfg_get(args, "rrd_axis_size", 0.0)),
                        axis_radius=float(_cfg_get(args, "rrd_axis_radius", 0.0)),
                    )

                    per_scene_saved[scene] += 1

                for k_metric in per_scene_results[scene].keys():
                    per_scene_results[scene][k_metric].append(float(metrics.get(k_metric, np.nan)))

        per_scene_json_path = os.path.join(
            args.output_dir, f"{benchmark_dataset_name}_per_scene_results.json"
        )
        with open(per_scene_json_path, "w", encoding="utf-8") as f:
            json.dump(per_scene_results, f, indent=4)

        across_dataset_results = {}
        for scene in per_scene_results.keys():
            for metric in per_scene_results[scene].keys():
                across_dataset_results.setdefault(metric, [])
                across_dataset_results[metric].extend(per_scene_results[scene][metric])

        for metric in list(across_dataset_results.keys()):
            across_dataset_results[metric] = float(np.nanmean(across_dataset_results[metric]))

        sim3_valid_mean = float(across_dataset_results.get("sim3_valid", np.nan))
        if np.isfinite(sim3_valid_mean):
            across_dataset_results["sim3_failure_rate"] = 1.0 - sim3_valid_mean
        else:
            across_dataset_results["sim3_failure_rate"] = float("nan")

        across_dataset_results["points_sim3_failure_rate"] = across_dataset_results["sim3_failure_rate"]

        pose_sim3_valid_mean = float(across_dataset_results.get("pose_sim3_valid", np.nan))
        if np.isfinite(pose_sim3_valid_mean):
            across_dataset_results["pose_sim3_failure_rate"] = 1.0 - pose_sim3_valid_mean
        else:
            across_dataset_results["pose_sim3_failure_rate"] = float("nan")

        avg_json_path = os.path.join(
            args.output_dir, f"{benchmark_dataset_name}_avg_across_all_scenes.json"
        )
        with open(avg_json_path, "w", encoding="utf-8") as f:
            json.dump(across_dataset_results, f, indent=4)

        print("Average results across all scenes for dataset: ", benchmark_dataset_name)
        for metric in across_dataset_results.keys():
            print(f"{metric}: {across_dataset_results[metric]}")

        per_dataset_results[benchmark_dataset_name] = across_dataset_results

    average_results = {}
    if len(per_dataset_results) > 0:
        all_metrics = set()
        for dataset_name in per_dataset_results.keys():
            all_metrics.update(per_dataset_results[dataset_name].keys())

        for metric in sorted(all_metrics):
            vals = [
                per_dataset_results[d].get(metric, np.nan)
                for d in per_dataset_results
                if metric in per_dataset_results[d]
            ]
            if metric in {"total_count", "failure_count"}:
                average_results[metric] = float(np.nansum(vals))
            elif metric == "failure_rate":
                continue
            else:
                average_results[metric] = float(np.nanmean(vals))

        sim3_valid_mean = float(average_results.get("sim3_valid", np.nan))
        if np.isfinite(sim3_valid_mean):
            average_results["sim3_failure_rate"] = 1.0 - sim3_valid_mean
        else:
            average_results["sim3_failure_rate"] = float("nan")

        average_results["points_sim3_failure_rate"] = average_results["sim3_failure_rate"]

        pose_sim3_valid_mean = float(average_results.get("pose_sim3_valid", np.nan))
        if np.isfinite(pose_sim3_valid_mean):
            average_results["pose_sim3_failure_rate"] = 1.0 - pose_sim3_valid_mean
        else:
            average_results["pose_sim3_failure_rate"] = float("nan")

        per_dataset_results["Average"] = average_results

    print("Benchmarking Done!")
    for metric in average_results.keys():
        print(f"{metric}: {average_results[metric]}")

    with open(os.path.join(args.output_dir, "per_dataset_results.json"), "w", encoding="utf-8") as f:
        json.dump(per_dataset_results, f, indent=4)
    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump({"alignment_mode": requested_alignment}, f, indent=2)


@hydra.main(version_base=None, config_path="../../configs", config_name="dense_n_view_benchmark")
def execute_benchmarking(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    
    configure_torch_hub(cfg.machine)

    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)

    benchmark(cfg)


if __name__ == "__main__":
    execute_benchmarking()  # noqa
