# -*- coding: utf-8 -*-
"""Depth-confidence filtering and debug visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PRED_MIN_DEPTH = 1.0e-6

def _pred_tensor_hw_map(
    tensor: Optional[torch.Tensor],
    *,
    reduce_channels: bool = False,
) -> Optional[torch.Tensor]:
    if tensor is None or not torch.is_tensor(tensor):
        return None
    value = tensor
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    elif value.ndim == 3 and reduce_channels:
        value = value.float().mean(dim=-1)
    if value.ndim != 2:
        return None
    return value


def _zero_pred_depth_at_mask(pred: Dict[str, torch.Tensor], drop_mask: torch.Tensor) -> None:
    drop_mask = drop_mask.to(dtype=torch.bool)
    for key in ("pts3d_cam", "pts3d"):
        tensor = pred.get(key)
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if tensor.ndim == 4 and tensor.shape[0] == 1 and tensor.shape[1:3] == drop_mask.shape:
            tensor = tensor.clone()
            tensor[0][drop_mask] = 0
            pred[key] = tensor
        elif tensor.ndim == 3 and tensor.shape[:2] == drop_mask.shape:
            tensor = tensor.clone()
            tensor[drop_mask] = 0
            pred[key] = tensor

    for key in ("depth_along_ray", "depth", "depthmap"):
        tensor = pred.get(key)
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if tensor.ndim == 4 and tensor.shape[0] == 1 and tensor.shape[1:3] == drop_mask.shape:
            tensor = tensor.clone()
            if tensor.shape[-1] == 1:
                tensor[0, ..., 0][drop_mask] = 0
            else:
                tensor[0][drop_mask] = 0
            pred[key] = tensor
        elif tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[1:] == drop_mask.shape:
            tensor = tensor.clone()
            tensor[0][drop_mask] = 0
            pred[key] = tensor
        elif tensor.ndim == 2 and tensor.shape == drop_mask.shape:
            tensor = tensor.clone()
            tensor[drop_mask] = 0
            pred[key] = tensor


def _tensor_to_np_float(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach()
    if value.dtype in (torch.float16, torch.bfloat16):
        value = value.float()
    return value.cpu().numpy().astype(np.float32, copy=False)


def _tensor_bool_to_np(mask: torch.Tensor) -> np.ndarray:
    return mask.detach().to(dtype=torch.bool).cpu().numpy()


def apply_optional_keep_masks_to_valid_masks(
    pred_valid_masks: Sequence[np.ndarray],
    keep_masks: Optional[Sequence[Optional[np.ndarray]]],
) -> List[np.ndarray]:
    out: List[np.ndarray] = []

    for i, valid in enumerate(pred_valid_masks):
        mask = np.asarray(valid, dtype=bool).copy()

        if keep_masks is not None and i < len(keep_masks):
            keep = keep_masks[i]
            if keep is not None:
                keep = np.asarray(keep, dtype=bool)
                if keep.shape == mask.shape:
                    mask &= keep
                else:
                    print(
                        f"[WARN] confidence keep mask shape {keep.shape} "
                        f"does not match valid mask shape {mask.shape}; skip for view {i}"
                    )

        out.append(mask)

    return out


def _colorize_debug_scalar(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    cmap_name: str = "turbo",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    import matplotlib

    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if vmin is None or vmax is None:
        if valid.any():
            selected = values[valid]
            lo, hi = np.percentile(selected, [2.0, 98.0])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = float(np.nanmin(selected))
                hi = float(np.nanmax(selected))
            if hi <= lo:
                hi = lo + 1.0
            vmin = float(lo)
            vmax = float(hi)
        else:
            vmin = 0.0
            vmax = 1.0
    norm = np.clip((values - float(vmin)) / max(float(vmax) - float(vmin), 1e-8), 0.0, 1.0)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb, float(vmin), float(vmax)


def _save_depth_conf_filter_debug_view(
    *,
    path: Path,
    rgb: np.ndarray,
    conf: np.ndarray,
    depth_before: np.ndarray,
    depth_after: np.ndarray,
    drop_mask: np.ndarray,
    threshold: float,
    quantile: float,
    chunk_id: int,
    stem: str,
) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = conf.shape
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)

    conf_valid = np.isfinite(conf)
    depth_valid = np.isfinite(depth_before) & (depth_before > 0)
    _, conf_vmin, conf_vmax = _colorize_debug_scalar(conf, conf_valid, cmap_name="viridis")
    _, depth_vmin, depth_vmax = _colorize_debug_scalar(depth_before, depth_valid, cmap_name="turbo")
    mask_vis = np.zeros((height, width, 3), dtype=np.uint8)
    mask_vis[..., 0] = np.asarray(drop_mask, dtype=np.uint8) * 255

    panels = [
        ("RGB", rgb, None, None, None),
        ("confidence", None, conf, conf_valid, (conf_vmin, conf_vmax)),
        ("drop mask", mask_vis, None, None, None),
        ("depth before", None, depth_before, depth_valid, (depth_vmin, depth_vmax)),
        (
            "depth after",
            None,
            depth_after,
            np.isfinite(depth_after) & (depth_after > 0),
            (depth_vmin, depth_vmax),
        ),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.0), dpi=150)
    for ax, (title, image, scalar, valid, value_range) in zip(axes, panels):
        if scalar is None:
            ax.imshow(image)
        else:
            cmap_name = "viridis" if title == "confidence" else "turbo"
            cmap = matplotlib.colormaps.get_cmap(cmap_name).copy()
            cmap.set_bad("black")
            masked = np.ma.masked_where(~valid, scalar)
            im = ax.imshow(masked, cmap=cmap, vmin=value_range[0], vmax=value_range[1])
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"chunk {chunk_id:03d} {stem} | conf q={quantile:.3f}, threshold={threshold:.6g}"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def apply_depth_confidence_filter_to_preds(
    *,
    preds: Sequence[Dict[str, torch.Tensor]],
    rgbs: Sequence[np.ndarray],
    stems: Sequence[str],
    chunk_id: int,
    conf_quantile: float,
    debug_dir: Optional[Path] = None,
) -> Tuple[Dict[str, object], List[Optional[np.ndarray]]]:
    q = float(conf_quantile)
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"conf_quantile must be in [0, 1], got {q}")

    keep_masks: List[Optional[np.ndarray]] = [None for _ in preds]

    if q <= 0.0:
        return {
            "enabled": False,
            "quantile": q,
            "num_views": int(len(preds)),
            "note": "confidence filtering disabled; original predicted depth/points are preserved",
        }, keep_masks

    per_view: List[Dict[str, object]] = []
    total_valid = 0
    total_dropped = 0
    total_missing_conf = 0

    for local_i, pred in enumerate(preds):
        stem = str(stems[local_i]) if local_i < len(stems) else f"view_{local_i:03d}"

        conf_t = _pred_tensor_hw_map(pred.get("conf"), reduce_channels=True)
        pts_cam_t = pred.get("pts3d_cam")

        if conf_t is None or pts_cam_t is None or not torch.is_tensor(pts_cam_t):
            total_missing_conf += 1
            per_view.append(
                {
                    "local_index": int(local_i),
                    "stem": stem,
                    "valid": False,
                    "reason": "missing_conf_or_pts3d_cam",
                }
            )
            continue

        pts_cam = pts_cam_t[0] if pts_cam_t.ndim == 4 and pts_cam_t.shape[0] == 1 else pts_cam_t

        if pts_cam.ndim != 3 or pts_cam.shape[-1] != 3 or pts_cam.shape[:2] != conf_t.shape:
            total_missing_conf += 1
            per_view.append(
                {
                    "local_index": int(local_i),
                    "stem": stem,
                    "valid": False,
                    "reason": "shape_mismatch",
                    "conf_shape": list(conf_t.shape),
                    "pts3d_cam_shape": list(pts_cam.shape),
                }
            )
            continue

        depth_before_t = pts_cam[..., 2].detach().float()
        conf = conf_t.detach().float()

        valid = (
            torch.isfinite(conf)
            & torch.isfinite(pts_cam).all(dim=-1)
            & torch.isfinite(depth_before_t)
            & (depth_before_t > PRED_MIN_DEPTH)
        )

        num_valid = int(valid.sum().item())
        if num_valid <= 0:
            per_view.append(
                {
                    "local_index": int(local_i),
                    "stem": stem,
                    "valid": False,
                    "reason": "no_valid_depth",
                }
            )
            continue

        threshold = torch.quantile(conf[valid], q)
        keep_mask = valid & (conf >= threshold)
        drop_mask = valid & ~keep_mask

        keep_masks[local_i] = _tensor_bool_to_np(keep_mask)

        num_drop = int(drop_mask.sum().item())
        total_valid += num_valid
        total_dropped += num_drop

        per_view.append(
            {
                "local_index": int(local_i),
                "stem": stem,
                "valid": True,
                "threshold": float(threshold.detach().cpu().item()),
                "num_valid": num_valid,
                "num_dropped": num_drop,
                "drop_fraction": float(num_drop / max(1, num_valid)),
            }
        )

        if debug_dir is not None:
            depth_before_np = _tensor_to_np_float(depth_before_t)
            keep_mask_np = keep_masks[local_i]
            depth_after = np.where(keep_mask_np, depth_before_np, 0.0).astype(np.float32)

            debug_path = debug_dir / f"chunk_{chunk_id:03d}" / f"{local_i:02d}_{stem}_depth_conf_filter.png"
            _save_depth_conf_filter_debug_view(
                path=debug_path,
                rgb=np.asarray(rgbs[local_i], dtype=np.uint8),
                conf=_tensor_to_np_float(conf),
                depth_before=depth_before_np,
                depth_after=depth_after,
                drop_mask=_tensor_bool_to_np(drop_mask),
                threshold=float(threshold.detach().cpu().item()),
                quantile=q,
                chunk_id=int(chunk_id),
                stem=stem,
            )

    return {
        "enabled": True,
        "quantile": q,
        "num_views": int(len(preds)),
        "num_views_missing_conf": int(total_missing_conf),
        "num_valid_pixels": int(total_valid),
        "num_dropped_pixels": int(total_dropped),
        "drop_fraction": float(total_dropped / max(1, total_valid)),
        "per_view": per_view,
        "note": "confidence filtering is stored as masks; original predicted depth/points are preserved",
    }, keep_masks


