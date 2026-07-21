# -*- coding: utf-8 -*-
"""Paper-style DSM elevation and contour visualizations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def dom_axis_label(axis: str) -> str:
    return f"Ground {str(axis).upper()} (m)"


def _nice_contour_step(span: float, target_levels: int = 10) -> float:
    raw = float(span) / max(float(target_levels), 1.0)
    if not np.isfinite(raw) or raw <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(raw))
    base = 10.0 ** exponent
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiplier * base
        if raw <= step:
            return float(step)
    return float(10.0 * base)


def _contour_levels(vmin: float, vmax: float, target_levels: int = 10) -> np.ndarray:
    span = float(vmax) - float(vmin)
    if not np.isfinite(span) or span <= 0.0:
        return np.empty((0,), dtype=np.float64)
    step = _nice_contour_step(span, target_levels=target_levels)
    start = math.ceil(float(vmin) / step) * step
    stop = math.floor(float(vmax) / step) * step
    levels = np.arange(start, stop + 0.5 * step, step, dtype=np.float64)
    levels = levels[(levels > float(vmin)) & (levels < float(vmax))]
    if levels.size < 3:
        levels = np.linspace(float(vmin), float(vmax), num=min(8, target_levels + 1))[1:-1]
    if levels.size > 14:
        levels = levels[:: int(math.ceil(float(levels.size) / 14.0))]
    return levels.astype(np.float64)


def _contour_label_format(levels: np.ndarray):
    if levels.size >= 2:
        step = float(np.nanmedian(np.diff(np.sort(levels))))
    else:
        step = 1.0
    if step >= 10.0:
        decimals = 0
    elif step >= 1.0:
        decimals = 1
    elif step >= 0.1:
        decimals = 2
    else:
        decimals = 3

    def _fmt(value: float) -> str:
        return f"{float(value):.{decimals}f} m"

    return _fmt


def _prepare_dsm(
    dsm: np.ndarray,
    *,
    max_vis_pixels: int,
) -> Tuple[Dict[str, object], np.ndarray, float, float, int]:
    meta: Dict[str, object] = {
        "saved": False,
        "png_path": None,
        "svg_path": None,
        "num_valid_pixels": 0,
    }
    dsm = np.asarray(dsm, dtype=np.float32)
    if dsm.ndim != 2 or dsm.size == 0:
        meta["reason"] = "empty DSM"
        return meta, np.empty((0, 0), dtype=np.float32), 0.0, 0.0, 1

    valid = np.isfinite(dsm)
    num_valid = int(valid.sum())
    meta["num_valid_pixels"] = num_valid
    if num_valid == 0:
        meta["reason"] = "no finite DSM pixels"
        return meta, np.empty((0, 0), dtype=np.float32), 0.0, 0.0, 1

    h, w = dsm.shape
    stride = 1
    if int(h) * int(w) > int(max_vis_pixels):
        stride = int(math.ceil(math.sqrt(float(h * w) / float(max_vis_pixels))))
    dsm_vis = dsm[::stride, ::stride]

    finite_vals = dsm_vis[np.isfinite(dsm_vis)]
    if finite_vals.size == 0:
        finite_vals = dsm[np.isfinite(dsm)]
    vmin = float(np.nanpercentile(finite_vals, 1.0))
    vmax = float(np.nanpercentile(finite_vals, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite_vals))
        vmax = float(np.nanmax(finite_vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        delta = max(abs(float(vmin)), 1.0) * 0.01
        vmin = float(vmin) - delta
        vmax = float(vmax) + delta

    return meta, dsm_vis, float(vmin), float(vmax), int(stride)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _make_dsm_figure(plt, dsm_vis: np.ndarray, dom_axes: str):
    h, w = dsm_vis.shape
    axes = str(dom_axes).lower().strip()
    x_label = dom_axis_label(axes[0] if len(axes) > 0 else "x")
    y_label = dom_axis_label(axes[1] if len(axes) > 1 else "y")

    aspect = max(float(w) / max(float(h), 1.0), 0.2)
    fig_w = 3.35
    fig_h = min(max(fig_w / aspect, 2.0), 5.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f3f3f3")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(direction="out", length=2.5, width=0.6, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    return fig, ax


def _add_matched_colorbar(fig, ax, im, *, label: str):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label, rotation=90, labelpad=4)
    cbar.ax.tick_params(labelsize=7, direction="out", length=2.5, width=0.6, pad=2)
    cbar.outline.set_linewidth(0.6)
    return cbar


def save_dsm_elevation_visualization(
    dsm: np.ndarray,
    *,
    output_dir: Path,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    dom_axes: str,
    max_vis_pixels: int = 25000000,
) -> Dict[str, object]:
    """Save a paper-style DSM elevation visualization with a height colorbar."""
    meta, dsm_vis, vmin, vmax, stride = _prepare_dsm(
        dsm,
        max_vis_pixels=int(max_vis_pixels),
    )
    if dsm_vis.size == 0:
        return meta

    try:
        plt = _setup_matplotlib()
    except Exception as exc:
        meta["reason"] = f"matplotlib unavailable: {exc}"
        print(f"[DOM][WARN] Failed to import matplotlib for DSM visualization: {exc}")
        return meta

    fig, ax = _make_dsm_figure(plt, dsm_vis, str(dom_axes))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f3f3f3")
    masked = np.ma.masked_invalid(dsm_vis)
    im = ax.imshow(
        masked,
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        extent=[float(u_min), float(u_max), float(v_min), float(v_max)],
        origin="upper",
        interpolation="nearest",
        rasterized=True,
    )
    _add_matched_colorbar(fig, ax, im, label="Elevation (m)")

    png_path = Path(output_dir) / "dom_dsm_elevation.png"
    svg_path = Path(output_dir) / "dom_dsm_elevation.svg"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    meta.update(
        {
            "saved": True,
            "png_path": str(png_path),
            "svg_path": str(svg_path),
            "vmin": float(vmin),
            "vmax": float(vmax),
            "downsample_stride": int(stride),
        }
    )
    print(f"[DOM] saved DSM elevation PNG: {png_path}")
    print(f"[DOM] saved DSM elevation SVG: {svg_path}")
    return meta


def save_dsm_contour_visualization(
    dsm: np.ndarray,
    *,
    output_dir: Path,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    dom_axes: str,
    max_vis_pixels: int = 25000000,
) -> Dict[str, object]:
    """Save DSM elevation visualization overlaid with labeled contour lines."""
    meta, dsm_vis, vmin, vmax, stride = _prepare_dsm(
        dsm,
        max_vis_pixels=int(max_vis_pixels),
    )
    if dsm_vis.size == 0:
        return meta

    try:
        plt = _setup_matplotlib()
    except Exception as exc:
        meta["reason"] = f"matplotlib unavailable: {exc}"
        print(f"[DOM][WARN] Failed to import matplotlib for DSM contours: {exc}")
        return meta

    fig, ax = _make_dsm_figure(plt, dsm_vis, str(dom_axes))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f3f3f3")
    masked = np.ma.masked_invalid(dsm_vis)
    im = ax.imshow(
        masked,
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        extent=[float(u_min), float(u_max), float(v_min), float(v_max)],
        origin="upper",
        interpolation="nearest",
        rasterized=True,
    )

    levels = _contour_levels(float(vmin), float(vmax), target_levels=10)
    if levels.size > 0:
        contour = ax.contour(
            masked,
            levels=levels,
            colors="white",
            linewidths=0.55,
            alpha=0.95,
            extent=[float(u_min), float(u_max), float(v_min), float(v_max)],
            origin="upper",
        )
        labels = ax.clabel(
            contour,
            contour.levels,
            inline=True,
            inline_spacing=3,
            fmt=_contour_label_format(levels),
            fontsize=3.0,
            colors="white",
        )
        for label in labels:
            label.set_fontweight("normal")

    _add_matched_colorbar(fig, ax, im, label="Elevation (m)")

    png_path = Path(output_dir) / "dom_dsm_elevation_contours.png"
    svg_path = Path(output_dir) / "dom_dsm_elevation_contours.svg"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    meta.update(
        {
            "saved": True,
            "png_path": str(png_path),
            "svg_path": str(svg_path),
            "vmin": float(vmin),
            "vmax": float(vmax),
            "downsample_stride": int(stride),
            "contour_levels": [float(x) for x in levels.tolist()],
        }
    )
    print(f"[DOM] saved DSM contour PNG: {png_path}")
    print(f"[DOM] saved DSM contour SVG: {svg_path}")
    return meta
