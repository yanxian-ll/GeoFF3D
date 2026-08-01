# -*- coding: utf-8 -*-
"""Chunk artifact colors and footprint visualization helpers."""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def make_chunk_color_lookup(
    chunks: Sequence[Dict[str, object]],
) -> Tuple[Dict[int, Tuple[float, float, float, float]], Dict[int, Tuple[int, int, int]]]:
    n_chunks = int(len(chunks))
    rgba_by_id: Dict[int, Tuple[float, float, float, float]] = {}
    rgb_by_id: Dict[int, Tuple[int, int, int]] = {}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cmap_name = "tab20" if n_chunks <= 20 else "gist_ncar"
        cmap = plt.colormaps[cmap_name].resampled(max(n_chunks, 1))
        for draw_idx, chunk in enumerate(chunks):
            chunk_id = int(chunk.get("chunk_id", draw_idx))
            rgba = tuple(float(v) for v in cmap(draw_idx))
            rgba_by_id[chunk_id] = rgba  # type: ignore[assignment]
            rgb_by_id[chunk_id] = tuple(
                int(np.clip(round(255.0 * rgba[i]), 0, 255))
                for i in range(3)
            )  # type: ignore[assignment]
        return rgba_by_id, rgb_by_id
    except Exception:
        pass

    for draw_idx, chunk in enumerate(chunks):
        chunk_id = int(chunk.get("chunk_id", draw_idx))
        h = (draw_idx / max(n_chunks, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        rgba_by_id[chunk_id] = (r, g, b, 1.0)
        rgb_by_id[chunk_id] = (
            int(round(255.0 * r)),
            int(round(255.0 * g)),
            int(round(255.0 * b)),
        )
    return rgba_by_id, rgb_by_id


def save_chunk_footprint_xy_visualization(
    *,
    meta: Dict[str, object],
    grid_meta: Dict[str, object],
    chunks: Sequence[Dict[str, object]],
    output_dir: Path,
    file_stem: str = "chunk_footprint_xy",
    point_size: float = 12.0,
    bg_point_size: float = 3.0,
    point_alpha: float = 0.78,
    bg_alpha: float = 0.22,
    label_size: float = 7.0,
    legend_cols: int = 0,
    legend_max_rows: int = 16,
    rgba_by_chunk_id: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as path_effects
    except Exception as exc:
        print(f"[WARN] Failed to import matplotlib for chunk visualization: {exc}")
        return None

    footprint_centers = grid_meta.get("footprint_centers", None)
    if footprint_centers is None:
        print("[WARN] Chunk footprint visualization skipped: footprint_centers missing.")
        return None

    centers = np.asarray(footprint_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        print(
            "[WARN] Chunk footprint visualization skipped: "
            f"invalid footprint_centers shape={centers.shape}, expected [N, 2]."
        )
        return None

    finite_all = np.isfinite(centers).all(axis=1)
    if not bool(finite_all.any()):
        print("[WARN] Chunk footprint visualization skipped: no finite footprint centers.")
        return None

    chunk_core_xy: List[Tuple[int, np.ndarray]] = []
    for fallback_chunk_id, chunk in enumerate(chunks):
        chunk_id = int(chunk.get("chunk_id", fallback_chunk_id))
        core_indices = np.asarray(chunk.get("core_indices", []), dtype=np.int64)
        if core_indices.size == 0:
            continue
        valid = (core_indices >= 0) & (core_indices < centers.shape[0])
        xy = centers[core_indices[valid]]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if xy.shape[0] > 0:
            chunk_core_xy.append((chunk_id, xy))

    if not chunk_core_xy:
        print("[WARN] Chunk footprint visualization skipped: no valid chunk centers.")
        return None

    if rgba_by_chunk_id is None:
        rgba_by_chunk_id, _ = make_chunk_color_lookup(chunks)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{file_stem}.svg"
    png_path = output_dir / f"{file_stem}.png"

    all_xy = centers[finite_all]
    n_chunks = len(chunk_core_xy)
    n_points = int(all_xy.shape[0])

    def _auto_marker_size(value: float, default: float, min_size: float) -> float:
        if float(value) > 0:
            return float(value)
        return float(max(min_size, min(default, 1500.0 / max(np.sqrt(max(n_points, 1)), 1.0))))

    bg_marker_size = _auto_marker_size(bg_point_size, 3.0, 0.6)
    core_marker_size = _auto_marker_size(point_size, 12.0, 2.0)
    point_alpha = float(np.clip(point_alpha, 0.0, 1.0))
    bg_alpha = float(np.clip(bg_alpha, 0.0, 1.0))
    label_size = float(label_size)
    legend_max_rows = max(1, int(legend_max_rows))
    legend_ncol = int(legend_cols) if int(legend_cols) > 0 else int(
        max(1, min(4, np.ceil(n_chunks / legend_max_rows)))
    )
    legend_rows = int(np.ceil(n_chunks / max(legend_ncol, 1)))

    bounds_xy = all_xy
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 8.5 if n_chunks > 20 else 9.5,
            "axes.linewidth": 0.8,
        }
    ):
        fig_w = 7.2 + 1.35 * max(0, legend_ncol - 1)
        fig_h = 5.2 + 0.10 * max(0, min(legend_rows, 18) - 12)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

        ax.scatter(
            all_xy[:, 0],
            all_xy[:, 1],
            s=bg_marker_size,
            c="0.78",
            alpha=bg_alpha,
            linewidths=0,
            zorder=1,
        )

        legend_handles = []
        for chunk_id, xy in chunk_core_xy:
            color = rgba_by_chunk_id.get(chunk_id, (0.1, 0.3, 0.9, 1.0))
            handle = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=core_marker_size,
                c=[color],
                alpha=point_alpha,
                edgecolors="white",
                linewidths=max(0.0, min(0.25, core_marker_size / 80.0)),
                zorder=2,
                label=f"Chunk {chunk_id}",
            )
            legend_handles.append(handle)

            if label_size > 0:
                center = np.mean(xy, axis=0)
                txt = ax.text(
                    center[0],
                    center[1],
                    str(chunk_id),
                    fontsize=label_size,
                    ha="center",
                    va="center",
                    color="black",
                    zorder=3,
                )
                txt.set_path_effects(
                    [
                        path_effects.Stroke(
                            linewidth=max(1.4, label_size * 0.22),
                            foreground="white",
                            alpha=0.86,
                        ),
                        path_effects.Normal(),
                    ]
                )

        xmin, ymin = np.min(bounds_xy, axis=0)
        xmax, ymax = np.max(bounds_xy, axis=0)
        dx = max(float(xmax - xmin), 1e-6)
        dy = max(float(ymax - ymin), 1e-6)
        axes_name = str(grid_meta.get("axes", "xy"))
        x_name = axes_name[0] if len(axes_name) >= 1 else "x"
        y_name = axes_name[1] if len(axes_name) >= 2 else "y"
        ax.set_xlim(xmin - dx * 0.05, xmax + dx * 0.05)
        ax.set_ylim(ymin - dy * 0.05, ymax + dy * 0.05)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Ground {x_name.upper()} (m)")
        ax.set_ylabel(f"Ground {y_name.upper()} (m)")
        ax.tick_params(direction="out", length=3.5, width=0.8)
        ax.grid(True, linewidth=0.35, alpha=0.25, color="0.65")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                ncol=legend_ncol,
                handletextpad=0.4,
                columnspacing=0.8,
                borderaxespad=0.0,
                labelspacing=0.32 if n_chunks > 20 else 0.42,
                markerscale=1.15,
            )

        fig.tight_layout()
        fig.savefig(svg_path, bbox_inches="tight", format="svg")
        fig.savefig(png_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    print(f"[INFO] Saved chunk footprint XY visualization: {svg_path} and {png_path}")
    return svg_path
