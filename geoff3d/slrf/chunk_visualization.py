# -*- coding: utf-8 -*-
"""Spatial chunk footprint visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from geoff3d.slrf.chunk_artifacts import make_chunk_color_lookup

def save_spatial_chunk_core_footprint_xy_visualization(
    meta: Dict[str, object],
    grid_meta: Dict[str, object],
    chunks: Sequence[Dict[str, object]],
    output_path: Path,
    point_size: float = 12.0,
    bg_point_size: float = 3.0,
    point_alpha: float = 0.78,
    bg_alpha: float = 0.22,
    label_size: float = 11.0,
    font_scale: float = 1.35,
    padding_ratio: float = 0.02,
    show_legend: bool = False,
    legend_cols: int = 0,
    legend_max_rows: int = 16,
) -> Optional[Path]:
    """
    Save a paper-style chunk visualization on the footprint plane.

    Points are footprint centers, not camera centers. Gray background points
    show all selected footprints; colored points show core footprints per chunk.
    Seam/overlap points are not drawn to avoid clutter.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Failed to import matplotlib for chunk visualization: {exc}")
        return None

    stems = list(meta.get("stems", []))
    footprint_centers = grid_meta.get("footprint_centers", None)

    if footprint_centers is None:
        print(
            "[WARN] Chunk footprint visualization skipped: "
            "grid_meta['footprint_centers'] is missing. "
            "Please store footprint_centers in build_footprint_tree_chunks()."
        )
        return None

    centers = np.asarray(footprint_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        print(
            "[WARN] Chunk footprint visualization skipped: "
            f"invalid footprint_centers shape={centers.shape}, expected [N, 2]."
        )
        return None

    if len(stems) > 0 and centers.shape[0] != len(stems):
        print(
            "[WARN] Chunk footprint visualization: "
            f"num footprint centers={centers.shape[0]} != num stems={len(stems)}. "
            "Will still draw valid indices."
        )

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

        valid = (
            (core_indices >= 0)
            & (core_indices < centers.shape[0])
        )
        core_indices = core_indices[valid]
        if core_indices.size == 0:
            continue

        xy = centers[core_indices]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if xy.shape[0] == 0:
            continue

        chunk_core_xy.append((chunk_id, xy))

    if not chunk_core_xy:
        print("[WARN] Chunk footprint visualization skipped: no valid chunk core footprint centers.")
        return None

    output_dir = Path(output_path).expanduser().resolve()
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_path = vis_dir / "chunk_core_footprint_xy.svg"
    png_path = vis_dir / "chunk_core_footprint_xy.png"

    all_xy = centers[finite_all]
    n_chunks = len(chunk_core_xy)
    n_points = int(all_xy.shape[0])

    def _auto_marker_size(value: float, default: float, min_size: float) -> float:
        if float(value) > 0:
            return float(value)
        if n_points <= 0:
            return float(default)
        return float(max(min_size, min(default, 1500.0 / max(np.sqrt(n_points), 1.0))))

    bg_marker_size = _auto_marker_size(bg_point_size, 3.0, 0.6)
    core_marker_size = _auto_marker_size(point_size, 12.0, 2.0)
    point_alpha = float(np.clip(point_alpha, 0.0, 1.0))
    bg_alpha = float(np.clip(bg_alpha, 0.0, 1.0))
    font_scale = max(0.1, float(font_scale))
    padding_ratio = max(0.0, float(padding_ratio))
    label_size = float(label_size) * font_scale
    axes_label_size = (
        label_size if label_size > 0 else 13.0 * font_scale
    )
    legend_max_rows = max(1, int(legend_max_rows))
    if int(legend_cols) > 0:
        legend_ncol = int(legend_cols)
    else:
        legend_ncol = int(max(1, min(4, np.ceil(n_chunks / legend_max_rows))))
    legend_rows = int(np.ceil(n_chunks / max(legend_ncol, 1)))

    split_lines = grid_meta.get("footprint_split_lines", [])
    split_lines_frame = str(grid_meta.get("footprint_split_lines_frame", "original"))
    flight_frame = grid_meta.get("footprint_flight_frame", None)

    def _flight_to_plot_xy(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        shape = pts.shape
        pts = pts.reshape(-1, 2)

        if split_lines_frame != "flight_aligned" or not isinstance(flight_frame, dict):
            return pts.reshape(shape)

        try:
            origin = np.asarray(flight_frame["origin"], dtype=np.float64).reshape(2)
            rot_to_flight = np.asarray(
                flight_frame["rot_to_flight"],
                dtype=np.float64,
            ).reshape(2, 2)
        except Exception:
            return pts.reshape(shape)

        # Inverse of:
        #   p_local = (p_world - origin) @ rot_to_flight.T
        # is:
        #   p_world = p_local @ rot_to_flight + origin
        out = pts @ rot_to_flight + origin[None, :]
        return out.reshape(shape)

    # Use actual footprint centers for the plot bounds. Including the rotated
    # root-region corners creates large empty margins that contain neither
    # observations nor chunk cores. Split lines outside these compact bounds
    # are intentionally clipped by the axes.
    bounds_xy = all_xy

    # Keep SVG text editable and explicitly tagged as Times New Roman. Raster
    # output requires Times New Roman to be installed in the runtime environment.
    try:
        from matplotlib import font_manager

        font_manager.findfont("Times New Roman", fallback_to_default=False)
    except Exception:
        print(
            "[WARN] Times New Roman is not installed. SVG text will retain the "
            "Times New Roman font-family declaration, but raster PNG rendering "
            "may use a fallback."
        )

    with plt.rc_context(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.sf": "Times New Roman",
            "mathtext.cal": "Times New Roman:italic",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11.0 * font_scale,
            "axes.titlesize": 14.0 * font_scale,
            "axes.labelsize": axes_label_size,
            "xtick.labelsize": 11.0 * font_scale,
            "ytick.labelsize": 11.0 * font_scale,
            "legend.fontsize": (8.5 if n_chunks > 20 else 9.5) * font_scale,
            "axes.linewidth": 0.8,
        }
    ):
        # An optional legend is placed inside the axes and therefore does not
        # require an extra empty strip beside the plot.
        fig_w = 7.2
        fig_h = 5.2 + 0.10 * max(0, min(legend_rows, 18) - 12)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

        # Draw adaptive-tree split lines first.
        if isinstance(split_lines, list) and len(split_lines) > 0:
            for item in split_lines:
                if not isinstance(item, dict):
                    continue

                try:
                    axis = int(item.get("axis", -1))
                    threshold = float(item.get("threshold", np.nan))
                    rmin = np.asarray(
                        item.get("region_min", None),
                        dtype=np.float64,
                    ).reshape(-1)
                    rmax = np.asarray(
                        item.get("region_max", None),
                        dtype=np.float64,
                    ).reshape(-1)
                    depth = int(item.get("depth", 0))
                except Exception:
                    continue

                if (
                    axis not in (0, 1)
                    or rmin.shape[0] < 2
                    or rmax.shape[0] < 2
                    or not np.isfinite(threshold)
                    or not np.isfinite(rmin[:2]).all()
                    or not np.isfinite(rmax[:2]).all()
                ):
                    continue

                # Root split is darker/thicker; deeper splits are lighter/thinner.
                alpha = max(0.18, 0.85 - 0.07 * float(depth))
                linewidth = max(0.45, 1.8 - 0.12 * float(depth))

                if axis == 0:
                    # Local vertical split line: local_x = threshold.
                    # In original XY this becomes a line perpendicular to the
                    # main flight direction.
                    line_local = np.asarray(
                        [
                            [threshold, float(rmin[1])],
                            [threshold, float(rmax[1])],
                        ],
                        dtype=np.float64,
                    )
                else:
                    # Local horizontal split line: local_y = threshold.
                    line_local = np.asarray(
                        [
                            [float(rmin[0]), threshold],
                            [float(rmax[0]), threshold],
                        ],
                        dtype=np.float64,
                    )

                line_xy = _flight_to_plot_xy(line_local)

                ax.plot(
                    line_xy[:, 0],
                    line_xy[:, 1],
                    color="0.18",
                    alpha=alpha,
                    linewidth=linewidth,
                    zorder=1.5,
                )

        # All footprint centers as an unlabeled background layer.
        ax.scatter(
            all_xy[:, 0],
            all_xy[:, 1],
            s=bg_marker_size,
            c="0.78",
            alpha=bg_alpha,
            linewidths=0,
            zorder=1,
        )

        rgba_by_chunk_id, _ = make_chunk_color_lookup(chunks)

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

            # Write chunk id at the chunk core footprint centroid.
            if label_size > 0:
                center = np.mean(xy, axis=0)
                # Preserve the chunk hue while darkening the text for stronger
                # contrast against its semi-transparent point markers.
                label_color = tuple(
                    float(np.clip(channel, 0.0, 1.0)) * 0.65
                    for channel in color[:3]
                )
                txt = ax.text(
                    center[0],
                    center[1],
                    str(chunk_id),
                    fontsize=label_size,
                    ha="center",
                    va="center",
                    color=label_color,
                    bbox={
                        "boxstyle": "round,pad=0.06",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.62,
                    },
                    zorder=3,
                )

        xmin, ymin = np.min(bounds_xy, axis=0)
        xmax, ymax = np.max(bounds_xy, axis=0)
        dx = max(float(xmax - xmin), 1e-6)
        dy = max(float(ymax - ymin), 1e-6)
        pad_x = dx * padding_ratio
        pad_y = dy * padding_ratio

        axes_name = str(grid_meta.get("axes", "xy"))
        x_name = axes_name[0] if len(axes_name) >= 1 else "x"
        y_name = axes_name[1] if len(axes_name) >= 2 else "y"

        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Ground {x_name.upper()} (m)")
        ax.set_ylabel(f"Ground {y_name.upper()} (m)")
        ax.tick_params(direction="out", length=3.5, width=0.8)
        ax.grid(True, linewidth=0.35, alpha=0.25, color="0.65")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show_legend and legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="best",
                frameon=True,
                facecolor="white",
                framealpha=0.82,
                edgecolor="none",
                ncol=legend_ncol,
                handletextpad=0.4,
                columnspacing=0.8,
                borderaxespad=0.55,
                labelspacing=0.32 if n_chunks > 20 else 0.42,
                markerscale=1.15,
            )

        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, format="svg")
        fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(fig)

    print(
        "[INFO] Saved chunk core footprint XY visualization: "
        f"{save_path} and {png_path}"
    )
    return save_path


# Main
# ---------------------------------------------------------------------------
