#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot CVPR-style camera trajectories from eval camera npz files.

Expected eval directory layout:
  eval/
    gt_cameras.npz               # stems, T_c2w, valid
    pred_cameras.npz             # stems, T_c2w, valid
    aligned_metrics/
      aligned_pred_cameras.npz   # stems, T_c2w, gt_T_c2w, ...

Example:
  python scripts/visualize_eval_camera_trajectory.py \
    --eval_dir outputs/spatial/geoff3d/npu_dronemap/phantom3-grass-kfs/eval
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "gt": "#1f77b4",
    "pred": "#d95f02",
}


def load_camera_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        out = {key: data[key] for key in data.files}
    if "T_c2w" not in out:
        raise KeyError(f"{path} does not contain key 'T_c2w'")
    return out


def valid_pose_mask(poses: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    poses = np.asarray(poses)
    mask = poses.shape[-2:] == (4, 4) and poses.ndim == 3
    if not mask:
        raise ValueError(f"Expected poses with shape (N, 4, 4), got {poses.shape}")
    finite = np.isfinite(poses).all(axis=(1, 2))
    if valid is None:
        return finite
    return finite & np.asarray(valid, dtype=bool)


def centers_from_poses(poses: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    mask = valid_pose_mask(poses, valid)
    return np.asarray(poses, dtype=np.float64)[mask, :3, 3]


def match_by_stem(
    pred: Dict[str, np.ndarray],
    gt: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    pred_stems = [str(x) for x in pred.get("stems", [])]
    gt_stems = [str(x) for x in gt.get("stems", [])]
    if len(pred_stems) == 0 or len(gt_stems) == 0:
        return (
            centers_from_poses(pred["T_c2w"], pred.get("valid")),
            centers_from_poses(gt["T_c2w"], gt.get("valid")),
        )

    gt_index = {stem: i for i, stem in enumerate(gt_stems)}
    pred_ids = []
    gt_ids = []
    for i, stem in enumerate(pred_stems):
        if stem in gt_index:
            pred_ids.append(i)
            gt_ids.append(gt_index[stem])

    if not pred_ids:
        return (
            centers_from_poses(pred["T_c2w"], pred.get("valid")),
            centers_from_poses(gt["T_c2w"], gt.get("valid")),
        )

    pred_T = np.asarray(pred["T_c2w"])[pred_ids]
    gt_T = np.asarray(gt["T_c2w"])[gt_ids]
    pred_valid = np.asarray(pred.get("valid", np.ones(len(pred["T_c2w"]), bool)))[pred_ids]
    gt_valid = np.asarray(gt.get("valid", np.ones(len(gt["T_c2w"]), bool)))[gt_ids]
    keep = (
        valid_pose_mask(pred_T, pred_valid)
        & valid_pose_mask(gt_T, gt_valid)
    )
    return pred_T[keep, :3, 3].astype(np.float64), gt_T[keep, :3, 3].astype(np.float64)


def load_trajectories(eval_dir: Path, pred_mode: str) -> Tuple[np.ndarray, np.ndarray, str]:
    aligned_path = eval_dir / "aligned_metrics" / "aligned_pred_cameras.npz"
    if pred_mode in ("auto", "aligned") and aligned_path.exists():
        aligned = load_camera_npz(aligned_path)
        if "gt_T_c2w" not in aligned:
            if pred_mode == "aligned":
                raise KeyError(f"{aligned_path} does not contain key 'gt_T_c2w'")
        else:
            pred_xyz = centers_from_poses(aligned["T_c2w"])
            gt_xyz = centers_from_poses(aligned["gt_T_c2w"])
            n = min(len(pred_xyz), len(gt_xyz))
            return pred_xyz[:n], gt_xyz[:n], "aligned"

    if pred_mode == "aligned":
        raise FileNotFoundError(aligned_path)

    pred = load_camera_npz(eval_dir / "pred_cameras.npz")
    gt = load_camera_npz(eval_dir / "gt_cameras.npz")
    pred_xyz, gt_xyz = match_by_stem(pred, gt)
    return pred_xyz, gt_xyz, "raw"


def set_axes_equal(ax, xyz: np.ndarray, pad_ratio: float = 0.08) -> None:
    mins = np.nanmin(xyz, axis=0)
    maxs = np.nanmax(xyz, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius * (1.0 + pad_ratio), 1e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def sparse_mark_indices(n: int, stride: int) -> np.ndarray:
    if n <= 0 or stride <= 0:
        return np.empty((0,), dtype=int)
    return np.arange(0, n, stride, dtype=int)


def plot_trajectory(
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    output: Path,
    title: str,
    dpi: int,
    figsize: Tuple[float, float],
    elev: float,
    azim: float,
    marker_stride: int,
    linewidth: float,
    show_legend: bool,
) -> None:
    if len(pred_xyz) == 0 and len(gt_xyz) == 0:
        raise ValueError("No valid camera centers to plot.")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    if len(gt_xyz):
        ax.plot(
            gt_xyz[:, 0],
            gt_xyz[:, 1],
            gt_xyz[:, 2],
            color=COLORS["gt"],
            linewidth=linewidth,
            label="GT",
        )
        ids = sparse_mark_indices(len(gt_xyz), marker_stride)
        if len(ids):
            ax.scatter(
                gt_xyz[ids, 0],
                gt_xyz[ids, 1],
                gt_xyz[ids, 2],
                color=COLORS["gt"],
                s=12,
                alpha=0.55,
                depthshade=False,
            )
        ax.scatter(
            *gt_xyz[0],
            color=COLORS["gt"],
            marker="x",
            s=90,
            linewidths=2.2,
            depthshade=False,
        )
        ax.scatter(
            *gt_xyz[-1],
            facecolors="white",
            edgecolors=COLORS["gt"],
            marker="o",
            s=82,
            linewidths=2.2,
            depthshade=False,
        )

    if len(pred_xyz):
        ax.plot(
            pred_xyz[:, 0],
            pred_xyz[:, 1],
            pred_xyz[:, 2],
            color=COLORS["pred"],
            linewidth=linewidth,
            label="Pred",
        )
        ids = sparse_mark_indices(len(pred_xyz), marker_stride)
        if len(ids):
            ax.scatter(
                pred_xyz[ids, 0],
                pred_xyz[ids, 1],
                pred_xyz[ids, 2],
                color=COLORS["pred"],
                s=12,
                alpha=0.55,
                depthshade=False,
            )
        ax.scatter(
            *pred_xyz[0],
            color=COLORS["pred"],
            marker="x",
            s=90,
            linewidths=2.2,
            depthshade=False,
        )
        ax.scatter(
            *pred_xyz[-1],
            facecolors="white",
            edgecolors=COLORS["pred"],
            marker="o",
            s=82,
            linewidths=2.2,
            depthshade=False,
        )

    xyz = np.concatenate([x for x in (pred_xyz, gt_xyz) if len(x)], axis=0)
    set_axes_equal(ax, xyz)
    ax.set_box_aspect((1.0, 1.0, 1.0))

    ax.set_xlabel(r"$X$ (m)", labelpad=12)
    ax.set_ylabel(r"$Y$ (m)", labelpad=12)
    ax.set_zlabel(r"$Z$ (m)", labelpad=8)
    if title:
        ax.set_title(title, pad=12, fontsize=15)

    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linestyle"] = "--"
        axis._axinfo["grid"]["linewidth"] = 0.7
        axis._axinfo["grid"]["color"] = (0.72, 0.72, 0.72, 0.8)
        axis._axinfo["pane_color"] = (1.0, 1.0, 1.0, 1.0)

    if show_legend:
        ax.legend(loc="upper right", frameon=False, handlelength=2.8, borderpad=0.2)
    fig.tight_layout(pad=0.55)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_interactive_html(
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    output: Path,
    title: str,
    marker_stride: int,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Interactive HTML output requires plotly. Install it with: pip install plotly"
        ) from exc

    fig = go.Figure()

    def add_traj(name: str, xyz: np.ndarray, color: str) -> None:
        if len(xyz) == 0:
            return
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="lines",
                name=name,
                line=dict(color=color, width=6),
            )
        )
        ids = sparse_mark_indices(len(xyz), marker_stride)
        if len(ids):
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[ids, 0],
                    y=xyz[ids, 1],
                    z=xyz[ids, 2],
                    mode="markers",
                    name=f"{name} samples",
                    marker=dict(color=color, size=3, opacity=0.55),
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter3d(
                x=[xyz[0, 0], xyz[-1, 0]],
                y=[xyz[0, 1], xyz[-1, 1]],
                z=[xyz[0, 2], xyz[-1, 2]],
                mode="markers",
                name=f"{name} start/end",
                marker=dict(
                    color=[color, "white"],
                    size=[7, 8],
                    symbol=["x", "circle"],
                    line=dict(color=color, width=3),
                ),
                showlegend=False,
            )
        )

    add_traj("GT", gt_xyz, COLORS["gt"])
    add_traj("Pred", pred_xyz, COLORS["pred"])

    fig.update_layout(
        title=title or None,
        template="plotly_white",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            xaxis=dict(showbackground=True, gridcolor="#c8c8c8", zeroline=False),
            yaxis=dict(showbackground=True, gridcolor="#c8c8c8", zeroline=False),
            zaxis=dict(showbackground=True, gridcolor="#c8c8c8", zeroline=False),
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=40 if title else 0, b=0),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs=True, full_html=True)


def parse_figsize(text: str) -> Tuple[float, float]:
    parts = [x.strip() for x in text.lower().replace(",", "x").split("x") if x.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("figsize must look like '5.2x4.8'")
    return float(parts[0]), float(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval_dir",
        type=Path,
        default=Path(
            "outputs/spatial/geoff3d/npu_dronemap/"
            "phantom3-grass-kfs/eval"
        ),
        help="Evaluation directory containing *_cameras.npz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to eval_dir/trajectory_cvpr.png.",
    )
    parser.add_argument(
        "--pred_mode",
        choices=["auto", "aligned", "raw"],
        default="auto",
        help="Use aligned prediction when available, require aligned, or use raw pred_cameras.npz.",
    )
    parser.add_argument("--title", default="", help="Optional plot title.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figsize", type=parse_figsize, default=(5.2, 4.8))
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--marker_stride", type=int, default=20)
    parser.add_argument("--linewidth", type=float, default=3.0)
    parser.add_argument(
        "--no_legend",
        action="store_true",
        help="Hide the legend for compact paper figures.",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Optional interactive HTML output for manual rotate/pan/zoom in a browser.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_dir = args.eval_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else eval_dir / "trajectory_cvpr.png"
    )

    pred_xyz, gt_xyz, loaded_mode = load_trajectories(eval_dir, args.pred_mode)
    plot_trajectory(
        pred_xyz=pred_xyz,
        gt_xyz=gt_xyz,
        output=output,
        title=args.title,
        dpi=args.dpi,
        figsize=args.figsize,
        elev=args.elev,
        azim=args.azim,
        marker_stride=args.marker_stride,
        linewidth=args.linewidth,
        show_legend=not bool(args.no_legend),
    )
    if args.html is not None:
        html_output = args.html.expanduser().resolve()
        plot_interactive_html(
            pred_xyz=pred_xyz,
            gt_xyz=gt_xyz,
            output=html_output,
            title=args.title,
            marker_stride=args.marker_stride,
        )
        print(f"Saved interactive HTML: {html_output}")
    print(f"Saved trajectory plot: {output}")
    print(f"Prediction mode: {loaded_mode}")
    print(f"GT poses: {len(gt_xyz)}, Pred poses: {len(pred_xyz)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
