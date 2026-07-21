#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run predict_scene_to_rrd with a reversible translation recenter step.

This wrapper keeps the original predicted/GT logging pipeline from
`scripts/predict_scene_to_rrd.py`, but changes the coordinate frame seen by the
model:

1. Build the original views from the scene.
2. Compute a translation anchor from the input camera centers.
3. Subtract that anchor from camera_pose/camera_pose_trans/world_translation and
   optional world-frame point maps before model inference.
4. Run the selected model in the recentered local frame.
5. Add the same anchor back to predicted points, point maps, and camera centers
   before scale alignment/logging, so the saved Rerun file is in the original
   scene coordinate system.

Only translation is changed. Rotations, intrinsics, rays, RGB, depth, and masks
are left unchanged.


python scripts/predict_scene_to_rrd_recenter.py    \
 --scene_dir /opt/data/private/dataset/data/NPU_Dronemap/gopro-npu-kfs     \
 --model geoff3d     \
 --checkpoint experiments/dom/uav_training/pi3x_zup_translation_yaw_aug_8v_6d_16ipg_2g/checkpoint-last.pth \
 --output_rrd outputs/gopro_recenter.rrd     --num_views 30     --stride 1     --max_side 518     --align none

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

import predict_scene_to_rrd as base


RECENTER_CHOICES = ("none", "zero", "first_camera", "mean_camera")


def parse_recenter_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--recenter",
        choices=RECENTER_CHOICES,
        default="mean_camera",
        help=(
            "Translation-only coordinate recentering before model inference. "
            "Predictions are translated back before logging. Default: mean_camera."
        ),
    )
    parser.add_argument(
        "--no_recenter",
        action="store_true",
        help="Compatibility alias for --recenter none.",
    )
    recenter_args, remaining = parser.parse_known_args(list(argv))
    if recenter_args.no_recenter:
        recenter_args.recenter = "none"
    return recenter_args, remaining


def _torch_anchor_from_views(views: Sequence[Dict[str, object]], mode: str) -> torch.Tensor:
    if mode in {"none", "zero"}:
        device = views[0]["img"].device
        return torch.zeros(3, dtype=torch.float32, device=device)

    centers = []
    for view in views:
        pose = view.get("camera_pose", None)
        if pose is None:
            continue
        if not torch.is_tensor(pose):
            pose = torch.as_tensor(pose)
        centers.append(pose.to(device=views[0]["img"].device, dtype=torch.float32)[0, :3, 3])

    if not centers:
        device = views[0]["img"].device
        return torch.zeros(3, dtype=torch.float32, device=device)

    stacked = torch.stack(centers, dim=0)
    if mode == "first_camera":
        return stacked[0]
    if mode == "mean_camera":
        return stacked.mean(dim=0)
    raise ValueError(f"Unsupported recenter mode: {mode!r}")


def _has_camera_pose_priors(views: Sequence[Dict[str, object]]) -> bool:
    return bool(views) and all("camera_pose" in view for view in views)


def _recenter_torch_views_inplace(
    views: Sequence[Dict[str, object]],
    anchor: torch.Tensor,
) -> None:
    """Subtract anchor from world-frame tensors consumed by the model.

    The model wrapper uses camera_pose to derive world_translation/world_rotation
    priors, so camera_pose is the critical field. The other fields are updated to
    keep the view dictionary internally consistent and useful for debugging.
    """
    if not views:
        return
    device = views[0]["img"].device
    anchor = anchor.to(device=device, dtype=torch.float32)

    for view in views:
        if "camera_pose" in view:
            pose = view["camera_pose"]
            pose = pose.clone()
            pose[..., :3, 3] = pose[..., :3, 3] - anchor.view(1, 3).to(pose.device, pose.dtype)
            view["camera_pose"] = pose

        if "camera_pose_trans" in view:
            trans = view["camera_pose_trans"]
            view["camera_pose_trans"] = trans - anchor.view(1, 3).to(trans.device, trans.dtype)

        if "world_translation" in view:
            trans = view["world_translation"]
            view["world_translation"] = trans - anchor.view(1, 3).to(trans.device, trans.dtype)

        if "pts3d" in view:
            pts = view["pts3d"]
            view["pts3d"] = pts - anchor.view(1, 1, 1, 3).to(pts.device, pts.dtype)


def _add_translation_to_points(points: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points
    return (points.astype(np.float64) + translation.reshape(1, 3).astype(np.float64)).astype(np.float32)


def _add_translation_to_point_maps(point_maps: Sequence[np.ndarray], translation: np.ndarray) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for point_map in point_maps:
        if point_map.size == 0:
            out.append(point_map)
            continue
        shape = point_map.shape
        out.append(_add_translation_to_points(point_map.reshape(-1, 3), translation).reshape(shape))
    return out


def _add_translation_to_cameras(cams: Sequence[Dict[str, object]], translation: np.ndarray) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    for cam in cams:
        T = np.asarray(cam["T_c2w"], dtype=np.float64)
        T_out = T.copy().astype(np.float32)
        T_out[:3, 3] = (T[:3, 3] + t).astype(np.float32)
        out.append({**cam, "T_c2w": T_out})
    return out


def _restore_predictions_to_original_frame(
    pred_points: np.ndarray,
    pred_maps: Sequence[np.ndarray],
    pred_cams: Sequence[Dict[str, object]],
    anchor_np: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray], List[Dict[str, object]]]:
    pred_points_out = _add_translation_to_points(pred_points, anchor_np)
    pred_maps_out = _add_translation_to_point_maps(pred_maps, anchor_np)
    pred_cams_out = _add_translation_to_cameras(pred_cams, anchor_np)
    return pred_points_out, pred_maps_out, pred_cams_out


def _append_recenter_sidecar(output_rrd: str, recenter_meta: Dict[str, object]) -> None:
    sidecar = Path(output_rrd).expanduser().resolve().with_suffix(".json")
    if not sidecar.exists():
        return
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] failed to read sidecar metadata for recenter update: {exc}")
        return
    payload["recenter"] = recenter_meta
    sidecar.write_text(json.dumps(base.json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated sidecar recenter metadata: {sidecar}")


@torch.no_grad()
def main() -> None:
    recenter_args, remaining_argv = parse_recenter_args(sys.argv[1:])

    # Let the original script own all existing CLI arguments.  We temporarily
    # remove only this wrapper's recenter flags so base.parse_args remains strict.
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining_argv]
        args = base.parse_args()
    finally:
        sys.argv = old_argv

    device = base.resolve_device(args.device)
    views, meta = base.build_views_from_scene(args, device=device)

    gt_cams = [
        {"stem": stem, "T_c2w": np.asarray(meta["cams"][stem]["T_c2w"], dtype=np.float32)}
        for stem in meta["stems"]
        if stem in meta["cams"]
    ]

    recenter_mode = str(recenter_args.recenter)
    recenter_requested = recenter_mode != "none"
    can_recenter = _has_camera_pose_priors(views)
    if recenter_requested and can_recenter:
        anchor = _torch_anchor_from_views(views, recenter_mode)
        _recenter_torch_views_inplace(views, anchor)
        anchor_np = anchor.detach().cpu().numpy().astype(np.float32)
        print(
            f"Recentering model input: mode={recenter_mode}, "
            f"anchor=[{anchor_np[0]:.6g}, {anchor_np[1]:.6g}, {anchor_np[2]:.6g}]"
        )
    else:
        anchor_np = np.zeros(3, dtype=np.float32)
        if recenter_requested and not can_recenter:
            print("[WARN] --recenter requested but not all views have camera_pose; using raw coordinates")

    recenter_meta = {
        "requested_mode": recenter_mode,
        "applied": bool(recenter_requested and can_recenter),
        "anchor_translation": anchor_np.tolist(),
        "restore_policy": "predicted points, point maps, and camera centers are translated back before alignment/logging",
        "rotation_changed": False,
        "scale_changed": False,
    }

    model, _ = base.init_model_from_hydra(
        model_name=args.model,
        machine=args.machine,
        hydra_overrides=args.hydra_override,
        device=device,
    )
    base.load_optional_checkpoint(model, args.checkpoint)
    model.eval()

    print(f"Running model={args.model} on {len(views)} recentered views ...")
    preds = model(views)

    raw_pred_points, raw_pred_colors, pred_maps, pred_valid_masks, raw_pred_cams = base.collect_pred_outputs(
        preds=preds,
        rgbs=meta["rgbs"],
        args=args,
        stems=meta["stems"],
    )

    if bool(recenter_meta["applied"]):
        raw_pred_points, pred_maps, raw_pred_cams = _restore_predictions_to_original_frame(
            pred_points=raw_pred_points,
            pred_maps=pred_maps,
            pred_cams=raw_pred_cams,
            anchor_np=anchor_np,
        )
        print("Restored raw predictions from recentered local frame to original scene coordinates")

    print(f"Raw prediction summary: points={raw_pred_points.shape[0]}, cameras={len(raw_pred_cams)}")

    pred_points, pred_colors, pred_maps_aligned, pred_cams, align_meta = base.estimate_and_apply_alignment(
        args=args,
        meta=meta,
        pred_points=raw_pred_points,
        pred_colors=raw_pred_colors,
        pred_maps=pred_maps,
        pred_valid_masks=pred_valid_masks,
        pred_cams=raw_pred_cams,
    )
    print(f"Logged prediction summary: points={pred_points.shape[0]}, cameras={len(pred_cams)}, align={align_meta['mode']}")

    base.save_rrd(
        args=args,
        meta=meta,
        gt_cams=gt_cams,
        pred_points=pred_points,
        pred_colors=pred_colors,
        pred_cams=pred_cams,
        raw_pred_points=raw_pred_points,
        raw_pred_colors=raw_pred_colors,
        raw_pred_cams=raw_pred_cams,
        align_meta=align_meta,
    )
    _append_recenter_sidecar(args.output_rrd, recenter_meta)


if __name__ == "__main__":
    main()
