#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run MASt3R-SfM and save dense point cloud outputs in benchmark format."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_vggt_slam_to_rrd import (
    align_prediction_to_gt_pose_sim3,
    json_safe,
    load_gt_artifacts,
    materialize_images,
    sample_points,
    save_final_eval_outputs,
    select_images,
    write_rrd,
)


def add_mast3r_paths(repo_root: Path) -> None:
    for rel in (
        "third_party/mast3r",
        "third_party/dust3r",
        "third_party/croco",
    ):
        path = repo_root / rel
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def resolve_device(device_arg: str) -> str:
    value = str(device_arg)
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.isdigit():
        return f"cuda:{value}"
    return value


def prepared_stem_map(prepared_metadata: Sequence[Dict[str, object]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for meta in prepared_metadata:
        prepared = Path(str(meta["prepared"]))
        stem = str(meta["stem"])
        out[prepared.name] = stem
        out[prepared.stem] = stem
        out[str(prepared)] = stem
    return out


def scene_to_dense_points(
    scene,
    *,
    min_conf_thr: float,
    clean_depth: bool,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from dust3r.utils.device import to_numpy

    pts3d, _depthmaps, confs = to_numpy(
        scene.get_dense_pts3d(clean_depth=bool(clean_depth))
    )
    rgbimgs = to_numpy(scene.imgs)

    point_parts: List[np.ndarray] = []
    color_parts: List[np.ndarray] = []
    for pts, conf, rgb in zip(pts3d, confs, rgbimgs):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        conf = np.asarray(conf).reshape(-1)
        rgb = np.asarray(rgb)
        if rgb.ndim == 3:
            rgb = rgb.reshape(-1, rgb.shape[-1])
        else:
            rgb = rgb.reshape(-1, 3)

        n = min(pts.shape[0], conf.shape[0], rgb.shape[0])
        if n <= 0:
            continue
        pts = pts[:n]
        conf = conf[:n]
        rgb = rgb[:n, :3]

        mask = (conf > float(min_conf_thr)) & np.isfinite(pts).all(axis=-1)
        if not bool(mask.any()):
            continue
        point_parts.append(pts[mask].astype(np.float32))
        color = np.clip(rgb[mask].reshape(-1, 3), 0.0, 1.0)
        color_parts.append(np.round(color * 255.0).astype(np.uint8))

    if not point_parts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    points = np.concatenate(point_parts, axis=0)
    colors = np.concatenate(color_parts, axis=0)
    return sample_points(points, colors, int(max_points), int(seed))


def scene_to_cameras(scene, prepared_metadata: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    from dust3r.utils.device import to_numpy

    name_map = prepared_stem_map(prepared_metadata)
    cam2w = to_numpy(scene.get_im_poses())
    cams: List[Dict[str, object]] = []
    for i, (path, T) in enumerate(zip(scene.img_paths, cam2w)):
        key = str(path)
        stem = name_map.get(key, name_map.get(Path(key).name, name_map.get(Path(key).stem, Path(key).stem)))
        T = np.asarray(T, dtype=np.float32)
        if T.shape == (4, 4) and np.isfinite(T).all():
            cams.append({"frame_id": float(i), "stem": stem, "T_c2w": T})
    return cams


def run_mast3r_sfm(
    *,
    image_paths: Sequence[Path],
    model_path: Path,
    retrieval_model_path: Optional[Path],
    output_dir: Path,
    device: str,
    image_size: int,
    scene_graph: str,
    prefilter: Optional[str],
    symmetrize: bool,
) -> object:
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.image_pairs import make_pairs
    from mast3r.model import AsymmetricMASt3R

    filelist = [str(p) for p in image_paths]
    print(f"[MASt3R-SfM] loading model: {model_path}")
    model = AsymmetricMASt3R.from_pretrained(str(model_path)).to(device)
    model.eval()

    imgs = load_images(
        filelist,
        size=int(image_size),
        verbose=True,
        patch_size=getattr(model, "patch_size", 16),
    )
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1
        filelist = [filelist[0], filelist[0] + "_2"]

    sim_matrix = None
    if scene_graph.startswith("retrieval"):
        if retrieval_model_path is None:
            raise RuntimeError("MASt3R-SfM retrieval scene graph requires --retrieval_model.")
        from mast3r.retrieval.processor import Retriever

        print(f"[MASt3R-SfM] computing retrieval graph with {retrieval_model_path}")
        retriever = Retriever(str(retrieval_model_path), backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)
        del retriever
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"[MASt3R-SfM] making image pairs with scene_graph={scene_graph}.")
    pairs = make_pairs(
        imgs,
        scene_graph=scene_graph,
        prefilter=prefilter,
        symmetrize=bool(symmetrize),
        sim_mat=sim_matrix,
    )
    print(f"[MASt3R-SfM] scene graph edges: {len(pairs)}")
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[MASt3R-SfM] running sparse_global_alignment with demo defaults.")
    scene = sparse_global_alignment(
        filelist,
        pairs,
        str(cache_dir),
        model,
        lr1=0.07,
        niter1=300,
        lr2=0.01,
        niter2=300,
        device=device,
        opt_depth=True,
        shared_intrinsics=False,
        matching_conf_thr=0.0,
    )
    return scene


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--output_rrd", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--cams_dir", default="cams")
    parser.add_argument("--depth_dir", default="depth")
    parser.add_argument("--frame_glob", default="*")
    parser.add_argument("--num_views", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_side", type=int, default=512)
    parser.add_argument("--size_multiple", type=int, default=16)
    parser.add_argument("--copy_images", action="store_true")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--depth_min", type=float, default=1e-6)
    parser.add_argument("--depth_max", type=float, default=1e6)
    parser.add_argument(
        "--model_path",
        default="checkpoints/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    )
    parser.add_argument(
        "--retrieval_model",
        default=None,
        help="MASt3R retrieval checkpoint. Required for retrieval-* scene graphs.",
    )
    parser.add_argument(
        "--scene_graph",
        default="complete",
        help="MASt3R scene graph, e.g. complete, swin-5-noncyclic, logwin-4, oneref-0, retrieval-20-1.",
    )
    parser.add_argument("--prefilter", default=None)
    parser.add_argument("--no_symmetrize", action="store_true")
    parser.add_argument("--max_pred_points", type=int, default=2000000)
    parser.add_argument("--max_gt_points", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_images", action="store_true")
    parser.add_argument("--view_coordinates", default="RDF")
    parser.add_argument("--background", type=int, nargs=3, default=(255, 255, 255))
    parser.add_argument("--hide_grid", action="store_true")
    parser.add_argument("--point_radius", type=float, default=0.0)
    parser.add_argument("--camera_axis_size", type=float, default=0.0)
    parser.add_argument("--camera_axis_radius", type=float, default=0.0)
    parser.add_argument("--show_world_axes", action="store_true", default=True)
    parser.add_argument("--no_world_axes", action="store_false", dest="show_world_axes")
    parser.add_argument("--world_axis_size", type=float, default=0.0)
    parser.add_argument("--world_axis_radius", type=float, default=0.0)
    return parser.parse_known_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, passthrough = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    add_mast3r_paths(repo_root)

    output_rrd = Path(args.output_rrd).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else output_rrd.with_suffix("")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stems, image_paths = select_images(args)
    prepared_dir = output_dir / "source" / "images"
    prepared_images, prepared_metadata = materialize_images(
        image_paths=image_paths,
        stems=stems,
        out_dir=prepared_dir,
        max_side=int(args.max_side),
        size_multiple=int(args.size_multiple),
        copy_images=bool(args.copy_images),
    )

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = repo_root / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Missing MASt3R checkpoint: {model_path}")
    retrieval_model_path = Path(args.retrieval_model) if args.retrieval_model else None
    if retrieval_model_path is not None:
        if not retrieval_model_path.is_absolute():
            retrieval_model_path = repo_root / retrieval_model_path
        if not retrieval_model_path.exists():
            raise FileNotFoundError(f"Missing MASt3R retrieval checkpoint: {retrieval_model_path}")

    device = resolve_device(args.device)
    t0 = time.perf_counter()
    scene = run_mast3r_sfm(
        image_paths=prepared_images,
        model_path=model_path,
        retrieval_model_path=retrieval_model_path,
        output_dir=output_dir / "mast3r_sfm",
        device=device,
        image_size=int(args.max_side) if int(args.max_side) > 0 else 512,
        scene_graph=str(args.scene_graph),
        prefilter=args.prefilter,
        symmetrize=not bool(args.no_symmetrize),
    )
    processing_time = {"processing_time_seconds": float(time.perf_counter() - t0)}

    points, colors = scene_to_dense_points(
        scene,
        min_conf_thr=1.5,
        clean_depth=True,
        max_points=int(args.max_pred_points),
        seed=int(args.seed),
    )
    cams = scene_to_cameras(scene, prepared_metadata)
    gt_cams, gt_points, gt_colors, gt_meta = load_gt_artifacts(
        args=args,
        stems=stems,
        image_paths=image_paths,
    )

    points, cams, align_meta = align_prediction_to_gt_pose_sim3(
        pred_points=points,
        pred_cams=cams,
        gt_cams=gt_cams,
    )

    save_final_eval_outputs(
        eval_dir=output_dir / "eval",
        pred_cams=cams,
        gt_cams=gt_cams,
        pred_points=points,
        pred_colors=colors,
        gt_points=gt_points,
        gt_colors=gt_colors,
        meta={
            "schema": "final_eval_v1",
            "script": "scripts/run_mast3r_sfm_to_rrd.py",
            "scene_dir": str(Path(args.scene_dir).expanduser().resolve()),
            "method": "mast3r-sfm",
            "method_display": "MASt3R-SfM",
            "pose_convention": "T_c2w",
            "points_coordinate": "same_as_pred_cameras",
            "processing_time": processing_time,
            "mast3r_sfm": {
                "model_path": str(model_path),
                "retrieval_model": str(retrieval_model_path) if retrieval_model_path else None,
                "scene_graph": str(args.scene_graph),
                "prefilter": args.prefilter,
                "symmetrize": not bool(args.no_symmetrize),
                "min_conf_thr": 1.5,
                "clean_depth": True,
            },
            "post_align": {
                "enabled": True,
                "type": "pose_sim3",
                "target": "gt_pose",
                **align_meta,
                "valid": bool(align_meta.get("valid", False)),
            },
        },
    )

    sidecar = output_rrd.with_suffix(".json")
    payload = {
        "method": "mast3r-sfm",
        "method_display": "MASt3R-SfM",
        "scene_dir": Path(args.scene_dir).expanduser().resolve(),
        "output_rrd": output_rrd,
        "output_dir": output_dir,
        "model_path": model_path,
        "stems": stems,
        "prepared_images": prepared_metadata,
        "num_poses": len(cams),
        "num_pred_points_logged": int(points.shape[0]),
        "num_gt_cameras": int(len(gt_cams)),
        "num_gt_points_logged": int(gt_points.shape[0]),
        "gt": gt_meta,
        "alignment": align_meta,
        "processing_time": processing_time,
    }
    sidecar.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved sidecar metadata: {sidecar}")

    write_rrd(
        args=args,
        method="mast3r-sfm",
        output_rrd=output_rrd,
        prepared_metadata=prepared_metadata,
        prepared_images=prepared_images,
        cams=cams,
        points=points,
        colors=colors,
        gt_cams=gt_cams,
        gt_points=gt_points,
        gt_colors=gt_colors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
