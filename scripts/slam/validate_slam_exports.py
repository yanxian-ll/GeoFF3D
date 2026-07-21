#!/usr/bin/env python
"""Validate Generic Feed-forward SLAM export folders."""

from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import json
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_FILES = [
    "summary.json",
    "export_manifest.json",
    "trajectory.npz",
    "points.npy",
    "points.ply",
    "trajectory.csv",
    "trajectory_tum.txt",
    "cameras.json",
    "colmap_text/cameras.txt",
    "colmap_text/images.txt",
    "colmap_text/points3D.txt",
]


def validate_export_dir(output_dir, *, require_prediction_arrays=True, strict_optional_tools=False):
    output_dir = Path(output_dir)
    report = {
        "output_dir": str(output_dir),
        "ok": True,
        "errors": [],
        "warnings": [],
        "counts": {},
        "optional_tools": _optional_tool_status(),
    }
    if not output_dir.is_dir():
        _error(report, f"Output directory does not exist: {output_dir}")
        return report

    summary = _read_json(output_dir / "summary.json", report, "summary")
    manifest = _read_json(output_dir / "export_manifest.json", report, "manifest")
    for relpath in REQUIRED_FILES:
        if relpath == "trajectory.npz" and not (output_dir / relpath).exists():
            explicit_npz = _manifest_file(manifest, "trajectory_npz")
            if explicit_npz is not None and Path(explicit_npz).exists():
                continue
        _require_file(output_dir / relpath, report)

    _validate_summary(summary, report)
    _validate_manifest(manifest, report)
    _validate_npz(output_dir, manifest, report)
    _validate_points(output_dir, report)
    _validate_ply(output_dir, report)
    _validate_tum(output_dir, summary, report)
    _validate_cameras(output_dir, summary, report)
    _validate_colmap_text(output_dir, summary, report)
    _validate_prediction_arrays(output_dir, summary, report, required=require_prediction_arrays)
    _validate_chunk_predictions(output_dir, summary, report)

    if strict_optional_tools:
        for name, available in report["optional_tools"].items():
            if not available:
                _error(report, f"Optional tool/module is unavailable: {name}")
    return report


def _validate_summary(summary, report):
    if not isinstance(summary, dict):
        return
    for key in ("input_frames", "trajectory_frames", "world_prediction_chunks", "map_summary", "backend_diagnostics"):
        if key not in summary:
            _error(report, f"summary.json missing key: {key}")
    report["counts"]["input_frames"] = int(summary.get("input_frames", 0))
    report["counts"]["trajectory_frames"] = int(summary.get("trajectory_frames", 0))
    if summary.get("trajectory_frames", 0) <= 0:
        _error(report, "summary trajectory_frames must be positive")


def _validate_manifest(manifest, report):
    if not isinstance(manifest, dict):
        return
    if manifest.get("format_version") != 1:
        _error(report, "export_manifest.json format_version must be 1")
    exported = manifest.get("exported_files", {})
    if not isinstance(exported, dict):
        _error(report, "export_manifest.json exported_files must be a dict")
        return
    for key, value in exported.items():
        if isinstance(value, str) and not Path(value).exists():
            _error(report, f"manifest exported file for {key!r} does not exist: {value}")


def _validate_npz(output_dir, manifest, report):
    npz_path = output_dir / "trajectory.npz"
    explicit_npz = _manifest_file(manifest, "trajectory_npz")
    if not npz_path.exists() and explicit_npz is not None:
        npz_path = Path(explicit_npz)
    if not npz_path.exists():
        return
    try:
        data = np.load(npz_path)
    except Exception as exc:
        _error(report, f"Cannot load trajectory npz {npz_path}: {exc}")
        return
    for key in ("frame_ids", "T_world_cam", "points"):
        if key not in data:
            _error(report, f"trajectory npz missing array: {key}")
    if "T_world_cam" in data and data["T_world_cam"].ndim != 3:
        _error(report, "T_world_cam must have shape [N, 4, 4]")


def _validate_points(output_dir, report):
    path = output_dir / "points.npy"
    if not path.exists():
        return
    try:
        points = np.load(path)
    except Exception as exc:
        _error(report, f"Cannot load points.npy: {exc}")
        return
    if points.ndim != 2 or points.shape[1] != 3:
        _error(report, f"points.npy must have shape [N, 3], got {points.shape}")
    if not np.isfinite(points).all():
        _error(report, "points.npy contains non-finite values")
    report["counts"]["points"] = int(points.shape[0]) if points.ndim == 2 else 0


def _validate_ply(output_dir, report):
    path = output_dir / "points.ply"
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or lines[0] != "ply":
        _error(report, "points.ply does not start with ply header")
        return
    vertex_lines = [line for line in lines[:20] if line.startswith("element vertex ")]
    if not vertex_lines:
        _error(report, "points.ply missing element vertex header")


def _validate_tum(output_dir, summary, report):
    path = output_dir / "trajectory_tum.txt"
    if not path.exists():
        return
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    expected = int(summary.get("trajectory_frames", 0)) if isinstance(summary, dict) else None
    if expected is not None and len(rows) != expected:
        _error(report, f"trajectory_tum.txt row count {len(rows)} != trajectory_frames {expected}")
    for idx, row in enumerate(rows[:5]):
        if len(row.split()) != 8:
            _error(report, f"trajectory_tum.txt row {idx} must have 8 columns")


def _validate_cameras(output_dir, summary, report):
    payload = _read_json(output_dir / "cameras.json", report, "cameras")
    if not isinstance(payload, dict):
        return
    cameras = payload.get("cameras")
    if not isinstance(cameras, list):
        _error(report, "cameras.json missing cameras list")
        return
    expected = int(summary.get("trajectory_frames", 0)) if isinstance(summary, dict) else None
    if expected is not None and len(cameras) != expected:
        _error(report, f"cameras.json camera count {len(cameras)} != trajectory_frames {expected}")
    report["counts"]["cameras"] = len(cameras)


def _validate_colmap_text(output_dir, summary, report):
    colmap_dir = output_dir / "colmap_text"
    cameras_path = colmap_dir / "cameras.txt"
    images_path = colmap_dir / "images.txt"
    points_path = colmap_dir / "points3D.txt"
    if not (cameras_path.exists() and images_path.exists() and points_path.exists()):
        return
    expected = int(summary.get("trajectory_frames", 0)) if isinstance(summary, dict) else None
    image_rows = [
        line
        for line in images_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    pose_rows = image_rows
    if expected is not None and len(pose_rows) != expected:
        _error(report, f"COLMAP images.txt pose rows {len(pose_rows)} != trajectory_frames {expected}")
    if not cameras_path.read_text(encoding="utf-8").startswith("# Camera list"):
        _error(report, "COLMAP cameras.txt has unexpected header")
    if not points_path.read_text(encoding="utf-8").startswith("# 3D point list"):
        _error(report, "COLMAP points3D.txt has unexpected header")


def _validate_prediction_arrays(output_dir, summary, report, required):
    index_path = output_dir / "prediction_arrays.json"
    if not index_path.exists():
        if required:
            _error(report, "prediction_arrays.json is missing")
        return
    index = _read_json(index_path, report, "prediction_arrays")
    if not isinstance(index, dict):
        return
    expected = int(summary.get("trajectory_frames", 0)) if isinstance(summary, dict) else None
    if expected is not None and len(index) != expected:
        _error(report, f"prediction_arrays entries {len(index)} != trajectory_frames {expected}")
    for frame_id, entry in index.items():
        if not isinstance(entry, dict):
            _error(report, f"prediction_arrays entry {frame_id} is not a dict")
            continue
        for key in ("depth", "confidence"):
            path = entry.get(key)
            if path is None:
                _warning(report, f"prediction_arrays entry {frame_id} missing {key}")
                continue
            try:
                arr = np.load(path)
            except Exception as exc:
                _error(report, f"Cannot load {key} array for frame {frame_id}: {exc}")
                continue
            if arr.ndim < 2:
                _error(report, f"{key} array for frame {frame_id} must be at least 2D")
            if not np.isfinite(arr).all():
                _error(report, f"{key} array for frame {frame_id} contains non-finite values")
        mask_path = entry.get("valid_mask")
        if mask_path is not None:
            mask = np.load(mask_path)
            if mask.dtype != np.bool_:
                _error(report, f"valid_mask array for frame {frame_id} must be bool")
    report["counts"]["prediction_arrays"] = len(index)


def _validate_chunk_predictions(output_dir, summary, report):
    index_path = output_dir / "chunk_predictions" / "index.json"
    if not index_path.exists():
        _warning(report, "chunk_predictions/index.json is missing")
        return
    index = _read_json(index_path, report, "chunk_predictions")
    if not isinstance(index, dict):
        return
    if index.get("format_version") != 1:
        _error(report, "chunk_predictions/index.json format_version must be 1")
    chunks = index.get("chunks")
    if not isinstance(chunks, list):
        _error(report, "chunk_predictions/index.json missing chunks list")
        return
    expected = int(summary.get("world_prediction_chunks", 0)) if isinstance(summary, dict) else None
    if expected is not None and len(chunks) != expected:
        _error(report, f"chunk_predictions entries {len(chunks)} != world_prediction_chunks {expected}")
    for chunk_idx, entry in enumerate(chunks):
        if not isinstance(entry, dict):
            _error(report, f"chunk_predictions entry {chunk_idx} is not a dict")
            continue
        packages = entry.get("packages")
        if not isinstance(packages, dict):
            _error(report, f"chunk_predictions entry {chunk_idx} missing packages dict")
            continue
        for name in ("world", "raw"):
            path = packages.get(name)
            if path is None:
                _warning(report, f"chunk_predictions entry {chunk_idx} missing {name} package")
                continue
            _validate_prediction_npz(Path(path), report, f"chunk {chunk_idx} {name}")
        local_path = packages.get("local")
        if local_path is not None:
            _validate_prediction_npz(Path(local_path), report, f"chunk {chunk_idx} local")
    report["counts"]["chunk_predictions"] = len(chunks)


def _validate_prediction_npz(path, report, label):
    if not path.exists():
        _error(report, f"{label} package is missing: {path}")
        return
    try:
        data = np.load(path)
    except Exception as exc:
        _error(report, f"Cannot load {label} package {path}: {exc}")
        return
    if "frame_ids" not in data:
        _error(report, f"{label} package missing frame_ids")
    if label.endswith("world"):
        if "points_world" not in data and "T_world_cam" not in data:
            _error(report, f"{label} package must contain points_world or T_world_cam")
    elif "points_model" not in data and "T_model_cam" not in data:
        _error(report, f"{label} package must contain points_model or T_model_cam")


def _optional_tool_status():
    return {
        "open3d": importlib.util.find_spec("open3d") is not None,
        "rerun": importlib.util.find_spec("rerun") is not None,
        "evo": importlib.util.find_spec("evo") is not None,
        "pycolmap": importlib.util.find_spec("pycolmap") is not None,
    }


def _manifest_file(manifest, key):
    if not isinstance(manifest, dict):
        return None
    exported = manifest.get("exported_files", {})
    if not isinstance(exported, dict):
        return None
    value = exported.get(key)
    return value if isinstance(value, str) else None


def _read_json(path, report, label):
    path = Path(path)
    if not path.exists():
        _error(report, f"Missing {label} json: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _error(report, f"Cannot parse {label} json {path}: {exc}")
        return None


def _require_file(path, report):
    if not Path(path).exists():
        _error(report, f"Missing required file: {path}")


def _error(report, message):
    report["ok"] = False
    report["errors"].append(str(message))


def _warning(report, message):
    report["warnings"].append(str(message))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report_json", default=None)
    parser.add_argument("--no_prediction_arrays", action="store_true")
    parser.add_argument("--strict_optional_tools", action="store_true")
    args = parser.parse_args(argv)

    report = validate_export_dir(
        args.output_dir,
        require_prediction_arrays=not args.no_prediction_arrays,
        strict_optional_tools=args.strict_optional_tools,
    )
    text = json.dumps(report, indent=2)
    if args.report_json is not None:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
