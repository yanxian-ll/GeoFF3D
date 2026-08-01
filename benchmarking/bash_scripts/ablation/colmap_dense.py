#!/usr/bin/env python3
"""Run the COLMAP dense reconstruction ablation and aligned evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Iterator, Tuple

import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from common.colmap_io import read_model


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class GpuMonitor:
    def __init__(self, device: str) -> None:
        self.device = str(device)
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(0.2):
            result = subprocess.run(
                ["nvidia-smi", "-i", self.device, "--query-compute-apps=used_gpu_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False,
            )
            values = []
            for line in result.stdout.splitlines():
                try:
                    values.append(float(line.strip()))
                except ValueError:
                    pass
            self.peak_mib = max(self.peak_mib, sum(values))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=Path(__file__).with_name("ablation_scenes.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "experiments/benchmarking/ablation",
    )
    parser.add_argument("--cuda-device", "--device", default="0")
    parser.add_argument("--colmap-bin", default=os.environ.get("COLMAP_BIN", "colmap"))
    parser.add_argument("--matcher", choices=("sequential", "exhaustive"), default=os.environ.get("COLMAP_MATCHER", "sequential"))
    parser.add_argument("--scratch-root", type=Path, default=Path(os.environ.get("COLMAP_SCRATCH_ROOT", f"/tmp/colmap-ablation-{os.environ.get('USER', 'user')}")))
    parser.add_argument("--max-image-size", type=int, default=518)
    parser.add_argument("--size-multiple", type=int, default=14)
    parser.add_argument("--max-num-features", type=int, default=8192)
    parser.add_argument("--max-rrd-points", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-metrics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def flatten_params(value: object) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict):
                out.update(flatten_params(item))
            else:
                out[str(key).lower().replace("-", "_")] = item
    return out


def iter_scenes(config: dict) -> Iterator[Tuple[str, str, Path, Dict[str, object]]]:
    for dataset in config.get("datasets", []):
        if not isinstance(dataset, dict) or not bool(dataset.get("enabled", True)):
            continue
        dataset_name = str(dataset.get("name", "default"))
        dataset_root = Path(str(dataset.get("root", "."))).expanduser()
        dataset_params = flatten_params(dataset.get("params", {}))
        for entry in dataset.get("scenes", []):
            if isinstance(entry, str):
                name, raw_path, scene_params = entry, Path(entry), {}
            elif isinstance(entry, dict):
                name = str(entry.get("name") or entry.get("path"))
                raw_path = Path(str(entry.get("path", name))).expanduser()
                scene_params = flatten_params(entry.get("params", {}))
            else:
                continue
            path = raw_path if raw_path.is_absolute() else dataset_root / raw_path
            yield dataset_name, name, path, {**dataset_params, **scene_params}


def run(command: list[str], *, env: dict | None = None, dry_run: bool = False) -> None:
    print("[CMD] " + " ".join(command))
    if not dry_run:
        subprocess.run(command, env=env, check=True)


def gpu_options(colmap: str, command: str, new_prefix: str, old_prefix: str) -> list[str]:
    help_text = subprocess.run([colmap, command, "-h"], capture_output=True, text=True, check=False).stdout
    prefix = new_prefix if f"--{new_prefix}.use_gpu" in help_text else old_prefix
    return [f"--{prefix}.use_gpu", "1", f"--{prefix}.gpu_index", "0"]


def prepare_images(scene_dir: Path, destination: Path, stride: int, max_side: int, multiple: int) -> None:
    import cv2

    sources = sorted(path for path in (scene_dir / "images").iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)[::stride]
    if not sources:
        raise RuntimeError(f"No images found under {scene_dir / 'images'}")
    reference = cv2.imread(str(sources[0]), cv2.IMREAD_COLOR)
    if reference is None:
        raise RuntimeError(f"Cannot read image: {sources[0]}")
    height, width = reference.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    target_h = max(multiple, int(height * scale) // multiple * multiple)
    target_w = max(multiple, int(width * scale) // multiple * multiple)
    destination.mkdir(parents=True, exist_ok=True)
    names = []
    for source in sources:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read image: {source}")
        if image.shape[:2] != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        target = destination / f"{source.stem}.png"
        if not cv2.imwrite(str(target), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise RuntimeError(f"Cannot write image: {target}")
        names.append(target.name)
    (destination.parent / "image_list.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def largest_sparse_model(root: Path) -> Path:
    candidates = []
    for path in root.iterdir():
        if not path.is_dir() or not (path / "images.bin").is_file():
            continue
        try:
            _cameras, images, _points = read_model(str(path), ext=".bin")
            candidates.append((len(images), path))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError(f"COLMAP produced no sparse model under {root}")
    return max(candidates, key=lambda item: item[0])[1]


def evaluate(root: Path, scene_dir: Path, eval_dir: Path, stride: int) -> None:
    script = root / "benchmarking/common/evaluate.py"
    run([
        sys.executable, str(script), "--scene_dir", str(scene_dir), "--eval_dir", str(eval_dir),
        "--output_json", str(eval_dir / "metrics.json"), "--output_csv", str(eval_dir / "metrics_summary.csv"),
        "--stride", str(stride), "--max_side", "518", "--size_multiple", "14",
        "--thresholds", "0.5,1.0,2.0,5.0", "--rpe_steps", "1,5,10",
        "--max_gt_points", "300000", "--max_pred_points_eval", "300000",
        "--max_gt_points_eval", "300000", "--max_align_points", "100000",
        "--icp_iterations", "8", "--icp_trim_quantile", "0.7", "--gt_io_workers", "4", "--seed", "0",
    ])


def process_scene(args: argparse.Namespace, root: Path, dataset: str, name: str, scene_dir: Path, params: Dict[str, object]) -> None:
    output = args.output_root / "09_colmap_dense" / dataset / name
    output_rrd = output.with_suffix(".rrd")
    if output_rrd.is_file() and (output / "eval/metrics.json").is_file() and not args.overwrite:
        print(f"[SKIP] {dataset}/{name}")
        return
    stride = int(params.get("stride", 1))
    if args.dry_run:
        print(f"[DRY_RUN] {dataset}/{name}: {scene_dir} -> {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    monitor = GpuMonitor(args.cuda_device)
    monitor.start()
    try:
        with tempfile.TemporaryDirectory(prefix=f"{dataset}-{name}-", dir=args.scratch_root) as temporary:
            work = Path(temporary)
            images = work / "images"
            prepare_images(scene_dir, images, stride, args.max_image_size, args.size_multiple)
            database, sparse, dense = work / "database.db", work / "sparse", output / "dense"
            sparse.mkdir()
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.cuda_device)}
            feature_gpu = gpu_options(args.colmap_bin, "feature_extractor", "FeatureExtraction", "SiftExtraction")
            match_gpu = gpu_options(args.colmap_bin, f"{args.matcher}_matcher", "FeatureMatching", "SiftMatching")
            run([args.colmap_bin, "feature_extractor", "--database_path", str(database), "--image_path", str(images), "--image_list_path", str(work / "image_list.txt"), "--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.single_camera", "1", *feature_gpu, "--SiftExtraction.max_num_features", str(args.max_num_features)], env=env)
            run([args.colmap_bin, f"{args.matcher}_matcher", "--database_path", str(database), *match_gpu], env=env)
            run([args.colmap_bin, "mapper", "--database_path", str(database), "--image_path", str(images), "--output_path", str(sparse)])
            model = largest_sparse_model(sparse)
            run([args.colmap_bin, "image_undistorter", "--image_path", str(images), "--input_path", str(model), "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", str(args.max_image_size)])
            run([args.colmap_bin, "patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.gpu_index", "0", "--PatchMatchStereo.max_image_size", str(args.max_image_size)], env=env)
            fused = dense / "fused.ply"
            run([args.colmap_bin, "stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP", "--input_type", "geometric", "--output_path", str(fused)])
            shutil.copy2(database, output / "database.db")
            adapter = root / "benchmarking/adapters/colmap_dense.py"
            run([sys.executable, str(adapter), "--scene-dir", str(scene_dir), "--sparse-model", str(model), "--fused-ply", str(fused), "--output-rrd", str(output_rrd), "--stride", str(stride), "--max-side", str(args.max_image_size), "--size-multiple", str(args.size_multiple), "--max-rrd-points", str(args.max_rrd_points), "--seed", "0"])
    finally:
        monitor.stop()
    timing = {"processing_time_seconds": time.perf_counter() - started, "num_chunks": 1, "peak_gpu_memory_allocated_mib": monitor.peak_mib}
    (output / "processing_time.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    if not args.no_metrics:
        evaluate(root, scene_dir, output / "eval", stride)


def main() -> int:
    args = parse_args()
    if shutil.which(args.colmap_bin) is None and not args.dry_run:
        raise RuntimeError(f"COLMAP executable not found: {args.colmap_bin}")
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(args.scene_list.read_text(encoding="utf-8"))
    for dataset, name, scene_dir, params in iter_scenes(config):
        if not scene_dir.is_dir():
            print(f"[WARN] Missing scene: {scene_dir}")
            continue
        process_scene(args, root, dataset, name, scene_dir.resolve(), params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
