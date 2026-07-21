#!/usr/bin/env python
"""Run one VGGT chunk and save standardized prediction tensors for inspection."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

import numpy as np

from slam.core.data_types import Chunk
from slam.io.folder_dataset import load_folder_frames
from slam.models.vggt_adapter import VGGTAdapter


class MockVGGTModel:
    """Small deterministic VGGT-like model used for smoke tests.

    This class is intentionally local to the debug script so the script can be
    exercised in CI without a GPU, checkpoint, or external VGGT dependency.
    """

    def __call__(self, views):
        n = len(views)
        return {
            "extrinsics": np.repeat(np.eye(3, 4, dtype=np.float32)[None], n, axis=0),
            "camera_intrinsics": np.repeat(np.eye(3, dtype=np.float32)[None], n, axis=0),
            "depth": np.ones((n, 2, 2, 1), dtype=np.float32),
            "point_maps": np.ones((n, 2, 2, 3), dtype=np.float32),
            "conf": np.ones((n, 2, 2, 1), dtype=np.float32),
            "tracks": {"mock": True, "num_views": n},
        }


def _array_or_empty(value, shape):
    if value is None:
        return np.empty(shape, dtype=np.float32)
    return np.asarray(value)


def _shape(value):
    return None if value is None else list(np.asarray(value).shape)


def save_prediction_npz(pred, output_npz):
    np.savez(
        output_npz,
        frame_ids=np.asarray(pred.frame_ids, dtype=np.int64),
        T_model_cam=_array_or_empty(pred.T_model_cam, (0, 4, 4)),
        intrinsics=_array_or_empty(pred.intrinsics, (0, 3, 3)),
        depth=_array_or_empty(pred.depth, (0,)),
        points_model=_array_or_empty(pred.points_model, (0, 3)),
        confidence=_array_or_empty(pred.confidence, (0,)),
    )


def save_summary_json(pred, output_json):
    summary = {
        "frame_ids": list(pred.frame_ids),
        "coord_type": pred.coord_type,
        "scale_type": pred.scale_type,
        "model_name": pred.model_name,
        "shapes": {
            "T_model_cam": _shape(pred.T_model_cam),
            "intrinsics": _shape(pred.intrinsics),
            "depth": _shape(pred.depth),
            "points_model": _shape(pred.points_model),
            "confidence": _shape(pred.confidence),
        },
        "diagnostics": pred.diagnostics,
        "has_tracks": pred.tracks is not None,
    }
    Path(output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_adapter(args):
    if args.mock_model:
        return VGGTAdapter(model=MockVGGTModel(), device=None)
    hydra_overrides = []
    if args.model_checkpoint is not None:
        checkpoint = Path(args.model_checkpoint)
        pretrained_location = checkpoint.parent if checkpoint.is_file() else checkpoint
        hydra_overrides.append(f"model.model_config.pretrained_model_name_or_path={pretrained_location}")
    return VGGTAdapter(
        auto_load=True,
        device=args.device,
        model_name=args.model_name,
        machine=args.machine,
        hydra_overrides=hydra_overrides,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one VGGT chunk and save standardized outputs")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", default="vggt")
    parser.add_argument("--machine", default="default")
    parser.add_argument("--model_checkpoint", default=None)
    parser.add_argument("--mock_model", action="store_true", help="Use a deterministic mock VGGT-like model for CI smoke tests")
    args = parser.parse_args(argv)

    frames = load_folder_frames(args.image_dir, max_frames=args.num_frames)
    chunk = Chunk(chunk_id=0, frames=frames)
    adapter = build_adapter(args)
    pred = adapter.infer(chunk)

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    save_prediction_npz(pred, output_npz)
    if args.output_json is not None:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        save_summary_json(pred, output_json)

    print(f"saved_npz: {output_npz}")
    if args.output_json is not None:
        print(f"saved_json: {args.output_json}")
    print(f"frame_ids: {pred.frame_ids}")
    print(f"coord_type: {pred.coord_type}")
    print(f"scale_type: {pred.scale_type}")
    print(f"T_model_cam: {_shape(pred.T_model_cam)}")
    print(f"intrinsics: {_shape(pred.intrinsics)}")
    print(f"depth: {_shape(pred.depth)}")
    print(f"points_model: {_shape(pred.points_model)}")
    print(f"confidence: {_shape(pred.confidence)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
