#!/usr/bin/env python3
"""Convert old DA3-train checkpoints to the new namespace.

Old training wrapper saved keys like:
    model.backbone.pretrained.cls_token

New training wrapper (self.model = api) saves / expects:
    model.model.backbone.pretrained.cls_token

This script updates checkpoint['model'] while preserving other fields such as
optimizer, loss_scaler, epoch, args, etc., so the output can be used for
continued training with the modified wrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


OLD_PREFIX = "model."
NEW_PREFIX = "model.model."


def remap_state_dict_keys(state_dict: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    converted = {}
    changed = 0
    skipped = 0

    for k, v in state_dict.items():
        if k.startswith(NEW_PREFIX):
            converted[k] = v
            skipped += 1
        elif k.startswith(OLD_PREFIX):
            converted[k.replace(OLD_PREFIX, NEW_PREFIX, 1)] = v
            changed += 1
        else:
            converted[k] = v

    return converted, changed, skipped


def load_checkpoint(path: Path, map_location: str = "cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def save_checkpoint(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to old checkpoint (.pth)")
    parser.add_argument("--output", required=True, help="Path to converted checkpoint (.pth)")
    parser.add_argument(
        "--raw-state-dict",
        action="store_true",
        help="Treat input as a raw state_dict instead of a full training checkpoint dict.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    ckpt = load_checkpoint(input_path)

    if args.raw_state_dict:
        if not isinstance(ckpt, dict):
            raise TypeError("Raw-state-dict mode expects the loaded object to be a dict.")
        new_state_dict, changed, skipped = remap_state_dict_keys(ckpt)
        save_checkpoint(new_state_dict, output_path)
        print(f"[done] raw state_dict converted: {input_path} -> {output_path}")
        print(f"[info] changed={changed}, already_new={skipped}, total={len(new_state_dict)}")
        return

    if not isinstance(ckpt, dict):
        raise TypeError(
            "Expected a full checkpoint dict. Use --raw-state-dict if the input is only a state_dict."
        )

    if "model" not in ckpt:
        raise KeyError("The checkpoint does not contain a 'model' field.")
    if not isinstance(ckpt["model"], dict):
        raise TypeError("checkpoint['model'] must be a state_dict-like dict.")

    new_ckpt = dict(ckpt)
    new_model, changed, skipped = remap_state_dict_keys(ckpt["model"])
    new_ckpt["model"] = new_model

    save_checkpoint(new_ckpt, output_path)

    print(f"[done] checkpoint converted: {input_path} -> {output_path}")
    print(f"[info] changed={changed}, already_new={skipped}, total={len(new_model)}")
    if changed == 0:
        print("[warn] No keys were converted. The checkpoint may already be in the new format.")


if __name__ == "__main__":
    main()

