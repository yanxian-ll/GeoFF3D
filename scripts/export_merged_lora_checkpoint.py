#!/usr/bin/env python3
"""
按 benchmark.py + Hydra overrides 的方式，导出 LoRA checkpoint 为可直接推理的 merged checkpoint。

示例：
python scripts/export_merged_lora_checkpoint.py \
  --input experiments/mapanything/training/mapa_finetuning_16v_6d_16ipg_2g_encoder_lora/checkpoint-best.pth \
  --output experiments/mapanything/training/mapa_finetuning_16v_6d_16ipg_2g_encoder_lora/checkpoint-best-merged.pth \
  --model mapanything_v1 \
  --model-task images_only \
  --set model.encoder.uses_torch_hub=false

说明：
- 模型构建方式直接参考 benchmark.py：
    cfg = Hydra compose(...)
    init_model(cfg.model.model_str, cfg.model.model_config, ...)
- 不再从 ckpt 读取 model_str / model_config
- LoRA 的 submodule_configs 仍从 ckpt['args'] 读取
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, List

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mapanything.models import init_model
from mapanything.peft.lora_utils import apply_lora_from_submodule_configs


def _try_merge_lora_inplace(module: torch.nn.Module) -> torch.nn.Module:
    if hasattr(module, "merge_and_unload") and callable(module.merge_and_unload):
        return module.merge_and_unload()

    for name, child in list(module.named_children()):
        merged_child = _try_merge_lora_inplace(child)
        if merged_child is not child:
            setattr(module, name, merged_child)
    return module


def _get_cfg_attr(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _has_lora_keys(state_dict: dict[str, Any]) -> bool:
    return any(
        ("lora_A" in k) or ("lora_B" in k) or ("base_layer" in k)
        for k in state_dict.keys()
    )


def _resolve_configs_dir() -> Path:
    """
    假设脚本放在 repo_root/scripts 下，对应 configs 在 repo_root/configs。
    如果你把脚本放到别处，可以手动传 --configs_dir。
    """
    return (Path(__file__).resolve().parent.parent / "configs").resolve()


def _build_cfg_from_hydra(configs_dir: Path, model: str, model_task: str | None, extra_sets: List[str]):
    overrides = [f"model={model}"]
    if model_task:
        overrides.append(f"model/task={model_task}")
    overrides.extend(extra_sets)

    with initialize_config_dir(version_base=None, config_dir=str(configs_dir)):
        cfg = compose(config_name="dense_n_view_benchmark", overrides=overrides)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="experiments/mapanything/training/mapa_finetuning_16v_6d_16ipg_2g_full_lora/checkpoint-best.pth",
        help="输入 checkpoint 路径",
    )
    parser.add_argument(
        "--output",
        default="experiments/mapanything/training/mapa_finetuning_16v_6d_16ipg_2g_full_lora/checkpoint-best-merged.pth",
        help="输出 merged checkpoint 路径",
    )
    parser.add_argument(
        "--model",
        default="mapanything_v1",
        help="对应 benchmark 中的 model=...，例如 mapanything_v1",
    )
    parser.add_argument(
        "--model-task",
        default="images_only",
        help="对应 benchmark 中的 model/task=...，例如 images_only",
    )
    parser.add_argument(
        "--set",
        dest="extra_sets",
        action="append",
        default=[],
        help="额外的 Hydra override，可重复传入，例如 --set model.encoder.uses_torch_hub=false",
    )
    parser.add_argument(
        "--configs_dir",
        default="configs",
        help="configs 目录路径；默认自动取 repo_root/configs",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="加载 checkpoint 时使用 strict=True，默认 False",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    if "args" not in ckpt or "model" not in ckpt:
        raise KeyError("checkpoint 必须包含 'args' 和 'model' 字段")

    train_args = ckpt["args"]
    submodule_configs = _get_cfg_attr(train_args, "train_params.submodule_configs")

    configs_dir = Path(args.configs_dir).resolve() if args.configs_dir else _resolve_configs_dir()
    if not configs_dir.exists():
        raise FileNotFoundError(f"configs_dir 不存在: {configs_dir}")

    cfg = _build_cfg_from_hydra(
        configs_dir=configs_dir,
        model=args.model,
        model_task=args.model_task,
        extra_sets=args.extra_sets,
    )

    print("[INFO] Final Hydra cfg.model =")
    print(OmegaConf.to_yaml(cfg.model))

    from mapanything.utils.torch_hub_setup import configure_torch_hub
    configure_torch_hub(cfg.machine)

    print(f"[INFO] Rebuilding model from Hydra config: model={args.model}")
    model = init_model(
        cfg.model.model_str,
        cfg.model.model_config,
        torch_hub_force_reload=False,
    )

    print("[INFO] Applying LoRA wrappers from checkpoint train_params.submodule_configs ...")
    model = apply_lora_from_submodule_configs(model, submodule_configs)

    incompatible = model.load_state_dict(ckpt["model"], strict=args.strict)
    print(f"[INFO] load_state_dict(strict={args.strict}) => {incompatible}")

    print("[INFO] Merging LoRA weights ...")
    merged_model = _try_merge_lora_inplace(copy.deepcopy(model).cpu())
    merged_state_dict = merged_model.state_dict()

    if _has_lora_keys(merged_state_dict):
        raise RuntimeError("导出失败：merged 后的 state_dict 仍包含 LoRA 专属键")

    out_ckpt = {
        "args": train_args,
        "model": merged_state_dict,
        "epoch": ckpt.get("epoch", None),
    }
    if "best_so_far" in ckpt:
        out_ckpt["best_so_far"] = ckpt["best_so_far"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, output_path)
    print(f"[OK] Saved merged checkpoint to: {output_path}")


if __name__ == "__main__":
    main()

