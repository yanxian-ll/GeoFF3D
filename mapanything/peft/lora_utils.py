from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Mapping, Any
import copy
from pathlib import Path
from peft import LoraConfig

from mapanything.peft.lora import (
    get_renormalized_peft_model,
    mark_only_lora_trainable,
)
from mapanything.utils.train_tools import save_on_master

def _get_submodule(root, path: str):
    module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module


def _set_submodule(root, path: str, new_module):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _has_enabled_lora(submodule_configs: Mapping, lora_default: Mapping | None = None) -> bool:
    default_enabled = False
    if lora_default is not None:
        default_enabled = bool(getattr(lora_default, "get", lambda *_: None)("enabled", False))

    for cfg in submodule_configs.values():
        lora_cfg = cfg.get("lora", None)
        if lora_cfg and lora_cfg.get("enabled", default_enabled):
            return True
    return False


def _discover_target_module_names(module, target_modules: list[str]) -> list[str]:
    matched = []
    for name, submodule in module.named_modules():
        if not name:
            continue
        leaf = name.split(".")[-1]
        if leaf in target_modules:
            matched.append(name)
    return matched


def _summarize_trainable_parameters(model) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _to_plain_dict(cfg: Mapping | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    return {k: cfg[k] for k in cfg.keys()}


def _merge_lora_cfg(lora_default: Mapping | None, lora_cfg: Mapping | None) -> dict[str, Any]:
    merged = {}
    merged.update(_to_plain_dict(lora_default))
    merged.update(_to_plain_dict(lora_cfg))
    return merged


def apply_lora_from_submodule_configs(model, submodule_configs, lora_default=None):
    """
    Apply Fin3R-style renormalized LoRA according to train_params.submodule_configs.

    The submodule selection logic follows the same prefix-based grouping idea used by
    the LR configuration in train_tools.py, while the LoRA implementation itself
    follows Fin3R's qkv-only renormalized LoRA.

    Example:
    submodule_configs:
      aggregator.patch_embed:
        lr: 1e-4
        lora:
          enabled: true

    lora_default:
      impl: fin3r
      r: 8
      alpha: 8
      dropout: 0.1
      target_modules: ["qkv"]
      bias: "none"
      freeze_base_params: true
    """

    if submodule_configs is None:
        return model

    if not _has_enabled_lora(submodule_configs, lora_default=lora_default):
        return model

    # A prefix is considered a LoRA prefix if either the local config enables LoRA,
    # or the local config provides a lora section and the global default says enabled.
    lora_prefixes = []
    default_enabled = bool(_to_plain_dict(lora_default).get("enabled", False))
    for prefix, cfg in submodule_configs.items():
        local_lora = cfg.get("lora", None)
        if local_lora is not None and local_lora.get("enabled", default_enabled):
            lora_prefixes.append(prefix)

    # keep only the most specific configured prefixes
    lora_prefixes = sorted(lora_prefixes, key=len, reverse=True)
    kept = []
    for prefix in lora_prefixes:
        if not any(existing == prefix or existing.startswith(prefix + ".") for existing in kept):
            kept.append(prefix)
    lora_prefixes = sorted(kept, key=len)

    if not lora_prefixes:
        return model

    for prefix in lora_prefixes:
        cfg = deepcopy(submodule_configs[prefix])
        lora_cfg = _merge_lora_cfg(lora_default, cfg.get("lora", {}))

        if not lora_cfg.get("enabled", False):
            continue

        target = _get_submodule(model, prefix)
        target_modules = list(lora_cfg.get("target_modules", ["qkv"]))
        matched_targets = _discover_target_module_names(target, target_modules)

        if not matched_targets:
            sample_names = []
            for name, _ in list(target.named_modules())[:30]:
                if name:
                    sample_names.append(name)
            print(
                f"No target modules {target_modules} found under submodule {prefix!r}. "
                f"First discovered module names: {sample_names[:15]}"
            )
            continue

        peft_cfg = LoraConfig(
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 8),
            target_modules=target_modules,
            lora_dropout=lora_cfg.get("dropout", 0.1),
            bias=lora_cfg.get("bias", "none"),
        )

        wrapped = get_renormalized_peft_model(target, peft_cfg)

        if lora_cfg.get("freeze_base_params", True):
            mark_only_lora_trainable(wrapped)

        _set_submodule(model, prefix, wrapped)
        print(
            f"[LoRA] Applied Fin3R LoRA to '{prefix}' with targets={target_modules}, "
            f"matched={matched_targets[:8]}{'...' if len(matched_targets) > 8 else ''}"
        )

    trainable, total = _summarize_trainable_parameters(model)
    pct = 100.0 * trainable / max(total, 1)
    print(f"[LoRA] Trainable params after injection: {trainable}/{total} ({pct:.4f}%)")
    return model


def has_enabled_lora_from_train_params(train_params) -> bool:
    submodule_configs = getattr(train_params, "submodule_configs", None)
    lora_default = getattr(train_params, "lora_default", None)

    if submodule_configs is None:
        return False

    return _has_enabled_lora(
        submodule_configs=submodule_configs,
        lora_default=lora_default,
    )


def _try_merge_lora_inplace(module):
    """
    Recursively merge PEFT/LoRA modules in-place if supported.
    Returns the merged module.
    """
    # Case 1: current module itself supports merge_and_unload
    if hasattr(module, "merge_and_unload") and callable(module.merge_and_unload):
        try:
            return module.merge_and_unload()
        except Exception as e:
            print(f"[WARN] merge_and_unload failed on {type(module)}: {e}")

    # Case 2: recurse into children
    for name, child in list(module.named_children()):
        merged_child = _try_merge_lora_inplace(child)
        if merged_child is not child:
            setattr(module, name, merged_child)

    return module


def get_merged_state_dict(model_without_ddp):
    """
    Return a merged plain-model state_dict if model contains LoRA wrappers.
    If merge is not needed or fails, return None.
    """
    try:
        model_cpu = copy.deepcopy(model_without_ddp).cpu()
    except Exception as e:
        print(f"[WARN] deepcopy model failed, skip merged export: {e}")
        return None

    try:
        merged_model = _try_merge_lora_inplace(model_cpu)
        state_dict = merged_model.state_dict()
        return state_dict
    except Exception as e:
        print(f"[WARN] failed to export merged state_dict: {e}")
        return None


def save_merged_checkpoint(args, epoch, model_without_ddp, fname, best_so_far=None):
    """
    Save an inference-ready merged checkpoint if LoRA can be merged.
    """
    merged_state_dict = get_merged_state_dict(model_without_ddp)
    if merged_state_dict is None:
        return

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f"checkpoint-{fname}-merged.pth"

    to_save = {
        "args": args,
        "model": merged_state_dict,
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far

    print(f">> Saving merged inference checkpoint to {checkpoint_path} ...")
    save_on_master(to_save, checkpoint_path)

