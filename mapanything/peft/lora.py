"""
https://github.com/Visual-AI/Fin3R

"""
import types

import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer


def renormalized_forward_qkv(self, x):
    """Fin3R custom forward for qkv LoRA with q/k/v re-normalization."""
    original_output = self.base_layer(x)

    if (
        getattr(self, "merged", False)
        or not hasattr(self, "active_adapters")
        or not self.active_adapters
    ):
        return original_output

    lora_output = 0
    for adapter_name in self.active_adapters:
        lora_A = self.lora_A[adapter_name]
        lora_B = self.lora_B[adapter_name]
        scaling = self.scaling[adapter_name]
        lora_output += scaling * (x @ lora_A.weight.T @ lora_B.weight.T)

    combined_output = original_output + lora_output
    hidden_size = original_output.shape[-1] // 3

    q_orig = original_output[..., :hidden_size]
    k_orig = original_output[..., hidden_size : 2 * hidden_size]
    v_orig = original_output[..., 2 * hidden_size :]

    q_comb = combined_output[..., :hidden_size]
    k_comb = combined_output[..., hidden_size : 2 * hidden_size]
    v_comb = combined_output[..., 2 * hidden_size :]

    q_orig_norm = torch.norm(q_orig, dim=-1, keepdim=True)
    k_orig_norm = torch.norm(k_orig, dim=-1, keepdim=True)
    v_orig_norm = torch.norm(v_orig, dim=-1, keepdim=True)

    q_comb_norm = torch.norm(q_comb, dim=-1, keepdim=True)
    k_comb_norm = torch.norm(k_comb, dim=-1, keepdim=True)
    v_comb_norm = torch.norm(v_comb, dim=-1, keepdim=True)

    q_renorm = q_comb * (q_orig_norm / (q_comb_norm + 1e-6))
    k_renorm = k_comb * (k_orig_norm / (k_comb_norm + 1e-6))
    v_renorm = v_comb * (v_orig_norm / (v_comb_norm + 1e-6))

    return torch.cat([q_renorm, k_renorm, v_renorm], dim=-1)


def get_renormalized_peft_model(model, lora_config: LoraConfig):
    """Wrap a module with PEFT LoRA and patch qkv forward exactly as Fin3R does."""
    peft_model = get_peft_model(model, lora_config)

    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLayer) and hasattr(module, "base_layer"):
            if name.endswith(".qkv") or name.split(".")[-1] == "qkv":
                module.forward = types.MethodType(renormalized_forward_qkv, module)

    return peft_model


def mark_only_lora_trainable(module):
    """Extra safety: keep only LoRA parameters trainable inside a wrapped module."""
    for name, param in module.named_parameters():
        param.requires_grad = "lora_" in name
