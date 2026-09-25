"""Utilities for loading Qwen and managing its LoRA adapter.
...
"""

from typing import Any
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_base_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Loads and returns (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def apply_lora(model, rank: int = 8, alpha: int = 16):
    """Wraps the base model with a LoRA adapter (peft), returns the wrapped model."""
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def get_adapter_parameters(model: Any):
    """Return LoRA adapter weights only as a list of NumPy arrays."""
    return [
        param.detach().cpu().numpy()
        for name, param in model.named_parameters()
        if "lora" in name.lower() and param.requires_grad
    ]


def set_adapter_parameters(model: Any, parameters):
    """Load a list of NumPy arrays into the LoRA adapter only."""
    lora_params = [
        (name, param) for name, param in model.named_parameters()
        if "lora" in name.lower() and param.requires_grad
    ]
    for (name, param), new_value in zip(lora_params, parameters):
        param.data = torch.from_numpy(new_value).to(param.device)


if __name__ == "__main__":
    model, tokenizer = load_base_model_and_tokenizer()
    print(f"Loaded model with {model.num_parameters():,} parameters.")

    model = apply_lora(model)
    model.print_trainable_parameters()

    params = get_adapter_parameters(model)
    print(f"Extracted {len(params)} adapter arrays, first shape: {params[0].shape}")

    zeroed = [np.zeros_like(p) for p in params]
    set_adapter_parameters(model, zeroed)
    check = get_adapter_parameters(model)
    print(f"Round-trip check — all zero: {all((c == 0).all() for c in check)}")