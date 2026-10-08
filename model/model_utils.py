"""Loads the Qwen base model and manages its LoRA adapter.

Provides the four locked interfaces: load_base_model_and_tokenizer(),
apply_lora(), get_adapter_parameters(), set_adapter_parameters().
Adapter arrays are returned in a fixed order, so every client and the server
must use the same rank and target modules.

Run a self-check from the repo root:
    python model/model_utils.py
"""

from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_base_model_and_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
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


def _lora_named_params(model):
    """All LoRA parameters, in a stable order (same order on every client/server)."""
    return [
        (name, param)
        for name, param in model.named_parameters()
        if "lora" in name.lower()
    ]


def get_adapter_parameters(model: Any):
    """Return LoRA adapter weights only as a list of NumPy arrays (copies)."""
    params = _lora_named_params(model)
    if not params:
        raise ValueError("No LoRA parameters found. Did you call apply_lora()?")
    return [p.detach().cpu().numpy().copy() for _, p in params]


def set_adapter_parameters(model: Any, parameters):
    """Load a list of NumPy arrays into the LoRA adapter only."""
    params = _lora_named_params(model)
    if len(params) != len(parameters):
        raise ValueError(
            f"Expected {len(params)} adapter arrays, got {len(parameters)}"
        )
    with torch.no_grad():
        for (name, param), new_value in zip(params, parameters):
            if tuple(new_value.shape) != tuple(param.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"expected {tuple(param.shape)}, got {tuple(new_value.shape)}"
                )
            param.copy_(
                torch.from_numpy(np.array(new_value)).to(
                    device=param.device, dtype=param.dtype
                )
            )


if __name__ == "__main__":
    model, tokenizer = load_base_model_and_tokenizer()
    print(f"Loaded model with {model.num_parameters():,} parameters.")

    model = apply_lora(model)
    model.print_trainable_parameters()

    params = get_adapter_parameters(model)
    print(f"Extracted {len(params)} adapter arrays, first shape: {params[0].shape}")

    # Round-trip 1: zeros
    zeroed = [np.zeros_like(p) for p in params]
    set_adapter_parameters(model, zeroed)
    check = get_adapter_parameters(model)
    print(f"Round-trip check — all zero: {all((c == 0).all() for c in check)}")

    # Round-trip 2: random values must come back exactly
    rand = [np.random.randn(*p.shape).astype(p.dtype) for p in params]
    set_adapter_parameters(model, rand)
    back = get_adapter_parameters(model)
    print(f"Random round-trip exact: {all(np.array_equal(a, b) for a, b in zip(rand, back))}")

    # Validation: a wrong-length list must raise, not silently truncate
    try:
        set_adapter_parameters(model, params[:-1])
    except ValueError as e:
        print(f"Length check OK: {e}")