"""Utilities for loading Qwen and managing its LoRA adapter.

Day-0 scaffold only.

Locked public interfaces:
- load_base_model_and_tokenizer()
- apply_lora()
- get_adapter_parameters()
- set_adapter_parameters()

Run tests from the repository root:
    pytest -q
"""

from typing import Any


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_base_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
):
    """Load and return (model, tokenizer)."""
    raise NotImplementedError("Day-0 scaffold: model loading not implemented yet.")


def apply_lora(model: Any, rank: int = 8, alpha: int = 16):
    """Wrap the base model with a LoRA adapter and return it."""
    raise NotImplementedError("Day-0 scaffold: LoRA application not implemented yet.")


def get_adapter_parameters(model: Any):
    """Return LoRA adapter weights only as a list of NumPy arrays."""
    raise NotImplementedError("Day-0 scaffold: adapter extraction not implemented yet.")


def set_adapter_parameters(model: Any, parameters):
    """Load a list of NumPy arrays into the LoRA adapter only."""
    raise NotImplementedError("Day-0 scaffold: adapter loading not implemented yet.")
