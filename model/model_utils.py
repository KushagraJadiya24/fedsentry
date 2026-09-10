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
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"



def load_base_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Loads and returns (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_base_model_and_tokenizer()
    print(f"Loaded model with {model.num_parameters():,} parameters.")

def apply_lora(model: Any, rank: int = 8, alpha: int = 16):
    """Wrap the base model with a LoRA adapter and return it."""
    raise NotImplementedError("Day-0 scaffold: LoRA application not implemented yet.")


def get_adapter_parameters(model: Any):
    """Return LoRA adapter weights only as a list of NumPy arrays."""
    raise NotImplementedError("Day-0 scaffold: adapter extraction not implemented yet.")


def set_adapter_parameters(model: Any, parameters):
    """Load a list of NumPy arrays into the LoRA adapter only."""
    raise NotImplementedError("Day-0 scaffold: adapter loading not implemented yet.")
