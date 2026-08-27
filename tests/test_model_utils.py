"""Contract tests for model utility interfaces.

These are intentionally lightweight Day-0 tests. Real model tests should be
added once model_utils.py is implemented.
"""

import inspect

from model.model_utils import (
    apply_lora,
    get_adapter_parameters,
    load_base_model_and_tokenizer,
    set_adapter_parameters,
)


def test_locked_model_default():
    """The locked default model name must remain unchanged."""
    signature = inspect.signature(load_base_model_and_tokenizer)
    assert (
        signature.parameters["model_name"].default
        == "Qwen/Qwen2.5-0.5B-Instruct"
    )


def test_locked_function_names_exist():
    """All four locked model utility functions exist."""
    assert callable(load_base_model_and_tokenizer)
    assert callable(apply_lora)
    assert callable(get_adapter_parameters)
    assert callable(set_adapter_parameters)
