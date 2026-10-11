"""Try the federated model: compare answers before and after loading the aggregated adapter.

Answers each prompt with the untrained adapter (identical to the base model, because LoRA
starts at zero), then loads the aggregated adapter saved by the server and answers again.

Run from the repo root:
    python -m model.inference
    python -m model.inference --prompt "Give three tips for staying healthy."
    python -m model.inference --adapter_path server/federated_adapter.npz
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from model.model_utils import (
    load_base_model_and_tokenizer,
    apply_lora,
    set_adapter_parameters,
)

DEFAULT_ADAPTER_PATH = Path(__file__).resolve().parent.parent / "server" / "federated_adapter.npz"

DEFAULT_PROMPTS = [
    "Give three tips for staying healthy.",
    "Explain why environmental protection is important.",
    "What is the boiling point of water?",
]


def load_adapter_arrays(adapter_path):
    """Reads the server's .npz file and returns the adapter as an ordered list of arrays."""
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter file not found: {adapter_path}. Run the federated training first."
        )
    data = np.load(adapter_path)
    names = list(data.files)
    if all(re.fullmatch(r"arr_\d+", n) for n in names):
        names = sorted(names, key=lambda n: int(n.split("_")[1]))
    return [data[n] for n in names]


def build_model():
    """Loads the base model with a fresh (untrained) LoRA adapter on the best device."""
    model, tokenizer = load_base_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = apply_lora(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, instruction, max_new_tokens=80):
    """Answers one instruction using the same prompt format used for training."""
    prompt = f"Instruction: {instruction}\nResponse:"
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.split("\nInstruction:")[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Compare answers before/after the federated adapter.")
    parser.add_argument("--adapter_path", default=str(DEFAULT_ADAPTER_PATH), help="Path to federated_adapter.npz")
    parser.add_argument("--prompt", default=None, help="One instruction to try (default: three examples)")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    args = parser.parse_args()

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    arrays = load_adapter_arrays(args.adapter_path)
    print(f"Loaded {len(arrays)} adapter arrays from {args.adapter_path}")

    model, tokenizer = build_model()
    before = [generate_answer(model, tokenizer, p, args.max_new_tokens) for p in prompts]

    set_adapter_parameters(model, arrays)  # raises a clear error if the shapes do not match
    after = [generate_answer(model, tokenizer, p, args.max_new_tokens) for p in prompts]

    for p, b, a in zip(prompts, before, after):
        print("\n" + "=" * 70)
        print(f"PROMPT : {p}")
        print(f"BEFORE : {b}")
        print(f"AFTER  : {a}")


if __name__ == "__main__":
    main()