import numpy as np
import torch

from model.model_utils import (
    load_base_model_and_tokenizer,
    apply_lora,
    set_adapter_parameters,
)


ADAPTER_FILE = "server/federated_adapter.npz"

PROMPT = (
    "Explain federated learning in simple terms and give one practical example."
)


def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


def main():
    print("=" * 70)
    print("FedSentry: Base vs Federated Model Evaluation")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load base Qwen model
    # ---------------------------------------------------------
    model, tokenizer = load_base_model_and_tokenizer()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # ---------------------------------------------------------
    # BASE MODEL
    # ---------------------------------------------------------
    print("\n[BASE MODEL]")
    print(f"Prompt: {PROMPT}")

    base_response = generate_response(
        model,
        tokenizer,
        PROMPT,
    )

    print(f"Response: {base_response}")

    # ---------------------------------------------------------
    # Apply LoRA structure
    # ---------------------------------------------------------
    model = apply_lora(model)

    # ---------------------------------------------------------
    # Load federated adapter
    # ---------------------------------------------------------
    print("\nLoading federated LoRA adapter...")

    adapter_data = np.load(ADAPTER_FILE)

    federated_parameters = [
        adapter_data[name]
        for name in adapter_data.files
    ]

    print(f"Loaded adapter arrays: {len(federated_parameters)}")
    print(
        f"First adapter shape: "
        f"{federated_parameters[0].shape}"
    )

    set_adapter_parameters(
        model,
        federated_parameters,
    )

    model.eval()

    # ---------------------------------------------------------
    # FEDERATED MODEL
    # ---------------------------------------------------------
    print("\n[FEDERATED MODEL]")
    print(f"Prompt: {PROMPT}")

    federated_response = generate_response(
        model,
        tokenizer,
        PROMPT,
    )

    print(f"Response: {federated_response}")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()