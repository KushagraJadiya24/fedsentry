from pathlib import Path
import math
import torch
from torch.utils.data import DataLoader

from client.client import (
    InstructionDataset,
    _make_collate_fn,
    load_holdout_data,
)
from model.inference import build_model, load_adapter_arrays
from model.model_utils import set_adapter_parameters, get_adapter_parameters

ADAPTER = Path("server/federated_adapter.npz")
PARTITIONS = "data/partitions"


@torch.no_grad()
def loss_on_examples(model, tokenizer, examples):
    loader = DataLoader(
        InstructionDataset(examples, tokenizer),
        batch_size=1,
        collate_fn=_make_collate_fn(tokenizer),
    )

    device = next(model.parameters()).device
    model.eval()

    total = 0.0
    count = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        n = int((batch["labels"][:, 1:] != -100).sum())

        if n == 0:
            continue

        total += model(**batch).loss.item() * n
        count += n

    return total / max(count, 1)


print("=" * 70)
print("FedSentry - Per-Client Federated Evaluation")
print("=" * 70)

adapter_arrays = load_adapter_arrays(ADAPTER)

model, tokenizer = build_model()

base_arrays = get_adapter_parameters(model)

for client_id in range(3):
    examples = load_holdout_data(client_id, PARTITIONS)

    if not examples:
        print(f"\nClient {client_id}: NO HOLDOUT DATA")
        continue

    set_adapter_parameters(model, base_arrays)
    base_loss = loss_on_examples(model, tokenizer, examples)

    set_adapter_parameters(model, adapter_arrays)
    federated_loss = loss_on_examples(model, tokenizer, examples)

    improvement = (base_loss - federated_loss) / base_loss * 100

    print(f"\nClient {client_id}")
    print(f"  Holdout examples : {len(examples)}")
    print(f"  Base loss        : {base_loss:.4f}")
    print(f"  Federated loss   : {federated_loss:.4f}")
    print(f"  Improvement      : {improvement:+.2f}%")

print("\n" + "=" * 70)
print("Evaluation complete")
print("=" * 70)
