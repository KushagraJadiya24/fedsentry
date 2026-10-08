"""Local training and evaluation for one federated client.

Locked public interfaces:
- load_client_data()
- train()
- evaluate()
(FLClient class owned separately — implemented below by Kashish.)

Run:
    python client/client.py --client_id 0
    python client/client.py --client_id 1 --server_address localhost:8080
"""
"""Local training and evaluation for one federated client.

Locked public interfaces:
- load_client_data()
- train()
- evaluate()
(FLClient class owned separately — implemented below by Kashish.)

Speed knobs are environment variables (no CLI or signature changes):
    FEDSENTRY_MAX_LENGTH, FEDSENTRY_MAX_EXAMPLES, FEDSENTRY_MAX_STEPS,
    FEDSENTRY_MAX_EVAL_EXAMPLES, FEDSENTRY_BATCH_SIZE, FEDSENTRY_THREADS

Run from the repo root:
    python -m client.client --client_id 0
    python -m client.client --client_id 1 --server_address localhost:8080
"""

import argparse
import json
import os
import time
from pathlib import Path

import flwr as fl
import torch
from torch.utils.data import Dataset, DataLoader

from model.model_utils import (
    load_base_model_and_tokenizer,
    apply_lora,
    get_adapter_parameters,
    set_adapter_parameters,
)

MAX_LENGTH = int(os.environ.get("FEDSENTRY_MAX_LENGTH", "64"))
MAX_LOCAL_EXAMPLES = int(os.environ.get("FEDSENTRY_MAX_EXAMPLES", "0")) or None
MAX_STEPS_PER_ROUND = int(os.environ.get("FEDSENTRY_MAX_STEPS", "0")) or None
MAX_EVAL_EXAMPLES = int(os.environ.get("FEDSENTRY_MAX_EVAL_EXAMPLES", "16"))
BATCH_SIZE = int(os.environ.get("FEDSENTRY_BATCH_SIZE", "2"))

if os.environ.get("FEDSENTRY_THREADS"):
    torch.set_num_threads(int(os.environ["FEDSENTRY_THREADS"]))


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InstructionDataset(Dataset):
    """Tokenizes examples without padding; padding happens per batch in the collate fn."""

    def __init__(self, examples, tokenizer, max_length: int = MAX_LENGTH):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = f"Instruction: {ex['instruction']}\nResponse: {ex['response']}"
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def _make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(batch):
        longest = max(len(b["input_ids"]) for b in batch)
        input_ids, attention_mask, labels = [], [], []
        for b in batch:
            pad = longest - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            attention_mask.append(b["attention_mask"] + [0] * pad)
            labels.append(b["input_ids"] + [-100] * pad)  # padding excluded from loss
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    return collate


def load_client_data(client_id: int, partitions_dir: str = "data/partitions"):
    """Loads this client's .jsonl file, returns a list of {instruction, response} dicts."""
    path = Path(partitions_dir) / f"client_{client_id}.jsonl"
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if MAX_LOCAL_EXAMPLES:
        examples = examples[:MAX_LOCAL_EXAMPLES]
    return examples


def train(model, tokenizer, dataset, epochs: int = 1, max_steps=None):
    """Local LoRA fine-tuning loop for this client's data only."""
    max_steps = max_steps or MAX_STEPS_PER_ROUND
    device = _get_device()
    model.to(device)

    ds = InstructionDataset(dataset, tokenizer)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_make_collate_fn(tokenizer)
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    model.train()
    losses, step = [], 0
    for _ in range(epochs):
        for batch in loader:
            t0 = time.time()
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
            print(f"  step {step} — loss: {losses[-1]:.4f} ({time.time() - t0:.1f}s)")
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    avg_loss = sum(losses) / max(len(losses), 1)
    print(f"Train done — {step} steps, avg loss: {avg_loss:.4f}")
    return model, avg_loss


def evaluate(model, tokenizer, dataset):
    """Returns (loss, None) on a capped number of examples."""
    device = _get_device()
    model.to(device)

    ds = InstructionDataset(dataset[:MAX_EVAL_EXAMPLES], tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=_make_collate_fn(tokenizer))

    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total_loss += model(**batch).loss.item()
            count += 1
    return total_loss / max(count, 1), None  # no separate accuracy metric per Conventions

# ---------------------------------------------------------------------------
# FLClient — Kashish's part. Wires local training/evaluation into the Flower
# network layer. Treats train()/evaluate()/get_adapter_parameters()/
# set_adapter_parameters() as black boxes, per the task doc.
# ---------------------------------------------------------------------------
class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, partitions_dir: str = "data/partitions"):
        self.client_id = client_id
        self.dataset = load_client_data(client_id, partitions_dir)

        self.model, self.tokenizer = load_base_model_and_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = apply_lora(self.model)

    def get_parameters(self, config):
        return get_adapter_parameters(self.model)

    def fit(self, parameters, config):
        set_adapter_parameters(self.model, parameters)
        self.model, loss = train(
            self.model, self.tokenizer, self.dataset, epochs=config.get("epochs", 1)
        )
        updated_parameters = get_adapter_parameters(self.model)
        num_examples = len(self.dataset)
        return updated_parameters, num_examples, {"loss": loss, "client_id": self.client_id}

    def evaluate(self, parameters, config):
        set_adapter_parameters(self.model, parameters)
        loss, metric = evaluate(self.model, self.tokenizer, self.dataset)
        num_examples = len(self.dataset)
        return loss, num_examples, {"metric": metric or 0.0, "client_id": self.client_id}


def main():
    parser = argparse.ArgumentParser(description="FedSentry Flower client.")
    parser.add_argument("--client_id", type=int, required=True, help="Which client shard to load (0, 1, or 2).")
    parser.add_argument("--server_address", default="localhost:8080", help="Flower server address.")
    args = parser.parse_args()

    client = FLClient(client_id=args.client_id)
    fl.client.start_client(server_address=args.server_address, client=client.to_client())


if __name__ == "__main__":
    main()