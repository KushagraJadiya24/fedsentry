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

import argparse
import json
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


class InstructionDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = f"Instruction: {ex['instruction']}\nResponse: {ex['response']}"
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_client_data(client_id: int, partitions_dir: str = "data/partitions"):
    """Loads this client's .jsonl file, returns a list of {instruction, response} dicts."""
    path = Path(partitions_dir) / f"client_{client_id}.jsonl"
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def train(model, tokenizer, dataset, epochs: int = 1):
    """Local LoRA fine-tuning loop for this client's data only."""
    ds = InstructionDataset(dataset, tokenizer)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    model.train()
    last_loss = None
    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
            print(f"  step {i+1} — loss: {last_loss:.4f}")
        print(f"Epoch {epoch + 1}/{epochs} — loss: {last_loss:.4f}")
    return model, last_loss


def evaluate(model, tokenizer, dataset):
    """Returns (loss, some accuracy/quality metric) on held-out examples."""
    ds = InstructionDataset(dataset, tokenizer)
    loader = DataLoader(ds, batch_size=2)
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            count += 1
    avg_loss = total_loss / max(count, 1)
    return avg_loss, None  # no separate accuracy metric per Conventions Section 8


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