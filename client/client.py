"""Local training and evaluation for one federated client.

Locked public interfaces:
- load_client_data()
- train()
- evaluate()
(FLClient class owned separately — not implemented here.)

Run standalone test from the repo root:
    python -m client.client
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from model.model_utils import load_base_model_and_tokenizer, apply_lora


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
            print(f"  step {i+1} — loss: {last_loss:.4f}")  # NEW
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


if __name__ == "__main__":
    model, tokenizer = load_base_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = apply_lora(model)

    data = load_client_data(client_id=0)
   # data = data[:2]  # even smaller
    print(f"Loaded {len(data)} examples for client 0")

    model, train_loss = train(model, tokenizer, data, epochs=1)
    eval_loss, _ = evaluate(model, tokenizer, data)
    print(f"Eval loss: {eval_loss:.4f}")