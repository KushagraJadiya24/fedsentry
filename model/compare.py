"""Compare the base model with the model that carries the aggregated federated adapter.

Three checks:
  1. Weight check  - the adapter file really changed the model's weights, and nothing else.
  2. Held-out loss - loss on examples that no client trained on (lower is better).
  3. Generation    - answers to held-out instructions compared with the reference answers.

Run from the repo root after a federated training run:
    python -m model.compare
    python -m model.compare --num_samples 20
"""

import argparse
import hashlib
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from client.client import InstructionDataset, _make_collate_fn, load_holdout_data
from model.inference import DEFAULT_ADAPTER_PATH, build_model, generate_answer, load_adapter_arrays
from model.model_utils import get_adapter_parameters, set_adapter_parameters

ROOT = Path(__file__).resolve().parent.parent
PARTITIONS_DIR = ROOT / "data" / "partitions"
NUM_CLIENTS = 3
F1_WORDS = 40


def load_test_set():
    examples = []
    for k in range(NUM_CLIENTS):
        examples.extend(load_holdout_data(k, str(PARTITIONS_DIR)))
    return examples


@torch.no_grad()
def heldout_loss(model, tokenizer, examples, batch_size=4):
    """Token-weighted average loss over the examples."""
    device = next(model.parameters()).device
    loader = DataLoader(InstructionDataset(examples, tokenizer), batch_size=batch_size,
                        collate_fn=_make_collate_fn(tokenizer))
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        n = int((batch["labels"][:, 1:] != -100).sum())
        if n == 0:
            continue
        total += model(**batch).loss.item() * n
        count += n
    return total / max(count, 1)


def base_fingerprint(model):
    """Sum of |w| over every non-LoRA parameter. It changes if any base weight changes."""
    total = 0.0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora" not in name.lower():
                total += p.detach().float().abs().sum().item()
    return total


def lora_b_mean_abs(model):
    vals = [p.detach().float().abs().mean().item() for n, p in model.named_parameters() if "lora_B" in n]
    return float(np.mean(vals)) if vals else float("nan")


def word_f1(prediction, reference, max_words=F1_WORDS):
    pred = re.findall(r"\w+", prediction.lower())[:max_words]
    ref = re.findall(r"\w+", reference.lower())[:max_words]
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def run_comparison(adapter_path=DEFAULT_ADAPTER_PATH, num_samples=12, max_new_tokens=64, bundle=None):
    """bundle = (model, tokenizer, untrained_adapter_arrays); built here if not given."""
    start = time.time()
    adapter_path = Path(adapter_path)
    file_arrays = load_adapter_arrays(adapter_path)
    if bundle is None:
        model, tokenizer = build_model()
        base_arrays = get_adapter_parameters(model)
    else:
        model, tokenizer, base_arrays = bundle
        set_adapter_parameters(model, base_arrays)  # start from the untrained adapter

    test_set = load_test_set()
    if not test_set:
        raise RuntimeError("No held-out examples found. Update client/client.py (FEDSENTRY_HOLDOUT) "
                           "and run the federated training again.")
    idx = np.unique(np.linspace(0, len(test_set) - 1, min(num_samples, len(test_set))).astype(int))
    sample = [test_set[i] for i in idx]

    fp_before, b_before = base_fingerprint(model), lora_b_mean_abs(model)
    loss_before = heldout_loss(model, tokenizer, test_set)
    answers_before = [generate_answer(model, tokenizer, ex["instruction"], max_new_tokens) for ex in sample]

    set_adapter_parameters(model, file_arrays)
    after_arrays = get_adapter_parameters(model)
    fp_after, b_after = base_fingerprint(model), lora_b_mean_abs(model)
    loss_after = heldout_loss(model, tokenizer, test_set)
    answers_after = [generate_answer(model, tokenizer, ex["instruction"], max_new_tokens) for ex in sample]

    examples = []
    for ex, b, a in zip(sample, answers_before, answers_after):
        examples.append({"instruction": ex["instruction"], "reference": ex["response"], "before": b, "after": a,
                         "f1_before": word_f1(b, ex["response"]), "f1_after": word_f1(a, ex["response"])})

    as64 = lambda x: x.astype(np.float64)
    update_norm = math.sqrt(sum(float(np.sum((as64(a) - as64(b)) ** 2)) for a, b in zip(base_arrays, after_arrays)))
    return {
        "adapter_file": str(adapter_path),
        "adapter_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "adapter_size_kb": adapter_path.stat().st_size / 1024,
        "adapter_modified": time.ctime(adapter_path.stat().st_mtime),
        "num_arrays": len(file_arrays),
        "num_params": int(sum(a.size for a in file_arrays)),
        "arrays_changed": sum(1 for a, b in zip(base_arrays, after_arrays) if not np.array_equal(a, b)),
        "update_norm": update_norm,
        "lora_b_before": b_before, "lora_b_after": b_after,
        "file_max_diff": max(float(np.max(np.abs(as64(a) - as64(b)))) for a, b in zip(file_arrays, after_arrays)),
        "base_unchanged": abs(fp_before - fp_after) <= 1e-9 * max(abs(fp_before), 1.0),
        "n_test": len(test_set),
        "loss_before": loss_before, "loss_after": loss_after,
        "ppl_before": math.exp(loss_before), "ppl_after": math.exp(loss_after),
        "n_gen": len(examples),
        "f1_before": float(np.mean([e["f1_before"] for e in examples])),
        "f1_after": float(np.mean([e["f1_after"] for e in examples])),
        "f1_wins": sum(1 for e in examples if e["f1_after"] > e["f1_before"]),
        "examples": examples,
        "seconds": time.time() - start,
    }


def format_report(r):
    gain = (r["loss_before"] - r["loss_after"]) / r["loss_before"] * 100
    lines = [
        "# FedSentry: base model vs federated model", "",
        f"Adapter file: `{Path(r['adapter_file']).name}` (saved {r['adapter_modified']}, "
        f"{r['adapter_size_kb']:.0f} KB, SHA-256 `{r['adapter_sha256'][:16]}...`)", "",
        "## 1. Weight check", "",
        "| Check | Result |", "|---|---|",
        f"| Adapter arrays / parameters | {r['num_arrays']} / {r['num_params']:,} |",
        f"| Arrays whose values changed | {r['arrays_changed']} of {r['num_arrays']} |",
        f"| Size of the update (L2 norm) | {r['update_norm']:.4f} |",
        f"| Mean abs of B matrices, before to after | {r['lora_b_before']:.6f} to {r['lora_b_after']:.6f} |",
        f"| Loaded weights equal the file (max difference) | {r['file_max_diff']:.2e} |",
        f"| Base model weights unchanged | {'yes' if r['base_unchanged'] else 'NO'} |", "",
        f"## 2. Loss on {r['n_test']} held-out examples (never used for training)", "",
        "| Model | Loss | Perplexity |", "|---|---|---|",
        f"| Base model | {r['loss_before']:.4f} | {r['ppl_before']:.2f} |",
        f"| Federated model | {r['loss_after']:.4f} | {r['ppl_after']:.2f} |", "",
        f"Change in loss: {-gain:+.1f}%", "",
        f"## 3. Answers to {r['n_gen']} held-out instructions", "",
        f"Word overlap with the reference answer (F1, first {F1_WORDS} words): "
        f"base {r['f1_before']:.3f}, federated {r['f1_after']:.3f}; "
        f"federated higher on {r['f1_wins']} of {r['n_gen']}.", "",
    ]
    for i, e in enumerate(r["examples"][:4], 1):
        lines += [f"### Example {i}", f"**Instruction:** {e['instruction']}", "",
                  f"**Reference:** {e['reference']}", "",
                  f"**Base:** {e['before']}", "", f"**Federated:** {e['after']}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare the base model with the federated model.")
    parser.add_argument("--adapter_path", default=str(DEFAULT_ADAPTER_PATH))
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()
    report = format_report(run_comparison(args.adapter_path, args.num_samples, args.max_new_tokens))
    print(report)
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "comparison.md").write_text(report, encoding="utf-8")
    print(f"\nSaved to {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()