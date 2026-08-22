# FedSentry

**Federated LoRA fine-tuning of a small LLM — multiple participants collaboratively
improve a shared language model without ever sharing their private data.**

Final Year Major Project — B.Tech Computer Science & Engineering (AI & ML)

---

## What is FedSentry?

FedSentry lets multiple participants collaboratively fine-tune a shared small language
model — without any of them ever having to share their raw private data.

Each participant fine-tunes the model locally, on their own text data, using **LoRA**
(Low-Rank Adaptation) — a lightweight fine-tuning technique that trains a small set of
extra parameters instead of the entire model. Only those small LoRA adapter weights
(a few megabytes, never the raw data, never the full model) are sent to a central
server, which combines everyone's updates via **FedAvg** across multiple training
rounds — using [Flower](https://flower.ai) for federated orchestration, [PyTorch](https://pytorch.org)
for model training, and Hugging Face [`transformers`](https://huggingface.co/docs/transformers)
+ [`peft`](https://huggingface.co/docs/peft) for the model and LoRA implementation.

> **Why this matters:** organizations increasingly want to fine-tune language models
> on their own private data (support tickets, internal docs, domain-specific text) but
> can't pool that data with others, or even centralize it internally, due to privacy
> or regulatory constraints. Federated learning solves this by moving the model to the
> data instead of the data to the model. FedSentry demonstrates this specifically for
> modern LLM fine-tuning via LoRA, not just classical model training.

**Note:** this project originally prototyped federated learning using a small image
classifier (CNN on MNIST). It has since pivoted to federated LoRA fine-tuning of an
LLM, to better reflect the skills the team is building toward. See
[`FedSentry_Blueprint_v4`](#) and [`FEDSENTRY_CONVENTIONS_v2.md`](#) for the full
reasoning and current locked scope.

---

## Current Status

🚧 **In active development.** Built in two phases — see [Roadmap](#roadmap).

---

## Features

**Phase 1 — Core Pipeline**
- [ ] LoRA fine-tuning of a small base LLM, working locally (no federation yet)
- [ ] Federated fine-tuning via Flower (FedAvg), across 3 simulated participants
- [ ] Non-IID private data split across participants (`.jsonl` instruction/response data)
- [ ] Round-by-round loss logging (CSV)
- [ ] Before/after demonstration: base model vs. federated fine-tuned model

**Phase 2 — Bonus (only after Phase 1 works end-to-end)**
- [ ] Single-page Streamlit UI: configure a run, click "Start Training," view results

**Explicitly out of scope for this build** (see blueprint for full reasoning):
multi-model selection, differential privacy, FedProx comparison, SQLite, Jinja2 code
generation, multi-laptop physical deployment. All documented as honest future work.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model training | PyTorch |
| LoRA fine-tuning | Hugging Face `peft` |
| Base model + tokenizer | Hugging Face `transformers` |
| Federated orchestration | [Flower](https://flower.ai) (`flwr`), FedAvg only |
| Metrics | Plain CSV file |
| UI (Phase 2 only) | Streamlit (single page) |

No SQLite, no Jinja2, no Opacus, no Docker, no JS frontend in this build — the entire
project runs with plain Python. See the blueprint for why each was deliberately cut.

---

## Prerequisites — Read Before Writing Any Code

This project assumes you can already write Python. It does **not** assume prior
experience with PyTorch, LoRA, or Flower — if you're new to these (most of the team
is), work through them in this order, on a small scale, before touching the actual
federated pipeline. Trying to learn all of this *while* debugging the full pipeline at
the same time is the fastest way to get stuck and not understand your own project.

### 1. PyTorch fundamentals
Before anything else, get comfortable with:
- **Tensors** — PyTorch's core data structure (like a NumPy array, but able to track
  gradients)
- **A basic model class** (`nn.Module`) — what layers and a forward pass are
- **A training loop** — where loss, backpropagation (`.backward()`), and an optimizer
  step actually happen in real code
- **Evaluation** — measuring loss/accuracy on held-out data

Good starting point: [PyTorch's official "Learn the Basics" tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html).
Don't move on until you can explain, in your own words, what happens when you call
`.backward()` on a loss.

### 2. Hugging Face `transformers` — loading and running a real small LLM
- How to load a pretrained model + tokenizer (`AutoModelForCausalLM`, `AutoTokenizer`)
- How tokenization works (text → numbers the model can process)
- How to run basic inference (ask the model something, get a response) before trying
  to train anything

Good starting point: [Hugging Face's `transformers` quicktour](https://huggingface.co/docs/transformers/quicktour).

### 3. LoRA fine-tuning with `peft`
- What a LoRA adapter actually is (freeze the base model, train a small set of extra
  parameters attached to it) and why it's used instead of full fine-tuning
- `LoraConfig` — what `rank` and `alpha` control, at a basic intuitive level
- `get_peft_model()` — wrapping a base model with a trainable LoRA adapter
- How to fine-tune on a small custom dataset and see the adapter actually learn
  something (a visible before/after difference in the model's output)

Good starting point: [Hugging Face's `peft` quicktour](https://huggingface.co/docs/peft/quicktour).
**Do this fully on ONE machine, with no federation involved, before moving to Flower.**
If you can't get a single local LoRA fine-tune working and showing a clear before/after
difference, the federated version will not make sense either.

### 4. Flower fundamentals
- The client/server relationship — what a `NumPyClient` is, what `fit()` and
  `evaluate()` are for
- `get_parameters()` / `set_parameters()` — how model weights get sent back and forth
  as plain NumPy arrays
- FedAvg — what "averaging updates across clients" actually does, conceptually

Good starting point: [Flower's official PyTorch quickstart tutorial](https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html).

### 5. Putting it together — the one genuinely tricky part
Standard Flower tutorials send a model's **entire** set of weights back and forth.
This project only sends the **LoRA adapter's** weights (not the frozen base model's).
This means `get_parameters()`/`set_parameters()` need to extract and load only the
adapter weights, not the full model — this specific combination has fewer existing
tutorials to lean on, so budget real, dedicated time for it rather than assuming it'll
be a small tweak on top of the standard Flower example.

---

## Getting Started

### Prerequisites
- Python 3.10+
- `pip`
- Enough local compute to run a ~0.5B parameter model (a modern laptop CPU works,
  though slowly; a GPU, even a small one, will make iteration much faster)

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/fedsentry.git
cd fedsentry
```

### 2. Set up a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare the client data
```bash
cd data
python prepare_data.py
cd ..
```
This generates `data/partitions/client_0.jsonl`, `client_1.jsonl`, `client_2.jsonl` —
each participant's own private, non-identical instruction/response data.

### 5. Run the federated pipeline
Open 4 terminals (all with the virtual environment activated):

```bash
# Terminal 1 — server
cd server
python server.py --rounds 5 --min_clients 3
```

```bash
# Terminal 2
cd client
python client.py --client_id 0 --server_address localhost:8080
```

```bash
# Terminal 3
cd client
python client.py --client_id 1 --server_address localhost:8080
```

```bash
# Terminal 4
cd client
python client.py --client_id 2 --server_address localhost:8080
```

You should see per-round loss print in each terminal, and results logged to
`server/metrics.csv`. Start the server first, then the 3 clients within a few seconds
of each other.

---

## Project Structure

```
fedsentry/
├── data/
│   ├── prepare_data.py     # splits instruction/response data into 3 non-IID client shards
│   └── partitions/          # generated: client_0.jsonl, client_1.jsonl, client_2.jsonl
├── model/
│   └── model_utils.py        # loads base model + tokenizer, applies/extracts LoRA adapter
├── server/
│   ├── server.py               # Flower server: FedAvg aggregation of adapter weights
│   └── metrics.csv              # round-by-round loss (generated)
├── client/
│   └── client.py                 # Flower client: local LoRA fine-tuning
├── app/
│   └── Home.py                    # Phase 2 only — Streamlit single-page UI
├── requirements.txt
└── README.md
```

See `FEDSENTRY_CONVENTIONS_v2.md` for exact locked function signatures, CLI argument
names, and file naming — read that before writing any new module.

---

## Roadmap

- [x] **Prototype (superseded):** federated CNN image classifier on MNIST — proved the
      core Flower/FedAvg mechanics work, before pivoting to LoRA/LLM
- [ ] **Milestone 1:** Local LoRA fine-tuning of the base LLM, working on one machine,
      no federation yet — visible before/after improvement
- [ ] **Milestone 2:** Federate it — 3 terminal-based clients + 1 server, FedAvg,
      adapter-only weight exchange
- [ ] **Milestone 3:** Hardening — reliable end-to-end runs, clean before/after demo
- [ ] **Milestone 4 (Phase 2):** Single-page Streamlit UI on top of the working pipeline
- [ ] **Future work:** multi-model selection, differential privacy (Opacus), FedProx
      comparison, persistent database, automatic code generation, multi-laptop
      physical deployment, Dockerized deployment

---

## Team

| Name | 
|---|
| Kushagra Jadiya | 
| Sunny Kapoor | 
| Kashish Pherwani | 

Guide: [Guide Name]

---

## References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017
- Li et al., *Federated Optimization in Heterogeneous Networks*, MLSys 2020
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- Flower documentation: https://flower.ai/docs
- Hugging Face `peft` documentation: https://huggingface.co/docs/peft
- Hugging Face `transformers` documentation: https://huggingface.co/docs/transformers

---

## License

This project is for academic purposes as part of a final year major project submission.
