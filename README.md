# FedSentry

Federated LoRA fine-tuning for a small language model.

FedSentry lets three simulated participants fine-tune a shared small LLM locally on private, non-IID text data. Only LoRA adapter weights are exchanged with a Flower server and aggregated with FedAvg.

## Current scope

### Phase 1 — Core pipeline
- PyTorch fundamentals
- LoRA fine-tuning
- Locked base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Flower + FedAvg
- 3 simulated terminal clients
- Plain CSV metrics
- Before/after generation

### Phase 2 — Bonus
- Single-page Streamlit UI

### Explicitly out of scope
- Multi-model selection
- Differential privacy / Opacus
- FedProx / strategy comparison
- SQLite
- Jinja2 code generation
- JavaScript frontend
- Multi-laptop demo as a requirement

Do not add these without an explicit team decision.

## Repository structure

```text
fedsentry/
├── data/
│   ├── prepare_data.py
│   └── partitions/
├── model/
│   └── model_utils.py
├── server/
│   └── server.py
├── client/
│   └── client.py
├── app/
│   └── Home.py
├── tests/
│   ├── test_data.py
│   ├── test_model_utils.py
│   └── test_smoke.py
├── requirements.txt
├── .gitignore
├── CONTRIBUTING.md
└── README.md
```

The paths and public interfaces above are locked by `FEDSENTRY_CONVENTIONS_v2.md`.

## Day-0 setup

### 1. Clone

```bash
git clone <your-repository-url>
cd fedsentry
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Python

```bash
python --version
```

Python 3.10+ is required.

## Golden path

The final Phase-1 developer flow is:

```text
prepare_data.py
      ↓
client_0.jsonl ─┐
client_1.jsonl ─┼──→ 3 Flower clients
client_2.jsonl ─┘          ↓
                       local LoRA
                           ↓
                    adapter weights
                           ↓
                    Flower server
                           ↓
                         FedAvg
                           ↓
                     metrics.csv
                           ↓
                 before/after generation
```

During Day 0/early development, each subsystem can be tested independently with stubs/dummy clients.

## Development order

1. Data partitions and schema validation
2. Qwen loading + LoRA locally
3. Local training/evaluation
4. Flower server with dummy clients
5. Flower client wiring with stubs
6. Real client/server integration
7. Multiple federated rounds
8. Federated evaluation: loss, adapter-only verification, client contribution, baseline comparison
9. Hardening + before/after demo
10. Streamlit only after Phase 1 is working

## Locked CLI interfaces

Server:

```bash
python server/server.py --rounds 3 --min_clients 3 --address 0.0.0.0:8080
```

Client:

```bash
python client/client.py --client_id 0 --server_address localhost:8080
python client/client.py --client_id 1 --server_address localhost:8080
python client/client.py --client_id 2 --server_address localhost:8080
```

These commands are the intended final shape. Day-0 stubs may not perform the real training yet.

## Testing

Run:

```bash
pytest -q
```

The tests currently focus on contracts and scaffolding. They should become stricter as each implementation lands.

## Team ownership

### Kushagra — AI / Model Engineering
- `model/model_utils.py`
- `client/client.py`: `load_client_data()`, `train()`, `evaluate()`
- Local LoRA fine-tuning and adapter serialization

### Sunny — Federated Systems + Evaluation
- `server/server.py`
- FedAvg orchestration and metrics
- Adapter-only aggregation verification
- Multi-round federated evaluation
- Non-IID contribution analysis
- Baseline vs federated comparison
- End-to-end integration/robustness testing

### Kashish — Data + Client Engineering + UI
- `data/prepare_data.py`
- `client/client.py`: `FLClient`
- `app/Home.py` only after Phase 1 is frozen and working

See the individual task documents for exact ownership boundaries.

## Git workflow

Create a feature branch:

```bash
git checkout -b feature/<short-description>
```

Examples:

```text
feature/lora-model-utils
feature/lora-client-training
feature/data-prep
feature/flower-client
feature/flower-server
feature/streamlit-ui
```

Commit format:

```text
[area] short description
```

Examples:

```text
[model] add LoRA adapter utilities
[client] add local training loop
[data] add non-IID partitions
[server] add FedAvg aggregation
```

