# Contributing to FedSentry

## Before coding

1. Pull the latest `main`.
2. Read `FEDSENTRY_CONVENTIONS_v2.md`.
3. Work only inside your assigned file/function boundaries.
4. Create a feature branch.
5. Keep changes small enough for a focused PR.

## Ownership

### Kushagra — LoRA / AI Engineering
- `model/model_utils.py`
- `client/client.py`: `load_client_data()`, `train()`, `evaluate()`

### Teammate A — Federated Server
- `server/server.py`

### Teammate B — Data + Flower Client + UI
- `data/prepare_data.py`
- `client/client.py`: `FLClient`
- `app/Home.py` only after Phase 1 is frozen and working

If two people need the same file, coordinate on the locked function boundaries before editing.

## Branch naming

Use:

```text
feature/<short-description>
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

## Commit messages

Use:

```text
[area] short description
```

Examples:

```text
[model] add LoRA adapter utilities
[client] add local training loop
[data] add non-IID partitions
[server] add FedAvg aggregation
[ui] add training controls
```

## Pull requests

Every PR should state:

- What changed
- How it was tested
- What it depends on
- Whether a locked interface was touched

## Guardrails

Do not:
- rename locked files, functions, classes, or CLI arguments
- introduce new top-level architecture
- introduce a new library without team agreement
- add explicitly cut features
- commit `server/metrics.csv`
- commit model artifacts such as `*.safetensors`
- silently change the base model

The shared data partitions are intentionally committed to Git so every teammate trains against the identical split.
