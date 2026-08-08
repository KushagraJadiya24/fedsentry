# FedSentry

**A no-code platform for configuring and running privacy-preserving federated learning pipelines.**

Final Year Major Project — B.Tech Computer Science & Engineering (AI & ML)

---

## What is FedSentry?

FedSentry lets someone train a shared machine learning model across multiple
independent data sources — without any of them ever having to share their raw data.

Instead of hand-writing federated learning code (which normally requires an ML
engineer), FedSentry provides a simple guided setup: pick a task, choose your privacy
and aggregation settings, and the platform generates and runs the training pipeline for
you — using [Flower](https://flower.ai) for federated orchestration, [PyTorch](https://pytorch.org)
for model training, and [Opacus](https://opacus.ai) for differential privacy.

> **Why this matters:** regulations like GDPR, HIPAA, and India's DPDP Act increasingly
> prevent organizations from centralizing sensitive data — even when pooling that data
> would produce a better shared AI model. Federated learning solves this by moving the
> model to the data instead of the data to the model. FedSentry makes that technique
> usable without requiring a dedicated ML engineering team.

---

## Current Status

🚧 **In active development.** This is a final year academic project, built in
progressive milestones. See [Roadmap](#roadmap) below for what's done vs. planned.

---

## Features

- [x] Federated training pipeline using FedAvg (Flower + PyTorch)
- [ ] FedProx as an alternate aggregation strategy, benchmarked against FedAvg
- [ ] Differential privacy toggle (Opacus), with measured accuracy tradeoff
- [ ] Round-by-round metrics logging (SQLite)
- [ ] Live-updating dashboard (Streamlit)
- [ ] No-code configuration wizard (Streamlit)
- [ ] Automatic code generation from configuration (Jinja2)
- [ ] Downloadable participant bundle

---

## Tech Stack

| Layer | Technology |
|---|---|
| Federated orchestration | [Flower](https://flower.ai) (`flwr`) |
| Model training | PyTorch |
| Differential privacy | Opacus |
| Config & metrics storage | SQLite |
| Code generation | Jinja2 |
| Wizard & dashboard UI | Streamlit |

No Docker, no separate frontend framework, no external database server required —
the entire project runs with plain Python. (A containerized/production-grade version
is noted as future work — see [Roadmap](#roadmap).)

---

## Getting Started

### Prerequisites
- Python 3.10+
- `pip`

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

### 4. Partition the dataset into non-IID client shards
```bash
cd data
python partition_data.py --num_clients 3 --alpha 0.5
cd ..
```

### 5. Run the federated pipeline
Open 4 terminals (all with the virtual environment activated):

```bash
# Terminal 1 — server
cd server
python server.py --rounds 5 --strategy fedavg --min_clients 3
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

You should see per-round accuracy print in each terminal, and results logged to
`server/metrics.csv`.

---

## Project Structure

```
fedsentry/
├── server/
│   ├── model.py       # shared PyTorch model definition
│   ├── server.py       # Flower server: FedAvg / FedProx aggregation
│   └── metrics.csv     # round-by-round training results (generated)
├── client/
│   └── client.py        # Flower client: local training, optional DP
├── data/
│   ├── partition_data.py # non-IID data partitioning script
│   └── partitions/       # generated per-client data indices
├── requirements.txt
└── README.md
```

*(Wizard, dashboard, and code-generation directories will be added as those
milestones are built — see Roadmap.)*

---

## Roadmap

- [x] **Milestone 1:** Working plain federated pipeline (FedAvg, 3 clients, MNIST)
- [ ] **Milestone 2:** Metrics logging + live dashboard
- [ ] **Milestone 3:** Jinja2 code generation from a config
- [ ] **Milestone 4:** Streamlit wizard for no-code configuration
- [ ] **Milestone 5:** Differential privacy + FedProx comparison
- [ ] **Future work:** Dockerized deployment, multi-project support, secure aggregation

---

---

## References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017
- Li et al., *Federated Optimization in Heterogeneous Networks*, MLSys 2020
- Wei et al., *Federated Learning with Differential Privacy: Algorithms and Performance Analysis*, IEEE TIFS 2020
- Flower documentation: https://flower.ai/docs
- Opacus documentation: https://opacus.ai

---

## License

This project is for academic purposes as part of a final year major project submission.
