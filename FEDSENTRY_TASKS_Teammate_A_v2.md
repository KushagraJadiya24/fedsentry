# FedSentry — Task Doc: Teammate A (Federated Systems + Evaluation)

> Paste this + `FEDSENTRY_CONVENTIONS_v2.md` into your Claude session before starting work.

## Your role in one line

You build and validate the "teacher" — the federated server that sends the shared
LoRA adapter out, collects each participant's update, aggregates the updates with
FedAvg, records the training results, and runs the experiments that demonstrate
that the federated system actually works.

Your role is deliberately broader than only writing `server/server.py`: you own the
federated-systems/evaluation side of the project, while still respecting the locked
scope and interfaces in `FEDSENTRY_CONVENTIONS_v2.md`.

## What you own

| Area | What it does |
|---|---|
| `server/server.py` | Flower server, FedAvg strategy, round orchestration, metrics logging |
| Federated verification | Verify that the federated pipeline exchanges/aggregates adapter-only parameters, not the full base model |
| Federated evaluation | Analyze loss across rounds and whether all 3 clients contribute meaningfully |
| Experiment coordination | Run baseline/federated comparisons and document the results used in the final demo/report |
| Integration testing | Verify the real server + 3 real clients complete end-to-end rounds |
| Robustness testing | Test practical server/client behavior such as client availability and multiple rounds without expanding the product scope |

You do **not** own:
- `model/model_utils.py`
- `train()` / `evaluate()` in `client/client.py`
- `data/prepare_data.py`
- `FLClient` implementation in `client/client.py`
- `app/Home.py`

Those remain with the teammates assigned in their task docs.

## Locked interfaces you must implement

### `server/server.py`

```python
METRICS_FILE = "metrics.csv"

def init_metrics_file(): ...
def weighted_average(metrics): ...
```

CLI args (LOCKED names):

```text
--rounds
--min_clients
--address
```

Defaults:

```text
--address 0.0.0.0:8080
```

Strategy is always **FedAvg**.

Do not add:
- `--strategy`
- FedProx
- strategy comparison
- any other aggregation strategy

These are explicitly cut.

### Metrics schema

`server/metrics.csv` has exactly:

```text
round, loss, num_clients
```

Do not add extra columns without updating `FEDSENTRY_CONVENTIONS_v2.md` first.

## Your branches

Primary implementation branch:

```text
feature/flower-server
```

If the team agrees that evaluation work should be separated into focused PRs,
use branches such as:

```text
feature/federated-evaluation
feature/federated-integration-tests
```

Do not create a new top-level application architecture for evaluation. Keep
evaluation/testing lightweight and compatible with the locked repository structure.

## How to start — Day 1

### 1. Build the server against a dummy client

You do not need to wait for anyone.

Set up Flower, read through the `FedAvg` strategy, and build `server.py` against
a fake/dummy client first — a client that returns controlled/random NumPy arrays.

The first target is:

```text
server starts
    ↓
dummy clients connect
    ↓
parameters are returned
    ↓
FedAvg aggregation occurs
    ↓
metrics are produced
```

This proves the Flower server wiring before real LoRA training exists.

### 2. Implement the locked server functions

Implement:

- `init_metrics_file()`
- `weighted_average()`
- FedAvg configuration
- locked CLI arguments
- round/metrics handling

### 3. Establish adapter-only verification

Once Teammate Kushagra's `model_utils.py` is available, verify the contract:

```text
client model
    ↓
get_adapter_parameters()
    ↓
Flower
    ↓
FedAvg
    ↓
aggregated adapter parameters
    ↓
set_adapter_parameters()
```

The server/integration tests should make it possible to demonstrate that the
communicated parameter set corresponds to the LoRA adapter rather than the full
Qwen base model.

Do not modify `model_utils.py` yourself to achieve this. Coordinate with Kushagra
if the locked interface is insufficient.

### 4. Integrate the real client

Once Teammate B's real `FLClient` is ready:

1. Pull the latest `main`.
2. Replace the dummy client.
3. Start the real server.
4. Connect all 3 real clients.
5. Run at least one real federated round.
6. Confirm that real adapter parameters are exchanged.
7. Confirm that a metrics row is produced.

This is the first major end-to-end integration milestone.

## Federated evaluation responsibilities

### A. Loss across rounds

Run multiple federated rounds and inspect:

```text
round 0
round 1
round 2
...
```

Use the locked CSV schema:

```text
round, loss, num_clients
```

The goal is to determine whether the federated process shows a sensible training
signal rather than simply proving that the server did not crash.

Do not assume loss must decrease monotonically on every round; document what the
actual experiment shows.

### B. All-client contribution

The blueprint requires checking that the final result reflects contribution
from more than one participant.

Use the team's non-IID client partitions to investigate:

```text
Client 0 → its local distribution
Client 1 → its local distribution
Client 2 → its local distribution
```

Then evaluate the final shared model on representative examples/metrics that
can show contribution from more than one client's data distribution.

The result should be reported honestly. Do not claim that all clients contributed
equally unless the experiment actually supports that conclusion.

### C. Baseline vs federated comparison

Coordinate a simple comparison:

```text
Base model
    ↓
no fine-tuning

versus

Federated fine-tuned model
    ↓
3 clients + FedAvg
```

Where practical, include a local-only reference run as an additional comparison:

```text
Base model
Local-only fine-tuning
Federated fine-tuning
```

Keep this comparison lightweight. It is an evaluation responsibility, not a new
product feature.

The final demonstration should include the project's required before/after
example generation.

### D. Multi-round integration

Verify the system over multiple rounds, not just one:

```text
Round 1 → clients train → adapters aggregated
Round 2 → clients train → adapters aggregated
Round 3 → clients train → adapters aggregated
...
```

Check that:
- all intended clients participate
- the server remains stable across rounds
- adapter parameter shapes remain compatible
- metrics are recorded correctly
- the final shared adapter can be used for the before/after demonstration

## Robustness testing

Perform a small number of practical tests that improve confidence in the system,
without expanding the locked product scope.

Examples:

- Start the server with the required `--min_clients`.
- Verify behavior when fewer than the required clients are available.
- Run multiple rounds.
- Verify that malformed/incompatible adapter parameters are not silently treated
  as valid training.
- Verify that generated `server/metrics.csv` follows the locked schema.

These are engineering tests, not new features.

Do **not** turn this into a distributed-systems research project.

## Coordination with the other teammates

### With Kushagra

Coordinate around:

- `get_adapter_parameters()`
- `set_adapter_parameters()`
- adapter parameter shapes/order
- what constitutes a valid adapter-only parameter payload
- baseline/final generation evaluation

Kushagra owns the model implementation. You own the federated-system verification
around it.

### With Teammate B

Coordinate around:

- `FLClient`
- Flower client/server connection
- client IDs
- `--server_address`
- `--min_clients`
- real multi-client rounds

B owns the client-side Flower wiring. You own the server-side orchestration and
federated integration verification.

## Definition of done — Phase 1

### Server implementation

- [ ] `server.py` runs with the locked CLI arguments.
- [ ] FedAvg is configured correctly.
- [ ] `init_metrics_file()` works.
- [ ] `weighted_average()` works.
- [ ] `metrics.csv` uses exactly `round, loss, num_clients`.
- [ ] Server can complete multiple rounds with 3 clients.

### Federated-system verification

- [ ] Real `FLClient` instances connect successfully.
- [ ] Real adapter parameters are exchanged.
- [ ] The system does not exchange the full base model as the federated payload.
- [ ] Adapter parameter shapes remain compatible across rounds.
- [ ] `--min_clients` behavior has been tested.

### Evaluation

- [ ] Loss across multiple rounds has been recorded and inspected.
- [ ] The non-IID client distributions have been considered when interpreting results.
- [ ] Evidence has been collected showing contribution from more than one client/data distribution.
- [ ] A base-vs-federated comparison has been run.
- [ ] A clean before/after generation example is available for the final demo.
- [ ] The team has a short, defensible explanation of what the experiments show.

## Guardrails from the Conventions doc

- FedAvg only — no FedProx.
- No strategy-selection CLI argument.
- Metrics go to a plain CSV — no SQLite.
- CLI args always long-form `--snake_case`.
- Commit format:

```text
[server] short description
```

For evaluation/integration commits, use:

```text
[server] add federated evaluation
[test] verify adapter aggregation
```

- Do not introduce a new library without team agreement.
- Do not change the locked base model without updating the conventions and notifying
  all teammates.
- Do not add multi-model selection, differential privacy, Jinja2 code generation,
  or other explicitly cut features.
