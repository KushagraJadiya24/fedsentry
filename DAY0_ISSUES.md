# FedSentry — Day-0 GitHub Issues

Create these as GitHub Issues/Project cards. Keep the conventions document as the source of truth.

## P0 — Repository bootstrap
**Owner:** Kushagra

- [ ] Add locked repository structure
- [ ] Add `.gitignore`
- [ ] Add `requirements.txt`
- [ ] Add README golden path
- [ ] Add CONTRIBUTING.md
- [ ] Add contract/smoke tests
- [ ] Confirm all teammates can create venv + install dependencies

**Done when:** fresh clone installs and `pytest -q` runs.

## P0 — Data partitions
**Owner:** Teammate B

Branch: `feature/data-prep`

- [ ] Select small instruction dataset
- [ ] Document source in `prepare_data.py` docstring
- [ ] Implement non-IID 3-way split
- [ ] Generate `client_0.jsonl`, `client_1.jsonl`, `client_2.jsonl`
- [ ] Validate schema
- [ ] Commit generated partitions

**Done when:** all three locked JSONL files exist and validate.

## P0 — Local LoRA model utilities
**Owner:** Kushagra

Branch: `feature/lora-model-utils`

- [ ] Load Qwen/Qwen2.5-0.5B-Instruct
- [ ] Apply LoRA
- [ ] Extract adapter-only parameters
- [ ] Restore adapter-only parameters
- [ ] Test parameter round-trip

**Done when:** model loads, LoRA attaches, adapter weights round-trip.

## P0 — Local LoRA training
**Owner:** Kushagra

Branch: `feature/lora-client-training`

- [ ] Implement `load_client_data`
- [ ] Implement tokenization/training dataset path
- [ ] Implement local `train`
- [ ] Implement `evaluate`
- [ ] Demonstrate before/after output change

**Done when:** one client can fine-tune locally without Flower.

## P0 — Flower server with dummy client
**Owner:** Teammate A

Branch: `feature/flower-server`

- [ ] Configure FedAvg
- [ ] Implement locked CLI
- [ ] Implement metrics initialization
- [ ] Implement weighted metric aggregation
- [ ] Run one dummy round
- [ ] Verify `metrics.csv`

**Done when:** server completes a dummy round and writes the locked CSV schema.

## P0 — Adapter-only federated verification
**Owner:** Teammate A

- [ ] Verify Flower payloads correspond to LoRA adapter parameters
- [ ] Verify parameter shapes/order remain compatible
- [ ] Verify full base-model weights are not being exchanged
- [ ] Coordinate with Kushagra rather than changing `model_utils.py`

**Done when:** the team can demonstrate and explain the adapter-only exchange path.

## P1 — Federated evaluation
**Owner:** Teammate A

- [ ] Run multiple federated rounds
- [ ] Inspect loss across rounds
- [ ] Analyze all three non-IID client distributions
- [ ] Collect evidence that more than one client contributes to the final result
- [ ] Run base-vs-federated comparison
- [ ] Optionally include a local-only reference run
- [ ] Prepare the before/after example used in the final demo

**Done when:** the team has a short, evidence-based explanation of what the federated experiments demonstrate.

## P1 — Integration and robustness testing
**Owner:** Teammate A

- [ ] Run 3 real clients against the real server
- [ ] Run multiple rounds end-to-end
- [ ] Test `--min_clients` behavior
- [ ] Verify metrics schema
- [ ] Check incompatible/malformed adapter payload behavior

**Done when:** the Phase-1 pipeline is reliably demoable and the federated behavior is defensible.

## P0 — Flower client wiring
**Owner:** Teammate B

Branch: `feature/flower-client`

- [ ] Implement `FLClient`
- [ ] Use stub `train`/`evaluate` initially
- [ ] Implement adapter get/set through model_utils
- [ ] Implement locked CLI
- [ ] Connect to dummy server

**Done when:** client can complete a round using stubs.

## P1 — First real integration
**Owners:** All

- [ ] Replace dummy/stub components with real implementations
- [ ] Run one real federated round
- [ ] Verify only adapter weights cross Flower boundary
- [ ] Verify all 3 clients participate

**Done when:** one real round completes end-to-end.

## P1 — Multi-round hardening
**Owners:** All

- [ ] Run 3+ rounds
- [ ] Log loss per round
- [ ] Inspect client data distribution
- [ ] Produce before/after generation
- [ ] Fix reproducibility/path issues

**Done when:** terminal Phase 1 is reliably demoable.

## P2 — Streamlit
**Owner:** Teammate B

Branch: `feature/streamlit-ui`

Do not start until Phase 1 is frozen and working.

- [ ] Single-page UI
- [ ] Rounds input
- [ ] Start Training button
- [ ] Loss display
- [ ] Before/after generation

**Done when:** UI calls the existing Python training pipeline directly.
