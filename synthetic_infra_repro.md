# Synthetic 20k Challenge + v6 Infra Repro Guide

This guide reproduces the full cycle requested by the consortium:

1. Generate a schema-valid synthetic BEACH dataset using `datavalgen` + `datavalgen-model-beach`
2. Iteratively enforce strong predictive signal until centralized logistic regression reaches:
   - train ROC-AUC >= `0.90`
   - validation ROC-AUC >= `0.80`
3. Split data for `3-6` organizations with center-based partitioning (a center is never split across nodes)
4. Start local `v6-infrastructure-sh`
5. Run federated ADMM logistic regression (`20kChallengeVantage6`)
6. Save results and tear down infra

## Prerequisites

- Docker daemon running
- Git installed
- Python `3.13+` available on PATH
  - `datavalgen-model-beach` currently requires Python 3.13+
- Open ports:
  - `5070` (v6 server API)
  - `5001` (default local Docker registry; configurable)
  - `80` (v6 UI when `--enable-ui` is used; configurable via `--ui-port`)

## One-command Run (recommended)

From the root of this repository (`20kChallengeVantage6`):

```bash
./scripts/run_20k_v6_infra_cycle.sh \
  --workdir /tmp/20k-v6-work \
  --venv-dir /tmp/20k-v6-venv \
  --node-count 4 \
  --num-subjects 2000 \
  --num-rounds 25
```

Notes:
- The script creates a fresh reusable venv in `/tmp/20k-v6-venv`.
- The script clones dependency repos into `/tmp/20k-v6-work`.
- The script generates and stores synthetic data in:
  - `generated_data/consortium_signal_v1/`
- Age values are normalized to realistic adult years (`40-90`).
- `patient_overall_stage` is regenerated from TNM to keep staging combinations clinically coherent.
- The script always shuts infra down, even on failure (trap cleanup).
- The script prunes Docker cache and unused images/containers after each run by default.
- Federated task submission defaults to `gamma-user/gamma-password` in this setup.
- Python auto-detection checks `python3.13`, then common `pyenv` 3.13 paths.
- UI is disabled by default (headless mode).

## UI Access (optional)

Enable UI explicitly:

```bash
./scripts/run_20k_v6_infra_cycle.sh --enable-ui --ui-port 80
```

Valid users are organization users imported from generated entities:

- `alpha-user / alpha-password`
- `beta-user / beta-password`
- `gamma-user / gamma-password`

`root/root` is not created by this harness.

Linux note: if your browser cannot resolve `host.docker.internal`, UI login may fail even with correct credentials. Add a host alias (for example `127.0.0.1 host.docker.internal`) or run in headless mode.

## Run with Different Node Counts

`--node-count` accepts values `3`, `4`, `5`, or `6`.

Example (`6` nodes):

```bash
./scripts/run_20k_v6_infra_cycle.sh --node-count 6
```

## Artifacts Produced

Main outputs under:

- `generated_data/consortium_signal_v1/synthetic_beach_signal_full.csv`
- `generated_data/consortium_signal_v1/generation_report.json`
- `generated_data/consortium_signal_v1/splits_<N>nodes/*.csv`
- `generated_data/consortium_signal_v1/federated_vs_centralized_<N>nodes.json`

`generation_report.json` includes the achieved train/validation AUC and selected signal parameters.

## Reproduce from `/tmp` on a Fresh Session

If you want to execute entirely from `/tmp` (including this repo copy):

```bash
cd /tmp
rm -rf /tmp/20kChallengeVantage6-local
cp -a /path/to/your/20kChallengeVantage6 /tmp/20kChallengeVantage6-local
cd /tmp/20kChallengeVantage6-local

./scripts/run_20k_v6_infra_cycle.sh \
  --workdir /tmp/20k-v6-work \
  --venv-dir /tmp/20k-v6-venv \
  --node-count 4 \
  --python-bin /path/to/python3.13
```

## Troubleshooting

1. Python version too low

- Symptom: install fails for `datavalgen-model-beach`
- Fix: use Python `3.13+` and pass it explicitly (if not auto-detected):

```bash
./scripts/run_20k_v6_infra_cycle.sh --python-bin /path/to/python3.13
```

2. Registry port conflict (`5001`)

- Symptom: cannot start local registry
- Fix: choose another port:

```bash
./scripts/run_20k_v6_infra_cycle.sh --registry-port 5002
```

3. Stale vantage6 local state / permissions errors

- Symptom: task create/permissions issues after restarts
- Fix: rerun the script (it calls `infra.sh down` before `up`).
- If permissions still fail, submit with an account that has task-create rights:
  - `./scripts/run_20k_v6_infra_cycle.sh --v6-username gamma-user --v6-password gamma-password`
- If needed, manually remove local state:
  - Linux: `~/.local/share/vantage6/`
  - macOS: `~/Library/Application Support/vantage6/`

4. Docker host resolution

- The infra scripts use `SERVER_URL=http://host.docker.internal`.
- This works on macOS and modern Docker on Linux with host-gateway support.

5. Keep the registry container alive for repeated runs

```bash
./scripts/run_20k_v6_infra_cycle.sh --keep-registry
```

6. Disable Docker prune (if you prefer to keep local cache/images)

```bash
./scripts/run_20k_v6_infra_cycle.sh --no-prune-docker
```

## Manual Components (if you need them separately)

Generate only synthetic data:

```bash
python scripts/generate_signal_synthetic_dataset.py \
  --output-dir generated_data/consortium_signal_v1 \
  --num-subjects 2000 \
  --node-counts 3,4,5,6 \
  --default-node-count 4
```

Submit centralized-vs-federated comparison only (infra must already be up):

```bash
python scripts/run_centralized_and_federated_eval.py \
  --full-data generated_data/consortium_signal_v1/synthetic_beach_signal_full.csv \
  --algo-image localhost:5001/20klogregchallenge:synth-4n \
  --output-json generated_data/consortium_signal_v1/federated_vs_centralized_4nodes.json \
  --username gamma-user \
  --password gamma-password \
  --collaboration-name challenge-20k-synth-4n
```
