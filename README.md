# How to Run: run_on_v6_network.py, MockClient.py, ADMM_Local.py, LocalLogReg.py

This guide explains how to run each of the four main scripts in the project.

---

## Reference (basis of this code)

This project is **based on** the code and ideas from:

**Repository:** [RadiationOncologyOntology/20kChallenge](https://github.com/RadiationOncologyOntology/20kChallenge/)  
*Distributed learning over 20k+ patients*

That repository accompanies the manuscript:

> **Distributed learning on 20 000+ lung cancer patients – The Personal Health Train**  
> T.M. Deist, F.J.W.M. Dankers, P. Ojha, S.M. Marshall, T. Janssen, C. Faivre-Finn, C. Masciocchi, V. Valentini, J. Wang, J. Chen, Z. Zhang, E. Spezi, M. Button, J.J. Nuyttens, R. Vernhout, J. van Soest, A. Jochems, R. Monshouwer, J. Bussink, G. Price, P. Lambin, A. Dekker  
> *(Bolded and italicized authors in the original publication contributed equally.)*

The [20kChallenge](https://github.com/RadiationOncologyOntology/20kChallenge/) repository provides code to:

- train a logistic regression model  
- validate a logistic regression model  

in a **distributed** setting (master/site architecture). The upstream project targets **MATLAB** and the **Varian Learning Portal 2.1**; this codebase adapts the same algorithmic ideas (e.g. ADMM for distributed logistic regression) in **Python** with vantage6.

See the **wiki** linked from the [20kChallenge README](https://github.com/RadiationOncologyOntology/20kChallenge/).

---

## Overview: What Each Script Does

| Script | Description |
|--------|-------------|
| **LocalLogReg.py** | Local **centralised** logistic regression model. Used as a **baseline** for comparison. Trains on pooled data (single CSV). |
| **ADMM_Local.py** | **Distributed** ADMM logistic regression run on a **single machine** without real nodes. Simulates the federated setup locally. |
| **MockClient.py** | Same algorithm as ADMM_Local.py, but uses vantage6’s **MockAlgorithmClient** instead of a custom simulation. |
| **run_on_v6_network.py** | **True distributed** run on the vantage6 network: 3 nodes + server on SURF Research Cloud. |

**Result consistency:** With the same hyperparameters, the global model coefficients (z) are **identical** between `run_on_v6_network.py`, `ADMM_Local.py`, and `MockClient.py`. `LocalLogReg.py` (pooled baseline) shows an average of about **0.5% difference** in model parameters compared to the distributed ADMM approaches.

---

## Prerequisites

**Python packages** (install in your environment, e.g. `v6-workshop`):

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch
pip install vantage6-client vantage6-algorithm-tools
```

**Data files** (must exist):

- `20kLogRegChallenge/test/fakebeach_merged_0.csv`
- `20kLogRegChallenge/test/fakebeach_merged_1.csv`
- `20kLogRegChallenge/test/fakebeach_merged_2.csv`
- `20kLogRegChallenge/test/fakebeach_merged_FULL.csv` (for LocalLogReg.py)

---

## 1. run_on_v6_network.py

**Purpose:** Submits an ADMM logistic regression task to the **real vantage6 network** (3 nodes + server on SURF Research Cloud). True distributed federated learning. Runs on your machine; the algorithm runs on the remote nodes.

**Location:** `20kLogRegChallenge/run_on_v6_network.py`

**Requirements:**

- vantage6 server running and reachable
- Nodes running and connected (see `NODE1_SETUP.md`)
- Algorithm image pushed to Docker Hub: `surfzare/20kLogRegChallenge`
- `vantage6-client` installed

**How to run:**

```bash
cd 20kLogRegChallenge
python run_on_v6_network.py
```

Or with full path:

```bash
python "/path/to/20kLogRegChallenge/run_on_v6_network.py"
```

**Configuration:** Edit the script to change server URL, credentials, or ADMM parameters (`input_params`).

**Output:** Prints task status and, when complete, coefficients, patient counts, accuracy, AUC, and calibration.

---

## 2. MockClient.py

**Purpose:** Same distributed ADMM algorithm as ADMM_Local.py, but uses vantage6’s **MockAlgorithmClient**. Simulates 3 nodes with local CSVs, no real vantage6 server or nodes. Produces identical z (global model) to run_on_v6_network.py and ADMM_Local.py with the same hyperparameters.

**Location:** `20kLogRegChallenge/test/MockClient.py`

**Requirements:**

- `20kLogRegChallenge` package installed (editable)
- `vantage6-algorithm-tools` (provides `MockAlgorithmClient`)
- Test CSVs in `20kLogRegChallenge/test/`

**How to run:**

```bash
# 1. Install the algorithm package (one-time)
cd 20kLogRegChallenge
pip install -e .

# 2. Run the test
python test/MockClient.py
```

Or from the project root:

```bash
cd 20kLogRegChallenge
python -m test.MockClient
```

**Output:** Prints ADMM results and opens matplotlib plots (convergence, accuracy per node, ROC curves, calibration).

---

## 3. ADMM_Local.py

**Purpose:** **Distributed** ADMM logistic regression run on a **single machine** without real nodes. Simulates the federated setup locally (no vantage6, no Docker). Produces identical z (global model) to run_on_v6_network.py and MockClient.py with the same hyperparameters.

**Location:** `local_simulation/ADMM_Local.py` (or `Ivan_Code/Code/local_simulation/ADMM_Local.py`)

**Requirements:**

- numpy, pandas, scipy, scikit-learn, matplotlib
- CSV paths set in the `__main__` section (see below)

**How to run:**

```bash
cd local_simulation
python ADMM_Local.py
```

Or with full path:

```bash
python "/path/to/local_simulation/ADMM_Local.py"
```

**Configuring CSV paths:** Edit the `my_csv_paths` list at the bottom of `ADMM_Local.py`:

```python
if __name__ == "__main__":
    my_csv_paths = [
        "/path/to/fakebeach_merged_0.csv",
        "/path/to/fakebeach_merged_1.csv",
        "/path/to/fakebeach_merged_2.csv",
    ]
    run_local_simulation(my_csv_paths, num_rounds=100, ...)
```

**Output:** Prints coefficients, validation metrics, and saves `logistic_regression_parameters123.xlsx`.

---

## 4. LocalLogReg.py

**Purpose:** **Centralised** logistic regression baseline. Trains on pooled data (single CSV) with all data combined. No ADMM, no federated learning. Used to compare against the distributed approaches; model parameters differ by ~0.5% on average.

**Location:** `my-fl-project/LocalLogReg.py`

**Requirements:**

- numpy, pandas, scipy, torch
- Path to `fakebeach_merged_FULL.csv` set in the script

**How to run:**

```bash
cd my-fl-project
python LocalLogReg.py
```

Or with full path:

```bash
python "/path/to/my-fl-project/LocalLogReg.py"
```

**Configuring CSV path:** Edit the path in the `__main__` section:

```python
if __name__ == "__main__":
    local_csvs = "/path/to/20kLogRegChallenge/test/fakebeach_merged_FULL.csv"
    # ...
```

**Output:** Prints coefficients and variable names.

---

## Quick reference

| Script | Purpose | Needs vantage6? | Needs Docker image? |
|--------|---------|-----------------|---------------------|
| `run_on_v6_network.py` | True distributed (3 nodes + server on SURF) | Yes | Yes |
| `MockClient.py` | Distributed ADMM via MockClient (same z as above) | No | No |
| `ADMM_Local.py` | Distributed ADMM on single machine (same z as above) | No | No |
| `LocalLogReg.py` | Centralised baseline (~0.5% diff in params) | No | No |

---

## Typical workflow

1. **Baseline:** `LocalLogReg.py` – centralised pooled regression
2. **Develop:** `ADMM_Local.py` – distributed ADMM on single machine
3. **Validate:** `MockClient.py` – same algorithm via vantage6 MockClient (identical z)
4. **Deploy:** `run_on_v6_network.py` – true distributed on SURF Research Cloud (identical z)
