"""
Partial (node-side) functions for the ADMM logistic regression algorithm.

Each function in this module is executed on the data-owning nodes. The central
algorithm (see ``central.py``) orchestrates the ADMM rounds by repeatedly
calling these functions via vantage6 tasks.

This version is adapted to 20kChallenge-style data. On each node, the local
table/CSV is expected to contain at least the following columns:

    - patient_identifier                               (ignored)
    - patient_t_stage
    - patient_n_stage
    - patient_m_stage
    - patient_overall_stage
    - year_of_diagnosis                               (used only for time split)
    - interval_diagnosis_to_last_visit_in_days        (used only for outcome)
    - vital_status                                    (alive / dead)
    - centre                                          (ignored)
    - age_at_diagnosis                                (ignored)

We:
  - derive a binary outcome ``twoYearSurvival`` from ``vital_status`` and
    ``interval_diagnosis_to_last_visit_in_days`` following the 20kChallenge
    definition;
  - use only the four stage-like variables as features:
        {'patient_t_stage', 'patient_n_stage',
         'patient_m_stage', 'patient_overall_stage'};
  - split data into train/validation by year_of_diagnosis:
        * train: year_of_diagnosis <= 2011
        * val:   year_of_diagnosis >= 2012
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from vantage6.algorithm.tools.decorators import data
from vantage6.algorithm.tools.util import info


np.random.seed(67)


# ============================================================================
# Helper functions: data loading & preprocessing
# ============================================================================

# Features analogous to {'tLabel','nLabel','mLabel','stageLabel'}
FEATURE_COLUMNS = (
    "patient_t_stage", "patient_n_stage", "patient_m_stage", "patient_overall_stage"
)
OUTCOME_COLUMN = "SurvivalStatus"

EXPECTED_CATEGORIES = {
    "patient_t_stage": ["Tx","Tis","T0","T1","T1mi","T1a","T1b","T1c","T2","T2a","T2b","T3","T4"],
    "patient_n_stage": ["Nx","N0","N1","N2","N3"],
    "patient_m_stage": ["Mx","M0","M1","M1a","M1b","M1c"],
    "patient_overall_stage": ["0","Occult","I","IA","IA1","IA2","IA3","IB","II","IIA","IIB","III","IIIA","IIIB","IIIC","IV","IVA","IVB","x"],
}

def _compute_two_year_survival(
    vital_status: pd.Series, days_until_last_visit: pd.Series
) -> pd.Series:
    """
    Compute the binary two-year survival variable in line with the
    20kChallenge rules.

    Rules (2 years = 2 * 365.24 days):
      - if vital status missing                 -> NaN
      - if dead  and days <= 2y                 -> 0
      - if dead  and days  >  2y                -> 1
      - if alive and days <= 2y                 -> NaN (insufficient follow-up)
      - if alive and days  >  2y                -> 1
    """
    vs = vital_status.astype(str).str.lower().str.strip()
    days = pd.to_numeric(days_until_last_visit, errors="coerce")

    out = pd.Series(np.nan, index=vital_status.index, dtype=float)
    threshold = 2 * 365.24

    dead = vs == "dead"
    alive = vs == "alive"

    out[dead] = 0
    out[alive] = 1
    # out[dead & (days <= threshold)] = 0 For right data
    # out[dead & (days > threshold)] = 1
    # out[alive & (days > threshold)] = 1
    # out[alive & (days <= threshold)] = 

    return out


def _preprocess_local_dataframe(
    df: pd.DataFrame,
    logging: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Node-local preprocessing:
      - derive twoYearSurvival from vital_status and follow-up time,
      - encode the four stage features as one-hot (K-1) dummies,
      - split into train/validation sets based on diagnosis year.
    """
    df = df.copy()

    # 1. Check and derive twoYearSurvival
    if "vital_status" not in df.columns:
        raise ValueError("Expected column 'vital_status' not found in input data")
    if "interval_diagnosis_to_last_visit_in_days" not in df.columns:
        raise ValueError(
            "Expected column 'interval_diagnosis_to_last_visit_in_days' not found; "
            "cannot compute twoYearSurvival"
        )
    # df[OUTCOME_COLUMN] = df[OUTCOME_COLUMN].astype(str).str.lower().str.strip()
    df[OUTCOME_COLUMN] = _compute_two_year_survival(
        df["vital_status"], df["interval_diagnosis_to_last_visit_in_days"]
    )

    # num_dead = (df[OUTCOME_COLUMN] == 0).sum()
    # num_alive = (df[OUTCOME_COLUMN] == 1).sum()
    # num_nan = df[OUTCOME_COLUMN].isna().sum()

    # print(f"Number of dead: {num_dead}")
    # print(f"Number of alive: {num_alive}")
    # print(f"Number of censored/unknown (NaN): {num_nan}")

    # 2. Apply fixed category sets to the four stage features
    for col in FEATURE_COLUMNS:
        # if col not in df.columns:
        #     raise ValueError(f"Expected feature column '{col}' not found in input data")
        df[col] = pd.Categorical(df[col], categories=EXPECTED_CATEGORIES[col])

    # 3. Ensure diagnosis year is present for time-based split
    if "year_of_diagnosis" not in df.columns:
        raise ValueError(
            "Expected column 'year_of_diagnosis' not found; required for train/val split"
        )
    df["__diag_year__"] = pd.to_numeric(df["year_of_diagnosis"], errors="coerce")

    # 4. Drop rows with missing outcome, features, or diagnosis year
    initial_count = len(df)
    df = df.dropna(subset=list(FEATURE_COLUMNS) + [OUTCOME_COLUMN, "__diag_year__"])
    # if len(df) < initial_count:
        # info(
            # f"Dropped {initial_count - len(df)} rows due to invalid/missing data "
            # f"for {FEATURE_COLUMNS + (OUTCOME_COLUMN, '__diag_year__')}"
        # )
    if logging == True:
        print(f"Total pooled rows after cleaning: {len(df)}")

    y = df[OUTCOME_COLUMN].astype(int).to_numpy()
    
    # 5. Encode features (one-hot, K-1 encoding) and outcome
    df_cat = pd.get_dummies(df[list(FEATURE_COLUMNS)], drop_first=True)
    X = df_cat.to_numpy(dtype=float)
    years = df["__diag_year__"].to_numpy().astype(int)

    # 6. Time-based split: train <= 2011, val >= 2012
    train_mask = years <= 2011
    val_mask = years >= 2012

    if not train_mask.any():
        info("Warning: no training patients with diagnosis year <= 2011 on this node")
    if not val_mask.any():
        info("Warning: no validation patients with diagnosis year >= 2012 on this node")

    # 2. Define the dimensions
    # Fix: Get the number of columns correctly
    num_cols = X.shape[1] 

    return X[train_mask], y[train_mask], X[val_mask], y[val_mask]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _predict_proba(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """
    Probabilities P(y=1 | X, coef)
    """
    
    n = X.shape[0]
    X_design = np.hstack([np.ones((n, 1)), X])
    return _sigmoid(X_design @ coef)


def _logistic_admm_objective(
    x: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    rho: float,
    features: np.ndarray,
    outcome: np.ndarray,
    total_patients: int,
) -> Tuple[float, np.ndarray]:
    """
    Objective function and gradient for the local X-update (site optimization),
    copied from your standalone script.
    """
    n_local = features.shape[0]
    X_design = np.hstack([np.ones((n_local, 1)), features])  # Add intercept
    logits = X_design @ x
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    eps = 1e-15
    probs = np.clip(probs, eps, 1.0 - eps)
    # logits = X_design @ x
    # probs = 1.0 / (1.0 + np.exp(-logits))

    # Logistic deviance (scaled by -2/N)
    val_logistic = (-2.0 / total_patients) * np.sum( outcome * np.log(probs ) + (1.0 - outcome) * np.log(1.0 - probs) )

    # ADMM penalty: (rho/2) ||x - z + u||^2
    diff = x - z + u
    val_quad = 0.5 * rho * np.dot(diff, diff)

    value = val_logistic + val_quad

    # Gradient
    grad_sum = np.zeros_like(x)
    for i in range(n_local):
        xi = X_design[i, :]  # row vector
        yi = outcome[i]
        exp_term = np.exp(xi @ x)
        grad_i = xi * (yi + (yi - 1) * exp_term) / (1 + exp_term)
        grad_sum += grad_i

    grad_logistic = (-2.0 / total_patients) * grad_sum
    grad_quad = rho * diff
    grad = grad_logistic + grad_quad
    return value, grad


# ============================================================================
# Node-side functions called from the central orchestrator
# ============================================================================


@data(1)
def init_node(df1: pd.DataFrame, logging:bool) -> Dict[str, Any]:
    """
    Initialization call from the central function.

    It runs the full preprocessing pipeline once to determine the local number
    of features and the local patient count. The central algorithm uses this
    to set up the ADMM state and hyperparameters.
    """
    X_train, y_train, X_val, y_val = _preprocess_local_dataframe(df1,logging)

    num_features = X_train.shape[1]
    patient_count = int(X_train.shape[0])

    return {
        "num_features": int(num_features),
        "patient_count": patient_count,
    }


@data(1)
def admm_x_update_partial(
    df1: pd.DataFrame,
    z: Any,
    u: Any,
    rho: float,
    total_patients: int,
    x_prev: Any | None = None,
    logging: bool = True
) -> Dict[str, Any]:
    """
    Local X-update for one ADMM round.

    Parameters are provided by the central function:
    - z: current global consensus parameter vector
    - u: current local dual variable for this node
    - rho: ADMM penalty parameter
    - total_patients: global number of patients across all nodes
    """

    X_train, y_train, X_val, y_val = _preprocess_local_dataframe(df1, logging)

    num_features = X_train.shape[1]

    z_arr = np.asarray(z, dtype=float).reshape(-1)
    u_arr = np.asarray(u, dtype=float).reshape(-1)


    res = minimize(
        fun=_logistic_admm_objective,
        x0=x_prev,
        args=(z_arr, u_arr, float(rho), X_train, y_train, int(total_patients)),
        jac=True,
        method="BFGS",
        options={"disp": False},
    )

    x_new = res.x

    # Training accuracy (as in your script)
    probs = _predict_proba(X_train, x_new)
    preds = (probs >= 0.5).astype(int)
    acc = float(np.mean(preds == y_train))
    if logging == True:
        info(f"Local training accuracy: {acc:.4f}")

    # --- Training SSE and objective value for global logging ---
    residuals = y_train - probs
    sum_square_error = float(np.sum(residuals**2))
    obj = float(res.fun)

    patient_count = int(X_train.shape[0])

    return {
        "x": x_new.tolist(),
        "patient_count": patient_count,
        "num_features": int(num_features),
        "train_acc": acc,
        "sum_square_error": sum_square_error,
        "obj": obj,
    }


@data(1)
def evaluate_global_model(
    df1: pd.DataFrame,
    z: Any,
    logging: bool = True
) -> Dict[str, Any]:
    """
    Evaluate the current global consensus vector ``z`` on the local validation
    data of this node. This mirrors the validation logic in your standalone
    simulation.
    """
    X_train, y_train, X_val, y_val = _preprocess_local_dataframe(df1, logging)

    z_arr = np.asarray(z, dtype=float).reshape(-1)
    probs = _predict_proba(X_val, z_arr)
    preds = (probs >= 0.5).astype(int)
    val_acc = float(np.mean(preds == y_val))
    if logging == True:
        info(f"Validation accuracy for global model on this node: {val_acc:.4f}")

    # Validation SSE and patient count for global RMSE logging
    residuals_val = y_val - probs
    val_sum_square_error = float(np.sum(residuals_val**2))
    val_patient_count = int(y_val.shape[0])

    return {
        "val_acc": val_acc,
        "val_sum_square_error": val_sum_square_error,
        "val_patient_count": val_patient_count,
    }


@data(1)
def collect_predictions(
    df1: pd.DataFrame,
    z: Any,
    logging: bool,
) -> Dict[str, Any]:
    """
    Return predictions and outcomes (train + val) for the final global model.

    This is used by the central function after convergence to compute
    ROC/AUC and calibration metrics in a way analogous to the standalone
    local simulation.
    """
    X_train, y_train, X_val, y_val = _preprocess_local_dataframe(df1,logging)

    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    z_arr = np.asarray(z, dtype=float).reshape(-1)
    probs_all = _predict_proba(X_all, z_arr)

    return {
        "y_all": y_all.tolist(),
        "probs_all": probs_all.tolist(),
    }
