import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from typing import Any, Dict, Tuple

# ============================================================================
# 1. Configuration & Constants
# ============================================================================
np.random.seed(67)

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

def LossLogist(x,X_Train,Y_Train, patients)-> Tuple[float, np.ndarray]:
    """
    Objective function and gradient for the local X-update (site optimization),
    copied from your standalone script.
    """
    n_local = X_Train.shape[0]
    X_design = np.hstack([np.ones((n_local, 1)), X_Train])  # Add intercept
    logits = X_design @ x
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    eps = 1e-15
    probs = np.clip(probs, eps, 1.0 - eps)

    value = (-2.0 / patients) * np.sum( Y_Train * np.log(probs ) + (1.0 - Y_Train) * np.log(1.0 - probs) )

    # Gradient
    grad_sum = np.zeros_like(x)
    for i in range(n_local):
        xi = X_design[i, :]  # row vector
        yi = Y_Train[i]
        exp_term = np.exp(xi @ x)
        grad_i = xi * (yi + (yi - 1) * exp_term) / (1 + exp_term)
        grad_sum += grad_i

    grad = (-2.0 / patients) * grad_sum
    return value, grad
# ============================================================================
# 2. Exact Preprocessing Matches
# ============================================================================
def _compute_two_year_survival(vital_status: pd.Series, days_until_last_visit: pd.Series) -> pd.Series:
    vs = vital_status.astype(str).str.lower().str.strip()
    days = pd.to_numeric(days_until_last_visit, errors="coerce")
    out = pd.Series(np.nan, index=vital_status.index, dtype=float)
    
    dead = vs == "dead"
    alive = vs == "alive"
    out[dead] = 0
    out[alive] = 1
    return out

def load_and_preprocess_pooled_data(csv_file) -> tuple:
    df = pd.read_csv(csv_file)
    
    # 1. Compute Outcome
    df[OUTCOME_COLUMN] = _compute_two_year_survival(df["vital_status"], df["interval_diagnosis_to_last_visit_in_days"])

    # 2. Force Categorical Levels
    for col in FEATURE_COLUMNS:
        df[col] = pd.Categorical(df[col], categories=EXPECTED_CATEGORIES[col])

    # 3. Time Variable
    df["__diag_year__"] = pd.to_numeric(df["year_of_diagnosis"], errors="coerce")
    
    # 4. Drop NaNs
    df = df.dropna(subset=list(FEATURE_COLUMNS) + [OUTCOME_COLUMN, "__diag_year__"])
    print(f"Total pooled rows after cleaning: {len(df)}")

    # 5. Dummy Encoding (Drop First)
    y = df[OUTCOME_COLUMN].astype(int).to_numpy()
    df_cat = pd.get_dummies(df[list(FEATURE_COLUMNS)], drop_first=True)
    X = df_cat.to_numpy(dtype=float)
    feature_names = df_cat.columns.tolist()
    
    # 6. Time Split
    years = df["__diag_year__"].to_numpy().astype(int)
    train_mask = years <= 2011
    val_mask = years >= 2012

    return X[train_mask], y[train_mask], X[val_mask], y[val_mask], feature_names

# ============================================================================
# 3. Main Execution
# ============================================================================
if __name__ == "__main__":
    local_csvs =  "20kLogRegChallenge/test/fakebeach_merged_FULL.csv"
    # # 1. Preprocess
    X_train, y_train, X_val, y_val, feature_names = load_and_preprocess_pooled_data(local_csvs)

    num_features = X_train.shape[1]
    total_patients = X_train.shape[0]
    x_prev = np.zeros(num_features + 1)

    res = minimize(
        fun=LossLogist,
        x0=x_prev,
        args=(X_train, y_train, int(total_patients)),
        jac=True,
        method="BFGS",
        options={"disp": False},
    )
    
    print(f"\nTraining set: {X_train.shape[0]} patients, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} patients")

    original_style_params = res.x

    print("\n" + "="*60)
    print(f"{'Index':<8} {'Variable Name':<40} {'Value':<12}")
    print("-" * 60)
    print(f"{0:<8} {'(Intercept)':<40} {original_style_params[0]:.6f}")
    for i, name in enumerate(feature_names):
        print(f"{i+1:<8} {name:<40} {original_style_params[i+1]:.6f}")
    print("="*60)

    print("\nVector representation (z_arr):")
    print(original_style_params.tolist())
