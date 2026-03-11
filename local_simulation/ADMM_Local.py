"""
Single-file ADMM Logistic Regression Simulation (Local CSV Version).

This script runs a fully local simulation using your specific CSV files.
It includes the exact data loading and preprocessing logic from your original
package (handling 'Dead'/'Alive' mapping, specific categorical stages, etc.).

Instructions:
1. Scroll to the bottom ("__main__" section).
2. Update the `csv_paths` list with the actual locations of your CSV files.
3. Run the script.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression

np.random.seed(67)

# ==============================================================================
# 1. Data Loading & Preprocessing
# ==============================================================================

@dataclass
class LocalData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray


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
    """Exactly matches your provided outcome logic."""
    vs = vital_status.astype(str).str.lower().str.strip()
    # Note: Threshold logic remains as you provided (0 for dead, 1 for alive)
    out = pd.Series(np.nan, index=vital_status.index, dtype=float)
    
    dead = vs == "dead"
    alive = vs == "alive"
    
    out[dead] = 0
    out[alive] = 1
    return out

def load_local_patient_data(csv_path: Path | str) -> LocalData:
    """
    100% aligned with _preprocess_local_dataframe requirements:
    - Survival derivation
    - Time-based split (2011/2012)
    - K-1 dummy encoding
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # 1. Compute Outcome
    if "vital_status" not in df.columns or "interval_diagnosis_to_last_visit_in_days" not in df.columns:
        raise ValueError("Missing columns for survival computation")
        
    df[OUTCOME_COLUMN] = _compute_two_year_survival(
        df["vital_status"], df["interval_diagnosis_to_last_visit_in_days"]
    )

    # 2. Categorical Setup (Fixed category sets)
    for col in FEATURE_COLUMNS:
        df[col] = pd.Categorical(df[col], categories=EXPECTED_CATEGORIES[col])

    # 3. Preparation for Time-based Split
    if "year_of_diagnosis" not in df.columns:
        raise ValueError("year_of_diagnosis missing - required for time-based split")
    
    df["__diag_year__"] = pd.to_numeric(df["year_of_diagnosis"], errors="coerce")

    # 4. Cleaning
    # Drop rows missing outcome, features, or the split year
    initial_count = len(df)
    clean_subset = list(FEATURE_COLUMNS) + [OUTCOME_COLUMN, "__diag_year__"]
    df = df.dropna(subset=clean_subset)


    # 5. Encoding (K-1 Dummies)
    # Using drop_first=True to match your _preprocess_local_dataframe
    df_cat = pd.get_dummies(df[list(FEATURE_COLUMNS)], drop_first=True)
    X = df_cat.to_numpy(dtype=float)
    y = df[OUTCOME_COLUMN].astype(int).to_numpy()
    years = df["__diag_year__"].to_numpy().astype(int)
    feature_names = df_cat.columns.tolist()

    # 6. Time-based split (<= 2011 Training, >= 2012 Validation)
    train_mask = years <= 2011
    val_mask = years >= 2012

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    return LocalData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val), feature_names


# ==============================================================================
# 2. State Containers
# ==============================================================================

@dataclass
class InstanceState:
    rho: float
    alpha: float
    lambda_: float
    abs_tol: float
    rel_tol: float
    patient_counts: List[int]
    x_init: np.ndarray
    u_init: np.ndarray
    z_init: np.ndarray


@dataclass
class SiteState:
    x: np.ndarray
    u: np.ndarray
    z: np.ndarray
    z_old: np.ndarray | None = None
    patient_count: int | None = None

    # Per-site metrics for training progress logging
    sum_square_error: float = 0.0  # training SSE
    obj: float = 0.0               # total objective (logistic + ADMM penalty)
    reg_obj: float = 0.0           # global regularization term (filled from z-update)


@dataclass
class LocalNode:
    index: int
    data: LocalData
    state: SiteState


# ==============================================================================
# 3. ADMM Math & Updates
# ==============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict_proba(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    "Just to check "
    n = X.shape[0]
    X_design = np.hstack([np.ones((n, 1)), X])
    return sigmoid(X_design @ coef)


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

def admm_x_update(node: LocalNode, instance: InstanceState) -> SiteState:
    """
    Performs local optimization (x-update) for one site.
    Matches the original partial function functionality exactly.
    """
    
    X_train = node.data.X_train
    y_train = node.data.y_train
    state = node.state

    total_patients = int(sum(instance.patient_counts))
    rho = float(instance.rho)
    
    x_start = np.asarray(state.x, dtype=float).reshape(-1)
    z_arr = np.asarray(state.z, dtype=float).reshape(-1)
    u_arr = np.asarray(state.u, dtype=float).reshape(-1)

    res = minimize(
        fun=_logistic_admm_objective,
        x0=x_start,
        args=(z_arr, u_arr, rho, X_train, y_train, total_patients),
        jac=True,
        method="BFGS",
        options={"disp": False},
    )
    
    # Update state with new local x
    state.x = res.x

    probs = predict_proba(X_train, state.x)
    preds = (probs >= 0.5).astype(int)
    acc = float(np.mean(preds == y_train))
    
    # Print training accuracy as requested
    print(f"Node || Training Accuracy: {acc:.4f}")
    residuals = y_train - probs
    state.sum_square_error = float(np.sum(residuals**2))
    
    # ADMM objective value at optimum x
    state.obj = float(res.fun)
    
    state.patient_count = int(X_train.shape[0])
    state.num_features = int(X_train.shape[1])
    state.train_acc = acc

    return state


def _z_objective(
    z: np.ndarray,
    x_hat_mean: np.ndarray,
    u_mean: np.ndarray,
    rho: float,
    lambda_: float,
    num_sites: int,
) -> Tuple[float, np.ndarray]:
    """
    Exact Python implementation of the Z-update objective from your script.
    """
    
    z_reg = z.copy()
    z_reg[0] = 0.0  # Skip intercept penalty

    # Value
    l1_val = lambda_ * np.linalg.norm(z_reg, 1)
    diff = z - x_hat_mean - u_mean
    quad_val = (num_sites * rho / 2.0) * np.dot(diff, diff)
    value = l1_val + quad_val

    # Gradient
    eps = 1e-64
    abs_z = np.abs(z_reg)
    abs_z[0] = eps

    grad_l1 = lambda_ * (z_reg / abs_z)
    # grad_l1[0] = 0.0

    grad_quad = num_sites * rho * diff
    grad = grad_l1 + grad_quad
    return value, grad

def admm_z_u_update(nodes: List[LocalNode], instance: InstanceState):
    x_mat = np.column_stack([node.state.x for node in nodes])
    z_mat = np.column_stack([node.state.z for node in nodes])
    u_mat = np.column_stack([node.state.u for node in nodes])

    rho = float(instance.rho)
    alpha = float(instance.alpha)
    lambda_ = float(instance.lambda_)
    num_sites = len(nodes)

    # This matches: x_hat = alpha * x_mat + (1.0 - alpha) * z_mat
    x_hat_mat = alpha * x_mat + (1.0 - alpha) * z_mat

    x_hat_mean = np.mean(x_hat_mat, axis=1)
    u_mean = np.mean(u_mat, axis=1)
    z_start = np.mean(z_mat, axis=1)

    # res = minimize(
    #     fun=_z_objective,
    #     x0=z_start,
    #     args=(x_hat_mean, u_mean, rho, lambda_, num_sites),
    #     jac=True,
    #     method="BFGS",
    #     options={"disp": False}
    # )

    # z_new = res.x

    target = x_hat_mean + u_mean
    kappa = lambda_ / (num_sites * rho)

    z_new = np.zeros_like(target)
    z_new[0] = target[0] 
    z_new[1:] = np.sign(target[1:]) * np.maximum(np.abs(target[1:]) - kappa, 0)


    # z_new = x_hat_mean + u_mean
    z_reg = z_new.copy()
    z_reg[0] = 0.0  
    reg_obj = lambda_ * np.linalg.norm(z_reg, 1)

    # 5. ----- Update Node States (Broadcast & Dual Update) -----
    for i, node in enumerate(nodes):
        node.state.z_old = node.state.z.copy()
        
        node.state.z = z_new.copy()
        node.state.u = node.state.u + x_hat_mat[:, i] - z_new
        node.state.reg_obj = float(reg_obj)

    # Update instance-level tracking if necessary
    instance.current_reg_obj = reg_obj

def check_convergence(nodes: List[LocalNode], instance: InstanceState, logging: bool = True) -> Tuple[float, float, float, float]:
    """
    Computes primal/dual residuals and tolerances.
    Exactly mirrors the _check_convergence logic using tiled global vectors.
    """
    # 1. Gather local states into matrices
    x_mat = np.column_stack([n.state.x for n in nodes])
    u_mat = np.column_stack([n.state.u for n in nodes])
    
    # In the reference, z and z_old are treated as vectors then tiled
    # We take the z from the first node (consensus) or the instance
    z_vec = nodes[0].state.z 
    z_old_vec = nodes[0].state.z_old

    rho = float(instance.rho)
    abs_tol = float(instance.abs_tol)
    rel_tol = float(instance.rel_tol)
    
    num_features, num_sites = x_mat.shape

    # 2. Tile the global vectors to match the x_mat shape (num_features, num_sites)
    z_mat = np.tile(z_vec.reshape(-1, 1), (1, num_sites))
    z_old_mat = np.tile(z_old_vec.reshape(-1, 1), (1, num_sites))

    # 3. Compute Residuals
    r_norm = np.linalg.norm(x_mat - z_mat, ord="fro")
    s_norm = np.linalg.norm(rho * (z_mat - z_old_mat), ord="fro")

    # 4. Compute Tolerances
    eps_abs = np.sqrt(num_sites * num_features) * abs_tol
    
    # Primal Tolerance
    eps_rel = rel_tol * max(
        np.linalg.norm(x_mat, "fro"),
        np.linalg.norm(z_mat, "fro"),
    )
    eps_pri = eps_abs + eps_rel

    # Dual Tolerance
    eps_dual_base = np.sqrt(num_sites * num_features) * abs_tol
    eps_dual_rel = rel_tol * np.linalg.norm(rho * u_mat, "fro")
    eps_dual = eps_dual_base + eps_dual_rel

    # 5. Exact Logging Format
    if logging:
        print(
            "Convergence info: "
            f"r_norm - eps_pri = {r_norm - eps_pri:.4e}, "
            f"s_norm - eps_dual = {s_norm - eps_dual:.4e}"
        )

    return r_norm, s_norm, eps_pri, eps_dual


# ==============================================================================
# 4. Training / Evaluation Metrics 
# ==============================================================================

def record_admm_metrics(nodes: List[LocalNode], instance: InstanceState, history: dict):
    """
    Record SSE, 'RMSE' , and coefficient trajectory.
    Mirrors the MATLAB recordAdmmVariables.m logic.
    """

    # per-site metrics
    site_obj = np.array([node.state.obj for node in nodes])
    site_reg_obj = np.array([node.state.reg_obj for node in nodes])
    site_sse = np.array([node.state.sum_square_error for node in nodes])

    # training objective components
    obj_loss = float(np.sum(site_obj))
    obj_reg = float(site_reg_obj[0])   # global regularization (same across sites)
    obj_total = obj_loss + obj_reg

    history.setdefault("obj_loss_log", []).append(obj_loss)
    history.setdefault("obj_reg_log", []).append(obj_reg)
    history.setdefault("obj_log", []).append(obj_total)

    # SSE and "RMSE" (MATLAB style: no sqrt in training log)
    sum_square_error = float(np.sum(site_sse))
    history.setdefault("sum_square_error_log", []).append(sum_square_error)

    total_patients = sum(instance.patient_counts)
    root_mean_square_error = float(np.sqrt(sum_square_error / total_patients))
    history.setdefault("root_mean_square_error_log", []).append(root_mean_square_error)

    # Coefficient trajectory (z is global)
    z_current = nodes[0].state.z.copy()
    history.setdefault("z_log", []).append(z_current)

    # rho trajectory
    history.setdefault("rho_log", []).append(instance.rho)


def evaluate_final_model(nodes: List[LocalNode]) -> Tuple[List[dict], dict, dict]:
    """
    Compute ROC/AUC per sitpyenv e, global ROC/AUC, and calibration metrics on
    all patients (train + val).
    """

    all_y = []
    all_probs = []
    roc_site = []

    for i, node in enumerate(nodes):
        z_global = node.state.z

        X_all = np.vstack([node.data.X_train, node.data.X_val])
        y_all = np.concatenate([node.data.y_train, node.data.y_val])

        probs_all = predict_proba(X_all, z_global)
        fpr, tpr, _ = roc_curve(y_all, probs_all)
        auc_site = roc_auc_score(y_all, probs_all)

        roc_site.append({
            "site": i,
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc_site
        })

        all_y.append(y_all)
        all_probs.append(probs_all)

    all_y = np.concatenate(all_y)
    all_probs = np.concatenate(all_probs)

    # Global ROC / AUC (micro-averaged)
    fpr_g, tpr_g, _ = roc_curve(all_y, all_probs)
    auc_g = roc_auc_score(all_y, all_probs)
    roc_global = {"fpr": fpr_g, "tpr": tpr_g, "auc": auc_g}

    # --- Calibration metrics (calibration-in-the-large, slope, quantiles) ---
    eps = 1e-15
    # Avoid division by zero
    probs_clipped = np.clip(all_probs, eps, 1.0 - eps)
    lp = np.log(probs_clipped / (1.0 - probs_clipped))  # linear predictor

    # Logistic recalibration: logit(p) = a + b * LP
    lr = LogisticRegression(fit_intercept=True, solver="lbfgs")
    lr.fit(lp.reshape(-1, 1), all_y)
    calib_intercept = float(lr.intercept_[0])  # calibration-in-the-large
    calib_slope = float(lr.coef_[0, 0])        # calibration slope

    # Quantile-based calibration curve
    n_quantiles = 10
    quantile_edges = np.quantile(all_probs, np.linspace(0, 1, n_quantiles + 1))

    mean_pred = []
    mean_obs = []
    bin_centers = []

    for q in range(n_quantiles):
        lo, hi = quantile_edges[q], quantile_edges[q + 1]
        if q == n_quantiles - 1:
            mask = (all_probs >= lo) & (all_probs <= hi)
        else:
            mask = (all_probs >= lo) & (all_probs < hi)

        if not np.any(mask):
            continue

        probs_q = all_probs[mask]
        y_q = all_y[mask]

        mean_pred.append(float(np.mean(probs_q)))
        mean_obs.append(float(np.mean(y_q)))
        bin_centers.append(float(np.mean([lo, hi])))

    calibration = {
        "intercept": calib_intercept,
        "slope": calib_slope,
        "bin_centers": np.array(bin_centers),
        "mean_pred": np.array(mean_pred),
        "mean_obs": np.array(mean_obs),
    }

    return roc_site, roc_global, calibration


# ==============================================================================
# 5. Visualization
# ==============================================================================

def plot_admm_convergence(history: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Residuals (Convergence) ---
    ax1.semilogy(history["round"], history["r_norm"], label='Primal Residual ($r^k$)', color='blue', lw=2)
    ax1.semilogy(history["round"], history["eps_pri"], label='Primal Tolerance', color='blue', linestyle='--', alpha=0.6)

    ax1.semilogy(history["round"], history["s_norm"], label='Dual Residual ($s^k$)', color='red', lw=2)
    ax1.semilogy(history["round"], history["eps_dual"], label='Dual Tolerance', color='red', linestyle='--', alpha=0.6)

    ax1.set_title("ADMM Residuals (Log Scale)", fontsize=14)
    ax1.set_xlabel("Iteration Round")
    ax1.set_ylabel("Norm")
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.legend()

    # --- Plot 2: Model Performance (Per Node, Accuracy) ---
    val_acc_matrix = np.array(history["val_acc_per_node"])  # shape = (rounds, num_nodes)

    for node_idx in range(val_acc_matrix.shape[1]):
        ax2.plot(
            history["round"],
            val_acc_matrix[:, node_idx],
            marker='o',
            label=f'Node {node_idx}'
        )

    ax2.set_title("Global Model Accuracy per Node", fontsize=14)
    ax2.set_xlabel("Iteration Round")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_training_progress(history: dict, coefficient_labels: Optional[List[str]] = None):
    """
    Dual-axis training progress:
      - left axis: training 'RMSE' and validation RMSE
      - right axis: coefficient trajectories (subset or all)
    """

    rmse_log = np.array(history.get("root_mean_square_error_log", []))
    z_log = np.array(history.get("z_log", []))  # shape: (iterations, num_coeffs)

    if rmse_log.size == 0 or z_log.size == 0:
        print("Training progress logs are empty; skipping progress plot.")
        return

    num_iter, num_coeffs = z_log.shape
    iters = np.arange(1, num_iter + 1)

    # Validation RMSE per round (true RMSE); align to training length
    val_rmse = np.array(history.get("val_rmse", []))
    if val_rmse.size >= iters.size:
        val_rmse_plot = val_rmse[: iters.size]
    else:
        pad = np.full(iters.size - val_rmse.size, np.nan)
        val_rmse_plot = np.concatenate([val_rmse, pad])

    if coefficient_labels is None or len(coefficient_labels) != num_coeffs:
        coefficient_labels = [f"coef_{j}" for j in range(num_coeffs)]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left axis: training and validation error
    color_train = "tab:blue"
    color_val = "tab:green"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Error", color=color_train)
    tr_line, = ax1.plot(
        iters,
        rmse_log,
        marker="*",
        linestyle="-",
        color=color_train,
        label="Training 'RMSE' (SSE / N)",
    )
    val_line, = ax1.plot(
        iters,
        val_rmse_plot,
        marker="o",
        linestyle="--",
        color=color_val,
        label="Validation RMSE",
    )
    ax1.tick_params(axis="y", labelcolor=color_train)
    ax1.grid(alpha=0.3)

    # Right axis: Coefficients
    ax2 = ax1.twinx()
    ax2.set_ylabel("Coefficient value", color="tab:red")
    coef_lines = []
    for j in range(num_coeffs):
        line, = ax2.plot(iters, z_log[:, j], label=coefficient_labels[j], alpha=0.4)
        coef_lines.append(line)
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Combined legend (training+validation error + coefficients)
    lines = [tr_line, val_line] + coef_lines
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc="upper right")

    plt.title("Training & Validation Error and Coefficients")
    fig.tight_layout()
    plt.show()


def plot_roc_curves(roc_site: List[dict], roc_global: dict):
    plt.figure(figsize=(8, 6))

    # Per-site ROC
    for rs in roc_site:
        plt.plot(rs["fpr"], rs["tpr"], alpha=0.6, label=f"Site {rs['site']} (AUC={rs['auc']:.3f})")

    # Global ROC
    plt.plot(
        roc_global["fpr"],
        roc_global["tpr"],
        color="black",
        lw=2.5,
        label=f"Global (AUC={roc_global['auc']:.3f})"
    )

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Per Site and Global)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_calibration_curve(calibration: dict):
    plt.figure(figsize=(7, 6))

    x = calibration["mean_pred"]
    y = calibration["mean_obs"]

    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(x, y, "o-", label="Quantile means")

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(
        f"Calibration Plot\n"
        f"Intercept={calibration['intercept']:.3f}, Slope={calibration['slope']:.3f}"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 6. Main Simulation Loop
# ==============================================================================

def run_local_simulation(
        csv_paths: List[str], 
        num_rounds = 300, 
        rho = 0.25, 
        alpha = 1.6, 
        lambda_ = 0, 
        abs_tol = 1e-4, 
        rel_tol  = 1e-4
        ):
    print(f"--- Loading data from {len(csv_paths)} files ---")

    history = {
        "round": [],
        "r_norm": [],
        "s_norm": [],
        "eps_pri": [],
        "eps_dual": [],
        "val_acc_mean": [],
        "val_acc_per_node": [],
        "val_rmse": [],            # validation RMSE per round (global)
        "roc_site": [],            # list of per-site ROC info at final stage
        "roc_global": None,        # global ROC at final stage
        "calibration": None        # global calibration info at final stage
    }

    # 1. Load Data
    local_data_list: List[LocalData] = []
    for path in csv_paths:
        try:
            ld, feature_names = load_local_patient_data(path)
            local_data_list.append(ld)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return

    if not local_data_list:
        print("No data loaded; exiting.")
        return

    # 2. Correctly derive dimensions and counts from the data itself
    # This matches: patient_count = int(X_train.shape[0])
    patient_counts = [int(ld.X_train.shape[0]) for ld in local_data_list]
    
    num_features = local_data_list[0].X_train.shape[1] + 1

    # 2. Initialize
    instance = InstanceState(
        rho=rho,
        alpha=alpha,
        lambda_=lambda_,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        patient_counts=patient_counts,
        x_init=np.zeros(num_features),
        u_init=np.zeros(num_features),
        z_init=np.ones(num_features)
    )
    # instance.z_init[0] = 0  # intercept is 0

    nodes: List[LocalNode] = []
    for i, data in enumerate(local_data_list):
        state = SiteState(
            x=instance.x_init.copy(),
            u=instance.u_init.copy(),
            z=instance.z_init.copy(),
            patient_count=data.X_train.shape[0]
        )
        nodes.append(LocalNode(i, data, state))

    # 3. Loop
    print(f"\n--- Starting ADMM ({num_rounds} rounds) ---")

    for r in range(1, num_rounds + 1):
        # A. Local X Updates
        for node in nodes:
            admm_x_update(node, instance)

        # B. Global Aggregation
        admm_z_u_update(nodes, instance)

        # C. Check Convergence
        r_norm, s_norm, eps_pri, eps_dual = check_convergence(nodes, instance)

        # D. Validation Metrics (Global Model on each Node Val Set)
        val_acc_per_node = []
        val_sse_per_node = []

        for i, node in enumerate(nodes):
            z_global = node.state.z  # same global z on all nodes
            val_probs = predict_proba(node.data.X_val, z_global)
            val_preds = (val_probs >= 0.5).astype(int)

            val_acc_i = np.mean(val_preds == node.data.y_val)
            val_acc_per_node.append(val_acc_i)

            # Validation SSE (for RMSE, like stageEvaluation.m)
            val_residuals = node.data.y_val - val_probs
            val_sse_i = float(np.sum(val_residuals ** 2))
            val_sse_per_node.append(val_sse_i)

            print(f"Node {i:02d} | Val Accuracy: {val_acc_i:.4f}")

        # --- Global validation RMSE   ---
        total_val_patients = sum(len(node.data.y_val) for node in nodes)
        total_val_sse = float(np.sum(val_sse_per_node))
        val_rmse_global = float(np.sqrt(total_val_sse / total_val_patients))

        history["round"].append(r)
        history["r_norm"].append(r_norm)
        history["s_norm"].append(s_norm)
        history["eps_pri"].append(eps_pri)
        history["eps_dual"].append(eps_dual)
        history["val_acc_per_node"].append(val_acc_per_node)
        history["val_rmse"].append(val_rmse_global)

        mean_val_acc = float(np.mean(val_acc_per_node))
        history["val_acc_mean"].append(mean_val_acc)

        print(
            f"Round {r:03d}: mean acc={mean_val_acc:.4f}, "
            f"val RMSE={val_rmse_global:.4f} | "
            f"r_norm={r_norm:.4f} (eps={eps_pri:.4f}) | "
            f"s_norm={s_norm:.4f} (eps={eps_dual:.4f})"
        )

        # --- Training progress metrics (SSE, 'RMSE', z-log) ---
        record_admm_metrics(nodes, instance, history)

        if r_norm < eps_pri and s_norm < eps_dual and r > 0:
            print(f"*** Converged at round {r} ***")
            break

    # After the loop finishes: convergence plots and training-progress plots
    plot_admm_convergence(history)
    # Example coefficient labels (Intercept + example TNM/stage style labels)
    coefficient_labels = ['Intercept'] + [f'coef_{j}' for j in range(1, nodes[0].state.z.shape[0])]
    plot_training_progress(history, coefficient_labels=coefficient_labels)

    # Final ROC/AUC and calibration (all patients, train + val)
    roc_site, roc_global, calibration = evaluate_final_model(nodes)
    history["roc_site"] = roc_site
    history["roc_global"] = roc_global
    history["calibration"] = calibration

    plot_roc_curves(roc_site, roc_global)
    plot_calibration_curve(calibration)

    print("\nFinal Global Coefficients (z):")
    original_style_params = nodes[0].state.z
    print("\n" + "="*60)
    print(f"{'Index':<8} {'Variable Name':<40} {'Value':<12}")
    print("-" * 60)
    print(f"{0:<8} {'(Intercept)':<40} {original_style_params[0]:.6f}")
    for i, name in enumerate(feature_names):
        print(f"{i+1:<8} {name:<40} {original_style_params[i+1]:.6f}")
    print("="*60)

    print("\nVector representation (z_arr):")
    print(original_style_params.tolist())
    # print(nodes[0].state.z)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    my_csv_paths = [
        "20kLogRegChallenge/test/fakebeach_merged_0.csv",
        "20kLogRegChallenge/test/fakebeach_merged_1.csv",
        "20kLogRegChallenge/test/fakebeach_merged_2.csv"
    ]

    # Run
    # run_local_simulation(my_csv_paths, num_rounds=100)
    run_local_simulation(
        my_csv_paths, 
        num_rounds = 400, 
        rho = 0.25, 
        alpha = 1, 
        lambda_ = 0, 
        abs_tol = 0.001, 
        rel_tol  = 0.001
        )