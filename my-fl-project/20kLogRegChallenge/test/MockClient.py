"""
Local test of the ADMM logistic regression algorithm using MockAlgorithmClient.

Run as:

    python test/MockClient.py

from the 20kLogRegChallenge folder, or from anywhere if the package is installed (pip install -e .).
"""
import sys
from pathlib import Path

# Ensure the algorithm package is importable when running this script directly
# (MockAlgorithmClient uses import_module("logreg_challenge_20k")).
_algorithm_root = Path(__file__).resolve().parent.parent
if str(_algorithm_root) not in sys.path:
    sys.path.insert(0, str(_algorithm_root))

from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

# current directory = .../20kLogRegChallenge/test
current_path = Path(__file__).parent


# datasets= [

#         [  # Org 0
#         {
#             "database": current_path / "fakebeach_merged_FULL.csv",
#             "db_type": "csv",
#             "input_data": {},
#         }
#     ],
# ]
# Point each "organization" to its own CSV. orignal is node0.csv, node1.csv, node2.csv
datasets = [
    [  # Org 0
        {
            "database": current_path / "fakebeach_merged_0.csv",
            "db_type": "csv",
            "input_data": {},
        }
    ],
    [  # Org 1
        {
            "database": current_path / "fakebeach_merged_1.csv",
            "db_type": "csv",
            "input_data": {},
        }
    ],
    [  # Org 2
        {
            "database": current_path / "fakebeach_merged_2.csv",
            "db_type": "csv",
            "input_data": {},
        }
    ],
]

client = MockAlgorithmClient(
    datasets=datasets,
    module="logreg_challenge_20k",
)

# list mock organizations
organizations = client.organization.list()
print("Organizations:", organizations)
org_ids = [organization["id"] for organization in organizations]

# Run central ADMM function on one "central" org (e.g. org 0)
central_task = client.task.create(
    input_={
        "method": "central_function",
        "kwargs": {
            "num_rounds": 400,
            "rho": 0.25,
            "alpha": 1,
            "lambda_": 0,
            "abs_tol": 0.001,
            "rel_tol": 0.001,
            "logging": False
        },
    },
    organizations=[org_ids[0]],
)

results = client.wait_for_results(central_task.get("id"))
res = results[0]

print("Central ADMM result (summary):")
print("  Final coefficients (z):", res.get("coefficients"))
print("  Patient counts per site:", res.get("patient_counts"))

history = res.get("history", {})
# print("  Rounds:", history.get("round"))
print("  Mean validation accuracy per round:", history.get("val_acc_mean"))
print("  Validation RMSE per round:", history.get("val_rmse"))

roc_site = res.get("roc_site", [])
roc_global = res.get("roc_global", {})
calibration = res.get("calibration", {})

print("  Global AUC:", roc_global.get("auc"))
print(
    "  Calibration intercept, slope:",
    calibration.get("intercept"),
    calibration.get("slope"),
)

# ----------------------------------------------------------------------
# Plotting (for local testing with matplotlib & numpy)
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

rounds = np.array(history["round"])

# Convergence plot: residuals vs tolerances
plt.figure(figsize=(10, 5))
plt.semilogy(rounds, history["r_norm"], label="Primal residual r^k", color="blue")
plt.semilogy(
    rounds,
    history["eps_pri"],
    "--",
    label="Primal tolerance",
    color="blue",
    alpha=0.6,
)
plt.semilogy(rounds, history["s_norm"], label="Dual residual s^k", color="red")
plt.semilogy(
    rounds,
    history["eps_dual"],
    "--",
    label="Dual tolerance",
    color="red",
    alpha=0.6,
)
plt.xlabel("Iteration")
plt.ylabel("Norm")
plt.title("ADMM residuals")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.tight_layout()

# Accuracy per node
val_acc_per_node = history["val_acc_per_node"]  # list[round][node]
val_acc_matrix = np.array(val_acc_per_node)     # shape (R, num_nodes)

plt.figure(figsize=(10, 5))
for node_idx in range(val_acc_matrix.shape[1]):
    plt.plot(rounds, val_acc_matrix[:, node_idx], marker="o", label=f"Node {node_idx}")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.title("Global model accuracy per node")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Training 'RMSE' (sumSquareError / N), validation RMSE, and coefficient trajectories
if "root_mean_square_error_log" in history and "z_log" in history:
    rmse_log = np.array(history["root_mean_square_error_log"])
    z_log = np.array(history["z_log"])  # shape: (iterations, num_coeffs)
    iters = np.arange(1, len(rmse_log) + 1)

    # Validation RMSE is stored per round; align to same length if needed
    val_rmse = np.array(history.get("val_rmse", []))
    if val_rmse.size >= iters.size:
        val_rmse_plot = val_rmse[: iters.size]
    else:
        # pad with NaNs if shorter, so plot ignores missing points
        pad = np.full(iters.size - val_rmse.size, np.nan)
        val_rmse_plot = np.concatenate([val_rmse, pad])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_train = "tab:blue"
    color_val = "tab:green"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Error", color=color_train)
    ax1.plot(
        iters,
        rmse_log,
        marker="*",
        linestyle="-",
        color=color_train,
        label="Training 'RMSE' (SSE / N)",
    )
    ax1.plot(
        iters,
        val_rmse_plot,
        marker="o",
        linestyle="--",
        color=color_val,
        label="Validation RMSE",
    )
    ax1.tick_params(axis="y", labelcolor=color_train)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Coefficient value", color="tab:red")
    for j in range(z_log.shape[1]):
        ax2.plot(iters, z_log[:, j], alpha=0.4)
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.title("Training & Validation Error and Coefficients")
    fig.tight_layout()

# ROC curves (per-site and global)
if roc_site or roc_global:
    plt.figure(figsize=(8, 6))

    # Per-site ROC curves
    for rs in roc_site:
        fpr = np.array(rs.get("fpr", []))
        tpr = np.array(rs.get("tpr", []))
        auc_val = rs.get("auc", None)
        if fpr.size == 0 or tpr.size == 0:
            continue
        label = f"Site {rs.get('site', '?')}"
        if auc_val is not None:
            label += f" (AUC={auc_val:.3f})"
        plt.plot(fpr, tpr, alpha=0.6, label=label)

    # Global ROC curve
    fpr_g = np.array(roc_global.get("fpr", []))
    tpr_g = np.array(roc_global.get("tpr", []))
    auc_g = roc_global.get("auc", None)
    if fpr_g.size > 0 and tpr_g.size > 0:
        label_g = "Global"
        if auc_g is not None:
            label_g += f" (AUC={auc_g:.3f})"
        plt.plot(fpr_g, tpr_g, color="black", lw=2.0, label=label_g)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Per Site and Global)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

# Calibration plot (quantile-based)
mean_pred = np.array(calibration.get("mean_pred", []))
mean_obs = np.array(calibration.get("mean_obs", []))
intercept = calibration.get("intercept", None)
slope = calibration.get("slope", None)

if mean_pred.size > 0 and mean_obs.size > 0:
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(mean_pred, mean_obs, "o-", label="Quantile means")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")

    title = "Calibration Plot"
    if intercept is not None and slope is not None:
        title += f"\nIntercept={intercept:.3f}, Slope={slope:.3f}"
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

plt.show()
