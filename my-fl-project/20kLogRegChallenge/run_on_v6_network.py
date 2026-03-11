import time
from typing import Dict, List, Tuple
from vantage6.client import Client

## Demo network stuff
SERVER_URL = "http://localhost"
SERVER_PORT = 7601
SERVER_API = "/api"
USERNAME = "dev_admin"
PASSWORD = "password"


IMAGE = "surfzare/20klogregchallenge"

client = Client(SERVER_URL, SERVER_PORT, SERVER_API)
client.authenticate(USERNAME, PASSWORD)
client.setup_encryption(None)

def _list_data(resp):
    if isinstance(resp, dict) and "data" in resp:
        return resp["data"]
    return resp if isinstance(resp, list) else []

def setup_network_entities() -> Tuple[int, List[int]]:
    """
    Use the first available collaboration on the server and return its
    organizations. You do not need to know collaboration or org names:
    create any collaboration in the UI and add the organizations that
    have nodes; this script will use it.
    """
    collabs = _list_data(client.collaboration.list())
    if not collabs:
        raise RuntimeError(
            "No collaboration found on the server. In the server UI go to "
            "Administration → Collaborations → Create collaboration, set a "
            "name and add the organizations that have nodes, then run again."
        )

    collaboration = collabs[0]
    collab_id = collaboration["id"]
    print(f"Using collaboration: {collaboration['name']} (id={collab_id})")

    orgs_in_collab = client.organization.list(collaboration=collab_id)
    org_ids = [org["id"] for org in orgs_in_collab["data"]]
    if not org_ids:
        raise RuntimeError(
            f"Collaboration '{collaboration['name']}' has no organizations. "
            "In the UI, edit the collaboration and add organizations."
        )
    print(f"Organizations in collaboration: {org_ids}")

    return collab_id, org_ids

collab_id, org_ids = setup_network_entities()

input_params = {
    "method": "central_function",
    "kwargs": {
        "num_rounds": 400,
        "rho": 0.25,
        "alpha": 1,
        "lambda_": 0.0,
        "abs_tol": 1e-3,
        "rel_tol": 1e-3,
        "logging": False,

    },
}

print(f"\nSubmitting ADMM task to organization {org_ids[0]} (central)...")
task = client.task.create(
    collaboration=collab_id,
    organizations=[org_ids[0]],
    name="ADMM Logistic Regression",
    image=IMAGE,
    description="Federated ADMM logistic regression on fakebeach data",
    input_=input_params,
    databases=[{"label": "default"}],
)

task_id = task["id"]
print(f"Task created with id={task_id}")

print("Waiting for results (this may take several minutes)...")
while True:
    task_info = client.task.get(task_id)
    status = task_info.get("status")
    print(f"  Status: {status}", end="\r")

    if status in ("completed", "crashed", "failed"):
        break

    complete_count = task_info.get("complete", "?")
    print(f"  Status: {status} | complete: {complete_count}")
    time.sleep(10)

print(f"\nTask finished with status: {status}")

if status == "completed":
    results = client.result.list(task=task_id)
    for r in results["data"]:
        result_data = r.get("result")
        if result_data:
            print("\n=== ADMM Results ===")
            if isinstance(result_data, dict):
                print(f"  Coefficients: {result_data.get('coefficients')}")
                print(f"  Patient counts: {result_data.get('patient_counts')}")
                history = result_data.get("history", {})
                if history:
                    print(f"  Rounds completed: {len(history.get('round', []))}")
                    accs = history.get("val_acc_mean", [])
                    if accs:
                        print(f"  Final mean val accuracy: {accs[-1]:.4f}")
                    rmses = history.get("val_rmse", [])
                    if rmses:
                        print(f"  Final val RMSE: {rmses[-1]:.4f}")
                roc = result_data.get("roc_global", {})
                if roc:
                    print(f"  Global AUC: {roc.get('auc', 'N/A')}")
                cal = result_data.get("calibration", {})
                if cal:
                    print(f"  Calibration intercept: {cal.get('intercept', 'N/A')}")
                    print(f"  Calibration slope: {cal.get('slope', 'N/A')}")
            else:
                print(f"  Raw result: {str(result_data)[:500]}")
else:
    print("Task did not complete successfully. Check node logs for details.")
    runs = client.run.list(task=task_id)
    for run in runs.get("data", []):
        print(f"  Run {run['id']}: status={run.get('status')}")
        log = run.get("log")
        if log:
            print(f"  Log (last 500 chars): ...{str(log)[-500:]}")