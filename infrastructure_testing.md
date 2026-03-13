# Infrastructure Testing Manifest (Portable)

This manifest is machine-agnostic. Replace placeholders and run step-by-step.

Pinned infra commit:
- `v6-infrastructure-sh`: `e234435b5eb2f2ff8cb7f5e157c167913882475e`
- Reference: https://github.com/mdw-nl/v6-infrastructure-sh/commit/e234435b5eb2f2ff8cb7f5e157c167913882475e

Pinned challenge commit:
- `20kChallengeVantage6`: `3f4719051cb189e65cb7ef6e90ffff3190b4bda9`
- Reference: https://github.com/Health-AI-Consortium/20kChallengeVantage6/commit/3f4719051cb189e65cb7ef6e90ffff3190b4bda9

## 1. Fill placeholders

Set these values for your machine:

```bash
export WORKDIR="<ABS_PATH_TO_WORKDIR>"                    # e.g. $HOME/code
export PYTHON_INTERPRETER="<PYTHON_3_12_EXECUTABLE>"      # e.g. python3.12
```

## 2. Clone repos

```bash
mkdir -p "$WORKDIR"
cd "$WORKDIR"

git clone https://github.com/mdw-nl/v6-infrastructure-sh.git
cd v6-infrastructure-sh
git checkout e234435b5eb2f2ff8cb7f5e157c167913882475e
cd "$WORKDIR"

git clone https://github.com/Health-AI-Consortium/20kChallengeVantage6.git 20kChallengeVantage6
cd 20kChallengeVantage6
git checkout 3f4719051cb189e65cb7ef6e90ffff3190b4bda9
```

## 3. Define project paths

```bash
export INFRA_DIR="$WORKDIR/v6-infrastructure-sh/infrastructure"
export CHALLENGE_DIR="$WORKDIR/20kChallengeVantage6"
export ALGO_DIR="$CHALLENGE_DIR/my-fl-project/20kLogRegChallenge"
```

## 4. Start local registry + build/push algorithm image

```bash
docker run -d --restart unless-stopped -p 5000:5000 --name v6-local-registry registry:2 || true

cd "$ALGO_DIR"
docker build -t localhost:5000/20klogregchallenge:dev .
docker push localhost:5000/20klogregchallenge:dev
```

## 5. Create node config with placeholders resolved from env vars

```bash
cat >/tmp/20kchallenge_nodes.env <<EOF
alpha|844a7d92-1cc9-4856-bf33-0613252d5b3c|$ALGO_DIR/test/fakebeach_merged_0.csv|csv|default
beta|57143784-19ef-456b-94c9-ba68c8cb079b|$ALGO_DIR/test/fakebeach_merged_1.csv|csv|default
gamma|57143784-19ef-456b-94c9-ba68c8cb079c|$ALGO_DIR/test/fakebeach_merged_2.csv|csv|default
EOF
```

## 6. Clean infra state, then start infra

```bash
cd "$INFRA_DIR"

PYTHON_INTERPRETER="$PYTHON_INTERPRETER" \
ENVIRONMENT=CI UI_ENABLED=false \
NODES_CONFIG=/tmp/20kchallenge_nodes.env \
COLLABORATION_NAME=challenge-20k \
SERVER_URL=http://host.docker.internal \
./infra.sh down

rm -rf "$HOME/.local/share/vantage6/server/demoserver"
rm -rf "$HOME/.local/share/vantage6/node/alpha"
rm -rf "$HOME/.local/share/vantage6/node/beta"
rm -rf "$HOME/.local/share/vantage6/node/gamma"

PYTHON_INTERPRETER="$PYTHON_INTERPRETER" \
ENVIRONMENT=CI UI_ENABLED=false \
NODES_CONFIG=/tmp/20kchallenge_nodes.env \
COLLABORATION_NAME=challenge-20k \
SERVER_URL=http://host.docker.internal \
./infra.sh up
```

## 7. Infra smoke tests

```bash
cd "$INFRA_DIR"
UI_ENABLED=false NODES_CONFIG=/tmp/20kchallenge_nodes.env ./infra.sh test
```

Expected: container count and naming checks pass.

## 8. MockClient test

```bash
cd "$ALGO_DIR"
"$PYTHON_INTERPRETER" -m venv .venv
. .venv/bin/activate

# requirements.txt currently contains invalid 'vantage6-tools'; skip it.
python -m pip install --upgrade pip setuptools wheel
python -m pip install $(grep -v '^vantage6-tools$' requirements.txt | tr '\n' ' ')
python -m pip install -e .

MPLBACKEND=Agg python test/MockClient.py
```

Expected: ADMM converges/runs, then prints coefficients/AUC/calibration.

## 9. Real infra-backed smoke task

Run from `"$ALGO_DIR"` with `.venv` active:

```bash
python - <<'PY'
import time
from vantage6.client import Client

client = Client("http://localhost", 5070, "/api")
client.authenticate("gamma-user", "gamma-password")
client.setup_encryption(None)

collab_id = client.collaboration.list()["data"][0]["id"]
org_ids = [o["id"] for o in client.organization.list(collaboration=collab_id)["data"]]

task = client.task.create(
    collaboration=collab_id,
    organizations=[org_ids[0]],
    name="20k challenge ADMM short smoke",
    image="localhost:5000/20klogregchallenge:dev",
    description="short smoke run",
    input_={
        "method": "central_function",
        "kwargs": {
            "num_rounds": 5,
            "rho": 0.25,
            "alpha": 1,
            "lambda_": 0.0,
            "abs_tol": 1e-3,
            "rel_tol": 1e-3,
            "logging": False,
        },
    },
    databases=[{"label": "default"}],
)
task_id = task["id"]

status = None
for _ in range(180):
    status = client.task.get(task_id)["status"]
    if status in {"completed", "failed", "crashed", "cancelled"}:
        break
    time.sleep(1)

print("task_id=", task_id, "status=", status)
print(client.run.list(task=task_id))
print(client.result.from_task(task_id=task_id))
PY
```

Expected:
- parent task reaches `completed`
- run log contains ADMM rounds
- `result.from_task(task_id=...)` returns output payload

## 10. Teardown

```bash
cd "$INFRA_DIR"
PYTHON_INTERPRETER="$PYTHON_INTERPRETER" \
ENVIRONMENT=CI UI_ENABLED=false \
NODES_CONFIG=/tmp/20kchallenge_nodes.env \
COLLABORATION_NAME=challenge-20k \
SERVER_URL=http://host.docker.internal \
./infra.sh down
```

## Validation note

This sequence was executed on 2026-03-13 with clean restart and passed:
- `infra.sh test`
- `MockClient.py`
- real 5-round infra task (`completed`, result decrypted).
