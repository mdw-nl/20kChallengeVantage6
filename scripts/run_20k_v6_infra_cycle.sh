#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_20k_v6_infra_cycle.sh [options]

Options:
  --workdir <path>        Clone/build workspace for dependency repos (default: /tmp/20k-v6-work)
  --venv-dir <path>       Python venv used for generation + task submission (default: /tmp/20k-v6-venv)
  --python-bin <path>     Python 3.13 executable (default: python3.13; fallback pyenv/python3)
  --v6-username <name>    v6 account used to submit federated task (default: gamma-user)
  --v6-password <pass>    v6 password used to submit federated task (default: gamma-password)
  --node-count <3-6>      Number of nodes for infra run (default: 4)
  --num-subjects <int>    Synthetic cohort size (>=500, default: 2000)
  --num-rounds <int>      ADMM rounds for federated run (default: 25)
  --registry-port <int>   Local Docker registry port (default: 5001)
  --enable-ui             Start the v6 UI container (default: disabled)
  --ui-port <int>         UI port when enabled (default: 80)
  --keep-registry         Keep local registry container after script exits
  --no-prune-docker       Skip Docker cleanup (cache/images/containers) at end
  -h, --help              Show this help
USAGE
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "[error] required command not found: $cmd" >&2
    exit 1
  }
}

WORKDIR="${WORKDIR:-/tmp/20k-v6-work}"
VENV_DIR="${VENV_DIR:-/tmp/20k-v6-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.13}"
V6_USERNAME="${V6_USERNAME:-gamma-user}"
V6_PASSWORD="${V6_PASSWORD:-gamma-password}"
NODE_COUNT="${NODE_COUNT:-4}"
NUM_SUBJECTS="${NUM_SUBJECTS:-2000}"
NUM_ROUNDS="${NUM_ROUNDS:-25}"
REGISTRY_PORT="${REGISTRY_PORT:-5001}"
KEEP_REGISTRY="false"
UI_ENABLED="${UI_ENABLED:-false}"
UI_PORT="${UI_PORT:-80}"
PRUNE_DOCKER="false"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-harbor2.vantage6.ai/infrastructure}"
TASK_TIMEOUT_S="${TASK_TIMEOUT_S:-1800}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --workdir)
      WORKDIR="$2"; shift 2 ;;
    --venv-dir)
      VENV_DIR="$2"; shift 2 ;;
    --python-bin)
      PYTHON_BIN="$2"; shift 2 ;;
    --v6-username)
      V6_USERNAME="$2"; shift 2 ;;
    --v6-password)
      V6_PASSWORD="$2"; shift 2 ;;
    --node-count)
      NODE_COUNT="$2"; shift 2 ;;
    --num-subjects)
      NUM_SUBJECTS="$2"; shift 2 ;;
    --num-rounds)
      NUM_ROUNDS="$2"; shift 2 ;;
    --registry-port)
      REGISTRY_PORT="$2"; shift 2 ;;
    --enable-ui)
      UI_ENABLED="true"; shift 1 ;;
    --ui-port)
      UI_PORT="$2"; shift 2 ;;
    --keep-registry)
      KEEP_REGISTRY="true"; shift 1 ;;
    --no-prune-docker)
      PRUNE_DOCKER="false"; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[error] unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

if ! [[ "$NODE_COUNT" =~ ^[0-9]+$ ]] || (( NODE_COUNT < 3 || NODE_COUNT > 6 )); then
  echo "[error] --node-count must be an integer between 3 and 6" >&2
  exit 1
fi

if ! [[ "$NUM_SUBJECTS" =~ ^[0-9]+$ ]] || (( NUM_SUBJECTS < 500 )); then
  echo "[error] --num-subjects must be an integer >= 500" >&2
  exit 1
fi

if ! [[ "$NUM_ROUNDS" =~ ^[0-9]+$ ]] || (( NUM_ROUNDS < 1 )); then
  echo "[error] --num-rounds must be a positive integer" >&2
  exit 1
fi

if ! [[ "$UI_PORT" =~ ^[0-9]+$ ]] || (( UI_PORT < 1 || UI_PORT > 65535 )); then
  echo "[error] --ui-port must be an integer between 1 and 65535" >&2
  exit 1
fi

require_cmd git
require_cmd docker

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHALLENGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if [[ -x "$HOME/.pyenv/shims/python3.13" ]]; then
    PYTHON_BIN="$HOME/.pyenv/shims/python3.13"
  elif [[ -x "$HOME/.pyenv/versions/3.13.0/bin/python" ]]; then
    PYTHON_BIN="$HOME/.pyenv/versions/3.13.0/bin/python"
  elif [[ -x "$CHALLENGE_DIR/../.venv/bin/python" ]]; then
    PYTHON_BIN="$CHALLENGE_DIR/../.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] Could not find requested Python executable: $PYTHON_BIN" >&2
  exit 1
fi

PY_VERSION="$($PYTHON_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION##*.}"
if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 13) )); then
  echo "[error] Python >= 3.13 is required (found $PY_VERSION)." >&2
  echo "        datavalgen-model-beach currently requires Python 3.13+." >&2
  exit 1
fi

INFRA_DIR="$WORKDIR/v6-infrastructure-sh/infrastructure"
DATAVALGEN_DIR="$WORKDIR/datavalgen"
DATAVALGEN_BEACH_DIR="$WORKDIR/datavalgen-model-beach"

NODES_ENV="$WORKDIR/20kchallenge_nodes_${NODE_COUNT}.env"
COLLAB_NAME="challenge-20k-synth-${NODE_COUNT}n"
ALGO_DIR="$CHALLENGE_DIR/my-fl-project/20kLogRegChallenge"
DATA_DIR="$CHALLENGE_DIR/generated_data/consortium_signal_v1"
RESULTS_JSON="$DATA_DIR/federated_vs_centralized_${NODE_COUNT}nodes.json"
REPORT_JSON="$DATA_DIR/generation_report.json"

REGISTRY_CONTAINER_NAME="v6-local-registry-${REGISTRY_PORT}"
REGISTRY_STARTED_BY_SCRIPT="false"
INFRA_STARTED="false"

run_infra() {
  (
    cd "$INFRA_DIR"
    ENVIRONMENT=CI \
    UI_ENABLED="$UI_ENABLED" \
    UI_PORT="$UI_PORT" \
    UI_URL="http://localhost:${UI_PORT}" \
    NODES_CONFIG="$NODES_ENV" \
    COLLABORATION_NAME="$COLLAB_NAME" \
    PYTHON_INTERPRETER="$VENV_DIR/bin/python" \
    VENV_PATH="$WORKDIR/.v6-infra-venv" \
    SERVER_URL="http://host.docker.internal" \
    DOCKER_REGISTRY="$DOCKER_REGISTRY" \
    STRICT_DATA_CHECKS=true \
    ./infra.sh "$@"
  )
}

cleanup() {
  set +e

  if [[ "$INFRA_STARTED" == "true" && -d "$INFRA_DIR" ]]; then
    echo "[cleanup] shutting down infrastructure"
    run_infra down >/dev/null 2>&1 || true
  fi

  if [[ "$REGISTRY_STARTED_BY_SCRIPT" == "true" && "$KEEP_REGISTRY" != "true" ]]; then
    echo "[cleanup] removing local registry container: $REGISTRY_CONTAINER_NAME"
    docker rm -f "$REGISTRY_CONTAINER_NAME" >/dev/null 2>&1 || true
  fi

  if [[ "$PRUNE_DOCKER" == "true" ]]; then
    echo "[cleanup] pruning Docker cache + unused images/containers"
    docker container prune -f >/dev/null 2>&1 || true
    docker image prune -af >/dev/null 2>&1 || true
    docker builder prune -af >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

clone_if_missing() {
  local repo_url="$1"
  local target_dir="$2"
  if [[ -d "$target_dir/.git" ]]; then
    echo "[info] reusing existing clone: $target_dir"
  else
    echo "[info] cloning $repo_url -> $target_dir"
    git clone "$repo_url" "$target_dir"
  fi
}

echo "[step] cloning dependency repos into $WORKDIR"
mkdir -p "$WORKDIR"
clone_if_missing "https://github.com/mdw-nl/v6-infrastructure-sh.git" "$WORKDIR/v6-infrastructure-sh"
clone_if_missing "https://github.com/mdw-nl/datavalgen.git" "$DATAVALGEN_DIR"
clone_if_missing "https://github.com/MaastrichtU-CDS/datavalgen-model-beach.git" "$DATAVALGEN_BEACH_DIR"

if [[ ! -d "$ALGO_DIR" ]]; then
  echo "[error] 20k challenge algorithm directory not found: $ALGO_DIR" >&2
  exit 1
fi

echo "[step] creating reusable venv in $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install \
  -e "$DATAVALGEN_DIR" \
  -e "$DATAVALGEN_BEACH_DIR" \
  -e "$ALGO_DIR" \
  vantage6-client==4.13.3

echo "[step] generating synthetic signal dataset"
"$VENV_DIR/bin/python" "$CHALLENGE_DIR/scripts/generate_signal_synthetic_dataset.py" \
  --output-dir "$DATA_DIR" \
  --num-subjects "$NUM_SUBJECTS" \
  --node-counts "$NODE_COUNT" \
  --default-node-count "$NODE_COUNT" \
  --target-train-auc 0.90 \
  --target-val-auc 0.80

if [[ ! -f "$REPORT_JSON" ]]; then
  echo "[error] generation report not found: $REPORT_JSON" >&2
  exit 1
fi

FULL_DATA="$("$VENV_DIR/bin/python" - <<PY
import json
from pathlib import Path
p=Path('$REPORT_JSON')
report=json.loads(p.read_text())
print(report['full_dataset_csv'])
PY
)"

SPLIT_DIR="$("$VENV_DIR/bin/python" - <<PY
import json
from pathlib import Path
p=Path('$REPORT_JSON')
report=json.loads(p.read_text())
print(report['default_split_dir'])
PY
)"

if [[ ! -f "$FULL_DATA" ]]; then
  echo "[error] full dataset file not found: $FULL_DATA" >&2
  exit 1
fi

if [[ ! -d "$SPLIT_DIR" ]]; then
  echo "[error] split directory not found: $SPLIT_DIR" >&2
  exit 1
fi

echo "[step] creating nodes env: $NODES_ENV"
NODE_NAMES=(alpha beta gamma delta epsilon zeta)
NODE_KEYS=(
  844a7d92-1cc9-4856-bf33-0613252d5b3c
  57143784-19ef-456b-94c9-ba68c8cb079b
  57143784-19ef-456b-94c9-ba68c8cb079c
  57143784-19ef-456b-94c9-ba68c8cb079d
  57143784-19ef-456b-94c9-ba68c8cb079e
  57143784-19ef-456b-94c9-ba68c8cb079f
)

: > "$NODES_ENV"
for ((i=0; i<NODE_COUNT; i++)); do
  node_name="${NODE_NAMES[$i]}"
  api_key="${NODE_KEYS[$i]}"
  data_file="$SPLIT_DIR/${node_name}.csv"
  if [[ ! -f "$data_file" ]]; then
    echo "[error] expected node split not found: $data_file" >&2
    exit 1
  fi
  printf '%s|%s|%s|csv|default\n' "$node_name" "$api_key" "$data_file" >> "$NODES_ENV"
done

echo "[step] ensuring local docker registry on port $REGISTRY_PORT"
if docker ps --format '{{.Ports}}' | grep -q ":${REGISTRY_PORT}->5000/tcp"; then
  echo "[info] registry already running on port $REGISTRY_PORT"
else
  if docker ps -a --format '{{.Names}}' | grep -Fxq "$REGISTRY_CONTAINER_NAME"; then
    docker start "$REGISTRY_CONTAINER_NAME" >/dev/null
  else
    docker run -d --restart unless-stopped -p "${REGISTRY_PORT}:5000" --name "$REGISTRY_CONTAINER_NAME" registry:2 >/dev/null
    REGISTRY_STARTED_BY_SCRIPT="true"
  fi
fi

ALGO_IMAGE="localhost:${REGISTRY_PORT}/20klogregchallenge:synth-${NODE_COUNT}n"
echo "[step] building + pushing algorithm image: $ALGO_IMAGE"
docker build -t "$ALGO_IMAGE" "$ALGO_DIR"
docker push "$ALGO_IMAGE"

echo "[step] starting local vantage6 infrastructure"
run_infra down >/dev/null 2>&1 || true
run_infra preflight
run_infra up
INFRA_STARTED="true"

if [[ "$UI_ENABLED" == "true" ]]; then
  echo "[info] UI enabled at http://localhost:${UI_PORT}"
  echo "[info] Valid users are organization users from entities import:"
  echo "       alpha-user / alpha-password"
  echo "       beta-user  / beta-password"
  echo "       gamma-user / gamma-password"
  echo "[info] root/root is not created by this harness."
  if [[ "$(uname -s)" == "Linux" ]]; then
    if ! getent hosts host.docker.internal >/dev/null 2>&1; then
      echo "[warn] host.docker.internal is not resolvable on this host."
      echo "[warn] UI may fail to login from browser despite valid credentials."
      echo "[warn] Add host alias (e.g. in /etc/hosts: 127.0.0.1 host.docker.internal) or run headless."
    fi
  fi
fi

# small settle period for node startup and registration
echo "[step] waiting briefly for infra to settle"
sleep 8

echo "[step] running infrastructure smoke checks"
run_infra test

echo "[step] submitting federated task and comparing with centralized baseline"
"$VENV_DIR/bin/python" "$CHALLENGE_DIR/scripts/run_centralized_and_federated_eval.py" \
  --full-data "$FULL_DATA" \
  --algo-image "$ALGO_IMAGE" \
  --output-json "$RESULTS_JSON" \
  --server-url "http://localhost" \
  --server-port 5070 \
  --api-path "/api" \
  --username "$V6_USERNAME" \
  --password "$V6_PASSWORD" \
  --collaboration-name "$COLLAB_NAME" \
  --num-rounds "$NUM_ROUNDS" \
  --timeout-seconds "$TASK_TIMEOUT_S"

echo "[step] shutting down infrastructure"
run_infra down
INFRA_STARTED="false"

echo "[done] results saved to: $RESULTS_JSON"
"$VENV_DIR/bin/python" - <<PY
import json
from pathlib import Path
p=Path('$RESULTS_JSON')
obj=json.loads(p.read_text())
print('  centralized_train_auc=', round(obj['centralized']['train_auc'], 4))
print('  centralized_val_auc  =', round(obj['centralized']['val_auc'], 4))
print('  federated_status     =', obj['federated']['status'])
print('  federated_task_id    =', obj['federated']['task_id'])
print('  federated_roc_global =', obj['federated']['payload'].get('roc_global', {}).get('auc'))
print('  coeff_comparison     =', obj['comparison'])
PY
