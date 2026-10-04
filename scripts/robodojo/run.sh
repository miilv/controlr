#!/usr/bin/env bash
# One RoboDojo evaluation with controlr as the controller (run ON the RoboDojo host, compute2).
#
#   scripts/robodojo/run.sh --task general_pickup --seed 0 --eval-num 5 [-c configs/robodojo.yaml]
#                           [--fake-llm] [--set key=value ...]
#
# Starts three processes and stops all of them on exit:
#   1. controlr robodojo-serve  (our venv)  — the controller, one run dir per episode
#   2. XPolicyLab's ws policy server with the no-op `controlr` model (RoboDojo's handshake)
#   3. RoboDojo's eval client (`robodojo.sh client`, Isaac Sim 5.1) — owns the episodes and
#      loads XPolicyLab/policy/controlr/deploy.py (copied from controlr/robot/robodojo/shim)
# RoboDojo's own results (_result.json, mp4s) are copied to runs/robodojo_eval/<stamp>/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"         # controlr checkout
RD_HOME="${RD_HOME:-/root/controlr-robodojo}"
RD="$RD_HOME/RoboDojo"
CONDA="$RD_HOME/miniconda3"
TASK="" SEED=0 NUM=1 CFG="configs/robodojo.yaml" FAKE="" SETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --eval-num) NUM="$2"; shift 2 ;;
    -c|--config) CFG="$2"; shift 2 ;;
    --fake-llm) FAKE="--fake-llm"; shift ;;
    --set) SETS+=(--set "$2"); shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TASK" ]] || { echo "--task is required" >&2; exit 2; }
cd "$HERE"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGS="$HERE/runs/robodojo_eval/${STAMP}_${TASK}_s${SEED}"
mkdir -p "$LOGS"

# 1. the shim, as XPolicyLab policy "controlr"
POL="$RD/XPolicyLab/policy/controlr"
mkdir -p "$POL"
cp controlr/robot/robodojo/shim/{__init__.py,deploy.py,model.py,deploy.yml} controlr/robot/robodojo/protocol.py "$POL/"

free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
export CONTROLR_ROBODOJO_HOST=127.0.0.1
export CONTROLR_ROBODOJO_PORT="$(free_port)"
export CONTROLR_ROBODOJO_AUTHKEY="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
export CONTROLR_ROBODOJO_READY_FILE="$LOGS/serve.ready"
WS_PORT="$(free_port)"

PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -TERM -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null || true; done
  # RoboDojo's results for this run
  if compgen -G "$RD/eval_result/RoboDojo/$TASK/controlr/*" >/dev/null; then
    newest="$(ls -td "$RD"/eval_result/RoboDojo/"$TASK"/controlr/*/*/* 2>/dev/null | head -1 || true)"
    [[ -n "$newest" ]] && cp -r "$newest" "$LOGS/robodojo_result" && echo "RoboDojo result -> $LOGS/robodojo_result"
  fi
}
trap cleanup EXIT

# 2. controlr
setsid "$HERE/.venv/bin/controlr" robodojo-serve -c "$CFG" $FAKE "${SETS[@]}" > "$LOGS/serve.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 60); do [[ -f "$CONTROLR_ROBODOJO_READY_FILE" ]] && break; sleep 1; done
[[ -f "$CONTROLR_ROBODOJO_READY_FILE" ]] || { echo "controlr robodojo-serve did not start:"; cat "$LOGS/serve.log"; exit 1; }

# 3. XPolicyLab ws policy server (no-op model) in its own env (websockets>=14; install.sh policy_env)
( cd "$RD/XPolicyLab" && PYTHONPATH="$RD" setsid "$RD_HOME/policy_venv/bin/python" setup_policy_server.py \
    --config_path "$POL/deploy.yml" --overrides port="$WS_PORT" host=127.0.0.1 policy_name=controlr \
    > "$LOGS/policy_server.log" 2>&1 ) &
PIDS+=($!)
for _ in $(seq 60); do (echo > /dev/tcp/127.0.0.1/"$WS_PORT") 2>/dev/null && break; sleep 1; done
(echo > /dev/tcp/127.0.0.1/"$WS_PORT") 2>/dev/null || { echo "policy server did not start:"; tail -20 "$LOGS/policy_server.log"; exit 1; }

source "$CONDA/bin/activate" "$CONDA/envs/robodojo"
export OMNI_KIT_ACCEPT_EULA=YES

# 4. RoboDojo's eval client: the episodes
echo "RoboDojo $TASK seed=$SEED eval-num=$NUM -> logs $LOGS"
set +e
( cd "$RD" && bash scripts/robodojo.sh client --task "$TASK" --policy-name controlr \
    --policy-host 127.0.0.1 --policy-port "$WS_PORT" --seed "$SEED" --eval-num "$NUM" \
    --ckpt controlr --action-type joint ) 2>&1 | tee "$LOGS/eval_client.log"
rc=${PIPESTATUS[0]}
set -e
tail -3 "$LOGS/serve.log" || true
exit "$rc"
