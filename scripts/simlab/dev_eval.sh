#!/usr/bin/env bash
# Dev-eval: RoboDojo's own eval client (unchanged main.py) on OUR dev layouts, with any XPolicyLab policy.
# Needs a GPU where Isaac Sim 5.1 renders (compute2 / rented; NOT compute3: 5.1 RTX segfaults there).
#
#   scripts/simlab/dev_eval.sh --task stack_blocks --policy controlr [--eval-num 25]
#        [--layouts /root/dojo-data/simlab/layouts] [--out /root/dojo-data/simlab/dev_eval]
#
# Layout source: $LAYOUTS/<task>/<id>.json (dojo/simlab/dev_eval.py patches SeedManager; RoboDojo's
# result "layout_id" i = the i-th smallest dev id). Policy: XPolicyLab/policy/<policy>/ must exist in
# the RoboDojo tree (its deploy.yml drives the ws policy server, as in scripts/robodojo/run.sh).
# Holds the team GPU lock; results land in $OUT/<stamp>_<task>_<policy>/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RD_HOME="${RD_HOME:-/root/controlr-robodojo}"; RD="${RD_ROOT:-$RD_HOME/RoboDojo}"
TASK="" POLICY="" NUM=25 LAYOUTS="${DOJO_DATA:-/root/dojo-data}/simlab/layouts" OUT="${DOJO_DATA:-/root/dojo-data}/simlab/dev_eval"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;; --policy) POLICY="$2"; shift 2 ;; --eval-num) NUM="$2"; shift 2 ;;
    --layouts) LAYOUTS="$2"; shift 2 ;; --out) OUT="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TASK" && -n "$POLICY" ]] || { echo "--task and --policy are required" >&2; exit 2; }
[[ -d "$RD/XPolicyLab/policy/$POLICY" ]] || { echo "no XPolicyLab policy $POLICY in $RD" >&2; exit 2; }
[[ "$LAYOUTS" != *Eval_Layout* ]] || { echo "refusing eval layouts" >&2; exit 2; }
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; RUN="$OUT/${STAMP}_${TASK}_${POLICY}"; mkdir -p "$RUN"
LOCK="${DOJO_GPU_LOCK:-$HOME/dojo-gpu.lock}"; exec 9>>"$LOCK"
flock -n 9 || { echo "[dev_eval] waiting for $LOCK: $(cat "$LOCK")" >&2; flock -w "${SIMLAB_LOCK_WAIT_S:-43200}" 9 || exit 75; }
printf 'pid=%s stream=simlab start=%s cmd=dev_eval %s %s\n' "$$" "$(date -u +%FT%TZ)" "$TASK" "$POLICY" > "$LOCK"
free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
WS_PORT="$(free_port)"; PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -TERM -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
( cd "$RD/XPolicyLab" && PYTHONPATH="$RD" exec setsid "${POLICY_PY:-$RD_HOME/policy_venv/bin/python}" setup_policy_server.py \
    --config_path "$RD/XPolicyLab/policy/$POLICY/deploy.yml" --overrides port="$WS_PORT" host=127.0.0.1 policy_name="$POLICY" \
    > "$RUN/policy_server.log" 2>&1 ) &
PIDS+=($!)
for _ in $(seq 120); do (echo > /dev/tcp/127.0.0.1/"$WS_PORT") 2>/dev/null && break; sleep 1; done
source "$RD_HOME/miniconda3/bin/activate" "$RD_HOME/miniconda3/envs/robodojo"
export OMNI_KIT_ACCEPT_EULA=YES SIMLAB_DEV_LAYOUTS="$LAYOUTS" RD_ROOT="$RD" EVAL_NUM="$NUM"
export PYTHONPATH="$HERE:$RD:$RD/XPolicyLab:${PYTHONPATH:-}" ROBODOJO_RUN_ID="$STAMP"
cd "$RUN"            # main.py writes eval_result/ relative to cwd
python -u "$HERE/dojo/simlab/dev_eval.py" --task_name "$TASK" --env_cfg_type arx_x5 --num_envs 1 --enable_cameras \
  --kit_args " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera" --device_id 0 \
  --policy_name "$POLICY" --port "$WS_PORT" --protocol ws --policy_server_url "ws://127.0.0.1:$WS_PORT" \
  --additional_info simlab_dev --seed 0 --host 127.0.0.1 --headless 2>&1 | tee "$RUN/eval_client.log"
find "$RUN/eval_result" -name _result.json -exec cat {} \; 2>/dev/null | head -c 2000
