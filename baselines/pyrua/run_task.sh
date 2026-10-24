#!/usr/bin/env bash
# One RoboDojo evaluation of PyRUA-Lean (upstream adapter, unmodified) on compute2.
#
#   bash /root/dojo-pyrua/glue/run_task.sh --task stack_blocks --seed 0 --eval-num 1
#
# What it starts, and stops on exit:
#   1. XPolicyLab's ws policy server with policy PyRUA_Lean (our venv /root/dojo-pyrua/venv: torch
#      cu128 + sam3 + pyrualean), i.e. what upstream's setup_eval_policy_server.sh runs;
#   2. RoboDojo's eval client (`scripts/robodojo.sh client`, Isaac Sim 5.1, the shared conda env),
#      which owns the episodes: layouts 0..N-1 of --seed, step limits, success, _result.json, mp4s.
# Upstream's eval.sh does the same but wants two conda env names; ours are a venv and a prefix env.
#
# LLM: GPT-6 Astra through omniroute only. PyRUA reads PYRUALEAN_BASE_URL and the key from the env
# variable named by PYRUALEAN_API_KEY_ENV, so the omniroute key is never copied into OPENAI_*.
# Output: /root/dojo-pyrua/runs/<stamp>_<task>_s<seed>/ (logs, episodes/ = PyRUA records,
# robodojo_result/ = RoboDojo's _result.json + mp4s).
set -euo pipefail
ROOT=/root/dojo-pyrua
RD_HOME=/root/controlr-robodojo
RD="$RD_HOME/RoboDojo"
XPL="$RD/XPolicyLab"
POLICY=PyRUA_Lean
TASK="" SEED=0 NUM=1 MODEL="${PYRUALEAN_MODEL:-cx/gpt-6-astra}" MIN_FREE_MB="${MIN_FREE_MB:-12000}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --eval-num) NUM="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TASK" ]] || { echo "--task is required" >&2; exit 2; }
[[ -L "$XPL/policy/$POLICY" ]] || { echo "missing $XPL/policy/$POLICY (see baselines/pyrua/setup.sh)" >&2; exit 1; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/runs/${STAMP}_${TASK}_s${SEED}"
mkdir -p "$OUT/episodes"

# credentials: omniroute only (never print them)
set -a; . /root/controlr/.env; set +a
unset OPENAI_API_KEY OPENAI_BASE_URL OPENROUTER_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY || true
export PYRUALEAN_BASE_URL="$OMNIROUTE_BASE_URL" PYRUALEAN_API_KEY_ENV=OMNIROUTE_API_KEY PYRUALEAN_MODEL="$MODEL"
export SAM3_CHECKPOINT_PATH="$ROOT/checkpoints/sam3.pt" PYRUALEAN_RESULT_DIR="$OUT/episodes"
# 5xx / transport retries per request (upstream default 10, linear backoff); 429s are waited out
# by upstream regardless (5 s doubling to 120 s, within the episode's 90 min budget)
export PYRUALEAN_OPENAI_RETRIES="${PYRUALEAN_OPENAI_RETRIES:-30}"

# shared GPU: wait for room (SAM3 ~8 GB + Isaac ~5-8 GB)
while :; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [[ "$free" -ge "$MIN_FREE_MB" ]] && break
  echo "[run_task] GPU free ${free} MB < ${MIN_FREE_MB} MB, waiting 60 s"; sleep 60
done

port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
WS_PORT="$(port)"
echo "$TASK seed=$SEED eval_num=$NUM model=$MODEL port=$WS_PORT" > "$OUT/meta.txt"
git -C "$ROOT/PyRUA-Lean" rev-parse HEAD >> "$OUT/meta.txt"

SERVER_PID=""
cleanup() {
  [[ -n "$SERVER_PID" ]] && kill -TERM -- -"$SERVER_PID" 2>/dev/null || true
  pgrep -f "src/eval_client/main.py.*--port $WS_PORT" | xargs -r kill 2>/dev/null || true
  sleep 5
  pgrep -f "src/eval_client/main.py.*--port $WS_PORT" | xargs -r kill -9 2>/dev/null || true
  pgrep -f "setup_policy_server.py.*port=$WS_PORT" | xargs -r kill -9 2>/dev/null || true
  newest="$(ls -td "$RD"/eval_result/RoboDojo/"$TASK"/"$POLICY"/*/*/* 2>/dev/null | head -1 || true)"
  if [[ -n "$newest" && "$newest" -nt "$OUT/meta.txt" ]]; then
    cp -r "$newest" "$OUT/robodojo_result" && echo "RoboDojo result -> $OUT/robodojo_result"
  fi
}
trap cleanup EXIT

# 1. policy server (as setup_eval_policy_server.sh, with our venv)
( cd "$XPL" && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning PYTHONPATH="$RD" \
    exec setsid "$ROOT/venv/bin/python" setup_policy_server.py --config_path "$XPL/policy/$POLICY/deploy.yml" \
    --overrides port="$WS_PORT" host=127.0.0.1 bench_name=RoboDojo task_name="$TASK" ckpt_name=no-checkpoint \
    env_cfg_type=arx_x5 seed="$SEED" policy_name="$POLICY" action_type=joint \
    > "$OUT/policy_server.log" 2>&1 ) &
SERVER_PID=$!
for _ in $(seq 300); do (echo > /dev/tcp/127.0.0.1/"$WS_PORT") 2>/dev/null && break; sleep 1; done
(echo > /dev/tcp/127.0.0.1/"$WS_PORT") 2>/dev/null || { echo "policy server did not start:"; tail -30 "$OUT/policy_server.log"; exit 1; }

# 2. RoboDojo's eval client: the episodes
source "$RD_HOME/miniconda3/bin/activate" "$RD_HOME/miniconda3/envs/robodojo"
export OMNI_KIT_ACCEPT_EULA=YES ROBODOJO_MAX_BASH_RETRIES="${ROBODOJO_MAX_BASH_RETRIES:-2}"
echo "[run_task] $TASK seed=$SEED eval-num=$NUM -> $OUT"
set +e
( cd "$RD" && bash scripts/robodojo.sh client --task "$TASK" --policy-name "$POLICY" \
    --policy-host 127.0.0.1 --policy-port "$WS_PORT" --seed "$SEED" --eval-num "$NUM" \
    --ckpt no-checkpoint --action-type joint ) > "$OUT/eval_client.log" 2>&1
rc=$?
set -e
echo "[run_task] eval client exit $rc"
exit "$rc"
