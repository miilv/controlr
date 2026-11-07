#!/usr/bin/env bash
# One official-client call on our layouts; always one owned GPU slot.
set -euo pipefail
ROOT=/home/robotics/controlr-rsi-repro
DATA=/home/robotics/dojo-data/rsi-repro
TASK="${1:?task}" LAYOUTS="${2:?layout root}" NUM="${3:-1}" PHASE="${4:-smoke}"
# Re-exec with an allowlist. Never source dojo-env.sh or a dotenv in the runtime.
if [[ ${RSI_CLEAN:-0} != 1 ]]; then
  exec env -i HOME="$ROOT/home" USER=robotics PATH=/usr/bin:/bin RSI_CLEAN=1 bash "$0" "$@"
fi
mkdir -p "$HOME" "$DATA"
PYTHONPATH="$ROOT/code" python3 "$ROOT/manifest.py" "$LAYOUTS" --verify
exec 9>"$DATA/gpu.lock"
flock 9
# Fail closed on telemetry errors; pause admission while above the agreed budget.
while ! nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | python3 -c 'import math,sys; v=[float(x) for x in sys.stdin]; sys.exit(0 if v and all(math.isfinite(x) and 0 <= x <= 450 for x in v) else 1)' ; do sleep 30; done
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XDG_CACHE_HOME="$ROOT/cache" CUDA_CACHE_PATH="$ROOT/cache/cuda" TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
RD="$ROOT/rdhome/RoboDojo" XPL="$ROOT/rdhome/RoboDojo/XPolicyLab"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
[[ "$PHASE" == smoke || "$PHASE" == yardstick ]] || exit 2
OUT="$DATA/$PHASE/${STAMP}_${TASK}"
mkdir -p "$OUT/episodes"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
export SAM3_CHECKPOINT_PATH="$ROOT/PyRUA-Lean/integrations/xpolicylab/PyRUA_RSI/checkpoints/sam3.pt"
export PYRUALEAN_RESULT_DIR="$OUT/episodes"
PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -TERM -- -"$p" 2>/dev/null || true; done
  sleep 3
  for p in "${PIDS[@]}"; do kill -KILL -- -"$p" 2>/dev/null || true; done
}
trap cleanup EXIT
(cd "$OUT"; PYTHONPATH="$RD:$XPL" exec setsid "$ROOT/venv/bin/python" "$XPL/setup_policy_server.py" \
  --config_path "$XPL/policy/PyRUA_RSI/deploy.yml" --overrides port="$PORT" host=127.0.0.1 \
  policy_name=PyRUA_RSI bench_name=RoboDojo task_name="$TASK" ckpt_name=offline env_cfg_type=arx_x5 seed=0 action_type=joint \
  > "$OUT/policy.log" 2>&1) &
PIDS+=($!); echo "${PIDS[0]}" > "$OUT/policy.pid"
for _ in $(seq 180); do
  kill -0 "${PIDS[0]}" || { tail -30 "$OUT/policy.log"; exit 1; }
  (echo > /dev/tcp/127.0.0.1/"$PORT") 2>/dev/null && break
  sleep 1
done
export RD_HOME="$ROOT/rdhome" RD_ROOT="$RD" DOJO_DATA="$DATA"
export SIMLAB_DEV_LAYOUTS="$LAYOUTS" EVAL_NUM="$NUM" OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$ROOT/code:$RD:$XPL" ROBODOJO_RUN_ID="$STAMP"
source "$RD_HOME/miniconda3/bin/activate" "$RD_HOME/miniconda3/envs/robodojo"
cd "$OUT"
setsid python -u "$ROOT/code/dojo/simlab/dev_eval.py" --task_name "$TASK" --env_cfg_type arx_x5 --num_envs 1 --enable_cameras \
  --kit_args ' --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera' --device_id 0 \
  --policy_name PyRUA_RSI --port "$PORT" --protocol ws --policy_server_url "ws://127.0.0.1:$PORT" \
  --additional_info rsi_yardstick --seed 0 --host 127.0.0.1 --headless > "$OUT/eval.log" 2>&1 &
PIDS+=($!); echo "${PIDS[1]}" > "$OUT/eval.pid"
wait "${PIDS[1]}"
echo "$OUT"
