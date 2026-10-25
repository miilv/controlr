#!/usr/bin/env bash
# One strict full-system RoboDojo episode through the OFFICIAL eval client (docs/BRIDGE-RUNBOOK.md):
#   scripts/dojo/bridge_episode.sh --id ID --task T --layout L --seed S [--policy module:Class]
#       [--perception local|server] [--verify bundle|off] [--config JSON] [--twin-envs 8] [--out DIR] [--fake]
# Processes (one GPU box, one session lock ~/dojo-gpu.lock held for the whole episode):
#   robot-harness (Mikhail's lane, $BENCH): strict server with joint_steps + no-op ws policy server + RoboDojo's eval
#     client (Isaac 5.1 with rendering, XPolicyLab policy harness_strict), started by $ROBOT_START;
#   ours (this checkout): perception service (.venv, EGL; --perception local), the bridge driver
#     (python -m dojo.bridge.driver: RuntimePolicy / AgentPolicy over the server socket), and the twin service it
#     spawns for the instruction's scene (RoboDojo python, physics-only; --verify bundle).
# --task / --layout / --seed only START the official run (they pick the eval layout); nothing of ours reads them.
# --fake: our fake strict server on the kinematic env instead of $ROBOT_START (CPU dry run of this script; no lock).
# Env: BENCH (robot-harness install; default isr-lab's path), ROBOT_START (default $BENCH/bin/robot-start-j2),
#   RD_HOME, DOJO_PERCEPTION_DATA (catalog/, bg_folds/), DOJO_BG_FOLDS / DOJO_COLOR_CALIB (default the 6.0-built
#   sets; bg_folds_51/ and catalog/color_calib_51.json are the 5.1-built ones: within noise on genuine 5.1 frames,
#   journal 2026-10-25-dojo-perception), J1_PERCEIVER / J1_PERCEIVER_PATH (the server's perceive verb; default our
#   adapter in this checkout), DOJO_GPU_LOCK.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export BENCH="${BENCH:-/home/anywherevla/data2/bench}"
ROBOT_START="${ROBOT_START:-$BENCH/bin/robot-start-j2}"
ID="" TASK="" LAYOUT="" SEED=0 POLICY="dojo.agent.policy:AgentPolicy" PERC=local VERIFY=bundle CONFIG='{}' TWIN_ENVS=8
OUT="${DOJO_DATA:-$HOME/dojo-data}/bridge/runs" FAKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --id) ID="$2"; shift 2 ;;            --task) TASK="$2"; shift 2 ;;
    --layout) LAYOUT="$2"; shift 2 ;;    --seed) SEED="$2"; shift 2 ;;
    --policy) POLICY="$2"; shift 2 ;;    --perception) PERC="$2"; shift 2 ;;
    --verify) VERIFY="$2"; shift 2 ;;    --config) CONFIG="$2"; shift 2 ;;
    --twin-envs) TWIN_ENVS="$2"; shift 2 ;;  --out) OUT="$2"; shift 2 ;;
    --fake) FAKE=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -n "$ID" ]] || { echo "--id is required" >&2; exit 2; }
[[ "$FAKE" == 1 || ( -n "$TASK" && -n "$LAYOUT" ) ]] || { echo "--task and --layout are required" >&2; exit 2; }
mkdir -p "$OUT/$ID"
LOG="$OUT/$ID/bridge_episode.log"
log() { echo "[bridge_episode $(date -u +%T)] $*" | tee -a "$LOG" >&2; }
PY="$HERE/.venv/bin/python"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; }   # only what this script started
trap cleanup EXIT

if [[ "$FAKE" != 1 ]]; then
  LOCK="${DOJO_GPU_LOCK:-$HOME/dojo-gpu.lock}"; exec 9>>"$LOCK"
  log "waiting for $LOCK: $(head -c 160 "$LOCK" | tr '\n' ' ')"
  flock -w "${BRIDGE_LOCK_WAIT_S:-3600}" 9 || { log "gave up waiting for $LOCK"; exit 75; }
  printf 'pid=%s stream=bridge start=%s id=%s\n' "$$" "$(date -u +%FT%TZ)" "$ID" > "$LOCK"
fi

# 1. the official run: robot-harness's strict server + ws policy server + RoboDojo's eval client
export J1_PERCEIVER="${J1_PERCEIVER:-dojo.perception.j1_adapter:perceive}" J1_PERCEIVER_PATH="${J1_PERCEIVER_PATH:-$HERE}"
PD="${DOJO_PERCEPTION_DATA:-$HOME/dojo-perception-data}"
export DOJO_CATALOG="${DOJO_CATALOG:-$PD/catalog}" DOJO_X5_DIR="${DOJO_X5_DIR:-$PD/catalog/robot/x5}"
export DOJO_BG_FOLDS="${DOJO_BG_FOLDS:-$PD/bg_folds}" DOJO_COLOR_CALIB="${DOJO_COLOR_CALIB:-$PD/catalog/color_calib.json}"
if [[ "$FAKE" == 1 ]]; then
  "$PY" -m dojo.bridge.fake_server --id "$ID" --bench "$BENCH" > "$OUT/$ID/fake_server.log" 2>&1 &
  PIDS+=($!)
  for _ in $(seq 60); do [[ -S "$BENCH/run/$ID.sock" ]] && break; sleep 0.5; done
  START='{"ok":true,"fake":true}'
else
  START="$("$ROBOT_START" --task "$TASK" --layout "$LAYOUT" --seed "$SEED" --dev --reserve 20 --id "$ID" --wait-gpu 900)"
fi
log "start: $START"
[[ "$START" == *'"ok":true'* || "$START" == *'"ok": true'* ]] || { log "server start refused"; exit 1; }

# 2. our perception service (local route; the server route uses the server's perceive verb)
if [[ "$PERC" == local ]]; then
  key() { python3 -c 'import secrets; print(secrets.token_hex(16))'; }
  port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
  export DOJO_PERCEPTION_AUTHKEY="$(key)" DOJO_PERCEPTION_PORT="$(port)" DOJO_PERCEPTION_READY="$OUT/$ID/perception.ready"
  ( cd "$HERE" && DOJO_ILIA=1 PYOPENGL_PLATFORM=egl nice "$PY" -u -m dojo.runtime.perception_service ) \
    > "$OUT/$ID/perception_service.log" 2>&1 &
  PIDS+=($!)
  for _ in $(seq 150); do [[ -f "$DOJO_PERCEPTION_READY" ]] && break; sleep 2; done
  [[ -f "$DOJO_PERCEPTION_READY" ]] || { log "perception service not ready: see perception_service.log"; exit 1; }
fi

# 3. the driver: waits for the episode, resets, spawns the twin for the instruction's scene, runs the policy, closes
TWIN=spawn; [[ "$VERIFY" == bundle ]] || TWIN=off
( cd "$HERE" && DOJO_ILIA=1 DOJO_AGENT_LOG="$OUT/$ID/agent" "$PY" -u -m dojo.bridge.driver --server-id "$ID" \
    --bench "$BENCH" --policy "$POLICY" --perception "$PERC" --verify "$VERIFY" --twin "$TWIN" \
    --twin-envs "$TWIN_ENVS" --hold-gpu-lock --config "$CONFIG" --out "$OUT" --close ) 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

# 4. the official run writes its result when the server closes; wait for the launcher (never read the verdict here)
if [[ "$FAKE" != 1 ]]; then
  for _ in $(seq 150); do [[ -f "$BENCH/run/$ID/launch_done.json" ]] && break; sleep 2; done
  [[ -f "$BENCH/run/$ID/launch_done.json" ]] && log "launcher done" || log "no launch_done.json after 300 s: check $BENCH/run/$ID/launch.log"
fi
log "driver rc=$RC; outputs in $OUT/$ID"
exit "$RC"
