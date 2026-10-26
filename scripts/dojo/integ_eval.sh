#!/usr/bin/env bash
# PROXY end-to-end eval of dojo.runtime.RuntimePolicy on compute3 (one GPU job):
#   scripts/dojo/integ_eval.sh --task T --layouts L1.json ... --mode gt|perception --verify off|bundle
#        [--twin-envs 8] [--out DIR] [--config JSON] [--max-minutes 25]
# Processes, all in ONE systemd user scope (24G / 14 cores with a twin or perception, else 16G / 12 cores):
#   render service (Isaac 6.0), [twin service (physics-only 5.1 Twin, --verify bundle)],
#   [perception service (controlr venv, EGL; --mode perception)], the driver (dojo.runtime.eval, RoboDojo python).
# Team rules: starts only with MemAvailable >= 25 GB (waits <= 10 min), holds ~/dojo-gpu.lock (blocking flock -w)
# for its lifetime (keep a job <= 30 min: --max-minutes), every child runs with SIMLAB_GPU=0; services stopped
# on exit; after releasing the lock it yields 90 s (INTEG_YIELD_S) before returning.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
OUT="$DOJO_DATA/integ/runs/$(date -u +%Y%m%dT%H%M%SZ)"
MODE=gt VERIFY=off TWIN_ENVS=8 TASK="" args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --mode) MODE="$2"; args+=("$1" "$2"); shift 2 ;;
    --verify) VERIFY="$2"; args+=("$1" "$2"); shift 2 ;;
    --twin-envs) TWIN_ENVS="$2"; shift 2 ;;
    --task) TASK="$2"; args+=("$1" "$2"); shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
[[ -n "$TASK" ]] || { echo "--task is required" >&2; exit 2; }
mkdir -p "$OUT"
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
for i in $(seq 30); do
  [[ "$(avail)" -ge 25 ]] && break
  [[ "$i" -eq 30 ]] && { echo "[integ] MemAvailable $(avail) GB < 25 GB: refusing to start" >&2; exit 75; }
  echo "[integ] MemAvailable $(avail) GB < 25 GB, waiting..." >&2; sleep 20
done
LOCK="${DOJO_GPU_LOCK:-$HOME/dojo-gpu.lock}"; exec 9>>"$LOCK"
echo "[integ] waiting for $LOCK: $(head -c 160 "$LOCK" | tr '\n' ' ')" >&2
flock -w "${INTEG_LOCK_WAIT_S:-10800}" 9 || { echo "[integ] gave up waiting for $LOCK" >&2; exit 75; }
printf 'pid=%s stream=integ start=%s cmd=integ_eval %s\n' "$$" "$(date -u +%FT%TZ)" "${args[*]}" > "$LOCK"
[[ "$(avail)" -ge 25 ]] || { echo "[integ] MemAvailable dropped below 25 GB while waiting for the lock" >&2; exit 75; }
key() { python3 -c 'import secrets; print(secrets.token_hex(16))'; }
port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
export SIMLAB_GPU=0 SIMLAB_MIN_FREE_MB=0
export SIMLAB_RENDER_AUTHKEY="$(key)" SIMLAB_RENDER_PORT="$(port)" SIMLAB_RENDER_READY="$OUT/render.ready"
export SIMLAB_TWIN_AUTHKEY="$(key)" SIMLAB_TWIN_PORT="$(port)" SIMLAB_TWIN_READY="$OUT/twin.ready"
export DOJO_PERCEPTION_AUTHKEY="$(key)" DOJO_PERCEPTION_PORT="$(port)" DOJO_PERCEPTION_READY="$OUT/perception.ready"
PD="${DOJO_PERCEPTION_DATA:-$HOME/dojo-perception-data}"
export DOJO_CATALOG="${DOJO_CATALOG:-$PD/catalog}" DOJO_X5_DIR="${DOJO_X5_DIR:-$PD/catalog/robot/x5}"
export DOJO_BG_FOLDS="${DOJO_BG_FOLDS:-$PD/bg_folds}" DOJO_COLOR_CALIB="${DOJO_COLOR_CALIB:-$PD/catalog/color_calib.json}"
# agentic perceiver (only read when the config sets "agentic"): omniroute keys from the controlr .env, transcripts here
export AGENTIC_ENV_FILE="${AGENTIC_ENV_FILE:-$HOME/controlr/.env}" DOJO_AGENTIC_LOG="${DOJO_AGENTIC_LOG:-$OUT/agentic}"
export OUT HERE MODE VERIFY TWIN_ENVS TASK
if [[ "$VERIFY" == bundle || "$MODE" == perception ]]; then MEM=24G CPU=1400%; else MEM=16G CPU=1200%; fi
systemd-run --user --scope -q -p MemoryMax=$MEM -p CPUQuota=$CPU bash -c '
  cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1)
  cpu0=$(awk "/usage_usec/ {print \$2}" $cg/cpu.stat); t0=$(date +%s.%N)
  "$HERE/scripts/simlab/isaac6.sh" dojo/simlab/render_service.py > "$OUT/render_service.log" 2>&1 &
  svc=$!; twin=""; perc=""
  if [[ "$VERIFY" == bundle ]]; then
    "$HERE/scripts/simlab/rdpy.sh" -u -m dojo.simlab.twin_service --task "$TASK" --num-envs "$TWIN_ENVS" > "$OUT/twin_service.log" 2>&1 &
    twin=$!
  fi
  if [[ "$MODE" == perception ]]; then
    ( cd "$HERE" && PYOPENGL_PLATFORM=egl nice "$HERE/.venv/bin/python" -u -m dojo.runtime.perception_service ) > "$OUT/perception_service.log" 2>&1 &
    perc=$!
  fi
  for i in $(seq 200); do
    [[ -f "$SIMLAB_RENDER_READY" ]] && { [[ -z "$twin" ]] || [[ -f "$SIMLAB_TWIN_READY" ]]; } \
      && { [[ -z "$perc" ]] || [[ -f "$DOJO_PERCEPTION_READY" ]]; } && break
    sleep 2
  done
  ok=1
  [[ -f "$SIMLAB_RENDER_READY" ]] || { echo "[integ] render service not ready: see render_service.log"; ok=0; }
  [[ -z "$twin" || -f "$SIMLAB_TWIN_READY" ]] || { echo "[integ] twin service not ready: see twin_service.log"; ok=0; }
  [[ -z "$perc" || -f "$DOJO_PERCEPTION_READY" ]] || { echo "[integ] perception service not ready: see perception_service.log"; ok=0; }
  if [[ "$ok" == 1 ]]; then
    "$HERE/scripts/simlab/rdpy.sh" -u -m dojo.runtime.eval --out "$OUT" "$@" 2>&1 | tee "$OUT/integ_eval.log" | grep -a "^\[integ\]"
  fi
  python3 - <<PY
import os
from multiprocessing.connection import Client
for p, k, on in (("SIMLAB_RENDER_PORT", "SIMLAB_RENDER_AUTHKEY", True),
                 ("SIMLAB_TWIN_PORT", "SIMLAB_TWIN_AUTHKEY", os.environ["VERIFY"] == "bundle"),
                 ("DOJO_PERCEPTION_PORT", "DOJO_PERCEPTION_AUTHKEY", os.environ["MODE"] == "perception")):
    if not on:
        continue
    try:
        c = Client(("127.0.0.1", int(os.environ[p])), authkey=bytes.fromhex(os.environ[k])); c.send({"op": "close"}); c.recv()
    except Exception as e:
        print(p, "close:", e)
PY
  sleep 5; kill $svc $twin $perc 2>/dev/null; wait $svc $twin $perc 2>/dev/null
  cpu1=$(awk "/usage_usec/ {print \$2}" $cg/cpu.stat); t1=$(date +%s.%N)
  peak=$(cat $cg/memory.peak 2>/dev/null || echo 0)
  python3 -c "import json,sys; c0,c1,t0,t1,pk=map(float,sys.argv[1:6]); json.dump({\"mem_peak_gb\": pk/1e9, \"cpu_cores_avg\": (c1-c0)/1e6/(t1-t0), \"wall_s\": t1-t0, \"mode\": sys.argv[6], \"verify\": sys.argv[7]}, open(sys.argv[8], \"w\"), indent=1)" $cpu0 $cpu1 $t0 $t1 $peak "$MODE" "$VERIFY" "$OUT/scope_stats.json"
  cat "$OUT/scope_stats.json"
' _ "${args[@]}"
rm -f "$OUT/stage_current.usd"; rm -rf "$OUT/work"          # scratch (37 MB stage export per job; disk is shared)
echo "[integ] results: $OUT"
exec 9>&-
sleep "${INTEG_YIELD_S:-90}"
