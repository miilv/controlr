#!/usr/bin/env bash
# PROXY dev-eval on compute3 (physics-only Isaac 5.1 episodes + warm Isaac 6.0 render service).
# The official harness is scripts/simlab/dev_eval.sh (needs a 5.1-rendering GPU).
#   scripts/simlab/proxy_eval.sh --task T --layouts L1.json [L2.json ...] --policy replay:TRAJ|module:Class
#        [--bundle [--twin-envs N]] [--dev-layout-info] [--out DIR]
# --dev-layout-info (passed through to proxy_eval.py): DEV ONLY, gives the policy info["layout_file"].
# Default: episode env + render service in ONE scope (16G, 12 cores). --bundle adds a twin service
# (physics-only Twin, N <= 16 envs, default 8; the policy talks to it via dojo.simlab.twin_service.TwinClient)
# and the scope becomes 24G / 14 cores. Either way: ONE GPU job (one lock), services stopped on exit,
# peak memory / CPU of the scope written to $OUT/bundle_stats.json.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
OUT="$DOJO_DATA/simlab/proxy_eval/$(date -u +%Y%m%dT%H%M%SZ)"
BUNDLE=0 TWIN_ENVS=8 TASK="" args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --bundle) BUNDLE=1; shift ;;
    --twin-envs) TWIN_ENVS="$2"; shift 2 ;;
    --task) TASK="$2"; args+=("$1" "$2"); shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
mkdir -p "$OUT"
# shared machine: start only with >= 25 GB available (wait up to 10 min, then refuse)
for i in $(seq 30); do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  [[ "$avail" -ge "${SIMLAB_MIN_AVAIL_GB:-25}" ]] && break
  [[ "$i" -eq 30 ]] && { echo "[proxy_eval] MemAvailable ${avail} GB < ${SIMLAB_MIN_AVAIL_GB:-25} GB: refusing to start" >&2; exit 75; }
  echo "[proxy_eval] MemAvailable ${avail} GB, waiting..." >&2; sleep 20
done
LOCK="${DOJO_GPU_LOCK:-$HOME/dojo-gpu.lock}"; exec 9>>"$LOCK"
flock -n 9 || { echo "[proxy_eval] waiting for $LOCK: $(head -c 120 "$LOCK")" >&2; flock -w "${SIMLAB_LOCK_WAIT_S:-43200}" 9 || exit 75; }
printf 'pid=%s stream=simlab start=%s cmd=proxy_eval bundle=%s %s\n' "$$" "$(date -u +%FT%TZ)" "$BUNDLE" "${args[*]}" > "$LOCK"
key() { python3 -c 'import secrets; print(secrets.token_hex(16))'; }
port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
export SIMLAB_GPU=0 SIMLAB_RENDER_AUTHKEY="$(key)" SIMLAB_RENDER_PORT="$(port)" SIMLAB_RENDER_READY="$OUT/render.ready"
export SIMLAB_TWIN_AUTHKEY="$(key)" SIMLAB_TWIN_PORT="$(port)" SIMLAB_TWIN_READY="$OUT/twin.ready"
export OUT HERE BUNDLE TWIN_ENVS TASK
if [[ "$BUNDLE" == 1 ]]; then MEM=24G CPU=1400%; else MEM=16G CPU=1200%; fi
systemd-run --user --scope -q -p MemoryMax=$MEM -p CPUQuota=$CPU bash -c '
  cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1)
  cpu0=$(awk "/usage_usec/ {print \$2}" $cg/cpu.stat); t0=$(date +%s.%N)
  ( peak=0; while true; do c=$(cat $cg/memory.current 2>/dev/null || echo 0); [[ $c -gt $peak ]] && peak=$c && echo $peak > "$OUT/.mem_peak"; sleep 2; done ) &
  sampler=$!
  "$HERE/scripts/simlab/isaac6.sh" dojo/simlab/render_service.py > "$OUT/render_service.log" 2>&1 &
  svc=$!; twin=""
  if [[ "$BUNDLE" == 1 ]]; then
    "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.twin_service --task "$TASK" --num-envs "$TWIN_ENVS" > "$OUT/twin_service.log" 2>&1 &
    twin=$!
  fi
  for i in $(seq 150); do [[ -f "$SIMLAB_RENDER_READY" ]] && { [[ "$BUNDLE" != 1 ]] || [[ -f "$SIMLAB_TWIN_READY" ]]; } && break; sleep 2; done
  "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.proxy_eval --out "$OUT" "$@" 2>&1 | tee "$OUT/proxy_eval.log" | grep -a "^\[proxy_eval\]"
  python3 - <<PY
import os
from multiprocessing.connection import Client
for p, k in (("SIMLAB_RENDER_PORT", "SIMLAB_RENDER_AUTHKEY"), ("SIMLAB_TWIN_PORT", "SIMLAB_TWIN_AUTHKEY")):
    if p == "SIMLAB_TWIN_PORT" and os.environ.get("BUNDLE") != "1":
        continue
    try:
        c = Client(("127.0.0.1", int(os.environ[p])), authkey=bytes.fromhex(os.environ[k])); c.send({"op": "close"}); c.recv()
    except Exception as e:
        print(p, "close:", e)
PY
  sleep 5; kill $svc $twin 2>/dev/null; wait $svc $twin 2>/dev/null
  kill $sampler 2>/dev/null
  cpu1=$(awk "/usage_usec/ {print \$2}" $cg/cpu.stat); t1=$(date +%s.%N)
  peak=$(cat $cg/memory.peak 2>/dev/null || cat "$OUT/.mem_peak" 2>/dev/null || echo 0)
  python3 -c "import json,sys; c0,c1,t0,t1,pk=map(float,sys.argv[1:6]); json.dump({\"bundle\": sys.argv[6]==\"1\", \"twin_envs\": int(sys.argv[7]), \"mem_peak_gb\": pk/1e9, \"cpu_cores_avg\": (c1-c0)/1e6/(t1-t0), \"wall_s\": t1-t0}, open(sys.argv[8], \"w\"), indent=1)" $cpu0 $cpu1 $t0 $t1 $peak $BUNDLE $TWIN_ENVS "$OUT/bundle_stats.json"
  cat "$OUT/bundle_stats.json"
' _ "${args[@]}"
echo "[proxy_eval] results: $OUT/_result.json"
