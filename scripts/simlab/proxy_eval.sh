#!/usr/bin/env bash
# PROXY dev-eval on compute3 (physics-only Isaac 5.1 episodes + warm Isaac 6.0 render service).
# The official harness is scripts/simlab/dev_eval.sh (needs a 5.1-rendering GPU).
#   scripts/simlab/proxy_eval.sh --task T --layouts L1.json [L2.json ...] --policy replay:TRAJ.json|module:Class [--out DIR]
# One GPU lock; both processes in ONE systemd scope (MemoryMax 16G, 12 cores); the service is stopped on exit.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
OUT="$DOJO_DATA/simlab/proxy_eval/$(date -u +%Y%m%dT%H%M%SZ)"
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in --out) OUT="$2"; shift 2 ;; *) args+=("$1"); shift ;; esac
done
mkdir -p "$OUT"
LOCK="${DOJO_GPU_LOCK:-$HOME/dojo-gpu.lock}"; exec 9>>"$LOCK"
until flock -n 9; do echo "[proxy_eval] waiting for $LOCK: $(cat "$LOCK")" >&2; sleep 30; done
printf 'pid=%s stream=simlab start=%s cmd=proxy_eval %s\n' "$$" "$(date -u +%FT%TZ)" "${args[*]}" > "$LOCK"
export SIMLAB_GPU=0 SIMLAB_RENDER_AUTHKEY="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
export SIMLAB_RENDER_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
export SIMLAB_RENDER_READY="$OUT/render.ready" OUT HERE
systemd-run --user --scope -q -p MemoryMax=16G -p CPUQuota=1200% bash -c '
  "$HERE/scripts/simlab/isaac6.sh" dojo/simlab/render_service.py > "$OUT/render_service.log" 2>&1 &
  svc=$!
  for i in $(seq 120); do [[ -f "$SIMLAB_RENDER_READY" ]] && break; sleep 2; done
  "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.proxy_eval --out "$OUT" "$@" 2>&1 | tee "$OUT/proxy_eval.log" | grep -a "^\[proxy_eval\]"
  python3 - <<PY
import os
from multiprocessing.connection import Client
try:
    c = Client(("127.0.0.1", int(os.environ["SIMLAB_RENDER_PORT"])), authkey=bytes.fromhex(os.environ["SIMLAB_RENDER_AUTHKEY"]))
    c.send({"op": "close"}); c.recv()
except Exception as e:
    print("service close:", e)
PY
  sleep 5; kill $svc 2>/dev/null; wait $svc 2>/dev/null
' _ "${args[@]}"
echo "[proxy_eval] results: $OUT/_result.json"
