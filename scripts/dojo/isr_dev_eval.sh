#!/usr/bin/env bash
# Rendered Isaac 5.1 dev eval of our runtime on OUR settled layouts (>= 100000) on isr-lab: the proxy driver
# (dojo.runtime.eval) with --renderer isaac51 (RenderedEpisode: one RoboDojo EvalEnv with RTX, the eval client's app
# flags), plus our perception service and twin; no 6.0 render service, no robot server.
#   scripts/dojo/isr_dev_eval.sh NAME ETA_MIN --task T --layouts L... [--policy module:Class] [--config JSON]
#        [--twin-envs 8] [--max-minutes 100]
# One coord gpu.lock hold (isr_job.sh); every episode is gated by scripts/dojo/isr_gate.sh (inbox, gpu.reserve, disk)
# through DOJO_EVAL_GATE. Mode perception + verify bundle always (the strict-path setup). Results:
# $DOJO_DATA/bridge/dev_eval/NAME/ (dojo.runtime.eval's episodes.jsonl / summary.json / reports).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ $# -ge 4 ]] || { echo "usage: isr_dev_eval.sh NAME ETA_MIN --task T --layouts L... [options]" >&2; exit 2; }
NAME="$1" ETA="$2"; shift 2
export CR="$HERE"
source "$HERE/scripts/dojo/isr_env.sh"
if [[ "${ISR_IN_JOB:-0}" != 1 ]]; then
  ISR_IN_JOB=1 ISR_MAX_S="${ISR_MAX_S:-7200}" exec "$HERE/scripts/dojo/isr_job.sh" "$NAME" "$ETA" "$0" "$NAME" "$ETA" "$@"
fi
TASK="" TWIN_ENVS=8 args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; args+=("$1" "$2"); shift 2 ;;
    --twin-envs) TWIN_ENVS="$2"; shift 2 ;;
    --mode|--verify|--renderer|--out) echo "isr_dev_eval.sh: $1 is fixed here" >&2; exit 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
[[ -n "$TASK" ]] || { echo "--task is required" >&2; exit 2; }
OUT="$DOJO_DATA/bridge/dev_eval/$NAME"; mkdir -p "$OUT"
LOCK="$DOJO_GPU_LOCK"; exec 9>>"$LOCK"; flock -w 600 9 || { echo "[dev_eval] session lock busy" >&2; exit 75; }
printf 'pid=%s stream=bridge-isr start=%s dev_eval %s\n' "$$" "$(date -u +%FT%TZ)" "$NAME" > "$LOCK"
key() { python3 -c 'import secrets; print(secrets.token_hex(16))'; }
port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
export SIMLAB_GPU=0 SIMLAB_MIN_FREE_MB=0            # this session holds the lock; children never queue on it
export SIMLAB_TWIN_AUTHKEY="$(key)" SIMLAB_TWIN_PORT="$(port)" SIMLAB_TWIN_READY="$OUT/twin.ready"
export DOJO_PERCEPTION_AUTHKEY="$(key)" DOJO_PERCEPTION_PORT="$(port)" DOJO_PERCEPTION_READY="$OUT/perception.ready"
export DOJO_AGENTIC_LOG="$OUT/agentic" DOJO_AGENT_LOG="$OUT/agent" DOJO_EVAL_GATE="$HERE/scripts/dojo/isr_gate.sh"
rm -f "$SIMLAB_TWIN_READY" "$DOJO_PERCEPTION_READY"
PIDS=()
cleanup() {
  python3 - <<'PY'
import os
from multiprocessing.connection import Client
for p, k in (("SIMLAB_TWIN_PORT", "SIMLAB_TWIN_AUTHKEY"), ("DOJO_PERCEPTION_PORT", "DOJO_PERCEPTION_AUTHKEY")):
    try:
        c = Client(("127.0.0.1", int(os.environ[p])), authkey=bytes.fromhex(os.environ[k])); c.send({"op": "close"}); c.recv()
    except Exception as e:  # noqa: BLE001
        print(p, "close:", e)
PY
  sleep 5
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  wait 2>/dev/null
  rm -rf "$OUT/work" "$OUT"/stage_current.usd
}
trap cleanup EXIT
( cd "$HERE" && exec "$HERE/scripts/simlab/rdpy.sh" -u -m dojo.simlab.twin_service --task "$TASK" --num-envs "$TWIN_ENVS" 9>&- ) \
  > "$OUT/twin_service.log" 2>&1 &
PIDS+=($!)
( cd "$HERE" && PYOPENGL_PLATFORM=egl exec nice "$HERE/.venv/bin/python" -u -m dojo.runtime.perception_service 9>&- ) \
  > "$OUT/perception_service.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 300); do [[ -f "$SIMLAB_TWIN_READY" && -f "$DOJO_PERCEPTION_READY" ]] && break; sleep 2; done
[[ -f "$SIMLAB_TWIN_READY" && -f "$DOJO_PERCEPTION_READY" ]] || { echo "[dev_eval] services not ready (see $OUT/*.log)" >&2; exit 1; }
echo "[dev_eval] $(date -u +%T) $NAME: twin x$TWIN_ENVS + perception ready; driver starts" >&2
"$HERE/scripts/simlab/rdpy.sh" -u -m dojo.runtime.eval --out "$OUT" --mode perception --verify bundle --renderer isaac51 \
  "${args[@]}" 9>&- 2>&1 | tee "$OUT/dev_eval.log" | grep -a "^\[integ\]"
exit "${PIPESTATUS[0]}"
