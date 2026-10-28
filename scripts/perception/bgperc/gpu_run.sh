#!/usr/bin/env bash
# Stream C: run one command as a light GPU job (GL silhouettes / SAM 3.1) under the slot's flock.
#   scripts/perception/bgperc/gpu_run.sh [--min-free-mb N] CMD ...
# Lock: $BGPERC_LOCK (default ~/dojo-gpu-light3.lock, shared with stream V: blocking flock, fair queue). Needs
# MemAvailable >= 10 GB and free VRAM >= N MiB (default 4000) at the start; yields BGPERC_YIELD_S (30) after.
set -uo pipefail
MIN_MB=4000
if [[ "${1:-}" == --min-free-mb ]]; then MIN_MB="$2"; shift 2; fi
LOCK="${BGPERC_LOCK:-$HOME/dojo-gpu-light3.lock}"
exec 9>>"$LOCK"
flock -w "${BGPERC_LOCK_WAIT_S:-7200}" 9 || { echo "[gpu_run] gave up waiting for $LOCK" >&2; exit 75; }
printf 'pid=%s stream=bgperc-C start=%s cmd=%s\n' "$$" "$(date -u +%FT%TZ)" "$(echo "$*" | cut -c1-120)" > "$LOCK"
ok=0
for i in $(seq 60); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  mem=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [[ "$free" -ge "$MIN_MB" && "$mem" -ge 10 ]]; then ok=1; break; fi
  [[ $((i % 6)) == 1 ]] && echo "[gpu_run] waiting: VRAM free $free MiB (need $MIN_MB), MemAvailable $mem GB" >&2
  sleep 10
done
if [[ "$ok" != 1 ]]; then        # never start below the floor: give the slot back and report
  echo "[gpu_run] VRAM free ${free} MiB < ${MIN_MB} MiB (or MemAvailable ${mem} GB < 10) after 10 min: not started" >&2
  exec 9>&-
  exit 75
fi
"$@"
rc=$?
exec 9>&-
sleep "${BGPERC_YIELD_S:-30}"
exit $rc
