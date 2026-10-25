#!/usr/bin/env bash
# G3 / probes on compute3 under the team's resource rules:
#   scripts/skills/g3.sh run --task stack_blocks --seeds 100000-100024 --out ~/dojo-data/skills/g3/<run>
#   scripts/skills/g3.sh probe --out ~/dojo-data/skills/probe [--curobo]
# - starts only when MemAvailable >= 25 GB (waits up to 2 h);
# - holds the shared GPU lock ~/dojo-gpu.lock (one Isaac process per machine) for its lifetime,
#   taken with a blocking flock -w (no polling); keep one job <= 30 min (<= 10 layouts per job:
#   scripts/skills/g3_chunks.sh splits sweeps and pauses 90 s between jobs);
# - runs RoboDojo's python inside a systemd user scope: MemoryMax=10G, CPUQuota=600%.
# Physics-only light lane (no render service; parallel to the shared lane):
#   DOJO_GPU_LOCK=~/dojo-gpu-light.lock G3_MIN_MEM_GB=20 G3_MEMORY_MAX=8G G3_ABORT_MEM_GB=8 (abort below 8 GB)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
# lane settings (defaults = the shared GPU lane); the physics-only light lane:
#   DOJO_GPU_LOCK=~/dojo-gpu-light.lock G3_MIN_MEM_GB=20 G3_MEMORY_MAX=8G G3_ABORT_MEM_GB=8
MIN_GB="${G3_MIN_MEM_GB:-25}"; MEMMAX="${G3_MEMORY_MAX:-10G}"; ABORT_GB="${G3_ABORT_MEM_GB:-0}"
for i in $(seq 360); do
  [[ "$(avail)" -ge "$MIN_GB" ]] && break
  [[ $((i % 6)) -eq 1 ]] && echo "[g3.sh] MemAvailable $(avail) GB < $MIN_GB, waiting..." >&2
  sleep 20
done
[[ "$(avail)" -ge "$MIN_GB" ]] || { echo "[g3.sh] not enough memory, giving up" >&2; exit 75; }
LOCK="${DOJO_GPU_LOCK:-$HOME/dojo-gpu.lock}"
exec 9>>"$LOCK"
echo "[g3.sh] waiting for $LOCK (held by: $(tr '\n' ' ' < "$LOCK" | cut -c1-120))" >&2
flock -w "${G3_LOCK_WAIT_S:-14400}" 9 || { echo "[g3.sh] gave up waiting for $LOCK" >&2; exit 75; }
printf 'pid=%s stream=%s start=%s cmd=g3 %s\n' "$$" "${DOJO_STREAM:-skills}" "$(date -u +%FT%TZ)" "$*" > "$LOCK"
[[ "$(avail)" -ge "$MIN_GB" ]] || { echo "[g3.sh] memory dropped below $MIN_GB GB while waiting for the lock" >&2; exit 75; }
export SIMLAB_GPU=0 SIMLAB_MIN_FREE_MB=0      # the lock is held here
if ! systemd-run --user --scope -q true 2>/dev/null; then
  echo "[g3.sh] systemd-run --user --scope unavailable; refusing to run unscoped" >&2; exit 76
fi
if [[ "$ABORT_GB" -le 0 ]]; then
  exec systemd-run --user --scope -q -p MemoryMax="$MEMMAX" -p CPUQuota=600% \
    "$HERE/scripts/simlab/rdpy.sh" -u -m dojo.skills.g3 "$@"
fi
# with a memory floor: run as a child and abort our own job if MemAvailable drops below it
systemd-run --user --scope -q -p MemoryMax="$MEMMAX" -p CPUQuota=600% \
  "$HERE/scripts/simlab/rdpy.sh" -u -m dojo.skills.g3 "$@" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null; wait "$child"; exit 143' TERM INT
while kill -0 "$child" 2>/dev/null; do
  if [[ "$(avail)" -lt "$ABORT_GB" ]]; then
    echo "[g3.sh] MemAvailable $(avail) GB < $ABORT_GB: aborting our job" >&2
    kill -TERM "$child" 2>/dev/null; sleep 10; kill -KILL "$child" 2>/dev/null
    wait "$child"; exit 77
  fi
  sleep 5
done
wait "$child"
