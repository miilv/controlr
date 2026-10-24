#!/usr/bin/env bash
# G3 / probes on compute3 under the team's resource rules:
#   scripts/skills/g3.sh run --task stack_blocks --seeds 100000-100024 --out ~/dojo-data/skills/g3/<run>
#   scripts/skills/g3.sh probe --out ~/dojo-data/skills/probe [--curobo]
# - starts only when MemAvailable >= 25 GB (waits up to 2 h);
# - holds the shared GPU lock ~/dojo-gpu.lock (one Isaac process per machine) for its lifetime;
# - runs RoboDojo's python inside a systemd user scope: MemoryMax=10G, CPUQuota=600%.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
for i in $(seq 360); do
  [[ "$(avail)" -ge 25 ]] && break
  [[ $((i % 6)) -eq 1 ]] && echo "[g3.sh] MemAvailable $(avail) GB < 25, waiting..." >&2
  sleep 20
done
[[ "$(avail)" -ge 25 ]] || { echo "[g3.sh] not enough memory, giving up" >&2; exit 75; }
LOCK="${DOJO_GPU_LOCK:-$HOME/dojo-gpu.lock}"
exec 9>>"$LOCK"
for i in $(seq 1440); do
  flock -n 9 && break
  [[ $((i % 10)) -eq 1 ]] && echo "[g3.sh] GPU lock held by: $(tr '\n' ' ' < "$LOCK"), waiting..." >&2
  sleep 30
done
flock -n 9 || { echo "[g3.sh] gave up waiting for $LOCK" >&2; exit 75; }
printf 'pid=%s stream=skills start=%s cmd=g3 %s\n' "$$" "$(date -u +%FT%TZ)" "$*" > "$LOCK"
[[ "$(avail)" -ge 25 ]] || { echo "[g3.sh] memory dropped below 25 GB while waiting for the lock" >&2; exit 75; }
export SIMLAB_GPU=0 SIMLAB_MIN_FREE_MB=0      # the lock is held here
if ! systemd-run --user --scope -q true 2>/dev/null; then
  echo "[g3.sh] systemd-run --user --scope unavailable; refusing to run unscoped" >&2; exit 76
fi
exec systemd-run --user --scope -q -p MemoryMax=10G -p CPUQuota=600% \
  "$HERE/scripts/simlab/rdpy.sh" -u -m dojo.skills.g3 "$@"
