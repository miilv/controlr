#!/usr/bin/env bash
# Full task coverage on compute3: per task (in the given order) generate N dev layouts (CPU), then pre-settle
# them (one GPU job, lock + 90 s yield). Status per task in ~/dojo-data/simlab/coverage/progress.txt.
#   scripts/simlab/coverage.sh N task [task ...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}" SIMLAB_MIN_FREE_MB="${SIMLAB_MIN_FREE_MB:-6000}"
N="$1"; shift
LOG="$DOJO_DATA/simlab/coverage"; mkdir -p "$LOG"
for t in "$@"; do
  SIMLAB_GPU=0 nice -n 10 "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.gen_layouts --task "$t" --n "$N" > "$LOG/${t}_gen.log" 2>&1
  rc=$?; n=$(grep -c "objects" "$LOG/${t}_gen.log")
  err=$(grep -aE "^\w*Error|^\w*Exception" "$LOG/${t}_gen.log" | tail -1 | cut -c1-160)
  echo "$(date -u +%T) $t gen rc=$rc layouts=$n ${err}" >> "$LOG/progress.txt"
  [[ "$n" -gt 0 ]] || continue
  "$HERE/scripts/simlab/presettle_all.sh" "$t" > /dev/null 2>&1
  echo "$(date -u +%T) $t presettle done" >> "$LOG/progress.txt"
done
echo DONE >> "$LOG/progress.txt"
