#!/usr/bin/env bash
# Pre-settled records for every task that has dev layouts (or the tasks given). One process per task,
# each in a 20G / 12-core scope under the GPU lock (rdpy.sh).   -> layouts/<task>/<id>.settled.json
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}" SIMLAB_MIN_FREE_MB="${SIMLAB_MIN_FREE_MB:-6000}"
L="$DOJO_DATA/simlab/layouts"; LOG="$DOJO_DATA/simlab/presettle"; mkdir -p "$LOG"
tasks=("$@"); [[ ${#tasks[@]} -gt 0 ]] || mapfile -t tasks < <(ls "$L")
for t in "${tasks[@]}"; do
  ls "$L/$t"/*.json 2>/dev/null | grep -qE '/[0-9]+\.json$' || continue
  systemd-run --user --scope -q -p MemoryMax=20G -p CPUQuota=1200% "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.presettle \
      --task "$t" ${SIMLAB_PRESETTLE_ARGS:-} > "$LOG/$t.log" 2>&1
  echo "$(date -u +%T) $t rc=$? $(tr '\r' '\n' < "$LOG/$t.log" | grep -a 'settled records' | tail -1)" >> "$LOG/progress.txt"
  sleep "${SIMLAB_LOCK_YIELD_S:-90}"   # team lock policy: yield before re-taking the GPU lock
done
echo DONE >> "$LOG/progress.txt"
