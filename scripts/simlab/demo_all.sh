#!/usr/bin/env bash
# Support-arm demos for imitate_sorting_sequence dev layouts, in chunks (each chunk one GPU-lock hold well
# under the team's 30 min; 90 s yield between chunks).   scripts/simlab/demo_all.sh [ids...]  (default 100000..100024)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
ids=("$@"); [[ ${#ids[@]} -gt 0 ]] || ids=($(seq 100000 100024))
LOG="$DOJO_DATA/simlab/traj/logs"; mkdir -p "$LOG"
CHUNK="${SIMLAB_DEMO_CHUNK:-3}"
for ((i = 0; i < ${#ids[@]}; i += CHUNK)); do
  part=("${ids[@]:i:CHUNK}")
  systemd-run --user --scope -q -p MemoryMax=20G -p CPUQuota=1200% "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.demo_gen \
      --ids "${part[@]}" --tries "${SIMLAB_DEMO_TRIES:-3}" > "$LOG/demo_${part[0]}.log" 2>&1
  echo "$(date -u +%T) ${part[*]} rc=$? $(tr '\r' '\n' < "$LOG/demo_${part[0]}.log" | grep -a '^\[demo\] [0-9]' | cut -c1-40 | tr '\n' ' ')" >> "$LOG/progress.txt"
  sleep "${SIMLAB_LOCK_YIELD_S:-90}"
done
echo "$(date -u +%T) DONE" >> "$LOG/progress.txt"
