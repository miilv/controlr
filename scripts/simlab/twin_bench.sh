#!/usr/bin/env bash
# Twin throughput bench (deliverable 5).  scripts/simlab/twin_bench.sh [--after PID] [task] [extra args]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == --after ]]; then while kill -0 "$2" 2>/dev/null; do sleep 20; done; shift 2; fi
TASK="${1:-stack_blocks}"; shift || true
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
OUT="$DOJO_DATA/simlab/twin_bench"; mkdir -p "$OUT"
for n in 1 4 8 16; do
  SIMLAB_MIN_FREE_MB="${SIMLAB_MIN_FREE_MB:-6000}" "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.twin --task "$TASK" \
      --num-envs "$n" --steps 100 --out "$OUT" "$@" > "$OUT/bench_${TASK}_n${n}.log" 2>&1
  echo "$(date -u +%T) n=$n rc=$? $(tr '\r' '\n' < "$OUT/bench_${TASK}_n${n}.log" | grep -a '^\[twin\]')" >> "$OUT/progress.txt"
done
