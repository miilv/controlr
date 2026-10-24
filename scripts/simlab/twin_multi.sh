#!/usr/bin/env bash
# N separate single-env twin processes in parallel (called by twin_bench.sh inside its scope + lock).
#   scripts/simlab/twin_multi.sh N TASK OUT
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
N="$1" TASK="$2" OUT="$3"
for i in $(seq "$N"); do
  SIMLAB_BENCH_TAG="_proc${i}of${N}" "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.twin --task "$TASK" --num-envs 1 \
      --steps 100 --out "$OUT" --physics-only > "$OUT/bench_${TASK}_n1_proc${i}of${N}.log" 2>&1 &
done
wait
