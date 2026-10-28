#!/usr/bin/env bash
# The segmenter's render queue: one render_job.sh per task, in order, a 90 s yield between jobs (team lock policy).
# Resumable: seg_render skips finished layouts, so a task cut by the job's time cap continues in its next pass.
#   SEG_GPU_LOCK=~/dojo-gpu-heavy2.lock SEG_REQUIRE_IDLE=0 SEG_MAX_S=1500 \
#     scripts/perception/seg/render_queue.sh "--ids 100030-100049 --variants 1 --poses 4" task1 task2 ...
# A file ~/dojo-data/segtrain/STOP stops the queue before its next job.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS="$1"; shift
PASSES="${SEG_PASSES:-3}"
for pass in $(seq "$PASSES"); do
  for t in "$@"; do
    [[ -f "$HOME/dojo-data/segtrain/STOP" ]] && { echo "$(date -u +%FT%TZ) queue stopped (STOP file)" >> "$HOME/dojo-data/segtrain/logs/render_jobs.txt"; exit 0; }
    # shellcheck disable=SC2086
    "$HERE/render_job.sh" "$t" $ARGS
    sleep "${SEG_YIELD_S:-90}"
  done
done
