#!/usr/bin/env bash
# DEV (stream C): per-step frame sequences of recorded proxy episodes, one integ_eval.sh job per episode (heavy lock,
# mode gt, no twin / perception service): dojo.perception.seq_capture replays the run's executed actions.
#   scripts/perception/seq_render.sh JOBS_FILE OUT_ROOT
# JOBS_FILE lines: "<task> <recorded run dir> <layout json>" (# comments). Frames go to OUT_ROOT/seq/<task>_<lid>/,
# the replay's eval rows (dev_gt_probe: compare with the recorded run's) to OUT_ROOT/runs/<task>_<lid>/.
# One job at a time; integ_eval.sh yields 90 s after each (INTEG_YIELD_S). Stops when OUT_ROOT's disk use passes
# SEQ_MAX_GB (default 4).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOBS="$1" ROOT="$2"
MAX_GB="${SEQ_MAX_GB:-4}"
mkdir -p "$ROOT/seq" "$ROOT/runs" "$ROOT/logs"
while read -r task src lay; do
  [[ -z "$task" || "$task" == \#* ]] && continue
  lid=$(basename "$lay"); lid=${lid%%.*}
  if [[ -s "$ROOT/seq/${task}_${lid}/steps.jsonl" && -f "$ROOT/runs/${task}_${lid}/_done" ]]; then
    echo "[seq] $task $lid: done already"; continue
  fi
  used=$(du -sm "$ROOT" | cut -f1)
  if (( used > MAX_GB * 1024 )); then echo "[seq] $ROOT uses ${used} MB > ${MAX_GB} GB: stop"; break; fi
  echo "[seq] $(date -u +%H:%M:%S) $task $lid from $src"
  DOJO_STREAM=bgperc-C DOJO_SEQ_SRC="$src" DOJO_SEQ_OUT="$ROOT/seq" \
    "$HERE/scripts/dojo/integ_eval.sh" --task "$task" --layouts "$lay" --mode gt --verify off \
      --policy dojo.perception.seq_capture:SeqReplayPolicy --out "$ROOT/runs/${task}_${lid}" \
      --max-minutes 25 --episode-max-minutes 25 > "$ROOT/logs/${task}_${lid}.log" 2>&1
  rc=$?
  grep -a "^\[integ\] ${task}_gt_verify-off\|dev GT probe" "$ROOT/logs/${task}_${lid}.log" | head -2
  [[ $rc == 0 ]] && touch "$ROOT/runs/${task}_${lid}/_done" || echo "[seq] $task $lid: rc=$rc"
done < "$JOBS"
echo "[seq] $(date -u +%H:%M:%S) all done"
