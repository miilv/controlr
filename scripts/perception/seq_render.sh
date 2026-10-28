#!/usr/bin/env bash
# DEV (stream C): per-step frame sequences of recorded proxy episodes, one integ_eval.sh job per episode (heavy lock,
# mode gt, no twin / perception service): dojo.perception.seq_capture replays the run's executed actions.
#   scripts/perception/seq_render.sh JOBS_FILE OUT_ROOT
# JOBS_FILE lines: "<task> <recorded run dir> <layout json>" (# comments). Frames go to OUT_ROOT/seq/<name>/, the
# replay's eval rows (dev_gt_probe: compare with the recorded run's, scripts/perception/bgperc/replay_check.py) to
# OUT_ROOT/runs/<name>/; <name> = <task>_<lid>, or the recorded run dir's name with SEQ_NAME_BY_SRC=1 (two arms of one
# layout). SEQ_RENDERER=isaac51: genuine 5.1 frames (a role-full box), else the 6.0 proxy render service.
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
  name="${task}_${lid}"
  [[ "${SEQ_NAME_BY_SRC:-0}" == 1 ]] && name=$(basename "$src")
  if [[ -s "$ROOT/seq/$name/steps.jsonl" && -f "$ROOT/runs/$name/_done" ]]; then
    echo "[seq] $name: done already"; continue
  fi
  used=$(du -sm "$ROOT" | cut -f1)
  if (( used > MAX_GB * 1024 )); then echo "[seq] $ROOT uses ${used} MB > ${MAX_GB} GB: stop"; break; fi
  echo "[seq] $(date -u +%H:%M:%S) $name ($task $lid) from $src"
  rend=()
  [[ -n "${SEQ_RENDERER:-}" ]] && rend=(--renderer "$SEQ_RENDERER")
  DOJO_STREAM=bgperc-C DOJO_SEQ_SRC="$src" DOJO_SEQ_OUT="$ROOT/seq" DOJO_SEQ_NAME="$name" \
    "$HERE/scripts/dojo/integ_eval.sh" --task "$task" --layouts "$lay" --mode gt --verify off "${rend[@]}" \
      --policy dojo.perception.seq_capture:SeqReplayPolicy --out "$ROOT/runs/$name" \
      --max-minutes 25 --episode-max-minutes 25 > "$ROOT/logs/$name.log" 2>&1
  rc=$?
  grep -a "^\[integ\] ${task}_gt_verify-off\|dev GT probe" "$ROOT/logs/$name.log" | head -2
  [[ $rc == 0 ]] && touch "$ROOT/runs/$name/_done" || echo "[seq] $name: rc=$rc"
done < "$JOBS"
echo "[seq] $(date -u +%H:%M:%S) all done"
