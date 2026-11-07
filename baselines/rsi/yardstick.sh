#!/usr/bin/env bash
# Start only after documenting the Phase 1 layout decision.
set -euo pipefail
ROOT=/home/robotics/controlr-rsi-repro
DATA=/home/robotics/dojo-data/rsi-repro
if [[ ${RSI_CLEAN:-0} != 1 ]]; then
  exec env -i HOME="$ROOT/home" PATH=/usr/bin:/bin RSI_CLEAN=1 bash "$0"
fi
export PYTHONDONTWRITEBYTECODE=1 RD_HOME="$ROOT/rdhome" RD_ROOT="$ROOT/rdhome/RoboDojo"
export PYTHONPATH="$ROOT/code:$RD_ROOT:$RD_ROOT/XPolicyLab" TMPDIR="$ROOT/tmp"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XDG_CACHE_HOME="$ROOT/cache" CUDA_CACHE_PATH="$ROOT/cache/cuda" OMNI_KIT_ACCEPT_EULA=YES
export DOJO_DATA="$DATA"
PY="$RD_HOME/miniconda3/envs/robodojo/bin/python"
mkdir -p "$DATA/yardstick"
# Explicit marker: nothing can bypass the smoke decision accidentally.
[[ -f "$DATA/phase1-approved-path-a" ]] || { echo 'Phase 1 decision marker missing'; exit 2; }
[[ -f "$DATA/yardstick.start" ]] || date -u +%FT%TZ > "$DATA/yardstick.start"
if [[ ! -f "$DATA/yardstick-layouts/manifest.json" ]]; then
  "$PY" "$ROOT/prepare.py"
fi
"$PY" "$ROOT/manifest.py" "$DATA/yardstick-layouts" --verify
"$PY" -c 'from dojo.scoreboard.board import cells; print("\n".join(t for t,_,_ in cells()))' > "$DATA/tasks.txt"
while read -r task; do
  [[ -f "$DATA/yardstick/$task.done" ]] && continue
  # Existing partial output needs diagnosis, not automatic score-driven retries.
  [[ ! -f "$DATA/yardstick/$task.started" ]] || { echo "unfinished call: $task"; exit 3; }
  date -u +%FT%TZ > "$DATA/yardstick/$task.started"
  bash "$ROOT/run_task.sh" "$task" "$DATA/yardstick-layouts" 10 yardstick
  date -u +%FT%TZ > "$DATA/yardstick/$task.done"
done < "$DATA/tasks.txt"
"$PY" "$ROOT/summarize.py" "$DATA/yardstick-layouts" "$DATA"/yardstick/20*T*_* --require-complete > "$DATA/per_cell.json"
date -u +%FT%TZ > "$DATA/yardstick.done"
