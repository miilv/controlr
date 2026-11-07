#!/usr/bin/env bash
# Companion for an already-running batch; publish only a complete 540-row result.
set -euo pipefail
ROOT=/home/robotics/controlr-rsi-repro
DATA=/home/robotics/dojo-data/rsi-repro
if [[ ${RSI_CLEAN:-0} != 1 ]]; then
  exec env -i HOME="$ROOT/home" PATH=/usr/bin:/bin RSI_CLEAN=1 bash "$0"
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/code"
PID=$(cat "$DATA/yardstick.pid")
while [[ ! -f "$DATA/yardstick.done" ]]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo 'batch exited before completion; inspect yardstick.log' > "$DATA/collection.failed"
    exit 1
  fi
  sleep 60
done
python3 "$ROOT/manifest.py" "$DATA/yardstick-layouts" --verify
python3 "$ROOT/summarize.py" "$DATA/yardstick-layouts" "$DATA"/yardstick/20*T*_* \
  --require-complete > "$DATA/per_cell.json.partial"
mv "$DATA/per_cell.json.partial" "$DATA/per_cell.json"
date -u +%FT%TZ > "$DATA/collection.done"
