#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/robotics/controlr-rsi-repro
DATA=/home/robotics/dojo-data/rsi-repro
ASSET="$ROOT/PyRUA-Lean/integrations/xpolicylab/PyRUA_RSI/checkpoints/sam3.pt"
for _ in $(seq 360); do
  [[ -f "$ASSET" ]] && [[ $(stat -c %s "$ASSET") == 3450062241 ]] && break
  sleep 5
done
printf '%s  %s\n' 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e "$ASSET" | sha256sum -c -
for task in stack_blocks press_by_number insert_tubes; do
  bash "$ROOT/run_task.sh" "$task" "$DATA/smoke-layouts" 2
done
date -u +%FT%TZ > "$DATA/smoke.done"
