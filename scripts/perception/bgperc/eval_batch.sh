#!/usr/bin/env bash
# Stream C: replay_eval over sequences, one light3 GPU hold (GL silhouettes) per sequence.
#   scripts/perception/bgperc/eval_batch.sh TAG "ARMS" SEQ_DIR... (env BGPERC_BG='{...}', BGPERC_LAG=15)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TAG="$1" ARMS="$2"; shift 2
export BGPERC_GL=1
source "$HERE/scripts/perception/bgperc/env.sh"
OUT="$HOME/dojo-data/bgperc/eval/$TAG"; mkdir -p "$OUT"
for d in "$@"; do
  n=$(basename "$d")
  [[ -s "$OUT/$n.json" ]] && { echo "[eval] $n: done"; continue; }
  echo "[eval] $(date -u +%T) $n"
  "$HERE/scripts/perception/bgperc/gpu_run.sh" --min-free-mb 2000 nice "$PY" "$HERE/scripts/perception/bgperc/replay_eval.py" "$d" \
    --arms "$ARMS" --lag "${BGPERC_LAG:-15}" --bg "${BGPERC_BG:-{\}}" --out "$OUT/$n.json" 2>&1 | grep -a "^\[bgperc\]\|Error\|Traceback" | cut -c1-400
done
echo "[eval] $(date -u +%T) all done"
