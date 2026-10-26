#!/usr/bin/env bash
# Held-object captures in the physics-only twin (G3 snippet) on ONE light lane:
#   scripts/perception/held_capture.sh <task> <seed> [<seed> ...]
# -> ~/dojo-data/perc_held/cap/<task>/<seed>/ (frame_C<k>.json, stage.usd, cameras.json, done.json)
# Lane rules (g3.sh): MemAvailable >= 20 GB to start, MemoryMax 8G, CPUQuota 600 %, abort below 8 GB.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK="$1"; shift
LAYD="${HP_LAYD:-$HOME/dojo-data/simlab/layouts/$TASK}"
files=()
for s in "$@"; do
  f="$LAYD/$s.settled.json"; [[ -f "$f" ]] || f="$LAYD/$s.json"
  [[ -f "$f" ]] && files+=("$f")
done
[[ ${#files[@]} -gt 0 ]] || { echo "[held_capture] no layouts" >&2; exit 2; }
export HP_OUT="${HP_OUT:-$HOME/dojo-data/perc_held/cap}" HP_TASK="$TASK" HP_LAYOUTS="$(IFS=,; echo "${files[*]}")"
export DOJO_STREAM=perc_held
export DOJO_GPU_LOCKS="${DOJO_GPU_LOCKS:-$HOME/dojo-gpu-light.lock $HOME/dojo-gpu-light2.lock $HOME/dojo-gpu-light3.lock $HOME/dojo-gpu-light4.lock}"
export G3_MIN_MEM_GB="${G3_MIN_MEM_GB:-20}" G3_MEMORY_MAX="${G3_MEMORY_MAX:-8G}" G3_ABORT_MEM_GB="${G3_ABORT_MEM_GB:-8}"
export DOJO_CATALOG="${DOJO_CATALOG:-$HOME/dojo-data/skills5/catalog:$HOME/dojo-data/skills4/catalog:$HOME/dojo-data/skills3/catalog:$HOME/dojo-perception-data/catalog}"
mkdir -p "$HP_OUT/logs"
exec "$HERE/scripts/skills/g3.sh" snippet --task "$TASK" --layout "${files[0]}" --out "$HP_OUT/work_$TASK" \
    --code "$HERE/scripts/perception/held_capture_snippet.py" >> "$HP_OUT/logs/capture_$TASK.log" 2>&1
