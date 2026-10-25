#!/usr/bin/env bash
# A G3 sweep on the physics-only light lanes (4 slots), chunked, from this deployed copy:
#   scripts/skills/g3_light.sh <task> <first_seed> <last_seed> <run_name> [extra g3 run args]
# Output ~/dojo-data/<stream>/g3/<run_name>_<commit>/, log ~/dojo-data/<stream>/logs/. Lane rules: start at
# MemAvailable >= 20 GB, MemoryMax 8G, our job aborts below 8 GB, <= 5 layouts per chunk, 90 s between chunks.
# Raw (unsettled) records: SIMLAB_DEV_LAYOUT_SUFFIX= (empty) in the environment.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STREAM="${DOJO_STREAM:-skills4}"
COMMIT="$(cut -c1-7 "$HERE/.deployed_commit" 2>/dev/null || echo nocommit)"
TASK="$1"; A="$2"; B="$3"; RUN="$4"; shift 4
export DOJO_STREAM="$STREAM" G3_LOGD="${G3_LOGD:-$HOME/dojo-data/$STREAM/logs}" G3_CHUNK="${G3_CHUNK:-5}" G3_PAUSE_S="${G3_PAUSE_S:-90}"
export DOJO_GPU_LOCKS="${DOJO_GPU_LOCKS:-$HOME/dojo-gpu-light.lock $HOME/dojo-gpu-light2.lock $HOME/dojo-gpu-light3.lock $HOME/dojo-gpu-light4.lock}"
export G3_MIN_MEM_GB="${G3_MIN_MEM_GB:-20}" G3_MEMORY_MAX="${G3_MEMORY_MAX:-8G}" G3_ABORT_MEM_GB="${G3_ABORT_MEM_GB:-8}"
export DOJO_CATALOG="${DOJO_CATALOG:-$HOME/dojo-data/skills4/catalog:$HOME/dojo-data/skills3/catalog:$HOME/dojo-perception-data/catalog}"
exec "$HERE/scripts/skills/g3_chunks.sh" "$TASK" "$A" "$B" "$HOME/dojo-data/$STREAM/g3/${RUN}_${COMMIT}" "$@"
