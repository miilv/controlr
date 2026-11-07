#!/usr/bin/env bash
# Dev-only compute3 POUR job: one stream slot, physics-only, no model calls.
# Usage: scripts/skills/pour_g3_job.sh -m dojo.skills.g3 run ...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="$HOME/dojo-data/pour" SIMLAB_GPU=0
mkdir -p "$DOJO_DATA"
exec 8>>"$DOJO_DATA/stream.lock"
flock -w 14400 8
exec 9>>"$HOME/dojo-gpu-light7.lock"
flock -w 14400 9
who
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[[ "$used" =~ ^[0-9]+$ && "$used" -le 24576 ]] || { echo "POUR: VRAM admission refused" >&2; exit 75; }
free=$(df --output=avail -BG "$DOJO_DATA" | tail -1 | tr -dc 0-9)
[[ "$free" -ge 10 ]] || { echo "POUR: disk admission refused" >&2; exit 6; }
printf 'stream=pour pid=%s used_MiB=%s free_GB=%s start=%s\n' "$$" "$used" "$free" "$(date -u +%FT%TZ)" > "$HOME/dojo-gpu-light7.lock"
export SIMLAB_PHYSICS_KIT_ARGS="${SIMLAB_PHYSICS_KIT_ARGS:-} --/app/userConfigPath=$DOJO_DATA/kit-config --/app/cachePath=$DOJO_DATA/kit-cache --/app/log/file=$DOJO_DATA/kit.log"
ulimit -c 0
exec systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=600% \
 timeout -k 60 3600 "$ROOT/scripts/simlab/rdpy.sh" -u "$@"
