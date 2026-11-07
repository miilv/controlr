#!/usr/bin/env bash
# Dev-only compute3 MATCH2 job (match_and_pick_from_conveyor): one light slot, physics-only, no model calls.
# Usage: MATCH2_LOCK=light4 scripts/skills/match2_g3_job.sh -m dojo.skills.g3 run ...
#        MATCH2_LOCK=light3 scripts/skills/match2_g3_job.sh scripts/perception/m95_replay.py ...
# Admission: <= 24 GiB used VRAM, >= 10 GB free disk; an 8G / 600 % systemd scope; a bounded KILL tail.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK="${MATCH2_LOCK:-light4}"
case "$LOCK" in light3|light4) ;; *) echo "MATCH2: lock must be light3 or light4" >&2; exit 64;; esac
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${MATCH2_DATA:-$HOME/dojo-data/match2}" SIMLAB_GPU=0
mkdir -p "$DOJO_DATA"
exec 9>>"$HOME/dojo-gpu-$LOCK.lock"
flock -w 14400 9
who
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[[ "$used" =~ ^[0-9]+$ && "$used" -le 24576 ]] || { echo "MATCH2: VRAM admission refused" >&2; exit 75; }
free=$(df --output=avail -BG "$DOJO_DATA" | tail -1 | tr -dc 0-9)
[[ "$free" -ge 10 ]] || { echo "MATCH2: disk admission refused" >&2; exit 6; }
printf 'pid=%s stream=MATCH2 used_MiB=%s free_GB=%s start=%s\n' "$$" "$used" "$free" "$(date -u +%FT%TZ)" > "$HOME/dojo-gpu-$LOCK.lock"
kit="$DOJO_DATA/kit-$LOCK"
export SIMLAB_PHYSICS_KIT_ARGS="${SIMLAB_PHYSICS_KIT_ARGS:-} --portable-root $kit --/log/file=$kit.log"
ulimit -c 0
exec systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=600% \
 timeout -k 60 "${MATCH2_TIMEOUT:-7200}" "$ROOT/scripts/simlab/rdpy.sh" -u "$@"
