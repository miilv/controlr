#!/usr/bin/env bash
# Start an integ_eval job fully detached on compute3 (survives the ssh session):
#   scripts/dojo/launch.sh NAME --task T --mode gt|perception --verify off|bundle --layouts ...
# Output dir and log: $DOJO_DATA/integ/runs/NAME{,.log}. Our jobs run strictly one after another: each holds
# ~/dojo-integ-queue.lock for its whole life (including integ_eval.sh's 90 s GPU-lock yield), then queues on
# the team GPU lock like any job.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="${DOJO_DATA:-$HOME/dojo-data}/integ/runs"
name=$1; shift
mkdir -p "$RUNS"
cd "$HERE"
printf '%q ' "$@" > "$RUNS/$name.args"
setsid nohup flock "$HOME/dojo-integ-queue.lock" bash scripts/dojo/integ_eval.sh --out "$RUNS/$name" "$@" \
  > "$RUNS/$name.log" 2>&1 < /dev/null &
echo "launched $name pid $!"
