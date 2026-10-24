#!/usr/bin/env bash
# Start an integ_eval job fully detached on compute3 (survives the ssh session):
#   scripts/dojo/launch.sh NAME --task T --mode gt|perception --verify off|bundle --layouts ...
# Output dir and log: $DOJO_DATA/integ/runs/NAME{,.log}. Jobs queue serially: a new job first waits until no
# other integ_eval.sh of ours is running (that includes its 90 s lock yield), then queues on the GPU lock.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="${DOJO_DATA:-$HOME/dojo-data}/integ/runs"
name=$1; shift
mkdir -p "$RUNS"
cd "$HERE"
printf '%q ' "$@" > "$RUNS/$name.args"
setsid nohup bash -c 'while pgrep -u "$(id -u)" -f "^bash scripts/dojo/integ_eval.sh" >/dev/null; do sleep 10; done
  exec bash scripts/dojo/integ_eval.sh --out "$0" "$@"' "$RUNS/$name" "$@" > "$RUNS/$name.log" 2>&1 < /dev/null &
echo "launched $name pid $!"
