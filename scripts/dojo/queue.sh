#!/usr/bin/env bash
# Ordered job queue for integ evals on compute3 (one runner; jobs strictly in file order):
#   scripts/dojo/queue.sh add NAME --task T --mode M --verify V [--repo ~/other-checkout] --layouts ...  # append
#   scripts/dojo/queue.sh start                                                 # start the runner (detached)
#   scripts/dojo/queue.sh list
# Queue file: $DOJO_DATA/integ/queue.txt (one "NAME ARGS..." per line; edit to reorder). Each job runs
# scripts/dojo/integ_eval.sh (GPU lock, one scope, 90 s yield after the release). Keep a job <= 5 episodes
# (team rule: <= 30 min per lock hold, bundle boot included).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D="${DOJO_DATA:-$HOME/dojo-data}/integ"; Q="$D/queue.txt"
mkdir -p "$D/runs"; touch "$Q"
case "${1:-}" in
  add) shift; name=$1; shift; flock "$Q.lock" bash -c 'printf "%s\n" "$2" >> "$1"' _ "$Q" "$name $*"; echo "queued $name" ;;
  list) cut -c1-150 "$Q" ;;
  start)
    if pgrep -u "$(id -u)" -f "scripts/dojo/queue_runner.sh" >/dev/null; then echo "runner already running"; exit 0; fi
    cd "$HERE"
    setsid nohup bash scripts/dojo/queue_runner.sh >> "$D/queue_runner.log" 2>&1 < /dev/null &
    echo "runner started pid $!" ;;
  *) echo "usage: queue.sh add|start|list" >&2; exit 2 ;;
esac
