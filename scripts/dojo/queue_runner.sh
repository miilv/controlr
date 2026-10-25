#!/usr/bin/env bash
# The ordered integ queue's runner (started by scripts/dojo/queue.sh start). Pops the first line of the queue
# file ("NAME ARGS...") and runs scripts/dojo/integ_eval.sh for it, until the file is empty.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D="${DOJO_DATA:-$HOME/dojo-data}/integ"; Q="$D/queue.txt"; RUNS="$D/runs"
cd "$HERE"
while true; do
  line=$(flock "$Q.lock" bash -c 'head -1 "$1"; sed -i 1d "$1"' _ "$Q")
  [[ -z "$line" ]] && break
  name=${line%% *}; args=${line#* }
  echo "$(date -u +%FT%TZ) start $name"
  # shellcheck disable=SC2086
  bash scripts/dojo/integ_eval.sh --out "$RUNS/$name" $args > "$RUNS/$name.log" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) done $name rc=$?"
done
echo "$(date -u +%FT%TZ) queue empty"
