#!/usr/bin/env bash
# Run a G3 sweep in chunks so no single job holds the shared GPU lock for long:
#   scripts/skills/g3_chunks.sh <task> <first_seed> <last_seed> <out_dir> [extra g3 run args]
# Chunks of G3_CHUNK layouts (default 10); after each chunk the lock is released and the next one
# waits G3_PAUSE_S (default 90 s) before queueing again (team rule). Episodes append to one jsonl.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK="$1"; A="$2"; B="$3"; OUT="$4"; shift 4
CHUNK="${G3_CHUNK:-10}"; PAUSE="${G3_PAUSE_S:-90}"
LOGD="${DOJO_DATA:-$HOME/dojo-data}/skills/logs"; mkdir -p "$LOGD"
s="$A"
while [[ "$s" -le "$B" ]]; do
  e=$(( s + CHUNK - 1 )); [[ "$e" -gt "$B" ]] && e="$B"
  "$HERE/scripts/skills/g3.sh" run --task "$TASK" --seeds "$s-$e" --out "$OUT" "$@" \
      >> "$LOGD/g3_${TASK}_$(basename "$OUT").log" 2>&1
  s=$(( e + 1 ))
  [[ "$s" -le "$B" ]] && sleep "$PAUSE"
done
