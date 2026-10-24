#!/usr/bin/env bash
# Build dataset D0 on compute2: 6 tasks x N dev layouts (default 25), one Isaac process per task.
#   scripts/simlab/d0_all.sh [N]      -> /root/dojo-data/d0/<task>/<id>/, log /root/dojo-data/d0/logs/
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
N="${1:-25}"
OUT=/root/dojo-data/d0
mkdir -p "$OUT/logs"
for t in stack_blocks push_T build_tower stack_bowls put_bottles_into_dustbin pack_objects_into_box_random; do
  "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.make_d0 --task "$t" --n "$N" --out "$OUT" > "$OUT/logs/$t.log" 2>&1
  echo "$(date -u +%FT%TZ) $t rc=$? $(tr '\r' '\n' < "$OUT/logs/$t.log" | grep -a '^\[d0\].*layouts in')" >> "$OUT/logs/progress.txt"
  sleep "${SIMLAB_LOCK_YIELD_S:-90}"   # team lock policy: yield before re-taking the GPU lock
done
