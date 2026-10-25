#!/usr/bin/env bash
# D0 on compute3 (RTX 5090): states from the physics-only Isaac 5.1 twin, RGB from Isaac 6.0 as a
# proxy renderer (5.1 RTX segfaults on this driver). Serial: each step takes the GPU lock.
#   scripts/simlab/d0_proxy.sh N task [task ...]      -> ~/dojo-data/d0/<task>/<id>/
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
N="$1"; shift
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
OUT="$DOJO_DATA/d0"; mkdir -p "$OUT/logs"
for t in "$@"; do
  SIMLAB_MIN_FREE_MB="${SIMLAB_MIN_FREE_MB:-6000}" "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.make_d0 --task "$t" --n "$N" \
      --out "$OUT" --physics-only > "$OUT/logs/${t}_states.log" 2>&1
  echo "$(date -u +%FT%TZ) $t states rc=$? $(tr '\r' '\n' < "$OUT/logs/${t}_states.log" | grep -a '^\[d0\].*layouts in')" >> "$OUT/logs/progress.txt"
  sleep "${SIMLAB_LOCK_YIELD_S:-90}"   # team lock policy: yield before re-taking the GPU lock
  "$HERE/scripts/simlab/isaac6.sh" dojo/simlab/render_d0_proxy.py --root "$OUT" --tasks "$t" > "$OUT/logs/${t}_render.log" 2>&1
  echo "$(date -u +%FT%TZ) $t render rc=$? $(grep -ac '^\[render\].*frames in' "$OUT/logs/${t}_render.log") layouts" >> "$OUT/logs/progress.txt"
done
