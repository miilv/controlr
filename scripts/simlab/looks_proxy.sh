#!/usr/bin/env bash
# Look-pose frames (frame_L1) for perception on compute3: states in the physics-only 5.1 twin, then the
# Isaac 6.0 proxy render. Each step is one GPU job (lock), 90 s yield in between (team lock policy).
#   scripts/simlab/looks_proxy.sh LOOKS.json [task ...]   (default: every task in the file)
#   FRAME=L2 SKIP_SAME_AS=v1.json scripts/simlab/looks_proxy.sh v2.json ...   (frame_L2, skip unchanged layouts)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}" SIMLAB_MIN_FREE_MB="${SIMLAB_MIN_FREE_MB:-6000}"
LOOKS="$1"; shift
OUT="$DOJO_DATA/d0"; mkdir -p "$OUT/logs"
tasks=("$@")
[[ ${#tasks[@]} -gt 0 ]] || mapfile -t tasks < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('\n'.join(sorted({k.split('/')[0] for k,v in d['layouts'].items() if v['looks']})))" "$LOOKS")
for t in "${tasks[@]}"; do
  systemd-run --user --scope -q -p MemoryMax=20G -p CPUQuota=1200% "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.look_render \
      --looks "$LOOKS" --task "$t" --frame "${FRAME:-L1}" ${SKIP_SAME_AS:+--skip-same-as "$SKIP_SAME_AS"} \
      > "$OUT/logs/${t}_looks_${FRAME:-L1}.log" 2>&1
  echo "$(date -u +%FT%TZ) $t looks ${FRAME:-L1} rc=$? $(tr '\r' '\n' < "$OUT/logs/${t}_looks_${FRAME:-L1}.log" | grep -ac '^\[looks\]') layouts" >> "$OUT/logs/progress_looks.txt"
  sleep "${SIMLAB_LOCK_YIELD_S:-90}"
  "$HERE/scripts/simlab/isaac6.sh" dojo/simlab/render_d0_proxy.py --root "$OUT" --tasks "$t" > "$OUT/logs/${t}_looks_${FRAME:-L1}_render.log" 2>&1
  echo "$(date -u +%FT%TZ) $t looks ${FRAME:-L1} render rc=$? $(grep -ac '^\[render\].*frames in' "$OUT/logs/${t}_looks_${FRAME:-L1}_render.log") layouts" >> "$OUT/logs/progress_looks.txt"
  sleep "${SIMLAB_LOCK_YIELD_S:-90}"
done
