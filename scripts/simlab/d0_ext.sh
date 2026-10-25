#!/usr/bin/env bash
# D0-style frames for more tasks (perception request): N pre-settled dev layouts per task, frame 0 + frame 1
# (left-arm look at a task object). States: physics-only Isaac 5.1; RGB: Isaac 6.0 proxy. index.json appears
# only after the task is rendered (perception's watcher keys on it). Stops past SIMLAB_D0_EXT_MB (300).
#   scripts/simlab/d0_ext.sh N task [task ...]      -> ~/dojo-data/d0_ext/<task>/<id>/
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
N="$1"; shift
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
OUT="$DOJO_DATA/d0_ext"; mkdir -p "$OUT/logs"
CAP_MB="${SIMLAB_D0_EXT_MB:-300}"
for t in "$@"; do
  used=$(du -sm "$OUT" | cut -f1)
  if [[ "$used" -ge "$CAP_MB" ]]; then echo "$(date -u +%FT%TZ) stop: $used MB >= $CAP_MB MB" >> "$OUT/logs/progress.txt"; break; fi
  SIMLAB_MIN_FREE_MB="${SIMLAB_MIN_FREE_MB:-6000}" systemd-run --user --scope -q -p MemoryMax=20G -p CPUQuota=1200% \
      "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.make_d0 --task "$t" --n "$N" --out "$OUT" --physics-only \
      --settled --max-frame 1 --index-name states_index.json > "$OUT/logs/${t}_states.log" 2>&1
  echo "$(date -u +%FT%TZ) $t states rc=$? $(tr '\r' '\n' < "$OUT/logs/${t}_states.log" | grep -a '^\[d0\].*layouts in')" >> "$OUT/logs/progress.txt"
  sleep "${SIMLAB_LOCK_YIELD_S:-90}"   # team lock policy: yield before re-taking the GPU lock
  "$HERE/scripts/simlab/isaac6.sh" dojo/simlab/render_d0_proxy.py --root "$OUT" --tasks "$t" > "$OUT/logs/${t}_render.log" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) $t render rc=$rc $(grep -ac '^\[render\].*frames in' "$OUT/logs/${t}_render.log") layouts" >> "$OUT/logs/progress.txt"
  if [[ "$rc" -eq 0 && -f "$OUT/$t/states_index.json" ]]; then
    python3 - "$OUT/$t" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
idx = json.loads((d / "states_index.json").read_text())
for r in idx:
    fr = d / str(r["id"])
    r["frames"] = sorted(int(p.stem.split("_")[1]) for p in fr.glob("frame_*.json")) if fr.is_dir() else []
    r["rendered"] = all((fr / f"frame_{k}_cam_head.png").exists() for k in r["frames"]) if r["frames"] else False
(d / "index.json").write_text(json.dumps(idx, indent=1))
EOF
  fi
  sleep "${SIMLAB_LOCK_YIELD_S:-90}"
done
echo "$(date -u +%FT%TZ) DONE $(du -sm "$OUT" | cut -f1) MB" >> "$OUT/logs/progress.txt"
