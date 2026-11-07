#!/usr/bin/env bash
# CLASSIFY3 camera A/B: compute4 deployment of merged main by the MOVE34 / KONG recipe (run on compute4 after the source
# tree was unpacked from `git archive` into $D). No lane, queue, coord or running job is touched.
#   bash deploy-c4.sh 2f198642900d5b6b280b6ee17918ed72c94b9365
set -euo pipefail
SHA=$1
D=$HOME/dojo-integ-c4-${SHA:0:8}
R=$HOME/dojo-integ-c4-416a2a03            # compute4's working lane venv (the board rows ran on it)
M=$HOME/dojo-data/classify3-cam/meta
mkdir -p "$M"
[[ -d "$D" && ! -e "$D/.venv" ]] || { echo "need a fresh source tree at $D without .venv" >&2; exit 2; }
cp -a "$R/.venv" "$D/.venv"
python3 - "$R" "$D" <<'PY'
import sys, pathlib
r, d = map(pathlib.Path, sys.argv[1:]); a, b = str(r).encode(), str(d).encode()
for p in (d / '.venv/bin').iterdir():
    if p.is_file() and not p.is_symlink():
        data = p.read_bytes()
        if a in data:
            p.write_bytes(data.replace(a, b))
s = d / '.venv/lib/python3.12/site-packages'
for p in list(s.iterdir()) + list(s.glob('*/direct_url.json')):
    if p.is_file() and (p.suffix in ('.pth', '.py') or p.name == 'direct_url.json'):
        data = p.read_bytes()
        if a in data:
            p.write_bytes(data.replace(a, b))
PY
"$R/.venv/bin/python" "$R/scripts/dojo/pkg_versions.py" > "$M/packages-ref.json"
"$D/.venv/bin/python" "$D/scripts/dojo/pkg_versions.py" > "$M/packages-new.json"
"$HOME/.local/bin/uv" pip freeze --python "$R/.venv/bin/python" | sed "s|$R|$D|g" > "$M/freeze-ref.txt"
"$HOME/.local/bin/uv" pip freeze --python "$D/.venv/bin/python" > "$M/freeze-new.txt"
diff "$M/freeze-ref.txt" "$M/freeze-new.txt" > "$M/freeze.diff" && echo "freeze: identical" > "$M/freeze.status" \
  || echo "freeze: DIFFERENT (see freeze.diff)" > "$M/freeze.status"
cd "$D"
.venv/bin/python -c 'import dojo, torch, cv2, transformers; print(dojo.__file__, torch.__version__, cv2.__version__, transformers.__version__)' > "$M/imports.txt"
grep -q "$D/dojo" "$M/imports.txt"
echo "$SHA" > .deployed_commit; echo "$SHA" > .deploy-sha
source "$HOME/dojo-env.sh"
export DOJO_BG_FOLDS="$HOME/dojo-data/bg_folds_51_ilia"
.venv/bin/python scripts/dojo/candidate_parity.py --candidate scripts/dojo/queues/2026-11-06-candidate.txt \
  --packages-ref "$M/packages-ref.json" --json "$M/parity.json" > "$M/parity.log" 2>&1 || true
PY="$D/.venv/bin/python" bash scripts/dojo/catalog_gate.sh > "$M/catalog-gate.log" 2>&1
python3 - "$D" "$SHA" <<'PY'
import json, sys, datetime, hashlib, subprocess
from pathlib import Path
d, sha = Path(sys.argv[1]), sys.argv[2]
meta = {"commit": sha, "head": sha[:8], "host": "compute4", "stream": "CLASSIFY3-CAM", "family_prefix": "c4b_",
        "candidate_model": "claude/claude-sonnet-5-5", "venv_from": "dojo-integ-c4-416a2a03",
        "deployed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "dirty": False,
        "note": "merged main containing #431; existing lane environment and Kit overlay retained (BASH_ENV kit-env)"}
(d / ".deployed.json").write_text(json.dumps(meta, indent=2) + "\n")
PY
echo "DEPLOYED $D $SHA"; cat "$M/freeze.status"; tail -3 "$M/parity.log"; tail -2 "$M/catalog-gate.log"
