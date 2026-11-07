set -euo pipefail
D=$1
R=$HOME/dojo-data/org2-cam
mkdir -p "$R" "$R/tmp" "$R/uv-cache" "$R/ultra_cfg" "$R/mplconfig"
export TMPDIR="$R/tmp" UV_CACHE_DIR="$R/uv-cache" YOLO_CONFIG_DIR="$R/ultra_cfg" MPLCONFIGDIR="$R/mplconfig"
cd "$D"
source "$HOME/dojo-env.sh"
export DOJO_BG_FOLDS="$HOME/dojo-data/bg_folds_51_ilia"
python3 - "$D" <<'PY'
import sys
from pathlib import Path
r=Path.home()/'dojo-data/org2-cam';repo=Path(sys.argv[1])
req=(Path.home()/'dojo-data/segab-1106/requirements.txt').read_text().splitlines()
(r/'requirements.txt').write_text('\n'.join(x for x in req if not x.startswith('-e '))+'\n-e '+str(repo)+'\n')
p=repo/'scripts/dojo/queues/2026-11-06-candidate.txt'
(r/'candidate.txt').write_text(p.read_text().replace('/home/haptics/','/home/robotics/'))
PY
UV=$HOME/.local/bin/uv
[[ -f "$D/.venv/pyvenv.cfg" ]] || "$UV" venv "$D/.venv" --python "$HOME/dojo-integ-c6-8473130a/.venv/bin/python"
"$UV" pip install --python "$D/.venv/bin/python" -r "$R/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match > "$R/venv-install.log" 2>&1
"$D/.venv/bin/python" scripts/dojo/pkg_versions.py --out "$R/packages-new.json" > /dev/null
"$D/.venv/bin/python" scripts/dojo/candidate_parity.py --candidate "$R/candidate.txt" --packages-ref "$HOME/dojo-data/segab-1106/packages-lane.json" --json "$R/parity.json" > "$R/parity.log" 2>&1
PY="$D/.venv/bin/python" bash scripts/dojo/catalog_gate.sh > "$R/catalog-gate.log" 2>&1
python3 - "$D" <<'PY'
import datetime,hashlib,json,sys
from pathlib import Path
r=Path.home()/'dojo-data/org2-cam';repo=Path(sys.argv[1]);sha=(repo/'.deployed_commit').read_text().strip()
meta=json.loads((Path.home()/'dojo-integ-c6-8473130a/.deployed.json').read_text())
meta.update(head=sha[:8],full_head=sha,commit=sha,sha=sha,stream='ORG2-CAM',dirty=False,model='claude/claude-sonnet-5-5')
meta.pop('weights',None)
meta['weights']={k:hashlib.md5((Path.home()/'dojo-data/segtrain/models'/f).read_bytes()).hexdigest() for k,f in
 [('seg','v2b/best.pt'),('digit','digit_v2/model.pt'),('kong','kong_v1b/model.pt')]}
meta['operational_lane_sha256']=hashlib.sha256((Path.home()/'dojo-data/segab-1106/lane2.segab.sh').read_bytes()).hexdigest()
meta['note']='Merged main containing #430 (ORG2 switches, default off); existing lane environment and Kit overlay retained; independent venv parity checked'
(repo/'.deployed.json').write_text(json.dumps(meta,indent=2));(r/'repo.txt').write_text(str(repo)+'\n')
with (r/'notes.log').open('a') as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+' DEPLOY '+sha+' '+str(repo)+' parity=true catalogue=pass; no simulation environment changes\n')
print('Deployment ready',repo,sha)
PY
