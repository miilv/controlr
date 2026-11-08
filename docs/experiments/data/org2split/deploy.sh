set -euo pipefail
D=$HOME/dojo-integ-c4-83887dae
R=$HOME/dojo-integ-c4-416a2a03
M=$HOME/dojo-data/org2-cam/meta
mkdir -p "$M"
python3 - "$R" "$D" <<'INNER'
import sys,pathlib
r,d=map(pathlib.Path,sys.argv[1:]);a,b=str(r).encode(),str(d).encode()
for p in (d/'.venv/bin').iterdir():
 if p.is_file() and not p.is_symlink():
  v=p.read_bytes()
  if a in v:p.write_bytes(v.replace(a,b))
s=d/'.venv/lib/python3.12/site-packages'
for p in list(s.iterdir())+list(s.glob('*/direct_url.json')):
 if p.is_file() and (p.suffix in ('.pth','.py') or p.name=='direct_url.json'):
  v=p.read_bytes()
  if a in v:p.write_bytes(v.replace(a,b))
INNER
"$R/.venv/bin/python" "$R/scripts/dojo/pkg_versions.py" > "$M/packages-ref.json"
"$D/.venv/bin/python" "$D/scripts/dojo/pkg_versions.py" > "$M/packages-new.json"
"$HOME/.local/bin/uv" pip freeze --python "$R/.venv/bin/python" | sed "s|$R|$D|g" > "$M/freeze-ref.txt"
"$HOME/.local/bin/uv" pip freeze --python "$D/.venv/bin/python" > "$M/freeze-new.txt"
diff "$M/freeze-ref.txt" "$M/freeze-new.txt" > "$M/freeze.diff"
cd "$D"
.venv/bin/python -c 'import dojo,torch,cv2,transformers; print(dojo.__file__,torch.__version__,cv2.__version__,transformers.__version__)' > "$M/imports.txt"
grep -q "$D/dojo" "$M/imports.txt"
export DOJO_BG_FOLDS="$HOME/dojo-data/bg_folds_51_ilia"
export DOJO_COLOR_CALIB="$DOJO_BG_FOLDS/frozen_d0_100040_49.color_calib.json"
.venv/bin/python scripts/dojo/candidate_parity.py --candidate scripts/dojo/queues/2026-11-06-candidate.txt --packages-ref "$M/packages-ref.json" --json "$M/parity.json" > "$M/parity.log" 2>&1
PY="$D/.venv/bin/python" bash scripts/dojo/catalog_gate.sh > "$M/catalog-gate.log" 2>&1
cp "$HOME/dojo-data/classify3-cam/meta/kit-env.sh" "$M/kit-env.sh"
sha256sum "$HOME/dojo-data/classify3-cam/meta/kit-env.sh" "$M/kit-env.sh" > "$M/kit-env.sha256"
printf '%s\n' 83887dae2ee50d582520b60d13191279a5a58f32 > "$D/.deployed_commit"
printf '%s\n' 83887dae2ee50d582520b60d13191279a5a58f32 > "$D/.deploy-sha"
echo CHECKS_PASS
