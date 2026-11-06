#!/usr/bin/env bash
# Back-to-back dev-only gates. Starts only after training releases the heavy lock.
set -euo pipefail
N=$(cd "$(dirname "$0")/../../.." && pwd)
D=$HOME/dojo-data/s-v2n-seg
PY=$HOME/controlr-identity/.venv/bin/python
exec 9>>"$HOME/dojo-gpu.lock"
flock -w 10800 9
source "$HOME/dojo-c/scripts/perception/bgperc/env.sh"
export PYTHONPATH="$N:$HOME/dojo-w-ext/pylib:$HOME/dojo-w-ext/pyseg"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export YOLO_CONFIG_DIR=$D/ultra_cfg TORCH_HOME=$D/torch_home DOJO_AGENTIC_LOG=$D/agentic_log
export DOJO_GL=1 PYOPENGL_PLATFORM=egl
source "$N/scripts/dojo/disk_guard.sh"
CPUS=$(lscpu -p=CPU,CORE,SOCKET | python3 -c 'import sys
r=[tuple(map(int,s.split(","))) for s in sys.stdin if not s.startswith("#")]
a=set(sorted({(s,c) for _,c,s in r})[:-4])
print(",".join(str(cpu) for cpu,c,s in r if (s,c) in a))')
taskset -pc "$CPUS" "$$"
preflight() { who; uptime; nvidia-smi; disk_guard SEG; }
cd "$N"
[[ -f "$D/weights.md5" ]] || { echo 'training did not complete cleanly'; exit 1; }
preflight
"$PY" scripts/perception/seg/additive_hitcheck.py --base "$HOME/dojo-w-ext/seg_v2b.pt" \
 --base-classes "$HOME/dojo-w-ext/classes_v2b.json" --extra "$D/runs/v2n/weights/best.pt" \
 --extra-classes "$D/yolo_v2n/classes.json" --out "$D/hitcheck.json" > "$D/hitcheck.log" 2>&1
BASE_CFG=$(cat "$HOME/dojo-data/w-ready/cfg_link_off.json")
gate() {
 local name=$1 cfg=$2
 preflight
 SEG_PY=$PY SEG_WEIGHTS=$HOME/dojo-w-ext/seg_v2b.pt SEG_CLASSES=$HOME/dojo-w-ext/classes_v2b.json \
 SEG_SERVER_LOG=$D/server_$name.log SEG_SOCK=$D/server_$name.sock \
 scripts/perception/seg/with_seg_server.sh --extra-weights "$D/runs/v2n/weights/best.pt" \
 --extra-classes "$D/yolo_v2n/classes.json" -- "$PY" scripts/perception/seg/bake_seg.py run \
 --out "$D/bake/$name" --cfg "$cfg" --sets d0_51,d0_ext,series,w51,p60 --workers 6 --gl gpu > "$D/gate_$name.log" 2>&1
 "$PY" scripts/perception/seg/dev_gate.py compare "$HOME/dojo-data/s-v2e-seg/bake/v2b-clean" "$D/bake/$name" --out "$D/$name-per-class.md"
 "$PY" scripts/perception/seg/dev_gate.py compare "$HOME/dojo-data/s-v2e-seg/bake/v2b-clean" "$D/bake/$name" --by-asset --out "$D/$name-per-asset.md"
}
gate v2b-off "$BASE_CFG"
for theta in 0.1 0.3 0.5; do
 CFG=$("$PY" - "$BASE_CFG" "$theta" <<'PY'
import json,sys
c=json.loads(sys.argv[1]);c.update(seg_additive=True,seg_additive_iou=float(sys.argv[2]),seg_additive_min_score=.5)
print(json.dumps(c))
PY
)
 gate "add-$theta" "$CFG"
done
md5sum "$HOME/dojo-w-ext/seg_v2b.pt" "$D/runs/v2n/weights/best.pt" > "$D/gated_weights.md5"
echo GATES_COMPLETE
