#!/usr/bin/env bash
# Frozen data -> cheap matched rehearsal ablation -> v2e. Inherit the heavy lock.
set -euo pipefail
N=$(cd "$(dirname "$0")/../../.." && pwd)
D=${SEG_DATA:?stream data root}
PY=$HOME/controlr-identity/.venv/bin/python
source "$HOME/dojo-c/scripts/perception/bgperc/env.sh"
export PYTHONPATH="$N:$HOME/dojo-w-ext/pylib:$HOME/dojo-w-ext/pyseg"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export YOLO_CONFIG_DIR=$D/ultra_cfg TORCH_HOME=$D/torch_home DOJO_AGENTIC_LOG=$D/agentic_log
export DOJO_GL=1 PYOPENGL_PLATFORM=egl
source "$N/scripts/dojo/disk_guard.sh"
CPUS=$(lscpu -p=CPU,CORE,SOCKET | python3 -c 'import sys
rows=[tuple(map(int,s.split(","))) for s in sys.stdin if not s.startswith("#")]
allowed=set(sorted({(s,c) for _,c,s in rows})[:-4])
print(",".join(str(cpu) for cpu,c,s in rows if (s,c) in allowed))')
taskset -pc "$CPUS" "$$" >/dev/null
preflight() { who; uptime; nvidia-smi; disk_guard SEG; }
preflight
cd "$N"
# Never export unfinished state-only layouts as training images.
if find "$D/render0" "$D/render1" -name stage.usd -print -quit | grep -q .; then
  echo 'unfinished proxy renders'; exit 1
fi
"$PY" scripts/perception/seg/export_yolo.py --render "$D/render0,$D/render1" --out "$D/yolo_v2e" \
  --base-classes "$HOME/dojo-w-ext/classes_v2b.json" --new-keep "$D/new_keep_v2e.txt" \
  --drop-prefix Fluid/,Garment/ --rfs-t 0.02 --workers 16 --val-ids 100049 > "$D/export_full.log" 2>&1
"$PY" scripts/perception/seg/audit_rehearsal.py --base-classes "$HOME/dojo-w-ext/classes_v2b.json" \
  --export "$D/yolo_v2e" --out "$D/support_full.json"
# The control arm keeps the same 543-name head, but omits the repaired rehearsal.
python3 - "$D" <<'PY'
import sys
from pathlib import Path
D=Path(sys.argv[1])
tasks=['hang_mugs_random','sort_nesting_dolls_by_size_random','make_toast_random',
       'store_laptop_and_headphones_random','pour_liquid_into_cup_random']
for i in (0,1):
 root=D/f'narrow{i}';root.mkdir(exist_ok=True)
 for task in tasks:
  p=root/task
  if not p.exists():p.symlink_to(D/f'render{i}'/task,target_is_directory=True)
PY
"$PY" scripts/perception/seg/export_yolo.py --render "$D/narrow0,$D/narrow1" --out "$D/yolo_narrow" \
  --base-classes "$D/yolo_v2e/classes.json" --new-keep "$D/new_keep_v2e.txt" \
  --drop-prefix Fluid/,Garment/ --rfs-t 0.02 --workers 16 --val-ids 100049 > "$D/export_narrow.log" 2>&1
"$PY" scripts/perception/seg/audit_rehearsal.py --base-classes "$HOME/dojo-w-ext/classes_v2b.json" \
  --export "$D/yolo_narrow" --out "$D/support_narrow.json" || [[ $? == 1 ]]
"$PY" scripts/perception/seg/expand_head.py --model "$HOME/dojo-w-ext/seg_v2b.pt" \
  --classes "$D/yolo_v2e/classes.json" --out "$D/v2e_init.pt" > "$D/expand.log" 2>&1
train() {
  local name=$1 data=$2 epochs=$3 close=$4 batch=$5
  preflight
  "$PY" scripts/perception/seg/train.py --data "$data" --model "$D/v2e_init.pt" \
    --epochs "$epochs" --imgsz 960 --batch "$batch" --batch-headroom-gb 8 --workers 12 --close-mosaic "$close" \
    --power-target 550 --power-max 580 --no-amp-check --no-font-check --project "$D/runs" --name "$name" \
    > "$D/train_$name.log" 2>&1
}
gate() {
  local name=$1
  preflight
  SEG_PY=$PY SEG_WEIGHTS=$D/runs/$name/weights/last.pt SEG_CLASSES=$D/yolo_v2e/classes.json \
    SEG_SERVER_LOG=$D/server_$name.log SEG_SOCK=$D/server_$name.sock \
    scripts/perception/seg/with_seg_server.sh "$PY" scripts/perception/seg/bake_seg.py run \
    --out "$D/bake/$name" --cfg "$(cat "$HOME/dojo-data/w-ready/cfg_link_off.json")" \
    --sets d0_51,d0_ext,series,w51,p60 --workers 6 --gl gpu > "$D/gate_$name.log" 2>&1
  "$PY" scripts/perception/seg/dev_gate.py compare "$D/bake/v2b-clean" "$D/bake/$name" --out "$D/$name-per-class.md"
  "$PY" scripts/perception/seg/dev_gate.py compare "$D/bake/v2b-clean" "$D/bake/$name" --by-asset --out "$D/$name-per-asset.md"
}
train narrow2 "$D/yolo_narrow/data.yaml" 2 0 -1
BATCH=$("$PY" - "$D/runs/narrow2/resolved_batch.json" <<'PY'
import sys,json
print(json.load(open(sys.argv[1]))['batch'])
PY
)
# Never silently mismatch resolved batch sizes between the paired arms.
[[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || { echo "unresolved batch $BATCH; read AutoBatch log and freeze integer"; exit 1; }
gate narrow2
train rehearsal2 "$D/yolo_v2e/data.yaml" 2 0 "$BATCH"
gate rehearsal2
# Candidate is a fresh eight-epoch run from expanded v2b, not the ablation's weights.
train v2e "$D/yolo_v2e/data.yaml" 8 3 -1
gate v2e
preflight
"$PY" "$HOME/dojo-data/s-v2d/hitcheck.py" \
  "v2b,$HOME/dojo-w-ext/seg_v2b.pt,$HOME/dojo-w-ext/classes_v2b.json" \
  "v2e,$D/runs/v2e/weights/last.pt,$D/yolo_v2e/classes.json" > "$D/hitcheck_v2e.txt" 2>&1
md5sum "$D/runs/"{narrow2,rehearsal2,v2e}/weights/last.pt > "$D/weights.md5"
