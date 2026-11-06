#!/usr/bin/env bash
# Standalone new-only model; never fine-tune or overwrite the production base.
set -euo pipefail
N=$(cd "$(dirname "$0")/../../.." && pwd)
D=$HOME/dojo-data/s-v2n-seg
PY=$HOME/controlr-identity/.venv/bin/python
mkdir -p "$D"
exec 9>>"$HOME/dojo-gpu.lock"
flock -w 60 9
source "$HOME/dojo-c/scripts/perception/bgperc/env.sh"
export PYTHONPATH="$N:$HOME/dojo-w-ext/pylib:$HOME/dojo-w-ext/pyseg"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export YOLO_CONFIG_DIR=$D/ultra_cfg TORCH_HOME=$D/torch_home
source "$N/scripts/dojo/disk_guard.sh"
CPUS=$(lscpu -p=CPU,CORE,SOCKET | python3 -c 'import sys
r=[tuple(map(int,s.split(","))) for s in sys.stdin if not s.startswith("#")]
a=set(sorted({(s,c) for _,c,s in r})[:-4])
print(",".join(str(cpu) for cpu,c,s in r if (s,c) in a))')
taskset -pc "$CPUS" "$$"
who; uptime; nvidia-smi; disk_guard SEG
cd "$N"
"$PY" scripts/perception/seg/new_only_export.py --full "$HOME/dojo-data/s-v2e-seg/yolo_v2e" \
 --keep "$HOME/dojo-data/s-v2e-seg/new_keep_v2e.txt" --out "$D/yolo_v2n" > "$D/export.log" 2>&1
"$PY" scripts/perception/seg/select_head.py --model "$HOME/dojo-data/s-v2e-seg/runs/v2e/weights/last.pt" \
 --classes "$D/yolo_v2n/classes.json" --out "$D/v2n_init.pt" > "$D/select.log" 2>&1
who; uptime; nvidia-smi; disk_guard SEG
"$PY" scripts/perception/seg/train.py --data "$D/yolo_v2n/data.yaml" --model "$D/v2n_init.pt" \
 --epochs 40 --patience 10 --imgsz 960 --batch -1 --batch-headroom-gb 8 --workers 12 --close-mosaic 5 \
 --power-target 550 --power-max 580 --no-amp-check --no-font-check --project "$D/runs" --name v2n \
 > "$D/train.log" 2>&1
md5sum "$HOME/dojo-w-ext/seg_v2b.pt" "$D/runs/v2n/weights/"{best,last}.pt > "$D/weights.md5"
echo TRAIN_COMPLETE
