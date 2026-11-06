#!/usr/bin/env bash
# compute3 only: explicit 6.0 proxy-domain pilot, never a genuine 5.1 render.
set -euo pipefail
N=$(cd "$(dirname "$0")/../../.." && pwd)
D=${1:?stream data root}
export RD_HOME=${RD_HOME:?} DOJO_DATA=$D/data
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
source "$N/scripts/dojo/disk_guard.sh"
exec 9>>"${DOJO_GPU_LOCK:-$HOME/dojo-gpu.lock}"
flock -w 7200 9
who; uptime; nvidia-smi
disk_guard SEG
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
(( free >= 18000 )) || { echo 'insufficient VRAM for parallel pilot + reserve'; exit 75; }
# Reserve four complete physical cores, including all their hyperthreads.
CPUS=$(lscpu -p=CPU,CORE,SOCKET | python3 -c 'import sys
rows=[tuple(map(int,s.split(","))) for s in sys.stdin if not s.startswith("#")]
cores=sorted({(s,c) for _,c,s in rows}); allowed=set(cores[:-4])
print(",".join(str(cpu) for cpu,c,s in rows if (s,c) in allowed))')
export SIMLAB_GPU=0
mkdir -p "$D/logs" "$D/data"
ulimit -c 0
pids=()
for task in arrange_largest_number general_pickup; do
  taskset -c "$CPUS" timeout -k 30 600 "$N/scripts/simlab/rdpy.sh" -m dojo.simlab.seg_render \
    --task "$task" --ids 100030-100031 --variants 1 --poses 2 --physics-only --out "$D/proxy_pilot" \
    > "$D/logs/pilot_states_$task.log" 2>&1 & pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
((rc == 0)) || exit 1
who; uptime; nvidia-smi
disk_guard SEG
pids=()
for task in arrange_largest_number general_pickup; do
  taskset -c "$CPUS" timeout -k 30 600 "$N/scripts/simlab/isaac6.sh" "$N/dojo/simlab/render_d0_proxy.py" \
    --root "$D/proxy_pilot" --tasks "$task" --any-dirs --instance --jpeg 90 \
    > "$D/logs/pilot_rgb_$task.log" 2>&1 & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
exit "$rc"
