#!/usr/bin/env bash
# DEV: D0-style frames (6.0 proxy RGB, 5.1 physics-only states) for the numbered-series tasks, in the SAM stream's
# light slot: sweep_blocks 100000-09 + 100020-24, classify_objects 100000-09 -> ~/dojo-data/sam/d0_series
set -uo pipefail
cd ~/controlr-sam
source ~/dojo-env.sh
export DOJO_GPU_LOCK=$HOME/dojo-gpu-light.lock SIMLAB_QUEUE_LOCK= RD_HOME=$HOME/dojo-robodojo
OUT=$HOME/dojo-data/sam/d0_series; mkdir -p $OUT/logs
st() {  # task start n
  systemd-run --user --scope -q -p MemoryMax=14G -p CPUQuota=600% scripts/simlab/rdpy.sh -m dojo.simlab.make_d0 \
    --task $1 --start $2 --n $3 --out $OUT --physics-only --settled --max-frame 1 --index-name states_index_$2.json \
    > $OUT/logs/$1_$2_states.log 2>&1
  echo "$(date -u +%FT%TZ) states $1 $2 rc=$?"
}
st sweep_blocks 100000 10
st sweep_blocks 100020 5
st classify_objects 100000 10
sleep 20
systemd-run --user --scope -q -p MemoryMax=16G -p CPUQuota=600% scripts/simlab/isaac6.sh dojo/simlab/render_d0_proxy.py \
  --root $OUT --tasks sweep_blocks classify_objects > $OUT/logs/render.log 2>&1
echo "$(date -u +%FT%TZ) render rc=$?"
