#!/usr/bin/env bash
# The isr-lab gate before each episode (dojo.runtime.eval runs it as DOJO_EVAL_GATE; isr_block.sh has the same
# checks): rc 0 = go; 76 = an unprocessed inbox message or gpu.reserve set (the lead reads / Mikhail's side has
# priority); 77 = data2 below ISR_DISK_FLOOR_GB free (default 7; Mikhail's floor is 6).
COORD=/home/anywherevla/data2/coord
m="$(find "$COORD/inbox/ilia" -maxdepth 1 -type f 2>/dev/null | head -1)"
[[ -z "$m" ]] || { echo "unprocessed inbox message $m"; exit 76; }
[[ -s "$COORD/gpu.reserve" ]] && { echo "gpu.reserve is set"; exit 76; }
free_mb=$(df -BM --output=avail /home/anywherevla/data2 | tail -1 | tr -dc 0-9); free_gb=$(awk -v m="$free_mb" 'BEGIN {printf "%.1f", m / 1024}')
(( free_mb >= ${ISR_DISK_FLOOR_GB:-7} * 1024 )) || { echo "only ${free_gb} GB free on data2"; exit 77; }
exit 0
