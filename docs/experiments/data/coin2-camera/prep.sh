#!/usr/bin/env bash
# COIN2-CAM: generate (fresh salt) + default two-pass presettle for every cell's id pool, on compute3 (COIN2 light5
# lane), one scoped job each; selection happens afterwards by stability only (select.py). Run from a deployment root.
#   bash prep.sh DESIGN.json DATA_ROOT
set -uo pipefail
DESIGN=$1; ROOT=$2
SALT=$(python3 -c "import json;print(json.load(open('$DESIGN'))['salt'])")
BLOCK=$(python3 -c "import json;print(json.load(open('$DESIGN'))['block'])")
B=$ROOT/$BLOCK; mkdir -p $B
python3 -c "import json
d=json.load(open('$DESIGN'))
for c in d['cells']:
  for s,n in c['pool']: print(c['task'],s,n)" | while read task start n; do
  COIN2_DATA=$B COIN2_LOCK=light5 COIN2_TIMEOUT=3000 scripts/skills/coin2_g3_job.sh -m dojo.simlab.gen_layouts \
    --task $task --start $start --n $n --seed-salt $SALT > $B/gen_${task}_$start.log 2>&1
  echo "gen $task $start rc=$?" >> $B/prep.txt
done
for task in $(python3 -c "import json;print(' '.join(c['task'] for c in json.load(open('$DESIGN'))['cells']))"); do
  COIN2_DATA=$B COIN2_LOCK=light5 COIN2_TIMEOUT=3600 scripts/skills/coin2_g3_job.sh -m dojo.simlab.presettle \
    --task $task > $B/presettle_$task.log 2>&1
  echo "presettle $task rc=$?" >> $B/prep.txt
done
echo DONE >> $B/prep.txt
