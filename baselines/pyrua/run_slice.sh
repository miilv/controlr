#!/usr/bin/env bash
# The M0 10-task slice: 2 tasks per dimension, seed 0, layouts 0..4, one task after another
# (one Isaac at a time on the shared 4090).
#   setsid nohup bash /root/dojo-pyrua/glue/run_slice.sh > /root/dojo-pyrua/slice.log 2>&1 &
set -uo pipefail
TASKS="${TASKS:-stack_blocks put_bottles_into_dustbin build_tower insert_tubes play_tic_tac_toe make_toast swap_blocks cover_blocks general_pickup solve_equation}"
for t in $TASKS; do
  echo "$(date -u +%FT%TZ) start $t"
  bash /root/dojo-pyrua/glue/run_task.sh --task "$t" --seed "${SEED:-0}" --eval-num "${EVAL_NUM:-5}"
  echo "$(date -u +%FT%TZ) end $t rc=$?"
done
