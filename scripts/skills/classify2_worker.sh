#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for task in classify_objects classify_objects_by_language; do
  scripts/simlab/rdpy.sh -m dojo.simlab.gen_layouts --task "$task" --start 100000 --n 3 \
    --seed-salt classify2-20261107-native-diagnostic-01
  scripts/simlab/rdpy.sh -m dojo.simlab.presettle --task "$task" --ids 100000 100001 100002 \
    > "$DOJO_DATA/${task}_settle.log" 2>&1
  for arm in A B C D; do
    opts=(--skill-attr ported_routes=on --skill-attr ported_classify=p2)
    if [[ "$arm" != A ]]; then opts+=(--skill-attr classify_route=native); fi
    if [[ "$arm" == C || "$arm" == D ]]; then opts+=(--skill-attr classify_carry=fixed); fi
    if [[ "$arm" == D ]]; then opts+=(--skill-attr classify_repair=budget); fi
    scripts/simlab/rdpy.sh -m dojo.skills.g3 run --task "$task" --seeds 100000-100002 \
      --trace --out "$DOJO_DATA/$arm/$task" "${opts[@]}" > "$DOJO_DATA/${task}_${arm}.log" 2>&1
  done
done
