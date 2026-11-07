#!/usr/bin/env bash
# Execute only an already registered private POUR panel. No generation or selection here.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="$HOME/dojo-data/pour"
REG="$DATA/panel-registration.json"
mkdir -p "$DATA/panel-logs"
export DOJO_CATALOG="${DOJO_CATALOG:-$HOME/dojo-data/skills3/catalog:$HOME/dojo-data/skills2/catalog:$HOME/dojo-perception-data/catalog}"
python3 - "$REG" "$ROOT/.deployed_commit" "$DATA" > "$DATA/panel-order.tsv" <<'PY'
import hashlib, json, pathlib, sys
reg, stamp, root = map(pathlib.Path, sys.argv[1:])
r = json.loads(reg.read_text())
assert r['controller_commit'] == stamp.read_text().strip(), 'source differs from registration'
assert r['registered_utc'] and set(r['arms']) == {'A', 'B'}
assert r['arms']['A'] == {'pace': 'v1', 'aim': 'v1', 'profile': 'off'}
assert set(r['cells']) == {'pour_by_language', 'pour_liquid_into_cup'}
for task, cell in r['cells'].items():
    ids = cell['ids']
    assert len(ids) >= 10 and len(set(ids)) == len(ids)
    assert all(isinstance(i, int) and 100080 <= i < 100100 for i in ids)
    index = json.loads((root / 'simlab/layouts' / task / 'settled_index.json').read_text())
    for j, seed in enumerate(ids):
        assert index[str(seed)]['status'] == 'ok', f'unstable {task}/{seed}'
        for suffix, key in (('', 'raw_sha256'), ('.settled', 'settled_sha256')):
            p = root / 'simlab/layouts' / task / f'{seed}{suffix}.json'
            assert hashlib.sha256(p.read_bytes()).hexdigest() == cell['layouts'][str(seed)][key]
        for arm in (('A', 'B') if j % 2 == 0 else ('B', 'A')):
            a = r['arms'][arm]
            assert a['pace'] in ('v1', 'overlap', 'budget')
            assert a['aim'] in ('v1', 'b10', 'b20') and a['profile'] == 'off'
            print(task, seed, arm, a['pace'], a['aim'], a['profile'], sep='\t')
PY
REG_SHA=$(sha256sum "$REG" | cut -d " " -f1)
while IFS=$'\t' read -r task seed arm pace aim profile; do
 [[ "$(sha256sum "$REG" | cut -d " " -f1)" == "$REG_SHA" ]] || { echo "registration changed" >&2; exit 2; }
 out="$DATA/panel-$arm-$task"
 if python3 - "$out/${task}_episodes.jsonl" "$seed" "$ROOT/.deployed_commit" "$pace" "$aim" "$profile" <<'PY'
import json, pathlib, sys
try:
    p, seed, stamp = pathlib.Path(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3])
    rows = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
    hits = [r for r in rows if r['seed'] == seed]
    assert len(hits) <= 1, 'duplicate episode; do not silently pick one'
    if hits:
        assert hits[0]['code']['commit'] == stamp.read_text().strip(), 'mixed source in panel'
        attrs = hits[0]['skill_attrs']
        assert [attrs[k] for k in ('ported_pour_pace', 'ported_pour_aim', 'ported_pour_profile')] == sys.argv[4:7], 'mixed settings'
        raise SystemExit(0)
    raise SystemExit(1)
except Exception as exc:
    print(f'[pour-panel] invalid existing data: {exc}', file=sys.stderr)
    raise SystemExit(2)
PY
 then
  echo "[pour-panel] already recorded $task $seed $arm"
  continue
 else
  rc=$?
  [[ "$rc" -eq 1 ]] || exit "$rc"
 fi
 # Verify the selected settled record again immediately before each episode.
 python3 - "$REG" "$DATA" "$task" "$seed" <<'PY'
import hashlib, json, pathlib, sys
r=json.loads(pathlib.Path(sys.argv[1]).read_text()); root=pathlib.Path(sys.argv[2]); task=sys.argv[3]; seed=sys.argv[4]
p=root/'simlab/layouts'/task/f'{seed}.settled.json'
assert hashlib.sha256(p.read_bytes()).hexdigest() == r['cells'][task]['layouts'][seed]['settled_sha256']
PY
 echo "[pour-panel] start $task $seed $arm"
 "$ROOT/scripts/skills/pour_g3_job.sh" -m dojo.skills.g3 run --task "$task" --seeds "$seed" \
  --layouts "$DATA/simlab/layouts/$task" --routes pour_order --skill-attr ported_routes=on \
  --skill-attr "ported_pour_pace=$pace" --skill-attr "ported_pour_aim=$aim" \
  --skill-attr "ported_pour_profile=$profile" --trace --out "$out" \
  > "$DATA/panel-logs/$task-$seed-$arm.log" 2>&1
 echo "[pour-panel] done $task $seed $arm"
done < "$DATA/panel-order.tsv"
date -u +%FT%TZ > "$DATA/panel.done"
