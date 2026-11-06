"""Apply the registered additive gate to complete paired dev bakes (no model/GT access)."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from dev_gate import compare, load, metrics


def groups(rows, by_asset=False):
    out = defaultdict(list)
    for row in rows:
        key = (row['set'], row['task'], row['category'])
        if by_asset:
            key += (row['gt_idx'],)
        out[key].append(row)
    return out


def phantom_groups(frames):
    out = defaultdict(int)
    for row in frames:
        out[(row['set'], row['task'])] += row.get('phantoms') or 0
    return out


def evaluate(base, candidate, recipe, naming, old_classes):
    # The shared comparator rejects unpaired keys and changed GT asset identities.
    compare(base, candidate, by_asset=True)
    a, fa = load(base)
    b, fb = load(candidate)
    gate = recipe['gate']
    if len(a) != gate['objects'] or len(fa) != gate['frames']:
        raise ValueError('gate cardinality differs from registration')
    old = {(k.split('/')[1], int(k.split('/')[2])) for k in old_classes}
    old_a = [r for r in a if (r['category'], r['gt_idx']) in old]
    old_b = [r for r in b if (r['category'], r['gt_idx']) in old]
    reports = {}
    for tag, aa, bb, asset in [('category', a, b, False), ('asset', a, b, True),
                              ('old_category', old_a, old_b, False), ('old_asset', old_a, old_b, True)]:
        ga, gb = groups(aa, asset), groups(bb, asset)
        rows = [dict(key=list(k), n=len(ga[k]), base=list(metrics(ga[k])), candidate=list(metrics(gb[k])))
                for k in sorted(ga)]
        losers = [r for r in rows if any(y < x for x, y in zip(r['base'], r['candidate']))]
        reports[tag] = dict(losers=len(losers), rows=rows)
    pa, pb = phantom_groups(fa), phantom_groups(fb)
    phantoms = [dict(set=k[0], task=k[1], base=pa[k], candidate=pb[k], delta=pb[k]-pa[k]) for k in sorted(pa)]
    total_a, total_b = sum(pa.values()), sum(pb.values())
    phantoms_ok = (total_b <= total_a + gate['phantom_total_increase_max'] and
                   all(r['delta'] <= gate['phantom_increase_per_set_task_max'] for r in phantoms))
    if sum(naming.get(s, 0) for s in ('hit', 'merged', 'missed')) != gate['naming_denominator']:
        raise ValueError('naming denominator differs from registration')
    passed = (all(reports[tag]['losers'] == 0 for tag in ('old_category', 'old_asset')) and phantoms_ok and
              naming.get('hit', 0) >= gate['new_naming_target'])
    return dict(pass_gate=passed, totals=dict(base=list(metrics(a)), candidate=list(metrics(b))),
                naming=naming, phantoms=dict(base=total_a, candidate=total_b, pass_gate=phantoms_ok, rows=phantoms),
                **reports)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--base-classes', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    recipe = json.loads(a.recipe.read_text())
    hit = json.loads((a.root / 'hitcheck.json').read_text())
    old = json.loads(a.base_classes.read_text())['names']
    result = dict(recipe=recipe, parity=hit['parity_gate'], thresholds={})
    for theta in recipe['merge']['mask_iou_grid']:
        naming = next(r for r in hit['rows'] if float(r['theta']) == theta and r['key'] == '_all')
        naming = {k: naming.get(k, 0) for k in ('hit', 'merged', 'missed')}
        result['thresholds'][str(theta)] = evaluate(a.baseline, a.root / 'bake' / f'add-{theta}', recipe, naming, old)
    qualified = [(v['naming']['hit'], -float(t), t) for t, v in result['thresholds'].items() if v['pass_gate']]
    result['selected_theta'] = max(qualified)[2] if qualified else None
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({t: {k: v[k] for k in ('pass_gate', 'totals', 'naming')} for t, v in result['thresholds'].items()}, indent=2))
    print('selected_theta', result['selected_theta'])


if __name__ == '__main__':
    main()
