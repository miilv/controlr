"""Dev-only HANG phase audit from G3 records and GT traces; never a policy input.

Run in RoboDojo's Python with DOJO_CATALOG set:
  .../rdpy.sh scripts/skills/hang_phase_report.py RUN --layouts LAYOUT_DIR
Only the supplied dev files are read. Output retains attempt-level diagnostics.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from dojo.simlab import paths
from dojo.simlab.layout import iter_objects
from dojo.skills.geometry import T_from, inv_T
from dojo.skills.kinematics import X5Kin
from dojo.skills.library import functional_frames, handle_hole, hook_frames, threads_along_x
from dojo.skills.world import AssetDB, objects_from_gt


def annotate_segments(frames, segments):
    """Attach current action names using completed trajectory lengths, not delayed trace labels."""
    cursor = 0
    for seg in segments:
        end = cursor + seg['steps']
        for frame in frames:
            if cursor < frame['step'] <= end:
                frame['current_segment'] = seg['name']
        cursor = end


def audit(root, layouts):
    assets, kin = AssetDB(paths.OBJECTS), X5Kin()
    counts, rows = Counter(), []
    for report in sorted(root.glob('*_episodes.jsonl')):
        for r in map(json.loads, report.read_text().splitlines()):
            seed = int(r['seed'])
            if not (100000 <= seed <= 100004 or 100030 <= seed <= 100099):
                raise ValueError(f'not a tuning id: {seed}')
            layout = json.loads((layouts / r['layout_file']).read_text())
            if paths.training_refusal(seed, layout, allow_variants=True):
                raise ValueError('non-variant training layout')
            counts['episodes'] += 1
            if r.get('category') in ('unstable_layout', 'infra_failure', 'error'):
                counts[r['category']] += 1
                continue
            counts['budget_exhausted'] += int(r['steps'] >= r['step_lim'])
            objects = objects_from_gt([
                {'label': i['label'], 'type': typ, 'category': cat, 'category_idx': i['category_idx'],
                 'pos': i['default_pos'], 'quat': i['default_ori'], 'clutter': i.get('type') == 'cluttered'}
                for typ, cat, i in iter_objects(layout)], assets)
            counts['mug14_instances'] += sum(o.category == 'mug' and o.idx == 14 for o in objects.values())
            for s in r.get('skills', []):
                if s['skill'] == 'pick':
                    counts['pick_attempts'] += 1
                    counts['loose_pick' if s['status'] == 'loose' else
                           'failed_lift' if not s['ok'] else 'good_pick'] += 1
                if s['skill'] == 'hang':
                    counts['hang_ok' if s['ok'] else 'hang_miss'] += 1
            counts['no_plan_exits'] += sum('[hang] no plan for ' in x for x in r.get('log', []))
            trace = root / f"{r['task']}_{seed}_trace.json.gz"
            if not trace.exists():
                continue  # G3 intentionally saves no success trace.
            frames = json.load(gzip.open(trace, 'rt'))['frames']
            # Trace `seg` names the previously completed segment, not the current move.
            # Recover true intervals from trajectory segment lengths instead.
            trajectory = root / f"{r['task']}_{seed}_actions.json.gz"
            segments = json.load(gzip.open(trajectory, 'rt'))['segments']
            annotate_segments(frames, segments)
            hooks = dict((i, (H, tip)) for i, H, tip in hook_frames(objects['cup_holder'], 'handle_support_point'))
            assignments = [re.search(r'\[hang\] (\S+) -> hook (\d+): (left|right)', x)
                           for x in r.get('log', [])]
            for match in filter(None, assignments):
                lab, hi, arm = match.group(1), int(match.group(2)), match.group(3)
                hs = [f for f in frames if f.get('current_segment') == f'hang {lab}']
                closed = [f for f in hs if f['g'][arm] < .5]
                if not closed:
                    continue  # failed lift / put-down, never inserted
                f = closed[-1]
                o = copy.copy(objects[lab])
                pose = f['obj'][lab]
                o.pos, o.quat = np.array(pose[:3]), np.array(pose[3:])
                H, tip = hooks[hi]
                handles = functional_frames(o, 'handle', active=True) or functional_frames(o, 'handle')
                coords = []
                for h in handles[:2]:
                    depth = handle_hole(o, h)
                    if not threads_along_x(o, h, depth):
                        continue
                    p = o.T @ np.r_[h[:3, 3] - h[:3, 2] * depth, 1.]
                    coords.append((inv_T(H) @ p)[:3])
                if not coords:
                    continue
                c = min(coords, key=lambda p: float(np.linalg.norm(p[1:])))
                pick = [p for p in frames if p.get('current_segment', '').startswith(f'pick {lab}') and p['step'] < f['step']]
                shift = None
                if pick:
                    p = pick[-1]
                    a = inv_T(kin.tcp(arm, p['qm'][arm])) @ T_from(p['obj'][lab][:3], quat=p['obj'][lab][3:])
                    b = inv_T(kin.tcp(arm, f['qm'][arm])) @ o.T
                    shift = float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000)
                rows.append({'task': r['task'], 'seed': seed, 'label': lab, 'hook': hi,
                             'pre_release_step': f['step'], 'hole_hook_mm': np.round(c * 1000, 2).tolist(),
                             'hook_tip_mm': round(tip * 1000, 2),
                             'carry_shift_mm': None if shift is None else round(shift, 2),
                             'j5_command_max': max(abs(p['q'][arm][4]) for p in hs)})
    return {'counts': dict(counts), 'pre_release': rows,
            'note': 'Attempt counts overlap; success traces absent. Coordinates/shift are diagnostics, not checker terms.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--layouts', required=True, type=Path)
    a = p.parse_args()
    paths.assert_not_eval(a.run)
    paths.assert_not_eval(a.layouts)
    print(json.dumps(audit(a.run, a.layouts), indent=2))


if __name__ == '__main__':
    main()
