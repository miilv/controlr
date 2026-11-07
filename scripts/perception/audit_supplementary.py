"""Re-score copied, frozen write-audit inputs offline; never changes registered verdicts or source reports.

--sources contains episodes/<run>/*_{report,actions}.json.gz and replays/<run>_<id>.json.gz.
--manifest is the original camera verdict (rows) or audit snapshot (episodes).
--writes is its frozen JSONL write ledger, or JSON snapshot with write_rows.
--geometry is a frozen {verts, groups, missing} cache: two bbox extrema per asset suffice for body centres.
Optional --repair-legacy-held reconstructs old show-write priors at the current TCP. Missing exact anchor
frames use commanded anchor FK and are explicitly marked approximate; this is supplementary, not a new gate.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dojo.perception.grasp_metric import Footprints, Groups, _R_quat, pose_T
from dojo.skills.kinematics import X5Kin


def legacy_show_prior(report, write, replay, actions, kin):
    """Return a transported old before pose plus its reconstruction provenance, or None for a hover write."""
    read = next(e for e in report['events'] if e['kind'] == 'held_pose' and e['step'] == write['step'])
    if read['view'] != 'show':
        return None
    arm = read['arm']
    anchor = max(e['step'] for e in report['events'] if e['kind'] == 'held_pose' and e['arm'] == arm
                 and e.get('view') != 'show' and e['step'] < write['step'])
    def tcp(frame):
        p = frame['flange'][arm]
        return kin.tcp_from_flange(pose_T(p[:3], p[3:]))
    exact = str(anchor) in replay['frames']
    old_tcp = (tcp(replay['frames'][str(anchor)]) if exact
               else kin.tcp(arm, actions[anchor - 1][arm + '_arm_joint_state']))
    current_tcp = tcp(replay['frames'][str(write['step'])])
    before = current_tcp @ np.linalg.inv(old_tcp) @ pose_T(write['before'], write['q_before'])
    return list(before[:3, 3]) + list(_R_quat(before[:3, :3])), {
        'step': write['step'], 'label': write['label'], 'anchor_step': anchor,
        'source': 'exact_flange' if exact else 'commanded_anchor_fk',
        'before_sync7': write['before'] + write['q_before']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'manifest', 'writes', 'geometry', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--repair-legacy-held', action='store_true')
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text())
    episodes = manifest.get('rows', manifest.get('episodes'))
    writes = ([json.loads(l) for l in a.writes.read_text().splitlines() if l.strip()]
              if a.writes.suffix == '.jsonl' else json.loads(a.writes.read_text())['write_rows'])
    geometry = json.loads(a.geometry.read_text())
    if geometry['missing']:
        raise ValueError('missing body geometry; refusing a root-position substitute')
    fp = Footprints(loader=lambda key: np.asarray(geometry['verts'][key]))
    def group(task):
        g = Groups.__new__(Groups)
        g.lab2g, g.series = geometry['groups'][task]['lab2g'], geometry['groups'][task]['series']
        return g
    spec = importlib.util.spec_from_file_location('m95_score', Path(__file__).with_name('m95_score.py'))
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    by_run = defaultdict(list)
    for w in writes:
        by_run[w['run']].append(w)
    inputs, replays = a.out / 'inputs', a.out / 'replays'
    inputs.mkdir(parents=True, exist_ok=True)
    replays.mkdir(exist_ok=True)
    repairs, hashes = [], {}
    kin = X5Kin()
    for ep in episodes:
        name, lid = ep['name'], ep['layout_id']
        if not scorer.on_slice(lid) or not name.startswith(('c4b_', 'c6b_')):
            raise ValueError(f'not an allowed saved dev arm: {name}')
        directory = a.sources / 'episodes' / name
        def one(pattern):
            paths = list(directory.glob(pattern))
            if len(paths) != 1:
                raise ValueError(f'expected one {pattern} in {directory}')
            return paths[0]
        report_path, actions_path = one('*_report.json.gz'), one('*_actions.json.gz')
        replay_path = a.sources / 'replays' / f'{name}_{lid}.json.gz'
        for path in (report_path, actions_path, replay_path):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = json.load(gzip.open(report_path, 'rt'))
        replay = json.load(gzip.open(replay_path, 'rt'))
        actions = json.load(gzip.open(actions_path, 'rt'))['actions']
        events = [e for e in report['events'] if e['kind'] not in ('belief_write', 'watch_lay', 'watch_relocate')]
        for w in by_run[name]:
            event = {'kind': 'belief_write', 'writer': w['writer'], 'label': w['label'], 'step': w['step'],
                     'before7': w['before'] + w['q_before'], 'after7': w['after'] + w['q_after']}
            if a.repair_legacy_held and w['writer'] == 'held_pose':
                repaired = legacy_show_prior(report, w, replay, actions, kin)
                if repaired is not None:
                    event['before7'], provenance = repaired
                    repairs.append({'run': name, 'reconstructed_before7': event['before7'], **provenance})
            events.append(event)
        normalized = inputs / name
        normalized.mkdir(exist_ok=True)
        with gzip.open(normalized / f'episode_{lid}_report.json.gz', 'wt') as f:
            json.dump({**report, 'events': events}, f)
        (normalized / 'episode_episodes.jsonl').write_text(json.dumps({**ep, 'mode': 'perception'}) + '\n')
        for dest, source in ((normalized / f'episode_{lid}_actions.json.gz', actions_path),
                             (replays / replay_path.name, replay_path)):
            if not dest.exists():
                dest.symlink_to(source.resolve())
    scorer.cmd_audit(SimpleNamespace(runs=[str(inputs / ep['name']) for ep in episodes], replays=str(replays),
                                   out=str(a.out / 'audit'), back=0, harm_mm=2., by='task'),
                     footprints=fp, group_factory=group)
    hashes['manifest'] = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    hashes['write_ledger'] = hashlib.sha256(a.writes.read_bytes()).hexdigest()
    hashes['body_geometry'] = hashlib.sha256(a.geometry.read_bytes()).hexdigest()
    (a.out / 'provenance.json').write_text(json.dumps({
        'supplementary_only': True, 'registered_verdict_changed': False, 'hashes': hashes,
        'legacy_transport_repairs': repairs}, indent=2) + '\n')


if __name__ == '__main__':
    main()
