"""Cut archived frame-0 bakes to dev ids and compare candidates without GPU/GT access.

python scripts/perception/seg/dev_gate.py cut SOURCE OUT
python scripts/perception/seg/dev_gate.py compare BASE CANDIDATE [--out report.md]

The cut reuses archived predictions; it is not a fresh inference run. Measurement
rows are discarded before validation, aggregation or comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def dev_id(value) -> bool:
    s = str(value)
    return s.isdigit() and (100000 <= int(s) <= 100004 or 100030 <= int(s) <= 100049)


def frame_key(r):
    return (r['set'], r['task'], str(r['episode']), r['frame'])


def object_key(r):
    # Repeated labels are legal (interchangeable objects); position disambiguates them.
    return (*frame_key(r), r['label'], tuple(r['gt_pos']))


def load(root, *, allow_errors=False):
    data = []
    for name in ('objects', 'frames'):
        rows = [json.loads(s) for s in (Path(root) / f'{name}.jsonl').read_text().splitlines() if s.strip()]
        if not rows or any(not dev_id(r['episode']) or r['frame'] != 0 for r in rows):
            raise ValueError(f'{root}/{name}: empty or outside dev frame-0')
        key = object_key if name == 'objects' else frame_key
        if len({key(r) for r in rows}) != len(rows):
            raise ValueError(f'{root}/{name}: duplicate keys')
        data.append(rows)
    objects, frames = data
    if {frame_key(r) for r in objects} - {frame_key(r) for r in frames}:
        raise ValueError('objects without frames')
    if not allow_errors and any(r.get('error') for r in frames):
        raise ValueError('frame errors: gate is invalid')
    return objects, frames


def cut(source: Path, out: Path):
    # Refuse to overwrite an existing gate (or the source).
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'source': str(source.resolve()), 'kind': 'archived prediction dev cut', 'files': {}}
    for name in ('objects', 'frames'):
        raw = (source / f'{name}.jsonl').read_bytes()
        rows = [json.loads(s) for s in raw.splitlines() if s.strip()]
        kept = [r for r in rows if dev_id(r['episode']) and r['frame'] == 0]
        encoded = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in kept).encode()
        (out / f'{name}.jsonl').write_bytes(encoded)
        manifest['files'][name] = {'source_sha256': hashlib.sha256(raw).hexdigest(),
                                   'cut_sha256': hashlib.sha256(encoded).hexdigest(),
                                   'kept': len(kept), 'discarded': len(rows) - len(kept)}
    if (source / 'config.json').exists():
        manifest['source_config'] = json.loads((source / 'config.json').read_text())
    load(out, allow_errors=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def metrics(rows):
    found = sum(bool(r['found']) for r in rows)
    identity = sum(bool(r['found'] and r.get('id_ok')) for r in rows)
    pose10 = sum(bool(r['found'] and r.get('t_err_mm') is not None and r['t_err_mm'] <= 10) for r in rows)
    id10 = sum(bool(r['found'] and r.get('id_ok') and r.get('t_err_mm') is not None
                    and r['t_err_mm'] <= 10) for r in rows)
    return found, identity, pose10, id10


def compare(base, candidate, *, diagnostic=False):
    a, fa = load(base, allow_errors=diagnostic)
    b, fb = load(candidate, allow_errors=diagnostic)
    if {object_key(r) for r in a} != {object_key(r) for r in b} or {frame_key(r) for r in fa} != {frame_key(r) for r in fb}:
        raise ValueError('unpaired gate: object/frame keys differ')
    groups = defaultdict(lambda: [[], []])
    for arm, rows in enumerate((a, b)):
        for r in rows:
            groups[(r['set'], r['task'], r['category'])][arm].append(r)
    lines = ['| set/task/category | n | found A → B | identity A → B | pose10 A → B | identity+10 A → B | gate |',
             '|---|---:|---:|---:|---:|---:|---|']
    losers = 0
    for k, (ra, rb) in sorted(groups.items()):
        ma, mb = metrics(ra), metrics(rb)
        loss = any(y < x for x, y in zip(ma, mb))
        losers += loss
        cells = ' | '.join(f'{x} → {y}' for x, y in zip(ma, mb))
        lines.append(f"| {'/'.join(k)} | {len(ra)} | {cells} | {'LOSS' if loss else ''} |")
    lines.append(f"| **all** | {len(a)} | " + ' | '.join(f'{x} → {y}' for x, y in zip(metrics(a), metrics(b))) + f' | {losers} class losers |')
    lines.append(f"\nFrames: {len(fa)}; phantoms A → B: {sum(r.get('phantoms') or 0 for r in fa)} → {sum(r.get('phantoms') or 0 for r in fb)}.")
    lines.append('\nNew-class hitcheck is a separate required gate; this table alone cannot qualify a candidate.')
    errors = [sum(bool(r.get('error')) for r in fs) for fs in (fa, fb)]
    if any(errors):
        lines.insert(0, f'**INVALID GATE: {errors[0]} / {errors[1]} errored frames in A / B. Diagnostic counts only.**\n')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('cut')
    p.add_argument('source', type=Path)
    p.add_argument('out', type=Path)
    p = sub.add_parser('compare')
    p.add_argument('base', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--diagnostic', action='store_true', help='report errors explicitly; never a qualifying gate')
    a = ap.parse_args()
    if a.cmd == 'cut':
        cut(a.source, a.out)
    else:
        report = compare(a.base, a.candidate, diagnostic=a.diagnostic)
        if a.out:
            a.out.write_text(report)
        else:
            print(report, end='')


if __name__ == '__main__':
    main()
