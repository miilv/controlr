"""Freeze fresh salted dev inputs; no policy code is imported or copied.

Run under RoboDojo's Python with this repo on PYTHONPATH. Raw records are
passed through the official client's native reset/stability check. Never
substitute an old layout when generation fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dojo.simlab.paths import is_our_measurement_id

SALT = 'rsi-yardstick-20261107'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_ids(ids: list[int]) -> None:
    if not ids or len(set(ids)) != len(ids) or any(not is_our_measurement_id(i) for i in ids):
        raise ValueError('unique OUR_MEASUREMENT_SLICES ids required')


def generate(root: Path, tasks: list[str], ids: list[int]) -> dict:
    from dojo.simlab.gen_layouts import GENERATOR, generate as draw

    validate_ids(ids)
    root.mkdir(parents=True, exist_ok=False)
    entries = {}
    for task in tasks:
        directory = root / task
        directory.mkdir()
        entries[task] = []
        for position, layout_id in enumerate(sorted(ids)):
            record = draw(task, layout_id, seed_salt=SALT)
            path = directory / f'{layout_id}.json'
            path.write_text(json.dumps(record, indent=1))
            entries[task].append(dict(id=layout_id, position=position,
                                      file=str(path.relative_to(root)), sha256=digest(path)))
    result = dict(schema='rsi-yardstick/1', salt=SALT, generator=GENERATOR,
                  layout_form='raw', tasks=entries)
    path = root / 'manifest.json'
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    (root / 'manifest.sha256').write_text(digest(path) + '  manifest.json\n')
    return result


def verify(root: Path) -> dict:
    manifest = root / 'manifest.json'
    if digest(manifest) != (root / 'manifest.sha256').read_text().split()[0]:
        raise ValueError('manifest hash mismatch')
    result = json.loads(manifest.read_text())
    if result['schema'] != 'rsi-yardstick/1' or result['salt'] != SALT:
        raise ValueError('unexpected manifest provenance')
    for task, entries in result['tasks'].items():
        if not task or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789' for c in task):
            raise ValueError('invalid task name')
        validate_ids([e['id'] for e in entries])
        for position, entry in enumerate(entries):
            expected = f"{task}/{entry['id']}.json"
            if entry['file'] != expected or entry['position'] != position:
                raise ValueError('path or call-order mismatch')
            if digest(root / expected) != entry['sha256']:
                raise ValueError('layout hash mismatch')
            if 'demo_sha256' in entry:
                demo = root.parent / f"simlab/traj/{task}/{entry['id']}.pkl"
                if digest(demo) != entry['demo_sha256']:
                    raise ValueError('demo hash mismatch')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--ids', nargs='+', type=int, default=list(range(100000, 100010)))
    args = parser.parse_args()
    if args.verify:
        verify(args.root)
    else:
        if not args.tasks:
            parser.error('--tasks required for generation')
        generate(args.root, args.tasks, args.ids)
