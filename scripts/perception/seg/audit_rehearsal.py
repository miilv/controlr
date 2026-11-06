"""Audit an export before incremental training; counts unique images, never RFS repeats.

Exit 1 when any old class lacks training support. A manifest-only audit does not
establish that deleted images are available for retraining.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit(base: list[str], names: list[str], frames: list[dict], val_ids=('100049',)):
    if not base or names[:len(base)] != base or len(set(names)) != len(names):
        raise ValueError('class names must preserve the exact nonempty base prefix without duplicates')
    counts = {split: Counter() for split in ('train', 'val')}
    seen = set()
    vocab = set(names)
    for f in frames:
        # Only new training draws; official stored dev layouts are not rehearsal.
        layout = f['layout'].split('_v')
        if len(layout) != 2 or not layout[0].isdigit() or not layout[1].isdigit():
            raise ValueError(f"not a variant training draw: {f['layout']}")
        if not 100030 <= int(layout[0]) <= 100049 or int(layout[1]) < 1:
            raise ValueError(f"outside training draws: {f['layout']}")
        if f['img'] in seen:
            raise ValueError(f"duplicate image: {f['img']}")
        seen.add(f['img'])
        keys = set(f['keys'])
        if keys - vocab:
            raise ValueError('frame labels outside class vocabulary')
        counts['val' if layout[0] in val_ids else 'train'].update(keys)
    if not frames:
        raise ValueError('empty rehearsal')
    rows = {k: {'old': i < len(base), 'train_images': counts['train'][k], 'val_images': counts['val'][k]}
            for i, k in enumerate(names)}
    missing = [k for k in base if counts['train'][k] == 0]
    return {'images': len(seen), 'old_classes': len(base), 'missing_old': missing,
            'all_old_supported': not missing, 'classes': rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-classes', required=True, type=Path)
    ap.add_argument('--export', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    paths = {'base': a.base_classes, 'classes': a.export / 'classes.json', 'frames': a.export / 'frames.jsonl'}
    raw = {k: p.read_bytes() for k, p in paths.items()}
    classes = json.loads(raw['classes'])
    result = audit(json.loads(raw['base'])['names'], classes['names'],
                   [json.loads(s) for s in raw['frames'].splitlines() if s.strip()], classes['val_ids'])
    result['sha256'] = {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}
    result['kind'] = 'manifest support audit; image availability not checked'
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(f"{len(result['missing_old'])}/{result['old_classes']} old classes lack training images")
    raise SystemExit(0 if result['all_old_supported'] else 1)


if __name__ == '__main__':
    main()
