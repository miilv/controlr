"""Derive a matched no-rehearsal control from an already labelled full export.

Keep the v2d random-task images plus every image containing an appended class.
Only old-class-only rehearsal images differ between the two arms. Labels are
symlinked from the full export; vocabulary and train/validation split are exact.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_yolo import frame_stem  # noqa: E402

TASKS = {'hang_mugs_random', 'sort_nesting_dolls_by_size_random', 'make_toast_random',
         'store_laptop_and_headphones_random', 'pour_liquid_into_cup_random'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full', type=Path, required=True)
    p.add_argument('--base-classes', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    base = set(json.loads(a.base_classes.read_text())['names'])
    classes = json.loads((a.full / 'classes.json').read_text())
    raw = (a.full / 'frames.jsonl').read_bytes()
    frames = [json.loads(s) for s in raw.splitlines() if s.strip()]
    kept = [f for f in frames if f['task'] in TASKS or set(f['keys']) - base]
    if not kept or len(kept) == len(frames):
        raise ValueError('control must be nonempty and omit some old-only rehearsal')
    a.out.mkdir(parents=True, exist_ok=False)
    val = set(classes['val_ids'])
    splits = Counter()
    counts = Counter()
    for f in kept:
        split = 'val' if f['layout'].split('_')[0] in val else 'train'
        splits[split] += 1
        counts.update(f['keys'])
        stem = frame_stem(f)
        for kind, suffix in (('images', Path(f['img']).suffix), ('labels', '.txt')):
            dest = a.out / kind / split / (stem + suffix)
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = a.full / kind / split / (stem + suffix)
            if not source.exists():
                raise ValueError(f'missing full-export file: {source}')
            dest.symlink_to(source.resolve())
    train = [f for f in kept if f['layout'].split('_')[0] not in val]
    keep_names = {frame_stem(f) + Path(f['img']).suffix for f in train}
    # Inherit full-export repetitions: shared images have identical exposure in
    # both arms, rather than recomputing RFS after removing the rehearsal.
    lines = [str(a.out / 'images/train' / Path(s).name)
             for s in (a.full / 'train_rfs.txt').read_text().splitlines()
             if Path(s).name in keep_names]
    (a.out / 'train_rfs.txt').write_text('\n'.join(lines) + '\n')
    classes.update(counts={k: counts[k] for k in classes['names']}, frames=len(kept), splits=dict(splits),
                   sources=dict(Counter(f['source'] for f in kept)))
    (a.out / 'classes.json').write_text(json.dumps(classes, indent=2) + '\n')
    (a.out / 'frames.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in kept))
    yaml = [f'path: {a.out}', 'train: train_rfs.txt', 'val: images/val', 'names:']
    yaml.extend(f"  {i}: '{k}'" for i, k in enumerate(classes['names']))
    (a.out / 'data.yaml').write_text('\n'.join(yaml) + '\n')
    (a.out / 'selection.json').write_text(json.dumps({'full_frames_sha256': hashlib.sha256(raw).hexdigest(),
        'full_images': len(frames), 'kept_images': len(kept), 'rule': __doc__,
        'rfs': 'inherited exact full-export repetitions for retained images'}, indent=2) + '\n')
    print(f'control {len(kept)}/{len(frames)} images; added-class images shared exactly')


if __name__ == '__main__':
    main()
