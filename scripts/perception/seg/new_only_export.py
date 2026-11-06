"""Project a full export onto a disjoint new-only vocabulary, retaining old-only images as negatives."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_yolo import frame_stem, repeat_factors  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full', type=Path, required=True)
    p.add_argument('--keep', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    old = json.loads((a.full / 'classes.json').read_text())
    names = sorted(a.keep.read_text().split())
    if len(set(names)) != len(names) or not names:
        raise ValueError('new class list must be nonempty and unique')
    ids = {i: names.index(k) for i, k in enumerate(old['names']) if k in names}
    if len(ids) != len(names):
        raise ValueError('requested class absent from source vocabulary')
    frames = [json.loads(s) for s in (a.full / 'frames.jsonl').read_text().splitlines()]
    for frame in frames:
        seed, sep, variant = frame['layout'].partition('_v')
        if not (seed.isdigit() and 100030 <= int(seed) <= 100049 and sep and
                variant.isdigit() and int(variant) > 0):
            raise ValueError('new-only training accepts training IDs with positive variants only')
    a.out.mkdir(parents=True, exist_ok=False)
    val = set(old['val_ids'])
    counts, splits, negatives = Counter(), Counter(), Counter()
    train = []
    for f in frames:
        split = 'val' if f['layout'].split('_')[0] in val else 'train'
        stem = frame_stem(f)
        for kind in ('images', 'labels'):
            (a.out / kind / split).mkdir(parents=True, exist_ok=True)
        image = stem + Path(f['img']).suffix
        (a.out / 'images' / split / image).symlink_to((a.full / 'images' / split / image).resolve())
        lines = []
        for line in (a.full / 'labels' / split / (stem + '.txt')).read_text().splitlines():
            cls, poly = line.split(' ', 1)
            if int(cls) in ids:
                lines.append(f'{ids[int(cls)]} {poly}')
        (a.out / 'labels' / split / (stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
        f['keys'] = [k for k in f['keys'] if k in names]
        counts.update(f['keys'])
        splits[split] += 1
        negatives[split] += not bool(f['keys'])
        if split == 'train':
            train.append((str(a.out / 'images/train' / image), set(f['keys'])))
    rnd = random.Random(0)
    rfs = []
    for (image, _), factor in zip(train, repeat_factors([ks for _, ks in train], .05)):
        rfs.extend([image] * (int(factor) + int(rnd.random() < factor - int(factor))))
    (a.out / 'train_rfs.txt').write_text('\n'.join(rfs) + '\n')
    meta = dict(names=names, counts=dict(counts), splits=dict(splits), negative_images=dict(negatives),
                val_ids=sorted(val), frames=len(frames), source=str(a.full), rfs_t=.05)
    (a.out / 'classes.json').write_text(json.dumps(meta, indent=2) + '\n')
    (a.out / 'frames.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in frames))
    yaml = [f'path: {a.out}', 'train: train_rfs.txt', 'val: images/val', 'names:']
    yaml.extend(f"  {i}: '{k}'" for i, k in enumerate(names))
    (a.out / 'data.yaml').write_text('\n'.join(yaml) + '\n')
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
