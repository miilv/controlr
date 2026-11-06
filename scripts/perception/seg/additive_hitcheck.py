"""DEV ONLY: genuine 49-presence naming check and exact standalone/base-prefix equality.

The naming unit is image-level scene membership, not localisation or visibility. Reuses the
registered archived hitcheck; the estimator never reads its layout/GT files.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/perception/seg'))
from dojo.perception import d0  # noqa: E402
from dojo.perception.seg_twin import _AdditiveSegModel, _SegModel  # noqa: E402
import bake_seg  # noqa: E402


def gt_keys(layout):
    return {f'{typ}/{cat}/{int(o["category_idx"]):05d}'
            for typ, cats in layout.items() if not typ.startswith('_') and isinstance(cats, dict)
            for cat, insts in cats.items() for o in (insts if isinstance(insts, list) else [])
            if isinstance(o, dict) and 'category_idx' in o}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--base-classes', required=True)
    p.add_argument('--extra', required=True)
    p.add_argument('--extra-classes', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    names = lambda path: json.loads(Path(path).read_text())['names']
    standalone = _SegModel(a.base, names(a.base_classes))
    combined = _AdditiveSegModel(_SegModel(a.base, names(a.base_classes)), _SegModel(a.extra, names(a.extra_classes)))
    parity = Counter()

    def check(image, conf, theta):
        req = dict(op='detect', conf=conf, max_det=100)
        base = standalone.handle(req, image)
        off = combined.handle(req, image)
        result = combined.handle({**req, 'additive': True, 'extra_iou': theta, 'extra_conf': .5}, image)
        assert pickle.dumps(off) == pickle.dumps(base), 'disabled additive changed base response'
        assert pickle.dumps(result['dets'][:len(base['dets'])]) == pickle.dumps(base['dets']), 'base prefix changed'
        assert result['sources'][:len(base['dets'])] == ['base'] * len(base['dets'])
        parity['images'] += 1
        parity['base_detections'] += len(base['dets'])
        return result

    home = Path.home()
    for name, genuine in (('w51', True), ('p60', False)):
        bake_seg.bake.SETS[name] = (home / 'dojo-data/w-ready' / name,
            home / 'dojo-perception-data' / ('bg_folds_51' if genuine else 'bg_folds'), None)
    for _, episode in bake_seg.bake.frames(['d0_51', 'd0_ext', 'series', 'w51', 'p60']):
        if not bake_seg.held_out(int(Path(episode).name)):
            continue
        obs = d0.load_obs(Path(episode), 0)
        for image in obs['images'].values():
            check(image, .4, .3)
    parity_gate = dict(parity)
    root = home / 'dojo-data/s-v2d/hitcheck'
    new = set((home / 'dojo-data/s-v2d/new_keep.txt').read_text().split())
    counts = defaultdict(Counter)
    records = []
    for path in sorted(root.glob('*_cam_*.png')):
        layout = root / (path.name.split('_cam_')[0] + '.layout.json')
        if not layout.exists():
            continue
        gt = gt_keys(json.loads(layout.read_text())) & new
        if not gt:
            continue
        image = np.asarray(Image.open(path).convert('RGB'))
        for theta in (.1, .3, .5):
            result = check(image, .25, theta)
            predicted = {d[2] for d in result['dets']}
            for key in sorted(gt):
                status = 'hit' if key in predicted else (
                    'merged' if any(k.split('/')[1] == key.split('/')[1] for k in predicted) else 'missed')
                counts[(str(theta), key)][status] += 1
                counts[(str(theta), '_all')][status] += 1
            records.append(dict(image=path.name, theta=theta, gt=sorted(gt), predicted=sorted(predicted),
                                extra=[d[2:] for d, source in zip(result['dets'], result['sources']) if source == 'extra']))
    output = dict(unit='image-level scene asset presence; no localisation/visibility filtering', extra_conf=.5,
                  parity_gate=parity_gate, parity_total=dict(parity), rows=[dict(theta=t, key=k, **v)
                  for (t, k), v in sorted(counts.items())], records=records)
    a.out.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'records'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
