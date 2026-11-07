"""DIAGNOSTIC ONLY: historical task-scoped mask routing. Never enable for a board candidate.

Task is the instruction-derived vocabulary, never launcher metadata. The board path
must use one general configuration and no per-task prompt banks.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np

from .sam_masks import SamClient, SideServerDown

# Explicit opt-ins; no task enables these merely by being recognized.
RECOMMENDED = {
    'pour_by_language': ('sam', 'sam3'),
    'fill_egg_holder': ('sam', 'sam3.1_text'),
    'classify_objects_by_language': ('union', 'sam3'),
    'imitate_sorting_sequence': ('union', 'sam3'),
    'match_and_pick_from_conveyor': ('sam', 'sam3.1_text'),
    'pick_from_conveyor_by_image': ('sam', 'sam3.1_text'),
}


@lru_cache(maxsize=1)
def banks():
    return json.loads((Path(__file__).parent / 'data/sam_task_banks.json').read_text())


def resolve(routes, task):
    for key, route in routes.items():
        if key not in banks() or route.get('mode', 'v2b') not in ('v2b', 'sam', 'union'):
            raise ValueError(f'invalid SAM route {key}')
        if route.get('model', 'sam3') not in ('sam3', 'sam3.1_text'):
            raise ValueError('SAM route requires an explicit detector model')
        allowed = {'mode', 'model', 'bank', 'threshold', 'dedup_iou', 'roi', 'labels'}
        if set(route) - allowed:
            raise ValueError('unknown SAM route fields')
        if not .05 <= route.get('threshold', banks()[key]['api_min']) <= 1:
            raise ValueError('SAM threshold must be in [0.05, 1]')
        if not 0 <= route.get('dedup_iou', .5) <= 1:
            raise ValueError('SAM dedup_iou must be in [0, 1]')
        roi = route.get('roi')
        if roi is not None and (len(roi) != 4 or not all(0 <= v <= 1 for v in roi)
                                or roi[0] >= roi[2] or roi[1] >= roi[3]):
            raise ValueError('SAM roi requires normalized x0,y0,x1,y1')
        if route.get('mode', 'v2b') != 'v2b' and 'conveyor' in key and roi is None:
            raise ValueError('conveyor SAM requires an explicit normalized roi')
        if not set(route.get('labels', [])).issubset(banks()[key]['catalogue_labels']):
            raise ValueError('SAM labels outside routed task')
        if route.get('bank', key) != key:
            raise ValueError('SAM bank must match routed task')
    return {'mode': 'v2b', **routes.get(task, {})}


@dataclass
class Proposal:
    mask: np.ndarray
    key: str
    score: float
    candidates: tuple
    labels: tuple = ()
    box: tuple = ()
    alts: list = field(default_factory=list)
    source: str = 'sam'


def iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return float((a & b).sum()) / max(1, int((a | b).sum()))


def detect(base, image, groups, task, route):
    """SAM categories restrict candidate fitting; never fabricate an exact asset class."""
    addr = os.environ.get('DOJO_SAM_ADDR')
    if not addr:
        raise SideServerDown('SAM routing requires DOJO_SAM_ADDR')
    bank = banks()[task]
    c = SamClient(addr, loud=True)
    try:
        masks = c.text(image, bank['phrases'], thr=route.get('threshold', bank['api_min']),
                       model=route.get('model', RECOMMENDED.get(task, ('', 'sam3'))[1]))
    finally:
        c.conn.close()
    out = list(base) if route['mode'] == 'union' else []
    for m in sorted(masks, key=lambda m: -m.score):
        cats = bank['mapping'].get(m.prompt, {}).get('categories', [])
        keys = tuple(k for g in groups for k in g['candidates'] if k.split('/')[1] in cats)
        if not keys or not np.any(m.mask):
            continue
        # Conveyor region is normalized RGB coordinates and must be explicitly calibrated.
        # SAMBAKE requested region/competition filters but did not measure their thresholds.
        if 'conveyor' in task:
            roi = route.get('roi')
            if roi is None or len(roi) != 4:
                raise ValueError('conveyor SAM requires an explicit normalized roi')
            ys, xs = np.nonzero(m.mask)
            x, y = (xs.mean() + .5) / image.shape[1], (ys.mean() + .5) / image.shape[0]
            if not (roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]):
                continue
        if any(iou(m.mask, d.mask) > route.get('dedup_iou', .5) for d in out):
            continue
        labels = tuple(route.get('labels', bank['mapping'].get(m.prompt, {}).get('labels', [])))
        out.append(Proposal(m.mask, 'SAM/category/proposal', m.score, keys, labels))
    return out
