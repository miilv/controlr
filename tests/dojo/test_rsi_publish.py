"""A published bar cannot be replaced, imputed or taken from another slice."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('rsi_publish', Path(__file__).parents[2] / 'baselines/rsi/publish.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def payload():
    cells = {task: dict(n=0, successes=0, official_score=None, episodes=[]) for task, _, _ in P.cells()}
    return dict(schema='rsi-yardstick-results/1', expected_per_cell=5, per_cell=cells)


def completed():
    p = payload()
    p['per_cell']['deposit_coin'] = dict(n=5, successes=4, official_score=84,
        episodes=[dict(layout_id=100000 + i, position=i) for i in range(5)])
    return p


def test_append_only_completed_bars():
    old = completed()
    assert P.appended(payload(), old) == {'deposit_coin'}
    assert P.appended(old, copy.deepcopy(old)) == set()
    changed = copy.deepcopy(old)
    changed['per_cell']['deposit_coin']['official_score'] = 90
    with pytest.raises(ValueError, match='published cell changed'):
        P.appended(old, changed)
    with pytest.raises(ValueError, match='published cell changed'):
        P.appended(old, payload())


@pytest.mark.parametrize('layout_id', [100010, 100030, 100100])
def test_refuses_other_slices(layout_id):
    p = completed()
    p['per_cell']['deposit_coin']['episodes'][-1]['layout_id'] = layout_id
    with pytest.raises(ValueError, match='off-slice'):
        P.validate(p)


def test_refuses_short_or_imputed_cell():
    p = completed()
    p['per_cell']['deposit_coin']['n'] = 4
    with pytest.raises(ValueError, match='five-episode'):
        P.validate(p)
    p = payload()
    p['per_cell']['deposit_coin']['official_score'] = 0
    with pytest.raises(ValueError, match='null'):
        P.validate(p)
