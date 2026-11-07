"""Synthetic masks/calibration only: no models, sockets, datasets or live calls."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dojo.perception.cameras import Camera
from dojo.perception.general_filters import (
    FilterConfig, Geometry, enabled, geometries, pair_geometry, plausible, select,
    size_ranges, supported,
)
from dojo.perception.general_prompts import instruction_prompts


def camera(x):
    t = np.eye(4); t[0, 3] = x
    return Camera(str(x), np.array([[100.,0,50],[0,100,50],[0,0,1]]), t, 100, 100)


def proposal(x, prompt='object'):
    m = np.zeros((100,100), np.uint8); m[45:56, x-5:x+6] = 1
    return SimpleNamespace(mask=m, score=.7, prompt=prompt)


def test_instruction_only_no_catalogue_sweep_or_task_argument():
    bank = instruction_prompts('Put the amber widget into a grey container.')
    assert {'object','container','tool','amber widget','grey container'} <= set(bank)
    assert 'bottle' not in bank
    with pytest.raises(TypeError):
        instruction_prompts('anything', task='forbidden')
    with pytest.raises(TypeError):
        instruction_prompts('anything', catalogue={})


def test_bare_compounds_and_deterministic_prompt_count():
    text = 'Put wooden_toy objects into the left basket, then take all pens.'
    assert {'wooden toy','wooden toy objects','left basket','pens'} <= set(instruction_prompts(text))
    assert instruction_prompts(text) == sorted(set(instruction_prompts(text)))
    assert instruction_prompts('') == ['container','object','tool']


def test_triangulation_reprojects_without_semantic_pruning():
    a, b = proposal(50, 'widget'), proposal(40, 'unknown label')
    g = pair_geometry(a, camera(0), b, camera(.2))
    assert g is not None and g[2] > .9
    assert g[0][2] == pytest.approx(2, abs=.05)
    found = geometries({'a':[a], 'b':[b]}, {'a':camera(0),'b':camera(.2)})
    assert found['a'][0] is not None and found['b'][0] is not None
    assert np.allclose(found['a'][0].low, g[0])


def test_no_baseline_or_wrong_epipolar_match_has_no_geometry():
    a = proposal(50); b = proposal(40); b.mask = np.roll(b.mask, 35, axis=0)
    assert pair_geometry(a, camera(0), b, camera(0)) is None
    assert pair_geometry(a, camera(0), b, camera(.2)) is None


def test_support_table_and_another_object_but_not_floating():
    make = lambda z: Geometry(np.array([0,0,z]),np.array([.1,.1,z+.1]),('other',0),1,0)
    base, stacked, floating = make(.765), make(.865), make(1.2)
    assert supported(base, [])
    assert supported(stacked, [base])
    assert not supported(floating, [base])
    assert not supported(None, [base])


def test_size_uses_every_asset_not_roles_or_task_specific_priors():
    index = {'Rigid/widget/00000':{'extents':[.01,.02,.03],'names':['widget']},
             'Rigid/widget/00001':{'extents':[.2,.3,.4],'names':['widget'],'task':'unused','place':['ignored']}}
    ranges = size_ranges(index)
    assert np.allclose(ranges['widget'][1], [.2,.3,.4])
    g = Geometry(np.zeros(3),np.array([.2,.3,.4]),('v',0),1,0)
    assert plausible(g,'red widget',ranges)
    assert plausible(g,'unknown noun',ranges)
    assert not plausible(None,'widget',ranges)
    assert not plausible(Geometry(np.zeros(3),np.ones(3)*10,('v',0),1,0),'widget',ranges)


def test_single_filter_ablations_do_not_require_other_flags():
    a, b = Mock(), Mock()
    flags = [dict(multiview=False,support=False,size=False,agreement=False,score=True),
             dict(multiview=True,support=False,size=True,agreement=True,score=False)]
    assert select([a,b], flags, 0) == [a,b]
    assert select([a,b], flags, 16) == [a]
    assert select([a,b], flags, 1) == [b]
    assert select([a,b], flags, 31) == []
    assert enabled(31) == ('multiview','support','size','agreement','score')
