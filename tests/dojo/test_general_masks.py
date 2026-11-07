"""The general proposer cannot see task identity, task config, priors or checker values."""
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dojo.perception import general_masks as gm
from dojo.perception.sam_masks import SamMask


@pytest.fixture
def catalog():
    return {'Rigid/wine_bottle/00000': {'names': ['bottle']},
            'Rigid/egg/00000': {'names': ['white egg']},
            'Rigid/pen/00000': {'names': ['pen']},
            'Rigid/wine_bowl/00000': {'names': ['bowl']},
            'Geometry/plate/00000': {'names': ['bowl']}}


def test_every_instruction_gets_the_whole_shared_catalog(catalog):
    for instruction in ['Pick the pen.', 'Pour from the red bottle.', 'Put the eggs away.', '']:
        bank = gm.prompts(instruction, catalog)
        assert set(gm.catalogue_labels(catalog)) <= set(bank)
        assert {'egg', 'white egg', 'pen', 'wine bottle', 'wine bowl'} <= set(bank)


def test_instruction_nouns_and_modifiers_without_a_task_bank(catalog):
    bank = gm.prompts('Put the turquoise widget into the first red bottle.', catalog)
    assert 'turquoise widget' in bank
    assert 'red bottle' in bank
    assert not any('first' in p for p in bank)


def test_priors_roles_and_checker_fields_are_ignored(catalog):
    expected = gm.prompts('Pick up the white egg.', catalog)
    poisoned = {k: {**v, 'xlim': [1, 9], 'place': ['secret'], 'task': 'unknown', 'score': 100,
                    'labels': ['target0'], 'checker': {'success': True}} for k, v in catalog.items()}
    poisoned['tasks'] = {'make_kong': {'phrases': ['forbidden']}}
    assert gm.prompts('Pick up the white egg.', poisoned) == expected
    with pytest.raises(TypeError):
        gm.prompts('instruction', catalog, task='anything')


def test_ambiguous_categories_are_not_an_exact_identity(catalog):
    assert gm.categories_for('bowl', gm.catalogue_labels(catalog)) == ('plate', 'wine_bowl')
    p = gm.GeneralProposal(np.ones((2, 2)), .9, ('egg',), 'white egg')
    assert not hasattr(p, 'key')


def test_uniform_union_preserves_base_identity_and_keeps_large_masks(catalog):
    cfg = gm.GeneralMaskConfig()
    image = np.zeros((20, 20, 3), np.uint8)
    a = np.zeros((20, 20), np.uint8); a[:3, :3] = 1
    b = 1-a
    base = SimpleNamespace(mask=a, score=.3, key='Rigid/pen/00000')
    v2b, sam = Mock(), Mock()
    v2b.detect.return_value = [base]
    sam.text.return_value = [SamMask(a, .99, 'egg'), SamMask(b, .8, 'bowl')]
    out, audit = gm.propose(image, 'Pick up the white egg.', catalog, v2b, sam, cfg)
    assert out[0] is base and out[1].mask.sum() > .08 * 400
    assert len(out) == 2 and audit['sam'] == 2
    v2b.detect.assert_called_once_with(image, conf=.25)
    assert sam.text.call_args.kwargs == {'thr': .2, 'model': 'sam3'}
    assert asdict(cfg)['dedup_iou'] == .5


def test_deterministic_order_and_no_input_mutation(catalog):
    import copy
    old = copy.deepcopy(catalog)
    a = gm.prompts('Put the eggs in the bowl.', catalog)
    assert a == gm.prompts('Put the eggs in the bowl.', dict(reversed(list(catalog.items()))))
    assert catalog == old


@pytest.mark.parametrize('value', [.01, 1.1])
def test_below_processor_floor_and_invalid_threshold_rejected(value):
    with pytest.raises(ValueError):
        gm.GeneralMaskConfig(sam_threshold=value)
