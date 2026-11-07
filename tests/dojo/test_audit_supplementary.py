"""Historical repairs explicitly distinguish an exact saved anchor from commanded-FK approximation."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.skills.geometry import T_from

spec = importlib.util.spec_from_file_location('supp', Path(__file__).resolve().parents[2] /
                                              'scripts/perception/audit_supplementary.py')
supp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supp)


@pytest.mark.parametrize('exact', [True, False])
def test_historical_transport_repair_preserves_inputs_and_labels_approximation(exact):
    report = {'events': [{'kind': 'held_pose', 'step': s, 'arm': 'left', 'view': view}
                         for s, view in ((2, 'hover'), (5, 'show'))]}
    write = {'step': 5, 'label': 'ring', 'before': [0, 0, .8], 'q_before': [1, 0, 0, 0]}
    frames = {'5': {'flange': {'left': [.1, 0, .8, 1, 0, 0, 0]}}}
    if exact:
        frames['2'] = {'flange': {'left': [0, 0, .8, 1, 0, 0, 0]}}
    replay = {'frames': frames}
    actions = [{'left_arm_joint_state': [0.] * 6}] * 5
    kin = SimpleNamespace(tcp_from_flange=lambda T: T, tcp=lambda arm, q: T_from([q[0], 0, .8]))
    before = copy.deepcopy((report, write, replay, actions))
    pose, provenance = supp.legacy_show_prior(report, write, replay, actions, kin)
    np.testing.assert_allclose(pose[:3], [.1, 0, .8], atol=1e-12)
    assert provenance['source'] == ('exact_flange' if exact else 'commanded_anchor_fk')
    assert (report, write, replay, actions) == before
