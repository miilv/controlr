"""Tools-articulated stream: hang_mugs helpers without Isaac (which mug is on which hook)."""

import numpy as np

from dojo.skills import library as L
from dojo.skills.geometry import T_from
from dojo.skills.world import AssetInfo, ObjectState


def _mug(pos, quat=(1.0, 0.0, 0.0, 0.0)):
    # the handle frame 45 mm out along the mug's x, 30 mm up from its origin
    info = AssetInfo(key="Rigid/mug/00000", bbox_min=np.array([-0.035, -0.032, 0.0]),
                     bbox_max=np.array([0.056, 0.032, 0.076]),
                     active_functional={"handle": {"frame": [[0.045, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0]]}})
    return ObjectState("mug0", "Rigid", "mug", 0, np.asarray(pos, float), np.asarray(quat, float), info)


def test_on_hook_finds_the_hook_a_mug_hangs_on_and_none_for_one_that_fell():
    hooks = [(0, T_from(np.array([0.0, 0.0, 1.0]), np.eye(3)), 0.05),
             (4, T_from(np.array([0.2, 0.0, 1.0]), np.eye(3)), 0.05)]
    hung = _mug((0.2 - 0.045 + 0.01, 0.0, 1.0 - 0.03 - 0.01))           # handle 14 mm from hook 4
    assert np.allclose(L.handle_point(hung), [0.21, 0.0, 0.99])
    assert L.on_hook(hung, hooks) == 4
    fell = _mug((0.15, -0.1, 0.78))                                      # on the table under the rack
    assert L.on_hook(fell, hooks) is None
    below = _mug((0.2 - 0.045, 0.0, 1.0 - 0.03 - 0.035))                 # handle 35 mm under the hook: slipped off
    assert L.on_hook(below, hooks) is None
