from types import SimpleNamespace

import numpy as np
import pytest

from dojo.skills.egg_reach import practical_limits, screen
from dojo.skills.kinematics import X5Kin
from dojo.runtime.policy import RuntimeConfig, RuntimePolicy


def test_practical_limits_restore_on_exception_and_invalidate_cache():
    kin = X5Kin()
    old = kin.limits
    kin._ik_cache = {'old': 1}
    with pytest.raises(RuntimeError), practical_limits(kin):
        assert kin.limits[3].tolist() == [-1.6, 1.6]
        assert kin.limits[4].tolist() == [-1.56, 1.56]
        assert kin.limits[5].tolist() == [-3., 3.]
        assert '_ik_cache' not in kin.__dict__
        kin._ik_cache = {'scoped': 1}
        raise RuntimeError()
    assert kin.limits is old
    assert '_ik_cache' not in kin.__dict__


def test_screen_checks_all_endpoints_and_keeps_candidate_identity():
    kin = X5Kin()
    calls = []
    def ik(arm, T, q, **kw):
        calls.append(T[2, 3])
        return SimpleNamespace(ok=T[2, 3] != 2., q=np.zeros(6), pos_err=.01)
    kin.ik = ik
    sk = SimpleNamespace(kin=kin, ex=SimpleNamespace(q={'left': np.zeros(6)}), log=lambda _: None)
    def poses(c):
        for z in c:
            T = np.eye(4)
            T[:3, 3] = kin.base('left')[:3, 3]
            T[2, 3] = z
            yield T
    a, b = [0., 2.], [0., 1.]
    assert screen(sk, 'left', [a, b], poses, 'test') == [b]
    assert calls == [0., 2., 0., 1.]


def test_switch_is_default_off_and_validated():
    assert RuntimeConfig().egg_reach == 'off'
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS['egg_reach'] == ('off', 'search')
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(egg_reach='typo'), assets=None)
