"""Stream H: ``RuntimeConfig.rehover_siblings`` / ``Skills.rehover_siblings`` ("follow" = today | "keep"): a pick
refine's "seen elsewhere" hint is not followed for an object with an identical sibling within ``SIBLING_NEAR``
(make_toast camera, dev 100000: the hint moved a bread slice 5 mm across its 12 mm thickness; the pinch closed on air).
CPU only."""

from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import skills as S


def _obj(key, c):
    c = np.asarray(c, float)
    return SimpleNamespace(asset=SimpleNamespace(key=key), center_w=c, pos=c.copy(), quat=np.array([1.0, 0, 0, 0]))


def _sk(mode, objs):
    sk = SimpleNamespace(world=SimpleNamespace(objects=objs, obj=lambda l: objs[l]), rehover_siblings=mode,
                         max_rehovers=1, logs=[], refreshed=0)
    sk.log = sk.logs.append
    sk.refresh = lambda: None
    sk._near_sibling = lambda label: S.Skills._near_sibling(sk, label)
    hint = (np.array([-0.126, -0.037, 0.841]), np.array([1.0, 0, 0, 0]))
    sk._refine = lambda label, arm, kind: hint
    return sk


def _rack():
    return {"bread_0": _obj("bread", [-0.106, -0.019, 0.841]), "bread_1": _obj("bread", [-0.131, -0.025, 0.841]),
            "toaster": _obj("toaster", [0.1, -0.1, 0.84])}


def test_keep_ignores_the_hint_beside_a_sibling():
    objs = _rack()
    sk = _sk("keep", objs)
    moved = []
    S.Skills._refine_hover(sk, "bread_1", "left", "pick", rehover=lambda: moved.append(1) or True)
    assert moved == [] and objs["bread_1"].pos == pytest.approx([-0.131, -0.025, 0.841])
    assert any("not followed" in m for m in sk.logs)


def test_follow_and_a_lone_object_still_rehover():
    for mode, label in (("follow", "bread_1"), ("keep", "toaster")):
        objs = _rack()
        sk = _sk(mode, objs)
        moved = []
        S.Skills._refine_hover(sk, label, "left", "pick", rehover=lambda: moved.append(1) or True)
        assert moved == [1] and objs[label].pos == pytest.approx([-0.126, -0.037, 0.841])


def test_default_and_validation():
    assert RuntimeConfig().rehover_siblings == "follow"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", rehover_siblings="never"), assets=None)
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["rehover_siblings"] == ("follow", "keep")
