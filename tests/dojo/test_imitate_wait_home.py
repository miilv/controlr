"""imitate_wait "home" (default "still"): copy_demonstration holds until the support robot reads back at its home (head
view appearance at eval, GT joints in G3) after its arm reads still; a motion read alone called imitate's demonstrator
still while it returned out of the head view (dev 100002: ours moved at 664, RoboDojo's query ended the episode at 669)."""
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.skills import interactive as I


class _Ex:
    def __init__(self, left=1000):
        self.held, self._left = 0, left

    def hold(self, n):
        self.held += n
        self._left -= n

    @property
    def steps_left(self):
        return self._left


def _sk(reads, left=1000, mode="home"):
    it = iter(reads)
    prov = SimpleNamespace(support_at_home=lambda: next(it), actors_moving=lambda: False)
    return SimpleNamespace(provider=prov, ex=_Ex(left), imitate_wait=mode)


def test_wait_support_home_holds_until_home():
    sk = _sk([False, False, False, True])
    assert I.wait_support_home(sk) == f"home after {3 * I.HOME_EVERY}" and sk.ex.held == 3 * I.HOME_EVERY
    sk = _sk([True])
    assert I.wait_support_home(sk) == "home after 0" and sk.ex.held == 0


def test_wait_support_home_unknown_proceeds_and_budget_is_kept():
    sk = _sk([None])
    assert I.wait_support_home(sk) == "home unknown" and sk.ex.held == 0
    sk = _sk([False] * 1000, left=I.COPY_STEPS_PER_OBJECT + 170)
    r = I.wait_support_home(sk)
    assert r.startswith("not home after") and "budget kept" in r
    sk = _sk([False] * 1000, left=100000)
    assert I.wait_support_home(sk) == f"not home after {I.HOME_MAX_WAIT}"


def test_wait_demo_done_home_mode_adds_the_home_wait_only_in_home_mode():
    sk = _sk([False, True])
    r = I.wait_demo_done(sk, after_last=10, n_copy=5)
    assert r.startswith("still after") and r.endswith(f"home after {I.HOME_EVERY}")
    sk = _sk([False, True], mode="origin")
    assert "home" not in I.wait_demo_done(sk, after_last=10, n_copy=5)


def test_gt_provider_support_at_home_compares_with_the_episode_start():
    from dojo.skills.skills import GTProvider
    q = {"franka": np.zeros(9)}
    env = SimpleNamespace(actors_gt=lambda: {k: v.copy() for k, v in q.items()})
    p = GTProvider(env, None)
    assert p.support_at_home() is True
    q["franka"][1] = 0.5                                      # away
    assert p.support_at_home() is False
    q["franka"][1] = 0.01                                     # back within tol
    assert p.support_at_home() is True
    assert GTProvider(SimpleNamespace(), None).support_at_home() is None


def test_support_home_reads_the_robots_appearance(monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import inverse_layout as IL
    ref = np.zeros((6, 8, 3), np.uint8)
    ref[1:3, 2:5] = 240
    m = np.zeros((6, 8), bool)
    m[1:3, 2:5] = True
    monkeypatch.setattr(IL, "_support_models", lambda: (("franka_home", ref, {"cam_head": m}),))
    r = IL.support_home(ref.copy())
    assert r["at_home"] is True and r["name"] == "franka_home" and r["diff"] == 0.0
    away = ref.copy()
    away[m] = 60
    assert IL.support_home(away)["at_home"] is False
    assert IL.support_home(np.zeros((4, 4, 3), np.uint8)) == {"at_home": None, "why": "no support robot model"}


def test_runtime_config_accepts_home_and_defaults_still():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    assert RuntimeConfig().imitate_wait == "still"
    RuntimePolicy(RuntimeConfig(imitate_wait="home"))


def test_keep_sides_reuses_the_first_entrys_demonstrator_half():
    """Re-entry after a re-plan: two of ours moved into basket0 (near our arms) shift the far-half line; with
    imitate_keep_sides the first entry's halves stand (dev 100001: our toy car counted as the demonstrator's)."""
    import inspect
    src = inspect.getsource(I.copy_demonstration)
    i = src.index('if getattr(sk, "imitate_keep_sides", False) in (True, "on")')
    assert 'st.setdefault("demo_side", demo_side)' in src[i:i + 300]
    from dojo.runtime.policy import RuntimeConfig
    assert RuntimeConfig().imitate_keep_sides is False
