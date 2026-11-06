"""Geometry gating preserves baseline paths and scopes the fill profile."""
from types import SimpleNamespace
import math

import numpy as np
import pytest

from dojo.skills import library
from dojo.skills.geometry import axis_angle_R


def scene(xy, rolled=False, rigid=True):
    holder = SimpleNamespace(center_w=np.r_[xy, .825], obj_type="Rigid" if rigid else "Geometry")
    shaft = SimpleNamespace(asset=SimpleNamespace(extents=np.array([.18, .018, .020])),
                            R=axis_angle_R(np.array([1., 0, 0]), math.radians(26 if rolled else 0)))
    world = SimpleNamespace(obj=lambda label: {"h": holder, "s": shaft}[label], objects={"h": holder, "s": shaft})
    def base(arm):
        T = np.eye(4)
        T[:3, 3] = [-.3 if arm == "left" else .3, -.45, .765]
        return T
    return SimpleNamespace(refresh=lambda: world, kin=SimpleNamespace(base=base))


@pytest.mark.parametrize("xy,rolled,rigid,active", [
    ((0., -.18), False, True, False),  # overlap: both arms comfortably reach
    ((.06, -.14), False, True, False), # near arm has clearance to spare
    ((.286, -.044), False, True, True),
    ((.342, -.04), False, True, True),
    ((.1, .1), False, True, True),
    ((.234, .004), True, True, False), # all slender shafts rolled
    ((.3, 0.), False, False, False),
])
def test_geometry_gate(xy, rolled, rigid, active):
    sk = scene(xy, rolled, rigid)
    assert library._fill_guard_features(sk, ["s"], "h", .42)["active"] is active


def test_wide_body_does_not_count_as_flat_shaft():
    sk = scene((.3, 0.))
    sk.refresh().objects["s"].asset.extents = np.array([.14, .033, .023])
    assert not library._fill_guard_features(sk, ["s"], "h", .42)["active"]


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_profile_is_scoped_and_restored(monkeypatch, active, fail):
    sk = SimpleNamespace(fill_guard="geometry", log=lambda _: None)
    for key in library.FILL_GUARD_PROFILE:
        setattr(sk, key, "off")
    monkeypatch.setattr(library, "_fill_guard_features", lambda *a: {
        "active": active, "nearest_m": .45, "farthest_m": .7, "flat_shaft": True})
    def inner(*args):
        assert {k: getattr(sk, k) for k in library.FILL_GUARD_PROFILE} == (
            library.FILL_GUARD_PROFILE if active else dict.fromkeys(library.FILL_GUARD_PROFILE, "off"))
        if fail:
            raise RuntimeError("test")
        return {"ok": True}
    monkeypatch.setattr(library, "_fill_with_joint_limits", inner)
    if fail:
        with pytest.raises(RuntimeError): library.fill_container(sk, ["s"], "h")
    else:
        assert library.fill_container(sk, ["s"], "h")["ok"]
    assert all(getattr(sk, k) == "off" for k in library.FILL_GUARD_PROFILE)
    assert sk.last_fill_guard["active"] is active


def test_off_does_not_even_refresh_geometry(monkeypatch):
    sk = SimpleNamespace(fill_guard="off")
    monkeypatch.setattr(library, "_fill_guard_features", lambda *a: pytest.fail("off gate evaluated"))
    monkeypatch.setattr(library, "_fill_with_joint_limits", lambda *a: {"ok": True})
    assert library.fill_container(sk, [], "h")["ok"]


def test_defaults_and_conflicting_axes_rejected():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().fill_guard == "off"
    assert SKILL_ATTRS["fill_guard"] == ("off", "geometry")
    for cfg in [RuntimeConfig(fill_guard="bad"), RuntimeConfig(fill_guard="geometry", fill_drop="inset")]:
        with pytest.raises(ValueError): RuntimePolicy(cfg, assets=None)
