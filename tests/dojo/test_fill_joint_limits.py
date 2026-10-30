"""Stream H: ``RuntimeConfig.fill_joint_limits`` / ``Skills.fill_joint_limits`` ("inherit" = today | "measured"):
``fill_container`` runs under the measured joint range (joint 5 inside the sim's pi/2 stop) and restores the episode's
range after (fill_pen_holder: drops planned past the stop let go beside the holder). CPU only."""

from types import SimpleNamespace

import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import library


def _run(monkeypatch, mode, raise_=False):
    seen = []

    def inner(sk, objects, container, drop, comfy, tries):
        seen.append(sk.joint_limits)
        if raise_:
            raise RuntimeError("boom")
        return {"ok": True}
    monkeypatch.setattr(library, "_fill_container", inner)
    sk = SimpleNamespace(joint_limits="wide", fill_joint_limits=mode)
    if raise_:
        with pytest.raises(RuntimeError):
            library.fill_container(sk, ["target0"], "pen_holder")
    else:
        assert library.fill_container(sk, ["target0"], "pen_holder") == {"ok": True}
    return seen, sk.joint_limits


def test_measured_only_while_filling(monkeypatch):
    assert _run(monkeypatch, "measured") == (["measured"], "wide")
    assert _run(monkeypatch, "measured", raise_=True) == (["measured"], "wide")


def test_inherit_leaves_the_range_alone(monkeypatch):
    assert _run(monkeypatch, "inherit") == (["wide"], "wide")


def test_default_and_validation():
    assert RuntimeConfig().fill_joint_limits == "inherit"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", fill_joint_limits="wide"), assets=None)
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["fill_joint_limits"] == ("inherit", "measured")


def test_fill_lean_default_and_validation():
    assert RuntimeConfig().fill_lean == "mixed"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", fill_lean="upright"), assets=None)
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["fill_lean"] == ("mixed", "fallback")
