"""Stream H: ``RuntimeConfig.ported_pour_aim`` / ``Skills.ported_pour_aim`` ("v1" = today | "b10" / "b20": pour_bottle's
``spout_back``, the stream aimed 10 / 20 mm back from the cup's centre against the pour direction). The effect is screened
in G3; here: the planner's args, the default, validation. CPU only."""

from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import planner
from dojo.skills.world import AssetInfo


def _world():
    o = SimpleNamespace(obj_type="Rigid", category="bottle", label="bottle", pos=np.zeros(3), clutter=False)
    c = SimpleNamespace(obj_type="Rigid", category="cup", label="cup", pos=np.ones(3), clutter=False)
    objs = {"bottle": o, "cup": c}
    return SimpleNamespace(objects=objs, task_objects=lambda: list(objs.values()),
                           instruction="Pour the liquid from the bottle into the cup.")


@pytest.mark.parametrize("aim, back", [(None, None), ("v1", None), ("b10", 0.010), ("b20", 0.020)])
def test_pour_bottle_gets_spout_back(aim, back):
    w = _world()
    p = planner.ported_plan(w.instruction, w, opts=({} if aim is None else {"pour_aim": aim}))
    assert p is not None and p.args["skill"] == "pour_bottle"
    assert p.args["args"].get("spout_back") == back


class _Assets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.015), bbox_max=np.full(3, 0.015))


def test_default_and_validation():
    assert RuntimeConfig().ported_pour_aim == "v1"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", ported_pour_aim="b30"), assets=_Assets())
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["ported_pour_aim"][0] == RuntimeConfig().ported_pour_aim
