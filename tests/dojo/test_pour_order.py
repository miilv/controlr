"""Stream N: pour_by_language's pairs from the instruction, matched by colour, poured in order through stream P's
ported pour (``dojo.skills.pour_order``). CPU only; the ported skill is replaced by a recorder."""

from __future__ import annotations

import numpy as np

from dojo.skills import library, planner, pour_order
from dojo.skills.world import AssetInfo, ObjectState, WorldState

POUR = ("Pour the liquid from the first red bottle into the first black bowl, from the second turquoise bottle into the "
        "second white bowl, and from the third violet bottle into the third brown bowl. Then reset the robot arm.")
# catalogue mean colours (dojo-perception-data/catalog: wine_bottle 0-2, wine_bowl 0-2)
BOTTLE_RGB = {0: (255, 0, 0), 1: (0, 200, 151), 2: (123, 97, 255)}
BOWL_RGB = {0: (11, 11, 11), 1: (221, 221, 221), 2: (63, 48, 38)}


def _obj(label, cat, idx, xyz, rgb):
    info = AssetInfo(key=f"Rigid/{cat}/{idx:05d}", bbox_min=-np.full(3, 0.04), bbox_max=np.full(3, 0.04),
                     rgb=None if rgb is None else np.asarray(rgb, float))
    return ObjectState(label, "Rigid", cat, idx, np.asarray(xyz, float), np.array([1.0, 0, 0, 0]), info)


def _world(bottle_idx=(2, 0, 1), bowl_idx=(1, 2, 0), routes=("pour_order",)):
    objs = [_obj(f"b{i}", "wine_bottle", k, (-0.2 + 0.2 * i, 0.0, 0.8), BOTTLE_RGB[k]) for i, k in enumerate(bottle_idx)]
    objs += [_obj(f"w{i}", "wine_bowl", k, (-0.15 + 0.15 * i, -0.15, 0.78), BOWL_RGB[k]) for i, k in enumerate(bowl_idx)]
    return WorldState("", "", {o.label: o for o in objs}, {}, {}, notes={"routes": list(routes)} if routes else {})


def test_pairs_follow_the_words_and_the_colours():
    w = _world()                                    # bottles left to right: violet, red, turquoise
    p = planner.plan(POUR.lower(), w)
    assert p.skill == "pour_in_order"
    assert p.args["pairs"] == [["b1", "w2"], ["b2", "w0"], ["b0", "w1"]]          # red->black, turquoise->white, ...
    assert planner.plan(POUR.lower(), _world(routes=())).skill != "pour_in_order"   # off by default


def test_no_route_without_colours():
    w = _world()
    w.objects["b0"].asset.rgb = None
    assert pour_order.route(POUR.lower(), w, ["pour_order"]) is None


def test_route_only_and_runs_in_order_stopping_at_a_failed_pour(monkeypatch):
    assert "pour_in_order" in library.ROUTE_ONLY and "pour_in_order" not in library.SKILLS
    calls = []

    def fake(sk, skill, args, reserve=None):
        calls.append((skill, args["container"], args["target"]))
        return {"ok": len(calls) < 2, "status": "ok" if len(calls) < 2 else "logic", "verified": True, "steps": 10}
    import dojo.skills.ported as ported_mod
    monkeypatch.setattr(ported_mod, "ported", fake)

    class Sk:
        def refresh(self):
            return None

        def log(self, m):
            pass
    r = pour_order.pour_in_order(Sk(), [["b1", "w2"], ["b2", "w0"], ["b0", "w1"]])
    assert not r["ok"] and calls == [("pour_bottle", "@b1", "@w2"), ("pour_bottle", "@b2", "@w0")]
