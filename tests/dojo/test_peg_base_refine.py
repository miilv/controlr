"""peg_base_refine (default off): stack_on_pegs' base gets ONE local place refine (no radius search, applied at a high
iou even under the noise gate), later base refines are skipped; the skill re-reads the peg axes from the base."""

import numpy as np

from dojo.runtime.policy import PEG_BASE_MAX_SHIFT, RuntimeConfig
from test_runtime import _env, _place_setup


class _Seq:
    """Place refines answering scripted (dx mm, iou); records each call's max_shift and kwargs."""

    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None, **kw):
        o = objects[0]
        self.calls.append({"max_shift": max_shift_m, **kw})
        dx, iou = self.answers.pop(0)
        x = o["pos"][0] + dx / 1000
        return {o["label"]: {"ok": True, "pos": [x, o["pos"][1], o["pos"][2]], "quat": o["quat"], "shift_mm": abs(dx),
                             "iou": iou, "visible_frac": 0.5, "turned_deg": 0.2}}


class _Sk:
    def __init__(self, base):
        self.peg_base = base
        self.refine_ctx = {"kind": "place", "label": base, "arm": "left", "held": None}


def _cfg(mode):
    return RuntimeConfig(mode="perception", refine=True, refine_max_shift_m=0.07, place_confirm=True,
                         refine_search_radius=0.07, peg_base_refine=mode)


def test_local_once_applies_a_small_good_fix_and_skips_later_refines():
    env = _env()
    pc = _Seq([(1.5, 0.97), (60.0, 0.95)])
    pol, prov = _place_setup(_cfg("local"), pc, env)
    x0 = float(prov.belief.objects["block_0"].pos[0])
    out = pol._refine_hook(prov, _Sk("block_0"), "block_0", "left", "place")
    assert "search_radius" not in pc.calls[0] and pc.calls[0]["max_shift"] == PEG_BASE_MAX_SHIFT
    assert "block_0" in out and np.isclose(float(prov.belief.objects["block_0"].pos[0]), x0 + 0.0015)
    assert pol.report["events"][-1]["peg_base_refine"] == "applied"
    assert pol._refine_hook(prov, _Sk("block_0"), "block_0", "left", "place") is None    # measured once
    assert len(pc.calls) == 1 and pol.report["events"][-1]["skipped"].startswith("peg_base_refine")


def test_local_rejects_a_poor_fit_and_gives_up_after_two_places():
    env = _env()
    pc = _Seq([(1.5, 0.80), (1.2, 0.90)])
    pol, prov = _place_setup(_cfg("local"), pc, env)
    x0 = float(prov.belief.objects["block_0"].pos[0])
    for _ in range(2):
        assert pol._refine_hook(prov, _Sk("block_0"), "block_0", "left", "place") is None
    assert float(prov.belief.objects["block_0"].pos[0]) == x0 and len(pc.calls) == 2
    assert pol._refine_hook(prov, _Sk("block_0"), "block_0", "left", "place") is None and len(pc.calls) == 2


def test_off_or_another_object_keeps_today_s_refine():
    env = _env()
    pc = _Seq([(1.5, 0.97)])
    pol, prov = _place_setup(_cfg("off"), pc, env)
    pol._refine_hook(prov, _Sk("block_0"), "block_0", "left", "place")
    assert pc.calls[0]["search_radius"] == 0.07                       # the radius search as before
    assert pol.report["events"][-1].get("place_confirm") == "below_noise"
    pc = _Seq([(1.5, 0.97)])
    pol, prov = _place_setup(_cfg("local"), pc, env)
    pol._refine_hook(prov, _Sk("block_9"), "block_0", "left", "place")   # not the pegs' base: untouched
    assert pc.calls[0]["search_radius"] == 0.07 and "peg_base_refine" not in pol.report["events"][-1]


def test_pegs_skill_marks_its_base_and_rereads_the_axes_only_with_the_flag(monkeypatch):
    from dojo.skills import pegs

    seen = []

    class W:
        def __init__(self, x):
            self.objects = {"base": None, "p": None}
            self._x = x

        def obj(self, lab):
            class B:
                asset = type("A", (), {"mesh": None})()
                pos = np.array([self._x, 0.0, 0.0])

                def yaw(self_):
                    return 0.0
            return B()

    class Sk:
        def __init__(self, flag):
            self.peg_base_refine, self.calls, self.x = flag, 0, 0.0
            self.world = W(0.0)
            self.coll = type("C", (), {"include": set(), "parts": {}, "margin_of": {}})()
            self.held = {"left": None, "right": None}

        def refresh(self):
            self.calls += 1
            self.world = W(0.001 * self.calls)
            return self.world

    monkeypatch.setattr(pegs, "peg_geometry", lambda b, tag: seen.append(round(float(b.pos[0]), 4)) or {"top": 0.8})
    for flag, n_geo in (("off", 1), ("local", 2)):
        seen.clear()
        sk = Sk(flag)
        monkeypatch.setattr(pegs, "on_peg", lambda sk_, lab, peg: (seen.append("on") or True))
        r = pegs.stack_on_pegs(sk, [["p"]], "base", ["pole/0"])
        assert r["ok"] and sk.peg_base is None
        assert len([x for x in seen if x != "on"]) == n_geo           # off: the axes once; local: again before the piece
