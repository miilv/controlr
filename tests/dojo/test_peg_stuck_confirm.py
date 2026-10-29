"""peg_stuck_confirm (default off): a "rests tilted" judgment needs a second read before the re-pick; a re-pick that
slips is followed by a fresh read of the piece (the runtime provider's recheck)."""

import numpy as np

from dojo.skills.skills import SkillResult


class _Prov:
    def __init__(self, resolve=None):
        self.calls, self.resolve = [], resolve

    def recheck(self, label, arm=None):
        self.calls.append((label, arm))
        if self.resolve is not None:
            self.resolve()
        return {"ok": True, "shift_mm": 4.0}


def _sk(flag, prov):
    class W:
        objects = {"base": None, "p": None}

        def obj(self, lab):
            class B:
                asset = type("A", (), {"mesh": None})()
                pos = np.zeros(3)

                def yaw(self_):
                    return 0.0
            return B()

    class Sk:
        def __init__(self):
            self.peg_stuck_confirm, self.peg_base_refine = flag, "off"
            self.world, self.provider = W(), prov
            self.coll = type("C", (), {"include": set(), "parts": {}, "margin_of": {}})()
            self.held = {"left": None, "right": None}
            self.records, self.logs = [], []

        def refresh(self):
            return self.world

        def bank(self):
            return None

        def _record(self, r):
            self.records.append(r)
            return r

        def log(self, m):
            self.logs.append(m)
    return Sk()


def _run(monkeypatch, flag, stuck_states, statuses, resolve_on_recheck=False):
    from dojo.skills import pegs
    state = {"stuck": list(stuck_states), "n": 0}
    prov = _Prov(resolve=(lambda: state.update(stuck=[False] * 5)) if resolve_on_recheck else None)
    sk = _sk(flag, prov)
    picks = []

    def one_piece(sk_, lab, b, base, peg, on_this, seat, thread, drop):
        st = statuses[len(picks)]
        picks.append(st)
        return SkillResult(st == "ok", "place", st, 10, "left")

    monkeypatch.setattr(pegs, "peg_geometry", lambda b, tag: {"top": 0.8})
    monkeypatch.setattr(pegs, "_one_piece", one_piece)
    monkeypatch.setattr(pegs, "on_peg", lambda sk_, lab, peg: False)
    monkeypatch.setattr(pegs, "stuck", lambda sk_, lab, peg, on_this: state["stuck"].pop(0) if state["stuck"] else True)
    pegs.stack_on_pegs(sk, [["p"]], "base", ["pole/0"])
    return sk, prov, picks


def test_off_re_picks_on_one_stuck_judgment(monkeypatch):
    sk, prov, picks = _run(monkeypatch, "off", [True, True], ["ok", "grasp_slip"])
    assert len(picks) == 2 and prov.calls == []                       # today: once more, no re-read


def test_on_needs_a_confirming_read_before_the_re_pick(monkeypatch):
    sk, prov, picks = _run(monkeypatch, "on", [True], ["ok", "grasp_slip"], resolve_on_recheck=True)
    assert len(picks) == 1 and prov.calls == [("p", "left")]             # the second read: not stuck -> no re-pick
    assert any("does not confirm" in m for m in sk.logs)
    sk, prov, picks = _run(monkeypatch, "on", [True, True, True], ["ok", "grasp_slip"])
    assert len(picks) == 2                                               # confirmed: the re-pick runs
    assert prov.calls == [("p", "left"), ("p", "left")]                  # ... and after its slip, a fresh read


def test_runtime_provider_recheck_runs_a_post_place_read_from_the_given_arm():
    from dojo.runtime.belief import RuntimeProvider
    from dojo.runtime.bridge import Bridge
    from dojo.runtime.estimate import objects_from_gt_rows
    from test_runtime import STACK, FakeAssets, FakeEpisode, _env

    class Checker:
        def __init__(self):
            self.calls = []

        def refine(self, obs, o, others, arm, held):
            self.calls.append((o.label, arm))
            return {"ok": True, "pos": [float(o.pos[0]) + 0.004, float(o.pos[1]), float(o.pos[2])],
                    "quat": [float(v) for v in o.quat], "shift_mm": 4.0, "iou": 0.9, "visible_frac": 1.0}

    env = _env()
    b = Bridge()
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    ck = Checker()
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets(), checker=ck, replan_on_divergence=False)
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    x0 = float(prov.belief.objects["block_0"].pos[0])
    ev = prov.recheck("block_0", "right")
    assert ck.calls == [("block_0", "right")] and ev["recheck"] and ev["ok"]
    assert np.isclose(float(prov.belief.objects["block_0"].pos[0]), x0 + 0.004)    # the measured pose is kept
    assert prov.recheck("nosuch", "left") is None


def test_the_switch_is_validated_and_off_by_default():
    import pytest

    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    assert RuntimeConfig().peg_stuck_confirm == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(peg_stuck_confirm="maybe"))
