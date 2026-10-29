"""robot-harness G5 review (docs/notes/G5-VERIFICATION-2026-10-29.md, "For Ilia" items 1, 2, 10, 11): the guards."""

import json
import threading
import time
from multiprocessing.connection import Listener

import numpy as np
import pytest

from dojo.bridge.backend import FOREIGN_REFUSED, RobotServerBackend
from dojo.bridge.client import RobotServerClient, ServerError
from dojo.bridge.contract import BANNED, ContractViolation
from test_bridge import bench, serve  # noqa: F401  (the fake strict server over a real socket)


# ------------------------------------------------------------------ 1: a foreign skill's handle
def test_a_foreign_skill_never_reaches_the_grader_or_the_episode_verbs(bench):  # noqa: F811
    srv, be, _ = serve(bench)
    be.connect()
    be.reset()
    n = len(srv.requests)
    rob = be.robot
    for verb in FOREIGN_REFUSED:
        for send in (lambda v: getattr(rob, v)(), lambda v: rob.call(v), lambda v: rob.ok(v),
                     lambda v: rob.call(v.upper())):
            with pytest.raises(ContractViolation, match="refused before sending"):
                send(verb)
    assert len(srv.requests) == n                       # nothing reached the socket
    assert rob.hold(steps=1)["ok"] and [r["cmd"] for r in srv.requests[n:]] == ["hold"]
    with pytest.raises(AttributeError):
        rob._client_private                             # noqa: B018  (no private attribute through the proxy)
    srv.stop()


def test_a_leaking_reply_to_a_foreign_skill_is_refused():
    sent = []

    def transport(req):
        sent.append(req["cmd"])
        return {"ok": True, "cmd": req["cmd"], "step": 0, "steps_left": 10, "ended": False,
                "steps": 1, "task": "stack_blocks"}

    be = RobotServerBackend(RobotServerClient(transport=transport))
    assert "task" in BANNED
    for send in (lambda: be.robot.hold(steps=1), lambda: be.robot.call("hold", steps=1),
                 lambda: be.robot.ok("hold", steps=1)):
        with pytest.raises(ContractViolation, match="task"):
            send()
    assert sent == ["hold"] * 3


def test_a_failed_foreign_verb_raises_on_ok_only():
    be = RobotServerBackend(RobotServerClient(transport=lambda req: {"ok": False, "cmd": req["cmd"], "error": "no"}))
    assert be.robot.call("hold")["ok"] is False
    with pytest.raises(ServerError):
        be.robot.ok("hold")


# ------------------------------------------------------------------ 2: bridge.json env notes
def test_agentic_extra_body_is_fingerprinted_in_bridge_json(tmp_path):
    from dojo.bridge.driver import env_notes, write_outputs

    token = "sk-or-v1-SYNTHETICTOKEN0123456789abcdef"
    notes = env_notes({"AGENTIC_EXTRA_BODY": json.dumps({"api_key": token, "reasoning_effort": "low"}),
                       "AGENTIC_MODEL": "cx/gpt-5", "OMNIROUTE_API_KEY": token})
    assert notes["AGENTIC_MODEL"] == "cx/gpt-5" and "OMNIROUTE_API_KEY" not in notes
    fp = notes["AGENTIC_EXTRA_BODY"]
    assert fp.startswith("sha256:") and len(fp) == len("sha256:") + 16
    write_outputs({"summary": {"env": notes}, "report": {}, "executed": []}, tmp_path)
    text = (tmp_path / "bridge.json").read_text()
    assert token not in text and "SYNTHETIC" not in text and fp in text


# ------------------------------------------------------------------ 10: missing asset metadata
def test_missing_asset_metadata_is_unknown_geometry_never_a_confident_cube(tmp_path):
    from dojo.runtime.estimate import objects_from_record
    from dojo.skills.world import AssetDB, asset_from_metadata

    rec = {"Rigid": {"block": [{"label": "obj0", "category_idx": 5, "default_pos": [0.3, -0.1, 0.78],
                                "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.9}}]}}
    o = objects_from_record(rec, AssetDB(objects_root=tmp_path), meshes=False)["obj0"]   # no metadata.json
    assert o.asset.unknown_geometry and o.confidence == 0.0
    assert asset_from_metadata("Clutter/vehicle/00000", {"geometry": {}}).unknown_geometry          # no aligned bbox
    assert asset_from_metadata("Rigid/x/00000", {"geometry": {"aligned_bbox": {"vertices": []}}}).unknown_geometry
    ok = asset_from_metadata("Rigid/block/00005", {"geometry": {"aligned_bbox": {"vertices": [[-0.01] * 3, [0.01] * 3]}}})
    assert not ok.unknown_geometry and np.allclose(ok.extents, 0.02)
    o = objects_from_record(rec, None, meshes=False)["obj0"]          # no catalogue at all (offline tools): as before
    assert not o.asset.unknown_geometry and o.confidence == 0.9


def test_planning_refuses_an_object_of_unknown_geometry():
    from dojo.skills.skills import GTProvider, Skills
    from test_skills import _blocks_env

    env = _blocks_env([(0.3, -0.1, 0.78)])
    sk = Skills(env, GTProvider(env, None, instruction=env.instruction))
    assert not isinstance(sk.plan_pick("obj0"), type(None))
    sk.world.objects["obj0"].asset.unknown_geometry = True
    n0 = sk.ex.steps_used
    for r in (sk.plan_pick("obj0"), sk.plan_transfer("obj0", [np.eye(4)]), sk.plan_topple("obj0", 2),
              sk.push("obj0", (0.35, -0.1))):
        assert not r.ok and r.status == "unknown_geometry", r
    assert sk.ex.steps_used == n0


# ------------------------------------------------------------------ 11: the perception reply deadline
def _listener(reply: bool):
    lst = Listener(("127.0.0.1", 0), authkey=b"k" * 16)
    got = []

    def run():
        conn = lst.accept()
        try:
            while True:
                req = conn.recv()
                got.append(req["op"])
                if reply:
                    conn.send({"ok": True, "check": json.dumps({})})
        except (EOFError, OSError):
            pass

    threading.Thread(target=run, daemon=True).start()
    return lst, got


def test_a_stalled_perception_reply_raises_within_the_deadline():
    from dojo.runtime.perception_service import PerceptionClient

    lst, got = _listener(reply=False)
    pc = PerceptionClient(port=lst.address[1], authkey=b"k" * 16, timeout_s=5.0, op_timeout_s=0.5)
    t0 = time.perf_counter()
    with pytest.raises(TimeoutError, match="no check reply within"):
        pc._call({"op": "check"})
    assert time.perf_counter() - t0 < 3.0 and pc.dead and pc.calls[-1]["timeout"]
    t0 = time.perf_counter()
    with pytest.raises(TimeoutError, match="connection closed"):   # dead: a late reply never answers a later op
        pc._call({"op": "check"})
    assert time.perf_counter() - t0 < 0.5 and got == ["check"]
    lst.close()


def test_a_prompt_perception_reply_is_unchanged():
    from dojo.runtime.perception_service import PerceptionClient

    lst, got = _listener(reply=True)
    pc = PerceptionClient(port=lst.address[1], authkey=b"k" * 16, timeout_s=5.0, op_timeout_s=5.0)
    assert pc._call({"op": "check"})["ok"] and pc._call({"op": "check"})["ok"] and pc.dead is None
    assert got == ["check", "check"] and "timeout" not in pc.calls[-1]
    pc.close()
    lst.close()


def test_the_runtime_reconnects_a_dead_perception_client_at_the_next_episode():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy

    class Dead:
        dead = "no estimate reply within 600 s"
        calls = []

    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off"))
    pol._pc = Dead()
    pol.reset({"instruction": "Stack the three blocks with different textures.", "step_lim": 100})
    assert pol._pc is None
    live = type("Live", (), {"dead": None, "calls": []})()
    pol._pc = live
    pol.reset({"instruction": "Stack the three blocks with different textures.", "step_lim": 100})
    assert pol._pc is live


# ------------------------------------------------------------------ 6: server perception's missing ops
def test_server_perception_forwards_local_ops_and_says_when_nothing_can_check():
    from dojo.bridge.perception import UNAVAILABLE, ServerPerception, local_only_features
    from dojo.runtime.policy import RuntimeConfig

    class Local:
        def scan_looks(self):
            return [{"arm": "left"}]

        def agentic_estimate(self, *a, **k):
            return {"via": "local"}

        def check(self, frame, objects, fold):
            return {"a": {"head": {"prec": 1.0}}}

    sp = ServerPerception(backend=None, local=Local())
    assert sp.scan_looks() == [{"arm": "left"}] and sp.agentic_estimate(None, "x", 0) == {"via": "local"}
    assert sp.check(None, [], 0) == {"a": {"head": {"prec": 1.0}}}
    bare = ServerPerception(backend=None)
    with pytest.raises(AttributeError, match="no `scan_looks` and no local client"):
        bare.scan_looks()
    assert bare.check(None, [], 0) == {UNAVAILABLE: "server perception: no local checker"}
    assert bare.refine(None, [], 0) == {} and bare.search_regions(None, {}, "") == [] and bare.look_poses(None, "left") == []
    assert local_only_features(RuntimeConfig()) == []
    assert local_only_features(RuntimeConfig(scan="trigger", held_look="b1", watch=True)) == ["held_look", "scan", "watch"]


def test_the_driver_refuses_server_perception_without_a_local_client(monkeypatch, tmp_path):
    from dojo.bridge import driver

    monkeypatch.delenv("DOJO_PERCEPTION_PORT", raising=False)
    with pytest.raises(SystemExit, match=r"enables \['scan'\]"):     # before any server is contacted
        driver.main(["--server-id", "nosuch", "--sock", str(tmp_path / "none.sock"), "--perception", "server",
                     "--config", '{"scan": "trigger"}', "--twin", "off", "--out", str(tmp_path)])


def test_resync_logs_an_unavailable_check_as_unverified():
    from dojo.runtime.belief import RuntimeProvider
    from dojo.runtime.bridge import Bridge
    from dojo.runtime.estimate import objects_from_gt_rows
    from test_runtime import STACK, FakeAssets, FakeEpisode, _env

    class Checker:
        def check(self, obs, objects):
            return {"_unavailable": "server perception: no local checker"}

        def estimate(self, obs):
            raise AssertionError("no re-estimate without a divergence")

    env = _env()
    b = Bridge()
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets(), checker=Checker(), replan_on_divergence=False)
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    ev = prov.resync(b.obs)
    assert ev["diverged"] == [] and ev["unverified"] == "server perception: no local checker" and ev["check"] == {}


# ------------------------------------------------------------------ 7: a search find's identity
@pytest.mark.parametrize("cands", [["Rigid/block/00005", "Rigid/block/00006"], ["Rigid/block/00006", "Rigid/block/00005"]])
def test_search_identity_unknown_when_several_candidates_fit(cands):
    from dojo.runtime.estimate import objects_from_record
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from test_runtime import FakeAssets

    class P:
        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None):
            return {"block_1": {"ok": True, "pos": [0.2, -0.1, 0.78], "quat": [1, 0, 0, 0], "iou": 0.9}}

    rec = {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": "block_0", "default_pos": [0.1, -0.1, 0.78],
                                "default_ori": [1, 0, 0, 0]}]},
           "_not_found": [{"label": "block_1", "candidates": list(cands)}]}
    for mode, known in (("first", True), ("unknown", False)):
        pol = RuntimePolicy(RuntimeConfig(mode="perception", search=True, search_identity=mode), assets=FakeAssets(),
                            perception_client=None)
        pol.perception = lambda: P()
        pol.report, pol.fold = {"events": []}, 0
        out, found = pol._search_refine(rec, {"target": [0.2, -0.1, 0.78], "why": "test"}, {}, "right")
        assert found == ["block_1"]
        o = objects_from_record(out, None)["block_1"]
        assert o.identity_known is known and o.confidence == pytest.approx(0.9)    # pose IoU stays pose confidence
    one = {**rec, "_not_found": [{"label": "block_1", "candidates": [cands[0]]}]}
    pol = RuntimePolicy(RuntimeConfig(mode="perception", search=True, search_identity="unknown"), assets=FakeAssets(),
                        perception_client=None)
    pol.perception = lambda: P()
    pol.report, pol.fold = {"events": []}, 0
    out, _ = pol._search_refine(one, {"target": [0.2, -0.1, 0.78], "why": "test"}, {}, "right")
    assert objects_from_record(out, None)["block_1"].identity_known is True      # one candidate: its identity is known
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(search_identity="bogus"))
