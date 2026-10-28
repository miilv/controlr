"""Legality audit 2026-10-28, P1 finding 4 (robot-harness docs/notes/ILIA-POLICY-LEGALITY-AUDIT-2026-10-28.md): the
launcher's ``--task`` decided whether the verifier ran (the policy read ``tc.info()["task"]`` of the bundle's twin and
turned verification off when it differed from the instruction-derived scene). Now the policy builds its verifier for
the scene it derived from the instruction and never asks a twin what it runs; the dev drivers' ``BundleTwins`` hand it
a twin for that scene (the bundle's own when it matches, else one started for the scene). Regression test (the
audit's): identical observations and instruction, only the launcher task behind the twin changed -> the same verifier
activation and the same decisions."""

import numpy as np

from dojo.runtime.firewall import DEV_GT_KEY
from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills.env import KinematicEnv
from dojo.skills.world import AssetInfo

STACK = "Stack the three blocks with different textures."
POS = [(0.412, -0.09, 0.78), (-0.26, -0.189, 0.78), (-0.344, -0.1, 0.78)]


def _env():
    rows = [{"label": f"block_{i}", "category": "block", "category_idx": 5, "pos": p, "quat": [1, 0, 0, 0],
             "half": [0.015] * 3} for i, p in enumerate(POS)]
    return KinematicEnv(rows, step_lim=550, instruction=STACK)


class FakeAssets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.015), bbox_max=np.full(3, 0.015))


class Twin:
    """A twin stand-in that runs ``launched`` (the launcher's task) and rolls every candidate out on a kinematic copy
    of the estimate. ``info()`` answers like the real service; a policy must never need it."""

    def __init__(self, launched):
        self.launched, self.info_calls, self.base, self.snaps = launched, 0, None, {}

    def info(self):
        self.info_calls += 1
        return {"task": self.launched, "num_envs": 4}

    def _env_from(self, layout, robot):
        rows = [{"label": r["label"], "category": c, "category_idx": r["category_idx"], "pos": r["default_pos"],
                 "quat": r["default_ori"], "half": [0.015] * 3} for c, lst in (layout.get("Rigid") or {}).items()
                for r in lst]
        env = KinematicEnv(rows, step_lim=10_000)
        for a in ("left", "right"):
            env.q[a] = np.asarray(robot[f"{a}_q"], float)
            env.g[a] = float(robot[f"{a}_grip"])
        return env

    def load_from_estimate(self, layout, robot=None, settle_steps=50):
        self.base = self._env_from(layout, robot)
        return {"fast": True, "stable": True}

    def verify(self, world, candidates, perturb=None, k=1, record_traj=False, seed=0, keep=None):
        import copy
        if "layout" in world:                       # the real service loads the estimate inside verify
            self.load_from_estimate(world["layout"], world.get("robot"))
        base = self.snaps[world["snapshot_id"]] if "snapshot_id" in world else self.base
        out = []
        for ci, c in enumerate(candidates):
            for si in range(k):
                e = copy.deepcopy(base)
                for a in c["actions"]:
                    e.step(a)
                o = {"candidate": ci, "sample": si, "success": False, "score": 0.0, "ended": False,
                     "reward": {"success": False, "success_flag": True, "ended": False, "steps_used": 0,
                                "step_lim": 1000},
                     "final_objects": {r["label"]: list(r["pos"]) + list(r["quat"]) for r in e.gt_objects()},
                     "final_robot": {f"{a}_q": e.q[a].tolist() for a in ("left", "right")}}
                if keep == "nominal" and si == 0:
                    sid = max(self.snaps, default=0) + 1
                    self.snaps[sid] = e
                    o["snapshot_id"] = sid
                out.append(o)
        return out

    def drop_snapshot(self, sid):
        self.snaps.pop(int(sid), None)


def _episode(launched):
    import dojo.runtime.estimate as E
    env = _env()
    tw = Twin(launched)
    pol = RuntimePolicy(RuntimeConfig(mode="gt", verify="bundle", n_cands=2, verify_k=1), assets=FakeAssets(),
                        twin_client=tw)
    orig = E.estimate_to_layout
    E.estimate_to_layout = lambda scene, objs: {"Rigid": {"block": [
        {"label": o["label"], "category_idx": o["category_idx"], "default_pos": o["pos"], "default_ori": o["quat"]}
        for o in objs]}}
    actions = []
    try:
        pol.reset({"instruction": STACK, "step_lim": env.step_lim, "task": launched, "layout_id": 1})
        for _ in range(2000):
            obs = {"images": {c: np.zeros((48, 64, 3), np.uint8) for c in ("cam_head", "cam_left_wrist",
                                                                          "cam_right_wrist")},
                   "state": env.proprio(), "instruction": STACK, "steps_used": env.steps_used,
                   "step_lim": env.step_lim, DEV_GT_KEY: env.gt_objects()}
            d = pol.next(obs)
            if d.get("done"):
                break
            for a in d["actions"]:
                actions.append({k: np.round(np.asarray(v, float), 9).tolist() for k, v in a.items()})
                if env.step(a):
                    break
            else:
                continue
            break
    finally:
        E.estimate_to_layout = orig
    return pol.episode_report(), actions, tw


def test_the_launcher_task_never_decides_verification_or_the_decisions():
    rep_a, act_a, tw_a = _episode("stack_blocks")           # the bundle launched for the instruction's scene
    rep_b, act_b, tw_b = _episode("press_by_number")        # the audit's mismatch: another launcher task
    for rep in (rep_a, rep_b):
        assert rep["error"] is None, rep["error"]
        assert isinstance(rep["verify"], dict) and not rep["verify"].get("disabled"), rep["verify"]
        assert not [e for e in rep["events"] if e.get("kind") == "verify_unavailable"]
    assert tw_a.info_calls == tw_b.info_calls == 0              # the policy never asks the twin what it runs
    assert rep_a["verify"]["calls"] == rep_b["verify"]["calls"] > 0
    assert rep_a["decisions"] == rep_b["decisions"] and act_a == act_b and len(act_a) > 0


def test_a_factory_twin_is_per_scene():
    started = []
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="bundle"))
    pol._twin_factory = lambda scene: started.append(scene) or Twin(scene)
    a = pol.twin("cover_blocks")
    assert pol.twin("cover_blocks") is a and started == ["cover_blocks"]
    b = pol.twin("build_tower")
    assert b is not a and started == ["cover_blocks", "build_tower"]


def test_bundle_twins_serve_the_policy_scene_never_the_launcher_task(monkeypatch, tmp_path):
    import dojo.simlab.twin_service as TS
    from dojo.bridge.twin import BundleTwins

    class Client:
        def __init__(self, *a, **k):
            self.kind = "bundle"
    monkeypatch.setattr(TS, "TwinClient", Client)
    monkeypatch.setenv("SIMLAB_TWIN_PORT", "1")
    stopped = []

    class Proc:
        def __init__(self, scene):
            self.scene = scene

        def stop(self):
            stopped.append(self.scene)

    tw = BundleTwins("stack_blocks", 4, tmp_path, spawn=lambda scene: (f"spawned:{scene}", Proc(scene)))
    assert tw.client("stack_blocks").kind == "bundle"
    assert tw.client("press_by_number") == "spawned:press_by_number"
    assert tw.served == [("stack_blocks", "bundle"), ("press_by_number", "spawned")]
    tw.stop()
    assert stopped == ["press_by_number"]
    monkeypatch.delenv("SIMLAB_TWIN_PORT")                    # no bundle twin at all: every scene gets its own
    assert BundleTwins("stack_blocks", 4, tmp_path, spawn=lambda s: (f"spawned:{s}", Proc(s))).client(
        "stack_blocks") == "spawned:stack_blocks"


def test_no_runtime_source_asks_a_twin_which_task_it_runs():
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "dojo"
    pat = re.compile(r"""(tc|twin\([^)]*\)|_tc|client)\.info\(\)""")
    for f in [*(root / "runtime").glob("*.py"), *(root / "agent").glob("*.py"), *(root / "skills").glob("*.py")]:
        for ln in f.read_text().splitlines():
            assert not pat.search(ln), (f.name, ln)
