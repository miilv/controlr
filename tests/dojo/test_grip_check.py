"""Stream H: RoboDojo's official gripper value as an empty-jaw check (``RuntimeConfig.grip_check``).

The value reaches the runtime under ``<arm>_ee_joint_state_official``: robot-harness's strict server v2
(``eef.<arm>.ee_joint_state_official``, D-C11) through the bridge contract, or ``DOJO_GRIP_SOURCE=v2`` in the dev
harnesses (the command stays in ``<arm>_ee_joint_state``). A path that hands over RoboDojo's observation as it is carries
it in ``<arm>_ee_joint_state`` itself: ``grip_official_key="ee_joint_state"`` (the command is then the runtime's own).
CPU only: the kinematic env, no Isaac, no perception service.
"""

import numpy as np
import pytest

from dojo.bridge.contract import ContractViolation, state_from_eef
from dojo.runtime.belief import RuntimeProvider, expected_official, official_width
from dojo.runtime.bridge import Bridge
from dojo.runtime.estimate import objects_from_gt_rows
from dojo.runtime.firewall import FirewallError, check_obs
from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.simlab.grip_source import GripSource
from dojo.skills.env import KinematicEnv
from dojo.skills.world import AssetInfo

STACK = "Stack the three blocks with different textures."
POS = [(0.412, -0.09, 0.78), (-0.26, -0.189, 0.78), (-0.344, -0.1, 0.78)]
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _env(positions=POS, half=0.015):
    rows = [{"label": f"block_{i}", "category": "block", "category_idx": 5, "pos": p, "quat": [1, 0, 0, 0],
             "half": [half] * 3} for i, p in enumerate(positions)]
    return KinematicEnv(rows, step_lim=550, instruction=STACK)


class _Assets:
    def __init__(self, half=(0.015, 0.015, 0.015)):
        self.half = np.asarray(half, float)

    def get(self, typ, cat, idx, clutter=False, mesh=False):
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=-self.half, bbox_max=self.half)


def _official(env, arm: str) -> float:
    """What RoboDojo's applied gripper control reads on the kinematic env: the command with nothing in the jaws,
    the measured finger minus 0.2 on a 30 mm block (``expected_official``)."""
    return expected_official(0.03, env.g[arm]) if arm in env.attached else env.g[arm]


class _Episode:
    """proxy_eval's ProxyEpisode stand-in; ``official`` adds the strict v2 key per arm; ``robodojo`` hands over RoboDojo's
    observation as it is: ``<arm>_ee_joint_state`` IS the official value, no command anywhere in the obs."""

    def __init__(self, env, official: bool, robodojo: bool = False):
        self.env, self.official, self.robodojo = env, official, robodojo

    def info(self):
        return {"stable": True, "task": "stack_blocks", "layout_id": 1, "instruction": self.env.instruction,
                "step_lim": self.env.step_lim, "reset_s": 0.0, "render_stage_s": 0.0}

    def observe(self):
        st = self.env.proprio()
        if self.robodojo:
            for a in ("left", "right"):
                st[f"{a}_ee_joint_state"] = np.array([round(_official(self.env, a), 4)])
        if self.official:
            for a in ("left", "right"):
                st[f"{a}_ee_joint_state_official"] = np.array([round(_official(self.env, a), 4)])
        img = np.zeros((48, 64, 3), np.uint8)
        return {"images": {c: img for c in CAMS}, "state": st, "instruction": self.env.instruction,
                "steps_used": self.env.steps_used, "step_lim": self.env.step_lim, "renderer": "fake"}


def _run(policy, ep, max_calls=2000):
    policy.reset(ep.info())
    for _ in range(max_calls):
        d = policy.next(ep.observe())
        if d.get("done"):
            return policy.episode_report()
        for a in d["actions"]:
            if ep.env.step(a):
                return policy.episode_report()
    raise AssertionError("episode did not finish")


# ------------------------------------------------------------------ the observation path
def _obs(**state_extra):
    st = {"left_arm_joint_state": np.zeros(6), "right_arm_joint_state": np.zeros(6), "left_ee_pose": np.zeros(7),
          "right_ee_pose": np.zeros(7), "left_ee_joint_state": [0.0], "right_ee_joint_state": [1.0], **state_extra}
    return {"images": {"cam_head": np.zeros((4, 4, 3), np.uint8)}, "state": st, "instruction": "x", "steps_used": 0,
            "step_lim": 10}


def test_firewall_takes_the_official_value_per_arm_and_nothing_else():
    clean, _ = check_obs(_obs(left_ee_joint_state_official=[0.3271]), "perception")
    assert float(clean["state"]["left_ee_joint_state_official"][0]) == 0.3271
    assert "right_ee_joint_state_official" not in clean["state"]          # optional, per arm
    clean, _ = check_obs(_obs(), "perception")                              # strict v1: no key, as before
    assert "left_ee_joint_state_official" not in clean["state"]
    with pytest.raises(FirewallError):                                      # one value, like the command
        check_obs(_obs(left_ee_joint_state_official=[0.3, 0.3]), "perception")
    for bad in ({"left_finger_q": [0.01]}, {"left_ee_joint_state_measured": [0.5]}):
        with pytest.raises(FirewallError):
            check_obs(_obs(**bad), "perception")
    o = _obs()
    del o["state"]["right_ee_joint_state"]                                  # the contract's own fields stay required
    with pytest.raises(FirewallError):
        check_obs(o, "perception")


def _st(gl, gr):
    return {"left_arm_joint_state": np.zeros(6), "left_ee_joint_state": np.array([gl]),
            "right_arm_joint_state": np.zeros(6), "right_ee_joint_state": np.array([gr])}


def _act(gl, gr):
    return {"left_arm_joint_state": np.zeros(6, np.float32), "left_ee_joint_state": np.array([gl], np.float32),
            "right_arm_joint_state": np.zeros(6, np.float32), "right_ee_joint_state": np.array([gr], np.float32)}


def test_grip_source_v2_keeps_the_command_and_adds_robodojos_value_under_its_own_key():
    g = GripSource("v2")
    out = g.apply(_st(1.0, 1.0))
    assert float(out["left_ee_joint_state"][0]) == 1.0 and float(out["left_ee_joint_state_official"][0]) == 1.0
    g.stepped(_act(0.0, 1.0))
    out = g.apply(_st(0.327123, 0.98))                                      # blocked jaws, a settling right arm
    assert float(out["left_ee_joint_state"][0]) == 0.0                      # the command, exactly as strict v1 / command
    assert float(out["left_ee_joint_state_official"][0]) == 0.3271          # RoboDojo's value, rounded as the server does
    assert float(out["right_ee_joint_state"][0]) == 1.0 and float(out["right_ee_joint_state_official"][0]) == 0.98
    assert "left_ee_joint_state_official" not in GripSource("command").apply(_st(1.0, 1.0))


def _eef(official="absent"):
    e = {"q": [0.0] * 6, "flange": {"pos": [0.0, 0.0, 1.0], "quat": [1.0, 0.0, 0.0, 0.0]}, "grip_cmd": 0.0,
         "grip_obs": 0.0}
    if official != "absent":
        e["ee_joint_state_official"] = official
    return {"left": e, "right": dict(e)}


def test_bridge_contract_passes_the_strict_v2_official_value_through():
    st = state_from_eef(_eef([0.3271]))
    assert float(st["left_ee_joint_state"][0]) == 0.0 and float(st["left_ee_joint_state_official"][0]) == 0.3271
    assert "left_ee_joint_state_official" not in state_from_eef(_eef())     # v1 server
    assert "left_ee_joint_state_official" not in state_from_eef(_eef(None))  # v2: null when the obs lacks the field
    with pytest.raises(ContractViolation):
        state_from_eef(_eef([float("nan")]))


def test_the_reading_model_matches_the_replays():
    # M's replays: a 40 mm stack block reads 0.30-0.35, a 3-5 mm mug wall 0.01-0.05; nothing in the jaws reads the command
    assert 0.30 < expected_official(0.040) < 0.36
    assert 0.0 < expected_official(0.004) < 0.05
    assert expected_official(0.0) == 0.0 and expected_official(0.0, 0.4) == 0.4
    assert abs(official_width(expected_official(0.031)) - 0.031) < 1e-9


# ------------------------------------------------------------------ the provider
def _prov(half=(0.015, 0.015, 0.015), official=None, cmd=0.0, still=10):
    env = _env()
    b = Bridge()
    obs = _Episode(env, official=False).observe()
    b.obs, b.fresh = obs, True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", _Assets(half))
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    prov.grip_check = True
    bel = prov.belief
    o = bel.objects["block_1"]
    o.asset = _Assets(half).get("Rigid", "block", 5)                        # e.g. a coin: 1.9 mm thick
    pose0 = np.r_[o.pos, o.quat].copy()
    z0 = float(bel.kin.tcp("left", bel.q["left"])[2, 3])
    bel.attached["left"] = ("block_1", np.eye(4))
    bel.grasp_z["left"] = (z0 - 0.05, 1, pose0)
    bel.n_since_g["left"] = still
    o.pos = o.pos + np.array([0.0, 0.0, 0.05])                              # the kinematic belief lifted it
    obs["state"]["left_ee_joint_state"] = np.array([cmd])
    if official is not None:
        obs["state"]["left_ee_joint_state_official"] = np.array([official])
    calls = []
    prov.held_checker = lambda obs_, arm, lab: calls.append((arm, lab)) or {"held": True, "confidence": 0.9}
    return prov, b.obs, pose0, calls


def test_an_empty_reading_takes_the_object_out_of_the_jaws_where_it_was_grasped():
    prov, obs, pose0, calls = _prov(official=0.0)
    prov._grip_check(obs)
    prov._held_check(obs)
    bel = prov.belief
    assert "left" not in bel.attached and calls == []                       # decided: no vision look either
    assert np.allclose(bel.objects["block_1"].pos, pose0[:3], atol=1e-6)    # closed on air: it never moved
    ev = [e for e in prov.events if e["kind"] == "grip_check"][-1]
    assert ev["verdict"] == "empty" and ev["dropped"] == "air" and ev["official"] == 0.0
    assert "block_1" in prov.touched and "block_1" in prov.grip_missed and prov.stats["grip_empty"] == 1


def test_a_held_reading_keeps_the_grasp_and_skips_the_vision_look():
    prov, obs, _, calls = _prov(official=0.2631)
    prov._grip_check(obs)
    prov._held_check(obs)
    assert "left" in prov.belief.attached and calls == []
    prov._grip_check(obs)                                                   # the same verdict is logged once per grasp
    evs = [e for e in prov.events if e["kind"] == "grip_check"]
    assert len(evs) == 1 and evs[0]["verdict"] == "held" and prov.stats["grip_reads"] == 2


@pytest.mark.parametrize("why, kw", [("thin object", {"half": (0.014, 0.014, 0.001), "official": 0.0}),
                                     ("settling", {"official": 0.0, "still": 2}),
                                     ("partial close", {"official": 0.4, "cmd": 0.4}),
                                     (None, {"official": None})])
def test_an_inconclusive_or_missing_reading_leaves_it_to_the_vision_look(why, kw):
    prov, obs, _, calls = _prov(**kw)
    prov._grip_check(obs)
    assert "left" in prov.belief.attached                                   # never dropped on these
    evs = [e for e in prov.events if e["kind"] == "grip_check"]
    assert (evs[-1]["why"] if evs else None) == why and (not evs or evs[-1]["verdict"] == "inconclusive")
    prov._held_check(obs)
    assert calls == [("left", "block_1")]                                   # the vision held_look runs as before


def test_a_slip_after_a_held_reading_is_a_release_under_the_jaws_for_the_next_resync():
    prov, obs, pose0, _ = _prov(official=0.2631)
    prov._grip_check(obs)
    prov.belief.objects["block_1"].pos += np.array([0.10, 0.0, 0.0])        # carried 10 cm before it slipped out
    carried = prov.belief.objects["block_1"].pos.copy()
    obs["state"]["left_ee_joint_state_official"] = np.array([0.0])
    prov._grip_check(obs)
    o = prov.belief.objects["block_1"]
    assert "left" not in prov.belief.attached
    assert np.allclose(o.pos[:2], carried[:2]) and not np.allclose(o.pos[:2], pose0[:2])   # fell from the jaws
    ev = [e for e in prov.events if e["kind"] == "grip_check"][-1]
    assert ev["dropped"] == "slip" and "block_1" in prov._pp_doubt
    assert prov.belief.pop_released() == ["block_1"] and prov.belief.released_by["block_1"] == "left"


class _ReadChecker:
    """post_place's checker: the refine and the 6-DoF read find nothing; the wide search finds the object at ``found``
    (stream W's release read). Records what each was asked."""

    def __init__(self, found):
        self.found, self.reads, self.searches = found, [], []

    def refine(self, obs, o, others, arm, held=None):
        return {"ok": False, "iou": 0.0}

    def release_read(self, obs, o, others, arm, cfg=None):
        self.reads.append(np.r_[o.pos, o.quat].copy())
        return {"ok": False, "why": "fit_poor"}

    def release_search(self, obs, o, others, arm, pre_pose7=None, cfg=None):
        self.searches.append((np.r_[o.pos, o.quat].copy(), None if pre_pose7 is None else np.asarray(pre_pose7)))
        return {"ok": True, "via": "head_search", "pose7": list(self.found)}


def test_grip_check_and_the_release_read_compose_without_moving_an_object_twice():
    """grip_check (H) and post_place_read "6dof_search" (W) on the same grasp. Closed on air: grip_check puts the object
    back where it was grasped and nothing is released, so no release read runs (no second move). A slip after a held
    reading: grip_check drops it under the jaws as a release; the next post-place read starts from that drop and its
    search moves it once, to where it is seen."""
    # closed on air
    prov, obs, pose0, _ = _prov(official=0.0)
    ck = _ReadChecker(found=[0.3, -0.2, 0.78, 1.0, 0.0, 0.0, 0.0])
    prov.checker, prov.post_place_read = ck, "6dof_search"
    prov._grip_check(obs)
    assert prov.post_place(obs, prov.belief.pop_released()) == []
    assert ck.reads == [] and ck.searches == []
    assert np.allclose(prov.belief.objects["block_1"].pos, pose0[:3], atol=1e-6)
    # a slip after a held reading
    prov, obs, pose0, _ = _prov(official=0.2631)
    ck = _ReadChecker(found=[0.3, -0.2, 0.78, 1.0, 0.0, 0.0, 0.0])
    prov.checker, prov.post_place_read = ck, "6dof_search"
    prov._grip_check(obs)
    prov.belief.objects["block_1"].pos += np.array([0.10, 0.0, 0.0])
    obs["state"]["left_ee_joint_state_official"] = np.array([0.0])
    prov._grip_check(obs)
    dropped = np.r_[prov.belief.objects["block_1"].pos, prov.belief.objects["block_1"].quat].copy()
    evs = prov.post_place(obs, prov.belief.pop_released())
    assert len(evs) == 1 and len(ck.reads) == 1 and len(ck.searches) == 1
    assert np.allclose(ck.reads[0], dropped) and np.allclose(ck.searches[0][0], dropped)  # the read starts from the drop
    assert ck.searches[0][1] is None            # it rose with the jaws (grip_check said held): no "never left" seed
    assert np.allclose(prov.belief.objects["block_1"].pos, [0.3, -0.2, 0.78])  # moved once, to where it was seen
    assert evs[0]["search"]["via"] == "head_search" and evs[0]["pose7"][:3] == pytest.approx([0.3, -0.2, 0.78])
    assert "block_1" not in prov._pp_doubt


def test_an_empty_hand_in_a_handover_releases_only_that_hand():
    prov, obs, _, _ = _prov(official=0.0)
    prov.belief.attached["right"] = ("block_1", np.eye(4))
    prov._grip_check(obs)
    assert "left" not in prov.belief.attached and prov.belief.attached["right"][0] == "block_1"
    assert prov.events[-1]["dropped"].startswith("this hand only")


def test_off_by_default_and_validated():
    assert RuntimeConfig().grip_check == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", grip_check="maybe"), assets=_Assets())
    assert RuntimeConfig().grip_official_key == "official_key"                # today's key unless a deployment says so
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", grip_check="on", grip_official_key="auto"), assets=_Assets())


# ------------------------------------------------------------------ grip_official_key: where the official value is
def _prov_ee(obs_value: float, own_cmd: float, mode: str = "ee_joint_state"):
    """A held block_1 where the obs's left_ee_joint_state reads ``obs_value`` (no _official key) and the runtime's
    own last command (``belief.g``) is ``own_cmd``."""
    prov, obs, pose0, calls = _prov(official=None, cmd=obs_value)
    prov.grip_official_key = mode
    prov.belief.g["left"] = own_cmd
    return prov, obs, pose0, calls


def test_ee_joint_state_mode_reads_the_obs_value_against_our_own_command():
    # RoboDojo's observation as it is: 0.2631 on a held 30 mm block while we commanded 0.0
    prov, obs, _, calls = _prov_ee(0.2631, 0.0)
    prov._grip_check(obs)
    prov._held_check(obs)
    ev = [e for e in prov.events if e["kind"] == "grip_check"]
    assert ev[-1]["verdict"] == "held" and ev[-1]["official"] == 0.2631 and ev[-1]["cmd"] == 0.0
    assert "left" in prov.belief.attached and calls == []                   # decided: no vision look
    # closed on nothing: the applied control reads the command itself
    prov, obs, pose0, calls = _prov_ee(0.0, 0.0)
    prov._grip_check(obs)
    ev = [e for e in prov.events if e["kind"] == "grip_check"][-1]
    assert ev["verdict"] == "empty" and ev["dropped"] == "air" and "left" not in prov.belief.attached
    assert np.allclose(prov.belief.objects["block_1"].pos, pose0[:3], atol=1e-6)


def test_the_mode_must_match_the_path_it_cannot_be_guessed():
    # the trap, both ways round:
    # (1) the default key on a path whose obs carries the official value in <arm>_ee_joint_state: no reading at all
    prov, obs, _, calls = _prov_ee(0.2631, 0.0, mode="official_key")
    prov._grip_check(obs)
    assert not any(e["kind"] == "grip_check" for e in prov.events)          # a silent no-op
    prov._held_check(obs)
    assert calls == [("left", "block_1")]                                   # the vision held_look runs as before
    # (2) "ee_joint_state" on a command-semantics path (strict v1: the obs's value IS our command, 0.0): a block that is
    # really held reads official - command = 0, exactly an empty grip -> wrongly dropped. Equal values carry no
    # information about which semantics the path has, so the deployment must say which key it provides
    prov, obs, _, _ = _prov_ee(0.0, 0.0)
    prov._grip_check(obs)
    assert [e for e in prov.events if e["kind"] == "grip_check"][-1]["verdict"] == "empty"
    # (3) and the default key on that strict-v2 path still reads the official key against the obs's command
    prov, obs, _, _ = _prov(official=0.2631, cmd=0.0)
    prov._grip_check(obs)
    assert [e for e in prov.events if e["kind"] == "grip_check"][-1]["verdict"] == "held"


def test_episode_ee_joint_state_mode_on_robodojo_semantics():
    def episode(key, off_m):
        env = _env()
        pc = _Perception(off_m)
        pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off", held_look="b1_then_agent", tracker="on",
                                          grip_check="on", grip_official_key=key), assets=_Assets(), perception_client=pc)
        return _run(pol, _Episode(env, official=False, robodojo=True)), pc
    rep, pc = episode("ee_joint_state", 0.0)
    assert rep["error"] is None and rep["grip_check"] == {"official_at_start": True, "key": "ee_joint_state"}
    picks = sum(1 for h in rep["history"] if h["skill"] == "pick" and h["ok"])
    held = [e for e in rep["events"] if e["kind"] == "grip_check" and e["verdict"] == "held"]
    assert picks >= 2 and len(held) >= picks and pc.held == []              # the reads replace the vision looks
    rep, _ = episode("ee_joint_state", 0.028)                               # block_0 believed 28 mm off: a close on air
    ev = [e for e in rep["events"] if e["kind"] == "grip_check" and e["verdict"] == "empty"]
    assert ev and ev[0]["label"] == "block_0" and ev[0]["dropped"] == "air"
    rep, pc = episode("official_key", 0.0)                                  # the default key on this path: a no-op
    assert rep["grip_check"]["official_at_start"] is False and not any(e["kind"] == "grip_check" for e in rep["events"])
    assert pc.held


# ------------------------------------------------------------------ whole episodes
class _Perception:
    """Frame-0 estimate with block_0 (the first one picked) ``off_m`` off along x: a close there meets nothing."""

    def __init__(self, off_m: float):
        self.off, self.held = off_m, []

    def estimate(self, frames, vocabulary, fold, prior_record=None):
        pos = [list(p) for p in POS]
        pos[0][0] += self.off
        return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": f"block_{i}", "default_pos": p,
                                     "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.9}}
                                    for i, p in enumerate(pos)]}}

    def check(self, frame, objects, fold):
        return {o["label"]: {"head": {"iou": 0.9, "vis": 1.0, "n_px": 100}} for o in objects}

    def look_plan(self, frame, record, max_looks=2):
        return []

    def held_look(self, frame, arm, mode):
        self.held.append(arm)
        return {"held": True, "confidence": 0.9, "calls": 1, "source": "agent"}


def _episode(grip_check: str, official: bool, off_m: float):
    env = _env()
    pc = _Perception(off_m)
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off", held_look="b1_then_agent", tracker="on",
                                      grip_check=grip_check), assets=_Assets(), perception_client=pc)
    return _run(pol, _Episode(env, official)), pc, env


def test_episode_held_reads_replace_the_vision_looks():
    rep, pc, env = _episode("on", official=True, off_m=0.0)
    assert rep["error"] is None
    picks = sum(1 for h in rep["history"] if h["skill"] == "pick" and h["ok"])
    held = [e for e in rep["events"] if e["kind"] == "grip_check" and e["verdict"] == "held"]
    assert picks >= 2 and len(held) >= picks and pc.held == [] and rep["agentic"] == []   # 0 held-look calls
    assert rep["grip_check"]["official_at_start"] and rep["provider"]["grip_reads"] >= picks
    rep1, pc1, _ = _episode("off", official=True, off_m=0.0)               # off: the key is there, nothing reads it
    assert not any(e["kind"] == "grip_check" for e in rep1["events"]) and len(pc1.held) >= picks


def test_episode_a_close_on_air_is_not_carried_to_the_place():
    """block_0 believed 28 mm off: inside the belief's attach margin (half + 15 mm), outside the env's grasp (half + 12)."""
    rep, _, env = _episode("on", official=True, off_m=0.028)
    ev = [e for e in rep["events"] if e["kind"] == "grip_check" and e["verdict"] == "empty"]
    assert ev and ev[0]["label"] == "block_0" and ev[0]["dropped"] == "air"
    first = rep["history"][0]
    assert first["skill"] == "pick" and first["status"] == "grasp_slip"     # the pick's own check now sees no rise
    rep0, pc0, _ = _episode("off", official=True, off_m=0.028)              # today: the vision look says "held" ...
    assert not any(e["kind"] == "grip_check" for e in rep0["events"]) and pc0.held
    assert rep0["history"][0]["status"] == "ok" and rep0["history"][1]["skill"] == "place"   # ... a phantom carry
