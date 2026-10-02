"""CPU tests for the Isaac wire protocol and ``IsaacRobot`` against an
in-process fake server (no Isaac): action -> waypoint mapping, durations,
event translation, Observation/CameraInfo construction, TCP-offset agreement
between controlr's kinematics and the sim."""

from __future__ import annotations

import threading
from multiprocessing.connection import Listener

import numpy as np
import pytest

from controlr.config import SafetyConfig
from controlr.robot.isaac import protocol, server, tasks
from controlr.robot.isaac.client import IsaacRobot
from controlr.robot.kinematics import DEFAULT_TCP_OFFSET, UR3Kinematics
from controlr.types import Action, ActionMode, EventLevel

KIN = UR3Kinematics()


def test_protocol_roundtrip_and_errors():
    req = protocol.request("execute", q=np.zeros((1, 6)))
    assert req["op"] == "execute" and req["version"] == protocol.PROTOCOL_VERSION
    with pytest.raises(ValueError):
        protocol.request("teleport")
    assert protocol.unwrap(protocol.ok({"a": 1}, 0.0)) == {"a": 1}
    with pytest.raises(protocol.RemoteError, match="boom"):
        protocol.unwrap(protocol.error("boom", "tb"))
    assert protocol.authkey_from_env({protocol.AUTHKEY_ENV: "s3cret"}) == b"s3cret"


def test_sim_tcp_matches_controlr_kinematics_offset():
    # The sim TCP is tool0 + TCP_OFFSET_M along tool z (PHANTOM's convention);
    # controlr's IK/safety must use the same point or every MOVE is off.
    assert DEFAULT_TCP_OFFSET == (0.0, 0.0, server.TCP_OFFSET_M, 0.0, 0.0, 0.0)


class FakeRig:
    """Mimics the server ops with ideal kinematic tracking."""

    def __init__(self):
        self.q = np.asarray(tasks.START_Q, float)
        self.grip = 0.09
        self.calls: list[tuple[str, dict]] = []
        self.next_contacts: dict = {}
        self.stop = False

    def _state(self):
        p, rv = KIN.fk(self.q)
        return {"t": 1.0, "q": self.q.copy(), "qd": np.zeros(6), "tcp_pos": p, "tcp_rotvec": rv, "tcp_fk_pos": p,
                "tcp_fk_rotvec": rv, "gripper_m": self.grip, "gripper_closure": 0.1,
                "gripper_closed": self.grip < 0.02, "holding": False, "pad_object_force_n": np.zeros(2)}

    def handle(self, op, args):
        self.calls.append((op, args))
        if op == "info":
            return {"camera": {"name": "scene", "width": 640, "height": 480, "K": np.eye(3) * 600,
                               "T_cam_base": np.eye(4)},
                    "gripper": {"max_m": 0.0913, "min_m": 0.0009, "tcp_offset_m": 0.18},
                    "scene_info": {"mat": {"top_z": -0.0095}}, "timings": {}}
        if op == "reset":
            return {"obs": {"rgb": np.zeros((480, 640, 3), np.uint8), "state": self._state()},
                    "episode": {"task": args["task"], "seed": args["seed"]}, "reset_s": 0.1}
        if op == "observe":
            return {"rgb": np.full((480, 640, 3), 7, np.uint8), "state": self._state(), "render_s": 0.01}
        if op == "state":
            return self._state()
        if op == "execute":
            q, g = np.asarray(args["q"]), np.asarray(args["gripper"])
            n = 1 if self.stop else len(q)
            self.q = q[n - 1].copy()
            if np.isfinite(g[n - 1]):
                self.grip = float(g[n - 1])
            return {"state": self._state(), "stopped": self.stop, "stop_reason": "contact force 99 N > 80 N",
                    "segments_done": n, "sim_s": float(np.sum(args["durations"])) + 0.1, "settle_s": 0.1,
                    "settled": True, "tcp_err_m": 0.0, "contacts_peak_n": self.next_contacts,
                    "gripper_mechanics_ok": True}
        if op == "check_goal":
            return {"success": True, "progress": 1.0, "message": "goal reached", "metrics": {"x": 1}}
        if op in ("close", "shutdown"):
            return None
        raise ValueError(op)


@pytest.fixture()
def fake():
    rig = FakeRig()
    listener = Listener(("127.0.0.1", 0), authkey=protocol.authkey_from_env())
    port = listener.address[1]

    def serve():
        conn = listener.accept()
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break
            try:
                conn.send(protocol.ok(rig.handle(msg["op"], msg["args"]), 0.0))
            except Exception as exc:  # noqa: BLE001
                conn.send(protocol.error(str(exc)))
            if msg["op"] in ("close", "shutdown"):
                break
        conn.close()

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    robot = IsaacRobot({"port": port, "launch": False}, safety=SafetyConfig())
    yield robot, rig
    robot.close()
    listener.close()


def test_spec_camera_and_observation(fake):
    robot, rig = fake
    assert robot.spec.gripper_max_mm == pytest.approx(91.3)
    assert robot.spec.table_z == pytest.approx(-0.0095)
    assert robot.spec.tcp_offset[2] == pytest.approx(0.18)
    assert "RIGHT" in robot.spec.base_frame_doc and "AWAY from the camera" in robot.spec.base_frame_doc
    obs = robot.reset({"name": "pick_place", "params": {"nominal": True}}, seed=3)
    assert rig.calls[-1][1] == {"task": "waffle_pick_place", "seed": 3, "params": {"nominal": True}}
    assert robot.task_instruction == tasks.TASKS["waffle_pick_place"].instruction
    cam = obs.cameras["scene"]
    assert obs.images["scene"].shape == (480, 640, 3) and cam.K.shape == (3, 3) and cam.T_cam_base.shape == (4, 4)
    np.testing.assert_allclose(obs.state.q, tasks.START_Q)
    assert obs.state.gripper_mm == pytest.approx(90.0)


def test_unfiltered_ee_delta_resolved_and_timed(fake):
    robot, rig = fake
    robot.reset({"name": "reach"}, seed=0)
    p0 = robot.state().tcp_pos
    rep = robot.execute([Action(ActionMode.EE_DELTA, (0.03, 0.0, -0.04), gripper=0.0)])
    op, args = rig.calls[-1]
    assert op == "execute" and args["q"].shape == (1, 6)
    np.testing.assert_allclose(KIN.fk(args["q"][0])[0], p0 + [0.03, 0, -0.04], atol=1e-3)
    # 50 mm at the default 0.15 m/s
    assert args["durations"][0] >= 0.05 / 0.15 - 1e-9
    assert args["gripper"][0] == 0.0 and args["force_stop_n"] == 80.0
    np.testing.assert_allclose(rep.state_after.tcp_pos - rep.state_before.tcp_pos, [0.03, 0, -0.04], atol=1e-3)
    assert rep.executed[0].q_target is not None and not rep.stopped


def test_filtered_q_target_used_verbatim_and_events(fake):
    robot, rig = fake
    robot.reset({"name": "reach"}, seed=0)
    q1 = np.asarray(tasks.START_Q) + 0.05
    q2 = q1 + 0.05
    acts = [Action(ActionMode.JOINT_ABS, tuple(q1), q_target=tuple(q1)),
            Action(ActionMode.JOINT_ABS, tuple(q2), gripper=0.05, q_target=tuple(q2))]
    rig.next_contacts = {"arm-box": 12.0, "gripper-object": 3.0, "gripper-table": 0.5}
    rep = robot.execute(acts)
    args = rig.calls[-1][1]
    np.testing.assert_allclose(args["q"], [q1, q2])
    assert np.isnan(args["gripper"][0]) and args["gripper"][1] == 0.05
    kinds = {(e.kind, e.level) for e in rep.events}
    assert ("collision", EventLevel.WARN) in kinds and ("contact", EventLevel.INFO) in kinds
    assert not any("table" in e.message for e in rep.events)          # below the report threshold
    rig.stop = True
    rep = robot.execute(acts)
    assert rep.stopped and rep.events[0].level is EventLevel.STOP and len(rep.executed) == 1


def test_goal_and_hold(fake):
    robot, _ = fake
    robot.reset({"name": "push"}, seed=1)
    g = robot.check_goal()
    assert g.success and g.progress == 1.0 and g.metrics == {"x": 1}
    rep = robot.execute([])
    assert rep.executed == [] and rep.duration_s == 0.0
