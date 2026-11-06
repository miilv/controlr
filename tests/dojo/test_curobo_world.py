from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dojo.skills.curobo_world import attachment_spheres, plan_curobo, scene_boxes
from dojo.skills.geometry import T_from
from dojo.skills.kinematics import X5Kin
from dojo.skills.motion import Collision, HeldBox, Motion


def test_attachment_spheres_cover_box_and_follow_tcp_offset():
    held = HeldBox(T_from((.02, .03, .04)), np.zeros(3), np.array([.10, .02, .01]))
    spheres = attachment_spheres(held, .145)
    assert len(spheres) <= 256
    assert np.allclose(spheres[:, :3].mean(0), [.165, .03, .04])
    for x in np.linspace(-.1, .1, 9):
        for y in np.linspace(-.02, .02, 5):
            for z in np.linspace(-.01, .01, 5):
                point = np.array([x+.165, y+.03, z+.04])
                assert np.any(np.linalg.norm(spheres[:, :3]-point, axis=1) <= spheres[:, 3])


def test_scene_uses_planner_base_for_both_table_and_belief_boxes():
    kin = X5Kin()
    center = np.array([.1, .2, .9])
    coll = SimpleNamespace(obstacles=lambda ignore: [(center, np.eye(3), np.ones(3)*.02, "belief")])
    for arm in ("left", "right"):
        boxes = scene_boxes(kin, coll, arm, ())
        b = kin.base(arm)
        got = b[:3, :3] @ np.array(boxes[1]["pose"][:3]) + b[:3, 3]
        assert np.allclose(got, center)
        assert np.allclose(boxes[1]["dims"], [.04]*3)
        assert boxes[0]["pose"][2] == pytest.approx(-.025)


@pytest.mark.parametrize("bad", ["collision", "limits", "nan", "endpoint", None])
def test_gpu_proposal_is_validated_before_segment_creation(bad):
    kin = X5Kin()
    q0 = np.array([0., 1., 1., 0., 0., 0.])
    q1 = q0.copy()
    q1[0] += .1
    target = kin.tcp("left", q1)
    proposed = np.stack([q0, q1])
    if bad == "limits":
        proposed[-1, 0] = 5.
    if bad == "nan":
        proposed[-1, 0] = np.nan
    if bad == "endpoint":
        proposed[-1] = q0
    coll = Collision(kin, SimpleNamespace(objects={}))
    coll.touching = Mock(return_value=set())
    coll.obstacles = Mock(return_value=[])
    coll.check = Mock(return_value=(bad != "collision", "test obstacle"))
    motion = Motion(kin, coll)
    motion._world_gpu = SimpleNamespace(propose=Mock(return_value=proposed))
    held = object()
    result = plan_curobo(motion, "left", q0, target, .05, "test", 0., ("tool",), held, 2)
    if bad is not None:
        assert result is None and motion.world_plans[-1]["fallback"]
    else:
        assert result.ok and result.segments[0].step_rad == .05 and result.segments[0].ease == 2
        assert coll.check.call_args.kwargs["held"] is held


def test_optional_backend_failure_is_cached_and_falls_back(monkeypatch):
    from dojo.skills import curobo_world
    motion = Motion(X5Kin())
    backend = Mock(side_effect=ImportError("cuRobo unavailable"))
    monkeypatch.setattr(curobo_world, "CuroboWorld", backend)
    for _ in range(2):
        assert plan_curobo(motion, "left", np.zeros(6), np.eye(4), .05, "test", 1., (), None, 2) is None
    assert backend.call_count == 1
    assert all(p["fallback"] and "unavailable" in p["reason"] for p in motion.world_plans)


def test_config_and_dispatch(monkeypatch):
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills import curobo_world, world_plan
    assert RuntimeConfig().plan_solver == "routes"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(plan_solver="bad"))
    motion = Motion()
    motion.plan_solver = "curobo"
    expected = object()
    monkeypatch.setattr(curobo_world, "plan_curobo", Mock(return_value=expected))
    assert world_plan.plan_objects(motion, "left", np.zeros(6), np.eye(4), .05,
                                   "test", 1., (), None, 2) is expected


def test_worker_request_timeout_closes_only_owned_worker():
    from dojo.skills.curobo_world import CuroboWorld
    client = object.__new__(CuroboWorld)
    client.connection = SimpleNamespace(send=Mock(), poll=Mock(return_value=False))
    client.close = Mock()
    motion = Motion(X5Kin())
    coll = SimpleNamespace(obstacles=lambda ignore: [])
    with pytest.raises(TimeoutError):
        client.propose(motion, coll, "left", np.zeros(6), np.eye(4), (), None)
    client.close.assert_called_once()
    payload = client.connection.send.call_args.args[0]
    assert set(payload) == {"q0", "boxes", "spheres", "flange"}
    assert 0 < client.connection.poll.call_args.args[0] <= 2.0


def test_robot_schema_conversion_preserves_source_and_drops_legacy_keys():
    from dojo.skills.curobo_world import robot_config
    raw = {"robot_cfg": {"kinematics": {"ee_link": "link6", "add_object_link": False,
           "cspace": {"joint_names": ["joint1"], "retract_config": [0.], "old_key": 10}}}}
    config = robot_config(raw)["robot_cfg"]["kinematics"]
    assert config["tool_frames"] == ["link6"] and config["format_version"] == 2.
    assert config["cspace"] == {"joint_names": ["joint1"], "default_joint_position": [0.]}
    assert "add_object_link" not in config and "ee_link" not in config
    assert "retract_config" in raw["robot_cfg"]["kinematics"]["cspace"]


def test_motion_close_releases_owned_worker_once():
    m = Motion()
    worker = SimpleNamespace(close=Mock())
    m._world_gpu = worker
    m.close()
    m.close()
    worker.close.assert_called_once()


def test_runtime_cleanup_closes_retained_motion_instances():
    from dojo.runtime.policy import RuntimePolicy
    policy = object.__new__(RuntimePolicy)
    closers = [Mock(), Mock()]
    policy._live = {"sks": [SimpleNamespace(motion=SimpleNamespace(close=closer)) for closer in closers]}
    policy._close_motion_workers()
    assert all(closer.call_count == 1 for closer in closers)


def test_contact_escape_does_not_exempt_a_large_in_place_rotation():
    kin = X5Kin()
    q0 = np.array([0., 1., 1., 0., 0., 0.])
    q1 = q0.copy()
    q1[-1] = .6  # roll about the tool axis, almost no TCP translation
    coll = Collision(kin, SimpleNamespace(objects={}))
    coll.touching = Mock(return_value={"reference"})
    coll.obstacles = Mock(return_value=[(np.zeros(3), np.eye(3), np.ones(3), "reference")])
    coll.check = Mock(return_value=(True, ""))
    motion = Motion(kin, coll)
    motion._world_gpu = SimpleNamespace(propose=Mock(return_value=np.stack([q0, q1])))
    assert plan_curobo(motion, "left", q0, kin.tcp("left", q1), .05,
                       "rotate", 0., (), None, 2).ok
    calls = coll.check.call_args_list
    assert not calls[0].kwargs["obstacles"]
    assert calls[-1].kwargs["obstacles"]  # full checks resume beyond 20 degrees


def test_failed_startup_is_not_double_counted_as_query_time(monkeypatch):
    from dojo.skills import curobo_world
    worker = SimpleNamespace(process=SimpleNamespace(pid=123),
                             wait_ready=Mock(side_effect=RuntimeError("startup failed")))
    monkeypatch.setattr(curobo_world, "CuroboWorld", Mock(return_value=worker))
    monkeypatch.setattr(curobo_world.time, "perf_counter", Mock(side_effect=[0., 1., 11., 11., 12.]))
    motion = Motion()
    assert plan_curobo(motion, "left", np.zeros(6), np.eye(4), .05,
                       "test", 1., (), None, 2) is None
    assert motion.world_plans[-1]["init_s"] == 10.
    assert motion.world_plans[-1]["seconds"] == 1.
    assert motion.world_plans[-1]["worker_pid"] == 123


def test_rotated_attachment_uses_object_rotation_after_tcp_translation():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    held = HeldBox(T_from((.02, .03, .04), rotation),
                   np.array([.01, 0., 0.]), np.array([.10, .02, .01]))
    spheres = attachment_spheres(held, .145)
    # Independent 90-degree-frame oracle: object center (1cm,0,0)
    # becomes (0,1cm,0), then object-root and native TCP translations.
    assert np.allclose(spheres[:, :3].mean(0), [.165, .04, .04])
    corner = np.array([.145, .14, .05])
    assert np.any(np.linalg.norm(spheres[:, :3]-corner, axis=1) <= spheres[:, 3])


def test_asymmetric_world_box_orientation_is_transformed_to_each_arm_base():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    coll = SimpleNamespace(obstacles=lambda ignore: [
        (np.array([.1, .2, .9]), rotation, np.array([.03, .015, .07]), "holder")])
    for arm, expected in (("left", [.65, -.4, .135]), ("right", [.65, .2, .135])):
        box = scene_boxes(X5Kin(), coll, arm, ())[1]
        assert np.allclose(box["pose"][:3], expected)
        assert abs(box["pose"][3]) == pytest.approx(1.)
        assert np.allclose(box["pose"][4:], 0., atol=1e-8)
        assert np.allclose(box["dims"], [.06, .03, .14])


@pytest.mark.parametrize("evidence, expected", [
    ({"start": {"constraints": {"robot_world_collision": .002}}}, "start_state_collision"),
    ({"ik": {"success": [False], "position_error": [.0005], "rotation_error": [.001],
             "constraints": {"robot_world_collision": .002}}}, "goal_collision"),
    ({"ik": {"success": [False], "position_error": [.02]}}, "ik_fail"),
    ({"ik": {"success": [True]}, "graph": {"success": [False]}}, "graph_fail"),
    ({"ik": {"success": [True]}, "trajopt": {"success": [False]}}, "trajopt_fail"),
])
def test_stage_status_uses_actual_flags(evidence, expected):
    from dojo.skills.plan_diagnostics import failure_category
    assert failure_category(evidence) == expected


def test_progress_does_not_restart_request_deadline(monkeypatch):
    from dojo.skills import curobo_world
    client = object.__new__(curobo_world.CuroboWorld)
    client.close = Mock()
    client.connection = SimpleNamespace(send=Mock(), poll=Mock(side_effect=[True, False]),
        recv=Mock(return_value={"progress": {"stage": "ik", "diagnostics": {"ik": {"success": [False]}}}}))
    monkeypatch.setattr(curobo_world.time, "perf_counter", Mock(side_effect=[0., .1, 1.5]))
    with pytest.raises(TimeoutError) as raised:
        client.propose(Motion(), SimpleNamespace(obstacles=lambda ignore: []),
                       "left", np.zeros(6), np.eye(4), (), None)
    assert raised.value.diagnostics == {"ik": {"success": [False]}}
    assert [c.args[0] for c in client.connection.poll.call_args_list] == [1.9, .5]
