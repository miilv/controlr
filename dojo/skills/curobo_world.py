"""Optional cuRobo v2 proposer. Inputs are belief geometry, never the eval env.

cuRobo paths are proposals: the skill Collision model and current joint limits
validate every path before existing Segment/Executor rate rules can execute it.
"""
from __future__ import annotations

import copy
import itertools
import multiprocessing as mp
import os
from pathlib import Path
import time
import threading

import numpy as np

from dojo.skills.geometry import R_to_quat, T_from, angle_between_R, inv_T
from dojo.skills.kinematics import TABLE_Z

BUDGET_S = 2.0
SPHERES = 256
MAX_BOXES = 2048


def attachment_spheres(held, tcp_offset):
    """Cover the held OBB with spheres, in link6 coordinates (including TCP offset)."""
    half = np.maximum(np.asarray(held.half), .0001)
    cell = .015
    while True:
        counts = np.maximum(1, np.ceil(2 * half / cell).astype(int))
        if int(np.prod(counts)) <= SPHERES:
            break
        cell *= 1.25
    sizes = 2 * half / counts
    axes = [np.linspace(-h + d/2, h - d/2, n) for h, d, n in zip(half, sizes, counts)]
    centers = np.array(list(itertools.product(*axes))) + held.center
    transform = T_from((tcp_offset, 0, 0)) @ held.T_tcp_obj
    centers = centers @ transform[:3, :3].T + transform[:3, 3]
    radius = float(np.linalg.norm(sizes / 2)) + .006
    return np.c_[centers, np.full(len(centers), radius)]


def scene_boxes(kin, coll, arm, ignore):
    """Current Collision OBBs, including its container parts, in the planner base."""
    base_inv = inv_T(kin.base(arm))
    boxes = [([0., -.05, TABLE_Z - .025], np.eye(3), np.array([.7, .55, .025]), "table")]
    boxes += coll.obstacles(ignore)
    result = []
    for index, (center, rotation, half, label) in enumerate(boxes):
        center = base_inv[:3, :3] @ np.asarray(center) + base_inv[:3, 3]
        rotation = base_inv[:3, :3] @ rotation
        result.append(dict(name=f"obstacle_{index}", dims=(2*np.asarray(half)).tolist(),
                           pose=np.r_[center, R_to_quat(rotation)].tolist()))
    if len(result) > MAX_BOXES:
        raise ValueError(f"world has {len(result)} boxes (capacity {MAX_BOXES})")
    return result


class _GPUBackend:
    def __init__(self):
        # All optional/GPU imports stay behind the default-off switch.
        import yaml
        import torch
        torch.set_num_threads(2)
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import DeviceCfg
        home = Path(os.environ.get("RD_HOME", "/root/controlr-robodojo"))
        path = Path(os.environ.get("RD_ROOT", str(home / "RoboDojo")))
        raw = yaml.safe_load((path / "Assets/Robots/x5/curobo.yml").read_text())
        robot = robot_config(raw)
        k = robot["robot_cfg"]["kinematics"]
        # Rendering mesh links must not become additional IK tool targets.
        k.pop("mesh_link_names", None)
        # The fixed mounting base intersects the table by design. The skill
        # validator still checks the arm/table model used before this proposer.
        k["collision_link_names"] = [x for x in k["collision_link_names"] if x != "base_link"]
        k["self_collision_ignore"] = {name: [x for x in ignores if x != "base_link"]
                                       for name, ignores in k["self_collision_ignore"].items()
                                       if name != "base_link"}
        k["collision_link_names"].append("attached_object")
        k["extra_links"] = {"attached_object": {
            "link_name": "attached_object", "parent_link_name": "link6",
            "joint_name": "attachment_fixed", "joint_type": "FIXED",
            "fixed_transform": [0., 0., 0., 1., 0., 0., 0.]}}
        k["extra_collision_spheres"] = {"attached_object": SPHERES}
        k.setdefault("self_collision_ignore", {})["attached_object"] = ["link6", "link5", "camera", "camera_base"]
        self.device = DeviceCfg()
        cfg = MotionPlannerCfg.create(
            robot=robot, scene_model={"cuboid": {"warmup_floor": {
                "dims": [.1, .1, .1], "pose": [0, 0, -5, 1, 0, 0, 0]}}},
            collision_cache={"cuboid": MAX_BOXES},
            device_cfg=self.device, num_ik_seeds=16, num_trajopt_seeds=1,
            self_collision_check=True, use_cuda_graph=False,
            position_tolerance=.001, orientation_tolerance=.02,
            optimizer_collision_activation_distance=.006)
        cfg.graph_planner_config.max_path_finding_iterations = 4
        cfg.graph_planner_config.min_finetune_iterations = 1
        self.planner = MotionPlanner(cfg)
        self.planner.warmup(enable_graph=True, num_warmup_iterations=1)
        # The deployed v2 MotionPlanner.attachment_manager property references
        # a missing TrajOptSolver property. Bind managers to its actual models.
        from curobo._src.collision.attachment_manager import AttachmentManager
        models = [self.planner.kinematics, self.planner.ik_solver.kinematics,
                  self.planner.graph_planner.kinematics]
        self.attachments = [AttachmentManager(model, self.planner.scene_collision_checker, self.device)
                            for model in models]

    def propose(self, payload):
        import torch
        from curobo.scene import Cuboid, Scene
        from curobo.types import GoalToolPose, JointState, Pose
        p = self.planner
        device = self.device.device
        names = list(p.joint_names)
        active = [names.index(f"joint{i}") for i in range(1, 7)]
        full = p.default_joint_state.position.detach().reshape(-1).clone()
        full[active] = torch.as_tensor(payload["q0"], device=device, dtype=full.dtype)
        state = JointState.from_position(full.reshape(1, -1), joint_names=names)
        for manager in self.attachments:
            manager.detach("attached_object")
        p.update_world(Scene(cuboid=[Cuboid(**b) for b in payload["boxes"]]))
        if payload["spheres"] is not None:
            spheres = torch.as_tensor(payload["spheres"], dtype=torch.float32, device=device)
            # Already link-local: do not run FK against any simulator pose.
            for manager in self.attachments:
                manager.update(spheres, state, link_name="attached_object")
        flange = payload["flange"]
        pose = Pose(position=torch.as_tensor(flange[:3], device=device, dtype=torch.float32).reshape(1, 3),
                    quaternion=torch.as_tensor(flange[3:], device=device,
                                               dtype=torch.float32).reshape(1, 4))
        goal = GoalToolPose.from_poses({"link6": pose}, ordered_tool_frames=list(p.tool_frames), num_goalset=1)
        p.reset_seed()
        result = p.plan_pose(goal_tool_poses=goal, current_state=state,
                             max_attempts=2, enable_graph_attempt=1)
        if result is None or result.success is None or not bool(result.success.any().item()):
            return None
        trajectory = result.get_interpolated_plan()
        q = trajectory.position.detach().cpu().numpy()
        while q.ndim > 2:
            q = q[0]
        trajectory_names = list(trajectory.joint_names)
        return q[:, [trajectory_names.index(f"joint{i}") for i in range(1, 7)]]



def robot_config(raw):
    """Public RoboDojo robot YAML -> cuRobo v2 schema (same conversion as its wrapper)."""
    k = copy.deepcopy(raw["robot_cfg"]["kinematics"])
    k["format_version"] = 2.0
    k["tool_frames"] = k.get("tool_frames", [k.get("ee_link", "link6")])
    c = k["cspace"]
    if "retract_config" in c:
        c["default_joint_position"] = c.pop("retract_config")
    cspace_keys = {"joint_names", "default_joint_position", "cspace_distance_weight", "null_space_weight",
                   "null_space_maximum_distance", "max_acceleration", "max_jerk", "velocity_scale",
                   "acceleration_scale", "jerk_scale", "position_limit_clip"}
    k["cspace"] = {key: value for key, value in c.items() if key in cspace_keys}
    keys = {"format_version", "base_link", "tool_frames", "collision_link_names", "collision_spheres",
            "collision_sphere_buffer", "extra_collision_spheres", "self_collision_ignore", "self_collision_buffer",
            "asset_root_path", "mesh_link_names", "lock_joints", "extra_links", "cspace", "urdf_path",
            "use_global_cumul", "grasp_contact_link_names"}
    return {"robot_cfg": {"kinematics": {key: value for key, value in k.items() if key in keys}},
            "load_dynamics": bool(raw.get("load_dynamics", False))}


def _worker(connection):
    # Spawned without Isaac. Fatal CUDA errors can poison only this owned process.
    try:
        backend = _GPUBackend()
        connection.send({"ready": True})
        while True:
            payload = connection.recv()
            if payload is None:
                break
            connection.send({"q": backend.propose(payload)})
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        import traceback
        traceback.print_exc()
        try:
            connection.send({"error": f"{type(error).__name__}: {error}"})
        except (EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()
        # Do not let a poisoned CUDA context hang in library destructors.
        os._exit(0)


class CuroboWorld:
    """A single owned GPU worker; startup is separate from the 2 s request budget."""
    def __init__(self):
        self._close_lock = threading.Lock()
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child,), daemon=True)
        self.process.start()
        child.close()

    def wait_ready(self):
        try:
            if not self.connection.poll(60):
                raise TimeoutError("cuRobo worker startup budget (60 s)")
            reply = self.connection.recv()
            if not reply.get("ready"):
                raise RuntimeError(reply.get("error", "cuRobo worker failed"))
        except Exception:
            self.close()
            raise

    def close(self):
        lock = getattr(self, "_close_lock", None)
        if lock is None:
            return
        with lock:
            process = getattr(self, "process", None)
            if process is not None and process.pid is not None:
                if process.is_alive():
                    process.terminate()
                    process.join(.25)
                    if process.is_alive():
                        process.kill()
                process.join(.25)
            connection = getattr(self, "connection", None)
            if connection is not None:
                connection.close()

    def __del__(self):
        self.close()

    def propose(self, motion, coll, arm, q0, target, ignore, held):
        flange = inv_T(motion.kin.base(arm)) @ motion.kin.flange_from_tcp(target)
        self.connection.send(dict(
            q0=np.asarray(q0), boxes=scene_boxes(motion.kin, coll, arm, ignore),
            spheres=None if held is None else attachment_spheres(held, motion.kin.tcp_offset),
            flange=np.r_[flange[:3, 3], R_to_quat(flange[:3, :3])]))
        if not self.connection.poll(BUDGET_S):
            self.close()
            raise TimeoutError("cuRobo planning budget (2 s)")
        reply = self.connection.recv()
        if "error" in reply:
            self.close()
            raise RuntimeError(reply["error"])
        return reply["q"]


def plan_curobo(motion, arm, q0, target, step_rad, name, gripper, ignore, held, ease):
    from dojo.skills.motion import MoveResult, Segment
    q0, target = np.asarray(q0, float), np.asarray(target, float)
    event = dict(arm=arm, name=name, solver="curobo", fallback=True, routes=1)
    started = time.perf_counter()
    try:
        if not hasattr(motion, "_world_gpu"):
            init = time.perf_counter()
            try:
                motion._world_gpu = CuroboWorld()
                motion._world_gpu.wait_ready()
            except Exception as error:  # missing optional backend is a logged legacy fallback
                motion._world_gpu = f"{type(error).__name__}: {error}"
            event["init_s"] = round(time.perf_counter() - init, 6)
        if isinstance(motion._world_gpu, str):
            event["reason"] = motion._world_gpu
            return None
        started = time.perf_counter()
        process = getattr(motion._world_gpu, "process", None)
        if process is not None:
            event["worker_pid"] = process.pid
        coll = copy.copy(motion.coll)
        coll.include = set(coll.include) | {o.label for o in coll.world.objects.values() if o.obj_type == "Geometry"}
        q = motion._world_gpu.propose(motion, coll, arm, q0, target, ignore, held)
        if q is None:
            event["reason"] = "no_plan"
            return None
        q = np.vstack([q0, np.asarray(q).reshape(-1, 6)])
        if not np.isfinite(q).all() or np.any(q < motion.kin.limits[:, 0]) or np.any(q > motion.kin.limits[:, 1]):
            event["reason"] = "joint_limits"
            return None
        goal = motion.kin.tcp(arm, q[-1])
        if np.linalg.norm(goal[:3, 3]-target[:3, 3]) > .003 or angle_between_R(goal[:3, :3], target[:3, :3]) > .035:
            event["reason"] = "endpoint"
            return None
        # Recheck every <=0.04 rad of joint travel, including every cuRobo knot.
        # Keep the legacy initial-contact escape allowance for only the first 5 cm.
        touch = coll.touching(arm, q0, gripper, ignore, held)
        obs = coll.obstacles(ignore)
        start_pose = motion.kin.tcp(arm, q0)
        for a, b in zip(q[:-1], q[1:]):
            count = max(1, int(np.ceil(np.max(np.abs(b-a))/.04)))
            for t in np.linspace(0, 1, count+1)[1:]:
                if time.perf_counter()-started > BUDGET_S:
                    event["reason"] = "budget"
                    return None
                qi = a+t*(b-a)
                pose = motion.kin.tcp(arm, qi)
                near = (np.linalg.norm(pose[:3, 3]-start_pose[:3, 3]) < .05
                        and angle_between_R(pose[:3, :3], start_pose[:3, :3]) < np.deg2rad(20))
                ok, reason = coll.check(arm, qi, gripper, ignore=ignore, held=held,
                                       obstacles=[o for o in obs if not near or o[3] not in touch])
                if not ok:
                    event["reason"] = f"validation: {reason}"
                    return None
        event.update(fallback=False, route="curobo")
        return MoveResult("reached", arm, [Segment(q, step_rad, name, ease)], goal)
    except Exception as error:
        motion._world_gpu = f"{type(error).__name__}: {error}"
        event["reason"] = motion._world_gpu
        return None
    finally:
        event["seconds"] = round(time.perf_counter()-started, 6)
        motion.world_plans.append(event)
