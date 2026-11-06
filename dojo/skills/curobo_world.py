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
RECOVER_BUDGET_S = 10.0
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


def scene_boxes(kin, coll, arm, ignore, profile="legacy"):
    """Current Collision OBBs, including its container parts, in the planner base."""
    base_inv = inv_T(kin.base(arm))
    boxes = [([0., -.05, TABLE_Z - .025], np.eye(3), np.array([.7, .55, .025]), "table")]
    obstacles = coll.obstacles(ignore)
    if profile == "recover":
        from dojo.skills.plan_geometry import heightmap_boxes
        expanded, seen = [], set()
        for obstacle in obstacles:
            label = obstacle[3]
            hm = getattr(coll, "heightmaps", {}).get(label)
            if hm is not None and label not in getattr(coll, "parts", {}):
                if label not in seen:
                    expanded.extend(heightmap_boxes(coll.world.objects[label], hm,
                        coll.margin_of.get(label, coll.obj_margin)))
                    seen.add(label)
            else:
                expanded.append(obstacle)
        obstacles = expanded
    boxes += obstacles
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
    def __init__(self, profile="legacy", limits=None):
        self.profile = profile
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
        if profile == "recover":
            k["collision_link_names"].append("native_hand")
            k["extra_links"]["native_hand"] = {
                "link_name": "native_hand", "parent_link_name": "link6",
                "joint_name": "native_hand_fixed", "joint_type": "FIXED",
                "fixed_transform": [0., 0., 0., 1., 0., 0., 0.]}
            k["extra_collision_spheres"]["native_hand"] = SPHERES
            k["self_collision_ignore"]["native_hand"] = ["link6", "link5", "camera", "camera_base", "attached_object"]
            k["self_collision_ignore"]["attached_object"].append("native_hand")
        self.device = DeviceCfg()
        if profile == "recover":
            from curobo._src.types.robot import RobotCfg
            from dojo.skills.kinematics import JOINT_LIMITS
            robot = RobotCfg.create(robot, device_cfg=self.device)
            native = np.asarray(JOINT_LIMITS if limits is None else limits)
            joint_limits = robot.kinematics.get_joint_limits()
            requested = torch.as_tensor(native.T,device=self.device.device,dtype=joint_limits.position.dtype)
            # Tighten the GPU model before constructing every solver and cost.
            # URDF ±10-radian ranges are not the native working envelope.
            joint_limits.position[0].copy_(torch.maximum(joint_limits.position[0],requested[0]))
            joint_limits.position[1].copy_(torch.minimum(joint_limits.position[1],requested[1]))
        cfg = MotionPlannerCfg.create(
            robot=robot, scene_model={"cuboid": {"warmup_floor": {
                "dims": [.1, .1, .1], "pose": [0, 0, -5, 1, 0, 0, 0]}}},
            collision_cache={"cuboid": MAX_BOXES},
            device_cfg=self.device, num_ik_seeds=64 if profile == "recover" else 16, num_trajopt_seeds=1,
            self_collision_check=True, use_cuda_graph=False,
            position_tolerance=.002 if profile == "recover" else .001,
            orientation_tolerance=.03 if profile == "recover" else .02,
            optimizer_collision_activation_distance=.002 if profile == "recover" else .006)
        cfg.graph_planner_config.max_path_finding_iterations = 16 if profile == "recover" else 4
        cfg.graph_planner_config.min_finetune_iterations = 1
        self.planner = MotionPlanner(cfg)
        self.model_limits = self.planner.kinematics.config.kinematics_config.joint_limits.position.detach().cpu().numpy().T.copy()
        self.planner.warmup(enable_graph=True, num_warmup_iterations=1)
        # The deployed v2 MotionPlanner.attachment_manager property references
        # a missing TrajOptSolver property. Bind managers to its actual models.
        from curobo._src.collision.attachment_manager import AttachmentManager
        models = [self.planner.kinematics, self.planner.ik_solver.kinematics,
                  self.planner.graph_planner.kinematics]
        self.attachments = [AttachmentManager(model, self.planner.scene_collision_checker, self.device)
                            for model in models]
        self.emit = lambda event: None
        self.diagnostics = {}
        self.goal_hint = None
        self.ik_attempt = 0
        # Observe the deployed v2 stages without replacing their algorithms.
        for stage, owner, method in (("ik", self.planner.ik_solver, "solve_pose"),
                                     ("trajopt", self.planner.trajopt_solver, "solve_pose"),
                                     ("graph", self.planner.graph_planner, "find_path")):
            original = getattr(owner, method)
            def observed(*args, _original=original, _stage=stage, **kwargs):
                self.emit({"stage": _stage})
                if _stage == "ik" and self.profile == "recover":
                    import torch
                    from dojo.skills.plan_seeds import ik_seed_bank
                    count = self.planner.ik_solver.config.num_seeds
                    seeds=ik_seed_bank(self.query_q0,self.goal_hint,self.query_limits,count,self.ik_attempt)
                    self.ik_attempt+=1
                    kwargs["seed_config"] = torch.as_tensor(seeds,device=self.device.device,dtype=torch.float32)
                result = _original(*args, **kwargs)
                from dojo.skills.plan_diagnostics import result_flags
                self.diagnostics[_stage] = result_flags(result)
                if _stage == "ik":
                    self.diagnostics[_stage]["configured_position_tolerance"] = .002 if self.profile == "recover" else .001
                    self.diagnostics[_stage]["configured_orientation_tolerance"] = .03 if self.profile == "recover" else .02
                if _stage == "ik" and result is not None:
                    try:
                        best = self.planner.kinematics.get_active_js(result.js_solution).position.reshape(-1, self.planner.action_dim)[0]
                        self.diagnostics[_stage]["goal_joints"] = best.detach().cpu().tolist()
                        self.diagnostics[_stage]["goal_branch_delta"] = float(np.max(np.abs(np.asarray(self.diagnostics[_stage]["goal_joints"])-self.query_q0)))
                    except Exception as error:
                        self.diagnostics[_stage]["joint_diagnostic_error"] = str(error)
                if _stage == "ik" and result is not None and not any(self.diagnostics[_stage].get("success", [])):
                    try:
                        self.diagnostics[_stage]["constraints"] = self._constraints(result.js_solution)
                    except Exception as error:
                        self.diagnostics[_stage]["diagnostic_error"] = str(error)
                self.emit({"stage": _stage, "diagnostics": self.diagnostics})
                return result
            setattr(owner, method, observed)

    def _constraints(self, state):
        from curobo.types import JointState
        active = self.planner.kinematics.get_active_js(state)
        js = JointState.from_position(active.position.reshape(1, -1, self.planner.action_dim),
                                      joint_names=self.planner.joint_names)
        model = self.planner.compute_kinematics(js)
        out = {}
        rollout = self.planner.trajopt_solver.metrics_rollout
        # Query only collision terms: cspace costs require trajectory goal/dt
        # buffers and are not valid for a stand-alone configuration diagnostic.
        for manager in (rollout.metrics_constraint_manager, rollout.metrics_hybrid_cost_constraint_manager):
            if manager is None:
                continue
            manager.setup_batch_tensors(1, js.position.shape[1])
            for name in ("self_collision", "scene_collision"):
                if not manager.has_cost(name):
                    continue
                cost = manager.get_cost(name)
                data = cost.forward(model.robot_spheres) if name == "self_collision" else cost.forward(model)
                out[name] = float(data.detach().max().cpu())
        return out

    def propose(self, payload):
        from dojo.skills.plan_geometry import world_penetrations
        import torch
        from curobo.scene import Cuboid, Scene
        from curobo.types import GoalToolPose, JointState, Pose
        p = self.planner
        self.diagnostics = {}
        self.goal_hint = payload.get("goal_hint")
        self.ik_attempt = 0
        self.query_q0 = np.asarray(payload["q0"])
        self.query_limits = payload.get("limits")
        if self.profile == "recover":
            self.query_limits = np.asarray(self.query_limits).copy()
            self.query_limits[:,0]=np.maximum(self.query_limits[:,0],self.model_limits[:,0])
            self.query_limits[:,1]=np.minimum(self.query_limits[:,1],self.model_limits[:,1])
            self.diagnostics["model_limits"]=self.model_limits.tolist()
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
        if self.profile == "recover" and payload.get("hand") is not None:
            hand = torch.as_tensor(payload["hand"], dtype=torch.float32, device=device)
            for manager in self.attachments:
                manager.update(hand, state, link_name="native_hand")
        flange = payload["flange"]
        pose = Pose(position=torch.as_tensor(flange[:3], device=device, dtype=torch.float32).reshape(1, 3),
                    quaternion=torch.as_tensor(flange[3:], device=device,
                                               dtype=torch.float32).reshape(1, 4))
        goal = GoalToolPose.from_poses({"link6": pose}, ordered_tool_frames=list(p.tool_frames), num_goalset=1)
        # Use cuRobo's own collision/feasibility constraints for the start state.
        try:
            self.diagnostics["start"] = {"constraints": self._constraints(state)}
        except Exception as error:
            self.diagnostics["start"] = {"diagnostic_error": f"{type(error).__name__}: {error}"}
        self.emit({"stage": "start", "diagnostics": self.diagnostics})
        repair_path = None
        if self.profile == "recover" and payload.get("prefix") is not None:
            from dojo.skills.plan_repair import check_escape_prefix
            def prefix_state(q):
                return JointState.from_position(torch.as_tensor(q,device=device,dtype=full.dtype).reshape(1,-1),joint_names=names)
            def prefix_penetration(q):
                spheres=p.compute_kinematics(prefix_state(q)).robot_spheres.detach().cpu().numpy()
                return world_penetrations(spheres,payload["boxes"])
            valid, reason=check_escape_prefix(payload["prefix"],lambda q:self._constraints(prefix_state(q)),prefix_penetration)
            self.diagnostics["prefix"]={"valid":valid,"reason":reason}
            if not valid:
                return {"q":None,"diagnostics":self.diagnostics}
        if self.profile == "recover" and any(v > 0 for v in self.diagnostics.get("start", {}).get("constraints", {}).values()):
            from dojo.skills.plan_repair import repair_coordinates
            def joint_state(q):
                return JointState.from_position(torch.as_tensor(q,device=device,dtype=full.dtype).reshape(1,-1),joint_names=names)
            def evaluate(q):
                return self._constraints(joint_state(q))
            def penetrations(q):
                spheres=p.compute_kinematics(joint_state(q)).robot_spheres.detach().cpu().numpy()
                return world_penetrations(spheres,payload["boxes"])
            repair_path, repair = repair_coordinates(payload["q0"],self.query_limits,evaluate,penetrations)
            self.diagnostics["repair"] = repair
            self.emit({"stage":"repair","diagnostics":self.diagnostics})
            if repair_path is None:
                return {"q": None, "diagnostics": self.diagnostics}
            self.diagnostics["initial_start"] = self.diagnostics["start"]
            state=joint_state(repair_path[-1])
            self.diagnostics["start"]={"constraints":self._constraints(state)}
        p.reset_seed()
        result = p.plan_pose(goal_tool_poses=goal, current_state=state,
                             max_attempts=4 if self.profile == "recover" else 2, enable_graph_attempt=1)
        if result is None or result.success is None or not bool(result.success.any().item()):
            return {"q": None, "diagnostics": self.diagnostics}
        trajectory = result.get_interpolated_plan()
        q = trajectory.position.detach().cpu().numpy()
        while q.ndim > 2:
            q = q[0]
        trajectory_names = list(trajectory.joint_names)
        q=q[:,[trajectory_names.index(f"joint{i}") for i in range(1,7)]]
        if repair_path is not None:
            q=np.vstack([repair_path,q])
        return {"q":q,"diagnostics":self.diagnostics}



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


def _worker(connection, profile="legacy", limits=None):
    # Spawned without Isaac. Fatal CUDA errors can poison only this owned process.
    try:
        backend = _GPUBackend(profile, limits)
        backend.emit = lambda event: connection.send({"progress": event})
        connection.send({"ready": True})
        while True:
            payload = connection.recv()
            if payload is None:
                break
            connection.send(backend.propose(payload))
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
    def __init__(self, profile="legacy", limits=None):
        self.profile = profile
        self.limits = None if limits is None else np.asarray(limits).copy()
        self._close_lock = threading.Lock()
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, profile, limits), daemon=True)
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

    def propose(self, motion, coll, arm, q0, target, ignore, held, *, budget_s=None, gripper=1., prefix_q=None):
        flange = inv_T(motion.kin.base(arm)) @ motion.kin.flange_from_tcp(target)
        profile = getattr(motion, "plan_profile", "legacy")
        spheres = None
        if held is not None:
            if profile == "recover":
                from dojo.skills.plan_geometry import mesh_spheres
                guarded = getattr(held,"points",None) is None and bool(getattr(coll,"heightmaps",{}))
                spheres = mesh_spheres(held, motion.kin.tcp_offset, capacity=SPHERES-27 if guarded else SPHERES)
                if guarded and spheres is not None:
                    # Add, rather than replace, the native box samples used
                    # against height fields. The mesh surface remains covered.
                    corners=held.corners_w(T_from((motion.kin.tcp_offset,0,0)))
                    spheres=np.vstack([spheres,np.c_[corners,np.full(len(corners),.001)]])
            if spheres is None:
                spheres = attachment_spheres(held, motion.kin.tcp_offset)
        payload = dict(
            q0=np.asarray(q0).tolist(), boxes=scene_boxes(motion.kin, coll, arm, ignore, profile),
            spheres=None if spheres is None else spheres.tolist(),
            flange=np.r_[flange[:3, 3], R_to_quat(flange[:3, :3])].tolist())
        if prefix_q is not None:
            payload["prefix"]=np.asarray(prefix_q).tolist()
        if profile == "recover":
            from dojo.skills.plan_geometry import hand_spheres
            payload["hand"] = hand_spheres(motion.kin, gripper).tolist()
            # Warm-start cuRobo IK on the same nearby branch as native routing.
            # It remains only a seed: GPU collision checks and final validation apply.
            hint = motion.kin.ik(arm, target, q0, n_random=4)
            payload["goal_hint"] = hint.q.tolist() if hint.ok else None
            payload["limits"] = motion.kin.limits.tolist()
        if profile == "recover" and getattr(motion, "plan_capture", "off") == "requests":
            from dojo.skills.plan_diagnostics import native_goal_evidence
            try:
                payload["native_goal"] = native_goal_evidence(motion, coll, arm, hint, gripper, ignore, held)
            except Exception as error:
                payload["native_goal"] = {"diagnostic_error": f"{type(error).__name__}: {error}"}
        if getattr(motion, "plan_capture", "off") == "requests":
            payload["source_model"] = {"profile": profile, "ignore": list(ignore), "gripper": float(gripper),
                "heightmaps": list(getattr(coll, "heightmaps", {})), "parts": list(getattr(coll, "parts", {})),
                "held_label": getattr(held, "label", None)}
        self.last_payload = payload
        self.last_diagnostics = {}
        self.last_stage = None
        self.connection.send(payload)
        if budget_s is None:
            budget_s = RECOVER_BUDGET_S if profile == "recover" else BUDGET_S
        deadline = time.perf_counter() + budget_s
        while True:
            if not self.connection.poll(max(0., deadline-time.perf_counter())):
                self.close()
                error = TimeoutError(f"cuRobo planning budget ({budget_s:g} s)")
                error.diagnostics = self.last_diagnostics
                error.last_stage = self.last_stage
                raise error
            reply = self.connection.recv()
            if "progress" in reply:
                self.last_stage = reply["progress"]["stage"]
                self.last_diagnostics = reply["progress"].get("diagnostics", self.last_diagnostics)
                continue
            if "error" in reply:
                self.close()
                raise RuntimeError(reply["error"])
            self.last_diagnostics = reply.get("diagnostics", self.last_diagnostics)
            return reply["q"]


def plan_curobo(motion, arm, q0, target, step_rad, name, gripper, ignore, held, ease):
    from dojo.skills.motion import MoveResult, Segment
    q0, target = np.asarray(q0, float), np.asarray(target, float)
    profile = getattr(motion, "plan_profile", "legacy")
    budget = RECOVER_BUDGET_S if profile == "recover" else BUDGET_S
    if profile == "recover" and held is not None and getattr(held, "label", None):
        ignore = tuple(set(ignore) | {held.label})
    event = dict(arm=arm, name=name, solver="curobo", profile=profile, fallback=True, routes=1)
    started = time.perf_counter()
    q = None
    try:
        if profile == "recover":
            old_limits = getattr(getattr(motion, "_world_gpu", None), "limits", None)
            if old_limits is not None and not np.array_equal(old_limits,motion.kin.limits):
                motion.close()
        if not hasattr(motion, "_world_gpu"):
            init = time.perf_counter()
            try:
                motion._world_gpu = CuroboWorld(profile, motion.kin.limits.tolist() if profile == "recover" else None)
                event["worker_pid"] = motion._world_gpu.process.pid
                motion._world_gpu.wait_ready()
            except Exception as error:  # missing optional backend is a logged legacy fallback
                motion._world_gpu = f"{type(error).__name__}: {error}"
            event["init_s"] = round(time.perf_counter() - init, 6)
        started = time.perf_counter()
        if isinstance(motion._world_gpu, str):
            event["reason"] = motion._world_gpu
            event["cached"] = True
            event["cached_from"] = getattr(motion, "_world_gpu_failure_event", None)
            event["failure_category"] = getattr(motion, "_world_gpu_failure", "backend_error")
            return None
        process = getattr(motion._world_gpu, "process", None)
        if process is not None:
            event["worker_pid"] = process.pid
        coll = copy.copy(motion.coll)
        coll.include = set(coll.include) | {o.label for o in coll.world.objects.values() if o.obj_type == "Geometry"}
        q = motion._world_gpu.propose(motion, coll, arm, q0, target, ignore, held,
                                     budget_s=budget, gripper=gripper)
        attempts = [getattr(motion._world_gpu, "last_diagnostics", {})]
        if q is None and profile == "recover":
            from dojo.skills.plan_diagnostics import failure_category
            if failure_category(attempts[-1]) == "start_state_collision":
                # Retract through the original checked Cartesian primitive. No
                # partial prefix executes unless the entire joined path validates.
                from dojo.skills.motion import Motion
                pose = motion.kin.tcp(arm, q0)
                offsets = [np.array([0., 0., z]) for z in (.03, .06, .10)]
                offsets += [-pose[:3, 0]*d for d in (.03, .06)]
                inward = motion.kin.base(arm)[:3, 3]-pose[:3, 3]
                inward[2] = 0
                norm = np.linalg.norm(inward)
                if norm > 1e-6:
                    offsets += [inward/norm*d+np.array([0., 0., .02]) for d in (.03, .06)]
                for offset in offsets:
                    if time.perf_counter()-started >= budget:
                        break
                    lift = pose.copy(); lift[:3, 3] += offset
                    prefix = Motion(motion.kin, coll).line(arm, q0, lift, step_rad=step_rad,
                        name=name, gripper=gripper, ignore=ignore, held=held)
                    if not prefix.ok:
                        continue
                    remaining = budget-(time.perf_counter()-started)
                    if remaining <= 0:
                        break
                    tail = motion._world_gpu.propose(motion, coll, arm, prefix.q_end, target, ignore, held,
                                                    budget_s=remaining, gripper=gripper,
                                                    prefix_q=np.vstack([s.q for s in prefix.segments]))
                    attempts.append(getattr(motion._world_gpu, "last_diagnostics", {}))
                    if tail is not None:
                        q = np.vstack([s.q for s in prefix.segments]+[tail])
                        event["retract_offset"] = offset.tolist()
                        break
        event["attempts"] = attempts
        event["routes"] = len(attempts)
        event["diagnostics"] = getattr(motion._world_gpu, "last_diagnostics", {})
        event["last_stage"] = getattr(motion._world_gpu, "last_stage", None)
        if getattr(motion, "plan_capture", "off") == "requests":
            event["request"] = getattr(motion._world_gpu, "last_payload", None)
        if q is None:
            from dojo.skills.plan_diagnostics import failure_category
            event["failure_category"] = failure_category(event["diagnostics"])
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
                if time.perf_counter()-started > budget:
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
                    event["validation_q"] = qi.tolist()
                    if held is not None:
                        event["validation_held_only"] = coll.check(arm, qi, gripper, ignore=ignore, held=None,
                            obstacles=[o for o in obs if not near or o[3] not in touch])[0]
                    return None
        event.update(fallback=False, route="curobo")
        return MoveResult("reached", arm, [Segment(q, step_rad, name, ease)], goal)
    except Exception as error:
        event["diagnostics"] = getattr(error, "diagnostics", getattr(getattr(motion, "_world_gpu", None), "last_diagnostics", {}))
        if getattr(motion, "plan_capture", "off") == "requests":
            event["request"] = getattr(getattr(motion, "_world_gpu", None), "last_payload", None)
        event["failure_category"] = "timeout" if isinstance(error, TimeoutError) else "backend_error"
        event["last_stage"] = getattr(error, "last_stage", getattr(getattr(motion, "_world_gpu", None), "last_stage", None))
        motion._world_gpu_failure = event["failure_category"]
        motion._world_gpu_failure_event = len(motion.world_plans)
        event["reason"] = f"{type(error).__name__}: {error}"
        if profile == "recover" and isinstance(error, TimeoutError):
            # This owned worker was stopped; the next query starts a fresh one.
            # Never disable an entire episode because one query exhausted time.
            motion.close()
        else:
            motion._world_gpu = event["reason"]
        return None
    finally:
        if q is not None and getattr(motion, "plan_capture", "off") == "requests":
            event["proposal_q"] = np.asarray(q).tolist()
        if event["fallback"] and "failure_category" not in event:
            event["failure_category"] = "validation_reject"
        event["seconds"] = round(time.perf_counter()-started, 6)
        motion.world_plans.append(event)
