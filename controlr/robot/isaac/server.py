"""Isaac Sim 6.0 server: PHANTOM's calibrated UR3 CB3 + Robotiq/W2L + D435 rig,
driven turn by turn over ``multiprocessing.connection``.

Runs ONLY inside Isaac's python (``scripts/isaac_server.sh``). The stage is
built once with PHANTOM's own builders (``phantom.sim.scene.import_robot`` /
``build_scene``, the measured camera from the scene config, PHANTOM's contact
observers) and then reused: ``reset`` moves the packet / markers and teleports
the arm instead of rebuilding, so the ~1 min Kit + URDF + shader startup is
paid once per server process.

Turn-based physics: nothing steps between requests. ``execute`` interpolates
joint targets (min-jerk profile) through PHANTOM's position drives, actuates
the gripper after the arm segment, steps until settled (joint speed threshold
or timeout), then reports contacts / tracking / holding. ``observe`` renders
the D435 view without stepping physics.

PHANTOM is imported, never modified (``--phantom`` repo root on sys.path).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Inside Isaac's python controlr is not installed: put the repo root on sys.path so the
# numpy-only modules (protocol, tasks, motion, and the kinematics/spec they use) import
# as ``controlr.*`` — one copy of each, same as in the client.
if str(HERE.parents[2]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[2]))
from controlr.robot.isaac import motion  # noqa: E402
from controlr.robot.isaac import protocol  # noqa: E402
from controlr.robot.isaac import tasks as task_registry  # noqa: E402

TCP_OFFSET_M = 0.18      # PHANTOM tcp: tool0 + 180 mm along tool z (configs/hardware.yaml)
# Bound of the gravity-compensating integral term (see _integrate). At PHANTOM's drive
# stiffness (3500 N m/rad) 0.004 rad is ~14 N m per joint: enough for the ~3 mrad elbow
# sag, far below the 0.01 rad (~35 N m, ~90 N at the TCP) that could push into an obstacle.
BIAS_LIMIT_RAD = 0.004
ANTI_WINDUP_N = 5.0         # no integration while robot-environment contact exceeds this
FORCE_STOP_RISE_N = 20.0    # a force stop needs this much more force than before the motion
UNSTABLE_FORCE_N = 5000.0   # no real contact on this rig gets near this: the solver blew up


def log(*parts) -> None:
    print("[controlr-isaac]", *parts, flush=True)


class IsaacRig:
    """One built stage + handles. Methods map 1:1 to protocol ops."""

    def __init__(self, app, phantom: Path, scene_rel: str, out: Path, *, solver_iterations: int | None = None):
        t_start = time.perf_counter()
        self.app, self.phantom, self.out = app, phantom, out
        self.scene_rel = scene_rel
        cfg = json.loads((phantom / scene_rel).read_text())
        if solver_iterations:
            cfg["physics"]["solver_position_iterations"] = int(solver_iterations)
        self.cfg = cfg

        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim, SingleArticulation, SingleRigidPrim
        from isaacsim.sensors.camera import Camera
        from pxr import PhysxSchema, UsdPhysics
        from scipy.spatial.transform import Rotation

        from phantom.sim.camera import configure_camera_intrinsics, validate_camera_intrinsics
        from phantom.sim.geometry import bin_geometry
        from phantom.sim.gripper_articulation import (
            closure_from_joint_positions, drive_targets, finger_joint_names, is_adaptive,
            joint_targets)
        from phantom.sim.kinematics import JOINT_NAMES, forward_pose
        from phantom.sim.scene import build_scene, import_robot
        from phantom.sim.task_objects import object_config
        from tools.sim.object_contacts import PAD_OBJECT_CONTACT_CAPACITY
        from tools.sim.robot_environment_contacts import RobotEnvironmentContactViews

        self.Rotation, self.forward_pose = Rotation, forward_pose
        self._drive_targets, self._joint_targets = drive_targets, joint_targets
        self._closure_from_q = closure_from_joint_positions
        if is_adaptive(cfg):
            from phantom.sim.gripper_adaptive import validate_physics_timestep
            validate_physics_timestep(cfg)
        self.dt = float(cfg["physics"]["dt"])
        robot_usd = import_robot(phantom, out, cfg)
        t_import = time.perf_counter()
        self.world = World(stage_units_in_meters=1, physics_dt=self.dt, rendering_dt=self.dt)
        self.stage = omni.usd.get_context().get_stage()
        self.paths = paths = build_scene(self.stage, phantom, out, cfg, robot_usd)
        self._build_markers()
        roots = [str(p.GetPath()) for p in self.stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(roots) != 1:
            raise RuntimeError(f"expected one articulation root, found {roots}")
        art = PhysxSchema.PhysxArticulationAPI.Apply(self.stage.GetPrimAtPath(roots[0]))
        art.CreateSolverPositionIterationCountAttr(cfg["physics"]["solver_position_iterations"])
        art.CreateSolverVelocityIterationCountAttr(cfg["physics"]["solver_velocity_iterations"])
        art.CreateEnabledSelfCollisionsAttr(True)
        root = roots[0]
        if self.stage.GetPrimAtPath(root).IsA(UsdPhysics.Joint):
            root = paths["robot_path"]
        self.robot = self.world.scene.add(SingleArticulation(prim_path=root, name="ur3"))
        self.packet = self.world.scene.add(SingleRigidPrim(prim_path=paths["object_path"], name="object"))
        self.tool = SingleRigidPrim(prim_path=paths["tool_path"], name="tool_feedback",
                                    reset_xform_properties=False)
        self.pads = [RigidPrim(prim_paths_expr=p, name=f"pad_{i}", track_contact_forces=True,
                               contact_filter_prim_paths_expr=[paths["object_path"]],
                               max_contact_count=PAD_OBJECT_CONTACT_CAPACITY, reset_xform_properties=False)
                     for i, p in enumerate(paths["pad_paths"])]
        robot_bodies = [str(p.GetPath()) for p in self.stage.Traverse()
                        if p.HasAPI(UsdPhysics.RigidBodyAPI) and str(p.GetPath()).startswith(paths["robot_path"] + "/")]
        self.contacts = RobotEnvironmentContactViews(robot_bodies, environment_paths=paths["environment_paths"],
                                                     rigid_prim_cls=RigidPrim)
        gripper_root = paths["gripper_housing_path"]
        self.body_group = {p: ("gripper" if p.startswith(gripper_root) or p in paths["pad_paths"] else "arm")
                           for p in robot_bodies}
        self.env_group = {}
        for p in paths["environment_paths"]:
            self.env_group[p] = ("object" if p == paths["object_path"] else "box" if p.startswith("/World/Bin")
                                 else "table")

        cam = cfg["camera"]
        self.camera = Camera("/World/SceneCamera", frequency=15, resolution=tuple(cam["resolution"]))
        twc = np.asarray(cam["world_from_cv"], float)
        orient = Rotation.from_matrix(twc[:3, :3] @ np.diag([1, -1, -1])).as_quat()
        self.camera.set_world_pose(position=twc[:3, 3], orientation=orient[[3, 0, 1, 2]], camera_axes="usd")
        configure_camera_intrinsics(self.camera, cam)
        self.camera.set_clipping_range(0.02, 10)
        self.camera.set_focus_distance(1.0)
        self.world.reset()
        for pad in self.pads:
            pad.initialize()
        self.contacts.initialize()
        self.tool.initialize()
        self.camera.initialize()
        # Turn-based: frames are requested explicitly after settling, never on a
        # sim-time cadence. Camera's own rate limiter counts *simulation* time
        # between renders (none passes between two observes), so disable it:
        # -1 is the class's "acquire on every rendered frame" state.
        self.camera._frequency = -1
        self.camera_report = validate_camera_intrinsics(self.camera, cam)
        from phantom.sim.camera import camera_projection
        self.K = camera_projection(cam).matrix
        self.T_cam_base = np.linalg.inv(twc)          # world frame == UR controller base frame
        self.resolution = tuple(int(v) for v in cam["resolution"])

        names = list(self.robot.dof_names)
        self.ids = np.array([names.index(n) for n in JOINT_NAMES])
        self.finger_names = finger_joint_names(cfg)
        self.fingers = np.array([names.index(n) for n in self.finger_names])
        self.n_dof = len(names)
        self.obj_cfg = object_config(cfg)
        bg = bin_geometry(cfg["bin"])
        lower, upper = bg.interior_bounds
        wall = 0.02
        if "outer_size" in cfg["bin"] and "opening_size" in cfg["bin"]:
            wall = float(np.min(np.asarray(cfg["bin"]["outer_size"][:2], float)
                                - np.asarray(cfg["bin"]["opening_size"], float)) / 2)
        self.bin_info = {"lower": lower, "upper": upper, "center": np.asarray(bg.center, float),
                         "yaw": float(bg.yaw), "wall": wall}
        self.scene_info = {
            "object": {"center": list(self.obj_cfg["center"]), "yaw": float(self.obj_cfg.get("yaw", 0.0)),
                       "size": list(self.obj_cfg["size"])},
            "table_top_z": float(cfg["table"]["top_z"]),
            "mat": {"center": list(cfg["mat"]["center"]), "size": list(cfg["mat"]["size"]),
                    "top_z": float(cfg["mat"]["center"][2] + cfg["mat"]["size"][2] / 2)},
            "bin": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.bin_info.items()},
            "camera": {"K": self.K.tolist(), "T_cam_base": self.T_cam_base.tolist()},
        }
        # commanded state
        self.q_cmd = np.asarray(task_registry.START_Q, float)
        self.closure_cmd = 0.0
        self.grip_intent = "open"          # last commanded gripper intent: open | closed | width
        self.q_bias = np.zeros(6)
        self.render_updates = 4
        self.t0 = 0.0
        self.episode: dict = {}
        self._teleport(self.q_cmd, 0.0)
        for _ in range(30):                      # renderer warm-up (shader compile on first frames)
            self.world.step(render=False)
            self.world.render()
        self._calibrate_gripper()
        self.timings = {"import_robot_s": t_import - t_start, "build_total_s": time.perf_counter() - t_start}
        log("rig ready", json.dumps({k: round(v, 2) for k, v in self.timings.items()}))

    # ------------------------------------------------------------------ scene
    def _build_markers(self) -> None:
        """Visual-only task markers (no colliders): reach ball + pole, push zone."""
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        def mat(name, rgb, emissive=0.0):
            m = UsdShade.Material.Define(self.stage, f"/World/Looks/{name}")
            sh = UsdShade.Shader.Define(self.stage, m.GetPath().AppendChild("Surface"))
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(c * emissive for c in rgb)))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
            m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            return m

        red, green, grey = mat("ControlrRed", (0.85, 0.05, 0.04), 0.3), mat("ControlrGreen", (0.05, 0.7, 0.15), 0.25), \
            mat("ControlrGrey", (0.6, 0.6, 0.6))
        UsdGeom.Xform.Define(self.stage, "/World/Controlr")
        self.marker_prims = {}
        ball = UsdGeom.Sphere.Define(self.stage, "/World/Controlr/Ball")
        ball.CreateRadiusAttr(0.012)
        pole = UsdGeom.Cylinder.Define(self.stage, "/World/Controlr/Pole")
        pole.CreateRadiusAttr(0.0015)
        pole.CreateHeightAttr(1.0)
        zone = UsdGeom.Cube.Define(self.stage, "/World/Controlr/Zone")
        zone.CreateSizeAttr(1.0)
        for prim, m in ((ball, red), (pole, grey), (zone, green)):
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(m)
            self.marker_prims[prim.GetPrim().GetName()] = (prim, prim.AddTranslateOp(), prim.AddScaleOp())
            UsdGeom.Imageable(prim).MakeInvisible()

    def _place_marker(self, name: str, pos=None, scale=(1.0, 1.0, 1.0)) -> None:
        from pxr import Gf, UsdGeom
        prim, translate, scale_op = self.marker_prims[name]
        if pos is None:
            UsdGeom.Imageable(prim).MakeInvisible()
            return
        translate.Set(Gf.Vec3d(*map(float, pos)))
        scale_op.Set(Gf.Vec3f(*map(float, scale)))
        UsdGeom.Imageable(prim).MakeVisible()

    def _full_targets(self, q_arm, closure) -> np.ndarray:
        full = np.zeros(self.n_dof)
        full[self.ids] = q_arm
        full[self.fingers] = self._drive_targets(float(closure), self.cfg)
        return full

    def _apply(self, q_arm, closure) -> None:
        from isaacsim.core.utils.types import ArticulationAction
        target = self._full_targets(np.asarray(q_arm, float) + self.q_bias, closure)
        self.robot.apply_action(ArticulationAction(joint_positions=target))

    def _integrate(self, gain: float = 0.5, limit: float = BIAS_LIMIT_RAD) -> float:
        """Integral action on the arm drives while holding a pose.

        PHANTOM's force drives are pure PD (stiffness 3500 / damping 180), so
        gravity leaves a ~3 mrad elbow sag (~2 mm at the TCP); the real UR
        controller tracks to ~0.1 mm. A bounded bias (<= 0.01 rad, so it can
        never push meaningfully into an obstacle) emulates that integral term.
        Returns the remaining max joint error (rad)."""
        err = self.q_cmd - np.asarray(self.robot.get_joint_positions()[self.ids], float)
        self.q_bias = np.clip(self.q_bias + gain * err, -limit, limit)
        return float(np.max(np.abs(err)))

    def _teleport(self, q_arm, closure) -> None:
        full = np.zeros(self.n_dof)
        full[self.ids] = q_arm
        full[self.fingers] = self._joint_targets(float(closure), self.cfg)
        self.robot.set_joint_positions(full)
        self.robot.set_joint_velocities(np.zeros(self.n_dof))
        self._apply(q_arm, closure)

    def _pad_positions(self) -> np.ndarray:
        return np.array([np.asarray(p.get_world_poses()[0]).reshape(-1, 3)[0] for p in self.pads])

    def _calibrate_gripper(self) -> None:
        """Closure (Robotiq POS/255) -> opening table, measured on the actual
        sim linkage (fingers teleported, one physics step each, arm in free
        space). Opening = distance between the two gel pad frames minus one
        gel thickness, i.e. the free gap between the gel faces. The W2L pads
        sit 5.5 mm outward of the nominal Robotiq fingertip (PHANTOM mount
        correction), so even full closure leaves a gap of ~1 cm."""
        closures = np.linspace(0.0, 1.0, 21)
        gel = float(self.cfg["gripper"].get("gel_geometry", {}).get("pad_thickness_m", 0.0))
        self.gel_thickness = gel
        width = []
        for c in closures:
            self._teleport(self.q_cmd, c)
            self.world.step(render=False)
            p = self._pad_positions()
            width.append(max(float(np.linalg.norm(p[0] - p[1])) - gel, 0.0))
        self.grip_closures, self.grip_widths = closures, np.asarray(width)
        self.grip_max_m = float(self.grip_widths[0])
        self.grip_min_m = float(self.grip_widths.min())
        self._teleport(self.q_cmd, 0.0)
        self.world.step(render=False)
        tool_pos, tool_quat = self.tool.get_world_pose()
        R = self.Rotation.from_quat(np.asarray(tool_quat)[[1, 2, 3, 0]])
        pads = self._pad_positions()
        self.pad_midpoint_in_tool = R.inv().apply(pads.mean(axis=0) - np.asarray(tool_pos))
        axis_tool = R.inv().apply(pads[0] - pads[1])
        self.closing_axis_in_tool = axis_tool / np.linalg.norm(axis_tool)
        log("gripper calibration", json.dumps({
            "max_width_mm": round(self.grip_max_m * 1000, 1), "min_width_mm": round(self.grip_min_m * 1000, 1),
            "pad_mid_in_tool_mm": np.round(self.pad_midpoint_in_tool * 1000, 1).tolist(),
            "closing_axis_in_tool": np.round(self.closing_axis_in_tool, 3).tolist()}))

    def width_to_closure(self, width_m: float) -> float:
        """Commanded opening -> closure. Anything at or below the smallest
        reachable gap means "close" (full closure; the grasp latch in
        ``_actuate_gripper`` stops tightening at the object)."""
        w = float(width_m)
        if w <= self.grip_min_m + 1e-4:
            return 1.0
        w = min(w, self.grip_max_m)
        order = np.argsort(self.grip_widths)
        return float(np.interp(w, self.grip_widths[order], self.grip_closures[order]))

    # ------------------------------------------------------------------ state
    def _measured_closure(self) -> float:
        return self._closure_from_q(np.asarray(self.robot.get_joint_positions()[self.fingers], float), self.cfg)

    def _gripper_width(self) -> float:
        p = self._pad_positions()
        return float(max(np.linalg.norm(p[0] - p[1]) - self.gel_thickness, 0.0))

    def _pad_forces(self) -> np.ndarray:
        """Per-pad contact force with the OBJECT only (the pads' contact filter),
        like PHANTOM run_waffles.py's packet_forces. The unfiltered net force also
        counts pad-on-pad contact, so a gripper closed on nothing read as
        "holding: yes" (integration run 2026-10-02)."""
        return np.array([float(np.linalg.norm(np.asarray(p.get_contact_force_matrix(dt=self.dt)).reshape(-1, 3).sum(axis=0)))
                         for p in self.pads])

    def _tcp_measured(self):
        pos, quat = self.tool.get_world_pose()
        R = self.Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])
        return np.asarray(pos) + R.apply([0, 0, TCP_OFFSET_M]), R.as_rotvec()

    def state(self) -> dict:
        q = np.asarray(self.robot.get_joint_positions()[self.ids], float)
        tcp_pos, tcp_rv = self._tcp_measured()
        fk = self.forward_pose(q)
        pads = self._pad_forces()
        return {"t": float(self.world.current_time - self.t0), "q": q, "qd": np.asarray(
                    self.robot.get_joint_velocities()[self.ids], float),
                "tcp_pos": tcp_pos, "tcp_rotvec": tcp_rv, "tcp_fk_pos": fk[:3], "tcp_fk_rotvec": fk[3:],
                "gripper_m": self._gripper_width(), "gripper_closure": self._measured_closure(),
                # contract: "last commanded state is closed" (a GRIP close that latched on a wide
                # object is still "closed"); holding needs a commanded closure, not just contact
                "gripper_closed": self.grip_intent == "closed",
                "holding": bool(self.closure_cmd > 0.05 and np.all(pads > 0.1)),
                "pad_object_force_n": pads}

    def render(self, updates: int | None = None) -> tuple[np.ndarray, float]:
        """Render the D435 view of the current (paused) physics state.

        The RTX annotator is pipelined: the frame read after an app update
        shows the scene from a few updates earlier. ``render_updates`` app
        updates are issued before reading (measured: see README)."""
        t = time.perf_counter()
        for _ in range(int(updates or self.render_updates)):
            self.world.render()
        rgba = self.camera.get_rgba()
        if rgba is None or rgba.size == 0:
            raise RuntimeError("camera returned no frame")
        return np.ascontiguousarray(rgba[:, :, :3]).astype(np.uint8), time.perf_counter() - t

    def observe(self, render_updates: int | None = None) -> dict:
        rgb, dt = self.render(render_updates)
        return {"rgb": rgb, "state": self.state(), "render_s": dt}

    def info(self) -> dict:
        return {"protocol": protocol.PROTOCOL_VERSION, "scene": self.scene_rel, "physics_dt": self.dt,
                "camera": {"name": "scene", "width": self.resolution[0], "height": self.resolution[1],
                           "K": self.K, "T_cam_base": self.T_cam_base, "report": self.camera_report},
                "gripper": {"max_m": self.grip_max_m, "min_m": self.grip_min_m, "closures": self.grip_closures, "widths_m": self.grip_widths,
                            "pad_midpoint_in_tool_m": self.pad_midpoint_in_tool,
                            "closing_axis_in_tool": self.closing_axis_in_tool, "tcp_offset_m": TCP_OFFSET_M},
                "scene_info": self.scene_info, "tasks": sorted(task_registry.TASKS), "timings": self.timings}

    # ------------------------------------------------------------------ episode
    def reset(self, task: str, seed: int | None, params: dict | None) -> dict:
        t = time.perf_counter()
        spec = task_registry.get_task(task)
        if spec.scene != self.scene_rel:
            raise ValueError(f"task {task!r} needs scene {spec.scene}; this server runs {self.scene_rel}")
        params = dict(params or {})
        rng = np.random.default_rng(seed)
        plan = spec.sample(rng, params, self.scene_info)
        q0 = np.asarray(plan["start_q"], float)
        g = plan["start_gripper"]
        closure = 0.0 if g == "open" else 1.0 if g == "closed" else self.width_to_closure(float(g))
        self.q_cmd, self.closure_cmd = q0.copy(), closure
        self.grip_intent = "open" if g == "open" else "closed" if g == "closed" else "width"
        self._teleport(q0, closure)
        quat = task_registry.yaw_quat_wxyz(plan["object_yaw"])
        self.packet.set_world_pose(position=plan["object_pos"], orientation=quat)
        self.packet.set_linear_velocity(np.zeros(3))
        self.packet.set_angular_velocity(np.zeros(3))
        top = self.scene_info["table_top_z"]
        if plan.get("marker") is not None:
            m = np.asarray(plan["marker"], float)
            self._place_marker("Ball", m)
            h = m[2] - top
            self._place_marker("Pole", (m[0], m[1], top + h / 2), (1, 1, h))
        else:
            self._place_marker("Ball", None)
            self._place_marker("Pole", None)
        if plan.get("zone") is not None:
            z = np.asarray(plan["zone"], float)
            self._place_marker("Zone", (z[0], z[1], self.scene_info["mat"]["top_z"] + 0.0004), (0.05, 0.05, 0.0008))
        else:
            self._place_marker("Zone", None)
        settle_s = float(params.get("reset_settle_s", 1.0))
        self.q_bias = np.zeros(6)
        for k in range(int(round(settle_s / self.dt))):
            self._apply(q0, closure)
            self.world.step(render=False)
            if k % 10 == 9:
                self._integrate()
        pos, quat_now = self.packet.get_world_pose()
        drift = float(np.linalg.norm(np.asarray(pos)[:2] - np.asarray(plan["object_pos"])[:2]))
        self.t0 = float(self.world.current_time)
        st = self.state()
        self.episode = {"task": spec.name, "seed": seed, "params": params,
                        "object_pos": np.asarray(pos, float), "object_quat_wxyz": np.asarray(quat_now, float),
                        "marker": plan.get("marker"), "zone": plan.get("zone"),
                        "tcp_pos0": st["tcp_pos"], "start_q": q0, "settle_drift_m": drift,
                        "instruction": spec.instruction,
                        # running record for "pushed, not carried" (tasks.evaluate_push)
                        "max_lift_m": 0.0, "ever_held": False}
        obs = self.observe()
        return {"obs": obs, "episode": self.episode, "reset_s": time.perf_counter() - t}

    def _sample_contacts(self, peak: dict) -> dict[str, float]:
        """Accumulate peak normal force per (robot group, environment group) into
        ``peak``; return THIS sample's force per pair key (``"unstable": inf`` when the
        contact buffers hold NaN). Per-sample, not the running peak: the force stop
        compares it with the force of the same pair at the start of the motion, so a
        stopped arm resting against the box can still back away."""
        try:
            data = self.contacts.get_all(self.dt)
        except RuntimeError:          # NaN / saturated contact buffers: the solver diverged
            peak["unstable"] = float("inf")
            return {"unstable": float("inf")}
        now: dict[str, float] = {}
        for actor in data["per_actor"]:
            group = self.body_group.get(actor["actor_path"], "arm")
            for c in actor["contacts"]:
                key = f"{group}-{self.env_group.get(c['filter_path'], 'env')}"
                f = float(c["normal_force_magnitude_n"])
                peak[key] = max(peak.get(key, 0.0), f)
                now[key] = max(now.get(key, 0.0), f)
        return now

    def execute(self, q, gripper, durations, force_stop_n: float = 80.0, settle: dict | None = None,
                contact_every: int = 1, group=None, object_force_stop_n: float = 0.0) -> dict:
        """Rows of ``q`` with the same ``group`` id form one action: the arm passes through
        all of them in ONE min-jerk motion (``motion.interpolate_group``), the group's
        duration is the sum of its rows' ``durations``, and its gripper command (the
        group's last finite ``gripper``) acts after the motion. Without ``group`` every
        row is its own action (protocol v1 behaviour). ``segments_done`` counts groups.

        Force stop: contacts are sampled every ``contact_every`` physics steps during
        motion (every step by default: at 10 ms sampling the logged stops overshot the
        80 N threshold up to 1297 N) and every 10 steps while settling. A pair stops the
        motion when its force exceeds max(threshold, its force before the motion + 20 N,
        capped at 2x threshold): arm/gripper vs table/box with ``force_stop_n``, vs the
        manipulated object with ``object_force_stop_n`` (not while the gripper itself
        is closing — that contact is the grasp). A STOP at ANY point (motion, gripper,
        settle) freezes the drive target at the measured pose and clears the integral
        term, so the arm never keeps pushing into what it hit."""
        wall = time.perf_counter()
        q = np.asarray(q, float).reshape(-1, 6)
        gripper = np.asarray(gripper, float).reshape(-1)
        durations = np.asarray(durations, float).reshape(-1)
        group = np.arange(len(q)) if group is None else np.asarray(group, int).reshape(-1)
        settle = {"joint_speed_rad_s": 0.03, "tcp_speed_m_s": 0.005, "joint_err_rad": 5e-4, "min_s": 0.05, "max_s": 2.0,
                  "grip_min_s": 0.3, "grip_max_s": 1.2, **(settle or {})}
        t_begin = float(self.world.current_time)
        peak: dict[str, float] = {}
        pad_peak = np.zeros(2)
        st_flags = {"stopped": False, "reason": "", "gripping": False, "env_force": 0.0}
        n_steps = 0
        done_segments = 0
        prof = {"physics_s": 0.0, "contacts_s": 0.0}
        thr_env = float(force_stop_n or 0.0)
        thr_obj = float(object_force_stop_n or 0.0)
        f0 = self._sample_contacts({})
        floors = {}
        for key, f in f0.items():
            thr = thr_obj if key.endswith("-object") else thr_env
            if thr > 0 and np.isfinite(f):
                floors[key] = max(thr, min(f + FORCE_STOP_RISE_N, 2.0 * thr))
        obj_z0 = float(np.asarray(self.episode.get("object_pos", np.zeros(3)), float)[2]) if self.episode else 0.0

        def floor_for(key: str) -> float:
            if key in floors:
                return floors[key]
            return thr_obj if key.endswith("-object") else thr_env

        def stop(reason: str) -> None:
            if not st_flags["stopped"]:
                st_flags["stopped"], st_flags["reason"] = True, reason
                # hold where we are: the drive target becomes the measured pose
                self.q_cmd = np.asarray(self.robot.get_joint_positions()[self.ids], float)
                self.q_bias = np.zeros(6)

        def step(q_target, closure, every: int = 10):
            nonlocal n_steps, pad_peak
            t_s = time.perf_counter()
            self._apply(self.q_cmd if st_flags["stopped"] else q_target, closure)
            self.world.step(render=False)
            n_steps += 1
            prof["physics_s"] += time.perf_counter() - t_s
            if n_steps % every == 0:
                t_c = time.perf_counter()
                now = self._sample_contacts(peak)
                env = [f for k, f in now.items() if not k.endswith("-object") and k != "unstable"]
                st_flags["env_force"] = max(env) if env else 0.0
                if "unstable" in now or any(f > UNSTABLE_FORCE_N for f in now.values()):
                    stop("physics became unstable (reset the episode)")       # also with the stop disabled
                else:
                    for key, f in now.items():
                        is_obj = key.endswith("-object")
                        thr = thr_obj if is_obj else thr_env
                        if thr <= 0 or (is_obj and st_flags["gripping"]):
                            continue
                        if (key.endswith("-table") or key.endswith("-box") or is_obj) and f > floor_for(key):
                            what = "the packet" if is_obj else key.split("-", 1)[1]
                            stop(f"contact force {f:.0f} N against {what} > {floor_for(key):.0f} N")
                            break
                prof["contacts_s"] += time.perf_counter() - t_c
            if n_steps % 10 == 0 and self.episode:
                pads = self._pad_forces()
                pad_peak = np.maximum(pad_peak, pads)
                if self.closure_cmd > 0.05 and np.all(pads > 0.1):
                    self.episode["ever_held"] = True
                z = float(np.asarray(self.packet.get_world_pose()[0], float)[2])
                self.episode["max_lift_m"] = max(float(self.episode.get("max_lift_m", 0.0)), z - obj_z0)

        groups = [np.flatnonzero(group == gid) for gid in dict.fromkeys(group.tolist())]
        for rows in groups:
            q_from = self.q_cmd.copy()
            dur = float(np.sum(durations[rows]))
            n = max(1, int(round(max(dur, self.dt) / self.dt)))
            for q_k in motion.interpolate_group(q_from, q[rows], n):
                self.q_cmd = q_k
                step(q_k, self.closure_cmd, every=max(1, int(contact_every)))
                if st_flags["stopped"]:
                    break
            if st_flags["stopped"]:
                break
            g = gripper[rows][np.isfinite(gripper[rows])]
            if len(g):
                st_flags["gripping"] = True
                self._actuate_gripper(float(g[-1]), lambda qq, cc: step(qq, cc), settle,
                                      lambda: st_flags["stopped"])
                st_flags["gripping"] = False
            done_segments += 1
            if st_flags["stopped"]:
                break
        # Settled = joints slow, the TCP barely moving and the joints on target
        # (integral action removes the gravity sag). The arm keeps a ~0.01
        # rad/s elbow micro-oscillation under PHANTOM's drive gains, so the
        # speed thresholds are loose and the timeout is the backstop.
        t_settle, settled = 0.0, False
        tcp_prev = self._tcp_measured()[0]
        while t_settle < settle["max_s"] - 1e-9:
            for _ in range(10):
                step(self.q_cmd, self.closure_cmd)
            t_settle += 10 * self.dt
            # anti-windup: no integration after a stop or while pressing on something
            integrate = not st_flags["stopped"] and st_flags["env_force"] < ANTI_WINDUP_N
            q_err = self._integrate() if integrate else 0.0
            qd = np.abs(np.asarray(self.robot.get_joint_velocities()[self.ids], float))
            tcp_now = self._tcp_measured()[0]
            v_tcp = float(np.linalg.norm(tcp_now - tcp_prev)) / (10 * self.dt)
            tcp_prev = tcp_now
            if t_settle >= settle["min_s"] and qd.max() < settle["joint_speed_rad_s"] \
                    and v_tcp < settle["tcp_speed_m_s"] and q_err < settle["joint_err_rad"]:
                settled = True
                break
        self._sample_contacts(peak)
        st = self.state()
        pad_peak = np.maximum(pad_peak, st["pad_object_force_n"])
        fk_cmd = self.forward_pose(self.q_cmd)
        from phantom.sim.gripper_adaptive import mechanical_diagnostics
        try:
            mech = mechanical_diagnostics(np.asarray(self.robot.get_joint_positions()[self.fingers], float))
            mech_ok, mech_note = bool(mech["passed"]), ""
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            mech_ok, mech_note = False, str(exc)
        stopped, stop_reason = st_flags["stopped"], st_flags["reason"]
        return {
            "state": st, "stopped": stopped, "stop_reason": stop_reason, "segments_done": done_segments,
            "sim_s": float(self.world.current_time - t_begin), "settle_s": t_settle,
            "settled": settled, "physics_steps": n_steps,
            "unstable": "unstable" in stop_reason, "q_cmd": self.q_cmd.copy(), "q_err_max_rad": float(np.max(np.abs(st["q"] - self.q_cmd))),
            "tcp_err_m": float(np.linalg.norm(st["tcp_pos"] - fk_cmd[:3])),
            "contacts_peak_n": peak, "pad_object_peak_n": pad_peak,
            "gripper_mechanics_ok": mech_ok, "gripper_mechanics_note": mech_note,
            "qd_max_rad_s": float(np.max(np.abs(st["qd"]))), "profile": prof,
            "wall_s": time.perf_counter() - wall,
        }

    def _actuate_gripper(self, width_m: float, step, settle: dict, is_stopped) -> None:
        """Robotiq-like position command with object detection.

        Closing ramps the command and, once both pads press on the packet,
        stops tightening at ``grip_squeeze`` closure past first contact —
        the real Robotiq also stops at the object and holds with its force
        limit. Driving the W2L linkage to full closure through a 35 mm packet
        destabilises PhysX (observed: the arm and packet were flung), and
        PHANTOM's own expert only commands 0.62 (scripted_expert.grip_close)."""
        target = self.width_to_closure(width_m)
        self.grip_intent = "closed" if target >= 1.0 else "open" if width_m >= self.grip_max_m - 1e-4 else "width"
        chunk = 20
        rate = float(settle.get("grip_rate_per_s", 1.5))          # closure units / s
        contact_n = float(settle.get("grip_contact_n", 0.5))
        squeeze = float(settle.get("grip_squeeze", 0.12))
        closing = target > self.closure_cmd
        t_g, prev, latched = 0.0, self._measured_closure(), False
        while t_g < settle["grip_max_s"] and not is_stopped():
            if closing and not latched:
                self.closure_cmd = min(target, self.closure_cmd + rate * chunk * self.dt)
                if np.all(self._pad_forces() > contact_n):
                    self.closure_cmd = min(target, self._measured_closure() + squeeze)
                    latched = True
            elif not closing:
                self.closure_cmd = target
            for _ in range(chunk):
                step(self.q_cmd, self.closure_cmd)
            t_g += chunk * self.dt
            cur = self._measured_closure()
            ramp_done = latched or not closing or self.closure_cmd >= target
            if ramp_done and t_g >= settle["grip_min_s"] and abs(cur - prev) < 0.002:
                break
            prev = cur

    def snapshot(self) -> dict:
        pos, quat = self.packet.get_world_pose()
        try:
            robot_obj = float(self.contacts.get_all(self.dt)["normal_force_by_environment_n"].get(
                self.paths["object_path"], 0.0))
        except RuntimeError:          # diverged solver: never report the packet as released
            robot_obj = float("inf")
        st = self.state()
        return {"object_pos": np.asarray(pos, float), "object_quat_wxyz": np.asarray(quat, float),
                "object_lin_vel": np.asarray(self.packet.get_linear_velocity(), float),
                "object_ang_vel": np.asarray(self.packet.get_angular_velocity(), float),
                "object_size": np.asarray(self.obj_cfg["size"], float),
                "pad_object_force_n": st["pad_object_force_n"],
                "robot_object_force_n": robot_obj,
                "tcp_pos": st["tcp_pos"], "bin": self.bin_info, "t": st["t"]}

    def check_goal(self) -> dict:
        if not self.episode:
            raise RuntimeError("reset first")
        snap = self.snapshot()
        res = task_registry.get_task(self.episode["task"]).evaluate(snap, self.episode)
        res["metrics"] = {**res["metrics"], "object_pos": snap["object_pos"].tolist()}
        return res


def serve(rig: IsaacRig, host: str, port: int, authkey: bytes, once: bool) -> None:
    listener = Listener((host, port), authkey=authkey)
    log(protocol.READY_MARKER, f"{host}:{port}")
    ready = os.environ.get("CONTROLR_ISAAC_READY_FILE")
    if ready:
        Path(ready).write_text(json.dumps({"host": host, "port": port, "pid": os.getpid()}))
    try:
        while True:
            conn = listener.accept()
            log("client connected")
            shutdown = False
            try:
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    started = time.perf_counter()
                    op, args = msg.get("op"), msg.get("args", {})
                    try:
                        if op == "close":
                            conn.send(protocol.ok(None, started))
                            break
                        if op == "shutdown":
                            conn.send(protocol.ok(None, started))
                            shutdown = True
                            break
                        handler = {"info": rig.info, "reset": rig.reset, "observe": rig.observe,
                                   "state": rig.state, "execute": rig.execute,
                                   "check_goal": rig.check_goal}.get(op)
                        if handler is None:
                            raise ValueError(f"unknown op {op!r}")
                        conn.send(protocol.ok(handler(**args), started))
                    except Exception as exc:  # noqa: BLE001 - report to client, keep serving
                        conn.send(protocol.error(f"{type(exc).__name__}: {exc}", traceback.format_exc()))
            finally:
                conn.close()
                log("client disconnected")
            if shutdown or once:
                break
    finally:
        listener.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phantom", type=Path, default=Path(os.environ.get(
        "PHANTOM_ROOT", "/home/physicalai/phantom-icra-2027/phantom")))
    ap.add_argument("--scene", default=task_registry.WAFFLE_SCENE, help="PHANTOM scene config (repo-relative)")
    ap.add_argument("--host", default=protocol.DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=protocol.DEFAULT_PORT)
    ap.add_argument("--out", type=Path, default=HERE.parents[2] / "runs" / ".isaac_cache",
                    help="imported robot USD + URDF (~0.5 GB, rewritten at every start)")
    ap.add_argument("--solver-iterations", type=int, default=None)
    ap.add_argument("--render-updates", type=int, default=4, help="app updates per observation (RTX pipeline depth)")
    ap.add_argument("--once", action="store_true", help="exit after the first client disconnects")
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost"):
        raise SystemExit("the Isaac server only listens on localhost")
    sys.path.insert(0, str(args.phantom))
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 640, "height": 480, "renderer": "RaytracedLighting",
                         "anti_aliasing": 3, "multi_gpu": False, "sync_loads": True})
    code = 0
    try:
        t_app = time.perf_counter()
        rig = IsaacRig(app, args.phantom, args.scene, args.out, solver_iterations=args.solver_iterations)
        rig.render_updates = max(1, args.render_updates)
        rig.timings["app_start_s"] = t_app - t0
        rig.timings["startup_total_s"] = time.perf_counter() - t0
        log("startup", json.dumps({k: round(v, 2) for k, v in rig.timings.items()}))
        serve(rig, args.host, args.port, protocol.authkey_from_env(), args.once)
    except BaseException:  # noqa: BLE001 - make failures visible before Kit tears down
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        app.close(exit_code=code) if "exit_code" in app.close.__code__.co_varnames else app.close()
        os._exit(code)


if __name__ == "__main__":
    main()
