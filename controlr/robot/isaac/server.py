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

On top of PHANTOM's stage (applied after it is built, PHANTOM untouched): the blue box
(``/World/Bin``) becomes ONE dynamic rigid body with its five parts as colliders (task
param ``box_dynamic``; kinematic = immovable otherwise), and the packet gets a contact
view against the environment so the held-packet stop uses packet-vs-environment force,
never the grip's own pad forces (``controlr.robot.isaac.contacts``).

Speed (``--dt``, ``--solver-iterations``, ``--no-usd-writeback``; per execute: ``reader``,
``substeps``, ``contact_every``, ``direct``): one PhysX contact-force matrix for all robot
bodies instead of one wrapper view per body, PhysX stepped and drive targets set without
the isaacsim.core wrappers, transforms written to USD only before a render. ``execute``
returns a ``profile`` (seconds per part) for the run log.

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
from controlr.robot.isaac import cameras as virtual_cams  # noqa: E402
from controlr.robot.isaac import contacts as contact_rules  # noqa: E402
from controlr.robot.isaac import motion  # noqa: E402
from controlr.robot.isaac import protocol  # noqa: E402
from controlr.robot.isaac import tasks as task_registry  # noqa: E402

TCP_OFFSET_M = 0.18      # PHANTOM tcp: tool0 + 180 mm along tool z (configs/hardware.yaml)
# Bound of the gravity-compensating integral term (see _integrate). At PHANTOM's drive
# stiffness (3500 N m/rad) 0.004 rad is ~14 N m per joint: enough for the ~3 mrad elbow
# sag, far below the 0.01 rad (~35 N m, ~90 N at the TCP) that could push into an obstacle.
BIAS_LIMIT_RAD = 0.004
ANTI_WINDUP_N = 5.0         # no integration while robot-environment contact exceeds this
FORCE_STOP_RISE_N = contact_rules.FORCE_STOP_RISE_N
UNSTABLE_FORCE_N = contact_rules.UNSTABLE_FORCE_N
BIN_PATH = "/World/Bin"
DEFAULT_BOX_MASS_KG = 0.4    # a light plastic tote (task param box_mass_kg)


def log(*parts) -> None:
    print("[controlr-isaac]", *parts, flush=True)


class IsaacRig:
    """One built stage + handles. Methods map 1:1 to protocol ops."""

    def __init__(self, app, phantom: Path, scene_rel: str, out: Path, *, solver_iterations: int | None = None,
                 physics: dict | None = None, extra_cameras: list[str] | None = None):
        t_start = time.perf_counter()
        self.app, self.phantom, self.out = app, phantom, out
        self.scene_rel = scene_rel
        cfg = json.loads((phantom / scene_rel).read_text())
        phys = dict(physics or {})
        if solver_iterations:
            phys.setdefault("solver_position_iterations", solver_iterations)
        for key in ("dt", "solver_position_iterations", "solver_velocity_iterations"):
            if phys.get(key):
                cfg["physics"][key] = type(cfg["physics"][key])(phys[key])
        if phys.get("forearm_collision_approximation"):
            cfg["physics"]["forearm_collision_approximation"] = str(phys["forearm_collision_approximation"])
        self.num_threads = int(phys["num_threads"]) if phys.get("num_threads") else None
        self.self_collisions = bool(phys.get("self_collisions", True))
        self.cfg = cfg
        self.usd_writeback = bool(phys.get("usd_writeback", True))
        # PHANTOM's per-body diagnostic contact views (reader "phantom"); without them only the
        # contact matrix exists (fewer contact views for PhysX to fill every step)
        self.legacy_views = bool(phys.get("legacy_contact_views", True))

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
        self.dt = float(cfg["physics"]["dt"])
        if is_adaptive(cfg):
            if self.dt > 0.001 + 1e-12:
                # PHANTOM rejects > 1 ms for the native W2L linkage (its 4 ms contact tests
                # diverged); a coarser step is only run on explicit request, for speed trials.
                log(f"WARNING: physics dt {self.dt * 1000:.1f} ms > 1 ms with PHANTOM's adaptive W2L gripper "
                    f"(PHANTOM's validate_physics_timestep rejects it: 4 ms contact tests diverged)")
            else:
                from phantom.sim.gripper_adaptive import validate_physics_timestep
                validate_physics_timestep(cfg)
        robot_usd = import_robot(phantom, out, cfg)
        t_import = time.perf_counter()
        self.world = World(stage_units_in_meters=1, physics_dt=self.dt, rendering_dt=self.dt)
        self.stage = omni.usd.get_context().get_stage()
        self.paths = paths = build_scene(self.stage, phantom, out, cfg, robot_usd)
        roots = [str(p.GetPath()) for p in self.stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(roots) != 1:
            raise RuntimeError(f"expected one articulation root, found {roots}")
        art = PhysxSchema.PhysxArticulationAPI.Apply(self.stage.GetPrimAtPath(roots[0]))
        art.CreateSolverPositionIterationCountAttr(cfg["physics"]["solver_position_iterations"])
        art.CreateSolverVelocityIterationCountAttr(cfg["physics"]["solver_velocity_iterations"])
        art.CreateEnabledSelfCollisionsAttr(self.self_collisions)
        self.has_bin = self._make_bin_dynamic()
        # the packet reports its own contacts (packet-vs-environment view)
        PhysxSchema.PhysxContactReportAPI.Apply(self.stage.GetPrimAtPath(paths["object_path"])).CreateThresholdAttr(0.0)
        root = roots[0]
        if self.stage.GetPrimAtPath(root).IsA(UsdPhysics.Joint):
            root = paths["robot_path"]
        self.robot = self.world.scene.add(SingleArticulation(prim_path=root, name="ur3"))
        self.packet = self.world.scene.add(SingleRigidPrim(prim_path=paths["object_path"], name="object"))
        self.tool = SingleRigidPrim(prim_path=paths["tool_path"], name="tool_feedback",
                                    reset_xform_properties=False)
        self.bin = (SingleRigidPrim(prim_path=BIN_PATH, name="bin", reset_xform_properties=False)
                    if self.has_bin else None)
        if self.legacy_views:
            self.pads = [RigidPrim(prim_paths_expr=p, name=f"pad_{i}", track_contact_forces=True,
                                   contact_filter_prim_paths_expr=[paths["object_path"]],
                                   max_contact_count=PAD_OBJECT_CONTACT_CAPACITY, reset_xform_properties=False)
                         for i, p in enumerate(paths["pad_paths"])]
        else:
            self.pads = [RigidPrim(prim_paths_expr=p, name=f"pad_{i}", reset_xform_properties=False)
                         for i, p in enumerate(paths["pad_paths"])]
        robot_bodies = [str(p.GetPath()) for p in self.stage.Traverse()
                        if p.HasAPI(UsdPhysics.RigidBodyAPI) and str(p.GetPath()).startswith(paths["robot_path"] + "/")]
        self.robot_bodies = robot_bodies
        # legacy reader: PHANTOM's per-body diagnostic views (one wrapper view per body)
        self.contacts = None
        if self.legacy_views:
            self.contacts = RobotEnvironmentContactViews(robot_bodies, environment_paths=paths["environment_paths"],
                                                         rigid_prim_cls=RigidPrim)
        else:
            for p in robot_bodies:              # the contact matrix needs contact reports on its sensors
                PhysxSchema.PhysxContactReportAPI.Apply(self.stage.GetPrimAtPath(p)).CreateThresholdAttr(0.0)
        gripper_root = paths["gripper_housing_path"]
        self.body_group = {p: ("gripper" if p.startswith(gripper_root) or p in paths["pad_paths"] else "arm")
                           for p in robot_bodies}
        self.env_group = {p: contact_rules.env_group(p, paths["object_path"]) for p in paths["environment_paths"]}
        self.packet_env_paths = [p for p in paths["environment_paths"] if p != paths["object_path"]]

        cam = cfg["camera"]
        self.camera = Camera("/World/SceneCamera", frequency=15, resolution=tuple(cam["resolution"]))
        twc = np.asarray(cam["world_from_cv"], float)
        orient = Rotation.from_matrix(twc[:3, :3] @ np.diag([1, -1, -1])).as_quat()
        self.camera.set_world_pose(position=twc[:3, 3], orientation=orient[[3, 0, 1, 2]], camera_axes="usd")
        configure_camera_intrinsics(self.camera, cam)
        self.camera.set_clipping_range(0.02, 10)
        self.camera.set_focus_distance(1.0)
        # virtual views (cameras.py): name -> (Camera, config); aimed between the mat and the box
        self.extra: dict[str, tuple] = {}
        if extra_cameras:
            mat_xy = np.asarray(cfg["mat"]["center"][:2], float)
            centre = (mat_xy + np.asarray(bin_geometry(cfg["bin"]).center, float)[:2]) / 2
            for name in extra_cameras:
                vc = virtual_cams.virtual_camera(name, centre, float(cfg["table"]["top_z"]), cam["resolution"])
                c = Camera(f"/World/VirtualCamera_{name}", frequency=15, resolution=tuple(vc["resolution"]))
                vt = np.asarray(vc["world_from_cv"], float)
                q = Rotation.from_matrix(vt[:3, :3] @ np.diag([1, -1, -1])).as_quat()
                c.set_world_pose(position=vt[:3, 3], orientation=q[[3, 0, 1, 2]], camera_axes="usd")
                configure_camera_intrinsics(c, vc)
                c.set_clipping_range(0.02, 10)
                c.set_focus_distance(1.0)
                self.extra[name] = (c, vc)
        self.world.reset()
        for pad in self.pads:
            pad.initialize()
        if self.contacts is not None:
            self.contacts.initialize()
        self.tool.initialize()
        if self.bin is not None:
            self.bin.initialize()
        self.camera.initialize()
        self._init_fast_paths()
        if self.num_threads:
            self._physx.set_thread_count(self.num_threads)
        # Turn-based: frames are requested explicitly after settling, never on a
        # sim-time cadence. Camera's own rate limiter counts *simulation* time
        # between renders (none passes between two observes), so disable it:
        # -1 is the class's "acquire on every rendered frame" state.
        self.camera._frequency = -1
        self.camera_report = validate_camera_intrinsics(self.camera, cam)
        self.extra_info: dict[str, dict] = {}
        for name, (c, vc) in self.extra.items():
            c.initialize()
            c._frequency = -1
            validate_camera_intrinsics(c, vc)
            self.extra_info[name] = {"name": name, "width": vc["resolution"][0], "height": vc["resolution"][1],
                                     "K": virtual_cams.intrinsics(vc),
                                     "T_cam_base": np.linalg.inv(np.asarray(vc["world_from_cv"], float))}
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
        wall_xy = (wall, wall)
        if "outer_size" in cfg["bin"] and "opening_size" in cfg["bin"]:
            d = (np.asarray(cfg["bin"]["outer_size"][:2], float) - np.asarray(cfg["bin"]["opening_size"], float)) / 2
            wall, wall_xy = float(np.min(d)), (float(d[0]), float(d[1]))
        self.bin_info = {"lower": lower, "upper": upper, "center": np.asarray(bg.center, float),
                         "yaw": float(bg.yaw), "wall": wall, "wall_xy": wall_xy, "dynamic": False}
        self.scene_info = {
            "object": {"center": list(self.obj_cfg["center"]), "yaw": float(self.obj_cfg.get("yaw", 0.0)),
                       "size": list(self.obj_cfg["size"])},
            "table_top_z": float(cfg["table"]["top_z"]),
            "mat": {"center": list(cfg["mat"]["center"]), "size": list(cfg["mat"]["size"]),
                    "top_z": float(cfg["mat"]["center"][2] + cfg["mat"]["size"][2] / 2)},
            "bin": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.bin_info.items()},
            "camera": {"K": self.K.tolist(), "T_cam_base": self.T_cam_base.tolist()},
        }
        # the box body's AUTHORED pose (PHANTOM: /World/Bin translate = the bin centre, rotateZ =
        # yaw): reset puts it back there; its geometry follows the body from there
        self.bin_pose0 = None
        if self.bin is not None:
            self.bin_pose0 = (np.asarray(bg.center, float), task_registry.yaw_quat_wxyz(float(bg.yaw)))
            pos, quat = self.bin.get_world_pose()
            if np.linalg.norm(np.asarray(pos, float) - self.bin_pose0[0]) > 0.005:
                log("WARNING: /World/Bin body pose", np.round(np.asarray(pos, float), 4).tolist(),
                    "is not the bin centre", np.round(self.bin_pose0[0], 4).tolist())
        # commanded state
        self.q_cmd = np.asarray(task_registry.START_Q, float)
        self.closure_cmd = 0.0
        self.grip_intent = "open"          # last commanded gripper intent: open | closed | width
        self.q_bias = np.zeros(6)
        self.render_updates = 4
        self.t0 = 0.0
        self.reader = "phantom"            # contact reader of the current execute (state() uses it too)
        self.direct = False
        self._finger_cache: dict[float, np.ndarray] = {}
        self.episode: dict = {}
        self._teleport(self.q_cmd, 0.0)
        for _ in range(30):                      # renderer warm-up (shader compile on first frames)
            self.world.step(render=False)
            self.world.render()
        self._calibrate_gripper()
        if not self.usd_writeback:
            # PhysX writes every body transform to USD after each step by default; nothing but
            # the renderer reads USD here (poses come from the physics views), so write only
            # before a render (render()).
            import carb
            st = carb.settings.get_settings()
            st.set_bool("/physics/updateToUsd", False)
            st.set_bool("/physics/updateVelocitiesToUsd", False)
        self.timings = {"import_robot_s": t_import - t_start, "build_total_s": time.perf_counter() - t_start}
        log("rig ready", json.dumps({k: round(v, 2) for k, v in self.timings.items()}),
            json.dumps(self.physics_info()))

    # ------------------------------------------------------------------ scene
    def _make_bin_dynamic(self) -> bool:
        """``/World/Bin`` (PHANTOM: an Xform with translate + rotateZ and five collider
        cubes, no rigid body) -> one rigid body with those colliders. Mass and dynamic /
        kinematic are set per reset (task params ``box_dynamic``, ``box_mass_kg``); the
        friction is PHANTOM's (its BinContact material stays bound to the colliders)."""
        from pxr import PhysxSchema, UsdGeom, UsdPhysics
        prim = self.stage.GetPrimAtPath(BIN_PATH)
        if not prim.IsValid():
            return False
        if not prim.IsA(UsdGeom.Xform):
            prim = UsdGeom.Xform.Define(self.stage, BIN_PATH).GetPrim()
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(DEFAULT_BOX_MASS_KG)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        self._bin_rb = rb
        return True

    def _init_fast_paths(self) -> None:
        """Raw PhysX views (after world.reset): one contact-force matrix for every robot body
        against the environment, one for the packet against the environment, the
        articulation's DOF view for drive targets and the simulation interface."""
        sv = None
        try:
            from isaacsim.core.simulation_manager import SimulationManager
            sv = SimulationManager.get_physics_sim_view()
        except Exception:  # noqa: BLE001 - older layouts
            sv = getattr(self.world, "physics_sim_view", None)
        if sv is None:
            raise RuntimeError("no physics simulation view after world.reset()")
        env = list(self.paths["environment_paths"])
        self.cv_robot = sv.create_rigid_contact_view(list(self.robot_bodies), [env] * len(self.robot_bodies), 0)
        rows = list(getattr(self.cv_robot, "sensor_paths", self.robot_bodies))
        if [str(r) for r in rows] != list(self.robot_bodies):
            log("contact view sensor order differs from the body list; mapping by path")
        self.cv_rows = [str(r) for r in rows]
        self.cv_row_groups = [self.body_group.get(r, "arm") for r in self.cv_rows]
        self.cv_cols = env
        self.cv_col_groups = [self.env_group.get(p, "table") for p in env]
        self.cv_pad_rows = [self.cv_rows.index(p) for p in self.paths["pad_paths"]]
        self.cv_obj_col = env.index(self.paths["object_path"])
        self.cv_packet = sv.create_rigid_contact_view([self.paths["object_path"]], [list(self.packet_env_paths)], 0)
        self.cv_packet_parts = [contact_rules.packet_part(p) for p in self.packet_env_paths]
        self._dof_view = self.robot._articulation_view._physics_view
        self._idx0 = np.array([0], dtype=np.uint32)
        self._sim_iface = self.world.get_physics_context()._physics_sim_interface
        import omni.physx
        self._physx = omni.physx.get_physx_interface()

    def physics_info(self) -> dict:
        p = self.cfg["physics"]
        return {"dt": self.dt, "solver_position_iterations": int(p["solver_position_iterations"]),
                "solver_velocity_iterations": int(p["solver_velocity_iterations"]),
                "usd_writeback": self.usd_writeback,
                "forearm_collision_approximation": str(p.get("forearm_collision_approximation", "convexHull")),
                "num_threads": self.num_threads, "legacy_contact_views": self.legacy_views,
                "self_collisions": self.self_collisions}

    def _bin_now(self) -> dict | None:
        """The box's CURRENT geometry (bin_info moved with the rigid body)."""
        if self.bin is None or self.bin_pose0 is None:
            return dict(self.bin_info)
        pos, quat = self.bin.get_world_pose()
        b = task_registry.bin_info_at(self.bin_info, self.bin_pose0, np.asarray(pos, float), np.asarray(quat, float))
        b["dynamic"] = bool(self.episode.get("box_dynamic", True)) if self.episode else True
        return b

    def _full_targets(self, q_arm, closure) -> np.ndarray:
        full = np.zeros(self.n_dof)
        full[self.ids] = q_arm
        key = round(float(closure), 6)
        fin = self._finger_cache.get(key)
        if fin is None:
            fin = np.asarray(self._drive_targets(float(closure), self.cfg), float)
            if len(self._finger_cache) > 4096:
                self._finger_cache.clear()
            self._finger_cache[key] = fin
        full[self.fingers] = fin
        return full

    def _apply(self, q_arm, closure) -> None:
        target = self._full_targets(np.asarray(q_arm, float) + self.q_bias, closure)
        if self.direct:
            self._dof_view.set_dof_position_targets(target.astype(np.float32)[None, :], self._idx0)
            return
        from isaacsim.core.utils.types import ArticulationAction
        self.robot.apply_action(ArticulationAction(joint_positions=target))

    def _sim_step(self) -> None:
        if self.direct:
            self._sim_iface.simulate(self.dt, self.world.current_time)
        else:
            self.world.step(render=False)

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

    def _robot_matrix(self) -> np.ndarray:
        return np.asarray(self.cv_robot.get_contact_force_matrix(self.dt), float).reshape(
            len(self.cv_rows), len(self.cv_cols), 3)

    def _pad_forces(self, matrix: np.ndarray | None = None) -> np.ndarray:
        """Per-pad contact force with the OBJECT only, like PHANTOM run_waffles.py's
        packet_forces. The unfiltered net force also counts pad-on-pad contact, so a
        gripper closed on nothing read as "holding: yes" (integration run 2026-10-02).
        ``reader == "matrix"``: from the robot contact matrix (one PhysX call)."""
        if self.reader == "matrix" or matrix is not None:
            m = self._robot_matrix() if matrix is None else matrix
            return np.linalg.norm(m[self.cv_pad_rows, self.cv_obj_col, :], axis=1)
        return np.array([float(np.linalg.norm(np.asarray(p.get_contact_force_matrix(dt=self.dt)).reshape(-1, 3).sum(axis=0)))
                         for p in self.pads])

    def _holding_grip(self) -> bool:
        """Robotiq-style object detection (gOBJ): after a close command the fingers are held
        open by something — they stopped clearly short of the commanded closure and clearly
        above the fully closed gap. Logged only (the model never sees `holding`)."""
        if self.grip_intent != "closed" or self.closure_cmd <= 0.05:
            return False
        return bool(self.closure_cmd - self._measured_closure() > 0.03 and self._gripper_width() > self.grip_min_m + 0.003)

    def _tcp_measured(self):
        """Measured TCP (pos, rotvec); NaN when PhysX has diverged (invalid transforms: zero or
        NaN quaternion — live run 20261002T194251Z crashed the server here instead of stopping)."""
        pos, quat = self.tool.get_world_pose()
        quat = np.asarray(quat, float)
        if not (np.all(np.isfinite(quat)) and np.all(np.isfinite(pos)) and np.linalg.norm(quat) > 1e-6):
            return np.full(3, np.nan), np.full(3, np.nan)
        R = self.Rotation.from_quat(quat[[1, 2, 3, 0]])
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
                "gripper_closure_cmd": float(self.closure_cmd),
                # contract: "last commanded state is closed" (a GRIP close that latched on a wide
                # object is still "closed"); holding needs a commanded closure, not just contact
                "gripper_closed": self.grip_intent == "closed",
                "holding": bool(self.closure_cmd > 0.05 and np.all(pads > 0.1)),
                "holding_grip": self._holding_grip(),
                "pad_object_force_n": pads,
                "bin": self._bin_now()}

    def render(self, updates: int | None = None) -> tuple[np.ndarray, float]:
        """Render the D435 view of the current (paused) physics state.

        The RTX annotator is pipelined: the frame read after an app update
        shows the scene from a few updates earlier. ``render_updates`` app
        updates are issued before reading (measured: see README). Without USD
        write-back the body transforms are pushed to USD once here."""
        t = time.perf_counter()
        if not self.usd_writeback:
            self._physx.update_transformations(False, True, False, False)
        for _ in range(int(updates or self.render_updates)):
            self.world.render()
        rgba = self.camera.get_rgba()
        if rgba is None or rgba.size == 0:
            raise RuntimeError("camera returned no frame")
        return np.ascontiguousarray(rgba[:, :, :3]).astype(np.uint8), time.perf_counter() - t

    def observe(self, render_updates: int | None = None) -> dict:
        rgb, dt = self.render(render_updates)
        out = {"rgb": rgb, "state": self.state(), "render_s": dt}
        if self.extra:          # rendered by the same world.render() calls
            extra = {}
            for name, (c, _) in self.extra.items():
                rgba = c.get_rgba()
                if rgba is None or rgba.size == 0:
                    raise RuntimeError(f"virtual camera {name} returned no frame")
                extra[name] = np.ascontiguousarray(rgba[:, :, :3]).astype(np.uint8)
            out["extra"] = extra
        return out

    def info(self) -> dict:
        return {"protocol": protocol.PROTOCOL_VERSION, "scene": self.scene_rel, "physics_dt": self.dt,
                "physics": self.physics_info(),
                "camera": {"name": "scene", "width": self.resolution[0], "height": self.resolution[1],
                           "K": self.K, "T_cam_base": self.T_cam_base, "report": self.camera_report},
                "extra_cameras": self.extra_info,
                "gripper": {"max_m": self.grip_max_m, "min_m": self.grip_min_m, "closures": self.grip_closures, "widths_m": self.grip_widths,
                            "pad_midpoint_in_tool_m": self.pad_midpoint_in_tool,
                            "closing_axis_in_tool": self.closing_axis_in_tool, "tcp_offset_m": TCP_OFFSET_M},
                "scene_info": self.scene_info, "tasks": sorted(task_registry.TASKS), "timings": self.timings}

    # ------------------------------------------------------------------ episode
    def _reset_bin(self, dynamic: bool, mass: float) -> None:
        if self.bin is None or self.bin_pose0 is None:
            return
        self._bin_rb.GetKinematicEnabledAttr().Set(not dynamic)
        self.bin.set_world_pose(position=self.bin_pose0[0], orientation=self.bin_pose0[1])
        if dynamic:
            self.bin.set_linear_velocity(np.zeros(3))
            self.bin.set_angular_velocity(np.zeros(3))
            self.bin.set_mass(float(mass))

    def reset(self, task: str, seed: int | None, params: dict | None, settings: dict | None = None) -> dict:
        """``settings``: the client's per-execute speed settings (``reader``, ``direct``,
        ``substeps``) — the reset settle runs with them too."""
        t = time.perf_counter()
        st_ = dict(settings or {})
        self.reader = str(st_.get("reader", self.reader)) if self.contacts is not None else "matrix"
        self.direct = bool(st_.get("direct", self.direct))
        k_sub = max(1, int(st_.get("substeps", 1)))
        spec = task_registry.get_task(task)
        if spec.scene != self.scene_rel:
            raise ValueError(f"task {task!r} needs scene {spec.scene}; this server runs {self.scene_rel}")
        params = dict(params or {})
        box_dynamic = bool(params.get("box_dynamic", True))
        self._reset_bin(box_dynamic, float(params.get("box_mass_kg", DEFAULT_BOX_MASS_KG)))
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
        settle_s = float(params.get("reset_settle_s", 1.0))
        self.q_bias = np.zeros(6)
        n_ticks = int(round(settle_s / (self.dt * k_sub)))
        every = max(1, int(round(10 / k_sub)))
        for k in range(n_ticks):
            self._apply(q0, closure)
            for _ in range(k_sub):
                self._sim_step()
            if k % every == every - 1:
                self._integrate()
        pos, quat_now = self.packet.get_world_pose()
        drift = float(np.linalg.norm(np.asarray(pos)[:2] - np.asarray(plan["object_pos"])[:2]))
        self.t0 = float(self.world.current_time)
        self.episode = {"box_dynamic": box_dynamic}
        st = self.state()
        bin0 = st["bin"]
        self.episode = {"task": spec.name, "seed": seed, "params": params,
                        "object_pos": np.asarray(pos, float), "object_quat_wxyz": np.asarray(quat_now, float),
                        "reach_target": plan.get("reach_target"), "push_target": plan.get("push_target"),
                        "tcp_pos0": st["tcp_pos"], "start_q": q0, "settle_drift_m": drift,
                        "instruction": plan.get("instruction") or spec.instruction,
                        "object_pos_sampled": np.asarray(plan["object_pos"], float),
                        "object_yaw_sampled": float(plan["object_yaw"]),
                        "start_yaw_offset": float(plan.get("start_yaw_offset", 0.0)), "start_gripper": g,
                        "box_dynamic": box_dynamic, "bin0": bin0,
                        # running record for "pushed, not carried" (tasks.evaluate_push)
                        "max_lift_m": 0.0, "ever_held": False}
        # the sampled scene in loggable units (setup.json "scene"; tasks.scene_record)
        self.episode["scene"] = task_registry.scene_record(self.episode, {**self.scene_info, "bin": bin0})
        obs = self.observe()
        return {"obs": obs, "episode": self.episode, "reset_s": time.perf_counter() - t}

    def _sample_contacts(self, peak: dict) -> dict[str, float]:
        """Legacy reader (PHANTOM's per-body views): accumulate peak normal force per (robot
        group, environment group) into ``peak``; return THIS sample's force per pair key
        (``"unstable": inf`` when the contact buffers hold NaN). Per-sample, not the running
        peak: the force stop compares it with the force of the same pair at the start of the
        motion, so a stopped arm resting against the box can still back away."""
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

    def _sample_all(self, peak: dict, reader: str) -> tuple[dict[str, float], np.ndarray]:
        """This sample's pair forces (robot pairs + ``packet-<part>``) and pad forces; peaks
        accumulate into ``peak``."""
        if reader == "matrix":
            m = self._robot_matrix()
            now = contact_rules.pair_forces(m, self.cv_row_groups, self.cv_col_groups)
            pads = (np.full(2, np.inf) if "unstable" in now else
                    np.linalg.norm(m[self.cv_pad_rows, self.cv_obj_col, :], axis=1))
        else:
            now = self._sample_contacts({})
            pads = self._pad_forces()
        pm = np.asarray(self.cv_packet.get_contact_force_matrix(self.dt), float)
        now.update({k: v for k, v in contact_rules.pair_forces(
            pm.reshape(1, len(self.cv_packet_parts), 3), ["packet"], self.cv_packet_parts).items()
            if k != "unstable" or "unstable" not in now})
        for k, f in now.items():
            peak[k] = max(peak.get(k, 0.0), f)
        return now, pads

    def execute(self, q, gripper, durations, force_stop_n: float = 80.0, settle: dict | None = None,
                contact_every: int = 1, group=None, object_force_stop_n: float = 0.0,
                box_force_stop_n: float | None = None, held_object_force_stop_n: float | None = None,
                reader: str = "phantom", substeps: int = 1, direct: bool = False,
                predict_stop: bool = False, box_push_stop_m: float = 0.005) -> dict:
        """Rows of ``q`` with the same ``group`` id form one action: the arm passes through
        all of them in ONE min-jerk motion (``motion.interpolate_group``), the group's
        duration is the sum of its rows' ``durations``, and its gripper command (the
        group's last finite ``gripper``) acts after the motion. Without ``group`` every
        row is its own action (protocol v1 behaviour). ``segments_done`` counts groups.

        Ticks: every ``substeps`` physics steps the drive target is updated once (min-jerk
        samples at that rate) — ``substeps=1`` is one target per 1 ms step as before.
        Contacts are sampled every ``contact_every`` ticks during motion (default every
        step: at 10 ms sampling the logged stops overshot the 80 N threshold up to 1297 N)
        and every ~10 steps while settling / gripping. ``reader``: ``matrix`` (one PhysX
        contact matrix) or ``phantom`` (PHANTOM's per-body views, the legacy path);
        ``direct``: step PhysX and set drive targets without the isaacsim.core wrappers.

        Force stop (``contacts.check_stop``): a pair stops the motion when its force exceeds
        max(threshold, its force before the motion + 20 N, capped at 2x threshold):
        arm/gripper vs table with ``force_stop_n``, vs the box with ``box_force_stop_n``
        (None -> ``force_stop_n``), vs the object with ``object_force_stop_n`` only while the
        object is NOT held (nor being grasped); while it IS held, the object vs the
        environment with ``held_object_force_stop_n`` (None -> ``object_force_stop_n``) —
        the grip's own pad forces never stop the arm. ``predict_stop``: also stop when the
        force extrapolated one sample ahead exceeds the floor. A STOP at ANY point (motion,
        gripper, settle) freezes the drive target at the measured pose and clears the
        integral term, so the arm never keeps pushing into what it hit."""
        wall = time.perf_counter()
        self.reader, self.direct = str(reader), bool(direct)
        if self.contacts is None:
            self.reader = "matrix"              # the legacy views were not built (--no-legacy-contact-views)
        k_sub = max(1, int(substeps))
        q = np.asarray(q, float).reshape(-1, 6)
        gripper = np.asarray(gripper, float).reshape(-1)
        durations = np.asarray(durations, float).reshape(-1)
        group = np.arange(len(q)) if group is None else np.asarray(group, int).reshape(-1)
        settle = {"joint_speed_rad_s": 0.03, "tcp_speed_m_s": 0.005, "joint_err_rad": 5e-4, "min_s": 0.05, "max_s": 2.0,
                  "grip_min_s": 0.3, "grip_max_s": 1.2, **(settle or {})}
        limits = contact_rules.StopLimits(
            env_n=float(force_stop_n or 0.0),
            box_n=float(force_stop_n or 0.0) if box_force_stop_n is None else float(box_force_stop_n),
            object_n=float(object_force_stop_n or 0.0),
            held_object_n=(float(object_force_stop_n or 0.0) if held_object_force_stop_n is None
                           else float(held_object_force_stop_n)),
            box_push_m=float(box_push_stop_m or 0.0))
        t_begin = float(self.world.current_time)
        peak: dict[str, float] = {}
        pad_peak = np.zeros(2)
        st_flags = {"stopped": False, "reason": "", "gripping": False, "env_force": 0.0, "stop": None,
                    "pads": np.zeros(2), "prev": {}}
        n_steps = 0
        done_segments = 0
        prof = {"targets_s": 0.0, "simulate_s": 0.0, "contacts_s": 0.0, "bookkeeping_s": 0.0, "settle_checks_s": 0.0,
                "ticks": 0, "samples": 0}
        f0, pads0 = self._sample_all({}, self.reader)
        floors = contact_rules.stop_floors(f0, limits)
        st_flags["pads"] = pads0
        bin_before = self._bin_now()
        box_c0 = (np.asarray(self.bin.get_world_pose()[0], float) if self.bin is not None else None)
        obj_z0 = float(np.asarray(self.episode.get("object_pos", np.zeros(3)), float)[2]) if self.episode else 0.0

        def held_now() -> bool:
            return bool(self.grip_intent == "closed" and self.closure_cmd > 0.05 and np.all(st_flags["pads"] > 0.1))

        def stop(reason: str, info: dict | None = None) -> None:
            if not st_flags["stopped"]:
                st_flags["stopped"], st_flags["reason"], st_flags["stop"] = True, reason, info
                # hold where we are: the drive target becomes the measured pose
                self.q_cmd = np.asarray(self.robot.get_joint_positions()[self.ids], float)
                self.q_bias = np.zeros(6)

        def sample() -> None:
            t_c = time.perf_counter()
            held = held_now()
            now, pads = self._sample_all(peak, self.reader)
            st_flags["pads"] = pads
            env = [f for kk, f in now.items() if kk.split("-", 1)[-1] in ("table", "box") and not kk.startswith("packet")]
            st_flags["env_force"] = max(env) if env else 0.0
            q_now = np.asarray(self.robot.get_joint_positions(), float)
            if not np.all(np.isfinite(q_now)):
                now = {**now, "unstable": float("inf")}
            moved = (float(np.linalg.norm(np.asarray(self.bin.get_world_pose()[0], float)[:2] - box_c0[:2]))
                     if box_c0 is not None else 0.0)
            hit = contact_rules.check_stop(now, floors, limits, held=held, gripping=st_flags["gripping"],
                                           prev=st_flags["prev"], predict=predict_stop, box_moved_m=moved)
            st_flags["prev"] = now
            if hit is not None:
                hit["held"] = held
                if hit["kind"] == "unstable":
                    stop("physics became unstable (reset the episode)", hit)       # also with the stop disabled
                else:
                    key = hit["pair"]
                    if hit.get("pushed"):
                        who = "the held packet" if key.startswith("packet-") else \
                            {"arm": "the arm", "gripper": "the gripper"}.get(key.split("-", 1)[0], "the robot")
                        reason = f"{who} pushed the blue box {hit['moved_m'] * 1000:.0f} mm"
                    elif hit["kind"] == "held_object":
                        what = contact_rules.PART_TEXT.get(key.split("-", 1)[1], "an obstacle")
                        reason = (f"the held packet pushed against {what} (force {hit['force']:.0f} N "
                                  f"> {hit['threshold']:.0f} N; the grip is still closed)")
                    else:
                        what = {"box": "the blue box", "table": "the table/mat", "object": "the packet"}[
                            key.split("-", 1)[1]]
                        rel = "rising toward" if hit.get("predicted") else ">"
                        reason = f"contact force {hit['force']:.0f} N against {what} {rel} {hit['threshold']:.0f} N"
                    stop(reason, hit)
            prof["contacts_s"] += time.perf_counter() - t_c
            prof["samples"] += 1

        def tick(q_target, closure, do_sample: bool) -> None:
            nonlocal n_steps, pad_peak
            t_s = time.perf_counter()
            self._apply(self.q_cmd if st_flags["stopped"] else q_target, closure)
            t_m = time.perf_counter()
            prof["targets_s"] += t_m - t_s
            for _ in range(k_sub):
                self._sim_step()
            prof["simulate_s"] += time.perf_counter() - t_m
            before = n_steps
            n_steps += k_sub
            prof["ticks"] += 1
            if do_sample:
                sample()
            if n_steps // 10 != before // 10 and self.episode:      # ~every 10 ms: episode record
                t_b = time.perf_counter()
                pads = st_flags["pads"] if do_sample else self._pad_forces()
                pad_peak = np.maximum(pad_peak, pads)
                if self.closure_cmd > 0.05 and np.all(pads > 0.1):
                    self.episode["ever_held"] = True
                z = float(np.asarray(self.packet.get_world_pose()[0], float)[2])
                self.episode["max_lift_m"] = max(float(self.episode.get("max_lift_m", 0.0)), z - obj_z0)
                prof["bookkeeping_s"] += time.perf_counter() - t_b

        def run_steps(n_phys: int, q_target, closure) -> None:
            """Hold one target for ~n_phys physics steps, sampling contacts at the end."""
            n_t = max(1, int(round(n_phys / k_sub)))
            for i in range(n_t):
                tick(q_target, closure, do_sample=(i == n_t - 1))

        every = max(1, int(contact_every))
        groups = [np.flatnonzero(group == gid) for gid in dict.fromkeys(group.tolist())]
        for rows in groups:
            q_from = self.q_cmd.copy()
            dur = float(np.sum(durations[rows]))
            n = max(1, int(round(max(dur, self.dt * k_sub) / (self.dt * k_sub))))
            for i, q_k in enumerate(motion.interpolate_group(q_from, q[rows], n)):
                self.q_cmd = q_k
                tick(q_k, self.closure_cmd, do_sample=((i + 1) % every == 0 or i == n - 1))
                if st_flags["stopped"]:
                    break
            if st_flags["stopped"]:
                break
            g = gripper[rows][np.isfinite(gripper[rows])]
            if len(g):
                st_flags["gripping"] = True
                self._actuate_gripper(float(g[-1]), lambda qq, cc, nn: run_steps(nn, qq, cc), settle,
                                      lambda: st_flags["stopped"], lambda: st_flags["pads"])
                st_flags["gripping"] = False
                # a new grasp: its own load on the packet's contacts is the new baseline
                base = {k: v for k, v in (st_flags["prev"] or {}).items() if k.startswith("packet-")}
                floors.update(contact_rules.stop_floors(base, limits))
            done_segments += 1
            if st_flags["stopped"]:
                break
        # Settled = joints slow, the TCP barely moving and the joints on target
        # (integral action removes the gravity sag). The arm keeps a ~0.01
        # rad/s elbow micro-oscillation under PHANTOM's drive gains, so the
        # speed thresholds are loose and the timeout is the backstop.
        t_settle, settled = 0.0, False
        tcp_prev = self._tcp_measured()[0]
        block = max(k_sub, 10)
        while t_settle < settle["max_s"] - 1e-9:
            run_steps(block, self.q_cmd, self.closure_cmd)
            t_settle += block * self.dt
            t_k = time.perf_counter()
            # anti-windup: no integration after a stop or while pressing on something
            integrate = not st_flags["stopped"] and st_flags["env_force"] < ANTI_WINDUP_N
            q_err = self._integrate() if integrate else 0.0
            qd = np.abs(np.asarray(self.robot.get_joint_velocities()[self.ids], float))
            tcp_now = self._tcp_measured()[0]
            prof["settle_checks_s"] += time.perf_counter() - t_k
            if not np.all(np.isfinite(tcp_now)):
                stop("physics became unstable (reset the episode)", {"kind": "unstable", "pair": "unstable",
                                                                    "force": float("inf"), "threshold": UNSTABLE_FORCE_N})
                break
            v_tcp = float(np.linalg.norm(tcp_now - tcp_prev)) / (block * self.dt)
            tcp_prev = tcp_now
            if t_settle >= settle["min_s"] and qd.max() < settle["joint_speed_rad_s"] \
                    and v_tcp < settle["tcp_speed_m_s"] and q_err < settle["joint_err_rad"]:
                settled = True
                break
        final, _ = self._sample_all(peak, self.reader)
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
        prof["physics_s"] = prof["targets_s"] + prof["simulate_s"]          # legacy name
        bin_after = st["bin"]
        shift = (float(np.linalg.norm(np.asarray(bin_after["center"], float) - np.asarray(bin_before["center"], float)))
                 if bin_after is not None and bin_before is not None else 0.0)
        return {
            "state": st, "stopped": stopped, "stop_reason": stop_reason, "stop": st_flags["stop"],
            "segments_done": done_segments,
            "sim_s": float(self.world.current_time - t_begin), "settle_s": t_settle,
            "settled": settled, "physics_steps": n_steps,
            "unstable": "unstable" in stop_reason, "q_cmd": self.q_cmd.copy(), "q_err_max_rad": float(np.max(np.abs(st["q"] - self.q_cmd))),
            "tcp_err_m": float(np.linalg.norm(st["tcp_pos"] - fk_cmd[:3])),
            "contacts_peak_n": peak, "pad_object_peak_n": pad_peak,
            "contacts_final_n": final, "box_shift_m": shift,
            "gripper_mechanics_ok": mech_ok, "gripper_mechanics_note": mech_note,
            "qd_max_rad_s": float(np.max(np.abs(st["qd"]))), "profile": prof,
            "settings": {"reader": self.reader, "substeps": k_sub, "contact_every": every, "direct": self.direct,
                         "predict_stop": bool(predict_stop), "box_push_stop_m": float(box_push_stop_m or 0.0),
                         **self.physics_info()},
            "wall_s": time.perf_counter() - wall,
        }

    def _actuate_gripper(self, width_m: float, run, settle: dict, is_stopped, pads_now) -> None:
        """Robotiq-like position command with object detection.

        Closing ramps the command and, once both pads press on the packet,
        stops tightening at ``grip_squeeze`` closure past first contact —
        the real Robotiq also stops at the object and holds with its force
        limit. Driving the W2L linkage to full closure through a 35 mm packet
        destabilises PhysX (observed: the arm and packet were flung), and
        PHANTOM's own expert only commands 0.62 (scripted_expert.grip_close).
        ``run(q, closure, n_steps)`` steps the sim (sampling contacts at the end);
        ``pads_now()`` the pad forces of that last sample."""
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
                if np.all(np.asarray(pads_now()) > contact_n):
                    self.closure_cmd = min(target, self._measured_closure() + squeeze)
                    latched = True
            elif not closing:
                self.closure_cmd = target
            run(self.q_cmd, self.closure_cmd, chunk)
            t_g += chunk * self.dt
            cur = self._measured_closure()
            ramp_done = latched or not closing or self.closure_cmd >= target
            if ramp_done and t_g >= settle["grip_min_s"] and abs(cur - prev) < 0.002:
                break
            prev = cur

    def snapshot(self) -> dict:
        pos, quat = self.packet.get_world_pose()
        try:
            if self.reader == "matrix":
                m = self._robot_matrix()
                if not np.all(np.isfinite(m)):
                    raise RuntimeError("non-finite contact matrix")
                robot_obj = float(np.linalg.norm(m[:, self.cv_obj_col, :], axis=1).sum())
            else:
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
                "tcp_pos": st["tcp_pos"], "bin": st["bin"], "t": st["t"]}

    def check_goal(self) -> dict:
        if not self.episode:
            raise RuntimeError("reset first")
        snap = self.snapshot()
        res = task_registry.get_task(self.episode["task"]).evaluate(snap, self.episode)
        res["metrics"] = {**res["metrics"], "object_pos": snap["object_pos"].tolist(),
                          "box_center": np.asarray(snap["bin"]["center"], float).tolist(),
                          "box_yaw": float(snap["bin"]["yaw"]), "box_tilt_deg": float(snap["bin"].get("tilt_deg", 0.0)),
                          "box_shift_m": float(snap["bin"].get("shift_m", 0.0))}
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
    ap.add_argument("--solver-iterations", type=int, default=None, help="articulation position iterations "
                    "(PHANTOM scene: 64)")
    ap.add_argument("--solver-velocity-iterations", type=int, default=None, help="(PHANTOM scene: 8)")
    ap.add_argument("--dt", type=float, default=None, help="physics step, s (PHANTOM scene: 0.001; > 1 ms is "
                    "rejected by PHANTOM for the adaptive W2L gripper and runs only with a warning)")
    ap.add_argument("--forearm-approx", default=None, help="forearm collider approximation (PHANTOM scene: "
                    "convexDecomposition)")
    ap.add_argument("--num-threads", type=int, default=None, help="PhysX CPU worker threads")
    ap.add_argument("--no-self-collisions", action="store_true", help="articulation self-collisions off")
    ap.add_argument("--no-legacy-contact-views", action="store_true",
                    help="build only the contact matrix, not PHANTOM's per-body views (reader is then matrix)")
    ap.add_argument("--no-usd-writeback", action="store_true",
                    help="do not write body transforms to USD after every physics step (only before renders)")
    ap.add_argument("--render-updates", type=int, default=4, help="app updates per observation (RTX pipeline depth)")
    ap.add_argument("--once", action="store_true", help="exit after the first client disconnects")
    ap.add_argument("--extra-cameras", default="", help="virtual views next to the D435, comma list "
                    "(top, side; controlr/robot/isaac/cameras.py)")
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
        rig = IsaacRig(app, args.phantom, args.scene, args.out, physics={
            "dt": args.dt, "solver_position_iterations": args.solver_iterations,
            "solver_velocity_iterations": args.solver_velocity_iterations,
            "forearm_collision_approximation": args.forearm_approx, "num_threads": args.num_threads,
            "legacy_contact_views": not args.no_legacy_contact_views,
            "self_collisions": not args.no_self_collisions,
            "usd_writeback": not args.no_usd_writeback},
            extra_cameras=[c for c in args.extra_cameras.split(",") if c])
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
