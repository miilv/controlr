"""Kinematic mock robot: no physics, no GPU, synthetic but informative frames.

WHY: the whole harness (prompt, grammar, safety, feedback, caching, logging)
must be testable — and cheaply runnable against the real LLM — without Isaac.
The mock therefore uses the real UR3 CB3 spec and kinematics, the calibrated
D435 pose of PHANTOM's Isaac scene (so overlays and the model's sense of
perspective match the sim), and renders a scene a vision model can actually
act on: table with a 50 mm-grid mat, coloured objects with drop shadows, the
arm, the gripper with its current opening and a TCP marker with a drop line.

Execution is a jump: the state goes straight to each action's ``q_target``
(time advances by the joint-space travel time at the spec's joint speed plus
a settle time). Grasping is a rule, not contact physics: closing within
``grasp_radius_m`` (15 mm) of the cube centre attaches the cube; opening
releases it and it drops onto the table.

Tasks (``task_cfg["name"]``): ``reach`` (TCP to a red ball) and
``pick_place`` (red cube into the green square). ``task_cfg["params"]``
overrides positions/tolerances; positions are randomised from ``seed``.

Cameras (``params.cameras``, default ``[scene]``): ``scene`` is the Isaac
D435 view; ``top`` is a synthetic top-down camera 1 m above the mat — enough
to exercise the multi-camera axis (``observation.cameras: [scene, top]``,
tiled or separate) without a GPU.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from controlr.config import SafetyConfig
from controlr.robot.base import Robot
from controlr.robot.kinematics import UR3Kinematics, matrix_to_rpy
from controlr.robot.safety import SafetyEnvelope
from controlr.robot.spec import TABLE_Z, ur3_cb3_spec
from controlr.types import (
    Action, CameraInfo, EventLevel, ExecReport, GoalReport, Observation, RobotState, SafetyEvent,
)

# The calibrated D435 the Isaac backend renders with (PHANTOM's current waffle
# scene, configs/sim/waffles_w2l_adaptive_parallel_dt1ms_20260909.json, as
# reported by controlr.robot.isaac at reset; 640x480). Integration check
# 2026-10-02: the older waffles.json fit differed by up to 135 px in K and
# 75 mm in pose, so mock overlays did not match Isaac frames.
D435_K = np.array([[609.28173828125, 0.0, 337.8138732910156],
                   [0.0, 608.1331176757812, 249.65126037597656],
                   [0.0, 0.0, 1.0]])
D435_CAM_FROM_BASE = np.array([
    [0.9999720414746368, -0.007215535335001406, 0.0019627325028448842, 0.348887964518517],
    [-0.005669109656577375, -0.9026929426481266, -0.43024796628103645, 0.005241988497421483],
    [0.0048762141821374816, 0.4302248102365734, -0.9027086103456389, 1.0545256885198455],
    [0.0, 0.0, 0.0, 1.0],
])
D435_WORLD_FROM_CV = np.linalg.inv(D435_CAM_FROM_BASE)
TABLE_CENTER_XY = (-0.23, 0.0)        # waffles.json table
TABLE_SIZE_XY = (1.7, 0.9)
MAT_CENTER_XY = (-0.31, -0.10)        # mock mat: placed in the tool-down reachable area
MAT_SIZE_XY = (0.30, 0.45)

TASK_INSTRUCTIONS = {
    "reach": "Move the gripper's TCP (the point between the fingertips) to the centre of the red ball.",
    "pick_place": "Pick up the red cube and place it inside the green square on the table, then open the gripper.",
}


def top_camera(width: int = 640, height: int = 480, name: str = "top") -> CameraInfo:
    """Synthetic top-down camera above the mock mat (image right = +x, image up = +y)."""
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ np.array([MAT_CENTER_XY[0], MAT_CENTER_XY[1], 1.0])
    f = 0.9 * width
    K = np.array([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])
    return CameraInfo(name=name, width=width, height=height, K=K, T_cam_base=T)


def d435_camera(width: int = 640, height: int = 480, name: str = "scene") -> CameraInfo:
    """The Isaac scene's calibrated D435, scaled to ``width``x``height``."""
    K = D435_K.copy()
    K[0] *= width / 640.0
    K[1] *= height / 480.0
    return CameraInfo(name=name, width=width, height=height, K=K,
                      T_cam_base=D435_CAM_FROM_BASE.copy())


@dataclass
class _Scene:
    task: str = "reach"
    target: np.ndarray | None = None          # reach: ball centre
    tolerance_m: float = 0.015
    cube: np.ndarray | None = None            # pick_place: cube centre
    cube_size: float = 0.04
    cube_yaw: float = 0.0
    zone: np.ndarray | None = None            # pick_place: square centre (x, y)
    zone_size: float = 0.10
    held: bool = False
    hold_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    start_dist: float = 1.0


class MockRobot(Robot):
    """Kinematic UR3 CB3 + 2F-85 with synthetic rendering (see module doc)."""

    def __init__(self, params: dict | None = None, *, safety: SafetyConfig | None = None) -> None:
        p = dict(params or {})
        spec_kw = {k: tuple(p[k]) if isinstance(p[k], list) else p[k]
                   for k in ("home_q", "table_z", "tcp_offset", "gripper_max_mm") if k in p}
        self.spec = ur3_cb3_spec(**spec_kw)
        self.kin = UR3Kinematics(tcp_offset=self.spec.tcp_offset)
        w, h = int(p.get("width", 640)), int(p.get("height", 480))
        self.camera = d435_camera(w, h, str(p.get("camera", "scene")))
        self.cameras = {self.camera.name: self.camera}
        for name in p.get("cameras") or []:
            if name == "top":
                self.cameras["top"] = top_camera(w, h)
            elif name != self.camera.name:
                raise ValueError(f"MockRobot: unknown camera {name!r} (scene | top)")
        self.settle_s = float(p.get("settle_s", 0.2))
        self.grasp_radius_m = float(p.get("grasp_radius_m", 0.015))
        self.supersample = int(p.get("supersample", 2))
        # Actions arriving without q_target (not filtered) are resolved with the
        # configured envelope so the mock never executes an unchecked target.
        self._fallback = SafetyEnvelope(self.spec, safety or SafetyConfig(), self.kin)
        self.task_instruction = TASK_INSTRUCTIONS["reach"]
        self._t = 0.0
        self._q = np.array(self.spec.home_q, dtype=float)
        self._grip = self.spec.gripper_max_mm / 1000.0
        self._grip_closed = False
        self._scene = _Scene()

    # ------------------------------------------------------------------ reset
    def reset(self, task_cfg: dict, seed: int | None = None) -> Observation:
        if dataclasses.is_dataclass(task_cfg):
            task_cfg = dataclasses.asdict(task_cfg)
        task_cfg = dict(task_cfg or {})
        name = task_cfg.get("name") or "reach"
        prm = dict(task_cfg.get("params") or {})
        rng = np.random.default_rng(seed)
        self._t = 0.0
        self._q = np.array(prm.get("home_q", self.spec.home_q), dtype=float)
        self._grip = self.spec.gripper_max_mm / 1000.0
        self._grip_closed = False
        tz = self.spec.table_z if self.spec.table_z is not None else TABLE_Z
        sc = _Scene(task=name)
        sc.tolerance_m = float(prm.get("tolerance_m", 0.015))
        if name == "reach":
            t = prm.get("target") or [rng.uniform(-0.38, -0.25), rng.uniform(-0.25, 0.05),
                                      tz + rng.uniform(0.04, 0.12)]
            sc.target = np.array(t, dtype=float)
            sc.start_dist = float(np.linalg.norm(self._tcp()[0] - sc.target))
        elif name == "pick_place":
            sc.cube_size = float(prm.get("cube_size", 0.04))
            sc.zone_size = float(prm.get("zone_size", 0.10))
            c = prm.get("cube_xy") or [rng.uniform(-0.37, -0.27), rng.uniform(-0.26, -0.12)]
            z = prm.get("zone_xy") or [rng.uniform(-0.35, -0.27), rng.uniform(0.0, 0.08)]
            sc.cube = np.array([c[0], c[1], tz + sc.cube_size / 2], dtype=float)
            sc.cube_yaw = float(prm.get("cube_yaw", rng.uniform(-0.4, 0.4)))
            sc.zone = np.array(z, dtype=float)
            sc.start_dist = float(np.linalg.norm(sc.cube[:2] - sc.zone))
        else:
            raise ValueError(f"MockRobot: unknown task {name!r} (reach | pick_place)")
        self._scene = sc
        self.task_instruction = task_cfg.get("instruction") or TASK_INSTRUCTIONS[name]
        obs = self.observe()
        self._fallback.reset(obs.state)
        return obs

    # ------------------------------------------------------------------ state
    def _tcp(self, q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        return self.kin.fk(self._q if q is None else q)

    def state(self) -> RobotState:
        pos, rv = self._tcp()
        return RobotState(t=self._t, q=self._q.copy(), tcp_pos=pos, tcp_rotvec=rv,
                          gripper_mm=self._grip * 1000.0, gripper_closed=self._grip_closed,
                          holding=self._scene.held if self._scene.task == "pick_place" else None)

    def observe(self) -> Observation:
        st = self.state()
        return Observation(t=self._t, images={n: self.render(c) for n, c in self.cameras.items()},
                           cameras=dict(self.cameras), state=st)

    # ---------------------------------------------------------------- execute
    def execute(self, actions: list[Action]) -> ExecReport:
        before = self.state()
        events: list[SafetyEvent] = []
        executed: list[Action] = []
        dur = 0.0
        speeds = np.array([j.max_velocity for j in self.spec.joints])
        for a in actions:
            if a.q_target is None:
                filt, ev = self._fallback.filter([a], self.state())
                events.extend(ev)
                if not filt:
                    continue
                a = filt[0]
            executed.append(a)
            qt = np.asarray(a.q_target, dtype=float)
            dur += float(np.max(np.abs(qt - self._q) / speeds)) + self.settle_s
            yaw0 = matrix_to_rpy(self.kin.fk_matrix(self._q)[:3, :3])[2]
            self._q = qt
            sc = self._scene
            if sc.held:
                T = self.kin.fk_matrix(self._q)
                sc.cube = T[:3, 3] + sc.hold_offset
                tz = self.spec.table_z if self.spec.table_z is not None else TABLE_Z
                sc.cube[2] = max(sc.cube[2], tz + sc.cube_size / 2)   # a held cube cannot sink into the table
                sc.cube_yaw += matrix_to_rpy(T[:3, :3])[2] - yaw0
            if a.gripper is not None:
                events.extend(self._gripper(float(a.gripper)))
        self._t += dur
        return ExecReport(requested=list(actions), executed=executed, events=events,
                          state_before=before, state_after=self.state(), duration_s=dur)

    def _gripper(self, width: float) -> list[SafetyEvent]:
        sc = self._scene
        closing = width < self._grip - 1e-6
        self._grip_closed = width < 0.5 * self.spec.gripper_max_mm / 1000.0
        out: list[SafetyEvent] = []
        if sc.cube is None:
            self._grip = width
            return out
        tcp = self._tcp()[0]
        if closing and not sc.held:
            if np.linalg.norm(tcp - sc.cube) <= self.grasp_radius_m and width < sc.cube_size:
                sc.held = True
                sc.hold_offset = sc.cube - tcp
                self._grip = sc.cube_size            # fingers stop on the cube
                out.append(SafetyEvent(EventLevel.INFO, "grasp", "gripper closed on the cube (holding)"))
                return out
            out.append(SafetyEvent(EventLevel.INFO, "grasp", "gripper closed on nothing"))
        elif sc.held and width > sc.cube_size:
            sc.held = False
            tz = self.spec.table_z if self.spec.table_z is not None else TABLE_Z
            sc.cube = np.array([sc.cube[0], sc.cube[1], tz + sc.cube_size / 2])
            out.append(SafetyEvent(EventLevel.INFO, "release", "gripper opened, cube released onto the table"))
        elif sc.held:
            width = sc.cube_size                     # cannot close further on a held cube
        self._grip = width
        return out

    # ------------------------------------------------------------------- goal
    def check_goal(self) -> GoalReport:
        sc = self._scene
        tcp = self._tcp()[0]
        if sc.task == "reach":
            d = float(np.linalg.norm(tcp - sc.target))
            ok = d <= sc.tolerance_m
            prog = float(np.clip(1.0 - d / max(sc.start_dist, 1e-6), 0.0, 1.0))
            msg = "target reached" if ok else f"TCP is {d * 1000:.0f} mm from the target"
            return GoalReport(ok, 1.0 if ok else prog, msg, {"dist_m": d})
        # pick_place
        dz = float(np.linalg.norm(sc.cube[:2] - sc.zone))
        inside = bool(np.all(np.abs(sc.cube[:2] - sc.zone) <= sc.zone_size / 2))
        tz = self.spec.table_z if self.spec.table_z is not None else TABLE_Z
        on_table = abs(sc.cube[2] - (tz + sc.cube_size / 2)) < 1e-3
        ok = inside and on_table and not sc.held
        if ok:
            return GoalReport(True, 1.0, "cube placed in the square", {"cube_zone_dist_m": dz})
        if sc.held:
            prog = 0.5 + 0.4 * float(np.clip(1.0 - dz / max(sc.start_dist, 1e-6), 0.0, 1.0))
            msg = f"holding the cube, {dz * 1000:.0f} mm from the square centre"
        else:
            dg = float(np.linalg.norm(tcp - sc.cube))
            prog = 0.4 * float(np.clip(1.0 - dg / 0.3, 0.0, 1.0))
            msg = (f"cube not in the square ({dz * 1000:.0f} mm from its centre); "
                   f"TCP is {dg * 1000:.0f} mm from the cube")
        return GoalReport(False, prog, msg, {"cube_zone_dist_m": dz, "held": sc.held})

    # -------------------------------------------------------------- rendering
    def render(self, camera: CameraInfo | None = None) -> np.ndarray:
        return _Renderer(self, self.supersample, camera).draw()


class _Renderer:
    """Pinhole rendering with PIL polygons/lines, painter's ordering by depth.
    Deterministic for identical state (needed for cache-stable JPEGs)."""

    def __init__(self, robot: MockRobot, ss: int, camera: CameraInfo | None = None) -> None:
        self.r = robot
        cam = camera or robot.camera
        self.ss = max(1, ss)
        self.W, self.H = cam.width * self.ss, cam.height * self.ss
        self.K = cam.K.copy()
        self.K[:2] *= self.ss
        self.T = cam.T_cam_base
        self.img = Image.new("RGB", (self.W, self.H), (58, 60, 66))
        self.d = ImageDraw.Draw(self.img)
        self.tz = robot.spec.table_z if robot.spec.table_z is not None else TABLE_Z

    # -- projection
    def cam(self, P) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, float))
        return P @ self.T[:3, :3].T + self.T[:3, 3]

    def proj(self, P) -> np.ndarray:
        C = self.cam(P)
        z = np.maximum(C[:, 2], 1e-6)
        return np.stack([self.K[0, 0] * C[:, 0] / z + self.K[0, 2], self.K[1, 1] * C[:, 1] / z + self.K[1, 2]], 1)

    def poly(self, P, fill, outline=None, width=1) -> None:
        """Polygon with near-plane clipping (camera z > 1 cm)."""
        C = self.cam(P)
        near = 0.01
        out = []
        n = len(C)
        for i in range(n):
            a, b = C[i], C[(i + 1) % n]
            if a[2] >= near:
                out.append(a)
            if (a[2] >= near) != (b[2] >= near):
                t = (near - a[2]) / (b[2] - a[2])
                out.append(a + t * (b - a))
        if len(out) < 3:
            return
        C = np.array(out)
        uv = np.stack([self.K[0, 0] * C[:, 0] / C[:, 2] + self.K[0, 2],
                       self.K[1, 1] * C[:, 1] / C[:, 2] + self.K[1, 2]], 1)
        self.d.polygon([tuple(p) for p in uv], fill=fill, outline=outline, width=width * self.ss)

    def line(self, P, fill, width=1) -> None:
        uv = self.proj(P)
        self.d.line([tuple(p) for p in uv], fill=fill, width=max(1, int(round(width * self.ss))), joint="curve")

    def dot(self, P, r_px, fill, outline=None) -> None:
        u, v = self.proj(P)[0]
        r = r_px * self.ss
        self.d.ellipse([u - r, v - r, u + r, v + r], fill=fill, outline=outline, width=self.ss)

    def rect_xy(self, cx, cy, sx, sy, z, yaw=0.0) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        pts = []
        for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            x, y = dx * sx / 2, dy * sy / 2
            pts.append([cx + c * x - s * y, cy + s * x + c * y, z])
        return np.array(pts)

    def depth(self, P) -> float:
        return float(self.cam(P)[:, 2].mean())

    # -- scene
    def draw(self) -> np.ndarray:
        r, sc, tz = self.r, self.r._scene, self.tz
        self.poly(self.rect_xy(*TABLE_CENTER_XY, *TABLE_SIZE_XY, tz), fill=(150, 146, 140))
        self.poly(self.rect_xy(*MAT_CENTER_XY, *MAT_SIZE_XY, tz + 1e-4), fill=(70, 86, 104))
        x0, y0 = MAT_CENTER_XY[0] - MAT_SIZE_XY[0] / 2, MAT_CENTER_XY[1] - MAT_SIZE_XY[1] / 2
        for k in range(int(round(MAT_SIZE_XY[0] / 0.05)) + 1):
            x = x0 + 0.05 * k
            self.line([[x, y0, tz], [x, y0 + MAT_SIZE_XY[1], tz]], (110, 128, 148), 1)
        for k in range(int(round(MAT_SIZE_XY[1] / 0.05)) + 1):
            y = y0 + 0.05 * k
            self.line([[x0, y, tz], [x0 + MAT_SIZE_XY[0], y, tz]], (110, 128, 148), 1)
        # robot base disc (may be off-screen)
        self.poly(self.rect_xy(0.0, 0.0, 0.13, 0.13, tz), fill=(90, 90, 96))
        if sc.zone is not None:
            self.poly(self.rect_xy(sc.zone[0], sc.zone[1], sc.zone_size, sc.zone_size, tz + 2e-4),
                      fill=(70, 160, 80), outline=(30, 110, 40), width=2)
        tcp_T = r.kin.fk_matrix(r._q)
        tcp = tcp_T[:3, 3]
        # drop shadows: TCP and objects on the table
        self.line([tcp, [tcp[0], tcp[1], tz]], (230, 220, 90), 1)
        self.dot([tcp[0], tcp[1], tz], 4, (40, 40, 40))
        items: list = []
        if sc.target is not None:
            t = sc.target
            self.line([t, [t[0], t[1], tz]], (200, 60, 60), 1)
            self.dot([t[0], t[1], tz], 5, (60, 50, 50))
            items.append((self.depth(t), lambda: self._ball(t)))
        if sc.cube is not None:
            c = sc.cube
            if c[2] > tz + sc.cube_size / 2 + 1e-3:
                self.poly(self.rect_xy(c[0], c[1], sc.cube_size, sc.cube_size, tz + 3e-4, sc.cube_yaw),
                          fill=(60, 60, 60))
            items.append((self.depth(c), self._cube))
        items.append((self.depth(tcp), self._arm))
        for _, f in sorted(items, key=lambda it: -it[0]):     # far first
            f()
        img = self.img
        if self.ss > 1:
            img = img.resize((self.W // self.ss, self.H // self.ss), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8).copy()

    def _ball(self, t) -> None:
        # radius ~12 mm projected
        u, v = self.proj(t)[0]
        z = self.cam(t)[0, 2]
        rad = self.K[0, 0] * 0.012 / z
        self.d.ellipse([u - rad, v - rad, u + rad, v + rad], fill=(220, 40, 40), outline=(120, 10, 10),
                       width=self.ss)

    def _cube(self) -> None:
        sc = self.r._scene
        c, s, yaw = sc.cube, sc.cube_size, sc.cube_yaw
        bot = self.rect_xy(c[0], c[1], s, s, c[2] - s / 2, yaw)
        top = self.rect_xy(c[0], c[1], s, s, c[2] + s / 2, yaw)
        cam_pos = np.linalg.inv(self.T)[:3, 3]
        faces = [(top, (225, 60, 50))]
        for i in range(4):
            j = (i + 1) % 4
            faces.append((np.array([bot[i], bot[j], top[j], top[i]]), (170, 40, 35) if i % 2 else (195, 50, 42)))
        for f, col in faces:
            ctr = f.mean(0)
            nrm = np.cross(f[1] - f[0], f[2] - f[0])
            if np.dot(nrm, ctr - c) < 0:
                nrm = -nrm
            if np.dot(nrm, cam_pos - ctr) > 0:            # back-face culling
                self.poly(f, fill=col, outline=(90, 20, 20))

    def _arm(self) -> None:
        r = self.r
        F = r.kin.frames(r._q)
        pts = [F[i][:3, 3] for i in range(7)]
        # links between DH origins (elbow/wrist offsets give a recognisable UR silhouette)
        for a, b in zip(pts[:-1], pts[1:]):
            self.line([a, b], (175, 180, 190), 9)
        for p in pts[1:]:
            self.dot(p, 6, (120, 160, 210), (60, 80, 110))
        tool = F[6]
        z, y = tool[:3, 2], tool[:3, 1]
        o = tool[:3, 3]
        body_end = o + z * 0.11                  # Robotiq housing
        self.line([o, body_end], (40, 40, 44), 14)
        tcp = F[7][:3, 3]
        w = r._grip / 2
        for sgn in (-1, 1):
            base = body_end + sgn * y * (w + 0.008)
            tip = tcp + sgn * y * (w + 0.004)
            self.line([body_end, base, tip], (30, 30, 30), 5)
            self.dot(tip, 3, (205, 205, 215), (30, 30, 30))      # finger pads
        col = (255, 120, 0) if r._grip_closed else (255, 230, 40)
        self.dot(tcp, 4, col, (0, 0, 0))
