"""HarnessRobot: robot-harness's robot-server verbs over our Skills, so Mikhail's RoboDojo skills run unchanged.

Stream P's port. Mikhail's skills (``dojo/skills/ported/skills/<dir>/``, copied unchanged from robot-harness, see
``SOURCE.txt``) call ``robot.obs / reach / move_to / move_delta / gripper / hold / return_home / skill`` on his robot
server. This class answers those calls with the same reply fields and the same motion semantics, executed through
our ``Executor`` (every env step goes through the env the runtime / G3 drives):

* semantics copied from robot-harness a21dd6f ``harness/robodojo_server.py`` (``Server.cmd_*``: reply envelope,
  orientation arguments, the reserve rule, home tolerances) and ``harness/rd_shim/deploy.py`` (``Rig.op_*``: a joint
  line to the IK goal from the MEASURED joints, the FK line check, the Cartesian-line fallback, trapezoidal timing at
  0.08 rad / joint / step empty-handed and 0.05 holding, the gripper ramp of 0.25 per step, ``cloud_grasp``);
* differences: our own IK (``X5Kin`` with the harness's 0.1501 m TCP) stands in for cuRobo's ``solve_ik_to_joint``,
  and our table-only ``Motion.to_pose`` (a joint move, else up / across / down) for cuRobo's ``plan_path``
  (``path_mode`` "curobo_plan", which the drag skills refuse as free-space either way); object AABBs and grasp
  candidates come from the catalogue mesh at the object's pose (the shim: the USD mesh at the live pose).

Where object state comes from (the legality rule of the port):

* ``truth=True`` only in G3 mode gt (the provider is ``GTProvider`` itself): objects from its WorldState (the
  simulator's poses), finger joints from the env's dev-only measured gripper. This is the planning ceiling.
* otherwise (``truth=False``, every runtime mode and every G3 mode-B hook): objects from ``sk.refresh()``, i.e. the
  runtime's belief; the env is behind ``TruthGuard`` (its ground-truth accessors raise ``TruthDenied``); the finger
  state comes from RoboDojo's official gripper reading when the observation has it
  (``<arm>_ee_joint_state_official``), else from the belief's own attachment (``provider.belief.attached``).
* never: checker labels. ``obs.objects[inst]["label"]`` is always None; instance names are
  ``<category>_<asset index>_<n>`` from the object's identity (``<category>_<n>`` when the identity is unread,
  ``ObjectState.identity_known`` False: a skill that needs the index then cannot ground it and says so).
  A route that must name an object passes ``"@<belief label>"``, translated here into that object's instance name.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
import traceback

import numpy as np

from dojo.skills.executor import BudgetExceeded, EpisodeEnded
from dojo.skills.kinematics import ARMS, X5Kin

PORTED_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
SOURCE_COMMIT = "a21dd6f6f2db323bd0fe84f8afd20312cb8dffc6"   # robot-harness, the vendored skills' origin

# robot-harness harness/rd_shim/deploy.py + harness/robodojo_server.py constants (a21dd6f)
GRASP_OFF = 0.1501                      # flange (link6) -> grasp point, along flange +x
TABLE_Z = 0.765
LIM_HOLD = 0.05                         # rad / joint / env step with the gripper closed
LIM_EMPTY = 0.08                        # rad / joint / env step empty-handed
GRIP_STEP = 0.25                        # gripper range per env step
FINGER_MAX = 0.044                      # m per finger, fully open
EMPTY_Q = 0.003                         # m per finger: stopped above this while closing = something held
MIN_GRASP_Z = TABLE_Z + 0.010
TOP_DOWN = np.array([0.5, -0.5, 0.5, 0.5])        # tool axis world -z, jaws close along world x at yaw 0
HOME_TOL_M, HOME_TOL_DEG = 0.15, 20.0
REACH_MAX = 0.70                        # flange distance from the arm's mount
MOUNTS = {"left": np.array([-0.30, -0.45, 0.765]), "right": np.array([0.30, -0.45, 0.765])}
GRIP_SCALE = (-0.01, 0.044)             # finger joint for g = 0 / 1 (robot_config.yml gripper_scale)
COMMANDS = ("obs", "move_to", "move_delta", "gripper", "hold", "return_home", "reach", "skill", "held_look")
# the runtime's control-flow exceptions (dojo.runtime.belief: a confirmed divergence re-plans, a lost object is searched):
# they leave a vendored skill untouched, as they leave our own skills (matched by name: skills never import the runtime)
PASS_THROUGH = ("Divergence", "LostObject")
NO_OBJECT_TYPES = ("Fluid", "Garment")  # no mesh body to grasp (the shim's cloud of a fluid is empty: no aabb)
# the env's ground-truth accessors (dojo.skills.env.RunnerEnv / KinematicEnv): denied outside G3 mode gt
TRUTH_NAMES = frozenset({"gt_objects", "reward", "score", "garment_gt", "actors_gt", "measured_gripper", "robot_gt",
                         "finger_joints", "objects", "attached", "r"})


class CmdError(Exception):
    def __init__(self, msg, hint=""):
        super().__init__(msg)
        self.hint = hint


class TruthDenied(BaseException):
    """A ground-truth read outside G3 mode gt (the port's legality guard). A BaseException: no ``except Exception`` in
    a vendored skill can swallow it."""


class TruthGuard:
    """The env as the port sees it outside G3 mode gt: actions and proprioception pass, ground truth raises."""

    def __init__(self, env):
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name):
        if name in TRUTH_NAMES:
            raise TruthDenied(f"env.{name} is ground truth: the ported skills read the belief outside G3 mode gt")
        return getattr(object.__getattribute__(self, "_env"), name)


# ------------------------------------------------------------------ math (robot-harness rd_shim / server, unchanged)
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qmat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_q(R):
    R = np.asarray(R, float)
    t = float(np.trace(R))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def axis_q(axis, rad):
    q = np.zeros(4)
    q[0] = math.cos(rad / 2)
    q[1 + axis] = math.sin(rad / 2)
    return q


def rotvec_q(rv):
    rv = np.asarray(rv, float)
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.array([1.0, 0, 0, 0])
    return np.concatenate([[math.cos(th / 2)], math.sin(th / 2) * rv / th])


def tool_quat(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """Top-down flange quaternion rotated about world x (pitch: tilts the tool toward +y), y (roll), z (yaw)."""
    r = axis_q(0, math.radians(pitch_deg))
    r = qmul(axis_q(1, math.radians(roll_deg)), r)
    r = qmul(axis_q(2, math.radians(yaw_deg)), r)
    return qmul(r, TOP_DOWN)


def flange_from_tcp(pos, quat):
    quat = np.asarray(quat, float) / np.linalg.norm(quat)
    return np.concatenate([np.asarray(pos, float) - GRASP_OFF * qmat(quat)[:, 0], quat])


def tcp_of(flange):
    f = np.asarray(flange, float).reshape(-1)
    return f[:3] + GRASP_OFF * qmat(f[3:7])[:, 0]


def qangle_deg(a, b):
    d = abs(float(np.dot(np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b))))
    return math.degrees(2 * math.acos(min(1.0, d)))


def slerp(a, b, t):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def wrap_half(a):
    return (a + math.pi / 2) % math.pi - math.pi / 2


def polyline_arclen(path):
    path = np.asarray(path, float)
    if len(path) < 2:
        return np.zeros(1)
    seg = np.abs(np.diff(path, axis=0)).max(axis=1)          # inf-norm per segment
    return np.concatenate([[0.0], np.cumsum(seg)])


def trapezoid(L, vmax, ramp):
    """Minimal n and per-step arc positions s_1..s_n (s_n = L) so that every step moves <= vmax,
    with `ramp` steps of linear speed-up and slow-down at the ends."""
    if L <= 1e-9:
        return 0, np.zeros(0)
    n = max(1, int(math.ceil(L / vmax - 1e-9)))
    while True:
        t = np.arange(1, n + 1, dtype=float)
        v = vmax * np.minimum(1.0, np.minimum(t, n + 1 - t) / (ramp + 1.0)) if ramp > 0 else np.full(n, vmax)
        tot = float(v.sum())
        if tot >= L - 1e-9:
            s = np.cumsum(v * (L / tot))
            s[-1] = L
            return n, s
        n += 1


def sample_path(path, arclen, s):
    path = np.asarray(path, float)
    if len(path) < 2:
        return np.repeat(path[-1:], len(s), axis=0)
    return np.stack([np.interp(s, arclen, path[:, j]) for j in range(path.shape[1])], axis=1)


def compact(x, nd=4):
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return compact(x.tolist(), nd)
    if isinstance(x, dict):
        return {k: compact(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [compact(v, nd) for v in x]
    return x


def _l(x, nd=5):
    return [round(float(v), nd) for v in np.asarray(x, float).reshape(-1)]


def cloud_grasp(P, finger_half=0.010, max_open=0.085):
    """Top-down grasp candidates from the mesh cloud (robot-harness rd_shim ``Rig.cloud_grasp``, unchanged)."""
    P = P[P[:, 2] > TABLE_Z + 0.002]
    if len(P) < 5:
        return []
    xy = P[:, :2]
    c0 = xy.mean(0)
    _, evec = np.linalg.eigh(np.cov((xy - c0).T))
    major = evec[:, 1]
    half_len = float(np.abs((xy - c0) @ major).max())
    found = []
    for frac in (0.0, 0.15, -0.15, 0.3, -0.3, 0.45, -0.45):
        c = c0 + frac * half_len * major
        for yaw in np.radians(np.arange(-90, 90, 15)):
            d = np.array([math.cos(yaw), math.sin(yaw)])
            n = np.array([-d[1], d[0]])
            rel = xy - c
            S = np.abs(rel @ n) < finger_half
            if S.sum() < 5:
                continue
            zs = P[S, 2]
            zt, zb = float(zs.max()), float(zs.min())
            h = zt - zb
            zg = max(zt - min(0.025, 0.5 * h), MIN_GRASP_Z)
            band = S & (P[:, 2] >= zg - 0.010)
            if band.sum() < 5:
                continue
            sd = rel[band] @ d
            lo, hi = float(sd.min()), float(sd.max())
            w = hi - lo
            if w > max_open:
                continue
            score = w + abs(frac) * half_len + 2.0 * max(0.0, 0.02 - h)
            mid = c + d * (lo + hi) / 2
            found.append({"score": score, "xy": mid, "yaw": wrap_half(yaw), "w": w, "zt": zt, "zg": zg, "h": h})
    found.sort(key=lambda g: g["score"])
    ranked = []
    for g in found:
        if all(np.linalg.norm(g["xy"] - r["xy"]) > 0.02 or abs(wrap_half(g["yaw"] - r["yaw"])) > math.radians(30)
               for r in ranked):
            ranked.append(g)
        if len(ranked) >= 4:
            break
    return [{"pos": _l([g["xy"][0], g["xy"][1], g["zg"]], 4), "yaw_deg": round(math.degrees(g["yaw"]), 1),
             "jaw_width_m": round(g["w"], 4), "top_z": round(g["zt"], 4), "height_m": round(g["h"], 4)} for g in ranked]


def object_cloud(o, max_pts=6000) -> np.ndarray:
    """World points of an object's catalogue mesh at its (believed or true) pose; a bbox surface grid when the
    catalogue has no mesh. The shim samples its USD mesh points the same way (<= 6000, rng 0)."""
    mesh = getattr(o.asset, "mesh", None)
    if mesh is not None and len(mesh[0]):
        V = np.asarray(mesh[0], float)
    else:
        mn, mx = np.asarray(o.asset.bbox_min, float), np.asarray(o.asset.bbox_max, float)
        g = np.linspace(0.0, 1.0, 7)
        pts = []
        for ax in range(3):
            for side in (mn[ax], mx[ax]):
                o1, o2 = [a for a in range(3) if a != ax]
                for u in g:
                    for v in g:
                        p = np.empty(3)
                        p[ax] = side
                        p[o1] = mn[o1] + u * (mx[o1] - mn[o1])
                        p[o2] = mn[o2] + v * (mx[o2] - mn[o2])
                        pts.append(p)
        V = np.asarray(pts)
    if len(V) > max_pts:
        V = V[np.random.default_rng(0).choice(len(V), max_pts, replace=False)]
    return V @ np.asarray(o.R, float).T + np.asarray(o.pos, float)


def is_gt_provider(provider) -> bool:
    """G3 mode gt: the plain ``GTProvider`` (no mode-B wrapper, no runtime belief)."""
    from dojo.skills.skills import GTProvider
    return type(provider) is GTProvider


# ------------------------------------------------------------------ the adapter
class HarnessRobot:
    """``robot`` for a vendored skill: ``robot.<verb>(**kw) -> reply dict`` (robot-harness's ``LocalRobot``)."""

    def __init__(self, sk, reserve: int = 20, truth: bool | None = None, root: str = PORTED_ROOT, log=None):
        self.sk = sk
        self.ex = sk.ex
        self.truth = is_gt_provider(sk.provider) if truth is None else bool(truth)
        if self.truth and not is_gt_provider(sk.provider):
            raise TruthDenied("truth=True needs G3 mode gt (a plain GTProvider)")
        self.env = sk.env if self.truth else TruthGuard(sk.env)
        self.kin = X5Kin(tcp_offset=GRASP_OFF)
        self.reserve = int(reserve)
        self.root = root
        self.log = log or getattr(sk, "log", None) or (lambda *a, **k: None)
        self.skill_depth = 0
        self.calls: list[dict] = []            # (cmd, steps, short reply) per call, for the episode record
        self.inst: dict[str, str] = {}         # belief label -> instance name (fixed once given)
        self.label_of: dict[str, str] = {}
        self._prev_finger = {a: self.finger_q(a) for a in ARMS}
        w = self._world()
        self.home_q = {a: np.asarray(w.home_q[a], float).copy() for a in ARMS}
        self.home_flange = {a: self._fk_pose7(a, self.home_q[a]) for a in ARMS}

    # ---------------------------------------------------------- state
    def _world(self):
        return self.sk.refresh()

    def _proprio(self) -> dict:
        return self.env.proprio()

    def q(self, arm) -> np.ndarray:
        return np.asarray(self._proprio()[f"{arm}_arm_joint_state"], float).reshape(-1)[:6]

    def flange(self, arm) -> np.ndarray:
        return np.asarray(self._proprio()[f"{arm}_ee_pose"], float).reshape(-1)[:7]

    def _fk_pose7(self, arm, q) -> np.ndarray:
        T = self.kin.flange(arm, q)
        return np.r_[T[:3, 3], mat_q(T[:3, :3])]

    def holding(self, arm) -> bool:
        return self.ex.g[arm] < 0.5

    def steps(self) -> int:
        return int(self.ex.steps_used)

    def step_lim(self) -> int:
        return int(self.env.step_lim)

    def ended(self) -> bool:
        return bool(self.ex.ended)

    def facts(self) -> dict:
        return {"steps_used": self.steps(), "step_lim": self.step_lim(), "steps_left": self.step_lim() - self.steps(),
                "ended": self.ended()}

    def finger_q(self, arm) -> np.ndarray:
        """Finger joints (m per finger). G3 mode gt: measured (dev-only). Otherwise from RoboDojo's official gripper
        reading (the applied control: the measured fingers at most 0.2 of the range past the command), else from the
        belief's own attachment (an object it believes in this arm's jaws: half its narrow side)."""
        g = float(self.ex.g[arm])
        free = GRIP_SCALE[0] + g * (GRIP_SCALE[1] - GRIP_SCALE[0])
        if self.truth:
            fj = getattr(self.env, "finger_joints", None)
            if fj is not None:
                v = np.asarray(fj()[arm], float).reshape(-1)
                return v[:2] if len(v) >= 2 else np.repeat(v[:1], 2)
            mg = getattr(self.env, "measured_gripper", None)
            if mg is not None:
                m = mg().get(arm)
                if m is not None:
                    f = GRIP_SCALE[0] + float(m) * (GRIP_SCALE[1] - GRIP_SCALE[0])
                    return np.array([f, f])
            return np.array([free, free])
        st = self._proprio()
        off = st.get(f"{arm}_ee_joint_state_official")
        if off is not None:
            v = float(np.asarray(off, float).reshape(-1)[0])
            if v > g + 0.001:                                  # the jaws stopped short of the command: an object
                f = GRIP_SCALE[0] + min(1.0, v + 0.2) * (GRIP_SCALE[1] - GRIP_SCALE[0])
                return np.array([f, f])
            return np.array([free, free])
        bel = getattr(getattr(self.sk, "provider", None), "belief", None)
        att = (getattr(bel, "attached", None) or {}).get(arm) if bel is not None else None
        if att is not None and g < 0.5:
            o = (getattr(bel, "objects", None) or {}).get(att[0])
            half = 0.5 * float(min(o.asset.extents[:2])) if o is not None else 0.01
            f = float(np.clip(half, EMPTY_Q + 0.001, FINGER_MAX))
            return np.array([f, f])
        return np.array([max(free, GRIP_SCALE[0]), max(free, GRIP_SCALE[0])])

    def eef(self) -> dict:
        st = self._proprio()
        out = {}
        for arm in ARMS:
            fl = np.asarray(st[f"{arm}_ee_pose"], float).reshape(-1)[:7]
            fq = self.finger_q(arm)
            hf = self.home_flange[arm]
            dpos, ddeg = float(np.abs(fl[:3] - hf[:3]).max()), qangle_deg(fl[3:7], hf[3:7])
            g = float(self.ex.g[arm])
            obs_g = st.get(f"{arm}_ee_joint_state_official", st.get(f"{arm}_ee_joint_state", [g]))
            out[arm] = {"pos": _l(tcp_of(fl)), "quat": _l(fl[3:7]), "flange": {"pos": _l(fl[:3]), "quat": _l(fl[3:7])},
                        "gripper": round(g, 4), "grip_cmd": round(g, 4),
                        "grip_obs": round(float(np.asarray(obs_g, float).reshape(-1)[0]), 4),
                        "finger_q": _l(fq), "gap_m": round(float(fq.sum()), 5),
                        "q": _l(np.asarray(st[f"{arm}_arm_joint_state"], float).reshape(-1)[:6]),
                        "home_dpos_max_m": round(dpos, 4), "home_ddeg": round(ddeg, 2),
                        "at_home": bool(dpos <= HOME_TOL_M and ddeg <= HOME_TOL_DEG)}
        return out

    # ---------------------------------------------------------- objects (belief or G3 truth; never checker labels)
    def _names(self, world) -> None:
        """Instance names for objects not named yet, in a pose order (category, index, x, y): never the belief's labels,
        whose text may carry a config's names."""
        used = set(self.inst.values())
        new = [o for lab, o in world.objects.items() if lab not in self.inst and o.obj_type not in NO_OBJECT_TYPES]
        new.sort(key=lambda o: (o.category, int(o.idx) if getattr(o, "identity_known", True) else -1,
                                round(float(o.pos[0]), 4), round(float(o.pos[1]), 4)))
        for o in new:
            stem = f"{o.category}_{int(o.idx)}" if getattr(o, "identity_known", True) else f"{o.category}"
            n = 0
            while f"{stem}_{n}" in used:
                n += 1
            name = f"{stem}_{n}"
            self.inst[o.label] = name
            self.label_of[name] = o.label
            used.add(name)

    def objects(self, world=None, grasp_for=()) -> dict:
        world = world or self._world()
        self._names(world)
        out = {}
        for lab, name in self.inst.items():
            o = world.objects.get(lab)
            if o is None:
                continue
            rec = {"label": None, "category": o.category, "pos": _l(o.pos, 4), "quat": _l(o.quat, 4)}
            if name in grasp_for or "*" in grasp_for:
                try:
                    P = object_cloud(o)
                    if len(P):
                        rec["aabb"] = [_l(P.min(0), 4), _l(P.max(0), 4)]
                        rec["grasps"] = cloud_grasp(P)
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {e}"
            out[name] = rec
        return out

    def resolve_args(self, args: dict | None) -> dict:
        """``"@<belief label>"`` -> that object's instance name (routes name objects by the belief's own handles)."""
        self._names(self._world())

        def tr(v):
            if isinstance(v, str) and v.startswith("@"):
                if v[1:] not in self.inst:
                    raise CmdError(f"no object {v[1:]!r} in the world", f"known: {sorted(self.inst)}")
                return self.inst[v[1:]]
            if isinstance(v, (list, tuple)):
                return type(v)(tr(x) for x in v)
            return v
        return {k: tr(v) for k, v in (args or {}).items()}

    # ---------------------------------------------------------- stepping
    def _run(self, lists: dict, name: str) -> int:
        """Per-step (q, g) lists through the executor; the episode ending stops it (the shim's loop checks ``ended``)."""
        n0 = self.steps()
        if self.ended() or not lists:
            return 0
        try:
            self.ex.run_lists(lists, name=name, use_reserve=True)
        except (EpisodeEnded, BudgetExceeded):
            pass
        return self.steps() - n0

    # ---------------------------------------------------------- kinematics
    def ik(self, arm, flange, q0):
        fl = np.asarray(flange, float).reshape(-1)
        T = np.eye(4)
        T[:3, :3] = qmat(fl[3:7])
        T[:3, 3] = tcp_of(fl)
        r = self.kin.ik(arm, T, np.asarray(q0, float))
        return np.asarray(r.q, float) if r.ok and r.q is not None else None

    def check_line(self, arm, path, start_flange, goal_flange, samples=12):
        idx = np.unique(np.linspace(0, len(path) - 1, min(samples, len(path))).round().astype(int))
        a, b = tcp_of(start_flange), tcp_of(goal_flange)
        ab = b - a
        dev, ztcp, zfl = 0.0, 9.0, 9.0
        for i in idx:
            T = self.kin.flange(arm, path[i])
            fl = T[:3, 3]
            tcp = fl + GRASP_OFF * T[:3, 0]
            tt = float(np.clip((tcp - a) @ ab / max(float(ab @ ab), 1e-12), 0, 1))
            dev = max(dev, float(np.linalg.norm(tcp - (a + tt * ab))))
            ztcp, zfl = min(ztcp, float(tcp[2])), min(zfl, float(fl[2]))
        return {"line_dev_m": round(dev, 4), "min_tcp_z": round(ztcp, 4), "min_flange_z": round(zfl, 4)}

    def cart_chain(self, arm, start_flange, goal_flange, q0, step_m=0.02, step_deg=10.0):
        a, b = np.asarray(start_flange, float), np.asarray(goal_flange, float)
        pa, pb = tcp_of(a), tcp_of(b)
        n = max(1, int(math.ceil(max(np.linalg.norm(pb - pa) / step_m, qangle_deg(a[3:], b[3:]) / step_deg))))
        qs, qprev = [np.asarray(q0, float)], np.asarray(q0, float)
        for i in range(1, n + 1):
            t = i / n
            qu = slerp(a[3:], b[3:], t)
            tcp = pa + t * (pb - pa)
            fl = np.concatenate([tcp - GRASP_OFF * qmat(qu)[:, 0], qu])
            qi = self.ik(arm, fl, qprev)
            if qi is None or float(np.abs(qi - qprev).max()) > 0.6:
                return None
            qs.append(qi)
            qprev = qi
        return np.stack(qs)

    def plan_path(self, arm, flange, q0):
        """Stand-in for cuRobo's ``plan_path``: our table-only free-space move (a joint move to the IK goal, else up,
        across and down)."""
        from dojo.skills.motion import Collision, Motion
        fl = np.asarray(flange, float)
        T = np.eye(4)
        T[:3, :3] = qmat(fl[3:7])
        T[:3, 3] = tcp_of(fl)
        mv = Motion(self.kin, Collision(self.kin, None)).to_pose(arm, np.asarray(q0, float), T,
                                                               gripper=float(self.ex.g[arm]))
        if not mv.ok or not mv.segments:
            return None
        qs = [np.asarray(q0, float)]
        for s in mv.segments:
            for q in np.asarray(s.q, float):
                if float(np.abs(q - qs[-1]).max()) > 1e-9:
                    qs.append(q)
        return np.stack(qs) if len(qs) > 1 else None

    def plan_arm(self, arm, cmd, ramp):
        q0 = self.q(arm)                                   # plan from measured joints
        start = self.flange(arm)
        info = {"arm": arm}
        if cmd.get("q") is not None:
            goal_q = np.asarray(cmd["q"], float).reshape(-1)
            path, mode = np.stack([q0, goal_q]), "joint_q"
            goal_flange = None
        else:
            goal_flange = np.asarray(cmd["flange"], float).reshape(-1)
            want = cmd.get("path", "auto")
            tcp_goal = tcp_of(goal_flange)
            reach = float(np.linalg.norm(goal_flange[:3] - MOUNTS[arm]))
            info["flange_reach_m"] = round(reach, 3)
            if tcp_goal[2] < TABLE_Z - 0.002:
                info["plan_fail_reason"] = "table_collision"
                return None, info
            if reach > REACH_MAX:
                info["plan_fail_reason"] = "workspace_limit"
                return None, info
            path, mode = None, None
            if want in ("auto", "joint"):
                qg = self.ik(arm, goal_flange, q0)
                if qg is None:
                    info["plan_fail_reason"] = "ik_unreachable"
                else:
                    line = np.stack([q0, qg])
                    chk = self.check_line(arm, line, start, goal_flange)
                    info["line_check"] = chk
                    max_dev = float(cmd.get("max_dev", 0.05))
                    ok = (chk["min_tcp_z"] >= TABLE_Z + 0.003 and chk["min_flange_z"] >= TABLE_Z + 0.03
                          and chk["line_dev_m"] <= max_dev)
                    if ok or want == "joint":
                        path, mode = line, "joint_line"
            if path is None and want in ("auto", "line"):
                ch = self.cart_chain(arm, start, goal_flange, q0)
                if ch is not None:
                    path, mode = ch, "cart_line"
                elif want == "line":
                    info["plan_fail_reason"] = info.get("plan_fail_reason") or "ik_unreachable"
            if path is None and want in ("auto", "plan"):
                pp = self.plan_path(arm, goal_flange, q0)
                if pp is not None:
                    path, mode = pp, "curobo_plan"
                    info.pop("plan_fail_reason", None)
                else:
                    info["plan_fail_reason"] = info.get("plan_fail_reason") or "plan_failed"
            if path is None:
                return None, info
        info["path_mode"] = mode
        info.pop("plan_fail_reason", None)
        lim = LIM_HOLD if self.holding(arm) else LIM_EMPTY
        sp = cmd.get("speed", "normal")
        if sp == "slow":
            v = min(LIM_HOLD, 0.6 * lim) if not self.holding(arm) else 0.6 * lim
        elif isinstance(sp, (int, float)) and not isinstance(sp, bool):
            v = float(np.clip(sp, 0.05, 1.0)) * lim
        else:
            v = lim
        info["rad_per_step"] = round(v, 4)
        info["goal_flange"] = None if goal_flange is None else _l(goal_flange)
        info["goal_q"] = _l(path[-1])
        arclen = polyline_arclen(path)
        info["joint_travel_rad"] = round(float(arclen[-1]), 4)
        n, _ = trapezoid(float(arclen[-1]), v, ramp)
        info["min_steps"] = n
        return (path, arclen, v), info

    def op_move(self, a) -> dict:
        n0 = self.steps()
        ramp = int(a.get("ramp", 2))
        plans, infos, refused = {}, {}, None
        for arm, cmd in (a.get("arms") or {}).items():
            if arm not in ARMS:
                return {"error": f"unknown arm {arm!r}"}
            p, info = self.plan_arm(arm, cmd, ramp)
            infos[arm] = info
            if p is None:
                refused = arm
            else:
                plans[arm] = p
        res = {"arms": infos, "steps": 0, **self.facts()}
        if refused is not None:                          # atomic: neither arm moves
            res.update(moved=False, refused_arm=refused, plan_fail_reason=infos[refused].get("plan_fail_reason"))
            return res
        n = max([infos[x]["min_steps"] for x in plans] or [0])
        res["planned_steps"] = n
        left = self.step_lim() - self.steps()
        reserve = int(a.get("reserve", 0))
        if a.get("max_steps") is not None and n > int(a["max_steps"]):
            res.update(moved=False, plan_fail_reason="lease_too_short", hint=f"needs {n} env steps > max_steps {a['max_steps']}")
            return res
        if n > left - reserve:
            res.update(moved=False, plan_fail_reason="budget",
                       hint=f"needs {n} env steps, {left} left, reserve {reserve} (pass reserve=0 to spend it)")
            return res
        if a.get("dry_run"):
            res.update(moved=False, dry_run=True)
            return res
        lists = {}
        for arm, (path, arclen, v) in plans.items():
            L = float(arclen[-1])
            _, s = trapezoid(L, v, ramp)
            if len(s) == 0:
                way = np.repeat(path[-1:], n, axis=0)
            else:
                if len(s) != n:                          # same n for every arm: rescale (only ever slower)
                    t = np.arange(1, n + 1, dtype=float)
                    vv = np.minimum(1.0, np.minimum(t, n + 1 - t) / (ramp + 1.0)) if ramp > 0 else np.ones(n)
                    s = np.cumsum(vv) / vv.sum() * L
                way = sample_path(path, arclen, s)
            lists[arm] = [(np.asarray(w, float), float(self.ex.g[arm])) for w in way]
        self._run(lists, "ported_move")
        for arm in plans:
            info = infos[arm]
            qm = self.q(arm)
            info["tracking_rad"] = round(float(np.abs(qm - plans[arm][0][-1]).max()), 4)
            if info.get("goal_flange") is not None:
                fl, g = self.flange(arm), np.asarray(info["goal_flange"], float)
                info["residual_m"] = round(float(np.linalg.norm(tcp_of(fl) - tcp_of(g))), 4)
                info["residual_deg"] = round(qangle_deg(fl[3:7], g[3:7]), 2)
                info["tcp"] = _l(tcp_of(fl))
                info["flange"] = _l(fl)
        res.update(moved=True, steps=self.steps() - n0, **self.facts())
        return res

    def grip_report(self, arm) -> dict:
        fq = self.finger_q(arm)
        g = float(self.ex.g[arm])
        target = float(np.clip(GRIP_SCALE[0] + g * (GRIP_SCALE[1] - GRIP_SCALE[0]), 0.0, FINGER_MAX))
        moved = float(np.abs(fq - self._prev_finger[arm]).max())
        stopped = moved < 0.0005
        return {"grip_cmd": round(g, 4), "cmd_width_m": round(2 * target, 4), "finger_q": _l(fq),
                "gap_m": round(float(fq.sum()), 5), "fingers_stopped": bool(stopped),
                "blocked": bool(stopped and float(fq.min()) > target + 0.002),
                "grasp_detected": bool(g < 0.5 and float(fq.min()) > EMPTY_Q)}

    def _grip_step(self, goals: dict | None = None) -> bool:
        """One env step with the arms still (gripper goals applied); False when no step ran."""
        if self.ended():
            return False
        self._prev_finger = {a: self.finger_q(a) for a in ARMS}
        lists = {}
        for arm in ARMS:
            g = float(self.ex.g[arm])
            if goals and arm in goals:
                g = float(np.clip(goals[arm], g - GRIP_STEP, g + GRIP_STEP))
            lists[arm] = [(self.ex.q[arm].copy(), g)]
        return self._run(lists, "ported_grip") > 0

    def op_grip(self, a) -> dict:
        n0 = self.steps()
        goals = {arm: float(np.clip(g, 0.0, 1.0)) for arm, g in (a.get("arms") or {}).items() if arm in ARMS}
        for _ in range(8):
            todo = {arm: g for arm, g in goals.items() if abs(g - self.ex.g[arm]) > 1e-6}
            if not todo or self.ended():
                break
            if not self._grip_step(todo):
                break
        for _ in range(int(a.get("wait_steps", 0))):
            if self.ended() or self.steps() >= self.step_lim():
                break
            if not self._grip_step():
                break
        return {"arms": {arm: self.grip_report(arm) for arm in goals}, "steps": self.steps() - n0, **self.facts()}

    def op_hold(self, a) -> dict:
        n0 = self.steps()
        for _ in range(int(a.get("n", 1))):
            if self.ended() or not self._grip_step():
                break
        return {"steps": self.steps() - n0, **self.facts()}

    def op_home(self, a) -> dict:
        arms = [x for x in (a.get("arms") or ARMS) if x in ARMS]
        n0 = self.steps()
        g = None
        if a.get("open", True) and any(self.ex.g[x] < 0.999 for x in arms):
            g = self.op_grip({"arms": {x: 1.0 for x in arms}, "wait_steps": int(a.get("open_wait", 0))})
        m = self.op_move({"arms": {x: {"q": self.home_q[x].tolist()} for x in arms}, "ramp": int(a.get("ramp", 2))})
        rep = {"grip": g, "move": m, "steps": self.steps() - n0, "home": {}}
        for x in ARMS:
            fl, hf = self.flange(x), self.home_flange[x]
            rep["home"][x] = {"dpos_max_m": round(float(np.abs(fl[:3] - hf[:3]).max()), 4),
                              "ddeg": round(qangle_deg(fl[3:7], hf[3:7]), 2),
                              "gap_m": round(float(self.finger_q(x).sum()), 5)}
        rep.update(self.facts())
        return rep

    def op_reach(self, a) -> dict:
        arm = a["arm"]
        q0 = self.q(arm)
        lim = LIM_HOLD if self.holding(arm) else LIM_EMPTY
        out = []
        for fl in a["flanges"]:
            fl = np.asarray(fl, float)
            reach = float(np.linalg.norm(fl[:3] - MOUNTS[arm]))
            if reach > REACH_MAX or tcp_of(fl)[2] < TABLE_Z - 0.002:
                out.append({"ok": False, "why": "workspace_limit" if reach > REACH_MAX else "table_collision"})
                continue
            fq = np.asarray(a.get("from_q") if a.get("from_q") is not None else q0, float)
            qg = self.ik(arm, fl, fq)
            if qg is None:
                out.append({"ok": False, "why": "ik_unreachable"})
            else:
                d = float(np.abs(qg - fq).max())
                out.append({"ok": True, "q": _l(qg), "joint_travel_rad": round(d, 4),
                            "steps_est": int(math.ceil(d / lim)) + 2})
        return {"results": out, **self.facts()}

    # ---------------------------------------------------------- server commands (robodojo_server.Server.cmd_*)
    def target_quat(self, a, cur_quat):
        if a.get("quat") is not None:
            q = np.asarray(a["quat"], float)
            return q / np.linalg.norm(q), "quat"
        if a.get("yaw_deg") is not None or a.get("down"):
            return tool_quat(float(a.get("yaw_deg") or 0.0), float(a.get("pitch_deg") or 0.0),
                             float(a.get("roll_deg") or 0.0)), "top_down"
        return np.asarray(cur_quat, float), "keep"

    @staticmethod
    def arm_arg(a):
        arm = a.get("arm")
        if arm not in ARMS:
            raise CmdError(f"arm must be 'left' or 'right', got {arm!r}", "two ARX X5 arms")
        return arm

    @staticmethod
    def motion_reply(res, arm, tol, rot_tol):
        info = (res.get("arms") or {}).get(arm, {})
        out = {"moved": bool(res.get("moved")), "steps": res.get("steps", 0), "planned_steps": res.get("planned_steps"),
               "plan_fail_reason": res.get("plan_fail_reason") or info.get("plan_fail_reason"),
               "path_mode": info.get("path_mode"), "rad_per_step": info.get("rad_per_step"),
               "residual_m": info.get("residual_m"), "residual_deg": info.get("residual_deg"),
               "tracking_rad": info.get("tracking_rad"), "line_check": info.get("line_check"),
               "tcp": info.get("tcp"), "hint": res.get("hint")}
        out["reached"] = bool(out["moved"] and out["residual_m"] is not None and out["residual_m"] <= tol
                              and (out["residual_deg"] or 0) <= rot_tol)
        if not out["moved"] and not out["plan_fail_reason"]:
            out["plan_fail_reason"] = "refused"
        return out

    def cmd_obs(self, a) -> dict:
        world = self._world()
        out = {"images": {}, "eef": self.eef(), "instruction": str(world.instruction or "")}
        if a.get("objects", True):
            out["objects"] = self.objects(world, grasp_for=list(a.get("grasps") or []))
        out.update(self.facts())
        return out

    def cmd_move_to(self, a) -> dict:
        arm = self.arm_arg(a)
        if a.get("pos") is None:
            raise CmdError("move_to needs pos=[x,y,z] (TCP, world m)", "optional: quat | yaw_deg [pitch_deg roll_deg]")
        cur = self.flange(arm)
        q, qsrc = self.target_quat(a, cur[3:7])
        fl = flange_from_tcp(a["pos"], q)
        return self._move(arm, fl, a, qsrc)

    def _move(self, arm, fl, a, qsrc) -> dict:
        cmd = {"flange": fl.tolist(), "path": a.get("path", "auto"), "speed": a.get("speed", "normal")}
        if a.get("max_dev") is not None:
            cmd["max_dev"] = float(a["max_dev"])
        res = self.op_move({"arms": {arm: cmd}, "ramp": int(a.get("ramp", 2)), "max_steps": a.get("max_steps"),
                            "reserve": int(a.get("reserve", self.reserve)), "dry_run": bool(a.get("dry_run"))})
        out = self.motion_reply(res, arm, float(a.get("tol", 0.01)), float(a.get("rot_tol_deg", 5.0)))
        out.update(arm=arm, orientation=qsrc, target_tcp=[round(float(v), 4) for v in a.get("pos", fl[:3])])
        return out

    def cmd_move_delta(self, a) -> dict:
        arm = self.arm_arg(a)
        fl0 = self.flange(arm)
        dpos = np.asarray(a.get("dpos") or [0, 0, 0], float)
        q = np.asarray(fl0[3:7], float)
        if a.get("drot_deg") is not None:
            q = qmul(rotvec_q(np.radians(np.asarray(a["drot_deg"], float))), q)
        elif a.get("dyaw_deg") is not None:
            q = qmul(axis_q(2, math.radians(float(a["dyaw_deg"]))), q)
        tgt = tcp_of(fl0) + dpos
        fl = flange_from_tcp(tgt, q)
        return self._move(arm, fl, dict(a, pos=tgt.tolist()), "delta")

    def cmd_gripper(self, a) -> dict:
        arm = self.arm_arg(a)
        act = a.get("action", a.get("value"))
        if act in ("open", None):
            g = 1.0
        elif act == "close":
            g = 0.0
        else:
            try:
                g = float(act)
            except (TypeError, ValueError):
                raise CmdError(f"gripper action {act!r}", "open | close | 0..1 (1 = open)") from None
        wait = int(a.get("wait_steps", 3 if g < 0.5 else 0))
        res = self.op_grip({"arms": {arm: g}, "wait_steps": wait})
        r = dict((res.get("arms") or {}).get(arm, {}))
        r.update(arm=arm, steps=res.get("steps", 0))
        return r

    def cmd_hold(self, a) -> dict:
        res = self.op_hold({"n": int(a.get("steps", 1))})
        return {"steps": res.get("steps", 0)}

    def cmd_return_home(self, a) -> dict:
        arms = a.get("arms") or ["left", "right"]
        res = self.op_home({"arms": arms, "open": bool(a.get("open", True)), "ramp": int(a.get("ramp", 2))})
        within = {x: bool(h["dpos_max_m"] <= HOME_TOL_M and h["ddeg"] <= HOME_TOL_DEG) for x, h in res["home"].items()}
        mv = res.get("move") or {}
        return {"home": res["home"], "within_tolerance": within, "all_home": all(within.values()), "steps": res.get("steps", 0),
                "move_steps": mv.get("steps"), "grip_steps": (res.get("grip") or {}).get("steps", 0),
                "plan_fail_reason": mv.get("plan_fail_reason"), "tolerance": {"m_per_axis": HOME_TOL_M, "deg": HOME_TOL_DEG}}

    def cmd_reach(self, a) -> dict:
        arm = self.arm_arg(a)
        cur = self.flange(arm)
        tg = a.get("targets") or [a]
        fls = []
        for t in tg:
            q, _ = self.target_quat(t, cur[3:7])
            fls.append(flange_from_tcp(t["pos"], q).tolist())
        res = self.op_reach({"arm": arm, "flanges": fls, "from_q": a.get("from_q")})
        out = []
        for r, fl in zip(res["results"], fls):
            r = dict(r)
            r["quat"] = [round(v, 5) for v in fl[3:7]]
            if not a.get("with_q"):
                r.pop("q", None)
            out.append(r)
        return {"arm": arm, "results": out}

    def cmd_held_look(self, a) -> dict:
        """The strict server's wrist look, answered from the finger state above (G3: measured fingers; otherwise the
        official gripper reading or the belief's attachment). PR2's strict-skill paths call it."""
        arm = self.arm_arg(a)
        fq = self.finger_q(arm)
        held = bool(self.ex.g[arm] < 0.5 and float(fq.min()) > EMPTY_Q)
        return {"arm": arm, "stage": a.get("stage"), "held": held, "answer": "yes" if held else "no",
                "source": "measured fingers (G3 mode gt)" if self.truth else "official gripper reading / belief"}

    def cmd_skill(self, a) -> dict:
        name = a.get("name") or ""
        if not name or "/" in name or name.startswith("."):
            raise CmdError(f"bad skill name {name!r}", f"skills in {self.root}")
        d = os.path.join(self.root, name)
        path = os.path.join(d, "skill_robodojo.py")
        if not os.path.isfile(path):
            path = os.path.join(d, "skill.py")
        if not os.path.isfile(path):
            raise CmdError(f"no skill {name!r}", f"see {self.root}")
        if self.skill_depth >= 4:
            raise CmdError("skill nesting too deep", "max depth 4")
        spec = importlib.util.spec_from_file_location(f"ported_skill_{name}_{time.time_ns()}", path)
        mod = importlib.util.module_from_spec(spec)
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        spec.loader.exec_module(mod)
        s0 = self.steps()
        self.skill_depth += 1
        try:
            rep = mod.run(self, **(a.get("args") or {}))
        except TypeError as e:
            raise CmdError(f"skill {name} bad arguments: {e}", f"see {d}") from None
        except (CmdError, TruthDenied):
            raise
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ in PASS_THROUGH:
                raise
            raise CmdError(f"skill {name} raised {type(e).__name__}: {e}", traceback.format_exc(limit=4)[-800:]) from None
        finally:
            self.skill_depth -= 1
        rep = dict(rep or {})
        rep.update(skill=name, skill_steps_used=self.steps() - s0)
        return rep

    # ---------------------------------------------------------- dispatch (robodojo_server.LocalRobot.call)
    def call(self, cmd, **kw) -> dict:
        kw = dict(kw)
        if "cmd" in kw:
            kw["action"] = kw.pop("cmd")
        t0 = time.time()
        n0 = self.steps()
        try:
            if cmd not in COMMANDS:
                raise CmdError(f"unknown command {cmd!r}", f"commands: {list(COMMANDS)}")
            rep = {"ok": True, **getattr(self, "cmd_" + cmd)(kw)}
        except TruthDenied:
            raise                                         # a legality breach is never a skill-level error
        except CmdError as e:
            rep = {"ok": False, "error": str(e), "hint": e.hint}
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ in PASS_THROUGH:
                raise
            rep = {"ok": False, "error": f"{type(e).__name__}: {e}", "hint": traceback.format_exc(limit=3)[-600:]}
        f = self.facts()
        rep.update(cmd=cmd, t_wall=round(time.time() - t0, 4), step=f["steps_used"], steps_left=f["steps_left"],
                   ended=f["ended"])
        rep = compact(rep)
        if cmd != "obs" or len(self.calls) < 4000:
            self.calls.append({"cmd": cmd, "step": n0, "steps": self.steps() - n0, "ok": rep.get("ok"),
                               **{k: rep.get(k) for k in ("moved", "reached", "plan_fail_reason", "path_mode",
                                                          "verified", "error") if rep.get(k) is not None}})
        return rep

    def __getattr__(self, cmd):
        if cmd.startswith("_"):
            raise AttributeError(cmd)
        return lambda **kw: self.call(cmd, **kw)

    def run_skill(self, name: str, args: dict | None = None) -> dict:
        """The top-level skill call (robot-harness ``skill name=<dir> args={...}``)."""
        return self.call("skill", name=name, args=self.resolve_args(args))


__all__ = ["HarnessRobot", "TruthGuard", "TruthDenied", "PORTED_ROOT", "SOURCE_COMMIT", "cloud_grasp", "object_cloud",
           "tool_quat", "flange_from_tcp", "tcp_of", "trapezoid", "is_gt_provider"]
