"""G3 (dev only): the in-hand pose as mode B has it, and perception's ``held_pose`` / ``show_pose`` answered from GT.

G3 runs physics-only (no rendering), so perception's render-and-compare ``held_pose`` (``dojo.perception.held``) cannot
run there. ``HeldBeliefProvider`` wraps the G3 world provider (``g3.OffsetBeliefProvider`` / ``GTProvider``) so the
skills' mode-B held-pose protocol (``Skills._held_protocol``) runs in G3 with the same control flow and step costs:

- **belief**: like the runtime's ``KinematicBelief``, an object is attached to an arm when its gripper closes around it
  (the commanded TCP at the closing step, the object's BELIEVED pose then: ``OffsetBeliefProvider``'s offset included),
  and while attached it is reported at ``T_tcp(measured joints) @ T_attached``: the in-hand transform predicted at
  grasp time, not the true one. Opening the gripper releases it (the base provider again);
- ``held_pose(arm, prior_T_tcp_obj, asset_key, cfg_)``: the TRUE in-hand transform (GT) moved by an error resampled from
  perception-held's dev evaluation for the object's category and the view (``cfg_["view"]``: hover / look / show; the
  table ``data/held_noise.json``, ``scripts/skills/held_noise_table.py``), with that read's ok / why / sigma; an ok
  read re-anchors the attachment (as the runtime's ``_held_pose`` does). ``mode="gt"``: the exact transform;
- ``show_pose(arm, T_tcp_obj, asset_key, cfg_)``: generic candidates in perception's ``show_pose.propose`` format:
  observer looks of the other arm's wrist camera at the held object from beside / across the pads, then show poses of
  the holding arm with the object in front of the head camera (+ an observer look there); reachable by our IK, in a
  fixed preference order (perception ranks by the predicted information of the silhouettes, which needs rendering).

Camera geometry from perception-held's ``dojo/perception/show_pose.py`` (PR #59: the wrist camera in the TCP frame, the
head camera, the look distance), copied so G3 does not need the perception stack.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from dojo.skills.geometry import T_to_pose7, inv_T
from dojo.skills.kinematics import ARMS, TABLE_Z, X5Kin

NOISE_FILE = Path(__file__).with_name("data") / "held_noise.json"
VIEWS = ("hover", "look", "show")
ATTACH_MARGIN = 0.015          # m: the runtime belief's attach margin (``KinematicBelief``)

# --- from dojo/perception/show_pose.py (perception-held, PR #59)
T_TCP_CAM = np.array([[-0.000007, -0.500030, 0.866008, -0.060158],
                      [-1.000000, 0.000004, -0.000006, 0.000000],
                      [0.000000, -0.866008, -0.500030, 0.050940],
                      [0.0, 0.0, 0.0, 1.0]])
HEAD_EYE = np.array([0.0, -0.41, 1.308])
HEAD_AXIS = np.array([0.0, 0.5, -0.8660254])
SHOW_HEAD_DIST = 0.42
SHOW_SIDE = 0.06
LOOK_DIST = (0.14, 0.24)


def look_at_camera(eye, target, roll_deg: float = 0.0) -> np.ndarray:
    """Camera-to-world (OpenCV axes) at ``eye`` looking at ``target``, image up = world up (then rolled)."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    y = -(up - z * float(up @ z))
    if np.linalg.norm(y) < 1e-6:
        y = np.array([0.0, 1.0, 0.0]) - z * z[1]
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    r = math.radians(roll_deg)
    x, y = math.cos(r) * x + math.sin(r) * y, -math.sin(r) * x + math.cos(r) * y
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = eye
    return T


def look_distance(extent_m: float) -> float:
    return float(np.clip(2.6 * float(extent_m), *LOOK_DIST))


def observer_dirs(T_tcp) -> list[tuple[str, np.ndarray]]:
    """Viewing directions (object -> eye) in the TCP frame: beside the pads (+-y), across them (+-z), tilted toward the
    protruding side (+x), and end-on; the holding gripper stays out of the line of sight."""
    R = np.asarray(T_tcp, float)[:3, :3]
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    out = []
    for t in (20.0, 0.0, 45.0, -20.0):
        for name, s in (("side+y", y), ("side-y", -y), ("across+z", z), ("across-z", -z)):
            r = math.radians(t)
            out.append((f"{name}@{t:.0f}", math.cos(r) * s + math.sin(r) * x))
    out.append(("end_on", x))
    return out


def show_point(arm: str) -> np.ndarray:
    side = -SHOW_SIDE if arm == "left" else SHOW_SIDE
    return HEAD_EYE + SHOW_HEAD_DIST * HEAD_AXIS + np.array([side, 0.0, 0.0])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotvec_R(v) -> np.ndarray:
    v = np.asarray(v, float)
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def lead_point(o, T_obj_target=None) -> np.ndarray:
    """The leading point (object frame): the mean of the mesh points within 1 mm of the lowest at the place target (the
    bbox corners without a mesh); without a target, at the object's current pose."""
    V = np.asarray(o.asset.mesh[0], float) if o.asset.mesh is not None else np.array(
        [[x, y, z] for x in (o.asset.bbox_min[0], o.asset.bbox_max[0]) for y in (o.asset.bbox_min[1], o.asset.bbox_max[1])
         for z in (o.asset.bbox_min[2], o.asset.bbox_max[2])], float)
    T = np.asarray(T_obj_target, float) if T_obj_target is not None else o.T
    z = V @ T[:3, :3].T[:, 2] + T[2, 3]
    return V[z <= z.min() + 0.001].mean(0)


class HeldBeliefProvider:
    """See the module doc. ``mode``: None (belief only: no ``held_pose`` / ``show_pose``), "noise" or "gt"."""

    def __init__(self, base, env, kin: X5Kin | None = None, mode: str | None = "noise", seed: int = 0,
                 table: dict | None = None):
        self.base = base
        self.env = env
        self.kin = kin or X5Kin()
        self.mode = mode
        self.rng = np.random.default_rng(int(seed))
        self.att: dict = {}                 # arm -> (label, T_tcp_obj as the belief has it)
        self._g = {a: 1.0 for a in ARMS}
        self._free: dict = {}               # label -> ObjectState as last reported while not attached
        self.stats: dict = {"reads": {v: 0 for v in VIEWS}, "ok": {v: 0 for v in VIEWS}, "show_pose": 0,
                            "attached": 0, "reanchored": 0}
        self.events: list = []
        if mode:
            if table is None:
                table = json.loads(NOISE_FILE.read_text())["table"] if mode == "noise" else {}
            self.table = table
            self.held_pose = self._held_pose        # instance attributes: absent (getattr -> base) when mode is None
            self.show_pose = self._show_pose
        self._step = env.step
        env.step = self._on_step

    def __getattr__(self, name):
        return getattr(self.base, name)

    # ------------------------------------------------------------------ belief
    def _on_step(self, action: dict):
        for a in ARMS:
            g_new = float(np.clip(np.asarray(action[f"{a}_ee_joint_state"]).reshape(-1)[0], 0, 1))
            if g_new < 0.5 and self._g[a] > g_new + 1e-6 and a not in self.att:
                T_c = self.kin.tcp(a, np.asarray(action[f"{a}_arm_joint_state"], float).reshape(-1)[:6])
                lab = self._object_at(T_c[:3, 3])
                if lab is not None:
                    self.att[a] = (lab, inv_T(T_c) @ self._free[lab].T)
                    self.stats["attached"] += 1
            if g_new > 0.6 and a in self.att:
                self.att.pop(a)
            self._g[a] = g_new
        return self._step(action)

    def _object_at(self, p) -> str | None:
        held = {v[0] for v in self.att.values()}
        best, bd = None, 1e9
        for lab, o in self._free.items():
            if lab in held or o.obj_type == "Geometry" or getattr(o, "clutter", False):
                continue
            loc = o.R.T @ (np.asarray(p, float) - o.center_w)
            if np.all(np.abs(loc) <= o.asset.extents / 2 + ATTACH_MARGIN):
                d = float(np.linalg.norm(loc))
                if d < bd:
                    best, bd = lab, d
        return best

    def refresh(self, holding: dict | None = None):
        w = self.base.refresh(holding)
        attached = {}
        for a, (lab, T_rel) in self.att.items():
            if lab not in w.objects:
                continue
            p7 = T_to_pose7(self.kin.tcp(a, w.arms[a].q) @ T_rel)
            o = w.objects[lab]
            o.pos, o.quat = np.asarray(p7[:3], float), np.asarray(p7[3:], float)
            o.source = "predicted"
            attached[lab] = a
        import copy
        for lab, o in w.objects.items():
            if lab not in attached:
                self._free[lab] = copy.copy(o)
        return w

    # ------------------------------------------------------------------ reads
    def _true_T_rel(self, arm: str, label: str) -> np.ndarray:
        st = self.env.proprio()
        q = np.asarray(st[f"{arm}_arm_joint_state"], float).reshape(-1)[:6]
        row = next(r for r in self.env.gt_objects() if r["label"] == label)
        from dojo.skills.geometry import T_from
        return inv_T(self.kin.tcp(arm, q)) @ T_from(np.asarray(row["pos"], float), quat=np.asarray(row["quat"], float))

    def _rows(self, cat: str, view: str) -> list:
        rows = (self.table.get(cat) or {}).get(view)
        if rows:
            return rows
        return [r for c in self.table.values() for r in (c.get(view) or [])]

    def _held_pose(self, arm: str, prior_T_tcp_obj=None, asset_key: str = "", cfg_: dict | None = None) -> dict:
        cfg = dict(cfg_ or {})
        view = cfg.get("view", "hover") if cfg.get("view", "hover") in VIEWS else "hover"
        prior = np.asarray(prior_T_tcp_obj, float) if prior_T_tcp_obj is not None else None
        if arm not in self.att:
            return {"ok": False, "why": "not_held", "T_tcp_obj": None if prior is None else prior.tolist(), "sigma": None}
        lab = self.att[arm][0]
        self.stats["reads"][view] += 1
        T_true = self._true_T_rel(arm, lab)
        o = self._free.get(lab) or self.base.refresh().obj(lab)
        if self.mode == "gt":
            row = {"ok": 1, "why": "", "err": [0.0] * 6, "sig": [0.05, 0.05, 0.05, 0.05, 0.1]}
        else:
            rows = self._rows(lab_cat(asset_key, o), view)
            row = rows[int(self.rng.integers(len(rows)))] if rows else {"ok": 0, "why": "no_table"}
        ev = {"arm": arm, "label": lab, "view": view, "ok": bool(row["ok"]), "why": row.get("why", ""),
              "step": int(self.env.steps_used)}
        if not row["ok"]:
            self.events.append(ev)
            return {"ok": False, "why": row.get("why") or "fit_poor",
                    "T_tcp_obj": (prior if prior is not None else self.att[arm][1]).tolist(), "sigma": None}
        e = np.asarray(row["err"], float)
        lead = lead_point(o, cfg.get("T_obj_target"))
        lead_t = T_true[:3, :3] @ lead + T_true[:3, 3]
        D = np.eye(4)
        D[:3, :3] = _rotvec_R(np.radians(e[3:6]))
        D[:3, 3] = lead_t + e[:3] / 1000.0 - D[:3, :3] @ lead_t
        T_meas = D @ T_true
        s = row.get("sig") or [99.0] * 5
        self.att[arm] = (lab, T_meas)                       # the runtime re-anchors the belief on an ok read
        self.stats["ok"][view] += 1
        self.stats["reanchored"] += 1
        ev.update({"err_lead_mm": [round(float(v), 3) for v in e[:3]], "sig_lat_mm": s[3]})
        self.events.append(ev)
        return {"ok": True, "why": "", "T_tcp_obj": T_meas.tolist(),
                "sigma": {"lead_mm": [float(v) for v in s[:3]], "lead_lat_mm": float(s[3]), "axis_deg": float(s[4])}}

    def _show_pose(self, arm: str, T_tcp_obj=None, asset_key: str = "", cfg_: dict | None = None) -> dict:
        """Candidates as ``dojo.perception.show_pose.propose`` returns them (best first)."""
        self.stats["show_pose"] += 1
        obs = "right" if arm == "left" else "left"
        st = self.env.proprio()
        q_h = np.asarray(st[f"{arm}_arm_joint_state"], float).reshape(-1)[:6]
        q_o = np.asarray(st[f"{obs}_arm_joint_state"], float).reshape(-1)[:6]
        T_tcp = self.kin.tcp(arm, q_h)
        T_rel = np.asarray(T_tcp_obj, float)
        o = next((x for x in self._free.values() if x.asset.key == asset_key), None)
        if o is None:
            o = next((x for x in self.base.refresh().objects.values() if x.asset.key == asset_key), None)
        if o is None:
            return {"poses": []}
        ext = float(np.max(o.asset.extents))
        poses = []

        def looks(T_t, n):
            """Up to ``n`` observer looks at the object held at ``T_t``: the first reachable roll per direction (the
            skills check collisions and the arms' clearance)."""
            c = (T_t @ T_rel)[:3, :3] @ o.asset.center + (T_t @ T_rel)[:3, 3]
            d = look_distance(ext)
            out = []
            for name, u in observer_dirs(T_t):
                if u[2] < -0.35:
                    continue
                eye = c + d * u
                if eye[2] < TABLE_Z + 0.05:
                    continue
                for roll in (0.0, 90.0, -90.0, 45.0, -45.0):
                    T_o = look_at_camera(eye, c, roll) @ inv_T(T_TCP_CAM)
                    r = self.kin.ik(obs, T_o, q_o)
                    if r.ok:
                        out.append({"name": f"{name}_r{roll:+.0f}", "arm": obs, "tcp": T_o.tolist(), "q": r.q.tolist(),
                                    "eye": eye.tolist(), "look_at": c.tolist(), "dist": d})
                        break
                if len(out) >= n:
                    break
            return out
        for L in looks(T_tcp, 4):
            poses.append({"name": "hover_look", "arm": arm, "tcp": T_tcp.tolist(), "q": q_h.tolist(), "observer": L})
        c_now = (T_tcp @ T_rel)[:3, :3] @ o.asset.center + (T_tcp @ T_rel)[:3, 3]
        P = show_point(arm)
        for yd in (0, 45, -45, 90, -90, 135, -135, 180):
            Rz = _rot_z(math.radians(yd))
            T = np.eye(4)
            T[:3, :3] = Rz @ T_tcp[:3, :3]
            T[:3, 3] = P + Rz @ (T_tcp[:3, 3] - c_now)
            r = self.kin.ik(arm, T, q_h)
            if not r.ok:
                continue
            L = looks(T, 1)
            poses.append({"name": f"show_yaw{yd:+d}", "arm": arm, "tcp": T.tolist(), "q": r.q.tolist(),
                          "observer": L[0] if L else None})
            if sum(p["name"].startswith("show") for p in poses) >= 3:
                break
        return {"poses": poses}


def lab_cat(asset_key: str, o=None) -> str:
    """The noise table's category key: the asset key's category (``Rigid/<category>/<idx>``)."""
    parts = str(asset_key).split("/")
    if len(parts) >= 2:
        return parts[1]
    return getattr(o, "category", "")
