"""DEV data for the held-object pose (``held.held_pose``): physics captures in G3, kinematic variants, scoring.

Three steps, all on OUR dev layouts (seeds >= 100000; never eval data):

1. ``capture`` (RoboDojo's python, a ``g3 snippet``; physics-only 5.1 twin): the real skills run each layout
   (``planner.solve``, GT world as in G3). At every place hover with an object in the jaws (the refine hook, kind
   "place": the pre-place / pre-insert pose, right where G3 re-measures) the state is written as
   ``frame_C<k>.json``: the official proprio, ``run.gt()`` (objects, every robot link pose), the skills' in-hand
   transform (``Held.T_tcp_obj``, measured at the lift in G3), the PLANNED one (``PickPlan.T_tcp_obj``, what mode B
   believes at grasp time), the place target and the planned costs of observer looks / show poses
   (``show_pose.plan_costs``). Plus ``stage.usd`` + ``cameras.json`` for the Isaac 6.0 proxy render.
2. ``variants`` (any python with numpy; the perception venv): kinematic re-posings of each capture, rendered later
   from the same stage (``render_d0_proxy`` writes every robot link and object pose, no physics):
   - ``V<k>g<j>``: the object turned / slid in the pads (rotation about the closing axis, a slide in the pad plane):
     small grasp variations around the physics grasp;
   - ``V<k>o<j>``: the observer arm at a planned look at the hover (``show_pose`` / the skills' IK + collision model);
   - ``V<k>s<j>``: the holding arm at a planned show pose (the object rigidly with the TCP, the fingers rigidly with
     link6) + the observer looking at it.
   The GT in-hand transform of every frame is in ``rec["held_gt"]`` (scorer only).
3. ``score`` helpers: errors in the TCP frame, at the leading point and on the insertion axis.

GT is used only to build frames and to score; ``held.held_pose`` never reads it.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np

TCP_OFFSET = 0.145          # dojo.skills.kinematics.TCP_OFFSET (the skills' TCP; T_tcp_obj is relative to it)
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


# ------------------------------------------------------------------------------------------------ small math
def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat_wxyz(R) -> list:
    R = np.asarray(R, float)
    t = np.trace(R)
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
    q = np.asarray(q)
    q = q / np.linalg.norm(q)
    return (q if q[0] >= 0 else -q).tolist()


def pose7_to_T(p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_R(p[3:7])
    T[:3, 3] = p[:3]
    return T


def T_to_pose7(T) -> list:
    T = np.asarray(T, float)
    return [float(v) for v in T[:3, 3]] + [float(v) for v in R_to_quat_wxyz(T[:3, :3])]


def inv_T(T) -> np.ndarray:
    T = np.asarray(T, float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def rotvec_R(v) -> np.ndarray:
    v = np.asarray(v, float)
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def R_rotvec(R) -> np.ndarray:
    R = np.asarray(R, float)
    c = float(np.clip((np.trace(R) - 1) / 2, -1, 1))
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    if a > math.pi - 1e-6:
        w, v = np.linalg.eigh((R + R.T) / 2)
        k = v[:, int(np.argmax(w))]
        return k * a
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(a))
    return k * a


def tcp_from_flange_T(T_flange) -> np.ndarray:
    T = np.asarray(T_flange, float).copy()
    T[:3, 3] = T[:3, 3] + T[:3, 0] * TCP_OFFSET
    return T


def catalog_key(row: dict) -> str:
    return f"{row['type']}/{row['category']}/{int(row['category_idx']):05d}"


# ------------------------------------------------------------------------------------------------ 1. capture
class _Stop(Exception):
    pass


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def capture_snippet(g: dict) -> None:
    """``g3 snippet`` entry (``scripts/perception/held_capture.sh``). Env: HP_OUT (root), HP_TASK, HP_LAYOUTS (comma
    list of layout files; the snippet's ``--layout`` is the first), HP_MAX_CAPS (captures per episode, default 3),
    HP_PLAN_COSTS (1: plan observer looks / show poses at each capture)."""
    from dojo.simlab import paths
    from dojo.skills import planner
    from dojo.skills.env import RunnerEnv
    from dojo.skills.skills import GTProvider, Skills
    from dojo.skills.world import AssetDB

    from . import show_pose

    run = g["run"]
    out = Path(os.environ["HP_OUT"]).expanduser()
    paths.assert_not_eval(out)
    task = os.environ["HP_TASK"]
    layouts = [p for p in os.environ.get("HP_LAYOUTS", "").split(",") if p]
    max_caps = int(os.environ.get("HP_MAX_CAPS", "3"))
    plan = os.environ.get("HP_PLAN_COSTS", "1") == "1"
    assets = AssetDB(paths.OBJECTS)
    for lp in layouts:
        paths.assert_not_eval(Path(lp))
        lay = json.loads(Path(lp).read_text())
        seed = int(Path(lp).name.split(".")[0])
        d = out / task / str(seed)
        if (d / "done.json").exists():
            print(f"[held_capture] {task} {seed}: done already", flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        lay.setdefault("_simlab", {})["seed"] = seed
        info = run.load(lay)
        if not info["stable"]:
            (d / "done.json").write_text(json.dumps({"seed": seed, "stable": False}))
            continue
        run.export_stage(d / "stage.usd")
        (d / "cameras.json").write_text(json.dumps(_jsonable(run.camera_spec()), indent=1))
        env = RunnerEnv(run)
        prov = GTProvider(env, assets, instruction=env.instruction)
        logs: list[str] = []
        caps: list[dict] = []
        planned: dict = {}
        cur_place: dict = {}

        def hook(sk_, label, arm, kind):
            h = sk_.held.get(arm)
            if kind != "place" or h is None:
                return None
            k = len(caps)
            tag = f"C{k}"
            st = {kk: np.asarray(v, float).reshape(-1).tolist() for kk, v in run.obs_state().items()}
            gt = run.gt()
            o = sk_.world.obj(h.label)
            row = next((r for r in gt["objects"] if r.get("label") == h.label), None)
            held = {"arm": arm, "label": h.label, "key": catalog_key(row) if row else None,
                    "T_tcp_obj_g3": np.asarray(h.T_tcp_obj).tolist(),
                    "T_tcp_obj_plan": planned.get(h.label, {}).get("T_tcp_obj"),
                    "grasp": planned.get(h.label, {}).get("grasp"),
                    "place": dict(cur_place) if cur_place.get("label") == h.label else None,
                    "ref_label": label, "q_cmd": {a: np.asarray(sk_.ex.q[a], float).tolist() for a in ("left", "right")},
                    "g_cmd": {a: float(sk_.ex.g[a]) for a in ("left", "right")}, "steps_used": int(sk_.ex.steps_used)}
            if plan:
                try:
                    bmin = np.asarray(o.asset.bbox_min, float)
                    held["plan_costs"] = show_pose.plan_costs(sk_, arm, h.label, h.T_tcp_obj, bmin,
                                                              bmin + np.asarray(o.asset.extents, float))
                except Exception as e:  # noqa: BLE001
                    held["plan_costs"] = {"error": f"{type(e).__name__}: {e}"}
            rec = {"frame": tag, "obs": {"instruction": env.instruction, "state": st,
                                         "images": {c: f"frame_{tag}_{c}.png" for c in CAMS}, "image_shape": {}},
                   "gt": gt, "held": held,
                   "meta": {"renderer": None, "physics": "isaac-5.1 physics-only", "capture": "place_hover",
                            "task": task, "seed": seed}}
            (d / f"frame_{tag}.json").write_text(json.dumps(_jsonable(rec)))
            caps.append({"tag": tag, "label": h.label, "arm": arm, "steps": int(sk_.ex.steps_used)})
            print(f"[held_capture] {task} {seed} {tag}: {h.label} in {arm} at step {sk_.ex.steps_used}", flush=True)
            if len(caps) >= max_caps:
                raise _Stop()
            return None

        sk = Skills(env, prov, reserve=30, log=lambda m: logs.append(str(m)), verifier=None, verify_k=2, n_cands=4,
                    remeasure=True, refine=hook)
        vp, rp = sk.verify_pick, sk.run_place

        def verify_pick(pp, before):
            gr = pp.grasp
            planned[pp.label] = {"T_tcp_obj": np.asarray(pp.T_tcp_obj).tolist(),
                                 "grasp": {"method": gr.method, "T": np.asarray(gr.T).tolist(), "width": float(gr.width),
                                           "note": str(gr.note)[:120]}}
            return vp(pp, before)

        def run_place(pp, *a, **kw):
            cur_place.clear()
            cur_place.update({"label": pp.label, "arm": pp.arm, "T_obj_target": np.asarray(pp.T_obj_target).tolist(),
                              "servo": bool((getattr(pp, "tail_kw", None) or {}).get("servo"))})
            return rp(pp, *a, **kw)

        sk.verify_pick, sk.run_place = verify_pick, run_place
        res = {}
        try:
            res = planner.solve(sk) or {}
        except _Stop:
            res = {"stopped": "max captures"}
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}"}
        summ = {"seed": seed, "stable": True, "captures": caps, "result": {k: str(v)[:200] for k, v in res.items()},
                "steps_used": int(sk.ex.steps_used), "wall_s": round(time.time() - t0, 1), "log": logs[-40:]}
        (d / "done.json").write_text(json.dumps(_jsonable(summ)))
        print(f"[held_capture] {task} {seed}: {len(caps)} captures in {summ['wall_s']} s", flush=True)


# ------------------------------------------------------------------------------------------------ 2. variants
def held_gt(rec: dict, model=None) -> dict:
    """GT in-hand transform of a frame record: object pose (gt) in the TCP of the measured joints (FK)."""
    from .robot_model import X5Model

    model = model or X5Model.default()
    h = rec["held"]
    arm = h["arm"]
    q = rec["obs"]["state"][f"{arm}_arm_joint_state"]
    T6 = model.fk_world(arm, q)["link6"]
    row = next(r for r in rec["gt"]["objects"] if r.get("label") == h["label"])
    T_obj = pose7_to_T(list(row["pos"]) + list(row["quat"]))
    return {"T_tcp_obj": (inv_T(tcp_from_flange_T(T6)) @ T_obj).tolist(), "T_obj": T_obj.tolist()}


def _set_arm(rec: dict, arm: str, q, model, rigid_from: dict | None = None) -> None:
    """Re-pose ``arm`` to joints ``q``: every link from FK, except links in ``rigid_from`` ({link: T_link6_link}) which
    follow link6 rigidly (the fingers at their physical opening). Updates the official state (joints, flange)."""
    q = [float(v) for v in q]
    R = rec["gt"]["robot"][arm]
    g = R.get("gripper_open_measured")
    P = model.fk_world(arm, q, g)
    for ln in list(R["links"]):
        if rigid_from and ln in rigid_from:
            R["links"][ln] = T_to_pose7(P["link6"] @ rigid_from[ln])
        elif ln in P:
            R["links"][ln] = T_to_pose7(P[ln])
    jp = list(R.get("joint_pos") or [])
    if len(jp) >= 6:
        R["joint_pos"] = q + jp[6:]
    st = rec["obs"]["state"]
    st[f"{arm}_arm_joint_state"] = q
    st[f"{arm}_ee_pose"] = T_to_pose7(P["link6"])
    R["flange"] = T_to_pose7(P["link6"])


def make_variants(cap_path: Path, rng_seed: int = 0, n_grasp: int = 2, n_looks: int = 2, n_shows: int = 2,
                  slide_mm=(1.0, 3.0), turn_deg=(2.0, 6.0)) -> list[Path]:
    """Kinematic variants of one capture (see the module doc). Returns the written frame files."""
    from .robot_model import X5Model

    model = X5Model.default()
    cap_path = Path(cap_path)
    rec0 = json.loads(cap_path.read_text())
    h = rec0["held"]
    arm, obs = h["arm"], ("right" if h["arm"] == "left" else "left")
    tag0 = rec0["frame"]
    k = tag0[1:]
    gt0 = held_gt(rec0, model)
    T_tco = np.asarray(gt0["T_tcp_obj"])
    R0 = rec0["gt"]["robot"][arm]
    q_h = rec0["obs"]["state"][f"{arm}_arm_joint_state"]
    T6 = model.fk_world(arm, q_h)["link6"]
    rigid = {ln: inv_T(T6) @ pose7_to_T(R0["links"][ln]) for ln in ("link7", "link8") if ln in R0["links"]}
    rng = np.random.default_rng(rng_seed + int(k) * 7919 + int(rec0["meta"].get("seed", 0)))
    written = []

    def emit(tag, rec, T_tcp_obj_gt, variant):
        rec = copy.deepcopy(rec)
        rec["frame"] = tag
        rec["obs"]["images"] = {c: f"frame_{tag}_{c}.png" for c in CAMS}
        rec["obs"]["image_shape"] = {}
        rec["meta"] = {**rec0["meta"], "renderer": None, "variant": variant, "of": tag0}
        rec["gt"].pop("cameras", None)
        a = rec["held"]["arm"]
        T6v = model.fk_world(a, rec["obs"]["state"][f"{a}_arm_joint_state"])["link6"]
        T_obj = tcp_from_flange_T(T6v) @ np.asarray(T_tcp_obj_gt)
        for r in rec["gt"]["objects"]:
            if r.get("label") == rec["held"]["label"]:
                r["pos"], r["quat"] = T_to_pose7(T_obj)[:3], T_to_pose7(T_obj)[3:]
        p = cap_path.parent / f"frame_{tag}.json"
        p.write_text(json.dumps(rec))
        written.append(p)

    # grasp variations at the hover: a turn about the closing axis (TCP y) through the TCP and a slide in the pad plane
    for j in range(n_grasp):
        ang = math.radians(rng.uniform(*turn_deg)) * rng.choice([-1, 1])
        sl = rng.uniform(*slide_mm) / 1000 * np.array([rng.choice([-1, 1]), 0.0, rng.choice([-1, 1])]) \
            * rng.uniform(0.3, 1.0, 3)
        D = np.eye(4)
        D[:3, :3] = rotvec_R([0.0, ang, 0.0])
        D[:3, 3] = sl
        emit(f"V{k}g{j}", rec0, D @ T_tco, {"kind": "grasp", "turn_deg": math.degrees(ang), "slide_mm": (sl * 1000).tolist()})
    pc = h.get("plan_costs") or {}
    looks = [L for L in pc.get("looks", []) if L.get("ok")][:n_looks]
    for j, L in enumerate(looks):
        rec = copy.deepcopy(rec0)
        _set_arm(rec, obs, L["q_end"], model)
        emit(f"V{k}o{j}", rec, T_tco, {"kind": "observer_look", "name": L["name"], "steps": L["steps"],
                                       "eye": L["eye"], "dist": L["dist"]})
    shows = [S for S in pc.get("shows", []) if S.get("ok")][:n_shows]
    for j, S in enumerate(shows):
        rec = copy.deepcopy(rec0)
        _set_arm(rec, arm, S["q_end"], model, rigid_from=rigid)
        sl = [L for L in S.get("looks", []) if L.get("ok")]
        if sl:
            _set_arm(rec, obs, sl[0]["q_end"], model)
        emit(f"V{k}s{j}", rec, T_tco, {"kind": "show", "name": S["name"], "steps_out": S["steps_out"],
                                       "steps_back": S.get("steps_back"), "look": sl[0]["name"] if sl else None,
                                       "look_steps": sl[0]["steps"] if sl else None})
    return written


# ------------------------------------------------------------------------------------------------ 3. scoring
def lead_axis_from_target(V: np.ndarray, T_obj_target) -> tuple[np.ndarray, np.ndarray]:
    """Insertion geometry in the object frame from the place target: the axis = world down at the target, the leading
    point = the centroid of the mesh points within 1 mm of the lowest one there."""
    T = np.asarray(T_obj_target, float)
    axis = T[:3, :3].T @ np.array([0.0, 0.0, -1.0])
    z = V @ T[2, :3]                                   # world height of each mesh point at the target (up to a shift)
    lead = V[z <= z.min() + 0.001].mean(0)
    return lead, axis / np.linalg.norm(axis)


def pose_errors(T_est, T_gt, lead_obj, axis_obj) -> dict:
    """Errors of an in-hand estimate (both 4x4 TCP-frame transforms): translation of the object origin and of the
    leading point (TCP frame, mm), rotation (deg, about TCP axes as a rotation vector) and the insertion-axis angle."""
    T_est, T_gt = np.asarray(T_est, float), np.asarray(T_gt, float)
    lead_e = T_est[:3, :3] @ lead_obj + T_est[:3, 3]
    lead_g = T_gt[:3, :3] @ lead_obj + T_gt[:3, 3]
    ax_e, ax_g = T_est[:3, :3] @ axis_obj, T_gt[:3, :3] @ axis_obj
    rv = R_rotvec(T_est[:3, :3] @ T_gt[:3, :3].T)
    return {"pos_mm": ((T_est[:3, 3] - T_gt[:3, 3]) * 1000).tolist(),
            "lead_mm": ((lead_e - lead_g) * 1000).tolist(),
            "lead_norm_mm": float(np.linalg.norm(lead_e - lead_g) * 1000),
            "rot_deg": np.degrees(rv).tolist(), "rot_norm_deg": float(np.degrees(np.linalg.norm(rv))),
            "axis_deg": float(np.degrees(math.acos(float(np.clip(ax_e @ ax_g, -1, 1)))))}
