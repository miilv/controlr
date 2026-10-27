"""Dev probe (``g3 snippet``) for fold_clothes: can the X5 pads pinch the garment's cloth and carry a sleeve onto the
opposite chest? Prints the garment's particle frame, the checker's sleeve / fold terms before and after.

    scripts/skills/g3.sh snippet --task fold_clothes --layout ~/dojo-data/simlab/layouts/fold_clothes/100000.settled.json \\
        --out ~/dojo-data/gennew/probe_fold --code scripts/skills/probe_fold.py
"""
# names in scope: sk, env, run, np, lay, rebuild
import json
import math
import os

from dojo.skills.executor import ArmPlan
from dojo.skills.geometry import T_from
from dojo.skills.kinematics import TABLE_Z, TIP_BEYOND_TCP, down_R
from dojo.skills.motion import FAST, MEDIUM, SLOW

OPEN, CLOSED = 1.0, 0.0
lm = run.env.scene_manager.layout_manager
rm = run.env.reward_manager
name = lm.get_instance_name(label="target", env_idx=0)
g = lm.get_scene_object(env_idx=0, inst_name=name)
meta = lm.get_instance_metadata(inst_name=name, env_idx=0)
fps = {k: v.get("id", []) for k, v in meta["passive"]["functional"].items()}
org = np.asarray(run.env.scene_manager.env_origins[0].cpu() if hasattr(run.env.scene_manager.env_origins[0], "cpu")
                 else run.env.scene_manager.env_origins[0], float).reshape(-1)[:3]
print("[fold] device", getattr(g, "_device", None), "fps", fps, flush=True)


def pts():
    P, _, pw, ow = g.sample_mesh_vertices()
    P = np.asarray(P, float) - org
    return P, pw, ow


def fp(P, k):
    ids = fps.get(k) or []
    return np.mean(P[ids], axis=0) if ids else None


def terms(tag):
    out = {}
    for side, (a, b, xu, xl, yu, yl) in {"left": ("left_sleeve", "right_chest", 0.14, -0.125, 0.14, -0.125),
                                         "right": ("right_sleeve", "left_chest", 0.125, -0.14, 0.14, -0.125)}.items():
        tx = rm.is_garment_pointA_close_to_pointB_by_x_range(label="target", point_A=a, point_B=b, x_upper=xu, x_lower=xl)
        ty = rm.is_garment_pointA_close_to_pointB_by_y_range(label="target", point_A=a, point_B=b, y_upper=yu, y_lower=yl)
        try:
            out[side] = (rm.check_once(tx, 0), rm.check_once(ty, 0))
        except Exception as e:  # noqa: BLE001
            out[side] = f"error {e!r}"[:200]
    for side in ("left", "right"):
        ty = rm.is_garment_pointA_close_to_pointB_by_y_range(label="target", point_A=f"{side}_hem",
                                                             point_B=f"{side}_shoulder", y_upper=0.17, y_lower=-0.095)
        tx = rm.is_garment_pointA_close_to_pointB_by_x_range(label="target", point_A=f"{side}_hem",
                                                             point_B=f"{side}_shoulder", x_upper=0.12, x_lower=-0.12)
        try:
            out[f"hem_{side}"] = (rm.check_once(tx, 0), rm.check_once(ty, 0))
        except Exception as e:  # noqa: BLE001
            out[f"hem_{side}"] = f"error {e!r}"[:200]
    P, pw, ow = pts()
    print(f"[fold] {tag}: terms {out}; score {run.score()}; " +
          " ".join(f"{k} {np.round(fp(P, k), 3).tolist()}" for k in fps), flush=True)
    return P


P0 = terms("start")
for k in range(6):
    for _ in range(int(os.environ.get("FOLD_SETTLE", "20"))):
        run.step(run.hold_action())
    P_, _, _ = pts()
    print(f"[fold] settle {k}: z range {P_[:, 2].min():.4f} {P_[:, 2].max():.4f} mean {P_[:, 2].mean():.4f}", flush=True)
P0 = terms("settled")
_, pw, ow = pts()
print("[fold] garment pose", pw, ow, "points", P0.shape, "z range", float(P0[:, 2].min()), float(P0[:, 2].max()),
      "xy", np.round(P0[:, :2].min(0), 3).tolist(), np.round(P0[:, :2].max(0), 3).tolist(), flush=True)

TIP_Z = float(os.environ.get("FOLD_TIP_Z", "0.003"))         # fingertips this far over the table at the pinch
LIFT = float(os.environ.get("FOLD_LIFT", "0.06"))


def run_plan(arm, plan, nm):
    sk.ex.run({arm: plan}, name=nm)


def pinch_move(arm, src, dst, yaw):
    """Pinch the cloth at ``src`` (world xy), carry it at LIFT to over ``dst``, lower to 2 cm, let go, back up."""
    q = sk.ex.q[arm].copy()
    z_p = TABLE_Z + TIP_Z + TIP_BEYOND_TCP
    R = down_R(yaw)
    above = T_from([src[0], src[1], z_p + 0.08], R)
    at = T_from([src[0], src[1], z_p], R)
    up = T_from([src[0], src[1], z_p + LIFT], R)
    over = T_from([dst[0], dst[1], z_p + LIFT], R)
    low = T_from([dst[0], dst[1], z_p + 0.02], R)
    p = ArmPlan(arm, q, sk.ex.g[arm])
    m1 = sk.motion.to_pose(arm, q, above, FAST, "fold_above", gripper=OPEN, check=False)
    print("[fold] above", m1.status, m1.note[:100], flush=True)
    if not m1.ok:
        return False
    p.grip(OPEN, 3, "open").move(m1, "above")
    m2 = sk.motion.line(arm, m1.q_end, at, SLOW, "fold_down", gripper=OPEN, check=False)
    print("[fold] down", m2.status, m2.note[:100], flush=True)
    p.move(m2, "down").grip(CLOSED, 10, "close").hold(4, "squeeze")
    m3 = sk.motion.line(arm, m2.q_end, up, SLOW, "fold_up", gripper=CLOSED, check=False)
    p.move(m3, "up")
    run_plan(arm, p, "pinch")
    P = terms(f"{arm} pinched + lifted")
    m4 = sk.motion.line(arm, sk.ex.q[arm], over, MEDIUM, "fold_carry", gripper=CLOSED, check=False)
    print("[fold] carry", m4.status, m4.note[:100], flush=True)
    if not m4.ok:
        return False
    m5 = sk.motion.line(arm, m4.q_end, low, SLOW, "fold_lower", gripper=CLOSED, check=False)
    p2 = ArmPlan(arm, sk.ex.q[arm].copy(), CLOSED).move(m4, "carry").move(m5, "lower").hold(3, "settle")
    p2.grip(OPEN, 6, "release").hold(10, "let_go")
    if m5.ok:
        m6 = sk.motion.line(arm, m5.q_end, over, MEDIUM, "fold_back", gripper=OPEN, check=False)
        p2.move(m6, "back")
    run_plan(arm, p2, "carry")
    terms(f"{arm} released")
    return True


Pn = pts()[0]
ls, rs, lc, rc = fp(Pn, "left_sleeve"), fp(Pn, "right_sleeve"), fp(Pn, "left_chest"), fp(Pn, "right_chest")
arm_l = "left" if ls[0] < rs[0] else "right"
print("[fold] left_sleeve at x", ls[0], "-> arm", arm_l, flush=True)
pinch_move(arm_l, ls[:2], rc[:2], 0.0)
Pn = pts()[0]
rs, lc = fp(Pn, "right_sleeve"), fp(Pn, "left_chest")
pinch_move("right" if arm_l == "left" else "left", rs[:2], lc[:2], 0.0)
f = sk.finish()
print("[fold] finish", f.status, flush=True)
for _ in range(10):
    run.step(run.hold_action())
terms("end")
print("[fold] reward", json.dumps(run.reward()), "score", run.score(), flush=True)
