"""Our dev demo lengths and their length under the public RoboDojo timing (same geometric paths).
Public facts used: sim dt 0.004 s, collect_freq 25 -> 10 sub-steps per env step (env_cfg); the Franka's URDF joint
velocity limits (2.175 rad/s j1-4, 2.61 j5-7), cuRobo max_acceleration 15 rad/s^2 (Assets/Robots/franka/curobo.yml,
no velocity_scale), RoboDojo's planner interpolation_dt = dt; gripper moves 100 sub-steps (robot_manager
plan_endeffector_joint default)."""
import glob, json, pickle, sys
import numpy as np
VMAX = np.array([2.175] * 4 + [2.61] * 3); AMAX = 15.0; DT = 0.004; SUB = 10
def seg_arrays(seg):
    q = np.array([s["support_arm0_joint_state"]["position"] for s in seg], float)
    g = np.array([s["support_ee0_joint_state"]["position"] for s in seg], float)
    return q, g
def t_min(q):
    L = np.abs(np.diff(q, axis=0)).sum(axis=0)          # per-joint path length (rad)
    t = 0.0
    for l, v in zip(L, VMAX):
        if l <= 1e-6: continue
        tj = l / v + v / AMAX if l > v * v / AMAX else 2 * np.sqrt(l / AMAX)
        t = max(t, tj)
    return t
rows = []
for f in sorted(sum((glob.glob(p) for p in sys.argv[1:]), [])):
    d = pickle.load(open(f, "rb"))
    ours = sum(len(s) for s in d["joint_path"])
    pub_motion, pub_grip, n_grip, n_move, hold = 0, 0, 0, 0, 0
    our_motion, our_grip = 0, 0
    for seg in d["joint_path"]:
        q, g = seg_arrays(seg)
        moves = np.abs(np.diff(q, axis=0)).sum() > 1e-4
        gmove = np.abs(np.diff(g, axis=0)).sum() > 1e-5
        if moves:
            pub_motion += int(np.ceil(t_min(q) / DT)); n_move += 1; our_motion += len(seg)
        elif gmove:
            pub_grip += 100; n_grip += 1; our_grip += len(seg)
        else:
            hold += len(seg)
    rows.append({"file": f.split("/")[-1], "ours_env": ours / SUB,
                 "public_env": (pub_motion + pub_grip) / SUB, "public_motion_env": pub_motion / SUB,
                 "grip_env": pub_grip / SUB, "ours_motion_env": our_motion / SUB, "ours_grip_env": our_grip / SUB, "holds_env": hold / SUB, "segments": len(d["joint_path"]),
                 "moves": n_move, "grips": n_grip})
for r in rows: print(json.dumps(r))
o = np.array([r["ours_env"] for r in rows]); p = np.array([r["public_env"] for r in rows])
print("n", len(rows), "ours env steps: min %.0f median %.0f max %.0f" % (o.min(), np.median(o), o.max()),
      "| public timing: min %.0f median %.0f max %.0f" % (p.min(), np.median(p), p.max()))

for k in ("ours_motion_env", "public_motion_env", "ours_grip_env", "grip_env"):
    v = np.array([r[k] for r in rows]); print(k, "median %.0f" % np.median(v))
