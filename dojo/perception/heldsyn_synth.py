"""Synthetic held-object wrist frames (DEV, held_look calibration): from a stack_blocks layout's stage (watch_seq
--steps 0) and the D0 look poses (d0 frame_1 / frame_2 robot state), put one block between one arm's fingers (fingers
at the opening that clamps it) or close the fingers on nothing, and write frame_H<k>.json (images: render_d0_proxy).
GT (the block in the jaws or none) is kept in rec["held_gt"] for the scorer only."""
import copy, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dojo.perception.robot_model import X5Model
from dojo.perception.render import robot_link_meshes, load_asset
from dojo.perception.tracker import tcp_from_flange
from dojo.perception.cameras import quat_wxyz_to_R

root = Path(sys.argv[1]); d0 = Path(sys.argv[2]).expanduser()
model = X5Model.default(); meshes = robot_link_meshes()


def T7(T):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return [float(v) for v in T[:3, 3]] + [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def gap(arm, q6, o, Rf, pf):
    P = model.fk_world(arm, q6, o)
    W = {ln: meshes[ln][0] @ P[ln][:3, :3].T + P[ln][:3, 3] for ln in ("link7", "link8")}
    L = {ln: (W[ln] - pf) @ Rf for ln in W}  # flange frame
    tip = max(L["link7"][:, 0].max(), L["link8"][:, 0].max())
    A = {ln: L[ln][L[ln][:, 0] > tip - 0.03] for ln in L}  # outer 3 cm of the fingers
    ax = int(np.argmax(np.abs(A["link7"].mean(0) - A["link8"].mean(0))[1:])) + 1
    lo, hi = sorted(("link7", "link8"), key=lambda ln: A[ln][:, ax].mean())
    return float(A[hi][:, ax].min() - A[lo][:, ax].max()), P


for d in sorted(p for p in (root / "stack_blocks").iterdir() if p.is_dir() and (p / "frame_W0000.json").exists()):
    rec0 = json.load(open(d / "frame_W0000.json"))
    k = 0
    for fk in (1, 2):
        src = d0 / "stack_blocks" / d.name / f"frame_{fk}.json"
        if not src.exists():
            continue
        look = json.load(open(src))
        st = look["obs"]["state"]
        for arm in ("left", "right"):
            q6 = np.asarray(st[f"{arm}_arm_joint_state"], float)
            if np.abs(q6).max() < 0.02:
                continue
            fl = np.asarray(st[f"{arm}_ee_pose"], float)
            Rf, pf = quat_wxyz_to_R(fl[3:7]), fl[:3]
            tcp = tcp_from_flange(fl)
            # sanity: our FK at the recorded opening vs the recorded finger link poses
            g_rec = float(np.asarray(look["gt"]["robot"][arm].get("gripper_open_measured", 1.0)).reshape(-1)[0])
            P_rec = model.fk_world(arm, q6, g_rec)
            dev = max(np.linalg.norm(P_rec[ln][:3, 3] - np.asarray(look["gt"]["robot"][arm]["links"][ln][:3])) for ln in ("link7", "link8"))
            blocks = [o for o in rec0["gt"]["objects"] if (o.get("label") or "").startswith("block_")]
            os_ = np.linspace(0.0, 1.0, 101)
            gaps = [gap(arm, q6, o, Rf, pf)[0] for o in os_]
            for case in [b["label"] for b in blocks] + [None]:
                rec = copy.deepcopy(rec0)
                rec["obs"]["state"] = copy.deepcopy(st)
                rec["gt"]["robot"] = copy.deepcopy(look["gt"]["robot"])
                if case is None:
                    o_f = 0.0  # closed on nothing (joint limit)
                    held = None
                else:
                    b = next(x for x in rec["gt"]["objects"] if x.get("label") == case)
                    w = float(np.min(load_asset(f"Rigid/block/{b['category_idx']:05d}").extents))
                    o_f = float(os_[int(np.argmin([abs(g - w) for g in gaps]))])
                    yaw = np.random.default_rng(k + int(d.name)).uniform(-0.15, 0.15)
                    Tb = np.eye(4); Tb[:3, :3] = Rf @ Rotation.from_euler("x", yaw).as_matrix(); Tb[:3, 3] = tcp
                    b["pos"], b["quat"] = T7(Tb)[:3], T7(Tb)[3:]
                    held = {"label": case, "key": f"Rigid/block/{b['category_idx']:05d}", "pose7": T7(Tb), "width": w}
                _, P = gap(arm, q6, o_f, Rf, pf)
                for ln in ("link7", "link8"):
                    rec["gt"]["robot"][arm]["links"][ln] = T7(P[ln])
                cmd = float(np.random.default_rng(k).choice([0.0, 0.3]))
                rec["obs"]["state"][f"{arm}_ee_joint_state"] = [cmd]
                tag = f"H{k:02d}"
                rec["frame"] = tag
                rec["obs"]["images"] = {c: f"frame_{tag}_{c}.png" for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
                rec["meta"] = {"renderer": None, "synthetic": "block in the jaws (dev held_look calibration)"}
                rec["held_gt"] = {"arm": arm, "held": held, "opening_set": o_f, "cmd": cmd, "look_frame": fk,
                                  "fk_dev_m": round(float(dev), 4)}
                (d / f"frame_{tag}.json").write_text(json.dumps(rec))
                k += 1
    print(d.name, k, "frames", flush=True)
