"""DEV validation of refine(): wrist frames of D0 frames 1/2 (scripted looks) and L1 (our looks); priors =
GT perturbed (dx, dy ~ U(-20, 20) mm, yaw ~ U(-10, 10) deg). GT is used only to build priors and to score.

    DOJO_D0=<root> VAL_OUT=<json> python -m dojo.perception.val_refine stack_blocks,push_T 1,2,L1 25
"""
import sys, os, json, zlib, time, numpy as np
sys.path.insert(0, os.environ.get("DOJO_CODE", os.path.expanduser("~/dojo-perception")))
from pathlib import Path
from scipy.spatial.transform import Rotation
from dojo.perception import d0
from dojo.perception.eval_g1 import fold_backgrounds, rot_err_deg, gt_world_symmetry
from dojo.perception.refine import refine
from dojo.perception.cameras import wrist_camera, pose_to_T
root = Path(os.path.expanduser(os.environ.get("DOJO_D0", "~/dojo-data/d0"))); rng = np.random.default_rng(0)
tasks = sys.argv[1].split(","); frames = sys.argv[2].split(","); nmax = int(sys.argv[3])
rows = []
for t in tasks:
    for ep in d0.episodes(root, t)[:nmax]:
        bgs = fold_backgrounds(ep)
        for k in frames:
            if not all((ep / f"frame_{k}_{c}.png").exists() for c in d0.CAMS) or not (ep / f"frame_{k}.json").exists():
                continue
            obs = d0.load_obs(ep, k if not k.isdigit() else int(k)); gt = d0.load_gt(ep, k if not k.isdigit() else int(k))
            for arm in ("left", "right"):
                cam = wrist_camera(arm, obs["state"][f"{arm}_arm_joint_state"])
                if np.abs(np.asarray(obs["state"][f"{arm}_arm_joint_state"])).max() < 0.02:
                    continue  # arm at home: not a look
                for o in gt["objects"]:
                    if o.get("clutter") or not o.get("label") or o["category"] in ("camera_stand", "dustbin"):
                        continue
                    uv, z = cam.project(np.array([o["pos"]]))
                    if z[0] <= 0.03 or not (60 < uv[0, 0] < 580 and 60 < uv[0, 1] < 420):
                        continue
                    key = f"{o['type']}/{o['category']}/{o['category_idx']:05d}"
                    T = pose_to_T(o["pos"], o["quat"])
                    d = rng.uniform(-0.02, 0.02, 2); a = np.deg2rad(rng.uniform(-10, 10))
                    P = T.copy(); P[:2, 3] += d; P[:3, :3] = Rotation.from_euler("z", a).as_matrix() @ T[:3, :3]
                    prior = list(P[:3, 3]) + list(Rotation.from_matrix(P[:3, :3]).as_quat()[[3, 0, 1, 2]])
                    oth = []
                    if os.environ.get("WITH_OTHERS"):
                        for o2 in gt["objects"]:
                            if o2 is o or o2.get("clutter") or o2["category"] in ("camera_stand",) or o2.get("pos") is None:
                                continue
                            p2 = list(np.array(o2["pos"]) + np.r_[rng.uniform(-0.005, 0.005, 2), 0]) + list(o2["quat"])
                            oth.append((f"{o2['type']}/{o2['category']}/{o2['category_idx']:05d}", p2))
                    out = refine(key, obs["images"][f"cam_{arm}_wrist"], obs["state"], prior, arm=arm, bgs=bgs, others=oth)
                    sym = gt_world_symmetry(key, o["quat"])
                    e0 = float(np.linalg.norm(d) * 1000); e1 = float(np.linalg.norm(np.array(out["pose"][:2]) - np.array(o["pos"][:2])) * 1000)
                    r0 = rot_err_deg(prior[3:], o["quat"], sym); r1 = rot_err_deg(out["pose"][3:], o["quat"], sym)
                    rows.append({"task": t, "ep": ep.name, "frame": k, "label": o["label"], "center_px": float(np.hypot(uv[0, 0] - 320, uv[0, 1] - 240)), "xy0": e0, "xy1": e1, "rot0": r0, "rot1": r1, **{kk: out[kk] for kk in ("iou", "visible_frac", "ok", "time_s")}})
json.dump(rows, open(os.environ.get("VAL_OUT", "/tmp/val_refine.json"), "w"))
R = rows; ok = [r for r in R if r["ok"]]
print("n", len(R), "ok", len(ok), "median time", np.median([r["time_s"] for r in R]), "p90", np.percentile([r["time_s"] for r in R], 90))
cen = [r for r in R if r["center_px"] < 160]
print("centred (<160 px from the image centre):", len(cen), "ok", sum(r["ok"] for r in cen))
for name, S in (("all", R), ("ok", ok), ("centred", cen), ("centred_ok", [r for r in cen if r["ok"]])):
    if S: print(name, "xy prior med %.1f -> %.1f mm, p90 %.1f -> %.1f; rot med %.1f -> %.1f deg; within 5 mm %.2f" % (np.median([r["xy0"] for r in S]), np.median([r["xy1"] for r in S]), np.percentile([r["xy0"] for r in S], 90), np.percentile([r["xy1"] for r in S], 90), np.median([r["rot0"] for r in S]), np.median([r["rot1"] for r in S]), np.mean([r["xy1"] <= 5 for r in S])))
for t in tasks:
    S = [r for r in ok if r["task"] == t]
    if S: print(" ", t, len(S), "xy after med %.1f p90 %.1f mm; rot after p90 %.1f" % (np.median([r["xy1"] for r in S]), np.percentile([r["xy1"] for r in S], 90), np.percentile([r["rot1"] for r in S], 90)))
