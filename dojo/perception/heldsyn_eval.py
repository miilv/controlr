"""Score B1 held_look (+ what) on the synthetic held frames: asked = the held block (identity check), asked = another
block (label-permutation case), and closed-on-nothing frames. GT from rec["held_gt"] (scorer only)."""
import json, sys
from pathlib import Path
import numpy as np
from dojo.perception import j1_adapter as J
from dojo.perception.cameras import cameras_from_state
root = Path(sys.argv[1]); out = []
rng = np.random.default_rng(0)
for fj in sorted(root.glob("stack_blocks/*/frame_H*.json")):
    rec = json.load(open(fj)); hg = rec.get("held_gt"); d = fj.parent
    if not rec.get("meta", {}).get("renderer") or hg is None:
        continue
    st = rec["obs"]["state"]; sta = {k: np.asarray(v, float) for k, v in st.items()}
    cams = cameras_from_state(sta)
    calib = {k: {"K": c.K.tolist(), "T_world_cam": c.T_wc.tolist()} for k, c in cams.items()}
    frames = {c: str(d / rec["obs"]["images"][c]) for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
    flange = {a: st[f"{a}_ee_pose"] for a in ("left", "right")}
    arm = hg["arm"]
    grip = {a: (st[f"{a}_ee_joint_state"][0] if a == arm else 1.0) for a in ("left", "right")}
    joints = {a: st[f"{a}_arm_joint_state"] for a in ("left", "right")}
    blocks = [o for o in rec["gt"]["objects"] if (o.get("label") or "").startswith("block_")]
    keys = {o["label"]: f"Rigid/block/{o['category_idx']:05d}" for o in blocks}
    held = hg["held"]
    pose = held["pose7"] if held else None
    if pose is None:  # expected in-jaw pose for an empty grasp: where a block would sit (TCP), from another frame's GT
        from dojo.perception.tracker import tcp_from_flange
        pose = list(tcp_from_flange(flange[arm])) + list(flange[arm][3:7])
    for asked in sorted(keys):
        p = list(np.asarray(pose[:3]) + np.r_[rng.uniform(-0.003, 0.003, 2), 0]) + list(pose[3:])
        cands = [{"label": asked, "key": keys[asked], "pose7": p}] + [{"label": l, "key": keys[l], "pose7": None} for l in sorted(keys) if l != asked]
        r = J._held_look_impl(frames, calib, flange, grip, joints=joints, candidates=cands, arm=arm)
        s = r.get(arm, {})
        w = s.get("what") or {}
        out.append({"frame": f"{d.name}/{fj.stem}", "arm": arm, "true": held["label"] if held else None,
                    "true_key": held["key"] if held else None, "asked": asked, "asked_key": keys[asked],
                    "held": s.get("held"), "conf": s.get("confidence"), "evidence": s.get("evidence"),
                    "opening": s.get("opening"), "what_key": w.get("key"), "what_label": w.get("label"),
                    "what_conf": w.get("conf"), "margin": w.get("margin"), "asked_held": s.get("asked_held"),
                    "scored": s.get("scored"), "region": s.get("what_region"), "cmd": hg["cmd"], "fk_dev_m": hg["fk_dev_m"]})
        print(json.dumps({k: out[-1][k] for k in ("frame", "true", "asked", "held", "conf", "what_key", "margin", "asked_held")}), flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
R = out
pos = [r for r in R if r["true"]]; neg = [r for r in R if not r["true"]]
print("held recall", np.mean([r["held"] is True for r in pos]), "n", len(pos), "| empty -> not held", np.mean([r["held"] is False for r in neg]), "n", len(neg))
P = [r for r in pos if r["held"] and r["what_key"]]
print("what top1", np.mean([r["what_key"] == r["true_key"] for r in P]), "n", len(P))
for thr in (0.05, 0.1, 0.15, 0.2):
    S = [r for r in P if (r["margin"] or 0) >= thr]
    print(f"  margin>={thr}: share {len(S)/max(len(P),1):.2f} precision {np.mean([r['what_key'] == r['true_key'] for r in S]) if S else float('nan'):.3f}")
perm = [r for r in pos if r["asked"] != r["true"]]
print("permutation asked_held False", np.mean([r["asked_held"] is False for r in perm]), "None", np.mean([r["asked_held"] is None for r in perm]), "True(wrong)", np.mean([r["asked_held"] is True for r in perm]), "n", len(perm))
same = [r for r in pos if r["asked"] == r["true"]]
print("asked is held: asked_held True", np.mean([r["asked_held"] is True for r in same]), "False(wrong)", np.mean([r["asked_held"] is False for r in same]), "n", len(same))
