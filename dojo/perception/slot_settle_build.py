# DEV: our physics-settled D0 frames (simlab make_d0) -> dojo/perception/slot_settle.json entries.
"""Settled pose of each support-slot object relative to its base (base asset frame), averaged over a D0 set's
frame-0 GT (our physics-only settle). argv: TASK D0_ROOT"""
import json, os, sys, numpy as np
from scipy.spatial.transform import Rotation
from dojo.perception.inverse_layout import task_map, support_slot
from dojo.perception.cameras import pose_to_T
task, root = sys.argv[1], os.path.expanduser(sys.argv[2]); R = f"{root}/{task}"; tm = task_map()[task]["labels"]
acc = {}
for ep in sorted(os.listdir(R)):
    if not ep.isdigit(): continue
    gt = {o["label"]: o for o in json.load(open(f"{R}/{ep}/frame_0.json"))["gt"]["objects"] if o.get("label")}
    for lab, o in gt.items():
        sl = support_slot((tm.get(lab) or {}).get("placement") or {})
        if not sl or sl[0] not in gt: continue
        b = gt[sl[0]]
        bk = f"{b['type']}/{b['category']}/{b['category_idx']:05d}"; ck = f"{o['type']}/{o['category']}/{o['category_idx']:05d}"
        Tr = np.linalg.inv(pose_to_T(b["pos"], b["quat"])) @ pose_to_T(o["pos"], o["quat"])
        acc.setdefault(f"{bk}|{sl[1]}|{ck}", []).append(Tr)
out = {}
for k, Ts in acc.items():
    P = np.array([T[:3, 3] for T in Ts]); q = Rotation.from_matrix(np.array([T[:3, :3] for T in Ts])).mean().as_quat()
    out[k] = {"pos": np.round(P.mean(0), 5).tolist(), "quat": np.round([q[3], q[0], q[1], q[2]], 6).tolist(), "n": len(Ts),
              "spread_mm": round(float(np.linalg.norm(P - P.mean(0), axis=1).max() * 1000), 2)}
print(json.dumps(out, indent=1))
