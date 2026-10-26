# DEV: run in the perception venv on compute3 (GPU light lane, EGL).
"""Support-slot objects on our D0 frames (GT = physics-settled): frame 0 and frame 0 + frame 1 (wrist look at the
base), variants main / slots + assembly. argv: TASK D0_ROOT OUT.json"""
import json, os, sys, time
from pathlib import Path
import numpy as np
from dojo.perception import d0
from dojo.perception.inverse_layout import EstimatorConfig, estimate, support_slot, task_map
from dojo.perception.eval_g1 import fold_backgrounds
task, root, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
tm = task_map()[task]["labels"]
slot_labels = [l for l, v in tm.items() if support_slot(v.get("placement") or {})]
bases = sorted({support_slot(tm[l]["placement"])[0] for l in slot_labels})
VAR = {"main": dict(support_slots=False), "slots_assembly": dict(support_slots=True, slot_assembly=True)}
rows = []
for ep in d0.episodes(root, task):
    o0 = d0.load_obs(ep, 0); gt = {o["label"]: o for o in d0.load_gt(ep, 0)["objects"] if o.get("label")}
    bgs = fold_backgrounds(ep)
    o1 = d0.load_obs(ep, 1) if (ep / "frame_1.json").exists() else None
    for name, kw in VAR.items():
        for look in ((False, True) if o1 is not None else (False,)):
            t0 = time.time()
            rec = estimate(o0["images"], o0["state"], task, o0.get("instruction", ""), bgs=bgs, cfg=EstimatorConfig(use_wrists=True, **kw),
                           extra_frames=[(o1["images"], o1["state"])] if look else None)
            found = {o["label"]: o for t in ("Rigid", "Geometry", "Articulation") for c, l in (rec.get(t) or {}).items() for o in l}
            r = {"ep": ep.name, "variant": name, "look": look, "s": round(time.time() - t0, 1), "objs": {}}
            for lab in slot_labels + bases:
                f, g = found.get(lab), gt.get(lab)
                r["objs"][lab] = None if (f is None or g is None) else {
                    "xy_mm": round(1000 * float(np.linalg.norm(np.asarray(f["default_pos"][:2]) - np.asarray(g["pos"][:2]))), 1),
                    "dz_mm": round(1000 * float(f["default_pos"][2] - g["pos"][2]), 1), "conf": f["_perception"].get("confidence")}
            rows.append(r)
            print(json.dumps({"ep": ep.name, "v": name, "look": look, "objs": {k: (v or {}).get("xy_mm") for k, v in r["objs"].items()}}), flush=True)
json.dump(rows, open(out, "w"), indent=1)
