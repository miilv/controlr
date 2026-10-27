"""Settled dev objects outside B1's generator region + margin, under the yawed-centre rule (old) and the generator's
own (the asset turns about its origin after the centre is placed: ``inverse_layout._gen_offset``). DEV layouts only.

    python scripts/perception/region_check.py TASK        (perception venv, DOJO_CATALOG)
"""
import json, sys, glob, os
import numpy as np
from dojo.perception import inverse_layout as IL
from dojo.perception.cameras import quat_wxyz_to_R
task = sys.argv[1]
groups = IL.label_groups(task)
lab2g = {l: g for g in groups for l in g["labels"]}
bad_old = bad_new = n = 0
rows = []
for f in sorted(glob.glob(os.path.expanduser(f"~/dojo-data/simlab/layouts/{task}/1*.settled.json"))):
    lay = json.load(open(f))
    for typ in ("Rigid",):
        for cat, lst in (lay.get(typ) or {}).items():
            for o in lst:
                g = lab2g.get(o.get("label"))
                if g is None or g["type"] != typ:
                    continue
                key = f"{typ}/{cat}/{int(o['category_idx']):05d}"
                R = quat_wxyz_to_R(o["default_ori"]); p = np.asarray(o["default_pos"])
                best = None
                for sm in IL.shape_models(key):
                    Rr = R @ sm.R0.T  # world rot relative to the rest pose: should be a yaw
                    tilt = np.degrees(np.arccos(np.clip(Rr[2, 2], -1, 1)))
                    if best is None or tilt < best[0]:
                        best = (tilt, sm, Rr)
                tilt, sm, Rr = best
                c = IL._bbox_centre_local(sm)
                old = p[:2] + (R @ c)[:2]
                new = p[:2] + (sm.R0 @ c)[:2]
                ok_o = IL.region_ok(g, *old); ok_n = IL.region_ok(g, *new)
                n += 1; bad_old += not ok_o; bad_new += not ok_n
                if not ok_o or not ok_n:
                    rows.append((os.path.basename(f), o["label"], key, round(tilt, 1), np.round(old, 3).tolist(), ok_o, np.round(new, 3).tolist(), ok_n))
print(task, "objects", n, "outside region+margin: old", bad_old, "new", bad_new)
for r in rows[:30]:
    print(" ", *r)
