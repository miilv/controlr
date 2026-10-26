"""``g3 snippet`` code: print every rigid task object's simulated centre of mass and mass as JSON lines (``COM {...}``)
for ``dojo/skills/data/physics_com.json`` (asset key -> {"com": object-frame xyz, "mass"}).

    ~/dojo-skillsN/scripts/skills/g3.sh snippet --task T --layout L --out OUT --code scripts/skills/physics_com.py
"""
import json  # noqa: E402

import numpy as np  # noqa: E402

lm = run.env.scene_manager.layout_manager  # noqa: F821  (snippet scope)
for r in run.objects_gt(0):  # noqa: F821
    if r.get("type") != "Rigid" or r.get("clutter"):
        continue
    obj = lm.get_scene_object(0, r["inst_name"])
    pos, _ = obj.get_com()
    key = f"Rigid/{r['category']}/{int(r['category_idx']):05d}"
    print("COM", json.dumps({key: {"com": np.round(np.asarray(pos, float).reshape(-1)[:3], 5).tolist(),
                                   "mass": round(float(np.asarray(obj.get_mass()).reshape(-1)[0]), 4)}}), flush=True)
