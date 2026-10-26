# DEV: integration breadth look frames (frame 0) on compute3.
"""Frame-0 estimates on integration's breadth look frames: are the support-slot objects found? argv: TASK RUN_DIR OUT.json"""
import glob, json, os, sys
import numpy as np, cv2
from pathlib import Path
from dojo.perception.inverse_layout import estimate, EstimatorConfig, support_slot, task_map
from dojo.perception.segment import load_backgrounds
task, run, out = sys.argv[1], os.path.expanduser(sys.argv[2]), sys.argv[3]
BG = Path(os.environ["DOJO_BG_FOLDS"]) / "fold0"; bgs = load_backgrounds(BG)
um = BG / "cam_head_clean_unseen.png"
if um.exists():
    bgs["cam_head_clean_unseen"] = (cv2.imread(str(um), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
tm = task_map()[task]["labels"]; slot_labels = [l for l, v in tm.items() if support_slot(v.get("placement") or {})]
rows = []
for d in sorted(glob.glob(f"{run}/*_look_frames")):
    st = {k: np.asarray(v, float) for k, v in json.load(open(f"{d}/frame0_state.json")).items()}
    imgs = {c: cv2.cvtColor(cv2.imread(f"{d}/frame0_{c}.png"), cv2.COLOR_BGR2RGB) for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
    for slots in (False, True):
        rec = estimate(imgs, st, task, "", bgs=bgs, cfg=EstimatorConfig(use_wrists=True, support_slots=slots))
        found = {o["label"] for t in ("Rigid", "Geometry", "Articulation") for c, l in (rec.get(t) or {}).items() for o in l}
        rows.append({"ep": Path(d).name.split("_")[-3], "slots": slots, "slot_found": sorted(set(slot_labels) & found),
                     "all_found": sorted(found), "not_found": [(n["label"], n["reason"]) for n in rec.get("_not_found", [])]})
        print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(out, "w"), indent=1)
