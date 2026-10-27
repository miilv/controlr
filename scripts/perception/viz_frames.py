"""Overlay a val_frames record (yellow) and the frame's GT (green) on the head view. argv: ROOT TASK REC_DIR OUT FRAME ID..."""
import json, os, sys
from pathlib import Path
import numpy as np, cv2
from dojo.perception import d0
from dojo.perception.cameras import cameras_from_state, pose_to_T
from dojo.perception.inverse_layout import shape_models
root, task, recd, out, frame = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
out.mkdir(parents=True, exist_ok=True)
for lid in sys.argv[6:]:
    ep = root / task / lid
    obs = d0.load_obs(ep, frame); gt = d0.load_gt(ep, frame)
    cam = cameras_from_state(obs["state"])["cam_head"]
    ov = obs["images"]["cam_head"].copy()
    if os.environ.get("FG"):  # foreground (cyan) and unknown (magenta) against the run's backgrounds
        from dojo.perception.segment import load_backgrounds, segment_views
        bgs = load_backgrounds(Path(os.environ["DOJO_BG_FOLDS"]) / "fold0")
        v = segment_views({"cam_head": ov}, {"cam_head": cam}, obs["state"], bgs)["cam_head"]
        f = v.fg.astype(bool); u = (v.known == 0) & (v.robot == 0)
        ov[f] = (0.5 * ov[f] + [0, 127, 127]).astype(np.uint8)
        ov[u] = (0.5 * ov[u] + [127, 0, 127]).astype(np.uint8)
    rec = json.loads((recd / f"rec_{lid}.json").read_text())
    for col, items in (((0, 255, 0), [(f"{o['type']}/{o['category']}/{o['category_idx']:05d}", o["pos"], o["quat"], o["label"]) for o in gt["objects"]]),
                       ((255, 255, 0), [(f"{t}/{c}/{int(r['category_idx']):05d}", r["default_pos"], r["default_ori"], r["label"])
                                        for t in ("Rigid", "Geometry", "Articulation") for c, l in (rec.get(t) or {}).items() for r in l])):
        for key, p, q, lab in items:
            try:
                s = shape_models(key)[0].sil(cam, pose_to_T(np.asarray(p), np.asarray(q))).astype(np.uint8)
            except Exception:
                continue
            cs, _ = cv2.findContours(s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cs, -1, col, 1)
            if cs and col[0] == 255:
                x, y, w, h = cv2.boundingRect(np.concatenate(cs))
                cv2.putText(ov, lab[:8], (x, max(8, y - 2)), cv2.FONT_HERSHEY_PLAIN, 0.8, col, 1)
    cv2.imwrite(str(out / f"{task}_{lid}_{frame}.png"), cv2.cvtColor(cv2.resize(ov, (1280, 960), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))
    print(lid)
