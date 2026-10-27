"""DEV ONLY: a D0-style frame-0 set (layouts 100000-04) from one of Integration's rendered (genuine 5.1) runs: frame-0
images from the agentic perceiver's call dirs, home-pose state, GT from our own settled layout records.

    python scripts/perception/sam/frames_from_run.py RUN_DIR OUT_ROOT [TASK]   (default store_laptop_and_headphones)
"""
import gzip, json, shutil, sys
from pathlib import Path

run = Path(sys.argv[1]); out = Path(sys.argv[2]); task = sys.argv[3] if len(sys.argv) > 3 else "store_laptop_and_headphones"
lay = Path.home() / "dojo-data/simlab/layouts" / task
calls = sorted((run / "agentic").glob("call_*"))
k = 0
for i in range(5):
    lid = 100000 + i
    rep = json.load(gzip.open(run / f"{task}_perception_verify-bundle_{lid}_report.json.gz"))
    ag = rep.get("agentic") or []
    if k >= len(calls):
        print(lid, "no call dir left, skipped"); continue
    first = calls[k]; k += len(ag)
    if not ag or ag[0]["tag"] != "frame0":
        print(lid, "no frame-0 agentic call, skipped"); continue
    src = next(p for p in first.iterdir() if p.is_dir())
    d = out / task / str(lid); d.mkdir(parents=True, exist_ok=True)
    for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        shutil.copy(src / f"{cam}.png", d / f"frame_0_{cam}.png")
    calib = json.loads((src / "calib0.json").read_text())
    rec = json.loads((lay / f"{lid}.settled.json").read_text())
    objs = []
    for typ in ("Rigid", "Geometry", "Articulation"):
        for cat, lst in (rec.get(typ) or {}).items():
            for o in lst:
                objs.append({"label": o["label"], "type": typ, "category": o["category"], "category_idx": o["category_idx"],
                             "pos": o["default_pos"], "quat": o["default_ori"], "clutter": bool(o.get("clutter", False))})
    import re
    first_user = json.loads((src / "transcript.jsonl").read_text().splitlines()[0])["first_user"][0]
    instr = re.search(r"Instruction: np\.str_\('(.+?)'\)", first_user).group(1)
    state = {"left_arm_joint_state": [0.0] * 6, "right_arm_joint_state": [0.0] * 6,
             "left_ee_joint_state": [1.0], "right_ee_joint_state": [1.0]}
    (d / "frame_0.json").write_text(json.dumps({"frame": 0, "obs": {"state": state, "instruction": instr}, "gt": {"objects": objs},
                                                "meta": {"renderer": "isaac-5.1", "source": str(src)}, "calib0": calib}))
    print(lid, src.parent.name, len(objs), [o["label"] for o in objs])
