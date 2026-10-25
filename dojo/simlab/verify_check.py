"""Live check of Twin.verify / load_from_estimate on the G2 ``stack`` record (compute3, physics-only).

    scripts/simlab/rdpy.sh -m dojo.simlab.verify_check --g2-dir ~/dojo-data/simlab/g2_c3b [--num-envs 4]

1. ``load_from_estimate`` full (RoboDojo reset) vs fast (same object set: restore + write poses +
   50-step settle): time and pose agreement after settling.
2. ``verify`` with 2 candidates x k=2 (sample 0 nominal, sample 1 perturbed 2 mm / 1 deg):
   A = the recorded stack trajectory (success expected), B = the same with both grippers held open.
   Sample 0 of A must reproduce the G2 replay bit-exactly (same start state, same actions).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from dojo.simlab.runner import jsonable


def main(argv=None) -> None:
    from dojo.simlab.twin import Twin
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2-dir", required=True)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    g2 = Path(a.g2_dir)
    lay = json.loads((g2 / "stack_layout.json").read_text())
    traj = json.loads((g2 / "stack_traj.json").read_text())["actions"]
    ref = json.loads((g2 / "stack_record.json").read_text())["replay1"]
    tw = Twin("stack_blocks", num_envs=a.num_envs, physics_only=True, need_planner=False)
    res: dict = {"num_envs": tw.num_envs}

    r1 = tw.load_from_estimate(lay)
    p_full = {o["label"]: o["pos"] for o in tw.run.objects_gt(0) if o["label"]}
    r2 = tw.load_from_estimate(lay)
    p_fast = {o["label"]: o["pos"] for o in tw.run.objects_gt(0) if o["label"]}
    res["load_from_estimate"] = {"full_s": r1["s"], "fast_s": r2["s"], "fast": r2["fast"],
                                 "fast_vs_full_max_mm": max(float(np.linalg.norm(np.subtract(p_full[k], p_fast[k]))) * 1000
                                                            for k in p_full)}
    shifted = copy.deepcopy(lay)
    for inst in shifted["Rigid"]["block"]:
        inst["default_pos"][0] += 0.01                      # an "estimate" 1 cm off in x
    r3 = tw.load_from_estimate(shifted)
    p_sh = {o["label"]: o["pos"] for o in tw.run.objects_gt(0) if o["label"]}
    res["load_from_estimate"]["shifted_fast"] = r3["fast"]
    res["load_from_estimate"]["shifted_s"] = r3["s"]
    res["load_from_estimate"]["shifted_dx_mm"] = {k: round((p_sh[k][0] - p_full[k][0]) * 1000, 3) for k in p_full
                                                  if k.startswith("block")}

    cand_a = {"actions": traj, "meta": {"name": "stack_recorded"}}
    b_actions = copy.deepcopy(traj)
    for act in b_actions:
        act["left_ee_joint_state"] = [1.0]
        act["right_ee_joint_state"] = [1.0]
    cand_b = {"actions": b_actions, "meta": {"name": "grippers_open"}}
    segs = json.loads((g2 / "stack_traj.json").read_text()).get("segments", [])
    n_lift, acc = None, 0
    for sg in segs:
        acc += sg["steps"]
        if sg["name"].startswith("lift_block_1"):
            n_lift = acc
            break
    cand_c = {"actions": traj[:n_lift or 60], "meta": {"name": "pick_block_1_only", "steps": n_lift}}
    t0 = time.perf_counter()
    outs = tw.verify({"layout": lay}, [cand_a, cand_b, cand_c], perturb={"pos_mm": 2.0, "yaw_deg": 1.0}, k=2,
                     record_traj=True)
    res["verify_s"] = time.perf_counter() - t0
    res["verify"] = [{k: v for k, v in o.items() if k != "traj"} for o in outs]
    a0 = next(o for o in outs if o["candidate"] == 0 and o["sample"] == 0)
    ref_final = ref["frames"][min(len(a0["traj"]), len(ref["frames"])) - 1]["objects"]
    got_final = a0["traj"][len(ref["frames"]) - 2] if len(a0["traj"]) >= len(ref["frames"]) - 1 else a0["traj"][-1]
    res["nominal_vs_g2_replay_max_mm"] = max(float(np.linalg.norm(np.subtract(ref_final[k][:3], got_final[k][:3]))) * 1000
                                             for k in ref_final if k in got_final)
    for o in res["verify"]:
        print(f"[verify] cand {o['candidate']} ({o['meta']['name']}) sample {o['sample']}: success={o['success']} "
              f"score={o['score']} lifted={o['lifted']} dropped={o['dropped']} held={o['held']} "
              f"moved={ {k: round(v, 1) for k, v in o['moved_mm'].items()} }", flush=True)
    print(f"[verify] load_from_estimate {res['load_from_estimate']}", flush=True)
    print(f"[verify] {len(outs)} rollouts in {res['verify_s']:.1f}s; nominal vs G2 replay max "
          f"{res['nominal_vs_g2_replay_max_mm']:.4f} mm", flush=True)
    out = Path(a.out) if a.out else g2 / f"verify_check_n{tw.num_envs}.json"
    out.write_text(json.dumps(jsonable(res), indent=1))
    tw.close()


if __name__ == "__main__":
    main()
