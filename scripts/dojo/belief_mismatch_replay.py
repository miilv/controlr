"""Dev only (stream H): replay a camera row's executed actions open-loop in the physics-only twin on its dev layout,
then (1) evaluate each condition of the task's remaining success stage, (2) send both arms home with open grippers and
re-check. Used for the belief_mismatch census (journal 2026-10-30-dojo-h95-belief-mismatch).
    RD_HOME=.. DOJO_DATA=.. scripts/simlab/rdpy.sh -u scripts/dojo/belief_mismatch_replay.py --task T --layout L.json \\
        --actions A_actions.json.gz --out O.json"""
import argparse
import gzip
import json
import os
from pathlib import Path
import numpy as np

ARMS = ("left", "right")


def describe(chk):
    if isinstance(chk, tuple) and len(chk) == 2:
        n, a = chk
        keys = {k: v for k, v in (a or {}).items() if k in ("label", "label_A", "label_B", "label_A_list", "tag", "point",
                                                              "arm_tag", "label_list", "labels", "threshold")}
        return f"{n} {keys}"
    if isinstance(chk, list):
        return "ANY[" + " | ".join(describe(c) for c in chk[:4]) + (" ..." if len(chk) > 4 else "") + "]"
    return str(chk)[:80]


def main():
    ap = argparse.ArgumentParser()
    for k in ("--task", "--layout", "--actions", "--out"):
        ap.add_argument(k)
    ap.add_argument("--home-steps", type=int, default=60)
    a = ap.parse_args()
    from dojo.simlab import paths
    from dojo.simlab.runner import RoboDojoRunner
    paths.assert_not_eval(a.layout)
    out = Path(a.out)
    run = RoboDojoRunner(a.task, workdir=str(out.parent / f"work_{out.stem}"), physics_only=True, need_planner=False)
    run.load(json.loads(Path(a.layout).read_text()))
    acts = json.load(gzip.open(a.actions, "rt"))["actions"]
    s0 = run.obs_state(0)
    q_home = {arm: np.asarray(s0[f"{arm}_arm_joint_state"], float).reshape(-1) for arm in ARMS}
    env, rm = run.env, run.env.reward_manager
    ended = False
    for act in acts:
        if run.step({k: np.asarray(v, np.float32) for k, v in act.items()})[0]:
            ended = True
            break
    res = {"task": a.task, "layout": a.layout, "steps": int(env.take_action_cnt[0]), "ended": ended,
           "reward_end": run.reward(0), "score_end": run.score(0), "conditions": []}
    if rm.check_list[0]:
        for chk in rm.check_list[0][0]:
            try:
                ok = bool(rm.check_once(chk, 0))
            except Exception as ex:     # noqa: BLE001
                ok = f"error {ex}"
            res["conditions"].append([describe(chk), ok])
    if not ended and a.home_steps > 0:
        q_last = {arm: np.asarray(acts[-1][f"{arm}_arm_joint_state"], float).reshape(-1) for arm in ARMS}
        for i in range(1, a.home_steps + 1):
            f = min(1.0, i / (0.6 * a.home_steps))
            act = {}
            for arm in ARMS:
                act[f"{arm}_arm_joint_state"] = (q_last[arm] + f * (q_home[arm] - q_last[arm])).astype(np.float32)
                act[f"{arm}_ee_joint_state"] = np.asarray([1.0], np.float32)
            if run.step(act)[0]:
                break
        res["reward_home"] = run.reward(0)
        res["score_home"] = run.score(0)
        res["conditions_home"] = []
        if rm.check_list[0]:
            for chk in rm.check_list[0][0]:
                try:
                    ok = bool(rm.check_once(chk, 0))
                except Exception as ex:  # noqa: BLE001
                    ok = f"error {ex}"
                res["conditions_home"].append([describe(chk), ok])
    out.write_text(json.dumps(res, default=str))
    print("[bm]", json.dumps({k: res[k] for k in ("task", "steps", "ended", "score_end")}, default=str),
          "end:", [c for c in res["conditions"] if c[1] is not True], "| home:",
          res.get("reward_home", {}).get("success_flag"), res.get("score_home"),
          [c for c in res.get("conditions_home", []) if c[1] is not True], flush=True)
    try:
        run.close()
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
