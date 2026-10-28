"""Stream C: the validity gate of per-step replays (``seq_capture``). A replay counts only if it reproduces the recorded
run's ``dev_gt_probe`` closes exactly: the same number, the same (step, arm, GT label), and the same jaw distance (mm,
as logged, to 0.0). Rises are not compared: they are maxima over the observation steps, which differ by construction.

    python scripts/perception/bgperc/replay_check.py OUT_ROOT [--json OUT.json]

OUT_ROOT holds ``seq/<name>/meta.json`` (the recorded actions file: its dir has the recorded ``*_episodes.jsonl``) and
``runs/<name>/*_episodes.jsonl`` (the replay's row). Prints one line per replay and the list of failures.
"""
import argparse
import glob
import json
import os


def rows(path_glob: str) -> list[dict]:
    out = []
    for f in glob.glob(path_glob):
        out += [json.loads(ln) for ln in open(f) if ln.strip()]
    return out


def check(root: str, name: str) -> dict:
    meta = json.load(open(os.path.join(root, "seq", name, "meta.json")))
    src = os.path.dirname(meta["actions"])
    lid = meta["layout_id"]
    rec = [r for r in rows(os.path.join(src, "*_episodes.jsonl")) if r.get("layout_id") == lid and r.get("dev_gt_probe")]
    rep = [r for r in rows(os.path.join(root, "runs", name, "*_episodes.jsonl")) if r.get("layout_id") == lid]
    out = {"name": name, "task": meta["task"], "layout_id": lid}
    if not rec or not rep:
        return {**out, "valid": False, "why": "no recorded row" if not rec else "no replay row"}
    a, b = rec[-1]["dev_gt_probe"].get("closes") or [], rep[-1]["dev_gt_probe"].get("closes") or []
    out.update(n_rec=len(a), n_rep=len(b), steps_rec=rec[-1].get("steps_used"), steps_rep=rep[-1].get("steps_used"))
    if len(a) != len(b):
        return {**out, "valid": False, "why": f"closes {len(a)} vs {len(b)}"}
    d = 0.0
    for x, y in zip(a, b):
        if (x.get("step"), x.get("arm"), x.get("label")) != (y.get("step"), y.get("arm"), y.get("label")):
            return {**out, "valid": False, "why": f"close {x} vs {y}"}
        if x.get("dist_mm") is not None or y.get("dist_mm") is not None:
            d = max(d, abs(float(x.get("dist_mm") or 0) - float(y.get("dist_mm") or 0)))
    out["max_dist_diff_mm"] = round(d, 2)
    steps = {json.loads(ln)["step"] for ln in open(os.path.join(root, "seq", name, "steps.jsonl")) if ln.strip()}
    forced = [s for s in meta.get("forced_steps") or [] if s <= (out["steps_rep"] or 0)]
    out["decision_frames"] = {"expected": len(forced), "present": sum(1 for s in forced if s in steps)}
    out["observations"] = len(steps)
    out["valid"] = d == 0.0 and out["steps_rec"] == out["steps_rep"]
    if not out["valid"]:
        out["why"] = f"jaw distance differs by {d:.1f} mm" if d else f"steps {out['steps_rec']} vs {out['steps_rep']}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = []
    for d in sorted(glob.glob(os.path.join(a.root, "runs", "*", "_done"))):
        name = os.path.basename(os.path.dirname(d))
        if not os.path.exists(os.path.join(a.root, "seq", name, "meta.json")):
            continue
        r = check(a.root, name)
        res.append(r)
        print(f"{'OK  ' if r['valid'] else 'FAIL'} {name}: closes {r.get('n_rec')} / {r.get('n_rep')}, "
              f"max |d| {r.get('max_dist_diff_mm')} mm, steps {r.get('steps_rec')} / {r.get('steps_rep')}, "
              f"decision frames {(r.get('decision_frames') or {}).get('present')} / "
              f"{(r.get('decision_frames') or {}).get('expected')}"
              f"{'' if r['valid'] else ' -- ' + r.get('why', '')}")
    bad = [r["name"] for r in res if not r["valid"]]
    print(f"{len(res) - len(bad)} / {len(res)} valid" + (f"; failing: {bad}" if bad else ""))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
