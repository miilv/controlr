"""ORG2 offline perception census of recorded organize-table camera runs (read-only; dev analysis, no policy input).

    ssh compute4 'python3 - ROOT' < scripts/skills/org2_perc.py > rows.jsonl

For every ``c4b_organize_*`` / ``c6b_organize_*`` run dir under ROOT whose layout id is in our measurement slices
(100000-09, 100020-29): the first planning-time belief snapshot (``dev_belief_probe``) against the episode's own
settled layout record (frame-0 GT; nothing has moved before the first manipulation), per object: xy error of the asset
origin (mm), yaw error (deg, mod 90 for the symmetric fixtures is NOT applied: raw), asset key right / wrong, and the
grasp-time close distances from ``dev_gt_probe``. Prints one JSON row per run. Writes nothing."""
import glob
import gzip
import json
import math
import os
import sys


def yaw(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/dojo-data/integ-c4/runs2")
    for d in sorted(glob.glob(os.path.join(root, "c[46]b_organize_*"))):
        if not os.path.isdir(d):
            continue
        eps = glob.glob(os.path.join(d, "*_episodes.jsonl"))
        reps = glob.glob(os.path.join(d, "*_report.json.gz"))
        if not eps or not reps:
            continue
        try:
            ep = json.loads(open(eps[0]).read().splitlines()[-1])
            rep = json.load(gzip.open(reps[0]))
        except Exception as e:                       # a truncated or running dir
            print(json.dumps({"run": os.path.basename(d), "error": str(e)[:80]}))
            continue
        lid = ep.get("layout_id")
        lay_p = ep.get("layout")
        if lid is None and lay_p:
            lid = int(os.path.basename(lay_p).split(".")[0])
        if lid is None or not (100000 <= lid <= 100009 or 100020 <= lid <= 100029):
            continue
        try:
            lay = json.load(open(lay_p))
        except Exception:
            print(json.dumps({"run": os.path.basename(d), "id": lid, "error": "layout unreadable"}))
            continue
        gt = {}
        for typ in ("Geometry", "Rigid"):
            for _, recs in lay.get(typ, {}).items():
                for r in recs:
                    gt[r["label"]] = (f"{typ}/{r['category']}/{int(r['category_idx']):05d}", r["default_pos"],
                                      r["default_ori"])
        snaps = (rep.get("dev_belief_probe") or {}).get("snaps") or []
        row = {"run": os.path.basename(d), "id": lid, "salt": (lay.get("_simlab") or {}).get("seed_salt"),
               "score": ep.get("score"), "success": ep.get("success"), "result": str(ep.get("result"))[:120]}
        if snaps:
            s0 = snaps[0]
            row["snap_step"] = s0.get("step")
            objs = {}
            for lab, (key, pos, ori) in gt.items():
                b = (s0.get("objects") or {}).get(lab)
                if b is None:
                    objs[lab] = {"missing": True}
                    continue
                e = math.hypot(b["pos"][0] - pos[0], b["pos"][1] - pos[1])
                dy = (yaw(b["quat"]) - yaw(ori) + 180) % 360 - 180
                objs[lab] = {"xy_mm": round(1000 * e, 1), "yaw_deg": round(dy, 1), "key_ok": b.get("key") == key,
                             "source": b.get("source")}
            row["objects"] = objs
        nf = ((rep.get("records") or {}).get("initial") or {}).get("_not_found") or []
        row["not_found_initial"] = [x.get("label") for x in nf]
        row["closes"] = [{k: c.get(k) for k in ("step", "arm", "label", "dist_mm")}
                         for c in ((ep.get("dev_gt_probe") or {}).get("closes") or [])]
        print(json.dumps(row))


if __name__ == "__main__":
    main()
