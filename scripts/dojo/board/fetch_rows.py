"""stream M north star: pull Integration's episode rows (small fields + the RuntimeConfig subset from each run's report)
from the permitted compute4 integration tree -> STATE_DIR/rows_c4.json. Configs are cached locally per run dir (reports are big)."""
import json, os, subprocess, sys, shlex

from state import options
ARGS = options()
D = str(ARGS.state_dir)
KEYS = ["preset", "grip_check", "agentic", "held_pose", "put_into_opts", "refine_recentre_guard", "seg_propose",
        "post_place_read", "a1", "wrist_check", "bg_perception", "place_surface", "refine_search_radius", "refine_empty_hand",
        "ported_routes", "ported_swap", "new_routes", "drop_verdict", "select_ties", "joint_limits", "ported_drag", "belt_wrist_veto", "ported_pour", "slot_read", "held_thin", "ported_pickup", "align_measure", "ported_pour_aim", "series_labels", "tile_grasp", "tile_recover", "vocab_found_needs_score", "ported_classify", "imitate_pairing", "kong_draw_read", "key_handover", "key_lean", "imitate_wait", "imitate_look", "ported_pour_pace"]
REMOTE = r'''
import json, glob, os, gzip, sys, re
SKIP = json.loads(sys.argv[2]); KEYS = json.loads(sys.argv[3])   # {run: report mtime already parsed}
out, cfgs = [], {}
def permitted_episode_files(pattern):
    root = os.path.expanduser(pattern).split("/*/")[0]
    for entry in os.scandir(root):
        # Reject before traversing the run directory; never follow symlink aliases.
        if not entry.is_dir(follow_symlinks=False):
            continue
        run = entry.name
        ids = [int(x) for x in re.findall(r"(?<![0-9])([0-9]{6,})(?![0-9])", run)]
        if not ids or any(x >= 100100 for x in ids) or "dc19" in run.lower():
            continue
        for path in sorted(glob.glob(entry.path + "/*_episodes.jsonl")):
            if not os.path.islink(path):
                yield path


for f in permitted_episode_files(sys.argv[1]):
    d = os.path.dirname(f); run = os.path.basename(d)
    rep = sorted(p for p in glob.glob(d + "/*_report.json.gz") if not os.path.islink(p))
    if rep and SKIP.get(run) != os.path.getmtime(rep[0]):     # a re-queued run reuses its dir name: re-parse on change
        if rep:
            try:
                R = json.load(gzip.open(rep[0]))
                c = R.get("config") or {}
                cfgs[run] = {k: c.get(k) for k in KEYS}
                cfgs[run]["_report_mtime"] = os.path.getmtime(rep[0])
                # W: _perception.series[<label>] has a "seg" key exactly when the seg server counted (None: no series)
                seg = None
                for rec in (R.get("records") or {}).values():
                    ser = ((rec or {}).get("_perception") or {}).get("series") if isinstance(rec, dict) else None
                    if isinstance(ser, dict) and ser:
                        seg = bool(seg) or any(isinstance(v, dict) and "seg" in v for v in ser.values())
                cfgs[run]["seg_counted"] = seg
                # W: entries of the frame-0 record carry _perception.glyph_read when the digit / tile classifier ran:
                # numerals, and from #290 (96d5fb4) also config-slot tiles (solve's t0 / t1 / t2 / t4, make_kong slots)
                glyph = None
                ini = (R.get("records") or {}).get("initial") or {}
                for typ, cats in (ini.items() if isinstance(ini, dict) else []):
                    if not isinstance(cats, dict):
                        continue
                    for cat, ents in cats.items():
                        if not isinstance(ents, list):
                            continue
                        glyphy = any(w in str(cat).lower() for w in ("number", "tile", "operator", "digit", "equation"))
                        for e in ents:
                            if isinstance(e, dict):
                                ran = isinstance((e.get("_perception") or {}).get("glyph_read"), dict)
                                if ran:
                                    glyph = True
                                elif glyphy and glyph is None:
                                    glyph = False
                cfgs[run]["glyph_ran"] = glyph
                # W (29 Oct): imitate detection lever (support_mask + wrist_pool support_masked): the gate-off "why" values
                # wherever they appear in the records (make_kong guard: both must show their gate-off reason)
                gw = {"support_mask": set(), "wrist_pool": set()}
                stack = [R.get("records") or {}]
                seen = 0
                while stack and seen < 200000:
                    o = stack.pop(); seen += 1
                    if isinstance(o, dict):
                        for k_, v_ in o.items():
                            if k_ in gw and isinstance(v_, dict) and len(gw[k_]) < 8:
                                gw[k_].add(str(v_.get("why")) if "why" in v_ else "<no why: active>")
                            if isinstance(v_, (dict, list)):
                                stack.append(v_)
                    elif isinstance(o, list):
                        stack.extend(x for x in o if isinstance(x, (dict, list)))
                cfgs[run]["support_mask_why"] = sorted(gw["support_mask"])
                cfgs[run]["wrist_pool_why"] = sorted(gw["wrist_pool"])
                # W: the initial record (B1's first stage) must carry both fields; agentic-only records may lack _perception
                ini_s = json.dumps((R.get("records") or {}).get("initial") or {})
                cfgs[run]["smask_initial_both"] = ('"support_mask"' in ini_s) and ('"wrist_pool"' in ini_s)
                # W (30 Oct, bundle arm C): the initial record's _perception gate fields + wrist_identity diagnostics
                ip = (((R.get("records") or {}).get("initial") or {}).get("_perception") or {}) if isinstance((R.get("records") or {}).get("initial"), dict) else {}
                sm, wp, wi = ip.get("support_mask"), ip.get("wrist_pool"), ip.get("wrist_identity")
                cfgs[run]["init_gate"] = {
                    "support_mask_applied": (sm.get("applied") if isinstance(sm, dict) else None),
                    "wrist_pool_why": (wp.get("why") if isinstance(wp, dict) else None),
                    "wrist_identity_why": (wi.get("why") if isinstance(wi, dict) else None),
                    "wrist_identity_changed": (wi.get("changed") if isinstance(wi, dict) else None),
                    "wrist_identity_checked": (wi.get("checked") if isinstance(wi, dict) else None),
                    "wrist_identity_time_s": (wi.get("time_s") if isinstance(wi, dict) else None),
                    "wrist_identity_budget_hit": (wi.get("budget_hit") if isinstance(wi, dict) else None)}
                # the imitate log line when repair fires ("[imitate] ... dropped after 2 tries"): count in the run dir's logs
                # W: the imitate skill log lives in the report's "log" list (not the run dir's *.log)
                cfgs[run]["imitate_repair_drops"] = sum("dropped after 2 tries" in str(l_) for l_ in (R.get("log") or []))
                try:      # the run's catalogue: <runs2>/<run>.catalog sidecar (index_md5)
                    cfgs[run]["catalog_index_md5"] = json.load(open(d + ".catalog")).get("index_md5") if not os.path.islink(d + ".catalog") else None
                except Exception:
                    cfgs[run]["catalog_index_md5"] = None
            except Exception:
                pass
    try:
        lines = open(f).read().splitlines()
    except Exception:
        continue
    for ln in lines:
        try:
            r = json.loads(ln)
        except Exception:
            continue
        lid = r.get("layout_id")
        if not isinstance(lid, int) or lid >= 100100:
            continue
        if "success" not in r and not r.get("unstable_episode"):
            continue
        code = r.get("code") if isinstance(r.get("code"), dict) else {}
        ag = r.get("agentic") or []
        out.append(dict(run=run, task=r.get("task"), layout_id=r.get("layout_id"), success=r.get("success"),
                        score=r.get("score"), mode=r.get("mode"), preset=r.get("preset"), renderer=r.get("renderer"),
                        head=code.get("head") or r.get("head"), grip_source=r.get("grip_source"),
                        layout=os.path.basename(str(r.get("layout") or "")), steps=r.get("steps_used"),
                        skill=str(r.get("skill") or (r.get("result") or {}).get("skill") or "")[:80],
                        layout_gen=r.get("layout_gen"), why=str(r.get("why") or "")[:160],
                        first_close=min([c.get("step") for c in ((r.get("dev_gt_probe") or {}).get("closes") or [])
                                         if isinstance(c, dict) and c.get("step") is not None] or [None], key=lambda x: (x is None, x)),
                        catalog_md5=(r.get("catalog_md5") or (r.get("perception") or {}).get("catalog_md5")
                                     if isinstance(r.get("perception"), dict) or r.get("catalog_md5") else None),
                        category=r.get("category"), unstable=bool(r.get("unstable_episode")),
                        agentic_calls=sum(a.get("calls", 0) for a in ag if isinstance(a, dict)), mtime=os.path.getmtime(f)))
norow = []  # do not enumerate unrelated or sealed runs
print(json.dumps({"rows": out, "cfgs": cfgs, "norow": norow}))
'''
# Only the explicitly permitted integration tree on compute4. No other hosts or trees.
for box, glob_ in (("c4", "~/dojo-data/integ-c4/runs2/*/*_episodes.jsonl"),):
    host = "compute4"
    cache_f = f"{D}/cfg_{box}.json"
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    try:
        command = shlex.join(["python3", "-", glob_,
                              json.dumps({k: v.get("_report_mtime") for k, v in cache.items()}),
                              json.dumps(KEYS)])
        p = subprocess.run(["timeout", "180", "ssh", "-o", "BatchMode=yes", host, command],
                           input=REMOTE, capture_output=True, text=True, check=True)
        res = json.loads(p.stdout)
    except Exception as e:
        print(f"fetch {host} failed: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    cache.update(res["cfgs"])
    json.dump(cache, open(cache_f, "w"))
    for r in res["rows"]:
        r["cfg"] = cache.get(r["run"])
    json.dump(res.get("norow", []), open(f"{D}/norow_{box}.json", "w"))
    json.dump(res["rows"], open(f"{D}/rows_{box}.json.tmp", "w"))
    os.replace(f"{D}/rows_{box}.json.tmp", f"{D}/rows_{box}.json")
    print(host, len(res["rows"]), "rows,", len(res["cfgs"]), "new configs")
