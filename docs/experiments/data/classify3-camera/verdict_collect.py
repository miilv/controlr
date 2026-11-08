"""Read-only collector for the CLASSIFY3 camera rows (run via `ssh compute4 python3 - < verdict_collect.py`).
Reads only ~/dojo-data/integ-c4/runs2/c4b_*_cls3[AB]r1_1000?? dirs; prints one JSON document."""
import collections
import glob
import gzip
import json
import os
import re

R = os.path.expanduser("~/dojo-data/integ-c4/runs2")
out = []
for d in sorted(glob.glob(R + "/c4b_*_cls3[AB]r1_1000[0-9][0-9]")):
    name = os.path.basename(d)
    m = re.match(r"c4b_(\w+?)_cls3([AB])r1_(\d+)$", name)
    row = {"name": name, "cell": m.group(1), "arm": m.group(2), "id": int(m.group(3))}
    eps = glob.glob(d + "/*_episodes.jsonl")
    if not eps:
        row["error"] = "no episodes.jsonl"
        out.append(row)
        continue
    ep = [json.loads(ln) for ln in open(eps[0]) if ln.strip()]
    e = ep[-1]
    row.update(n_episode_records=len(ep), task=e.get("task"), layout=e.get("layout"), score=e.get("score"),
               success=e.get("success"), steps=e.get("steps_used"), step_lim=e.get("step_lim"),
               category=e.get("category"), wall_s=e.get("wall_s"), unstable=e.get("unstable_episode"),
               code=(e.get("code") or {}).get("head"))
    rp = glob.glob(d + "/*_report.json.gz")
    rep = json.load(gzip.open(rp[0])) if rp else {}
    rep = rep if isinstance(rep, dict) else {}
    cfg = rep.get("config") or {}
    row["cfg"] = {k: cfg.get(k) for k in ("sort_contain", "sort_park", "instance_evidence", "ported_classify", "preset")}
    row["cfg_full_sha"] = __import__("hashlib").sha256(
        json.dumps({k: v for k, v in cfg.items() if k != "sort_contain"}, sort_keys=True).encode()).hexdigest()
    row["history"] = [h.get("skill") if isinstance(h, dict) else str(h) for h in rep.get("history") or []]
    evs = [x for x in rep.get("events") or [] if isinstance(x, dict)]
    row["writers"] = dict(collections.Counter(x.get("writer") for x in evs if x.get("kind") == "belief_write"))
    row["event_kinds"] = dict(collections.Counter(x.get("kind") for x in evs))
    port = [x for x in evs if x.get("kind") == "ported"]
    claims = []
    for x in port:
        det = x.get("details")
        det = (json.loads(det) if isinstance(det, str) else det) or {}
        row["ported_skill"] = x.get("skill")
        row["contain"] = det.get("contain")
        for rec in [r_ for r_ in det.get("log") or [] if isinstance(r_, dict)]:
            if "truth_in" in rec:
                claims.append(bool(rec["truth_in"]))
    row["claims"] = len(claims)
    row["claims_in"] = sum(claims)
    row["evidence"] = bool(rep.get("evidence"))
    row["n_candidates"] = len((rep.get("evidence") or {}).get("candidates") or [])
    # route: every agentic call record and every transcript request
    models, calls, tok = collections.Counter(), 0, collections.Counter()
    fns = collections.Counter()
    for cj in glob.glob(d + "/agentic/call_*/calls.jsonl"):
        for ln in open(cj):
            if not ln.strip():
                continue
            c = json.loads(ln)
            models["calls.jsonl:" + str(c.get("model"))] += 1
            fns[str(c.get("fn"))] += 1
            calls += int(c.get("calls") or 0)
            t = c.get("tokens") or {}
            if isinstance(t, str):
                t = eval(t, {}, {})  # noqa: S307 (our own log: a dict repr)
            for k, v in t.items():
                tok[k] += int(v or 0)
    tm = collections.Counter()
    for tr in glob.glob(d + "/agentic/call_*/*/transcript.jsonl"):
        for mm in re.findall(r'"model": "([^"]+)"', open(tr, errors="replace").read()):
            tm[mm] += 1
    ag = [a for a in rep.get("agentic") or [] if isinstance(a, dict)]
    rm = collections.Counter(str(a.get("model")) for a in ag)
    row.update(call_records=dict(models), call_fns=dict(fns), transcript_models=dict(tm), report_models=dict(rm),
               llm_calls=calls, tokens=dict(tok),
               report_calls=sum(int(a.get("calls") or 0) for a in ag))
    lg = open(d + "/integ_eval.log", errors="replace").read() if os.path.exists(d + "/integ_eval.log") else ""
    row["agentic_model_env"] = sorted(set(re.findall(r"AGENTIC_MODEL=(\S+)", lg)))
    row["http_429"] = len(re.findall(r"\b429\b", lg))
    out.append(row)
print(json.dumps(out))
