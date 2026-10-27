"""PROXY end-to-end eval of ``RuntimePolicy`` on our dev layouts (compute3; ``scripts/dojo/integ_eval.sh``).

Episodes go through simlab's ``ProxyEpisode`` (physics-only Isaac 5.1 RoboDojo EvalEnv: its reset, step
accounting, step_lim, reward latches) with frames from the warm Isaac 6.0 render service; marked "proxy".
This driver adds, around simlab's loop:

* mode ``gt``: the dev-only GT rows (``objects_gt``), captured in the main thread at each observation and
  put under ``firewall.DEV_GT_KEY``; mode ``perception``: nothing but the official observation (the
  policy's firewall raises on anything else);
* ``info["dev_calib_fold"]`` = crc32(layout id) % 2, the D0 half the perception backgrounds / colour map
  must NOT come from (they are built from the other half);
* a dev-only scorer: perception's estimate vs GT at episode start, and the failure taxonomy
  (perception_gross / grasp_slip / plan / verify_reject / budget / not_home / divergence / error);
* results per (task, mode, verify) in separate files: ``<task>_<mode>_<verify>_{episodes.jsonl,summary.json}``.

GT is read here (the scorer) and handed to the policy only in mode gt.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np

from dojo.simlab import paths
from dojo.simlab.proxy_eval import PROXY_TAG, RENDERED_TAG, ProxyEpisode, RenderClient, RenderedEpisode
from dojo.simlab.runner import jsonable

from .estimate import objects_from_gt_rows, objects_from_record, planning_record, pose_errors
from .firewall import DEV_GT_KEY
from .policy import RuntimeConfig, RuntimePolicy

GROSS_XY_MM = 20.0
TAXONOMY = ("perception_gross", "grasp_slip", "plan", "verify_reject", "budget", "time_cap", "belief_mismatch", "not_home",
            "divergence", "error", "other")


def event_counts(events: list) -> dict:
    """{event kind: count} over an episode's whole event list."""
    out: dict = {}
    for e in events or []:
        k = str(e.get("kind")) if isinstance(e, dict) else "?"
        out[k] = out.get(k, 0) + 1
    return out

def classify(res: dict, reward: dict, rep: dict, perc: dict | None, mode: str) -> tuple[str | None, str]:
    if res.get("success"):
        return None, ""
    if rep.get("error") or rep.get("error_handled"):
        return "error", (rep.get("error") or rep["error_handled"]).splitlines()[0][:200]
    if mode == "perception" and perc:
        bad = {k: v for k, v in perc.items()
               if not v.get("found") or v.get("xy_mm", 0) > GROSS_XY_MM or not v.get("shape_ok", True)}
        if bad:
            return "perception_gross", json.dumps(bad)[:300]
    fail = next((h for h in rep.get("history", []) if not h["ok"] and h["status"] not in ("ok", "skipped")), None)
    if res.get("steps_used", 0) >= res.get("step_lim", 1) or (fail and fail["status"] in ("step_budget", "ended")):
        return "budget", (fail or {}).get("note", "")[:200]
    v = rep.get("verify")
    if fail is not None:
        if isinstance(v, dict) and v.get("none_ok", 0) > 0:
            return "verify_reject", f"{fail['skill']} {fail['status']}: {fail['note'][:160]}"
        if fail["status"] == "grasp_slip":
            return "grasp_slip", fail["note"][:200]
        return "plan", f"{fail['skill']} {fail['status']}: {fail['note'][:160]}"
    if any(e.get("kind") == "divergence" for e in rep.get("events", [])):
        return "divergence", ""
    if mode == "perception" and (rep.get("result") or {}).get("ok"):
        # the skill finished on the BELIEF (predicted poses) but the checker disagrees: an estimate error that
        # neither verification nor resync caught
        return "belief_mismatch", f"stages_left {reward.get('stages_left')}"
    if reward.get("stages_left", 0) == 1:
        return "not_home", ""
    return "other", str(rep.get("result"))[:200]


def perception_eval(rep: dict, gt_rows: list[dict], assets) -> dict | None:
    """DEV-ONLY scorer: the estimate the skills planned with (``planning_record``; before: the fused record else the
    first, which missed what the scan / agentic / search records found or broke: breadth2 scored 5 episodes on a stale
    record) vs GT at episode start; ``xy_mm_first`` / ``id_ok_first``: the frame-0 record."""
    recs = rep.get("records") or {}
    _, rec = planning_record(recs)
    if not rec:
        return None
    est = objects_from_record(rec, assets, meshes=False)
    gt = {k: o for k, o in objects_from_gt_rows(gt_rows, assets, meshes=False).items()
          if not o.clutter and k != "camera_stand"}
    gt = {k: o for k, o in gt.items() if any(k == r.get("label") for t in ("Rigid", "Geometry", "Articulation")
                                             for lst in (rec.get(t) or {}).values() for r in lst)
          or o.obj_type != "Geometry"}
    out = pose_errors(est, gt)
    if recs.get("initial") and rec is not recs.get("initial"):
        first = pose_errors(objects_from_record(recs["initial"], assets, meshes=False), gt)
        for k, v in out.items():
            if k in first and first[k].get("found"):
                v["xy_mm_first"] = first[k]["xy_mm"]
                v["id_ok_first"] = first[k]["id_ok"]
    return out


def load_policy_factory(spec: str | None):
    """``module:Class`` (called as ``Class(cfg)``); default ``RuntimePolicy``."""
    if not spec:
        return RuntimePolicy
    import importlib
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


def deployed_code() -> dict | None:
    """``.deployed.json`` written by scripts/deploy.sh in this checkout (head, skills / perception / simlab /
    runtime commits), or None (e.g. a local run)."""
    p = Path(__file__).resolve().parents[2] / ".deployed.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def run(task: str, layout_files: list[str], cfg: RuntimeConfig, out: Path, max_minutes: float = 25.0,
        policy_factory=None, renderer: str = "isaac60", episode_max_minutes: float | None = 25.0) -> dict:
    """``renderer``: ``isaac60`` = the proxy (physics-only 5.1 + the 6.0 render service, compute3); ``isaac51`` =
    one 5.1 EvalEnv with RTX (``RenderedEpisode``; a box where 5.1 renders, e.g. isr-lab)."""
    from dojo.skills.world import AssetDB
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{task}_{cfg.mode}_verify-{cfg.verify}"
    if renderer == "isaac51":
        rc, ep = None, RenderedEpisode(task, out)
    else:
        rc = RenderClient()
        ep = ProxyEpisode(task, rc, out)
    pol = (policy_factory or RuntimePolicy)(cfg)
    assets = AssetDB(paths.OBJECTS)
    rows, t_all = [], time.perf_counter()
    code = deployed_code()
    gate = os.environ.get("DOJO_EVAL_GATE")          # a command run before each episode (isr-lab: inbox / reserve / disk)
    stop = None
    for lf in layout_files:
        if (time.perf_counter() - t_all) / 60 > max_minutes:
            rows.append({"layout": lf, "skipped": "time budget of this job"})
            print(f"\n[integ] {tag}: time budget reached, skipping {Path(lf).name}", flush=True)
            continue
        if gate and stop is None:
            g = subprocess.run(gate, shell=True, capture_output=True, text=True)
            if g.returncode != 0:
                stop = (g.stdout + g.stderr).strip()[:200] or f"rc {g.returncode}"
        if stop is not None:
            rows.append({"layout": lf, "skipped": f"gate: {stop}"})
            print(f"\n[integ] {tag}: gate refused, skipping {Path(lf).name}: {stop}", flush=True)
            continue
        paths.assert_not_eval(lf)
        lay = json.loads(Path(lf).read_text())
        info = ep.reset(lay)
        if not info["stable"]:
            rows.append({"layout": lf, "unstable": True})
            continue
        lid = info.get("layout_id")
        info["dev_calib_fold"] = zlib.crc32(str(lid).encode()) % 2
        gt0 = ep.run.objects_gt(0)                  # scorer side (dev only)
        pol.reset(info)
        t0 = time.perf_counter()
        sim_s, obs_s, ended, capped = 0.0, [], False, False
        executed: list[dict] = []                   # what the env ran (the policy's own list misses the batch the latch ended)
        while not ended:
            if episode_max_minutes and time.perf_counter() - t0 > 60.0 * episode_max_minutes:
                capped = True                       # the GPU-lock rule: no episode holds the lane past this
                print(f"\n[integ] {tag} {lid}: episode wall-time cap {episode_max_minutes} min reached at step "
                      f"{ep.result().get('steps_used')}: ending it", flush=True)
                break
            t1 = time.perf_counter()
            obs = ep.observe()
            if cfg.mode == "gt":
                obs[DEV_GT_KEY] = ep.run.objects_gt(0)
            obs_s.append(time.perf_counter() - t1)
            d = pol.next(obs)
            if d.get("done"):
                break
            t2 = time.perf_counter()
            for a in d["actions"]:
                executed.append(a)
                if ep.step(a):
                    ended = True
                    break
            sim_s += time.perf_counter() - t2
        res = ep.result()
        reward = ep.run.reward(0)
        rep = pol.episode_report()
        perc = perception_eval(rep, gt0, assets) if cfg.mode == "perception" else None
        cat, why = classify(res, reward, rep, perc, cfg.mode)
        if capped and not res.get("success"):
            cat, why = "time_cap", f"episode wall-time cap {episode_max_minutes} min at step {res.get('steps_used')}"
        wall = time.perf_counter() - t0
        v = rep.get("verify")
        row = {"task": task, "layout": lf, "layout_id": lid, "mode": cfg.mode, "verify": cfg.verify, "preset": cfg.preset,
               "time_capped": capped,
               "renderer": renderer,
               "success": bool(res["success"]), "score": float(res["score"]), "steps_used": int(res["steps_used"]),
               "step_lim": int(res["step_lim"]), "observations": int(res["observations"]), "wall_s": round(wall, 1),
               "sim_s": round(sim_s, 1), "observe_s": round(float(np.sum(obs_s)), 1),
               "worker_s": rep["timings"].get("worker_s"), "perceive_s": rep["timings"].get("perceive_s"),
               "look_s": rep["timings"].get("look_s"), "reset_s": round(info["reset_s"], 1),
               "n_looks": sum(1 for lk in rep.get("looks", []) if lk.get("status") == "ok"),
               "look_steps": rep.get("look_steps", 0), "flags": rep.get("flags"), "skill": rep.get("skill"),
               "category": cat, "why": why, "reward": reward,
               "verify_stats": v if isinstance(v, dict) else v, "provider": rep.get("provider"),
               "perception": perc, "events": rep.get("events", [])[-30:], "decisions": rep.get("decisions"),
               # the row keeps the last 30 events; the episode's setup events (scan, gates at step 0) and per-kind
               # counts of all of them stay readable from the row too (the full list is in the report.json.gz)
               "events_head": rep.get("events", [])[:8], "event_counts": event_counts(rep.get("events", [])),
               "history": rep.get("history", [])[-40:], "result": rep.get("result"), "info_dropped": rep.get("info_dropped"),
               "agent": rep.get("agent"), "agentic": rep.get("agentic"), "policy": type(pol).__module__ + ":" + type(pol).__name__, "code": code}
        rows.append(row)
        with (out / f"{tag}_episodes.jsonl").open("a") as f:
            f.write(json.dumps(jsonable(row), default=str) + "\n")
        frames = rep.pop("look_frames", None)
        if frames:
            fd = out / f"{tag}_{lid}_look_frames"
            fd.mkdir(exist_ok=True)
            for ftag, fr in frames.items():
                for cam, png in fr["png"].items():
                    (fd / f"{ftag}_{cam}.png").write_bytes(png)
                (fd / f"{ftag}_state.json").write_text(json.dumps(fr["state"]))
        with gzip.open(out / f"{tag}_{lid}_report.json.gz", "wt") as f:
            json.dump(jsonable(rep), f, default=str)
        with gzip.open(out / f"{tag}_{lid}_actions.json.gz", "wt") as f:
            json.dump({"actions": [{k: np.asarray(x, np.float64).tolist() for k, x in a.items()}
                                   for a in executed]}, f)
        print(f"\n[integ] {tag} {lid}: success={row['success']} score={row['score']:.2f} "
              f"steps={row['steps_used']}/{row['step_lim']} looks={row['n_looks']} wall={row['wall_s']}s "
              f"(sim {row['sim_s']} obs {row['observe_s']} worker {row['worker_s']}) cat={cat} {why[:120]}", flush=True)
    summ = summarize(rows, task, cfg)
    if renderer == "isaac51":
        summ["proxy"] = RENDERED_TAG
        summ["renderer"] = renderer
    (out / f"{tag}_summary.json").write_text(json.dumps(jsonable(summ), indent=1, default=str))
    print(f"\n[integ] SUMMARY {json.dumps(jsonable(summ), default=str)}", flush=True)
    if rc is not None:
        rc.close()
    if renderer == "isaac51":
        try:                                        # Kit with RTX hangs at interpreter exit (a zenity dialog, no display)
            ep.run.close()
        except Exception as e:  # noqa: BLE001
            print(f"[integ] runner close: {type(e).__name__}: {e}", flush=True)
    return summ


def summarize(rows: list[dict], task: str, cfg: RuntimeConfig) -> dict:
    ok = [r for r in rows if "success" in r]
    n = len(ok)
    cats: dict = {}
    for r in ok:
        if r.get("category"):
            cats[r["category"]] = cats.get(r["category"], 0) + 1
    med = (lambda k: round(float(np.median([r[k] for r in ok if r.get(k) is not None])), 1)
           if any(r.get(k) is not None for r in ok) else None)
    return {"proxy": PROXY_TAG, "task": task, "mode": cfg.mode, "verify": cfg.verify,
            "verify_note": None if cfg.verify == "bundle" else "verification OFF (stub): single plan, no twin",
            "n": n, "unstable": sum(1 for r in rows if r.get("unstable")),
            "skipped": sum(1 for r in rows if r.get("skipped")),
            "success": sum(1 for r in ok if r["success"]), "success_rate": round(np.mean([r["success"] for r in ok]), 3) if n else None,
            "score": round(float(np.mean([r["score"] for r in ok])) * 100, 1) if n else None,
            "steps_median": med("steps_used"), "steps_max": max((r["steps_used"] for r in ok), default=None),
            "step_lim": ok[0]["step_lim"] if ok else None, "looks_total": sum(r.get("n_looks", 0) for r in ok),
            "wall_s_median": med("wall_s"), "worker_s_median": med("worker_s"), "categories": cats,
            "failed": [r["layout_id"] for r in ok if not r["success"]], "config": cfg.__dict__}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--layouts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("gt", "perception"), default=None, help="default: the preset's, else gt")
    ap.add_argument("--verify", choices=("off", "bundle"), default=None, help="default: the preset's, else off")
    ap.add_argument("--preset", default=None, help="named RuntimeConfig preset (dojo/runtime/presets.py), e.g. submit_v1")
    ap.add_argument("--config", default="{}", help="extra RuntimeConfig fields as JSON (over the preset)")
    ap.add_argument("--max-minutes", type=float, default=25.0, help="no new layout after this many minutes")
    ap.add_argument("--episode-max-minutes", type=float, default=25.0,
                    help="end an episode after this many wall-clock minutes (0: no cap); the GPU-lock rule")
    ap.add_argument("--policy", default=None, help="module:Class, called as Class(cfg) (default RuntimePolicy)")
    ap.add_argument("--renderer", choices=("isaac60", "isaac51"), default="isaac60",
                    help="isaac60: the proxy (6.0 render service); isaac51: one 5.1 EvalEnv with RTX (isr-lab)")
    a = ap.parse_args(argv)
    from .presets import check_env, resolve
    if a.preset and check_env(a.preset):
        raise SystemExit(f"preset {a.preset}: this environment contradicts it: {check_env(a.preset)}")
    d = resolve(a.preset, json.loads(a.config), a.mode, a.verify)
    d.setdefault("mode", "gt")
    d.setdefault("verify", "off")
    cfg = RuntimeConfig(**d)
    run(a.task, a.layouts, cfg, Path(a.out), a.max_minutes, policy_factory=load_policy_factory(a.policy),
        episode_max_minutes=a.episode_max_minutes or None,
        renderer=a.renderer)
    if a.renderer == "isaac51":                     # results are written; never wait on Kit's own shutdown
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
