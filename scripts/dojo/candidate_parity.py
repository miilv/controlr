"""Parity of the candidate on the two paths: integ_eval (as Integration's lane2.sh runs a queue line) vs the official
path (scripts/dojo/official_eval.sh --candidate). Zero diff is the gate before a scored official run (D-C19).

    python scripts/dojo/candidate_parity.py --candidate ~/dojo-integ-c4-state/CANDIDATE.txt [--models DIR] [--json OUT]

Compared: the effective RuntimeConfig (``dojo.runtime.eval.build_config`` vs ``dojo.bridge.driver.build_config``), the
policy class, the perception data (DOJO_BG_FOLDS: lane2.sh runs ~/dojo-data/bg_folds_51_ilia; colour map, catalogue,
calibration set), the candidate's environment (``--setenv``: DOJO_PERCEPTION_CFG, ...), which side-server addresses each
path exports (DOJO_SEG_ADDR, DOJO_GLYPH_ADDR[_DIGIT|_KONG]) and the side servers' model files (md5). The integ side
mirrors lane2.sh's queue-line rules (Integration's script, read here, never run: tokens split on whitespace;
``--seg V`` / ``--glyph M`` / ``--glyph-kong M`` start servers on ``models/<V>``; ``--setenv K=V`` enters the job's
environment; everything else goes to integ_eval / dojo.runtime.eval). The official side is official_eval.sh's own
``--dry-run``. Both paths get the candidate's environment verbatim; ``expected`` notes a variable that acts on one
path only (DOJO_GRIP_SOURCE=v2: the proxy's emulation of what the strict v2 server sends itself).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIDE = {"--seg": ("seg", ("DOJO_SEG_ADDR",), ("best.pt", "classes.json")),
        "--glyph": ("glyph", ("DOJO_GLYPH_ADDR", "DOJO_GLYPH_ADDR_DIGIT"), ("model.pt",)),
        "--glyph-kong": ("glyphk", ("DOJO_GLYPH_ADDR_KONG",), ("model.pt",))}
# candidate environment that acts on one path only (reported with the result; both paths still get it verbatim)
EXPECTED_ENV_ONLY = {"integ": {"DOJO_GRIP_SOURCE": "proxy emulation of strict v2; on the official path inert: the strict "
                                                   "v2 server sends eef.<arm>.ee_joint_state_official itself"}}


def md5(p: Path) -> str | None:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.is_file() else None


def candidate_tokens(text: str) -> list[str]:
    """lane2.sh's split: the non-comment lines, whitespace-separated."""
    return " ".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#")).split()


def lane2_perception_env(environ=None) -> dict:
    """The perception data an integ_eval job of lane2.sh runs with: lane2.sh exports DOJO_BG_FOLDS (LANE_BG_FOLDS, else
    ~/dojo-data/bg_folds_51_ilia: the 5.1 set with the scene backgrounds); integ_eval.sh fills the rest (isaac51)."""
    e = os.environ if environ is None else environ
    home = str(Path.home())
    pd = e.get("DOJO_PERCEPTION_DATA", f"{home}/dojo-perception-data")
    return {"DOJO_CATALOG": e.get("DOJO_CATALOG", f"{pd}/catalog"), "DOJO_X5_DIR": e.get("DOJO_X5_DIR", f"{pd}/catalog/robot/x5"),
            "DOJO_BG_FOLDS": e.get("LANE_BG_FOLDS", f"{home}/dojo-data/bg_folds_51_ilia"),
            "DOJO_COLOR_CALIB": e.get("DOJO_COLOR_CALIB", f"{pd}/catalog/color_calib_51.json"),
            "DOJO_CALIB_SET": e.get("DOJO_CALIB_SET", "frozen_d0_100040_49")}


def integ_side(tokens: list[str], models: Path) -> dict:
    from dojo.runtime import eval as EV
    args, setenv, servers = [], {}, {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in SIDE:
            kind, envs, files = SIDE[t]
            servers[kind] = {"model": tokens[i + 1], "env": list(envs),
                             "md5": {f: md5(models / tokens[i + 1] / f) for f in files}}
            i += 2
        elif t == "--setenv":
            k, v = tokens[i + 1].split("=", 1)
            setenv[k] = v
            i += 2
        elif t in ("--sam", "--repo", "--mv3d"):          # lane2.sh's other job options: not policy inputs here
            i += 2
        else:
            args.append(t)
            i += 1
    opt = {k: args[j + 1] for j, k in enumerate(args) if k.startswith("--") and j + 1 < len(args)}
    cfg = EV.build_config(opt.get("--preset"), opt.get("--config", "{}"), opt.get("--mode"), opt.get("--verify"))
    pol = EV.load_policy_factory(opt.get("--policy"))
    return {"policy": f"{pol.__module__}:{pol.__name__}", "runtime_config": cfg.__dict__, "setenv": setenv,
            "side_servers": servers, "perception_env": lane2_perception_env()}


def official_side(cand: Path, models: Path, script: Path) -> dict:
    env = {**os.environ, "DOJO_SEG_MODELS": str(models)}
    r = subprocess.run(["bash", str(script), "--candidate", str(cand), "--dry-run"], capture_output=True, text=True,
                       env=env, timeout=300)
    if r.returncode != 0:
        raise SystemExit(f"official_eval.sh --dry-run failed: {r.stderr[-800:]}")
    d = json.loads(r.stdout.strip().splitlines()[-1])
    d["side_servers"] = {k: {"model": v["model"], "env": v["env"], "md5": {f: x["md5"] for f, x in v["files"].items()}}
                         for k, v in d["side_servers"].items()}
    return d


def diff(a: dict, b: dict) -> dict:
    out: dict = {}
    for k in sorted(set(a["runtime_config"]) | set(b["runtime_config"])):
        va, vb = a["runtime_config"].get(k, "<absent>"), b["runtime_config"].get(k, "<absent>")
        if json.dumps(va, sort_keys=True, default=str) != json.dumps(vb, sort_keys=True, default=str):
            out.setdefault("runtime_config", {})[k] = [va, vb]
    if a["policy"] != b["policy"]:
        out["policy"] = [a["policy"], b["policy"]]
    ea, eb = a["setenv"], b["setenv"]
    for k in sorted(set(ea) | set(eb)):
        if ea.get(k) != eb.get(k):
            out.setdefault("setenv", {})[k] = [ea.get(k), eb.get(k)]
    pa, pb = a.get("perception_env") or {}, b.get("perception_env") or {}
    for k in sorted(set(pa) | set(pb)):          # the perception data (backgrounds, colour map, calibration set)
        if pa.get(k) != pb.get(k):
            out.setdefault("perception_env", {})[k] = [pa.get(k), pb.get(k)]
    for kind in sorted(set(a["side_servers"]) | set(b["side_servers"])):
        sa, sb = a["side_servers"].get(kind), b["side_servers"].get(kind)
        if sa != sb:
            out.setdefault("side_servers", {})[kind] = [sa, sb]
    for kind, s in a["side_servers"].items():          # a model file missing on this box is no parity
        if any(v is None for v in s["md5"].values()):
            out.setdefault("missing_models", []).append(kind)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--models", default=str(Path.home() / "dojo-data/segtrain/models"))
    ap.add_argument("--official", default=str(ROOT / "scripts/dojo/official_eval.sh"))
    ap.add_argument("--json", default=None, help="write the full report here")
    a = ap.parse_args(argv)
    cand, models = Path(a.candidate).expanduser(), Path(a.models).expanduser()
    integ = integ_side(candidate_tokens(cand.read_text()), models)
    official = official_side(cand, models, Path(a.official))
    d = diff(integ, official)
    expected = {k: {"value": integ["setenv"][k], "why": why} for k, why in EXPECTED_ENV_ONLY["integ"].items()
                if k in integ["setenv"]}
    rep = {"candidate": str(cand), "candidate_md5": md5(cand), "models": str(models), "diff": d, "expected": expected,
           "integ": integ, "official": official, "parity": not d}
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, sort_keys=True, default=str))
    print(json.dumps({"parity": not d, "diff": d, "expected": expected}, indent=1, sort_keys=True, default=str))
    return 0 if not d else 1


if __name__ == "__main__":
    sys.exit(main())
