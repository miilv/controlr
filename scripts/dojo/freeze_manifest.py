"""The freeze manifest of our candidate for a joint held-out block (robot-harness D-C19). DEV ONLY: reads files, runs
nothing scored, starts no server, never reads a layout.

    python scripts/dojo/freeze_manifest.py --candidate ~/dojo-integ-c4-state/CANDIDATE.txt [--sha SHA]
        [--models ~/dojo-data/segtrain/models] [--rd-python PY] [--rd-home DIR] [--bench DIR] [--out manifest.json]

Run it on the box that will run the block, from a checkout AT the frozen sha (refused otherwise, and on a dirty tree,
unless --allow-mismatch for a dry run). It writes:
- controlr: the sha, whether origin/main contains it, the tree state;
- the candidate: path, md5, text verbatim;
- the launcher: the exact official_eval.sh line and its resolved form (``--dry-run``: policy, full RuntimeConfig,
  candidate environment, side servers with the md5 of every weight file);
- the parity diff integ_eval vs official path (``candidate_parity.py``; empty is the gate);
- model routes by NAME only (agentic perceiver model, the preset's agentic / held-look modes); never a key or a URL
  (OMNIROUTE_* values are not read);
- versions: python and the perception stack (numpy, torch, ultralytics), Isaac Sim (RoboDojo's python), the RoboDojo
  and robot-harness commits; the perception calibration set (``DOJO_CALIB_SET``);
- the gripper semantics: the strict v2 server supplies ``eef.<arm>.ee_joint_state_official`` (D-C11);
- with ``--route-probe``: one tiny call on the agentic route (``route_probe.py``: ok / rate-limited / error).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "dojo"))


def sh(*cmd, cwd=None, timeout=60) -> str | None:
    try:
        r = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def md5(p: Path) -> str | None:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.is_file() else None


def versions(rd_python: str | None) -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("numpy", "torch", "ultralytics", "opencv-python-headless", "scipy", "pyrender"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    if rd_python:
        out["robodojo_python"] = sh(rd_python, "-c", "import sys; print(sys.version.split()[0])")
        out["isaacsim"] = sh(rd_python, "-c", "import importlib.metadata as m; print(m.version('isaacsim'))")
    return out


def routes(runtime_config: dict, setenv: dict) -> dict:
    """Model routes by name: the agentic perceiver's (AGENTIC_MODEL, else the code default) and the modes that call it."""
    import re
    src = (ROOT / "dojo/perception/agentic/j1.py").read_text()
    m = re.search(r'os\.environ\.get\("AGENTIC_MODEL", "([^"]+)"\)', src)
    env_model = setenv.get("AGENTIC_MODEL") or os.environ.get("AGENTIC_MODEL")
    return {"agentic_model": env_model or (m.group(1) if m else None),
            "agentic_model_source": "AGENTIC_MODEL" if env_model else "code default (dojo/perception/agentic/j1.py)",
            "agentic": runtime_config.get("agentic"), "agentic_max": runtime_config.get("agentic_max"),
            "held_look": runtime_config.get("held_look"), "policy_llm": None,
            "router": "omniroute (names only; the key file is AGENTIC_ENV_FILE / DOJO_AGENT_DOTENV, never recorded)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--sha", default=None, help="the frozen controlr sha (default: this checkout's HEAD)")
    ap.add_argument("--models", default=str(Path.home() / "dojo-data/segtrain/models"))
    ap.add_argument("--rd-python", default=None, help="RoboDojo's python (Isaac Sim version)")
    ap.add_argument("--rd-home", default=os.environ.get("RD_HOME"), help="RoboDojo install (its git commit)")
    ap.add_argument("--bench", default=os.environ.get("BENCH", str(Path.home() / "dojo-official-rh")),
                    help="the robot-harness tree official_eval.sh uses (its COMMIT)")
    ap.add_argument("--allow-mismatch", action="store_true", help="dry run: HEAD != --sha or a dirty tree is noted")
    ap.add_argument("--route-probe", action="store_true",
                    help="pre-flight: one tiny call on the agentic route (route_probe.py), recorded; costs one call")
    ap.add_argument("--packages-ref", default=None,
                    help="pkg_versions.py JSON of a compute4 lane venv: recorded, and a version mismatch is a parity diff")
    ap.add_argument("--packages-python", default=None, help="the official side's venv python (default <repo>/.venv)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import candidate_parity as CP

    cand, models = Path(a.candidate).expanduser(), Path(a.models).expanduser()
    problems = []
    if (ROOT / ".git").exists():
        head = sh("git", "rev-parse", "HEAD", cwd=ROOT)
        sha = sh("git", "rev-parse", "--verify", f"{a.sha or 'HEAD'}^{{commit}}", cwd=ROOT)
        dirty = bool(sh("git", "status", "--porcelain", "--untracked-files=no", cwd=ROOT))
    else:                               # a deploy dir (git archive + .deployed.json, as the boxes run it)
        dep = json.loads((ROOT / ".deployed.json").read_text()) if (ROOT / ".deployed.json").exists() else {}
        head = dep.get("head")
        sha = a.sha or head
        dirty = bool(dep.get("dirty", True))
        if head and sha and not (sha.startswith(head) or head.startswith(sha)):
            problems.append(f"this deploy is {head}, not the frozen {sha}")
    if sha is None:
        problems.append(f"sha {a.sha} is not a commit here")
    elif head and not (head.startswith(sha) or sha.startswith(head)):
        problems.append(f"this checkout is at {head}, not the frozen {sha}")
    if dirty:
        problems.append("the checkout has uncommitted changes")
    if problems and not a.allow_mismatch:
        raise SystemExit("; ".join(problems) + " (run from a clean checkout at the frozen sha)")
    on_main = None
    if sha and sh("git", "rev-parse", "--verify", "origin/main", cwd=ROOT):
        on_main = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"], cwd=ROOT).returncode == 0
    integ = CP.integ_side(CP.candidate_tokens(cand.read_text()), models)
    official = CP.official_side(cand, models, ROOT / "scripts/dojo/official_eval.sh")
    d = CP.diff(integ, official)
    pk = CP.packages_check(a.packages_ref, a.packages_python)
    if pk.get("diff"):
        d["packages"] = pk["diff"]
    full = json.loads(subprocess.run(["bash", str(ROOT / "scripts/dojo/official_eval.sh"), "--candidate", str(cand),
                                      "--dry-run"], capture_output=True, text=True, timeout=300,
                                     env={**os.environ, "DOJO_SEG_MODELS": str(models)}).stdout.strip().splitlines()[-1])
    rh = Path(a.bench).expanduser()
    calib = os.environ.get("DOJO_CALIB_SET", "frozen_d0_100040_49")
    man = {
        "kind": "controlr freeze manifest (D-C19)", "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "dry_run_notes": problems,
        "controlr": {"sha": sha, "head": head, "on_origin_main": on_main, "dirty": dirty},
        "candidate": {"path": str(cand), "md5": md5(cand), "text": cand.read_text()},
        "launcher": {"line": f"scripts/dojo/official_eval.sh --candidate {cand} --task <task> --layouts <ids>",
                     "policy": full["policy"], "preset": full["preset"], "config_arg": full["config_arg"],
                     "setenv": full["setenv"], "driver_args": full["driver_args"], "runtime_config": full["runtime_config"]},
        "side_servers": {k: {"model": v["model"], "env": v["env"],
                             "files": {f: {**x, "sha256": hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()
                                           if Path(x["path"]).is_file() else None} for f, x in v["files"].items()}}
                         for k, v in full["side_servers"].items()},
        "packages": {"official": pk["official"], "lane_ref": pk["ref"], "lane_ref_path": pk["ref_path"],
                     "lane_ref_md5": md5(Path(pk["ref_path"])) if pk["ref_path"] else None, "diff": pk["diff"]},
        "parity": {"zero_diff": not d, "diff": d,
                   "expected": {k: v for k, v in CP.EXPECTED_ENV_ONLY["integ"].items() if k in integ["setenv"]}},
        "routes": routes(full["runtime_config"], full["setenv"]),
        "versions": versions(a.rd_python),
        "robodojo_commit": sh("git", "rev-parse", "HEAD", cwd=Path(a.rd_home) / "RoboDojo") if a.rd_home else None,
        "robot_harness": {"tree": str(rh), "commit": (rh / "COMMIT").read_text().strip() if (rh / "COMMIT").exists() else None},
        "perception": {"calib_set": calib, "bg_folds": full["perception_env"]["DOJO_BG_FOLDS"],
                       "color_calib": full["perception_env"]["DOJO_COLOR_CALIB"],
                       "color_calib_md5": md5(Path(full["perception_env"]["DOJO_COLOR_CALIB"])),
                       "color_calib_sha256": hashlib.sha256(Path(full["perception_env"]["DOJO_COLOR_CALIB"]).read_bytes())
                       .hexdigest() if Path(full["perception_env"]["DOJO_COLOR_CALIB"]).is_file() else None,
                       "calib_set_files_md5": {f.name: md5(f) for f in sorted(
                           (Path(full["perception_env"]["DOJO_BG_FOLDS"]) / calib).glob("*")) if f.is_file()}},
        "grip": "the strict v2 server supplies eef.<arm>.ee_joint_state_official (D-C11); DOJO_GRIP_SOURCE is inert there",
    }
    if a.route_probe:                  # headroom / errors of the perceiver's route, before tranche A
        import route_probe
        man["route_probe"] = route_probe.probe(man["routes"]["agentic_model"])
    text = json.dumps(man, indent=1, sort_keys=True, default=str)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0 if not d else 1


if __name__ == "__main__":
    sys.exit(main())
