"""LIBERO variant of the agentic perceiver: one fresh agent per capture -> picks.json (PERCEIVER-PROTOCOL.md section 3).

    python -m dojo.perception.agentic.libero --cap DIR --instruction "..." --cell object_task --out picks.json
        [--reply rgbd_reply_01.json] [--model M] [--max-turns 8]

Runs on the box that holds the capture and robot-harness's LIBERO helpers (the policy's machine). Tools, mirroring
tools/perceiver/libero/template/ep_cell_blind.sh (the commands the in-session perceiver ran):
    grid.py full|crops ...                     the vendored grid tool on the workspace images (agentview.png, wrist.png)
    probe u,v [u,v ...]                        depth back-projection: <LIBERO_PY> <HARNESS>/libero_rgbd.py point
    locate '<picks json>'                      the cell's helper (libero_rgbd_v3.py locate | locate-spatial): diagnostics
                                               + overlay.png, the template's `locate` check (0 env steps)
Answer: {"answer": {"picks": {...}, "notes": "..."}} -> picks.json (+ robot_base / eef from the capture reply).
Environment: LIBERO_PY (the helper python), LIBERO_HARNESS (robot-harness harness/ dir), AGENTIC_* as in j1.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from controlr.keyguard import child_env

from .tools_rd import _fit, _png_state

VENDOR = Path(__file__).resolve().parent / "vendor"
HELPER = {"object_task": "locate", "object_swap": "locate", "spatial_task": "locate-spatial", "spatial_swap": "locate-spatial"}


class LiberoTools:
    def __init__(self, ws: Path, cap: Path, cell: str, reply: dict | None, py: str, harness: str, timeout_s: float = 120.0):
        self.ws, self.cap, self.cell, self.reply = Path(ws), Path(cap), cell, reply or {}
        self.py, self.harness, self.timeout_s = py, harness, timeout_s
        self.protected = {"agentview.png", "wrist.png"}

    def _sub(self, argv: list[str]) -> tuple[int, str, str]:
        env = child_env({"OMP_NUM_THREADS": "1"})   # a tool never calls the model: no key, no key file path
        p = subprocess.run(argv, cwd=self.ws, capture_output=True, text=True, timeout=self.timeout_s, env=env)
        return p.returncode, p.stdout or "", p.stderr or ""

    def run(self, cmd: str):
        try:
            argv = shlex.split(cmd)
        except ValueError as e:
            return f"error: cannot parse the command ({e})", []
        while argv and argv[0] in ("python", "python3"):
            argv = argv[1:]
        if not argv:
            return "error: empty command", []
        prog = argv[0].rsplit("/", 1)[-1]
        try:
            if prog == "grid.py":
                return self._grid(argv[1:])
            if prog == "probe":
                return self._probe(argv[1:])
            if prog == "locate":
                return self._locate(argv[1:])
        except subprocess.TimeoutExpired:
            return f"error: {prog} timed out", []
        except Exception as e:
            return f"error: {prog}: {type(e).__name__}: {e}", []
        return f"error: '{prog}' is not an available command. Available: grid.py, probe, locate", []

    def _grid(self, args):
        if len(args) < 3 or args[0] not in ("full", "crops"):
            return "error: grid.py full IMG OUT [--step N] [--scale N] | grid.py crops IMG OUT --box NAME:u0,v0,u1,v1 ...", []
        for a in args[1:3]:
            if "/" in a or ".." in a:
                return f"error: {a!r}: workspace file names only", []
        if args[2] in self.protected:
            return f"error: {args[2]} is an input image; name the output differently", []
        before = _png_state(self.ws)
        rc, out, err = self._sub([sys.executable, str(VENDOR / "libero" / "grid.py")] + args)
        imgs = [(n, _fit((self.ws / n).read_bytes())) for n, mt in sorted(_png_state(self.ws).items())
                if n not in self.protected and before.get(n) != mt]
        return (out.strip() or "(no output)") + (f"\nerror (exit {rc}): {err.strip()[-300:]}" if rc else ""), imgs

    def _probe(self, args):
        uvs = [a for a in args if "," in a and not a.startswith("cam=")]
        cam = next((a.split("=", 1)[1] for a in args if a.startswith("cam=")), "agentview")
        if not uvs:
            return "error: probe u,v [u,v ...]", []
        argv = [self.py, str(Path(self.harness) / "libero_rgbd.py"), "point", "--cap", str(self.cap), "--cam", cam]
        for uv in uvs:
            argv += ["--uv", uv]
        rc, out, err = self._sub(argv)
        if rc:
            return f"error (exit {rc}): {err.strip()[-400:]}", []
        d = json.loads(out)
        fz = d["floor_z"]
        lines = [f"table z {fz}"] + [f"{p['uv']} {p['world']} h {round(p['world'][2] - fz, 4)} spread {p['spread_3x3_m']}"
                                     for p in d["points"]]
        return "\n".join(lines), []

    def _locate(self, args):
        if not args:
            return "error: locate '<picks json>'", []
        picks = json.loads(" ".join(args))
        if self.reply:
            picks["robot_base"] = self.reply.get("robot_base")
            picks["eef"] = ((self.reply.get("eef") or {}).get("single") or {}).get("pos")
        (self.ws / "picks.json").write_text(json.dumps(picks))
        helper = HELPER.get(self.cell, "locate")
        rc, out, err = self._sub([self.py, str(Path(self.harness) / "libero_rgbd_v3.py"), helper, "--cap", str(self.cap),
                                  "--picks", str(self.ws / "picks.json"), "--out", str(self.ws / "world.json"),
                                  "--overlay", str(self.ws / "overlay.png")])
        if rc:
            return f"error (exit {rc}): {err.strip()[-600:]}", []
        txt = summarize_locate(json.loads(out)) if out.strip().startswith("{") else out.strip()[-2000:]
        imgs = [("overlay.png", _fit((self.ws / "overlay.png").read_bytes()))] if (self.ws / "overlay.png").exists() else []
        return txt, imgs


def summarize_locate(d: dict) -> str:
    """The same fields the template's `locate` mode prints."""
    lines = [f"floor {d.get('floor_z')} table_visual {d.get('table_visual_z')} support {(d.get('support') or {}).get('rule')} "
             f"{(d.get('support') or {}).get('gap_mm')}"]
    objs = d.get("objects") or {"target": d.get("target")}
    for n, o in (objs or {}).items():
        o = o or {}
        v = o.get("v3_outline") or {}
        lines.append(f"{n} " + json.dumps({k: o.get(k) for k in ("centre_xy", "r_out", "top_z", "height", "length", "width", "yaw_deg",
                                                                  "yaw_deg_used", "top_face_px", "error", "fallback")})
                     + " | v3 " + json.dumps({k: v.get(k) for k in ("used", "edge_points", "shift_from_v2_mm", "fallback")}))
    r = d.get("region_v3")
    if r:
        lines.append(f"REGION centre {r.get('v3_centre_xy')} {r.get('centre_rule')} half {r.get('v3_half_xy')} {r.get('half_rule')}")
    lines.append(f"cells {d.get('obstacle_cells') or d.get('obstacle_cells_all')}")
    return "\n".join(lines)


def perceive_cell(cap: str, instruction: str, cell: str = "object_task", out: str | None = None, reply: dict | None = None,
                  model: str | None = None, max_turns: int | None = None, chat=None, log_dir: str | None = None) -> dict:
    from .llm import image_part, make_chat
    from .prompt import pin, system_prompt
    from .runner import run_agent

    model = model or os.environ.get("AGENTIC_MODEL", "claude/claude-sonnet-5-5")
    max_turns = int(max_turns or os.environ.get("AGENTIC_MAX_TURNS", 8))
    eb = os.environ.get("AGENTIC_EXTRA_BODY")
    cap = Path(cap)
    ws = Path(log_dir or os.environ.get("AGENTIC_LOG_DIR", str(Path.home() / "agentic-runs"))) / f"{time.strftime('%Y%m%d-%H%M%S')}-libero-{os.getpid()}"
    ws.mkdir(parents=True, exist_ok=True)
    for n in ("agentview.png", "wrist.png"):
        if (cap / n).exists():
            shutil.copy(cap / n, ws / n)
    subprocess.run([sys.executable, str(VENDOR / "libero" / "grid.py"), "full", "agentview.png", "av_grid.png", "--step", "32",
                    "--scale", "2"], cwd=ws, capture_output=True, timeout=60, env=child_env())
    tools = LiberoTools(ws, cap, cell, reply, os.environ.get("LIBERO_PY", sys.executable), os.environ.get("LIBERO_HARNESS", "harness"))
    # ``cell`` only routes the locate helper (HELPER); it is never shown to the model: a cell name is the suite name
    # (PERCEIVER-PROTOCOL §1, robot-harness audit A2 finding 1)
    parts = [{"type": "text", "text": f"LIBERO capture. Instruction: {instruction!r}\n"
                                      f"Workspace: agentview.png, wrist.png, av_grid.png (agentview with the 2x grid, attached first)."}]
    for n in ("av_grid.png", "wrist.png"):
        if (ws / n).exists():
            parts += [{"type": "text", "text": f"{n}:"}, image_part(_fit((ws / n).read_bytes()))]
    t0 = time.time()
    st = run_agent(chat or make_chat(), model, system_prompt("libero"), parts, tools, max_turns=max_turns,
                   extra_body=json.loads(eb) if eb else None, log_dir=ws)
    picks = (st["answer"] or {}).get("picks")
    if isinstance(picks, dict):
        if reply:
            picks["robot_base"] = reply.get("robot_base")
            picks["eef"] = ((reply.get("eef") or {}).get("single") or {}).get("pos")
        if out:
            Path(out).write_text(json.dumps(picks))
    p = pin("libero", model)
    row = {"t": time.time(), "fn": "libero", "ws": str(ws), "model": model, "cell": cell, "ok": isinstance(picks, dict),
           "turns": st["turns"], "calls": st["calls"], "commands": st["commands"], "stop": st["stop"], "tokens": st["tokens"],
           "agent_s": round(time.time() - t0, 2), "system_md5": p["system_md5"], "protocol_md5": p["protocol_md5"]}
    with open(ws.parent / "calls.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return {"picks": picks, "notes": (st["answer"] or {}).get("notes"), "stats": row}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--cell", default="object_task", choices=sorted(HELPER))
    ap.add_argument("--out", required=True)
    ap.add_argument("--reply")
    ap.add_argument("--model")
    ap.add_argument("--max-turns", type=int)
    a = ap.parse_args(argv)
    reply = json.loads(Path(a.reply).read_text()) if a.reply else None
    r = perceive_cell(a.cap, a.instruction, a.cell, a.out, reply, a.model, a.max_turns)
    print(json.dumps(r["stats"]))
    sys.exit(0 if r["picks"] is not None else 2)


if __name__ == "__main__":
    main()
