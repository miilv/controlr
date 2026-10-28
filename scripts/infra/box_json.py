"""~/dojo-box.json writer (scripts/infra/box_steps.sh box_json): what agents read to pick a GPU box.

Runs on the box with the system python3 (stdlib only). Inputs come as BOXJ_* environment variables from
box_steps.sh; smoke results are read from ~/dojo-bootstrap/state/<step>.{rc,result}. Schema "dojo-box/1"."""

import json
import os
import socket
import subprocess
import time
from pathlib import Path

e = os.environ


def sh(c: str) -> str:
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def scope(v: str) -> dict:
    return dict(x.split("=", 1) for x in v.split() if "=" in x)


gpus = []
for line in sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits").splitlines():
    p = [x.strip() for x in line.split(",")]
    if len(p) == 3:
        try:
            gpus.append({"name": p[0], "driver": p[1], "vram_mib": int(float(p[2]))})
        except ValueError:
            pass
mem = {}
for line in open("/proc/meminfo"):
    k, v = line.split(":", 1)
    mem[k] = int(v.split()[0]) // 1048576
st = Path(e["BOXJ_BOX"]) / "state"


def step(s: str) -> dict:
    rc, res = st / f"{s}.rc", st / f"{s}.result"
    return {"ok": (rc.read_text().strip() == "0") if rc.exists() else None,
            "result": res.read_text().strip() if res.exists() else None}


manifests = {}                                          # box_steps.sh manifest DIR NAME: data trees, md5-checked
for f in sorted(st.glob("manifest_*.json")):
    try:
        manifests[f.stem[len("manifest_"):]] = json.loads(f.read_text())
    except (OSError, ValueError):
        pass
smokes = {s: step(s) for s in ("smoke_twin", "smoke_render", "smoke_dev_eval", "smoke_proxy_eval", "smoke_perception",
                               "smoke_bridge")}


def ok(s: str) -> bool:
    return smokes[s]["ok"] is True and "skipped" not in (smokes[s]["result"] or "")


role = e.get("BOXJ_ROLE", "")
h = Path.home()
caps = {
    "isaac51_physics": ok("smoke_twin"),                     # Twin / RoboDojoRunner(physics_only=True) / G3
    "isaac51_render": role == "full" and ok("smoke_render"),  # genuine 5.1 frames
    "official_dev_eval": role == "full" and ok("smoke_dev_eval"),
    "bridge_strict": role == "full" and ok("smoke_render"),  # robot-harness's strict server behind the official client
    "bridge_fake": ok("smoke_bridge"),
    "isaac60_render": ok("smoke_proxy_eval"),                # proxy_eval.sh: 5.1 physics + the 6.0 render service
    "perception_egl": ok("smoke_perception"),
    "systemd_scopes": e.get("BOXJ_SCOPES") == "1",
    "llm_env": (h / "controlr" / ".env").exists(),           # OMNIROUTE_* for the agent stream (presence only)
}
use = {"full": ["official dev_eval / strict bridge episodes", "genuine 5.1 frames", "twin", "G3", "perception"],
       "physics": ["twin", "G3 (heavy + light lane)", "proxy_eval with the Isaac 6.0 render", "perception"]}.get(role, [])
disk = sh(f"df -BG --output=avail {e.get('BOXJ_ROOT', str(h))} | tail -1").rstrip("G").strip()
out = {
    "schema": "dojo-box/1", "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "host": socket.gethostname(), "ssh_alias": e.get("BOXJ_ALIAS") or None, "user": e.get("USER") or sh("id -un"),
    "role": role, "role_reason": e.get("BOXJ_REASON") or None, "use_for": use,
    "gpus": gpus,
    "cpu": {"threads": os.cpu_count(), "model": sh("lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1")},
    "ram_gb": {"total": mem.get("MemTotal"), "available": mem.get("MemAvailable")},
    "disk_free_gb": int(disk) if disk.isdigit() else None,
    "os": sh(". /etc/os-release && echo $PRETTY_NAME"),
    "capabilities": caps, "smokes": smokes, "manifests": manifests,
    "paths": {"env_file": str(h / "dojo-env.sh"), "rd_home": e.get("BOXJ_RD"), "controlr": e.get("BOXJ_CODE"),
              "dojo_data": e.get("BOXJ_DATA"), "perception_data": e.get("BOXJ_PERC"),
              "isaac_sim_6": e.get("BOXJ_I6") if role == "physics" else None},
    "locks": {"heavy": str(h / "dojo-gpu.lock"), "light": [str(h / x) for x in e.get("BOXJ_LOCKS", "").split()],
              "rule": "one heavy job per machine, <= 30 min per hold, 90 s yield; light = physics-only, one slot each"},
    "scopes": {"bundle": scope(e.get("BOXJ_SB", "")), "proxy": scope(e.get("BOXJ_SP", "")),
               "batch": scope(e.get("BOXJ_SBA", "")), "g3": scope(e.get("BOXJ_SG", "")),
               "light": scope(e.get("BOXJ_SL", "")), "min_mem_available_gb": int(e.get("BOXJ_MIN", "25"))},
    "pins": {"controlr": e.get("BOXJ_SHA"), "robodojo": e.get("BOXJ_RDC"), "hf_assets": e.get("BOXJ_HF"),
             "assets": e.get("BOXJ_ASSETS"), "isaac_sim_6": e.get("BOXJ_I6V") if role == "physics" else None},
}
print(json.dumps(out, indent=1))
