# Run: git clone --depth 1 https://github.com/msannikov03/robot-harness /tmp/rh-lock; python3 scripts/dojo/isr_lockmock.py /tmp/lockmock/t1
"""Mock: OUR isr_job.sh (coord paths -> temp) holding gpu.lock around robot-harness's lock-aware J2 starter in
ROBOT_START_DRYRUN mode (robot-harness main, harness/launch_lock.py delegated mode (a): the gpu.owner pid is an ancestor)."""
import os, subprocess, sys, time
from pathlib import Path
RH = Path("/tmp/rh-lock"); CR = Path(__file__).resolve().parents[2]
tmp = Path(sys.argv[1]); tmp.mkdir(parents=True, exist_ok=True)
coord, bench, rd = tmp / "coord", tmp / "bench", tmp / "rd"
(coord / "inbox" / "ilia").mkdir(parents=True); (coord / "gpu.lock").touch(); (coord / "gpu.owner").touch()
bench.mkdir(); (bench / "harness").symlink_to(RH / "harness"); (bench / "bin").symlink_to(RH / "bin")
(bench / "COORDINATION.md").write_text("# test log\n")
lay = rd / "RoboDojo" / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"; lay.mkdir(parents=True)
(lay / "stack_blocks_0.json").write_text("{}\n")
(tmp / "env.sh").write_text(f"export BENCH={bench}\nexport RD_HOME={rd}\n")
shim = tmp / "shim"; shim.mkdir()
(shim / "setsid").write_text('#!/usr/bin/env python3\nimport os, sys\ntry:\n    os.setsid()\nexcept OSError:\n    if os.fork():\n        os._exit(0)\n    os.setsid()\nos.execvp(sys.argv[1], sys.argv[1:])\n')
(shim / "setsid").chmod(0o755)
job = (CR / "scripts/dojo/isr_job.sh").read_text().replace("COORD=/home/anywherevla/data2/coord", f"COORD={coord}").replace(
    "CLOG=/home/anywherevla/data2/bench/COORDINATION.md RUN=/home/anywherevla/data2/ilia/run", f"CLOG={tmp}/ours.md RUN={tmp}/run")
(tmp / "isr_job.sh").write_text(job)
env = {k: v for k, v in os.environ.items() if not k.startswith(("ROBOT_GPU_", "ROBOT_START_", "J1_", "COORD_DIR", "HARNESS_ENV"))}
env.update(PATH=f"{shim}:{env['PATH']}", COORD_DIR=str(coord), HARNESS_ENV=str(tmp / "env.sh"), J1_PERCEIVER="tests.fake:perceive",
           PYTHONDONTWRITEBYTECODE="1", ROBOT_START_DRYRUN="2", ROBOT_GPU_SIDE="ilia", ROBOT_GPU_LOCK_HELD="1",
           DOJO_ILIA="1", ISR_SKIP_PREFLIGHT="1", ISR_YIELD_S="0")
starter = [str(bench / "bin/robot-start-j2"), "--task", "stack_blocks", "--layout", "9", "--id", "t-ilia-deleg", "--wait-gpu", "0"]
cmd = ["bash", str(tmp / "isr_job.sh"), "mock_j2", "5", "bash", "-c",
       "out=$(\"$@\"); rc=$?; echo \"$out\"; echo \"starter_rc=$rc\"; for i in $(seq 60); do [ -f " + str(bench) +
       "/run/t-ilia-deleg/launch_done.json ] && break; sleep 0.5; done; echo owner_during:; cat " + str(coord) + "/gpu.owner", "_", *starter]
r = subprocess.run(cmd, cwd=bench, env=env, capture_output=True, text=True, timeout=120)
print("rc", r.returncode); print(r.stdout[-2500:]); print("ERR", r.stderr[-1500:])
print("--- their COORDINATION.md:"); print((bench / "COORDINATION.md").read_text()[-1200:])
print("--- ours:"); print((tmp / "ours.md").read_text())
print("--- owner after:", repr((coord / "gpu.owner").read_text()))
print("--- launchers dir:", sorted(os.listdir(coord / "launchers")) if (coord / "launchers").exists() else None)
