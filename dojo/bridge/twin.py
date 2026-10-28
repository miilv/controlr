"""The twin on OUR side of a robot server: a physics-only ``Twin`` service for the instruction-derived scene.

In the proxy bundle the launcher started the twin for the eval task. Behind the official client we never know
the task: the scene is derived from the instruction by the same function the runtime uses
(``dojo.runtime.supervisor.perception_vocabulary``), so the twin follows the instruction exactly like perception
and the skills (one fixed system, no routing by task name).

``TwinProcess.start(scene)`` spawns ``dojo.simlab.twin_service`` in RoboDojo's python (``scripts/simlab/rdpy.sh``:
Isaac 5.1 physics-only, no RTX) as its own process group, waits for its ready file and returns a ``TwinClient``;
``stop()`` closes it and kills only that group. The runtime's ``BundleVerifier`` then loads it from OUR estimate
(``estimate_to_layout`` of perception's record) + the measured proprio (``load_from_estimate`` inside
``verify``) and rolls the candidates (``Twin.verify``).
"""

from __future__ import annotations

import os
import secrets
import signal
import socket
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def twin_scene(instruction: str) -> str | None:
    from dojo.runtime.supervisor import perception_vocabulary
    v = perception_vocabulary(instruction)
    return v.name if v else None


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class TwinProcess:
    def __init__(self, num_envs: int = 8, log_dir: str | Path = ".", rdpy: str | None = None,
                 hold_gpu_lock: bool = False, ready_timeout_s: float = 900.0):
        """``hold_gpu_lock``: the session already holds the GPU lock (one scope for server + client + twin), so the
        twin must not queue on it (``SIMLAB_GPU=0`` skips rdpy's lock; Isaac still uses the GPU)."""
        self.num_envs = num_envs
        self.log_dir = Path(log_dir)
        self.rdpy = rdpy or str(REPO / "scripts" / "simlab" / "rdpy.sh")
        self.hold_gpu_lock = hold_gpu_lock
        self.ready_timeout_s = ready_timeout_s
        self.proc: subprocess.Popen | None = None
        self.client = None
        self.scene: str | None = None
        self.boot_s: float | None = None

    def start(self, scene: str):
        from dojo.simlab.twin_service import TwinClient
        self.log_dir.mkdir(parents=True, exist_ok=True)
        port, key = _free_port(), secrets.token_hex(16)
        ready = self.log_dir / f"twin_{scene}.ready"
        ready.unlink(missing_ok=True)
        env = dict(os.environ, SIMLAB_TWIN_PORT=str(port), SIMLAB_TWIN_AUTHKEY=key, SIMLAB_TWIN_READY=str(ready),
                   DOJO_ILIA="1", PYTHONUNBUFFERED="1")
        if self.hold_gpu_lock:
            env["SIMLAB_GPU"] = "0"
        log = open(self.log_dir / f"twin_service_{scene}.log", "ab")
        t0 = time.time()
        self.proc = subprocess.Popen([self.rdpy, "-u", "-m", "dojo.simlab.twin_service", "--task", scene, "--num-envs",
                                      str(self.num_envs)], cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        log.close()
        while not ready.exists():
            if self.proc.poll() is not None:
                raise RuntimeError(f"twin service exited rc {self.proc.returncode}; see {log.name}")
            if time.time() - t0 > self.ready_timeout_s:
                self.stop()
                raise TimeoutError(f"twin service not ready after {self.ready_timeout_s:.0f} s; see {log.name}")
            time.sleep(1.0)
        self.boot_s = round(time.time() - t0, 1)
        self.scene = scene
        self.client = TwinClient(port=port, authkey=bytes.fromhex(key))
        return self.client

    def stop(self) -> None:
        if self.client is not None:
            self.client.close(stop_service=True)
            self.client = None
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGTERM)      # the group this object started (start_new_session)
                p.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


class BundleTwins:
    """Harness side (a dev driver: ``dojo.runtime.eval``, simlab's ``proxy_eval``): the twin for the scene the POLICY
    asks for (``RuntimePolicy._twin_factory``). integ_eval.sh / proxy_eval.sh --bundle pre-start one for the
    launcher's task (``SIMLAB_TWIN_PORT``); a policy whose instruction names another scene gets its own twin service
    (``TwinProcess``, inside this job's lock and scope), as behind the official client. So the launcher's task never
    decides whether or how the policy verifies, and the policy never learns which task was launched (legality audit
    2026-10-28, P1-4: the policy used to compare the twin's task with its scene and turn verification off on a
    mismatch)."""

    def __init__(self, launched_task: str, num_envs: int, log_dir: Path, spawn=None):
        self.launched, self.num_envs, self.log_dir = launched_task, int(num_envs), Path(log_dir)
        self.spawn = spawn                  # tests: scene -> (client, stoppable); default TwinProcess
        self.procs: dict = {}               # scene -> the process we started
        self.served: list = []              # (scene, "bundle" | "spawned"), in request order

    def client(self, scene: str):
        if scene == self.launched and os.environ.get("SIMLAB_TWIN_PORT"):
            from dojo.simlab.twin_service import TwinClient
            self.served.append((scene, "bundle"))
            return TwinClient()
        if self.spawn is not None:
            c, proc = self.spawn(scene)
        else:
            proc = TwinProcess(self.num_envs, self.log_dir, hold_gpu_lock=True)
            c = proc.start(scene)
        self.procs[scene] = proc
        self.served.append((scene, "spawned"))
        print(f"[twin] for scene {scene!r} started (the bundle's runs {self.launched!r})", flush=True)
        return c

    def stop(self) -> None:
        for proc in self.procs.values():
            try:
                proc.stop()
            except Exception as e:  # noqa: BLE001
                print(f"[twin] stop: {type(e).__name__}: {e}", flush=True)
        self.procs = {}


__all__ = ["BundleTwins", "TwinProcess", "twin_scene"]
