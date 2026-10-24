"""Render-on-demand service: a warm Isaac Sim 6.0 process rendering RoboDojo states (PROXY for 5.1).

Runs in Isaac 6.0's python (``scripts/simlab/isaac6.sh``); the physics-only 5.1 episode runner
(``proxy_eval.py``) is the client. Localhost only, HMAC-authenticated ``multiprocessing.connection``:

    {"op": "load_stage", "stage": <stage.usd path>, "cameras": <Runner.camera_spec()>}   -> {"ok", "s"}
    {"op": "render", "gt": <Runner.gt() subset: objects, robot, env_origin>, "subframes": 8}
                                         -> {"ok", "images": {cam: HxWx3 uint8}, "cameras": {...}, "s", "missing"}
    {"op": "ping"} / {"op": "close"}

Env: SIMLAB_RENDER_PORT, SIMLAB_RENDER_AUTHKEY (hex), SIMLAB_RENDER_READY (file written when listening).
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    from dojo.simlab.proxy_render import APP_CONFIG, RENDERER_TAG
    port = int(os.environ["SIMLAB_RENDER_PORT"])
    key = bytes.fromhex(os.environ["SIMLAB_RENDER_AUTHKEY"])
    from isaacsim import SimulationApp
    app = SimulationApp(APP_CONFIG)
    from dojo.simlab.proxy_render import ProxyRenderer
    pr = ProxyRenderer(app)
    with Listener(("127.0.0.1", port), authkey=key) as lst:
        ready = os.environ.get("SIMLAB_RENDER_READY")
        if ready:
            Path(ready).write_text(str(os.getpid()))
        print(f"[render_service] listening on 127.0.0.1:{port}", flush=True)
        stop = False
        while not stop:
            conn = lst.accept()
            while True:
                try:
                    req = conn.recv()
                except EOFError:
                    break
                t0 = time.perf_counter()
                try:
                    op = req.get("op")
                    if op == "load_stage":
                        pr.open(req["stage"], req["cameras"], warmup=int(req.get("warmup", 20)))
                        conn.send({"ok": True, "s": time.perf_counter() - t0, "renderer": RENDERER_TAG})
                    elif op == "render":
                        images, cams, missing = pr.render(req["gt"], subframes=int(req.get("subframes", 8)))
                        # raw bytes, not ndarrays: Isaac 6.0's numpy 2 pickles do not load on RoboDojo's numpy 1.26
                        imgs = {n: {"shape": list(im.shape), "data": im.tobytes()} for n, im in images.items()}
                        conn.send({"ok": True, "images": imgs, "cameras": cams, "missing": missing,
                                   "s": time.perf_counter() - t0, "renderer": RENDERER_TAG})
                    elif op == "ping":
                        conn.send({"ok": True})
                    elif op == "close":
                        conn.send({"ok": True})
                        stop = True
                        break
                    else:
                        conn.send({"ok": False, "error": f"unknown op {op!r}"})
                except Exception as e:  # noqa: BLE001 (report, keep serving)
                    conn.send({"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})
            conn.close()
    pr.close_stage()
    app.close()


if __name__ == "__main__":
    main()
