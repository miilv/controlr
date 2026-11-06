"""Proxy render CLI with bounded RGB and instance-buffer readiness checks.

This is an explicitly labelled Isaac 6.0 training domain, not genuine 5.1 RGB.
Arguments are render_d0_proxy's arguments. No renderer settings are changed.
"""
import sys
from pathlib import Path

N = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(N))
from dojo.simlab.proxy_render import ProxyRenderer  # noqa: E402

_original_render = ProxyRenderer.render


def ready_render(self, gt, subframes=8):
    for attempt in range(20):
        images, cameras, missing = _original_render(self, gt, subframes)
        ready = all(v.shape == (480, 640, 3) for v in images.values())
        if self.inst:
            for name in self.inst:
                ids, _ = self.instance_ids(name)
                ready = ready and ids is not None and ids.shape[:2] == (480, 640)
        if ready:
            print(f'[capture-ready] retries={attempt}', flush=True)
            return images, cameras, missing
        print(f'[capture-empty] attempt={attempt}', flush=True)
        for _ in range(20):
            self.app.update()
    raise RuntimeError('proxy RGB/instance buffers remained empty after 20 attempts')


if __name__ == '__main__':
    ProxyRenderer.render = ready_render
    from dojo.simlab import render_d0_proxy

    render_d0_proxy.main()
