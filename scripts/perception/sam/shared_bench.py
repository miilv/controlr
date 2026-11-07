"""Zero-call, two-client service benchmark on an existing permitted RGB capture.
Run under a light GPU lock; owns and terminates only its child server.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from dojo.perception.sam_masks import SamClient
from dojo.perception.sam_routing import banks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sock', required=True)
    ap.add_argument('--models', default='sam3,sam3.1_text')
    a = ap.parse_args()
    image = np.array(Image.open(a.image).convert('RGB'))
    started = time.monotonic()
    server = subprocess.Popen([sys.executable, '-u', '-m', 'dojo.perception.sam_masks', 'serve',
                               '--sock', a.sock, '--models', a.models, '--max-hours', '.5'])
    try:
        while not Path(a.sock).exists():
            if server.poll() is not None or time.monotonic() - started > 240:
                raise RuntimeError('server did not start')
            time.sleep(.5)
        c = SamClient(a.sock, loud=True)
        result = {'ready_s': time.monotonic()-started, 'models': a.models, 'cells': {}}

        def request(model, task):
            bank = banks()[task]
            cl = SamClient(a.sock, loud=True)
            try:
                t = time.monotonic()
                masks = cl.text(image, bank['phrases'], bank['api_min'], model=model)
                return {'s': time.monotonic()-t, 'masks': len(masks)}
            finally:
                cl.conn.close()

        for model in a.models.split(','):
            for task in ('pour_by_language', 'classify_objects_by_language', 'fill_egg_holder'):
                request(model, task)  # warmup
                single = [request(model, task) for _ in range(3)]
                with ThreadPoolExecutor(2) as pool:
                    t = time.monotonic()
                    concurrent = list(pool.map(lambda _: request(model, task), range(2)))
                    wall = time.monotonic() - t
                result['cells'][model+':'+task] = {'single': single, 'two_clients': concurrent, 'two_wall_s': wall}
        result['info'] = c.info()
        result['gpu_apps'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                                       '--format=csv,noheader'], text=True)
        result['server_pid'] = server.pid
        c.conn.close()
        Path(a.out).write_text(json.dumps(result, indent=2)+'\n')
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait()
        Path(a.sock).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
