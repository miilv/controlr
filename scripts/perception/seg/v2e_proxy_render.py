"""Explicit compute3 proxy-domain rehearsal render; run under the heavy GPU lock.

Two base workers / three focused workers, disjoint roots, four physical CPU cores reserved. Stops only
its own subprocess groups if free VRAM drops below 6 GiB or disk below 10 GiB.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--phase', choices=('full', 'extras'), default='full')
    a = p.parse_args()
    n = Path(__file__).resolve().parents[3]
    d = a.data.resolve()
    (d / 'logs').mkdir(exist_ok=True)
    rows = [tuple(map(int, s.split(','))) for s in subprocess.check_output(
        ['lscpu', '-p=CPU,CORE,SOCKET'], text=True).splitlines() if not s.startswith('#')]
    cores = set(sorted({(s, c) for _, c, s in rows})[:-4])
    cpus = {cpu for cpu, c, s in rows if (s, c) in cores}
    os.sched_setaffinity(0, cpus)
    env = dict(os.environ, SIMLAB_GPU='0', DOJO_DATA=str(d / 'data'), OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    active = set()
    lock = threading.Lock()
    stop = threading.Event()

    def terminate(proc):
        # Every subprocess has its own new session; no colleague process joins it.
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    def guard():
        while not stop.wait(2):
            free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                               '--format=csv,noheader,nounits'], text=True).splitlines()[0])
            disk = os.statvfs(d).f_bavail * os.statvfs(d).f_frsize / 2**30
            with (d / 'resources.jsonl').open('a') as f:
                f.write(json.dumps(dict(t=time.time(), free_vram_mib=free, free_disk_gib=disk)) + '\n')
            if free < 6144 or disk < 10:
                stop.set()
                with lock:
                    for proc in active:
                        terminate(proc)
                return

    def run(cmd, log):
        if stop.is_set():
            raise RuntimeError('resource stop')
        subprocess.run(['who'], check=True)
        subprocess.run(['uptime'], check=True)
        subprocess.run(['nvidia-smi'], check=True)
        with log.open('w') as f:
            proc = subprocess.Popen(cmd, env=env, cwd=n, stdout=f, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            with lock:
                active.add(proc)
            try:
                rc = proc.wait(timeout=3600)
                if rc:
                    raise RuntimeError(f'{log}: exit {rc}')
            finally:
                terminate(proc)
                with lock:
                    active.discard(proc)

    def worker(i):
        root = d / (f'extras{i}' if a.phase == 'extras' else f'render{i}')
        ids = '100030-100039' if i == 0 else '100040-100049'
        if a.phase == 'extras':
            ids = ['100030-100036', '100037-100043', '100044-100049'][i]
        tasks = ['general_pickup', 'hang_mugs_random', 'sort_nesting_dolls_by_size_random',
                 'make_toast_random', 'store_laptop_and_headphones_random', 'pour_liquid_into_cup_random',
                 'arrange_largest_number', 'organize_table', 'classify_objects']
        if a.phase == 'extras':
            tasks = ['general_pickup', 'sort_nesting_dolls_by_size_random', 'store_laptop_and_headphones_random']
        for task in tasks:
            variants = '1,2,3,4,5' if task == 'general_pickup' else '1'
            extra = ['--extra-pool', str(d / 'all_classes_pool.txt'), '--extra-n', '6'] \
                if task == 'general_pickup' else []
            if a.phase == 'extras':
                variants = '11' if task == 'general_pickup' else '2'
                extra = ['--extra-pool', str(d / 'garage_negatives_pool.txt'), '--extra-n', '3'] \
                    if task == 'general_pickup' else []
            run([str(n / 'scripts/simlab/rdpy.sh'), '-m', 'dojo.simlab.seg_render', '--task', task,
                 '--ids', ids, '--variants', variants, '--poses', '4', '--physics-only', '--out', str(root), *extra],
                d / 'logs' / f'states_{i}_{task}.log')
            if not any((root / task).glob('*/stage.usd')):
                continue
            run([str(n / 'scripts/simlab/isaac6.sh'), str(n / 'scripts/perception/seg/proxy_capture.py'), '--root', str(root),
                 '--tasks', task, '--any-dirs', '--instance', '--jpeg', '90'],
                d / 'logs' / f'rgb_{i}_{task}.log')

    watcher = threading.Thread(target=guard, daemon=True)
    watcher.start()
    try:
        with ThreadPoolExecutor(max_workers=3 if a.phase == 'extras' else 2) as pool:
            futures = [pool.submit(worker, i) for i in range(3 if a.phase == 'extras' else 2)]
            for future in futures:
                future.result()
    finally:
        stop.set()
        with lock:
            for proc in active:
                terminate(proc)
        watcher.join()


if __name__ == '__main__':
    main()
