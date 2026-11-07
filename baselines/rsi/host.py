"""Explicit host paths and admission for external RSI measurements (stdlib only)."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

CHECKPOINT_SHA256 = '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'


@dataclass(frozen=True)
class Host:
    root: Path
    state: Path
    code: Path
    rdhome: Path
    checkpoint: Path
    lane: str
    lock: Path
    profile: str
    lane_state: Path | None = None
    boot_stamp: Path | None = None
    stream: str = ''

    def env(self) -> dict[str, str]:
        """Allowlist, including a private HOME; never inherit dotenv/API credentials."""
        return dict(HOME=str(self.root / 'home'), PATH='/usr/bin:/bin',
                    PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                    MKL_NUM_THREADS='4', XDG_CACHE_HOME=str(self.root / 'cache'),
                    CUDA_CACHE_PATH=str(self.root / 'cache/cuda'), TMPDIR=str(self.root / 'tmp'),
                    OMNI_KIT_ACCEPT_EULA='YES', RD_HOME=str(self.rdhome),
                    RD_ROOT=str(self.rdhome / 'RoboDojo'), DOJO_DATA=str(self.state),
                    SAM3_CHECKPOINT_PATH=str(self.checkpoint),
                    PYTHONPATH=f'{self.code}:{self.rdhome}/RoboDojo:{self.rdhome}/RoboDojo/XPolicyLab')

    @property
    def python(self) -> Path:
        return self.rdhome / 'miniconda3/envs/robodojo/bin/python'

    def args(self) -> list[str]:
        result = []
        for key in ('root', 'state', 'code', 'rdhome', 'checkpoint', 'lane', 'lock', 'profile',
                    'lane_state', 'boot_stamp', 'stream'):
            value = getattr(self, key)
            if value is not None and value != '':
                result += ['--' + key.replace('_', '-'), str(value)]
        return result

    def directories(self) -> None:
        for path in (self.state, self.root / 'home', self.root / 'tmp', self.root / 'cache/cuda'):
            path.mkdir(parents=True, exist_ok=True)


def arguments(parser: argparse.ArgumentParser) -> None:
    for key in ('root', 'state', 'code', 'rdhome', 'checkpoint', 'lock'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--lane', required=True, choices=('heavy1', 'heavy2'))
    parser.add_argument('--profile', choices=('desktop', 'camera'), required=True)
    parser.add_argument('--lane-state', type=Path)
    parser.add_argument('--boot-stamp', type=Path)
    parser.add_argument('--stream', default='')


def host_from(args: argparse.Namespace) -> Host:
    host = Host(**{k: getattr(args, k) for k in Host.__dataclass_fields__})
    if host.profile == 'desktop' and host.lane != 'heavy1':
        raise ValueError('desktop permission is one job: heavy1 only')
    if host.profile == 'camera' and (host.lane_state is None or not host.stream):
        raise ValueError('camera profile requires --lane-state and --stream')
    for key in ('root', 'state', 'code', 'rdhome', 'checkpoint', 'lock', 'lane_state', 'boot_stamp'):
        value = getattr(host, key)
        if value is not None and not value.is_absolute():
            raise ValueError(f'--{key.replace("_", "-")} must be absolute')
    return host


def telemetry() -> dict:
    line = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,power.draw',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    if len(line.splitlines()) != 1:
        raise ValueError('exactly one GPU required')
    used, free, power = map(float, line.split(','))
    if not all(math.isfinite(v) and v >= 0 for v in (used, free, power)):
        raise ValueError('invalid GPU telemetry')
    available_kb = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    return dict(used_mb=used, free_mb=free, power_w=power, load1=os.getloadavg()[0], available_gb=available_kb / 1024**2)


def decision(host: Host, sample: dict) -> tuple[bool, str]:
    if host.profile == 'desktop':
        # No sibling RSI job is permitted. Count ALL used VRAM and ALL load1 as
        # foreign: conservative (never subtract guessed desktop/own usage).
        if sample['used_mb'] > 2048:
            return False, 'foreign_vram_conservative_gt_2048'
        if sample['load1'] > 8:
            return False, 'foreign_load1_conservative_gt_8'
        if sample['power_w'] > 300:
            return False, 'power_gt_300'
    else:
        if sample['available_gb'] < 16:
            return False, 'camera_ram_gate'
        if sample['used_mb'] > 16384 or sample['free_mb'] < 16223:
            return False, 'camera_vram_gate'
        if sample['power_w'] > 450:
            return False, 'power_gt_450'
    return True, 'admit'


def validate_scope(host: Host) -> None:
    if host.profile == 'camera':
        if host.stream not in Path('/proc/self/cgroup').read_text():
            raise RuntimeError('camera jobs require the configured stream systemd scope')
        return
    group = Path('/proc/self/cgroup').read_text().strip().split('::')[-1]
    if '.scope' not in group or 'rsi' not in group:
        raise RuntimeError('desktop jobs require an RSI systemd user scope')
    directory = Path('/sys/fs/cgroup') / group.lstrip('/')
    memory = (directory / 'memory.max').read_text().strip()
    quota, period = (directory / 'cpu.max').read_text().split()
    if memory == 'max' or int(memory) > 16 * 1024**3 or quota == 'max' or int(quota) / int(period) > 8:
        raise RuntimeError('desktop scope must have MemoryMax<=16G and CPUQuota<=800%')


def log(host: Host, **entry) -> None:
    entry.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), lane=host.lane)
    with (host.state / 'admission.jsonl').open('a') as stream:
        stream.write(json.dumps(entry, sort_keys=True) + '\n')


def stopped(host: Host) -> bool:
    roots = [host.state] + ([host.lane_state] if host.lane_state else [])
    return any((root / name).exists() for root in roots for name in ('STOP', 'STOP.' + host.lane))


def camera_caps(host: Host) -> bool:
    """Same lane2 counters: scoped foreign GPU apps + running lanes + big sides."""
    state = host.lane_state
    caps = [int(v) for v in (state / 'lane2.caps').read_text().split()]
    if len(caps) != 2 or min(caps) < 1:
        return False
    running = 0
    for path in state.glob('running.*'):
        try:
            os.kill(int(path.read_text()), 0)
            running += 1
        except ProcessLookupError:
            pass
    sides = sum(int(p.read_text()) for p in state.glob('side.*') if p.name != 'side.' + host.lane)
    pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15)
    foreign = 0
    for pid in pids.splitlines():
        try:
            foreign += host.stream not in Path('/proc', str(int(pid)), 'cgroup').read_text()
        except FileNotFoundError:
            continue
    return running + 1 <= min(2, caps[0]) and foreign + running + sides + 1 <= caps[1]


def regular_lock(path: Path, create: bool = False):
    flags = os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0)
    fd = os.open(path, flags, 0o600)
    import stat
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError('lane lock must be a regular file')
    return os.fdopen(fd, 'r+')


@contextmanager
def admission(host: Host, sample_fn=telemetry, sleep=time.sleep):
    """Hold a real lane slot until all owned children finish. Fail closed."""
    validate_scope(host)
    host.directories()
    wait_s = 300 if host.profile == 'desktop' else 30
    startup = host.lane_state / 'wave-start.lock' if host.lane_state else host.state / 'startup.lock'
    with regular_lock(startup, create=True) as start, regular_lock(host.lock) as slot:
        while True:
            if stopped(host):
                raise RuntimeError('STOP file: no new start')
            try:
                fcntl.flock(start, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                sample = sample_fn()
                ok, reason = decision(host, sample)
                if ok and host.profile == 'camera' and not camera_caps(host):
                    ok, reason = False, 'lane2_caps'
                if shutil.disk_usage(host.state).free < 50 * 1024**3:
                    ok, reason = False, 'disk_below_50_GiB'
                if ok and host.boot_stamp:
                    epoch = int(host.boot_stamp.read_text().split()[0]) if host.boot_stamp.exists() else 0
                    if time.time() - epoch < 120:
                        ok, reason = False, 'shared_boot_gap'
                log(host, admitted=ok, reason=reason, sample=sample, own_load_subtracted=0)
                if ok:
                    break
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                log(host, admitted=False, reason=type(error).__name__)
            fcntl.flock(slot, fcntl.LOCK_UN)
            fcntl.flock(start, fcntl.LOCK_UN)
            sleep(wait_s)
        if host.boot_stamp:
            host.boot_stamp.write_text(f'{int(time.time())} {datetime.datetime.now(datetime.timezone.utc).isoformat()} ilia-ab\n')
        running = host.lane_state / ('running.' + host.lane) if host.lane_state else None
        side = host.lane_state / ('side.' + host.lane) if host.lane_state else None
        if running:
            running.write_text(str(os.getpid()))
            side.write_text('1')   # SAM3 is a big side server; conservative for prep too.
        started = time.monotonic()
        def release_startup():
            if host.profile == 'camera' and time.monotonic() - started >= 150:
                fcntl.flock(start, fcntl.LOCK_UN)
        try:
            yield release_startup
        finally:
            if running:
                running.unlink(missing_ok=True)
                side.unlink(missing_ok=True)


def terminate(process: subprocess.Popen, pidfile: Path) -> None:
    """Only this still-live child, recorded before use, can be signalled."""
    if process.poll() is None and int(pidfile.read_text()) == process.pid:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def wait_job(process: subprocess.Popen, timeout: int, release_startup) -> None:
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        release_startup()
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(process.args, timeout)
        time.sleep(1)


def install_signals() -> None:
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
