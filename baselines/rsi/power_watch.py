"""Five-minute GPU telemetry, owned foreground process; no system changes."""
import datetime
import pathlib
import subprocess
import time

root = pathlib.Path('/home/robotics/dojo-data/rsi-repro')
while True:
    result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw,memory.used',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True)
    with (root / 'power.log').open('a') as f:
        f.write(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} '
                f'{result.stdout.strip() if result.returncode == 0 else "query_failed"}\n')
    time.sleep(300)
