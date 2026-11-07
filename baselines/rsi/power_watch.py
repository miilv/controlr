"""Five-minute telemetry in the explicit state directory. Never starts on import."""
import argparse
import datetime
import fcntl
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    args.state.mkdir(parents=True, exist_ok=True)
    lock = (args.state / 'power-watch.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    while True:
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw,memory.used',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True)
        with (args.state / 'power.log').open('a') as f:
            f.write(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} '
                    f'{result.stdout.strip() if result.returncode == 0 else "query_failed"}\n')
        time.sleep(300)


if __name__ == '__main__':
    main()
