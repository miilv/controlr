"""Log compute6 power every five minutes; drain camera lane two after two readings >700 W."""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def reduce_lanes(state: Path, watts: float, previous_high: bool) -> bool:
    high = watts > 700
    if high and previous_high:
        (state / "STOP.heavy2").touch()
        (state / "lane2.caps").write_text("1 100\n")
    return high


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("state", type=Path)
    args = ap.parse_args()
    high = False
    for _ in range(432):  # bounded at 36 hours; camera booking is ~7.3 hours
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        row = {"time": datetime.now(timezone.utc).isoformat(), "gpu": result.stdout.strip(), "rc": result.returncode}
        if result.returncode == 0:
            watts = float(result.stdout.split(",")[0])
            high = reduce_lanes(args.state, watts, high)
        else:
            high = False
        row["lane2_draining"] = (args.state / "STOP.heavy2").exists()
        with (args.state / "power.jsonl").open("a") as out:
            out.write(json.dumps(row) + "\n")
        time.sleep(300)


if __name__ == "__main__":
    main()
