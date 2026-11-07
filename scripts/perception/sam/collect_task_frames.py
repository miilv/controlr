"""Read-only export of allowed c4b_/c6b_ frame-zero recordings to a tar stream.

Run on a recording host: python collect_task_frames.py ROOT > frames.tar.
Only ids 100000–09 / 100020–24, at most five layouts/task; no source writes.
Uses each episode manifest's actual layout path, including salted layouts.
"""

import io
import hashlib
import gzip
import json
import re
import sys
import tarfile
from pathlib import Path

TASKS = (
    "imitate_sorting_sequence",
    "match_and_pick_from_conveyor",
    "pour_by_language",
    "classify_objects",
    "classify_objects_by_language",
    "deposit_coin",
    "organize_table",
    "put_bottles_into_dustbin",
    "fill_pen_holder",
    "fill_egg_holder",
    "pick_from_conveyor_by_image",
)


def allowed(i):
    return 100000 <= i <= 100009 or 100020 <= i <= 100029


def belt_phase(layout, events, reference):
    """MATCH2 scoring reconstruction; no model/GT access in the runtime.

    Reference is target0 for match, the initially visible basket for by-image.
    Require an observed anchor; do not silently substitute the 51-step prior.
    """
    refs = [
        o
        for arr in layout.get("Rigid", {}).values()
        for o in arr
        if o.get("label") == reference
    ]
    if not refs:
        return None
    x, y = refs[0]["default_pos"][:2]
    reads = [e for e in events if e.get("kind") == "track_read"][:6]
    for e in reads:
        for label, confidence, category, xy in e.get("found", []):
            if category == refs[0]["category"] and abs(xy[1] - y) < 0.05 and xy[0] > x:
                phase = (xy[0] - x) / 0.004 - e["step"]
                if 0 <= phase <= 150:
                    return {
                        "native_steps": phase,
                        "anchor_step": e["step"],
                        "anchor_label": label,
                        "reference_label": reference,
                        "anchor_xy": xy,
                        "speed_m_per_step": 0.004,
                    }
    return None


def main():
    root = Path(sys.argv[1]).expanduser()
    selected = set()
    counts = dict.fromkeys(TASKS, 0)
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as tar:

        def add(name, raw):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))

        for run in sorted(root.iterdir()):
            m = re.fullmatch(r"c[46]b_.*_(\d+)", run.name)
            if not m or not run.is_dir() or not allowed(int(m[1])):
                continue
            lid = int(m[1])
            reports = list(run.glob(f"*_{lid}_report.json.gz"))
            if not reports:
                continue
            task = reports[0].name.split("_perception_")[0]
            if task not in TASKS or (task, lid) in selected or counts[task] >= 5:
                continue
            if lid >= 100025:
                continue  # historical v2b held-out slice
            episodes = list(run.glob("*_episodes.jsonl"))
            if not episodes:
                continue
            records = [
                json.loads(x) for x in episodes[0].read_text().splitlines() if x.strip()
            ]
            row = next(
                (
                    r
                    for r in records
                    if r.get("layout_id") == lid and r.get("task") == task
                ),
                None,
            )
            if row is None:
                continue
            settled = Path(row["layout"])
            if settled.parent.name != task or settled.name not in (
                f"{lid}.json",
                f"{lid}.settled.json",
            ):
                raise ValueError(
                    f"unexpected layout path for allowed episode: {settled}"
                )
            proxy = ".settled." not in settled.name
            phase = "look" if "--phase-look" in sys.argv else "frame0"
            if phase == "look" and (
                task not in ("fill_pen_holder", "organize_table", "deposit_coin")
                or counts[task] >= 2
            ):
                continue
            looks = list(run.glob(f"*_{lid}_look_frames/{phase}_state.json"))
            calls = sorted(run.glob("agentic/call_001/*/calib0.json"))
            if not settled.exists() or not looks:
                continue
            state = looks[0]
            imgs = {
                c: state.parent / f"{phase}_{c}.png"
                for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")
            }
            if not all(p.exists() for p in imgs.values()):
                continue
            base = f"{task}/{lid}"
            belt = None
            if "conveyor" in task:
                report = json.load(gzip.open(reports[0]))
                belt = belt_phase(
                    json.loads(settled.read_text()),
                    report.get("events", []),
                    "target0" if task == "match_and_pick_from_conveyor" else "basket",
                )
                if belt is None:
                    continue  # no safe scoring phase for this recording
            add(f"{base}/settled.json", settled.read_bytes())
            add(f"{base}/state.json", state.read_bytes())
            if calls:
                add(f"{base}/calib.json", calls[0].read_bytes())
            add(
                f"{base}/source.json",
                json.dumps(
                    {
                        "run": str(run),
                        "frame": phase,
                        "task": task,
                        "layout_id": lid,
                        "belt_phase": belt,
                        "gt_source": "belt-reconstructed"
                        if belt
                        else "loaded-raw"
                        if proxy
                        else "loaded-settled",
                        "layout_path": str(settled),
                        "layout_sha256": hashlib.sha256(
                            settled.read_bytes()
                        ).hexdigest(),
                    }
                ).encode(),
            )
            for c, p in imgs.items():
                add(f"{base}/{c}.png", p.read_bytes())
            selected.add((task, lid))
            counts[task] += 1
    print(counts, file=sys.stderr)


if __name__ == "__main__":
    main()
