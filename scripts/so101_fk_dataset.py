"""Build few-shot sessions for the FK agent from SO-101 teleop recordings.

    uv run python scripts/so101_fk_dataset.py /tmp/fkdata/gym_objects_demo_2 /tmp/fkdata/ball_demo_4 \
        --labels /tmp/fkdata/labels.json --out runs/fk_dataset

Writes ``<out>/sessions.json`` (one session per detected grasp: key points with MODEL joint
angles = measured + joint offsets, joint motion between key points, gripper, FK TCP, wrist photo),
``<out>/images/`` and ``<out>/sheets/`` (a contact sheet per session to check the key points by
eye), and prints a summary. ``--labels``: JSON ``{"<recording>#<cycle>" | "<recording>": "object"}``.
See docs/FK_AGENT.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controlr.fkagent import dataset as D  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recordings", nargs="+", help="recording dirs (joints.jsonl + frames/ + calib)")
    ap.add_argument("--out", default="runs/fk_dataset")
    ap.add_argument("--labels", default=None, help="JSON {session_id or recording: object name}")
    ap.add_argument("--keep-regrips", action="store_true", help="keep grasps that start while holding")
    ap.add_argument("--long-edge", type=int, default=640, help="px of the exported photos")
    a = ap.parse_args(argv)
    labels = json.loads(Path(a.labels).read_text()) if a.labels else {}
    recs, sessions = {}, []
    for p in a.recordings:
        try:
            rec = D.load(p)
        except (OSError, ValueError, KeyError) as e:
            print(f"skip {p}: {e}")
            continue
        recs[rec.name] = rec
        ss = D.build_sessions(rec, labels=labels)
        print(f"{rec.name}: {rec.t[-1]:.1f} s, {len(rec.t)} samples, {len(rec.frame_t)} frames, "
              f"{len(ss)} grasp(s)")
        sessions += [s for s in ss if a.keep_regrips or not s["regrip"]]
    out = D.export(sessions, recs, a.out, long_edge=a.long_edge)
    print("\n".join(D.summary_lines(sessions)))
    print(f"\n{len(sessions)} sessions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
