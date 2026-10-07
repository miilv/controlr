"""Perception probe for the decision head (live; OPENROUTER_API_KEY): can the model read the image
relation between the gripper fingertips and a VISIBLE object (mock: the red ball, Isaac: the
packet), per camera? Ground truth from the calibrated projections. One request per scene with
two choice questions per camera (left/right, higher/lower in that image).

    python scripts/decisions_perception.py -c configs/sim_reach_dec.yaml -n 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decisions_probe import sample  # noqa: E402
from controlr.config import load_config  # noqa: E402
from controlr.llm.decisions import build_state  # noqa: E402
from controlr.observation.renderers import image_labels, project_points  # noqa: E402
from controlr.robot import make_robot  # noqa: E402


def visible_object(robot) -> tuple[str, np.ndarray]:
    sc = getattr(robot, "_scene", None)
    if sc is not None and getattr(sc, "target", None) is not None:
        return "the red ball", np.asarray(sc.target, float)
    return "the waffle packet", np.asarray(robot.scene_record()["packet"]["pos_mm"], float) / 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-n", type=int, default=24)
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--subject", default="the gripper fingertips")
    ap.add_argument("--vertical", choices=["edge", "physical"], default="edge")
    a = ap.parse_args()
    S = a.subject
    cfg = load_config(a.config, a.sets)
    names = list(cfg.observation.cameras)
    robot = make_robot(cfg)
    items = []
    try:
        for i in range(a.n):
            msgs, _, (cams, _, _) = sample(cfg, i, robot, aim_fn=lambda r: visible_object(r)[1])
            obj, p = visible_object(robot)
            tcp = robot.state().tcp_pos
            truth = {}
            for n in names:
                uv, ok = project_points(cams[n], np.stack([tcp, p]))
                if ok.all():
                    truth[n] = (uv[1][0] - uv[0][0], uv[1][1] - uv[0][1])
            items.append((msgs, obj, truth))
    finally:
        getattr(robot, "close", lambda: None)()
    key = os.environ[cfg.decisions.api_key_env]

    fovea = "fovea" in cfg.observation.renderers
    labels = image_labels(cfg.observation)

    def ask(it):
        msgs, obj, truth0 = it
        q = {}
        truth = {(f"{n} fovea" if fovea else n): v for n, v in truth0.items()}
        for n in truth:
            q[f"lr_{n}"] = {"type": "choice", "instructions": f"Look at the `{n}` image only. Is {obj} to the left or to the right of {S} in that image?",
                            "criteria": {"left": f"{obj} is left of {S} in the `{n}` image",
                                         "right": f"{obj} is right of {S} in the `{n}` image"}}
            q[f"ud_{n}"] = {"type": "choice", "instructions": f"Look at the `{n}` image only. Is {obj} " + ("nearer the top edge or nearer the bottom edge of that image" if a.vertical == "edge" else "higher up or lower down in that image") + f" than {S}?",
                            "criteria": {"higher": f"{obj} is " + ("nearer the top edge of" if a.vertical == "edge" else "higher in") + f" the `{n}` image than {S}",
                                         "lower": f"{obj} is " + ("nearer the bottom edge of" if a.vertical == "edge" else "lower in") + f" the `{n}` image than {S}"}}
        body = {"model": cfg.llm.model, "state": build_state(msgs, cfg.decisions, labels), "questions": q}
        r = httpx.post(f"{cfg.decisions.base_url}/decisions", json=body,
                       headers={"Authorization": f"Bearer {key}"}, timeout=60).json()
        return truth, r.get("answers", {})
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(ask, items))
    print(f"object: {items[0][1]}")
    for n in ([f"{c} fovea" for c in names] if fovea else names):
        lr = [ans[f"lr_{n}"]["choice"] == ("left" if t[n][0] < 0 else "right") for t, ans in res if n in t and abs(t[n][0]) > 8]
        ud = [ans[f"ud_{n}"]["choice"] == ("higher" if t[n][1] < 0 else "lower") for t, ans in res if n in t and abs(t[n][1]) > 8]
        mr = [ans[f"lr_{n}"]["choice"] == "right" for t, ans in res if n in t and t[n][0] > 8]
        mh = [ans[f"ud_{n}"]["choice"] == "higher" for t, ans in res if n in t and t[n][1] < -8]
        print(f"{n:12s} left/right {sum(lr)}/{len(lr)} (truly right: {sum(mr)}/{len(mr)})   "
              f"higher/lower {sum(ud)}/{len(ud)} (truly higher: {sum(mh)}/{len(mh)})")
    out = Path(cfg.log.root) / f"perception_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl"
    out.write_text("".join(json.dumps({"truth": t, "answers": a2}) + "\n" for t, a2 in res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
