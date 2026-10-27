"""Replay probe: a logged AgentLoop episode rebuilt up to one message, the next cell sampled n times per prompt variant.

    python -m dojo.agent.replay_probe --log OUT/agent/<stamp>_epNNN --upto 5 --models cx/gpt-6.1-sol \\
        --n 3 --appendix agent_v4,agent_v5 --skill declare_kong --arg tiles --out runs/<probe>

* The transcript comes from the episode's ``messages.jsonl`` + ``images/<sha>.jpg``. These are the logged system text,
  the user turns with their images, and the assistant replies as stored (tool calls already as fenced cells).
  ``messages[0..upto]`` are kept; ``upto`` is the last user turn before the cell under test.
* Every sample gets a fresh ``RUN`` nonce: omniroute replays byte-identical requests, so without it the original
  reply comes back.
* The model is called the way ``AgentLoop`` calls it: cache markers on Claude routes, the ``python`` tool declared,
  ``max_tokens`` from the episode's ``agent_config.json``.
* A variant swaps the episode's rendered appendix (``agent_config.appendix`` with its placeholders filled) in the
  logged system text for another appendix version. Only that text changes; the manual and skill docs stay as the
  episode had them.
* ``--skill/--arg``: which ids the cell's first ``robo.skill(<skill>, <arg>=...)`` passes. The cell runs in the
  agent's restricted ``Session`` against a stub ``robo`` that records the call and stops there. Each sample's code
  and grade go to ``probe.jsonl``; no robot and no simulator are involved.
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path

from controlr.llm import Transcript, apply_cache_markers, cache_style_for
from controlr.llm.transcript import ImagePart
from dojo.agent.llmio import call_record, make_client
from dojo.agent.loop import PYTHON_TOOL, extract_code, tool_code

PROMPTS = Path(__file__).with_name("prompts")
NONCE_RE = re.compile(r"^RUN [0-9a-f]+")


def rendered_appendix(name: str, cfg: dict) -> str:
    """The appendix as ``loop.system_text`` renders it for that episode's limits."""
    app = (PROMPTS / f"{name}.md").read_text()
    return (app.replace("{max_cell_steps}", str(cfg.get("max_cell_steps", 250)))
            .replace("{max_cell_contacts}", str(cfg.get("max_cell_contacts", 4)))
            .replace("{max_calls}", str(cfg.get("max_calls", 30))))


def rebuild(log_dir: Path, upto: int, nonce: str, appendix: str | None = None) -> tuple[Transcript, dict]:
    """The episode's transcript up to message ``upto`` (inclusive), with a fresh nonce and, optionally, another
    appendix in the system text."""
    recs = [json.loads(line) for line in (log_dir / "messages.jsonl").read_text().splitlines() if line.strip()]
    cfg = json.loads((log_dir / "agent_config.json").read_text())
    sys_rec = recs[0]
    system = sys_rec["content"] if isinstance(sys_rec["content"], str) else "".join(
        p.get("text", "") for p in sys_rec["content"])
    if appendix and appendix != cfg.get("appendix"):
        old = rendered_appendix(cfg["appendix"], cfg)
        if old not in system:
            raise ValueError(f"the logged system text does not end with {cfg['appendix']} as rendered")
        system = system.replace(old, rendered_appendix(appendix, cfg))
    tr = Transcript(system)
    first_user = True
    for rec in recs[1:upto + 1]:
        parts = rec["content"] if isinstance(rec["content"], list) else [{"type": "text", "text": rec["content"]}]
        if rec["role"] == "assistant":
            tr.add_assistant("".join(p.get("text", "") for p in parts if p.get("type") == "text"))
            continue
        out: list = []
        for p in parts:
            if p.get("type") == "image":
                jpeg = (log_dir / "images" / f"{p['image_sha']}.jpg").read_bytes()
                out.append(ImagePart.from_jpeg(jpeg, p.get("label", "")))
            else:
                text = p.get("text", "")
                if first_user:
                    text = NONCE_RE.sub(f"RUN {nonce}", text, count=1)
                    first_user = False
                out.append(text)
        tr.add_user(out)
    return tr, cfg


class _StubResult:
    ok = False
    status = "probe"
    note = "replay probe: nothing ran"
    steps = 0

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return "Result(ok=False, status='probe')"


class _StubRobo:
    """``robo`` for a replayed cell: records the first ``skill(...)`` call and ends the cell there; every other action
    returns a failed result, perception returns nothing. No robot, no simulator, no host access (the cell runs in
    the agent's restricted ``Session``)."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def skill(self, name, **kw):
        from dojo.agent.session import CellCheckpoint
        self.calls.append((str(name), kw))
        raise CellCheckpoint("replay probe: the skill call is recorded")

    def scene(self):
        return {}

    def __getattr__(self, name):
        return lambda *a, **k: _StubResult()


def skill_ids(code: str, skill: str, arg: str) -> list[str] | None:
    """The ids the cell's first ``robo.skill(<skill>, <arg>=...)`` call passes, found by running the cell in the
    agent's ``Session`` against ``_StubRobo`` (any expression the model wrote: comprehensions with filters, slices,
    a list built in an earlier line). None when the cell makes no such call before its first other skill."""
    from dojo.agent.session import EpisodeFinished, Session
    robo = _StubRobo()

    def finish(*a, **k):
        raise EpisodeFinished("finish")
    try:
        Session({"robo": robo, "finish": finish}).run(code or "")
    except BaseException:  # noqa: BLE001  (a refused or broken cell: nothing recorded beyond what ran)
        pass
    for name, kw in robo.calls[:1]:
        if name == skill and isinstance(kw.get(arg), (list, tuple)):
            return [str(v) for v in kw[arg]]
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", required=True, help="an AgentLoop episode log dir (messages.jsonl, images/, agent_config.json)")
    ap.add_argument("--upto", type=int, required=True, help="index of the last message kept (a user turn)")
    ap.add_argument("--models", required=True, help="comma-separated model ids")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--appendix", default="", help="comma-separated appendix versions (default: the episode's)")
    ap.add_argument("--skill", default="")
    ap.add_argument("--arg", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    log_dir, out = Path(a.log), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = [v for v in a.appendix.split(",") if v] or [None]
    client = make_client(timeout_s=600, max_retries=3)
    try:
        with (out / "probe.jsonl").open("a") as f:
            for model in a.models.split(","):
                for app in variants:
                    for k in range(a.n):
                        tr, cfg = rebuild(log_dir, a.upto, uuid.uuid4().hex[:12], app)
                        msgs = apply_cache_markers(tr.to_messages(), cache_style_for(model, "auto"))
                        extra = dict(cfg.get("extra_body") or {})
                        if cfg.get("python_tool", True):
                            extra["tools"] = [PYTHON_TOOL]
                        res = client.complete(model, msgs, max_tokens=int(cfg.get("max_tokens", 16000)), extra_body=extra)
                        code = tool_code(getattr(res, "tool_calls", None) or []) or extract_code(res.text or "")
                        ids = skill_ids(code or "", a.skill, a.arg) if a.skill else None
                        rec = {"model": model, "appendix": app or cfg.get("appendix"), "k": k, "code": code,
                               "text": res.text, "ids": ids, "n_ids": None if ids is None else len(set(ids)),
                               "call": call_record(res, model)}
                        f.write(json.dumps(rec) + "\n")
                        print(f"[probe] {model} {rec['appendix']} #{k}: "
                              f"{'no ' + a.skill + ' call' if ids is None else f'{len(set(ids))} ids'}", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
