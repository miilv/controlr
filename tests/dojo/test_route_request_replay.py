"""Replay must refuse incomplete or changed source requests before any paid call."""
import importlib.util
import json
from pathlib import Path

import pytest

from dojo.perception.agentic import j1
from dojo.perception.agentic.llm import image_part
from dojo.perception.agentic.prompt import pin, system_prompt

SPEC = importlib.util.spec_from_file_location("route_request_replay", Path(__file__).resolve().parents[2] /
                                               "scripts/perception/route_request_replay.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_existing_route_config_precedence(monkeypatch):
    monkeypatch.delenv("AGENTIC_MODEL", raising=False)
    assert j1._cfg({})["model"] == "claude/claude-sonnet-5-5"
    monkeypatch.setenv("AGENTIC_MODEL", "cx/gpt-6-sol")
    assert j1._cfg({})["model"] == "cx/gpt-6-sol"
    assert j1._cfg({"model": "cx/other"})["model"] == "cx/other"


def fixture(ws):
    ws.mkdir(parents=True)
    (ws / "cam_head.png").write_bytes(b"original-image")
    model = "claude/claude-sonnet-5-5"
    messages = [{"role": "system", "content": system_prompt("robodojo")}, {"role": "user", "content": [
        {"type": "text", "text": "cam_head.png:"}, image_part(b"original-image")]}]
    body = {"model": model, "messages": messages, "max_tokens": 6000, "stream": True,
            "stream_options": {"include_usage": True}}
    start = {"event": "start", "model": model, "first_user": ["cam_head.png:", "<image>"]}
    reply = {"event": "reply", "turn": 0, "text": '{"answer": {"objects": []}}',
             "request_bytes": len(json.dumps(body).encode())}
    (ws / "transcript.jsonl").write_text(json.dumps(start) + "\n" + json.dumps(reply) + "\n")
    (ws.parent / "calls.jsonl").write_text(json.dumps({"ws": str(ws), "pin": pin("robodojo", model)}) + "\n")
    return messages


def test_rebuild_original_request(tmp_path):
    ws = tmp_path / "call_001" / "workspace"
    expected = fixture(ws)
    rebuilt = list(MODULE.requests(ws))
    assert rebuilt[0][0] == expected
    assert len(rebuilt[0][2]) == 64
    (ws / "cam_head.png").write_bytes(b"changed-image-size")
    with pytest.raises(AssertionError, match="request size mismatch"):
        list(MODULE.requests(ws))


def test_prompt_pin_mismatch_refused(tmp_path):
    ws = tmp_path / "call_001" / "workspace"
    fixture(ws)
    call = json.loads((ws.parent / "calls.jsonl").read_text())
    call["pin"]["system_md5"] = "wrong"
    (ws.parent / "calls.jsonl").write_text(json.dumps(call))
    with pytest.raises(AssertionError, match="prompt changed"):
        list(MODULE.requests(ws))
