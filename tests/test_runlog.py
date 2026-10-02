"""Run directory layout + the stats helpers shared by loop and report."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from controlr.config import Config
from controlr.runlog import (
    RunLog,
    cache_read_share,
    find_run_dirs,
    load_run,
    percentiles,
    read_jsonl,
    summarize_turns,
    to_jsonable,
    token_totals,
)
from controlr.types import EventLevel, SafetyEvent


class _Img:
    def __init__(self, data: bytes, sha: str):
        self.jpeg, self.sha = data, sha


def test_layout_and_append(tmp_path):
    log = RunLog(tmp_path, "my run/1", stamp="20260101T000000Z")
    assert log.path.name == "20260101T000000Z_my_run_1"
    log.write_config(Config())
    log.write_system_prompt("SYS")
    log.write_plan("PLAN")
    assert log.save_image(_Img(b"\xff\xd8abc", "aa")) == "aa"
    log.save_images([_Img(b"\xff\xd8abc", "aa"), _Img(b"\xff\xd8xyz", "bb")])
    assert (log.path / "images" / "aa.jpg").read_bytes() == b"\xff\xd8abc"
    assert len(list((log.path / "images").iterdir())) == 2

    recs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": [{"image_sha": "aa"}]}]
    log.sync_messages(recs)
    log.sync_messages(recs + [{"role": "assistant", "content": "STATUS OK"}])
    log.sync_messages(recs + [{"role": "assistant", "content": "STATUS OK"}])
    assert [m["role"] for m in read_jsonl(log.path / "messages.jsonl")] == ["system", "user", "assistant"]

    log.append_turn({"turn": 0, "x": np.float32(1.5), "ev": SafetyEvent(EventLevel.WARN, "clamp", "m")})
    log.append_turn({"turn": 1, "nan": float("nan")})
    t = read_jsonl(log.path / "turns.jsonl")
    assert t[0]["x"] == 1.5 and t[0]["ev"]["level"] == "warn" and t[1]["nan"] is None

    cfg = yaml.safe_load((log.path / "config.yaml").read_text())
    assert cfg["llm"]["api_key_env"] == "OMNIROUTE_API_KEY"
    assert (log.path / "plan.md").read_text() == "PLAN"


def test_same_second_runs_get_distinct_dirs(tmp_path):
    a = RunLog(tmp_path, "x", stamp="S")
    b = RunLog(tmp_path, "x", stamp="S")
    assert a.path != b.path and b.path.name == "S_x_1"


def test_save_images_disabled(tmp_path):
    log = RunLog(tmp_path, "x", save_images=False)
    log.save_image(_Img(b"j", "cc"))
    assert not (log.path / "images" / "cc.jpg").exists()


def test_truncated_jsonl_tolerated(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "tr')
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_stats():
    tot = token_totals([{"prompt_tokens": 100, "cache_read_tokens": 80}, None,
                        {"prompt_tokens": 100, "cache_read_tokens": 90, "completion_tokens": 5}])
    assert tot["prompt_tokens"] == 200 and tot["calls_without_usage"] == 1
    assert cache_read_share(tot) == pytest.approx(0.85)
    assert cache_read_share({"prompt_tokens": 0}) is None
    p = percentiles([1.0, 2.0, 3.0, None])
    assert p["p50"] == 2.0 and p["n"] == 3
    assert percentiles([])["p50"] is None
    s = summarize_turns([{"llm": {"t_end": 1.0}, "timings": {"cycle": 1.5}, "usage": {"prompt_tokens": 10}}])
    assert s["latency"]["llm_end_s"]["p50"] == 1.0 and s["totals"]["prompt_tokens"] == 10


def test_load_run_with_and_without_summary(tmp_path):
    log = RunLog(tmp_path / "sweep", "a")
    log.write_config(Config())
    log.append_turn({"turn": 0, "llm": {"t_end": 2.0}, "usage": {"prompt_tokens": 5, "cache_read_tokens": 0}})
    s = load_run(log.path)
    assert s["outcome"] == "incomplete" and s["turns"] == 1 and s["model"] == Config().llm.model
    log.write_summary({"outcome": "success", "turns": 1})
    assert load_run(log.path)["outcome"] == "success"
    log2 = RunLog(tmp_path / "sweep", "b")
    log2.write_summary({"outcome": "fail"})
    assert len(find_run_dirs(tmp_path)) == 2
    assert find_run_dirs(log.path) == [log.path]


def test_to_jsonable_roundtrip():
    d = to_jsonable({"a": np.arange(3), "b": (1, 2), "c": b"xx"})
    assert json.loads(json.dumps(d)) == {"a": [0, 1, 2], "b": [1, 2], "c": "<2 bytes>"}
