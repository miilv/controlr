"""CLI argument parsing, sweep expansion, latency matrix and report — offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from controlr.bench.latency import LatencyBench, Variant, load_bench, markdown_table, run_latency
from controlr.bench.sweep import Sweep, load_sweep, run_sweep
from controlr.cli import build_parser, format_table, main, report_rows
from controlr.config import load_config, parse_override
from controlr.llm.fake import FakeLLM
from controlr.runlog import RunLog

ROOT = Path(__file__).resolve().parents[1]


def test_parse_run():
    a = build_parser().parse_args(["run", "-c", "configs/mock.yaml", "--set", "llm.model=x",
                                   "--set", "seed=3", "--seeds", "0,1,2"])
    assert a.cmd == "run" and a.sets == ["llm.model=x", "seed=3"] and a.seeds == [0, 1, 2]
    a = build_parser().parse_args(["run", "-c", "c.yaml", "--episodes", "4"])
    assert a.episodes == 4 and a.seeds is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "-c", "c.yaml", "--episodes", "2", "--seeds", "1"])


def test_parse_other_commands():
    p = build_parser()
    a = p.parse_args(["bench-cache", "--model", "m", "--turns", "5", "--size", "224", "--frames", "d"])
    assert (a.model, a.turns, a.size, a.frames) == ("m", 5, 224, "d")
    assert p.parse_args(["bench-latency", "-c", "x.yaml", "--dry-run"]).dry_run
    assert p.parse_args(["sweep", "s.yaml"]).sweep == "s.yaml"
    assert p.parse_args(["report", "a", "b"]).paths == ["a", "b"]
    assert p.parse_args(["models", "--filter", "claude"]).filter == "claude"


@pytest.mark.parametrize("name", ["base.yaml", "sim_waffle.yaml", "sim_reach.yaml", "mock.yaml"])
def test_shipped_configs_load(name):
    cfg = load_config(ROOT / "configs" / name, dotenv=None)
    assert cfg.llm.model and cfg.episode.max_turns > 0


def test_base_defaults():
    cfg = load_config(ROOT / "configs" / "base.yaml", dotenv=None)
    assert cfg.llm.model == "claude/claude-sonnet-5-5"
    assert cfg.planner.model == "claude/claude-opus-5-5-xhigh"
    assert cfg.planner.extra_body == {}
    assert cfg.robot.backend == "isaac" and cfg.observation.size == 448
    assert cfg.action.mode == "ee_delta" and cfg.action.pos_unit == "mm" and cfg.episode.max_turns == 40
    assert load_config(ROOT / "configs" / "mock.yaml", dotenv=None).robot.backend == "mock"


def test_sweep_expansion(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({
        "config": "base.yaml", "set": ["planner.enabled=false"], "seeds": [0, 1],
        "grid": {"llm.model": ["claude/a", "b"], "observation.size": [224, 448],
                 "llm.extra_body": [{"reasoning_effort": "low"}]}}))
    sw = load_sweep(p)
    assert sw.name == "s" and sw.config == str((tmp_path / "base.yaml").resolve())
    pts = sw.points()
    assert len(pts) == 2 * 2 * 1 * 2
    assert [pt.seed for pt in pts[:2]] == [0, 1]
    assert pts[0].overrides[0] == "planner.enabled=false"
    vals = {".".join(k): v for k, v in (parse_override(o) for o in pts[0].overrides[1:])}
    assert vals == {"llm.model": "claude/a", "observation.size": 224,
                    "llm.extra_body": {"reasoning_effort": "low"}}
    assert "model=a" in pts[0].label and "seed=0" in pts[0].label


def test_example_sweep_file():
    sw = load_sweep(ROOT / "configs" / "sweeps" / "example.yaml")
    assert len(sw.points()) == 2 * 2 * 2
    for pt in sw.points():
        load_config(sw.config, pt.overrides, dotenv=None)


def test_run_sweep_with_stub_runner(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("name: t\n")
    sw = Sweep(name="t", config=str(base), seeds=[0, 1], grid={"observation.size": [224, 448]})
    seen = []

    class R:
        def __init__(self, cfg):
            self.cfg = cfg

        def summary(self):
            if self.cfg.observation.size == 448 and self.cfg.seed == 1:
                raise RuntimeError("boom")
            return {"outcome": "success", "success": True, "turns": 3, "run_dir": "x",
                    "latency": {"llm_end_s": {"p50": 1.0}}, "totals": {"prompt_tokens": 10}}

    def run_fn(cfg):
        seen.append((cfg.observation.size, cfg.seed, cfg.log.root))
        return R(cfg)

    rows, out = run_sweep(sw, run_fn=run_fn, out_dir=tmp_path / "out", printer=None)
    assert [(s, sd) for s, sd, _ in seen] == [(224, 0), (224, 1), (448, 0), (448, 1)]
    assert all(root == str(tmp_path / "out") for _, _, root in seen)
    assert rows[0]["outcome"] == "success" and rows[0]["llm_end_p50"] == 1.0
    assert rows[3]["outcome"] == "error" and "boom" in rows[3]["error"]
    assert (out / "aggregate.csv").read_text().count("\n") == 5


def test_sweep_dry_run_cli(capsys):
    assert main(["sweep", str(ROOT / "configs" / "sweeps" / "example.yaml"), "--dry-run"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 8


def test_latency_bench_file_and_dry_run(capsys):
    b = load_bench(ROOT / "configs" / "bench" / "latency.yaml")
    assert b.models == ["claude/claude-sonnet-5-5"] and b.sizes == [224, 448, 672] and b.history == [1, 10, 30]
    assert not any("haiku" in m for m in b.models)          # Ilia: Sonnet 5.5 only for now
    assert b.plan_calls() == 1 * 3 * 3 * len(b.variants) * (1 + b.repeats)
    assert main(["bench-latency", "-c", str(ROOT / "configs" / "bench" / "latency.yaml"), "--dry-run"]) == 0
    assert "calls" in capsys.readouterr().out


def test_latency_matrix_with_fake(tmp_path):
    b = LatencyBench(models=["claude/claude-sonnet-5", "gpt-x"], sizes=[64], history=[1, 3],
                     variants=[Variant(), Variant("low", {"reasoning_effort": "low"})], repeats=2)
    llm = FakeLLM(["MOVE ee_delta 1 0 0\nSTATUS OK"])
    rows, out = run_latency(b, llm, out_dir=tmp_path, printer=None)
    assert len(rows) == len(llm.calls) == b.plan_calls() == 2 * 2 * 2 * 3
    h3 = [c for c, r in zip(llm.calls, rows) if r["history"] == 3 and r["model"].startswith("claude")]
    n_img = lambda c: json.dumps(c["messages"]).count('"type": "image_url"')  # noqa: E731
    assert [n_img(c) for c in h3[:3]] == [3, 4, 5]
    assert "cache_control" in json.dumps(h3[0]["messages"])
    assert any(c["extra_body"] == {"reasoning_effort": "low"} for c in llm.calls)
    assert (tmp_path / "calls.csv").exists() and "| claude/claude-sonnet-5 |" in (tmp_path / "table.md").read_text()
    assert markdown_table(rows).count("\n") == 2 + 8


def test_cache_probe_with_fake(tmp_path):
    from controlr.bench.cache_probe import run_cache_probe

    llm = FakeLLM(["MOVE ee_delta 5 0 0\nSTATUS OK"])
    rows = run_cache_probe(llm, "claude/claude-sonnet-5-5", turns=4, size=96,
                           out_dir=tmp_path, printer=None)
    assert [r.n_images for r in rows] == [1, 2, 3, 4]
    assert len(llm.calls) == 4 and (tmp_path / "probe.csv").exists()
    # same prefix each turn: the earlier messages are byte-identical apart from markers
    a, b = llm.calls[1]["messages"], llm.calls[2]["messages"]
    assert json.dumps(a[0]) == json.dumps(b[0])


def test_report(tmp_path, capsys):
    log = RunLog(tmp_path, "r1")
    log.write_summary({"outcome": "success", "success": True, "turns": 4, "model": "m",
                       "latency": {"llm_end_s": {"p50": 1.234, "p90": 2.0}}, "cache_read_share": 0.9,
                       "totals": {"prompt_tokens": 1000, "completion_tokens": 50}})
    log2 = RunLog(tmp_path, "r2")
    log2.append_turn({"turn": 0, "llm": {"t_end": 1.0}, "usage": None})
    rows = report_rows([str(tmp_path / "*")])
    assert {r["outcome"] for r in rows} == {"success", "incomplete"}
    assert "1.23" in format_table(rows)
    assert main(["report", str(tmp_path), "--csv", str(tmp_path / "r.csv")]) == 0
    assert "success" in capsys.readouterr().out and (tmp_path / "r.csv").exists()
    assert main(["report", str(tmp_path / "nothing")]) == 1


def test_run_fake_llm_cli(tmp_path, capsys):
    """`controlr run --fake-llm` drives the full pipeline (mock robot) without network."""
    from controlr.cli import main

    rc = main(["run", "-c", str(ROOT / "configs" / "mock.yaml"), "--fake-llm",
               "--set", f"log.root={tmp_path}", "--set", "episode.max_turns=5"])
    assert rc == 0
    runs = list(tmp_path.iterdir())
    assert len(runs) == 1
    for f in ("config.yaml", "system_prompt.md", "messages.jsonl", "turns.jsonl", "summary.json"):
        assert (runs[0] / f).exists(), f
    assert any((runs[0] / "images").glob("*.jpg"))


# ---------------------------------------------------------------------------
# review fixes: config validation, base.yaml defaults, sweeps, report, prompt
# ---------------------------------------------------------------------------

def test_base_yaml_equals_dataclass_defaults():
    """Review contracts #15: one source for defaults (usage_grace_s, max_stops drifted)."""
    import dataclasses

    from controlr.config import Config
    cfg, d = load_config(ROOT / "configs" / "base.yaml", dotenv=None), Config()
    for sec in ("llm", "planner", "action", "safety", "prompt", "episode", "log"):
        for f in dataclasses.fields(getattr(d, sec)):
            if sec == "llm" and f.name == "base_url":
                continue
            assert getattr(getattr(cfg, sec), f.name) == getattr(getattr(d, sec), f.name), f"{sec}.{f.name}"


@pytest.mark.parametrize("bad", ["action.format=tool", "episode.goal_feedback=alwyas", "action.gripper=widht",
                                 "action.mode=ee", "action.rotation=pitch", "action.pos_unit=inch",
                                 "llm.cache=yes", "llm.cache_ttl=2h", "robot.backend=ur",
                                 "observation.renderers=[raw,gird]", "action.max_chunk=0", "observation.size=0"])
def test_enum_like_values_are_validated(bad):
    """Review contracts #8: typos in values silently fell back."""
    with pytest.raises(ValueError, match="invalid config"):
        load_config(None, [bad], dotenv=None)


def test_sweep_rejects_set_on_a_grid_key_and_uses_full_key_labels(tmp_path):
    from controlr.bench.sweep import check_overrides
    sw = Sweep(name="t", config=None, grid={"llm.model": ["a/x", "b/y"], "planner.model": ["c"]})
    labels = [p.label for p in sw.points()]
    assert labels[0] == "llm.model=x,planner.model=c,seed=0"
    with pytest.raises(ValueError, match="grid key"):
        check_overrides(sw, ["llm.model=z"])
    check_overrides(sw, ["episode.max_turns=3"])


def test_sweep_resume_skips_finished_points(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("name: t\n")
    sw = Sweep(name="t", config=str(base), seeds=[0, 1, 2])
    calls = []

    class R:
        def __init__(self, outcome):
            self.outcome = outcome

        def summary(self):
            return {"outcome": self.outcome, "turns": 1}

    def first(cfg):
        calls.append(cfg.seed)
        return R("error" if cfg.seed == 1 else "max_turns")

    rows, out = run_sweep(sw, run_fn=first, out_dir=tmp_path / "s", printer=None)
    assert [r["outcome"] for r in rows] == ["max_turns", "error", "max_turns"]
    calls.clear()
    rows, _ = run_sweep(sw, run_fn=lambda cfg: (calls.append(cfg.seed), R("success"))[1],
                        out_dir=out, resume=True, printer=None)
    assert calls == [1] and [r["outcome"] for r in rows] == ["max_turns", "success", "max_turns"]


def test_sweep_fake_llm_cli_and_exit_code(tmp_path, capsys):
    """Review contracts #12: sweeps can be smoke-tested offline; failures exit non-zero."""
    sw = tmp_path / "s.yaml"
    sw.write_text(yaml.safe_dump({"name": "t", "config": str(ROOT / "configs" / "mock.yaml"),
                                  "set": [f"log.root={tmp_path}", "episode.max_turns=3"], "seeds": [0],
                                  "grid": {"observation.size": [224, 448]}, "out_root": str(tmp_path)}))
    assert main(["sweep", str(sw), "--fake-llm"]) == 0
    agg = next(tmp_path.glob("*_sweep_t")) / "aggregate.csv"
    text = agg.read_text()
    assert text.count("\n") == 3 and "fake" in text
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"name": "b", "config": str(ROOT / "configs" / "mock.yaml"), "seeds": [0],
                                   "set": ["task.name=fly"], "out_root": str(tmp_path)}))
    assert main(["sweep", str(bad), "--fake-llm"]) == 1


def test_report_skips_fake_runs_unless_asked(tmp_path, capsys):
    from controlr.loop import run_episode
    from controlr.robot.mock import MockRobot

    cfg = load_config(ROOT / "configs" / "mock.yaml", [f"log.root={tmp_path}", "episode.max_turns=1"], dotenv=None)
    run_episode(cfg, robot=MockRobot({"width": 160, "height": 120}), llm=FakeLLM(["STATUS FAIL"]))
    assert report_rows([str(tmp_path)]) == []
    rows = report_rows([str(tmp_path)], include_fake=True)
    assert len(rows) == 1 and rows[0]["llm"] == "fake" and rows[0]["task"] == "reach" and rows[0]["seed"] == 0


def test_prompt_command(tmp_path, capsys):
    assert main(["prompt", "-c", str(ROOT / "configs" / "mock.yaml")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Operating manual") and "px per 100 mm" in out
    assert main(["prompt", "-c", str(ROOT / "configs" / "mock.yaml"), "--planner"]) == 0
    assert "PLANNING CALL" in capsys.readouterr().out


def test_yaml_off_means_off_for_box_collision():
    cfg = load_config(None, ["safety.box_collision=off"], dotenv=None)
    assert cfg.safety.box_collision == "off"
    with pytest.raises(ValueError, match="box_collision"):
        load_config(None, ["safety.box_collision=maybe"], dotenv=None)
    with pytest.raises(ValueError, match="feedback.level"):
        load_config(None, ["feedback.level=minimal"], dotenv=None)


def test_effort_none_on_claude_routes_is_rejected():
    import pytest

    from controlr.config import load_config

    with pytest.raises(ValueError, match="use 'low'"):
        load_config(None, ["llm.model=claude/claude-sonnet-5-5", "llm.extra_body={reasoning_effort: none}"], dotenv=None)
    cfg = load_config(None, ["llm.model=cx/gpt-6-luna", "llm.extra_body={reasoning_effort: none}"], dotenv=None)
    assert cfg.llm.extra_body["reasoning_effort"] == "none"
    load_config(None, ["llm.model=claude/claude-sonnet-5-5", "llm.extra_body={reasoning_effort: low}"], dotenv=None)
