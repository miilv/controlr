"""dojo.agent.replay_probe without network: rebuilding a logged episode (fresh nonce, appendix swap, images) and
grading which ids a replayed cell's skill call passes."""

import io
import json

from PIL import Image

from dojo.agent import replay_probe as P

CFG = {"model": "m", "appendix": "agent_v4", "max_cell_steps": 250, "max_cell_contacts": 4, "max_calls": 30,
       "max_tokens": 100, "python_tool": True}


def _log(tmp_path):
    system = "# manual\n" + P.rendered_appendix("agent_v4", CFG)
    buf = io.BytesIO()
    Image.new("RGB", (8, 6), (10, 20, 30)).save(buf, format="JPEG")
    (tmp_path / "images").mkdir()
    sha = "ab" * 32
    (tmp_path / "images" / f"{sha}.jpg").write_bytes(buf.getvalue())
    recs = [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": "RUN 0123456789ab\nINSTRUCTION: x"},
                                         {"type": "image", "image_sha": sha, "label": "cam_head start"}]},
            {"role": "assistant", "content": "```python\nprint(1)\n```"},
            {"role": "user", "content": [{"type": "text", "text": "STDOUT:\n1"}]},
            {"role": "assistant", "content": "```python\nprint(2)\n```"}]
    (tmp_path / "messages.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (tmp_path / "agent_config.json").write_text(json.dumps(CFG))
    return system


def test_rebuild_keeps_the_log_and_swaps_only_the_appendix(tmp_path):
    system = _log(tmp_path)
    tr, cfg = P.rebuild(tmp_path, 3, "feedfacecafe")
    msgs = tr.to_messages()
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]      # up to message 3
    assert msgs[0]["content"] == system and cfg["appendix"] == "agent_v4"
    first = msgs[1]["content"]
    assert first[0]["text"].startswith("RUN feedfacecafe\n") and first[1]["type"] == "image_url"
    tr5, _ = P.rebuild(tmp_path, 3, "feedfacecafe", "agent_v5")
    s5 = tr5.to_messages()[0]["content"]
    assert s5 != system and s5.startswith("# manual\n") and "never drops required ids" in s5
    assert s5 == "# manual\n" + P.rendered_appendix("agent_v5", CFG)


def test_skill_ids_runs_the_cell_against_a_stub_robo():
    ids22 = [f"mahjong_{i}" for i in range(22)]
    assert P.skill_ids("r = robo.skill('declare_kong', tiles=[f'mahjong_{i}' for i in range(22)])\nprint(r)",
                       "declare_kong", "tiles") == ids22
    assert sorted(P.skill_ids("t = ['mahjong_12'] + [f'mahjong_{i}' for i in range(22) if i != 12]\n"
                              "r = robo.skill('declare_kong', tiles=t)", "declare_kong", "tiles")) == sorted(ids22)
    assert P.skill_ids("robo.skill('declare_kong', tiles=[f'mahjong_{i}' for i in range(22)][::-1])",
                       "declare_kong", "tiles") == ids22[::-1]
    assert len(P.skill_ids("robo.skill('declare_kong', tiles=[f'mahjong_{i}' for i in range(22) if i not in (4, 5)])",
                           "declare_kong", "tiles")) == 20
    # no skill call: a re-measure first; a refused cell; another skill first
    assert P.skill_ids("print(robo.refine('mahjong_4', arm='left'))", "declare_kong", "tiles") is None
    assert P.skill_ids("import os\nrobo.skill('declare_kong', tiles=[])", "declare_kong", "tiles") is None
    assert P.skill_ids("robo.skill('put_into', objects=['a'], container='b')\nrobo.skill('declare_kong', tiles=['a'])",
                       "declare_kong", "tiles") is None


def test_agent_default_appendix_keeps_retries_complete():
    from dojo.agent.loop import AgentConfig, system_text
    assert AgentConfig().appendix == "agent_v5"
    assert "never drops required ids" in system_text(AgentConfig())
