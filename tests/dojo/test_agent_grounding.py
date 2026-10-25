"""dojo.agent Phase 1 pieces without network or RoboDojo: instruction stub, neutral ids, gold, judging."""

import numpy as np

from dojo.agent import dataset, gold, grounding
from dojo.agent.manual import SKILL_DOCS, render_manual
from dojo.agent.scene import assign_ids, scene_text
from dojo.skills import library
from dojo.skills.world import ArmState, AssetInfo, ObjectState, WorldState

LAYOUT = {"Rigid": {"cube": [{"category_idx": 3, "label": "block_0", "default_pos": [0.1, -0.1, 0.78],
                              "default_ori": [1, 0, 0, 0]},
                             {"category_idx": 1, "label": "block_1", "default_pos": [-0.1, -0.1, 0.78],
                              "default_ori": [1, 0, 0, 0]}],
                    "car": [{"category_idx": 2, "label": "cat0_0", "default_pos": [0, 0, 0.78], "default_ori": [1, 0, 0, 0]}]},
          "Geometry": {"basket": [{"category_idx": 0, "label": "basket0", "default_pos": [0, 0.1, 0.8],
                                   "default_ori": [1, 0, 0, 0]}]}}

SRC = '''
class X:
    def gen_instruction(self, env_idx):
        cat_index = self.reward_manager.func_parser.get_label_cat_index(labels=["block_0", "block_1"])
        color = {"1": "blue", "3": "red"}
        lab = self.reward_manager.func_parser.get_label_by_prefix("cat0")[env_idx][0]
        cat = self.reward_manager.func_parser.get_category_by_label(lab, env_idx=env_idx)
        return [f"{color[str(cat_index[0][env_idx])]} on {color[str(cat_index[1][env_idx])]}, {cat} <target>"]
    step_lim = 400
'''


def test_instruction_runs_the_tasks_own_method(tmp_path):
    t = dataset.instruction_for(SRC, LAYOUT, tmp_path, seed=1)
    assert t == "red on blue, car <target>"          # no description file: the slot stays
    assert dataset.step_lim_of("self.step_lim = 700") == 700


def _world():
    def ob(label, cat, x, typ="Rigid"):
        a = AssetInfo(key=f"{typ}/{cat}/00000", bbox_min=np.full(3, -0.02), bbox_max=np.full(3, 0.02))
        return ObjectState(label, typ, cat, 0, np.array([x, -0.1, 0.78]), np.array([1.0, 0, 0, 0]), a)
    objs = {o.label: o for o in [ob("target", "cube", 0.2), ob("red", "cube", -0.2), ob("camera_stand", "camera_stand", 0, "Geometry")]}
    arms = {a: ArmState(np.zeros(6), np.zeros(7), 1.0) for a in ("left", "right")}
    return WorldState("", "pick", objs, arms, {}, step=0, step_lim=200)


def test_scene_hides_checker_labels():
    w = _world()
    text, ids = scene_text(w)
    assert ids.label("cube_0") == "red" and ids.label("cube_1") == "target"   # left to right
    assert "target" not in text and "red" not in text and "camera_stand" not in text
    assert ids.map_args({"objects": ["cube_1", "cube_0"]}) == {"objects": ["target", "red"]}
    assert "200 of 200" in text


def test_manual_documents_every_library_skill():
    assert set(SKILL_DOCS) == set(library.SKILLS)
    m = render_manual()
    assert "{skills}" not in m and "{api}" not in m
    for name in library.SKILLS:
        assert name in m


def _item(task, objects):
    return {"task": task, "objects": {oid: {"label": lab, "category": cat, "idx": idx, "type": typ, "clutter": False}
                                      for oid, (lab, cat, idx, typ) in objects.items()}}


def test_gold_press_needs_a_confirm_after_each_group():
    it = _item("press_by_number", {"SpringButton_0": ("button0", "SpringButton", 2, "Articulation"),
                                   "SpringButton_1": ("button1", "SpringButton", 2, "Articulation"),
                                   "SpringButton_2": ("button2", "SpringButton", 3, "Articulation"),
                                   "number_card_0": ("num0", "number_card", 7, "Geometry"),
                                   "number_card_1": ("num1", "number_card", 4, "Geometry")})
    g = gold.gold_for(it)
    ok = {"skill": "press_sequence", "args": {"presses": [["SpringButton_0", 7], ["SpringButton_2", 1],
                                                          ["SpringButton_1", 4], ["SpringButton_2", 1]]}}
    one_confirm = {"skill": "press_sequence", "args": {"presses": [["SpringButton_0", 7], ["SpringButton_1", 4],
                                                                   ["SpringButton_2", 1]]}}
    assert grounding.judge(ok, g)["full_ok"]
    j = grounding.judge(one_confirm, g)
    assert j["skill_ok"] and not j["args_ok"] and j["bad_args"] == ["presses"]
    assert not grounding.judge({"code": True, "plan": "x"}, g)["decision_ok"]


def test_gold_code_item_and_target_grounding():
    it = _item("general_pickup", {"wooden_toy_0": ("target", "wooden_toy", 0, "Rigid"),
                                  "dice_0": ("dice1", "dice", 0, "Rigid")})
    g = gold.gold_for(it)
    assert g["kind"] == "code" and not g["home"]
    assert grounding.judge({"code": True, "plan": "pick it", "objects": ["wooden_toy_0"]}, g)["full_ok"]
    assert not grounding.judge({"code": True, "plan": "pick dice_0", "objects": ["dice_0"]}, g)["full_ok"]
    assert grounding.judge({"skill": "put_into", "args": {}}, g)["answer_kind"] == "skill"


def test_set_and_has_comparisons():
    assert grounding._cmp({"op": "set", "value": [["place", "a", "b"], ["push", "c", "d"]]},
                          [["push", "c", "d"], ["place", "a", "b"]])
    has = {"op": "has", "value": ["bank"], "allowed": ["bank"]}
    assert grounding._cmp(has, ["bank"]) and not grounding._cmp(has, ["holder", "bank"])
    assert grounding._cmp({"op": "approx", "value": 0.0, "tol": 5, "default": 0.0}, None)


def test_parse_answer_tolerates_fences_and_prose():
    assert grounding.parse_answer('```json\n{"skill": "x", "args": {}}\n```') == {"skill": "x", "args": {}}
    assert grounding.parse_answer('Answer: {"code": true, "plan": "p"} done')["code"] is True
    assert grounding.parse_answer("no json") is None
