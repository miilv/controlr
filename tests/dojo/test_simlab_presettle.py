"""Pre-settled records: write-back and rest checks (pure)."""
import pytest

from dojo.simlab import presettle as P


def record():
    def inst(label, x):
        return {"category": "block", "category_idx": 5, "label": label, "default_pos": [x, -0.1, 0.78],
                "default_ori": [1.0, 0.0, 0.0, 0.0], "physics": {"type": "rigid"}}
    return {"Rigid": {"block": [inst("block_0", 0.1), inst("block_1", 0.2)],
                      "pen": [{"category_idx": 0, "type": "cluttered", "clutter_idx": 0, "default_pos": [0.5, 0.3, 0.77],
                               "default_ori": [1.0, 0.0, 0.0, 0.0], "physics": {"type": "rigid"}}]},
            "Table": {"default": "material_0122"}}


def live_rows(lay, dz=0.0, v=0.0):
    rows = []
    for t, cat, inst in P.iter_objects(lay):
        rows.append({"inst_name": f"{cat}_{inst.get('label')}_{len(rows)}", "type": t, "category": cat, "category_idx": inst["category_idx"], "label": inst.get("label"),
                     "clutter": inst.get("type") == "cluttered",
                     "pos": [inst["default_pos"][0], inst["default_pos"][1], inst["default_pos"][2] + dz],
                     "quat": [1.0, 0.0, 0.0, 0.0], "layout_pos": inst["default_pos"], "layout_quat": inst["default_ori"],
                     "lin_vel": [v, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]})
    return rows


def test_write_back_moves_every_movable_object_and_keeps_the_original():
    lay = record()
    out, n = P.write_back(lay, live_rows(lay, dz=-0.005))
    assert n == 3
    assert out["Rigid"]["block"][0]["default_pos"][2] == lay["Rigid"]["block"][0]["default_pos"][2] - 0.005
    assert lay["Rigid"]["block"][0]["default_pos"][2] == 0.78                    # original untouched
    assert out["Table"] == lay["Table"]


def test_rest_and_fall_checks():
    lay = record()
    assert P.at_rest(live_rows(lay), live_rows(lay, v=0.02), True)[0]          # velocity noise is ignored
    ok, why = P.at_rest(live_rows(lay), live_rows(lay, dz=-0.003), True)
    assert not ok and "moving" in why[0]
    ok, why = P.at_rest(live_rows(lay, dz=-0.2), live_rows(lay, dz=-0.2), True)
    assert not ok and "fell" in why[0]


def test_drift_ignores_clutter():
    lay = record()
    rows = live_rows(lay)
    for r in rows:
        if r["clutter"]:
            r["pos"] = [9, 9, 9]
    assert P.drift(rows)[0] == 0.0


def test_layout_files_skip_settled(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    for n in ("100001.json", "100000.json", "100000.settled.json", "settled_index.json"):
        (d / n).write_text("{}")
    assert [p.name for p in P.layout_files("t", root=tmp_path)] == ["100000.json", "100001.json"]


def test_dev_demo_never_touches_eval_paired_demos(tmp_path, monkeypatch):
    import pickle

    from dojo.simlab import paths
    from dojo.simlab import runner as R
    monkeypatch.setattr(paths, "DATA", tmp_path)
    stub = R.dev_demo("imitate_sorting_sequence", paths.SEED_BASE + 3)
    assert stub["_simlab_no_demo"] and stub["joint_path"] == []
    assert sorted(stub["target_label"]) == [f"t{i}" for i in range(5)]
    assert [x.replace("aim", "t") for x in stub["aim_label"]] == stub["target_label"]
    assert R.dev_demo("imitate_sorting_sequence", (paths.SEED_BASE + 3) * 100 + 1) == stub   # shared-seed key
    with pytest.raises(RuntimeError):
        R.dev_demo("imitate_sorting_sequence", 7)                  # an eval seed
    d = tmp_path / "simlab" / "traj" / "imitate_sorting_sequence"
    d.mkdir(parents=True)
    demo = {"joint_path": [[{"support_arm0_joint_state": {"position": [0.0] * 6, "velocity": [0.0] * 6}}]],
            "target_label": ["t1", "t0", "t2", "t3", "t4"], "aim_label": ["aim1", "aim0", "aim2", "aim3", "aim4"],
            "target_place_tag": [None] * 5}
    (d / f"{paths.SEED_BASE + 3}.pkl").write_bytes(pickle.dumps(demo))
    assert R.dev_demo("imitate_sorting_sequence", paths.SEED_BASE + 3) == demo


def test_keep_raw_unmoved_keeps_only_objects_that_did_not_move():
    import math
    lay = record()
    live = live_rows(lay)
    live[0]["pos"] = [live[0]["pos"][0] + 0.0005, live[0]["pos"][1], live[0]["pos"][2]]      # block_0: 0.5 mm
    live[1]["pos"] = [live[1]["pos"][0] + 0.003, live[1]["pos"][1], live[1]["pos"][2]]       # block_1: 3 mm
    h = math.radians(2.0) / 2
    live[2]["quat"] = [math.cos(h), math.sin(h), 0.0, 0.0]                                   # pen: tilted 2 deg
    keep = P.unmoved(live)
    assert keep == {live[0]["inst_name"]}
    out, n = P.write_back(lay, live, keep)
    assert n == 2
    assert out["Rigid"]["block"][0]["default_pos"] == lay["Rigid"]["block"][0]["default_pos"]   # kept at the raw pose
    assert out["Rigid"]["block"][1]["default_pos"][0] == live[1]["pos"][0]
    assert P.write_back(lay, live)[1] == 3                                                     # default: every object


def test_keep_raw_unmoved_selects_only_dropped_ids_without_a_settled_file(tmp_path):
    import json
    d = tmp_path / "t"
    d.mkdir()
    index = {"100000": {"status": "ok"}, "100001": {"status": "dropped: settled record not stable"},
             "100002": {"status": "dropped: settled record not stable"}, "100004": {"status": "dropped: not at rest"},
             "100150": {"status": "dropped: settled record not stable"}}
    for n in ("100000", "100001", "100002", "100003", "100004", "100150"):
        (d / f"{n}.json").write_text("{}")
    for n in ("100000", "100002"):                       # 100002: a settled file next to a dropped entry
        (d / f"{n}.settled.json").write_text("{}")
    (d / "settled_index.json").write_text(json.dumps(index))
    names = lambda fs: [p.stem for p in fs]
    assert names(P.select_files("t", root=tmp_path)) == ["100000", "100001", "100002", "100003", "100004"]  # reserved out
    assert names(P.select_files("t", root=tmp_path, keep_raw_unmoved=True)) == ["100001"]
    assert names(P.select_files("t", ids=[100000, 100002], root=tmp_path, keep_raw_unmoved=True)) == []
