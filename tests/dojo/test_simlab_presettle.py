"""Pre-settled records: write-back and rest checks (pure)."""

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
