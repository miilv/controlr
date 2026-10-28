"""dojo.scoreboard.board.average never imputes a missing random cell silently (robot-harness
docs/notes/ILIA-POLICY-LEGALITY-AUDIT-2026-10-28.md, finding 5).

A missing cell counts as 0 and is listed; a missing ``<task>_random`` takes the standard score only with
``random_missing="standard"``, and the output says so.
"""

import math

import pytest

from dojo.scoreboard import board as B

# robot-harness results/ledger.jsonl, the breadth3 round-1 dev mix SB-VERIFICATION-2026-10-27 §3 reconstructed:
# lines 73-97 (ours, controlr 1f7e965) + 20-21 (J1) + 11-12 (basket r1). Standard config only: no random cell.
R1_MIX = {"push_T": 80, "swap_T": 100, "classify_objects": 88, "put_bottles_into_dustbin": 64, "build_tower": 100,
          "deposit_coin": 4, "fasten_screws": 80, "fill_egg_holder": 0, "fill_pen_holder": 8, "fold_clothes": 0,
          "hang_mugs": 34, "imitate_sorting_sequence": 0, "insert_key": 15, "insert_tubes": 76, "make_kong": 0,
          "make_toast": 10, "match_and_pick_from_conveyor": 0, "organize_table": 30, "pick_from_conveyor_by_image": 0,
          "play_Xylophone": 80, "play_stacking_toy": 0, "play_tic_tac_toe": 10, "plug_in_charger": 40,
          "press_by_number": 100, "stack_blocks": 80, "store_laptop_and_headphones": 0, "store_tools_in_toolbox": 30,
          "swap_blocks": 100, "sweep_blocks": 0}


def _cells(scores: dict) -> dict:
    return {t: {"score": float(v)} for t, v in scores.items()}


def _by_weights(per_task: dict) -> float:
    """The Average straight from cells()' weights, a missing cell = 0."""
    return sum(w * float(per_task.get(c, {}).get("score", 0.0)) for c, _, w in B.cells())


def test_a_lone_standard_cell_with_its_random_cell_missing_counts_the_random_half_as_zero():
    per = _cells({"stack_blocks": 100})
    for d in (B.average(per), B.average(per, random_missing="zero")):
        assert d["Average"] == pytest.approx(0.8333, abs=1e-4)
        assert d["Gen-std"] == pytest.approx(100 / 12) and d["Gen-rand"] == 0.0
        assert d["random_missing"] == "zero" and d["imputed_cells"] == []
        assert "stack_blocks_random" in d["missing_cells"] and "stack_blocks" not in d["missing_cells"]
        assert len(d["missing_cells"]) == len(B.cells()) - 1
    assert _by_weights(per) == pytest.approx(0.8333, abs=1e-4)


def test_the_random_cell_counts_only_when_measured_or_explicitly_imputed():
    measured = B.average(_cells({"stack_blocks": 100, "stack_blocks_random": 100}))
    assert measured["Average"] == pytest.approx(1.6667, abs=1e-4)
    assert "stack_blocks_random" not in measured["missing_cells"] and measured["imputed_cells"] == []
    imputed = B.average(_cells({"stack_blocks": 100}), random_missing="standard")
    assert imputed["Average"] == pytest.approx(1.6667, abs=1e-4)
    assert imputed["random_missing"] == "standard" and imputed["imputed_cells"] == ["stack_blocks_random"]
    assert "stack_blocks_random" in imputed["missing_cells"]          # imputed is still not measured
    # a random cell whose standard cell is missing too has nothing to impute from: 0, not listed as imputed
    assert "push_T_random" not in imputed["imputed_cells"] and "push_T_random" in imputed["missing_cells"]


def test_error_mode_refuses_missing_random_cells_and_names_them():
    full_gen = {t: 50 for t in B.DIMENSIONS["Generalization"]}
    full_gen.update({t + "_random": 40 for t in B.DIMENSIONS["Generalization"] if t != "hang_mugs"})
    with pytest.raises(ValueError, match=r"\['hang_mugs_random'\]"):
        B.average(_cells(full_gen), random_missing="error")
    full_gen["hang_mugs_random"] = 40
    d = B.average(_cells(full_gen), random_missing="error")
    assert d["Gen-std"] == 50 and d["Gen-rand"] == 40 and d["random_missing"] == "error"
    assert d["missing_cells"] == [c for c, dim, _ in B.cells() if dim != "Generalization"]
    with pytest.raises(ValueError, match="random_missing must be one of"):
        B.average({}, random_missing="std")


def test_missing_cells_of_every_dimension_are_listed_and_count_as_zero():
    per = _cells({"insert_key": 60, "swap_T": 90})
    per["press_by_number"] = {"successRate": 100.0}                  # no score: missing for key="score"
    d = B.average(per)
    assert d["Average"] == pytest.approx(60 / 40 + 90 / 30)
    assert d["Average"] == pytest.approx(_by_weights(per))
    assert "press_by_number" in d["missing_cells"] and "insert_key" not in d["missing_cells"]
    assert d["missing_cells"] == [c for c, _, _ in B.cells() if c not in ("insert_key", "swap_T")]
    assert B.average(per, key="successRate")["missing_cells"].count("press_by_number") == 0


def test_cells_weights_match_the_dimension_means():
    assert math.isclose(sum(w for _, _, w in B.cells()), 1.0)
    full = _cells({c: (i * 7) % 101 for i, (c, _, _) in enumerate(B.cells())})
    for mode in B.RANDOM_MISSING:
        d = B.average(full, random_missing=mode)
        assert d["Average"] == pytest.approx(_by_weights(full)) and d["missing_cells"] == [] == d["imputed_cells"]


def test_breadth3_round1_dev_mix_reproduces_the_audit():
    """Audit finding 5's probe: r1 mix std-imputed 29.0250, random-zero 27.3250."""
    per = _cells(R1_MIX)
    zero, std = B.average(per), B.average(per, random_missing="standard")
    assert zero["Average"] == pytest.approx(27.3250, abs=1e-4)
    assert zero["Average"] == pytest.approx(_by_weights(per))
    assert std["Average"] == pytest.approx(29.0250, abs=1e-4)
    gen_std_measured = sorted(t + "_random" for t in B.DIMENSIONS["Generalization"] if t in R1_MIX)
    assert sorted(std["imputed_cells"]) == gen_std_measured and len(gen_std_measured) == 7


def _bundle(entries: dict) -> str:
    """A JS-bundle-like text the board's extract() reads: "name":{"task":{"score":..},..} per entry."""
    import json
    return "var x=1;" + ",".join(f'"{n}":' + json.dumps(per, separators=(",", ":")) for n, per in entries.items()) + ";"


def test_board_cli_lists_missing_cells_and_records_the_choice(tmp_path, monkeypatch, capsys):
    full = {c: {"score": 50.0, "successRate": 50.0} for c, _, _ in B.cells()}
    std_only = {c: {"score": 50.0, "successRate": 50.0} for c, _, _ in B.cells() if not c.endswith("_random")}
    f = tmp_path / "bundle.js"
    f.write_text(_bundle({"Full": full, "StdOnly": std_only}))
    assert set(B.extract(f.read_text())) == {"Full", "StdOnly"}
    rows = {}
    for mode in ("zero", "standard"):
        monkeypatch.setattr("sys.argv", ["board", "--bundle", str(f), "--ref", "Full", "--random-missing", mode])
        B.main()
        out = capsys.readouterr().out
        assert f"random_missing={mode}" in out
        rows[mode] = {ln.split(" | ")[0][2:]: ln for ln in out.splitlines() if ln.startswith("| Full |")
                      or ln.startswith("| StdOnly |")}
        assert "- StdOnly: stack_bowls_random," in out and "- Full:" not in out
    assert "**50.00** | 0 | 0 |" in rows["zero"]["Full"]
    assert "**45.00** | 12 | 0 |" in rows["zero"]["StdOnly"]             # Gen 25 instead of 50: Average 45
    assert "**50.00** | 12 | 12 |" in rows["standard"]["StdOnly"]
    monkeypatch.setattr("sys.argv", ["board", "--bundle", str(f), "--random-missing", "error"])
    with pytest.raises(ValueError, match="stack_bowls_random"):
        B.main()
