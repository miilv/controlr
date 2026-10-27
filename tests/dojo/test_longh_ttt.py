"""Long-Horizon stream, tic-tac-toe in mode B: the opponent's moves come from the watch read, not the belief."""

from __future__ import annotations

from tests.dojo.test_skills6 import _board_world


def test_a_piece_the_watch_saw_land_counts_although_the_belief_never_moved_it():
    from dojo.skills.interactive import board_cells, board_state, with_watch_read
    import numpy as np
    w = _board_world()
    cells = board_cells(w.obj("board"))
    w.obj("mine2").pos = np.array([0.003, -0.077, 0.781])                          # ours on the centre (believed)
    occ = board_state(w, "board", cells, [f"mine{i}" for i in range(5)], [f"theirs{i}" for i in range(4)])
    assert occ == {4: ("own", "mine2")}                                            # the opponent's move is not in it
    read = {"cells": [{"k": k, "occupant": {4: "own", 0: "opponent"}.get(k)} for k in range(9)]}
    merged = with_watch_read(occ, read)
    assert merged == {4: ("own", "mine2"), 0: ("opp", "watch:0")}
    assert with_watch_read(occ, None) == occ
    # the belief wins where both have a piece
    assert with_watch_read({0: ("opp", "theirs1")}, read)[0] == ("opp", "theirs1")
