"""Long-Horizon stream, tic-tac-toe in mode B: the opponent's moves come from the watch read, not the belief."""

from __future__ import annotations

import numpy as np

from dojo.skills.world import AssetInfo, ObjectState, WorldState


def _obj(label, cat, idx, xyz, half, typ="Rigid"):
    info = AssetInfo(key=f"{typ}/{cat}/{idx:05d}", bbox_min=-np.asarray(half, float), bbox_max=np.asarray(half, float))
    return ObjectState(label, typ, cat, idx, np.asarray(xyz, float), np.asarray((1.0, 0, 0, 0), float), info)


def _board_world():
    """A 3 x 3 board (cells 74 mm apart), five pieces of ours near the robot, four of the opponent's across."""
    b = _obj("board", "checkerboard", 0, (0, -0.075, 0.765), (0.11, 0.11, 0.008), typ="Geometry")
    b.asset.bbox_min, b.asset.bbox_max = np.array([-0.11, -0.11, 0.0]), np.array([0.11, 0.11, 0.016])
    b.asset.support = {f"cell/{i}": [(np.array([(-0.074, 0.0, 0.074)[i % 3], (-0.074, 0.0, 0.074)[i // 3], 0.022,
                                                1, 0, 0, 0]), 0.032)] for i in range(9)}
    objs = [b]
    for i, x in enumerate((-0.2, -0.1, 0.0, 0.1, 0.2)):
        objs.append(_obj(f"mine{i}", "chessman", 0, (x, -0.25, 0.765), (0.025, 0.025, 0.005)))
    for i, x in enumerate((-0.15, -0.05, 0.05, 0.15)):
        objs.append(_obj(f"theirs{i}", "chessman", 1, (x, 0.2, 0.765), (0.025, 0.025, 0.005)))
    return WorldState("", "", {o.label: o for o in objs}, {}, {})


def test_a_piece_the_watch_saw_land_counts_although_the_belief_never_moved_it():
    from dojo.skills.interactive import board_cells, board_state, with_watch_read
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
