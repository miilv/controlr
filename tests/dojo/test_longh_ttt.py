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


# perception's head check of the opponent's believed start places on the proxy replay of B_play_tic_tac_toe_breadth2
# 100004 (stream C, 2026-10-28): after the opponent took opponent_piece0 to the centre, its place reads 0.0 (step 530;
# 164 px); the three still there read 0.83-1.00 (every present piece over 4 episodes: >= 0.70 on 666 checks, never < 0.3)
STEP530 = {"theirs0": (0.0, 164), "theirs1": (0.9955, 220), "theirs2": (0.9417, 223), "theirs3": (0.8318, 220)}
STEP510 = {"theirs0": (None, 0), "theirs1": (None, 0), "theirs2": (0.8947, 19), "theirs3": (0.8063, 222)}


class _Checker:
    def __init__(self, precs):
        self.precs, self.calls = precs, 0

    def check(self, obs, objects):
        self.calls += 1
        return {o.label: {"head": {"prec": self.precs[o.label][0], "n_vis": self.precs[o.label][1]}}
                for o in objects if o.label in self.precs}


class _Prov:
    def __init__(self, precs, adopt=True):
        self.checker = _Checker(precs) if precs is not None else None
        self.adopt, self.proposed = adopt, []
        self.bridge = type("B", (), {"observe": lambda _s: {"images": {}, "state": {}}})()

    def confirm_pose(self, label, pos, quat):
        self.proposed.append((label, np.asarray(pos, float)))
        return {"adopted": self.adopt}


class _Sk:
    def __init__(self, w, prov):
        self.world, self.provider, self.logs, self.held = w, prov, [], {}

    def refresh(self):
        return self.world

    def confirm_moved(self, label, T, why=""):
        r = self.provider.confirm_pose(label, np.asarray(T)[:3, 3], [1, 0, 0, 0])
        if r.get("adopted"):
            self.world.obj(label).pos = np.asarray(T, float)[:3, 3].copy()
        return r


def _into(precs, adopt=True, prev=None):
    from dojo.skills import interactive as I
    w = _board_world()
    if prev:
        prev(w)
    sk = _Sk(w, _Prov(precs, adopt))
    cells = I.board_cells(w.obj("board"))
    mine, theirs = [f"mine{i}" for i in range(5)], [f"theirs{i}" for i in range(4)]
    r = I.opponent_into_belief(sk, "board", cells, 4, mine, theirs, float(w.obj("board").top_z))
    return r, sk, I.board_state(w, "board", cells, mine, theirs)


def test_the_opponent_piece_whose_place_is_empty_moves_onto_the_watched_cell():
    r, sk, occ = _into(STEP530)
    assert r["moved"] == "theirs0" and [p[0] for p in sk.provider.proposed] == ["theirs0"]
    assert occ == {4: ("opp", "theirs0")}                                       # the belief has it on the centre cell
    assert np.allclose(sk.world.obj("theirs0").pos[:2], [0.0, -0.075], atol=1e-6)


def test_no_move_when_the_start_places_are_not_clearly_empty_or_ambiguous():
    r, sk, occ = _into(STEP510)                                                  # the emptied place hidden: abstain
    assert r["moved"] is None and sk.provider.proposed == [] and occ == {}
    both = dict(STEP530, theirs3=(0.1, 200))                                    # two empty places: which one went?
    r, sk, _ = _into(both)
    assert r["moved"] is None and sk.provider.proposed == []
    r, sk, occ = _into(STEP530, adopt=False)                                    # the head view did not prefer the cell
    assert r["moved"] is None and occ == {} and len(sk.provider.proposed) == 1
    r, sk, _ = _into(None)                                                       # G3 / no checker: the world is GT
    assert r["moved"] is None


def test_pieces_already_believed_on_the_board_are_not_candidates():
    def prev(w):
        w.obj("theirs0").pos = np.array([-0.074, -0.149, 0.781])                # believed in cell 0 already
    r, sk, occ = _into(dict(STEP530, theirs0=(0.0, 164)), prev=prev)
    assert r["moved"] is None and sk.provider.checker.calls == 1                # theirs0 is not a candidate any more
