"""dojo.runtime.belief: a released object and the static fixtures (Geometry) under it. Before, ``KinematicBelief._drop``
ignored fixtures, so an object let go on one fell through it to the table in the belief (integration-2's breadth run:
every nut 53 mm under its bolt's top, every tic-tac-toe piece 17 mm under the board, every hung mug on the table).
CPU only, synthetic meshes."""

import math

import numpy as np

from dojo.runtime.belief import KinematicBelief, fixture_rest
from dojo.skills.kinematics import ARMS, TABLE_Z, X5Kin
from dojo.skills.world import AssetInfo, ObjectState


def _merge(*meshes):
    V, F, n = [], [], 0
    for v, f in meshes:
        V.append(np.asarray(v, float))
        F.append(np.asarray(f, int) + n)
        n += len(v)
    return np.vstack(V), np.vstack(F)


def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    F = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
         [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return V, np.array(F)


def _tube(r_in, r_out, z0, z1, n=48):
    """An annulus (r_in > 0) or a solid cylinder (r_in == 0), axis z."""
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = lambda r, z: np.c_[r * np.cos(a), r * np.sin(a), np.full(n, z)]   # noqa: E731
    if r_in <= 0:
        V = np.vstack([ring(r_out, z0), ring(r_out, z1), [[0, 0, z0], [0, 0, z1]]])
        F = []
        for i in range(n):
            j = (i + 1) % n
            F += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
        return V, np.array(F)
    V = np.vstack([ring(r_out, z0), ring(r_out, z1), ring(r_in, z0), ring(r_in, z1)])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [2 * n + i, 3 * n + j, 2 * n + j], [2 * n + i, 3 * n + i, 3 * n + j],
              [n + i, n + j, 3 * n + j], [n + i, 3 * n + j, 3 * n + i], [i, 2 * n + j, j], [i, 2 * n + i, 2 * n + j]]
    return V, np.array(F)


def _obj(label, typ, mesh, pos, quat=(1, 0, 0, 0)):
    V = np.asarray(mesh[0], float)
    info = AssetInfo(key=f"{typ}/{label}/00000", bbox_min=V.min(0), bbox_max=V.max(0), mesh=mesh)
    return ObjectState(label, typ, label, 0, np.asarray(pos, float), np.asarray(quat, float), info)


def _belief(objs):
    st = {f"{a}_arm_joint_state": np.zeros(6) for a in ARMS}
    st.update({f"{a}_ee_joint_state": [1.0] for a in ARMS})
    return KinematicBelief({o.label: o for o in objs}, X5Kin(), st)


BOLT = _merge(_box([-0.03, -0.03, 0.0], [0.03, 0.03, 0.008]), _tube(0.0, 0.01176, 0.008, 0.050))
NUT = _tube(0.01015, 0.018, 0.0, 0.021)
BOARD = _box([-0.15, -0.15, 0.0], [0.15, 0.15, 0.017])
PIECE = _tube(0.0, 0.016, 0.0, 0.012)
PEG = _merge(_box([-0.04, -0.04, 0.0], [0.04, 0.04, 0.0103]), _tube(0.0, 0.00483, 0.0103, 0.063))
RING = _tube(0.0053, 0.02, 0.0, 0.010)
# an open box: floor 5 mm, walls 4 mm thick, 80 mm high
BIN = _merge(_box([-0.06, -0.06, 0.0], [0.06, 0.06, 0.005]), _box([-0.06, -0.06, 0.0], [-0.056, 0.06, 0.08]),
             _box([0.056, -0.06, 0.0], [0.06, 0.06, 0.08]), _box([-0.06, -0.06, 0.0], [0.06, -0.056, 0.08]),
             _box([-0.06, 0.056, 0.0], [0.06, 0.06, 0.08]))


def test_a_piece_let_go_over_a_board_rests_on_it():
    kb = _belief([_obj("board", "Geometry", BOARD, [0.0, -0.1, TABLE_Z]),
                  _obj("piece", "Rigid", PIECE, [0.05, -0.08, TABLE_Z + 0.017 + 0.002])])
    kb._drop("piece", release=True)
    assert abs(kb.objects["piece"].bottom_z - (TABLE_Z + 0.017)) < 0.0015
    kb.objects["piece"].pos[2] = TABLE_Z + 0.019
    kb._drop("piece")                                   # not a release (held look): fixtures are not consulted
    assert abs(kb.objects["piece"].bottom_z - TABLE_Z) < 1e-6


def test_a_nut_rests_on_its_bolt_top_and_stays_on_the_thread():
    bolt = _obj("bolt", "Geometry", BOLT, [0.2, -0.15, TABLE_Z])
    top = TABLE_Z + 0.050
    kb = _belief([bolt, _obj("nut", "Rigid", NUT, [0.2, -0.15, top + 0.001])])
    kb._drop("nut", release=True)
    assert abs(kb.objects["nut"].bottom_z - top) < 0.0015            # on the bolt's top, not 53 mm down on the table
    kb.objects["nut"].pos[2] = top - 0.006                            # turned 6 mm down the thread, let go
    kb._drop("nut", release=True)
    assert abs(kb.objects["nut"].bottom_z - (top - 0.006)) < 1e-6     # held by the thread where it was let go
    assert kb.drop_events[-1]["how"] == "held by a fixture"


def test_a_ring_slides_down_its_peg_and_a_drop_into_an_open_box_reaches_its_floor():
    kb = _belief([_obj("base", "Geometry", PEG, [0.0, -0.2, TABLE_Z]),
                  _obj("ring", "Rigid", RING, [0.0, -0.2, TABLE_Z + 0.063 - 0.004])])
    kb._drop("ring", release=True)                                    # 0.5 mm clearance on the peg: not held
    assert abs(kb.objects["ring"].bottom_z - (TABLE_Z + 0.0103)) < 0.0015
    kb = _belief([_obj("bin", "Geometry", BIN, [0.1, -0.1, TABLE_Z]),
                  _obj("piece", "Rigid", PIECE, [0.1, -0.1, TABLE_Z + 0.08 + 0.012])])
    kb._drop("piece", release=True)                                   # let go over the rim: falls to the floor
    assert abs(kb.objects["piece"].bottom_z - (TABLE_Z + 0.005)) < 0.0015


def test_without_meshes_or_a_still_release_nothing_changes():
    board = _obj("board", "Geometry", BOARD, [0.0, -0.1, TABLE_Z])
    piece = _obj("piece", "Rigid", PIECE, [0.05, -0.08, TABLE_Z + 0.02])
    piece.asset.mesh = None
    kb = _belief([board, piece])
    kb._drop("piece", release=True)
    assert abs(kb.objects["piece"].bottom_z - TABLE_Z) < 1e-6
    assert fixture_rest(_obj("p2", "Rigid", PIECE, [0.5, 0.3, TABLE_Z + 0.05]), [board]) == ("none", None)


def test_release_through_step_needs_a_still_arm():
    """``step``: the jaws opening on an attached object while the arm's command holds = a release onto the fixture."""
    kin = X5Kin()
    q = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    T = kin.tcp("left", q)
    piece = _obj("piece", "Rigid", PIECE, T[:3, 3] - np.array([0, 0, 0.006]))
    board = _obj("board", "Geometry", BOARD, [T[0, 3], T[1, 3], piece.pos[2] - 0.017 - 0.003])

    def run(still):
        kb = _belief([board, _obj("piece", "Rigid", PIECE, piece.pos.copy())])
        kb.q["left"] = q.copy()
        act = lambda qq, g: {"left_arm_joint_state": qq, "left_ee_joint_state": [g],          # noqa: E731
                             "right_arm_joint_state": np.zeros(6), "right_ee_joint_state": [1.0]}
        kb.step(act(q, 0.0))
        assert kb.attached.get("left", (None,))[0] == "piece"
        kb.step(act(q if still else q + 0.01, 1.0))
        return kb.objects["piece"].bottom_z
    assert abs(run(True) - float(board.top_z)) < 0.0015
    assert abs(run(False) - TABLE_Z) < 1e-6
