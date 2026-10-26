"""skills7: pieces threaded onto pegs (hole from the mesh, side bias, order), the rack as small boxes, hook order,
handle frames along the hole, dry-run pegs and hooks in the kinematic env (CPU only)."""

from __future__ import annotations

import numpy as np

from tests.dojo.test_skills2 import _box_mesh, _merge, _obj


def _plate_with_hole(hx: float, hy: float, half: float = 0.03, hole: float = 0.005, t: float = 0.013):
    """A square plate (``half`` m) with a square through hole of half size ``hole`` centred at (hx, hy)."""
    parts = [_box_mesh((-half, -half, 0), (hx - hole, half, t)),            # left of the hole
             _box_mesh((hx + hole, -half, 0), (half, half, t)),             # right of it
             _box_mesh((hx - hole, -half, 0), (hx + hole, hy - hole, t)),   # in front
             _box_mesh((hx - hole, hy + hole, 0), (hx + hole, half, t))]    # behind
    return _merge(*parts)


def test_through_hole_finds_an_off_centre_hole():
    from dojo.skills.pegs import through_hole
    m = _plate_with_hole(0.0, 0.004)
    o = _obj("p", "toy", 701, (0, 0, 0.77), (0.03, 0.03, 0.0065), mesh=m)
    h = through_hole(o)
    assert h is not None
    assert np.allclose(h[0][:2], [0.0, 0.004], atol=6e-4)
    assert 0.0045 < h[1] < 0.0065                      # equivalent radius of a 10 mm square hole: 5.6 mm
    # a plate without a hole has none
    o2 = _obj("q", "toy", 702, (0, 0, 0.77), (0.03, 0.03, 0.0065), mesh=_box_mesh((-0.03, -0.03, 0), (0.03, 0.03, 0.013)))
    assert through_hole(o2) is None


def test_piece_target_puts_the_hole_on_the_peg_with_the_bias():
    from dojo.skills.pegs import piece_target, side_bias, through_hole
    from tests.dojo.test_skills2 import _world
    m = _plate_with_hole(0.0, 0.004)
    o = _obj("p", "toy", 703, (0.2, -0.1, 0.77), (0.03, 0.03, 0.0065), mesh=m, place={"up": [1.0, 0, 0, 0]})

    class SK:
        world = _world([o])
    peg = {"p": np.array([0.05, -0.2, 0.82]), "top": 0.83, "radius": 0.0050}
    T, hole = piece_target(SK(), "p", peg, 0.3, 0.826, bias=(0.0, -0.0004))
    hw = T[:3, :3] @ hole + T[:3, 3]
    assert np.allclose(hw[:2], [0.05, -0.2004], atol=1e-6)
    low = float((m[0] @ T[:3, :3].T + T[:3, 3])[:, 2].min())
    assert abs(low - 0.826) < 1e-6
    b = side_bias(o, peg)
    assert abs(b - (through_hole(o)[1] - 0.0050 - 0.00015)) < 1e-9 and b > 0


def test_kinematic_env_slides_a_piece_down_its_peg():
    from dojo.skills.env import KinematicEnv
    rows = [{"label": "base", "type": "Geometry", "pos": [0, -0.2, 0.765], "quat": [1, 0, 0, 0], "half": [0.13, 0.03, 0.033],
             "center": [0, 0, 0.033], "pegs": [[0.0, -0.2, 0.775], [0.065, -0.2, 0.775]]},
            {"label": "a", "pos": [0.0005, -0.2, 0.84], "quat": [1, 0, 0, 0], "half": [0.029, 0.029, 0.0065],
             "center": [0, 0, 0]},
            {"label": "b", "pos": [0.0, -0.2003, 0.86], "quat": [1, 0, 0, 0], "half": [0.029, 0.029, 0.0065],
             "center": [0, 0, 0]},
            {"label": "c", "pos": [0.065, -0.2, 0.86], "quat": [1, 0, 0, 0], "half": [0.029, 0.029, 0.0065],
             "center": [0, 0, 0]}]
    env = KinematicEnv(rows)
    for lab in ("a", "b", "c"):
        env._drop(lab)
    z = {o["label"]: o["pos"][2] for o in env.objects}
    assert abs(z["a"] - (0.775 + 0.0065)) < 1e-9                     # on the plate, snapped onto the axis
    assert abs(z["b"] - (0.775 + 0.013 + 0.0065)) < 1e-9             # on the piece below it
    assert abs(z["c"] - (0.775 + 0.0065)) < 1e-9                     # the next peg: not on its neighbour
    assert abs(env._obj("a")["pos"][0]) < 1e-9


def test_kinematic_env_leaves_a_hung_object_on_its_hook():
    from dojo.skills.env import KinematicEnv
    rows = [{"label": "rack", "type": "Geometry", "pos": [0, 0, 0.765], "quat": [1, 0, 0, 0], "half": [0.09, 0.09, 0.17],
             "center": [0, 0, 0.17], "hang_points": [[0.06, 0.0, 0.93]]},
            {"label": "mug", "pos": [0.07, 0.0, 0.90], "quat": [1, 0, 0, 0], "half": [0.05, 0.04, 0.04], "center": [0, 0, 0]},
            {"label": "cup", "pos": [-0.2, 0.0, 0.90], "quat": [1, 0, 0, 0], "half": [0.05, 0.04, 0.04], "center": [0, 0, 0]}]
    env = KinematicEnv(rows)
    env._drop("mug")
    env._drop("cup")
    assert abs(env._obj("mug")["pos"][2] - 0.90) < 1e-9
    assert env._obj("cup")["pos"][2] < 0.85


def test_mesh_parts_model_a_thin_arm_not_its_box():
    """A rack-like L (a pole and one thin arm): a point under the arm is free, one in the arm collides; the box of the
    whole object would have refused both."""
    from dojo.skills.kinematics import X5Kin
    from dojo.skills.motion import Collision, HeldBox, mesh_parts
    from tests.dojo.test_skills2 import _world
    m = _merge(_box_mesh((-0.006, -0.006, 0.0), (0.006, 0.006, 0.3)), _box_mesh((0.0, -0.004, 0.2), (0.09, 0.004, 0.208)))
    C, H = mesh_parts(*m)
    assert len(C) > 5
    lo, hi = (C - H).min(0), (C + H).max(0)
    assert np.allclose(lo, m[0].min(0), atol=2.5e-3) and np.allclose(hi, m[0].max(0), atol=2.5e-3)   # min half 2 mm
    rack = _obj("rack", "cup_holder", 700, (0.0, 0.2, 0.765), (0.045, 0.006, 0.15), typ="Geometry", mesh=m)
    coll = Collision(X5Kin(), _world([rack]))
    coll.include.add("rack")
    coll.parts["rack"] = (C, H)
    obs = coll.obstacles()
    assert len(obs) == len(C) and all(o[3] == "rack" for o in obs)

    def hits(p):
        pts = np.asarray([p], float)
        return any(np.all(np.abs((pts - c) @ R) <= h, axis=1).any() for c, R, h, _ in obs)
    assert hits((0.05, 0.2, 0.765 + 0.204))                  # in the arm
    assert not hits((0.05, 0.2, 0.765 + 0.15))               # 5 cm under it, inside the object's box
    # a held object's mesh points (HeldBox.dense) are what is checked against parts
    hb = HeldBox(np.eye(4), np.zeros(3), np.full(3, 0.01), dense=np.zeros((1, 3)))
    assert np.allclose(hb.dense_w(np.eye(4)), 0.0)


def test_hook_order_fills_the_highest_hooks_lowest_first():
    from dojo.skills.geometry import T_from
    from dojo.skills.library import hook_order
    hooks = [(i, T_from((0, 0, z)), 0.03) for i, z in enumerate((0.93, 0.93, 1.01, 1.01, 1.09, 1.09))]
    order = [h[0] for h in hook_order(hooks, 3)]
    assert order[:3] in ([2, 4, 5], [3, 4, 5])                # one middle hook, then both top ones
    assert set(order) == set(range(6)) and order[3] in (2, 3)


def test_threads_along_x_tells_the_hole_axis():
    """A handle loop in the x-z plane... the frame whose x runs along the loop's hole passes; one along the bar does
    not."""
    from dojo.skills.geometry import T_from
    from dojo.skills.library import threads_along_x
    # a square loop around the y axis (hole along y), bars 4 mm thick, inner half size 8 mm, at x = 0.05
    loop = _merge(_box_mesh((0.038, -0.004, 0.008), (0.062, 0.004, 0.012)),
                  _box_mesh((0.038, -0.004, -0.012), (0.062, 0.004, -0.008)),
                  _box_mesh((0.058, -0.004, -0.012), (0.062, 0.004, 0.012)),
                  _box_mesh((0.038, -0.004, -0.012), (0.042, 0.004, 0.012)))
    o = _obj("mug", "mug", 799, (0, 0, 0), (0.03, 0.01, 0.015), mesh=loop)
    # handle frame on the outside of the bar (x = 0.062), z pointing out: the hole centre 12 mm inside it
    R_good = np.stack([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]], axis=1)      # x along +y (the hole), z along +x
    R_bad = np.stack([[0, 0, 1.0], [0, -1.0, 0], [1.0, 0, 0]], axis=1)      # x along +z (across the bars)
    assert threads_along_x(o, T_from((0.062, 0, 0), R_good), 0.012)
    assert not threads_along_x(o, T_from((0.062, 0, 0), R_bad), 0.012)


def test_free_spots_avoid_a_fixture():
    from dojo.skills.library import free_spots
    from tests.dojo.test_skills2 import _world
    piece = _obj("p", "toy", 704, (0.3, -0.1, 0.77), (0.029, 0.029, 0.0065))
    base = _obj("base", "stack_base", 705, (0.0, -0.2, 0.765), (0.13, 0.032, 0.033), typ="Geometry")

    class SK:
        world = _world([piece, base])
    plain = free_spots(SK(), "p", near_xy=(0.0, -0.2), clearance=0.03)
    clear = free_spots(SK(), "p", near_xy=(0.0, -0.2), clearance=0.03, avoid=("base",))
    assert any(abs(y + 0.2) < 1e-9 for _, y in plain)           # without: spots on the base
    assert clear and all(abs(y + 0.2) > 0.032 + 0.04 for _, y in clear)


def test_stack_top_and_stuck():
    from dojo.skills.geometry import rot_x
    from dojo.skills.pegs import stack_top, stuck
    from tests.dojo.test_skills2 import _world
    m = _plate_with_hole(0.0, 0.0)
    a = _obj("a", "toy", 706, (0.0, -0.2, 0.7753), (0.03, 0.03, 0.0065), mesh=m, place={"up": [1.0, 0, 0, 0]})
    q = np.array([np.cos(np.radians(10)), np.sin(np.radians(10)), 0, 0])        # 20 deg about x
    b = _obj("b", "toy", 706, (0.0, -0.2, 0.80), (0.03, 0.03, 0.0065), mesh=m, quat=q, place={"up": [1.0, 0, 0, 0]})

    class SK:
        world = _world([a, b])

        def refresh(self):
            return self.world
    peg = {"plate": 0.7753, "frame_z": 0.828}
    assert abs(stack_top(SK(), [], peg) - 0.7753) < 1e-9
    assert abs(stack_top(SK(), ["a"], peg) - (0.7753 + 0.013)) < 1e-6
    assert stuck(SK(), "b", peg, ["a"])                      # tilted 20 deg, high on the peg
    assert not stuck(SK(), "a", peg, [])
    assert rot_x(0.0).shape == (3, 3)


def test_held_measured_falls_back_to_blind_threading():
    import types
    from dojo.skills.pegs import held_measured
    eye = np.eye(4)
    assert held_measured(types.SimpleNamespace(remeasure=True, provider=None))           # G3: the true pose
    assert not held_measured(types.SimpleNamespace(remeasure=False, provider=object()))  # mode B, nothing to measure
    prov = types.SimpleNamespace(held_pose=lambda arm: (eye, 0.0003) if arm == "left" else {"T_tcp_obj": eye, "sigma": 0.002})
    sk = types.SimpleNamespace(remeasure=False, provider=prov)
    assert held_measured(sk)                       # a measurement exists: plan to seat
    assert held_measured(sk, "left")               # 0.3 mm: seat
    assert not held_measured(sk, "right")          # 2 mm: thread blind
    sk.provider = types.SimpleNamespace(held_pose=lambda arm: None)
    assert not held_measured(sk, "left")
