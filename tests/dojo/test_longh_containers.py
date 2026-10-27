"""Long-Horizon stream: a container / support as a mesh-height-map obstacle (container_obstacle)."""

from __future__ import annotations

import itertools

import numpy as np

from dojo.skills.env import KinematicEnv
from dojo.skills.kinematics import TABLE_Z
from dojo.skills.skills import GTProvider, Skills


def _box(lo, hi):
    V = np.array([[x, y, z] for x, y, z in itertools.product(*zip(lo, hi))], float)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def _open_box_scene(mesh: bool):
    """A 200 x 200 x 60 mm open box (5 mm floor, 5 mm walls) and nothing else."""
    parts = [_box([-0.1, -0.1, -0.03], [0.1, 0.1, -0.025]),                        # floor
             _box([-0.1, -0.1, -0.03], [-0.095, 0.1, 0.03]), _box([0.095, -0.1, -0.03], [0.1, 0.1, 0.03]),
             _box([-0.1, -0.1, -0.03], [0.1, -0.095, 0.03]), _box([-0.1, 0.095, -0.03], [0.1, 0.1, 0.03])]
    V = np.vstack([p[0] for p in parts])
    F = np.vstack([p[1] + 8 * i for i, p in enumerate(parts)])
    row = {"label": "box", "category": "toolbox", "type": "Geometry", "pos": [0.0, -0.1, TABLE_Z + 0.03],
           "quat": [1, 0, 0, 0], "half": [0.1, 0.1, 0.03]}
    if mesh:
        row["mesh"] = (V, F)
    env = KinematicEnv([row], step_lim=10)
    sk = Skills(env, GTProvider(env, None), log=lambda *a: None)
    sk.refresh()
    return sk


def _hits(sk, p) -> bool:
    """Would a point at ``p`` (world) hit the box obstacle? (the collision check's own test, on one point)"""
    from dojo.skills.motion import below_height_map
    for c, R, h, lab in sk.coll.obstacles():
        loc = (np.atleast_2d(p) - c) @ R
        inside = np.all(np.abs(loc) <= h, axis=1)
        if inside.any() and lab in sk.coll.heightmaps:
            inside &= below_height_map(sk.coll.heightmaps[lab], loc, sk.coll.margin_of.get(lab, sk.coll.obj_margin))
        if inside.any():
            return True
    return False


def test_an_open_box_blocks_its_walls_not_its_inside_or_the_air_beside_it():
    from dojo.skills.library import container_obstacle
    sk = _open_box_scene(mesh=True)
    container_obstacle(sk, "box")
    assert "box" in sk.coll.include and "box" in sk.coll.heightmaps
    assert _hits(sk, [0.0975, -0.1, TABLE_Z + 0.05])                 # in the wall
    assert not _hits(sk, [0.0, -0.1, TABLE_Z + 0.03])                # inside, over the floor
    assert not _hits(sk, [0.0, -0.1, TABLE_Z + 0.064])               # just over the open top
    assert not _hits(sk, [0.1045, -0.1, TABLE_Z + 0.02])             # 4.5 mm beside the wall (the solid box: 6 mm)


def test_without_a_mesh_the_box_stays_solid():
    from dojo.skills.library import container_obstacle
    sk = _open_box_scene(mesh=False)
    container_obstacle(sk, "box")
    assert "box" in sk.coll.include and "box" not in sk.coll.heightmaps
    assert _hits(sk, [0.0, -0.1, TABLE_Z + 0.03]) and _hits(sk, [0.1045, -0.1, TABLE_Z + 0.02])
