"""dojo.runtime.belief ``release_rest="container"`` (RuntimeConfig.release_rest, G3 ``--release-rest``; default legacy):
an object let go with its centre over a fixture's open interior rests on that interior, where the legacy rule rested it
on the rim or kept it at the release height when its predicted footprint dipped into a wall (CLASSIFY3: recorded
classify rows, physics had the object in the basket and the belief above the rim). CPU only, synthetic meshes."""

import numpy as np
import pytest

from dojo.runtime.belief import KinematicBelief, container_rest, drop_rest
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


def _obj(label, typ, mesh, pos, key=None):
    V = np.asarray(mesh[0], float)
    info = AssetInfo(key=key or f"{typ}/{label}/00000", bbox_min=V.min(0), bbox_max=V.max(0), mesh=mesh)
    return ObjectState(label, typ, label, 0, np.asarray(pos, float), np.array([1.0, 0, 0, 0]), info)


def _belief(objs, rest):
    st = {f"{a}_arm_joint_state": np.zeros(6) for a in ARMS}
    st.update({f"{a}_ee_joint_state": [1.0] for a in ARMS})
    kb = KinematicBelief({o.label: o for o in objs}, X5Kin(), st)
    kb.release_rest = rest
    return kb


# a basket like classify's (252 x 168 x 77 mm outside, 4 mm walls, 10 mm floor), its frame at its centre
H = 0.0385
BASKET = _merge(_box([-0.126, -0.084, -H], [0.126, 0.084, -H + 0.010]),
                _box([-0.126, -0.084, -H], [-0.122, 0.084, H]), _box([0.122, -0.084, -H], [0.126, 0.084, H]),
                _box([-0.126, -0.084, -H], [0.126, -0.080, H]), _box([-0.126, 0.080, -H], [0.126, 0.084, H]))
# a toy car, its frame at its bottom centre; 2 mm slices so its vertices sample its underside (a catalogue mesh is dense)
CAR = _merge(*[_box([x, -0.02, 0.0], [x + 0.002, 0.02, 0.035]) for x in np.arange(-0.045, 0.0429, 0.002)])
CUP = _box([-0.02, -0.02, 0.0], [0.02, 0.02, 0.03])
# a holder with one 30 mm hole 25 mm deep in a 60 mm block: an opening narrower than the car's footprint
HOLDER = _merge(_box([-0.03, -0.03, 0.0], [0.03, 0.03, 0.005]),
                _box([-0.03, -0.03, 0.0], [-0.015, 0.03, 0.03]), _box([0.015, -0.03, 0.0], [0.03, 0.03, 0.03]),
                _box([-0.015, -0.03, 0.0], [0.015, -0.015, 0.03]), _box([-0.015, 0.015, 0.0], [0.015, 0.03, 0.03]))

BX, BY, BZ = 0.28, 0.05, TABLE_Z + H
FLOOR = TABLE_Z + 0.010


def _same(a, b):
    return (a[0] is None and b[0] is None or np.allclose(a[0], b[0])) and a[1] == b[1]


def _basket(key="Geometry/basket/test0"):
    return _obj("basket0", "Geometry", BASKET, [BX, BY, BZ], key=key)


def test_a_car_over_the_wall_rests_on_the_rim_or_hangs_legacy_and_falls_in_with_container():
    # let go 20 mm above the rim with its centre 30 mm inside the right wall, the car's end over the wall
    rel = [BX + 0.122 - 0.030, BY, BZ + H + 0.020]
    kb = _belief([_basket(), _obj("car", "Rigid", CAR, rel)], "legacy")
    kb._drop("car", release=True)
    assert kb.objects["car"].bottom_z > BZ + H - 0.002                    # on the rim: above the basket's top
    kb = _belief([_basket(), _obj("car", "Rigid", CAR, rel)], "container")
    kb._drop("car", release=True)
    assert abs(kb.objects["car"].bottom_z - FLOOR) < 0.0015              # inside, on the floor
    assert kb.drop_events[-1]["how"] == "into a container"
    # the footprint dipping 5 mm into the wall: legacy keeps it at the release height ("held by a fixture")
    low = [BX + 0.122 - 0.030, BY, BZ + H - 0.005]
    kb = _belief([_basket(), _obj("car", "Rigid", CAR, low)], "legacy")
    kb._drop("car", release=True)
    assert kb.drop_events[-1]["how"] == "held by a fixture"
    kb = _belief([_basket(), _obj("car", "Rigid", CAR, low)], "container")
    kb._drop("car", release=True)
    assert abs(kb.objects["car"].bottom_z - FLOOR) < 0.0015


def test_an_object_already_inside_supports_the_next_one():
    under = _obj("cup", "Rigid", CUP, [BX, BY, FLOOR])
    car = _obj("car", "Rigid", CAR, [BX, BY, BZ + H + 0.03])
    pos, ev = drop_rest(car, [under], [_basket()], container=True)
    assert ev["how"] == "into a container"
    assert abs(pos[2] - (FLOOR + 0.03)) < 1e-6                          # on the cup's top, not through it


def test_centre_outside_the_opening_or_an_opening_too_small_keeps_the_legacy_rule():
    b = _basket()
    # centre over the wall (outside the opening): no container rest
    car = _obj("car", "Rigid", CAR, [BX + 0.124, BY, BZ + H + 0.02])
    assert container_rest(car, [b]) is None
    assert _same(drop_rest(car, [], [b], container=True), drop_rest(car, [], [b], container=False))
    # a 30 mm hole is narrower than the car's footprint: no container rest
    holder = _obj("holder", "Geometry", HOLDER, [0.0, -0.2, TABLE_Z], key="Geometry/holder/test0")
    car = _obj("car", "Rigid", CAR, [0.0, -0.2, TABLE_Z + 0.031])
    assert container_rest(car, [holder]) is None
    assert _same(drop_rest(car, [], [holder], container=True), drop_rest(car, [], [holder], container=False))
    # a small cube fits that hole: it goes in
    cube = _obj("cube", "Rigid", _box([-0.008, -0.008, 0.0], [0.008, 0.008, 0.016]), [0.0, -0.2, TABLE_Z + 0.031])
    z, lab = container_rest(cube, [holder])
    assert lab == "holder" and abs(z - (TABLE_Z + 0.005)) < 0.0015


@pytest.mark.parametrize("rest", ["legacy", "container"])
def test_on_the_table_or_without_fixtures_both_rules_agree(rest):
    b = _basket()
    car = _obj("car", "Rigid", CAR, [-0.1, -0.2, TABLE_Z + 0.05])     # far from the basket
    assert _same(drop_rest(car, [], [b], container=rest == "container"), drop_rest(car, [], [b]))
    kb = _belief([b, _obj("car", "Rigid", CAR, [BX, BY, BZ + H + 0.02])], rest)
    kb._drop("car", release=False)                                     # not a still release: fixtures are not consulted
    assert abs(kb.objects["car"].bottom_z - TABLE_Z) < 1e-6


def test_runtime_config_default_and_validation():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    assert RuntimeConfig().release_rest == "legacy"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(release_rest="basket"))


def test_heldsim_mirrors_the_switch():
    from types import SimpleNamespace

    from dojo.skills.heldsim import HeldBeliefProvider
    env = SimpleNamespace(step=lambda action: None)
    with pytest.raises(AssertionError):
        HeldBeliefProvider(object(), env, mode=None, drop="fixtures", release_rest="basket")
    assert HeldBeliefProvider(object(), env, mode=None, drop="fixtures").release_rest == "legacy"
