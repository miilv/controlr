"""CPU-only guards for the opt-in basket experiment."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from dojo.skills.basket import hook_pose, route


def test_config_defaults_and_validation():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().belt_lift == 'off'
    assert RuntimeConfig().new_routes == []
    assert SKILL_ATTRS['belt_lift'] == ('off', 'braced_hook')
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(belt_lift='typo'))
    RuntimePolicy(RuntimeConfig(belt_lift='braced_hook', new_routes=['basket_image']))


def obj(label, category, typ='Rigid'):
    return NS(label=label, category=category, obj_type=typ, clutter=False)


def test_route_needs_opt_in_instruction_and_observed_roles():
    objects = [obj('b', 'basket_grasp'), obj('p', 'photo', 'Rigid'), obj('t', 'apple'),
               obj('c', 'conveyor', 'Dynamic')]
    w = NS(task_objects=lambda: objects)
    text = 'lift the basket and pick the target on the conveyor according to the image'
    assert route(text, w, []) is None
    r = route(text, w, ['basket_image'])
    assert r[:2] == ('pick_into_held', dict(objects=['t'], container='b', picture='p', conveyor='c'))
    assert route('remember the first object on the conveyor', w, ['basket_image']) is None
    objects.pop(1)
    assert route(text, w, ['basket_image']) is None
    objects.append(obj('p', 'photo'))
    objects.append(obj('b2', 'basket_grasp'))
    assert route(text, w, ['basket_image']) is None


def test_hook_geometry_and_missing_mesh():
    x, y = np.meshgrid(np.linspace(-.07, .07, 10), [-.005, .005])
    v = np.c_[x.ravel(), y.ravel(), np.full(x.size, .2)]
    o = NS(asset=NS(mesh=(v, [])), R=np.eye(3), pos=np.array([.1, 0., .8]))
    t = hook_pose(o, np.array([.1, -.4, .8]))
    np.testing.assert_allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-9)
    assert np.linalg.det(t[:3, :3]) == pytest.approx(1)
    assert t[2, 3] == pytest.approx(.964)
    assert t[1, 0] > 0  # approach away from the base, across the bar
    o.asset.mesh = None
    assert hook_pose(o, np.zeros(3)) is None
