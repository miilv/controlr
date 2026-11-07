"""Default-off mesh table guard: full geometry, conservative footprint, no hand bypass."""
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.skills.geometry import T_from, inv_T, rot_y
from dojo.skills.kinematics import X5Kin, TABLE_Z
from dojo.skills.motion import Collision, HeldBox, held_table_collision


def shape(z=.8):
    # A tilted diagonal triangle is horizontal in world; its OBB has empty low corners.
    v=np.array([[-.1,0,-.1],[.1,0,.1],[0,.02,0]])
    return HeldBox(T_from([0,0,z],rot_y(np.pi/4)),np.array([0,.01,0]),
                   np.array([.1,.01,.1]),mesh=(v,np.array([[0,1,2]])))


def test_mesh_clears_only_the_empty_box_corner():
    h=shape()
    assert held_table_collision(h,np.eye(4))
    assert held_table_collision(h,np.eye(4),'bbox')
    assert not held_table_collision(h,np.eye(4),'mesh')
    assert held_table_collision(shape(.75),np.eye(4),'mesh')


def test_missing_or_invalid_mesh_keeps_original_guard():
    for mesh in (None,(np.empty((0,3)),None),(np.full((3,3),np.nan),None),([["bad"]],None)):
        h=shape();h.mesh=mesh
        assert held_table_collision(h,np.eye(4),'mesh')


def test_all_vertices_and_boundary_crossings_are_checked():
    # No triangle vertex lies inside the table, but the triangle crosses its footprint.
    v=np.array([[-2,-2,.75],[2,-2,.75],[0,2,.75]])
    h=HeldBox(np.eye(4),np.array([0,0,.8]),np.full(3,.01),mesh=(v,np.array([[0,1,2]])),
              dense=np.array([[0,0,.8]]),points=np.array([[0,0,.8]]))
    assert held_table_collision(h,np.eye(4),'mesh')
    # A low vertex missed by sparse/dense samples must still reject the mesh.
    h.mesh=(np.vstack([np.tile([0,0,.8],(601,1)),[0,0,.74]]),None)
    assert held_table_collision(h,np.eye(4),'mesh')
    # A completely disjoint footprint does not collide with the finite table.
    h.mesh=(np.array([[1.,0,.75],[1.1,.1,.75],[1.1,-.1,.75]]),None)
    assert not held_table_collision(h,np.eye(4),'mesh')


def test_collision_integration_keeps_hand_table_checks():
    kin=X5Kin();q=np.zeros(6);tcp=kin.tcp('left',q)
    h=shape();h.T_tcp_obj=inv_T(tcp)@h.T_tcp_obj
    c=Collision(kin)
    assert c.held_table=='bbox'
    assert c.check('left',q,held=h)==(False,'held object below the table')
    c.held_table='mesh'
    assert c.check('left',q,held=h)[0]
    # Even with clear held geometry, fingertips below the plane are rejected first.
    badkin=SimpleNamespace(frames=lambda *a:[np.eye(4)],
                           finger_points=lambda *a:np.array([[0,0,TABLE_Z-.01]]),
                           palm_points=lambda *a:np.array([[0,0,.9]]),
                           link_points=lambda *a,**kw:np.array([[0,0,.9]]),
                           tcp_from_flange=lambda *a:np.eye(4))
    c=Collision(badkin);c.held_table='mesh'
    ok,reason=c.check('left',q,held=shape())
    assert not ok and reason.startswith('fingertip at table')


@pytest.mark.parametrize('tag,mode,inside',[('headband_bridge','mesh','mesh'),
                                          ('handle','mesh','bbox'),('headband_bridge','bbox','bbox')])
def test_cradle_scope_restores_table_mode_on_exception(monkeypatch,tag,mode,inside):
    from dojo.skills import library as L
    sk=SimpleNamespace(headset_table=mode,coll=SimpleNamespace(held_table='bbox',held_skip={}))
    def fail(*a):
        assert sk.coll.held_table==inside
        raise RuntimeError('sentinel')
    monkeypatch.setattr(L,'_hang_on_hooks',fail)
    with pytest.raises(RuntimeError,match='sentinel'):
        L.hang_on_hooks(sk,[],'stand','cradle',tag)
    assert sk.coll.held_table=='bbox'


def test_mesh_table_config_defaults_and_validation():
    from dojo.runtime.policy import RuntimeConfig,RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    from dojo.skills.skills import Skills
    from dojo.skills.env import KinematicEnv
    from dojo.skills.world import WorldState
    assert RuntimeConfig().headset_table=='bbox'
    assert SKILL_ATTRS['headset_table']==('bbox','mesh')
    sk=Skills(KinematicEnv([]),SimpleNamespace(refresh=lambda:WorldState('','',{}, {}, {})))
    assert sk.headset_table=='bbox' and sk.coll.held_table=='bbox'
    with pytest.raises(ValueError,match='bad runtime config'):
        RuntimePolicy(RuntimeConfig(headset_table='skip'),assets=object())
