from types import SimpleNamespace

import numpy as np

from dojo.skills.geometry import T_from
from dojo.skills.motion import HeldBox
from dojo.skills.plan_geometry import heightmap_boxes, mesh_spheres


def test_heightmap_preserves_an_open_column_in_the_middle():
    obj = SimpleNamespace(label="holder", pos=np.array([0., 0., 1.]), R=np.eye(3),
                          asset=SimpleNamespace(center=np.array([0., 0., .05]), extents=np.array([.03, .03, .1])))
    z = np.full((3, 3), .1); z[1, 1] = -np.inf
    boxes = heightmap_boxes(obj, {"Z": z, "x0": 0., "y0": 0., "cell": .01}, .006)
    assert len(boxes) == 8
    point = np.array([.015, .015, 1.05])
    assert not any(np.all(np.abs((point-c)@r) < h) for c,r,h,_ in boxes)
    assert all(c[2]+h[2] >= 1.106-1e-9 for c,r,h,_ in boxes)


def test_heightmap_capacity_pooling_covers_the_highest_cell():
    obj = SimpleNamespace(label="holder", pos=np.zeros(3), R=np.eye(3),
                          asset=SimpleNamespace(center=np.array([0., 0., .05]), extents=np.array([.1, .1, .1])))
    z = np.zeros((20, 20)); z[7, 9] = .1
    boxes = heightmap_boxes(obj, {"Z":z, "x0":0., "y0":0., "cell":.005}, .006, capacity=25)
    assert len(boxes) <= 25
    point=np.array([.0475,.0375,.1])
    assert any(np.all(np.abs((point-c)@r) <= h+1e-9) for c,r,h,_ in boxes)


def test_mesh_cover_contains_triangle_interior_after_tcp_transform():
    vertices=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]])
    held=HeldBox(T_from((.01,.02,.03)), np.zeros(3), np.ones(3)*.1,
                 mesh=(vertices,np.array([[0,1,2]])))
    spheres=mesh_spheres(held,.145)
    assert len(spheres)<=256
    for a in np.linspace(0,1,13):
        for b in np.linspace(0,1-a,13):
            point=np.array([.155+.1*a,.02+.1*b,.03])
            assert np.any(np.linalg.norm(spheres[:,:3]-point,axis=1)<=spheres[:,3])


def test_hand_cover_follows_opening_and_covers_native_finger_surfaces():
    from dojo.skills.kinematics import X5Kin
    from dojo.skills.plan_geometry import hand_spheres
    kin=X5Kin()
    for opening in (0.,1.):
        spheres=hand_spheres(kin,opening)
        assert len(spheres)<=256
        for point in kin.finger_points(np.eye(4),opening):
            assert np.any(np.linalg.norm(spheres[:,:3]-point,axis=1)<=spheres[:,3]+1e-9)
    closed=hand_spheres(kin,0.);opened=hand_spheres(kin,1.)
    assert np.max(np.abs(opened[:180,1]))>np.max(np.abs(closed[:180,1]))


def test_mesh_cover_remains_valid_when_guard_samples_reduce_capacity():
    vertices=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]])
    held=HeldBox(np.eye(4),np.zeros(3),np.ones(3)*.1,
                 mesh=(vertices,np.array([[0,1,2]])))
    # A held mesh may be reused across worlds with different guard capacity.
    for capacity in (256,8,229,8):
        spheres=mesh_spheres(held,0.,capacity=capacity)
        assert len(spheres)<=capacity
        for point in ([0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[.03,.04,0.]):
            assert np.any(np.linalg.norm(spheres[:,:3]-point,axis=1)<=spheres[:,3])
