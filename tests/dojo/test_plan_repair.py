import numpy as np

from dojo.skills.plan_repair import repair_coordinates


def test_shallow_contact_repair_is_bounded_and_monotone():
    depth=lambda q:np.array([max(0.,.01-q[0])])
    cost=lambda q:{"scene_collision":depth(q)[0]**2,"self_collision":0.}
    path,info=repair_coordinates(np.zeros(6),np.tile([-1.,1.],(6,1)),cost,depth)
    assert path is not None and info["reason"]=="repaired"
    assert np.max(np.abs(path))<=.3 and np.max(np.abs(np.diff(path,axis=0)))<=.03+1e-9
    assert np.all(np.diff([depth(q)[0] for q in path])<=0)
    assert depth(path[-1])[0]==0


def test_repair_cannot_trade_old_contact_for_a_new_obstacle():
    depth=lambda q:np.array([max(0.,.01-q[0]),max(0.,q[0]-.005)])
    cost=lambda q:{"scene_collision":float(np.square(depth(q)).sum()),"self_collision":0.}
    path,info=repair_coordinates(np.zeros(6),np.tile([-1.,1.],(6,1)),cost,depth)
    assert path is None


def test_deep_start_and_self_contact_are_refused():
    for depth,self_cost in [(.02,0.),(.01,1.)]:
        path,info=repair_coordinates(np.zeros(6),np.tile([-1.,1.],(6,1)),
            lambda q:{"scene_collision":1.,"self_collision":self_cost},lambda q:np.array([depth]))
        assert path is None
        assert info["reason"] in ["start_depth_limit","start_self_collision"]


def test_prefix_cannot_dive_deeper_then_escape_at_its_endpoint():
    from dojo.skills.plan_repair import check_escape_prefix
    cost=lambda q:{"self_collision":0.}
    penetration=lambda q:np.array([q[0]])
    valid,why=check_escape_prefix(np.array([[.005],[.012],[0.]]),cost,penetration)
    assert not valid and why=="prefix_worsens_contact"
    assert check_escape_prefix(np.array([[.012],[.005],[0.]]),cost,penetration)[0]
