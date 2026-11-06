import numpy as np
from dojo.skills.plan_seeds import ik_seed_bank


def test_attempts_reproduce_but_do_not_repeat_the_same_global_bank():
    bounds=np.tile([-1.,1.],(6,1));hint=np.zeros(6)
    a=ik_seed_bank(hint,hint,bounds,64,0)
    b=ik_seed_bank(hint,hint,bounds,64,1)
    assert np.array_equal(a,ik_seed_bank(hint,hint,bounds,64,0))
    assert not np.array_equal(a[:,16:],b[:,16:])
    assert np.array_equal(a[:,0],hint.reshape(1,6))
    assert (a>=-1).all() and (a<=1).all()
    assert np.abs(a[:,:16]).max()<.2 and np.abs(a[:,16:]).max()>.8
