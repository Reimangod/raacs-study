import numpy as np
from benchmark.readout_noise import bitflip_distribution


def test_independent_readout_channel():
    p=np.array([1.,0,0,0]);r=.1
    assert np.allclose(bitflip_distribution(p,r),[(1-r)**2,r*(1-r),r*(1-r),r*r])
    assert np.array_equal(bitflip_distribution(p,0),p)
    assert np.isclose(bitflip_distribution(np.ones(16)/16,.01).sum(),1)
