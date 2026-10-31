import numpy as np
import pytest
from benchmark.ansatz import (make_pool,pool_catalog,hf_state,state_from_parameters,
                             energy_and_gradient,selection_gradients)
from benchmark.cuda_backend import ExcitationBackend
from scipy import sparse


@pytest.mark.parametrize("method",["ceo_ovp","ceo_mvp","qeb","qubit","gsd"])
def test_two_level_backend_exactly_matches_author_generators(method):
    pool=make_pool(method,4)
    rng=np.random.default_rng(602)
    h=sparse.diags(rng.normal(size=16),format="csr")
    engine=ExcitationBackend(h,pool_catalog(pool),"numpy")
    indices=list(range(pool.size));theta=rng.uniform(-.5,.5,len(indices))
    initial=hf_state(4,12)
    native=state_from_parameters(pool,indices,theta,initial)
    state=engine.prepare(indices,theta,12)
    assert np.allclose(native,state,atol=1e-11)
    a,ga=energy_and_gradient(pool,indices,theta,initial,h)
    b,gb=engine.energy_and_gradient(indices,theta,12)
    assert abs(a-b)<1e-10
    assert np.allclose(ga,gb,atol=1e-10)
    assert np.allclose(selection_gradients(pool,native,h),engine.selection_gradients(state),atol=1e-10)
