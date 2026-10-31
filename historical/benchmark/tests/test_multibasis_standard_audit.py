import numpy as np
from scipy.linalg import expm
from benchmark.multibasis_standard_audit import sector_vector,rotate_full_audit_only
from benchmark.multibasis_audit import exterior_vector
from benchmark.nonorthogonal_ci import occupied_columns
from benchmark.chemistry import determinant_ids


def test_fixed_sector_ffsim_is_independent_exterior_parity():
    rng=np.random.default_rng(341);n=4
    a=rng.normal(size=(n,n));u=expm(.3*(a-a.T));ids,_,_,_=determinant_ids(n,2,1)
    for idx in ids[::7]:
        actual=sector_vector(u,int(idx),2,1)
        expected=exterior_vector(occupied_columns(u,int(idx)))
        assert np.max(abs(actual-expected[ids]))<1e-11


def test_full_audit_rotation_handles_multiple_spin_sectors_not_sector_postselection():
    rng=np.random.default_rng(981);n=3
    a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));u=expm(.2*(a-a.conj().T))
    state=np.zeros(1<<(2*n),complex);state[[48,40,24,12]]=[.5,.5j,-.5,.5]
    actual=rotate_full_audit_only(state,u)
    expected=sum(state[i]*exterior_vector(occupied_columns(u,i)) for i in np.flatnonzero(state))
    assert np.max(abs(actual-expected))<1e-11
