import numpy as np
from scipy.sparse import csr_matrix
from benchmark.qsci import select_configurations, solve_selected


def test_frequency_hf_cap_and_no_padding():
    ids = np.array([0,3,5,7]); counts = np.array([5,2,2,1])
    assert select_configurations(ids,counts,{3,5,6},6,2,True).tolist() == [3,6]
    assert select_configurations(ids,counts,{3,5,6},6,8,False).tolist() == [3,5]


def test_monotone_ritz_and_phase_independence():
    h = csr_matrix([[1.,.3,0],[.3,2.,.1],[0,.1,3.]])
    a = solve_selected(h,np.array([0]))
    b = solve_selected(h,np.array([0,1]))
    c = solve_selected(h,np.array([0,1,2]))
    assert a["energy"] >= b["energy"] >= c["energy"]
    # No sampler amplitudes or phases enter selected-space diagonalization.
    assert abs(solve_selected(h,np.array([1,0]))["energy"]-b["energy"]) < 1e-13
    assert solve_selected(h,np.array([],dtype=int))["status"] == "EMPTY_VALID_SUBSPACE"
