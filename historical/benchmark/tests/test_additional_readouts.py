from collections import Counter
import numpy as np
from scipy import sparse
from benchmark.cumulative import merge_counts
from benchmark.qsci import solve_selected


def test_cumulative_counts_are_actual_sum_not_last_checkpoint():
    counts=Counter();merge_counts(counts,[0,3],[8,2]);merge_counts(counts,[0,2],[5,5])
    assert counts=={0:13,3:2,2:5} and sum(counts.values())==20


def test_large_projected_ci_uses_sparse_solver():
    h=sparse.diags(np.linspace(-2,5,512),format="csr")
    result=solve_selected(h,np.arange(512))
    assert result["solver"].startswith("sparse eigsh")
    assert abs(result["energy"]+2)<1e-9 and result["residual"]<1e-8
