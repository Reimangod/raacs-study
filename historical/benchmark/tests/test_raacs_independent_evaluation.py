import numpy as np
import pytest
from benchmark.raacs_independent_evaluation import remaining_budget, draw_components, distribution_id, normalized_probability


def test_unknown_cost_is_not_zero_and_exhausted_is_not_available():
    assert remaining_budget(1000, None) == (None, "UNKNOWN_QPU_CONSTRUCTION_COST")
    assert remaining_budget(1000, 1000) == (None, "TOTAL_BUDGET_INFEASIBLE")
    assert remaining_budget(1000, 1001) == (None, "TOTAL_BUDGET_INFEASIBLE")
    assert remaining_budget(1000, 992) == (8, "AVAILABLE")
    assert remaining_budget(1000, 0) == (1000, "AVAILABLE")
    with pytest.raises(ValueError): remaining_budget(1000, False)


def test_multicircuit_draw_is_actual_stratified_not_mean_probability():
    p = np.array([1., 0, 0, 0]); q = np.array([0., 0, 0, 1])
    components = [(p, 1), (q, 1)]; key = distribution_id("same physical problem", components)
    ids, counts = draw_components(components, 101, key, 0)
    assert ids.tolist() == [0, 3] and counts.tolist() == [51, 50]
    ids2, counts2 = draw_components(components, 101, key, 0)
    assert np.array_equal(ids, ids2) and np.array_equal(counts, counts2)
    assert distribution_id("other physical problem", components) != key


def test_final_readout_small_only_probability_contract():
    assert normalized_probability([1, 2, 0, 0], 2).sum() == 1
    with pytest.raises(ValueError): normalized_probability([1], 40)
    with pytest.raises(ValueError): normalized_probability([1, -1, 0, 0], 2)
    with pytest.raises(ValueError): normalized_probability([0, 0, 0, 0], 2)
