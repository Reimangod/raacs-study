import numpy as np
import pytest
from scipy.sparse import csr_matrix
from benchmark.stopping_pilot_bank import ci_features, PROBES


def test_external_residual_is_not_the_internal_Ritz_residual():
    h = csr_matrix([[0., 1.], [1., 2.]])
    ids = np.array([3, 5])
    one = ci_features(h, ids, [3], 3, {})
    assert one["qsci_energy_Ha"] == 0 and one["external_residual_norm_Ha"] == 1
    assert one["pt2_proxy_Ha"] == pytest.approx(-.5)
    both = ci_features(h, ids, [3, 5], 3, {5: 1})
    assert both["qsci_energy_Ha"] == pytest.approx(1-np.sqrt(2))
    assert both["external_residual_norm_Ha"] == 0


def test_unobserved_padding_and_probe_budget():
    h = csr_matrix([[0., 1.], [1., 2.]])
    with pytest.raises(ValueError): ci_features(h, np.array([3, 5]), [3, 5], 3, {})
    assert len(PROBES)*64 < 1000
    # An off-grid natural endpoint replaces unavailable future probes; the
    # episode never needs an extra probe beyond the11-at-64step schedule.
    for final_step in range(1, 65):
        probes = set(x for x in PROBES if x <= final_step) | {final_step}
        assert len(probes)*64 < 1000
