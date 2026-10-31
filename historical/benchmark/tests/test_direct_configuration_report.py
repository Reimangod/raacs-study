from copy import deepcopy
import pytest
from benchmark.direct_configuration_report import pair_fields


def records():
    d=dict(final_dimension=2,final_energy_Ha=-1.,history=[{'selected_ids':[1,3]}])
    q=dict(final_selected_ids=[3,1],actual_final_CI_dimension=2,algorithm={'seed':101},
        final_QSCI_energy_Ha=-1.,actual_total_model_shots=10000)
    return d,q


def test_actual_equal_dimension_is_not_matching_design_flag():
    d,q=records();r=pair_fields('toy',d,q)
    assert r['equal_actual_dimension'] and r['identical_selected_space']
    assert r['RAACS_space_subset_of_direct'] and not r['same_final_dimension_enforced']
    assert r['classical_CNOT'] is None and r['ground_state_error_Ha'] is None


def test_same_dimension_different_support_is_not_equal_space():
    d,q=records();q['final_selected_ids']=[1,5];r=pair_fields('toy',d,q)
    assert r['equal_actual_dimension'] and not r['identical_selected_space']
    assert not r['RAACS_space_subset_of_direct'] and r['intersecting_configurations']==1


def test_unequal_dimensions_and_difference_sign_preserved():
    d,q=records();q['final_selected_ids']=[1];q['actual_final_CI_dimension']=1;q['final_QSCI_energy_Ha']=-.9
    r=pair_fields('toy',d,q)
    assert not r['equal_actual_dimension'] and r['direct_minus_RAACS_mHa']==pytest.approx(-100.)
    assert r['RAACS_space_subset_of_direct']


def test_dimension_mismatch_is_rejected():
    d,q=records();d['final_dimension']=1
    with pytest.raises(ValueError):pair_fields('toy',d,q)
