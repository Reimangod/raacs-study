"""Pure selection tests: no numerical imports, sampling or new Ritz solves."""
import pytest
from benchmark.hci_paired_dimension import select_counts, select_coefficients


def test_frequency_order_not_numerical_id_prefix_and_hf_preserved():
    assert select_counts([1,2,3,7], {1:3,2:3,3:100},7,2) == [3,7]
    assert select_counts([1,2,3,7], {1:3,2:3,3:100},7,3) == [1,3,7]
    assert select_counts([1,2,3,7], {1:3,2:3,3:100},7,1) == [7]


def test_final_coefficient_magnitude_not_sign_or_id_order():
    assert select_coefficients([1,2,3,7],[.1,.2,-.9,.01],7,2) == [3,7]
    assert select_coefficients([1,2,3,7],[.2,-.2,.1,.01],7,2) == [1,7]
    assert select_coefficients([1,2,3,7],[.1,.2,-.9,.01],7,4) == [1,2,3,7]


@pytest.mark.parametrize('d', [0,5,True,1.5])
def test_invalid_dimension_is_rejected(d):
    with pytest.raises(ValueError): select_counts([1,2],{2:1},1,d)
    with pytest.raises(ValueError): select_coefficients([1,2],[.2,.8],1,d)


def test_no_padding_of_unobserved_sampler_configuration():
    with pytest.raises(ValueError, match='actually observed'):
        select_counts([1,2,3],{2:5},1,2)


@pytest.mark.parametrize('coefficient', [float('nan'),float('inf'),'0.2'])
def test_invalid_coefficient_is_not_imputed(coefficient):
    with pytest.raises(ValueError): select_coefficients([1,2],[.1,coefficient],1,2)


def test_duplicate_and_missing_hf_fail_closed():
    with pytest.raises(ValueError): select_counts([1,1],{1:1},1,1)
    with pytest.raises(ValueError): select_coefficients([1,2],[.2,.8],3,1)


def test_selection_sets_are_nested_at_growing_d():
    ids=[1,2,3,7]; counts={1:4,2:2,3:10}; coeff=[.4,.8,.2,.01]
    for d in range(1,4):
        assert set(select_counts(ids,counts,7,d)) <= set(select_counts(ids,counts,7,d+1))
        assert set(select_coefficients(ids,coeff,7,d)) <= set(select_coefficients(ids,coeff,7,d+1))
