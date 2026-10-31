"""Native SqDRIFT ensemble allocation and identity regression tests."""
import numpy as np
import pytest
from benchmark.sqdrift_ensemble import specs, member_specs
from benchmark.sqdrift_ensemble_evaluate import allocate, draw_members


def test_registered_ensemble_count_and_identity():
    methods = specs("standard")
    assert len(methods) == 12
    members = [m for s in methods for m in member_specs("problem", s)]
    assert len(members)*30 == 6480
    assert len(set(m["member_id"] for m in members)) == len(members)
    assert all(m["num_terms"] in (8, 32) for m in members)
    assert members == [m for s in methods for m in member_specs("problem", s)]


def test_fixed_allocation_not_multinomial_of_average_probability():
    allocation = allocate(11, 2)
    assert allocation == [6, 5] and sum(allocation) == 11
    # Mean-probability sampling has a random binomial split. Actual registered
    # per-member allocation deterministically yields6/5 in every repetition.
    for rep in range(20):
        ids, counts, records = draw_members([np.array([1., 0]), np.array([0., 1])], allocation, [rep], ["a", "b"])
        assert ids.tolist() == [0, 1] and counts.tolist() == [6, 5]
        assert sum(r["allocated_shots"] for r in records) == 11


def test_member_rng_reproducible_and_not_free_conditioning():
    probabilities = [np.array([.3, .7]), np.array([.8, .2])]
    a = draw_members(probabilities, allocate(100, 2), ["problem", 101], ["a", "b"])
    b = draw_members(probabilities, allocate(100, 2), ["problem", 101], ["a", "b"])
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[2][0]["seed"] != a[2][1]["seed"]
    assert a[1].sum() == 100
    with pytest.raises(ValueError):
        draw_members([np.array([.1, .1])], [10], [1], ["a"])
