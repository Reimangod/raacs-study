from dataclasses import replace
import pytest
from benchmark.online_stopping_rules import ObservedFrame, RuleSpec, OnlineStopRule, consume_until_stop, remaining_final_budget


def frame(step, energy=-1., **kwargs):
    return ObservedFrame("problem", "trajectory", step, energy, 4, 0, step*64, .7, step*13, step*9, float(step), **kwargs)


def test_stream_never_reads_a_future_checkpoint():
    consumed = []
    def observations():
        for i in range(1, 4):
            consumed.append(i); yield frame(i)
        raise AssertionError("Reading a future low-energy checkpoint would leak outcome information")
    decision = consume_until_stop(RuleSpec("energy_plateau"), observations())
    assert decision["stop"] and decision["step"] == 3 and consumed == [1, 2, 3]
    assert not decision["FCI_used"] and decision["cumulative_pilot_raw_shots"] == 192


def test_unknown_residual_and_gradient_are_not_zero():
    assert not consume_until_stop(RuleSpec("residual_pt2"), [frame(i) for i in range(1, 5)])["stop"]
    assert not consume_until_stop(RuleSpec("gradient"), [frame(i) for i in range(1, 5)])["stop"]
    policy = OnlineStopRule(RuleSpec("gradient"))
    assert policy.observe(frame(1, current_pool_gradient_norm=1e-5))["stop"]
    with pytest.raises(RuntimeError): policy.observe(frame(2))


def test_shot_and_cost_order_and_forbidden_reference_labels():
    assert remaining_final_budget(1000, 192) == 808
    with pytest.raises(ValueError): remaining_final_budget(1000, 1000)
    with pytest.raises(TypeError): ObservedFrame(**dict(frame(1).__dict__, fci_energy=-2.))
    p = OnlineStopRule(RuleSpec("discovery_plateau")); p.observe(frame(2))
    with pytest.raises(ValueError): p.observe(frame(1))
    with pytest.raises(ValueError): replace(frame(1), qsci_energy_Ha=float("nan"))
    with pytest.raises(ValueError): replace(frame(1), current_pool_gradient_norm=-1.)
    p = OnlineStopRule(RuleSpec("energy_plateau")); p.observe(frame(1))
    with pytest.raises(ValueError): p.observe(replace(frame(2), problem_id="different-problem"))


def test_reduced_gate_count_does_not_fake_small_improvement_per_added_gate():
    frames = [replace(frame(i), cnot_count=100-i) for i in range(1, 5)]
    assert not consume_until_stop(RuleSpec("improvement_per_cnot"), frames)["stop"]
