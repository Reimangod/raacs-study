from benchmark.raacs_comparison_cost_overlay import source_build_cost


def test_parent_conditioned_random_angles_do_not_make_topology_free():
    cost, reason = source_build_cost("ceo_random_101", {"config": {"parent_checkpoint_id": "trained-CEO-parent"}})
    assert cost is None and "UPSTREAM_TRAINED_TOPOLOGY" in reason


def test_HF_and_classical_fixed_LUCJ_differ_from_VQE_growth():
    assert source_build_cost("hf", {"config": {"parent_checkpoint_id": None}})[0] == 0
    assert source_build_cost("lucj_2", {"config": {"parent_checkpoint_id": None}})[0] == 0
    assert source_build_cost("ceo_ovp", {"config": {}})[0] is None
