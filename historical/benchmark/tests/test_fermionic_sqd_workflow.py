import numpy as np
import pytest
from benchmark.fermionic_sqd_workflow import full_fermionic_sqd, FermionicSQDSpec, cartesian_ids


def test_cartesian_spin_bit_order_is_interleaved_big_endian():
    assert cartesian_ids([1, 2], [1, 2], 2) == [3, 6, 9, 12]
    assert cartesian_ids([1], [2], 3) == [36]


def test_official_full_fermionic_loop_actual_integrals_without_FCI_labels():
    from benchmark.common import ROOT
    folder = ROOT/"results/standard-v1/references/H2_R0.7"
    if not folder.exists(): pytest.skip("Molecular reference integrals unavailable")
    import json
    from benchmark.resource_adaptive_sampler import integral_problem
    backend, info = integral_problem(folder)
    out = full_fermionic_sqd(backend.h1, backend.eri, backend.core, info["hf_id"], 1, 1,
        [12, 3, 0, 15], [40, 30, 15, 15], FermionicSQDSpec(max_spin_strings=2, batches=2, max_iterations=3, samples_per_batch=8))
    assert not out["FCI_labels_used"] and not out["full_FCI_solve"] and out["hardware_shots"] == 0
    assert out["raw_model_shots"] == 100 and out["raw_valid_sector_fraction"] == .7
    assert out["dimension"] <= 4 and len(out["solver_trace"]) >= 2
    assert out["new_quantum_shots_for_recovery"] == 0 and out["recovery_trace"]
    assert all(abs(x["energy_Ha"]-x["independent_named_CI_energy_Ha"]) < 1e-7 for x in out["solver_trace"])


def test_raw_counts_are_not_rounded_or_zero_cost_fabrications():
    with pytest.raises(ValueError, match="Integer"):
        full_fermionic_sqd(np.eye(2), np.zeros((2,)*4), 0., 12, 1, 1, [12], [1.5])
