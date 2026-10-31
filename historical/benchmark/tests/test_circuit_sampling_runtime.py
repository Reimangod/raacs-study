import numpy as np
import pytest
from qiskit import QuantumCircuit
from benchmark.circuit_sampling_runtime import CircuitCountSampler, SamplingSpec
from benchmark.local_sampler_primitives import LocalQE


def test_actual_local_QE_MPS_samples_known_support_and_cache():
    prep = QuantumCircuit(8)
    for wire in (4, 5, 6, 7): prep.x(wire)
    prep = LocalQE((0, 1), (4, 5), 8).append(prep, np.pi/4)
    sampler = CircuitCountSampler(SamplingSpec(max_bond=16))
    ids, counts, meta = sampler.draw(prep, 64, 771, 1, 1)
    assert set(map(int, ids)) <= {240, 60} and int(counts.sum()) == 64
    assert set(map(int, ids)) == {240, 60}
    assert not meta["full_statevector_exposed_to_algorithm"] and not meta["FCI_used"]
    assert meta["hardware_shots"] == 0 and meta["sample_metadata"]["status"] == "COMPLETED"
    assert sampler.MPS_builds == 1 and sampler.exact_small_builds == 0
    _, _, again = sampler.draw(prep, 8, 772, 1, 1)
    assert again["prepared_simulator_cache_hit"] and sampler.MPS_builds == 1
    assert sampler.raw_model_shots == 72


def test_exact_qualification_not_silently_used_for_large_register():
    sampler = CircuitCountSampler(SamplingSpec(backend="exact_small"))
    with pytest.raises(ValueError, match="16qubits"):
        sampler.draw(QuantumCircuit(20), 4, 1)
