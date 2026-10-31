from benchmark.random_structure_report import resource_kind


def test_classical_counts_not_charged_as_qpu_or_free_quantum_preparation():
    assert resource_kind("uniform_valid_classical") == "CLASSICAL_CONFIGURATION_PROPOSALS"
    assert resource_kind("random_ceo_s101") == "SIMULATED_QUANTUM_CIRCUIT_READOUTS"
