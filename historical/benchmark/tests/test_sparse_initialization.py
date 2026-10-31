import numpy as np
from qiskit.quantum_info import Statevector
from benchmark.sparse_initialization import prepare_two_configurations


def test_complex_signed_two_configuration_preparation():
    rng = np.random.default_rng(390)
    for n in (4, 8, 12, 16):
        for _ in range(4):
            ids = rng.choice(1 << n, 2, replace=False)
            amplitudes = rng.normal(size=2)+1j*rng.normal(size=2)
            amplitudes /= np.linalg.norm(amplitudes)
            qc = prepare_two_configurations(n, ids, amplitudes)
            target = np.zeros(1 << n, complex); target[ids] = amplitudes
            actual = Statevector.from_instruction(qc).data
            assert np.linalg.norm(actual-target) < 1e-11
            d = int(ids[0] ^ ids[1]).bit_count()
            assert qc.count_ops().get("cx", 0) == d-1
            assert qc.depth(filter_function=lambda x: x.operation.name == "cx") == int(np.ceil(np.log2(d)))


def test_large_register_circuit_without_statevector_or_fci():
    # A circuit and an algebraic construction certificate, not a 40-qubit simulation.
    first = (1 << 20)-1
    second = first ^ (1 << 0) ^ (1 << 4) ^ (1 << 21) ^ (1 << 25)
    qc = prepare_two_configurations(40, [first, second], [0.8, -0.6])
    assert qc.num_qubits == 40 and qc.count_ops()["cx"] == 3
