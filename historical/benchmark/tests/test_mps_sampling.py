import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from benchmark.mps_sampling import construct_mps, sample_mps


def test_qiskit_mps_endianness_nonadjacent_complex_gate_parity():
    qc = QuantumCircuit(8)
    qc.x(0); qc.h(3); qc.ry(.7, 6); qc.cx(3, 0); qc.rz(.31, 0)
    qc.cx(6, 3); qc.ry(-.23, 0); qc.sdg(6); qc.cx(0, 7)
    state, meta = construct_mps(qc, max_bond=32, cutoff=0)
    exact = Statevector.from_instruction(qc).data
    actual = np.asarray(state.psi.to_dense()).reshape(-1)
    assert np.allclose(abs(actual)**2, abs(exact)**2, atol=1e-10)
    assert abs(abs(np.vdot(actual, exact))-1) < 1e-10
    assert abs(meta["retained_norm_squared"]-1) < 1e-10
    ids, counts, _ = sample_mps(state, 200, 101)
    ids2, counts2, _ = sample_mps(state, 200, 101)
    assert np.array_equal(ids, ids2) and np.array_equal(counts, counts2)
    assert counts.sum() == 200
    ids3, counts3, partial = sample_mps(state, 200, 101, seconds_cap=-1)
    assert len(ids3) == len(counts3) == partial["raw_simulated_shots"] == 0
    assert partial["status"] == "RESOURCE_CAPPED_PARTIAL_COUNTS_PRESERVED"


def test_40_qubit_actual_entangled_circuit_samples_without_dense_state(monkeypatch):
    import quimb.tensor as qtn
    qc = QuantumCircuit(40); qc.h(0)
    for q in range(1, 40):
        qc.cx(q-1, q)
    def forbidden(*args, **kwargs):
        raise AssertionError("Full dense state is forbidden")
    monkeypatch.setattr(qtn.MatrixProductState, "to_dense", forbidden)
    state, meta = construct_mps(qc, max_bond=4, cutoff=1e-12)
    ids, counts, sampling = sample_mps(state, 40, 211)
    assert set(ids) == {0, (1 << 40)-1}
    assert counts.sum() == sampling["raw_simulated_shots"] == 40
    assert meta["max_observed_final_bond"] <= 2 and not meta["full_statevector_allocated"]
