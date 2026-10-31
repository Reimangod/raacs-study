import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.linalg import expm
from benchmark.time_evolution_primitives import pauli_hamiltonian, apply_sequence, evolution_circuit, trotter_sequence, qdrift_sequence
from benchmark.chemistry import load_reference
from benchmark.common import ROOT


def test_integral_jw_reconstruction_no_fci_and_actual_trotter_circuit():
    ref = load_reference(ROOT / "results/standard-v1/references/H2_R0.7")
    terms, constant = pauli_hamiltonian(ref["h1"], ref["eri"], ref["meta"]["hf_id"], ref["meta"]["rhf_energy"])
    h = SparsePauliOp.from_list(terms+[("I"*4, constant)]).to_matrix()
    assert np.allclose(h, ref["h"].toarray(), atol=1e-10)
    v = np.zeros(16, complex); v[ref["meta"]["hf_id"]] = 1
    sequence = trotter_sequence(terms, .5, 2)
    state = apply_sequence(v, sequence)
    actual = Statevector.from_instruction(evolution_circuit(4, ref["meta"]["hf_id"], sequence)).data
    assert np.allclose(state, actual, atol=1e-10)
    coarse = apply_sequence(v, trotter_sequence(terms, .5, 1))
    fine = apply_sequence(v, trotter_sequence(terms, .5, 16))
    exact = expm(-.5j*(h-constant*np.eye(16))) @ v
    assert np.linalg.norm(fine-exact) < np.linalg.norm(coarse-exact)


def test_qdrift_norm_probabilities_angles_and_determinism():
    terms = [("XX", .25), ("ZI", -.75)]
    first, indices, norm = qdrift_sequence(terms, .8, 10000, np.random.default_rng(17))
    second, _, _ = qdrift_sequence(terms, .8, 10000, np.random.default_rng(17))
    assert first == second and norm == 1
    assert abs(indices.count(0)/len(indices)-.25) < .02
    assert all(np.isclose(abs(angle), .8/10000) for _, angle in first)
    assert all(angle < 0 for label, angle in first if label == "ZI")
