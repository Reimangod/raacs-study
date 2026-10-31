import numpy as np
from benchmark.nonvariational_baselines import specs, orbital_control
from benchmark.nonvariational_audit import named_pauli_entries
from benchmark.common import ROOT
from benchmark.chemistry import load_reference
from benchmark.ansatz import verify_circuit_state


def test_registered_no_best_seed_selection_and_physical_orbital_rotation():
    assert len(specs("standard")) == 18
    assert len({x["name"] for x in specs("standard")}) == 18
    ref = load_reference(ROOT / "results/standard-v1/references/H4_R1.1")
    state, qc, details, params, slots = orbital_control(ref["meta"], 2, np.random.default_rng(101))
    assert verify_circuit_state(qc, state) < 1e-9
    assert abs(state[ref["meta"]["hf_id"]]) < .999
    assert params == slots == len(details["angles"])


def test_named_pauli_complex_phase_against_independent_matrix():
    from qiskit.quantum_info import SparsePauliOp
    terms = [("YX", .23), ("ZZ", -.17)]
    ids = [0, 1, 3]
    actual = named_pauli_entries(terms, .8, ids)
    expected = SparsePauliOp.from_list(terms+[("II", .8)]).to_matrix()[np.ix_(ids, ids)]
    assert np.allclose(actual, expected)
