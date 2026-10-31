import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from benchmark.integral_hamiltonian import IntegralHamiltonian
from benchmark.integral_connections import IntegralConnections
from benchmark.local_sampler_primitives import (LocalQE, excitation_between, observed_ci, ci_gradient,
    local_proposals, pack_counts64, unpack_counts64, valid_sector)


def test_uint62_counts_do_not_truncate_large_configurations():
    ids = [3, (1 << 39)+17, (1 << 60)+7]; counts = [1, 2, 100]
    a, b = unpack_counts64(pack_counts64(ids, counts))
    assert a.tolist() == ids and b.tolist() == counts and a.dtype == np.uint64
    for bad in ([3.5], [True]):
        with pytest.raises(ValueError): pack_counts64(bad, [1])
    with pytest.raises(ValueError): pack_counts64([3], [2.4])


def test_sparse_QE_generator_matches_actual_circuit_without_Fock_operator():
    n = 8; ids = [x for x in range(1 << n) if valid_sector(x, n, 2, 2)][:20]
    rng = np.random.default_rng(761); coeff = rng.normal(size=len(ids))+1j*rng.normal(size=len(ids)); coeff /= np.linalg.norm(coeff)
    initial = np.zeros(1 << n, complex); initial[ids] = coeff; eps = 1e-6
    for op in (LocalQE((0,), (4,), n), LocalQE((0, 1), (4, 5), n), LocalQE((1, 6), (3, 4), n)):
        expected = (Statevector(initial).evolve(op.circuit(eps)).data-Statevector(initial).evolve(op.circuit(-eps)).data)/(2*eps)
        action = op.action(dict(zip(ids, coeff))); actual = np.zeros_like(initial)
        for x, value in action.items(): actual[x] = value
        assert np.linalg.norm(actual-expected) < 1e-8
        assert all(valid_sector(x, n, 2, 2) for x in action)


def test_observed_only_CI_and_local_residual_proposals():
    h1 = np.array([[-1., .2], [.2, -.5]]); backend = IntegralHamiltonian(h1, np.zeros((2,)*4), .3)
    ci = observed_ci(backend, {}, 12, 1, 1)
    assert ci["selected_ids"] == [12] and ci["actual_dimension"] == 1
    proposals, scope = local_proposals(backend, IntegralConnections(backend, parent_cap=16), ci, 16, 4)
    assert len(proposals) == 2 and not scope["FCI_used"] and not scope["full_pool_constructed"]
    assert ci["selected_ids"] == [12]  # Proposal targets are not inserted as free observations.
    assert all(p["proposal_target_ids"] for p in proposals)
    ci2 = observed_ci(backend, {6: 4, 5: 20}, 12, 1, 1)
    assert ci2["selected_ids"] == [6, 12]  # Wrong Sz=beta/beta ID5 is not valid.
    hs = backend.matrix(range(16), range(16)); psi = np.zeros(16, complex); psi[12] = 1
    for p in proposals:
        op = p["operator"]; eps = 1e-5
        plus = Statevector(psi).evolve(op.circuit(eps)).data
        minus = Statevector(psi).evolve(op.circuit(-eps)).data
        numerical = (np.vdot(plus, hs@plus).real-np.vdot(minus, hs@minus).real)/(2*eps)
        assert ci_gradient(backend, op, ci) == pytest.approx(numerical, abs=1e-8)
    assert excitation_between(12, 6, 4) is not None
    assert excitation_between(12, 5, 4) is None  # Same N alone is insufficient.


def test_40qubit_local_pattern_has_constant_arity_no_full_pool():
    n = 40; hf = sum(1 << (n-1-q) for q in range(20))
    child = hf ^ (1 << 39) ^ (1 << 19)
    op = excitation_between(hf, child, n)
    assert op is not None and len(op.source+op.target) == 2
    action = op.action({hf: 1.})
    assert set(action) == {child} and all(x >= 1 << 19 for x in action)
    assert op.circuit(.1).num_qubits == 40
    # This is a primitive test, not a40qubit molecular scaling experiment.
