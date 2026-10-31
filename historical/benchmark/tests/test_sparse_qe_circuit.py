import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from benchmark.local_sampler_primitives import LocalQE
from benchmark.sparse_qe_circuit import SparseQECircuit
from benchmark.parameter_control_primitives import vqe_update


def test_physical_sparse_simulation_noncommuting_circuit_and_adjoint_gradient():
    from benchmark.integral_hamiltonian import IntegralHamiltonian
    n = 8; hf = 240
    ops = [LocalQE((0, 1), (4, 5), n), LocalQE((2,), (4,), n), LocalQE((1, 3), (5, 7), n)]
    rng = np.random.default_rng(88); h1 = rng.normal(size=(4, 4)); h1 = (h1+h1.T)/2
    backend = IntegralHamiltonian(h1, np.zeros((4,)*4), 0.)
    sim = SparseQECircuit(n, hf, ops, backend=backend); theta = [.31, -.27, .19]
    qc = QuantumCircuit(n)
    for q in range(n):
        if hf >> q & 1: qc.x(q)
    for op, angle in zip(ops, theta): qc = op.append(qc, angle)
    exact = Statevector.from_instruction(qc).data
    fresh = np.zeros(1 << n, complex); fresh[sim.ids.astype(np.int64)] = sim.state(theta)
    assert np.allclose(exact, fresh, atol=1e-11)
    E, grad = sim.energy_gradient(theta)
    assert E == pytest.approx(np.vdot(fresh, backend.matrix(range(1 << n), range(1 << n))@fresh).real, abs=1e-11)
    eps = 1e-6
    for j in range(len(theta)):
        a = np.array(theta); b = a.copy(); a[j] += eps; b[j] -= eps
        finite = (sim.energy_gradient(a)[0]-sim.energy_gradient(b)[0])/(2*eps)
        assert grad[j] == pytest.approx(finite, abs=1e-8)
    assert len(sim.ids) <= 8


def test_44qubit_physical_count_mode_never_expands_full_Fock_space(monkeypatch):
    def forbidden(*a, **k): raise AssertionError("No full Statevector")
    monkeypatch.setattr(Statevector, "from_instruction", forbidden)
    n = 44; hf = (1 << 43)+(1 << 42)
    sim = SparseQECircuit(n, hf, [LocalQE((0, 1), (40, 41), n)])
    ids, ns = sim.counts([np.pi/4], 128, 211)
    assert len(sim.ids) == 2 and int(ns.sum()) == 128


def test_one_block_full_controls_track_active_and_full_gradients_separately():
    from benchmark.integral_hamiltonian import IntegralHamiltonian
    n = 4; ops = [LocalQE((0,), (2,), n), LocalQE((1,), (3,), n)]
    backend = IntegralHamiltonian(np.array([[-1., .3], [.3, -.2]]), np.zeros((2,)*4), 0.)
    sim = SparseQECircuit(n, 12, ops, backend=backend)
    initial = [.4, .4]
    one = vqe_update(sim, initial, "one_parameter_vqe"); full = vqe_update(sim, initial, "full_vqe")
    assert one["theta"][0] == initial[0] and one["n_optimized_variables"] == 1
    assert one["n_independent_ansatz_parameters"] == 2 and full["n_optimized_variables"] == 2
    assert full["energy_Ha"] <= one["energy_Ha"]+1e-9
    assert full["actual_objective_evaluations"] == len(full["trace"])
    capped = vqe_update(sim, initial, "full_vqe", max_calls=1)
    assert capped["status"] == "OBJECTIVE_OR_TIME_CAP" and capped["actual_objective_evaluations"] == 1
