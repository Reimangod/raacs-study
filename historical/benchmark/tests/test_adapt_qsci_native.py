import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import expm, eigh
from qiskit.quantum_info import SparsePauliOp, Statevector
from benchmark.adapt_qsci_native import (Pauli, paper_pool, NamedHamiltonian, gradient_data,
    standard_angle, hf_gradient, hf_root, hf_angle, hf_matrices, update_state, circuit, prepare_target)


def test_exact_paper_pool_and_pauli_phase():
    assert len(paper_pool(8)) == 164 and len(paper_pool(12)) == 1050
    rng = np.random.default_rng(4)
    for p in paper_pool(4):
        state = rng.normal(size=16)+1j*rng.normal(size=16); state /= np.linalg.norm(state)
        actual = update_state(state, p, .237)
        expected = expm(1j*.237*SparsePauliOp(p.label).to_matrix()) @ state
        assert np.allclose(actual, expected, atol=1e-12)


def test_commutator_external_connections_and_angle():
    # CI vector is an eigenvector of H_S; projecting P first would wrongly give0.
    H = np.array([[0, .1, .2, 0], [.1, 1, 0, .2], [.2, 0, 2, .1], [0, .2, .1, 3.]])
    backend = NamedHamiltonian(sparse.csr_matrix(H)); ids = np.array([0]); c = np.array([1.])
    pool = [Pauli("IY"), Pauli("YI")]
    rows = gradient_data(backend, ids, c, pool)
    for p, row in zip(pool, rows):
        P = SparsePauliOp(p.label).to_matrix(); full = np.eye(4)[:, 0]
        expected = np.vdot(full, 1j*(H @ P-P @ H) @ full).real
        assert abs(row["C"]-expected) < 1e-12 and abs(row["C"]) > .1
        pi, pc = p.act(ids, c)
        A, B = backend.expectation(ids, c), backend.expectation(pi, pc)
        theta, energy = standard_angle(A, B, row["C"])
        v = expm(1j*theta*P) @ full
        assert abs(np.vdot(v, H @ v).real-energy) < 1e-12
        grid = [np.vdot(expm(1j*t*P) @ full, H @ (expm(1j*t*P) @ full)).real for t in np.linspace(-np.pi/2, np.pi/2, 101)]
        assert energy <= min(grid)+1e-12


def test_hf_gradient_with_nonzero_overlap_term():
    rng = np.random.default_rng(819)
    H = rng.normal(size=(16, 16)); H = H+H.T-20*np.eye(16)
    c = rng.normal(size=16); c[0] = 0; c /= np.linalg.norm(c)
    backend = NamedHamiltonian(sparse.csr_matrix(H)); ids = np.flatnonzero(c); values = c[ids]
    for p, data in zip(paper_pool(4), gradient_data(backend, ids, values, paper_pool(4), hf=0)):
        pi, pc = p.act(ids, values)
        coeff = dict(EHF=H[0, 0], A=backend.expectation(ids, values), B=backend.expectation(pi, pc),
                     v=float((H[0, ids] @ values).real), **data)
        g = hf_gradient(**{k: v for k, v in coeff.items() if k != "B"})
        eps = 1e-5
        fd = (hf_root(eps, **coeff)-hf_root(-eps, **coeff))/(2*eps)
        assert abs(g-fd) < 2e-7
        theta = .193
        P = SparsePauliOp(p.label).to_matrix(); rotated = expm(1j*theta*P) @ c
        basis = np.column_stack((np.eye(16)[:, 0], rotated))
        expected = eigh(basis.conj().T @ H @ basis, basis.conj().T @ basis, eigvals_only=True)[0]
        assert abs(hf_root(theta, **coeff)-expected) < 1e-10
        opt, energy, calls = hf_angle(**coeff)
        assert energy <= hf_root(0., **coeff)+1e-10 and calls > 100


def test_actual_preparation_and_qiskit_update():
    n, hf = 4, 12
    target_ids = np.array([3, 6]); target_c = np.array([.8, -.6])
    state, rotations, fidelity = prepare_target(n, hf, target_ids, target_c, paper_pool(n))
    assert fidelity > 1-1e-10
    quantum = Statevector.from_instruction(circuit(n, hf, rotations)).data
    assert abs(abs(np.vdot(quantum, state))-1) < 1e-10
    assert len(rotations) >= 2  # Not a free target state.


def test_singular_overlap_has_no_fake_second_dimension():
    assert np.isclose(hf_root(np.pi/2, EHF=-1., A=0., B=-1., C=0., v=0., rH=-1., rS=1.), -1.)
    with pytest.raises(RuntimeError, match="DEGENERATE"):
        hf_gradient(EHF=-1., A=-1., C=0., v=0., rH=0., rS=0.)


def test_growth_raw_shots_and_prefix_resume(tmp_path):
    import json
    import shutil
    from benchmark.common import ROOT
    from benchmark.chemistry import load_reference
    from benchmark.qsci_growth import grow
    ref = load_reference(ROOT / "results/standard-v1/references/H2_R0.7")
    config = {"max_steps": 4, "trajectory_seconds": 120, "worker_ram_bytes": 2*2**30, "free_floor_bytes": 0}
    first = grow(ref, "adapt_qsci_original", 1000, 32, 101, tmp_path / "a", config)
    prefix = tmp_path / "b"; prefix.mkdir()
    for name in ("step-0000.json", "counts-0000.npz"):
        shutil.copy(tmp_path / "a" / name, prefix / name)
    resumed = grow(ref, "adapt_qsci_original", 1000, 32, 101, prefix, config)
    assert resumed["status"] == first["status"]
    assert resumed["cumulative"]["raw_simulated_shots"] == first["cumulative"]["raw_simulated_shots"]
    for f in (tmp_path / "a").glob("step-*.json"):
        a = json.loads(f.read_text()); b = json.loads((prefix / f.name).read_text())
        assert a["rotations"] == b["rotations"] and a["selected_ids"] == b["selected_ids"]
        assert a["native_qsci_energy"] == b["native_qsci_energy"]
    assert first["cumulative"]["raw_simulated_shots"] == first["checkpoints"]*1000
    # Terminal checkpoint is sufficient after a crash before terminal-file commit.
    (prefix / "terminal.json").unlink()
    recovered = grow(ref, "adapt_qsci_original", 1000, 32, 101, prefix, config)
    assert recovered["status"] == first["status"] and recovered["recovered_terminal_from_checkpoint"]
