import numpy as np
from scipy import sparse
from benchmark.adapt_qsci_native import NamedHamiltonian
from benchmark.selected_ci_native import external_diagnostics, grow_space, solve


def test_external_residual_is_not_internal_ritz_residual():
    H = np.array([[0., .3, .1], [.3, 1., .2], [.1, .2, 2.]])
    b = NamedHamiltonian(sparse.csr_matrix(H))
    S = [0, 1]; E, c, internal = solve(b, S)
    out = external_diagnostics(b, S, c, E)
    full = np.zeros(3); full[S] = c
    assert internal < 1e-12
    assert out["external_residual_norm"] > 0
    assert np.isclose(out["external_residual_norm"], np.linalg.norm(H @ full-E*full))
    assert out["pt2_raw"] < 0


def test_hci_max_is_not_coherent_cipsi_sum():
    H = np.array([[0., 0., .2], [0., 0., .2], [.2, .2, 1.]])
    b = NamedHamiltonian(sparse.csr_matrix(H))
    out = external_diagnostics(b, [0, 1], np.array([1., -1.])/np.sqrt(2), 0.)
    assert out["hci_importance"][0] > .1
    assert abs(out["residual"][0]) < 1e-12
    assert out["cipsi_coefficient_importance"][0] < 1e-12


def test_intruders_unknown_raw_pt2_and_nested_variational_spaces():
    H = np.array([[0., .1, .2], [.1, 0., .3], [.2, .3, 1.]])
    b = NamedHamiltonian(sparse.csr_matrix(H))
    out = external_diagnostics(b, [0], [1.], 0.)
    assert out["pt2_raw"] is None and out["near_zero_denominators"] == 1
    for criterion in ("hci_importance", "cipsi_coefficient_importance", "en_energy_importance"):
        history, status = grow_space(b, 0, 3, criterion)
        assert np.all(np.diff([r["energy"] for r in history]) <= 1e-10)
        assert all(set(a["selected"]) <= set(z["selected"]) for a, z in zip(history, history[1:]))
        assert abs(history[-1]["energy"]-np.linalg.eigvalsh(H)[0]) < 1e-12
