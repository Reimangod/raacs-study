import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.linalg import eigh
from benchmark.config import Case, cases_for
from benchmark.chemistry import build_reference
from benchmark.ansatz import (make_pool, hf_state, state_from_parameters, energy_and_gradient,
                              build_circuit, verify_circuit_state,selection_gradients,apply_op)
from scipy.sparse.linalg import expm_multiply
from benchmark.metrics import discovery_probabilities, analytic_metrics


@pytest.fixture(scope="session")
def reference(tmp_path_factory):
    return build_reference(Case("H4",1.1), tmp_path_factory.mktemp("chem"))


def test_all30_cases():
    assert len(cases_for("full")) == 30
    assert len(cases_for("standard")) == 15
    assert Case("BeH2",2).geometry()[0][1][2] == -2


def test_hf_fci_mapping_and_normalization(reference):
    r = reference
    m = r["meta"]
    assert abs(r["h"][m["hf_id"],m["hf_id"]]-m["rhf_energy"]) < 1e-8
    assert abs(r["weights"].sum()-1) < 1e-10
    hs = r["h"][r["ids"]][:,r["ids"]].toarray()
    assert abs(eigh(hs,eigvals_only=True)[0]-m["fci_energy"]) < 1e-8
    assert np.linalg.norm(hs @ r["amplitudes"]-m["fci_energy"]*r["amplitudes"]) < 1e-7


@pytest.mark.parametrize("method",["ceo_ovp","ceo_mvp","qeb","qubit","gsd"])
def test_pool_state_parity_and_adjoint_gradient(reference,method):
    r = reference
    pool = make_pool(method,r["meta"]["n_qubits"])
    initial = hf_state(pool.n,r["meta"]["hf_id"])
    indices = [0,pool.size//2,pool.size-1]
    theta = np.array([.19,-.23,.37])
    state = state_from_parameters(pool,indices,theta,initial)
    qc = build_circuit(pool,indices,theta,r["meta"]["hf_id"])
    assert verify_circuit_state(qc,state) < 1e-8
    _, grad = energy_and_gradient(pool,indices,theta,initial,r["h"])
    numeric = approx_derivative(lambda x: energy_and_gradient(pool,indices,x,initial,r["h"])[0], theta).ravel()
    assert np.allclose(grad,numeric,atol=1e-7)
    actual=selection_gradients(pool,state,r["h"])
    independent=np.array([2*np.vdot(r["h"]@state,pool.get_imp_op(i)@state).real for i in range(pool.size)])
    assert np.allclose(actual,independent,atol=1e-10)
    for index in indices:
        assert np.allclose(apply_op(pool,index,.31,state),
                           expm_multiply(.31*pool.get_imp_op(index),state,traceA=0.),atol=1e-10)
    if method != "qubit":
        assert abs(np.abs(state[r["ids"]])@np.abs(state[r["ids"]])-1) < 1e-9


def test_coverage_exact_expectation(reference):
    r = reference
    p = np.zeros(r["h"].shape[0]); p[r["meta"]["hf_id"]] = 1
    rows = analytic_metrics(p,r)
    assert all(abs(x["expected_unique"]-1)<1e-12 for x in rows)
    assert all(x["nonhf_weighted_coverage"] == 0 for x in rows)
    assert np.allclose(discovery_probabilities([0,.5,1],2),[0,.75,1])
    with pytest.raises(ValueError):
        analytic_metrics(p*.9,r)
