import copy
import json
import numpy as np
import pytest
from benchmark.common import ROOT, file_digest
from benchmark.chemistry import load_reference
from benchmark.config import Case
from benchmark.strong_baselines.lucj import Objective, BudgetReached, initial_parameters, run_cell, method_specs


SOURCE = ROOT / "results/standard-v1"


def budget(**kw):
    return {"max_evals": 300, "cell_seconds": 120, "worker_ram_bytes": 2*2**30,
            "pilot_shots": 256, "maxiter": 2, "qsci_proposals": 2, **kw}


def test_parameter_roundtrip_and_no_fci_selection(tmp_path):
    ref = load_reference(SOURCE / "references/H2_R0.7")
    spec = next(s for s in method_specs() if s["name"] == "lucj_mp2_2_fixed")
    before = file_digest(SOURCE / "references/H2_R0.7/reference.npz")
    x, pairs, init = initial_parameters(Case("H2", .7), ref, spec, SOURCE, tmp_path)
    d = tmp_path / "objective"; d.mkdir()
    obj = Objective(ref, spec, pairs, d, budget())
    energy = obj(x)
    signed = obj.state(x)*obj.signs
    h = ref["h"][obj.ids][:, obj.ids]
    assert np.isclose(energy, np.vdot(signed, h @ signed).real, atol=1e-11)
    assert np.isclose(np.linalg.norm(obj.state(x)), 1)
    obj.meta["fci_energy"] = 100000.
    assert obj(x) == energy  # Oracle metadata cannot change objective value.
    assert file_digest(SOURCE / "references/H2_R0.7/reference.npz") == before
    assert init["dependency_new_seconds"] > 0
    obj.db.close(); obj.cache_db.close()


def test_exact_call_cap_and_gradient_accounting(tmp_path):
    ref = load_reference(SOURCE / "references/H2_R0.7")
    spec = dict(name="test-vqe", init="random", layers=1, objective="vqe", seed=101)
    x, pairs, _ = initial_parameters(Case("H2", .7), ref, spec, SOURCE, tmp_path)
    obj = Objective(ref, spec, pairs, tmp_path, budget(max_evals=2*len(x)+1))
    obj(x); grad = obj.gradient(x)
    assert len(grad) == len(x) and np.all(np.isfinite(grad))
    assert obj.calls == 2*len(x)+1 and obj.gradients == 1
    assert obj.db.execute("SELECT count(*) FROM calls").fetchone()[0] == obj.calls
    with pytest.raises(BudgetReached):
        obj(x)
    obj.db.close(); obj.cache_db.close()


def test_capped_result_and_resume_preserve_work(tmp_path):
    ref = load_reference(SOURCE / "references/H2_R0.7")
    spec = dict(name="test-capped-vqe", init="random", layers=1, objective="vqe", seed=211)
    result = run_cell(Case("H2", .7), ref, spec, SOURCE, tmp_path, budget(max_evals=3))
    assert result["status"] == "BOUNDED_SEARCH_TERMINATED"
    assert result["objective_evaluations"] == 3
    assert not result["optimizer_success"] and result["stationary"] is None
    assert result["state_parity_error"] < 1e-8
    again = run_cell(Case("H2", .7), ref, spec, SOURCE, tmp_path, budget(max_evals=3))
    assert result == again


def test_qsci_pilot_cost_and_independent_label(tmp_path):
    ref = load_reference(SOURCE / "references/H2_R0.7")
    spec = dict(name="test-qsci", init="random", layers=1, objective="qsci", seed=101)
    result = run_cell(Case("H2", .7), ref, spec, SOURCE, tmp_path, budget())
    assert result["status"] == "BOUNDED_SEARCH_TERMINATED"
    assert result["simulated_construction_shots"] == 3*256
    assert result["qsci_pilot_selection_bias"] and not result["fci_used_for_selection"]
    assert result["gradient_inf_at_returned_point"] is None
