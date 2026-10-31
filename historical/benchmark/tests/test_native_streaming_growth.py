import json
import numpy as np
import pytest
from benchmark.molecular_problem_no_fci import MolecularSpec, build_problem
from benchmark.native_streaming_growth import GrowthSpec, run_growth, advance
from benchmark.ansatz import make_pool, hf_state, selection_gradients


@pytest.fixture
def ample_native_test_storage(monkeypatch):
    # Hermetic *physics* tests must not depend on unrelated host disk usage.
    # Patch only this module's resource adapter, not stdlib shutil globally.
    # Production's frozen3GiB guard remains intact and is tested separately.
    from types import SimpleNamespace
    import benchmark.native_streaming_growth as native_module
    monkeypatch.setattr(native_module,"shutil",SimpleNamespace(disk_usage=lambda path:SimpleNamespace(free=4*2**30)))


def test_actual_constructor_has_no_FCI_and_current_gradient(tmp_path, monkeypatch, ample_native_test_storage):
    import pyscf.fci.direct_spin1
    def forbidden(*a, **k):
        raise AssertionError("FCI must not enter actual construction")
    monkeypatch.setattr(pyscf.fci.direct_spin1, "kernel", forbidden)
    p = build_problem(MolecularSpec("H2", .85), tmp_path/"physical", True)
    native = run_growth(p, GrowthSpec("ceo_ovp", "native_terminal", max_steps=3), tmp_path/"native")
    gradient = run_growth(p, GrowthSpec("ceo_ovp", "gradient", max_steps=3), tmp_path/"gradient")
    assert native["status"] == gradient["status"] == "NATIVE_GRADIENT_CONVERGED"
    assert native["physical_checkpoint_id"] == gradient["physical_checkpoint_id"]
    assert native["pilot_raw_model_shots"] == gradient["pilot_raw_model_shots"] == 0
    assert native["current_pool_gradient_norm"] < 1e-3 and native["objective_evaluations"] > 0
    cps = [json.loads(q.read_text()) for q in sorted((tmp_path/"native/steps").glob("*.json"))]
    assert len(cps) == 2 and all(not c["FCI_used"] for c in cps)
    # Only one current full-pool evaluation per committed step, not redundant
    # preappend+postappend evaluations. H2 OVP pool has4operators.
    assert native["pool_gradient_components_computed"] == len(cps)*make_pool("ceo_ovp", 4).size
    assert run_growth(p, GrowthSpec("ceo_ovp", "native_terminal", max_steps=3), tmp_path/"native") == native
    bad = {**p, "meta": {**p["meta"], "ground_state_reference_energy_Ha": -1.}}
    with pytest.raises(ValueError, match="no-FCI"):
        run_growth(bad, GrowthSpec("ceo_ovp", "native_terminal"), tmp_path/"bad")


def test_extension_reuses_only_current_gradient(tmp_path):
    p = build_problem(MolecularSpec("H4", 1.25), tmp_path/"physical", True)
    pool = make_pool("ceo_mvp", 8); psi = hf_state(8, p["meta"]["hf_id"])
    g = selection_gradients(pool, psi, p["h"])
    a = advance(p, pool, [], np.array([])); b = advance(p, pool, [], np.array([]), g)
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:4] == b[2:4]
    assert a[-1] == pool.size and b[-1] == 0
    assert np.array_equal(a[4], g) and np.array_equal(b[4], g)


def test_actual_H4_stopped_prefix_and_paid_counts(tmp_path, ample_native_test_storage):
    p = build_problem(MolecularSpec("H4", 1.25), tmp_path/"physical", True)
    run_growth(p, GrowthSpec("ceo_ovp", "native_terminal", max_steps=8), tmp_path/"native")
    stopped = run_growth(p, GrowthSpec("ceo_ovp", "discovery_plateau", max_steps=8), tmp_path/"stopped")
    a = {q.name: json.loads(q.read_text()) for q in (tmp_path/"native/steps").glob("*.json")}
    spent = 0
    for q in (tmp_path/"stopped/steps").glob("*.json"):
        b = json.loads(q.read_text()); n = a[q.name]
        assert b["selected_operators"] == n["selected_operators"]
        assert np.allclose(b["theta"], n["theta"], atol=1e-9, rtol=0)
        assert abs(b["energy_Ha"]-n["energy_Ha"]) < 1e-9
        if b["pilot"]:
            ev = b["pilot"]; assert sum(n for _, n in ev["raw_counts"]) == 64
            spent += 64
            assert ev["frame"]["cumulative_pilot_raw_shots"] == spent
    assert stopped["pilot_raw_model_shots"] == spent > 0
    if stopped["status"] == "POLICY_STOPPED":
        assert not (tmp_path/"stopped/steps"/f"step-{stopped['last_step']+1:04d}.json").exists()


def test_low_storage_guard_preserves_committed_evidence_and_never_reports_completion(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import benchmark.native_streaming_growth as native_module
    p=build_problem(MolecularSpec("H2",.85),tmp_path/"physical",True)
    monkeypatch.setattr(native_module,"shutil",SimpleNamespace(disk_usage=lambda path:SimpleNamespace(free=0)))
    root=tmp_path/"guarded"
    with pytest.raises(RuntimeError,match="Resource boundary; committed actual trajectory preserved"):
        run_growth(p,GrowthSpec("ceo_ovp","native_terminal",max_steps=3),root)
    checkpoint=root/"steps/step-0000.json"
    assert checkpoint.exists() and not (root/"result.json").exists()
    cp=json.loads(checkpoint.read_text());cfg=json.loads((root/"config.json").read_text())
    assert cfg["free_floor_bytes"]==3*2**30 and not cp["FCI_used"]
    assert cp["step"]==0 and cp["terminal"] is None
