import json
import sqlite3
from dataclasses import replace
import numpy as np
import pytest
from benchmark.resource_adaptive_sampler import integral_problem, run_sampler, RAACSSpec, allocation_counts
from benchmark.circuit_sampling_runtime import SamplingSpec


def physical_source(folder, oracle_value):
    folder.mkdir(parents=True, exist_ok=True)
    h1 = np.array([[-1., .2], [.2, -.5]])
    np.savez(folder/"reference.npz", h1=h1, eri=np.zeros((2,)*4), weights=np.array([oracle_value]))
    (folder/"reference.json").write_text(json.dumps({"n_qubits": 4, "nalpha": 1, "nbeta": 1,
        "hf_id": 12, "rhf_energy": -2., "fci_energy": oracle_value, "request_digest": "ignored-metadata-ID"}))


def test_FCI_label_and_archive_changes_do_not_enter_physical_identity(tmp_path):
    physical_source(tmp_path/"a", -2.1); physical_source(tmp_path/"b", -999.)
    _, a = integral_problem(tmp_path/"a"); _, b = integral_problem(tmp_path/"b")
    assert a == b and "fci_energy" not in a and not a["FCI_labels_passed_to_algorithm"]


def test_small_actual_sampler_paid_budget_and_reproducible_choices(tmp_path):
    physical_source(tmp_path/"physical", -2.1)
    settings = RAACSSpec(total_model_shots=512, probe_shots=16, max_steps=2, candidates_measured_per_step=2)
    sim = SamplingSpec(backend="exact_small")
    a = run_sampler(tmp_path/"physical", tmp_path/"first", settings, sim)
    b = run_sampler(tmp_path/"physical", tmp_path/"second", settings, sim)
    assert a["status"] == b["status"] == "COMPLETED"
    assert a["actual_total_model_shots"] == 512 and a["independent_final_raw_model_shots"] >= 256
    assert a["accepted_operators"] == b["accepted_operators"]
    assert a["final_QSCI_energy_Ha"] == b["final_QSCI_energy_Ha"]
    assert not a["CI_state_freely_prepared"] and not a["FCI_used"] and a["hardware_shots"] == 0
    db = sqlite3.connect(tmp_path/"first/events.sqlite")
    assert db.execute("SELECT sum(raw_shots) FROM draws").fetchone()[0] == 512
    assert db.execute("SELECT count(*) FROM draws WHERE role='independent_final'").fetchone()[0] == 1
    db.close()
    with pytest.raises(ValueError, match="Protect"):
        run_sampler(tmp_path/"physical", tmp_path/"physical", settings, sim)
    with pytest.raises(NotImplementedError):
        run_sampler(tmp_path/"physical", tmp_path/"unsupported", replace(settings, angle_mode="full_vqe"), sim)


def test_exact_integer_shot_allocation():
    assert allocation_counts(101, [1, 1, 1]) == [34, 34, 33]
    assert sum(allocation_counts(7, [0, 0])) == 7
    assert allocation_counts(20, [0, 1]) == [0, 20]
