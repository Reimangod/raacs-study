import copy
import sqlite3
import numpy as np
import pytest
from benchmark.common import ROOT
from benchmark.online_parameter_controls import construct, setting, SCHEMA, task_identity, MODES
from benchmark.resource_adaptive_sampler import integral_problem


def test_task_identity_has_no_case_FCI_label_or_reference_file_hash():
    cfg = setting("quick")
    _, info = integral_problem(ROOT/"results/standard-v1/references/H2_R0.7")
    first = task_identity(cfg, info, 101, "fixed_pi4")
    changed = copy.deepcopy(cfg); changed["raw_reference_sha"] = "mutated-FCI-evaluation-data"
    changed["fci_energy"] = -999
    assert first == task_identity(changed, info, 101, "fixed_pi4")
    assert first != task_identity(cfg, info, 211, "fixed_pi4")


@pytest.mark.parametrize("mode", MODES)
def test_actual_H2_online_fresh_growth_and_paid_final(mode):
    cfg = setting("quick"); cfg["final_repeats"] = 2
    db = sqlite3.connect(":memory:"); db.executescript(SCHEMA)
    r = construct(db, cfg, "H2_R0.7", 101, mode)
    assert r["status"] == "COMPLETED" and r["pilot_shots"]+r["final_shots_per_rep"] == 10000
    assert r["final_shots_per_rep"] >= 5000
    # A32-shot pilot can miss the weak optimized excitation. Rejecting the
    # candidate is an algorithm result, not a construction/validation failure.
    assert db.execute("SELECT count(*) FROM draws WHERE role='candidate'").fetchone()[0] >= 1
    assert r["new_actual_model_shots"] == r["pilot_shots"]+2*r["final_shots_per_rep"]
    assert not r["FCI_used"] and not r["full_Fock_vector_used"]
    assert (r["VQE_QPU_objective_shots"] is None) == mode.endswith("_vqe")
    assert (r["actual_classical_objective_calls"] > 0) == mode.endswith("_vqe")
    n = db.execute("SELECT count(*) FROM draws").fetchone()[0]
    assert construct(db,cfg,"H2_R0.7",101,mode) == r
    assert db.execute("SELECT count(*) FROM draws").fetchone()[0] == n
    assert db.execute("SELECT count(*) FROM final_qpy").fetchone()[0] == 1
    db.close()


def test_unknown_mode_refused_before_any_execution():
    db = sqlite3.connect(":memory:"); db.executescript(SCHEMA)
    with pytest.raises(ValueError): construct(db,setting("quick"),"H2_R0.7",101,"proxy")
    assert db.execute("SELECT count(*) FROM cells").fetchone()[0] == 0
