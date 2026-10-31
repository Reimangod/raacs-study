import sqlite3
from unittest.mock import patch
import numpy as np
import pytest
import qiskit_addon_sqd.fermion as official
from benchmark.common import ROOT
from benchmark.fermionic_sqd_durable import DurableTrace, full_sqd_durable, read_trace, encode
from benchmark.fermionic_sqd_workflow import FermionicSQDSpec, full_fermionic_sqd
from benchmark.resource_adaptive_sampler import integral_problem


def physical():
    backend, info = integral_problem(ROOT/"results/standard-v1/references/H2_R0.7")
    return backend, info, [12, 3, 0, 15], [40, 30, 15, 15]


def test_observer_preserves_real_official_numerical_trajectory(tmp_path):
    b, info, ids, counts = physical()
    spec = FermionicSQDSpec(max_spin_strings=2, batches=2, max_iterations=3, samples_per_batch=8)
    old = full_fermionic_sqd(b.h1, b.eri, b.core, info["hf_id"], 1, 1, ids, counts, spec)
    db = sqlite3.connect(tmp_path/"events.sqlite"); db.execute("PRAGMA synchronous=FULL")
    new = full_sqd_durable(b.h1, b.eri, b.core, info["hf_id"], 1, 1, ids, counts, spec, DurableTrace(db, "a"))
    for key in ("energy_Ha", "selected_interleaved_ids", "self_consistency_iterations", "recovery_trace", "solver_trace"):
        assert new[key] == old[key]
    trace = read_trace(db, "a")
    assert trace[0]["kind"] == "START" and trace[-1]["kind"] == "COMPLETED"
    for rec in trace:
        if rec["kind"] == "SOLVE_RESULT":
            p = rec["payload"]
            assert np.isclose(sum(p["coefficient_weights"]), 1.)
            assert len(p["selected_ids"]) == len(p["coefficient_weights"])
    db.close()


def test_failed_real_solver_keeps_prior_durable_work_and_no_free_retry(tmp_path):
    b, info, ids, counts = physical()
    path = tmp_path/"events.sqlite"; db = sqlite3.connect(path); db.execute("PRAGMA synchronous=FULL")
    original = official.solve_sci_batch; calls = []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2: raise RuntimeError("injected second-batch failure")
        return original(*args, **kwargs)
    with patch.object(official, "solve_sci_batch", fail_second), pytest.raises(RuntimeError, match="injected"):
        full_sqd_durable(b.h1, b.eri, b.core, info["hf_id"], 1, 1, ids, counts,
            FermionicSQDSpec(max_spin_strings=2, batches=2, max_iterations=3, samples_per_batch=8), DurableTrace(db, "a"))
    external = sqlite3.connect(path)
    trace = read_trace(external, "a")
    assert trace[-1]["kind"] == "FAILED_PRESERVED"
    assert any(r["kind"] == "SOLVE_RESULT" for r in trace)
    with pytest.raises(RuntimeError, match="partial trace"): DurableTrace(db, "a")
    external.close(); db.close()


def test_trace_detects_payload_and_order_mutation(tmp_path):
    db = sqlite3.connect(tmp_path/"events.sqlite")
    trace = DurableTrace(db, "a"); trace.emit("START", {"shots": 100})
    assert len(read_trace(db, "a")) == 1
    db.execute("UPDATE trace SET payload=?", (encode({"shots": 0}),)); db.commit()
    with pytest.raises(RuntimeError, match="digest"): read_trace(db, "a")
    db.close()
