import sqlite3
import numpy as np
import pytest
from benchmark.bounded_selected_sci import lowest_projected_root, projected_sqd
from benchmark.fermionic_sqd_durable import DurableTrace, read_trace
from benchmark.fermionic_sqd_workflow import FermionicSQDSpec, full_fermionic_sqd
from benchmark.common import ROOT
from benchmark.resource_adaptive_sampler import integral_problem


def test_lowest_root_does_not_depend_on_excited_initial_guess():
    H = np.diag([-2., -1., 5.]); calls = []
    def op(x):
        assert x.flags.c_contiguous
        calls.append(1); return H@x
    e, v, work = lowest_projected_root(op, [np.array([0., 0., 1.])])
    assert e == -2. and len(calls) == work["selected_H_vector_products"] == 3
    assert np.allclose(H@v, e*v) and not work["full_Fock_matrix_or_vector"]


def test_projection_cap_rejects_before_any_operator_work():
    calls = []
    def op(x): calls.append(1); return x
    with pytest.raises(RuntimeError, match="dimension cap"):
        lowest_projected_root(op, [np.ones(65)])
    assert not calls


def test_real_official_SQD_with_selected_ground_kernel(tmp_path):
    b, info = integral_problem(ROOT/"results/standard-v1/references/H2_R0.7")
    spec = FermionicSQDSpec(max_spin_strings=2, batches=2, max_iterations=3, samples_per_batch=8)
    ids, counts = [12, 3, 0, 15], [40, 30, 15, 15]
    old = full_fermionic_sqd(b.h1, b.eri, b.core, info["hf_id"], 1, 1, ids, counts, spec)
    db = sqlite3.connect(tmp_path/"events.sqlite")
    out = projected_sqd(b.h1, b.eri, b.core, info["hf_id"], 1, 1, ids, counts, spec, DurableTrace(db, "a"))
    assert abs(old["energy_Ha"]-out["energy_Ha"]) < 1e-10
    assert out["selected_solver_work"]["calls"] == len(out["solver_trace"])
    assert out["selected_solver_work"]["selected_H_vector_products"] >= len(out["solver_trace"])
    assert out["selected_solver_work"]["maximum_selected_dimension"] <= 4
    assert any(x["kind"] == "PROJECTED_GROUND_RESULT" for x in read_trace(db, "a"))
    db.close()


def test_real_stretched_N2_former_failure_with_bounded_kernel(tmp_path):
    from benchmark.fermionic_sqd_durable import decode
    from benchmark.local_sampler_primitives import unpack_counts64
    from benchmark.common import rng_for
    path = ROOT/"results/full-fermionic-SQD-durable-standard-v1/events.sqlite"
    if not path.exists(): pytest.skip("Original preserved broad cohort unavailable")
    olddb = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    raw, rawcounts = olddb.execute("SELECT result,counts FROM cells WHERE status='ENGINEERING_OR_RESOURCE_FAILURE_PRESERVED' AND task LIKE '%N2_R2.2%' ORDER BY id LIMIT 1").fetchone(); olddb.close()
    r = decode(raw); ids, counts = unpack_counts64(rawcounts)
    b, info = integral_problem(ROOT/"results/standard-v1/references"/r["task"]["case"])
    seed = rng_for("full-fermionic-SQD-first-qualification-v1", r["task"])[1]
    db = sqlite3.connect(tmp_path/"events.sqlite")
    out = projected_sqd(b.h1, b.eri, b.core, info["hf_id"], info["nalpha"], info["nbeta"],
        ids, counts, FermionicSQDSpec(seed=seed), DurableTrace(db, "former-failure"))
    assert out["dimension"] <= 64
    assert out["selected_solver_work"]["maximum_selected_dimension"] <= 64
    assert all(abs(s["energy_Ha"]-s["independent_named_CI_energy_Ha"]) < 1e-7 for s in out["solver_trace"])
    assert read_trace(db, "former-failure")[-1]["kind"] == "COMPLETED"
    db.close()
