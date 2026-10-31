"""Independent reconstruction of all broad-SQD counts, traces and CI spaces."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
from threadpoolctl import threadpool_limits
from .common import ROOT, digest, file_digest, rng_for, save_json
from .fermionic_sqd_durable import decode, read_trace
from .fermionic_sqd_workflow import cartesian_ids
from .full_sqd_durable_cohort import RNG_NAMESPACE, NAMESPACE, matched_d
from .full_sqd_qualification import corrupt
from .resource_adaptive_sampler import integral_problem
from .local_sampler_primitives import unpack_counts64, valid_sector, observed_ci
from .study.core import unpack_pairs


def audit(root):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text()); run = json.loads((root/"execution-status.json").read_text())
    assert run["status"] == "FULL_SQD_DURABLE_EXECUTED_PENDING_AUDIT"
    assert file_digest(root/"events.sqlite") == run["database_sha256"]
    for p, sha in cfg["implementation"].items(): assert file_digest(ROOT/p) == sha, p
    import qiskit_addon_sqd.fermion as official
    assert file_digest(official.__file__) == cfg["official_source_sha256"]
    source_audit = json.loads((root/"source-audit.json").read_text())
    assert source_audit["status"] == "ALL_FROZEN_SOURCE_HASHES_VERIFIED" and source_audit["files"] == 40944
    assert file_digest(source_audit["source_manifest"]) == source_audit["source_manifest_sha256"]
    db = sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro", uri=True)
    old = {}
    if cfg["mode"] == "quick":
        for p in (ROOT/"results/full-fermionic-SQD-quick-v1/cells").glob("*/*/result.json"):
            a = json.loads(p.read_text()); old[digest(a["task"])] = a
    seen = set(); acquisitions = {}; failures = []; missing = []; completed = events = solves = recovery = parity = 0
    full_spaces = 0; max_error = 0.
    with threadpool_limits(limits=1):
        for case in cfg["cases"]:
            backend, info = integral_problem(Path(cfg["source"])/"references"/case)
            sourcepath = Path(cfg["comparison"])/"readouts"/case/"readouts.sqlite"
            sourcesha = file_digest(sourcepath); original = sqlite3.connect(f"file:{sourcepath}?mode=ro", uri=True)
            for method in cfg["methods"]:
                for rep in cfg["reps"]:
                    hit = original.execute("SELECT d.id,d.counts FROM links l JOIN draws d ON d.id=l.draw WHERE l.method=? AND l.total_M=? AND l.scope='readout_only' AND l.rep=?", (method, cfg["M"], rep)).fetchone()
                    for prob in cfg["readout_probabilities"]:
                        task = {"case": case, "method": method, "rep": rep, "readout_flip_probability": prob,
                                "source_database_sha256": sourcesha, "source_draw": hit[0] if hit else None}
                        tid = digest([NAMESPACE, task]); seen.add(tid)
                        stored_task, status, counts_raw, result_raw = db.execute("SELECT task,status,counts,result FROM cells WHERE id=?", (tid,)).fetchone()
                        assert json.loads(stored_task) == task
                        result = decode(result_raw); assert result["task"] == task and result["status"] == status
                        if hit is None:
                            assert status == "SOURCE_INITIALIZATION_MISSING" and counts_raw is None
                            assert not read_trace(db, tid); missing.append(task); continue
                        ids, counts = unpack_pairs(hit[1]); seed = rng_for(RNG_NAMESPACE, task)[1]
                        ni, nn = (ids, counts) if prob == 0 else corrupt(ids, counts, info["n_qubits"], prob, seed)
                        actual_ids, actual_counts = unpack_counts64(counts_raw)
                        assert np.array_equal(actual_ids, ni) and np.array_equal(actual_counts, nn)
                        assert int(actual_counts.sum()) == cfg["M"]
                        acquisitions[(sourcesha, hit[0])] = int(sum(counts))
                        trace = read_trace(db, tid); events += len(trace)
                        if status != "COMPLETED":
                            assert status == "ENGINEERING_OR_RESOURCE_FAILURE_PRESERVED"
                            failures.append({"task": task, "error": result["error"], "durable_events": len(trace)})
                            continue
                        assert trace[0]["kind"] == "START" and trace[-1]["kind"] == "COMPLETED"
                        assert trace[0]["payload"]["raw_ids"] == list(map(int, ni))
                        assert trace[0]["payload"]["raw_counts"] == list(map(int, nn))
                        sqd = result["SQD"]; raw = set(map(int, ni))
                        assert sqd["raw_model_shots"] == cfg["M"] and set(sqd["raw_observed_ids"]) == raw
                        assert sqd["new_quantum_shots_for_recovery"] == result["new_quantum_sample_acquisitions"] == result["hardware_shots"] == 0
                        assert not sqd["FCI_labels_used"] and result["QPU_construction_cost"] is None
                        ci = observed_ci(backend, dict(zip(map(int, ni), map(int, nn))), info["hf_id"], info["nalpha"], info["nbeta"], cfg["R"])
                        assert ci["selected_ids"] == result["QSCI"]["selected_ids"]
                        assert abs(ci["energy_Ha"]-result["QSCI"]["energy_Ha"]) < 1e-9
                        sr = [r["payload"] for r in trace if r["kind"] == "SOLVE_RESULT"]
                        rr = [r["payload"] for r in trace if r["kind"] == "RECOVERY_RESULT"]
                        it = [r["payload"] for r in trace if r["kind"] == "ITERATION"]
                        assert len(sr) == len(sqd["solver_trace"]) and len(rr) == len(sqd["recovery_trace"])
                        assert len(it) == sqd["iterations_used"]
                        for rec in (r["payload"] for r in trace if r["kind"] == "SOLVE_REQUEST"):
                            assert len(rec["spaces"]) == cfg["SQD_settings"]["batches"]
                            for space in rec["spaces"]: assert len(space["alpha"])*len(space["beta"]) <= cfg["R"]
                        for rec, frozen in zip(sr, sqd["solver_trace"]):
                            xs = cartesian_ids(rec["alpha"], rec["beta"], info["n_qubits"]//2)
                            assert xs == rec["selected_ids"] == frozen["selected_interleaved_ids"]
                            assert info["hf_id"] in xs and len(xs) <= cfg["R"]
                            assert all(valid_sector(x, info["n_qubits"], info["nalpha"], info["nbeta"]) for x in xs)
                            assert rec["new_cartesian_ids_not_raw"] == sorted(set(xs)-raw)
                            ws = np.array(rec["coefficient_weights"])
                            assert len(ws) == len(xs) and np.isfinite(ws).all() and min(ws) >= 0 and abs(ws.sum()-1) < 1e-8
                            energy = float(np.linalg.eigvalsh(backend.matrix(xs, xs))[0])
                            error = abs(energy-rec["energy_Ha"]); assert error < 1e-7
                            assert abs(rec["energy_Ha"]-frozen["energy_Ha"]) < 1e-12
                            assert abs(rec["spin_square"]-frozen["spin_square_measured"]) < 1e-9
                            max_error = max(max_error, error); solves += 1
                        for rec, frozen in zip(rr, sqd["recovery_trace"]):
                            xs, ws = rec["ids"], np.array(rec["classical_weights"])
                            assert xs == frozen["returned_unique_ids"] and len(xs) == len(ws)
                            assert np.isfinite(ws).all() and min(ws) >= 0 and abs(ws.sum()-1) < 1e-8
                            assert all(valid_sector(x, info["n_qubits"], info["nalpha"], info["nbeta"]) for x in xs)
                            assert rec["new_recovered_ids_not_raw"] == sorted(set(xs)-raw)
                            assert rec["new_quantum_shots"] == 0; recovery += 1
                        assert abs(min(x["energy_Ha"] for x in sr)-sqd["energy_Ha"]) < 1e-8
                        md = matched_d(backend, ci, sqd, trace, info["hf_id"])
                        for k in ("d", "QSCI_ids", "SQD_ids"): assert md[k] == result["same_actual_d"][k]
                        for k in ("QSCI_energy_Ha", "SQD_energy_Ha"): assert abs(md[k]-result["same_actual_d"][k]) < 1e-9
                        assert result["SQD_selected_space_spans_full_sector"] == (sqd["dimension"] == result["registered_sector_dimension"])
                        full_spaces += int(result["SQD_selected_space_spans_full_sector"])
                        if old:
                            legacy = old[digest(task)]["SQD"]
                            for k in ("energy_Ha", "selected_interleaved_ids", "self_consistency_iterations", "recovery_trace", "solver_trace"):
                                assert legacy[k] == sqd[k], (case, method, rep, prob, k)
                            parity += 1
                        completed += 1
            original.close()
    assert len(seen) == run["planned"] == db.execute("SELECT count(*) FROM cells").fetchone()[0]
    assert completed+len(missing)+len(failures) == len(seen)
    db.close()
    out = {"status": "FULL_SQD_DURABLE_COUNTS_TRACES_AND_SPACES_VALIDATED", "planned": len(seen), "completed": completed,
        "missing_preserved": missing, "failures_preserved": failures, "old_quick_parity_cells": parity,
        "durable_hash_linked_events": events, "all_selected_Cartesian_spaces_independently_solved": solves,
        "recovery_records_checked": recovery, "maximum_projection_discrepancy_Ha": max_error,
        "selected_spaces_spanning_full_sector": full_spaces, "unique_preexisting_acquisitions": len(acquisitions),
        "unique_preexisting_model_shots_referenced": sum(acquisitions.values()),
        "new_quantum_sample_acquisitions": 0, "hardware_shots": 0, "database_sha256": run["database_sha256"],
        "auditor_sha256": file_digest(__file__), "whole_phase_F_complete": False, "whole_study_complete": False}
    save_json(root/"validation.json", out); return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); a = p.parse_args(); print(json.dumps(audit(a.output)))
