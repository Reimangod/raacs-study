"""All-record ledger/resource/selected-space audit plus bounded Aer replay."""
from __future__ import annotations
import argparse
from io import BytesIO
import json
from math import comb
from pathlib import Path
import sqlite3
from time import perf_counter
import zlib
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, digest, rng_for, save_json
from .gate_noise_study import NAMESPACE, unpack, merge_chunks
from .gate_noise_runtime import load_sampler, compile_sampler, logical_probabilities, simulate_counts
from .resource_adaptive_sampler import integral_problem
from .local_sampler_primitives import observed_ci, valid_sector
from .fermionic_sqd_durable import read_trace
from .fermionic_sqd_workflow import cartesian_ids
from .full_sqd_durable_cohort import matched_d


def audit(root):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text()); status = json.loads((root/"execution-status.json").read_text())
    assert status["status"] == "GATE_NOISE_FULL_COHORT_EXECUTED_PENDING_AUDIT"
    assert file_digest(root/"events.sqlite") == status["database_sha256"]
    for p, sha in cfg["implementation"].items(): assert file_digest(ROOT/p) == sha, p
    source_audit = json.loads((root/"source-audit.json").read_text()); assert source_audit["files"] == 40944
    assert file_digest(source_audit["source_manifest"]) == source_audit["source_manifest_sha256"]
    db = sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro", uri=True)
    circuits = {}; compiled_count = parity_count = 0; maximum = 0.; tasks = {}; replays = []; projections = 0
    raw_shots = 0; completed = 0; partial = []; missing = 0; sector_complete = 0; recovery_records = 0
    with threadpool_limits(limits=1):
        for cid, case, method, target, available, raw, qblob in db.execute("SELECT * FROM circuits"):
            expected = digest([NAMESPACE, cfg["mode"], case, method, target, cfg["implementation"]]); assert expected == cid
            source, origin = load_sampler(case, method)
            if available != "AVAILABLE":
                assert source is None and available == "SOURCE_INITIALIZATION_MISSING" and qblob is None
                circuits[cid] = {"available": False}; continue
            record = unpack(raw); assert record["source"] == origin
            qc = qpy.load(BytesIO(zlib.decompress(qblob)))[0]
            rebuilt, resource = compile_sampler(source, target)
            assert json.loads(json.dumps(resource)) == record["resources"]
            assert dict(qc.count_ops()) == resource["gate_counts"]
            assert qc.count_ops().get("cx", 0) == resource["native_2q_count"]
            p = Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
            error = float(max(abs(Statevector(source).probabilities()-logical_probabilities(p, resource["measurement_physical_to_logical_clbit"]))))
            assert error < 1e-9; maximum = max(maximum, error); parity_count += 1
            circuits[cid] = {"available": True, "qc": qc, "resource": resource}; compiled_count += 1
        for case in cfg["cases"]:
            backend, info = integral_problem(Path(cfg["source"])/"references"/case)
            for method in cfg["methods"]:
                for target in cfg["targets"]:
                    cid = digest([NAMESPACE, cfg["mode"], case, method, target, cfg["implementation"]])
                    for profile in cfg["profiles"]:
                        for seed in cfg["seeds"]:
                            task = {"case": case, "method": method, "target": target, "profile": profile, "seed": seed, "circuit": cid, "M": cfg["M"]}
                            tid = digest([NAMESPACE, cfg["mode"], task]); tasks[tid] = task
                            task_raw, terminal, raw_result = db.execute("SELECT task,status,result FROM tasks WHERE id=?", (tid,)).fetchone()
                            assert json.loads(task_raw) == task
                            r = unpack(raw_result); assert r["status"] == terminal and terminal != "RUNNING"
                            counts, shots = merge_chunks(db, tid); raw_shots += shots
                            assert shots == r["known_committed_model_shots"] <= cfg["M"]
                            chunks = db.execute("SELECT idx,shots,seed,counts,record FROM chunks WHERE task=? ORDER BY idx", (tid,)).fetchall()
                            for j, (idx, n, sd, blob, metadata_blob) in enumerate(chunks):
                                assert idx == j and n == cfg["chunk_shots"]
                                assert int(sd) == rng_for(NAMESPACE, cfg["mode"], tid, "chunk", idx)[1]
                                c = unpack(blob); assert sum(c.values()) == n
                                assert all(0 <= int(x) < 1 << info["n_qubits"] and isinstance(v, int) and v > 0 for x, v in c.items())
                                metadata = unpack(metadata_blob)
                                assert metadata["hardware_shots"] == 0 and metadata["backend"]["method"] == "statevector"
                                assert metadata["backend"]["device"] == "CPU" and metadata["backend"]["maximum_qubits"] == 16
                            if terminal == "SOURCE_INITIALIZATION_MISSING":
                                assert not circuits[cid]["available"] and shots == 0; missing += 1; continue
                            if terminal != "COMPLETED":
                                partial.append({"task": task, "status": terminal, "known_model_shots": shots, "error": r.get("error")})
                                continue
                            assert shots == cfg["M"] and len(chunks) == cfg["M"]//cfg["chunk_shots"]
                            assert r["hardware_shots"] == 0 and r["QPU_construction_cost"] is None
                            ci = observed_ci(backend, counts, info["hf_id"], info["nalpha"], info["nbeta"], cfg["R"])
                            assert ci["selected_ids"] == r["QSCI"]["selected_ids"]
                            assert abs(ci["energy_Ha"]-r["QSCI"]["energy_Ha"]) < 1e-9
                            sqd = r["SQD"]; trace = read_trace(db, tid)
                            assert trace[0]["kind"] == "START" and trace[-1]["kind"] == "COMPLETED"
                            assert dict(zip(trace[0]["payload"]["raw_ids"], trace[0]["payload"]["raw_counts"])) == counts
                            assert sqd["raw_model_shots"] == cfg["M"] and set(sqd["raw_observed_ids"]) == set(counts)
                            for s in sqd["solver_trace"]:
                                xs = cartesian_ids(s["alpha_strings"], s["beta_strings"], info["n_qubits"]//2)
                                assert xs == s["selected_interleaved_ids"] and len(xs) <= cfg["R"] and info["hf_id"] in xs
                                e = float(np.linalg.eigvalsh(backend.matrix(xs, xs))[0])
                                assert abs(e-s["energy_Ha"]) < 1e-7
                                assert s["cartesian_ids_not_raw_observed"] == sorted(set(xs)-set(counts))
                                projections += 1
                            trace_results = [e["payload"] for e in trace if e["kind"] == "SOLVE_RESULT"]
                            roots = [e["payload"] for e in trace if e["kind"] == "PROJECTED_GROUND_RESULT"]
                            assert len(roots) == len(trace_results) == len(sqd["solver_trace"])
                            assert sum(s["selected_H_vector_products"] for s in roots) == sqd["selected_solver_work"]["selected_H_vector_products"]
                            for s in sqd["recovery_trace"]:
                                xs, ws = s["returned_unique_ids"], s["returned_classical_weights"]
                                assert len(xs) == len(ws) and abs(sum(ws)-1) < 1e-8 and min(ws) >= 0
                                assert all(valid_sector(x, info["n_qubits"], info["nalpha"], info["nbeta"]) for x in xs)
                                assert s["new_recovered_ids_not_raw_observed"] == sorted(set(xs)-set(counts))
                                assert s["new_quantum_shots"] == 0; recovery_records += 1
                            assert abs(min(s["energy_Ha"] for s in sqd["solver_trace"])-sqd["energy_Ha"]) < 1e-8
                            md = matched_d(backend, ci, sqd, trace, info["hf_id"])
                            for k in ("d", "QSCI_ids", "SQD_ids"): assert md[k] == r["same_actual_d"][k]
                            for k in ("QSCI_energy_Ha", "SQD_energy_Ha"): assert abs(md[k]-r["same_actual_d"][k]) < 1e-9
                            ep = r["explicit_last_fullSQD_recovery_projection"]
                            if ep is not None:
                                assert len(ep["selected_ids"]) == ep["dimension"] <= cfg["R"]
                                assert abs(float(np.linalg.eigvalsh(backend.matrix(ep["selected_ids"], ep["selected_ids"]))[0])-ep["energy_Ha"]) < 1e-9
                                assert ep["new_ids_not_observed"] == sorted(set(ep["selected_ids"])-set(counts))
                            sector_complete += int(sqd["dimension"] == comb(info["n_qubits"]//2, info["nalpha"])*comb(info["n_qubits"]//2, info["nbeta"]))
                            completed += 1
                            # Fixed one chunk per molecular family; no energy-based choice.
                            first_case = next(x for x in cfg["cases"] if x.split("_R")[0] == case.split("_R")[0])
                            if case == first_case and method == cfg["methods"][1] and target == "line_cx" and profile == "combined" and seed == cfg["seeds"][0]:
                                c0 = chunks[0]; before = perf_counter()
                                replay, metadata = simulate_counts(circuits[cid]["qc"], profile, c0[1], int(c0[2]))
                                assert replay == {int(k): v for k, v in unpack(c0[3]).items()}
                                replays.append({"case": case, "model_shots_reexecuted_for_deterministic_audit": c0[1],
                                    "new_independent_acquisitions": 0, "audit_replay_seconds": perf_counter()-before})
    assert len(tasks) == status["planned"] == db.execute("SELECT count(*) FROM tasks").fetchone()[0]
    assert completed+missing+len(partial) == len(tasks)
    db.close()
    out = {"status": "GATE_NOISE_COUNTS_RESOURCES_AND_SELECTED_SPACES_VALIDATED", "planned": len(tasks), "completed": completed,
        "missing_preserved": missing, "failures_and_censoring_preserved": partial, "native_circuits_reconstructed": compiled_count,
        "independent_logical_probability_checks": parity_count, "maximum_logical_probability_error": maximum,
        "all_completed_selected_spaces_independently_diagonalized": projections, "all_recovery_records_checked": recovery_records,
        "selected_spaces_spanning_full_sector": sector_complete, "known_committed_new_model_shots": raw_shots,
        "deterministic_Aer_replay_checks": replays, "all_other_Aer_draws_ledger_validated_not_reexecuted": True,
        "hardware_shots": 0, "database_sha256": status["database_sha256"], "auditor_sha256": file_digest(__file__),
        "whole_study_complete": False}
    save_json(root/"validation.json", out); return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); a = p.parse_args(); print(json.dumps(audit(a.output)))
