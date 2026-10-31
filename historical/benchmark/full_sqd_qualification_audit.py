"""All32 actual full-SQD qualification records, raw provenance and projections."""
from __future__ import annotations
import json
from pathlib import Path
import sqlite3
import numpy as np
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, rng_for, save_json
from .local_sampler_primitives import unpack_counts64, valid_sector, observed_ci
from .resource_adaptive_sampler import integral_problem
from .study.core import unpack_pairs
from .full_sqd_qualification import corrupt
from .fermionic_sqd_workflow import cartesian_ids


def audit():
    root = ROOT/"results/full-fermionic-SQD-quick-v1"; cfg = json.loads((root/"manifest.json").read_text())
    for name, sha in cfg["implementation"].items(): assert file_digest(ROOT/name) == sha
    count = solves = recovered = 0; failures = []; acquisitions = set(); max_difference = 0.
    with threadpool_limits(limits=1):
        for path in sorted((root/"cells").glob("*/*/result.json")):
            r = json.loads(path.read_text()); task = r["task"]; case = task["case"]
            if r["status"] != "COMPLETED": failures.append({"task": task, "error": r["error"]}); continue
            backend, info = integral_problem(Path(cfg["source"])/"references"/case)
            dbpath = Path(cfg["comparison"])/"readouts"/case/"readouts.sqlite"
            assert file_digest(dbpath) == task["source_database_sha256"]
            db = sqlite3.connect(f"file:{dbpath}?mode=ro", uri=True)
            M, blob = db.execute("SELECT shots,counts FROM draws WHERE id=?", (task["source_draw"],)).fetchone(); db.close()
            ids, counts = unpack_pairs(blob)
            seed = rng_for("full-fermionic-SQD-first-qualification-v1", task)[1]
            ni, nn = (ids, counts) if task["readout_flip_probability"] == 0 else corrupt(ids, counts, info["n_qubits"], task["readout_flip_probability"], seed)
            raw = path.parent/"raw-counts.u64z"; assert file_digest(raw) == r["raw_count_file_sha256"]
            actual_ids, actual_counts = unpack_counts64(raw.read_bytes())
            assert np.array_equal(ni, actual_ids) and np.array_equal(nn, actual_counts)
            s = r["SQD"]; assert int(actual_counts.sum()) == s["raw_model_shots"] == M == 1000
            assert set(s["raw_observed_ids"]) == set(map(int, actual_ids))
            ci = observed_ci(backend, dict(zip(map(int, ni), map(int, nn))), info["hf_id"], info["nalpha"], info["nbeta"], 64)
            assert abs(ci["energy_Ha"]-r["explicit_QSCI_energy_Ha"]) < 1e-9
            for rec in s["recovery_trace"]:
                assert rec["new_quantum_shots"] == 0
                xs = rec["returned_unique_ids"]; ws = rec["returned_classical_weights"]
                assert len(xs) == len(ws) and np.isclose(sum(ws), 1.) and all(w >= 0 for w in ws)
                assert all(valid_sector(x, info["n_qubits"], info["nalpha"], info["nbeta"]) for x in xs)
                assert rec["new_recovered_ids_not_raw_observed"] == sorted(set(xs)-set(s["raw_observed_ids"]))
                recovered += 1
            energies = []
            for solve in s["solver_trace"]:
                selected = cartesian_ids(solve["alpha_strings"], solve["beta_strings"], info["n_qubits"]//2)
                assert selected == solve["selected_interleaved_ids"] and len(selected) == solve["dimension"] <= 64
                assert solve["cartesian_ids_not_raw_observed"] == sorted(set(selected)-set(s["raw_observed_ids"]))
                assert info["hf_id"] in selected
                e = float(np.linalg.eigvalsh(backend.matrix(selected, selected))[0])
                difference = abs(e-solve["energy_Ha"]); assert difference < 1e-7
                max_difference = max(max_difference, difference); energies.append(solve["energy_Ha"]); solves += 1
            assert abs(min(energies)-s["energy_Ha"]) < 1e-8
            assert any(t["selected_interleaved_ids"] == s["selected_interleaved_ids"] and abs(t["energy_Ha"]-s["energy_Ha"]) < 1e-8 for t in s["solver_trace"])
            assert not s["FCI_labels_used"] and not s["full_FCI_solve"] and s["hardware_shots"] == 0
            acquisitions.add((task["source_database_sha256"], task["source_draw"]))
            count += 1
    assert count+len(failures) == 32
    import qiskit_addon_sqd.fermion as official
    out = {"status": "FULL_FERMIONIC_SQD_QUICK_RAW_AND_PROJECTIONS_VALIDATED", "validated": count,
           "failures_preserved": failures, "all_selected_Cartesian_spaces_independently_diagonalized": solves,
           "all_recovery_output_records_checked": recovered, "max_projection_energy_discrepancy_Ha": max_difference,
           "unique_preexisting_acquisitions_reused": len(acquisitions), "new_quantum_sample_acquisitions": 0,
           "synthetic_noise_arms_not_independent_quantum_acquisitions": True, "hardware_shots": 0,
           "official_fermion_source_sha256": file_digest(official.__file__), "auditor_sha256": file_digest(__file__),
           "gate_noise_relaxation_routing_and_full_phase_F": "REMAIN_REQUIRED", "whole_study_complete": False}
    save_json(root/"validation.json", out); return out


if __name__ == "__main__": print(json.dumps(audit()))
