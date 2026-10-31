"""Actual official fermionic SQD on existing independent raw acquisitions."""
from __future__ import annotations
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
import numpy as np
from threadpoolctl import threadpool_limits
from .common import ROOT, digest, file_digest, rng_for, save_json
from .study.core import unpack_pairs
from .study.reanalysis import frozen_inventory
from .local_sampler_primitives import observed_ci, pack_counts64
from .resource_adaptive_sampler import integral_problem
from .fermionic_sqd_workflow import full_fermionic_sqd, FermionicSQDSpec


def corrupt(ids, counts, n, probability, seed):
    if not 0 <= probability <= 1: raise ValueError("Probability in[0,1]")
    expanded = np.repeat(np.asarray(ids, dtype=np.uint64), counts)
    rng = np.random.default_rng(seed)
    for bit in range(n): expanded ^= (rng.random(len(expanded)) < probability).astype(np.uint64) << np.uint64(bit)
    return np.unique(expanded, return_counts=True)


def run():
    root = ROOT/"results/full-fermionic-SQD-quick-v1"; root.mkdir(exist_ok=True)
    source = ROOT/"results/standard-v1"; comparisons = ROOT/"results/RAACS-independent-comparison-standard-sharded-v1"
    audit = json.loads((comparisons/"validation.json").read_text())
    assert audit["status"] == "RAACS_INDEPENDENT_COMPARISON_COUNTS_AND_PAIRS_VALIDATED"
    methods = ["B1:source_final_ceo_ovp", "B1:source_final_ceo_mvp", "B1:lucj_ccsd_2_vqe",
               "RAACS:full_cost-qsci_local_grid-last_only-s101-exact_small"]
    files = ["benchmark/fermionic_sqd_workflow.py", "benchmark/full_sqd_qualification.py", "benchmark/integral_hamiltonian.py", "benchmark/sqd.py"]
    cfg = {"schema": "full-fermionic-SQD-first-qualification-v1", "cases": ["H2_R0.7", "H4_R1.1"], "methods": methods,
           "M": 1000, "replicates": [0, 1], "readout_flip_probabilities": [0., .01],
           "settings": asdict(FermionicSQDSpec()), "source": str(source), "comparison": str(comparisons),
           "implementation": {p: file_digest(ROOT/p) for p in files}, "spec_sha256": file_digest(ROOT/"study-v1/FULL_FERMIONIC_SQD.md"),
           "actual_hardware_shots": 0, "new_quantum_sample_acquisitions": 0}
    with (root/"run.lock").open("a") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = root/"manifest.json"
        if path.exists() and json.loads(path.read_text()) != cfg: raise RuntimeError("FullSQD qualification drift")
        if not path.exists() and shutil.disk_usage(root).free < 3*2**30+64*2**20: raise RuntimeError("Qualification freefloor+bounded output")
        save_json(path, cfg); frozen_inventory(source, root); rows = []
        for case in cfg["cases"]:
            backend, info = integral_problem(source/"references"/case)
            db_path = comparisons/"readouts"/case/"readouts.sqlite"
            db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            for method in methods:
                for rep in cfg["replicates"]:
                    did, raw = db.execute("SELECT d.id,d.counts FROM links l JOIN draws d ON d.id=l.draw WHERE l.method=? AND l.total_M=1000 AND l.scope='readout_only' AND l.rep=?", (method, rep)).fetchone()
                    ids, counts = unpack_pairs(raw)
                    for probability in cfg["readout_flip_probabilities"]:
                        if shutil.disk_usage(root).free < 3*2**30: raise RuntimeError("SQD qualification storage boundary")
                        task = {"case": case, "method": method, "rep": rep, "readout_flip_probability": probability,
                                "source_database_sha256": file_digest(db_path), "source_draw": did}
                        seed = rng_for("full-fermionic-SQD-first-qualification-v1", task)[1]
                        folder = root/"cells"/case/digest(task)[:20]; folder.mkdir(parents=True, exist_ok=True)
                        result_path = folder/"result.json"
                        if result_path.exists():
                            r = json.loads(result_path.read_text()); assert r["task"] == task; rows.append(r); continue
                        ni, nn = (ids, counts) if probability == 0 else corrupt(ids, counts, info["n_qubits"], probability, seed)
                        (folder/"raw-counts.u64z").write_bytes(pack_counts64(ni, nn))
                        ci = observed_ci(backend, dict(zip(map(int, ni), map(int, nn))), info["hf_id"], info["nalpha"], info["nbeta"], 64)
                        try:
                            settings = FermionicSQDSpec(seed=seed)
                            sqd = full_fermionic_sqd(backend.h1, backend.eri, backend.core, info["hf_id"], info["nalpha"], info["nbeta"], ni, nn, settings)
                            r = {"status": "COMPLETED", "task": task, "explicit_QSCI_energy_Ha": ci["energy_Ha"],
                                 "explicit_QSCI_dimension": ci["actual_dimension"], "SQD": sqd,
                                 "raw_count_file_sha256": file_digest(folder/"raw-counts.u64z"),
                                 "same_raw_counts_for_QSCI_and_full_SQD": True, "extra_quantum_shots": 0}
                        except Exception as exc:
                            r = {"status": "ENGINEERING_OR_RESOURCE_FAILURE_PRESERVED", "task": task, "error": repr(exc)}
                        save_json(result_path, r); rows.append(r)
                        save_json(root/"execution-status.json", {"status": "RUNNING", "terminal": len(rows), "planned": 32})
            db.close()
        import pandas as pd
        summaries = []
        for r in rows:
            a = {**r["task"], "status": r["status"]}
            if r["status"] == "COMPLETED":
                s = r["SQD"]
                a.update(QSCI_energy_Ha=r["explicit_QSCI_energy_Ha"], SQD_energy_Ha=s["energy_Ha"],
                         QSCI_d=r["explicit_QSCI_dimension"], SQD_d=s["dimension"],
                         SQD_minus_QSCI_mHa=1000*(s["energy_Ha"]-r["explicit_QSCI_energy_Ha"]),
                         recovery_calls=len(s["recovery_trace"]), solver_calls=len(s["solver_trace"]),
                         solver_seconds=s["actual_wall_seconds"], iterations=s["iterations_used"])
            summaries.append(a)
        pd.DataFrame(summaries).to_csv(root/"qualification-summary.csv", index=False)
        out = {"status": "OFFICIAL_FULL_FERMIONIC_SQD_EXECUTED_PENDING_INDEPENDENT_AUDIT", "terminal": len(rows), "planned": 32,
               "completed": sum(r["status"] == "COMPLETED" for r in rows), "hardware_shots": 0,
               "new_quantum_sample_acquisitions": 0, "whole_phase_F_complete": False, "whole_study_complete": False}
        save_json(root/"execution-status.json", out); return out


if __name__ == "__main__": print(json.dumps(run()))
