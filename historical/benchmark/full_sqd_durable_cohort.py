"""All-molecule full-SQD successor on existing independent acquisitions."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import importlib.metadata
import json
from math import comb
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
import qiskit_addon_sqd.fermion as official
from .common import ROOT, digest, file_digest, rng_for, save_json
from .fermionic_sqd_workflow import FermionicSQDSpec
from .fermionic_sqd_durable import DurableTrace, full_sqd_durable, encode, decode, read_trace
from .full_sqd_qualification import corrupt
from .local_sampler_primitives import observed_ci, pack_counts64
from .resource_adaptive_sampler import integral_problem
from .study.core import unpack_pairs, rss_peak

NAMESPACE = "full-fermionic-SQD-durable-cohort-v1"
RNG_NAMESPACE = "full-fermionic-SQD-first-qualification-v1"
METHODS = ["B1:source_final_ceo_ovp", "B1:source_final_ceo_mvp", "B1:lucj_ccsd_2_vqe",
           "RAACS:full_cost-qsci_local_grid-last_only-s101-exact_small"]


def settings(mode):
    source = ROOT/"results/standard-v1"
    comparison = ROOT/"results/RAACS-independent-comparison-standard-sharded-v1"
    cases = ["H2_R0.7", "H4_R1.1"] if mode == "quick" else sorted(p.name for p in (source/"references").iterdir() if (p/"reference.json").exists())
    if mode == "standard" and len(cases) != 30: raise RuntimeError("Expected full30case cohort")
    files = ["fermionic_sqd_workflow.py", "fermionic_sqd_durable.py", "full_sqd_durable_cohort.py",
             "full_sqd_qualification.py", "integral_hamiltonian.py", "local_sampler_primitives.py",
             "resource_adaptive_sampler.py", "sqd.py", "common.py"]
    return {"schema": NAMESPACE, "mode": mode, "source": str(source), "comparison": str(comparison),
        "cases": cases, "methods": METHODS, "reps": [0, 1] if mode == "quick" else [0, 1, 2],
        "readout_probabilities": [0., .01], "M": 1000, "R": 64, "SQD_settings": asdict(FermionicSQDSpec()),
        "RNG_namespace_preserves_old_quick_parity": RNG_NAMESPACE,
        "free_floor": int(2.75*2**30), "output_cap": 128*2**20, "RSS_cap": 2*2**30,
        "implementation": {"benchmark/"+p: file_digest(ROOT/"benchmark"/p) for p in files},
        "official_source_sha256": file_digest(official.__file__),
        "packages": {p: importlib.metadata.version(p) for p in ("qiskit-addon-sqd", "pyscf", "numpy", "scipy")},
        "lock_sha256": file_digest(ROOT/"uv.lock"), "spec_sha256": file_digest(ROOT/"study-v1/FULL_FERMIONIC_SQD_DURABLE.md"),
        "new_quantum_sample_acquisitions": 0, "hardware_shots": 0}


def verify_source(source, output):
    pointer = json.loads((source/"REPRODUCIBILITY.json").read_text())
    path = source/pointer["manifest"]
    assert file_digest(path) == pointer["manifest_sha256"]
    artifacts = json.loads(path.read_text())["artifacts"]; total = 0
    for name, record in artifacts.items():
        p = source/name; h = hashlib.sha256()
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(2**20), b""): h.update(chunk)
        assert h.hexdigest() == record["sha256"], name
        total += p.stat().st_size
    out = {"status": "ALL_FROZEN_SOURCE_HASHES_VERIFIED", "source_manifest": str(path),
           "source_manifest_sha256": pointer["manifest_sha256"], "files": len(artifacts), "bytes": total,
           "source_inventory_not_copied": True}
    save_json(output/"source-audit.json", out); return out


def matched_d(backend, ci, sqd, trace_records, hf):
    # Match the actual returned best batch, not an arbitrary equal-energy batch.
    best = next(r["payload"] for r in trace_records if r["kind"] == "SOLVE_RESULT"
                and r["payload"]["selected_ids"] == sqd["selected_interleaved_ids"]
                and abs(r["payload"]["energy_Ha"]-sqd["energy_Ha"]) < 1e-8)
    d = min(ci["actual_dimension"], sqd["dimension"])
    order = [hf]+[x for x, w in sorted(zip(best["selected_ids"], best["coefficient_weights"]), key=lambda z: (-z[1], z[0])) if x != hf]
    a, b = sorted(ci["frequency_order"][:d]), sorted(order[:d])
    assert len(a) == len(b) == d and hf in a and hf in b
    return {"d": d, "QSCI_ids": a, "SQD_ids": b,
            "QSCI_energy_Ha": float(np.linalg.eigvalsh(backend.matrix(a, a))[0]),
            "SQD_energy_Ha": float(np.linalg.eigvalsh(backend.matrix(b, b))[0]),
            "SQD_coefficient_order_not_FCI_order": True,
            "truncated_SQD_space_not_required_to_remain_Cartesian": True,
            "same_d_not_same_total_classical_work": True}


def bounds(root, cfg):
    if shutil.disk_usage(root).free < cfg["free_floor"]: raise RuntimeError("SQD freefloor boundary")
    if rss_peak() > cfg["RSS_cap"]: raise RuntimeError("SQD RSS boundary")
    if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > cfg["output_cap"]:
        raise RuntimeError("SQD output boundary")


def run(root, mode):
    root = Path(root).resolve(); source = ROOT/"results/standard-v1"
    if root == source or source in root.parents or root in source.parents: raise ValueError("Protect source")
    if any((p/"REPRODUCIBILITY.json").exists() for p in (root, *root.parents)): raise ValueError("Protect sealed output")
    root.mkdir(parents=True, exist_ok=True)
    with (root/"run.lock").open("a") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg = settings(mode); manifest = root/"manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != cfg: raise RuntimeError("Frozen SQD cohort drift")
        else:
            if shutil.disk_usage(root).free < cfg["free_floor"]+cfg["output_cap"]:
                raise RuntimeError("Insufficient SQD floor plus output reserve")
            if mode == "standard":
                quick = json.loads((ROOT/"results/full-fermionic-SQD-durable-quick-v1/validation.json").read_text())
                assert quick["status"] == "FULL_SQD_DURABLE_COUNTS_TRACES_AND_SPACES_VALIDATED"
                assert quick["old_quick_parity_cells"] == 32
            save_json(manifest, cfg)
        if not (root/"source-audit.json").exists(): verify_source(source, root)
        comparison = Path(cfg["comparison"])
        assert json.loads((comparison/"validation.json").read_text())["status"] == "RAACS_INDEPENDENT_COMPARISON_COUNTS_AND_PAIRS_VALIDATED"
        db = sqlite3.connect(root/"events.sqlite")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS cells(id TEXT PRIMARY KEY,task TEXT,status TEXT,counts BLOB,result BLOB)"); db.commit()
        planned = len(cfg["cases"])*len(METHODS)*len(cfg["reps"])*2
        for case in cfg["cases"]:
            backend, info = integral_problem(source/"references"/case)
            dbpath = comparison/"readouts"/case/"readouts.sqlite"; dbsha = file_digest(dbpath)
            original = sqlite3.connect(f"file:{dbpath}?mode=ro", uri=True)
            for method in METHODS:
                for rep in cfg["reps"]:
                    hit = original.execute("SELECT d.id,d.counts FROM links l JOIN draws d ON d.id=l.draw WHERE l.method=? AND l.total_M=? AND l.scope='readout_only' AND l.rep=?", (method, cfg["M"], rep)).fetchone()
                    for prob in cfg["readout_probabilities"]:
                        bounds(root, cfg)
                        task = {"case": case, "method": method, "rep": rep, "readout_flip_probability": prob,
                                "source_database_sha256": dbsha, "source_draw": hit[0] if hit else None}
                        tid = digest([NAMESPACE, task]); existing = db.execute("SELECT status FROM cells WHERE id=?", (tid,)).fetchone()
                        if existing:
                            if existing[0] == "RUNNING": raise RuntimeError("Partial SQD cell preserved; no unrecorded retry")
                            continue
                        started = perf_counter()
                        if hit is None:
                            result = {"status": "SOURCE_INITIALIZATION_MISSING", "task": task,
                                      "reason": "No source independent acquisition: retain parent missingness", "new_quantum_shots": 0}
                            db.execute("INSERT INTO cells VALUES(?,?,?,?,?)", (tid, json.dumps(task), result["status"], None, encode(result))); db.commit()
                        else:
                            ids, counts = unpack_pairs(hit[1]); seed = rng_for(RNG_NAMESPACE, task)[1]
                            ni, nn = (ids, counts) if prob == 0 else corrupt(ids, counts, info["n_qubits"], prob, seed)
                            db.execute("INSERT INTO cells VALUES(?,?,?,?,NULL)", (tid, json.dumps(task), "RUNNING", pack_counts64(ni, nn))); db.commit()
                            tracer = DurableTrace(db, tid)
                            try:
                                ci = observed_ci(backend, dict(zip(map(int, ni), map(int, nn))), info["hf_id"], info["nalpha"], info["nbeta"], cfg["R"])
                                sqd = full_sqd_durable(backend.h1, backend.eri, backend.core, info["hf_id"], info["nalpha"], info["nbeta"], ni, nn, FermionicSQDSpec(seed=seed), tracer)
                                md = matched_d(backend, ci, sqd, read_trace(db, tid), info["hf_id"])
                                sector_dim = comb(info["n_qubits"]//2, info["nalpha"])*comb(info["n_qubits"]//2, info["nbeta"])
                                result = {"status": "COMPLETED", "task": task, "seed": seed, "problem": info,
                                    "QSCI": {k: ci[k] for k in ("selected_ids", "frequency_order", "energy_Ha", "actual_dimension", "internal_Ritz_residual")},
                                    "SQD": sqd, "same_actual_d": md, "registered_sector_dimension": sector_dim,
                                    "SQD_selected_space_spans_full_sector": sqd["dimension"] == sector_dim,
                                    "same_raw_counts": True, "new_quantum_sample_acquisitions": 0, "hardware_shots": 0,
                                    "QPU_construction_cost": None, "QPU_construction_cost_status": "NOT_REESTIMATED_FROM_CLASSICAL_CALLS",
                                    "cell_actual_classical_wall_seconds": perf_counter()-started, "process_RSS_HWM_bytes": rss_peak()}
                            except Exception as exc:
                                result = {"status": "ENGINEERING_OR_RESOURCE_FAILURE_PRESERVED", "task": task, "error": repr(exc),
                                    "cell_actual_classical_wall_seconds": perf_counter()-started, "trace_events_preserved": tracer.seq,
                                    "new_quantum_sample_acquisitions": 0, "hardware_shots": 0}
                            db.execute("UPDATE cells SET status=?,result=? WHERE id=?", (result["status"], encode(result), tid)); db.commit()
                        statuses = dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status"))
                        save_json(root/"execution-status.json", {"status": "RUNNING", "planned": planned, "terminal": sum(statuses.values()), "cells_by_status": statuses})
            original.close()
        statuses = dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status")); db.close()
        out = {"status": "FULL_SQD_DURABLE_EXECUTED_PENDING_AUDIT", "planned": planned, "terminal": sum(statuses.values()),
               "cells_by_status": statuses, "database_sha256": file_digest(root/"events.sqlite"),
               "new_quantum_sample_acquisitions": 0, "hardware_shots": 0, "whole_phase_F_complete": False, "whole_study_complete": False}
        assert out["terminal"] == planned
        save_json(root/"execution-status.json", out); return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="quick"); p.add_argument("--output")
    a = p.parse_args(); print(json.dumps(run(a.output or ROOT/f"results/full-fermionic-SQD-durable-{a.mode}-v1", a.mode)))
