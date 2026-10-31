"""Bounded all30 molecular Aer noise/routing experiment with durable chunks."""
from __future__ import annotations
import argparse
from io import BytesIO
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
from time import perf_counter, sleep
import zlib
import fcntl
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
from .common import ROOT, digest, file_digest, rng_for, save_json
from .gate_noise_runtime import load_sampler, compile_sampler, logical_probabilities, simulate_counts, PROFILES, THERMAL

METHODS = ["B1:source_final_ceo_ovp", "B1:source_final_ceo_mvp", "B1:lucj_ccsd_2_vqe",
           "RAACS:full_cost-qsci_local_grid-last_only-s101-exact_small"]
NAMESPACE = "gate-noise-all-molecule-cohort-v1"


def pack(value): return zlib.compress(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(), 6)
def unpack(value): return json.loads(zlib.decompress(value))


def settings(mode):
    source = ROOT/"results/standard-v1"
    cases = ["H2_R0.7", "H4_R1.1"] if mode == "quick" else sorted(p.name for p in (source/"references").iterdir() if (p/"reference.json").exists())
    if mode == "standard": assert len(cases) == 30
    files = ("gate_noise_study.py", "gate_noise_runtime.py", "bounded_selected_sci.py", "fermionic_sqd_durable.py",
             "fermionic_sqd_workflow.py", "full_sqd_durable_cohort.py", "integral_hamiltonian.py", "local_sampler_primitives.py",
             "resource_adaptive_sampler.py", "common.py", "ansatz.py", "sqd.py")
    return {"schema": NAMESPACE, "mode": mode, "source": str(source), "cases": cases, "methods": METHODS,
        "targets": ["all_to_all_cx", "line_cx"], "profiles": PROFILES, "thermal": THERMAL,
        "seeds": [101] if mode == "quick" else [101, 211, 307], "M": 250, "chunk_shots": 25, "R": 64,
        "simulation_cell_timeout_seconds": 300., "free_floor": int(2.75*2**30), "output_cap": 128*2**20,
        "RSS_cap": 2*2**30, "Aer_state_memory_cap_mb": 1024,
        "implementation": {"benchmark/"+p: file_digest(ROOT/"benchmark"/p) for p in files},
        "packages": {p: importlib.metadata.version(p) for p in ("qiskit-aer", "qiskit", "numpy", "pyscf", "qiskit-addon-sqd")},
        "base_lock_sha256": file_digest(ROOT/"uv.lock"), "spec_sha256": file_digest(ROOT/"study-v1/GATE_NOISE_FULL_COHORT.md"),
        "runtime": str(ROOT/".runtime-overlays/aer-noise-v1/bin/python"),
        "runtime_PYTHONPATH": str(ROOT/".venv/lib/python3.11/site-packages"), "hardware_shots": 0}


def database(root):
    db = sqlite3.connect(Path(root)/"events.sqlite"); db.execute("PRAGMA synchronous=FULL"); db.execute("PRAGMA busy_timeout=10000")
    db.execute("CREATE TABLE IF NOT EXISTS circuits(id TEXT PRIMARY KEY,case_id TEXT,method TEXT,target TEXT,status TEXT,record BLOB,qpy BLOB)")
    db.execute("CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,task TEXT,status TEXT,result BLOB)")
    db.execute("CREATE TABLE IF NOT EXISTS chunks(task TEXT,idx INTEGER,shots INTEGER,seed TEXT,counts BLOB,record BLOB,PRIMARY KEY(task,idx))")
    db.commit(); return db


def merge_chunks(db, tid):
    counts = {}; shots = 0
    for n, raw in db.execute("SELECT shots,counts FROM chunks WHERE task=? ORDER BY idx", (tid,)):
        block = unpack(raw); assert sum(block.values()) == n
        for x, value in block.items(): counts[int(x)] = counts.get(int(x), 0)+int(value)
        shots += n
    return counts, shots


def worker(root, tid):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text())
    assert all(file_digest(ROOT/p) == sha for p, sha in cfg["implementation"].items())
    db = database(root); task_raw, status = db.execute("SELECT task,status FROM tasks WHERE id=?", (tid,)).fetchone()
    assert status == "RUNNING" and not db.execute("SELECT 1 FROM chunks WHERE task=?", (tid,)).fetchone()
    task = json.loads(task_raw)
    circuit = qpy.load(BytesIO(zlib.decompress(db.execute("SELECT qpy FROM circuits WHERE id=?", (task["circuit"],)).fetchone()[0])))[0]
    with threadpool_limits(limits=1):
        for idx in range(cfg["M"]//cfg["chunk_shots"]):
            before = perf_counter(); seed = rng_for(NAMESPACE, cfg["mode"], tid, "chunk", idx)[1]
            counts, metadata = simulate_counts(circuit, task["profile"], cfg["chunk_shots"], seed)
            db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?)", (tid, idx, cfg["chunk_shots"], str(seed), pack(counts),
                pack({"simulator_seconds": perf_counter()-before, "backend": metadata, "hardware_shots": 0}))); db.commit()
    db.close()


def pid_is_owned_worker(pid, root, tid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    command = result.stdout.strip()
    return result.returncode == 0 and "benchmark.gate_noise_study" in command and str(root) in command and tid in command and "--worker-id" in command


def capacity(root, cfg):
    if shutil.disk_usage(root).free < cfg["free_floor"]: return "FREE_FLOOR_BOUNDARY"
    if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > cfg["output_cap"]: return "OUTPUT_BOUNDARY"
    return None


def stop_owned(proc):
    # Popen created this exact group; this is not a general PID cleanup facility.
    try: os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait(); return "ALREADY_EXITED_AT_BOUNDARY"
    try: proc.wait(timeout=10); return "SIGTERM"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL); proc.wait(); return "SIGTERM_THEN_SIGKILL"


def execute_child(root, tid, cfg):
    env = dict(os.environ, PYTHONPATH=cfg["runtime_PYTHONPATH"], OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    command = [cfg["runtime"], "-m", "benchmark.gate_noise_study", "--worker-root", str(root), "--worker-id", tid]
    started = perf_counter(); reason = None; termination = None
    with (root/"worker.log").open("ab") as logfile:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        save_json(root/"active-worker.json", {"pid": proc.pid, "task": tid, "command": command, "live_requires_OS_verification": True})
        while proc.poll() is None:
            if perf_counter()-started > cfg["simulation_cell_timeout_seconds"]: reason = "SIMULATION_TIME_CENSORED"
            else: reason = capacity(root, cfg)
            if not reason:
                rss = subprocess.run(["ps", "-p", str(proc.pid), "-o", "rss="], capture_output=True, text=True)
                if rss.returncode == 0 and rss.stdout.strip() and int(rss.stdout.strip())*1024 > cfg["RSS_cap"]:
                    reason = "WORKER_RSS_BOUNDARY"
            if reason:
                termination = stop_owned(proc); break
            sleep(.5)
    save_json(root/"active-worker.json", {"task": tid, "pid": proc.pid, "terminal_returncode": proc.returncode, "active": False})
    return {"returncode": proc.returncode, "boundary_reason": reason, "termination": termination,
        "parent_observed_child_seconds_including_startup": perf_counter()-started,
        "unreported_inflight_simulation_work_possible": proc.returncode != 0,
        "unreported_inflight_model_shots": None if proc.returncode != 0 else 0}


def postprocess(db, cfg, task, tid, counts):
    # Lazy imports keep every short Aer subprocess free of PySCF/SQD startup.
    from .resource_adaptive_sampler import integral_problem
    from .local_sampler_primitives import observed_ci
    from .fermionic_sqd_durable import DurableTrace, read_trace
    from .fermionic_sqd_workflow import FermionicSQDSpec
    from .bounded_selected_sci import projected_sqd
    from .full_sqd_durable_cohort import matched_d
    backend, info = integral_problem(Path(cfg["source"])/"references"/task["case"])
    before = perf_counter()
    ci = observed_ci(backend, counts, info["hf_id"], info["nalpha"], info["nbeta"], cfg["R"])
    out = {"problem": info, "QSCI": {k: ci[k] for k in ("selected_ids", "frequency_order", "actual_dimension", "energy_Ha")},
        "QSCI_seconds": perf_counter()-before, "QPU_construction_cost": None}
    seed = rng_for(NAMESPACE, cfg["mode"], tid, "full-SQD")[1]
    try:
        out["SQD"] = projected_sqd(backend.h1, backend.eri, backend.core, info["hf_id"], info["nalpha"], info["nbeta"],
            list(counts), list(counts.values()), FermionicSQDSpec(seed=seed), DurableTrace(db, tid))
        sqd = out["SQD"]; trace = read_trace(db, tid)
        out["same_actual_d"] = matched_d(backend, ci, sqd, trace, info["hf_id"])
        if sqd["recovery_trace"]:
            last = sqd["recovery_trace"][-1]
            xs, ws = last["returned_unique_ids"], last["returned_classical_weights"]
            ranked = [info["hf_id"]]+[x for x, w in sorted(zip(xs, ws), key=lambda z: (-z[1], z[0])) if x != info["hf_id"]]
            selected = sorted(ranked[:cfg["R"]])
            out["explicit_last_fullSQD_recovery_projection"] = {"energy_Ha": float(np.linalg.eigvalsh(backend.matrix(selected, selected))[0]),
                "dimension": len(selected), "selected_ids": selected, "classical_weights_not_quantum_counts": True,
                "new_ids_not_observed": sorted(set(selected)-set(counts)),
                "not_independent_standalone_recovery_algorithm": True, "new_quantum_shots": 0}
        else: out["explicit_last_fullSQD_recovery_projection"] = None
        out["status"] = "COMPLETED"
    except Exception as exc:
        out.update(status="READOUT_COMPLETE_SQD_FAILURE_PRESERVED", error=repr(exc))
    return out


def prepare_circuits(db, cfg):
    from .ansatz import circuit_bytes
    for case in cfg["cases"]:
        for method in METHODS:
            for target in cfg["targets"]:
                boundary = capacity(Path(db.execute("PRAGMA database_list").fetchone()[2]).parent, cfg)
                if boundary: raise RuntimeError("Native preparation capacity boundary: "+boundary)
                cid = digest([NAMESPACE, cfg["mode"], case, method, target, cfg["implementation"]])
                if db.execute("SELECT 1 FROM circuits WHERE id=?", (cid,)).fetchone(): continue
                qc, origin = load_sampler(case, method)
                if qc is None:
                    db.execute("INSERT INTO circuits VALUES(?,?,?,?,?,?,NULL)", (cid, case, method, target, "SOURCE_INITIALIZATION_MISSING", pack(origin))); db.commit(); continue
                before = perf_counter(); compiled, resources = compile_sampler(qc, target)
                original = Statevector(qc).probabilities()
                native_p = Statevector(compiled.remove_final_measurements(inplace=False)).probabilities()
                maximum = float(max(abs(original-logical_probabilities(native_p, resources["measurement_physical_to_logical_clbit"]))))
                if maximum > 1e-9: raise RuntimeError("Production native circuit logical mapping parity failed")
                record = {"source": origin, "resources": resources, "ideal_logical_distribution_error": maximum,
                    "compile_and_independent_parity_seconds": perf_counter()-before,
                    "compiler_parity_full_vector_scope": "independent <=16q check only, not sampler-policy input"}
                db.execute("INSERT INTO circuits VALUES(?,?,?,?,?,?,?)", (cid, case, method, target, "AVAILABLE", pack(record), zlib.compress(circuit_bytes(compiled), 6))); db.commit()


def run(root, mode):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    source = ROOT/"results/standard-v1"
    if root == source or source in root.parents or root in source.parents or any((p/"REPRODUCIBILITY.json").exists() for p in (root, *root.parents)):
        raise ValueError("Protect sealed sources")
    with (root/"run.lock").open("a") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg = settings(mode); manifest = root/"manifest.json"
        if manifest.exists(): assert json.loads(manifest.read_text()) == cfg
        else:
            if shutil.disk_usage(root).free < cfg["free_floor"]+cfg["output_cap"]: raise RuntimeError("Need noise output reserve plus freefloor")
            q = json.loads((ROOT/"results/gate-noise-runtime-qualification-v1/validation.json").read_text())
            assert q["status"] == "AER_SYNTHETIC_NOISE_AND_ROUTED_LOGICAL_MAPPING_QUALIFIED"
            if mode == "standard":
                q = json.loads((ROOT/"results/gate-noise-full-quick-v1/validation.json").read_text())
                assert q["status"] == "GATE_NOISE_COUNTS_RESOURCES_AND_SELECTED_SPACES_VALIDATED" and q["completed"] == 80
            save_json(manifest, cfg)
            from .gate_noise_runtime import noise_model
            save_json(root/"serialized-noise-models.json", {name: noise_model(name).to_dict(serializable=True) for name in cfg["profiles"]})
            capsule = root/"code-capsule"; capsule.mkdir()
            for p in cfg["implementation"]: (capsule/Path(p).name).write_bytes((ROOT/p).read_bytes())
            from .full_sqd_durable_cohort import verify_source
            verify_source(source, root)
        db = database(root); prepare_circuits(db, cfg)
        planned = len(cfg["cases"])*len(METHODS)*len(cfg["targets"])*len(cfg["profiles"])*len(cfg["seeds"])
        for cid, case, method, target, available in db.execute("SELECT id,case_id,method,target,status FROM circuits ORDER BY case_id,method,target").fetchall():
            for profile in cfg["profiles"]:
                for seed in cfg["seeds"]:
                    boundary = capacity(root, cfg)
                    if boundary:
                        save_json(root/"execution-status.json", {"status": "PENDING_CAPACITY_BOUNDARY", "reason": boundary, "planned": planned})
                        db.close(); return {"status": "PENDING_CAPACITY_BOUNDARY", "reason": boundary}
                    task = {"case": case, "method": method, "target": target, "profile": profile, "seed": seed, "circuit": cid, "M": cfg["M"]}
                    tid = digest([NAMESPACE, cfg["mode"], task]); existing = db.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()
                    if existing and existing[0] != "RUNNING": continue
                    if existing:
                        active = json.loads((root/"active-worker.json").read_text()) if (root/"active-worker.json").exists() else {}
                        if active.get("task") == tid and active.get("pid") and pid_is_owned_worker(active["pid"], root, tid):
                            while pid_is_owned_worker(active["pid"], root, tid): sleep(1.)
                        counts, shots = merge_chunks(db, tid)
                        result = {"status": "INTERRUPTED_ENGINEERING_PRESERVED", "known_committed_model_shots": shots,
                            "unreported_inflight_model_shots": None, "partial_trace_not_silently_retried": True}
                    elif available != "AVAILABLE":
                        result = {"status": "SOURCE_INITIALIZATION_MISSING", "known_committed_model_shots": 0, "new_quantum_acquisitions": 0}
                        db.execute("INSERT INTO tasks VALUES(?,?,?,NULL)", (tid, json.dumps(task), "RUNNING")); db.commit()
                    else:
                        db.execute("INSERT INTO tasks VALUES(?,?,?,NULL)", (tid, json.dumps(task), "RUNNING")); db.commit()
                        runtime = execute_child(root, tid, cfg); counts, shots = merge_chunks(db, tid)
                        if runtime["returncode"] == 0 and not runtime["boundary_reason"] and shots == cfg["M"]:
                            result = postprocess(db, cfg, task, tid, counts)
                        else:
                            result = {"status": runtime["boundary_reason"] or "SIMULATION_ENGINEERING_FAILURE_PRESERVED"}
                        result.update(simulation_runtime=runtime, known_committed_model_shots=shots,
                            new_model_acquisitions=1, hardware_shots=0)
                    db.execute("UPDATE tasks SET status=?,result=? WHERE id=?", (result["status"], pack(result), tid)); db.commit()
                    status = dict(db.execute("SELECT status,count(*) FROM tasks GROUP BY status"))
                    save_json(root/"execution-status.json", {"status": "RUNNING", "terminal": sum(status.values()), "planned": planned,
                        "cells_by_status": status, "known_committed_new_model_shots": db.execute("SELECT coalesce(sum(shots),0) FROM chunks").fetchone()[0],
                        "hardware_shots": 0})
        status = dict(db.execute("SELECT status,count(*) FROM tasks GROUP BY status")); db.close()
        out = {"status": "GATE_NOISE_FULL_COHORT_EXECUTED_PENDING_AUDIT", "planned": planned, "terminal": sum(status.values()),
            "cells_by_status": status, "database_sha256": file_digest(root/"events.sqlite"), "hardware_shots": 0, "whole_study_complete": False}
        assert out["terminal"] == planned
        save_json(root/"execution-status.json", out); return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="quick"); p.add_argument("--output")
    p.add_argument("--worker-root"); p.add_argument("--worker-id"); a = p.parse_args()
    if a.worker_root: worker(a.worker_root, a.worker_id)
    else: print(json.dumps(run(a.output or ROOT/f"results/gate-noise-full-{a.mode}-v1", a.mode)))
