"""Actual fixed-topology coefficient-control experiments with paid pilot counts."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter
import zlib
import numpy as np
from qiskit import QuantumCircuit
from threadpoolctl import threadpool_limits
from .common import ROOT, digest, file_digest, rng_for, save_json
from .resource_adaptive_sampler import integral_problem
from .sparse_qe_circuit import SparseQECircuit
from .parameter_control_primitives import vqe_update
from .local_sampler_primitives import LocalQE, observed_ci, pack_counts64
from .ansatz import circuit_bytes, measurement_equivalent_circuit, resources
from .study.core import rss_peak
from .study.reanalysis import frozen_inventory

MODES = ("source_qsci_grid", "fixed_pi4", "one_parameter_vqe", "blockwise_vqe", "full_vqe", "qsci_coordinate")
NAMESPACE = "RAACS-fixed-operator-parameter-control-v1"


def source_queue(mode):
    source = ROOT/f"results/RAACS-{'quick' if mode=='quick' else 'standard'}-v1"
    cases = ["H2_R0.7", "H4_R1.1"] if mode == "quick" else sorted(p.name for p in (source/"cells").iterdir() if p.is_dir())
    seeds = [101] if mode == "quick" else [101, 211, 307]
    return source, [{"case": case, "seed": seed, "source_cell": str(source/"cells"/case/(
        "full_cost-exact_small" if mode == "quick" else f"full_cost-qsci_local_grid-last_only-s{seed}-exact_small"))}
        for case in cases for seed in seeds]


def coordinate_update(initial, measure):
    theta = np.asarray(initial, float).copy(); E, initial_draw = measure(theta, "initial", 0)
    trials = []; used = 0
    for j in range(len(theta)):
        base = theta.copy(); choices = [(E, theta.copy(), None)]
        for shift in (-np.pi/4, -np.pi/8, np.pi/8, np.pi/4):
            if used >= 32: break
            trial = base.copy(); trial[j] += shift
            value, draw = measure(trial, "trial", used+1); used += 1
            trials.append({"trial": used, "coordinate": j, "theta": trial.tolist(), "energy_Ha_pilot": value, "draw": draw})
            choices.append((value, trial, draw))
        best = min(choices, key=lambda x: (x[0], tuple(x[1])))
        if best[0] < E-1e-6: E, theta = best[0], best[1].copy()
        if used >= 32: break
    return {"theta": theta.tolist(), "pilot_energy_Ha_selection_biased": E, "initial_draw": initial_draw,
            "trials": trials, "pilot_evaluations": used+1, "status": "REGISTERED_COORDINATE_SWEEP_FINISHED",
            "all_trials_paid": True, "full_Fock_statevector_allocated": False, "FCI_used": False}


def config(mode):
    construction, queue = source_queue(mode)
    files = ("benchmark/parameter_controls.py", "benchmark/parameter_control_primitives.py", "benchmark/sparse_qe_circuit.py",
             "benchmark/local_sampler_primitives.py", "benchmark/integral_hamiltonian.py", "benchmark/ansatz.py")
    return {"schema": "fixed-operator-RAACS-parameter-controls-v1", "mode": mode, "queue": queue, "modes": list(MODES),
            "construction": str(construction), "source": str(ROOT/"results/standard-v1"), "M": 10000, "R": 64,
            "final_repeats": 4 if mode == "quick" else 20, "support_cap": 512,
            "BFGS": {"max_calls": 300, "max_iterations": 80, "seconds": 30., "gtol": 1e-6},
            "implementation": {p: file_digest(ROOT/p) for p in files}, "spec_sha256": file_digest(ROOT/"study-v1/PARAMETER_CONTROLS.md"),
            "free_floor_bytes": 3*2**30, "output_boundary_bytes": 96*2**20, "RSS_boundary_bytes": 2*2**30,
            "post_construction_conditional_not_new_online_discovery": True, "hardware_shots": 0}


def execute_cell(root, cfg, item, mode):
    case = item["case"]; source_cell = Path(item["source_cell"])
    source_result = json.loads((source_cell/"result.json").read_text())
    if file_digest(source_cell/"events.sqlite") != source_result["event_database_sha256"]: raise RuntimeError("Frozen source decisions changed")
    backend, info = integral_problem(Path(cfg["source"])/"references"/case)
    actions = source_result["accepted_operators"]
    ops = [LocalQE(tuple(a["source"]), tuple(a["target"]), a["n_qubits"]) for a in actions]
    initial = np.array([a["theta"] for a in actions]); inherited = source_result["pilot_raw_model_shots"]
    folder = root/"cells"/case/f"s{item['seed']}"/mode; folder.mkdir(parents=True, exist_ok=True)
    task = {"case": case, "source_seed": item["seed"], "mode": mode,
            "source_database_sha256": source_result["event_database_sha256"], "problem": info,
            "operators": [o.identity for o in ops], "initial_theta": initial.tolist(), "inherited_pilot_shots": inherited,
            "experiment_manifest_sha256": file_digest(root/"manifest.json")}
    path = folder/"result.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["task"] != task: raise RuntimeError("Parameter control task identity drift")
        if file_digest(folder/"events.sqlite") != old["events_sha256"]: raise RuntimeError("Parameter control event digest mismatch")
        return old
    if (folder/"events.sqlite").exists(): raise RuntimeError("Partial parameter control evidence; no free retry")
    started = perf_counter(); sim = SparseQECircuit(info["n_qubits"], info["hf_id"], ops, cfg["support_cap"], backend)
    E0, g0 = sim.energy_gradient(initial)
    db = sqlite3.connect(folder/"events.sqlite"); db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE draws(id TEXT PRIMARY KEY,role TEXT,slot INTEGER,theta TEXT,shots INTEGER,seed TEXT,counts BLOB,energy_Ha REAL,selected_ids TEXT)")
    new_pilots = 0
    def measure(theta, role, slot, shots=64):
        nonlocal new_pilots
        identity = digest([task, role, slot, np.asarray(theta).tolist(), shots]); seed = rng_for(NAMESPACE, identity)[1]
        ids, counts = sim.counts(theta, shots, seed)
        ci = observed_ci(backend, dict(zip(map(int, ids), map(int, counts))), info["hf_id"], info["nalpha"], info["nbeta"], cfg["R"])
        db.execute("INSERT INTO draws VALUES(?,?,?,?,?,?,?,?,?)", (identity, role, slot, json.dumps(np.asarray(theta).tolist()), shots,
                   str(seed), pack_counts64(ids, counts), ci["energy_Ha"], json.dumps(ci["selected_ids"]))); db.commit()
        if role != "independent_final": new_pilots += shots
        return ci["energy_Ha"], identity
    if mode in ("one_parameter_vqe", "blockwise_vqe", "full_vqe"):
        control = vqe_update(sim, initial, mode, **cfg["BFGS"]); theta = np.array(control["theta"])
    elif mode == "qsci_coordinate":
        control = coordinate_update(initial, measure); theta = np.array(control["theta"])
    else:
        theta = initial.copy() if mode == "source_qsci_grid" else np.full(len(initial), np.pi/4)
        control = {"status": "REGISTERED_FIXED_COEFFICIENTS", "actual_objective_evaluations": 0, "theta": theta.tolist()}
    known_final = cfg["M"]-inherited-new_pilots
    if known_final <= 0: raise RuntimeError("Registered conditional parameter budget infeasible; do not fabricate final counts")
    final_energies = []; final_draws = []
    for rep in range(cfg["final_repeats"]):
        E, did = measure(theta, "independent_final", rep, known_final); final_energies.append(E); final_draws.append(did)
    prep = QuantumCircuit(info["n_qubits"], name="parameter-control")
    for q in range(info["n_qubits"]):
        if info["hf_id"] >> q & 1: prep.x(q)
    for op, angle in zip(ops, theta): prep = op.append(prep, angle)
    sampling, removed = measurement_equivalent_circuit(prep)
    (folder/"preparation.qpy").write_bytes(circuit_bytes(prep)); (folder/"sampling.qpy").write_bytes(circuit_bytes(sampling))
    save_json(folder/"optimization.json", control)
    E, gradient = sim.energy_gradient(theta)
    result = {"status": "COMPLETED", "task": task, "mode": mode, "initial_theta": initial.tolist(), "final_theta": theta.tolist(),
              "physical_VQE_energy_initial_Ha": E0, "physical_VQE_energy_final_Ha": E,
              "final_full_gradient_inf": float(max(abs(gradient))) if len(theta) else 0.,
              "independent_QSCI_energies_Ha": final_energies, "final_draw_ids": final_draws,
              "inherited_pilot_shots": inherited, "new_QSCI_pilot_shots": new_pilots,
              "final_shots_per_rep": known_final, "final_repeats": cfg["final_repeats"],
              "new_actual_model_shots": new_pilots+cfg["final_repeats"]*known_final,
              "known_per_branch_model_budget": cfg["M"], "VQE_objective_QPU_shots": None if "vqe" in mode else 0,
              "VQE_objective_calls_are_not_measurement_shots": True, "physical_sparse_support_size": len(sim.ids),
              "physical_sparse_support_not_padded_into_QSCI": True,
              "n_independent_ansatz_parameters": len(ops), "n_optimized_variables": control.get("n_optimized_variables", len(ops) if mode == "qsci_coordinate" else 0),
              "sampling_resources": resources(sampling, len(ops), len(ops)), "preparation_resources": resources(prep, len(ops), len(ops)),
              "removed_terminal_diagonals": removed, "FCI_used": False, "full_Fock_vector_used_in_control": False,
              "actual_wall_seconds": perf_counter()-started, "peak_process_RSS_bytes": rss_peak(), "named_H_requests": backend.matrix_entry_requests,
              "hardware_shots": 0, "whole_study_complete": False}
    db.close(); result["events_sha256"] = file_digest(folder/"events.sqlite")
    result["artifact_sha256"] = {p: file_digest(folder/p) for p in ("preparation.qpy", "sampling.qpy", "optimization.json")}
    save_json(path, result); return result


def run(root, mode):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    with (root/"run.lock").open("a") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg = config(mode); path = root/"manifest.json"
        if path.exists() and json.loads(path.read_text()) != cfg: raise RuntimeError("Fixed-sequence parameter experiment drift")
        if mode == "standard":
            quick = ROOT/"results/parameter-controls-quick-v1/validation.json"
            if not quick.exists() or json.loads(quick.read_text())["status"] != "PARAMETER_CONTROL_COUNTS_AND_CIRCUITS_VALIDATED":
                raise RuntimeError("Actual quick independent qualification required")
        if not path.exists() and shutil.disk_usage(root).free < cfg["free_floor_bytes"]+cfg["output_boundary_bytes"]:
            raise RuntimeError("Insufficient bounded parameter-control storage")
        save_json(path, cfg); frozen_inventory(Path(cfg["source"]), root); rows = []
        for item in cfg["queue"]:
            for method in cfg["modes"]:
                if shutil.disk_usage(root).free < cfg["free_floor_bytes"] or rss_peak() > cfg["RSS_boundary_bytes"]: raise RuntimeError("Parameter-control resource boundary")
                if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > cfg["output_boundary_bytes"]: raise RuntimeError("Parameter-control output boundary")
                cell = root/"cells"/item["case"]/f"s{item['seed']}"/method
                if (cell/"failure.json").exists():
                    rows.append({"case": item["case"], "seed": item["seed"], "mode": method, "status": "FAILED_PRESERVED"}); continue
                try:
                    result = execute_cell(root, cfg, item, method)
                    rows.append({"case": item["case"], "seed": item["seed"], "mode": method, "status": result["status"]})
                except Exception as exc:
                    save_json(cell/"failure.json", {"error": repr(exc), "partial_raw_records_preserved": True, "failed_cost_unknown_not_zero": True})
                    rows.append({"case": item["case"], "seed": item["seed"], "mode": method, "status": "FAILED_PRESERVED"})
                save_json(root/"execution-status.json", {"status": "RUNNING", "terminal": len(rows), "planned": len(cfg["queue"])*len(MODES), "rows": rows})
        result = {"status": "PARAMETER_CONTROLS_EXECUTED_PENDING_AUDIT", "terminal": len(rows), "planned": len(cfg["queue"])*len(MODES),
                  "rows": rows, "whole_study_complete": False, "hardware_shots": 0}
        save_json(root/"execution-status.json", result); return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="quick"); p.add_argument("--output")
    a = p.parse_args(); print(json.dumps(run(a.output or ROOT/f"results/parameter-controls-{a.mode}-v1", a.mode)))
