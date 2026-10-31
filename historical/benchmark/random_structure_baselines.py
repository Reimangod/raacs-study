"""Final-size-matched random diagnostics and classical uniform proposals."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
from time import perf_counter
import numpy as np
from qiskit.quantum_info import Statevector
from .common import ROOT, digest, file_digest, rng_for, save_json, save_npz, save_binary
from .chemistry import load_reference
from .ansatz import make_pool, hf_state, state_from_parameters, build_circuit, compile_logical, measurement_equivalent_circuit, verify_circuit_state, resources, circuit_bytes
from .adapt_qsci_native import paper_pool, update_state, circuit
from .study.core import rss_peak


def method_specs(mode):
    rows = [{"name": "uniform_valid_classical", "family": "uniform_classical"}]
    for family in ("random_pauli", "random_qeb", "random_ceo", "trained_ceo_random_angles"):
        for seed in ([101] if mode == "quick" else [101, 211, 307]):
            rows.append({"name": f"{family}_s{seed}", "family": family, "seed": seed})
    return rows


def construct(case, ref, source_cp, source_path, spec, root):
    directory = Path(root) / "cells" / case / spec["name"]; directory.mkdir(exist_ok=True, parents=True)
    identity = digest([ref["meta"]["request_digest"], source_cp["checkpoint_id"], spec, "B4-random-v1"])
    if (directory / "result.json").exists():
        old = json.loads((directory / "result.json").read_text()); assert old["task_id"] == identity
        return old
    t = perf_counter(); meta = ref["meta"]; n, hf = meta["n_qubits"], meta["hf_id"]
    nops = len(source_cp["selected_operators"])
    result = {"task_id": identity, "case_id": case, "method": spec["name"], "spec": spec,
              "problem_id": meta["request_digest"], "FCI_used_for_selection": False, "hardware_shots": 0,
              "source_CEO_checkpoint_id": source_cp["checkpoint_id"], "source_checkpoint_sha256": file_digest(source_path),
              "source_saved_operator_count": nops, "historical_CEO_trajectory_seconds_as_recorded": source_cp.get("trajectory_seconds"),
              "cost_scope": "retrospective final-size diagnostic; historical CEO length/structure dependency is not free",
              "construction_simulated_raw_shots": 0, "objective_evaluations": 0, "n_optimized_variables": 0}
    try:
        if spec["family"] == "uniform_classical":
            probability = np.zeros(1 << n); probability[ref["ids"]] = 1/len(ref["ids"])
            save_npz(directory / "final.npz", probability=probability)
            result.update(status="CLASSICAL_UNIFORM_PROPOSAL_NOT_A_QUANTUM_CIRCUIT", sampler_type="CLASSICAL_PROPOSAL",
                          resources={"cnot_count": None, "cnot_depth": None, "total_depth": None, "parameter_count": None,
                                     "reason": "NO_QUANTUM_PREPARATION_IMPLEMENTED_OR_ASSUMED"},
                          valid_sector_probability=1., input_state_energy=None, energy_error_mHa=None)
        else:
            rng, seed = rng_for("B4-random-structure-angle-v1", case, spec, source_cp["checkpoint_id"])
            angles = rng.uniform(-np.pi, np.pi, nops)
            if spec["family"] == "random_pauli":
                pool = paper_pool(n); indices = rng.integers(len(pool), size=nops).tolist()
                rotations = [(pool[i].label, float(theta)) for i, theta in zip(indices, angles)]
                state = hf_state(n, hf)
                for i, theta in zip(indices, angles):
                    state = update_state(state, pool[i], theta)
                preparation = circuit(n, hf, rotations)
                result["selected_pauli_labels"] = [label for label, _ in rotations]
            else:
                family = "qeb" if spec["family"] == "random_qeb" else "ceo_ovp"
                pool = make_pool(family, n)
                indices = source_cp["selected_operators"] if spec["family"] == "trained_ceo_random_angles" else rng.integers(pool.size, size=nops).tolist()
                state = state_from_parameters(pool, indices, angles, hf_state(n, hf))
                preparation = build_circuit(pool, indices, angles, hf)
            parity = verify_circuit_state(preparation, state)
            sampler, removed = measurement_equivalent_circuit(preparation)
            probability = abs(state)**2
            if not np.allclose(abs(Statevector.from_instruction(sampler).data)**2, probability, atol=1e-9):
                raise RuntimeError("Actual random sampler changed Z distribution")
            save_npz(directory / "final.npz", state=state, probability=probability, angles=angles, indices=np.array(indices))
            save_binary(directory / "preparation.qpy", circuit_bytes(preparation)); save_binary(directory / "sampling.qpy", circuit_bytes(sampler))
            energy = float(np.vdot(state, ref["h"] @ state).real)
            result.update(status="FIXED_RANDOM_DIAGNOSTIC", sampler_type="QUANTUM_CIRCUIT_SIMULATION", rng_seed=str(seed),
                          resources=resources(sampler, nops, nops), preparation_resources=resources(preparation, nops, nops),
                          state_parity_error=parity, removed_terminal_diagonal=removed, valid_sector_probability=float(probability[ref["ids"]].sum()),
                          input_state_energy=energy, energy_error_mHa=1000*(energy-meta["fci_energy"]),
                          input_energy_scope="NOT_target_sector_variational_if_N_or_Sz_leaks; QSCI is separately sector projected",
                          n_coefficient_slots=nops, n_independent_ansatz_parameters=nops)
        result["final_sha256"] = file_digest(directory / "final.npz")
    except Exception as exc:
        import traceback
        result.update(status="FAILED_PRESERVED", error=repr(exc), traceback=traceback.format_exc())
    result.update(construction_wall_seconds=perf_counter()-t, peak_process_rss=rss_peak())
    save_json(directory / "result.json", result)
    return result


def worker(case, root, source, methods, config):
    import fcntl
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        lock_path = Path(root) / "locks" / (case+".lock"); lock_path.parent.mkdir(exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ref = load_reference(Path(source) / "references" / case)
            path = sorted((Path(source) / "checkpoints" / case / "ceo_ovp").glob("step-*.json"))[-1]
            cp = json.loads(path.read_text())
            for spec in methods:
                if shutil.disk_usage(root).free < config["free_floor_bytes"] or rss_peak() > config["rss_bytes"]:
                    raise RuntimeError("Resource boundary; retain prefix")
                if sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file()) > config["output_cap_bytes"]:
                    raise RuntimeError("Output boundary; retain prefix")
                construct(case, ref, cp, path, spec, root)
            return case


def run(root, source, mode, workers):
    root, source = Path(root).resolve(), Path(source).resolve()
    if root == source or source in root.parents:
        raise RuntimeError("Protect immutable source")
    root.mkdir(exist_ok=True, parents=True)
    cases = ["H2_R0.7", "H4_R1.1"] if mode == "quick" else sorted(p.name for p in (source / "references").iterdir() if p.is_dir())
    methods = method_specs(mode)
    config = {"rss_bytes": 2*2**30, "free_floor_bytes": 3*2**30, "output_cap_bytes": 2**30}
    files = ["benchmark/random_structure_baselines.py", "benchmark/ansatz.py", "benchmark/adapt_qsci_native.py", "study-v1/BASELINES_B4_RANDOM.md"]
    manifest = {"schema": "B4-random-structure-angles-v1", "mode": mode, "source": str(source), "case_ids": cases, "methods": methods,
                "implementation": {p: file_digest(ROOT / p) for p in files}, "resource_config": config,
                "budget": {"M_grid": [1000, 10000, 100000], "R_grid": [8, 16, 32, 64, 128, 256, 512], "evaluation_repeats": 4 if mode == "quick" else 20}}
    if (root / "manifest.json").exists() and json.loads((root / "manifest.json").read_text()) != manifest:
        raise RuntimeError("Code/config drift; use successor")
    if mode == "standard":
        p = ROOT / "results/random-B4-quick-v1/validation.json"
        if not p.exists() or json.loads(p.read_text())["status"] != "QUICK_RANDOM_DIAGNOSTICS_VALIDATED":
            raise RuntimeError("Actual quick validation required")
    save_json(root / "manifest.json", manifest)
    with ProcessPoolExecutor(max_workers=min(2, max(1, workers)), max_tasks_per_child=1) as pool:
        for f in as_completed([pool.submit(worker, c, root, source, methods, config) for c in cases]):
            print("B4 random controls completed "+f.result(), flush=True)
    return {"status": "EXECUTED_PENDING_AUDIT", "cells": len(cases)*len(methods), "whole_study_complete": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="quick"); p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output"); p.add_argument("--source", default=str(ROOT / "results/standard-v1")); a = p.parse_args()
    print(json.dumps(run(a.output or ROOT / f"results/random-B4-{a.mode}-v1", a.source, a.mode, a.workers)))
