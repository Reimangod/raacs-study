"""Actual TE-QSCI / Pauli-qDRIFT / physical shallow orbital baseline cohort."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
from time import perf_counter
import numpy as np
import ffsim
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from .common import ROOT, digest, file_digest, rng_for, save_json, save_npz, save_binary
from .ansatz import compile_logical, measurement_equivalent_circuit, resources, verify_circuit_state, circuit_bytes
from .chemistry import load_reference, determinant_ids
from .time_evolution_primitives import pauli_hamiltonian, apply_sequence, evolution_circuit, trotter_sequence, qdrift_sequence
from .study.core import rss_peak


def specs(mode):
    if mode == "quick":
        return ([{"name": f"TE_t1_steps{s}", "family": "TE", "time": 1., "steps": s} for s in (1, 4)] +
                [{"name": f"Pauli_qDRIFT_t1_steps64_s{s}", "family": "Pauli_qDRIFT", "time": 1., "steps": 64, "seed": s} for s in (101, 211)] +
                [{"name": f"orbital_brickwork_layers{l}_s101", "family": "orbital", "layers": l, "seed": 101} for l in (1, 2)])
    return ([{"name": f"TE_t{t:g}_steps{s}", "family": "TE", "time": t, "steps": s} for t in (.5, 1., 2.) for s in (1, 4)] +
            [{"name": f"Pauli_qDRIFT_t1_steps{n}_s{s}", "family": "Pauli_qDRIFT", "time": 1., "steps": n, "seed": s} for n in (64, 256) for s in (101, 211, 307)] +
            [{"name": f"orbital_brickwork_layers{l}_s{s}", "family": "orbital", "layers": l, "seed": s} for l in (1, 2) for s in (101, 211, 307)])


def orbital_control(meta, layers, rng):
    norb = meta["n_orbitals"]; nelec = (meta["nalpha"], meta["nbeta"])
    matrix = np.eye(norb); angles = []; pairs = []
    first_parity = (meta["nalpha"]-1) % 2
    for layer in range(layers):
        for a in range((first_parity+layer) % 2, norb-1, 2):
            b = a+1; angle = float(rng.uniform(-np.pi/4, np.pi/4))
            rotation = np.eye(norb); co, si = np.cos(angle), np.sin(angle)
            rotation[a, a] = rotation[b, b] = co
            rotation[a, b] = -si; rotation[b, a] = si
            matrix = rotation @ matrix; angles.append(angle); pairs.append([a, b])
    vec = ffsim.apply_orbital_rotation(ffsim.hartree_fock_state(norb, nelec), matrix, norb, nelec)
    ids, signs, _, _ = determinant_ids(norb, *nelec)
    state = np.zeros(1 << (2*norb), complex); state[ids] = vec*signs
    blocked = QuantumCircuit(2*norb)
    blocked.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), range(2*norb))
    blocked.append(ffsim.qiskit.OrbitalRotationJW(norb, matrix), range(2*norb))
    wires = [2*norb-1-2*i for i in range(norb)]+[2*norb-2-2*i for i in range(norb)]
    qc = QuantumCircuit(2*norb); qc.compose(blocked, qubits=wires, inplace=True)
    for a in range(norb):
        for b in range(a):
            qc.cz(2*norb-1-2*a, 2*norb-2-2*b)
    return state, qc, {"orbital_rotation": matrix.tolist(), "pairs": pairs, "angles": angles}, len(angles), len(angles)


def build_cell(case, ref, spec, terms, setup, root):
    d = Path(root) / "cells" / case / spec["name"]; d.mkdir(parents=True, exist_ok=True)
    task_id = digest([ref["meta"]["request_digest"], spec, "B3-nonvariational-v1"])
    if (d / "result.json").exists():
        result = json.loads((d / "result.json").read_text())
        assert result["task_id"] == task_id
        return result
    start = perf_counter(); meta = ref["meta"]; n = meta["n_qubits"]
    result = {"case_id": case, "method": spec["name"], "task_id": task_id, "spec": spec,
              "problem_id": meta["request_digest"], "FCI_used_for_selection": False, "hardware_shots": 0,
              "construction_simulated_raw_shots": 0, "objective_evaluations": 0,
              "shared_case_setup": setup, "simulation_scope": "small-statevector qualification, not practical scaling evidence"}
    try:
        rng, seed = rng_for("B3-nonvariational-circuit-v1", case, spec)
        if spec["family"] == "orbital":
            state, qc, details, n_parameters, slots = orbital_control(meta, spec["layers"], rng)
        else:
            if spec["family"] == "TE":
                sequence = trotter_sequence(terms, spec["time"], spec["steps"])
                details = {"product_order": "OpenFermion JW terms sorted lexicographically", "trotter_order": 1}
            else:
                sequence, indices, norm = qdrift_sequence(terms, spec["time"], spec["steps"], rng)
                details = {"sampled_term_indices": indices, "coefficient_L1_norm": norm,
                           "not_grouped_fermionic_SqDRIFT": True}
            initial = np.zeros(1 << n, complex); initial[meta["hf_id"]] = 1
            state = apply_sequence(initial, sequence)
            qc = evolution_circuit(n, meta["hf_id"], sequence)
            n_parameters, slots = 1, len(sequence)
            details["sequence"] = sequence
            details["parameter_binding"] = "all angles linked to one total-time variable, fixed Hamiltonian/sequence"
        preparation = compile_logical(qc)
        sampler, removed = measurement_equivalent_circuit(qc)
        parity = verify_circuit_state(preparation, state)
        probability = abs(state)**2
        if not np.allclose(abs(Statevector.from_instruction(sampler).data)**2, probability, atol=1e-9):
            raise RuntimeError("Z-equivalent sampler parity failed")
        save_npz(d / "final.npz", state=state, probability=probability)
        save_binary(d / "preparation.qpy", circuit_bytes(preparation)); save_binary(d / "sampling.qpy", circuit_bytes(sampler))
        energy = float(np.vdot(state, ref["h"] @ state).real)
        result.update(status="FIXED_NONVARIATIONAL_CIRCUIT", final_sha256=file_digest(d / "final.npz"),
                      resources=resources(sampler, n_parameters, slots), preparation_resources=resources(preparation, n_parameters, slots),
                      n_coefficient_slots=slots, n_independent_ansatz_parameters=n_parameters, n_optimized_variables=0,
                      details=details, rng_seed=str(seed), state_parity_error=parity, removed_terminal_diagonal=removed,
                      input_state_energy=energy, energy_error_mHa=1000*(energy-meta["fci_energy"]),
                      valid_sector_probability=float(probability[ref["ids"]].sum()))
    except Exception as exc:
        import traceback
        result.update(status="FAILED_PRESERVED", error=repr(exc), traceback=traceback.format_exc())
    result.update(construction_wall_seconds=perf_counter()-start, peak_process_rss=rss_peak())
    save_json(d / "result.json", result)
    return result


def worker(case, root, source, methods, config):
    import fcntl
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        lock_path = Path(root) / "locks" / (case+".lock"); lock_path.parent.mkdir(exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            t = perf_counter(); ref = load_reference(Path(source) / "references" / case)
            load_seconds = perf_counter()-t; t = perf_counter()
            terms, constant = pauli_hamiltonian(ref["h1"], ref["eri"], ref["meta"]["hf_id"], ref["meta"]["rhf_energy"])
            setup = {"load_seconds": load_seconds, "JW_integral_setup_seconds": perf_counter()-t,
                     "pauli_terms": len(terms), "omitted_identity_global_phase": constant,
                     "scope": "shared once per case, not added separately for every circuit"}
            save_json(Path(root) / "case-setup" / (case+".json"), {"terms": terms, **setup})
            for spec in methods:
                if shutil.disk_usage(root).free < config["free_floor_bytes"] or rss_peak() > config["worker_ram_bytes"]:
                    raise RuntimeError("Resource boundary: preserve completed prefix")
                if sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file()) > config["output_cap_bytes"]:
                    raise RuntimeError("Output boundary cap: preserve prefix")
                build_cell(case, ref, spec, terms, setup, root)
            return case


def run(root, source, mode, workers):
    root, source = Path(root).resolve(), Path(source).resolve()
    if root == source or source in root.parents:
        raise RuntimeError("Protect immutable source")
    root.mkdir(exist_ok=True, parents=True)
    cases = ["H2_R0.7", "H4_R1.1"] if mode == "quick" else sorted(p.name for p in (source / "references").iterdir() if p.is_dir())
    methods = specs(mode)
    files = ["benchmark/nonvariational_baselines.py", "benchmark/time_evolution_primitives.py", "benchmark/integral_hamiltonian.py", "benchmark/ansatz.py", "study-v1/BASELINES_B3_PAULI_TE.md"]
    config = {"worker_ram_bytes": 2*2**30, "free_floor_bytes": 3*2**30, "output_cap_bytes": int(1.5*2**30)}
    manifest = {"schema": "B3-Pauli-TE-qDRIFT-physical-orbital-v1", "mode": mode, "source": str(source), "case_ids": cases,
                "methods": methods, "resource_config": config, "implementation": {p: file_digest(ROOT / p) for p in files},
                "budget": {"M_grid": [1000, 10000, 100000], "R_grid": [8, 16, 32, 64, 128, 256, 512], "evaluation_repeats": 4 if mode == "quick" else 20}}
    if (root / "manifest.json").exists() and json.loads((root / "manifest.json").read_text()) != manifest:
        raise RuntimeError("Frozen B3 config/code drift")
    if mode == "standard":
        p = ROOT / "results/nonvariational-B3-quick-v1/validation.json"
        if not p.exists() or json.loads(p.read_text())["status"] != "QUICK_NONVARIATIONAL_CIRCUITS_VALIDATED":
            raise RuntimeError("Actual quick circuit audit required")
    save_json(root / "manifest.json", manifest)
    with ProcessPoolExecutor(max_workers=min(2, max(1, workers))) as pool:
        for f in as_completed([pool.submit(worker, case, root, source, methods, config) for case in cases]):
            print("B3 actual circuits completed "+f.result(), flush=True)
    return {"status": "EXECUTED_PENDING_INDEPENDENT_AUDIT", "planned_cells": len(cases)*len(methods), "whole_study_complete": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="quick"); p.add_argument("--workers", type=int, default=2)
    p.add_argument("--output"); p.add_argument("--source", default=str(ROOT / "results/standard-v1"))
    args = p.parse_args()
    print(json.dumps(run(args.output or ROOT / f"results/nonvariational-B3-{args.mode}-v1", args.source, args.mode, args.workers)))
