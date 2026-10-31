"""Versioned HF-separated growth with actual sparse two-determinant preparation."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
from time import perf_counter
import numpy as np
from scipy.linalg import eigh
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from .common import ROOT, digest, file_digest, rng_for, save_json, save_npz, save_binary
from .chemistry import load_reference
from .sparse_initialization import prepare_two_configurations
from .adapt_qsci_native import NamedHamiltonian, paper_pool, perturbative_target, gradient_data, hf_gradient, hf_angle, Pauli, update_state
from .qsci_growth import orbital_energies, selected_subspace
from .ansatz import compile_logical, circuit_bytes, measurement_equivalent_circuit, resources
from .study.core import rss_peak


METHOD = "hf_separated_adapt_qsci_MP2det_exactprep"


def preparation(n, ids, coefficients, rotations):
    qc = prepare_two_configurations(n, ids, coefficients)
    for label, angle in rotations:
        qc.append(PauliEvolutionGate(SparsePauliOp(label), time=-float(angle)), range(n))
    return compile_logical(qc)


def restore(n, ids, coefficients, rotations):
    state = np.zeros(1 << n, complex); state[ids] = coefficients
    for label, theta in rotations:
        state = update_state(state, Pauli(label), theta)
    return state


def grow(ref, Ns, R, seed, directory, config):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    meta = ref["meta"]; n, hf = meta["n_qubits"], meta["hf_id"]
    task = {"problem_id": meta["request_digest"], "method": METHOD, "Ns": Ns, "R": R, "seed": seed, "config": config}
    task_id = digest(task)
    if (directory / "terminal.json").exists():
        previous = json.loads((directory / "terminal.json").read_text())
        if previous["task_id"] != task_id:
            raise RuntimeError("Task identity drift")
        return previous
    t0 = perf_counter(); backend = NamedHamiltonian(ref["h"]); pool = paper_pool(n)
    t = perf_counter()
    ids, coefficients = perturbative_target(backend, ref["ids"], hf, orbital_energies(ref), n)
    initial_parameter_count = int(len(ids) == 2)
    initial_qc = preparation(n, ids, coefficients, [])
    initial_state = restore(n, ids, coefficients, [])
    parity = 1-abs(np.vdot(Statevector.from_instruction(initial_qc).data, initial_state))**2
    if abs(parity) > 1e-9:
        raise RuntimeError("Physical initial preparation mismatch")
    init = {"kind": "EXACT_TWO_CONFIGURATION_RY_PHASE_X_CX", "target_ids": ids.tolist(),
            "coefficients_real": coefficients.real.tolist(), "coefficients_imag": coefficients.imag.tolist(),
            "resources": resources(initial_qc, 1 if len(ids) == 2 else 0, 0), "parity_error": float(parity),
            "initialization_seconds_current_attempt": perf_counter()-t,
            "not_paper_figure_reproduction": "explicit MP partition and sparse circuit initialization"}
    cps = sorted(directory.glob("step-*.json")); rotations = []; prior_wall = 0.; previous_energy = None
    cumulative = {"raw_simulated_shots": 0, "pool_evaluations": 0, "classical_angle_calls": 0,
                  "ci_seconds": 0., "ranking_seconds": 0., "sampling_seconds": 0.}
    if cps:
        last = json.loads(cps[-1].read_text())
        assert last["task_id"] == task_id and last["initialization"]["target_ids"] == ids.tolist()
        cumulative = last["cumulative"]; prior_wall = last["wall_seconds_total"]
        previous_energy = last["native_qsci_energy"]
        rotations = list(last["rotations"])
        if last["next_rotation"] is not None:
            rotations.append(last["next_rotation"])
        else:
            result = {"task_id": task_id, **task, "status": last["termination_reason"], "checkpoints": len(cps),
                      "cumulative": cumulative, "wall_seconds_total": prior_wall, "hardware_shots": 0, "recovered_from_checkpoint": True}
            save_json(directory / "terminal.json", result); return result
    state = restore(n, ids, coefficients, rotations)
    terminal_status = "UNREACHED"; count = len(cps)
    for step in range(len(cps), config["max_steps"]+1):
        if shutil.disk_usage(directory).free < config["free_floor_bytes"]:
            terminal_status = "DISK_FLOOR_CAPPED"; break
        if perf_counter()-t0+prior_wall > config["trajectory_seconds"]:
            terminal_status = "WALL_CAPPED"; break
        if rss_peak() > config["worker_ram_bytes"]:
            terminal_status = "RSS_CAPPED"; break
        rng, rng_seed = rng_for("B2-exactprep-growth-v1", task_id, step)
        t = perf_counter(); probability = abs(state)**2
        ns = rng.multinomial(Ns, probability/probability.sum())
        observed = np.flatnonzero(ns)
        count_path = directory / f"counts-{step:04d}.npz"
        if count_path.exists():
            with np.load(count_path) as z:
                if not np.array_equal(z["ids"], observed) or not np.array_equal(z["counts"], ns[observed]):
                    raise RuntimeError("Durable count / deterministic regeneration mismatch")
        else:
            save_npz(count_path, ids=observed, counts=ns[observed])
        cumulative["sampling_seconds"] += perf_counter()-t
        cumulative["raw_simulated_shots"] += Ns
        S = selected_subspace(observed, ns[observed], ref["ids"], hf, R, True)
        t = perf_counter(); hs = backend.matrix(S, S)
        w, v = eigh(hs, subset_by_index=(0, 0)); energy, c = float(w[0]), v[:, 0]
        residual = float(np.linalg.norm(hs @ c-energy*c))
        if residual > 1e-8:
            raise RuntimeError("CI residual failed")
        cumulative["ci_seconds"] += perf_counter()-t
        reason = "STEP_CAP" if step == config["max_steps"] else None
        if previous_energy is not None and abs(energy-previous_energy) < 1e-5:
            reason = "NATIVE_QSCI_ENERGY_PLATEAU_NOT_VQE_STATIONARITY"
        next_rotation = None; selected_gradient = predicted = None
        if not reason:
            t = perf_counter(); keep = S != hf; ci_ids, ci_c = S[keep], c[keep]
            norm = np.linalg.norm(ci_c)
            if norm < 1e-12:
                reason = "ZERO_NONHF_CI_COMPONENT"
            else:
                ci_c = ci_c/norm
                rows = gradient_data(backend, ci_ids, ci_c, pool, hf)
                A = backend.expectation(ci_ids, ci_c); EHF = float(ref["h"][hf, hf].real)
                coupling = float((backend.matrix([hf], ci_ids) @ ci_c)[0].real)
                gradients = [hf_gradient(EHF, A, row["C"], coupling, row["rH"], row["rS"]) for row in rows]
                cumulative["pool_evaluations"] += len(pool)
                chosen = min(range(len(pool)), key=lambda i: (-abs(gradients[i]), i))
                selected_gradient = float(gradients[chosen])
                if abs(selected_gradient) < 1e-12:
                    reason = "CI_GRADIENT_ZERO_NOT_FULL_VQE_CONVERGENCE"
                else:
                    pi, pc = pool[chosen].act(ci_ids, ci_c)
                    angle, predicted, calls = hf_angle(EHF=EHF, A=A, B=backend.expectation(pi, pc), v=coupling, **rows[chosen])
                    next_rotation = [pool[chosen].label, angle]
                    cumulative["classical_angle_calls"] += calls
            cumulative["ranking_seconds"] += perf_counter()-t
        qc = preparation(n, ids, coefficients, rotations)
        sampler, removed = measurement_equivalent_circuit(qc)
        actual = Statevector.from_instruction(qc).data
        parity = 1-abs(np.vdot(actual, state))**2
        if abs(parity) > 1e-8 or not np.allclose(abs(Statevector.from_instruction(sampler).data)**2, probability, atol=1e-9):
            raise RuntimeError("Actual preparation/sampling parity failed")
        save_binary(directory / "latest-preparation.qpy", circuit_bytes(qc))
        save_binary(directory / "latest-sampling.qpy", circuit_bytes(sampler))
        save_npz(directory / "latest-state.npz", state=state, probability=probability)
        cp = {"task_id": task_id, "step": step, "method": METHOD, "Ns": Ns, "R_cap": R, "seed": seed,
              "initialization": init, "rotations": rotations, "next_rotation": next_rotation,
              "selected_gradient": selected_gradient, "predicted_classical_surrogate_energy": predicted,
              "selected_ids": S.tolist(), "ci_coefficients_real": c.real.tolist(), "ci_residual": residual,
              "native_qsci_energy": energy, "native_qsci_error_mHa": 1000*(energy-meta["fci_energy"]),
              "native_HF_insertion": True, "input_state_energy": float(np.vdot(state, ref["h"] @ state).real),
              "resources": resources(sampler, initial_parameter_count+len(rotations), len(rotations)),
              "preparation_resources": resources(qc, initial_parameter_count+len(rotations), len(rotations)),
              "initial_fixed_coefficient_variables": initial_parameter_count, "growth_optimized_angle_count": len(rotations),
              "removed_terminal_diagonal": removed, "state_parity_error": float(parity),
              "draw_seed": str(rng_seed), "cumulative": dict(cumulative), "termination_reason": reason,
              "named_matrix_requests_current_attempt": backend.matrix_entry_requests,
              "named_sparse_actions_current_attempt": backend.sparse_actions,
              "wall_seconds_total": perf_counter()-t0+prior_wall, "peak_process_rss": rss_peak(),
              "FCI_used_for_selection": False, "hardware_shots": 0,
              "scope": "small-statevector sampling; actual initialization gates; not scalable-backend evidence"}
        save_json(directory / f"step-{step:04d}.json", cp); count += 1
        if reason:
            terminal_status = reason; break
        rotations = rotations+[next_rotation]
        state = update_state(state, Pauli(next_rotation[0]), next_rotation[1]); previous_energy = energy
    result = {"task_id": task_id, **task, "status": terminal_status, "checkpoints": count, "cumulative": cumulative,
              "wall_seconds_total": perf_counter()-t0+prior_wall, "hardware_shots": 0,
              "FCI_used_for_selection": False, "whole_study_complete": False}
    save_json(directory / "terminal.json", result)
    return result


def worker(case, root, source, config, mode):
    import fcntl
    from threadpoolctl import threadpool_limits
    from pyscf import lib
    with threadpool_limits(limits=1):
        lib.num_threads(1)
        lock_path = Path(root) / "locks" / (case+".lock"); lock_path.parent.mkdir(exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ref = load_reference(Path(source) / "references" / case)
            for Ns in ([1000] if mode == "quick" else [1000, 100000]):
                for R in ([32] if mode == "quick" else [32, 128]):
                    for seed in ([101] if mode == "quick" else [101, 211, 307]):
                        if shutil.disk_usage(root).free < config["free_floor_bytes"] or sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file()) > config["output_cap_bytes"]:
                            raise RuntimeError("Resource boundary; retain prefix")
                        d = Path(root) / "trajectories" / case / f"M{Ns}-R{R}-s{seed}"
                        if (d / "failure.json").exists():
                            continue
                        started = perf_counter()
                        try:
                            grow(ref, Ns, R, seed, d, config)
                        except Exception as exc:
                            import traceback
                            save_json(d / "failure.json", {"status": "FAILED_PRESERVED", "case_id": case, "Ns": Ns, "R": R, "seed": seed,
                                      "error": repr(exc), "traceback": traceback.format_exc(), "current_attempt_seconds": perf_counter()-started})
            return case


def run(root, source, mode, workers):
    root, source = Path(root).resolve(), Path(source).resolve()
    if root == source or source in root.parents:
        raise RuntimeError("Protect immutable source")
    root.mkdir(exist_ok=True, parents=True)
    cases = ["H2_R0.7", "H4_R1.1"] if mode == "quick" else sorted(p.name for p in (source / "references").iterdir() if p.is_dir())
    config = {"max_steps": 8 if mode == "quick" else 64, "trajectory_seconds": 180,
              "worker_ram_bytes": 2*2**30, "free_floor_bytes": 3*2**30, "output_cap_bytes": int(1.5*2**30)}
    files = ["benchmark/qsci_exact_initialization.py", "benchmark/sparse_initialization.py", "benchmark/adapt_qsci_native.py", "benchmark/qsci_growth.py", "benchmark/ansatz.py", "study-v1/BASELINES_B2_EXACT_INIT.md"]
    mf = {"schema": "B2-HF-separated-exact-physical-initialization-v2", "source": str(source), "mode": mode,
          "cases": cases, "config": config, "implementation": {p: file_digest(ROOT / p) for p in files}}
    if (root / "manifest.json").exists() and json.loads((root / "manifest.json").read_text()) != mf:
        raise RuntimeError("Code/config drift; successor version required")
    if mode == "standard":
        quick = ROOT / "results/sampler-baselines-B2-exactprep-quick-v2/validation.json"
        if not quick.exists() or json.loads(quick.read_text())["status"] != "QUICK_PHYSICAL_INIT_GROWTH_VALIDATED":
            raise RuntimeError("Quick actual growth validation required")
    save_json(root / "manifest.json", mf)
    with ProcessPoolExecutor(max_workers=min(2, max(1, workers))) as pool:
        futures = [pool.submit(worker, case, root, source, config, mode) for case in cases]
        for f in as_completed(futures):
            print("B2 physical initializer growth completed "+f.result(), flush=True)
    return {"status": "EXECUTED_PENDING_INDEPENDENT_AUDIT", "cases": len(cases), "whole_study_complete": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=["quick", "standard"], default="quick")
    parser.add_argument("--output"); parser.add_argument("--source", default=str(ROOT / "results/standard-v1")); parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.output or ROOT / f"results/sampler-baselines-B2-exactprep-{args.mode}-v2", args.source, args.mode, args.workers)))
