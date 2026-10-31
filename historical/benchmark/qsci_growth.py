"""Bounded, resumable paper-native QSCI circuit-growth experiments (B2)."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
from scipy.linalg import eigh

from .common import ROOT, digest, file_digest, provenance, rng_for, save_json, save_npz, save_binary
from .adapt_qsci_native import (NamedHamiltonian, paper_pool, gradient_data, standard_angle,
    hf_gradient, hf_angle, update_state, circuit, perturbative_target, prepare_target)


def orbital_energies(ref):
    # RHF active Fock; h1 already includes frozen-core Coulomb/exchange.
    na = ref["meta"]["nalpha"]
    J = np.einsum("pqii->pq", ref["eri"][:, :, :na, :na])
    K = np.einsum("piiq->pq", ref["eri"][:, :na, :na, :])
    f = ref["h1"]+2*J-K
    if np.max(abs(f-np.diag(np.diag(f)))) > 2e-5:
        raise RuntimeError("Active source orbitals are not RHF canonical")
    return np.diag(f)


def selected_subspace(ids, counts, valid, hf, R, hf_separated):
    allowed = set(map(int, valid))
    pairs = [(int(x), int(c)) for x, c in zip(ids, counts) if c > 0 and int(x) in allowed]
    order = [x for x, _ in sorted(pairs, key=lambda v: (-v[1], v[0]))]
    if hf_separated:
        order = [hf]+[x for x in order if x != hf]
    return np.array(order[:R], np.int64)


def restore_state(n, hf, rotations):
    from .adapt_qsci_native import Pauli
    state = np.zeros(1 << n, complex); state[hf] = 1
    for label, theta in rotations:
        state = update_state(state, Pauli(label), theta)
    return state


def grow(ref, method, Ns, R, seed, directory, config):
    from .ansatz import circuit_bytes, measurement_equivalent_circuit, resources, verify_circuit_state
    from .study.core import rss_peak
    from qiskit.quantum_info import Statevector
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    identity = {"problem": ref["meta"]["request_digest"], "method": method, "Ns": Ns, "R": R, "seed": seed, "config": config}
    task_id = digest(identity)
    if (directory / "terminal.json").exists():
        previous = json.loads((directory / "terminal.json").read_text())
        if previous["task_id"] != task_id:
            raise RuntimeError("Task identity changed")
        return previous
    start = perf_counter(); meta = ref["meta"]; n, hf = meta["n_qubits"], meta["hf_id"]
    backend = NamedHamiltonian(ref["h"]); pool = paper_pool(n)
    separated = method == "hf_separated_adapt_qsci_MP_m0_2"
    rotations = []; init_count = 0; initialization = {"kind": "HF_PRODUCT", "cnot_debt": 0}
    cumulative = {"raw_simulated_shots": 0, "pool_evaluations": 0, "classical_angle_calls": 0,
                  "ci_seconds": 0., "ranking_seconds": 0., "sampling_seconds": 0., "prior_wall_seconds": 0.}
    cps = sorted(directory.glob("step-*.json"))
    if cps:
        last = json.loads(cps[-1].read_text())
        if last["task_id"] != task_id:
            raise RuntimeError("Checkpoint identity changed")
        rotations = [tuple(v) for v in last["rotations"]]
        init_count, initialization = last["initial_rotation_count"], last["initialization"]
        cumulative = last["cumulative"]
        cumulative["prior_wall_seconds"] = last["wall_seconds_total"]
        step_start = last["step"]+1
        previous_energy = last["native_qsci_energy"]
        next_update = last["next_rotation"]
        backend.matrix_entry_requests = last["named_hamiltonian_entry_requests"]
        backend.sparse_actions = last["named_hamiltonian_sparse_actions"]
        if next_update is None:
            terminal = {"task_id": task_id, **identity, "status": last["termination_reason"], "checkpoints": len(cps),
                        "cumulative": cumulative, "hardware_shots": 0, "wall_seconds_total": last["wall_seconds_total"],
                        "FCI_used_for_growth": False, "whole_study_complete": False, "recovered_terminal_from_checkpoint": True}
            save_json(directory / "terminal.json", terminal)
            return terminal
        rotations.append(tuple(next_update))
    else:
        if separated:
            ids, c = perturbative_target(backend, ref["ids"], hf, orbital_energies(ref), n)
            state, rotations, fidelity = prepare_target(n, hf, ids, c, pool)
            init_count = len(rotations)
            initialization = {"kind": "TOP2_MP_FIRST_ORDER_GREEDY_POOL_PREPARATION", "target_ids": ids.tolist(),
                              "target_coefficients_real": c.real.tolist(), "fidelity": fidelity, "rotations": rotations,
                              "paper_setting_difference": "explicit_MP_partition; actual_preparation_gate_count_not_assumed12"}
        step_start = 0; previous_energy = None
    state = restore_state(n, hf, rotations)
    terminal_status = None; checkpoints = len(cps)
    for step in range(step_start, config["max_steps"]+1):
        if shutil.disk_usage(directory).free < config["free_floor_bytes"]:
            terminal_status = "DISK_FLOOR_CAPPED"; break
        if perf_counter()-start+cumulative["prior_wall_seconds"] > config["trajectory_seconds"]:
            terminal_status = "WALL_CAPPED"; break
        if rss_peak() > config["worker_ram_bytes"]:
            terminal_status = "RSS_CAPPED"; break
        rng, rng_seed = rng_for("B2-native-growth-draw-v1", task_id, step)
        t = perf_counter(); p = abs(state)**2
        count_path = directory / f"counts-{step:04d}.npz"
        if count_path.exists():
            with np.load(count_path) as z:
                counts = np.zeros(len(state), np.int64); counts[z["ids"]] = z["counts"]
            if counts.sum() != Ns:
                raise RuntimeError("Durable draw budget mismatch")
        else:
            counts = rng.multinomial(Ns, p/p.sum())
        cumulative["sampling_seconds"] += perf_counter()-t
        cumulative["raw_simulated_shots"] += Ns
        observed = np.flatnonzero(counts)
        S = selected_subspace(observed, counts[observed], ref["ids"], hf, R, separated)
        if not count_path.exists():
            save_npz(count_path, ids=observed, counts=counts[observed])
        if not len(S):
            terminal_status = "NO_VALID_OBSERVED_CONFIGURATION"; break
        t = perf_counter(); hs = backend.matrix(S, S)
        ew, ev = eigh(hs, subset_by_index=(0, 0)); energy, c = float(ew[0]), ev[:, 0]
        residual = float(np.linalg.norm(hs @ c-energy*c))
        if residual > 1e-8:
            raise RuntimeError("CI residual failed")
        cumulative["ci_seconds"] += perf_counter()-t
        reason = None; selected = None; selected_gradient = None; angle = None; predicted = None
        if previous_energy is not None and abs(energy-previous_energy) < 1e-5:
            reason = "NATIVE_QSCI_ENERGY_PLATEAU_NOT_VQE_STATIONARITY"
        elif step == config["max_steps"]:
            reason = "STEP_CAP"
        if not reason:
            t = perf_counter()
            ci_ids, ci_c = S, c
            if separated:
                keep = S != hf
                ci_ids, ci_c = S[keep], c[keep]
                norm = np.linalg.norm(ci_c)
                if norm < 1e-12:
                    reason = "ZERO_NONHF_CI_COMPONENT"
                else:
                    ci_c = ci_c/norm
            if not reason:
                rows = gradient_data(backend, ci_ids, ci_c, pool, hf if separated else None)
                A = backend.expectation(ci_ids, ci_c)
                if separated:
                    EHF = float(ref["h"][hf, hf].real)
                    v = float((backend.matrix([hf], ci_ids) @ ci_c)[0].real)
                    gs = [hf_gradient(EHF, A, row["C"], v, row["rH"], row["rS"]) for row in rows]
                else:
                    gs = [row["C"] for row in rows]
                cumulative["pool_evaluations"] += len(pool)
                chosen = min(range(len(pool)), key=lambda i: (-abs(gs[i]), i))
                selected_gradient = float(gs[chosen]); selected = pool[chosen].label
                if abs(selected_gradient) < 1e-12:
                    reason = "CI_GRADIENT_ZERO_NOT_FULL_VQE_CONVERGENCE"
                else:
                    pi, pc = pool[chosen].act(ci_ids, ci_c)
                    B = backend.expectation(pi, pc)
                    if separated:
                        angle, predicted, calls = hf_angle(EHF=EHF, A=A, B=B, v=v, **rows[chosen])
                    else:
                        angle, predicted = standard_angle(A, B, selected_gradient); calls = 1
                    cumulative["classical_angle_calls"] += calls
            cumulative["ranking_seconds"] += perf_counter()-t
        preparation = circuit(n, hf, rotations)
        sampler, removed = measurement_equivalent_circuit(preparation)
        verify = verify_circuit_state(preparation, state)
        sampling_state = Statevector.from_instruction(sampler).data
        if not np.allclose(abs(sampling_state)**2, p, atol=1e-9):
            raise RuntimeError("Sampler distribution parity failed")
        save_binary(directory / "latest-preparation.qpy", circuit_bytes(preparation))
        save_binary(directory / "latest-sampling.qpy", circuit_bytes(sampler))
        save_npz(directory / "latest-state.npz", state=state, probability=p)
        next_rotation = [selected, angle] if reason is None else None
        cp = {"task_id": task_id, "step": step, "method": method, "Ns": Ns, "R_cap": R,
              "draw_seed": str(rng_seed), "rotations": rotations, "initial_rotation_count": init_count,
              "initialization": initialization, "native_qsci_energy": energy,
              "native_qsci_error_mHa": 1000*(energy-meta["fci_energy"]),
              "native_HF_insertion": separated, "selected_ids": S.tolist(), "ci_coefficients_real": c.real.tolist(),
              "ci_residual": residual, "next_rotation": next_rotation, "selected_gradient": selected_gradient,
              "predicted_classical_surrogate_energy": predicted,
              "input_state_energy": float(np.vdot(state, ref["h"] @ state).real),
              "resources": resources(sampler, len(rotations), len(rotations)),
              "preparation_resources": resources(preparation, len(rotations), len(rotations)),
              "removed_terminal_diagonal": removed, "state_parity_error": verify,
              "cumulative": dict(cumulative), "named_hamiltonian_entry_requests": backend.matrix_entry_requests,
              "named_hamiltonian_sparse_actions": backend.sparse_actions,
              "wall_seconds_total": perf_counter()-start+cumulative["prior_wall_seconds"], "peak_rss": rss_peak(),
              "termination_reason": reason, "hardware_shots": 0, "fci_selection": False,
              "simulation_scope": "small-statevector sampler, sparse named-CI classical selection; not scalable backend evidence"}
        save_json(directory / f"step-{step:04d}.json", cp); checkpoints += 1
        if reason:
            terminal_status = reason; break
        from .adapt_qsci_native import Pauli
        rotations = rotations+[(selected, angle)]
        state = update_state(state, Pauli(selected), angle)
        previous_energy = energy
    terminal = {"task_id": task_id, **identity, "status": terminal_status, "checkpoints": checkpoints,
                "cumulative": cumulative, "hardware_shots": 0, "wall_seconds_total": perf_counter()-start+cumulative["prior_wall_seconds"],
                "FCI_used_for_growth": False, "whole_study_complete": False}
    save_json(directory / "terminal.json", terminal)
    return terminal


def case_worker(case_id, source, root, config, mode):
    import fcntl
    from threadpoolctl import threadpool_limits
    from .chemistry import load_reference
    from pyscf import lib
    with threadpool_limits(limits=1):
        lib.num_threads(1)
        lock_path = Path(root) / "locks" / f"{case_id}.lock"; lock_path.parent.mkdir(exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ref = load_reference(Path(source) / "references" / case_id)
            results = []
            for method in ("adapt_qsci_original", "hf_separated_adapt_qsci_MP_m0_2"):
                for Ns in ([1000] if mode == "quick" else [1000, 100000]):
                    for R in ([32] if mode == "quick" else [32, 128]):
                        for seed in ([101] if mode == "quick" else [101, 211, 307]):
                            directory = Path(root) / "trajectories" / case_id / method / f"M{Ns}-R{R}-s{seed}"
                            if shutil.disk_usage(root).free < config["free_floor_bytes"]:
                                raise RuntimeError("Disk floor before next trajectory; preserve completed prefix")
                            current_bytes = sum(f.stat().st_size for f in Path(root).rglob("*") if f.is_file())
                            if current_bytes > config["output_cap_bytes"]:
                                raise RuntimeError("Versioned output-size cap; preserve completed prefix")
                            if (directory / "failure.json").exists():
                                results.append(json.loads((directory / "failure.json").read_text()))
                                continue
                            try:
                                row = grow(ref, method, Ns, R, seed, directory, config)
                            except Exception as exc:
                                import traceback
                                cumulative_shots = 0
                                for f in directory.glob("counts-*.npz"):
                                    with np.load(f) as z:
                                        cumulative_shots += int(z["counts"].sum())
                                row = {"status": "FAILED_PRESERVED", "error": repr(exc), "traceback": traceback.format_exc(), "method": method, "Ns": Ns, "R": R, "seed": seed,
                                       "durable_raw_simulated_shots": cumulative_shots, "partial_checkpoint_count": len(list(directory.glob("step-*.json")))}
                                save_json(directory / "failure.json", row)
                            results.append(row)
                            print(json.dumps({"case": case_id, **{k: row.get(k) for k in ("method", "Ns", "R", "seed", "status", "checkpoints")}}), flush=True)
            save_json(Path(root) / "case-status" / f"{case_id}.json", results)
            return results


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--phase", choices=["baselines"], required=True)
    p.add_argument("--component", choices=["adapt_qsci"], default="adapt_qsci")
    p.add_argument("--mode", choices=["quick", "standard", "full"], default="quick")
    p.add_argument("--source", default=str(ROOT / "results/standard-v1")); p.add_argument("--output")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args(argv)
    if args.workers not in (1, 2):
        p.error("At most2 workers")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    source = Path(args.source).resolve(); root = Path(args.output or ROOT / f"results/sampler-baselines-B2-{args.mode}-v1").resolve()
    if root == source or source in root.parents:
        raise RuntimeError("Never modify source")
    root.mkdir(parents=True, exist_ok=True)
    config = {"max_steps": 8 if args.mode == "quick" else 64, "trajectory_seconds": 180,
              "worker_ram_bytes": 2*2**30, "free_floor_bytes": 3*2**30, "output_cap_bytes": 6*2**30}
    cases = ["H2_R0.7", "H4_R1.1"] if args.mode == "quick" else sorted(d.name for d in (source / "references").iterdir() if d.is_dir())
    paths = ["benchmark/adapt_qsci_native.py", "benchmark/qsci_growth.py", "benchmark/ansatz.py", "study-v1/BASELINES_B2.md"]
    manifest = {"schema": "B2-paper-QSCI-growth-v1", "mode": args.mode, "source": str(source), "config": config, "cases": cases,
                "code": {name: file_digest(ROOT / name) for name in paths}, "packages": provenance()["packages"],
                "hardware_shots": 0, "scope": "method-defined algorithm; common R rather than FCI-selected paperR; explicit MP initializer variant"}
    if (root / "manifest.json").exists() and json.loads((root / "manifest.json").read_text()) != manifest:
        raise RuntimeError("Frozen B2 identity changed; use successor version")
    save_json(root / "manifest.json", manifest)
    if args.mode != "quick":
        q = ROOT / "results/sampler-baselines-B2-quick-v1/validation.json"
        if not q.exists() or json.loads(q.read_text())["status"] != "QUICK_B2_VALIDATED":
            raise RuntimeError("Validate quick B2 first")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(case_worker, case, source, root, config, args.mode) for case in cases]
        for f in as_completed(futures):
            f.result()


if __name__ == "__main__":
    main()
