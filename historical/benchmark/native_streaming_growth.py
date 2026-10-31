"""Actual small-system FCI-free construction. Not a scalable sampler backend."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import fcntl
import json
from pathlib import Path
import shutil
from time import perf_counter, time_ns
import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
from .ansatz import (make_pool, pool_catalog, hf_state, energy_and_gradient, state_from_parameters,
                     selection_gradients, build_circuit, resources, verify_circuit_state,
                     measurement_equivalent_circuit, circuit_bytes)
from .common import ROOT, digest, file_digest, rng_for, save_binary, save_json, save_npz
from .online_stopping_rules import ObservedFrame, OnlineStopRule, registered_rules
from .stopping_pilot_bank import PROBES, ci_features
from .study.core import rss_peak


@dataclass(frozen=True)
class GrowthSpec:
    method: str
    policy: str
    max_steps: int = 64
    task_seconds: float = 180.
    optimizer_maxiter: int = 200
    optimizer_gtol: float = 1e-7
    gradient_tolerance: float = 1e-3
    probe_shots: int = 64
    probe_R: int = 64
    seed: int = 101


def advance(problem, pool, indices, theta, current_gradients=None):
    """Same author generators / independent common ADAPT BFGS as old source.

    Prepares the next zero-angle extension using only the current gradient.
    No FCI target, future checkpoint, final trajectory length or teacher input.
    An already computed current gradient is reused, not evaluated twice.
    """
    initial = hf_state(problem["meta"]["n_qubits"], problem["meta"]["hf_id"])
    h = problem["h"]; start = perf_counter()
    components = 0
    if current_gradients is None:
        before_state = state_from_parameters(pool, indices, theta, initial)
        before = selection_gradients(pool, before_state, h); components = pool.size
    else:
        before = current_gradients
    selected = int(np.argmax(abs(before)))
    if type(pool).__name__ == "MVP_CEO":
        support = pool.get_qubits(selected)
        additions = [i for i in range(pool.size) if pool.get_qubits(i) == support and abs(before[i]) > 1e-10]
    else:
        additions = [selected]
    new_indices = list(indices)+additions; initial_theta = np.r_[theta, np.zeros(len(additions))]
    return new_indices, initial_theta, selected, additions, before, perf_counter()-start, components


def run_growth(problem, spec, root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True); meta = problem["meta"]
    if meta.get("FCI_computed") is not False or meta.get("ground_state_reference_energy_Ha") is not None:
        raise ValueError("Constructor requires physical no-FCI problem, not a reference-label object")
    if not meta["full_Fock_H_constructed"] or meta["n_qubits"] > 16:
        raise ValueError("This qualification backend is explicitly small/full-statevector only")
    names = {r.name for r in registered_rules()}
    if spec.policy not in names|{"native_terminal"}:
        raise ValueError("Unregistered policy")
    if spec.max_steps < 0 or spec.optimizer_maxiter < 1 or spec.probe_shots < 1 or spec.probe_R < 1 or spec.task_seconds <= 0:
        raise ValueError("Invalid construction/probe budget")
    config = {"schema": "actual-FCI-free-growth-v1", "problem_id": meta["problem_id"], "spec": asdict(spec),
              "problem_config_digest": meta["config_digest"], "physical_h_sha256": meta["artifact_sha256"]["small-hamiltonian.npz"],
              "rules": [asdict(r) for r in registered_rules()],
              "implementation": {p: file_digest(ROOT/p) for p in ("benchmark/native_streaming_growth.py", "benchmark/ansatz.py",
                  "benchmark/online_stopping_rules.py", "benchmark/stopping_pilot_bank.py", "benchmark/molecular_problem_no_fci.py")},
              "FCI_used": False, "max_RSS_bytes": 2*2**30, "free_floor_bytes": 3*2**30}
    path = root/"config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("Frozen growth configuration changed; use a successor")
    save_json(path, config)
    with (root/"run.lock").open("a") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/"result.json").exists():
            result = json.loads((root/"result.json").read_text())
            for name, sha in result["artifact_sha256"].items():
                if file_digest(root/name) != sha: raise RuntimeError("Growth artifact mismatch")
            return result
        start = perf_counter(); attempt = str(time_ns())
        save_json(root/"attempts"/f"{attempt}-start.json", {"status": "STARTED", "config_digest": digest(config)})
        t = perf_counter(); pool = make_pool(spec.method, meta["n_qubits"]); catalog = pool_catalog(pool)
        pool_seconds = perf_counter()-t; save_json(root/"pool.json", catalog)
        initial = hf_state(meta["n_qubits"], meta["hf_id"]); h = problem["h"]
        hsector = h[problem["ids"]][:, problem["ids"]].tocsr(); valid = set(map(int, problem["ids"]))
        rule = None if spec.policy in ("native_terminal", "gradient") else OnlineStopRule(next(r for r in registered_rules() if r.name == spec.policy))
        indices = []; theta = np.array([]); state = initial; steps = []
        cumulative = {}; seen = set(); pilot_spent = 0; pilot_seconds = 0.
        total_objective = total_gradient = total_iterations = total_components = 0; optimizer_seconds = compile_seconds = gradient_seconds = 0.
        previous_wall = 0.; last_cp = None
        # Resume committed steps only; reconstruct policy from committed observations.
        for cp_path in sorted((root/"steps").glob("step-*.json")):
            cp = json.loads(cp_path.read_text())
            if cp["config_digest"] != digest(config): raise RuntimeError("Checkpoint configuration drift")
            steps.append(cp); last_cp = cp
            if cp.get("pilot"):
                ev = cp["pilot"]
                for x, n in ev["valid_nonHF_counts"]: cumulative[x] = cumulative.get(x, 0)+n
                seen.update(x for x, _ in ev["valid_nonHF_counts"])
                pilot_spent += spec.probe_shots; pilot_seconds = ev["frame"]["cumulative_classical_seconds"]
                rule.observe(ObservedFrame(**ev["frame"]))
            total_objective += cp["objective_evaluations"]; total_gradient += cp["gradient_evaluations"]
            total_iterations += cp["optimizer_iterations"]; total_components += cp["pool_gradient_components_computed"]
            optimizer_seconds += cp["optimization_seconds"]; gradient_seconds += cp["pool_gradient_seconds"]
            compile_seconds += cp["compilation_verification_seconds"]; previous_wall = cp["cumulative_wall_seconds"]
        if last_cp:
            indices = last_cp["selected_operators"]; theta = np.array(last_cp["theta"])
            state = state_from_parameters(pool, indices, theta, initial)
        next_step = 0 if last_cp is None else last_cp["step"]+1
        terminal = last_cp.get("terminal") if last_cp else None
        current_for_next = None
        for step in range(next_step, spec.max_steps+1):
            if terminal: break
            opt = None; selection_seconds = 0.; selection_components = 0; additions = []; selected = None
            if step:
                indices, x0, selected, additions, before, selection_seconds, selection_components = advance(problem, pool, indices, theta, current_for_next)
                opt_start = perf_counter()
                opt = minimize(lambda x: energy_and_gradient(pool, indices, x, initial, h), x0, method="BFGS", jac=True,
                               options={"maxiter": spec.optimizer_maxiter, "gtol": spec.optimizer_gtol})
                theta = opt.x; opt_seconds = perf_counter()-opt_start
                state = state_from_parameters(pool, indices, theta, initial)
            else: opt_seconds = 0.
            tgrad = perf_counter(); current = selection_gradients(pool, state, h); current_seconds = perf_counter()-tgrad
            current_for_next = current
            norm = float(np.linalg.norm(current)); E = float(np.vdot(state, h@state).real)
            total_components += pool.size+selection_components; gradient_seconds += current_seconds+selection_seconds
            total_objective += int(opt.nfev) if opt else 0; total_gradient += int(opt.njev) if opt else 0
            total_iterations += int(opt.nit) if opt else 0; optimizer_seconds += opt_seconds
            tc = perf_counter(); prep = build_circuit(pool, indices, theta, meta["hf_id"]); parity = verify_circuit_state(prep, state)
            sampler, removed = measurement_equivalent_circuit(prep); p = abs(state)**2
            sample_p = abs(Statevector.from_instruction(sampler).data)**2
            if not np.allclose(p, sample_p, atol=1e-9): raise RuntimeError("Sampling compiler probability mismatch")
            compile_seconds_step = perf_counter()-tc; compile_seconds += compile_seconds_step
            r = resources(sampler, len(theta), len(indices)); ev = None
            elapsed = previous_wall+perf_counter()-start
            terminal = "NATIVE_GRADIENT_CONVERGED" if norm <= spec.gradient_tolerance else "MAX_STEPS" if step >= spec.max_steps else "TIME_CENSORED" if elapsed >= spec.task_seconds else None
            if rule is not None and terminal is None and step in PROBES:
                tp = perf_counter(); rng, seed = rng_for("PhaseC-actual-policy-pilots-v1", meta["problem_id"], spec.method, spec.policy, spec.seed, step)
                counts = rng.multinomial(spec.probe_shots, p/p.sum()); ids = np.flatnonzero(counts)
                current_seen = {int(x) for x in ids if int(x) in valid and int(x) != meta["hf_id"]}
                new = len(current_seen-seen); seen.update(current_seen)
                pairs = [[int(x), int(counts[x])] for x in ids if int(x) in valid and int(x) != meta["hf_id"]]
                for x, n in pairs: cumulative[x] = cumulative.get(x, 0)+n
                ordered = [meta["hf_id"]]+[x for x, _ in sorted(cumulative.items(), key=lambda a: (-a[1], a[0]))][:spec.probe_R-1]
                feature = ci_features(hsector, problem["ids"], ordered, meta["hf_id"], cumulative)
                pilot_spent += spec.probe_shots; pilot_seconds += perf_counter()-tp
                frame = ObservedFrame(meta["problem_id"], digest([meta["problem_id"], asdict(spec)]), step,
                    feature["qsci_energy_Ha"], feature["actual_dimension"], new, pilot_spent,
                    float(counts[meta["hf_id"]]/spec.probe_shots), r["cnot_count"], r["cnot_depth"], pilot_seconds,
                    norm, feature["external_residual_norm_Ha"], feature["pt2_proxy_Ha"], feature["residual_scope"])
                decision = rule.observe(frame)
                ev = {"frame": asdict(frame), "seed": str(seed), "raw_counts": [[int(x), int(counts[x])] for x in ids],
                      "valid_nonHF_counts": pairs, "selected_ids": ordered, "decision": decision}
                if decision: terminal = "POLICY_STOPPED"
            cp = {"config_digest": digest(config), "step": step, "selected_operators": indices, "theta": theta.tolist(),
                  "selected_this_step": selected, "added_group": additions, "energy_Ha": E,
                  "current_pool_gradient_norm": norm, "pool_gradient_components_computed": pool.size+selection_components,
                  "pool_gradient_seconds": current_seconds+selection_seconds,
                  "objective_evaluations": int(opt.nfev) if opt else 0, "gradient_evaluations": int(opt.njev) if opt else 0,
                  "optimizer_iterations": int(opt.nit) if opt else 0, "optimization_seconds": opt_seconds,
                  "optimizer_success": bool(opt.success) if opt else None, "optimizer_message": str(opt.message) if opt else "HF",
                  "optimizer_gradient_inf": float(np.max(abs(opt.jac))) if opt else None,
                  "resources": r, "preparation_resources": resources(prep, len(theta), len(indices)),
                  "qiskit_state_error": parity, "removed_terminal_diagonal_gates": removed,
                  "compilation_verification_seconds": compile_seconds_step, "pilot": ev, "terminal": terminal,
                  "cumulative_wall_seconds": previous_wall+perf_counter()-start, "FCI_used": False}
            cp["physical_checkpoint_id"] = digest([meta["problem_id"], catalog["catalog_digest"], indices, theta.tolist()])
            save_json(root/"steps"/f"step-{step:04d}.json", cp); steps.append(cp)
            save_json(root/"status.json", {"status": terminal or "RUNNING", "step": step, "pilot_shots": pilot_spent})
            print(f"actual {spec.method}/{spec.policy} step{step} E={E:.10f} {terminal or ''}", flush=True)
            if shutil.disk_usage(root).free < config["free_floor_bytes"] or rss_peak() > config["max_RSS_bytes"]:
                raise RuntimeError("Resource boundary; committed actual trajectory preserved")
            if terminal: break
        last = steps[-1]
        # Restore final circuits on resume at a committed terminal checkpoint.
        prep = build_circuit(pool, indices, theta, meta["hf_id"]); sampler, _ = measurement_equivalent_circuit(prep)
        save_binary(root/"final.preparation.qpy", circuit_bytes(prep)); save_binary(root/"final.sampling.qpy", circuit_bytes(sampler))
        save_npz(root/"final.npz", probability=abs(state)**2, theta=theta)
        result = {"status": terminal, "problem_id": meta["problem_id"], "spec": asdict(spec), "config_digest": digest(config),
                  "physical_checkpoint_id": last["physical_checkpoint_id"], "last_step": last["step"], "energy_Ha": last["energy_Ha"],
                  "resources": last["resources"], "current_pool_gradient_norm": last["current_pool_gradient_norm"],
                  "pilot_raw_model_shots": pilot_spent, "pilot_classical_seconds": pilot_seconds,
                  "objective_evaluations": total_objective, "gradient_evaluations": total_gradient,
                  "optimizer_iterations": total_iterations, "pool_gradient_components_computed": total_components,
                  "pool_setup_seconds_this_attempt": pool_seconds, "optimizer_seconds": optimizer_seconds,
                  "pool_gradient_seconds": gradient_seconds, "compilation_verification_seconds": compile_seconds,
                  "actual_attempt_wall_seconds": perf_counter()-start, "committed_cumulative_wall_seconds": last["cumulative_wall_seconds"],
                  "resume_prior_completed_steps": next_step, "unknown_interrupted_inflight_work": next_step>0,
                  "measured_process_RSS_hwm_bytes": rss_peak(), "FCI_used": False, "hardware_shots": 0,
                  "scope": "actual independent construction; full-statevector <=16q qualification, not scalable backend",
                  "artifact_sha256": {p: file_digest(root/p) for p in ("final.preparation.qpy", "final.sampling.qpy", "final.npz", "pool.json")}}
        save_json(root/"result.json", result); save_json(root/"attempts"/f"{attempt}-finish.json", result)
        return result
