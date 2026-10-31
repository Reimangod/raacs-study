from __future__ import annotations
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from .ansatz import (make_pool, pool_catalog, hf_state, energy_and_gradient,
                     state_from_parameters, selection_gradients, build_circuit,
                     resources, verify_circuit_state, circuit_bytes,measurement_equivalent_circuit)
from qiskit.quantum_info import Statevector
from .common import digest, save_json, save_npz, provenance, file_digest, save_binary
from .metrics import analytic_metrics


def run_adapt(reference, method, output, max_steps=8, optimizer_maxiter=200,
              gradient_tolerance=1e-3, optimizer_gtol=1e-7, task_seconds=600,
              allow_extension=False):
    meta, h = reference["meta"], reference["h"]
    case_id = reference["directory"].name
    root = Path(output) / "checkpoints" / case_id / method
    root.mkdir(parents=True, exist_ok=True)
    pool = make_pool(method, meta["n_qubits"])
    catalog = pool_catalog(pool)
    save_json(root / "pool.json", catalog)
    config = {"schema": "adapt-2", "method": method, "problem": meta["request_digest"],
              "catalog_digest": catalog["catalog_digest"], "max_steps": max_steps,
              "optimizer": "BFGS;full-ansatz;analytic-adjoint-gradient;cold-identity-Hessian",
              "optimizer_maxiter": optimizer_maxiter, "optimizer_gtol": optimizer_gtol,
              "gradient_tolerance": gradient_tolerance, "task_seconds": task_seconds,
              "selection": "maximum absolute current gradient; smallest index tie-break",
              "no_fci_for_training_or_stopping": True}
    config_id = digest(config)
    status_path = root / "status.json"
    initial = hf_state(meta["n_qubits"], meta["hf_id"])
    indices, theta, next_step = [], np.array([]), 0
    elapsed_previous = 0.
    predecessor=None
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("source_execution_id") and status.get("runtime",{}).get("backend")=="cupy":
            old=status["config"]
            checks=("method","problem","catalog_digest","max_steps","optimizer_maxiter","optimizer_gtol","gradient_tolerance","task_seconds")
            if status["status"] not in ("CONVERGED","MAX_STEPS","TIME_CENSORED") or any(old[k]!=config[k] for k in checks):
                raise RuntimeError("CUDA trajectory scientific settings mismatch or incomplete import")
            return status
        if status["config_digest"] != config_id:
            old=status.get("config",{})
            same={k:v for k,v in old.items() if k!="max_steps"}=={k:v for k,v in config.items() if k!="max_steps"}
            if not (allow_extension and same and max_steps>old.get("max_steps",max_steps)):
                raise RuntimeError("ADAPT settings changed; only explicit step-cap extension may reuse this trajectory")
            if status["status"] in ("CONVERGED","TIME_CENSORED"):
                return status
            if status["status"]!="MAX_STEPS":
                raise RuntimeError("Finish/reconcile the current run before extending its cap")
            predecessor=status["config_digest"]
            save_json(root/"extensions"/f"{predecessor}.json",status)
        if status["status"] in ("CONVERGED", "TIME_CENSORED") or (status["status"]=="MAX_STEPS" and predecessor is None):
            return status
        checkpoints = sorted(root.glob("step-*.json"))
        if checkpoints:
            last = json.loads(checkpoints[-1].read_text())
            indices, theta = last["selected_operators"], np.array(last["optimized_parameters"])
            next_step = last["step"]+1
            elapsed_previous = last["trajectory_seconds"]
    start = perf_counter()
    save_json(status_path, {"status": "RUNNING", "config": config, "config_digest": config_id,
                            "provenance": provenance(),"predecessor_config_digest":predecessor})
    for step in range(next_step, max_steps+1):
        opt = None
        initial_parameters = theta.tolist()
        if step > 0:
            state = state_from_parameters(pool, indices, theta, initial)
            gradients = selection_gradients(pool, state, h)
            if np.linalg.norm(gradients) <= gradient_tolerance:
                terminal = "CONVERGED"
                break
            selected = int(np.argmax(np.abs(gradients)))
            if method == "ceo_mvp":
                support = pool.get_qubits(selected)
                additions = [i for i in range(pool.size) if pool.get_qubits(i)==support and abs(gradients[i])>1e-10]
            else:
                additions = [selected]
            indices = indices + additions
            theta = np.r_[theta,np.zeros(len(additions))]
            initial_parameters = theta.tolist()
            opt_start = perf_counter()
            opt = minimize(lambda x: energy_and_gradient(pool, indices, x, initial, h), theta,
                           method="BFGS", jac=True,
                           options={"maxiter": optimizer_maxiter, "gtol": optimizer_gtol})
            theta = opt.x
            opt_seconds = perf_counter()-opt_start
        else:
            selected, gradients, opt_seconds = None, selection_gradients(pool, initial, h), 0.
        state = state_from_parameters(pool, indices, theta, initial)
        energy = float(np.vdot(state, h @ state).real)
        qc = build_circuit(pool, indices, theta, meta["hf_id"])
        parity = verify_circuit_state(qc, state)
        sampler,removed = measurement_equivalent_circuit(qc)
        sampling_state = Statevector.from_instruction(sampler).data
        if not np.allclose(np.abs(sampling_state)**2,np.abs(state)**2,atol=1e-9):
            raise RuntimeError("Sampling compiler changed probability")
        metrics = resources(sampler, len(theta), len(indices))
        cp_path = root / f"step-{step:04d}.json"
        dist_path = Path(output) / "distributions" / case_id / method / f"step-{step:04d}.npz"
        probability = np.abs(state)**2
        save_npz(dist_path, state=state, sampling_state=sampling_state,probability=probability)
        qc_path = cp_path.with_suffix(".qpy")
        save_binary(qc_path,circuit_bytes(qc))
        sample_path=cp_path.with_suffix(".sampling.qpy")
        save_binary(sample_path,circuit_bytes(sampler))
        cp = {"schema": "checkpoint-1", "config_digest": config_id,
              "predecessor_config_digest":predecessor,
              "case_id": case_id, "problem_id": meta["request_digest"], "method": method,
              "step": step, "selected_operators": indices, "optimized_parameters": theta.tolist(),
              "initial_parameters": initial_parameters, "selected_this_step": selected,
              "selected_group": additions if step>0 else [],
              "pool_gradients_before_append": gradients.tolist(),
              "pool_gradient_norm_before_append":float(np.linalg.norm(gradients)),
              "pool_gradient_components_computed":int(pool.size),
              "vqe_energy": energy, "vqe_energy_error": energy-meta["fci_energy"],
              "resources": metrics, "qiskit_state_error": parity,
              "preparation_resources":resources(qc,len(theta),len(indices)),
              "sampling_equivalence":"exact computational-basis distribution; not VQE-state identity",
              "removed_terminal_diagonal_gates":removed,
              "sampling_circuit_qpy":str(sample_path.resolve()),
              "sampling_circuit_sha256":file_digest(sample_path),
              "optimization_seconds": opt_seconds,
              "objective_evaluations": int(opt.nfev) if opt else 0,
              "gradient_evaluations": int(opt.njev) if opt else 0,
              "optimizer_iterations": int(opt.nit) if opt else 0,
              "optimizer_success": bool(opt.success) if opt else None,
              "optimizer_message": str(opt.message) if opt else "HF_NOT_OPTIMIZED",
              "final_parameter_gradient_norm": float(np.linalg.norm(opt.jac)) if opt else None,
              "distribution": str(dist_path.resolve()), "distribution_sha256": file_digest(dist_path),
              "circuit_qpy": str(qc_path.resolve()), "circuit_sha256": file_digest(qc_path),
              "trajectory_seconds": elapsed_previous+perf_counter()-start,
              "analytic_metrics": analytic_metrics(probability, reference)}
        cp["checkpoint_id"] = digest({"config": config_id, "operators": indices, "theta": theta.tolist()})
        save_json(cp_path, cp)
        save_json(status_path, {"status": "RUNNING", "config_digest": config_id,
                               "last_checkpoint": step, "elapsed_seconds": cp["trajectory_seconds"]})
        print(f"{case_id} {method} step{step}: E={energy:.10f}, CNOT={metrics['cnot_count']}", flush=True)
        if elapsed_previous+perf_counter()-start > task_seconds:
            terminal = "TIME_CENSORED"
            break
    else:
        terminal = "MAX_STEPS"
    status = {"status": terminal, "config": config, "config_digest": config_id,
              "predecessor_config_digest":predecessor,
              "checkpoints": len(list(root.glob('step-*.json'))),
              "elapsed_seconds": elapsed_previous+perf_counter()-start}
    save_json(status_path, status)
    return status

