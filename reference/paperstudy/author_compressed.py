"""Run the unmodified pinned C1 initializer in a dedicated subprocess.

Instrumentation records classical work and enforces a boundary; it does not
change the author's optimization options, objective, gradients or stage order.
"""
import argparse
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import resource
from time import perf_counter, process_time
from unittest.mock import patch

import ffsim
import jax
import numpy as np
import scipy.optimize

from .input_adapter import array_bytes
from .io import ROOT, atomic_bytes, save_json, sha256
from .lucj import amplitudes_from_saved, interaction_pairs


def load_author():
    filename = ROOT / "vendor/C1/src/lucj/operator_task/lucj_compressed_t2_task_ffsim/compressed_t2.py"
    expected = json.loads((ROOT / "vendor/C1/source.json").read_text())["file_sha256"]
    if sha256(filename.read_bytes()) != expected:
        raise RuntimeError("Unmodified author source digest mismatch")
    spec = importlib.util.spec_from_file_location("pinned_C1_compressed_t2", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(case_id, seconds=1200., fresh=False):
    output = ROOT / "qualification" / (case_id+"-author-compressed"+("-fresh" if fresh else ""))
    if (output/"result.json").exists():
        return json.loads((output/"result.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    if (output/"operator.npz").exists() and (output/"progress.json").exists():
        # Earlier scientific work returned and was saved before a metadata-only
        # version lookup failed. Preserve that work; do not optimize again.
        progress = json.loads((output/"progress.json").read_text())
        stages = progress["stages"]
        if not stages or any(x["status"] != "RETURNED" for x in stages):
            raise RuntimeError("Incomplete initializer prefix requires explicit checkpoint reconciliation")
        report = {"status": "AUTHOR_INITIALIZER_RETURNED_METADATA_RECOVERED", "case_id": case_id,
                  "phase": "S1_SOURCE_NATIVE_INITIALIZATION_NOT_PRIMARY",
                  "operator_sha256": sha256((output/"operator.npz").read_bytes()),
                  "first_loss": None, "first_loss_reason": "Not persisted before metadata exception",
                  "last_optimizer_objective": stages[-1]["final_objective"],
                  "post_return_objective_separately_persisted": False,
                  "all_stages_optimizer_success": all(x["success"] for x in stages),
                  "stages": stages, "total_wall_seconds_lower_bound": progress["elapsed_seconds"],
                  "wall_bound_reason": "Final serialization time was not persisted before exception",
                  "ffsim": version("ffsim"), "jax": jax.__version__, "JAX_x64": True,
                  "source_digest": json.loads((ROOT/"vendor/C1/source.json").read_text())["file_sha256"],
                  "model_quantum_shots": 0, "actual_QPU_shots": 0, "candidate_energy_evaluated": False,
                  "whole_study_complete": False, "repeated_optimization_for_metadata_repair": False}
        save_json(output/"result.json", report)
        return report
    meta = json.loads((ROOT / "data/physical" / case_id / "problem.json").read_text())
    initialization_root = ROOT / "data" / ("initialization-fresh" if fresh else "initialization") / case_id
    initialization_evidence = None
    if fresh:
        initialization_evidence = json.loads((initialization_root/"result.json").read_text())
        if initialization_evidence["status"] != "FRESH_INITIALIZATION_COMPLETED" or sha256((initialization_root/"amplitudes.npz").read_bytes()) != initialization_evidence["amplitudes_sha256"]:
            raise RuntimeError("Fresh amplitude convergence/digest binding failure")
    t1, t2, kind = amplitudes_from_saved(initialization_root / "amplitudes.npz", meta["norb"], meta["nalpha"])
    if fresh:
        kind = "FRESH_CONVERGED_"+initialization_evidence["selected_method"]
    # This is an explicit double-precision runtime decision, not a claim about
    # an undocumented author environment. Store it in every result.
    jax.config.update("jax_enable_x64", True)
    module = load_author()
    start = perf_counter()
    cpu_start = process_time()
    stages = []
    original = scipy.optimize.minimize
    save_json(output/"started.json", {"case_id": case_id, "phase": "S1_SOURCE_NATIVE_INITIALIZATION",
                                       "wall_limit_seconds": seconds, "JAX_x64": True,
                                       "primary_scientific_run": False})

    def measured_minimize(fun, x0, *args, **kwargs):
        stage = {"stage": len(stages)+1, "n_variables": len(x0), "objective_calls": 0,
                 "options": kwargs.get("options"), "status": "RUNNING"}
        stages.append(stage)
        tick = perf_counter()

        def counted(x, *a):
            if perf_counter()-start > seconds:
                raise TimeoutError("Author initializer objective-boundary wall cap")
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 2*2**30:
                raise MemoryError("Author initializer2GiB local RSS cap")
            stage["objective_calls"] += 1
            value, gradient = fun(x, *a)
            if stage["objective_calls"] == 1:
                stage["initial_objective"] = float(value)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise RuntimeError("Nonfinite author objective/gradient; no substitute optimizer")
            return value, gradient

        save_json(output/"progress.json", {"stages": stages, "elapsed_seconds": perf_counter()-start})
        result = original(counted, x0, *args, **kwargs)
        stage.update(status="RETURNED", success=bool(result.success), message=str(result.message),
                     iterations=int(result.nit), nfev=int(result.nfev), final_objective=float(result.fun),
                     wall_seconds=perf_counter()-tick)
        save_json(output/"progress.json", {"stages": stages, "elapsed_seconds": perf_counter()-start})
        print("STAGE", stage["stage"], stage["success"], stage["final_objective"], flush=True)
        return result

    try:
        with patch.object(module.scipy.optimize, "minimize", measured_minimize):
            op, first_loss, last_loss = module.from_t_amplitudes_compressed(
                t2, t1=t1, n_reps=1, interaction_pairs=interaction_pairs(meta["norb"]), optimize=True,
                multi_stage_optimization=True, begin_reps=20, step=2, regularization=False)
        payload = array_bytes(dict(diag_coulomb_mats=op.diag_coulomb_mats, orbital_rotations=op.orbital_rotations,
                                   final_orbital_rotation=op.final_orbital_rotation))
        atomic_bytes(output/"operator.npz", payload)
        report = {"status": "AUTHOR_INITIALIZER_RETURNED", "first_loss": float(first_loss), "last_loss": float(last_loss),
                  "operator_sha256": sha256(payload), "all_stages_optimizer_success": all(x["success"] for x in stages)}
    except Exception as exc:
        report = {"status": "AUTHOR_INITIALIZER_FAILED_PRESERVED", "error": repr(exc)}
    report.update(case_id=case_id, problem_id=meta["problem_id"], phase="S1_SOURCE_NATIVE_INITIALIZATION_NOT_PRIMARY",
                  initialization=kind,
                  original_amplitude_creation_CPU_seconds=initialization_evidence["CPU_process_seconds"] if fresh else None,
                  saved_amplitude_creation_cost_unknown=not fresh, JAX_x64=True,
                  source_digest=json.loads((ROOT/"vendor/C1/source.json").read_text())["file_sha256"],
                  ffsim=version("ffsim"), jax=jax.__version__, stages=stages,
                  total_wall_seconds=perf_counter()-start, CPU_process_seconds=process_time()-cpu_start,
                  fresh_initialization_evidence=initialization_evidence,
                  actual_QPU_shots=0,
                  model_quantum_shots=0, candidate_energy_evaluated=False, whole_study_complete=False)
    save_json(output/"result.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="N2-32-R1p2-P2")
    p.add_argument("--seconds", type=float, default=1200.)
    p.add_argument("--fresh", action="store_true")
    a = p.parse_args()
    print(json.dumps(run(a.case, a.seconds, a.fresh), indent=2))
