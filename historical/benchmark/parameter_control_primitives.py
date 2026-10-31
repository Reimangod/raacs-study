"""Actual angle controls; classical VQE objectives are not QPU-shot estimates."""
from __future__ import annotations
from time import perf_counter
import numpy as np
from scipy.optimize import minimize


class ObjectiveBoundary(Exception): pass


def vqe_update(simulator, initial, mode, max_calls=300, max_iterations=80, seconds=30., gtol=1e-6):
    initial = np.asarray(initial, float); k = len(initial)
    if mode not in ("one_parameter_vqe", "blockwise_vqe", "full_vqe"): raise ValueError("Actual VQE control mode")
    indices = np.arange(k) if mode == "full_vqe" else np.arange(max(0, k-(2 if mode == "blockwise_vqe" else 1)), k)
    started = perf_counter(); trace = []; best = [float("inf"), initial.copy(), None]
    def objective(variables):
        if len(trace) >= max_calls or perf_counter()-started > seconds: raise ObjectiveBoundary()
        theta = initial.copy(); theta[indices] = variables
        E, gradient = simulator.energy_gradient(theta)
        row = {"call": len(trace)+1, "energy_Ha": E, "theta": theta.tolist(),
               "active_gradient_inf": float(max(abs(gradient[indices]))) if len(indices) else 0.,
               "full_gradient_inf": float(max(abs(gradient))) if k else 0.}
        trace.append(row)
        if E < best[0]: best[:] = [E, theta.copy(), gradient.copy()]
        return E, gradient[indices]
    status = "NO_OPTIMIZED_VARIABLES"; message = "Empty registered ansatz"; iterations = 0; success = None
    if len(indices):
        try:
            result = minimize(objective, initial[indices], method="BFGS", jac=True,
                              options={"gtol": gtol, "maxiter": max_iterations})
            status = "OPTIMIZER_TERMINATED"; message = str(result.message); iterations = int(result.nit); success = bool(result.success)
            if int(result.nfev) != len(trace): raise RuntimeError("Actual energy/gradient objective accounting mismatch")
        except ObjectiveBoundary:
            status = "OBJECTIVE_OR_TIME_CAP"; message = "Boundary reached; best actually evaluated point retained"
    else: objective(np.array([]))
    if best[2] is None: raise RuntimeError("No physical objective evaluated")
    grad = best[2]; active_inf = float(max(abs(grad[indices]))) if len(indices) else 0.
    full_inf = float(max(abs(grad))) if k else 0.
    return {"theta": best[1].tolist(), "energy_Ha": float(best[0]), "trace": trace,
            "actual_objective_evaluations": len(trace), "actual_complete_gradient_evaluations": len(trace),
            "optimizer_initial_state": "RESET_IDENTITY_BFGS", "initial_theta": initial.tolist(),
            "optimized_variable_indices": indices.tolist(), "n_optimized_variables": len(indices),
            "n_independent_ansatz_parameters": k, "status": status, "optimizer_message": message,
            "optimizer_success": success, "optimizer_iterations": iterations,
            "active_gradient_inf_at_returned_point": active_inf, "full_gradient_inf_at_returned_point": full_inf,
            "active_stationary": active_inf <= gtol, "full_stationary": full_inf <= gtol,
            "selection": "best actual physical objective evaluation", "wall_seconds": perf_counter()-started,
            "objective_scope": "exact bounded sparse physical-circuit simulation; not a QPU measurement model",
            "QPU_objective_measurement_shots": None, "full_Fock_statevector_allocated": False, "FCI_used": False}
