"""Exact ground root within a bounded SELECTED space, never a Fock-space solver.

The official SQD/PySCF APIs, selected strings, RDMs, recovery and batching stay
unchanged. Only the SelectedCI.eig kernel is replaced for these <=64-dimensional
projections. This is a versioned numerical successor, not a retry of v1 cells.
"""
from __future__ import annotations
from unittest.mock import patch
import numpy as np
from scipy.linalg import eigh
import pyscf.fci.selected_ci as selected_ci
import qiskit_addon_sqd.fermion as official
from .fermionic_sqd_durable import full_sqd_durable

ORIGINAL_BATCH = official.solve_sci_batch
ORIGINAL_CLASS = selected_ci.SelectedCI


def lowest_projected_root(op, guesses, dimension_cap=64):
    """Materialize only the already-selected CI operator using named-space H*v."""
    if not callable(op) or not guesses: raise ValueError("Selected-space callable and guesses required")
    d = np.asarray(guesses[0]).size
    if not 1 <= d <= dimension_cap: raise RuntimeError("Selected projection exceeds registered dimension cap")
    # No dependence on the values of initial guesses or a full Hilbert basis.
    eye = np.eye(d); columns = []
    for i in range(d):
        # PySCF's selected-CI C contraction expects contiguous amplitudes.
        # A strided column view of np.eye must not reach that kernel.
        column = np.asarray(op(np.ascontiguousarray(eye[:, i]))).reshape(-1).copy()
        if column.size != d or not np.isfinite(column).all(): raise RuntimeError("Corrupt selected-space action")
        columns.append(column)
    matrix = np.column_stack(columns)
    antihermitian = float(np.max(abs(matrix-matrix.T.conj())))
    if antihermitian > 1e-9: raise RuntimeError("Non-Hermitian selected-space action")
    matrix = (matrix+matrix.T.conj())*.5
    values, vectors = eigh(matrix, check_finite=False)
    residual = float(np.linalg.norm(matrix@vectors[:, 0]-values[0]*vectors[:, 0]))
    if residual > 1e-8: raise RuntimeError("Selected-space ground Ritz residual failed")
    return float(values[0]), vectors[:, 0], {"selected_dimension": d, "selected_H_vector_products": d,
        "selected_matrix_bytes": int(matrix.nbytes), "max_antihermitian_error": antihermitian,
        "ground_Ritz_residual": residual, "full_Fock_matrix_or_vector": False,
        "ground_root_in_selected_space_not_global_ground_certificate": True}


def projected_sqd(h1, eri, core, hf, na, nb, ids, counts, spec, trace):
    cap = spec.max_spin_strings**2
    if cap > 64: raise ValueError("This qualified projected solver is registered only through64")
    work = []
    class ProjectedSCI(ORIGINAL_CLASS):
        def eig(self, op, x0=None, precond=None, **kwargs):
            if kwargs.get("nroots", 1) != 1: raise ValueError("Registered single lowest root only")
            d = np.asarray(x0[0]).size
            if not 1 <= d <= cap: raise RuntimeError("Reject oversized selected projection before H*v")
            trace.emit("PROJECTED_GROUND_REQUEST", {"dimension": d, "dimension_cap": cap,
                "planned_selected_H_vector_products": d, "full_Fock_basis": False})
            e, v, record = lowest_projected_root(op, x0, cap)
            trace.emit("PROJECTED_GROUND_RESULT", {**record, "electronic_ground_energy_Ha": e,
                "new_quantum_shots": 0})
            work.append(record); self.converged = True
            return e, v

    def batch(*args, **kwargs):
        # The parent SDK constructs the selected vectors and RDMs in its own
        # fermionic ordering; no hand-written conversion of CI coefficient signs.
        with patch.object(selected_ci, "SelectedCI", ProjectedSCI):
            return ORIGINAL_BATCH(*args, **kwargs)

    with patch.object(official, "solve_sci_batch", batch):
        out = full_sqd_durable(h1, eri, core, hf, na, nb, ids, counts, spec, trace)
    out["selected_solver"] = "official_PySCF_selected_action_bounded_projected_ground_v1"
    out["selected_solver_work"] = {"calls": len(work), "selected_H_vector_products": sum(x["selected_H_vector_products"] for x in work),
        "maximum_selected_dimension": max((x["selected_dimension"] for x in work), default=0),
        "maximum_selected_matrix_bytes": max((x["selected_matrix_bytes"] for x in work), default=0),
        "additional_classical_work_not_free": True, "new_quantum_shots": 0}
    return out
