"""Named-space residual/PT2 and explicit HCI/CIPSI-ranked small-system controls.

These cardinality-bounded growing-space variants use the published importance
criteria, not an optimized SHCI implementation or semistochastic PT2 engine.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


def external_diagnostics(backend, selected, coefficients, energy, denominator_floor=1e-8):
    selected = np.asarray(selected, np.int64)
    c = np.asarray(coefficients)
    block = backend.h[:, selected].tocoo()
    external = ~np.isin(block.row, selected)
    row, col, values = block.row[external], block.col[external], block.data[external]
    if not len(row):
        return {"ids": np.array([], np.int64), "residual": np.array([], complex),
                "hci_importance": np.array([]), "cipsi_coefficient_importance": np.array([]),
                "en_energy_importance": np.array([]), "external_residual_norm": 0.,
                "pt2_raw": 0., "pt2_regularized": 0., "intruders": 0, "near_zero_denominators": 0,
                "connected_nonzero_terms_examined": int(block.nnz), "denominator_floor": denominator_floor}
    order = np.argsort(row, kind="stable")
    row, col, values = row[order], col[order], values[order]
    ids, first = np.unique(row, return_index=True)
    terms = values*c[col]
    residual = np.add.reduceat(terms, first)
    hci = np.maximum.reduceat(abs(terms), first)
    diagonal = np.asarray(backend.h[ids, ids]).reshape(-1).real
    backend.matrix_entry_requests += len(ids)
    denominator = energy-diagonal
    near = abs(denominator) < denominator_floor
    regularized = np.where(near, np.where(denominator > 0, denominator_floor, -denominator_floor), denominator)
    raw = None if np.any(near) else float(np.sum(abs(residual)**2/denominator))
    return {"ids": ids, "residual": residual, "hci_importance": hci,
            "cipsi_coefficient_importance": abs(residual/regularized),
            "en_energy_importance": abs(residual)**2/abs(regularized),
            "external_residual_norm": float(np.linalg.norm(residual)),
            "pt2_raw": raw, "pt2_regularized": float(np.sum(abs(residual)**2/regularized)),
            "intruders": int((denominator >= 0).sum()), "near_zero_denominators": int(near.sum()),
            "connected_nonzero_terms_examined": int(block.nnz), "denominator_floor": denominator_floor}


def solve(backend, selected):
    matrix = backend.h[np.asarray(selected)][:, np.asarray(selected)].tocsr()
    backend.matrix_entry_requests += len(selected)**2
    if len(selected) < 128:
        w, v = eigh(matrix.toarray(), subset_by_index=(0, 0))
    else:
        w, v = eigsh(matrix, k=1, which="SA", tol=1e-12, v0=np.random.default_rng(163).normal(size=len(selected)), maxiter=5000)
    energy, c = float(w[0]), v[:, 0]
    residual = float(np.linalg.norm(matrix @ c-energy*c))
    if residual > 1e-8:
        raise RuntimeError("SCI Ritz residual failed")
    return energy, c, residual


def grow_space(backend, hf, dimension_cap, criterion, max_iterations=64, denominator_floor=1e-8,
               max_seconds=None, rss_limit=None):
    from time import perf_counter
    from .study.core import rss_peak
    started = perf_counter()
    if criterion not in ("hci_importance", "cipsi_coefficient_importance", "en_energy_importance"):
        raise ValueError("Unknown criterion")
    S = [int(hf)]; history = []
    for iteration in range(max_iterations):
        if max_seconds is not None and perf_counter()-started > max_seconds:
            return history, "WALL_SLICE_CAP_AT_ITERATION_BOUNDARY"
        if rss_limit is not None and rss_peak() > rss_limit:
            return history, "RSS_CAP_AT_ITERATION_BOUNDARY"
        energy, c, residual = solve(backend, S)
        external = external_diagnostics(backend, S, c, energy, denominator_floor)
        summary = {k: v for k, v in external.items() if not isinstance(v, np.ndarray)}
        row = {"iteration": iteration, "energy": energy, "selected": S.copy(), "coefficients": c,
               "dimension": len(S), "internal_residual": residual, **summary}
        history.append(row)
        if len(S) >= dimension_cap:
            return history, "DIMENSION_CAP"
        candidates = [(int(x), float(value)) for x, value in zip(external["ids"], external[criterion]) if value > 1e-16]
        if not candidates:
            return history, "NO_EXTERNAL_IMPORTANCE_ABOVE_REGISTERED_NUMERICAL_FLOOR"
        candidates.sort(key=lambda v: (-v[1], v[0]))
        # Registered cardinality growth, not fixed-epsilon full HCI protocol.
        count = min(len(S), dimension_cap-len(S), len(candidates))
        S = sorted(S+[x for x, _ in candidates[:count]])
    return history, "ITERATION_CAP"
