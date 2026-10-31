"""Spin-conserving integral-driven external connections, never full Fock space.

Per-parent screening/truncation is explicit. A sum over truncated connections is
a partial residual, not an exact norm or a rigorous error bound.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numba import njit
from .integral_hamiltonian import _element, _diagonal


@njit(cache=True)
def _diagonals(ids, n, h1, eri, core):
    out = np.empty(len(ids))
    for j in range(len(ids)):
        out[j] = _diagonal(ids[j], n, h1, eri, core)
    return out


@njit(cache=True)
def _enumerate(ket, n, h1, eri, core, floor):
    occupied = np.empty(n, np.int64); virtual = np.empty(n, np.int64)
    no = nv = 0
    for q in range(n):
        if (ket >> np.uint64(n-1-q)) & np.uint64(1):
            occupied[no] = q; no += 1
        else:
            virtual[nv] = q; nv += 1
    capacity = no*nv+(no*(no-1)//2)*(nv*(nv-1)//2)
    ids = np.empty(capacity, np.uint64); values = np.empty(capacity)
    saved = visited = 0
    for i0 in range(no):
        i = occupied[i0]
        for a0 in range(nv):
            a = virtual[a0]
            if i % 2 != a % 2:
                continue
            bra = ket ^ (np.uint64(1) << np.uint64(n-1-i)) ^ (np.uint64(1) << np.uint64(n-1-a))
            value = _element(bra, ket, n, h1, eri, core); visited += 1
            if abs(value) > floor:
                ids[saved] = bra; values[saved] = value; saved += 1
    for i0 in range(no):
        i = occupied[i0]
        for j0 in range(i0+1, no):
            j = occupied[j0]
            for a0 in range(nv):
                a = virtual[a0]
                for b0 in range(a0+1, nv):
                    b = virtual[b0]
                    if (i % 2)+(j % 2) != (a % 2)+(b % 2):
                        continue
                    bra = ket ^ (np.uint64(1) << np.uint64(n-1-i)) ^ (np.uint64(1) << np.uint64(n-1-j)) ^ (np.uint64(1) << np.uint64(n-1-a)) ^ (np.uint64(1) << np.uint64(n-1-b))
                    value = _element(bra, ket, n, h1, eri, core); visited += 1
                    if abs(value) > floor:
                        ids[saved] = bra; values[saved] = value; saved += 1
    return ids[:saved], values[:saved], visited


@dataclass
class ConnectionStatistics:
    named_matrix_entries_examined: int = 0
    nonzero_connections_found: int = 0
    connections_returned: int = 0
    parent_truncations: int = 0


class IntegralConnections:
    def __init__(self, backend, coupling_floor=1e-12, parent_cap=512):
        if coupling_floor < 0 or (parent_cap is not None and parent_cap < 1):
            raise ValueError("Invalid screening/cap")
        self.backend = backend; self.floor = float(coupling_floor); self.parent_cap = parent_cap
        self.statistics = ConnectionStatistics()

    def connections(self, ket):
        if int(ket) < 0 or int(ket) >= 1 << self.backend.n_qubits:
            raise ValueError("Named configuration outside register")
        ids, values, visited = _enumerate(np.uint64(ket), self.backend.n_qubits, self.backend.h1,
                                          self.backend.eri, self.backend.core, self.floor)
        self.statistics.named_matrix_entries_examined += visited
        self.statistics.nonzero_connections_found += len(ids)
        # Charge every examined element, including screened-to-zero results.
        self.backend.matrix_entry_requests += visited
        order = np.lexsort((ids, -abs(values)))
        if self.parent_cap is not None and len(order) > self.parent_cap:
            order = order[:self.parent_cap]; self.statistics.parent_truncations += 1
        self.statistics.connections_returned += len(order)
        return ids[order].astype(np.int64), values[order]

    def external(self, selected, coefficients, energy, denominator_floor=1e-8):
        if denominator_floor <= 0:
            raise ValueError("Positive denominator floor required")
        selected = list(map(int, selected)); c = np.asarray(coefficients)
        if len(selected) != len(c) or len(set(selected)) != len(selected):
            raise ValueError("Invalid named CI support")
        internal = set(selected); residual = {}; hci = {}
        truncated_before = self.statistics.parent_truncations
        for y, coefficient in zip(selected, c):
            ids, values = self.connections(y)
            for x, value in zip(ids, values):
                x = int(x)
                if x in internal:
                    continue
                term = value*coefficient
                residual[x] = residual.get(x, 0j)+term
                hci[x] = max(hci.get(x, 0.), abs(term))
        ids = np.array(sorted(residual), np.int64)
        r = np.array([residual[int(x)] for x in ids])
        diagonal = _diagonals(ids.astype(np.uint64), self.backend.n_qubits, self.backend.h1, self.backend.eri, self.backend.core)
        self.backend.matrix_entry_requests += len(ids)
        denominator = energy-diagonal
        regularized = np.where(abs(denominator) < denominator_floor,
                               np.where(denominator > 0, denominator_floor, -denominator_floor), denominator)
        truncated = self.statistics.parent_truncations != truncated_before
        return {"ids": ids, "partial_residual": r, "hci_importance": np.array([hci[int(x)] for x in ids]),
                "cipsi_coefficient_importance": abs(r/regularized), "en_energy_importance": abs(r)**2/abs(regularized),
                "external_residual_norm_of_returned_terms": float(np.linalg.norm(r)),
                "parent_truncated": truncated, "coupling_floor": self.floor,
                "residual_scope": "partial_due_to_parent_cap_or_coupling_screen" if truncated or self.floor > 0 else "all_spin_conserving_connections",
                "near_zero_denominators": int((abs(denominator) < denominator_floor).sum()),
                "intruders": int((denominator >= 0).sum())}
