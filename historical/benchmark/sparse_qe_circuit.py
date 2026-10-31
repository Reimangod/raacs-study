"""Exact bounded-support physical QE simulation, not a free Ritz-state oracle.

Only configurations reachable by the ordered *physical circuit* are stored.
No2**n vector, full Fock Hamiltonian, FCI labels or global operator pool.
Support can grow exponentially with circuit length: explicit cap, no generic
large/deep simulation claim. Counts-only users never receive amplitudes.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from qiskit.quantum_info import Operator
from scipy.sparse import csr_matrix
from .local_sampler_primitives import LocalQE, _zero_phase
from adaptvqe.circuits import qe_circuit


@lru_cache(maxsize=128)
def local_rotation(src, dst, n):
    # These are compressed support coordinates, NOT physical spin-orbital
    # labels. Physical Nalpha/Nbeta was checked on the uncompressed LocalQE.
    # Applying spin-parity validation to compressed0/1 labels would reject
    # every same-spin physical single excitation incorrectly.
    def matrix(theta):
        return Operator(qe_circuit(list(src), list(dst), theta, n, big_endian=True)).data/_zero_phase(src, dst, n)
    plus = matrix(np.pi/2)
    minus = matrix(-np.pi/2)
    G = (plus-minus)/2
    Q = (plus+minus)/2-np.eye(1 << n)
    if not np.allclose(G.conj().T, -G, atol=1e-12) or not np.allclose(Q, G@G, atol=1e-12):
        raise RuntimeError("Physical QE normalization does not match the qualified trigonometric kernel")
    for theta in (.123, -.431):
        actual = matrix(theta)
        predicted = np.eye(1 << n)+np.sin(theta)*G+(1-np.cos(theta))*Q
        if not np.allclose(actual, predicted, atol=1e-12): raise RuntimeError("Local QE rotation parity failed")
    return G, Q


def action_on_id(operator, local_matrix, value):
    support = tuple(sorted(operator.source+operator.target)); k = len(support); n = operator.n_qubits
    column = sum(((int(value) >> (n-1-q)) & 1) << (k-1-i) for i, q in enumerate(support))
    mask = sum(1 << (n-1-q) for q in support); result = {}
    for row in np.flatnonzero(abs(local_matrix[:, column]) > 1e-13):
        target = int(value) & ~mask
        for i, q in enumerate(support): target |= ((int(row) >> (k-1-i)) & 1) << (n-1-q)
        result[target] = complex(local_matrix[row, column])
    return result


class SparseQECircuit:
    def __init__(self, n_qubits, hf_id, operators, support_cap=512, backend=None):
        if not 1 <= n_qubits <= 62 or not 0 <= hf_id < 1 << n_qubits: raise ValueError("Bounded physical register")
        if not 1 <= support_cap <= 4096: raise ValueError("Explicit physical support cap")
        if any(o.n_qubits != n_qubits for o in operators): raise ValueError("Same physical register")
        self.n_qubits = n_qubits; self.hf_id = int(hf_id); self.operators = tuple(operators)
        support = {int(hf_id)}; kernels = []
        for op in self.operators:
            local = tuple(sorted(op.source+op.target))
            src = tuple(local.index(q) for q in op.source); dst = tuple(local.index(q) for q in op.target)
            G, Q = local_rotation(src, dst, len(local)); kernels.append((G, Q))
            previous = list(support)
            for value in previous:
                support.update(action_on_id(op, G, value)); support.update(action_on_id(op, Q, value))
            if len(support) > support_cap: raise RuntimeError("Physical sparse support cap; no full-vector fallback")
        self.ids = np.array(sorted(support), dtype=np.uint64); positions = {int(x): i for i, x in enumerate(self.ids)}
        self.initial = np.zeros(len(self.ids), dtype=complex); self.initial[positions[int(hf_id)]] = 1.
        self.kernels = []
        for op, (G, Q) in zip(self.operators, kernels):
            projected = []
            for matrix in (G, Q):
                rows = []; cols = []; values = []
                for j, x in enumerate(self.ids):
                    for y, amplitude in action_on_id(op, matrix, int(x)).items():
                        if y in positions: rows.append(positions[y]); cols.append(j); values.append(amplitude)
                projected.append(csr_matrix((values, (rows, cols)), shape=(len(self.ids), len(self.ids)), dtype=complex))
            self.kernels.append(tuple(projected))
        self.hamiltonian = None if backend is None else backend.matrix(self.ids, self.ids)
        if self.hamiltonian is not None and not np.allclose(self.hamiltonian, self.hamiltonian.T.conj(), atol=1e-10):
            raise RuntimeError("Non-Hermitian named physical-support Hamiltonian")

    def apply(self, index, theta, state, adjoint=False, derivative=False):
        G, Q = self.kernels[index]; sign = -1 if adjoint else 1
        if derivative: return np.cos(theta)*(G@state)+np.sin(theta)*(Q@state)
        return state+sign*np.sin(theta)*(G@state)+(1-np.cos(theta))*(Q@state)

    def state(self, theta, keep=False):
        theta = np.asarray(theta, float)
        if theta.shape != (len(self.operators),) or not np.isfinite(theta).all(): raise ValueError("Coefficient binding mismatch")
        state = self.initial.copy(); history = [state]
        for i, angle in enumerate(theta):
            state = self.apply(i, angle, state); history.append(state)
            if abs(np.vdot(state, state).real-1) > 1e-9: raise RuntimeError("Truncated or nonunitary physical support evolution")
        return history if keep else state

    def energy_gradient(self, theta):
        if self.hamiltonian is None: raise ValueError("Physical integrals required for VQE diagnostic")
        history = self.state(theta, keep=True); adjoint = self.hamiltonian@history[-1]
        energy = float(np.vdot(history[-1], adjoint).real); gradient = np.zeros(len(theta))
        for j in range(len(theta)-1, -1, -1):
            gradient[j] = 2*np.vdot(adjoint, self.apply(j, theta[j], history[j], derivative=True)).real
            adjoint = self.apply(j, theta[j], adjoint, adjoint=True)
        return energy, gradient

    def counts(self, theta, shots, seed):
        if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1: raise ValueError("Positive integer shots")
        p = abs(self.state(theta))**2; p /= p.sum()
        counts = np.random.default_rng(seed).multinomial(shots, p); selected = counts > 0
        return self.ids[selected].copy(), counts[selected].astype(np.uint64)
