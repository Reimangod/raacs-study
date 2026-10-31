"""Independent implementation of paper-defined ADAPT-QSCI and HF-separated math.

Selection works with named sparse CI configurations, not FCI coefficients or the
quantum input state's amplitudes. The small-system sampler backend is separate.
Reference: arXiv:2311.01105v2 §3/S1 and2610.03114v1 §IV/AppendixA.
No author-repository code is copied (inspected example had no license).
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from .ansatz import compile_logical


@dataclass(frozen=True)
class Pauli:
    label: str

    @property
    def flip_mask(self):
        return sum(1 << (len(self.label)-1-q) for q, p in enumerate(self.label) if p in "XY")

    def act(self, ids, amplitudes):
        ids = np.asarray(ids, np.int64)
        mask = sum(1 << (len(self.label)-1-q) for q, p in enumerate(self.label) if p in "YZ")
        ny = self.label.count("Y")
        phase = (1j**ny)*np.where((np.bitwise_count(ids & mask) & 1) == 0, 1, -1)
        return ids ^ self.flip_mask, phase*np.asarray(amplitudes)


def paper_pool(n):
    """Exactly S1: same-spin XY, odd-support-parity-filtered four-body one-Y."""
    labels = []
    for p, q in combinations(range(n), 2):
        if p % 2 == q % 2:
            s = ["I"]*n; s[p], s[q] = "X", "Y"; labels.append(Pauli("".join(s)))
    for support in combinations(range(n), 4):
        if sum(support) % 2 == 0:
            for y in reversed(support):  # XXXY,XXYX,XYXX,YXXX in paper order.
                s = ["I"]*n
                for q in support:
                    s[q] = "Y" if q == y else "X"
                labels.append(Pauli("".join(s)))
    assert len({p.label for p in labels}) == len(labels)
    return labels


class NamedHamiltonian:
    """Named-entry access to saved small-system sparse H, with no oracle labels."""
    def __init__(self, h):
        self.h = h.tocsr()
        self.matrix_entry_requests = 0
        self.sparse_actions = 0

    def matrix(self, left, right):
        self.matrix_entry_requests += len(left)*len(right)
        return self.h[np.asarray(left)][:, np.asarray(right)].toarray()

    def action(self, ids, c):
        self.sparse_actions += 1
        v = sparse.csc_matrix((c, (ids, np.zeros(len(ids), int))), shape=(self.h.shape[0], 1))
        out = (self.h @ v).tocoo()
        return dict(zip(map(int, out.row), out.data))

    def expectation(self, ids, c):
        return float(np.vdot(c, self.matrix(ids, ids) @ c).real)


def gradient_data(backend, ids, c, pool, hf=None):
    """Project the full commutator, never [H_S,P_S] on its eigenvector."""
    hc = backend.action(ids, c)
    hf_row = backend.h.getrow(hf).tocoo() if hf is not None else None
    hf_elements = dict(zip(map(int, hf_row.col), hf_row.data)) if hf_row is not None else None
    result = []
    for p in pool:
        flipped, pc = p.act(ids, c)
        hp_overlap = sum(np.conjugate(hc.get(int(x), 0.))*v for x, v in zip(flipped, pc))
        row = {"C": float(-2*np.imag(hp_overlap))}
        if hf is not None:
            row["rS"] = float(np.real(-1j*sum(v for x, v in zip(flipped, pc) if x == hf)))
            backend.matrix_entry_requests += len(flipped)
            row["rH"] = float(np.real(-1j*sum(hf_elements.get(int(x), 0.)*v for x, v in zip(flipped, pc))))
        result.append(row)
    return result


def standard_angle(A, B, C):
    theta = .5*np.arctan2(-C, B-A)
    value = A*np.cos(theta)**2+B*np.sin(theta)**2+C*np.sin(theta)*np.cos(theta)
    return float(theta), float(value)


def hf_matrices(theta, EHF, A, B, C, v, rH, rS):
    co, si = np.cos(theta), np.sin(theta)
    h12 = v*co-rH*si
    h22 = A*co*co+B*si*si+C*co*si
    s12 = -rS*si
    return np.array([[EHF, h12], [h12, h22]]), np.array([[1., s12], [s12, 1.]])


def hf_root(theta, EHF, A, B, C, v, rH, rS, overlap_threshold=1e-10):
    # Shift before solving to reduce large total-energy cancellation.
    co, si = np.cos(theta), np.sin(theta)
    S = np.array([[1., -rS*si], [-rS*si, 1.]])
    K = np.array([[0., v*co-(rH-EHF*rS)*si],
                  [v*co-(rH-EHF*rS)*si, (A-EHF)*co*co+(B-EHF)*si*si+C*co*si]])
    sw, su = eigh(S)
    keep = sw > overlap_threshold
    if np.any(sw < -overlap_threshold):
        raise RuntimeError("Nonpositive physical overlap")
    W = su[:, keep]/np.sqrt(sw[keep])
    return float(EHF+eigh(W.T @ K @ W, eigvals_only=True)[0])


def hf_gradient(EHF, A, C, v, rH, rS):
    # Generalized Hellmann-Feynman includes S'(0), not just H'(0).
    k = np.array([[0., v], [v, A-EHF]])
    ew, ev = eigh(k)
    if ew[1]-ew[0] < 1e-12:
        raise RuntimeError("DEGENERATE_HF_OBJECTIVE_GRADIENT_UNDEFINED")
    E = EHF+ew[0]; u = ev[:, 0]
    Hprime = np.array([[0., -rH], [-rH, C]])
    Sprime = np.array([[0., -rS], [-rS, 0.]])
    return float(u @ (Hprime-E*Sprime) @ u)


def hf_angle(**coefficients):
    """Periodic1D search of the exact generalized-root objective, no device query."""
    xs = np.linspace(-np.pi/2, np.pi/2, 129)
    calls = 0
    def f(theta):
        nonlocal calls
        calls += 1
        return hf_root(theta, **coefficients)
    ys = [f(x) for x in xs]
    trials = [(ys[i], xs[i]) for i in (0, len(xs)-1)]
    trials.append((f(0.), 0.))
    for i in range(1, len(xs)-1):
        if ys[i] <= ys[i-1] and ys[i] <= ys[i+1]:
            opt = minimize_scalar(f, bounds=(xs[i-1], xs[i+1]), method="bounded", options={"xatol": 1e-11})
            trials.append((opt.fun, opt.x))
    value, theta = min(trials, key=lambda x: (x[0], abs(x[1]), x[1]))
    return float(theta), float(value), calls


def update_state(state, p, theta):
    ids = np.arange(len(state)); flipped, rotated = p.act(ids, state)
    output = state*np.cos(theta)
    output[flipped] += 1j*np.sin(theta)*rotated
    if abs(np.linalg.norm(output)-1) > 1e-9:
        raise RuntimeError("Unitary update failed")
    return output


def circuit(n, hf, rotations):
    qc = QuantumCircuit(n)
    for wire in range(n):
        if hf >> wire & 1:
            qc.x(wire)
    for label, theta in rotations:
        # Qiskit uses exp(-i*time*P); paper rotation is exp(+i*theta*P).
        qc.append(PauliEvolutionGate(SparsePauliOp(label), time=-float(theta)), range(n))
    return compile_logical(qc)


def perturbative_target(backend, valid_ids, hf, orbital_energies, n):
    """Explicit Moller-Plesset partition; paper does not specify its H0 in detail."""
    connected = backend.action([hf], np.array([1.]))
    occ_hf = sum(orbital_energies[q//2] for q in range(n) if hf >> (n-1-q) & 1)
    rows = []
    for x in valid_ids:
        x = int(x)
        if x == hf or abs(connected.get(x, 0)) < 1e-12:
            continue
        occ_x = sum(orbital_energies[q//2] for q in range(n) if x >> (n-1-q) & 1)
        denom = occ_hf-occ_x
        if abs(denom) < 1e-10:
            raise RuntimeError("DEGENERATE_FIRST_ORDER_INITIALIZER")
        rows.append((x, connected[x]/denom))
    chosen = sorted(rows, key=lambda v: (-abs(v[1]), v[0]))[:2]
    if not chosen:
        raise RuntimeError("NO_NON_HF_FIRST_ORDER_INITIALIZER")
    ids = np.array([x for x, _ in chosen]); c = np.array([v for _, v in chosen], complex)
    return ids, c/np.linalg.norm(c)


def prepare_target(n, hf, target_ids, target_coefficients, pool, max_gates=8):
    """Actual greedy pool-rotation preparation, including physical gate debt."""
    target = np.zeros(1 << n, complex); target[target_ids] = target_coefficients
    state = np.zeros_like(target); state[hf] = 1
    rotations = []
    for _ in range(max_gates):
        a = np.vdot(target, state)
        occupied = np.flatnonzero(state != 0)
        best = None
        for index, p in enumerate(pool):
            ids, ps = p.act(occupied, state[occupied])
            b = 1j*np.vdot(target[ids], ps)
            mat = np.array([[abs(a)**2, (a.conjugate()*b).real], [(a.conjugate()*b).real, abs(b)**2]])
            w, v = eigh(mat)
            theta = np.arctan2(v[1, -1], v[0, -1])
            trial = (float(w[-1]), -index, float(theta))
            if best is None or trial[:2] > best[:2]:
                best = trial
        fidelity, neg_index, theta = best
        if fidelity <= abs(a)**2+1e-13:
            break
        p = pool[-neg_index]
        state = update_state(state, p, theta); rotations.append((p.label, theta))
        if abs(np.vdot(target, state))**2 > 1-1e-10:
            return state, rotations, float(abs(np.vdot(target, state))**2)
    raise RuntimeError("INITIAL_PREPARATION_FIDELITY_NOT_REACHED; no free ideal-state substitution")
