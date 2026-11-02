"""Bounded local QEB proposals and observed-only integral CI for RAACS.

CI coefficients are classical proposal information, not a freely prepared
quantum state. No full operator pool, Fock vector, Fock matrix or FCI label is
required. 'Local' means support arity<=4, not hardware-nearest-neighbor.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from functools import lru_cache
from io import BytesIO
import zlib
import numpy as np
from scipy.linalg import eigh
from qiskit.quantum_info import Operator
from .ansatz import compile_logical
from adaptvqe.circuits import qe_circuit
from .common import digest


def pack_counts64(ids, counts):
    """Portable integer codec for up to62qubits, unlike PhaseA's int32 codec."""
    ids = list(ids); counts = list(counts)
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) for v in ids+counts):
        raise ValueError("Do not truncate floating IDs/counts or reinterpret boolean unknowns")
    ids = list(map(int, ids)); counts = list(map(int, counts))
    if len(ids) != len(counts) or len(set(ids)) != len(ids) or any(x < 0 or x >= 1 << 62 for x in ids) or any(n <= 0 for n in counts):
        raise ValueError("Unique uint62 configurations and positive integer counts required")
    buffer = BytesIO(); np.savez(buffer, ids=np.asarray(ids, dtype="<u8"), counts=np.asarray(counts, dtype="<u8"))
    return zlib.compress(buffer.getvalue(), level=3)


def unpack_counts64(raw):
    with np.load(BytesIO(zlib.decompress(raw)), allow_pickle=False) as z:
        return z["ids"].astype(np.uint64), z["counts"].astype(np.uint64)


def valid_sector(x, n, na, nb):
    if not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_)): return False
    if not 0 <= int(x) < 1 << n: return False
    alpha = sum((int(x) >> (n-1-q)) & 1 for q in range(0, n, 2))
    beta = sum((int(x) >> (n-1-q)) & 1 for q in range(1, n, 2))
    return alpha == na and beta == nb


@dataclass(frozen=True, order=True)
class LocalQE:
    source: tuple[int, ...]
    target: tuple[int, ...]
    n_qubits: int

    def __post_init__(self):
        if not 1 <= len(self.source) == len(self.target) <= 2 or not 1 <= self.n_qubits <= 62:
            raise ValueError("A localQE is a single/double exchange on a bounded register")
        allq = self.source+self.target
        if tuple(sorted(self.source)) != self.source or tuple(sorted(self.target)) != self.target or len(set(allq)) != len(allq):
            raise ValueError("Sorted disjoint source/target supports required")
        if any(q < 0 or q >= self.n_qubits for q in allq): raise ValueError("Qubit outside register")
        if sum(q % 2 for q in self.source) != sum(q % 2 for q in self.target): raise ValueError("Preserve Nalpha/Nbeta, not just total N")

    @property
    def identity(self):
        return digest({"family": "author-QE", **asdict(self), "indexing": "logical interleaved big-endian",
                       "normalization": "author qe_circuit; constant global phase removed so U(0)=I"})

    def circuit(self, theta):
        if not np.isfinite(theta): raise ValueError("Finite angle required")
        support = tuple(sorted(self.source+self.target))
        src = tuple(support.index(q) for q in self.source); dst = tuple(support.index(q) for q in self.target)
        qc = qe_circuit(list(self.source), list(self.target), float(theta), self.n_qubits, big_endian=True)
        qc.global_phase -= float(np.angle(_zero_phase(src, dst, len(support))))
        return qc

    def append(self, parent, theta):
        if parent.num_qubits != self.n_qubits or parent.num_clbits: raise ValueError("Bind the same unmeasured physical register")
        return compile_logical(parent.compose(self.circuit(theta)))

    def action(self, amplitudes):
        """Sparse action of the actual author circuit generator on named IDs."""
        support = tuple(sorted(self.source+self.target)); k = len(support)
        src = tuple(support.index(q) for q in self.source); dst = tuple(support.index(q) for q in self.target)
        generator = _local_generator(src, dst, k); output = {}
        mask = sum(1 << (self.n_qubits-1-q) for q in support)
        for raw, amplitude in amplitudes.items():
            raw = int(raw)
            if not 0 <= raw < 1 << self.n_qubits: raise ValueError("Invalid named amplitude ID")
            if not np.isfinite(amplitude): raise ValueError("Nonfinite classical coefficient")
            column = sum(((raw >> (self.n_qubits-1-q)) & 1) << (k-1-i) for i, q in enumerate(support))
            for row in np.flatnonzero(abs(generator[:, column]) > 1e-8):
                value = raw & ~mask
                for i, q in enumerate(support):
                    value |= ((int(row) >> (k-1-i)) & 1) << (self.n_qubits-1-q)
                output[value] = output.get(value, 0j)+generator[row, column]*amplitude
        return {x: value for x, value in output.items() if abs(value) > 1e-14}


@lru_cache(maxsize=128)
def _zero_phase(src, dst, k):
    zero = Operator(qe_circuit(list(src), list(dst), 0., k, big_endian=True)).data
    phase = zero[0, 0]/abs(zero[0, 0])
    if not np.allclose(zero, phase*np.eye(1 << k), atol=1e-9):
        raise RuntimeError("Author helper at zero is not a constant global phase")
    return phase


@lru_cache(maxsize=128)
def _local_generator(src, dst, k):
    # At most16×16. Never embed a unitary into the full n-qubit Fock space.
    eps = 1e-6
    plus = Operator(qe_circuit(list(src), list(dst), eps, k, big_endian=True)).data
    minus = Operator(qe_circuit(list(src), list(dst), -eps, k, big_endian=True)).data
    # The author's double-QE helper carries exp(-i*pi/4) even at theta=0.
    # It is physically harmless, but dU/dtheta without removing that constant
    # gauge is not the anti-Hermitian generator. Do not alter vendor code.
    generator = (plus-minus)/(2*eps*_zero_phase(src, dst, k))
    if not np.allclose(generator+generator.conj().T, 0, atol=1e-8): raise RuntimeError("Local generator is not anti-Hermitian")
    generator[abs(generator) < 1e-8] = 0
    return generator


def excitation_between(y, x, n):
    y, x = int(y), int(x)
    if any(v < 0 or v >= 1 << n for v in (y, x)): raise ValueError("Named configuration outside register")
    removed = tuple(q for q in range(n) if (y >> (n-1-q)) & 1 and not (x >> (n-1-q)) & 1)
    added = tuple(q for q in range(n) if (x >> (n-1-q)) & 1 and not (y >> (n-1-q)) & 1)
    if len(removed) != len(added) or len(removed) not in (1, 2): return None
    if sum(q % 2 for q in removed) != sum(q % 2 for q in added): return None
    # Reverse orientation is the same variational family with angle sign flipped.
    source, target = min((removed, added), (added, removed))
    return LocalQE(source, target, n)


def observed_ci(backend, observed_counts, hf, nalpha, nbeta, cap=64):
    if not 1 <= cap <= 512: raise ValueError("Bounded named CI cap1..512 required")
    n = backend.n_qubits
    if not valid_sector(hf, n, nalpha, nbeta): raise ValueError("Known HF does not belong to the registered physical sector")
    pairs = []
    for raw, count in observed_counts.items():
        if not isinstance(raw, (int, np.integer)) or isinstance(raw, (bool, np.bool_)) or not 0 <= int(raw) < 1 << n:
            raise ValueError("Corrupt register ID is not merely wrong-sector sampling")
        if not isinstance(count, (int, np.integer)) or isinstance(count, (bool, np.bool_)) or count <= 0: raise ValueError("Positive actual observed integer counts required")
        if valid_sector(raw, n, nalpha, nbeta) and int(raw) != int(hf): pairs.append((int(raw), int(count)))
    order = [int(hf)]+[x for x, _ in sorted(pairs, key=lambda a: (-a[1], a[0]))][:cap-1]
    selected = sorted(order); h = backend.matrix(selected, selected)
    if not np.allclose(h, h.T.conj(), atol=1e-10): raise RuntimeError("Non-Hermitian named CI")
    values, vectors = eigh(h, check_finite=False)
    E = float(values[0]); coefficients = vectors[:, 0]
    internal = float(np.linalg.norm(h@coefficients-E*coefficients))
    if internal > 1e-8: raise RuntimeError("CI Ritz residual failed")
    return {"selected_ids": selected, "frequency_order": order, "coefficients": coefficients, "energy_Ha": E,
            "actual_dimension": len(selected), "internal_Ritz_residual": internal,
            "first_excitation_gap_Ha": float(values[1]-E) if len(values)>1 else None,
            "observed_support_only_plus_knownHF": True, "FCI_used": False}


def ci_gradient(backend, operator, ci):
    amplitudes = dict(zip(ci["selected_ids"], ci["coefficients"]))
    action = operator.action(amplitudes)
    if not action: return 0.
    ids = sorted(action); hphi = backend.matrix(ids, ci["selected_ids"])@ci["coefficients"]
    # Gradient of the classical Ritz vector under this unitary, NOT necessarily
    # the gradient of the actually prepared physical quantum sampler state.
    return float(2*np.vdot(hphi, np.array([action[x] for x in ids])).real)


def local_proposals(backend, connections, ci, target_cap=128, candidate_cap=32, include_ci_gradients=False):
    if target_cap < 1 or candidate_cap < 1: raise ValueError("Positive bounded target/candidate caps")
    external = connections.external(ci["selected_ids"], ci["coefficients"], ci["energy_Ha"], denominator_floor=1e-4)
    ids = external["ids"]; importance = external["en_energy_importance"]
    # Shared local-bank preselection uses coefficient-weighted couplings,
    # not residual/EN ranks. Thus a gradient-ranking ablation is not secretly
    # receiving a residual-ranked bank. This is still a local-bank control,
    # not a substitute for native global-pool ADAPT/ADAPT-QSCI baselines.
    order = np.lexsort((ids, -external["hci_importance"]))[:target_cap]
    targets = ids[order]; weights = importance[order]
    matrix = backend.matrix(targets, ci["selected_ids"]) if len(targets) else np.empty((0, len(ci["selected_ids"])))
    proposals = {}
    for row, (x, weight) in enumerate(zip(targets, weights)):
        terms = abs(matrix[row]*ci["coefficients"]); total = float(terms.sum())
        if total <= 1e-14: continue
        for y, term in zip(ci["selected_ids"], terms):
            if term <= 1e-14: continue
            op = excitation_between(y, int(x), backend.n_qubits)
            if op is None: continue
            entry = proposals.setdefault(op, {"operator": op, "residual_priority": 0., "common_bank_priority": 0., "proposal_target_ids": set()})
            entry["residual_priority"] += float(weight*term/total); entry["proposal_target_ids"].add(int(x))
            entry["common_bank_priority"] = max(entry["common_bank_priority"], float(term))
    selected = sorted(proposals.values(), key=lambda p: (-p["common_bank_priority"], p["operator"]))[:candidate_cap]
    for p in selected:
        p["proposal_target_ids"] = sorted(p["proposal_target_ids"])
        before = backend.matrix_entry_requests
        p["ci_gradient_proxy"] = ci_gradient(backend, p["operator"], ci) if include_ci_gradients else None
        p["gradient_matrix_entries"] = backend.matrix_entry_requests-before
        p["gradient_scope"] = "classical Ritz-vector commutator; not actual sampler-state gradient" if include_ci_gradients else "NOT_EVALUATED"
    return selected, {"returned_external_targets": len(ids), "screened_targets_considered": len(targets),
                      "unique_local_operator_patterns": len(proposals), "returned_operator_candidates": len(selected),
                      "residual_scope": external["residual_scope"], "intruder_count": external["intruders"],
                      "near_zero_denominators": external["near_zero_denominators"],
                      "classical_proposal_targets_are_not_quantum_observations": True,
                      "FCI_used": False, "full_pool_constructed": False}
