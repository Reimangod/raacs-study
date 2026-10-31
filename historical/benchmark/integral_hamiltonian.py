"""Real-integral Slater–Condon named entries, without a Fock matrix or FCI vector.

Interleaved alpha/beta, OpenFermion big-endian occupation IDs. This is the
integral backend qualification primitive, not yet an executed scaling study.
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True)
def _popcount(x):
    count = 0
    while x:
        x &= x-np.uint64(1)
        count += 1
    return count


@njit(cache=True)
def _phase_one(x, q, n):
    # Occupations before logicalq are more-significant raw bits.
    parity = _popcount(x >> np.uint64(n-q))
    return -1. if parity % 2 else 1.


@njit(cache=True)
def _diagonal(x, n, h1, eri, core):
    energy = core
    for i in range(n):
        if not ((x >> np.uint64(n-1-i)) & np.uint64(1)):
            continue
        energy += h1[i//2, i//2]
        for j in range(n):
            if (x >> np.uint64(n-1-j)) & np.uint64(1):
                energy += .5*(eri[i//2, i//2, j//2, j//2]-(eri[i//2, j//2, j//2, i//2] if i % 2 == j % 2 else 0.))
    return energy


@njit(cache=True)
def _element(bra, ket, n, h1, eri, core):
    difference = bra ^ ket
    count = _popcount(difference)
    if count == 0:
        return _diagonal(ket, n, h1, eri, core)
    if count not in (2, 4):
        return 0.
    removed = np.empty(2, np.int64); added = np.empty(2, np.int64)
    nr, na = 0, 0
    for q in range(n):
        mask = np.uint64(1) << np.uint64(n-1-q)
        if difference & mask:
            if ket & mask:
                if nr == 2:
                    return 0.
                removed[nr] = q; nr += 1
            else:
                if na == 2:
                    return 0.
                added[na] = q; na += 1
    if nr != na:
        return 0.
    if nr == 1:
        i, a = removed[0], added[0]
        if i % 2 != a % 2:
            return 0.
        value = h1[a//2, i//2]
        for j in range(n):
            if (ket >> np.uint64(n-1-j)) & np.uint64(1):
                value += eri[a//2, i//2, j//2, j//2]
                if a % 2 == j % 2:
                    value -= eri[a//2, j//2, j//2, i//2]
        sign = _phase_one(ket, i, n)
        x = ket ^ (np.uint64(1) << np.uint64(n-1-i))
        sign *= _phase_one(x, a, n)
        return sign*value
    i, j = removed[0], removed[1]; a, b = added[0], added[1]
    value = 0.
    if a % 2 == i % 2 and b % 2 == j % 2:
        value += eri[a//2, i//2, b//2, j//2]
    if a % 2 == j % 2 and b % 2 == i % 2:
        value -= eri[a//2, j//2, b//2, i//2]
    # a†_a a†_b a_j a_i, applied right-to-left to the ket.
    x = ket; sign = 1.
    for q in (i, j, b, a):
        sign *= _phase_one(x, q, n)
        x ^= np.uint64(1) << np.uint64(n-1-q)
    return sign*value


@njit(cache=True)
def _matrix(left, right, n, h1, eri, core):
    out = np.empty((len(left), len(right)))
    for i in range(len(left)):
        for j in range(len(right)):
            out[i, j] = _element(left[i], right[j], n, h1, eri, core)
    return out


class IntegralHamiltonian:
    def __init__(self, h1, eri, core_energy):
        if np.max(abs(np.asarray(h1).imag)) > 1e-12 or np.max(abs(np.asarray(eri).imag)) > 1e-12:
            raise ValueError("This version qualifies real spatial integrals only")
        self.h1 = np.ascontiguousarray(np.asarray(h1).real, dtype=float)
        self.eri = np.ascontiguousarray(np.asarray(eri).real, dtype=float)
        self.norb = len(self.h1); self.n_qubits = 2*self.norb; self.core = float(core_energy)
        if self.h1.shape != (self.norb, self.norb) or self.eri.shape != (self.norb,)*4:
            raise ValueError("Integral dimensions")
        if self.n_qubits > 62:
            raise ValueError("This uint64 backend is qualified up to62 qubits")
        self.matrix_entry_requests = 0

    def matrix(self, left, right):
        for ids in (left, right):
            if any(int(x) < 0 or int(x) >= 1 << self.n_qubits for x in ids):
                raise ValueError("Named configuration outside register")
        self.matrix_entry_requests += len(left)*len(right)
        return _matrix(np.array(left, np.uint64), np.array(right, np.uint64), self.n_qubits, self.h1, self.eri, self.core)

    @classmethod
    def from_saved_integrals(cls, h1, eri, hf_id, hf_energy):
        # Determine only the scalar core offset from an independently known HF
        # energy. No FCI energy/coefficient/determinant list is accepted.
        n = 2*len(h1)
        no_core_hf = _diagonal(np.uint64(hf_id), n, np.asarray(h1).real, np.asarray(eri).real, 0.)
        return cls(h1, eri, hf_energy-no_core_hf)
