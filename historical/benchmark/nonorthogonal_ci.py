"""Integral-only Slater overlaps/H matrix elements, including zero overlaps.

Independent implementation of biorthogonal Slater--Condon contractions. Unlike
overlap * local-energy formulae, no inverse occupied-overlap matrix is used.
All occupied columns use ascending interleaved spin-orbital creator order.
This module never enumerates a Hilbert space or accepts FCI labels.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh


def occupied_columns(rotation, determinant):
    u = np.asarray(rotation, complex)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("Square spatial rotation required")
    n = len(u)
    if not np.allclose(u.conj().T @ u, np.eye(n), atol=1e-10, rtol=0):
        raise ValueError("Nonunitary orbital basis")
    if not 0 <= int(determinant) < 1 << (2*n):
        raise ValueError("Configuration outside register")
    occupied = [q for q in range(2*n) if int(determinant) >> (2*n-1-q) & 1]
    result = np.zeros((2*n, len(occupied)), complex)
    for j, q in enumerate(occupied):
        result[q % 2::2, j] = u[:, q//2]
    return result


class NonorthogonalHamiltonian:
    def __init__(self, h1, eri, core):
        self.h1 = np.asarray(h1, float)
        eri = np.asarray(eri, float)
        self.norb = len(self.h1); self.core = float(core)
        n = self.norb
        if self.h1.shape != (n,n) or eri.shape != (n,)*4:
            raise ValueError("Integral dimensions")
        if not np.isfinite(eri).all() or not np.isfinite(self.h1).all():
            raise ValueError("Nonfinite integrals")
        coulomb = eri.reshape(n*n,n*n)
        if not np.allclose(coulomb,coulomb.T,atol=1e-10,rtol=0):
            raise ValueError("Real chemist-order Coulomb integrals required")
        weights, vectors = eigh((coulomb+coulomb.T)/2)
        # Keep ALL factors, including signed roundoff eigenvalues. No CDF
        # truncation, fitted Hamiltonian, or Cholesky tolerance is introduced.
        self.factors = (vectors*np.sqrt(abs(weights))[None,:]).T.reshape(n*n,n,n)
        self.signs = np.sign(weights)
        rebuilt = np.einsum("t,tpq,trs->pqrs", self.signs,self.factors,self.factors)
        self.factorization_error = float(np.max(abs(rebuilt-eri)))
        if self.factorization_error > 1e-9:
            raise RuntimeError("Untruncated Coulomb factorization parity")
        self.pair_requests = 0

    def pair(self, bra, ket):
        a,b = np.asarray(bra,complex),np.asarray(ket,complex)
        self.pair_requests += 1
        if a.shape != b.shape or a.ndim != 2 or a.shape[0] != 2*self.norb:
            raise ValueError("Same-electron Slater columns required")
        ne = a.shape[1]
        if not np.allclose(a.conj().T@a,np.eye(ne),atol=1e-9,rtol=0) or not np.allclose(b.conj().T@b,np.eye(ne),atol=1e-9,rtol=0):
            raise ValueError("Occupied orbitals must be orthonormal")
        if ne == 0:
            return complex(self.core),1+0j
        left,s,right_h = np.linalg.svd(a.conj().T@b)
        right = right_h.conj().T
        phase = np.linalg.det(left)*np.linalg.det(right).conjugate()
        aa = (a@left).reshape(self.norb,2,ne)
        bb = (b@right).reshape(self.norb,2,ne)
        overlap = phase*np.prod(s)
        one = np.einsum("psi,pq,qsi->i",aa.conj(),self.h1,bb,optimize=True)
        transformed = np.einsum("psi,tpq,qsj->tij",aa.conj(),self.factors,bb,optimize=True)
        energy = self.core*np.prod(s)+0j
        # Products with omitted singular values remain defined at exactly zero
        # overlap. A double-excited orthogonal pair can have S=0 and H!=0.
        for i in range(ne):
            energy += np.prod(np.delete(s,i))*one[i]
            for j in range(i+1,ne):
                weight = np.prod(np.delete(s,[i,j]))
                jk = np.dot(self.signs,transformed[:,i,i]*transformed[:,j,j]-transformed[:,i,j]*transformed[:,j,i])
                energy += weight*jk
        return phase*energy,overlap

    def matrices(self, columns):
        n = len(columns)
        h = np.empty((n,n),complex); s = np.empty_like(h)
        for i in range(n):
            for j in range(i+1):
                hij,sij = self.pair(columns[i],columns[j])
                h[i,j],s[i,j] = hij,sij
                h[j,i],s[j,i] = hij.conjugate(),sij.conjugate()
        return h,s


def generalized_ground(h,s,relative_cutoff=1e-9):
    """Canonical orthogonalization with explicit retained rank and diagnostics."""
    h,s = np.asarray(h,complex),np.asarray(s,complex)
    if h.shape != s.shape or h.ndim != 2 or h.shape[0] != h.shape[1] or len(h)==0:
        raise ValueError("Nonempty equally sized projected matrices required")
    if not np.isfinite(h).all() or not np.isfinite(s).all():
        raise ValueError("Nonfinite projected matrices")
    if not 0 < relative_cutoff < 1:
        raise ValueError("Registered relative rank cutoff required")
    if max(np.max(abs(h-h.conj().T)),np.max(abs(s-s.conj().T))) > 1e-9:
        raise ValueError("Non-Hermitian projected matrices")
    w,v = eigh(s); maximum = float(w[-1])
    if maximum <= 0 or w[0] < -1e-9*maximum:
        raise ValueError("Overlap matrix not positive semidefinite")
    keep = w > relative_cutoff*maximum
    x = v[:,keep]/np.sqrt(w[keep])
    small = x.conj().T@h@x
    e,c = eigh((small+small.conj().T)/2)
    coefficients = x@c[:,0]
    residual = h@coefficients-e[0]*(s@coefficients)
    projected_residual = float(np.linalg.norm(x.conj().T@residual))
    if projected_residual > 1e-7:
        raise RuntimeError("Retained-space generalized eigen residual")
    return {"energy_Ha":float(e[0]),"rank":int(keep.sum()),"candidate_vectors":len(h),
            "overlap_eigenvalues":w.tolist(),"overlap_relative_cutoff":relative_cutoff,
            "retained_condition_number":float(maximum/min(w[keep])),
            "retained_residual_norm":projected_residual,
            "full_generalized_residual_norm":float(np.linalg.norm(residual)),
            "normalization":float(np.vdot(coefficients,s@coefficients).real)},coefficients
