import numpy as np
from benchmark.common import ROOT
from benchmark.chemistry import load_reference
from benchmark.integral_hamiltonian import IntegralHamiltonian


def test_named_integral_entries_match_independent_jw_sparse_backend():
    rng = np.random.default_rng(443)
    for case in ("H2_R0.7", "H4_R1.1", "LiH_R1.5", "BeH2_R2.0", "H6_R1.8", "N2_R1.4"):
        ref = load_reference(ROOT / "results/standard-v1/references" / case); m = ref["meta"]
        h = IntegralHamiltonian.from_saved_integrals(ref["h1"], ref["eri"], m["hf_id"], m["rhf_energy"])
        ids = rng.choice(ref["ids"], min(48, len(ref["ids"])), replace=False)
        actual = h.matrix(ids, ids)
        expected = ref["h"][ids][:, ids].toarray()
        assert np.max(abs(actual-expected)) < 1e-10, case
        # Include arbitrary occupations, not just the target FCI sector.
        ids = rng.choice(1 << m["n_qubits"], min(48, 1 << m["n_qubits"]), replace=False)
        assert np.max(abs(h.matrix(ids, ids)-ref["h"][ids][:, ids].toarray())) < 1e-10, case


def test_40qubit_named_matrix_needs_no_fock_matrix_or_statevector():
    n = 20; h1 = np.diag(np.linspace(-2, 1, n)); eri = np.zeros((n,)*4)
    h = IntegralHamiltonian(h1, eri, 1.2)
    hf = sum(1 << (39-q) for q in range(20))
    excited = hf ^ (1 << 39) ^ (1 << 19)
    matrix = h.matrix([hf, excited], [hf, excited])
    assert matrix.shape == (2, 2) and matrix[0, 1] == 0.
    assert h.h1.nbytes+h.eri.nbytes < 2*2**20


def test_40qubit_nonzero_long_range_fermionic_phases():
    n = 20; h1 = np.zeros((n, n)); h1[0, 19] = h1[19, 0] = .13
    W = np.zeros((n, n)); W[0, 19] = W[19, 0] = np.sqrt(.2)
    eri = np.einsum("pq,rs->pqrs", W, W)
    h = IntegralHamiltonian(h1, eri, 0.)
    hf = sum(1 << (39-q) for q in range(20))
    single = hf ^ (1 << 39) ^ (1 << 1)
    double = hf ^ (1 << 39) ^ (1 << 38) ^ (1 << 1) ^ (1 << 0)
    # Independent Python ladder action on named 40-bit occupations, no Hilbert vector.
    def amplitude(ket, operators):
        sign = 1
        for q, creation in reversed(operators):
            mask = 1 << (39-q)
            if bool(ket & mask) == creation:
                return None, 0
            sign *= -1 if (ket >> (40-q)).bit_count() % 2 else 1
            ket ^= mask
        return ket, sign
    actual_id, sign = amplitude(hf, [(38, True), (0, False)])
    assert actual_id == single and np.isclose(h.matrix([single], [hf])[0, 0], .13*sign)
    actual_id, sign = amplitude(hf, [(38, True), (39, True), (1, False), (0, False)])
    assert actual_id == double and np.isclose(h.matrix([double], [hf])[0, 0], .2*sign)
