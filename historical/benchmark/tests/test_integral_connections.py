import numpy as np
from benchmark.common import ROOT
from benchmark.chemistry import load_reference
from benchmark.integral_hamiltonian import IntegralHamiltonian
from benchmark.integral_connections import IntegralConnections


def test_integral_connections_against_independent_full_source_h():
    for case in ("H2_R0.7", "H4_R1.1", "LiH_R1.5", "BeH2_R1.3", "H6_R1.1", "N2_R1.1"):
        ref = load_reference(ROOT / "results/standard-v1/references" / case); meta = ref["meta"]
        h = IntegralHamiltonian.from_saved_integrals(ref["h1"], ref["eri"], meta["hf_id"], meta["rhf_energy"])
        backend = IntegralConnections(h, coupling_floor=1e-12, parent_cap=None)
        support = [int(meta["hf_id"]), int(ref["ids"][0])]
        support = sorted(set(support)); c = np.ones(len(support))/np.sqrt(len(support))
        data = backend.external(support, c, meta["rhf_energy"])
        exact = ref["h"][:, support] @ c; exact[support] = 0
        kept = np.array([x for x in np.flatnonzero(abs(exact) > 1e-10)], dtype=int)
        actual = dict(zip(data["ids"], data["partial_residual"]))
        assert all(abs(actual.get(int(x), 0)-exact[x]) < 1e-9 for x in kept)
        assert all(abs(value-exact[x]) < 1e-9 for x, value in actual.items())
        assert not data["parent_truncated"]


def test_parent_cap_is_explicit_and_40_qubit_no_hilbert_vector():
    norb = 20; h1 = np.eye(norb)
    for p in range(norb-1):
        h1[p, p+1] = h1[p+1, p] = .2
    h = IntegralHamiltonian(h1, np.zeros((norb,)*4), 0.)
    hf = sum(1 << (39-q) for q in range(20))
    connections = IntegralConnections(h, parent_cap=1)
    data = connections.external([hf], [1.], 20.)
    assert len(data["ids"]) == 1 and data["parent_truncated"]
    assert data["residual_scope"].startswith("partial")
    assert connections.statistics.named_matrix_entries_examined > 1000
