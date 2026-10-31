import numpy as np
import pytest
from benchmark.molecular_problem_no_fci import MolecularSpec, build_problem, sector_ids


def test_fresh_H2_without_ground_state_solver(tmp_path, monkeypatch):
    import pyscf.fci.direct_spin1
    def forbidden(*a, **k):
        raise AssertionError("A physical problem must not invoke FCI")
    monkeypatch.setattr(pyscf.fci.direct_spin1, "kernel", forbidden)
    p = build_problem(MolecularSpec("H2", .85), tmp_path, small_full_H=True)
    m = p["meta"]
    assert not m["FCI_computed"] and m["ground_state_reference_energy_Ha"] is None
    assert p["h"].shape == (16, 16) and len(p["ids"]) == 4
    assert abs(p["h"][m["hf_id"], m["hf_id"]]-m["hf_energy_Ha"]) < 1e-8
    restored = build_problem(MolecularSpec("H2", .85), tmp_path, small_full_H=True)
    assert restored["meta"] == m
    with pytest.raises(RuntimeError, match="drift"):
        build_problem(MolecularSpec("H2", .86), tmp_path, small_full_H=True)


def test_integral_only_allocates_no_full_H_or_sector(tmp_path, monkeypatch):
    import openfermion as of
    def forbidden(*a, **k):
        raise AssertionError("No full Fock matrix is allowed in integral-only mode")
    monkeypatch.setattr(of, "get_sparse_operator", forbidden)
    p = build_problem(MolecularSpec("H4", 1.25), tmp_path)
    assert p["meta"]["sector_dimension"] == 36
    assert not p["meta"]["full_Fock_H_constructed"] and "h" not in p and "ids" not in p
    assert p["meta"]["n_qubits"] == 8
    assert set(p["meta"]["artifact_sha256"]) == {"integrals.npz"}


def test_sector_interleaved_big_endian_ids():
    assert set(sector_ids(2, 1, 1)) == {12, 9, 6, 3}
    ids = sector_ids(3, 2, 1)
    assert len(ids) == 9 and np.all(ids >= 0)
