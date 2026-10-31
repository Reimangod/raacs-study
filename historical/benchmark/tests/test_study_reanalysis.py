import sqlite3
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from benchmark.common import rng_for
from benchmark.study.core import RitzCache, individual_m90, joint_discovery_m90, open_db, pack_pairs, ranks, unpack_pairs
from benchmark.study.mixtures import allocation, choose_diverse, selected_from_counts


def test_m90_categorical_not_independent_product():
    assert individual_m90(.5) == 4
    assert individual_m90(0) is None
    assert individual_m90(1) == 1
    assert joint_discovery_m90([.5, .5], "two", 4096)["m90_estimate"] == 5
    assert joint_discovery_m90([1.], "one", 64)["m90_estimate"] == 1
    assert joint_discovery_m90([.1, 0.], "zero")["status"] == "UNREACHABLE_ZERO_PROBABILITY"
    assert int(individual_m90(1e-25)) > 2**63


def test_ranked_spaces_are_nested_same_realized_dimension():
    r = ranks([0, 3, 5, 6, 15], [1, 7, 3, 2, 500], [0, 3, 5, 6], 0, "test")
    assert r["frequency"] == [0, 3, 5, 6]
    assert set(r["uniform_observed"]) == {0, 3, 5, 6}
    assert r["uniform_observed"][0] == 0
    assert len(r["frequency"][:2]) == len(r["uniform_observed"][:2])


def test_sparse_ritz_cache_residual_and_identity(tmp_path):
    db = open_db(tmp_path / "cache.sqlite")
    h = sparse.csr_matrix([[1, .1, 0], [.1, .7, .2], [0, .2, .6]])
    cache = RitzCache(h, [0, 1, 2], "physical-problem", db)
    i, e, hit, _ = cache.solve([0, 1])
    assert not hit
    j, e2, hit, _ = cache.solve([1, 0])
    assert i == j and hit and e == e2
    _, more, _, _ = cache.solve([0, 1, 2])
    assert more < e
    assert abs(more - np.linalg.eigvalsh(h.toarray())[0]) < 1e-12
    assert db.execute("SELECT count(*) FROM subspaces").fetchone()[0] == 2
    with pytest.raises(ValueError):
        cache.solve([0, 0])
    db.close()


def test_counts_compact_lossless_and_seed_namespaces():
    ids, counts = np.array([0, 2, 65535]), np.array([7, 3, 100000])
    a, b = unpack_pairs(pack_pairs(ids, counts))
    assert np.array_equal(a, ids) and np.array_equal(b, counts)
    assert rng_for("new-phase", "cp", 1000, 0)[1] != rng_for("finite-evaluation-v1", "cp", 1000, 0)[1]


def test_mixture_budget_and_pilot_only_diversity():
    for total in (10, 25, 1000):
        assert sum(allocation(total, 3)) == total
    selected = choose_diverse({1: {0: 9, 3: 1}, 2: {0: 8, 5: 2}, 3: {0: 10}}, [1, 2, 3], 2, 0)
    assert selected == [1, 2]
    assert selected_from_counts({0: 1, 3: 2, 5: 20, 7: 100}, [0, 3, 5], 0, 2) == [0, 5]


def test_ritz_iterative_path_matches_known_tridiagonal(tmp_path):
    db = open_db(tmp_path / "sparse.sqlite")
    h = sparse.diags([-np.ones(127), 2*np.ones(128), -np.ones(127)], [-1, 0, 1], format="csr")
    cache = RitzCache(h, np.arange(128), "tridiagonal-toy", db)
    _, value, _, _ = cache.solve(np.arange(128))
    assert abs(value - (2 - 2*np.cos(np.pi/129))) < 1e-11
    db.close()


def test_physical_h2_resume_does_not_duplicate_draws(tmp_path):
    import json
    from benchmark.common import ROOT, file_digest
    from benchmark.study.reanalysis import finite_case
    source = ROOT / "results/standard-v1"
    path = source / "checkpoints/H2_R0.7/ceo_ovp/step-0001.json"
    cp = json.loads(path.read_text())
    record = {"checkpoint_id": cp["checkpoint_id"], "case_id": cp["case_id"], "method": cp["method"],
              "step": cp["step"], "primary": True, "distribution_path": "distributions/H2_R0.7/ceo_ovp/step-0001.npz",
              "distribution_sha": cp["distribution_sha256"]}
    options = {"source": str(source), "output": str(tmp_path), "threads": 1, "namespace": "engineering-resume-test",
               "case_seconds": 60, "free_floor_bytes": 0, "output_cap_bytes": 100*2**20, "worker_ram_bytes": 8*2**30,
               "repeats": 1, "joint_mc_trials": 8}
    a = finite_case("H2_R0.7", [record], options)
    b = finite_case("H2_R0.7", [record], options)
    assert a["status"] == b["status"] == "COMPLETED"
    assert a["counts"] == b["counts"]
    assert b["counts"]["draws"] == 9 and b["counts"]["readouts"] == 126
    assert file_digest(source / record["distribution_path"]) == record["distribution_sha"]
