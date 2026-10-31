from __future__ import annotations

import io
import json
import math
import resource
import sqlite3
import sys
import zlib
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.stats import binom

from ..common import digest, rng_for
from ..metrics import analytic_metrics

SHOTS = (10, 25, 50, 100, 250, 500, 1000, 10000, 100000)
CAPS = (8, 16, 32, 64, 128, 256, 512)
TOPKS = (3, 5, 10, 20, 50)
ADAPT = ("ceo_ovp", "ceo_mvp", "qeb", "qubit", "gsd")
NAMESPACE = "sampler-study-A-independent-counts-v1"


def rss_peak():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def pack_pairs(ids, counts):
    a = np.column_stack((ids, counts)).astype("<i4")
    return zlib.compress(a.tobytes(), level=3)


def unpack_pairs(raw):
    a = np.frombuffer(zlib.decompress(raw), dtype="<i4").reshape(-1, 2)
    return a[:, 0].astype(np.int64), a[:, 1].astype(np.int64)


def individual_m90(probability):
    p = float(probability)
    if p <= 0:
        return None
    if p >= 1:
        return 1
    # Python integers avoid overflow for extremely rare configurations.
    return math.ceil(math.log(.1) / math.log1p(-p))


def joint_discovery_m90(probabilities, identity, trials=256):
    """Categorical coupon collector, not a product of individual discovery rates.

    Each waiting time is geometric at the remaining unseen probability mass.
    Conditional target identity is proportional to its probability. Float64
    inverse-CDF waiting times retain large rare-event scales beyond int64.
    """
    p = np.asarray(probabilities, float)
    if not len(p):
        return {"status": "NO_TARGETS", "trials": 0, "m90_estimate": None}
    if np.any(p <= 0):
        return {"status": "UNREACHABLE_ZERO_PROBABILITY", "trials": 0, "m90_estimate": None}
    if p.sum() > 1 + 1e-12:
        raise ValueError("Target probabilities exceed one")
    rng, seed = rng_for("joint-categorical-discovery-v1", identity, len(p), trials)
    remaining = np.tile(p, (trials, 1))
    times = np.zeros(trials)
    for _ in range(len(p)):
        mass = remaining.sum(axis=1)
        u = rng.random(trials)
        with np.errstate(divide="ignore", invalid="ignore"):
            waits = np.ceil(np.log1p(-u) / np.log1p(-np.minimum(mass, 1)))
        waits[mass >= 1] = 1
        times += np.maximum(waits, 1)
        target = rng.random(trials) * mass
        pick = (remaining.cumsum(axis=1) < target[:, None]).sum(axis=1)
        pick = np.minimum(pick, len(p) - 1)
        remaining[np.arange(trials), pick] = 0
    values = np.sort(times)
    if not np.all(np.isfinite(values)):
        return {"status": "NUMERIC_RANGE_UNRESOLVED", "trials": trials, "m90_estimate": None}
    # Order-statistic uncertainty in the MC estimate of the 90th percentile.
    low = max(0, int(binom.ppf(.05, trials, .9)) - 1)
    high = min(trials - 1, int(binom.ppf(.95, trials, .9)))
    return {"status": "MONTE_CARLO_NOT_EXACT", "trials": trials, "seed": str(seed),
            "m90_estimate": float(values[math.ceil(.9 * trials) - 1]),
            "m90_mc90_low": float(values[low]), "m90_mc90_high": float(values[high]),
            "scope": "categorical waiting-time MC; approximate quantile; float64 inverse CDF"}


def discovery(probability, reference, checkpoint_id, trials=256):
    p = np.asarray(probability, float)
    rows = analytic_metrics(p, reference, SHOTS, TOPKS)
    ids, weights, hf = reference["ids"], reference["weights"], reference["meta"]["hf_id"]
    targets = sorted(np.flatnonzero((ids != hf) & (weights > 1e-16)), key=lambda i: (-weights[i], int(ids[i])))
    individual = [{"id": int(ids[i]), "fci_weight": float(weights[i]), "probability": float(p[ids[i]]),
                   "m90_decimal": str(individual_m90(p[ids[i]])) if p[ids[i]] > 0 else None,
                   "status": "FINITE" if p[ids[i]] > 0 else "UNREACHABLE_ZERO_PROBABILITY"} for i in targets[:50]]
    joint = [{"K_requested": k, "actual_targets": min(k, len(targets)),
              **joint_discovery_m90(p[ids[targets[:k]]], checkpoint_id, trials)} for k in TOPKS]
    return {"rows": rows, "individual": individual, "joint": joint,
            "scope": "FCI_ORACLE_EVALUATION_ONLY; not an online selection feature"}


def ranks(ids, counts, valid_ids, hf, identity):
    valid = set(map(int, valid_ids))
    pairs = [(int(i), int(n)) for i, n in zip(ids, counts) if n > 0 and int(i) in valid and int(i) != hf]
    frequency = [i for i, _ in sorted(pairs, key=lambda v: (-v[1], v[0]))]
    random, _ = rng_for("uniform-observed-order-v1", identity)
    uniform = list(map(int, random.permutation(sorted(i for i, _ in pairs))))
    return {"frequency": [hf] + frequency, "uniform_observed": [hf] + uniform}


def open_db(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS checkpoints (
      id INTEGER PRIMARY KEY, sha TEXT UNIQUE NOT NULL, method TEXT NOT NULL,
      step INTEGER NOT NULL, primary_record INTEGER NOT NULL, metadata TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS analytics (cp INTEGER PRIMARY KEY, payload BLOB NOT NULL);
    CREATE TABLE IF NOT EXISTS subspaces (
      id INTEGER PRIMARY KEY, sha TEXT UNIQUE NOT NULL, ids BLOB NOT NULL,
      dimension INTEGER NOT NULL, energy REAL NOT NULL, residual REAL NOT NULL,
      seconds REAL NOT NULL, projected_bytes INTEGER NOT NULL, rss_hwm INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS draws (
      id INTEGER PRIMARY KEY, cp INTEGER NOT NULL, shots INTEGER NOT NULL, rep INTEGER NOT NULL,
      seed TEXT NOT NULL, counts BLOB NOT NULL, seconds REAL NOT NULL,
      UNIQUE(cp,shots,rep));
    CREATE TABLE IF NOT EXISTS outcomes (
      draw INTEGER NOT NULL, cap INTEGER NOT NULL, rule TEXT NOT NULL, subspace INTEGER NOT NULL,
      cache_hit INTEGER NOT NULL, lookup_seconds REAL NOT NULL,
      PRIMARY KEY(draw,cap,rule));
    CREATE TABLE IF NOT EXISTS progress (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    return db


class RitzCache:
    """Contains only Hamiltonian and physical-sector IDs, never FCI labels."""
    def __init__(self, h, valid_ids, problem_id, db):
        self.ids = np.asarray(valid_ids, np.int64)
        self.h = h.tocsr()[self.ids][:, self.ids].tocsr()
        self.index = {int(v): i for i, v in enumerate(self.ids)}
        self.problem_id, self.db = problem_id, db

    def solve(self, selected):
        start = perf_counter()
        ids = np.array(sorted(map(int, selected)), dtype="<i4")
        if len(set(ids)) != len(ids) or not len(ids):
            raise ValueError("Selected IDs must be nonempty and unique")
        sha = digest([self.problem_id, ids.tolist()])
        existing = self.db.execute("SELECT id,energy FROM subspaces WHERE sha=?", (sha,)).fetchone()
        if existing:
            return existing[0], existing[1], True, perf_counter() - start
        positions = [self.index[int(i)] for i in ids]
        hs = self.h[positions][:, positions].tocsr()
        nonhermitian = hs - hs.conj().T
        if nonhermitian.nnz and max(abs(nonhermitian.data)) > 1e-10:
            raise ValueError("Non-Hermitian subspace")
        projected_bytes = hs.data.nbytes + hs.indices.nbytes + hs.indptr.nbytes
        if len(ids) == 1:
            energy, residual = float(hs[0, 0].real), 0.
        elif len(ids) < 128:
            dense = hs.toarray()
            projected_bytes += dense.nbytes
            values, vectors = eigh(dense, subset_by_index=(0, 0))
            energy = float(values[0])
            residual = float(np.linalg.norm(hs @ vectors[:, 0] - energy * vectors[:, 0]))
        else:
            values, vectors = eigsh(hs, k=1, which="SA", tol=1e-12, maxiter=5000,
                                   v0=np.random.default_rng(7081).normal(size=len(ids)))
            energy = float(values[0])
            residual = float(np.linalg.norm(hs @ vectors[:, 0] - energy * vectors[:, 0]))
        seconds = perf_counter() - start
        if not np.isfinite(energy) or residual > 1e-8:
            raise RuntimeError("Ritz residual failed")
        cur = self.db.execute("INSERT INTO subspaces (sha,ids,dimension,energy,residual,seconds,projected_bytes,rss_hwm) VALUES(?,?,?,?,?,?,?,?)",
                              (sha, zlib.compress(ids.tobytes()), len(ids), energy, residual, seconds, projected_bytes, rss_peak()))
        return cur.lastrowid, energy, False, seconds
