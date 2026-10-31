"""Rebuild every new/reused draw, budget link, selected space and actual-d pair."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import zlib
import hashlib
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import load_npz
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, save_json, rng_for
from .study.core import unpack_pairs, ranks
from .raacs_independent_evaluation import (NAMESPACE, baseline_bank, raacs_distribution, distribution_id,
                                         normalized_probability, draw_components, remaining_budget)


def audit(output):
    output = Path(output); cfg = json.loads((output/"manifest.json").read_text())
    for p, sha in cfg["implementation"].items(): assert file_digest(ROOT/p) == sha
    status = json.loads((output/"execution-status.json").read_text())
    assert status["status"] == "RAACS_INDEPENDENT_COMPARISON_EXECUTED_PENDING_AUDIT"
    source = Path(cfg["source"]); construction = Path(cfg["construction"])
    draws = shots = fresh = reused = spaces_checked = pairs_checked = links_checked = numerical = 0
    maxdiff = 0.; missing = {}; cases = []
    with threadpool_limits(limits=1):
        for case in cfg["cases"]:
            ref = source/"references"/case; meta = json.loads((ref/"reference.json").read_text())
            with np.load(ref/"reference.npz") as z: valid = z["ids"].copy()
            probabilities = {}
            for path in sorted((construction/"cells"/case).glob("*/result.json")):
                components, _ = raacs_distribution(path.parent)
                probabilities[distribution_id(meta["request_digest"], components)] = components
            for b in baseline_bank(source, case):
                if b["failed"]: continue
                with np.load(b["distribution"]) as z: p = normalized_probability(z["probability"], meta["n_qubits"])
                comp = [(p, 1)]; probabilities[distribution_id(meta["request_digest"], comp)] = comp
            db = sqlite3.connect(f"file:{output/'readouts'/case/'readouts.sqlite'}?mode=ro", uri=True)
            checkpoint = dict(db.execute("SELECT id,sha FROM checkpoints")); orders = {}; shotmap = {}; origin_connections = {}
            for did, cp, M, rep, seed, blob in db.execute("SELECT id,cp,shots,rep,seed,counts FROM draws"):
                key = checkpoint[cp]; ids, ns = unpack_pairs(blob)
                assert int(ns.sum()) == M and all(int(n) > 0 for n in ns)
                kind, raw = db.execute("SELECT kind,metadata FROM origins WHERE draw=?", (did,)).fetchone(); origin = json.loads(raw)
                if kind == "NEW_INDEPENDENT_COMPONENT_DRAWS":
                    a, b = draw_components(probabilities[key], M, key, rep)
                    assert seed == str(rng_for(NAMESPACE, key, M, rep)[1]); fresh += 1
                elif kind == "EXACT_EXISTING_BASELINE_ACQUISITION_REUSE":
                    path = origin["source_db"]
                    if path not in origin_connections: origin_connections[path] = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                    original, original_seed = origin_connections[path].execute("SELECT counts,seed FROM draws WHERE id=?", (origin["source_draw"],)).fetchone()
                    assert hashlib.sha256(original).hexdigest() == origin["source_counts_sha256"]
                    assert original_seed == seed == origin["original_seed"]
                    a, b = unpack_pairs(original); reused += 1
                else: raise AssertionError("Unknown acquisition origin")
                assert np.array_equal(ids, a) and np.array_equal(ns, b)
                orders[did] = ranks(ids, ns, valid, meta["hf_id"], [key, M, rep])["frequency"]
                shotmap[did] = M; draws += 1; shots += M
            spaces = {sid: (np.frombuffer(zlib.decompress(blob), dtype="<i4").tolist(), E)
                      for sid, blob, E in db.execute("SELECT id,ids,energy FROM subspaces")}
            for did, R, sid in db.execute("SELECT draw,cap,subspace FROM outcomes"):
                assert spaces[sid][0] == sorted(orders[did][:R]); spaces_checked += 1
            buildmap = dict(db.execute("SELECT method,build_shots FROM cells")); linkmap = {}
            for method, M, scope, rep, did, state in db.execute("SELECT * FROM links"):
                remaining, reason = (M, "AVAILABLE") if scope == "readout_only" else remaining_budget(M, buildmap[method])
                assert state == reason
                if reason == "AVAILABLE": assert shotmap[did] == remaining
                else:
                    assert did is None
                    missing[reason] = missing.get(reason, 0)+1
                linkmap[(method, M, scope, rep)] = did; links_checked += 1
            for method, M, scope, rep, R, baseline, d, sa, sb in db.execute("SELECT * FROM pairs"):
                da = linkmap[(method, M, scope, rep)]; dbid = linkmap[(baseline, M, scope, rep)]
                assert da is not None and dbid is not None
                assert d == min(R, len(orders[da]), len(orders[dbid]))
                assert spaces[sa][0] == sorted(orders[da][:d]) and spaces[sb][0] == sorted(orders[dbid][:d])
                pairs_checked += 1
            h = load_npz(ref/"hamiltonian.npz")
            for sid in sorted(spaces)[::max(1, len(spaces)//5)][:5]:
                ids, energy = spaces[sid]; dense = h[ids][:, ids].toarray()
                E = float(eigh(dense, eigvals_only=True, subset_by_index=(0, 0))[0])
                diff = abs(E-energy); assert diff < 1e-8; maxdiff = max(maxdiff, diff); numerical += 1
            for c in origin_connections.values(): c.close()
            cells = db.execute("SELECT count(*) FROM cells").fetchone()[0]
            assert db.execute("SELECT count(*) FROM links").fetchone()[0] == cells*len(cfg["M"])*len(cfg["readout_scopes"])*cfg["repeats"]
            db.close(); cases.append(case)
    result = {"status": "RAACS_INDEPENDENT_COMPARISON_COUNTS_AND_PAIRS_VALIDATED", "cases": cases,
              "all_physical_draw_records_checked": draws, "all_linked_model_shots": shots,
              "new_draws_rebuilt": fresh, "reused_draws_checked_against_original_acquisition": reused,
              "all_common_R_selections_checked": spaces_checked, "all_budget_links_checked": links_checked,
              "all_actual_d_pairs_checked": pairs_checked, "fresh_independent_Ritz_solves": numerical,
              "maximum_fresh_Ritz_difference_Ha": maxdiff, "unavailable_budget_links": missing,
              "scope": "postconstruction exploratory, conditional on frozen samplers; reused counts not independent new evidence",
              "FCI_used": False, "hardware_shots": 0, "whole_study_complete": False,
              "auditor_sha256": file_digest(__file__)}
    save_json(output/"validation.json", result); return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(audit(p.parse_args().output)))
