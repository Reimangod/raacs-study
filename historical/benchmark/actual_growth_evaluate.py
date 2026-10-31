"""Independent final measurements for actual FCI-free construction.

No reference ground-state energy is an input. Pilot support is not inserted
into the primary independent final CI basis. Raw invalid outcomes remain paid.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .common import ROOT, digest, file_digest, rng_for, save_json
from .molecular_problem_no_fci import load_problem
from .online_stopping_rules import remaining_final_budget
from .study.core import RitzCache, open_db, pack_pairs, ranks, unpack_pairs

NAMESPACE = "PhaseC-actual-growth-independent-final-v1"


def evaluate(root):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text())
    valid = json.loads((root/"validation.json").read_text())
    assert valid["status"] == "ACTUAL_GROWTH_PREFIX_AND_PHYSICS_VALIDATED"
    binding = {"evaluator_sha256": file_digest(__file__), "manifest_sha256": file_digest(root/"manifest.json"),
               "validation_sha256": file_digest(root/"validation.json"), "namespace": NAMESPACE,
               "key": "mode / physical checkpoint / remaining final M / rep; mode separates quick/standard final draws",
               "Ritz_engine_sha256": file_digest(ROOT/"benchmark/study/core.py"),
               "all_pilot_shots_paid": True, "pilot_support_in_primary_final_CI": False, "FCI_used": False}
    path = root/"evaluation-binding.json"
    if path.exists() and json.loads(path.read_text()) != binding: raise RuntimeError("Final evaluation binding drift")
    if cfg["mode"] == "standard":
        q = ROOT/"results/PhaseC-actual-growth-quick-v1/evaluation-validation.json"
        if not q.exists() or json.loads(q.read_text())["status"] != "ACTUAL_FINAL_COUNTS_AND_PAIRS_VALIDATED":
            raise RuntimeError("Quick final measurement audit required")
    save_json(path, binding); results = []
    with threadpool_limits(limits=1):
        for item in cfg["physical_problems"]:
            case = item["case_id"]; physical = load_problem(root/"problems"/case); meta = physical["meta"]
            db = open_db(root/"readouts"/case/"readouts.sqlite")
            db.executescript("""
              CREATE TABLE IF NOT EXISTS cell_readouts(
                method TEXT NOT NULL,policy TEXT NOT NULL,total_M INTEGER NOT NULL,rep INTEGER NOT NULL,
                draw INTEGER NOT NULL,native_draw INTEGER NOT NULL,pilot_shots INTEGER NOT NULL,
                PRIMARY KEY(method,policy,total_M,rep));
              CREATE TABLE IF NOT EXISTS actual_d_pairs(
                method TEXT NOT NULL,policy TEXT NOT NULL,total_M INTEGER NOT NULL,rep INTEGER NOT NULL,
                cap INTEGER NOT NULL,d INTEGER NOT NULL,policy_subspace INTEGER NOT NULL,native_subspace INTEGER NOT NULL,
                PRIMARY KEY(method,policy,total_M,rep,cap));
            """)
            cache = RitzCache(physical["h"], physical["ids"], meta["problem_id"], db)
            cells = {}; probabilities = {}
            for method in cfg["methods"]:
                for policy in cfg["policies"]:
                    folder = root/"cells"/case/method/policy
                    r = json.loads((folder/"result.json").read_text()); key = r["physical_checkpoint_id"]
                    assert file_digest(folder/"final.npz") == r["artifact_sha256"]["final.npz"]
                    with np.load(folder/"final.npz") as z: p = z["probability"].copy()
                    if key in probabilities:
                        assert np.array_equal(probabilities[key], p), "Physical checkpoint ID collision"
                    else: probabilities[key] = p
                    db.execute("INSERT OR IGNORE INTO checkpoints(sha,method,step,primary_record,metadata) VALUES(?,?,?,?,?)",
                               (key, method, r["last_step"], 1, json.dumps({"physical_checkpoint_id": key, "problem_id": meta["problem_id"]})))
                    cells[(method, policy)] = r
            cp_ids = dict(db.execute("SELECT sha,id FROM checkpoints"))

            def draw_final(r, M, rep):
                key = r["physical_checkpoint_id"]; cp = cp_ids[key]
                row = db.execute("SELECT id,counts,seed FROM draws WHERE cp=? AND shots=? AND rep=?", (cp, M, rep)).fetchone()
                rng, seed = rng_for(NAMESPACE, cfg["mode"], key, M, rep)
                if row:
                    dr, blob, saved_seed = row; assert saved_seed == str(seed); ids, counts = unpack_pairs(blob)
                else:
                    t = perf_counter(); p = probabilities[key]; allcounts = rng.multinomial(M, p/p.sum())
                    ids = np.flatnonzero(allcounts); counts = allcounts[ids]
                    dr = db.execute("INSERT INTO draws(cp,shots,rep,seed,counts,seconds) VALUES(?,?,?,?,?,?)",
                                    (cp, M, rep, str(seed), pack_pairs(ids, counts), perf_counter()-t)).lastrowid
                order = ranks(ids, counts, physical["ids"], meta["hf_id"], [key, M, rep])["frequency"]
                for R in cfg["R_grid"]:
                    if db.execute("SELECT 1 FROM outcomes WHERE draw=? AND cap=?", (dr, R)).fetchone(): continue
                    sid, _, hit, sec = cache.solve(order[:R])
                    db.execute("INSERT INTO outcomes VALUES(?,?,?,?,?,?)", (dr, R, "frequency", sid, int(hit), sec))
                return dr, order

            for method in cfg["methods"]:
                native = cells[(method, "native_terminal")]
                for M in cfg["M_grid"]:
                    for rep in range(cfg["final_evaluation_repeats"]):
                        native_draw, native_order = draw_final(native, M, rep)
                        for policy in cfg["policies"]:
                            r = cells[(method, policy)]; pilot = r["pilot_raw_model_shots"]
                            remaining = remaining_final_budget(M, pilot)
                            dr, order = draw_final(r, remaining, rep)
                            db.execute("INSERT OR IGNORE INTO cell_readouts VALUES(?,?,?,?,?,?,?)", (method, policy, M, rep, dr, native_draw, pilot))
                            for R in cfg["R_grid"]:
                                d = min(R, len(order), len(native_order)); sa = cache.solve(order[:d])[0]; sb = cache.solve(native_order[:d])[0]
                                db.execute("INSERT OR IGNORE INTO actual_d_pairs VALUES(?,?,?,?,?,?,?,?)", (method, policy, M, rep, R, d, sa, sb))
                        db.commit()
            expected = len(cfg["methods"])*len(cfg["policies"])*len(cfg["M_grid"])*cfg["final_evaluation_repeats"]
            assert db.execute("SELECT count(*) FROM cell_readouts").fetchone()[0] == expected
            drs, shots = db.execute("SELECT count(*),sum(shots) FROM draws").fetchone()
            frame = pd.read_sql_query("""SELECT r.method,r.policy,r.total_M,r.rep,r.pilot_shots,d.shots AS final_shots,o.cap,
                    s.energy AS energy_Ha,s.dimension AS actual_d,ns.energy AS native_energy_Ha,
                    p.d AS paired_d,ps.energy AS paired_energy_Ha,pn.energy AS paired_native_energy_Ha
                FROM cell_readouts r JOIN draws d ON d.id=r.draw
                JOIN outcomes o ON o.draw=r.draw JOIN subspaces s ON s.id=o.subspace
                JOIN outcomes no ON no.draw=r.native_draw AND no.cap=o.cap JOIN subspaces ns ON ns.id=no.subspace
                JOIN actual_d_pairs p ON p.method=r.method AND p.policy=r.policy AND p.total_M=r.total_M AND p.rep=r.rep AND p.cap=o.cap
                JOIN subspaces ps ON ps.id=p.policy_subspace JOIN subspaces pn ON pn.id=p.native_subspace""", db)
            frame.to_parquet(root/"readouts"/case/"readouts.parquet", index=False); db.close()
            results.append({"case_id": case, "physical_final_draws": drs, "unique_final_raw_model_shots": shots, "R_readouts": len(frame)})
            print("independent actual-growth final readout "+case, flush=True)
    final = {"status": "ACTUAL_INDEPENDENT_FINAL_READOUT_EXECUTED", "cases": len(results),
             "physical_final_draws": sum(r["physical_final_draws"] for r in results),
             "unique_final_raw_model_shots": sum(r["unique_final_raw_model_shots"] for r in results),
             "policy_R_readouts": sum(r["R_readouts"] for r in results),
             "hardware_shots": 0, "FCI_used": False, "whole_study_complete": False}
    save_json(root/"evaluation-status.json", final)
    return final


def audit(root):
    """Rebuild every raw final draw and selection basis; sample Ritz numerics."""
    import zlib
    from scipy.linalg import eigh
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text())
    bind = json.loads((root/"evaluation-binding.json").read_text())
    assert file_digest(__file__) == bind["evaluator_sha256"]
    draws = shots = pairs = numerical = readouts = 0; max_difference = 0.
    with threadpool_limits(limits=1):
        for item in cfg["physical_problems"]:
            case = item["case_id"]; physical = load_problem(root/"problems"/case); valid = set(map(int, physical["ids"])); hf = physical["meta"]["hf_id"]
            db = sqlite3.connect(f"file:{root/'readouts'/case/'readouts.sqlite'}?mode=ro", uri=True)
            pmap = {}
            for method in cfg["methods"]:
                for policy in cfg["policies"]:
                    folder = root/"cells"/case/method/policy; r = json.loads((folder/"result.json").read_text())
                    with np.load(folder/"final.npz") as z: pmap[r["physical_checkpoint_id"]] = z["probability"].copy()
            keys = dict(db.execute("SELECT id,sha FROM checkpoints")); orders = {}; draw_shots = {}
            for did, cp, M, rep, seed, blob in db.execute("SELECT id,cp,shots,rep,seed,counts FROM draws"):
                key = keys[cp]; rng, sd = rng_for(NAMESPACE, cfg["mode"], key, M, rep)
                p = pmap[key]; fresh = rng.multinomial(M, p/p.sum()); ids, counts = unpack_pairs(blob)
                assert str(sd) == seed and counts.sum() == M
                assert np.array_equal(ids, np.flatnonzero(fresh)) and np.array_equal(counts, fresh[ids])
                orders[did] = [hf]+[int(x) for x, n in sorted(zip(ids, counts), key=lambda a: (-a[1], a[0])) if int(x) in valid and int(x) != hf]
                draw_shots[did] = M; draws += 1; shots += M
            spaces = {sid: (np.frombuffer(zlib.decompress(raw), dtype="<i4").tolist(), E)
                      for sid, raw, E in db.execute("SELECT id,ids,energy FROM subspaces")}
            for did, R, sid in db.execute("SELECT draw,cap,subspace FROM outcomes"):
                assert spaces[sid][0] == sorted(orders[did][:R]); readouts += 1
            for method, policy, M, rep, dr, native_dr, pilot in db.execute("SELECT * FROM cell_readouts"):
                assert draw_shots[dr]+pilot == M and draw_shots[native_dr] == M
                expected_pilot = json.loads((root/"cells"/case/method/policy/"result.json").read_text())["pilot_raw_model_shots"]
                assert pilot == expected_pilot
                matched = list(db.execute("SELECT cap,d,policy_subspace,native_subspace FROM actual_d_pairs WHERE method=? AND policy=? AND total_M=? AND rep=?", (method, policy, M, rep)))
                assert len(matched) == len(cfg["R_grid"])
                for R, d, sa, sb in matched:
                    assert d == min(R, len(orders[dr]), len(orders[native_dr]))
                    assert spaces[sa][0] == sorted(orders[dr][:d]) and spaces[sb][0] == sorted(orders[native_dr][:d]); pairs += 1
            keys_sorted = sorted(spaces)
            for i in sorted(set(np.linspace(0, len(keys_sorted)-1, min(5, len(keys_sorted)), dtype=int))):
                ids, energy = spaces[keys_sorted[i]]; hs = physical["h"][ids][:, ids].toarray()
                E = float(eigh(hs, subset_by_index=(0, 0), eigvals_only=True)[0]); difference = abs(E-energy)
                assert difference < 1e-8; max_difference = max(max_difference, difference); numerical += 1
            db.close()
    result = {"status": "ACTUAL_FINAL_COUNTS_AND_PAIRS_VALIDATED", "all_final_draws_rebuilt": draws,
              "unique_final_raw_model_shots": shots, "all_common_R_bases_verified": readouts,
              "all_actual_d_pairs_verified": pairs, "fresh_independent_Ritz_solves": numerical,
              "max_fresh_Ritz_difference_Ha": max_difference, "FCI_used": False, "hardware_shots": 0,
              "whole_study_complete": False, "auditor_sha256": file_digest(__file__)}
    save_json(root/"evaluation-validation.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); p.add_argument("--audit", action="store_true")
    a = p.parse_args(); print(json.dumps(audit(a.output) if a.audit else evaluate(a.output)))
