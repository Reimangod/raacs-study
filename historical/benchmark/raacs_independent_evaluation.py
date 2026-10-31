"""Conditional independent readout, known-build budgets, strong baseline reuse.

Small<=16q analysis only. The practical large molecular route never imports
or invokes the full-distribution extraction below.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter
import zlib
import hashlib
import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit.quantum_info import Statevector
from scipy.sparse import load_npz
from threadpoolctl import threadpool_limits
from .common import ROOT, digest, file_digest, rng_for, save_json
from .resource_adaptive_sampler import allocation_counts
from .mps_sampling import construct_mps
from .study.core import CAPS, RitzCache, open_db, pack_pairs, ranks, unpack_pairs, rss_peak
from .study.reanalysis import frozen_inventory

NAMESPACE = "RAACS-post-construction-independent-final-v1"
PAIRED = ["B1:source_final_"+m for m in ("ceo_ovp", "ceo_mvp", "qeb", "qubit", "gsd")]+[
    "B1:lucj_ccsd_2_vqe", "B3:TE_t1_steps4", "B3:Pauli_qDRIFT_t1_steps256_s101", "B2exact:M1000-R128-s101"]


def remaining_budget(total, build):
    if build is None: return None, "UNKNOWN_QPU_CONSTRUCTION_COST"
    if not isinstance(build, int) or isinstance(build, bool) or build < 0: raise ValueError("Known nonnegative integer construction shots")
    left = total-build
    return (left, "AVAILABLE") if left > 0 else (None, "TOTAL_BUDGET_INFEASIBLE")


def normalized_probability(p, n):
    p = np.asarray(p, float)
    if n > 16 or p.shape != (1 << n,) or not np.isfinite(p).all() or p.min() < -1e-15 or p.sum() <= 0:
        raise ValueError("Qualified small-cohort probability only")
    p = np.maximum(p, 0); return p/p.sum()


def raacs_distribution(folder):
    """Actual prepared sampler distribution; MPS expansion is <=16q analysis."""
    folder = Path(folder); r = json.loads((folder/"result.json").read_text())
    cfg = json.loads((folder/"config.json").read_text()); n = r["problem"]["n_qubits"]
    if n > 16: raise ValueError("Never expand a large molecular state")
    assert file_digest(folder/"events.sqlite") == r["event_database_sha256"]
    db = sqlite3.connect(f"file:{folder/'events.sqlite'}?mode=ro", uri=True); components = []
    for allocation in r["final_circuit_allocations"]:
        if not allocation["shots"]: continue
        raw = db.execute("SELECT sampling_qpy FROM circuits WHERE id=?", (allocation["circuit_id"],)).fetchone()[0]
        qc = qpy.load(BytesIO(zlib.decompress(raw)))[0]
        if cfg["sampling"]["backend"] == "mps":
            spec = cfg["sampling"]
            state, meta = construct_mps(qc, spec["max_bond"], spec["cutoff"], spec["construction_seconds_cap"], spec["RSS_cap_bytes"])
            psi = state.psi.copy(); psi.normalize(); p = abs(np.asarray(psi.to_dense()).reshape(-1))**2
        else: p = abs(Statevector.from_instruction(qc).data)**2
        components.append((normalized_probability(p, n), allocation["shots"]))
    db.close()
    return components, r


def distribution_id(problem_id, components):
    total = sum(w for _, w in components)
    return digest([problem_id, [[hashlib.sha256(p.astype("<f8").tobytes()).hexdigest(), w/total] for p, w in components]])


def draw_components(components, M, identity, rep):
    allocations = allocation_counts(M, [w for _, w in components]); merged = {}
    for j, ((p, _), shots) in enumerate(zip(components, allocations)):
        if not shots: continue
        rng, _ = rng_for(NAMESPACE, identity, M, rep, j)
        counts = rng.multinomial(shots, p)
        for x in np.flatnonzero(counts): merged[int(x)] = merged.get(int(x), 0)+int(counts[x])
    ids = np.array(sorted(merged), dtype=np.int64); ns = np.array([merged[int(x)] for x in ids])
    assert int(ns.sum()) == M
    return ids, ns


def baseline_bank(source, case):
    """Retain all saved methods; FCI is never a method-selection criterion."""
    bank = []
    roots = [("B1", ROOT/"results/sampler-baselines-B1-standard-v1"),
             ("B3", ROOT/"results/nonvariational-B3-standard-v1"),
             ("B2exact", ROOT/"results/sampler-baselines-B2-exactprep-standard-v2/independent-common-HF-readouts-v2")]
    for cohort, root in roots:
        paths = sorted((root/"cells"/case).glob("*/result.json"))
        for path in paths:
            r = json.loads(path.read_text()); method = path.parent.name
            build = r.get("construction_simulated_raw_shots", r.get("simulated_construction_shots"))
            if r.get("spec", {}).get("objective") == "vqe": build = None
            if build is not None: build = int(build)
            bank.append({"method": cohort+":"+method, "result": r, "distribution": path.parent/"final.npz",
                         "result_path": path, "build": build, "reuse_db": root/"readouts"/case/"readouts.sqlite",
                         "reuse_method": method, "failed": r["status"] == "FAILED_PRESERVED"})
    for folder in sorted((source/"checkpoints"/case).iterdir()):
        paths = sorted(folder.glob("step-*.json"))
        if not paths: continue
        cp = json.loads(paths[-1].read_text()); method = "source_final_"+folder.name
        # Native growth QPU objective/pool costs have no certified shot conversion.
        build = 0 if folder.name in ("hf", "lucj_1", "lucj_2", "lucj_truncated", "lucj_random_101", "ceo_random_101") else None
        bank.append({"method": "B1:"+method, "result": cp, "distribution": Path(cp["distribution"]),
                     "result_path": paths[-1], "build": build, "reuse_db": roots[0][1]/"readouts"/case/"readouts.sqlite",
                     "reuse_method": method, "failed": False})
    assert len({b["method"] for b in bank}) == len(bank)
    return bank


def manifest(construction, mode):
    source = Path(json.loads((construction/"manifest.json").read_text())["source"])
    return {"schema": "RAACS-postconstruction-independent-comparison-v1", "mode": mode,
            "construction": str(construction.resolve()), "source": str(source),
            "cases": ["H2_R0.7", "H4_R1.1"] if mode == "quick" else sorted(p.name for p in (source/"references").iterdir() if p.is_dir()),
            "M": [1000, 10000, 100000], "R": list(CAPS), "repeats": 4 if mode == "quick" else 20,
            "paired_baselines": PAIRED, "readout_scopes": ["readout_only", "known_build_plus_readout"],
            "construction_manifest_sha256": file_digest(construction/"manifest.json"),
            "implementation": {p: file_digest(ROOT/p) for p in ("benchmark/raacs_independent_evaluation.py", "benchmark/study/core.py", "benchmark/resource_adaptive_sampler.py")},
            "spec_sha256": file_digest(ROOT/"study-v1/RAACS_INDEPENDENT_COMPARISON.md"),
            "post_construction_exploratory": True, "FCI_used": False, "hardware_shots": 0,
            "free_floor_bytes": 3*2**30, "output_boundary_bytes": 640*2**20}


def evaluate(construction, output, mode):
    construction, output = Path(construction).resolve(), Path(output).resolve()
    if construction == output or construction in output.parents: raise ValueError("Use independent versioned analysis root")
    source_root = Path(json.loads((construction/"manifest.json").read_text())["source"]).resolve()
    if source_root == output or source_root in output.parents or (output/"REPRODUCIBILITY.json").exists():
        raise ValueError("Protect sealed benchmark inputs")
    output.mkdir(parents=True, exist_ok=True); cfg = manifest(construction, mode); mf = output/"manifest.json"
    if mf.exists() and json.loads(mf.read_text()) != cfg: raise RuntimeError("Independent comparison binding drift")
    if mode == "standard":
        q = ROOT/"results/RAACS-independent-comparison-quick-v1/validation.json"
        if not q.exists() or json.loads(q.read_text())["status"] != "RAACS_INDEPENDENT_COMPARISON_COUNTS_AND_PAIRS_VALIDATED":
            raise RuntimeError("Actual quick final readout audit required")
    if not mf.exists() and shutil.disk_usage(output).free < cfg["free_floor_bytes"]+cfg["output_boundary_bytes"]:
        raise RuntimeError("Insufficient bounded comparison capacity; no data deletion")
    save_json(mf, cfg); source = Path(cfg["source"]); frozen_inventory(source, output)
    case_results = []
    with threadpool_limits(limits=1):
        for case in cfg["cases"]:
            if shutil.disk_usage(output).free < cfg["free_floor_bytes"]: raise RuntimeError("Comparison freefloor; preserve prefix")
            if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > cfg["output_boundary_bytes"]:
                raise RuntimeError("Comparison output boundary; preserve prefix")
            refpath = source/"references"/case; meta = json.loads((refpath/"reference.json").read_text())
            with np.load(refpath/"reference.npz") as z: valid_ids = z["ids"].copy()
            h = load_npz(refpath/"hamiltonian.npz")
            db = open_db(output/"readouts"/case/"readouts.sqlite"); cache = RitzCache(h, valid_ids, meta["request_digest"], db)
            db.executescript("""
              CREATE TABLE IF NOT EXISTS cells(method TEXT PRIMARY KEY,kind TEXT,distribution_id TEXT,build_shots INTEGER,metadata TEXT);
              CREATE TABLE IF NOT EXISTS links(method TEXT,total_M INTEGER,scope TEXT,rep INTEGER,draw INTEGER,status TEXT,
                PRIMARY KEY(method,total_M,scope,rep));
              CREATE TABLE IF NOT EXISTS origins(draw INTEGER PRIMARY KEY,kind TEXT,metadata TEXT);
              CREATE TABLE IF NOT EXISTS pairs(method TEXT,total_M INTEGER,scope TEXT,rep INTEGER,cap INTEGER,baseline TEXT,
                actual_d INTEGER,RAACS_subspace INTEGER,baseline_subspace INTEGER,PRIMARY KEY(method,total_M,scope,rep,cap,baseline));
            """)
            items = {}; components_by_key = {}; distributions = {}
            for path in sorted((construction/"cells"/case).glob("*/result.json")):
                components, result = raacs_distribution(path.parent); method = "RAACS:"+path.parent.name
                key = distribution_id(meta["request_digest"], components)
                items[method] = {"kind": "RAACS", "key": key, "build": int(result["pilot_raw_model_shots"]),
                                 "metadata": {"result_path": str(path), "result_sha256": file_digest(path), "construction_wall_seconds": result["actual_wall_seconds"],
                                              "conditioning": "one saved construction, not a new training replication"}}
                components_by_key[key] = components
            bank = baseline_bank(source, case); source_rows = []
            for b in bank:
                source_rows.append({"method": b["method"], "failed": b["failed"], "result_path": str(b["result_path"]), "sha256": file_digest(b["result_path"])})
                if b["failed"]: continue
                expected = b["result"].get("final_sha256", b["result"].get("distribution_sha256"))
                assert file_digest(b["distribution"]) == expected
                with np.load(b["distribution"]) as z: p = normalized_probability(z["probability"], meta["n_qubits"])
                components = [(p, 1)]; key = distribution_id(meta["request_digest"], components)
                items[b["method"]] = {"kind": "BASELINE", "key": key, "build": b["build"], "baseline": b,
                    "metadata": {"result_path": str(b["result_path"]), "result_sha256": file_digest(b["result_path"]), "construction_shots": b["build"],
                                 "source_distribution_sha256": expected}}
                components_by_key[key] = components
            save_json(output/"readouts"/case/"source-records.json", source_rows)
            for method, item in items.items():
                key = item["key"]
                db.execute("INSERT OR IGNORE INTO checkpoints(sha,method,step,primary_record,metadata) VALUES(?,?,?,?,?)", (key, method, 0, 1, "{}"))
                db.execute("INSERT OR IGNORE INTO cells VALUES(?,?,?,?,?)", (method, item["kind"], key, item["build"], json.dumps(item["metadata"])))
            cp_map = dict(db.execute("SELECT sha,id FROM checkpoints")); connections = {}
            def get_draw(method, shots, rep):
                item = items[method]; key = item["key"]; cp = cp_map[key]
                old = db.execute("SELECT id,counts FROM draws WHERE cp=? AND shots=? AND rep=?", (cp, shots, rep)).fetchone()
                if old:
                    dr, blob = old; ids, ns = unpack_pairs(blob)
                else:
                    before = perf_counter(); origin = {"kind": "NEW_INDEPENDENT_COMPONENT_DRAWS", "namespace": NAMESPACE, "distribution_id": key}
                    reused = None
                    if item["kind"] == "BASELINE":
                        b = item["baseline"]; path = b["reuse_db"]
                        if path.exists():
                            if str(path) not in connections:
                                connections[str(path)] = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                            srcdb = connections[str(path)]
                            reused = srcdb.execute("SELECT d.id,d.counts,d.seed FROM draws d JOIN checkpoints c ON c.id=d.cp WHERE c.method=? AND d.shots=? AND d.rep=?", (b["reuse_method"], shots, rep)).fetchone()
                            if reused:
                                origin = {"kind": "EXACT_EXISTING_BASELINE_ACQUISITION_REUSE", "source_db": str(path), "source_draw": reused[0],
                                          "source_counts_sha256": hashlib.sha256(reused[1]).hexdigest(), "original_seed": reused[2]}
                    if reused: ids, ns = unpack_pairs(reused[1]); seed = reused[2]
                    else:
                        ids, ns = draw_components(components_by_key[key], shots, key, rep)
                        seed = str(rng_for(NAMESPACE, key, shots, rep)[1])
                    dr = db.execute("INSERT INTO draws(cp,shots,rep,seed,counts,seconds) VALUES(?,?,?,?,?,?)", (cp, shots, rep, seed, pack_pairs(ids, ns), perf_counter()-before)).lastrowid
                    db.execute("INSERT INTO origins VALUES(?,?,?)", (dr, origin["kind"], json.dumps(origin)))
                order = ranks(ids, ns, valid_ids, meta["hf_id"], [key, shots, rep])["frequency"]
                for R in cfg["R"]:
                    if db.execute("SELECT 1 FROM outcomes WHERE draw=? AND cap=?", (dr, R)).fetchone(): continue
                    sid, _, hit, sec = cache.solve(order[:R])
                    db.execute("INSERT INTO outcomes VALUES(?,?,?,?,?,?)", (dr, R, "frequency", sid, int(hit), sec))
                return dr, order
            for total_M in cfg["M"]:
                for rep in range(cfg["repeats"]):
                    for scope in cfg["readout_scopes"]:
                        orders = {}; draws = {}
                        for method, item in items.items():
                            remaining, reason = (total_M, "AVAILABLE") if scope == "readout_only" else remaining_budget(total_M, item["build"])
                            if remaining is None:
                                db.execute("INSERT OR IGNORE INTO links VALUES(?,?,?,?,?,?)", (method, total_M, scope, rep, None, reason)); continue
                            dr, order = get_draw(method, remaining, rep); orders[method] = order; draws[method] = dr
                            db.execute("INSERT OR IGNORE INTO links VALUES(?,?,?,?,?,?)", (method, total_M, scope, rep, dr, "AVAILABLE"))
                        for method, order in orders.items():
                            if items[method]["kind"] != "RAACS": continue
                            for baseline in cfg["paired_baselines"]:
                                if baseline not in orders: continue
                                for R in cfg["R"]:
                                    d = min(R, len(order), len(orders[baseline])); a = cache.solve(order[:d])[0]; b = cache.solve(orders[baseline][:d])[0]
                                    db.execute("INSERT OR IGNORE INTO pairs VALUES(?,?,?,?,?,?,?,?,?)", (method, total_M, scope, rep, R, baseline, d, a, b))
                        db.commit()
            for connection in connections.values(): connection.close()
            n, samples = db.execute("SELECT count(*),sum(shots) FROM draws").fetchone()
            fresh_n, fresh_samples = db.execute("SELECT count(*),sum(d.shots) FROM origins o JOIN draws d ON d.id=o.draw WHERE o.kind='NEW_INDEPENDENT_COMPONENT_DRAWS'").fetchone()
            frame = pd.read_sql_query("""SELECT l.method,c.kind,l.total_M,l.scope,l.rep,c.build_shots,d.shots AS final_shots,
                    o.cap,s.energy,s.dimension FROM links l JOIN cells c ON c.method=l.method JOIN draws d ON d.id=l.draw
                    JOIN outcomes o ON o.draw=l.draw JOIN subspaces s ON s.id=o.subspace""", db)
            frame.to_parquet(output/"readouts"/case/"individual-readouts.parquet", index=False)
            pd.read_sql_query("SELECT * FROM links WHERE status!='AVAILABLE'", db).to_csv(output/"readouts"/case/"budget-missingness.csv", index=False)
            row = {"case": case, "cells": len(items), "draws": n, "all_linked_model_shots": samples,
                   "new_independent_draws": fresh_n, "new_model_shots": fresh_samples or 0,
                   "readout_rows": len(frame), "actual_d_pairs": db.execute("SELECT count(*) FROM pairs").fetchone()[0]}
            db.close(); case_results.append(row)
            save_json(output/"execution-status.json", {"status": "RUNNING", "finished_cases": len(case_results), "planned_cases": len(cfg["cases"]), "rows": case_results})
            print("RAACS conditional readout "+case, flush=True)
    result = {"status": "RAACS_INDEPENDENT_COMPARISON_EXECUTED_PENDING_AUDIT", "cases": case_results,
              "hardware_shots": 0, "FCI_used": False, "whole_study_complete": False}
    save_json(output/"execution-status.json", result); return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="quick")
    p.add_argument("--construction"); p.add_argument("--output"); args = p.parse_args()
    construction = args.construction or ROOT/f"results/RAACS-{'quick' if args.mode=='quick' else 'standard'}-v1"
    output = args.output or ROOT/f"results/RAACS-independent-comparison-{args.mode}-v1"
    print(json.dumps(evaluate(construction, output, args.mode)))
