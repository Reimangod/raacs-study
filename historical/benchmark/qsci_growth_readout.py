"""Independent common/native final readouts after B2 growth; no training queries."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import beta

from .common import ROOT, digest, file_digest, rng_for, save_json
from .chemistry import load_reference
from .qsci_growth import restore_state, selected_subspace
from .study.core import RitzCache, open_db, pack_pairs, unpack_pairs


def samplers(case, root, source):
    ref = load_reference(source / "references" / case)
    records = []
    coverage = []
    trajectories = sorted((root / "trajectories" / case).glob("*/*"))
    mf = json.loads((root / "manifest.json").read_text())
    expected = 2 if mf["mode"] == "quick" else 24
    if len(trajectories) != expected:
        raise RuntimeError(f"Growth cohort incomplete: {case} {len(trajectories)}/{expected}")
    for d in trajectories:
        name = "/".join(d.parts[-2:])
        if (d / "failure.json").exists():
            row = json.loads((d / "failure.json").read_text())
            coverage.append({"case_id": case, "method": name, "status": "FAILED_PRESERVED", "error": row["error"]})
            continue
        terminal = json.loads((d / "terminal.json").read_text())
        cps = sorted(d.glob("step-*.json"))
        if not cps:
            coverage.append({"case_id": case, "method": name, "status": terminal["status"], "error": "NO_SAVED_CI_CHECKPOINT"})
            continue
        cp = json.loads(cps[-1].read_text())
        state = restore_state(ref["meta"]["n_qubits"], ref["meta"]["hf_id"], cp["rotations"])
        row = {"case_id": case, "method": name, "task_id": cp["task_id"], "status": terminal["status"],
               "source_step": cp["step"], "native_qsci_error_mHa": cp["native_qsci_error_mHa"],
               "construction_simulated_raw_shots": terminal["cumulative"]["raw_simulated_shots"],
               "construction_seconds": terminal["wall_seconds_total"], "resources": cp["resources"],
               "native_hf_insertion": cp["native_HF_insertion"], "construction_R": cp["R_cap"],
               "construction_Ns": cp["Ns"], "seed": terminal["seed"], "family": terminal["method"]}
        coverage.append(row)
        records.append({**row, "probability": abs(state)**2})
    for method in sorted((source / "checkpoints" / case).iterdir()):
        cps = sorted(method.glob("step-*.json"))
        if not cps:
            coverage.append({"case_id": case, "method": "source_final_"+method.name, "status": "SOURCE_MISSING_PRESERVED"})
            continue
        cp = json.loads(cps[-1].read_text())
        assert file_digest(cp["distribution"]) == cp["distribution_sha256"]
        with np.load(cp["distribution"]) as z:
            probability = z["probability"]
        records.append({"case_id": case, "method": "source_final_"+method.name, "task_id": cp["checkpoint_id"],
                        "source_step": cp["step"], "resources": cp["resources"], "native_hf_insertion": True,
                        "family": method.name, "construction_simulated_raw_shots": None,
                        "construction_seconds": cp.get("trajectory_seconds"), "probability": probability})
    return ref, records, coverage


def evaluate(root, repeats=None, derivative="independent-readouts-v2"):
    root = Path(root); mf = json.loads((root / "manifest.json").read_text()); source = Path(mf["source"])
    config = {"M_grid": [1000, 10000, 100000], "R_grid": [8, 16, 32, 64, 128, 256, 512],
              "repeats": repeats or (4 if mf["mode"] == "quick" else 20),
              "namespace": "B2-independent-final-evaluation-v1", "native_and_common_views": True}
    directory = root / derivative; directory.mkdir(exist_ok=True)
    bound = {"growth_manifest_sha256": file_digest(root / "manifest.json"), "config": config,
             "implementation_sha256": file_digest(Path(__file__)), "scope": "fixed final checkpoints; no new growth or parameter selection"}
    path = directory / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != bound:
        raise RuntimeError("Readout config/code drift; use successor derivative")
    save_json(path, bound)
    rows = []; all_cells = []; count_total = 0; shot_total = 0
    for case in mf["cases"]:
        ref, records, coverage = samplers(case, root, source); all_cells += coverage
        case_dir = directory / case; case_dir.mkdir(exist_ok=True)
        db = open_db(case_dir / "readouts.sqlite")
        db.execute("CREATE TABLE IF NOT EXISTS paired (reference_method TEXT,method TEXT,shots INTEGER,rep INTEGER,cap INTEGER,common_d INTEGER,reference_subspace INTEGER,subspace INTEGER,PRIMARY KEY(reference_method,method,shots,rep,cap))")
        cache = RitzCache(ref["h"], ref["ids"], ref["meta"]["request_digest"], db)
        index = {}
        for record in records:
            metadata = {k: v for k, v in record.items() if k != "probability"}
            cp_id = digest([record["task_id"], record["method"], record["source_step"]])
            db.execute("INSERT OR IGNORE INTO checkpoints(sha,method,step,primary_record,metadata) VALUES(?,?,?,?,?)",
                       (cp_id, record["method"], record["source_step"], 1, json.dumps(metadata)))
            index[record["method"]] = db.execute("SELECT id FROM checkpoints WHERE sha=?", (cp_id,)).fetchone()[0]
        for M in config["M_grid"]:
            for rep in range(config["repeats"]):
                orders = {}
                for record in records:
                    method = record["method"]; cp = index[method]
                    old = db.execute("SELECT id,counts FROM draws WHERE cp=? AND shots=? AND rep=?", (cp, M, rep)).fetchone()
                    if old:
                        dr, raw = old; ids, counts = unpack_pairs(raw)
                    else:
                        rng, seed = rng_for(config["namespace"], case, record["task_id"], method, M, rep)
                        t = perf_counter(); p = record["probability"]
                        ns = rng.multinomial(M, p/p.sum()); ids = np.flatnonzero(ns); counts = ns[ids]
                        cur = db.execute("INSERT INTO draws(cp,shots,rep,seed,counts,seconds) VALUES(?,?,?,?,?,?)",
                                         (cp, M, rep, str(seed), pack_pairs(ids, counts), perf_counter()-t))
                        dr = cur.lastrowid
                    common = selected_subspace(ids, counts, ref["ids"], ref["meta"]["hf_id"], 512, True)
                    orders[method] = common
                    for R in config["R_grid"]:
                        for view, insert_hf in (("common_HF_inserted", True), ("native", record["native_hf_insertion"])):
                            selected = selected_subspace(ids, counts, ref["ids"], ref["meta"]["hf_id"], R, insert_hf)
                            if not len(selected):
                                # Record explicit unprojectable draw rather than a fictitious energy.
                                db.execute("CREATE TABLE IF NOT EXISTS empty_native(draw INTEGER,cap INTEGER,PRIMARY KEY(draw,cap))")
                                db.execute("INSERT OR IGNORE INTO empty_native VALUES(?,?)", (dr, R))
                                continue
                            sid, _, hit, seconds = cache.solve(selected)
                            db.execute("INSERT OR IGNORE INTO outcomes VALUES(?,?,?,?,?,?)", (dr, R, view, sid, int(hit), seconds))
                for R in config["R_grid"]:
                    pairs = [(refname, method) for refname in ("source_final_ceo_ovp", "source_final_ceo_mvp")
                             if refname in orders for method in orders]
                    for method in orders:
                        if method.startswith("hf_separated_adapt_qsci_MP_m0_2/"):
                            original = "adapt_qsci_original/"+method.split("/", 1)[1]
                            if original in orders:
                                pairs.append((original, method))
                    for a, b in pairs:
                        d = min(R, len(orders[a]), len(orders[b]))
                        s0, _, _, _ = cache.solve(orders[a][:d]); s1, _, _, _ = cache.solve(orders[b][:d])
                        db.execute("INSERT OR IGNORE INTO paired VALUES(?,?,?,?,?,?,?,?)", (a, b, M, rep, R, d, s0, s1))
                db.commit()
        frame = pd.read_sql_query("SELECT c.method,d.shots,d.rep,o.cap AS R_cap,o.rule AS view,s.energy,s.dimension,s.residual FROM outcomes o JOIN draws d ON d.id=o.draw JOIN checkpoints c ON c.id=d.cp JOIN subspaces s ON s.id=o.subspace", db)
        frame["error_mHa"] = (frame.energy-ref["meta"]["fci_energy"])*1000
        frame.to_parquet(case_dir / "individual-readouts.parquet", index=False)
        groups = frame.groupby(["method", "shots", "R_cap", "view"])
        from itertools import product
        for method, M, R, view in product(index, config["M_grid"], config["R_grid"], ("common_HF_inserted", "native")):
            key = (method, M, R, view)
            g = groups.get_group(key) if key in groups.groups else frame.iloc[:0]
            n = config["repeats"]; k = int((g.error_mHa <= 1.5936014376).sum())
            rows.append({"case_id": case, "method": method, "shots": int(M), "R_cap": int(R), "view": view,
                         "valid_energy_repeats": len(g), "planned_repeats": n, "missing_energy_repeats": n-len(g),
                         "conditional_mean_error_mHa": float(g.error_mHa.mean()) if len(g) else None,
                         "std_error_mHa": float(g.error_mHa.std(ddof=1)) if len(g) > 1 else None,
                         "mean_actual_d": float(g.dimension.mean()) if len(g) else None, "unconditional_success_probability": k/n,
                         "success_ci90_low": float(beta.ppf(.05, k, n-k+1)) if k else 0.,
                         "success_ci90_high": float(beta.ppf(.95, k+1, n-k)) if k < n else 1.,
                         "scope": "independent fixed-final readout; native missing draws not called successful"})
        pair = pd.read_sql_query("SELECT r.reference_method,r.method,r.shots,r.rep,r.cap AS R_cap,r.common_d,s0.energy AS reference_energy,s1.energy FROM paired r JOIN subspaces s0 ON s0.id=r.reference_subspace JOIN subspaces s1 ON s1.id=r.subspace", db)
        pair.to_parquet(case_dir / "pairwise-matched-d.parquet", index=False)
        n, shots = db.execute("SELECT count(*),coalesce(sum(shots),0) FROM draws").fetchone()
        count_total += n; shot_total += shots; db.close()
        print(f"B2 independent readout {case}: {n} draws", flush=True)
    pd.DataFrame(rows).to_csv(directory / "summary.csv", index=False)
    pd.DataFrame([{**r, "resources": json.dumps(r.get("resources"))} for r in all_cells]).to_csv(directory / "all-growth-cells.csv", index=False)
    result = {"status": "INDEPENDENT_FINAL_READOUT_COMPLETE", "draws": count_total, "simulated_readout_raw_shots": int(shot_total),
              "hardware_shots": 0, "summary_rows": len(rows), "full_milestone2_complete": False, "whole_study_complete": False}
    save_json(directory / "status.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(evaluate(p.parse_args().output)))
