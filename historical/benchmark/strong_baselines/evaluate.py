"""Independent readout of new B1 samplers and source final checkpoints."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd

from ..chemistry import load_reference
from ..common import digest, file_digest, rng_for, save_json
from ..study.core import RitzCache, open_db, pack_pairs, ranks, unpack_pairs


def evaluate(root):
    root = Path(root); config = json.loads((root / "manifest.json").read_text()); budget = config["budget"]
    source = Path(config["source"]); summary = []; coverage = []
    for case in config["case_ids"]:
        ref = load_reference(source / "references" / case)
        samplers = []
        for name in config["methods"]:
            result_file = root / "cells" / case / name["name"] / "result.json"
            if not result_file.exists():
                raise RuntimeError("Missing registered B1 terminal cell: " + str(result_file))
            result = json.loads(result_file.read_text())
            coverage.append({"case_id": case, "method": name["name"], "status": result["status"],
                             "objective_evaluations": result.get("objective_evaluations", 0),
                             "construction_wall_seconds": result["construction_wall_seconds"],
                             "error": result.get("error"), "energy_error_mHa": result.get("energy_error_mHa")})
            if result["status"] == "FAILED_PRESERVED":
                continue
            dist = result_file.parent / "final.npz"
            if file_digest(dist) != result["final_sha256"]:
                raise RuntimeError("New sampler distribution mismatch")
            with np.load(dist) as z:
                p = z["probability"]
            samplers.append((name["name"], result["task_id"], p))
        for method in sorted((source / "checkpoints" / case).iterdir()):
            cps = sorted(method.glob("step-*.json"))
            if not cps:
                continue
            cp = json.loads(cps[-1].read_text())
            if file_digest(cp["distribution"]) != cp["distribution_sha256"]:
                raise RuntimeError("Source sampler distribution mismatch")
            with np.load(cp["distribution"]) as z:
                samplers.append(("source_final_"+method.name, cp["checkpoint_id"], z["probability"]))
        directory = root / "readouts" / case; directory.mkdir(parents=True, exist_ok=True)
        db = open_db(directory / "readouts.sqlite")
        cache = RitzCache(ref["h"], ref["ids"], ref["meta"]["request_digest"], db)
        db.execute("CREATE TABLE IF NOT EXISTS common_readouts (method TEXT, shots INTEGER, rep INTEGER, cap INTEGER, common_d INTEGER, subspace INTEGER, PRIMARY KEY(method,shots,rep,cap))")
        db.execute("CREATE TABLE IF NOT EXISTS pair_readouts (reference_method TEXT,method TEXT,shots INTEGER,rep INTEGER,cap INTEGER,common_d INTEGER,reference_subspace INTEGER,subspace INTEGER,PRIMARY KEY(reference_method,method,shots,rep,cap))")
        cp_index = {}
        for method, key, probability in samplers:
            db.execute("INSERT OR IGNORE INTO checkpoints(sha,method,step,primary_record,metadata) VALUES(?,?,?,?,?)", (key, method, 0, 1, "{}"))
            cp_index[method] = db.execute("SELECT id FROM checkpoints WHERE sha=?", (key,)).fetchone()[0]
        for M in budget["M_grid"]:
            for rep in range(budget["evaluation_repeats"]):
                orders = {}
                for method, key, probability in samplers:
                    cp_id = cp_index[method]
                    rng, seed = rng_for("B1-independent-final-evaluation-v1", case, method, key, M, rep)
                    old_draw = db.execute("SELECT counts FROM draws WHERE cp=? AND shots=? AND rep=?", (cp_id, M, rep)).fetchone()
                    started = perf_counter()
                    if old_draw:
                        ids, ns = unpack_pairs(old_draw[0])
                    else:
                        counts = rng.multinomial(M, probability/probability.sum())
                        valid = counts > 0; ids = np.flatnonzero(valid); ns = counts[valid]
                    draw_seconds = perf_counter()-started
                    order = ranks(ids, ns, ref["ids"], ref["meta"]["hf_id"], [case, method, M, rep])["frequency"]
                    orders[method] = order
                    db.execute("INSERT OR IGNORE INTO draws(cp,shots,rep,seed,counts,seconds) VALUES(?,?,?,?,?,?)", (cp_id, M, rep, str(seed), pack_pairs(ids, ns), draw_seconds))
                    dr = db.execute("SELECT id FROM draws WHERE cp=? AND shots=? AND rep=?", (cp_id, M, rep)).fetchone()[0]
                    for R in budget["R_grid"]:
                        subspace, energy, hit, seconds = cache.solve(order[:R])
                        db.execute("INSERT OR IGNORE INTO outcomes VALUES(?,?,?,?,?,?)", (dr, R, "frequency", subspace, int(hit), seconds))
                for R in budget["R_grid"]:
                    d = min(min(R, len(v)) for v in orders.values())
                    for method, order in orders.items():
                        sid, _, _, _ = cache.solve(order[:d])
                        db.execute("INSERT OR IGNORE INTO common_readouts VALUES(?,?,?,?,?,?)", (method, M, rep, R, d, sid))
                    for reference_method in ("source_final_ceo_ovp", "source_final_ceo_mvp"):
                        if reference_method not in orders:
                            continue
                        for method, order in orders.items():
                            pdim = min(R, len(order), len(orders[reference_method]))
                            s0, _, _, _ = cache.solve(orders[reference_method][:pdim])
                            s1, _, _, _ = cache.solve(order[:pdim])
                            db.execute("INSERT OR IGNORE INTO pair_readouts VALUES(?,?,?,?,?,?,?,?)", (reference_method, method, M, rep, R, pdim, s0, s1))
                db.commit()
        frame = pd.read_sql_query("SELECT c.method,d.shots,d.rep,o.cap,s.energy,s.dimension,s.residual FROM outcomes o JOIN draws d ON d.id=o.draw JOIN checkpoints c ON c.id=d.cp JOIN subspaces s ON s.id=o.subspace", db)
        matched = pd.read_sql_query("SELECT r.method,r.shots,r.rep,r.cap,r.common_d,s.energy FROM common_readouts r JOIN subspaces s ON s.id=r.subspace", db)
        frame["error_mHa"] = (frame.energy-ref["meta"]["fci_energy"])*1000
        frame["chemical_accuracy"] = frame.error_mHa <= 1.5936014376
        for (method, shots, cap), group in frame.groupby(["method", "shots", "cap"]):
            g = matched[(matched.method == method) & (matched.shots == shots) & (matched.cap == cap)]
            summary.append({"case_id": case, "method": method, "shots": int(shots), "R_cap": int(cap),
                            "repeats": len(group), "mean_error_mHa": float(group.error_mHa.mean()),
                            "std_error_mHa": float(group.error_mHa.std(ddof=1)), "mean_actual_d": float(group.dimension.mean()),
                            "success_probability_estimate": float(group.chemical_accuracy.mean()),
                            "mean_common_d_all_available_samplers": float(g.common_d.mean()),
                            "mean_error_matched_d_mHa": float((g.energy-ref["meta"]["fci_energy"]).mean()*1000),
                            "matched_scope": "all available samplers including HF-only; may force d=1; also report commonR", "hardware_shots": 0})
        frame.to_parquet(directory / "individual-readouts.parquet", index=False)
        matched.to_parquet(directory / "matched-d-readouts.parquet", index=False)
        pairs = pd.read_sql_query("SELECT r.reference_method,r.method,r.shots,r.rep,r.cap,r.common_d,s0.energy AS reference_energy,s1.energy FROM pair_readouts r JOIN subspaces s0 ON s0.id=r.reference_subspace JOIN subspaces s1 ON s1.id=r.subspace", db)
        pairs.to_parquet(directory / "pairwise-ceo-matched-d.parquet", index=False)
        db.close()
    pd.DataFrame(summary).to_csv(root / "readout-summary.csv", index=False)
    pd.DataFrame(coverage).to_csv(root / "all-cell-status.csv", index=False)
    result = {"status": "B1_INDEPENDENT_READOUT_COMPLETE", "planned_cells": len(config["case_ids"])*len(config["methods"]),
              "terminal_cells": len(coverage), "summary_rows": len(summary), "hardware_shots": 0,
              "FCI_used_in_sampler_selection": False, "full_milestone2_complete": False}
    save_json(root / "evaluation-status.json", result)
    print(json.dumps(result), flush=True)
    return result
