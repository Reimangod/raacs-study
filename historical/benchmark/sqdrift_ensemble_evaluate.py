"""Fixed-allocation member readouts; never sample the ensemble's mean p.

Existing B1 controls are referenced/copied as the SAME statistical draws, not
rerun, independent replications or newly acquired quantum measurements.
"""
from __future__ import annotations
import argparse
import gzip
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from time import perf_counter, time_ns
import numpy as np
import pandas as pd
from qiskit import qpy
from threadpoolctl import threadpool_limits
from .common import ROOT, digest, file_digest, rng_for, save_json
from .chemistry import load_reference
from .study.core import RitzCache, open_db, pack_pairs, unpack_pairs, ranks
from .sqdrift_ensemble import boundary


def allocate(total, members):
    if total < 0 or members < 1:
        raise ValueError("Nonnegative total and nonempty ensemble required")
    q, rem = divmod(int(total), int(members))
    return [q+int(i < rem) for i in range(members)]


def draw_members(probabilities, allocation, identity, member_ids):
    if len(probabilities) != len(allocation) or len(member_ids) != len(allocation):
        raise ValueError("Member count mismatch")
    total = np.zeros(len(probabilities[0]), dtype=np.int64); records = []
    for p, shots, member_id in zip(probabilities, allocation, member_ids):
        p = np.asarray(p, float)
        if p.shape != total.shape or np.min(p) < -1e-14 or abs(float(p.sum())-1) > 1e-8:
            raise ValueError("Invalid physical member probability")
        rng, seed = rng_for("native-SqDRIFT-member-readout-v1", identity, member_id)
        start = perf_counter(); counts = rng.multinomial(shots, np.maximum(p, 0)/p.sum())
        seconds = perf_counter()-start; ids = np.flatnonzero(counts)
        records.append({"member_id": member_id, "allocated_shots": shots, "seed": str(seed),
                        "ids": ids, "counts": counts[ids], "seconds": seconds})
        total += counts
    ids = np.flatnonzero(total)
    return ids, total[ids], records


def setting_id(path):
    circuit = qpy.load(BytesIO(gzip.decompress(Path(path).read_bytes())))[0]
    ops = [[inst.operation.name, [circuit.find_bit(q).index for q in inst.qubits],
            [float(p) for p in inst.operation.params]] for inst in circuit.data]
    return digest([circuit.num_qubits, ops])  # Global phase is not a sampling setting.


def load_members(root, case, result):
    records, probabilities = [], []
    for mid in result["member_ids"]:
        directory = root / "native" / case / "members" / mid
        member = json.loads((directory / "result.json").read_text())
        if member["status"] != "ACTUAL_SDK_MEMBER_PARITY_VALIDATED":
            raise RuntimeError("Cannot silently omit a failed ensemble member")
        if file_digest(directory / "probability.npz") != member["probability_sha256"]:
            raise RuntimeError("Probability digest mismatch")
        with np.load(directory / "probability.npz") as z:
            probabilities.append(z["probability"])
        records.append({**member, "setting_id": setting_id(directory / "sampling.qpy.gz")})
    return records, probabilities


def evaluate(root, controls):
    root, controls = Path(root).resolve(), Path(controls).resolve()
    config = json.loads((root / "manifest.json").read_text()); budget = config["budget"]
    source = Path(config["source"]); summaries = []
    bound = {"cohort_manifest_sha256": file_digest(root / "manifest.json"), "code_sha256": file_digest(__file__),
             "controls_dataset_id": json.loads((controls / "REPRODUCIBILITY.json").read_text())["dataset_id"],
             "allocation": config["allocation"], "draw_namespace": "native-SqDRIFT-member-readout-v1",
             "control_counts": "exact B1 draw reuse; not new independent acquisitions", "hardware_shots": 0}
    binding = root / "ensemble-evaluation-binding.json"
    if binding.exists() and json.loads(binding.read_text()) != bound:
        raise RuntimeError("Frozen ensemble evaluator/control binding drift")
    save_json(binding, bound)
    attempt = root / "evaluation-attempts" / f"{time_ns()}-started.json"
    started = perf_counter(); save_json(attempt, {"binding": digest(bound), "status": "STARTED", "hardware_shots": 0})
    with threadpool_limits(limits=1):
        for case in config["case_ids"]:
            boundary(root, config["resource_config"])
            ref = load_reference(source / "references" / case)
            directory = root / "readouts" / case; directory.mkdir(exist_ok=True, parents=True)
            db = open_db(directory / "readouts.sqlite")
            db.executescript("""
              CREATE TABLE IF NOT EXISTS member_draws (draw INTEGER, member_id TEXT, allocated_shots INTEGER,
                seed TEXT, counts BLOB, seconds REAL, PRIMARY KEY(draw,member_id));
              CREATE TABLE IF NOT EXISTS ensemble_resources (draw INTEGER PRIMARY KEY, cnot_shots INTEGER,
                depth_shots INTEGER, max_cnot INTEGER, max_depth INTEGER, unique_settings INTEGER,
                minimum_switches INTEGER, registered_order_switches INTEGER);
              CREATE TABLE IF NOT EXISTS reused_controls (draw INTEGER PRIMARY KEY, source_database TEXT,
                source_draw_id INTEGER, source_checkpoint_sha TEXT, acquisition_kind TEXT);
              CREATE TABLE IF NOT EXISTS pair_readouts (reference_method TEXT, method TEXT, shots INTEGER,
                rep INTEGER, cap INTEGER, common_d INTEGER, reference_subspace INTEGER, subspace INTEGER,
                PRIMARY KEY(reference_method,method,shots,rep,cap));
            """)
            cache = RitzCache(ref["h"], ref["ids"], ref["meta"]["request_digest"], db)
            ensembles = {}
            for spec in config["methods"]:
                result = json.loads((root / "cells" / case / spec["name"] / "result.json").read_text())
                if result["status"] == "FAILED_MEMBERS_PRESERVED":
                    continue
                records, probabilities = load_members(root, case, result)
                ensembles[spec["name"]] = (result, records, probabilities)
                db.execute("INSERT OR IGNORE INTO checkpoints(sha,method,step,primary_record,metadata) VALUES(?,?,?,?,?)",
                           (result["task_id"], spec["name"], 0, 1, json.dumps({"role": "NEW_ENSEMBLE", "member_count": len(records)})))
            origin = controls / "readouts" / case / "readouts.sqlite"
            src = sqlite3.connect(f"file:{origin}?mode=ro", uri=True)
            control_cps = list(src.execute("SELECT id,sha,method FROM checkpoints ORDER BY method"))
            for _, sha, name in control_cps:
                db.execute("INSERT OR IGNORE INTO checkpoints(sha,method,step,primary_record,metadata) VALUES(?,?,?,?,?)",
                           (sha, name, 0, 1, json.dumps({"role": "REUSED_B1_CONTROL", "source_database": str(origin)})))
            cp_index = dict(db.execute("SELECT method,id FROM checkpoints"))
            for M in budget["M_grid"]:
                for rep in range(budget["evaluation_repeats"]):
                    orders = {}
                    for name, (result, members, probabilities) in ensembles.items():
                        cp = cp_index[name]
                        existing = db.execute("SELECT id,counts FROM draws WHERE cp=? AND shots=? AND rep=?", (cp, M, rep)).fetchone()
                        if existing:
                            draw, raw = existing; ids, ns = unpack_pairs(raw)
                            if db.execute("SELECT COUNT(*) FROM member_draws WHERE draw=?", (draw,)).fetchone()[0] != len(members):
                                raise RuntimeError("Incomplete member transaction; do not use aggregate without its evidence")
                        else:
                            allocation = allocate(M, len(members)); mid = result["member_ids"]
                            ids, ns, records = draw_members(probabilities, allocation, [case, result["task_id"], M, rep], mid)
                            db.execute("INSERT INTO draws(cp,shots,rep,seed,counts,seconds) VALUES(?,?,?,?,?,?)",
                                       (cp, M, rep, "INDEPENDENT_MEMBER_SEEDS;see member_draws", pack_pairs(ids, ns), sum(r["seconds"] for r in records)))
                            draw = db.execute("SELECT last_insert_rowid()").fetchone()[0]
                            db.executemany("INSERT INTO member_draws VALUES(?,?,?,?,?,?)", [(draw, r["member_id"], r["allocated_shots"], r["seed"], pack_pairs(r["ids"], r["counts"]), r["seconds"]) for r in records])
                            settings = [r["setting_id"] for r, shots in zip(members, allocation) if shots]
                            db.execute("INSERT INTO ensemble_resources VALUES(?,?,?,?,?,?,?,?)", (
                                draw, sum(r["resources"]["cnot_count"]*m for r, m in zip(members, allocation)),
                                sum(r["resources"]["cnot_depth"]*m for r, m in zip(members, allocation)),
                                max(r["resources"]["cnot_count"] for r in members), max(r["resources"]["cnot_depth"] for r in members),
                                len(set(settings)), max(0, len(set(settings))-1), sum(a!=b for a, b in zip(settings, settings[1:]))))
                            db.commit()  # One complete ensemble counts transaction.
                        orders[name] = ranks(ids, ns, ref["ids"], ref["meta"]["hf_id"], [case, name, M, rep])["frequency"]
                    for old_cp, sha, name in control_cps:
                        source_draw = src.execute("SELECT id,seed,counts FROM draws WHERE cp=? AND shots=? AND rep=?", (old_cp, M, rep)).fetchone()
                        if source_draw is None:
                            raise RuntimeError("Required sealed B1 control readout missing")
                        source_id, seed, raw = source_draw; ids, ns = unpack_pairs(raw); cp = cp_index[name]
                        db.execute("INSERT OR IGNORE INTO draws(cp,shots,rep,seed,counts,seconds) VALUES(?,?,?,?,?,?)", (cp, M, rep, seed, raw, 0.))
                        draw = db.execute("SELECT id FROM draws WHERE cp=? AND shots=? AND rep=?", (cp, M, rep)).fetchone()[0]
                        db.execute("INSERT OR IGNORE INTO reused_controls VALUES(?,?,?,?,?)", (draw, str(origin), source_id, sha, "REUSE_NO_NEW_ACQUISITION"))
                        orders[name] = ranks(ids, ns, ref["ids"], ref["meta"]["hf_id"], [case, name, M, rep])["frequency"]
                    for name, order in orders.items():
                        draw = db.execute("SELECT id FROM draws WHERE cp=? AND shots=? AND rep=?", (cp_index[name], M, rep)).fetchone()[0]
                        for R in budget["R_grid"]:
                            sid, _, hit, seconds = cache.solve(order[:R])
                            db.execute("INSERT OR IGNORE INTO outcomes VALUES(?,?,?,?,?,?)", (draw, R, "frequency", sid, int(hit), seconds))
                            if name in ensembles:
                                for reference_method in ("source_final_ceo_ovp", "source_final_ceo_mvp"):
                                    d = min(R, len(order), len(orders[reference_method]))
                                    a, _, _, _ = cache.solve(orders[reference_method][:d]); b, _, _, _ = cache.solve(order[:d])
                                    db.execute("INSERT OR IGNORE INTO pair_readouts VALUES(?,?,?,?,?,?,?,?)", (reference_method, name, M, rep, R, d, a, b))
                    db.commit()
            frame = pd.read_sql_query("SELECT c.method,c.metadata,d.shots,d.rep,o.cap,s.energy,s.dimension,s.residual FROM outcomes o JOIN draws d ON d.id=o.draw JOIN checkpoints c ON c.id=d.cp JOIN subspaces s ON s.id=o.subspace", db)
            frame["error_mHa"] = (frame.energy-ref["meta"]["fci_energy"])*1000
            for (name, M, R), g in frame.groupby(["method", "shots", "cap"]):
                summaries.append({"case_id": case, "method": name, "shots": int(M), "R_cap": int(R), "repeats": len(g),
                                  "mean_error_mHa": float(g.error_mHa.mean()), "std_error_mHa": float(g.error_mHa.std(ddof=1)),
                                  "mean_actual_d": float(g.dimension.mean()), "chemical_accuracy_observed_rate": float((g.error_mHa <= 1.5936014376).mean()),
                                  "role": "NEW_ENSEMBLE" if name in ensembles else "REUSED_B1_CONTROL", "hardware_shots": 0})
            frame.drop(columns="metadata").to_parquet(directory / "individual-readouts.parquet", index=False)
            pairs = pd.read_sql_query("SELECT r.reference_method,r.method,r.shots,r.rep,r.cap,r.common_d,s0.energy AS reference_energy,s1.energy FROM pair_readouts r JOIN subspaces s0 ON s0.id=r.reference_subspace JOIN subspaces s1 ON s1.id=r.subspace", db)
            pairs.to_parquet(directory / "pairwise-ceo-matched-d.parquet", index=False)
            pd.read_sql_query("SELECT c.method,d.shots,d.rep,r.* FROM ensemble_resources r JOIN draws d ON d.id=r.draw JOIN checkpoints c ON c.id=d.cp", db).to_parquet(directory / "per-draw-ensemble-resources.parquet", index=False)
            counts = {"draws": db.execute("SELECT COUNT(*) FROM draws").fetchone()[0],
                      "component_draws": db.execute("SELECT COUNT(*) FROM member_draws").fetchone()[0],
                      "new_model_shots": db.execute("SELECT SUM(allocated_shots) FROM member_draws").fetchone()[0],
                      "reused_control_draws": db.execute("SELECT COUNT(*) FROM reused_controls").fetchone()[0]}
            save_json(directory / "completion.json", {"status": "COMPLETE", **counts, "control_database_sha256": file_digest(origin), "hardware_shots": 0})
            src.close(); db.close(); print("SqDRIFT independent ensemble readouts completed "+case, flush=True)
            boundary(root, config["resource_config"])
    pd.DataFrame(summaries).to_csv(root / "readout-summary.csv", index=False)
    status = {"status": "INDEPENDENT_ENSEMBLE_READOUT_COMPLETE", "summary_rows": len(summaries),
              "attempt_wall_seconds": perf_counter()-started, "hardware_shots": 0, "whole_study_complete": False}
    save_json(root / "evaluation-status.json", status)
    save_json(attempt.with_name(attempt.name.replace("started", "finished")), status)
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    p.add_argument("--controls", default=str(ROOT / "results/sampler-baselines-B1-standard-v1")); args = p.parse_args()
    print(json.dumps(evaluate(args.output, args.controls)))
