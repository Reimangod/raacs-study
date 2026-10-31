from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..chemistry import load_reference
from ..common import digest, file_digest, resolve_artifact, rng_for, save_json
from .core import ADAPT, CAPS, NAMESPACE, SHOTS, RitzCache, discovery, open_db, pack_pairs, ranks, rss_peak


def frozen_inventory(source, output, verify_all=True):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("New output cannot be inside the frozen source")
    pointer = json.loads((source / "REPRODUCIBILITY.json").read_text())
    manifest_path = source / pointer["manifest"]
    if file_digest(manifest_path) != pointer["manifest_sha256"]:
        raise RuntimeError("Source manifest digest mismatch")
    old_manifest = json.loads(manifest_path.read_text())
    required = ["STUDY_PROTOCOL.md", "report.md", "REQUIREMENTS_AUDIT.md", "execution-manifest-final.json", "summary.csv", "qsci-summary.csv", "all-cell-status.csv", "analysis/checkpoint_optima.csv", "analysis/analytic_metrics.csv", "analysis/construction_costs.csv", "analysis/shots_to_chemical_accuracy.csv", "sqd-matched-dimension/summary.csv"]
    required_paths = [source.parent.parent / "STUDY_PROTOCOL.md" if n == "STUDY_PROTOCOL.md" else source / n for n in required]
    if any(not p.is_file() for p in required_paths):
        raise FileNotFoundError("Required evidence missing")
    audit_path = output / "source-audit.json"
    if not audit_path.exists() and verify_all:
        checked, total = 0, 0
        for name, record in old_manifest["artifacts"].items():
            p = source / name
            h = hashlib.sha256()
            with p.open("rb") as stream:
                for chunk in iter(lambda: stream.read(2**20), b""):
                    h.update(chunk)
            if h.hexdigest() != record["sha256"] or p.stat().st_size != record["bytes"]:
                raise RuntimeError(f"Frozen artifact drift: {name}")
            checked += 1
            total += p.stat().st_size
        save_json(audit_path, {"source": str(source), "manifest_sha": pointer["manifest_sha256"], "checked": checked,
                               "bytes": total, "all_match": True, "source_modified": False})
    cells = pd.read_csv(source / "all-cell-status.csv").fillna("")
    latest = {}
    records = []
    for p in sorted((source / "checkpoints").glob("*/*/step-*.json")):
        cp = json.loads(p.read_text())
        row = {"case_id": cp["case_id"], "method": cp["method"], "step": cp["step"], "checkpoint_id": cp["checkpoint_id"],
               "checkpoint_path": str(p.relative_to(source)), "source_sha": file_digest(p),
               "distribution_path": str(resolve_artifact(source, cp["distribution"]).relative_to(source)),
               "distribution_sha": cp["distribution_sha256"], "vqe_energy": cp["vqe_energy"],
               "cnot": cp["resources"]["cnot_count"], "depth": cp["resources"]["cnot_depth"],
               "parameters": cp["resources"]["parameter_count"], "operators": cp.get("selected_operators", []),
               "added_group": cp.get("selected_group", []), "config_digest": cp["config_digest"]}
        key = (row["case_id"], row["method"])
        latest[key] = max(latest.get(key, -1), row["step"])
        records.append(row)
    for row in records:
        row["primary"] = row["method"] in ADAPT or row["step"] == latest[(row["case_id"], row["method"])]
    save_json(output / "checkpoint-inventory.json", records)
    save_json(output / "cell-coverage.json", {"planned": cells.to_dict("records"), "saved_checkpoints": len(records),
              "primary_checkpoints": sum(r["primary"] for r in records), "source_dataset": pointer,
              "fixed_control_archives_are_not_growth": True})
    return records, pointer


def retrospective(source, output):
    source, output = Path(source), Path(output)
    q = pd.read_csv(source / "qsci-summary.csv")
    q = q[q.include_hf & q.method.isin(ADAPT)]
    status = pd.read_csv(source / "all-cell-status.csv").set_index(["case_id", "method"])
    rows = []
    for (case, method, M), g in q.groupby(["case_id", "method", "shots"]):
        g = g.sort_values("step")
        final = g.iloc[-1]
        best = g.sort_values(["mean_energy_error_mHa", "cnot_count", "step"]).iloc[0]
        for tol in (0., .1, .5, 1.):
            candidates = g[g.mean_energy_error_mHa <= best.mean_energy_error_mHa + tol]
            selected = candidates.iloc[0]
            rows.append({"case_id": case, "molecule": case.split("_R")[0], "method": method, "shots": int(M),
                         "tolerance_mHa": tol, "status": status.loc[(case, method), "status"],
                         "best_step": int(best.step), "selected_step": int(selected.step), "last_saved_step": int(final.step),
                         "step_ratio": float(selected.step / final.step) if final.step else None,
                         "cnot_ratio": float(selected.cnot_count / final.cnot_count) if final.cnot_count else None,
                         "depth_ratio": float(selected.cnot_depth / final.cnot_depth) if final.cnot_depth else None,
                         "cnot_reduction_percent": 100 * (1 - selected.cnot_count / final.cnot_count) if final.cnot_count else None,
                         "depth_reduction_percent": 100 * (1 - selected.cnot_depth / final.cnot_depth) if final.cnot_depth else None,
                         "selected_error_mHa": float(selected.mean_energy_error_mHa), "best_error_mHa": float(best.mean_energy_error_mHa),
                         "final_error_mHa": float(final.mean_energy_error_mHa),
                         "final_minus_selected_error_mHa": float(final.mean_energy_error_mHa - selected.mean_energy_error_mHa),
                         "scope": "RETROSPECTIVE_SAME_SAMPLE; ratio_to_last_saved_not_true_convergence"})
    df = pd.DataFrame(rows)
    df.to_parquet(output / "retrospective-stops.parquet", index=False)
    df.to_csv(output / "retrospective-stops.csv", index=False)
    agg = df.groupby(["molecule", "method", "shots", "tolerance_mHa"]).agg(
        conditions=("case_id", "count"), median_step_ratio=("step_ratio", "median"), median_cnot_reduction=("cnot_reduction_percent", "median"),
        median_depth_reduction=("depth_reduction_percent", "median"), earlier_fraction=("step_ratio", lambda v: float((v < 1).mean()))).reset_index()
    agg.to_csv(output / "stops-by-molecule-pool.csv", index=False)
    return len(df)


def operator_tracking(source, output, records):
    """Trace probabilities; association includes full reoptimization, not causal attribution."""
    source, output = Path(source), Path(output)
    q = pd.read_csv(source / "qsci-summary.csv")
    q = q[q.include_hf]
    qlookup = {(r.case_id, r.method, int(r.step), int(r.shots)): r.mean_energy_error_mHa for r in q.itertuples()}
    counts = {"trajectories": 0, "transitions": 0, "configuration_lifecycles": 0, "important_probability_rows": 0}
    for case in sorted({r["case_id"] for r in records}):
        folder = output / "operator-tracking" / case
        terminal = folder / "status.json"
        if terminal.exists():
            for key, value in json.loads(terminal.read_text())["counts"].items():
                counts[key] += value
            continue
        ref = load_reference(source / "references" / case)
        ids, weights, hf = ref["ids"], ref["weights"], ref["meta"]["hf_id"]
        targets = sorted(np.flatnonzero((ids != hf) & (weights > 1e-16)), key=lambda i: (-weights[i], int(ids[i])))[:50]
        transitions, lives, probabilities = [], [], []
        for method in ADAPT:
            cps = sorted([r for r in records if r["case_id"] == case and r["method"] == method], key=lambda r: r["step"])
            if not cps:
                continue
            sector = []
            for cp in cps:
                with np.load(source / cp["distribution_path"]) as z:
                    p = z["probability"]
                sector.append(p[ids])
            series = np.array(sector)
            for i, state_id in enumerate(ids):
                maximum = int(np.argmax(series[:, i]))
                item = {"case_id": case, "method": method, "configuration_id": int(state_id),
                        "excitation_rank_from_hf": (int(state_id) ^ int(hf)).bit_count() // 2,
                        "fci_weight": float(weights[i]), "maximum_probability": float(series[maximum, i]),
                        "max_probability_step": cps[maximum]["step"], "final_probability": float(series[-1, i]),
                        "max_source_checkpoint": cps[maximum]["checkpoint_id"], "fci_scope": "ORACLE_DIAGNOSTIC_ONLY"}
                for threshold in (1e-12, 1e-8, 1e-4):
                    crossed = np.flatnonzero(series[:, i] > threshold)
                    item[f"first_step_gt_{threshold:g}"] = cps[int(crossed[0])]["step"] if len(crossed) else None
                lives.append(item)
            hf_pos = int(np.flatnonzero(ids == hf)[0])
            for t, cp in enumerate(cps):
                for i in targets:
                    probabilities.append({"case_id": case, "method": method, "step": cp["step"], "configuration_id": int(ids[i]),
                                          "probability": float(series[t, i]), "fci_weight": float(weights[i]), "fci_scope": "ORACLE_TOP50"})
                if t == 0:
                    continue
                prior = cps[t - 1]
                change = series[t] - series[t - 1]
                transitions.append({"case_id": case, "method": method, "step": cp["step"], "checkpoint_id": cp["checkpoint_id"],
                                    "prior_checkpoint_id": prior["checkpoint_id"], "added_group": cp["added_group"],
                                    "pool_binding_config_digest": cp["config_digest"],
                                    "new_ids_gt_1e-12": ids[(series[t] > 1e-12) & (series[t - 1] <= 1e-12)].tolist(),
                                    "important_increase_ids": [int(ids[i]) for i in targets if change[i] > 1e-12],
                                    "important_decrease_ids": [int(ids[i]) for i in targets if change[i] < -1e-12],
                                    "delta_hf_probability": float(change[hf_pos]), "delta_cnot": cp["cnot"] - prior["cnot"],
                                    "delta_depth": cp["depth"] - prior["depth"],
                                    **{f"delta_qsci_error_M{M}_mHa": qlookup[(case, method, cp["step"], M)] - qlookup[(case, method, prior["step"], M)] for M in (1000, 10000, 100000)},
                                    "attribution": "operator_addition_AND_full_reoptimization; association_not_isolated_causation"})
        folder.mkdir(parents=True, exist_ok=True)
        for name, rows in (("transitions", transitions), ("configuration-lifecycles", lives), ("important-probabilities", probabilities)):
            pd.DataFrame(rows).to_parquet(folder / f"{name}.parquet", index=False)
        local = {"trajectories": sum(bool([r for r in records if r["case_id"] == case and r["method"] == m]) for m in ADAPT),
                 "transitions": len(transitions), "configuration_lifecycles": len(lives), "important_probability_rows": len(probabilities)}
        save_json(terminal, {"status": "COMPLETED", "counts": local, "thresholds_are_registered_diagnostic_assumptions": [1e-12, 1e-8, 1e-4]})
        for k, v in local.items():
            counts[k] += v
    save_json(output / "operator-tracking-status.json", counts)
    return counts


def health(output, options):
    free = shutil.disk_usage(output).free
    used = sum(p.stat().st_size for p in Path(output).rglob("*") if p.is_file())
    if free < options["free_floor_bytes"]:
        raise RuntimeError("YIELD_STORAGE_FLOOR; preserve progress and resume after space available")
    if used >= options["output_cap_bytes"]:
        raise RuntimeError("YIELD_OUTPUT_CAP; preserve evidence, no data deletion")
    if rss_peak() > options["worker_ram_bytes"]:
        raise RuntimeError("YIELD_WORKER_MEMORY_CAP")


def finite_case(case, records, options):
    source, output = Path(options["source"]), Path(options["output"])
    folder = output / "finite" / case
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with threadpool_limits(limits=options["threads"]):
            return _finite_case_locked(case, records, options, source, output, folder)


def _finite_case_locked(case, records, options, source, output, folder):
    start = perf_counter()
    ref = load_reference(source / "references" / case)
    db = open_db(folder / "readouts.sqlite")
    cache = RitzCache(ref["h"], ref["ids"], ref["meta"]["request_digest"], db)
    namespace = options["namespace"]
    try:
        for cp in records:
            if perf_counter() - start > options["case_seconds"]:
                save_json(folder / "status.json", {"status": "YIELDED_WITH_PENDING_WORK", "case": case, "reason": "time_slice", "pid": os.getpid()})
                return {"case": case, "status": "YIELDED_WITH_PENDING_WORK"}
            health(output, options)
            db.execute("INSERT OR IGNORE INTO checkpoints (sha,method,step,primary_record,metadata) VALUES(?,?,?,?,?)",
                       (cp["checkpoint_id"], cp["method"], cp["step"], int(cp["primary"]), json.dumps(cp)))
            cp_id = db.execute("SELECT id FROM checkpoints WHERE sha=?", (cp["checkpoint_id"],)).fetchone()[0]
            expected = len(SHOTS) * options["repeats"] * len(CAPS) * 2
            done = db.execute("SELECT count(*) FROM outcomes o JOIN draws d ON d.id=o.draw WHERE d.cp=?", (cp_id,)).fetchone()[0]
            if done == expected and db.execute("SELECT 1 FROM analytics WHERE cp=?", (cp_id,)).fetchone():
                continue
            dist = source / cp["distribution_path"]
            if file_digest(dist) != cp["distribution_sha"]:
                raise RuntimeError("Frozen distribution hash drift")
            with np.load(dist) as z:
                p = z["probability"].astype(float)
            if p.min() < -1e-14 or abs(p.sum() - 1) > 1e-9:
                raise ValueError("Invalid saved probability")
            p = np.maximum(p, 0)
            p /= p.sum()
            if not db.execute("SELECT 1 FROM analytics WHERE cp=?", (cp_id,)).fetchone():
                payload = discovery(p, ref, cp["checkpoint_id"], options["joint_mc_trials"])
                db.execute("INSERT INTO analytics VALUES(?,?)", (cp_id, zlib.compress(json.dumps(payload).encode())))
            for M in SHOTS:
                for rep in range(options["repeats"]):
                    old = db.execute("SELECT id FROM draws WHERE cp=? AND shots=? AND rep=?", (cp_id, M, rep)).fetchone()
                    if old and db.execute("SELECT count(*) FROM outcomes WHERE draw=?", old).fetchone()[0] == len(CAPS) * 2:
                        continue
                    ts = perf_counter()
                    rng, seed = rng_for(namespace, case, cp["checkpoint_id"], M, rep)
                    all_counts = rng.multinomial(M, p)
                    ids = np.flatnonzero(all_counts)
                    counts = all_counts[ids]
                    if old:
                        draw_id = old[0]
                    else:
                        cursor = db.execute("INSERT INTO draws(cp,shots,rep,seed,counts,seconds) VALUES(?,?,?,?,?,?)",
                                            (cp_id, M, rep, str(seed), pack_pairs(ids, counts), perf_counter() - ts))
                        draw_id = cursor.lastrowid
                    order = ranks(ids, counts, ref["ids"], ref["meta"]["hf_id"], [cp["checkpoint_id"], M, rep, namespace])
                    for rule, ranked in order.items():
                        prior_energy = None
                        for cap in CAPS:
                            selected = ranked[:cap]
                            sub_id, energy, hit, seconds = cache.solve(selected)
                            if energy < ref["meta"]["fci_energy"] - 1e-8:
                                raise RuntimeError("Variational reference violation")
                            if prior_energy is not None and energy > prior_energy + 1e-8:
                                raise RuntimeError("Nested-space monotonicity violation")
                            prior_energy = energy
                            db.execute("INSERT OR IGNORE INTO outcomes VALUES(?,?,?,?,?,?)", (draw_id, cap, rule, sub_id, int(hit), seconds))
                db.commit()
                db.execute("INSERT OR REPLACE INTO progress VALUES('latest',?)", (json.dumps({"checkpoint": cp["checkpoint_id"], "method": cp["method"], "step": cp["step"], "M": M}),))
                db.commit()
            save_json(folder / "status.json", {"status": "RUNNING", "case": case, "pid": os.getpid(), "last_complete_checkpoint": cp["checkpoint_id"],
                                              "draws": db.execute("SELECT count(*) FROM draws").fetchone()[0], "peak_rss_bytes": rss_peak()})
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        counts = {"checkpoints": db.execute("SELECT count(*) FROM checkpoints").fetchone()[0],
                  "draws": db.execute("SELECT count(*) FROM draws").fetchone()[0],
                  "readouts": db.execute("SELECT count(*) FROM outcomes").fetchone()[0],
                  "unique_subspaces": db.execute("SELECT count(*) FROM subspaces").fetchone()[0],
                  "new_simulated_shots": db.execute("SELECT COALESCE(sum(shots),0) FROM draws").fetchone()[0]}
        terminal = {"status": "COMPLETED", "case": case, "counts": counts, "seconds_this_invocation": perf_counter() - start,
                    "peak_rss_bytes": rss_peak(), "peak_memory_scope": "measured process-lifetime RSS high-water mark, not per-solve allocation", "hardware_shots": 0}
        save_json(folder / "status.json", terminal)
        return terminal
    except Exception as error:
        db.rollback()
        failure = {"status": "FAILED_PRESERVED", "case": case, "error": repr(error), "traceback": traceback.format_exc(), "seconds": perf_counter() - start}
        save_json(folder / f"failure-{digest(failure)[:12]}.json", failure)
        save_json(folder / "status.json", failure)
        return failure
    finally:
        db.close()


def launch_finite(records, options):
    cases = sorted({r["case_id"] for r in records})
    results = []
    if options["workers"] == 1:
        return [finite_case(c, [r for r in records if r["case_id"] == c], options) for c in cases]
    with ProcessPoolExecutor(max_workers=options["workers"], max_tasks_per_child=1) as pool:
        futures = {pool.submit(finite_case, c, [r for r in records if r["case_id"] == c], options): c for c in cases}
        for future in as_completed(futures):
            results.append(future.result())
            save_json(Path(options["output"]) / "finite-progress.json", {"terminal_this_invocation": results, "planned_cases": cases})
    return results
