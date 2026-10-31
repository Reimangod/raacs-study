"""Full combinatorial readout audit, with explicitly sampled numerical parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import zlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .chemistry import load_reference
from .common import digest, rng_for, save_json
from .qsci import solve_selected
from .study.core import CAPS, SHOTS, ranks, unpack_pairs


def audit(root):
    start = perf_counter()
    root = Path(root)
    config = json.loads((root / "manifest.json").read_text())
    source = Path(config["source"])
    inventory = json.loads((root / "checkpoint-inventory.json").read_text())
    queue = json.loads((root / "execution-queue.json").read_text())
    chosen = set(queue["queue"])
    checks, totals = [], {"checkpoints": 0, "draws": 0, "readouts": 0, "subspaces": 0, "simulated_shots": 0, "sampled_numeric_parity": 0}
    failures = []
    with threadpool_limits(limits=1):
        for case in sorted({r["case_id"] for r in inventory if r["checkpoint_id"] in chosen}):
            case_records = [r for r in inventory if r["case_id"] == case and r["checkpoint_id"] in chosen]
            folder = root / "finite" / case
            db_path = folder / "readouts.sqlite"
            if not db_path.exists():
                failures.append({"case": case, "reason": "MISSING_DATABASE"})
                continue
            ref = load_reference(source / "references" / case)
            db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            method_lookup = {r[0]: (r[1], json.loads(r[2])) for r in db.execute("SELECT id,sha,metadata FROM checkpoints")}
            actual = {"checkpoints": len(method_lookup), "draws": db.execute("SELECT count(*) FROM draws").fetchone()[0],
                      "readouts": db.execute("SELECT count(*) FROM outcomes").fetchone()[0], "subspaces": db.execute("SELECT count(*) FROM subspaces").fetchone()[0],
                      "simulated_shots": db.execute("SELECT COALESCE(sum(shots),0) FROM draws").fetchone()[0]}
            errors = []
            if actual["checkpoints"] != len(case_records) or actual["draws"] != len(case_records)*len(SHOTS)*config["repeats"] or actual["readouts"] != actual["draws"]*len(CAPS)*2:
                errors.append("QUEUE_CARDINALITY_MISMATCH")
            bad_residual = db.execute("SELECT count(*) FROM subspaces WHERE residual>1e-8 OR dimension<1 OR dimension>512 OR projected_bytes<=0 OR rss_hwm<=0").fetchone()[0]
            if bad_residual:
                errors.append("SUBSPACE_NUMERIC_OR_MEMORY_CONTRACT")
            # Check every stored selected-ID set and its content address.
            for sha, raw, dimension in db.execute("SELECT sha,ids,dimension FROM subspaces"):
                ids = np.frombuffer(zlib.decompress(raw), dtype="<i4")
                if len(ids) != dimension or len(set(map(int, ids))) != dimension or sha != digest([ref["meta"]["request_digest"], ids.tolist()]):
                    errors.append("SUBSPACE_CONTENT_ADDRESS_MISMATCH")
                    break
            # Every draw has exact shot budget/seed and every R/rule matches its observed-only ranking.
            for draw_id, cp, M, rep, seed, raw in db.execute("SELECT id,cp,shots,rep,seed,counts FROM draws"):
                ids, counts = unpack_pairs(raw)
                sha, metadata = method_lookup[cp]
                if counts.sum() != M or np.any(counts <= 0) or len(ids) != len(np.unique(ids)):
                    errors.append("RAW_SHOT_BUDGET_OR_IDS_INVALID")
                    break
                _, expected_seed = rng_for(config["namespace"], case, sha, M, rep)
                if str(expected_seed) != seed:
                    errors.append("DRAW_SEED_BINDING_INVALID")
                    break
                ordered = ranks(ids, counts, ref["ids"], ref["meta"]["hf_id"], [sha, M, rep, config["namespace"]])
                rows = list(db.execute("SELECT o.cap,o.rule,s.sha,s.dimension,s.energy FROM outcomes o JOIN subspaces s ON s.id=o.subspace WHERE o.draw=? ORDER BY o.rule,o.cap", (draw_id,)))
                if len(rows) != 14:
                    errors.append("DRAW_READOUT_INCOMPLETE")
                    break
                prior = {}
                for cap, rule, sub_sha, dimension, energy in rows:
                    selected = sorted(ordered[rule][:cap])
                    if sub_sha != digest([ref["meta"]["request_digest"], selected]) or dimension != len(selected):
                        errors.append("OBSERVED_SELECTION_OR_DIMENSION_DRIFT")
                        break
                    if energy < ref["meta"]["fci_energy"]-1e-8 or energy > prior.get(rule, np.inf)+1e-8:
                        errors.append("VARIATIONAL_OR_NESTED_MONOTONICITY")
                        break
                    prior[rule] = energy
                if errors:
                    break
            # Independent solver parity is sampled, never claimed as full recomputation.
            sample = db.execute("SELECT ids,energy FROM subspaces ORDER BY sha LIMIT5".replace("LIMIT5", "LIMIT 5")).fetchall()
            for raw, energy in sample:
                ids = np.frombuffer(zlib.decompress(raw), dtype="<i4").astype(np.int64)
                independent = solve_selected(ref["h"], ids)
                if abs(independent["energy"]-energy) > 1e-8:
                    errors.append("SAMPLED_INDEPENDENT_NUMERIC_PARITY")
                totals["sampled_numeric_parity"] += 1
            for k in actual:
                totals[k] += actual[k]
            entry = {"case": case, "status": "PASS" if not errors else "FAIL", "counts": actual, "errors": errors}
            checks.append(entry)
            if errors:
                failures.append(entry)
            db.close()
            save_json(root / "audit-progress.json", {"cases_checked": len(checks), "planned_cases": queue["case_count"], "totals": totals, "failures": failures})
    mixes = []
    for case in sorted({r["case_id"] for r in inventory if r["checkpoint_id"] in chosen}):
        p = root / "mixtures" / case / "readouts.parquet"
        if not p.exists():
            mixes.append({"case": case, "status": "PENDING"})
            continue
        rows = pd.read_parquet(p)
        raw = pd.read_parquet(p.with_name("raw-counts.parquet"))
        rawmap = dict(zip(raw.row_id, raw.counts))
        error = None
        for r in rows.itertuples():
            ids, counts = unpack_pairs(rawmap[r.row_id])
            if counts.sum() != r.shots_total or sum(r.allocation_shots) != r.shots_total or r.pilot_shots_included > r.shots_total:
                error = "MIXTURE_BUDGET_DRIFT"
                break
            if r.matched_common_d > r.actual_dimension or r.matched_error_mHa < r.energy_error_mHa-1e-5:
                error = "MIXTURE_DIMENSION_OR_VARIATIONAL_DRIFT"
                break
        sizes = rows.groupby(["method", "shots_total", "replicate"]).size()
        common = rows.groupby(["method", "shots_total", "replicate"]).matched_common_d.nunique()
        if not (sizes == 5).all() or not (common == 1).all():
            error = "MIXTURE_POLICY_PAIR_INCOMPLETE"
        mixes.append({"case": case, "status": "PASS" if not error else "FAIL", "readouts": len(rows), "error": error})
        if error:
            failures.append(mixes[-1])
    pending = any(c["status"] == "PENDING" for c in mixes)
    result = {"status": "PASS_MILESTONE1_CORE_AUDIT" if not failures and not pending else "FAIL" if failures else "PENDING_MIXTURES",
              "whole_study_complete": False, "finite_cases": checks, "mixture_cases": mixes, "totals": totals,
              "seconds": perf_counter()-start, "failures": failures, "hardware_shots": 0,
              "coverage": "every count budget/seed, every selected-subspace content address, every R/rule observed-only selection and nested energy;150 sampled independent solver parity checks max; NOT full numerical recomputation",
              "remaining": "Figures/report QA and detailedA1–A5 interpretation; all phasesB–F remain"}
    save_json(root / "milestone1-audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("finite_cases", "mixture_cases")}, indent=2))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
