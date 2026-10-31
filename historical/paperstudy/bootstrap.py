"""Register the full intended cohort without pretending tasks were executed."""
import json
import sqlite3
from pathlib import Path
import platform
import shutil
from importlib.metadata import version

from .io import ROOT, PLAN_ROOT, digest, save_json, sha256


def main():
    plan = json.loads((PLAN_ROOT / "plan.json").read_text())
    database = ROOT / "run-queue.sqlite"
    db = sqlite3.connect(database)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS planned_cells (id TEXT PRIMARY KEY, case_id TEXT, method_id TEXT, budget INTEGER, seed INTEGER, status TEXT, reason TEXT, result_path TEXT)")
    for case in plan["primary_cases"]:
        for method in plan["primary_methods"]:
            for budget in plan["primary_design"]["total_model_shots_per_run"]:
                for seed in plan["primary_design"]["whole_run_seeds"]:
                    identity = ["primary-v1", case["id"], method["id"], budget, seed]
                    db.execute("INSERT OR IGNORE INTO planned_cells VALUES(?,?,?,?,?,?,?,?)",
                        (digest(identity), case["id"], method["id"], budget, seed, "NOT_STARTED", None, None))
    db.commit()
    count = db.execute("SELECT COUNT(*) FROM planned_cells").fetchone()[0]
    if count != 480:
        raise RuntimeError("Primary cohort cardinality drift")
    summary = {"status": "AUTHORIZED_EXECUTION_IN_PROGRESS", "objective": "Complete paper-aligned comparison plan v1.0",
               "plan_sha256": sha256((PLAN_ROOT/"plan.json").read_bytes()), "primary_planned_cells": count,
               "primary_status_counts": dict(db.execute("SELECT status,COUNT(*) FROM planned_cells GROUP BY status").fetchall()),
               "platform": platform.platform(),
               "runtime": {p: version(p) for p in ("numpy", "scipy", "ffsim", "pyscf", "qiskit", "qiskit-addon-sqd")},
               "free_disk_bytes": shutil.disk_usage(ROOT).free,
               "whole_experiment_complete": False, "candidate_performance_claim": "NONE"}
    save_json(ROOT / "execution-status.json", summary)
    db.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
