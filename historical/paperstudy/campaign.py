"""Frozen full-cohort controller, not a best-case or quick subset campaign."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
from time import perf_counter, time
import uuid

from .common_execution import load_physical, run_identity
from .io import ROOT, PLAN_ROOT, digest, save_json, sha256

PROJECT = ROOT.parents[2]/"ceo-sampler-benchmark"


def freeze():
    path = ROOT/"main-execution-freeze-v1.json"
    if path.exists():
        value = json.loads(path.read_text())
        for filename, expected in value["code_sha256"].items():
            if sha256(Path(filename).read_bytes()) != expected:
                raise RuntimeError("Frozen scientific execution code changed: "+filename)
        return value
    plan = json.loads((PLAN_ROOT/"plan.json").read_text())
    pilots = []
    for method in plan["primary_methods"]:
        for seed in plan["engineering_pilot"]["seeds"]:
            p = ROOT/"common-runs/S3_ENGINEERING_PILOT/N2-32-R1p2-P2"/method["id"]/("10000-"+str(seed))/"result.json"
            result = json.loads(p.read_text())
            if result["status"] != "COMPLETED" or result["budget_ledger"]["known_actual_shots"] != 10000:
                raise RuntimeError("All12 engineering pilot cells must complete their actual sampling")
            # Status, runtime and accounting are used here, not energies/rankings.
            pilots.append({"method": method["id"], "seed": seed, "result_sha256": sha256(p.read_bytes()),
                           "wall_seconds": result["this_attempt_wall_seconds"]})
    physical = {case["id"]: load_physical(case["id"])[0]["problem_id"] for case in plan["primary_cases"]}
    names = ["common_execution.py", "sqdrift_execution.py", "sqdrift_native_worker.py", "budget.py", "ci.py",
             "sparse_raacs.py", "lucj.py", "cell_worker.py"]
    files = [Path(__file__).with_name(n) for n in names]
    files += [PROJECT/"benchmark"/n for n in ("resource_adaptive_sampler.py", "local_sampler_primitives.py",
              "integral_connections.py", "integral_hamiltonian.py", "ansatz.py")]
    value = {"schema": "paper-common-execution-freeze-1", "plan_sha256": sha256((PLAN_ROOT/"plan.json").read_bytes()),
             "physical_problem_ids": physical, "code_sha256": {str(p): sha256(p.read_bytes()) for p in files},
             "pilot_status_and_runtime_evidence_only": pilots,
             "seed_roles_unchanged": True, "method_scores_and_angles_unchanged": True,
             "common_wall_cap_seconds_by_budget": {"10000": 600, "100000": 1800, "1000000": 5400},
             "wall_caps_rationale": "All-method common engineering margins above pilot duration; same caps per budget, not tuned to favorable energy",
             "local_lucj_sector_vector_limit_bytes": 2*2**30,
             "large_LUCJ_over_local_limit": "WAITING_REMOTE_EXACT_EXECUTOR, not scientific method failure",
             "total_CPU_core_hour_ceiling": 192, "total_A100_hour_ceiling": 96,
             "primary_planned_cells": 480, "primary_inference_uses_pilot_cells": False,
             "SqDRIFT_method_label": "P1-inspired native SDK grouped/common-budget adaptation; not P1 author-driver byte reproduction",
             "naphthalene_label": "Cited source geometry/AVAS reconstructed in current runtime; not author integral byte reproduction",
             "main_outcomes_used_for_this_freeze": False, "frozen_at_unix": time()}
    save_json(path, value)
    return value


def queue():
    db = sqlite3.connect(ROOT/"run-queue.sqlite", timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS active_claims(cell TEXT PRIMARY KEY, token TEXT, host TEXT, pid INTEGER, started REAL)")
    return db


def dispatch(maximum_cells=None):
    ROOT.mkdir(exist_ok=True)
    with (ROOT/"campaign-controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registration = freeze()
        plan = json.loads((PLAN_ROOT/"plan.json").read_text())
        db = queue()
        files = ROOT/"controller"
        files.mkdir(exist_ok=True)
        started = perf_counter()
        finished = 0
        for budget in plan["primary_design"]["total_model_shots_per_run"]:
            for case in plan["primary_cases"]:
                for seed in plan["primary_design"]["whole_run_seeds"]:
                    for method in plan["primary_methods"]:
                        mid, cid = method["id"], case["id"]
                        row = db.execute("SELECT id,status FROM planned_cells WHERE case_id=? AND method_id=? AND budget=? AND seed=?", (cid, mid, budget, seed)).fetchone()
                        if row[1] in ("COMPLETED", "FAILED_ENGINEERING_PRESERVED", "RESOURCE_CAPPED"):
                            continue
                        meta, _, _ = load_physical(cid)
                        sector_bytes = 16*math.comb(meta["norb"], meta["nalpha"])*math.comb(meta["norb"], meta["nbeta"])
                        if "LUCJ" in mid and sector_bytes > registration["local_lucj_sector_vector_limit_bytes"]:
                            if row[1]!='RUNNING': # Existing remote ownership is not a new waiting task.
                                db.execute("UPDATE planned_cells SET status=?,reason=? WHERE id=?", ("WAITING_REMOTE_EXECUTOR", "Exact sector vector exceeds registered local2GiB; not a scientific failure", row[0]))
                                db.commit()
                            continue
                        if shutil.disk_usage(ROOT).free < 3*2**30:
                            save_json(files/"resource-boundary.json", {"free_bytes": shutil.disk_usage(ROOT).free,
                                      "cell_id": row[0], "unstarted_cell_not_executed": True, "scientific_result": None})
                            return
                        consumed_cpu = sum(json.loads(p.read_text()).get("child_CPU_seconds", 0.)
                                           for p in files.glob("*-terminal.json"))
                        if consumed_cpu >= registration["total_CPU_core_hour_ceiling"]*3600:
                            save_json(files/"resource-boundary.json", {"kind": "CPU_ALLOCATION_EXHAUSTED",
                                      "consumed_CPU_seconds": consumed_cpu, "cell_not_executed": row[0]})
                            return
                        token = uuid.uuid4().hex
                        db.execute("BEGIN IMMEDIATE")
                        old = db.execute("SELECT host,pid FROM active_claims WHERE cell=?", (row[0],)).fetchone()
                        if old:
                            if old[0] != socket.gethostname():
                                raise RuntimeError("Remote queue owner must be reconciled before replacement")
                            try:
                                os.kill(old[1], 0)
                            except ProcessLookupError:
                                db.execute("DELETE FROM active_claims WHERE cell=?", (row[0],))
                            else:
                                raise RuntimeError("Existing controller owner still live")
                        db.execute("INSERT INTO active_claims VALUES(?,?,?,?,?)", (row[0], token, socket.gethostname(), os.getpid(), time()))
                        db.execute("UPDATE planned_cells SET status='RUNNING',reason=NULL WHERE id=?", (row[0],))
                        db.commit()
                        env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                   MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1", QISKIT_NUM_PROCS="1")
                        command = ["rtk", "proxy", sys.executable, "-u", "-m", "paperstudy.cell_worker", "--case", cid,
                                   "--method", mid, "--budget", str(budget), "--seed", str(seed)]
                        log = files/(row[0]+".log")
                        tick = perf_counter()
                        old_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                        with log.open("a") as stream:
                            proc = subprocess.Popen(command, env=env, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                            save_json(files/(row[0]+"-dispatch.json"), {"cell_id": row[0], "pid": proc.pid, "owner_token": token,
                                      "command": command, "parent_pid": os.getpid(), "started_unix": time(),
                                      "controller_sha256": sha256(Path(__file__).read_bytes())})
                            try:
                                code = proc.wait(timeout=registration["common_wall_cap_seconds_by_budget"][str(budget)])
                                reason = None if code == 0 else "Worker exit "+str(code)+"; raw attempts and budget reservations preserved"
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid, signal.SIGTERM)
                                code = proc.wait(timeout=30)
                                reason = "Common frozen wall cap reached; exact process group SIGTERM, preserve work"
                        result = ROOT/"common-runs/C_PRIMARY"/cid/mid/(str(budget)+"-"+str(seed))/"result.json"
                        status = "COMPLETED" if code == 0 and result.exists() else "FAILED_ENGINEERING_PRESERVED"
                        db.execute("BEGIN IMMEDIATE")
                        db.execute("UPDATE planned_cells SET status=?,reason=?,result_path=? WHERE id=?", (status, reason, str(result.relative_to(ROOT)) if result.exists() else None, row[0]))
                        db.execute("DELETE FROM active_claims WHERE cell=? AND token=?", (row[0], token))
                        db.commit()
                        new_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                        save_json(files/(row[0]+"-terminal.json"), {"cell_id": row[0], "status": status, "reason": reason,
                                  "wall_seconds": perf_counter()-tick, "exit_code": code,
                                  "child_CPU_seconds": new_usage.ru_utime+new_usage.ru_stime-old_usage.ru_utime-old_usage.ru_stime})
                        finished += 1
                        counts = dict(db.execute("SELECT status,COUNT(*) FROM planned_cells GROUP BY status").fetchall())
                        save_json(files/"live-status.json", {"controller_pid": os.getpid(), "primary_status_counts": counts,
                                  "elapsed_seconds_this_controller": perf_counter()-started, "whole_experiment_complete": False})
                        print(cid, mid, budget, seed, status, counts, flush=True)
                        if maximum_cells is not None and finished >= maximum_cells:
                            return
            remaining = db.execute("SELECT COUNT(*) FROM planned_cells WHERE budget=? AND status IN ('NOT_STARTED','RUNNING','WAITING_REMOTE_EXECUTOR')", (budget,)).fetchone()[0]
            if remaining:
                save_json(files/"budget-tier-pending.json", {"budget": budget, "pending_cells": remaining,
                          "higher_budget_execution_started": False,
                          "reason": "Finish the all-method low-budget tier before progressing; other tiers are retained in full queue"})
                return
        db.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze-only", action="store_true")
    p.add_argument("--maximum-cells", type=int)
    a = p.parse_args()
    if a.freeze_only:
        print(json.dumps(freeze(), indent=2))
    else:
        dispatch(a.maximum_cells)
