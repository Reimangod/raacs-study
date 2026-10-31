"""Fresh C cohort: separate real executions, not truncated completed trajectories."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import json
import multiprocessing
from pathlib import Path
import shutil
from time import perf_counter, time_ns
import traceback
from .common import ROOT, digest, file_digest, save_json
from .molecular_problem_no_fci import MolecularSpec, build_problem, load_problem
from .native_streaming_growth import GrowthSpec, run_growth
from .stopping_replay import POLICIES


def problems(mode):
    specs = [MolecularSpec("H2", .85), MolecularSpec("H4", 1.25)]
    if mode == "standard":
        specs += [MolecularSpec("H6", 1.25), MolecularSpec("LiH", 1.75, 1),
                  MolecularSpec("BeH2", 1.45, 1), MolecularSpec("N2", 1.25, 2),
                  MolecularSpec("LiH", 1.75, 0), MolecularSpec("BeH2", 1.45, 0)]
    return [(f"{s.molecule}_R{s.distance_A:g}_core{s.frozen_core}", s) for s in specs]


def execute_task(task):
    directory, physical, raw = task; root = Path(directory)
    try:
        return run_growth(load_problem(physical), GrowthSpec(**raw), root)
    except Exception as exc:
        result = {"status": "FAILED_ENGINEERING_PRESERVED", "error": str(exc), "traceback": traceback.format_exc(),
                  "spec": raw, "hardware_shots": 0, "failed_cost_not_assumed_zero": True}
        save_json(root/f"failure-{time_ns()}.json", result)
        return result


def run(root, mode, workers=2):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    if workers not in (1, 2): raise ValueError("At most2one-thread workers")
    impl = ("benchmark/actual_growth_cohort.py", "benchmark/native_streaming_growth.py", "benchmark/molecular_problem_no_fci.py",
            "benchmark/ansatz.py", "benchmark/online_stopping_rules.py", "benchmark/stopping_pilot_bank.py")
    cfg = {"schema": "PhaseC-fresh-actual-growth-cohort-v1", "mode": mode,
           "physical_problems": [{"case_id": name, "spec": asdict(s)} for name, s in problems(mode)],
           "methods": ["ceo_ovp", "ceo_mvp"], "policies": list(POLICIES), "construction_seed": 101,
           "max_steps": 8 if mode == "quick" else 64, "task_seconds": 60 if mode == "quick" else 180,
           "M_grid": [1000, 10000, 100000], "R_grid": [8, 16, 32, 64, 128, 256, 512],
           "final_evaluation_repeats": 4 if mode == "quick" else 20,
           "primary_policy": "gradient", "others": "fixed prospective diagnostics, not posthoc winners",
           "old_LOMO_validation_sha256": file_digest(ROOT/"results/PhaseC-replay-standard-v1/LOMO-v1/independent-validation.json"),
           "spec_sha256": file_digest(ROOT/"study-v1/PHASE_C_ACTUAL_GROWTH.md"),
           "implementation": {p: file_digest(ROOT/p) for p in impl},
           "free_floor_bytes": 3*2**30, "output_cap_bytes": 768*2**20,
           "FCI_solve": False, "FCI_labels": False, "hardware_shots": 0,
           "backend_scope": "small exact-statevector qualification <=16qubits; not scalable backend"}
    path = root/"manifest.json"
    if path.exists() and json.loads(path.read_text()) != cfg: raise RuntimeError("Cohort binding drift; use successor")
    if mode == "standard":
        quick = ROOT/"results/PhaseC-actual-growth-quick-v1/validation.json"
        if not quick.exists() or json.loads(quick.read_text())["status"] != "ACTUAL_GROWTH_PREFIX_AND_PHYSICS_VALIDATED":
            raise RuntimeError("Quick actual growth/prefix audit required before standard")
    save_json(path, cfg)
    with (root/"run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        start = perf_counter(); attempt = root/"attempts"/str(time_ns())
        save_json(attempt/"started.json", {"status": "STARTED", "cohort_digest": digest(cfg), "workers": workers})
        tasks = []; setup = []
        for case, spec in problems(mode):
            if shutil.disk_usage(root).free < cfg["free_floor_bytes"]: raise RuntimeError("Pre-setup free-floor guard")
            physical = root/"problems"/case; t = perf_counter()
            p = build_problem(spec, physical, small_full_H=True)
            setup.append({"case_id": case, "problem_id": p["meta"]["problem_id"],
                          "dependency_seconds_this_attempt": perf_counter()-t,
                          "original_problem_construction_seconds": p["meta"]["construction_wall_seconds"],
                          "problem_shared_across_all_policy_cells": True})
            for method in cfg["methods"]:
                for policy in cfg["policies"]:
                    growth = GrowthSpec(method, policy, max_steps=cfg["max_steps"], task_seconds=cfg["task_seconds"])
                    folder = root/"cells"/case/method/policy
                    tasks.append((str(folder), str(physical), asdict(growth)))
        save_json(attempt/"physical-setup.json", setup)
        if not (root/"physical-setup-first.json").exists(): save_json(root/"physical-setup-first.json", setup)
        completed = failed = 0; results = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as executor:
            # No failed task is silently counted successful. Existing result.json
            # is validated/skipped by the worker; unfinished committed steps resume.
            future_map = {executor.submit(execute_task, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]; result = future.result(); completed += 1
                failed += result["status"] == "FAILED_ENGINEERING_PRESERVED"
                results.append({"cell": str(Path(task[0]).relative_to(root)), "status": result["status"]})
                size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
                status = {"status": "RUNNING", "planned": len(tasks), "terminal": completed, "failed": failed,
                          "output_bytes": size, "hardware_shots": 0, "whole_study_complete": False}
                save_json(root/"execution-status.json", status)
                print(f"actual cohort {completed}/{len(tasks)} {task[0]} {result['status']}", flush=True)
                if size > cfg["output_cap_bytes"] or shutil.disk_usage(root).free < cfg["free_floor_bytes"]:
                    # Already-running tasks have their own boundary guard; do
                    # not kill them or delete evidence. Cancel only undispatched.
                    for pending in future_map: pending.cancel()
                    raise RuntimeError("Cohort storage boundary; running workers preserve committed prefixes")
        final = {"status": "ACTUAL_GROWTH_EXECUTED_PENDING_AUDIT", "planned": len(tasks), "terminal": completed,
                 "failed": failed, "cells": results, "wall_seconds_this_attempt": perf_counter()-start,
                 "hardware_shots": 0, "FCI_used": False, "whole_study_complete": False}
        save_json(root/"execution-status.json", final); save_json(attempt/"finished.json", final)
        return final


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="quick")
    p.add_argument("--output"); p.add_argument("--workers", type=int, default=2); a = p.parse_args()
    print(json.dumps(run(a.output or ROOT/f"results/PhaseC-actual-growth-{a.mode}-v1", a.mode, a.workers)))
