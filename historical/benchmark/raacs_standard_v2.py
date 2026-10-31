"""Launcher successor: JSON-stable config identity, unchanged qualified kernel.

The original qualification launcher serialized tuple variants as JSON lists;
direct dict equality would reject a subsequent resume. Preserve that launcher
and its qualification artifacts. This fixes representation only, not physics.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import fcntl
import json
import multiprocessing
from pathlib import Path
import shutil
from time import perf_counter
from .common import ROOT, file_digest, save_json
from .raacs_standard import run_case, variants
from .study.reanalysis import frozen_inventory


def configuration(root, source):
    files = ("benchmark/raacs_standard_v2.py", "benchmark/raacs_standard.py", "benchmark/resource_adaptive_sampler.py",
             "benchmark/local_sampler_primitives.py", "benchmark/circuit_sampling_runtime.py", "benchmark/integral_hamiltonian.py",
             "benchmark/integral_connections.py", "benchmark/mps_sampling.py", "benchmark/ansatz.py")
    cfg = {"schema": "RAACS-standard-construction-launcher-v2", "mode": "standard", "source": str(source),
           "case_ids": sorted(p.name for p in (source/"references").iterdir() if p.is_dir()),
           "total_model_shots": 10000, "max_steps": 8, "MPS_max_bond": 64,
           "variants": variants("standard"), "exact_small_cells": 810, "MPS_cells": 30,
           "spec_sha256": file_digest(ROOT/"study-v1/RAACS_STANDARD_EXECUTION.md"),
           "implementation": {p: file_digest(ROOT/p) for p in files}, "free_floor_bytes": 3*2**30,
           "aggregate_output_boundary_bytes": 768*2**20, "hardware_shots": 0,
           "representation_fix_only": "normalize tuples to JSON-native lists before config equality"}
    return json.loads(json.dumps(cfg))


def run(root, source, workers=2):
    root, source = Path(root).resolve(), Path(source).resolve()
    if root == source or source in root.parents: raise ValueError("Protect sealed source")
    if workers not in (1, 2): raise ValueError("At most2numericalworkers")
    root.mkdir(parents=True, exist_ok=True)
    with (root/"run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg = configuration(root, source); path = root/"manifest.json"
        if path.exists() and json.loads(path.read_text()) != cfg: raise RuntimeError("Configuration drift; use successor")
        for name in ("RAACS-quick-v1", "RAACS-variants-quick-v1"):
            q = ROOT/"results"/name/"validation.json"
            if not q.exists() or json.loads(q.read_text())["status"] != "RAACS_QUICK_COUNTS_PHYSICS_AND_FINAL_SUPPORT_VALIDATED":
                raise RuntimeError("Actual variants and independent quick audit required")
        if not path.exists() and shutil.disk_usage(root).free < cfg["free_floor_bytes"]+cfg["aggregate_output_boundary_bytes"]+256*2**20:
            raise RuntimeError("Insufficient floor+bounded-output+temporary-margin capacity")
        save_json(path, cfg); frozen_inventory(source, root); started = perf_counter(); rows = []
        tasks = [(str(root), str(source), cfg, case) for case in cfg["case_ids"]]
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as executor:
            futures = {executor.submit(run_case, task): task for task in tasks}
            for future in as_completed(futures):
                rows.extend(future.result()); size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
                save_json(root/"execution-status.json", {"status": "RUNNING", "terminal": len(rows),
                           "planned": len(tasks)*len(variants("standard")), "output_bytes": size, "whole_study_complete": False})
                if size > cfg["aggregate_output_boundary_bytes"]:
                    for f in futures: f.cancel()
                    raise RuntimeError("Output boundary; no runningworker killed")
        result = {"status": "RAACS_CONSTRUCTION_EXECUTED_PENDING_AUDIT_AND_COMPARISON", "terminal": len(rows),
                  "planned": len(tasks)*len(variants("standard")), "statuses": dict(Counter(r["status"] for r in rows)),
                  "wall_seconds_this_attempt": perf_counter()-started, "hardware_shots": 0, "whole_study_complete": False}
        save_json(root/"execution-status.json", result)
        return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--phase", default="new_sampler"); p.add_argument("--mode", choices=["standard"], default="standard")
    p.add_argument("--source", default=str(ROOT/"results/standard-v1")); p.add_argument("--output", default=str(ROOT/"results/RAACS-standard-v1")); p.add_argument("--workers", type=int, default=2)
    a = p.parse_args(); print(json.dumps(run(a.output, a.source, a.workers)))


if __name__ == "__main__": main()
