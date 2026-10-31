"""Versioned extended qualification and full30condition RAACS construction."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import shutil
from time import perf_counter
import traceback
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, save_json
from .resource_adaptive_sampler import RAACSSpec, SCORES, run_sampler
from .circuit_sampling_runtime import SamplingSpec
from .study.reanalysis import frozen_inventory


def variants(mode):
    if mode == "qualify_variants":
        return [("full_cost", "fixed_angle", "last_only", 101, "exact_small")]+[
            ("full_cost", "qsci_local_grid", a, 101, "exact_small") for a in ("uniform", "discovery", "energy_guided")]
    v = [(s, "qsci_local_grid", "last_only", seed, "exact_small") for s in SCORES for seed in (101, 211, 307)]
    v += [("full_cost", "fixed_angle", "last_only", seed, "exact_small") for seed in (101, 211, 307)]
    v += [("full_cost", "qsci_local_grid", a, seed, "exact_small") for a in ("uniform", "discovery", "energy_guided") for seed in (101, 211, 307)]
    return v+[("full_cost", "qsci_local_grid", "last_only", 101, "mps")]


def run_case(task):
    root, source, cfg, case = task; root, source = Path(root), Path(source); results = []
    with threadpool_limits(limits=1):
        for score, angle, allocation, seed, backend in variants(cfg["mode"]):
            name = f"{score}-{angle}-{allocation}-s{seed}-{backend}"
            cell = root/"cells"/case/name
            if shutil.disk_usage(root).free < cfg["free_floor_bytes"]:
                raise RuntimeError("Case boundary freefloor; preserve partialcohort")
            if list(cell.glob("failure-*.json")):
                results.append({"case_id": case, "method": name, "status": "FAILED_PRESERVED_NO_AUTOMATIC_RERUN"}); continue
            try:
                spec = RAACSSpec(score=score, angle_mode=angle, final_allocation=allocation,
                    total_model_shots=cfg["total_model_shots"], max_steps=cfg["max_steps"], seed=seed)
                r = run_sampler(source/"references"/case, cell, spec, SamplingSpec(backend=backend, max_bond=cfg["MPS_max_bond"]))
                row = {"case_id": case, "method": name, "status": r["status"], "energy_Ha": r["final_QSCI_energy_Ha"],
                       "accepted_steps": r["accepted_steps"], "actual_raw_model_shots": r["actual_total_model_shots"]}
            except Exception as exc:
                cell.mkdir(parents=True, exist_ok=True)
                from time import time_ns
                row = {"case_id": case, "method": name, "status": "FAILED_ENGINEERING_PRESERVED",
                       "error": str(exc), "traceback": traceback.format_exc(), "failed_cost_not_assumed_zero": True}
                save_json(cell/f"failure-{time_ns()}.json", row)
            results.append(row)
            save_json(root/"case-status"/f"{case}.json", {"terminal": len(results), "planned": len(variants(cfg["mode"])), "rows": results})
            print("RAACS construction "+case+" "+name+" "+row["status"], flush=True)
    return results


def run(root, source, mode, workers=2):
    root, source = Path(root).resolve(), Path(source).resolve(); root.mkdir(parents=True, exist_ok=True)
    if root == source or source in root.parents: raise ValueError("Protect source dataset")
    if workers not in (1, 2): raise ValueError("At most2numericalworkers")
    cfg = {"schema": "RAACS-standard-construction-v1", "mode": mode, "source": str(source),
           "case_ids": ["H2_R0.7", "H4_R1.1"] if mode == "qualify_variants" else sorted(p.name for p in (source/"references").iterdir() if p.is_dir()),
           "total_model_shots": 4000 if mode == "qualify_variants" else 10000,
           "max_steps": 4 if mode == "qualify_variants" else 8, "MPS_max_bond": 32 if mode == "qualify_variants" else 64,
           "variants": variants(mode), "exact_small_cells": 8 if mode == "qualify_variants" else 810,
           "MPS_cells": 0 if mode == "qualify_variants" else 30,
           "spec_sha256": file_digest(ROOT/"study-v1/RAACS_STANDARD_EXECUTION.md"),
           "implementation": {p: file_digest(ROOT/p) for p in ("benchmark/raacs_standard.py", "benchmark/resource_adaptive_sampler.py",
               "benchmark/local_sampler_primitives.py", "benchmark/circuit_sampling_runtime.py", "benchmark/integral_hamiltonian.py",
               "benchmark/integral_connections.py", "benchmark/mps_sampling.py", "benchmark/ansatz.py")},
           "free_floor_bytes": 3*2**30, "aggregate_output_boundary_bytes": 768*2**20, "hardware_shots": 0}
    if mode == "standard":
        for name in ("RAACS-quick-v1", "RAACS-variants-quick-v1"):
            p = ROOT/"results"/name/"validation.json"
            if not p.exists() or json.loads(p.read_text())["status"] != "RAACS_QUICK_COUNTS_PHYSICS_AND_FINAL_SUPPORT_VALIDATED":
                raise RuntimeError("Actual quick variants and independent audit required")
    path = root/"manifest.json"
    if path.exists() and json.loads(path.read_text()) != cfg: raise RuntimeError("Standard construction config drift")
    if not path.exists() and shutil.disk_usage(root).free < cfg["free_floor_bytes"]+cfg["aggregate_output_boundary_bytes"]+256*2**20:
        raise RuntimeError("Free floor plus bounded output and temporary margin not available")
    save_json(path, cfg); frozen_inventory(source, root); started = perf_counter(); rows = []
    tasks = [(str(root), str(source), cfg, case) for case in cfg["case_ids"]]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as executor:
        futures = {executor.submit(run_case, task): task for task in tasks}
        for future in as_completed(futures):
            rows.extend(future.result())
            size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            save_json(root/"execution-status.json", {"status": "RUNNING", "terminal": len(rows),
                       "planned": len(tasks)*len(variants(mode)), "output_bytes": size, "whole_study_complete": False})
            if size > cfg["aggregate_output_boundary_bytes"]:
                for f in futures: f.cancel()
                raise RuntimeError("Aggregate output boundary; no running worker killed")
    result = {"status": "RAACS_CONSTRUCTION_EXECUTED_PENDING_AUDIT_AND_COMPARISON", "terminal": len(rows),
              "planned": len(tasks)*len(variants(mode)), "statuses": dict(__import__("collections").Counter(r["status"] for r in rows)),
              "wall_seconds": perf_counter()-started, "hardware_shots": 0, "whole_study_complete": False}
    save_json(root/"execution-status.json", result)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--phase", default="new_sampler")
    p.add_argument("--mode", choices=["qualify_variants", "standard"], default="standard")
    p.add_argument("--output"); p.add_argument("--source", default=str(ROOT/"results/standard-v1")); p.add_argument("--workers", type=int, default=2)
    a = p.parse_args(); name = "RAACS-variants-quick-v1" if a.mode == "qualify_variants" else "RAACS-standard-v1"
    print(json.dumps(run(a.output or ROOT/"results"/name, a.source, a.mode, a.workers)))


if __name__ == "__main__": main()
