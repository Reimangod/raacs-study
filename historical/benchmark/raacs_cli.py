"""First actual RAACS qualification cohort; standard is not silently emulated."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from time import perf_counter
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, save_json
from .circuit_sampling_runtime import SamplingSpec
from .resource_adaptive_sampler import RAACSSpec, SCORES, run_sampler
from .study.reanalysis import frozen_inventory


def quick(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source or source in output.parents: raise ValueError("Protect frozen source")
    output.mkdir(exist_ok=True, parents=True)
    cfg = {"schema": "RAACS-quick-qualification-v1", "cases": ["H2_R0.7", "H4_R1.1"], "source": str(source),
           "scores": list(SCORES), "total_model_shots": 4000, "max_steps": 4, "seed": 101,
           "exact_small_cells": 10, "MPS_cells": 2,
           "spec_sha256": file_digest(ROOT/"study-v1/RAACS_PROTOTYPE.md"),
           "implementation": {p: file_digest(ROOT/p) for p in ("benchmark/raacs_cli.py", "benchmark/resource_adaptive_sampler.py",
              "benchmark/local_sampler_primitives.py", "benchmark/circuit_sampling_runtime.py", "benchmark/integral_connections.py",
              "benchmark/integral_hamiltonian.py", "benchmark/mps_sampling.py", "benchmark/ansatz.py")}}
    path = output/"manifest.json"
    if path.exists() and json.loads(path.read_text()) != cfg: raise RuntimeError("Qualification code drift; preserve output and use successor")
    save_json(path, cfg); frozen_inventory(source, output)
    rows = []; started = perf_counter()
    with threadpool_limits(limits=1):
        for case in cfg["cases"]:
            tasks = [(s, "exact_small") for s in SCORES]+[("full_cost", "mps")]
            for score, backend in tasks:
                if shutil.disk_usage(output).free < 3*2**30: raise RuntimeError("Free-floor boundary; preserve completed qualification cells")
                spec = RAACSSpec(score=score, total_model_shots=4000, max_steps=4)
                result = run_sampler(source/"references"/case, output/"cells"/case/(score+"-"+backend), spec, SamplingSpec(backend=backend, max_bond=32))
                rows.append({"case_id": case, "score": score, "backend": backend, "result": result})
                save_json(output/"execution-status.json", {"status": "RUNNING", "completed": len(rows), "planned": 12, "whole_study_complete": False})
                print(f"RAACS {case} {score} {backend} {result['status']} E_QSCI={result['final_QSCI_energy_Ha']:.10f}", flush=True)
    status = {"status": "RAACS_QUICK_EXECUTED_PENDING_INDEPENDENT_AUDIT", "cells": len(rows),
              "actual_model_shots": sum(r["result"]["actual_total_model_shots"] for r in rows),
              "wall_seconds": perf_counter()-started, "hardware_shots": 0, "FCI_used": False, "whole_study_complete": False}
    save_json(output/"execution-status.json", status)
    return status


def main():
    p = argparse.ArgumentParser(); p.add_argument("--phase", choices=["new_sampler"], default="new_sampler")
    p.add_argument("--mode", choices=["quick", "standard"], default="quick"); p.add_argument("--output"); p.add_argument("--source", default=str(ROOT/"results/standard-v1"))
    a = p.parse_args()
    if a.mode != "quick": raise NotImplementedError("Qualify and audit the actual quick sampler before freezing the standard cohort")
    print(json.dumps(quick(a.output or ROOT/"results/RAACS-quick-v1", a.source)))


if __name__ == "__main__": main()
