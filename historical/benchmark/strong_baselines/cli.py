from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil

from ..common import ROOT, digest, file_digest, provenance, save_json


def one_case(case_id, options):
    import fcntl
    from threadpoolctl import threadpool_limits
    from ..chemistry import load_reference
    from ..config import Case
    from .lucj import method_specs, run_cell
    from pyscf import lib
    root = Path(options["output"])
    lock_path = root / "locks" / f"{case_id}.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock, threadpool_limits(limits=options["threads"]):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lib.num_threads(options["threads"])
        name, distance = case_id.split("_R")
        case = Case(name, float(distance))
        ref = load_reference(Path(options["source"]) / "references" / case_id)
        rows = []
        for spec in method_specs():
            if shutil.disk_usage(root).free < options["budget"]["free_floor_bytes"]:
                raise RuntimeError("Disk floor before new cell; completed cells retained")
            row = run_cell(case, ref, spec, options["source"], root, options["budget"])
            rows.append({k: row.get(k) for k in ("case_id", "method", "status", "energy_error_mHa", "objective_evaluations", "construction_wall_seconds")})
            print(json.dumps(rows[-1]), flush=True)
        save_json(root / "case-status" / f"{case_id}.json", {"status": "ALL_REGISTERED_B1_CELLS_TERMINAL", "cells": rows})
        return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["baselines"], required=True)
    p.add_argument("--mode", choices=["quick", "standard", "full"], default="quick")
    p.add_argument("--source", default=str(ROOT / "results/standard-v1"))
    p.add_argument("--output")
    p.add_argument("--config")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--component", choices=["lucj", "evaluate", "all"], default="all")
    args = p.parse_args(argv)
    if args.workers not in (1, 2) or args.threads < 1:
        p.error("Controller profile supports1–2 workers, positive threads")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    root = Path(args.output or ROOT / f"results/sampler-baselines-B1-{args.mode}-v1").resolve()
    source = Path(args.source).resolve()
    if root == source or source in root.parents:
        raise RuntimeError("Never write new study inside source")
    root.mkdir(parents=True, exist_ok=True)
    from .lucj import method_specs
    quick = args.mode == "quick"
    budget = {"maxiter": 4 if quick else 200, "max_evals": 300 if quick else 10000,
              "cell_seconds": 120 if quick else 600, "worker_ram_bytes": 2*2**30,
              "free_floor_bytes": 3*2**30, "output_cap_bytes": 6*2**30,
              "pilot_shots": 256 if quick else 1024, "qsci_proposals": 8 if quick else 64,
              "evaluation_repeats": 4 if quick else 20, "M_grid": [1000, 10000, 100000],
              "R_grid": [8, 16, 32, 64, 128, 256, 512]}
    if args.config:
        custom = json.loads(Path(args.config).read_text())
        if set(custom) - set(budget):
            raise RuntimeError("Unknown budget keys")
        budget.update(custom)
    cases = ["H2_R0.7", "H4_R1.1"] if quick else sorted(d.name for d in (source / "references").iterdir() if d.is_dir())
    if not quick:
        q = ROOT / "results/sampler-baselines-B1-quick-v1/validation.json"
        if not q.exists() or json.loads(q.read_text())["status"] != "QUICK_B1_VALIDATED":
            raise RuntimeError("Execute and validate quick B1 before standard")
    env = provenance()
    implementation = {str(f.relative_to(ROOT)): file_digest(f) for f in sorted((ROOT / "benchmark/strong_baselines").glob("*.py"))}
    # Bind shared numerical dependencies too; the Phase A package stays unchanged.
    for name in ("benchmark/lucj.py", "benchmark/ansatz.py", "benchmark/chemistry.py", "benchmark/study/core.py"):
        implementation[name] = file_digest(ROOT / name)
    manifest = {"schema": "sampler-baselines-B1-v1", "mode": args.mode, "source": str(source),
                "source_dataset": json.loads((source / "REPRODUCIBILITY.json").read_text())["dataset_id"],
                "case_ids": cases, "methods": method_specs(), "budget": budget, "implementation": implementation,
                "packages": env["packages"], "lock_sha256": env["lock_sha256"],
                "license_ffsim_sha256": file_digest(ROOT / ".venv/lib/python3.11/site-packages/ffsim-0.0.84.dist-info/licenses/LICENSE"),
                "protocol_sha256": file_digest(ROOT / "study-v1/BASELINES_B1.md"),
                "hardware_shots": 0, "scope": "B1 only; B2–B4 and other milestones remain required"}
    mf = root / "manifest.json"
    if mf.exists() and json.loads(mf.read_text()) != manifest:
        raise RuntimeError("Frozen B1 code/config changed; use successor output version")
    save_json(mf, manifest)
    options = {"source": str(source), "output": str(root), "budget": budget, "threads": args.threads}
    if args.component in ("lucj", "all"):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one_case, case, options): case for case in cases}
            for f in as_completed(futures):
                f.result()
    if args.component in ("evaluate", "all"):
        from .evaluate import evaluate
        evaluate(root)


if __name__ == "__main__":
    main()
