from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..common import ROOT, digest, file_digest, save_json
from .core import CAPS, NAMESPACE, SHOTS


def main(argv=None):
    p = argparse.ArgumentParser(description="Versioned sampler study; frozen standard-v1 is read-only")
    p.add_argument("--phase", choices=["reanalysis", "baselines", "early_stopping", "new_sampler", "scaling", "sqd_noise"], required=True)
    p.add_argument("--mode", choices=["quick", "standard", "full"], default="quick")
    p.add_argument("--source", default=str(ROOT / "results/standard-v1"))
    p.add_argument("--output")
    p.add_argument("--config")
    p.add_argument("--component", choices=["all", "static", "finite", "mixtures", "analyze"], default="all")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args(argv)
    if args.phase != "reanalysis":
        raise SystemExit(f"{args.phase}: method-native implementation not yet available; do not substitute a proxy. Milestone 1 must run first.")
    output = Path(args.output or ROOT / f"results/sampler-study-{args.mode}-v1").resolve()
    output.mkdir(parents=True, exist_ok=True)
    options = {"schema": "sampler-study-A-v1", "source": str(Path(args.source).resolve()), "output": str(output),
               "mode": args.mode, "workers": args.workers, "threads": args.threads,
               "repeats": 4 if args.mode == "quick" else 20, "joint_mc_trials": 64 if args.mode == "quick" else 256,
               "case_seconds": 1800, "worker_ram_bytes": 2 * 2**30, "output_cap_bytes": 6 * 2**30,
               "free_floor_bytes": 3 * 2**30, "namespace": NAMESPACE + ("-quick" if args.mode == "quick" else "-standard"),
               "R_grid": list(CAPS), "M_grid": list(SHOTS), "quick_scope": "H2_R0.7,H4_R1.1; first2 and final ADAPT checkpoints; all fixed controls" if args.mode == "quick" else None}
    if args.config:
        config = json.loads(Path(args.config).read_text())
        unknown = set(config) - {"workers", "threads", "case_seconds", "worker_ram_bytes", "output_cap_bytes", "free_floor_bytes"}
        if unknown:
            p.error(f"Unknown configuration keys: {unknown}")
        options.update(config)
    if options["workers"] not in (1, 2) or options["threads"] < 1:
        p.error("This 16-GiB controller profile supports1–2 workers; threads must be positive")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(options["threads"])
    from .analysis import analyze
    from .reanalysis import frozen_inventory, launch_finite, operator_tracking, retrospective
    # Scientific config and code identity are immutable per new output version.
    implementation = {str(f.relative_to(ROOT)): file_digest(f) for f in sorted((ROOT / "benchmark/study").glob("*.py"))}
    manifest = {**options, "implementation": implementation, "goal_source": "/Users/rei/.codex/attachments/662165c0-336c-4a5f-88b9-6719d6fe76ce/goal-objective.md",
                "scope": "new independent reanalysis; all old results preserved; oracle diagnostics separate", "hardware_quantum_shots": 0}
    mf = output / "manifest.json"
    if mf.exists():
        old = json.loads(mf.read_text())
        if old != manifest:
            raise RuntimeError("Frozen reanalysis config/code changed; use a new versioned output directory")
    else:
        save_json(mf, manifest)
    records, pointer = frozen_inventory(args.source, output)
    chosen = records
    if args.mode == "quick":
        chosen = []
        for case in ("H2_R0.7", "H4_R1.1"):
            for method in sorted({r["method"] for r in records if r["case_id"] == case}):
                rs = [r for r in records if r["case_id"] == case and r["method"] == method]
                steps = sorted({r["step"] for r in rs})
                chosen += [r for r in rs if r["step"] in set(steps[:2] + steps[-1:])]
    save_json(output / "execution-queue.json", {"source_dataset": pointer["dataset_id"], "source_checkpoints": len(records),
              "selected_checkpoints": len(chosen), "case_count": len({r["case_id"] for r in chosen}), "queue": [r["checkpoint_id"] for r in chosen],
              "selection": options["quick_scope"] or "all5236 including5 historical fixed revisions; primary analyses flag5231"})
    if args.component in ("all", "static"):
        print("Retrospective rows:", retrospective(args.source, output), flush=True)
        print("Operator tracking:", operator_tracking(args.source, output, records), flush=True)
    if args.component in ("all", "finite"):
        result = launch_finite(chosen, options)
        save_json(output / "finite-run-status.json", result)
        print(json.dumps([{k: v for k, v in r.items() if k not in ("traceback",)} for r in result]), flush=True)
    if args.component in ("all", "mixtures"):
        from .mixtures import run_mixtures
        run_mixtures(args.source, output, chosen, options)
    if args.component in ("all", "finite", "mixtures", "analyze"):
        print(json.dumps(analyze(output)), flush=True)


if __name__ == "__main__":
    main()
