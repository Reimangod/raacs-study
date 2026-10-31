from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import traceback
from time import perf_counter
from .common import ROOT, save_json, provenance,digest
import json
from .config import cases_for, Case


def one_case(case, options):
    import fcntl
    from .task_registry import Registry
    record=Registry(ROOT/"results/task-registry.sqlite").get({"kind":"molecular-case-production-1","case_id":case.id,
                      "output":str(Path(options["output"]).resolve())})
    if record and record[2] not in ("COMPLETED","FAILED_PRESERVED"):
        raise RuntimeError(f"Case already owned by {record[1]} ({record[2]}); reconcile remote job, do not duplicate")
    path=Path(options["output"])/"locks"/f"{case.id}.lock"
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            return _one_case_locked(case,options)
        except Exception as error:
            result={"case":case.id,"status":"FAILED","error":repr(error),"traceback":traceback.format_exc()}
            save_json(Path(options["output"])/"failures"/case.id/"case.json",result)
            return result


def _one_case_locked(case, options):
    from threadpoolctl import threadpool_limits
    from .chemistry import build_reference
    from .adapt import run_adapt
    with threadpool_limits(limits=options["threads"]):
        reference = build_reference(case, options["output"], threads=options["threads"])
        result = {}
        for method in options["methods"]:
            failure_path=Path(options["output"])/"failures"/case.id/f"{method}.json"
            if failure_path.exists() and not options.get("retry_failed",False):
                status_path=Path(options["output"])/"checkpoints"/case.id/method/"status.json"
                successful=(status_path.exists() and json.loads(status_path.read_text()).get("status") not in ("RUNNING","FAILED"))
                if not successful:
                    result[method]=json.loads(failure_path.read_text())
                    continue
            try:
                attempt_started=perf_counter()
                if method.startswith("lucj_"):
                    from .lucj import run_lucj
                    result[method]=run_lucj(case,reference,method,options["output"])
                elif method in ("hf","ceo_random_101"):
                    from .baselines import run_baseline
                    result[method]=run_baseline(case,reference,method,options["output"])
                else:
                    result[method] = run_adapt(reference, method, options["output"],
                                           max_steps=options["max_steps"],
                                           task_seconds=options["task_seconds"],
                                           allow_extension=options.get("extend_max_steps",False))
            except Exception as e:
                result[method] = {"status": "FAILED", "error": repr(e), "traceback": traceback.format_exc(),
                                  "failed_attempt_seconds":perf_counter()-attempt_started}
                save_json(Path(options["output"])/"failures"/case.id/f"{method}.json", result[method])
        if options["finite_shots"]:
            from .qsci import run_qsci
            result["finite_qsci"] = run_qsci(reference,options["output"],options["methods"],
                                               repeats=options["repeats"],ci_size=options["ci_size"])
        return {"case": case.id, "methods": result}


def main():
    import sys
    if "--phase" in sys.argv or any(x.startswith("--phase=") for x in sys.argv):
        phase = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--phase=")), None)
        if phase is None:
            phase = sys.argv[sys.argv.index("--phase") + 1]
        if phase in ("scaling", "early_stopping"):
            # Additive routing only: historical commands without --phase retain
            # their old implementation. These dedicated CLIs do not consume it.
            remaining = [sys.argv[0]]
            i = 1
            while i < len(sys.argv):
                if sys.argv[i] == "--phase": i += 2; continue
                if sys.argv[i].startswith("--phase="): i += 1; continue
                remaining.append(sys.argv[i]); i += 1
            previous = sys.argv; sys.argv = remaining
            try:
                if phase == "scaling":
                    from .molecular_mps_scaling import main as selected_main
                else:
                    from .actual_growth_cli import main as selected_main
                return selected_main()
            finally:
                sys.argv = previous
        if phase == "new_sampler":
            mode = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--mode=")), None)
            if mode is None and "--mode" in sys.argv:
                mode = sys.argv[sys.argv.index("--mode")+1]
            if mode == "standard":
                from .raacs_standard_v2 import main as sampler_main
                return sampler_main()
            from .raacs_cli import main as sampler_main
            return sampler_main()
        if phase == "baselines":
            component = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--component=")), None)
            if component is None and "--component" in sys.argv:
                component = sys.argv[sys.argv.index("--component") + 1]
            if component == "adapt_qsci":
                from .qsci_growth import main as qsci_main
                return qsci_main()
            from .strong_baselines.cli import main as baseline_main
            return baseline_main()
        from .study.cli import main as study_main
        return study_main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "standard", "full"], default="quick")
    parser.add_argument("--output", default="results/quick")
    parser.add_argument("--methods", nargs="+", default=["ceo_ovp", "ceo_mvp", "qeb", "qubit", "gsd",
                                                        "lucj_1", "lucj_2", "lucj_truncated", "lucj_random_101",
                                                        "hf","ceo_random_101"])
    parser.add_argument("--cases", nargs="+", help="e.g. H4:1.1 LiH:1.5; explicit subset is reported")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--extend-max-steps",action="store_true",help="Explicitly extend only a previous MAX_STEPS cap; preserve old checkpoints")
    parser.add_argument("--retry-failed",action="store_true",help="Explicit retry; normal resume preserves failed outcomes without re-running")
    parser.add_argument("--exclude-molecules",nargs="*",default=[],help="Execution allocation only; full cohort remains visible as missing")
    parser.add_argument("--sqd",action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--task-seconds", type=float, default=600)
    parser.add_argument("--finite-shots", action="store_true",default=True)
    parser.add_argument("--analytic-only",dest="finite_shots",action="store_false")
    parser.add_argument("--repeats",type=int,default=20)
    parser.add_argument("--ci-size",type=int,default=32)
    args = parser.parse_args()
    cases = [Case(m, float(r)) for m,r in (s.split(":") for s in args.cases)] if args.cases else cases_for(args.mode)
    cohort=list(cases)
    cases=[c for c in cases if c.molecule not in args.exclude_molecules]
    if len({c.id for c in cases}) != len(cases):
        parser.error("Duplicate case IDs are not allowed")
    options = vars(args).copy()
    options["max_steps"] = args.max_steps if args.max_steps is not None else {"quick": 8,"standard":32,"full":64}[args.mode]
    options["output"] = str(Path(args.output).resolve())
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    environment=provenance()
    for name in ("environment.json","request.json","run-status.json"):
        path=Path(args.output)/name
        if path.exists():
            old=json.loads(path.read_text())
            save_json(Path(args.output)/"history"/(name.removesuffix(".json")+"-"+digest(old)+".json"),old)
    save_json(Path(args.output)/"environment.json", environment)
    save_json(Path(args.output)/"request.json", {"options": options, "cases": [c.specification() for c in cases],
                                               "cohort_cases":[c.specification() for c in cohort],
                                               "planned_full_case_count": 30,
                                               "stage": "ANALYTIC_PLUS_FINITE_QSCI" if args.finite_shots else "ANALYTIC_ONLY"})
    if args.workers == 1:
        result = [one_case(c, options) for c in cases]
    else:
        result = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            tasks = {executor.submit(one_case,c,options):c for c in cases}
            for task in as_completed(tasks):
                try:
                    result.append(task.result())
                except Exception as e:
                    result.append({"case": tasks[task].id, "status": "FAILED", "error": repr(e)})
    save_json(Path(args.output)/"run-status.json", result)
    if args.sqd is True or (args.sqd is None and args.mode=="full"):
        from .sqd import run_sqd
        run_sqd(args.output)
    from .report import make_report
    from .plot import plot_all
    make_report(args.output)
    plot_all(args.output)
    if any(any(v.get("status")=="FAILED" for v in r.get("methods",{}).values()) or r.get("status")=="FAILED" for r in result):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

