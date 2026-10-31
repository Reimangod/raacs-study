"""Sequential B1->B2 execution; avoids competing two-worker production pools."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import ROOT, save_json


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    result = subprocess.run(["rtk", "proxy", "ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    return "benchmark.run --phase baselines --mode standard" in result.stdout


def main():
    p = argparse.ArgumentParser(); p.add_argument("--wait-b1-pid", type=int, required=True)
    args = p.parse_args()
    root = ROOT / "results/baseline-sequence-v1"; root.mkdir(exist_ok=True)
    import fcntl
    with (root / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save_json(root / "identity.json", {"pid": os.getpid(), "wait_b1_pid": args.wait_b1_pid,
                  "scope": "B1 construction/readout/audit/report then B2 growth/readout/audit; not full study completion"})
        b1 = ROOT / "results/sampler-baselines-B1-standard-v1"
        stage_history = []
        def status(stage, **extra):
            save_json(root / "status.json", {"stage": stage, "stage_history": stage_history,
                      "pid": os.getpid(), "whole_study_complete": False, **extra})
        started = time.monotonic(); status("WAITING_FOR_EXISTING_B1_DO_NOT_DUPLICATE")
        while not (b1 / "evaluation-status.json").exists():
            if not alive(args.wait_b1_pid):
                status("B1_PROCESS_ENDED_BEFORE_FULL_READOUT", engineering_issue=True)
                raise RuntimeError("Existing B1 ended without readout completion; inspect its evidence, no free retry")
            if time.monotonic()-started > 6*3600:
                status("WAIT_SLICE_EXPIRED_B1_NOT_STOPPED", engineering_issue=False)
                return
            time.sleep(15)
        def run(name, module, *arguments):
            stamp = time.monotonic(); status(name)
            with (root / f"{name}.log").open("a") as stream:
                result = subprocess.run(["rtk", "proxy", sys.executable, "-m", module, *arguments], cwd=ROOT,
                                        stdout=stream, stderr=subprocess.STDOUT)
            record = {"stage": name, "exit_code": result.returncode, "wall_seconds": time.monotonic()-stamp}
            stage_history.append(record)
            print(json.dumps(record), flush=True)
            if result.returncode:
                status(name+"_REQUIRES_INSPECTION", engineering_issue=True)
                raise RuntimeError(f"{name} returned{result.returncode}; retained log and artifacts")
        run("B1_AUDIT", "benchmark.strong_baselines.audit", "--output", str(b1))
        run("B1_REPORT_FIGURES", "benchmark.b1_report", "--output", str(b1))
        b2 = ROOT / "results/sampler-baselines-B2-standard-v1"
        run("B2_NATIVE_GROWTH", "benchmark.run", "--phase", "baselines", "--component", "adapt_qsci", "--mode", "standard", "--workers", "2")
        run("B2_GROWTH_AUDIT", "benchmark.qsci_growth_audit", "--output", str(b2))
        run("B2_INDEPENDENT_READOUT", "benchmark.qsci_growth_readout", "--output", str(b2))
        status("B1_B2_SEQUENCE_FINISHED_REMAINING_MILESTONES_REQUIRED", engineering_issue=False)


if __name__ == "__main__":
    main()
