"""Finish actual C evaluation after an explicitly identified live constructor.

Waiting observes the same handle/command; it never restarts a timed-out worker.
This driver executes all actual-count audits/reports, not a synthetic proxy.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
from time import sleep
from .common import ROOT, file_digest, save_json


def run(pid=None):
    root = ROOT/"results/PhaseC-actual-growth-sequence-v1"; root.mkdir(exist_ok=True)
    files = ("benchmark/actual_growth_sequence.py", "benchmark/actual_growth_audit.py", "benchmark/actual_growth_evaluate.py",
             "benchmark/actual_growth_reference.py", "benchmark/actual_growth_report.py")
    binding = {"implementation": {p: file_digest(ROOT/p) for p in files}, "watched_constructor_PID": pid}
    if (root/"binding.json").exists() and json.loads((root/"binding.json").read_text()) != binding:
        raise RuntimeError("Sequence binding drift")
    save_json(root/"binding.json", binding)
    if pid:
        while True:
            try: os.kill(pid, 0)
            except ProcessLookupError: break
            command = subprocess.run(["ps", "-p", str(pid), "-o", "stat=", "-o", "command="], capture_output=True, text=True).stdout
            if not command.strip():
                continue  # A process may exit between kill(0) and ps; recheck.
            if command.strip().split()[0].startswith("Z"):
                break  # Reaped/terminal worker, not an observation timeout.
            if "benchmark.actual_growth_cohort" not in command:
                raise RuntimeError("Watched PID command changed; do not infer completion/restart")
            save_json(root/"status.json", {"stage": "WAITING_VERIFIED_CONSTRUCTOR", "pid": pid, "command": command.strip(), "whole_study_complete": False})
            sleep(10)
    standard = ROOT/"results/PhaseC-actual-growth-standard-v1"
    execution = json.loads((standard/"execution-status.json").read_text())
    if execution["status"] != "ACTUAL_GROWTH_EXECUTED_PENDING_AUDIT" or execution["failed"]:
        raise RuntimeError("Actual construction unfinished/failed; preserve evidence, no unqualified final readout")
    import sys
    def command(module, output, extra=()):
        args = [sys.executable, "-m", module, "--output", str(output), *extra]
        stage = module+" "+" ".join(extra)+" "+output.name
        save_json(root/"status.json", {"stage": stage, "whole_study_complete": False})
        path = root/(module.split(".")[-1]+"-"+output.name+("-audit" if extra else "")+".log")
        with path.open("a") as log:
            result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            save_json(root/"status.json", {"stage": "FAILED_PRESERVED", "command": args, "returncode": result.returncode, "log": str(path), "whole_study_complete": False})
            raise RuntimeError("Actual C sequence stage failed: "+stage)
    quick = ROOT/"results/PhaseC-actual-growth-quick-v1"
    command("benchmark.actual_growth_evaluate", quick)
    command("benchmark.actual_growth_evaluate", quick, ("--audit",))
    command("benchmark.actual_growth_reference", quick)
    command("benchmark.actual_growth_report", quick)
    command("benchmark.actual_growth_audit", standard)
    command("benchmark.actual_growth_evaluate", standard)
    command("benchmark.actual_growth_evaluate", standard, ("--audit",))
    command("benchmark.actual_growth_reference", standard)
    command("benchmark.actual_growth_report", standard)
    result = {"stage": "ACTUAL_FRESH_GROWTH_EVALUATED_AUDITED_REPORTED", "hardware_shots": 0,
              "remaining": "RAACS, large molecular matrix-free/MPS scaling, full SQD/noise, baseline gaps and final requirement audit",
              "whole_study_complete": False}
    save_json(root/"status.json", result)
    return result


if __name__ == "__main__":
    import fcntl
    p = argparse.ArgumentParser(); p.add_argument("--wait-constructor-pid", type=int)
    args = p.parse_args(); directory = ROOT/"results/PhaseC-actual-growth-sequence-v1"; directory.mkdir(exist_ok=True)
    with (directory/"run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(run(args.wait_constructor_pid)))
