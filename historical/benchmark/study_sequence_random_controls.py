"""Continue existing random-control work to audit/readout/report, without duplication."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep
import psutil
from .common import ROOT, save_json


def run(pid):
    root = ROOT / "results/random-B4-standard-v1"
    output = ROOT / "results/random-B4-sequence-v1"; output.mkdir(exist_ok=True)
    until = monotonic()+4*3600
    while len(list(root.glob("cells/*/*/result.json"))) < 390:
        n = len(list(root.glob("cells/*/*/result.json")))
        if not psutil.pid_exists(pid) or monotonic() > until:
            raise RuntimeError(f"Existing random worker stopped before fullcoverage {n}/390; preserve and diagnose, do not duplicate")
        save_json(output / "status.json", {"stage": "WAITING_EXISTING_RANDOM_WORKER", "terminal_cells": n, "whole_study_complete": False})
        sleep(10)
    commands = [("audit", "benchmark.random_structure_audit", ["--output", str(root)]),
                ("readout", "benchmark.evaluate_fixed_cohort", ["--output", str(root)]),
                ("report", "benchmark.random_structure_report", ["--output", str(root)])]
    for name, module, args in commands:
        with (output / (name+".log")).open("a") as log:
            child = subprocess.Popen([sys.executable, "-m", module, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            save_json(output / "status.json", {"stage": name, "pid": child.pid, "whole_study_complete": False})
            sleep(10)
        if child.returncode:
            raise RuntimeError(f"Random-control {name} failed with exit{child.returncode}; see preserved log")
    save_json(output / "status.json", {"stage": "RANDOM_COHORT_AUDITED_EVALUATED_REPORTED", "whole_study_complete": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--existing-pid", type=int, required=True)
    a = p.parse_args()
    try:
        run(a.existing_pid)
    except Exception as exc:
        save_json(ROOT / "results/random-B4-sequence-v1/error.json", {"error": repr(exc), "whole_study_complete": False})
        raise
