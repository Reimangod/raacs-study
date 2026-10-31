"""Bounded two-job CPU handoff, waiting for existing B2 rather than duplicating it."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep
import psutil
from .common import ROOT, save_json, file_digest


def sequence(existing_pid):
    out = ROOT / "results/baseline-successor-sequence-v1"; out.mkdir(exist_ok=True)
    b2 = ROOT / "results/sampler-baselines-B2-exactprep-standard-v2"
    b3q = ROOT / "results/nonvariational-B3-quick-v1"
    b3 = ROOT / "results/nonvariational-B3-standard-v1"
    def state(**kwargs):
        save_json(out / "status.json", {"whole_study_complete": False, **kwargs})
    def launch(name, args):
        log = (out / (name+".log")).open("a")
        p = subprocess.Popen([sys.executable, "-m", *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        save_json(out / (name+"-dispatch.json"), {"pid": p.pid, "command": [sys.executable, "-m", *args], "hardware_shots": 0})
        return p
    def wait_all(jobs):
        while any(p.poll() is None for _, p in jobs):
            state(stage="RUNNING_CHILDREN", jobs=[{"name": name, "pid": p.pid, "exit": p.poll()} for name, p in jobs])
            sleep(10)
        failed = [{"name": n, "exit": p.returncode} for n, p in jobs if p.returncode]
        if failed:
            raise RuntimeError("Child failures preserved: "+json.dumps(failed))
    deadline = monotonic()+4*3600
    while len(list(b2.glob("trajectories/*/*/terminal.json"))) < 360:
        terminal = len(list(b2.glob("trajectories/*/*/terminal.json")))
        failures = len(list(b2.glob("trajectories/*/*/failure.json")))
        if failures or monotonic() > deadline or not psutil.pid_exists(existing_pid):
            raise RuntimeError(f"Existing native job needs diagnosis; {terminal}/360, failures{failures}; no duplicate launch")
        state(stage="WAITING_EXISTING_NATIVE_GROWTH", pid=existing_pid, available=terminal, expected=360)
        sleep(10)
    state(stage="AUDITING_360_NATIVE_TRAJECTORIES")
    wait_all([("B2-exact-audit", launch("B2-exact-audit", ["benchmark.qsci_exact_init_audit", "--output", str(b2)]))])
    wait_all([("B3-quick", launch("B3-quick", ["benchmark.nonvariational_baselines", "--mode", "quick", "--workers", "1"]))])
    wait_all([("B3-quick-audit", launch("B3-quick-audit", ["benchmark.nonvariational_audit", "--output", str(b3q)]))])
    # Exactly two independent numerical processes: one readout, one case worker.
    wait_all([("B2-independent-readout", launch("B2-independent-readout", ["benchmark.qsci_exact_init_evaluate", "--output", str(b2)])),
              ("B3-standard", launch("B3-standard", ["benchmark.nonvariational_baselines", "--mode", "standard", "--workers", "1"]))])
    wait_all([("B3-standard-audit", launch("B3-standard-audit", ["benchmark.nonvariational_audit", "--output", str(b3)]))])
    # Uses the existing qualified common-HF engine, not a fake SqDRIFT evaluator.
    wait_all([("B3-independent-readout", launch("B3-independent-readout", ["benchmark.evaluate_fixed_cohort", "--output", str(b3)]))])
    state(stage="NUMERICAL_SUCCESSOR_COHORTS_COMPLETE_REPORTS_PENDING", whole_milestone2_complete=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--existing-pid", type=int, required=True)
    args = p.parse_args()
    try:
        sequence(args.existing_pid)
    except Exception as exc:
        save_json(ROOT / "results/baseline-successor-sequence-v1/error.json", {"error": repr(exc), "whole_study_complete": False})
        raise
