"""Serial verified handoff: D audit then actual E molecular/MPS execution.

Never starts a third numerical worker, restarts a live constructor, or treats
an observation timeout as terminal. A failed stage leaves durable logs.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep
from .common import ROOT, file_digest, save_json


def run(pid):
    root = ROOT/"results/RAACS-to-molecular-scaling-sequence-v1"; root.mkdir(exist_ok=True)
    files = ("raacs_scaling_sequence.py", "raacs_counts_audit.py", "raacs_standard_audit.py",
             "molecular_mps_scaling.py", "molecular_scaling_audit.py", "molecular_scaling_report.py")
    binding = {"watched_constructor_PID": pid, "files": {p: file_digest(ROOT/"benchmark"/p) for p in files}}
    with (root/"run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/"binding.json").exists() and json.loads((root/"binding.json").read_text()) != binding:
            raise RuntimeError("Sequence version/code binding changed")
        save_json(root/"binding.json", binding)
        while True:
            try: os.kill(pid, 0)
            except ProcessLookupError: break
            command = subprocess.run(["ps", "-p", str(pid), "-o", "stat=", "-o", "command="], capture_output=True, text=True).stdout.strip()
            if not command: continue
            if command.split()[0].startswith("Z"): break
            if "benchmark.run --phase new_sampler --mode standard" not in command:
                raise RuntimeError("Watched PID identity changed")
            save_json(root/"status.json", {"stage": "WAITING_VERIFIED_LIVE_D_CONSTRUCTOR", "pid": pid,
                                           "command": command, "whole_study_complete": False})
            sleep(10)
        d = ROOT/"results/RAACS-standard-v1"; state = json.loads((d/"execution-status.json").read_text())
        if state["terminal"] != 840 or state["status"] != "RAACS_CONSTRUCTION_EXECUTED_PENDING_AUDIT_AND_COMPARISON":
            raise RuntimeError("D constructor did not reach the authoritative840terminal state")
        def stage(name, args):
            marker = root/f"{name}-done.json"
            if marker.exists():
                if json.loads(marker.read_text())["command"] != args: raise RuntimeError("Completed stage identity changed")
                return
            save_json(root/"status.json", {"stage": name, "command": args, "whole_study_complete": False})
            env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}
            with (root/f"{name}.log").open("a") as log:
                code = subprocess.run(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                save_json(root/"status.json", {"stage": "FAILED_STAGE_PRESERVED", "failed_stage": name, "returncode": code,
                                               "log": str(root/f"{name}.log"), "whole_study_complete": False})
                raise RuntimeError("Sequence stage failed: "+name)
            save_json(marker, {"command": args, "returncode": code, "log_sha256": file_digest(root/f"{name}.log")})
        stage("qualification-tests", [sys.executable, "-m", "pytest", "benchmark/tests/test_molecular_mps_scaling.py",
              "benchmark/tests/test_resource_adaptive_sampler.py", "benchmark/tests/test_raacs_counts_audit.py", "-q"])
        stage("standard-D-count-audit", [sys.executable, "-m", "benchmark.raacs_standard_audit", "--output", str(d)])
        for mode in ("quick", "standard"):
            output = ROOT/f"results/molecular-MPS-scaling-{mode}-v1"
            stage(f"scaling-{mode}", [sys.executable, "-m", "benchmark.molecular_mps_scaling", "--mode", mode, "--output", str(output)])
            stage(f"scaling-{mode}-audit", [sys.executable, "-m", "benchmark.molecular_scaling_audit", "--output", str(output)])
            stage(f"scaling-{mode}-report", [sys.executable, "-m", "benchmark.molecular_scaling_report", "--output", str(output)])
        save_json(root/"status.json", {"stage": "D_COUNTS_AUDITED_MOLECULAR_SCALING_EXECUTED_AUDITED_REPORTED",
             "remaining": "D independent strong-baseline comparison/advanced controls; E strong large comparators; F fullSQD/noise; B multi-orbital bases; final full audit",
             "whole_study_complete": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--wait-constructor-pid", type=int, required=True)
    run(p.parse_args().wait_constructor_pid)
