"""Continue the registered native ensemble cohort, without duplicating producer."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import ROOT, file_digest, save_json


def run(root, producer_pid, status_root):
    root, status_root = Path(root).resolve(), Path(status_root).resolve()
    status_root.mkdir(exist_ok=True, parents=True)
    mf = json.loads((root / "manifest.json").read_text()); planned = len(mf["case_ids"])*len(mf["methods"])
    bound = {"producer_pid": producer_pid, "cohort_manifest_sha256": file_digest(root / "manifest.json"),
             "code": {name: file_digest(ROOT / "benchmark" / name) for name in
                      ("study_sequence_sqdrift.py", "sqdrift_ensemble_audit.py", "sqdrift_ensemble_evaluate.py", "sqdrift_ensemble_report.py")}}
    save_json(status_root / "sequence-binding.json", bound)
    def status(stage, **kwargs):
        save_json(status_root / "status.json", {"stage": stage, "planned_cells": planned, **kwargs, "whole_study_complete": False})
    while not (root / "construction-status.json").exists():
        n = len(list((root / "cells").glob("*/*/result.json")))
        try:
            os.kill(producer_pid, 0)
        except ProcessLookupError:
            status("PRODUCER_HANDLE_MISSING_COMPLETED_PREFIX_PRESERVED", terminal_cells=n)
            return 1
        status("WAITING_VERIFIED_NATIVE_PRODUCER", terminal_cells=n, producer_pid=producer_pid)
        time.sleep(15)
    for stage, module in (("AUDITING_ACTUAL_CIRCUITS", "sqdrift_ensemble_audit"),
                          ("INDEPENDENT_FIXED_ALLOCATION_READOUT", "sqdrift_ensemble_evaluate"),
                          ("AUDITING_COMPLETE_COUNTS", "sqdrift_ensemble_audit"),
                          ("BUILDING_ALL_SETTING_REPORT", "sqdrift_ensemble_report")):
        status(stage, terminal_cells=planned)
        command = [sys.executable, "-u", "-m", "benchmark."+module, "--output", str(root)]
        if stage == "AUDITING_ACTUAL_CIRCUITS":
            command.append("--construction-only")
        with (status_root / (stage.lower()+".log")).open("a") as log:
            code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            status("STAGE_FAILED_PRESERVED", failing_stage=stage, returncode=code)
            return code
    status("NATIVE_ENSEMBLES_EXECUTED_EVALUATED_AUDITED_REPORTED", terminal_cells=planned)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); p.add_argument("--producer-pid", type=int, required=True)
    p.add_argument("--status-output", default=str(ROOT / "results/SqDRIFT-ensemble-sequence-v1")); args = p.parse_args()
    raise SystemExit(run(args.output, args.producer_pid, args.status_output))
