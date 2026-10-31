"""Run frozen noise experiment, then actual all-record audit and report."""
import argparse
import json
from pathlib import Path
from .common import ROOT, save_json


def execute(root, mode):
    from .gate_noise_study import run
    from .gate_noise_audit import audit
    from .gate_noise_report import report
    result = run(root, mode)
    if result["status"] != "GATE_NOISE_FULL_COHORT_EXECUTED_PENDING_AUDIT": return result
    validation = audit(root); output = report(root)
    result = {"status": "GATE_NOISE_EXECUTED_AUDITED_REPORTED", "mode": mode, "planned": validation["planned"],
        "completed": validation["completed"], "missing": validation["missing_preserved"],
        "failures_and_censoring": len(validation["failures_and_censoring_preserved"]),
        "report": str(Path(root)/"report-v1"), "hardware_shots": 0, "whole_study_complete": False}
    save_json(Path(root)/"completion-sequence-v1.json", result); return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="standard"); p.add_argument("--output")
    a = p.parse_args(); print(json.dumps(execute(a.output or ROOT/f"results/gate-noise-full-{a.mode}-v1", a.mode)))
