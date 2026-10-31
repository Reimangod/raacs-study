"""Post-B4 regression/reuse audit; waits for the existing live coordinator.

Does not start another numerical worker while B4 owns the second slot.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from .common import ROOT, save_json
from .finite_draw_reuse_audit import audit


def run(pid):
    status_file = ROOT / "results/random-B4-sequence-v1/status.json"
    output = ROOT / "results/post-B4-regression-reuse-v1"; output.mkdir(exist_ok=True)
    while True:
        status = json.loads(status_file.read_text())
        if status["stage"] == "RANDOM_COHORT_AUDITED_EVALUATED_REPORTED":
            break
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            save_json(output / "status.json", {"stage": "B4_COORDINATOR_HANDLE_MISSING", "last_status": status, "whole_study_complete": False})
            return 1
        save_json(output / "status.json", {"stage": "WAITING_VERIFIED_B4_COORDINATOR", "pid": pid, "whole_study_complete": False})
        time.sleep(15)
    env = dict(os.environ); env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    save_json(output / "status.json", {"stage": "FULL_PROJECT_REGRESSION", "whole_study_complete": False})
    # Explicit discovery path: a prior `pytest tests` only checked3new tests,
    # not the configured benchmark/tests suite. Never label it a full run.
    with (output / "pytest.log").open("a") as log:
        code = subprocess.run([sys.executable, "-m", "pytest", "benchmark/tests", "-q", "--junitxml="+str(output / "regression.xml")],
                              cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        save_json(output / "status.json", {"stage": "REGRESSION_FAILURE_PRESERVED", "returncode": code, "whole_study_complete": False})
        return code
    cohorts = [ROOT / ("results/"+name) for name in (
        "sampler-baselines-B1-standard-v1", "sampler-baselines-B2-exactprep-standard-v2",
        "nonvariational-B3-standard-v1", "random-B4-standard-v1")]
    result = audit(cohorts, ROOT / "results/shared-source-final-draw-audit-v2")
    classical = 0; classical_draws = 0
    for p in (cohorts[-1] / "readouts").glob("*/readouts.sqlite"):
        with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as db:
            count, shots = db.execute("SELECT COUNT(*),SUM(d.shots) FROM draws d JOIN checkpoints c ON c.id=d.cp WHERE c.method='uniform_valid_classical'").fetchone()
            classical_draws += count; classical += shots or 0
    save_json(output / "classical-proposal-distinction.json", {
        "uniform_valid_classical_draws": classical_draws, "classical_proposals": classical,
        "included_in_generic_statistical_model_sample_count": True, "not_quantum_shots": True,
        "hardware_shots": 0, "do_not_call_all_model_samples_quantum_measurements": True})
    save_json(output / "status.json", {"stage": "FULL_REGRESSION_AND_REUSE_AUDIT_COMPLETE", "returncode": 0,
              "unique_statistical_draws": result["unique_statistical_draws_across_listed_cohorts"], "whole_study_complete": False})
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--existing-coordinator-pid", type=int, required=True)
    raise SystemExit(run(p.parse_args().existing_coordinator_pid))
