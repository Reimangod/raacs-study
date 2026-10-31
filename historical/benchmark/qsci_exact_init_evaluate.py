"""Reuse the qualified common-HF finite-readout engine for new exact-init B2.

This adapter exports real finalized circuit distributions, not proxy energies.
The reused engine's historical B1 filenames are explicitly an engine identity,
not a claim that these trajectories are LUCJ or have native HF-free readouts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, save_json
from .strong_baselines.evaluate import evaluate


def evaluate_successor(root):
    root = Path(root); mf = json.loads((root / "manifest.json").read_text())
    audit = json.loads((root / "validation.json").read_text())
    if audit["failed"]:
        raise RuntimeError("Unexplained native failures; audit required")
    output = root / "independent-common-HF-readouts-v2"; output.mkdir(exist_ok=True)
    methods = [d.name for d in sorted((root / "trajectories" / mf["cases"][0]).iterdir())]
    manifest = {"schema": "generic-common-HF-engine-adapter-for-B2-exactprep-v2", "mode": mf["mode"], "numeric_threads": 1,
                "source": mf["source"], "case_ids": mf["cases"], "methods": [{"name": m} for m in methods],
                "budget": {"M_grid": [1000, 10000, 100000], "R_grid": [8, 16, 32, 64, 128, 256, 512],
                           "evaluation_repeats": 4 if mf["mode"] == "quick" else 20},
                "growth_manifest_sha256": file_digest(root / "manifest.json"),
                "readout_implementation_sha256": file_digest(ROOT / "benchmark/strong_baselines/evaluate.py"),
                "adapter_implementation_sha256": file_digest(__file__),
                "view": "common HF inserted; matches native HF insertion for this successor",
                "tie_rule": "qualified study.core.ranks frequency ordering; see frozen engine",
                "namespace": "qualified B1-independent-final-evaluation-v1 with unique B2 taskIDs; disjoint from growth",
                "scope": "12 distinct growth Ns/R/seed results retained, not best-of12"}
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text()) != manifest:
        raise RuntimeError("Readout adapter identity drift")
    save_json(output / "manifest.json", manifest)
    for case in mf["cases"]:
        for method in methods:
            d = root / "trajectories" / case / method
            terminal = json.loads((d / "terminal.json").read_text())
            cp = json.loads(sorted(d.glob("step-*.json"))[-1].read_text())
            target = output / "cells" / case / method; target.mkdir(exist_ok=True, parents=True)
            destination = target / "final.npz"
            expected = file_digest(d / "latest-state.npz")
            if destination.exists() and file_digest(destination) != expected:
                raise RuntimeError("Readout distribution changed")
            if not destination.exists():
                shutil.copyfile(d / "latest-state.npz", destination)
            row = {"case_id": case, "method": method, "task_id": terminal["task_id"], "status": terminal["status"],
                   "construction_wall_seconds": terminal["wall_seconds_total"], "final_sha256": expected,
                   "objective_evaluations": terminal["cumulative"]["classical_angle_calls"],
                   "construction_simulated_raw_shots": terminal["cumulative"]["raw_simulated_shots"],
                   "energy_error_mHa": cp["native_qsci_error_mHa"], "input_state_energy": cp["input_state_energy"],
                   "resources": cp["resources"], "preparation_resources": cp["preparation_resources"],
                   "Ns": terminal["Ns"], "R": terminal["R"], "seed": terminal["seed"],
                   "initialization": cp["initialization"], "hardware_shots": 0,
                   "scope": "actual B2 HF-separated MP exact preparation, not LUCJ"}
            save_json(target / "result.json", row)
    with threadpool_limits(limits=1):
        result = evaluate(output)
    save_json(output / "B2-readout-status.json", {**result, "status": "B2_EXACT_INIT_INDEPENDENT_COMMON_HF_READOUT_COMPLETE",
                                               "view": manifest["view"], "whole_study_complete": False})
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(evaluate_successor(p.parse_args().output)))
