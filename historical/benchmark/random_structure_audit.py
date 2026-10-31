"""Qualify B4 real circuits, classical proposals, all seeds and count identities."""
from __future__ import annotations
import argparse
from io import BytesIO
import json
from pathlib import Path
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, save_json
from .chemistry import load_reference


def audit(root):
    root = Path(root); mf = json.loads((root / "manifest.json").read_text())
    for path, sha in mf["implementation"].items():
        assert file_digest(ROOT / path) == sha
    failures = []; circuits = classical = 0
    with threadpool_limits(limits=1):
        for case in mf["case_ids"]:
            ref = load_reference(Path(mf["source"]) / "references" / case)
            for spec in mf["methods"]:
                directory = root / "cells" / case / spec["name"]
                row = json.loads((directory / "result.json").read_text())
                if row["status"] == "FAILED_PRESERVED":
                    failures.append({"case_id": case, "method": spec["name"], "error": row["error"]}); continue
                assert file_digest(directory / "final.npz") == row["final_sha256"]
                assert not row["FCI_used_for_selection"]
                with np.load(directory / "final.npz") as z:
                    probability = z["probability"]
                    if spec["family"] == "uniform_classical":
                        assert set(np.flatnonzero(probability)) == set(ref["ids"])
                        assert np.allclose(probability[ref["ids"]], 1/len(ref["ids"]))
                        assert row["resources"]["cnot_count"] is None
                        assert row["sampler_type"] == "CLASSICAL_PROPOSAL"
                        classical += 1; continue
                    state, angles, indices = z["state"], z["angles"], z["indices"]
                assert len(angles) == len(indices) == row["source_saved_operator_count"]
                assert np.allclose(abs(state)**2, probability, atol=1e-12)
                assert abs(probability.sum()-1) < 1e-9
                prep = qpy.load(BytesIO((directory / "preparation.qpy").read_bytes()))[0]
                sampler = qpy.load(BytesIO((directory / "sampling.qpy").read_bytes()))[0]
                assert 1-abs(np.vdot(Statevector.from_instruction(prep).data, state))**2 < 1e-8
                assert np.allclose(abs(Statevector.from_instruction(sampler).data)**2, probability, atol=1e-9)
                assert sampler.count_ops().get("cx", 0) == row["resources"]["cnot_count"]
                assert abs(float(probability[ref["ids"]].sum())-row["valid_sector_probability"]) < 1e-9
                circuits += 1
    result = {"status": "UNEXPECTED_FAILURES_PRESERVED" if failures else ("QUICK_RANDOM_DIAGNOSTICS_VALIDATED" if mf["mode"] == "quick" else "STANDARD_RANDOM_DIAGNOSTICS_VALIDATED"),
              "actual_quantum_circuits_checked": circuits, "classical_uniform_controls_checked": classical,
              "failures": failures, "hardware_shots": 0, "whole_study_complete": False}
    save_json(root / "validation.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    result = audit(p.parse_args().output); print(json.dumps(result))
    if result["failures"]:
        raise SystemExit(1)
