"""Optional ORACLE reporting references, only after all actual final readouts.

This file is never imported by the problem builder, constructor or evaluator.
Small <=16q full-sector eigenvalues are reporting labels, not a practical input.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from pyscf import fci, lib
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits
from .common import file_digest, save_json
from .molecular_problem_no_fci import load_problem


def reference(root):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text())
    ev = json.loads((root/"evaluation-validation.json").read_text())
    assert ev["status"] == "ACTUAL_FINAL_COUNTS_AND_PAIRS_VALIDATED"
    directory = root/"oracle-reporting-reference-v1"; directory.mkdir(exist_ok=True)
    binding = {"implementation_sha256": file_digest(__file__), "manifest_sha256": file_digest(root/"manifest.json"),
               "final_evaluation_validation_sha256": file_digest(root/"evaluation-validation.json"),
               "scope": "POSTHOC_ORACLE_ENERGY_LABELS_ONLY; after all growth and independent final draws; not selection input"}
    path = directory/"binding.json"
    if path.exists() and json.loads(path.read_text()) != binding: raise RuntimeError("Oracle reporting binding drift")
    save_json(path, binding); references = []
    with threadpool_limits(limits=1):
        lib.num_threads(1)
        for item in cfg["physical_problems"]:
            case = item["case_id"]; out = directory/f"{case}.json"
            if out.exists(): references.append(json.loads(out.read_text())); continue
            p = load_problem(root/"problems"/case); m = p["meta"]
            assert m["n_qubits"] <= 16 and not m["FCI_computed"]
            start = perf_counter(); solver = fci.direct_spin1.FCI()
            solver.conv_tol = 1e-12; solver.conv_tol_residual = 1e-8; solver.lindep = 1e-18
            solver.max_space = 50; solver.max_cycle = 200
            E, coefficients = solver.kernel(p["h1"], p["eri"], m["n_orbitals"], (m["nalpha"], m["nbeta"]), ecore=m["core_constant_Ha"])
            if not solver.converged: raise RuntimeError("Oracle energy not converged; preserve unknown reference")
            # Independently check the lowest eigenvalue in the registered physical
            # sector, not merely PySCF's internal convergence flag.
            hs = p["h"][p["ids"]][:, p["ids"]].tocsr()
            if len(p["ids"]) <= 2:
                values, vec = np.linalg.eigh(hs.toarray()); E2 = float(values[0]); vector = vec[:, 0]
            else:
                values, vec = eigsh(hs, k=1, which="SA", tol=1e-12, maxiter=5000,
                                   v0=np.random.default_rng(16037).normal(size=len(p["ids"])))
                E2 = float(values[0]); vector = vec[:, 0]
            residual = float(np.linalg.norm(hs@vector-E2*vector))
            if abs(float(E)-E2) > 1e-7 or residual > 1e-8:
                raise RuntimeError("Independent physical-sector ground-energy parity failed")
            result = {"case_id": case, "problem_id": m["problem_id"], "energy_Ha": float(E),
                      "independent_JW_sector_energy_Ha": E2, "independent_JW_residual": residual,
                      "SCF_HF_energy_Ha": m["hf_energy_Ha"], "oracle_reference_seconds": perf_counter()-start,
                      "PySCF_converged": True, "FCI_used_in_construction_or_stopping": False,
                      "FCI_only_posthoc_reporting": True, "created_after_final_readout_validation": True,
                      "input_integrals_sha256": m["artifact_sha256"]["integrals.npz"],
                      "scope": "registered Nalpha/Nbeta sector ground-energy reference; not a hardware measurement"}
            save_json(out, result); references.append(result)
    final = {"status": "POSTHOC_ORACLE_REFERENCES_INDEPENDENTLY_VALIDATED", "cases": len(references),
             "max_energy_difference_Ha": max(abs(r["energy_Ha"]-r["independent_JW_sector_energy_Ha"]) for r in references),
             "total_oracle_reference_seconds": sum(r["oracle_reference_seconds"] for r in references),
             "FCI_used_in_policy": False, "whole_study_complete": False}
    save_json(directory/"status.json", final)
    return final


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(reference(p.parse_args().output)))
