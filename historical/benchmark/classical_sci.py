"""Actual common-R classical SCI ranking controls; not an optimized SHCI claim."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .common import ROOT, digest, file_digest, save_json, save_npz
from .chemistry import load_reference
from .adapt_qsci_native import NamedHamiltonian
from .selected_ci_native import grow_space
from .study.core import rss_peak


def run(root, source, mode):
    root = Path(root).resolve(); source = Path(source).resolve()
    if root == source or source in root.parents:
        raise RuntimeError("Never place a new SCI study inside immutable source")
    root.mkdir(parents=True, exist_ok=True)
    cases = ["H2_R0.7", "H4_R1.1"] if mode == "quick" else sorted(d.name for d in (source / "references").iterdir() if d.is_dir())
    code = ["benchmark/selected_ci_native.py", "benchmark/classical_sci.py", "benchmark/adapt_qsci_native.py", "study-v1/CLASSICAL_SCI_B4.md"]
    manifest = {"schema": "classical-SCI-ranked-cardinality-v2", "source": str(source.resolve()), "mode": mode,
                "cases": cases, "R_grid": [8, 16, 32, 64, 128, 256, 512], "max_iterations": 64,
                "denominator_floor_Ha": 1e-8, "numerical_importance_floor": 1e-16,
                "criterion_seconds": 120, "rss_limit_bytes": 2*2**30,
                "implementation": {n: file_digest(ROOT / n) for n in code}, "hardware_shots": 0,
                "scope": "small-system common-R ranking controls; not optimized SHCI/full practical scaling evidence"}
    if (root / "manifest.json").exists() and json.loads((root / "manifest.json").read_text()) != manifest:
        raise RuntimeError("SCI config/code changed; use successor version")
    save_json(root / "manifest.json", manifest)
    rows = []; statuses = []
    with threadpool_limits(limits=1):
        for case in cases:
            if shutil.disk_usage(root).free < 3*2**30 or rss_peak() > 2*2**30:
                raise RuntimeError("Resource floor before new SCI case; retained prefix")
            started = perf_counter(); ref = load_reference(source / "references" / case)
            load_seconds = perf_counter()-started
            for criterion in ("hci_importance", "cipsi_coefficient_importance", "en_energy_importance"):
                d = root / "cases" / case / criterion; d.mkdir(parents=True, exist_ok=True)
                if (d / "result.json").exists():
                    result = json.loads((d / "result.json").read_text())
                else:
                    backend = NamedHamiltonian(ref["h"])
                    t = perf_counter()
                    history, status = grow_space(backend, ref["meta"]["hf_id"], 512, criterion,
                                                 max_seconds=120, rss_limit=2*2**30)
                    numeric_seconds = perf_counter()-t
                    records = []
                    for row in history:
                        save_npz(d / f"step-{row['iteration']:04d}.npz", ids=np.array(row["selected"], np.int64), coefficients=row["coefficients"])
                        records.append({**{k: v for k, v in row.items() if k != "coefficients"},
                                        "error_mHa_posthoc": (row["energy"]-ref["meta"]["fci_energy"])*1000})
                    result = {"case_id": case, "criterion": criterion, "problem_id": ref["meta"]["request_digest"],
                              "status": status, "records": records, "classical_numeric_seconds": numeric_seconds,
                              "shared_source_load_seconds_once_per_case": load_seconds,
                              "named_entry_requests": backend.matrix_entry_requests,
                              "connected_nonzero_terms_examined": sum(r["connected_nonzero_terms_examined"] for r in history),
                              "peak_process_rss": rss_peak(), "quantum_shots": 0, "hardware_shots": 0,
                              "logical_cnot": None, "logical_cnot_reason": "CLASSICAL_METHOD_NOT_A_QUANTUM_PREPARATION",
                              "FCI_used_for_selection": False}
                    save_json(d / "result.json", result)
                statuses.append({k: result[k] for k in ("case_id", "criterion", "status", "classical_numeric_seconds", "named_entry_requests", "peak_process_rss")})
                for R in manifest["R_grid"]:
                    if not result["records"]:
                        rows.append({"case_id": case, "criterion": criterion, "R_cap": R, "actual_d": None,
                                     "variational_error_mHa": None, "status": result["status"], "quantum_shots": 0})
                        continue
                    row = max((r for r in result["records"] if r["dimension"] <= R), key=lambda r: r["dimension"])
                    rows.append({"case_id": case, "criterion": criterion, "R_cap": R, "actual_d": row["dimension"],
                                 "variational_error_mHa": row["error_mHa_posthoc"], "variational_energy": row["energy"],
                                 "external_residual_norm": row["external_residual_norm"], "pt2_raw": row["pt2_raw"],
                                 "pt2_regularized": row["pt2_regularized"], "intruders": row["intruders"],
                                 "near_zero_denominators": row["near_zero_denominators"], "quantum_shots": 0,
                                 "cost_scope": "numeric_seconds_for_full_common_prefix_to_d512; not independently recharged at everyR"})
            print(f"Classical SCI completed {case}", flush=True)
    pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)
    pd.DataFrame(statuses).to_csv(root / "all-cell-status.csv", index=False)
    status = {"status": "QUICK_CLASSICAL_SCI_EXECUTED" if mode == "quick" else "CLASSICAL_SCI_RANKED_CONTROLS_EXECUTED",
              "cells": len(statuses), "summary_rows": len(rows), "hardware_shots": 0, "whole_study_complete": False,
              "full_SHCI_implemented": False}
    save_json(root / "status.json", status)
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="quick")
    p.add_argument("--source", default=str(ROOT / "results/standard-v1")); p.add_argument("--output")
    args = p.parse_args()
    print(json.dumps(run(args.output or ROOT / f"results/classical-SCI-{args.mode}-v2", args.source, args.mode)))
