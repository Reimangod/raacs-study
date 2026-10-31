"""Cohort-complete RAACS-v1 audit, preserving censored and failed cells."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, save_json
from .raacs_counts_audit import audit_cell


def audit(root):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text())
    status = json.loads((root/"execution-status.json").read_text())
    expected = len(cfg["case_ids"])*len(cfg["variants"])
    assert status["status"] == "RAACS_CONSTRUCTION_EXECUTED_PENDING_AUDIT_AND_COMPARISON"
    assert status["terminal"] == status["planned"] == expected
    for p, sha in cfg["implementation"].items(): assert file_digest(ROOT/p) == sha
    rows = []; failures = []; seen = set()
    with threadpool_limits(limits=1):
        for case in cfg["case_ids"]:
            for score, angle, allocation, seed, backend in cfg["variants"]:
                name = f"{score}-{angle}-{allocation}-s{seed}-{backend}"
                folder = root/"cells"/case/name; seen.add(folder.resolve())
                errors = sorted(folder.glob("failure-*.json"))
                if errors:
                    failures.append({"case_id": case, "cell": name, "failure_files": [str(p.relative_to(root)) for p in errors],
                                     "status": "ENGINEERING_FAILURE_PRESERVED", "failed_costs_not_zero": True})
                    continue
                a = audit_cell(Path(cfg["source"])/"references"/case, folder)
                rows.append({"case_id": case, "cell": name, "score": score, "angle": angle, "allocation": allocation,
                             "construction_seed": seed, "backend": backend, **a})
                if len(rows) % 28 == 0:
                    save_json(root/"counts-audit-progress-v1.json", {"validated_cells": len(rows), "failures_preserved": len(failures), "planned": expected})
    assert {p.parent.resolve() for p in (root/"cells").glob("*/*/result.json")} <= seen
    import pandas as pd
    pd.json_normalize(rows).to_csv(root/"audited-standard-cells-v1.csv", index=False)
    out = {"status": "RAACS_STANDARD_COUNTS_BINDINGS_AND_FINAL_SUPPORT_VALIDATED", "planned": expected,
           "available_validated": len(rows), "engineering_failures_preserved": failures,
           "statuses": dict(Counter(r["status"] for r in rows)),
           "draw_records_checked": sum(r["raw_draws_checked"] for r in rows),
           "circuits_decoded_and_resources_checked": sum(r["circuits_decoded_and_resources_checked"] for r in rows),
           "symbolically_rebuilt_final_circuits": sum(r["final_circuits_symbolically_rebuilt"] for r in rows),
           "raw_model_pilot_shots": sum(r["pilot_shots"] for r in rows),
           "raw_model_final_shots": sum(r["final_shots"] for r in rows),
           "scope": "all available count records, identities, final observed support and fresh integral CI; not all draws reexecuted",
           "independent_final_replicates_and_strong_baseline_comparison": "REMAIN_REQUIRED",
           "hardware_shots": 0, "whole_study_complete": False,
           "auditor_implementation": {p: file_digest(ROOT/p) for p in ("benchmark/raacs_standard_audit.py", "benchmark/raacs_counts_audit.py")}}
    assert len(rows)+len(failures) == expected
    save_json(root/"counts-validation-v1.json", out); return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(audit(p.parse_args().output)))
