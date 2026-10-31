"""Seal only completed B1 evidence, not the full baseline/study objective."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import zipfile
from .common import ROOT, digest, file_digest, save_json


def seal(root):
    root = Path(root)
    if (root / "REPRODUCIBILITY.json").exists():
        return json.loads((root / "REPRODUCIBILITY.json").read_text())
    m = json.loads((root / "manifest.json").read_text())
    assert m["mode"] == "standard" and len(m["case_ids"]) == 30
    audit = json.loads((root / "validation.json").read_text())
    assert audit["status"] == "B1_VALIDATED_WITH_LIMITS" and not audit["failures"]
    assert len(audit["cells"]) == 390
    report = json.loads((root / "report-status.json").read_text())
    assert report["all_cells"] == 390 and report["figures"] == 90
    for name in report["names"]:
        for ext in report["formats"]:
            assert (root / "figures" / f"{name}.{ext}").stat().st_size > 100
    for name, sha in m["implementation"].items():
        assert file_digest(ROOT / name) == sha, name
    p = root / "provenance"; p.mkdir(exist_ok=True)
    paths = list((ROOT / "benchmark").rglob("*.py"))+list((ROOT / "study-v1").glob("*.md"))+[ROOT / "pyproject.toml", ROOT / "uv.lock"]
    code_hashes = {str(f.relative_to(ROOT)): file_digest(f) for f in sorted(paths)}
    code_id = digest(code_hashes)
    capsule = p / f"code-{code_id}.zip"
    with zipfile.ZipFile(capsule, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in code_hashes:
            archive.write(ROOT / name, name)
    save_json(p / "code-files.json", code_hashes)
    closure = {"status": "B1_LUCJ_COHORT_COMPLETE_WITH_CAPS_AND_MISSINGNESS", "planned_cells": 390,
               "status_counts": report["status_counts"], "figures": 90, "formats": report["formats"],
               "hardware_shots": 0, "full_milestone2_complete": False, "whole_study_complete": False,
               "limitations": ["finite-budget_L-BFGS_not_optimizer_independent_LUCJ_optimum", "no_linear_method_in_this_cohort",
                               "10_preserved_source_CCSD_dependent_failures", "historical_vs_new_wall_scopes_not_matched",
                               "VQE_calls_not_converted_to_QPU_shots", "B2_B3_B4_and_C_D_E_F_remain"]}
    save_json(root / "closure.json", closure)
    records = {}
    for f in sorted(root.rglob("*")):
        if f.is_file() and not f.name.endswith(("-wal", "-shm", ".lock")):
            records[str(f.relative_to(root))] = {"bytes": f.stat().st_size, "sha256": file_digest(f)}
    manifest = {"schema": "B1-completed-artifacts-v1", "source_dataset": m["source_dataset"],
                "code_capsule": str(capsule.relative_to(root)), "artifacts": records, "whole_study_complete": False}
    identity = digest(manifest); dest = p / f"artifacts-{identity}.json"
    save_json(dest, manifest)
    result = {"dataset_id": identity, "manifest": str(dest.relative_to(root)), "manifest_sha256": file_digest(dest),
              "artifacts": len(records), "bytes": sum(r["bytes"] for r in records.values()), **closure}
    save_json(root / "REPRODUCIBILITY.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(seal(p.parse_args().output)))
