"""Seal Milestone1 evidence only. Does not close the full six-milestone goal."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

from .common import ROOT, digest, file_digest, save_json
from .study_report import make_report


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def finalize(root):
    root = Path(root).resolve()
    if (root / "REPRODUCIBILITY.json").exists():
        raise RuntimeError("Already sealed; do not overwrite the sealed Milestone1 dataset")
    config = json.loads((root / "manifest.json").read_text())
    if config["mode"] not in ("standard", "full"):
        raise RuntimeError("Quick mode cannot close full Milestone1")
    for name, expected in config["implementation"].items():
        if file_digest(ROOT / name) != expected:
            raise RuntimeError(f"Numerical implementation drift: {name}")
    core = json.loads((root / "milestone1-audit.json").read_text())
    assert core["status"] == "PASS_MILESTONE1_CORE_AUDIT" and not core["failures"]
    assert len(core["finite_cases"]) == len(core["mixture_cases"]) == 30
    assert all(r["status"] == "PASS" for r in core["finite_cases"] + core["mixture_cases"])
    figures = json.loads((root / "figure-status.json").read_text())
    assert figures["generated_figures"] == len(figures["figures"]) == 334
    for name in figures["figures"]:
        for ext in figures["formats"]:
            assert (root / "figures" / f"{name}.{ext}").stat().st_size > 100
    interpreted = json.loads((root / "interpretation-status.json").read_text())
    assert interpreted["new_M_R_stopping_rows"] == 37800
    assert interpreted["historical_choices_independent_readout_rows"] == 1800
    assert interpreted["mixture_summary_groups"] == 6750
    assert len(pd.read_parquet(root / "analysis/individual-target-discovery.parquet")) == interpreted["individual_target_discovery_rows"]
    # Recheck the original dataset after all new numerical and reporting work.
    source = Path(config["source"])
    pointer = json.loads((source / "REPRODUCIBILITY.json").read_text())
    manifest = json.loads((source / pointer["manifest"]).read_text())
    for name, record in manifest["artifacts"].items():
        if sha(source / name) != record["sha256"]:
            raise RuntimeError("Original source drift: " + name)
    save_json(root / "source-audit-after.json", {"all_match": True, "artifacts_checked": len(manifest["artifacts"]),
              "source_dataset_id": pointer["dataset_id"], "source_modified": False})
    closure = {"status": "MILESTONE1_COMPLETE_WITH_EXPLICIT_LIMITS", "whole_study_complete": False,
               "core_counts": core["totals"], "mixture_readouts": 135000, "figures": 334, "figure_formats": figures["formats"],
               "interpretation": interpreted, "hardware_shots": 0,
               "numeric_audit_scope": core["coverage"],
               "remaining": ["M2_strong_baselines", "M3_FCI_free_stopping_and_generalization", "M4_RAACS_ablations", "M5_scaling", "M6_fullSQD_noise_hardware_conditional"],
               "claims_not_established": ["RAACS_superiority", "optimized_LUCJ_superiority", "end_to_end_quantum_shot_advantage", "unknown_molecule_generalization", "large_system_scaling", "hardware_advantage"]}
    save_json(root / "milestone1-closure.json", closure)
    make_report(root)
    code_paths = [ROOT / "pyproject.toml", ROOT / "uv.lock", ROOT / "AGENTS.md"]
    code_paths += sorted((ROOT / "benchmark").rglob("*.py")) + sorted((ROOT / "study-v1").glob("*.md"))
    provenance = root / "provenance"
    provenance.mkdir(exist_ok=True)
    code_id = digest({str(p.relative_to(ROOT)): sha(p) for p in code_paths})
    capsule = provenance / f"code-{code_id}.zip"
    with zipfile.ZipFile(capsule, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in code_paths:
            archive.write(path, str(path.relative_to(ROOT)))
        archive.write(config["goal_source"], "AUTHORITATIVE_GOAL_OBJECTIVE.md")
    # Transient SQLite WAL/SHM and lock files are not durable data products.
    artifacts = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(("-wal", "-shm", ".lock")):
            continue
        artifacts[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    record = {"schema": "sampler-study-M1-artifacts-v1", "source_dataset": pointer["dataset_id"],
              "code_capsule": str(capsule.relative_to(root)), "code_capsule_sha256": sha(capsule),
              "artifacts": artifacts, "whole_study_complete": False, "remaining_milestones": closure["remaining"]}
    record_id = digest(record)
    dest = provenance / f"artifacts-{record_id}.json"
    save_json(dest, record)
    result = {"dataset_id": record_id, "manifest": str(dest.relative_to(root)), "manifest_sha256": sha(dest),
              "artifacts": len(artifacts), "bytes": sum(x["bytes"] for x in artifacts.values()),
              "status": closure["status"], "whole_study_complete": False, "external_upload": False}
    save_json(root / "REPRODUCIBILITY.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print(json.dumps(finalize(args.output), indent=2))


if __name__ == "__main__":
    main()
