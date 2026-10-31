"""Storage-only continuation of the exact frozen30-case comparison queue.

Never modifies the stopped prefix or evaluator. Remaining cases are mechanical
disjoint shards, same seeds/M/R/physics. A read-only symlink view avoids copying
the raw databases. It is separately audited against the full original queue.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from .common import ROOT, file_digest, save_json
from . import raacs_independent_evaluation as engine


def execute(prefix, view):
    prefix, view = Path(prefix).resolve(), Path(view).resolve(); view.mkdir(parents=True, exist_ok=True)
    with (view/"run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        original = json.loads((prefix/"manifest.json").read_text()); state = json.loads((prefix/"execution-status.json").read_text())
        if original != engine.manifest(Path(original["construction"]), "standard"):
            raise RuntimeError("Frozen evaluator/protocol drift")
        completed = [r["case"] for r in state.get("rows", state.get("cases", []))]
        assert completed == original["cases"][:len(completed)] and len(set(completed)) == len(completed)
        physical_dirs = {p.name for p in (prefix/"readouts").iterdir() if p.is_dir()}
        if physical_dirs != set(completed): raise RuntimeError("Unreconciled partial case evidence; do not rerun")
        remaining = original["cases"][len(completed):]
        binding = {"schema": "RAACS-storage-only-disjoint-queue-shards-v1", "prefix": str(prefix),
                   "prefix_manifest_sha256": file_digest(prefix/"manifest.json"),
                   "prefix_execution_status_sha256": file_digest(prefix/"execution-status.json"),
                   "prefix_cases": completed, "remaining_cases": remaining, "maximum_cases_per_new_shard": 4,
                   "driver_sha256": file_digest(__file__), "science_seed_budget_unchanged": True,
                   "reason": "frozen640MiB per-output boundary reached after17complete cases; no scientific result discarded"}
        marker = view/"continuation-binding.json"
        if marker.exists() and json.loads(marker.read_text()) != binding: raise RuntimeError("Storage continuation drift")
        save_json(marker, binding)
        original_manifest = engine.manifest; destinations = [(case, prefix) for case in completed]
        for j in range(0, len(remaining), 4):
            cases = remaining[j:j+4]; shard = view.parent/f"{view.name}-shard-{j//4+2:02d}"
            def sharded_manifest(construction, mode):
                cfg = original_manifest(construction, mode)
                cfg["cases"] = cases
                cfg["queue_shard"] = {"original_manifest_sha256": binding["prefix_manifest_sha256"],
                                      "original_queue_indices": [original["cases"].index(c) for c in cases],
                                      "driver_sha256": file_digest(__file__), "only_storage_sharding_changed": True}
                return cfg
            stage = {"stage": "EXECUTING_SAME_QUEUE_REMAINDER_SHARD", "cases": cases, "output": str(shard), "whole_study_complete": False}
            save_json(view/"continuation-status.json", stage)
            # Config override changes only storage queue partition. The numerical
            # engine and its physical-distribution RNG identities stay untouched.
            with patch.object(engine, "manifest", sharded_manifest):
                result = engine.evaluate(Path(original["construction"]), shard, "standard")
            assert [r["case"] for r in result["cases"]] == cases
            destinations.extend((case, shard) for case in cases)
        assert [c for c, _ in destinations] == original["cases"]
        (view/"readouts").mkdir(exist_ok=True)
        for case, root in destinations:
            source = (root/"readouts"/case).resolve(); target = view/"readouts"/case
            if target.is_symlink():
                if target.resolve() != source: raise RuntimeError("Unified-view ownership changed")
            elif target.exists(): raise RuntimeError("Never overwrite a physical raw-data directory")
            else: target.symlink_to(source, target_is_directory=True)
        # Same full scientific manifest as the original queue; separate storage
        # provenance binds the new layout. No old manifest/status is overwritten.
        save_json(view/"manifest.json", original)
        rows = []
        for case, root in destinations:
            s = json.loads((root/"execution-status.json").read_text())
            rows.append(next(r for r in s.get("cases", s.get("rows", [])) if r["case"] == case))
        save_json(view/"storage-index.json", {"cases": [{"case": c, "root": str(r),
                  "raw_database_sha256": file_digest(r/"readouts"/c/"readouts.sqlite")} for c, r in destinations],
                  "all30_disjoint_queue_cases": True, "raw_data_copied": False})
        save_json(view/"execution-status.json", {"status": "RAACS_INDEPENDENT_COMPARISON_EXECUTED_PENDING_AUDIT", "cases": rows,
                  "hardware_shots": 0, "FCI_used": False, "whole_study_complete": False})
        save_json(view/"continuation-status.json", {"stage": "ALL_ORIGINAL30_CASES_LINKED_PENDING_INDEPENDENT_AUDIT", "whole_study_complete": False})
        return view


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default=str(ROOT/"results/RAACS-independent-comparison-standard-v1"))
    p.add_argument("--output", default=str(ROOT/"results/RAACS-independent-comparison-standard-sharded-v1"))
    a = p.parse_args(); output = execute(a.prefix, a.output)
    from .raacs_independent_audit import audit
    from .raacs_independent_report import report
    audit(output); print(json.dumps(report(output)))
