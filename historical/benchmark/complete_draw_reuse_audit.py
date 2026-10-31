"""Coverage-qualified fixed-cohort RNG audit (including nested B2 directory)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
from .common import ROOT, file_digest, save_json
from .finite_draw_reuse_audit import audit


def run(output):
    cohorts = [ROOT / ("results/"+name) for name in (
        "sampler-baselines-B1-standard-v1",
        "sampler-baselines-B2-exactprep-standard-v2/independent-common-HF-readouts-v2",
        "nonvariational-B3-standard-v1", "random-B4-standard-v1")]
    counts = []
    for root in cohorts:
        databases = list((root / "readouts").glob("*/readouts.sqlite"))
        if len(databases) != 30:
            raise ValueError(f"Complete-cohort audit needs30actual databases, found{len(databases)}: {root}")
        n = 0
        for p in databases:
            with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as db:
                k = db.execute("SELECT COUNT(*) FROM draws").fetchone()[0]
                if not k: raise ValueError("Empty readout database: "+str(p))
                n += k
        counts.append(n)
    assert counts == [42240, 41040, 51840, 42840]
    output = Path(output); output.mkdir(exist_ok=True, parents=True)
    result = audit(cohorts, output)
    assert result["stored_draw_occurrences"] == sum(counts)
    assert result["unique_statistical_draws_across_listed_cohorts"] == 119640
    save_json(output / "coverage.json", {"databases": [30]*4, "expected_actual_draw_counts": counts,
              "all_four_cohorts_present": True, "implementation_sha256": file_digest(__file__),
              "supersedes_incomplete_cohort_selection": "shared-source-final-draw-audit-v2", "hardware_shots": 0})
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", default=str(ROOT / "results/shared-source-final-draw-audit-v3"))
    print(json.dumps(run(p.parse_args().output)))
