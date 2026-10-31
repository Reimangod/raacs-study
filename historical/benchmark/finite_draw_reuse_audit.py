"""Distinguish cohort model budgets, repeated simulation, and unique RNG draws.

Fixed-readout engine intentionally uses the same source-final reference seeds
across additive cohorts. Those are shared statistical controls, not independent
replications or additional physical quantum measurements.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
from .common import ROOT, file_digest, save_json


def audit(cohorts, output):
    groups = {}; cohort_records = []; conflict = []; repeated = []
    for cohort in map(Path, cohorts):
        draws = samples = 0; shared_draws = shared_samples = 0
        for database in sorted((cohort / "readouts").glob("*/readouts.sqlite")):
            case = database.parent.name
            with sqlite3.connect(database) as db:
                for method, sha, M, rep, seed, counts in db.execute("SELECT c.method,c.sha,d.shots,d.rep,d.seed,d.counts FROM draws d JOIN checkpoints c ON c.id=d.cp"):
                    identity = (case, method, sha, M, rep, seed)
                    blob_sha = hashlib.sha256(counts).hexdigest()
                    draws += 1; samples += M
                    if identity in groups:
                        first = groups[identity]
                        if first["counts_sha256"] != blob_sha:
                            conflict.append({"case_id": case, "method": method, "first": first["cohort"], "cohort": str(cohort), "identity": identity})
                        shared_draws += 1; shared_samples += M
                    else:
                        groups[identity] = {"cohort": str(cohort), "counts_sha256": blob_sha, "shots": M, "method": method}
        cohort_records.append({"cohort": str(cohort), "stored_statistical_draw_occurrences": draws,
                               "cohort_specific_model_sample_budget": samples,
                               "draw_identities_already_present_in_prior_cohorts": shared_draws,
                               "model_samples_with_already_present_draw_identities": shared_samples})
    if conflict:
        raise RuntimeError("Repeated RNG evaluation identity has nonidentical counts: "+str(conflict[:3]))
    result = {"status": "COMMON_REFERENCE_RNG_REUSE_EXPLICITLY_ACCOUNTED", "cohorts": cohort_records,
              "unique_statistical_draws_across_listed_cohorts": len(groups),
              "unique_statistical_model_samples_across_listed_cohorts": sum(x["shots"] for x in groups.values()),
              "stored_draw_occurrences": sum(x["stored_statistical_draw_occurrences"] for x in cohort_records),
              "hardware_shots": 0, "conflicts": conflict,
              "interpretation": ["within each final sampler the20replicates are independent",
                                 "source-final controls regenerated with the same seed/key across cohorts are not independent replications",
                                 "do not sum shared-control occurrences as additional unique training/evaluation examples",
                                 "cohort-specific hypothetical budgets remain valid; repeated RNG/CI CPU work was actually performed and is not retroactively erased",
                                 "these are simulated categorical samples, not physically executed QPU shots"],
              "implementation_sha256": file_digest(__file__), "whole_study_complete": False}
    output = Path(output); output.mkdir(exist_ok=True, parents=True)
    save_json(output / "validation.json", result)
    (output / "README.md").write_text("# Shared source-final statistical controls\n\nThe same source-final reference draws are reused/regenerated deterministically across additive fixed-readout cohorts. This is useful for comparisons, but is not independent replication. See validation.json for deduplicated statistical draw identities and per-cohort hypothetical budgets. All actual hardware shots are0. CPU time spent repeating simulation/CI is real and is not erased by this statistical deduplication.\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--cohort", nargs="+", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); print(json.dumps(audit(a.cohort, a.output)))
