"""Non-destructive correction: trained-topology random CEO is not free growth.

All raw draws/energies and old interpretation remain. Only certification of
the standalone construction-inclusive endpoint is corrected to unknown.
"""
import json
from pathlib import Path
import pandas as pd
from .common import ROOT, file_digest, save_json


def source_build_cost(method, checkpoint):
    # A parent-conditioned skeleton cannot be called construction-free merely
    # because its new random coefficients were not optimized.
    if checkpoint.get("config", {}).get("parent_checkpoint_id") is not None:
        return None, "UNKNOWN_UPSTREAM_TRAINED_TOPOLOGY_QPU_CONSTRUCTION_COST"
    if method in ("hf", "lucj_1", "lucj_2", "lucj_truncated", "lucj_random_101"):
        return 0, "FIXED_CLASSICAL_INITIALIZER_NO_CONSTRUCTION_MEASUREMENTS"
    return None, "UNKNOWN_QPU_CONSTRUCTION_COST"


def apply():
    root = ROOT/"results/RAACS-independent-comparison-standard-sharded-v1"
    old = root/"derived-report-v1"; new = root/"cost-corrected-report-v2"; new.mkdir(exist_ok=True)
    cfg = json.loads((root/"manifest.json").read_text()); rows = []
    for case in cfg["cases"]:
        checkpoints = Path(cfg["source"])/"checkpoints"/case/"ceo_random_101"
        path = sorted(checkpoints.glob("step-*.json"))[-1]; cp = json.loads(path.read_text())
        cost, reason = source_build_cost("ceo_random_101", cp); assert cost is None
        rows.append({"case_id": case, "method": "B1:source_final_ceo_random_101", "build_model_shots_legacy": 0,
                     "build_model_shots_corrected": None, "corrected_status": reason,
                     "parent_checkpoint_id": cp["config"]["parent_checkpoint_id"], "checkpoint_sha256": file_digest(path)})
    frame = pd.read_csv(old/"all-cell-common-R-summary.csv")
    mask = (frame.method == "B1:source_final_ceo_random_101") & (frame.scope == "known_build_plus_readout")
    frame[mask].to_csv(new/"superseded-known-build-summary.csv", index=False)
    frame[~mask].to_csv(new/"all-cell-common-R-summary-cost-corrected.csv", index=False)
    pd.DataFrame(rows).to_csv(new/"trained-topology-cost-correction.csv", index=False)
    result = {"status": "TRAINED_TOPOLOGY_COST_SEMANTICS_CORRECTED_NONDESTRUCTIVELY", "cases": len(rows),
              "superseded_construction_inclusive_summary_rows": int(mask.sum()), "raw_data_or_energy_changed": False,
              "primary_paired_d_and_cluster_comparisons_changed": False,
              "reason": "source ceo_random_101 reuses already trained CEO-OVP operator sequence; unknown upstream QPU cost is not0",
              "readout_only_comparisons_retained": True,
              "legacy_manifest_and_links_immutable": True, "new_measurement_acquisitions": 0,
              "implementation_sha256": file_digest(__file__), "source_control_code_sha256": file_digest(ROOT/"benchmark/baselines.py"),
              "whole_study_complete": False}
    save_json(new/"cost-correction.json", result)
    (new/"READ_FIRST.md").write_text("""# Cost interpretation correction\n\n
Read the unchanged derived-report-v1/REPORT.md and its figures together with
this supplement. For common-R summary, prefer this directory's corrected CSV.
The source fixed-random CEO control inherits a trained CEO-OVP topology.
Generating its random angles needs no new optimization, but discovering its
topology was not free. Therefore its standalone construction-inclusive QPU
budget is unknown, not0. All630 affected common-R summary rows are retained in
superseded-known-build-summary.csv rather than discarded. Original raw draws,
links, manifest, energies and old report remain unchanged. No new samples.
The nine primary actual-d comparator methods do not include this control, so
their values and the primary molecule-cluster diagnostics are unaffected.
""")
    return result


if __name__ == "__main__": print(json.dumps(apply()))
