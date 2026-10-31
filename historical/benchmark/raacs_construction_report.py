"""Outer, post-construction RAACS reporting, not a baseline superiority test."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .common import file_digest, save_json


def report(root):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text())
    audit = json.loads((root/"counts-validation-v1.json").read_text())
    assert audit["status"] == "RAACS_STANDARD_COUNTS_BINDINGS_AND_FINAL_SUPPORT_VALIDATED"
    output = root/"construction-report-v2"; output.mkdir(exist_ok=True)
    source = Path(cfg["source"]); rows = []
    for path in sorted((root/"cells").glob("*/*/result.json")):
        cell = path.parent; case = cell.parent.name; r = json.loads(path.read_text()); a = r["algorithm"]
        ref = json.loads((source/"references"/case/"reference.json").read_text())
        db = sqlite3.connect(f"file:{cell/'events.sqlite'}?mode=ro", uri=True)
        resources = {cid: json.loads(raw) for cid, raw in db.execute("SELECT id,resources FROM circuits")}; db.close()
        allocations = r["final_circuit_allocations"]; total = sum(x["shots"] for x in allocations)
        resource = {key: sum(x["shots"]*resources[x["circuit_id"]][key] for x in allocations)/total
                    for key in ("cnot_count", "cnot_depth", "total_depth", "parameter_count")}
        row = {"case_id": case, "molecule": ref["case"]["molecule"], "distance_A": ref["case"]["distance"],
               "score": a["score"], "angle": a["angle_mode"], "allocation": a["final_allocation"], "seed": a["seed"],
               "backend": r["sampling"]["backend"], "error_mHa": 1000*(r["final_QSCI_energy_Ha"]-ref["fci_energy"]),
               "energy_Ha": r["final_QSCI_energy_Ha"], "actual_final_d": r["actual_final_CI_dimension"],
               "pilot_shots": r["pilot_raw_model_shots"], "final_shots": r["independent_final_raw_model_shots"],
               "total_raw_model_shots": r["actual_total_model_shots"], "wall_seconds": r["actual_wall_seconds"],
               "named_H_requests": r["classical_named_H_requests"], "accepted_steps": r["accepted_steps"],
               "construction_stop_reason": r["construction_stop_reason"], "status": r["status"],
               "resource_scope": "shot-weighted mean final sampling circuit; not all-pilot gate execution cost", **resource}
        rows.append(row)
    frame = pd.DataFrame(rows); frame.to_csv(output/"all-construction-cells.csv", index=False)
    common = frame[(frame.backend == "exact_small") & (frame.angle == "qsci_local_grid") & (frame.allocation == "last_only")]
    assert len(common) == 450
    means = common.groupby(["case_id", "molecule", "distance_A", "score"], as_index=False).agg(
        mean_error_mHa=("error_mHa", "mean"), mean_CNOT=("cnot_count", "mean"), mean_depth=("cnot_depth", "mean"),
        mean_wall_seconds=("wall_seconds", "mean"), mean_pilot_shots=("pilot_shots", "mean"),
        construction_seed_count=("seed", "count"))
    means.to_csv(output/"five-score-case-means.csv", index=False)
    by_mol = means.groupby(["molecule", "score"], as_index=False).mean(numeric_only=True)
    by_mol.to_csv(output/"molecule-cluster-descriptive-means.csv", index=False)
    figures = []
    for molecule, group in means.groupby("molecule"):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for score, sg in group.groupby("score"):
            sg = sg.sort_values("distance_A")
            for ax, col, label in zip(axes.flat, ["mean_error_mHa", "mean_CNOT", "mean_depth", "mean_wall_seconds"],
                                     ["QSCI error (mHa)", "Mean final CNOT", "Mean final CNOT depth", "Observed construction wall (s)"]):
                ax.plot(sg.distance_A, sg[col], "o-", label=score, ms=3)
                ax.set_xlabel("Distance / equal spacing (Å)"); ax.set_ylabel(label); ax.grid(alpha=.2)
        axes[0, 0].set_yscale("symlog", linthresh=.01); axes[1, 1].set_yscale("log")
        axes[0, 0].legend(fontsize=7); fig.suptitle(f"{molecule}: three construction seeds; one final draw each, total M=10000")
        fig.tight_layout(); name = f"score-ablation-{molecule}"
        for suffix in ("png", "svg", "pdf"): fig.savefig(output/f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig); figures.append(name)
    medians = means.groupby("score")[["mean_error_mHa", "mean_CNOT", "mean_depth", "mean_wall_seconds"]].median()
    text = f"""# RAACS standard construction: descriptive results\n\n
All840 planned cells returned COMPLETED; no engineering-failure records.
{audit['draw_records_checked']:,} draw records; {audit['raw_model_pilot_shots']:,} pilot and
{audit['raw_model_final_shots']:,} independent-final model shots. Hardware shots0.

## Five-score diagnostic (median of30 case means, NOT independent30 molecules)

```text
{medians.to_string(float_format=lambda x: f'{x:.4f}')}
```

Each case mean has three construction seeds and one independent final draw
per construction. This is not a high-precision conditional readout estimate.
Five geometries within one molecule are clustered, not independent molecules.
The six molecule-level descriptive means are separately exported; no p-value
or superiority claim is made from these small dependent groups.

## Boundaries

Every score uses the same bounded coupling-preselected local bank. The local
CI-gradient control is not full-pool or actual sampler-state ADAPT gradient.
Variant IDs change pilot random streams: these are end-to-end policy variants,
not isolated paired causal effects. Advanced one-parameter/blockwise/full-VQE
controls remain unimplemented in this version. MPS vs exact trajectories are
not a same-circuit bond-convergence test; the separate Phase E diagnostic is.

Reported CNOT/Depth are shot-weighted final sampling resources; all pilot gate
execution cost is separately available in audited-standard-cells-v1.csv.
FCI energies are loaded only here, after construction and audit, for reporting.
Independent repeated final readouts, same-total-budget/commonR/actuald strong
baseline comparisons and full classical cost interpretation remain required.
No claim of RAACS superiority, chemical-accuracy generalization or full-study
completion follows from a successful840-cell engineering execution.
"""
    (output/"REPORT.md").write_text(text)
    result = {"status": "RAACS_CONSTRUCTION_DESCRIPTIVE_REPORT_WRITTEN", "rows": len(frame), "figures": figures,
              "reporter_sha256": file_digest(__file__), "audit_sha256": file_digest(root/"counts-validation-v1.json"),
              "FCI_used_only_in_outer_reporting": True, "independent_strong_baseline_comparison_pending": True,
              "whole_study_complete": False}
    save_json(output/"manifest.json", result); return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(report(p.parse_args().output)))
