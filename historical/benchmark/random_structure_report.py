"""B4 diagnostic report with classical proposals separated from circuit readouts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import save_json, file_digest
from .study_figures import energy_limits


def resource_kind(method):
    return "CLASSICAL_CONFIGURATION_PROPOSALS" if method == "uniform_valid_classical" else "SIMULATED_QUANTUM_CIRCUIT_READOUTS"


def report(root):
    root = Path(root); mf = json.loads((root / "manifest.json").read_text()); source = Path(mf["source"])
    if not (root / "evaluation-status.json").exists():
        raise RuntimeError("Complete independent readout required")
    out = root / "derived-report-v1"; out.mkdir(exist_ok=True)
    records = []
    for case in mf["case_ids"]:
        for spec in mf["methods"]:
            row = json.loads((root / "cells" / case / spec["name"] / "result.json").read_text())
            records.append({"case_id": case, "method": spec["name"], "family": spec["family"], "seed": spec.get("seed"),
                            "status": row["status"], "resource_kind": resource_kind(spec["name"]),
                            "construction_wall_seconds": row["construction_wall_seconds"],
                            "historical_CEO_dependency_seconds": row["historical_CEO_trajectory_seconds_as_recorded"],
                            "valid_sector_probability": row.get("valid_sector_probability"),
                            "cnot_count": row.get("resources", {}).get("cnot_count"), "cnot_depth": row.get("resources", {}).get("cnot_depth"),
                            "parameter_count": row.get("resources", {}).get("parameter_count"), "error": row.get("error")})
    resources = pd.DataFrame(records); resources.to_csv(out / "all-planned-cell-resources.csv", index=False)
    data = pd.read_csv(root / "readout-summary.csv")
    data["resource_kind"] = data.method.map(resource_kind)
    data = data.merge(resources.drop(columns="resource_kind"), on=["case_id", "method"], how="left", validate="many_to_one")
    data.to_csv(out / "independent-summary-with-cost-kinds.csv", index=False)
    cost_rows = []
    for case in mf["case_ids"]:
        with sqlite3.connect(root / "readouts" / case / "readouts.sqlite") as db:
            for method, draws, amount in db.execute("SELECT c.method,count(*),sum(d.shots) FROM draws d JOIN checkpoints c ON c.id=d.cp GROUP BY c.method"):
                cost_rows.append({"case_id": case, "method": method, "draws": draws, "resource_kind": resource_kind(method),
                                  "raw_proposals_or_simulated_readouts": amount, "hardware_shots": 0})
    costs = pd.DataFrame(cost_rows); costs.to_csv(out / "independent-budget-by-resource-kind.csv", index=False)
    figs = out / "figures"; figs.mkdir(exist_ok=True)
    for case, g in data.groupby("case_id"):
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True); values = []
        for family in mf["methods"]:
            name = family["family"]
            # Only the first specification per family controls the legend/mean;
            # all three seeds remain in the grouped mean and range.
            if family != next(x for x in mf["methods"] if x["family"] == name):
                continue
            methods = [x["name"] for x in mf["methods"] if x["family"] == name]
            s = g[(g.R_cap == 128) & g.method.isin(methods)].groupby("shots").mean_error_mHa.agg(["mean", "min", "max"])
            ax.plot(s.index, s["mean"], "o-", label=name, ms=3)
            ax.fill_between(s.index, s["min"], s["max"], alpha=.1)
            values += s["min"].tolist()+s["max"].tolist()
        for method in ("source_final_ceo_ovp", "source_final_ceo_mvp"):
            s = g[(g.R_cap == 128) & (g.method == method)].sort_values("shots")
            ax.plot(s.shots, s.mean_error_mHa, ".--", label=method); values += s.mean_error_mHa.tolist()
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=.01); energy_limits(ax, values)
        ax.axhline(1.5936014376, color="gray", ls="--", lw=.8)
        ax.set(xlabel="Raw proposals / simulated circuit readouts (different resource kinds)", ylabel="Independent QSCI error (mHa)",
               title=f"{case} — final-size-matched random controls; R cap128\nShaded range of seed means, NOT confidence interval")
        ax.legend(loc="best", fontsize=7)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(figs / f"{case}-B4-random-and-trained.{ext}", dpi=220)
        plt.close(fig)
    totals = costs.groupby("resource_kind").raw_proposals_or_simulated_readouts.sum().to_dict()
    lines = ["# Random structure/angle diagnostics", "", f"All{len(resources)} planned cells retained. Budget by distinct resource kind: {totals}. Hardware shots0.", "",
             "- Classical uniform valid-sector proposals have no prepared quantum circuit. CNOT/depth are N/A, not0. Their proposal count is not QPU shots. The engine's raw summed samples must be separated by resource_kind before reporting quantum measurement budgets.",
             "- Quantum controls use three registered seeds each, no best seed. Trained CEO operators + random angles versus random CEO operators + random angles distinguishes operator-order/topology effects at the same final source operator count; it is not a full causal decomposition of training.",
             "- All random lengths are conditioned on the final saved CEO-OVP length. This retrospective matching uses historical construction information. It is not a zero-cost online alternative to growing CEO. source construction dependency is retained and has a historical cost scope, not a controlled same-device timing ratio.",
             "- Pauli controls can leave particle/Sz sectors. Invalid counts remain charged in raw readouts, with only physically valid observed configurations admitted to QSCI. No free conditional renormalization. S² not certified.",
             "- Common R and pairwise actual-d comparisons are both saved. All-method common d may be1 due to HF. Same geometry points are not independent molecule replicates. These are exploratory diagnostic comparisons, not unqualified general superiority claims.",
             "- Full SqDRIFT, online stopping, RAACS, large-system molecular/MPS results and full SQD/noise remain scope. This report alone does not complete the study.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    result = {"cells": len(resources), "figures": len(mf["case_ids"]), "formats": ["png", "svg", "pdf"],
              "raw_budgets_by_distinct_resource_kind": totals, "implementation_sha256": file_digest(__file__),
              "hardware_shots": 0, "whole_study_complete": False}
    save_json(out / "report-status.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(report(p.parse_args().output)))
