"""Measured common-R/cost/recovery reports for explicit new fixed-sampler cohorts."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import beta, t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import file_digest, save_json
from .study_figures import energy_limits
from .b2_report import correlation_recovery


def group_name(method):
    if method.startswith("M") and "-R" in method:
        return "HF-separated "+method.rsplit("-s", 1)[0]
    return re.sub(r"_s[0-9]+$", "", method)


def report(root, name):
    root = Path(root); mf = json.loads((root / "manifest.json").read_text()); source = Path(mf["source"])
    if not (root / "evaluation-status.json").exists():
        raise RuntimeError("Independent fixed-final readout not complete")
    output = root / "derived-report-v1"; output.mkdir(exist_ok=True)
    bound = {"manifest_sha256": file_digest(root / "manifest.json"), "summary_sha256": file_digest(root / "readout-summary.csv"),
             "implementation_sha256": file_digest(__file__), "name": name}
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text()) != bound:
        raise RuntimeError("Report identity changed; successor required")
    save_json(output / "manifest.json", bound)
    resource_rows = []; missing = []
    for case in mf["case_ids"]:
        for method in mf["methods"]:
            row = json.loads((root / "cells" / case / method["name"] / "result.json").read_text())
            item = {k: row.get(k) for k in ("case_id", "method", "status", "construction_wall_seconds", "construction_simulated_raw_shots", "objective_evaluations", "input_state_energy", "n_coefficient_slots", "n_independent_ansatz_parameters", "n_optimized_variables", "Ns", "R", "seed", "peak_process_rss", "energy_error_mHa")}
            item["shared_case_setup"] = json.dumps(row.get("shared_case_setup", {}))
            item["cost_scope"] = "new full circuit/growth construction, source-reference setup excluded"
            if row["status"] == "FAILED_PRESERVED":
                missing.append({"case_id": case, "method": method["name"], "status": row["status"], "error": row.get("error")})
            else:
                item.update(row["resources"])
                item.update({"preparation_"+k: v for k, v in row["preparation_resources"].items() if k in ("cnot_count", "cnot_depth", "total_depth")})
            resource_rows.append(item)
        for method in sorted((source / "checkpoints" / case).iterdir()):
            cps = sorted(method.glob("step-*.json"))
            if not cps:
                missing.append({"case_id": case, "method": "source_final_"+method.name, "status": "SOURCE_MISSING_PRESERVED"}); continue
            cp = json.loads(cps[-1].read_text())
            resource_rows.append({"case_id": case, "method": "source_final_"+method.name, "status": "SOURCE_FINAL_REUSED",
                                  "historical_trajectory_seconds": cp.get("trajectory_seconds"), "input_state_energy": cp["vqe_energy"],
                                  "source_step": cp["step"], "cost_scope": "historical source trajectory; NOT same-device complete wall",
                                  **cp["resources"]})
    resources = pd.DataFrame(resource_rows)
    resources.to_csv(output / "resources-and-construction-costs.csv", index=False)
    pd.DataFrame(missing, columns=["case_id", "method", "status", "error"]).to_csv(output / "missing-cells.csv", index=False)
    data = pd.read_csv(root / "readout-summary.csv").merge(resources, on=["case_id", "method"], how="left", validate="many_to_one")
    frames = []
    for case, g in data.groupby("case_id"):
        ref = json.loads((source / "references" / case / "reference.json").read_text()); g = g.copy()
        g["correlation_recovered_percent"] = correlation_recovery(g.mean_error_mHa, ref["rhf_energy"], ref["fci_energy"])
        frames.append(g)
    data = pd.concat(frames, ignore_index=True)
    data["mean_ci90_halfwidth_mHa"] = t.ppf(.95, data.repeats-1)*data.std_error_mHa/np.sqrt(data.repeats)
    k = np.rint(data.success_probability_estimate*data.repeats).astype(int)
    data["success_ci90_low"] = [float(beta.ppf(.05, ki, ni-ki+1)) if ki else 0. for ki, ni in zip(k, data.repeats)]
    data["success_ci90_high"] = [float(beta.ppf(.95, ki+1, ni-ki)) if ki < ni else 1. for ki, ni in zip(k, data.repeats)]
    data["group"] = data.method.map(group_name)
    data.to_csv(output / "comparison-with-recovery.csv", index=False)
    figures = output / "figures"; figures.mkdir(exist_ok=True); generated = []
    source_methods = {"source_final_ceo_ovp", "source_final_ceo_mvp", "source_final_lucj_2"}
    for case, g in data.groupby("case_id"):
        primary = g[(g.R_cap == 128) & (~g.method.str.startswith("source_final_") | g.method.isin(source_methods))]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        values = []
        for index, (group, part) in enumerate(primary.groupby("group", sort=False)):
            color = plt.get_cmap("tab20")(index % 20)
            # All independent circuit/growth seeds retained. Range is NOT CI.
            means = part.groupby("shots").mean_error_mHa.agg(["mean", "min", "max"])
            recovery = part.groupby("shots").correlation_recovered_percent.mean()
            axes[0, 0].plot(means.index, means["mean"], "o-", color=color, ms=3, lw=1, label=group)
            axes[0, 0].fill_between(means.index, means["min"], means["max"], color=color, alpha=.1)
            axes[0, 1].plot(recovery.index, recovery.values, "o-", color=color, ms=3, lw=1)
            point = part[part.shots == 1000]
            axes[1, 0].scatter(point.cnot_count, point.mean_error_mHa, color=color, s=20)
            axes[1, 1].scatter(point.cnot_depth, point.mean_error_mHa, color=color, s=20)
            values += part.mean_error_mHa.tolist()
        for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
            ax.set_yscale("symlog", linthresh=.01); ax.set_ylabel("Independent QSCI error (mHa)")
            energy_limits(ax, values); ax.axhline(1.5936014376, color="gray", lw=.8, ls="--")
        for ax in axes[0]:
            ax.set_xscale("log"); ax.set_xlabel("Raw simulated readout shots (construction separate)")
        axes[0, 1].set_ylabel("FCI correlation energy recovered (%)")
        axes[1, 0].set_xlabel("Logical sampling CNOT count, M1000")
        axes[1, 1].set_xlabel("Logical sampling CNOT depth, M1000")
        fig.suptitle(f"{case} — {name}; common HF, R cap128\nLines: mean over registered circuit seeds; shade: range of seed means, NOT confidence interval")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=6.5)
        basename = case+"-independent-common-R-comparison"; generated.append(basename)
        for extension in ("png", "svg", "pdf"):
            fig.savefig(figures / f"{basename}.{extension}", dpi=220)
        plt.close(fig)
    shots_total = draws = 0
    for path in (root / "readouts").glob("*/readouts.sqlite"):
        with sqlite3.connect(path) as db:
            n, shots = db.execute("SELECT count(*),sum(shots) FROM draws").fetchone()
            draws += n; shots_total += shots
    lines = [f"# {name}: independent finite-shot comparisons", "",
             f"{len(mf['case_ids'])} conditions ×{len(mf['methods'])} new methods/settings. All planned states retained; missing rows in missing-cells.csv. Independent draws {draws:,}, simulated raw readout shots {shots_total:,}, hardware shots0.", "",
             "## Interpretation and boundaries", "",
             "- Common Hamiltonian/MO/active space/M/R, HF insertion. Pairwise CEO matched actual-d readouts are stored separately; all-method common-d includes HF-only and may collapse to1. Common-R is not actual-d equality.",
             "- All seeds/settings are shown. Shaded ranges are circuit-seed variation, not uncertainty intervals. Pointwise readout t90% intervals and exact-binomial90% success intervals are in CSV. Geometries are not independent molecules.",
             "- Correlation recovery is100*(E_HF−E_QSCI)/(E_HF−E_FCI), posthoc evaluation only. It is NOT configuration recovery; full SQD/noise remains separate.",
             "- Circuit logical sampling resources preserve Z probabilities; preparation resources are separate. Native hardware resources remain unknown without a device target.",
             "- New full construction wall and raw construction/pilot shots are recorded separately. Historical source trajectories have different cost scope/hardware, so no unconditional speedup or end-to-end QPU advantage follows. Exact classical VQE calls have no assumed universal shot conversion.",
             "- Pauli-qDRIFT is not the grouped-fermionic SqDRIFT ensemble. Its particle/Sz validity is measured, not silently renormalized. Fixed Trotter time grid was not selected using FCI. No finite-circuit asymptotic guarantee is claimed.",
             "- Orbital rotations here change the prepared state then sample the same original canonical basis; no invalid raw-ID union across different orbital bases.",
             "- These are small-system simulation results. Large MPS/integral backend molecular runs, FCI-free online stopping, RAACS, full SQD/noise and the rest of Milestone2 remain required. Whole study is not complete.", "",
             "## All M1000/R128 results", "", "| case | method | error mHa | correlation recovered % | CNOT | CNOT depth |", "|---|---|---:|---:|---:|---:|"]
    for row in data[(data.shots == 1000) & (data.R_cap == 128)].itertuples():
        lines.append(f"| {row.case_id} | {row.method} | {row.mean_error_mHa:.6g} | {row.correlation_recovered_percent:.6g} | {row.cnot_count:g} | {row.cnot_depth:g} |")
    (output / "REPORT.md").write_text("\n".join(lines)+"\n")
    result = {"figures": len(generated), "formats": ["png", "svg", "pdf"], "independent_draws": draws,
              "simulated_raw_readout_shots": shots_total, "hardware_shots": 0, "whole_study_complete": False}
    save_json(output / "report-status.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); p.add_argument("--name", required=True)
    args = p.parse_args(); print(json.dumps(report(args.output, args.name)))
