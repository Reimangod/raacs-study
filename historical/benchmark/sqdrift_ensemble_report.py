"""All-setting native SqDRIFT tables and scientific figures, not seed picking."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta, t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import ROOT, file_digest, save_json
from .chemistry import load_reference


def report(root, controls):
    root, controls = Path(root), Path(controls)
    mf = json.loads((root / "manifest.json").read_text())
    validation = json.loads((root / "validation.json").read_text())
    if validation["issues"] or validation["construction_only"]:
        raise RuntimeError("Full actual-count audit required")
    out = root / "derived-report-v1"; out.mkdir(exist_ok=True)
    summary = pd.read_csv(root / "readout-summary.csv"); cost_rows = []; pair_frames = []
    b1_cost = pd.read_csv(controls / "resources-and-construction-costs.csv")
    for case in mf["case_ids"]:
        meta = load_reference(Path(mf["source"]) / "references" / case)["meta"]
        gap = (meta["rhf_energy"]-meta["fci_energy"])*1000
        ids = summary.case_id == case
        summary.loc[ids, "correlation_energy_recovered_percent"] = 100*(1-summary.loc[ids, "mean_error_mHa"]/gap)
        resources = pd.read_parquet(root / "readouts" / case / "per-draw-ensemble-resources.parquet")
        for spec in mf["methods"]:
            result = json.loads((root / "cells" / case / spec["name"] / "result.json").read_text())
            row = {"case_id": case, "method": spec["name"], "status": result["status"], "member_count": result["member_count"],
                   "construction_wall_seconds": result["construction_wall_seconds"], "construction_pilot_shots": 0,
                   "SDK_case_shared_setup": str(root / "native" / case / "session-setup.json"),
                   "switch_latency_seconds": None, "hardware_shots": 0, "native_2Q_count": None,
                   "native_status": "NOT_CONFIGURED_NO_HARDWARE_TARGET", "configuration_recovery_status": "NOT_EXECUTED_THIS_COHORT",
                   "no_statevector_scaling_claim": True}
            row.update({"max_member_"+k: v for k, v in result.get("member_max_resources", {}).items()})
            cost_rows.append(row)
        pairs = pd.read_parquet(root / "readouts" / case / "pairwise-ceo-matched-d.parquet")
        pairs["case_id"] = case; pairs["delta_mHa"] = (pairs.energy-pairs.reference_energy)*1000
        pair_frames.append(pairs)
        for (method, M), g in resources.groupby(["method", "shots"]):
            m = ids & (summary.method == method) & (summary.shots == M)
            for field in ("cnot_shots", "depth_shots", "max_cnot", "max_depth", "unique_settings", "minimum_switches", "registered_order_switches"):
                assert g[field].nunique() == 1
                summary.loc[m, field] = float(g[field].iloc[0])
            summary.loc[m, "shot_weighted_mean_cnot"] = float(g.cnot_shots.iloc[0])/M
            summary.loc[m, "shot_weighted_mean_depth"] = float(g.depth_shots.iloc[0])/M
        for method in sorted(summary.loc[ids & (summary.role == "REUSED_B1_CONTROL"), "method"].unique()):
            q = b1_cost[(b1_cost.case_id == case) & (b1_cost.method == method)]
            assert len(q) == 1
            m = ids & (summary.method == method)
            summary.loc[m, "shot_weighted_mean_cnot"] = float(q.cnot_count.iloc[0])
            summary.loc[m, "shot_weighted_mean_depth"] = float(q.cnot_depth.iloc[0])
    # Exact binomial90% intervals:20/20 is not a90%-success certificate.
    n = summary.repeats.to_numpy(int); k = np.rint(summary.chemical_accuracy_observed_rate*n).astype(int)
    summary["success_ci90_low"] = [0. if a == 0 else beta.ppf(.05, a, b-a+1) for a, b in zip(k, n)]
    summary["success_ci90_high"] = [1. if a == b else beta.ppf(.95, a+1, b-a) for a, b in zip(k, n)]
    summary["error_ci90_halfwidth_mHa"] = t.ppf(.95, summary.repeats-1)*summary.std_error_mHa/np.sqrt(summary.repeats)
    summary.to_csv(out / "comparison-all-settings.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(out / "ensemble-construction-resources.csv", index=False)
    pairs = pd.concat(pair_frames, ignore_index=True); paired_rows = []
    for keys, g in pairs.groupby(["case_id", "reference_method", "method", "shots", "cap"]):
        paired_rows.append(dict(zip(["case_id", "reference_method", "method", "shots", "R_cap"], keys),
                                delta_mHa=float(g.delta_mHa.mean()), mean_actual_d=float(g.common_d.mean()),
                                paired_error_ci90_halfwidth_mHa=float(t.ppf(.95, len(g)-1)*g.delta_mHa.std(ddof=1)/np.sqrt(len(g))), repeats=len(g)))
    pair_summary = pd.DataFrame(paired_rows); pair_summary.to_csv(out / "pairwise-ceo-actual-d.csv", index=False)
    m90 = []
    for keys, g in summary.groupby(["case_id", "method", "R_cap"]):
        for column in ("chemical_accuracy_observed_rate", "success_ci90_low"):
            eligible = g[g[column] >= .9]
            m90.append(dict(zip(["case_id", "method", "R_cap"], keys), criterion=column,
                            first_observed_grid_M=int(eligible.shots.min()) if len(eligible) else None,
                            status="OBSERVED_GRID_NOT_CONTINUOUS_M90" if len(eligible) else "RIGHT_CENSORED_ABOVE_GRID",
                            criterion_not_interchangeable=True))
    pd.DataFrame(m90).to_csv(out / "success90-observed-grid.csv", index=False)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "svg.fonttype": "none"})
    gallery = []
    colors = {"L8_Nr3": "#2466a3", "L8_Nr9": "#c15e24", "L32_Nr3": "#258664", "L32_Nr9": "#9155a2", "L8_Nr2": "#2466a3"}
    for case in mf["case_ids"]:
        fig, axs = plt.subplots(2, 3, figsize=(15, 9)); s = summary[summary.case_id == case]
        labels = set()
        for spec in mf["methods"]:
            group = spec["name"].rsplit("_s", 1)[0].removeprefix("SqDRIFT_"); color = colors[group]
            label = group if group not in labels else "_nolegend_"; labels.add(group)
            a = s[(s.method == spec["name"]) & (s.R_cap == 128)].sort_values("shots")
            axs[0, 0].plot(a.shots, a.mean_error_mHa, "o-", color=color, alpha=.65, lw=1, label=label)
            b = a[a.shots == 1000]
            axs[0, 1].scatter(b.shot_weighted_mean_cnot, b.mean_error_mHa, color=color, s=25)
            axs[0, 2].scatter(b.shot_weighted_mean_depth, b.mean_error_mHa, color=color, s=25)
            b = s[(s.method == spec["name"]) & (s.shots == 1000)].sort_values("R_cap")
            axs[1, 0].plot(b.R_cap, b.mean_error_mHa, "o-", color=color, alpha=.65, lw=1)
            p = pair_summary[(pair_summary.case_id == case) & (pair_summary.method == spec["name"]) & (pair_summary.shots == 1000) & (pair_summary.reference_method == "source_final_ceo_mvp")].sort_values("R_cap")
            axs[1, 1].plot(p.R_cap, p.delta_mHa, "o-", color=color, alpha=.65, lw=1)
            axs[1, 2].plot(a.shots, a.correlation_energy_recovered_percent, "o-", color=color, alpha=.65, lw=1)
        for method, color, marker in (("source_final_ceo_ovp", "black", "s"), ("source_final_ceo_mvp", ".5", "^")):
            a = s[(s.method == method) & (s.R_cap == 128)].sort_values("shots")
            axs[0, 0].plot(a.shots, a.mean_error_mHa, marker+"--", color=color, lw=1.5, label=method.removeprefix("source_final_"))
            b = a[a.shots == 1000]
            axs[0, 1].scatter(b.shot_weighted_mean_cnot, b.mean_error_mHa, color=color, marker=marker, s=50)
            axs[0, 2].scatter(b.shot_weighted_mean_depth, b.mean_error_mHa, color=color, marker=marker, s=50)
            b = s[(s.method == method) & (s.shots == 1000)].sort_values("R_cap")
            axs[1, 0].plot(b.R_cap, b.mean_error_mHa, marker+"--", color=color, lw=1.5)
            axs[1, 2].plot(a.shots, a.correlation_energy_recovered_percent, marker+"--", color=color, lw=1.5)
        for ax in axs.flat:
            ax.grid(alpha=.15)
        for ax in (axs[0, 0], axs[1, 2]): ax.set_xscale("log")
        for ax in (axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0]):
            ax.set_yscale("symlog", linthresh=.1); ax.set_ylabel("QSCI error (mHa)"); ax.axhline(1.5936014376, color=".6", ls="--", lw=.7)
        axs[0, 0].set(xlabel="Total ensemble shots M", title="Common R cap128; all ensemble seeds")
        axs[0, 1].set(xlabel="Shot-weighted mean CNOT", title="M1000 / R128")
        axs[0, 2].set(xlabel="Shot-weighted mean CNOT depth", title="M1000 / R128")
        axs[1, 0].set(xlabel="R cap", title="M1000: common-R performance")
        axs[1, 1].set(xlabel="R cap", ylabel="SqDRIFT − CEO-MVP (mHa)", title="M1000: pairwise realized-d match"); axs[1, 1].axhline(0, color=".5", lw=.7)
        axs[1, 2].set(xlabel="Total ensemble shots M", ylabel="Correlation energy recovered (%)", title="Correlation recovery, NOT configuration recovery")
        axs[0, 0].legend(fontsize=8); fig.suptitle(case+": native grouped-fermion SqDRIFT partition/settings variant", fontsize=14)
        fig.text(.01, .008, "Same total M across all physical members; no FCI/time/seed selection. Thin curves retain every registered ensemble seed.\nSmall statevector qualification only. Hardware shots0. B1 controls reused as same statistical draws. Uncertainty and all LUCJ controls in CSV. CNOT/depth maxima separately tabulated.", fontsize=8)
        fig.tight_layout(rect=(0, .06, 1, .96))
        for ext in ("png", "svg", "pdf"): fig.savefig(out / f"{case}.{ext}", dpi=220, facecolor="white", bbox_inches="tight")
        plt.close(fig); gallery.append(case)
    text = ["# Native SqDRIFT ensemble comparison", "", f"Cases: {len(mf['case_ids'])}; ensembles: {len(cost_rows)}; actual member circuits audited: {validation['member_circuits_audited']}.",
            f"New ensemble statistical model shots: {validation['new_statistical_model_shots']}; reused B1 control draws: {validation['reused_control_draws']}; hardware shots0.", "",
            "- Published native grouped-fermionic construction via official pinned SDK, magnitude-refined partition; all nonconstant diagonals retained. Not the earlier Pauli-qDRIFT baseline and not an exact published hardware-figure replica.",
            "- Total M belongs to an ensemble, not to each member. Member seeds/allocations/counts and exact gate-shot/depth-shot totals are retained in SQLite.",
            "- Every registered time/randomization/ensemble seed is retained; no best-seed or FCI-guided selection. A finite Nr does not establish the paper's asymptotic concentration guarantee.",
            "- All B1 controls, including stronger LUCJ, are available in full CSV; six-panel figures highlight CEO references for legibility. Reused controls are the SAME draws, not extra replications.",
            "- Shared chemistry/integral input and native cold startup are recorded separately; historical CEO CPU/A100 construction times are not matched-device full wall times. No end-to-end QPU cost superiority claim.",
            "- Ensemble resources are not one circuit's count or depth. Use per-shot weighted mean plus member maximum; switching latency unknown, not0. Configuration recovery and real native hardware metrics are not evaluated in this cohort.",
            "- Exact binomial90% success intervals and observed-grid90% endpoints are separate.20/20 successes do not prove a90% lower-confidence success rate.",
            "- Full C/D/E/F study remains incomplete. Negative or conditional sampler results do not resolve those missing experiments.", ""]
    (out / "REPORT.md").write_text("\n".join(text))
    items = "".join(f'<h2>{html.escape(c)}</h2><img style="width:100%" src="{c}.png" alt="{c} all-setting comparison"><p><a href="{c}.svg">SVG</a> · <a href="{c}.pdf">PDF</a></p>' for c in gallery)
    (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Native SqDRIFT ensemble comparison</title><body style="font:15px Arial;max-width:1400px;margin:30px auto"><h1>Native SqDRIFT ensemble comparison</h1><p>Same total shot budget. All registered seeds. Reused controls are not independent replications. No hardware shots.</p>'+items+'</body>')
    status = {"status": "ALL_SETTING_ENSEMBLE_REPORT_COMPLETE", "figures": len(gallery), "formats": ["png", "svg", "pdf"],
              "comparison_rows": len(summary), "paired_rows": len(pair_summary), "implementation_sha256": file_digest(__file__),
              "hardware_shots": 0, "whole_study_complete": False}
    save_json(out / "status.json", status)
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    p.add_argument("--controls", default=str(ROOT / "results/sampler-baselines-B1-standard-v1")); args = p.parse_args()
    print(json.dumps(report(args.output, args.controls)))
