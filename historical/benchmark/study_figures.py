"""Figures for the new sampler study; never write into standard-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .common import save_json

COLORS = {"ceo_ovp": "#2563a6", "ceo_mvp": "#dc7c24", "qeb": "#238b62", "qubit": "#bf4457", "gsd": "#8059a8", "lucj_1": "#976340", "lucj_2": "#ca6a99", "lucj_truncated": "#798742", "lucj_random_101": "#788b9d", "hf": "#333333", "ceo_random_101": "#269cb0"}


def energy_limits(ax, values, reference=True):
    """Pad in transformed coordinates, not linear margins across decades.

    Genuine negative CI tails remain; do not clip observations or uncertainty.
    """
    values = np.array(list(values) + ([1.59360143764] if reference else []), dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return
    ax.set_yscale("symlog", linthresh=1e-4)
    transform = ax.yaxis.get_transform()
    bounds = transform.transform(np.array([values.min(), values.max()]))
    # SymLog's transform carries its linthresh scale: an absolute .01 here
    # would accidentally span ~100 decades for linthresh=1e-4.
    pad = max(float(bounds[1]-bounds[0])*.06, 1e-6)
    ax.set_ylim(transform.inverted().transform(bounds + np.array([-pad, pad])))


def save(fig, path, note):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(.01, .008, note, fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, .085, 1, .98))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(str(path) + "." + ext, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_all(root):
    root = Path(root)
    out = root / "figures"
    manifest = json.loads((root / "manifest.json").read_text())
    source = Path(manifest["source"])
    planned = json.loads((root / "execution-queue.json").read_text())
    prefix = f"{manifest['mode']} cohort: {planned['case_count']} cases / {planned['selected_checkpoints']} checkpoints; simulated draws, hardware shots0."
    made = []
    stops = pd.read_parquet(root / "retrospective-stops.parquet")
    for tol in (0, .1, .5, 1.):
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        for j, column in enumerate(("step_ratio", "cnot_ratio", "depth_ratio")):
            groups = [stops[(stops.method == m) & (stops.shots == 1000) & (stops.tolerance_mHa == tol)][column].dropna().values for m in ("ceo_ovp", "ceo_mvp", "qeb", "qubit", "gsd")]
            axs[j].boxplot(groups, tick_labels=["OVP", "MVP", "QEB", "qubit", "GSD"], showmeans=True)
            axs[j].set(ylabel=column.replace("_", " "), title=f"Within {tol:g} mHa of best")
            axs[j].axhline(1, color=".5", lw=.7, ls="--")
        name = f"Fig06_retrospective_ratios_tol_{tol:g}"
        save(fig, out / name, "All30 conditions, all5 pools. Same-sample historicalM1000. Ratio to last SAVED checkpoint, not unknown convergence.\nGeometry points are correlated; boxplots are descriptive, not independent molecule-level confidence intervals.")
        made.append(name)
    readout_file = root / "analysis/readout-summary.parquet"
    if readout_file.exists():
        all_rows = pd.read_parquet(readout_file)
        df = all_rows[all_rows.primary_record & all_rows.complete_group & (all_rows.selection_rule == "frequency")]
        inventory = pd.DataFrame(json.loads((root / "checkpoint-inventory.json").read_text()))
        last = inventory[inventory.primary].groupby(["case_id", "method"]).step.max().to_dict()
        final = df[[r.step == last[(r.case_id, r.method)] for r in df.itertuples()]]
        for case, case_rows in final.groupby("case_id"):
            for fig_id, x, fixed, title in [("Fig01", "shots", ("R_cap", 32), "QSCI energy vs shots"), ("Fig17", "R_cap", ("shots", 1000), "CI-dimension sensitivity")]:
                fig, ax = plt.subplots(figsize=(7, 4.8))
                g = case_rows[case_rows[fixed[0]] == fixed[1]]
                for method, arm in g.groupby("method"):
                    arm = arm.sort_values(x)
                    ax.errorbar(arm[x], arm.mean_error_mHa, yerr=[arm.mean_error_mHa-arm.ci90_low_mHa, arm.ci90_high_mHa-arm.mean_error_mHa], fmt="o-", ms=3, lw=1, capsize=1, label=method, color=COLORS[method])
                ax.set_xscale("log")
                ax.set_yscale("symlog", linthresh=1e-4)
                ax.axhline(1.59360143764, lw=.8, ls="--", color=".4")
                ax.set(xlabel="Shots per draw" if x == "shots" else "Registered CI cap R (realized d is recorded separately)", ylabel="QSCI error (mHa)", title=f"{case}: {title}")
                energy_limits(ax, g[["mean_error_mHa", "ci90_low_mHa", "ci90_high_mHa"]].to_numpy().ravel())
                ax.legend(fontsize=6, ncol=3)
                ax.grid(alpha=.15)
                name = f"{case}/{fig_id}_{x}"
                save(fig, out / name, prefix + "\nFinal checkpoint within this execution scope; not necessarily full VQE convergence. Pointwise90% draw intervals; no independent superiority claim.")
                made.append(name)
        resources = inventory[["checkpoint_id", "cnot", "depth"]]
        curves = df[(df.shots == 1000) & (df.R_cap == 32)].merge(resources, on="checkpoint_id")
        for case, case_rows in curves.groupby("case_id"):
            for fig_id, x in (("Fig02", "cnot"), ("Fig03", "depth")):
                fig, ax = plt.subplots(figsize=(7, 4.8))
                for method, arm in case_rows.groupby("method"):
                    arm = arm.sort_values("step")
                    ax.plot(arm[x], arm.mean_error_mHa, "o-" if method in ("ceo_ovp", "ceo_mvp", "qeb", "qubit", "gsd") else "o", ms=2, lw=.8, color=COLORS[method], label=method)
                ax.set(xlabel="Z-sampling logical CNOT count" if x == "cnot" else "Z-sampling logical CNOT depth", ylabel="Mean QSCI error (mHa)", title=f"{case}: M1000 / R32", yscale="symlog")
                ax.axhline(1.59360143764, color=".4", ls="--", lw=.8)
                energy_limits(ax, case_rows.mean_error_mHa)
                ax.legend(fontsize=6, ncol=3)
                name = f"{case}/{fig_id}_{x}"
                save(fig, out / name, prefix + "\nIndependent new evaluation draws, recorded growth trajectories; fixed controls are points. Z-probability-equivalent resources, not full-state compression.")
                made.append(name)
        met_file = root / "analysis/discovery.parquet"
        if met_file.exists():
            met = pd.read_parquet(met_file)
            met = met[met.primary_record]
            for case, group in met.groupby("case_id"):
                fig, ax = plt.subplots(figsize=(7, 4.8))
                for method, arm in group.groupby("method"):
                    arm = arm[arm.step == last[(case, method)]].sort_values("shots")
                    ax.plot(arm.shots, arm.recall_top3, "o-", ms=3, label=method, color=COLORS[method])
                ax.set(xlabel="Shots M", ylabel="Expected top3 non-HF recall", title=f"{case}: oracle discovery diagnostic", xscale="log")
                ax.legend(fontsize=6, ncol=3)
                name = f"{case}/Fig04_top3_discovery"
                save(fig, out / name, prefix + "\nFCI weights identify evaluation targets only; these oracle values are not used for selection or stopping. Actual target count can be<3.")
                made.append(name)
    for path in sorted((root / "mixtures").glob("*/readouts.parquet")):
        df = pd.read_parquet(path)
        case = path.parent.name
        for method, group in df.groupby("method"):
            fig, ax = plt.subplots(figsize=(7, 4.8))
            for policy, arm in group.groupby("policy"):
                means = arm.groupby("shots_total").energy_error_mHa.mean()
                ax.plot(means.index, means, "o-", ms=3, label=policy)
            ax.set(xlabel="Total evaluation shots including pilots", ylabel="Mean QSCI error (mHa)", title=f"{case} / {method}: checkpoint mixtures", xscale="log", yscale="symlog")
            ax.axhline(1.59360143764, color=".4", ls="--", lw=.8)
            energy_limits(ax, group.groupby(["policy", "shots_total"]).energy_error_mHa.mean())
            ax.legend(fontsize=6)
            name = f"{case}/Fig16_multi_{method}"
            save(fig, out / name, "Post-growth checkpoint bank: historical construction is NOT free. Best-single calibration is retrospective and separately recorded.\nPilot costs included in evaluation budget; fixedR32 here, same-realized-d diagnostics in raw rows. Simulated shots only.")
            made.append(name)
    # Mechanistic source traces are complete independently of the new finite suite.
    for path in sorted((root / "operator-tracking").glob("*/important-probabilities.parquet")):
        case = path.parent.name
        data = pd.read_parquet(path)
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for ax, method in zip(axs, ("ceo_ovp", "qubit")):
            arm = data[data.method == method]
            targets = arm.groupby("configuration_id").fci_weight.max().sort_values(ascending=False).head(5).index
            for target in targets:
                g = arm[arm.configuration_id == target].sort_values("step")
                ax.plot(g.step, g.probability, label=str(target), lw=1)
            ax.set(xlabel="ADAPT growth step", ylabel="Measurement probability", title=f"{case} / {method}")
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=6, title="Configuration ID", title_fontsize=7)
        name = f"{case}/Fig11_probability_tracks"
        save(fig, out / name, "Oracle-important target identities for diagnosis only. Changes combine operator addition and full reoptimization, not isolated causality.\nFull configuration first/max/final records and step-level transitions are in operator-tracking Parquet tables.")
        made.append(name)
    save_json(root / "figure-status.json", {"generated_figures": len(made), "formats": ["png", "svg", "pdf"], "figures": made,
              "full_study_18_figure_requirements_not_all_satisfied": True, "no_placeholder_data": True})
    return len(made)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print("Figures:", plot_all(args.output))


if __name__ == "__main__":
    main()
