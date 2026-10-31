"""Measured B1 comparisons, all-cell coverage and clearly scoped costs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t, beta
from .common import save_json
from .study_figures import energy_limits


PRIMARY = ("source_final_ceo_ovp", "source_final_ceo_mvp", "source_final_lucj_2",
           "lucj_ccsd_2_compressed", "lucj_ccsd_1_vqe", "lucj_ccsd_2_vqe", "lucj_mp2_2_vqe", "lucj_ccsd_2_qsci")


def make_report(root):
    root = Path(root); mf = json.loads((root / "manifest.json").read_text()); source = Path(mf["source"])
    if not (root / "evaluation-status.json").exists():
        raise RuntimeError("Full registered B1 readout required before report")
    summary = pd.read_csv(root / "readout-summary.csv")
    summary["mean_ci90_halfwidth_mHa"] = t.ppf(.95, summary.repeats-1)*summary.std_error_mHa/np.sqrt(summary.repeats)
    success = np.rint(summary.success_probability_estimate*summary.repeats).astype(int)
    summary["success_ci90_low"] = [float(beta.ppf(.05, k, n-k+1)) if k else 0. for k, n in zip(success, summary.repeats)]
    summary["success_ci90_high"] = [float(beta.ppf(.95, k+1, n-k)) if k < n else 1. for k, n in zip(success, summary.repeats)]
    rows = []; source_costs = pd.read_csv(source / "analysis/construction_costs.csv")
    for case in mf["case_ids"]:
        for spec in mf["methods"]:
            r = json.loads((root / "cells" / case / spec["name"] / "result.json").read_text())
            row = {"case_id": case, "method": spec["name"], "construction_status": r["status"],
                   "optimizer_success": r.get("optimizer_success"), "stationary": r.get("stationary"),
                   "new_full_construction_seconds": r["construction_wall_seconds"],
                   "objective_evaluations": r.get("objective_evaluations", 0),
                   "simulated_construction_shots": r.get("simulated_construction_shots"),
                   "construction_qpu_shots": None, "input_vqe_error_mHa": r.get("energy_error_mHa")}
            if "initialization" in r:
                init = r["initialization"]
                row["initialization_dependency_total_seconds"] = init["dependency_total_seconds"]
                row["initialization_dependency_new_seconds"] = init["dependency_new_seconds"]
                row["from_scratch_construction_seconds_estimate"] = r["construction_wall_seconds"]+init["dependency_total_seconds"]-init["dependency_new_seconds"]
                row["resource_scope"] = "new logical Z-sampling circuit; preparation resources separate in result.json"
                row.update({k: r["resources"][k] for k in ("cnot_count", "cnot_depth", "total_depth", "parameter_count")})
            rows.append(row)
        for method in sorted((source / "checkpoints" / case).iterdir()):
            cps = sorted(method.glob("step-*.json"))
            if not cps:
                rows.append({"case_id": case, "method": "source_final_"+method.name, "construction_status": "SOURCE_MISSING_PRESERVED"})
                continue
            cp = json.loads(cps[-1].read_text()); g = source_costs[(source_costs.case_id == case) & (source_costs.method == method.name)]
            row = {"case_id": case, "method": "source_final_"+method.name, "construction_status": "SOURCE_FIXED_FINAL_REUSED",
                   "source_final_step": cp["step"], "input_vqe_error_mHa": 1000*cp["vqe_energy_error"],
                   "historical_trajectory_seconds_as_recorded": cp.get("trajectory_seconds"),
                   "historical_numerical_optimizer_seconds": float(g.optimizer_seconds.iloc[0]) if len(g) else None,
                   "historical_backend": str(g.numerical_backend.iloc[0]) if len(g) else None,
                   "historical_device": str(g.device.iloc[0]) if len(g) else None,
                   "method_required_ccsd_seconds": cp.get("ccsd_initialization_seconds", 0.) if cp.get("ccsd_initialization_required") else 0.,
                   "construction_qpu_shots": None, "cost_scope": "historical measurements, not a matched-device full-wall comparison",
                   **{k: cp["resources"][k] for k in ("cnot_count", "cnot_depth", "total_depth", "parameter_count")}}
            rows.append(row)
    resources = pd.DataFrame(rows)
    resources.to_csv(root / "resources-and-construction-costs.csv", index=False)
    data = summary.merge(resources, on=["case_id", "method"], how="left", validate="many_to_one")
    recovery = []
    for case, g in data.groupby("case_id"):
        meta = json.loads((source / "references" / case / "reference.json").read_text())
        g = g.copy(); correlation_mHa = (meta["rhf_energy"]-meta["fci_energy"])*1000
        g["recovered_correlation_percent"] = 100*(1-g.mean_error_mHa/correlation_mHa) if correlation_mHa > 1e-12 else np.nan
        g["recovery_mean_ci90_halfwidth_percent"] = 100*g.mean_ci90_halfwidth_mHa/correlation_mHa if correlation_mHa > 1e-12 else np.nan
        g["recovery_reference"] = "posthoc FCI correlation energy; not configuration recovery or a selection feature"
        recovery.append(g)
    data = pd.concat(recovery, ignore_index=True)
    data.to_csv(root / "comparison-with-recovery.csv", index=False)
    figures = root / "figures"; figures.mkdir(exist_ok=True); generated = []
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for case, g in data.groupby("case_id"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        yvalues = []
        for method in PRIMARY:
            s = g[(g.method == method) & (g.R_cap == 128)].sort_values("shots")
            if not len(s):
                continue
            axes[0].errorbar(s.shots, s.mean_error_mHa, yerr=s.mean_ci90_halfwidth_mHa, fmt="o-", markersize=3, capsize=2, label=method)
            axes[1].errorbar(s.shots, s.recovered_correlation_percent, yerr=s.recovery_mean_ci90_halfwidth_percent, fmt="o-", markersize=3, capsize=2)
            yvalues += (s.mean_error_mHa-s.mean_ci90_halfwidth_mHa).tolist()+(s.mean_error_mHa+s.mean_ci90_halfwidth_mHa).tolist()
        axes[0].set_xscale("log"); axes[0].set_yscale("symlog", linthresh=.01)
        axes[0].set(xlabel="Raw simulated readout shots", ylabel="QSCI error (mHa), R cap128")
        axes[1].set_xscale("log"); axes[1].set(xlabel="Raw simulated readout shots", ylabel="FCI correlation energy recovered (%)")
        axes[0].axhline(1.5936014376, color="gray", ls="--", lw=.8)
        energy_limits(axes[0], yvalues)
        scope = "QUICK VALIDATION" if mf["mode"] == "quick" else "BOUNDED STANDARD"
        fig.suptitle(f"{case} — {scope}; n={mf['budget']['evaluation_repeats']}, approximate pointwise t-90% mean intervals")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=7)
        name = f"{case}-B1-shots-and-recovery"; generated.append(name)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(figures / f"{name}.{ext}", dpi=220)
        plt.close(fig)
        for M in (1000, 100000):
            s = g[(g.shots == M) & (g.R_cap == 128)]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            for method, part in s.groupby("method"):
                if part.cnot_count.isna().all():
                    continue
                marker = "o" if method in PRIMARY else "."
                label = method if method in PRIMARY else None
                axes[0].scatter(part.cnot_count, part.mean_error_mHa, marker=marker, label=label)
                axes[1].scatter(part.cnot_depth, part.mean_error_mHa, marker=marker)
            for ax, xlab in zip(axes, ("Logical sampling CNOT count", "Logical sampling CNOT depth")):
                ax.set_yscale("symlog", linthresh=.01); ax.set(xlabel=xlab, ylabel="Independent QSCI error (mHa)")
                energy_limits(ax, s.mean_error_mHa)
            fig.suptitle(f"{case} — {scope}; M{M}, R cap128; small dots retain other controls/seeds")
            fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=7)
            name = f"{case}-B1-resources-M{M}"; generated.append(name)
            for ext in ("png", "svg", "pdf"):
                fig.savefig(figures / f"{name}.{ext}", dpi=220)
            plt.close(fig)
    cells = pd.read_csv(root / "all-cell-status.csv")
    status_counts = cells.status.value_counts().to_dict()
    lines = ["# B1 LUCJ強化比較 — 実測値と限界", "", f"Mode`{mf['mode']}`。{len(mf['case_ids'])}条件、{len(cells)} / {len(mf['case_ids'])*len(mf['methods'])}予定セル。全研究・Milestone2全体の完了ではない。", "",
             "## 実行状態", "", json.dumps(status_counts, ensure_ascii=False), "",
             "FIXEDは固定初期値、OPTIMIZER_TERMINATEDは登録optimizerの停止、BOUNDED_SEARCH_TERMINATEDは有限予算での終了。後者を勾配収束や大域最適化と呼ばない。stationaryがnullなら未確認。全セルをCSVに残し、良かったseedだけ選ばない。", "",
             "## 数値表（M1000、R128、全手法）", "", "| 条件 | 手法 | 誤差 mHa | 相関energy回収 % | CNOT | CNOT depth |", "|---|---|---:|---:|---:|---:|"]
    for r in data[(data.shots == 1000) & (data.R_cap == 128)].itertuples():
        lines.append(f"| {r.case_id} | {r.method} | {r.mean_error_mHa:.6g} | {r.recovered_correlation_percent:.5g} | {r.cnot_count:g} | {r.cnot_depth:g} |")
    lines += ["", "## 費用と比較の境界", "", "- 全て古典simulation。実機shotsは0。raw-count shot数は仮想測定回数であり、その回数だけ回路simulationを実行したという意味ではない。", "- 新LUCJの全構築wall、初期化依存費用、再利用したCCSD/MP2と新規作業を分離。source側は記録されたhistorical trajectory/optimizer timeで、同一機材・同一定義の完全な速度比較ではない。異なる列を割ってspeedupと呼ばない。", "- VQEの正確な古典objective callsはQPU shotsに変換できていない。QSCI-guidedのpilot shotsは別に実数計上。constructionQPUshotsの未知を0にしない。", "- 回収率はFCIに対する相関energy回収で、Configuration Recoveryの割合とは別。FCIは構築・選択に使っていない。", "- 共通R結果とpairwise-ceo-matched-d.parquetを併用。全手法共通dはHF-onlyを含むためd=1になりうる。これだけで方法の有用性を判定しない。", "- 1/2layerやtruncatedの有限予算結果であり、任意のLUCJ・全optimizerに対する優越性を証明していない。特にlinear-method最適化はまだ本cohortに含めていない。", "- B2–B4、FCI不要停止・RAACS・大規模化・fullSQD/noiseは残scope。", ""]
    (root / "REPORT_JA.md").write_text("\n".join(lines))
    result = {"figures": len(generated), "formats": ["png", "svg", "pdf"], "names": generated,
              "all_cells": len(cells), "status_counts": status_counts, "whole_study_complete": False}
    save_json(root / "report-status.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    make_report(p.parse_args().output)
