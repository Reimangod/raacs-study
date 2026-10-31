"""Derived B2 report. Never changes growth, counts, or the strict v1 audit."""
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
from .common import file_digest, save_json
from .study_figures import energy_limits


def correlation_recovery(error_mha, hf, fci):
    denominator = 1000*(hf-fci)
    return 100*(1-np.asarray(error_mha)/denominator) if denominator > 1e-10 else np.nan


def partition_coverage(cells):
    """Readout coverage additionally retains missing source controls, not growth."""
    source = cells.method.str.startswith("source_final_")
    return cells[~source].copy(), cells[source].copy()


def report(root):
    root = Path(root)
    mf = json.loads((root / "manifest.json").read_text())
    source = Path(mf["source"])
    readout = root / "independent-readouts-v2"
    status = json.loads((readout / "status.json").read_text())
    audit = json.loads((root / "validation-available-v2.json").read_text())
    if status["status"] != "INDEPENDENT_FINAL_READOUT_COMPLETE" or audit["unexpected_failures"]:
        raise RuntimeError("Complete validated available cohort required")
    out = root / "derived-report-v2"
    out.mkdir(exist_ok=True)
    bound = {"implementation_sha256": file_digest(__file__),
             "growth_manifest_sha256": file_digest(root / "manifest.json"),
             "readout_summary_sha256": file_digest(readout / "summary.csv"),
             "available_audit_sha256": file_digest(root / "validation-available-v2.json"),
             "scope": "posthoc evaluation and reporting only; no construction/selection changes"}
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()) != bound:
        raise RuntimeError("Derived report identity changed; use successor directory")
    save_json(out / "manifest.json", bound)
    resource_rows = []
    for case in mf["cases"]:
        with sqlite3.connect(readout / case / "readouts.sqlite") as db:
            for method, raw in db.execute("SELECT method,metadata FROM checkpoints"):
                meta = json.loads(raw)
                row = {k: v for k, v in meta.items() if k not in ("resources", "probability")}
                row.update(meta["resources"])
                row["method"] = method
                row["cost_scope"] = "source historical trajectory, not matched-device full wall" if method.startswith("source_final_") else "new growth full wall, including sampling/CI/compilation; shared reference setup separate"
                resource_rows.append(row)
    resources = pd.DataFrame(resource_rows)
    if resources.duplicated(["case_id", "method"]).any():
        raise RuntimeError("Nonunique sampler resource identity")
    resources.to_csv(out / "resources-and-construction-costs.csv", index=False)
    cells, source_missing = partition_coverage(pd.read_csv(readout / "all-growth-cells.csv"))
    if len(cells) != audit["planned_trajectories"]:
        raise RuntimeError("Missing planned growth cells")
    cells.to_csv(out / "all-planned-growth-cells.csv", index=False)
    source_missing.to_csv(out / "missing-source-controls.csv", index=False)
    data = pd.read_csv(readout / "summary.csv").merge(resources, on=["case_id", "method"], how="left", validate="many_to_one")
    frames = []
    for case, group in data.groupby("case_id"):
        meta = json.loads((source / "references" / case / "reference.json").read_text())
        group = group.copy()
        group["recovered_correlation_percent"] = correlation_recovery(group.conditional_mean_error_mHa, meta["rhf_energy"], meta["fci_energy"])
        group["recovery_definition"] = "posthoc FCI correlation energy; NOT configuration recovery"
        frames.append(group)
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(out / "comparison-with-recovery.csv", index=False)
    figs = out / "figures"; figs.mkdir(exist_ok=True)
    names = []
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    styles = [("source_final_ceo_ovp", "CEO-OVP final", "#2764a5"),
              ("source_final_ceo_mvp", "CEO-MVP final", "#008c69"),
              ("source_final_lucj_2", "Fixed CCSD LUCJ2", "#555555"),
              ("adapt_qsci_original/", "ADAPT-QSCI", "#d07716"),
              ("hf_separated_adapt_qsci_MP_m0_2/", "HF-separated MP2det", "#9958ae")]
    for case, g in data.groupby("case_id"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        plotted = []
        for prefix, label, color in styles:
            s = g[(g.view == "common_HF_inserted") & (g.R_cap == 128)]
            s = s[s.method == prefix] if prefix.startswith("source_final_") else s[s.method.str.startswith(prefix) & (s.construction_Ns == 1000) & (s.construction_R == 128)]
            for j, (method, part) in enumerate(s.groupby("method")):
                part = part.sort_values("shots")
                axes[0, 0].plot(part.shots, part.conditional_mean_error_mHa, "o-", color=color, lw=1, markersize=3, alpha=.7, label=label if j == 0 else None)
                axes[0, 1].plot(part.shots, part.recovered_correlation_percent, "o-", color=color, lw=1, markersize=3, alpha=.7)
                point = part[part.shots == 1000]
                axes[1, 0].scatter(point.cnot_count, point.conditional_mean_error_mHa, color=color, s=24)
                axes[1, 1].scatter(point.cnot_depth, point.conditional_mean_error_mHa, color=color, s=24)
                plotted.extend(part.conditional_mean_error_mHa.dropna())
        for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
            ax.set_yscale("symlog", linthresh=.01)
            ax.axhline(1.5936014376, color="gray", lw=.8, ls="--")
            ax.set_ylabel("Independent QSCI error (mHa)")
            energy_limits(ax, plotted)
        for ax in axes[0]:
            ax.set_xscale("log"); ax.set_xlabel("Raw simulated readout shots (pilot separate)")
        axes[0, 1].set_ylabel("FCI correlation energy recovered (%)")
        axes[1, 0].set_xlabel("Logical sampling CNOT count, M1000")
        axes[1, 1].set_xlabel("Logical sampling CNOT depth, M1000")
        fig.suptitle(f"{case} — B2; R cap128, 20 independent readout draws / growth seed\nNative growth M1000/R128, each of 3 seeds shown; no best-seed selection")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=8)
        name = case+"-B2-independent-comparison"; names.append(name)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(figs / f"{name}.{ext}", dpi=220)
        plt.close(fig)
    availability = cells.assign(available=cells.task_id.notna()).groupby("case_id").available.agg(["sum", "count"])
    lines = ["# B2 ADAPT-QSCI／HF分離版：独立評価", "",
             f"予定{len(cells)}軌道、利用可能{audit['validated_available']}、初期化不能{audit['initializer_missing']}。保存checkpoint {audit['checkpoint_count']}。",
             f"独立draw {status['draws']:,}、仮想readout shots {status['simulated_readout_raw_shots']:,}、実機shots 0。",
             "",
             "## 何を同じにしたか", "",
             "同一Hamiltonian・MO・活性空間、同じreadout M、同じR上限。common_HF_insertedと原手法nativeの両方を保存。pairwise-matched-d.parquetはCEOと同一実CI次元の対照。図はcommon HF viewで、HF分離による構築目的の差とreadout時HF処理の差を混同しない。",
             "原ADAPT-QSCIはCI波動関数由来のfull commutator勾配で演算子を選び、新規角度のみを最適化。HF分離版はHFと非HF-CIの一般化2×2固有値・overlap微分を含む。MP partition top2配置による初期化は登録variantであり、論文図の完全再現ではない。",
             "", "## 全条件の欠測", "", "| 条件 | 利用可能 / 予定 |", "|---|---:|"]
    for case, row in availability.iterrows():
        lines.append(f"| {case} | {int(row['sum'])} / {int(row['count'])} |")
    lines += ["", "## M1000・R128での全readout結果（3 growth seedsを個別表示）", "",
              "| 条件 | 手法・seed | view | 誤差 mHa | 相関energy回収 % | CNOT | depth | readout成功率 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    select = data[(data.shots == 1000) & (data.R_cap == 128) & (data.view == "common_HF_inserted")]
    for row in select.itertuples():
        lines.append(f"| {row.case_id} | {row.method} | common HF | {row.conditional_mean_error_mHa:.6g} | {row.recovered_correlation_percent:.6g} | {row.cnot_count:g} | {row.cnot_depth:g} | {row.unconditional_success_probability:.3f} |")
    lines += ["", "## 重要な限界", "",
              "- NATIVE_QSCI_ENERGY_PLATEAUはQSCI成長の停止であり、VQE勾配収束でも絶対精度達成でもない。STEP_CAPは打ち切り。",
              "- 96軌道は初期化で停止し、有効な回路・energyはない。8条件のHF分離版が欠測。単純なgreedy pool準備の制限であり、targetの大域的な準備不能やHF分離理論の否定ではない。失敗試行の個別wallは元handlerで欠測し、0としない。",
              "- 旧strict v1 auditは全720成功を要求してFAIL。available-v2 auditは624軌道の数学・カウントを検証し、96欠測を維持したPASS。旧FAILを消していない。",
              "- 回収率は相関energy回収率。Configuration Recoveryはこのcohortでは実施していない。",
              "- 成長pilot shotsはconstruction_simulated_raw_shots、最終独立評価shotsは別。共通R/native viewを二重に量子測定費用へ加算しない。",
              "- 20 drawsの成功率推定・90%区間を保存。20/20でも90%成功の確率下限保証とはしない。geometryは独立分子ではなく、この表から母集団優越性を推定しない。",
              "- 小系statevectorによるsampler実行。CI勾配はnamed配置に限定するが、これだけで20–40qubit実用backendの実証とはしない。",
              "- 論文poolは粒子数・Szを厳密には保存しないため、無効配置のshotsもraw budgetに含める。無料の条件付き再正規化は禁止。",
              "- source CEOはhistorical費用、新B2は現在CPUのfull wall。異なる費用scopeを割って速度優位を主張しない。",
              "- B1/B2以外のベースライン、FCI不要停止、RAACS、大規模化、full SQD/noiseは未完。", ""]
    (out / "REPORT_JA.md").write_text("\n".join(lines))
    result = {"status": "B2_AVAILABLE_COHORT_REPORTED_WITH_96_INITIALIZER_MISSING", "figures": len(names), "formats": ["png", "svg", "pdf"],
              "planned_growth": len(cells), "available_growth": audit["validated_available"], "missing_growth": audit["initializer_missing"],
              "summary_rows": len(data), "independent_readout": status, "whole_study_complete": False}
    save_json(out / "report-status.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True)
    print(json.dumps(report(parser.parse_args().output)))
