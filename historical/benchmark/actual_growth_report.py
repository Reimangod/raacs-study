"""Fresh actual-growth report, including capped cases and posthoc-only labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from .common import file_digest, save_json


def report(root):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text())
    validation = json.loads((root/"evaluation-validation.json").read_text())
    assert validation["status"] == "ACTUAL_FINAL_COUNTS_AND_PAIRS_VALIDATED"
    out = root/"derived-report-v1"; out.mkdir(exist_ok=True)
    frames = []; costs = []
    for item in cfg["physical_problems"]:
        case = item["case_id"]; problem = json.loads((root/"problems"/case/"problem.json").read_text())
        refpath = root/"oracle-reporting-reference-v1"/f"{case}.json"
        ref = json.loads(refpath.read_text()) if refpath.exists() else None
        frame = pd.read_parquet(root/"readouts"/case/"readouts.parquet")
        frame["case_id"] = case; frame["molecule"] = item["spec"]["molecule"]
        frame["fresh_active_space"] = case.endswith("core0") and frame["molecule"].iloc[0] in ("LiH", "BeH2")
        frame["regret_mHa"] = 1000*(frame.energy_Ha-frame.native_energy_Ha)
        frame["paired_regret_mHa"] = 1000*(frame.paired_energy_Ha-frame.paired_native_energy_Ha)
        frame["error_mHa"] = (frame.energy_Ha-ref["energy_Ha"])*1000 if ref else np.nan
        frame["chemical_success"] = (frame.error_mHa <= 1.5936014376).astype(float) if ref else np.nan
        for method in cfg["methods"]:
            native = json.loads((root/"cells"/case/method/"native_terminal/result.json").read_text())
            for policy in cfg["policies"]:
                folder = root/"cells"/case/method/policy; r = json.loads((folder/"result.json").read_text())
                pilot_gate_shots = sum(64*c["resources"]["cnot_count"] for path in (folder/"steps").glob("*.json") if (c := json.loads(path.read_text()))["pilot"])
                mask = (frame.method == method)&(frame.policy == policy)
                for key in ("cnot_count", "cnot_depth", "parameter_count", "total_depth"):
                    frame.loc[mask, key] = r["resources"][key]
                    base = native["resources"][key]
                    frame.loc[mask, key+"_reduction_percent"] = 100*(1-r["resources"][key]/base) if base else np.nan
                frame.loc[mask, "final_step"] = r["last_step"]
                frame.loc[mask, "construction_status"] = r["status"]
                frame.loc[mask, "native_status"] = native["status"]
                frame.loc[mask, "deployment_gate_shots"] = frame.loc[mask, "final_shots"]*r["resources"]["cnot_count"]+pilot_gate_shots
                costs.append({"case_id": case, "method": method, "policy": policy, "status": r["status"],
                    "native_status": native["status"], "last_step": r["last_step"], "native_last_step": native["last_step"],
                    "objective_calls": r["objective_evaluations"], "native_objective_calls": native["objective_evaluations"],
                    "objective_calls_avoided": native["objective_evaluations"]-r["objective_evaluations"],
                    "pool_gradient_components": r["pool_gradient_components_computed"],
                    "actual_construction_wall_seconds": r["actual_attempt_wall_seconds"],
                    "native_construction_wall_seconds": native["actual_attempt_wall_seconds"],
                    "pool_setup_seconds": r["pool_setup_seconds_this_attempt"], "optimizer_seconds": r["optimizer_seconds"],
                    "pilot_classical_seconds": r["pilot_classical_seconds"], "pilot_raw_model_shots": r["pilot_raw_model_shots"],
                    "process_RSS_hwm_bytes": r["measured_process_RSS_hwm_bytes"],
                    "shared_problem_construction_seconds_separate": problem["construction_wall_seconds"],
                    "posthoc_oracle_reference_seconds_separate": ref["oracle_reference_seconds"] if ref else None,
                    "hardware_shots": 0, "primary_selected_policy": policy == cfg["primary_policy"]})
        frames.append(frame)
    f = pd.concat(frames, ignore_index=True); f.to_parquet(out/"all-independent-final-readouts.parquet", index=False)
    cost = pd.DataFrame(costs); cost.to_csv(out/"actual-construction-costs.csv", index=False)
    metrics = ["energy_Ha", "native_energy_Ha", "regret_mHa", "paired_regret_mHa", "error_mHa", "chemical_success", "actual_d", "paired_d",
               "cnot_count", "cnot_depth", "parameter_count", "total_depth", "cnot_count_reduction_percent", "cnot_depth_reduction_percent",
               "parameter_count_reduction_percent", "total_depth_reduction_percent", "pilot_shots", "final_shots", "deployment_gate_shots"]
    summary = f.groupby(["case_id", "molecule", "method", "policy", "total_M", "cap", "construction_status", "native_status"], as_index=False)[metrics].mean()
    summary.to_csv(out/"all-case-means.csv", index=False)
    primary_cases = summary[summary.policy == cfg["primary_policy"]]
    primary_molecules = primary_cases.groupby(["molecule", "method", "total_M", "cap"], as_index=False)[metrics].mean()
    primary_summary = primary_molecules.groupby(["method", "total_M", "cap"], as_index=False)[metrics].mean()
    primary_summary.to_csv(out/"primary-equal-molecule-summary.csv", index=False)
    success = []; thresholds = []
    for key, g in f.groupby(["case_id", "method", "policy", "total_M", "cap"]):
        n = int(g.chemical_success.notna().sum()); k = int(g.chemical_success.sum())
        success.append(dict(zip(("case_id", "method", "policy", "total_M", "cap"), key), n=n, successes=k,
            estimate=k/n if n else None, lower90=float(beta.ppf(.05, k, n-k+1)) if k else 0. if n else None,
            upper90=float(beta.ppf(.95, k+1, n-k)) if n and k<n else 1. if n else None))
    sf = pd.DataFrame(success); sf.to_csv(out/"chemical-success-pointwise90.csv", index=False)
    for key, g in sf.groupby(["case_id", "method", "policy", "cap"]):
        observed = g[g.estimate >= .9]; lower = g[g.lower90 >= .9]
        thresholds.append(dict(zip(("case_id", "method", "policy", "cap"), key),
             minimum_tested_M_observed90=int(observed.total_M.min()) if len(observed) else None,
             minimum_tested_M_lowerCI90=int(lower.total_M.min()) if len(lower) else None,
             status="OBSERVED_TEST_GRID_ATTAINMENT" if len(observed) else "NO_ATTAINMENT_ON_TESTED_GRID",
             scope="R/frequency finite-shot errors may be nonmonotone in M; not a continuous shot lower bound"))
    pd.DataFrame(thresholds).to_csv(out/"shots-to90-tested-grid.csv", index=False)
    figures = []
    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "svg", "pdf"):
            p = out/"figures"/f"{name}.{ext}"; p.parent.mkdir(exist_ok=True)
            fig.savefig(p, dpi=180, bbox_inches="tight"); figures.append(str(p.relative_to(root)))
        plt.close(fig)
    for case, group in summary[summary.cap == 64].groupby("case_id"):
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        for (method, policy), g in group.groupby(["method", "policy"]):
            g = g.sort_values("total_M"); label = f"{method}/{policy}"
            ax[0].plot(g.total_M, g.error_mHa, ".-", label=label)
            ax[1].scatter(g.cnot_count, g.regret_mHa, s=15)
            ax[2].scatter(g.cnot_depth, g.paired_regret_mHa, s=15)
        ax[0].set_xscale("log"); ax[0].set_yscale("symlog", linthresh=.1); ax[0].set_xlabel("Total model shots (includes pilots)"); ax[0].set_ylabel("QSCI error (mHa; oracle reporting)")
        ax[0].axhline(1.5936014376, lw=.7, ls="--", color="grey")
        ax[1].set_xlabel("Logical sampling CNOT"); ax[1].set_ylabel("Regret vs native (mHa)")
        ax[2].set_xlabel("Logical sampling CNOT depth"); ax[2].set_ylabel("Paired actual-d regret (mHa)")
        for a in ax[1:]: a.set_yscale("symlog", linthresh=.1); a.axhline(0, color="grey", lw=.7)
        for a in ax: a.grid(alpha=.2)
        fig.suptitle(f"{case}: actual fresh growth / R64 / fixed policies, not posthoc winners; one construction seed")
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.01), fontsize=7, ncol=4)
        save(fig, case)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    primary = cost[cost.primary_selected_policy]
    for a, x, y, label in ((ax[0], "native_objective_calls", "objective_calls", "Actual optimizer calls"),
                            (ax[1], "native_construction_wall_seconds", "actual_construction_wall_seconds", "Observed construction wall seconds")):
        for method, g in primary.groupby("method"): a.scatter(g[x], g[y], label=method)
        bound = max(primary[x].max(), primary[y].max()); a.plot([0, bound], [0, bound], "--", color="grey", lw=.7)
        a.set_xlabel("Native: "+label); a.set_ylabel("Training-selected gradient: "+label); a.grid(alpha=.2)
    ax[0].legend(); fig.suptitle("Primary frozen policy / same M4 runtime, up to2concurrent workers; not a speedup certificate")
    save(fig, "primary-construction-cost")
    statuses = cost.status.value_counts().to_dict()
    text = ["# Phase C：新距離・活性空間で実際に成長・停止した比較", "",
        f"{len(cfg['physical_problems'])}物理問題、{len(cost)}実行。主方式は旧LOMOが選んだgradient。その他の固定ルールは診断対照であり、結果を見て主方式に昇格させない。", "",
        "## 終了状態", "", *[f"- {k}: {v}" for k, v in statuses.items()], "",
        "上限や時間切れは収束と呼ばない。構築は独立に実行し、停止後に後続optimizerを回してからprefixとして切り出したものではない。",
        "", "## 事前固定された主方式（R64、分子平均を等重み）", "",
        "| Pool | 総M | native比regret mHa | CNOT削減 % | Depth削減 % |", "|---|---:|---:|---:|---:|",
        *[f"|{r['method']}|{r['total_M']}|{r['regret_mHa']:.4f}|{r['cnot_count_reduction_percent']:.2f}|{r['cnot_depth_reduction_percent']:.2f}|" for r in primary_summary[primary_summary.cap == 64].to_dict("records")],
        "", "## 公平性・費用・限界", "",
        "- 総Mにはpilotが含まれ、最終評価は独立countsだけで行う。HF以外の未観測配置・FCI配置を挿入しない。共通Rとペア実d一致を両方保存。",
        "- 全構築と独立最終countsの監査終了後にのみ、別のoracle-reporting-reference-v1で小規模FCI参照を作る。PySCFとJW sector最低固有値を独立照合。FCIは本レポートの誤差ラベルのみで、構築・停止・候補選択・QSCIの入力ではない。",
        "- 1分子の複数距離・活性空間は独立分子ではない。構築seedは1つ、最終draw反復は独立初期値の反復ではない。quickとstandardで共通のpilot seed/重複物理prefixを使うため、その重複を独立replicationと数えない。最終draw namespaceはmodeで分離。",
        "- 実際に省けたoptimizer calls/iterations/pool componentsをconstruction-pairs.csvに保存。wall timeは観測値で、並列競合・cold setupを含むため速度向上の保証ではない。",
        "- 同じ物理問題の積分生成は共有依存として別計上。各セルのcold pool、optimizer、pilot、compile/verificationと、事後oracle参照費用は区別する。古典callsを量子shotsへ換算しない。",
        "- nativeが時間切れで先に終わった場合、そこを超えるpolicy点はnative-prefix一致未確認として保持する。真の収束点との圧縮率にしない。",
        "- 最大16量子ビットのfull-statevector qualification。実機shots0、native device未指定。20–40+qubit分子の拡張やfull SQD/noiseは未達。",
        "- 化学精度成功率は点ごとの90%二項区間。観測90%と区間下限90%を分離し、未達はtested-grid未達として残す。", ""]
    (out/"REPORT_JA.md").write_text("\n".join(text))
    result = {"status": "ACTUAL_GROWTH_AND_INDEPENDENT_FINAL_EVALUATION_REPORTED", "actual_construction_cells": len(cost),
              "all_policy_case_summary_rows": len(summary), "figures": len(figures)//3, "figure_files": figures,
              "hardware_shots": 0, "one_construction_seed_limitation": True, "scalable_backend": False,
              "whole_study_complete": False, "reporter_sha256": file_digest(__file__)}
    save_json(out/"manifest.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(report(p.parse_args().output)))
