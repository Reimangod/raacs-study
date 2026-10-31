"""Descriptive PhaseA tables; no selection/training or unseen-condition claims."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, t

from .common import rng_for, save_json
from .metrics import discovery_probabilities
from .study.core import ADAPT, SHOTS


def interpret(root):
    root = Path(root)
    output = root / "analysis"
    full = pd.read_parquet(output / "readout-summary.parquet")
    primary = full[full.primary_record & full.complete_group]
    inv = pd.DataFrame(json.loads((root / "checkpoint-inventory.json").read_text()))
    final_keys = inv[inv.primary].groupby(["case_id", "method"]).step.max().reset_index()
    final = primary.merge(final_keys, on=["case_id", "method", "step"])
    capacity = final[final.selection_rule == "frequency"].pivot_table(index=["case_id", "molecule", "method", "step", "shots"], columns="R_cap", values=["mean_error_mHa", "mean_dimension", "success_probability_estimate"])
    capacity.columns = [f"{a}_R{b}" for a, b in capacity.columns]
    capacity = capacity.reset_index()
    capacity["improvement_R32_to_R512_mHa"] = capacity.mean_error_mHa_R32 - capacity.mean_error_mHa_R512
    capacity.to_csv(output / "final-capacity-effects.csv", index=False)
    ranking = final[final.R_cap == 32].pivot_table(index=["case_id", "molecule", "method", "shots"], columns="selection_rule", values=["mean_error_mHa", "mean_dimension"])
    ranking.columns = [f"{a}_{b}" for a, b in ranking.columns]
    ranking = ranking.reset_index()
    ranking["uniform_minus_frequency_error_mHa"] = ranking.mean_error_mHa_uniform_observed - ranking.mean_error_mHa_frequency
    ranking["dimension_difference"] = ranking.mean_dimension_uniform_observed - ranking.mean_dimension_frequency
    assert np.allclose(ranking.dimension_difference, 0, atol=1e-12)
    ranking.to_csv(output / "final-ranking-control-effects.csv", index=False)
    # Individual target-discovery probabilities, not only mean Top-K recalls.
    targets = pd.read_parquet(output / "individual-m90.parquet")
    per_target = []
    for M in SHOTS:
        arm = targets.copy()
        arm["shots"] = M
        arm["discovery_probability"] = discovery_probabilities(arm.probability.to_numpy(), M)
        arm["target_scope"] = "FCI_ORACLE_TOP50_NONHF_EVALUATION_ONLY"
        per_target.append(arm)
    target_rows = pd.concat(per_target, ignore_index=True)
    target_rows.to_parquet(output / "individual-target-discovery.parquet", index=False)
    # All new M/R conditions, separate from old same-sample historical minima.
    stat = pd.read_csv(Path(json.loads((root / "manifest.json").read_text())["source"]) / "all-cell-status.csv").set_index(["case_id", "method"])
    growth = primary[(primary.selection_rule == "frequency") & primary.method.isin(ADAPT)].merge(inv[["checkpoint_id", "cnot", "depth"]], on="checkpoint_id")
    stopping = []
    for (case, method, M, R), g in growth.groupby(["case_id", "method", "shots", "R_cap"]):
        g = g.sort_values("step")
        last = g.iloc[-1]
        best = g.sort_values(["mean_error_mHa", "cnot", "step"]).iloc[0]
        for tol in (0., .1, .5, 1.):
            p = g[g.mean_error_mHa <= best.mean_error_mHa + tol].iloc[0]
            stopping.append({"case_id": case, "method": method, "shots": M, "R_cap": R, "tolerance_mHa": tol,
                             "actual_terminal": stat.loc[(case, method), "status"], "last_saved_step": int(last.step),
                             "best_step": int(best.step), "selected_step": int(p.step),
                             "step_ratio": float(p.step/last.step) if last.step else None,
                             "cnot_ratio": float(p.cnot/last.cnot) if last.cnot else None,
                             "depth_ratio": float(p.depth/last.depth) if last.depth else None,
                             "cnot_reduction_percent": 100*(1-p.cnot/last.cnot) if last.cnot else None,
                             "depth_reduction_percent": 100*(1-p.depth/last.depth) if last.depth else None,
                             "selected_error_mHa": float(p.mean_error_mHa), "best_error_mHa": float(best.mean_error_mHa), "final_error_mHa": float(last.mean_error_mHa),
                             "scope": "NEW_DRAWS_RETROSPECTIVE_SAME_SAMPLE; ratios_to_last_saved_not_true_convergence"})
    new_stops = pd.DataFrame(stopping)
    new_stops.to_csv(output / "new-draw-retrospective-stops.csv", index=False)
    new_stops.to_parquet(output / "new-draw-retrospective-stops.parquet", index=False)
    historical = pd.read_parquet(root / "retrospective-stops.parquet")
    new_eval = primary[(primary.selection_rule == "frequency") & (primary.R_cap == 32)]
    readout_lookup = {(r.case_id, r.method, int(r.step), int(r.shots)): r for r in new_eval.itertuples()}
    held_choices = []
    for old in historical.itertuples():
        a = readout_lookup[(old.case_id, old.method, int(old.selected_step), int(old.shots))]
        b = readout_lookup[(old.case_id, old.method, int(old.last_saved_step), int(old.shots))]
        variance = a.std_error_mHa**2/a.repeats + b.std_error_mHa**2/b.repeats
        denominator = (a.std_error_mHa**2/a.repeats)**2/(a.repeats-1) + (b.std_error_mHa**2/b.repeats)**2/(b.repeats-1)
        dof = variance**2/denominator if denominator else None
        half = float(t.ppf(.95, dof)*np.sqrt(variance)) if dof else 0.
        if a.step == b.step:
            half = 0.  # Same evaluation IDs: no duplicated independent error term.
        delta = a.mean_error_mHa - b.mean_error_mHa
        held_choices.append({"case_id": old.case_id, "method": old.method, "shots": old.shots, "tolerance_mHa": old.tolerance_mHa,
                             "historically_selected_step": old.selected_step, "last_saved_step": old.last_saved_step,
                             "cnot_reduction_percent": old.cnot_reduction_percent, "depth_reduction_percent": old.depth_reduction_percent,
                             "new_selected_error_mHa": a.mean_error_mHa, "new_final_error_mHa": b.mean_error_mHa,
                             "new_delta_selected_minus_final_mHa": delta, "welch_ci90_low_mHa": delta-half, "welch_ci90_high_mHa": delta+half,
                             "mean_gap_within_registered_tolerance": delta <= old.tolerance_mHa,
                             "scope": "old_counts_select_point; disjoint_new_counts_evaluate_same_point; NOT_online_or_unknown_molecule_validation"})
    held = pd.DataFrame(held_choices)
    held.to_csv(output / "historical-choices-independent-readout.csv", index=False)
    mixes = pd.concat([pd.read_parquet(p) for p in sorted((root / "mixtures").glob("*/readouts.parquet"))], ignore_index=True)
    rows = []
    for key, group in mixes.groupby(["case_id", "method", "shots_total", "policy"]):
        n, success = len(group), int(group.chemical_accuracy.sum())
        mean, std = float(group.energy_error_mHa.mean()), float(group.energy_error_mHa.std(ddof=1))
        half = float(t.ppf(.95, n-1) * std / np.sqrt(n)) if n > 1 else None
        rows.append({"case_id": key[0], "method": key[1], "shots_total": key[2], "policy": key[3], "repeats": n,
                     "mean_error_mHa": mean, "ci90_low_mHa": mean-half if half is not None else None, "ci90_high_mHa": mean+half if half is not None else None,
                     "mean_matched_d_error_mHa": float(group.matched_error_mHa.mean()), "mean_common_d": float(group.matched_common_d.mean()),
                     "success_probability_estimate": success/n,
                     "success_ci90_low": float(beta.ppf(.05, success, n-success+1)) if success else 0.,
                     "success_ci90_high": float(beta.ppf(.95, success+1, n-success)) if success < n else 1.,
                     "mean_pilot_shots_in_budget": float(group.pilot_shots_included.mean()),
                     "mean_pilot_ci_calls": float(group.pilot_ci_calls.mean()), "mean_min_switches": float(group.minimal_switch_count.mean()),
                     "mean_gate_shot_proxy": float(group.gate_shot_proxy.mean()), "mean_depth_shot_proxy": float(group.depth_shot_proxy.mean()),
                     "mean_classical_policy_seconds": float(group.seconds.mean()),
                     "scope": "post-grown_bank; retrospective_single_best_calibration_separate; no_hardware_shots"})
    mix = pd.DataFrame(rows)
    source = Path(json.loads((root / "manifest.json").read_text())["source"])
    costs = pd.read_csv(source / "analysis/construction_costs.csv")
    costs = costs.rename(columns={k: "source_full_construction_"+k for k in costs.columns if k not in ("case_id", "method")})
    mix = mix.merge(costs, on=["case_id", "method"], how="left", validate="many_to_one")
    mix["construction_qpu_shots"] = None
    mix["construction_qpu_shots_status"] = "NOT_ESTABLISHED_FROM_CLASSICAL_OBJECTIVE_CALLS"
    mix.to_parquet(output / "mixture-summary.parquet", index=False)
    mix.to_csv(output / "mixture-summary.csv", index=False)
    # Equal-weight molecule summaries: geometry points never bootstrap as independent molecules.
    pair = mix.pivot_table(index=["case_id", "method", "shots_total"], columns="policy", values=["mean_error_mHa", "mean_matched_d_error_mHa"])
    pair.columns = [f"{a}_{b}" for a, b in pair.columns]
    pair = pair.reset_index()
    pair["molecule"] = pair.case_id.str.split("_R").str[0]
    bootstrap = []
    for method, g in pair.groupby("method"):
        for M, arm in g.groupby("shots_total"):
            for policy in ("equal_three", "pilot_diversity", "pilot_qsci_energy", "best_single_retrospective"):
                for metric in ("mean_error_mHa", "mean_matched_d_error_mHa"):
                    delta = arm[f"{metric}_{policy}"]-arm[f"{metric}_final"]
                    by_molecule = pd.DataFrame({"molecule": arm.molecule, "delta": delta}).groupby("molecule").delta.mean()
                    rng, seed = rng_for("descriptive-molecule-cluster-bootstrap-v1", method, int(M), policy, metric)
                    b = by_molecule.to_numpy()
                    resampled = rng.choice(b, size=(2000, len(b)), replace=True).mean(axis=1)
                    bootstrap.append({"method": method, "shots_total": M, "policy": policy, "metric": metric,
                                      "molecule_clusters": len(b), "equal_weight_molecule_mean_delta_mHa": float(b.mean()),
                                      "cluster_bootstrap90_low": float(np.quantile(resampled, .05)), "cluster_bootstrap90_high": float(np.quantile(resampled, .95)),
                                      "seed": str(seed), "scope": "exploratory6-molecule_percentile_bootstrap; not simultaneous, not heldout validation"})
    pd.DataFrame(bootstrap).to_csv(output / "mixture-cluster-diagnostics.csv", index=False)
    lines = ["# PhaseA — 解釈と次の実験への含意", "", "全30条件の再解析。以下は独立の新しいmultinomial drawを使ったsimulationであり、量子実機・新アルゴリズムの優越性の実証ではない。", "",
             "## 1. R=32は容量制約だったか", "", "固定した最終保存点、同じcountsをR32から512へ拡張。正は誤差が下がった量。小Mと大Mを併記し、有利な予算だけを選ばない。", "",
             "| 分子 | Pool | M | 条件数 | 誤差低下中央値 mHa | R32平均dの中央値 | R512平均dの中央値 |", "|---|---|---:|---:|---:|---:|---:|"]
    selected = capacity[capacity.method.isin(["ceo_ovp", "ceo_mvp", "qubit", "lucj_2"]) & capacity.shots.isin([1000, 100000])]
    for (mol, method, M), g in selected.groupby(["molecule", "method", "shots"]):
        lines.append(f"| {mol} | {method} | {M} | {len(g)} | {g.improvement_R32_to_R512_mHa.median():.5f} | {g.mean_dimension_R32.median():.2f} | {g.mean_dimension_R512.median():.2f} |")
    lines += ["", "CI空間を広げた効果であり、回路そのものの改善ではない。HF-onlyのように発見配置が増えなければRを広げても改善しない。H6/N2の未収束trajectoryを収束済みと置換しない。", "",
              "## 2. 頻度ランキングだけで説明できるか", "", "同じcounts・同じ実dで、頻度順と観測済み配置の一様shuffleを比較。負はshuffle側の平均誤差が低い。FCIからの補充はしていない。", "",
              "| 分子 | M | method条件数 | shuffle−frequency中央値 mHa |", "|---|---:|---:|---:|"]
    for (mol, M), g in ranking[ranking.shots.isin([1000, 100000])].groupby(["molecule", "shots"]):
        lines.append(f"| {mol} | {M} | {len(g)} | {g.uniform_minus_frequency_error_mHa.median():.5f} |")
    lines += ["", "## 3. 複数checkpointは有効だったか", "", "同じ総M、pilot費用をそのMに含める。下表はM1000のpolicy−final誤差差を分子ごとに5距離平均し、6分子を等重みで平均した探索的診断。負ならpolicy側が良い。CI次元を揃えた列も分離する。", "",
              "| Pool | Policy | 同R差 mHa | 同実d差 mHa |", "|---|---|---:|---:|"]
    bdf = pd.DataFrame(bootstrap)
    for (method, policy), g in bdf[bdf.shots_total == 1000].groupby(["method", "policy"]):
        values = dict(zip(g.metric, g.equal_weight_molecule_mean_delta_mHa))
        lines.append(f"| {method} | {policy} | {values['mean_error_mHa']:.5f} | {values['mean_matched_d_error_mHa']:.5f} |")
    lines += ["", "この表はbest-singleに将来情報を許したretrospective上限を含む。既存bankの全構築費用を免除してはいない。新しいonline停止法の性能や未知分子一般化には読み替えない。policy間の多重比較・6分子という小規模cohortを考慮し、bootstrap区間は補助CSVの探索的記述に留める。", "",
              "## 4. 昔選んだ早い点は、新しいsamplingでも良いか", "", "旧countsで固定した点を、今回の独立seedで再評価した。下表はM1000・旧0.1mHa許容点。平均差の90% Welch区間を全1800組のCSVに保存。これは固定点の再samplingであり、FCI不要のonline停止policyの独立検証ではない。", "",
              "| Pool | 条件数 | 新しい誤差差（早い点−最終点）中央値 mHa | 新しい平均差≤0.1mHaの条件数 |", "|---|---:|---:|---:|"]
    for method, g in held[(held.shots == 1000) & (held.tolerance_mHa == .1)].groupby("method"):
        lines.append(f"| {method} | {len(g)} | {g.new_delta_selected_minus_final_mHa.median():.5f} | {int(g.mean_gap_within_registered_tolerance.sum())} |")
    lines += ["", "新しい全M/Rによる事後停止表も37,800行を別保存した。旧点の再評価と、新しい標本を見て再選択した点は区別する。", "",
              "## 5. 次へ何を持ち越すか", "", "- 容量R、ランキング、sampler選択を混同せず、強いbaselineにも同じR/実d対照を適用する。", "- VQE energy低下やoraclecoverageだけを実用停止基準にしない。", "- まず最適化済みLUCJ、正確なADAPT-QSCI/HF-separated、TE/SqDRIFTとclassicalSCIを追加する。固定LUCJに勝つことだけで新規性を主張しない。", "- FCI不要の残差/発見/pilot-energy選択は次段階で独立評価する。現在の表を見て設計した新policyを同じseedで確認済みと呼ばない。", "- RAACS、20–40+qubits、fullSQD/noiseはまだ未実施で、研究全体の残scopeである。", ""]
    (root / "INTERPRETATION_JA.md").write_text("\n".join(lines) + "\n")
    result = {"capacity_rows": len(capacity), "ranking_control_rows": len(ranking), "mixture_summary_groups": len(mix), "cluster_diagnostics": len(bootstrap),
              "individual_target_discovery_rows": len(target_rows), "new_M_R_stopping_rows": len(new_stops),
              "historical_choices_independent_readout_rows": len(held),
              "whole_study_complete": False, "hardware_shots": 0}
    save_json(root / "interpretation-status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(interpret(args.output)))


if __name__ == "__main__":
    main()
