"""Outer Phase C reporting. FCI reference labels enter only this module."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .common import file_digest, save_json


def report(root):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text())
    valid = json.loads((root/"validation.json").read_text())
    lomo = json.loads((root/"LOMO-v1/status.json").read_text())
    assert valid["status"] == "INDEPENDENT_REPLAY_VALIDATED" and lomo["status"] == "TRAINING_ONLY_LOMO_EXECUTED"
    out = root/"derived-report-v1"; out.mkdir(exist_ok=True)
    f = pd.read_parquet(root/"all-policy-independent-readouts.parquet")
    held = pd.read_parquet(root/"LOMO-v1/heldout-readouts.parquet")
    references = {c: json.loads((Path(cfg["source"])/"references"/c/"reference.json").read_text()) for c in cfg["case_ids"]}
    for frame in (f, held):
        frame["error_mHa"] = [(E-references[c]["fci_energy"])*1000 for c, E in zip(frame.case_id, frame.energy_Ha)]
        frame["native_error_mHa"] = [(E-references[c]["fci_energy"])*1000 for c, E in zip(frame.case_id, frame.native_energy_Ha)]
        frame["matched_regret_mHa"] = 1000*(frame.matched_energy_Ha-frame.matched_native_energy_Ha)
        frame["regret_mHa"] = 1000*(frame.energy_Ha-frame.native_energy_Ha)
        frame["CNOT_reduction_percent"] = 100*(1-frame.cnot_count/frame.native_cnot_count)
        frame["depth_reduction_percent"] = 100*(1-frame.cnot_depth/frame.native_cnot_depth)
        frame["chemical_success"] = frame.error_mHa <= 1.5936014376
        frame["saved_step_ratio"] = frame.selected_step/frame.native_last_saved_step
    f.to_parquet(out/"all-policy-report-readouts.parquet", index=False)
    held.to_parquet(out/"LOMO-heldout-report-readouts.parquet", index=False)
    metrics = ["error_mHa", "native_error_mHa", "regret_mHa", "matched_regret_mHa", "chemical_success",
               "CNOT_reduction_percent", "depth_reduction_percent", "pilot_shots", "remaining_final_shots",
               "cnot_count", "cnot_depth", "actual_d", "matched_d", "saved_step_ratio", "total_deployment_gate_shots"]
    full = f.groupby(["case_id", "molecule", "pool", "policy", "total_M", "cap"], as_index=False)[metrics].mean()
    full.to_csv(out/"all-policy-case-means.csv", index=False)
    summary = held.groupby(["case_id", "molecule", "pool", "fold_selected_policy", "total_M", "cap"], as_index=False)[metrics].mean()
    summary.to_csv(out/"LOMO-heldout-case-means.csv", index=False)
    clusters = pd.read_csv(root/"LOMO-v1/cluster-summary.csv")
    figures = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "svg", "pdf"):
            path = out/"figures"/f"{name}.{ext}"; path.parent.mkdir(exist_ok=True)
            fig.savefig(path, dpi=180, bbox_inches="tight"); figures.append(str(path.relative_to(root)))
        plt.close(fig)

    pools = list(cfg["methods"])
    for M in cfg["total_M"]:
        g = clusters[(clusters.total_M == M)&(clusters.cap == 64)].set_index("pool").reindex(pools)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
        for a, metric, label in zip(ax, ("regret_mHa", "CNOT_reduction_percent", "depth_reduction_percent"),
                                   ("QSCI regret vs native (mHa)", "CNOT reduction (%)", "CNOT depth reduction (%)")):
            a.errorbar(range(len(pools)), g[metric], yerr=g[metric+"_t90_halfwidth"].fillna(0), fmt="o", capsize=3)
            a.axhline(0, color="grey", lw=.7); a.set_xticks(range(len(pools)), pools, rotation=35, ha="right")
            a.set_ylabel(label); a.grid(alpha=.2)
        fig.suptitle(f"Training-only LOMO / M total={M}, R64 / molecular-cluster descriptive t90 / replay only")
        save(fig, f"LOMO-cluster-M{M}")
    for case in cfg["case_ids"]:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        q = summary[(summary.case_id == case)&(summary.cap == 64)]
        for pool, g in q.groupby("pool"):
            g = g.sort_values("total_M")
            ax[0].plot(g.total_M, g.error_mHa, "o-", label=pool)
            ax[1].plot(g.total_M, g.CNOT_reduction_percent, "o-")
            ax[2].plot(g.total_M, g.pilot_shots, "o-")
        for a, label in zip(ax, ("QSCI error (mHa)", "CNOT reduction vs native (%)", "Paid pilot model shots")):
            a.set_xscale("log"); a.set_xlabel("Total model shots (pilot + final)"); a.set_ylabel(label); a.grid(alpha=.2)
        ax[0].set_yscale("symlog", linthresh=.1); ax[0].axhline(1.5936014376, color="grey", ls="--", lw=.7)
        ax[0].legend(fontsize=7); fig.suptitle(f"{case}: heldout policy / R64 / independent final-only support; not actual stopped growth")
        save(fig, f"case-{case}")
    f64 = full[full.cap == 64]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for policy, g in f64.groupby("policy"):
        # A geometry cluster is a point, not a new independent molecule.
        ax[0].scatter(g.CNOT_reduction_percent, g.regret_mHa, s=9, alpha=.4, label=policy)
        ax[1].scatter(g.depth_reduction_percent, g.matched_regret_mHa, s=9, alpha=.4)
    ax[0].set_xlabel("CNOT reduction (%)"); ax[0].set_ylabel("Common-R regret (mHa)")
    ax[1].set_xlabel("CNOT depth reduction (%)"); ax[1].set_ylabel("Paired actual-d regret (mHa)")
    for a in ax: a.axhline(0, color="grey", lw=.7); a.set_yscale("symlog", linthresh=.1); a.grid(alpha=.2)
    ax[0].legend(fontsize=7); fig.suptitle("All registered rules / R64 / case means / fixed thresholds; no per-case winner selection")
    save(fig, "all-rules-resources-regret")
    switch_rows = []
    for path in sorted((root/"LOMO-v1/folds").glob("*.json")):
        a = json.loads(path.read_text())
        switch_rows.append({k: a[k] for k in ("heldout_molecule", "pool", "total_M", "selected_policy", "training_input_digest")})
    pd.DataFrame(switch_rows).to_csv(out/"LOMO-policy-selections.csv", index=False)
    costs = []
    for case in cfg["case_ids"]:
        db = sqlite3.connect(f"file:{root/'readouts'/case/'readouts.sqlite'}?mode=ro", uri=True)
        draws, shots, sample_seconds = db.execute("SELECT count(*),sum(shots),sum(seconds) FROM draws").fetchone()
        solves, solve_seconds, stored = db.execute("SELECT count(*),sum(seconds),sum(projected_bytes) FROM subspaces").fetchone()
        costs.append(dict(case_id=case, unique_final_draws=draws, unique_final_raw_model_shots=shots,
                          final_sampling_simulation_seconds=sample_seconds, unique_CI_solves=solves,
                          unique_CI_solve_seconds=solve_seconds, cumulative_projected_bytes_not_peak=stored))
        db.close()
    pd.DataFrame(costs).to_csv(out/"unique-acquisition-and-CI-costs.csv", index=False)
    text = ["# Phase C：独立評価・分子単位LOMO（保存軌道の再生）", "",
        "固定した6停止規則とnative最終点を比較。pilotと最終評価は独立seed、総Mからpilot分を控除。FCIは本レポートの事後誤差表示にのみ使用し、停止・calibrationには入力しない。", "",
        f"- {len(cfg['case_ids'])}条件、{lomo['molecular_clusters']}分子cluster、{lomo['folds']} folds。",
        f"- 全判断 {valid['all_policy_decisions_reconciled']:,}、独立最終draw {valid['raw_draws_rebuilt']:,}、unique final model shots {valid['unique_final_raw_model_shots']:,}。",
        f"- pilot全bankの取得費用は別途 {json.loads((Path(cfg['pilot_bank'])/'validation.json').read_text())['raw_model_pilot_shots']:,} model shots。foldやRごとに再加算しない。",
        "- 実機shots 0。全count再生成・全共通R/実d一致の選択ID検査済み。独立Ritz再対角化は代表最大5空間/条件のみで、全solveの再対角化ではない。", "",
        "## 主要結果（R64、分子平均を等重み）", "",
        "| Pool | 総M | QSCI regret mHa | CNOT削減 % | Depth削減 % | pilot shots |", "|---|---:|---:|---:|---:|---:|"]
    for r in clusters[clusters.cap == 64].to_dict("records"):
        text.append(f"|{r['pool']}|{r['total_M']}|{r['regret_mHa']:.4f}|{r['CNOT_reduction_percent']:.2f}|{r['depth_reduction_percent']:.2f}|{r['pilot_shots']:.1f}|")
    text += ["", "regret = 停止policyの独立最終QSCI energy − native最終QSCI energy。負なら改善。正のCNOT/Depth削減率は資源減少。これは平均値であり、全条件・全seedの非劣性保証ではない。",
        "", "## 解釈と限界", "",
        "- LOMO選択は他分子だけの絶対QSCI energiesと資源・shotsに限定。閾値は平均regret≤0.5mHaかつ最悪training分子平均≤1.6mHa。基底状態精度や投稿水準の保証ではない。",
        "- 6つの既知分子identityを分離した検証。未使用結合距離・活性空間のprospective validationは未実施。fold間でtrainingを共有するためcluster t90は記述的な不確かさ。",
        "- gradient規則は保存された検証済み終端勾配しか利用できず、native endpointと同じ回路・0pilot。途中の未保存勾配を0で補わない。",
        "- nativeは不要なQSCI pilotを払わず全Mを最終回路に使う。停止規則は消費したprefix pilotを差し引く。primary最終評価はpilot配置を混ぜない。",
        "- nativeのMAX_STEPSは収束ではない。QSCI停止も絶対精度の認証ではない。両手法がFCIから遠い条件は誤差表に残す。",
        "- 全軌道の構築費用は既に発生。短いprefix秒数は反実仮想の診断値であり、実際の構築費削減とは呼ばない。実際にFCI-freeで成長を止める実験が次に必要。",
        "- CI、pilot、最終samplingの古典時間は別記。古典VQE評価回数から量子Measurement Costを作らない。",
        "- この結果はPhase Cの再生部分であり、D新サンプラー、E20–40+qubit分子、F full SQD/noise、全研究の完了を意味しない。", ""]
    (out/"REPORT_JA.md").write_text("\n".join(text))
    result = {"status": "REPLAY_AND_TRAINING_ONLY_LOMO_REPORTED", "figures": len(figures)//3,
              "figure_files": figures, "all_policy_case_rows": len(full), "heldout_case_rows": len(summary),
              "hardware_shots": 0, "FCI_only_in_outer_reporting": True,
              "actual_stopped_growth_required_next": True, "whole_study_complete": False,
              "reporter_sha256": file_digest(__file__)}
    save_json(out/"manifest.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(report(p.parse_args().output)))
