"""Evidence-led report with full-scope pending work, never a narrowed completion."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from .common import save_json


def make_report(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    queue = json.loads((root / "execution-queue.json").read_text())
    total_cps = draws = readouts = simulated = subspaces = 0
    live = []
    for path in sorted((root / "finite").glob("*/readouts.sqlite")):
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        cp = db.execute("SELECT count(*) FROM analytics").fetchone()[0]
        dr, shots = db.execute("SELECT count(*),COALESCE(sum(shots),0) FROM draws").fetchone()
        ro = db.execute("SELECT count(*) FROM outcomes").fetchone()[0]
        ns = db.execute("SELECT count(*) FROM subspaces").fetchone()[0]
        db.close()
        total_cps += cp
        draws += dr
        readouts += ro
        simulated += shots
        subspaces += ns
        status_file = path.parent / "status.json"
        status = json.loads(status_file.read_text())["status"] if status_file.exists() else "COMMITTED_PREFIX_NO_TERMINAL"
        live.append({"case": path.parent.name, "status_record": status, "checkpoints_with_analytics": cp, "committed_draws": dr, "committed_readouts": ro})
    expected = queue["selected_checkpoints"] * len(manifest["M_grid"]) * manifest["repeats"] * len(manifest["R_grid"]) * 2
    static = root / "retrospective-stops.parquet"
    tracker = root / "operator-tracking-status.json"
    ratios = pd.read_parquet(static) if static.exists() else pd.DataFrame()
    lines = ["# Scalable quantum sampler study — 実行結果と未完了事項", "", "研究全体は継続中。PhaseAの実行成功をRAACSの性能優位や全研究完了とは呼ばない。", "",
             f"Mode: `{manifest['mode']}`。新finite cohort: {queue['case_count']}条件 / {queue['selected_checkpoints']} checkpoint。元データは全30条件/5,236 checkpoint。", "",
             "## 現在の実測進捗", "", f"- 配置解析のあるcheckpoint: {total_cps}", f"- committed独立draws: {draws:,}",
             f"- R/rule readouts: {readouts:,} / {expected:,}", f"- 実際に対角化したunique subspaces: {subspaces:,}",
             f"- 新しい仮想shots: {simulated:,}（R/ruleに再利用、二重計上なし）", "- 実機shots: 0", "",
             "JSON statusのRUNNINGだけで稼働証明とはしない。PID/sessionの確認が別途必要。上の数はSQLiteにcommit済みのデータから再構成。", "",
             "## Source保全", "", "`source-audit.json`で既存40,944 artifactのSHA-256一致を検証。旧manifest・失敗・checkpointは変更していない。", "",
             "## A3: retrospective停止点", ""]
    if len(ratios):
        lines += [f"全30条件×5pool×3M×4許容値の{len(ratios):,}行を保存。以下はM1000、最良energyから0.1mHa以内に最初に到達した点。", "",
                  "| Pool | 条件数 | step比中央値 | CNOT削減中央値 % | Depth削減中央値 % |", "|---|---:|---:|---:|---:|"]
        for method, g in ratios[(ratios.shots == 1000) & (ratios.tolerance_mHa == .1)].groupby("method"):
            lines.append(f"| {method} | {len(g)} | {g.step_ratio.median():.4f} | {g.cnot_reduction_percent.median():.3f} | {g.depth_reduction_percent.median():.3f} |")
        lines += ["", "同じ評価sampleで選んだ事後診断。MAX_STEPSを真の収束とせず、最後の保存点との比率。幾何点は独立分子ではなく、上の中央値に優越性検定の意味はない。", ""]
    if tracker.exists():
        d = json.loads(tracker.read_text())
        lines += ["## A4: 電子配置の追跡", "", f"{d['trajectories']}軌道、{d['transitions']:,}遷移、{d['configuration_lifecycles']:,}配置履歴、{d['important_probability_rows']:,}重要配置確率行。", "",
                  "operatorと全係数の再最適化が同時に起こるため、変化を単一operatorの因果効果とは呼ばない。FCI重要度は診断用oracleのみ。", ""]
    summary = root / "analysis/readout-summary.parquet"
    if summary.exists():
        frame = pd.read_parquet(summary)
        frame = frame[frame.primary_record & frame.complete_group & (frame.selection_rule == "frequency")]
        inventory = pd.DataFrame(json.loads((root / "checkpoint-inventory.json").read_text()))
        source_final = inventory[inventory.primary].groupby(["case_id", "method"]).step.max().reset_index()
        final = frame.merge(source_final, on=["case_id", "method", "step"])
        compare = final[(final.shots == 1000) & final.R_cap.isin([32, 512])].pivot_table(index=["case_id", "method", "step"], columns="R_cap", values="mean_error_mHa")
        if 32 in compare.columns and 512 in compare.columns:
            compare = compare.dropna()
            lines += ["## A1: R32→512感度（最終点・M1000）", "", "| 分子 | 比較可能method条件数 | 誤差低下の中央値 mHa |", "|---|---:|---:|"]
            c = compare.reset_index()
            c["molecule"] = c.case_id.str.split("_R").str[0]
            for molecule, g in c.groupby("molecule"):
                lines.append(f"| {molecule} | {len(g)} | {(g[32]-g[512]).median():.5f} |")
            lines += ["", "同じcountsの入れ子拡張。CI上限が制約だったかの診断であり、sampler自体の発見能力や量子優位の証明ではない。未完了cohortの途中値なら全体の結論にしない。", ""]
    closure_path = root / "milestone1-closure.json"
    closure = json.loads(closure_path.read_text()) if closure_path.exists() else None
    if closure:
        lines += ["## Milestone1の監査済み終端", "", f"`{closure['status']}`。A1–A5の全cohort計算、配置追跡、独立再sampling、1,800旧選択点の再評価、37,800新M/R停止点、334図×3形式を保存。", "",
                  "`INTERPRETATION_JA.md`に容量・ランキング・混合・独立seedでの旧選択点評価の数値と解釈を記載。`SECTOR_DIAGNOSTIC_JA.md`で粒子数とSzを分離した検査を追加。H4_R2.2/qubitの低い要求sector確率は粒子数ではなくSzに由来する。全研究の終端ではなく、Milestone2への入力である。", ""]
    lines += ["## 科学的な限界", "", "- 保存確率を使った古典再samplingで、実機結果ではない。仮想shots数はmultinomial countsが表す数であり、回路をその回数分simulationしたという意味でもない。", "- R差と頻度選択規則差は同じcountsから検査。FCI補充なし。", "- jointM90はcategorical Monte Carloの近似quantileで、個別発見確率の積ではない。", "- p-hat90%到達は確率保証ではない。未達はtested-grid打切りとして保存。", "- LUCJ sourceは固定CCSD。最適化済みLUCJとの比較は未実施。", "- end-to-end量子Measurement Cost、RAACSの性能、未知分子一般化、大規模化、fullSQD/gate-noiseは未実証。", "",
              "## 完了に必要な残りの全scope", "", "1. " + ("Milestone1はclosure監査済み。以後の手法比較へ引き継ぐ。" if closure else "A1/A2/A5の全cohortと図・解釈の最終closure監査。"), "2. B: 最適化LUCJ、MP2/multiseed/QSCI-guided、正確なADAPT-QSCI/HF-separated、TE/SqDRIFT、orbital/random、古典SCI。", "3. C: FCI不要停止、独立seed/held-out/分子cluster不確かさ、pilotを含む費用。", "4. D: residual/local/cost-awareRAACS、全ablation、軽量angle更新とmulti-shallow。", "5. E:20–40+qubitsを含むFCI不要・非full-statevector routeの実行とscaling。LevelCは実行基盤がある場合。", "6. F: fullfermionicSQD、gate-noise、dephasing/routing。実機は利用可能時だけ。", "7. 全10primarymetrics、18figuretypes、15研究質問のevidence/limitations監査。", ""]
    (root / "REPORT_JA.md").write_text("\n".join(lines) + "\n")
    audit = {"scope": "entire original objective preserved", "whole_study_complete": False,
             "phase_A_committed_readouts": readouts, "phase_A_expected_readouts": expected,
             "cases": live, "hardware_shots": 0,
             "milestones": {"M1": closure["status"] if closure else "IN_PROGRESS" if readouts != expected else "FINITE_COUNTS_COMPLETE_ADDITIONAL_AUDIT_REQUIRED",
                            "M2": "NOT_IMPLEMENTED", "M3": "NOT_IMPLEMENTED", "M4": "NOT_IMPLEMENTED", "M5": "NOT_IMPLEMENTED", "M6": "NOT_IMPLEMENTED"}}
    save_json(root / "requirements-progress.json", audit)
    return audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print(json.dumps(make_report(args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
