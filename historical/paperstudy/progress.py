"""Evidence-based progress and all-primary coverage; never claims completion."""
import csv
from io import StringIO
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

from .io import ROOT, atomic_bytes, save_json, sha256


def load_optional(path):
    return json.loads(path.read_text()) if path.exists() else None


def final_completion_record():
    """Recognize an explicit, evidence-bound S6 audit; counts alone never close it.

    This is a report-status reader, not an experiment authorization gate or a
    replacement for the final requirement-by-requirement scientific audit.
    """
    final=load_optional(ROOT/'FINAL_COMPLETION.json')
    if not final or final.get('whole_experiment_complete')is not True:return None
    sources=final.get('evidence_SHA256',{})
    required={'FULL_PLAN_AUDIT_JA.md','FINAL_REPORT_JA.md','test-results.json',
        'analysis-v1/requirement-coverage.json','analysis-v1/actual-molecular-data-integrity.json',
        'analysis-v1/all-planned-cells.csv','analysis-v1/all-observed-readouts.csv'}
    if not required.issubset(sources):return None
    for relative,expected in sources.items():
        path=ROOT/relative
        if not path.is_file()or ROOT.resolve()not in path.resolve().parents:return None
        if sha256(path.read_bytes())!=expected:return None
    return final


def report(run_tests=False):
    inputs = load_optional(ROOT/"data/input-preparation.json")
    plan = load_optional(ROOT.parent/"plan.json")
    current_inputs = {}
    for case in plan["primary_cases"]:
        value = load_optional(ROOT/"data/physical"/case["id"]/"problem.json")
        current_inputs[case["id"]] = None if value is None else {
                "problem_id": value["problem_id"], "source": value["source"], "qualification": value["qualification"]}
    pilot_results = []
    for method in plan["primary_methods"]:
        for seed in plan["engineering_pilot"]["seeds"]:
            path = ROOT/"common-runs/S3_ENGINEERING_PILOT/N2-32-R1p2-P2"/method["id"]/("10000-"+str(seed))/"result.json"
            value = load_optional(path)
            if value is not None:
                pilot_results.append({"method": method["id"], "seed": seed, "status": value["status"],
                                      "shots": value["budget_ledger"]["known_actual_shots"]})
    db = sqlite3.connect(ROOT/"run-queue.sqlite")
    rows = db.execute("SELECT case_id,method_id,budget,seed,status,reason,result_path FROM planned_cells ORDER BY budget,case_id,method_id,seed").fetchall()
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["case_id", "method_id", "total_model_shot_cap", "seed", "status", "reason", "result_path"])
    writer.writerows(rows)
    atomic_bytes(ROOT/"all-planned-primary.csv", output.getvalue().encode())
    primary_counts = dict(db.execute("SELECT status,COUNT(*) FROM planned_cells GROUP BY status").fetchall())
    tier_counts={B:dict(db.execute('SELECT status,COUNT(*) FROM planned_cells WHERE budget=? GROUP BY status',(B,)))for B in(10000,100000,1000000)}
    sensitivity_counts = dict(db.execute("SELECT status,COUNT(*) FROM sensitivity_cells GROUP BY status").fetchall()) if db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='sensitivity_cells'").fetchone()[0] else {}
    basis_counts = dict(db.execute("SELECT status,COUNT(*) FROM basis_cells GROUP BY status").fetchall()) if db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='basis_cells'").fetchone()[0] else {}
    db.close()
    tests = load_optional(ROOT/"test-results.json")
    if run_tests:
        project = ROOT.parents[2]/"ceo-sampler-benchmark"
        command = ["rtk", "proxy", sys.executable, "-m", "pytest", str(ROOT/"tests"), "-q"]
        result = subprocess.run(command, cwd=project, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        atomic_bytes(ROOT/"test-output.txt", result.stdout.encode())
        passed = re.search(r"(\d+) passed", result.stdout)
        tests = {"command": command, "returncode": result.returncode,
                 "passed": int(passed.group(1)) if passed else None,
                 "scope": "new input/method/compiler/CI/atomic-budget runtime tests; NOT full scientific study",
                 "code_hashes": {str(p.relative_to(ROOT)): sha256(p.read_bytes()) for p in sorted((ROOT/"paperstudy").glob("*.py"))}}
        save_json(ROOT/"test-results.json", tests)
    gpu = []
    for seed in (101, 211):
        evidence = load_optional(ROOT/"qualification/GPU"/str(seed)/"observed-status.json")
        if evidence:
            try:
                value = json.loads(evidence["result_or_stderr"])
            except json.JSONDecodeError:
                value = {"status": "NO_STRUCTURED_RESULT"}
            gpu.append({"job_id": evidence["job_id"], "result": value, "accounting": evidence["accounting"]})
    primary_completed = sum(count for status, count in primary_counts.items() if status == "COMPLETED")
    coverage=load_optional(ROOT/'analysis-v1/requirement-coverage.json')or{}
    tn_final=sum(1 for p in ROOT.glob('common-runs/S_TN_INDEPENDENT_FINAL/*/*/*/result.json')
        if json.loads(p.read_text()).get('status')=='COMPLETED')
    evidence = {
        "status": "ACTIVE_FROZEN_PRIMARY_AND_SENSITIVITY_EXECUTION",
        "primary_planned": len(rows), "primary_completed": primary_completed,
        "primary_status_counts": primary_counts, "physical_input_original_snapshot": inputs,
        "primary_by_model_shot_budget":tier_counts,
        "sensitivity_status_counts": sensitivity_counts,
        "orbital_basis_status_counts": basis_counts,
        "TN_engineering_successor_status": load_optional(ROOT/"TN-successor-v2/status.json"),
        "current_primary_inputs": current_inputs, "engineering_pilot_completed": pilot_results,
        "tests": tests, "GPU_small_qualifications": gpu,
        "GPU_full32q_qualifications": {m:load_optional(ROOT/"qualification/GPU-Givens-v1"/m/"observed-status.json") for m in ("B_LUCJ_NAIVE_L1","C_LUCJ_COMPRESSED_L1")},
        "analysis_and_figures": "analysis-v1/README_JA.md",
        "public_hardware_source_weights": load_optional(ROOT/"hardware-reanalysis-v1/summary.json"),
        "classical_controls": load_optional(ROOT/"classical-controls-v1/summary.json"),
        "GPU_concurrency_accounting": load_optional(ROOT/"analysis-v1/GPU-concurrency-evidence.json"),
        "GPU_simultaneous_execution_proven": bool((load_optional(ROOT/"analysis-v1/GPU-concurrency-evidence.json")or{}).get("two_GPU_simultaneous_allocation_observed",False)),
        "GPU_kernel_utilization_or_speedup_proved": False,
        "TN_independent_final_status": load_optional(ROOT/"TN-controls-v1/status.json"),
        "TN_original_plan_high_budget_completion": load_optional(ROOT/'TN-high-final-v1/status.json'),
        "P1_settings_bridge": load_optional(ROOT/"paper-anchors/P1-bridge-v1/terminal.json"),
        "P3_source_physical_identity": load_optional(ROOT/"hardware-physical-v1/summary.json"),
        "P3_current_runtime_reanalysis": load_optional(ROOT/"hardware-postprocessing-v2/status.json"),
        "CI_capacity_sensitivity": load_optional(ROOT/"CI-extension-v1/status.json"),
        "Rustiq_source_equivalent_resource_control": load_optional(ROOT/"rustiq-control-v1/summary.json"),
        "plain_decision_report": "analysis-v1/DECISION_REPORT_JA.md",
        "all_scope_coverage": load_optional(ROOT/"analysis-v1/requirement-coverage.json"),
        "observed_data_integrity": load_optional(ROOT/"analysis-v1/actual-molecular-data-integrity.json"),
        "control_capacity_sensitivity": load_optional(ROOT/"control-CI-extension-v1/status.json"),
        "P2_P3_source_report": "analysis-v1/SOURCE_REANALYSIS_JA.md",
        "TN_control_cost_and_quality": "analysis-v1/TN_CONTROL_JA.md",
        "N2_naive_exact_timing": load_optional(ROOT/"qualification/N2-32-R1p2-P2-naive/result.json"),
        "N2_native_resource_views": load_optional(ROOT/"qualification/N2-native-resource-views.json"),
        "RAACS_fixed_rule_fixture": load_optional(ROOT/"qualification/N2-32-R1p2-P2-RAACS-exact-sparse-v2/wrapper-result.json"),
        "P2_naive_current_runtime_anchor": load_optional(ROOT/"paper-anchors/P2/N2-32-R1p2-P2-naive/result.json"),
        "P2_compressed_postprocessing": load_optional(ROOT/"paper-anchors/P2/N2-32-R1p2-P2-compressed/postprocessing-v2/result.json"),
        "measured_initialization": {case: load_optional(ROOT/"data/initialization-fresh"/case/"result.json")
                for case in ("N2-32-R1p2-P2", "N2-32-R2p4-P2", "FE2S2-40-P2")},
        "initialization_driver": load_optional(ROOT/"initialization-driver.json"),
        "remaining":coverage.get('remaining',['Current full-scope coverage report required'])+['S6 final full-scope synthesis/audit'],
        "paper_fidelity_limitations_are_separate_from_pending_execution":True,
        "science_claim": "No method superiority claim from engineering fixtures or P-lane-only points",
        "whole_experiment_complete": False,
    }
    final=final_completion_record()
    closed=bool(final and coverage.get('all_registered_requirements_closed')is True
        and primary_counts=={'COMPLETED':480}and sensitivity_counts=={'COMPLETED':125}
        and basis_counts=={'COMPLETED':40}and tn_final==20
        and tests and tests.get('returncode')==0)
    if closed:
        evidence.update(status='COMPLETED_REGISTERED_PLAN_WITH_DOCUMENTED_REPRODUCTION_LIMITATIONS',
            whole_experiment_complete=True,remaining=[],final_completion_record='FINAL_COMPLETION.json',
            final_report='FINAL_REPORT_JA.md',full_plan_audit='FULL_PLAN_AUDIT_JA.md')
    save_json(ROOT/"PROGRESS.json", evidence)
    text = """# 実行状況：論文接続・公平比較 v1.0

実行は開始済み。研究全体は未完了。主比較は480予定cellのうち{completed}件完了。

| 工程 | 実証できたこと | 残り |
|---|---|---|
| S0 入力 | N₂2距離・Fe₂S₂・出典座標/AVAS再構成ナフタレンが準備済み | 旧runtime/著者積分の完全一致とは別 |
| S1 数値 | LUCJ/QE・SqDRIFT小系独立検算、共通CI、共有予算・resume | 残りの対照・容量感度 |
| GPU | 両LUCJの32q全状態がCPUと約1e-15以下で一致。S5の実状態も独立比較 | 同時割当の証拠と、kernel利用率・高速化の証明は別 |
| 初期化・TN | 新規CCSD/CISD・compressed経路終了。TNの独立finalは{tn_final}/20 | 時間切れや原著TNとの違いはTN_CONTROL_JA.mdに分離 |
| S2 論文anchor | P1設定bridge、P2全6設定、公開P3/Hを別レーンへ保存 | 元runtime/原論文数値の完全再現ではない。上限到達の欠測あり |
| S3 pilot | 共通10,000 Shotの4手法×3 seed、12件完了 | 主推論へは混ぜない |
| S4–S6 | 10k:{low}/160、100k:{medium}/160、1M:{high}/160。層数/長さ感度{sensitivity}/125、軌道感度{basis}/40 | 現在の未完了項目はREQUIREMENT_COVERAGE_JA.md、最後に全費用・図表を統合 |

## 混同してはいけないこと

- 通常LUCJのSDK標準簡約を使い、対照を不必要に大きくしない。
- Fe₂S₂のsource MO変換を全methodで共有。coreをHFに合わせて変更しない。
- 保存振幅を読むだけの時間をCCSD費用にしない。新規作成費用を別途測った。
- 原著設定を現行SDK/PySCFで実行したことと、原論文数値を再現できたことは違う。
- 改善LUCJの大CIは保存countsを再利用してmatrix-freeで完了。追加Shotは0。
- P2の10batchが同じ空間になった結果は、10個の独立whole-run seedに置き換えられない。
- S1の1000shot fixture・失敗時32shotも記録。主比較の成功例に混ぜない。
- 旧GPU fixtureの非同時稼働と、新S5の実Slurm区間を分ける。2台の同時割当はGPU-concurrency-evidence.jsonで確認し、高速化とは同一視しない。
- 旧研究の結果・環境は変更していない。
- P3公開archiveは浮動小数の分布重みであり、生の整数countsと偽って丸めていない。P2とFeの積分も未一致。
- 完了済みの新規rawファイルはSHAを照合した可逆アーカイブへ移行する。元の全バイトはraw-archives-v1/から読める。

詳細はPROGRESS.json、全予定表はall-planned-primary.csv、判断はDECISIONS_JA.md。
稼働状態はこの文書だけで判断せず、OS PID・tool session・Slurmで確認する。
""".format(completed=primary_completed,low=tier_counts[10000].get('COMPLETED',0),
    medium=tier_counts[100000].get('COMPLETED',0),high=tier_counts[1000000].get('COMPLETED',0),
    sensitivity=sensitivity_counts.get('COMPLETED',0),basis=basis_counts.get('COMPLETED',0),tn_final=tn_final)
    if closed:
        text='\n'.join(['# 実行状況：登録した全計画の処理・報告を完了','',
            '主比較480件、層数/長さ感度125件、局在軌道40件、TN中予算20件、TN高予算補完20件を含む最終監査を完了しました。','',
            '完了は原論文曲線の完全再現やRAACSの優越性の認定ではありません。原著との差分、上限停止、参照精度・sector・実機・費用の限界は最終報告に残しています。','',
            '- [最終結果と研究判断](FINAL_REPORT_JA.md)',
            '- [元計画の全要件との照合](FULL_PLAN_AUDIT_JA.md)',
            '- [最終監査時点の証拠結合](FINAL_COMPLETION.json)',
            '- [全表・図への入口](analysis-v1/README_JA.md)',''])
    atomic_bytes(ROOT/"PROGRESS_JA.md", text.encode())
    print(json.dumps({"primary_status_counts": primary_counts, "tests": tests and tests["passed"],
                      "physical_inputs_prepared": sum(x is not None for x in current_inputs.values()),
                      "engineering_pilot_completed": len(pilot_results),
                      "whole_experiment_complete": closed}, indent=2))


if __name__ == "__main__":
    report(run_tests="--test" in sys.argv)
