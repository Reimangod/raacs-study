"""Read-only scientific aggregation, all planned cells and explicit missingness.

Figures use observed grid points, not extrapolated or interpolated optima.
This generator does not alter selection, queues, raw draws, or references.
"""
from collections import Counter, defaultdict
import csv
from io import StringIO
import json
from pathlib import Path
import sqlite3

import numpy as np

from .io import ROOT, atomic_bytes, save_json


def reference(case):
    path = ROOT/"data/evaluation-only"/(case+".json")
    if not path.exists():
        return None, "REFERENCE_UNAVAILABLE"
    data = json.loads(path.read_text())
    values = data["energies_Ha"]
    for key in ("fci_energy", "sci_energy"):
        value = values.get(key)
        if value is not None and np.isfinite(value):
            return float(value), ("FCI" if key == "fci_energy" else "APPROXIMATE_SCI")+(
                    "_INDEPENDENTLY_CHECKED" if data.get("sector_and_reference_precision_verified") else "_AUTHOR_REPORTED_NOT_INDEPENDENTLY_CHECKED")
    return None, "REFERENCE_UNAVAILABLE"


def report():
    destination = ROOT/"report"
    destination.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(ROOT/"run-queue.sqlite")
    planned = db.execute("SELECT case_id,method_id,budget,seed,status,reason,result_path FROM planned_cells ORDER BY case_id,budget,seed,method_id").fetchall()
    db.close()
    rows, coverage = [], []
    for case, method, budget, seed, status, reason, path in planned:
        coverage.append({"case_id": case, "method_id": method, "total_shot_cap": budget, "seed": seed,
                         "status": status, "reason": reason, "result_path": path})
        if status != "COMPLETED" or not path:
            continue
        result = json.loads((ROOT/path).read_text())
        if result["status"] != "COMPLETED" or result["budget_ledger"]["known_actual_shots"] != budget:
            raise RuntimeError("Queue/result/cost mismatch; do not plot a phantom completed cell")
        meta = json.loads((ROOT/"data/physical"/case/"problem.json").read_text())
        Eref, quality = reference(case)
        resources = result.get("sampling_resources", result.get("member_shot_weighted_mean_resources"))
        for item in result["final_CI"]:
            energy = item["energy_Ha"]
            denominator = None if Eref is None else meta["hf_energy_Ha"]-Eref
            recovery = None if denominator is None or denominator <= 0 else 100*(meta["hf_energy_Ha"]-energy)/denominator
            row = {"case_id": case, "method_id": method, "budget": budget, "seed": seed,
                   "r_cap": item["r_cap"], "D_cap": item["r_cap"]**2, "actual_d": item["actual_d"],
                   "energy_Ha": energy, "energy_error_mHa": None if Eref is None else 1000*(energy-Eref),
                   "correlation_recovery_pct": recovery, "reference_quality": quality,
                   "CNOT": resources["cnot_count"], "CNOT_depth": resources["cnot_depth"],
                   "total_depth": resources.get("total_unitary_depth", resources.get("total_depth")),
                   "resource_is_ensemble_shot_weighted_mean": "member_shot_weighted_mean_resources" in result,
                   "coefficient_slots": result.get("coefficient_slots"),
                   "coefficient_slots_unknown_reason": None if "coefficient_slots" in result else "Ensemble: per-member values in raw result, not a single circuit dimension",
                   "independent_parameters": result.get("nonredundant_independent_parameters"),
                   "VQE_optimized_variables": result["VQE_optimized_variables"],
                   "pilot_model_shots": result["pilot_model_shots"], "final_model_shots": result["independent_final_model_shots"],
                   "actual_QPU_shots": result["actual_QPU_shots"],
                   "shot_weighted_CNOT_applications": result["total_model_cnot_applications"],
                   "CI_matvec_calls": item["matvec_calls"], "CI_wall_seconds": item["classical_wall_seconds"],
                   "CI_residual": item["residual_norm"], "CI_converged": item["converged"], "spin_square": item["spin_square"],
                   "whole_run_wall_seconds": result["this_attempt_wall_seconds"],
                   "whole_run_CPU_seconds": result["this_attempt_CPU_seconds"],
                   "FCI_used_for_construction": False, "source_result": path}
            rows.append(row)
    for name, data in (("all-planned-status.csv", coverage), ("all-observed-readouts.csv", rows)):
        if not data:
            continue
        stream = StringIO()
        writer = csv.DictWriter(stream, list(data[0]))
        writer.writeheader()
        writer.writerows(data)
        atomic_bytes(destination/name, stream.getvalue().encode())
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case_id"], row["method_id"], row["budget"], row["r_cap"]].append(row)
    summary = []
    for (case, method, budget, r), values in sorted(grouped.items()):
        energies = [x["energy_Ha"] for x in values]
        summary.append({"case_id": case, "method_id": method, "budget": budget, "r_cap": r,
                        "observed_runs": len(values), "planned_runs": 10,
                        "mean_energy_Ha": float(np.mean(energies)), "median_energy_Ha": float(np.median(energies)),
                        "std_energy_Ha": float(np.std(energies, ddof=1)) if len(values) > 1 else None,
                        "mean_CNOT": float(np.mean([x["CNOT"] for x in values])),
                        "mean_CNOT_depth": float(np.mean([x["CNOT_depth"] for x in values])),
                        "mean_total_depth": float(np.mean([x["total_depth"] for x in values]))})
    counts = Counter(x[4] for x in planned)
    evidence = {"primary_planned": len(planned), "primary_status_counts": dict(counts),
                "observed_readouts": len(rows), "groups": summary, "whole_experiment_complete": False,
                "scope": "Interim observed primary cells; sensitivities and P/H lanes are separate and incomplete"}
    save_json(destination/"summary.json", evidence)
    lines = ["# 公平比較の途中集計", "", "研究全体は未完了。以下は主比較の実測済み部分だけを示す。", "",
             "| 状態 | 予定480件中 |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    lines += ["", "## 読み方", "", "- 同じ分子・総Shot・r上限の組合せで比較する。r違いは同じcountsの再利用であり独立試行ではない。",
              "- LUCJは初期化改善を区別。SqDRIFTはnative SDKによる共通予算版で、元著者driverの完全再現とは呼ばない。",
              "- 平均値の途中変化を見てseed・score・回路長を変更しない。最良seedだけを表示しない。",
              "- CI energyとVQE期待値は異なる。Recoveryは参照の品質を添えた相関エネルギー回収率。configuration recoveryとは別。",
              "- n=10に達していない行は部分結果。主結果・投稿可否の判断に使わない。",
              "- 固定LUCJの10seedは10種類の構造ではなく、同じ固定回路の独立測定。",
              "- coefficient_slotsは非冗長な物理自由度の証明ではない。不明な独立parameter数を0にしない。",
              "- 初期化費用はconstruction/source artifactへ別記。whole_run時間のみでend-to-end費用としない。",
              "", "## r=32の観測値", "", "| 条件 | 手法 | Shot上限 | 観測/予定 | 平均CI energy/Ha | 平均CNOT | 平均CNOT Depth | 平均Total Depth |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for item in summary:
        if item["r_cap"] == 32:
            lines.append(f'| {item["case_id"]} | {item["method_id"]} | {item["budget"]} | {item["observed_runs"]}/10 | {item["mean_energy_Ha"]:.8f} | {item["mean_CNOT"]:.1f} | {item["mean_CNOT_depth"]:.1f} | {item["mean_total_depth"]:.1f} |')
    lines += ["", "## 現段階で不足するもの", "", "全480件、強いLUCJ/TN・層数等の感度、CI4096の条件付き追加、全論文anchor・公開実機counts、最終図と費用の統合は未完了。",
              "この表ができたことを、研究全体の完了やRAACSの一般的優越性に置き換えない。"]
    atomic_bytes(destination/"REPORT_JA.md", ("\n".join(lines)+"\n").encode())
    print(json.dumps({"primary_status_counts": dict(counts), "observed_readouts": len(rows), "whole_experiment_complete": False}))
    return evidence


if __name__ == "__main__":
    report()
