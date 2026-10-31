"""Descriptive qualification plots, not a matched-work superiority result."""
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
    validation = json.loads((root/"validation.json").read_text())
    assert validation["status"] == "RAACS_QUICK_COUNTS_PHYSICS_AND_FINAL_SUPPORT_VALIDATED"
    source = Path(cfg["source"]); rows = []
    for p in sorted((root/"cells").glob("*/*/result.json")):
        r = json.loads(p.read_text()); case = p.parent.parent.name
        # FCI is read here, after all construction/final counts and independent
        # audit, and never passed back into the numerical sampler.
        ref = json.loads((source/"references"/case/"reference.json").read_text())
        db = sqlite3.connect(f"file:{p.parent/'events.sqlite'}?mode=ro", uri=True)
        finals = list(db.execute("SELECT raw_shots,metadata FROM draws WHERE role='independent_final'"))
        resources = [(M, json.loads(raw)["sampling_resources"]) for M, raw in finals]
        total_final = sum(M for M, _ in resources)
        row = {"case_id": case, "cell": p.parent.name, "score": r["algorithm"]["score"],
               "angle_mode": r["algorithm"]["angle_mode"], "final_allocation": r["algorithm"]["final_allocation"],
               "backend": r["sampling"]["backend"], "total_model_shots": r["actual_total_model_shots"],
               "pilot_shots": r["pilot_raw_model_shots"], "final_shots": total_final,
               "error_mHa_ORACLE_REPORTING_ONLY": 1000*(r["final_QSCI_energy_Ha"]-ref["fci_energy"]),
               "actual_final_CI_dimension": r["actual_final_CI_dimension"], "accepted_operators": r["accepted_steps"],
               "construction_stop_reason": r["construction_stop_reason"], "named_H_entry_requests": r["classical_named_H_requests"],
               "wall_seconds": r["actual_wall_seconds"], "hardware_shots": 0}
        for metric in ("cnot_count", "cnot_depth", "parameter_count", "total_depth"):
            row["final_per_shot_weighted_"+metric] = sum(M*v[metric] for M, v in resources)/total_final
            row["final_max_"+metric] = max(v[metric] for _, v in resources)
        row["all_actual_gate_shots"] = sum(M*json.loads(raw)["sampling_resources"]["cnot_count"] for M, raw in db.execute("SELECT raw_shots,metadata FROM draws"))
        db.close(); rows.append(row)
    out = root/"derived-report-v1"; out.mkdir(exist_ok=True)
    f = pd.DataFrame(rows); f.to_csv(out/"all-qualification-cells.csv", index=False)
    figures = []
    labels = {"ci_gradient_local": "CI-grad(local)", "residual": "Residual", "residual_cost": "Residual/cost",
              "residual_cost_diversity": "Residual/cost/div", "full_cost": "Full cost"}
    for case, group in f.groupby("case_id"):
        names = [labels[r.score]+(" MPS" if r.backend == "mps" else "")+
                 (" "+r.final_allocation if r.final_allocation != "last_only" else "")+
                 (" fixed" if r.angle_mode == "fixed_angle" else "") for r in group.itertuples()]
        fig, ax = plt.subplots(1, 3, figsize=(13, 4)); y = np.arange(len(group))
        for a, metric, title in zip(ax, ("error_mHa_ORACLE_REPORTING_ONLY", "final_per_shot_weighted_cnot_count", "all_actual_gate_shots"),
                                    ("QSCI error (mHa)", "Final per-shot weighted CNOT", "All trial + final CNOT-shots")):
            a.barh(y, group[metric], color="#4c78a8"); a.set_yticks(y, names if a is ax[0] else [""]*len(y)); a.set_xlabel(title); a.grid(axis="x", alpha=.2)
        fig.suptitle(f"{case}: qualification only / one construction seed + one final readout / no matched baseline claim")
        fig.tight_layout()
        for ext in ("png", "svg", "pdf"):
            path = out/"figures"/f"{case}.{ext}"; path.parent.mkdir(exist_ok=True)
            fig.savefig(path, bbox_inches="tight", dpi=180); figures.append(str(path.relative_to(root)))
        plt.close(fig)
    text = ["# RAACS小規模実行：正しさの検証と初期観測", "",
        f"{len(f)}セル、{validation['all_raw_draws_rebuilt']}raw drawsを再生成。全回路・独立最終supportを監査済み。FCIは終了後の誤差ラベルにのみ使用。", "",
        "| 条件 | 方式 | backend | 誤差 mHa | 最終CNOT/shot | 最終Depth/shot | pilot/final shots |", "|---|---|---|---:|---:|---:|---:|"]
    for r in f.to_dict("records"):
        text.append(f"|{r['case_id']}|{r['cell']}|{r['backend']}|{r['error_mHa_ORACLE_REPORTING_ONLY']:.4f}|{r['final_per_shot_weighted_cnot_count']:.1f}|{r['final_per_shot_weighted_cnot_depth']:.1f}|{r['pilot_shots']}/{r['final_shots']}|")
    text += ["", "## 解釈の境界", "",
        "- 4step・1constructionseed・各セル1finaldrawのqualification。H2は小さく容易な例であり、一般的優位性の証拠ではない。H4に残る誤差を隠さない。",
        "- 同じ総M4000での強力baseline独立比較はこの図に含まれない。既存M1000/10000の値を補間して同予算baselineと呼ばない。",
        "- CI-grad(local)は古典Ritzベクトルに対する局所bank勾配。物理sampler勾配/原著ADAPT-QSCI/CEOの代用品ではない。",
        "- score/angle/allocation/backendがconfigに入り、別のpilot RNGstreamになる。これらはend-to-end policyの比較で、1因子だけのpaired causal ablationではない。multishallow効果だけを見る固定回路bankの対照は別に必要。",
        "- MPSは近似古典simulation。ここでの小系回路parityと40qローカルprimitiveテストだけで、大規模分子や実機での有効性を主張しない。",
        "- Named integral requests・CPU時間・量子sampling modelshotsは別の費用単位。古典VQEの測定換算は未確立で、end-to-end quantum優位性を主張しない。",
        "- blockwise/fullVQE/one-parameter-objective比較、standard multiseed・独立最終replicates、強いbaseline、20–40+q分子、fullSQD/noiseは未完。研究全体は継続中。", ""]
    (out/"REPORT_JA.md").write_text("\n".join(text))
    result = {"status": "RAACS_QUICK_QUALIFICATION_REPORTED", "cells": len(f), "figures": len(figures)//3,
              "figure_files": figures, "FCI_in_sampler": False, "hardware_shots": 0,
              "matched_baseline_superiority_tested": False, "whole_study_complete": False, "reporter_sha256": file_digest(__file__)}
    save_json(out/"manifest.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(report(p.parse_args().output)))
