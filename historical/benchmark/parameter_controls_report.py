"""Outer, post-validation analysis; only this layer reads existing FCI labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import ROOT, file_digest, save_json, rng_for
from .local_sampler_primitives import unpack_counts64
from .parameter_controls_audit import SUCCESS

ORDER = ["source_qsci_grid", "fixed_pi4", "one_parameter_vqe", "blockwise_vqe", "full_vqe", "qsci_coordinate"]
LABELS = ["Source QSCI grid", "Fixed pi/4", "One-parameter VQE", "Last-two VQE", "Full VQE", "Counts-only coordinate"]
COLORS = ["#333333", "#8666ac", "#2b6c9c", "#15968b", "#cf8034", "#bb4b65"]


def report(root, output=None):
    root = Path(root).resolve(); out = Path(output).resolve() if output else root/"report-v1"
    if out.exists(): raise RuntimeError("Use a new additive report directory")
    validation = json.loads((root/"validation.json").read_text())
    assert validation["status"] == SUCCESS and not validation["failures"]
    assert file_digest(root/"manifest.json") == validation["manifest_sha256"]
    cfg = json.loads((root/"manifest.json").read_text()); out.mkdir(parents=True)
    rows = []; acquisition_counts = {}; refs = {}
    for item in cfg["queue"]:
        case = item["case"]
        if case not in refs:
            path = Path(cfg["source"])/"references"/case/"reference.json"
            refs[case] = (json.loads(path.read_text()), file_digest(path))
        ref = refs[case][0]
        for mode in cfg["modes"]:
            folder = root/"cells"/case/f"s{item['seed']}"/mode
            r = json.loads((folder/"result.json").read_text()); control = json.loads((folder/"optimization.json").read_text())
            proof = next(v for v in validation["rows"] if (v["case"], v["source_seed"], v["mode"]) == (case, item["seed"], mode))
            assert file_digest(folder/"result.json") == proof["result_sha256"]
            energies = np.array(r["independent_QSCI_energies_Ha"]); errors = 1000*(energies-ref["fci_energy"])
            assert errors.min() >= -1e-6
            half = float(t.ppf(.95, len(energies)-1)*np.std(errors, ddof=1)/np.sqrt(len(energies)))
            d = []; unique = []; hf = []
            db = sqlite3.connect(f"file:{folder/'events.sqlite'}?mode=ro", uri=True)
            try:
                for raw, selected in db.execute("SELECT counts, selected_ids FROM draws WHERE role='independent_final'"):
                    ids, ns = unpack_counts64(raw); d.append(len(json.loads(selected))); unique.append(len(ids))
                    hf.append(sum(int(n) for x, n in zip(ids, ns) if int(x) == r["task"]["problem"]["hf_id"])/int(ns.sum()))
            finally: db.close()
            source_key = str(Path(item["source_cell"])/"events.sqlite")
            assert source_key not in acquisition_counts or acquisition_counts[source_key] == r["inherited_pilot_shots"]
            acquisition_counts[source_key] = r["inherited_pilot_shots"]
            res = r["sampling_resources"]
            rows.append({"case": case, "molecule": ref["case"]["molecule"], "distance_A": ref["case"]["distance"],
                         "source_seed": item["seed"], "mode": mode, "status": r["status"],
                         "n_qubits": ref["n_qubits"], "parameter_count": r["n_independent_ansatz_parameters"],
                         "optimized_variables": r["n_optimized_variables"], "physical_support": r["physical_sparse_support_size"],
                         "physical_VQE_error_mHa": 1000*(r["physical_VQE_energy_final_Ha"]-ref["fci_energy"]),
                         "QSCI_mean_error_mHa": float(errors.mean()), "QSCI_readout_t90_halfwidth_mHa": half,
                         "QSCI_error_std_mHa": float(errors.std(ddof=1)), "QSCI_observed_chemical_success": float(np.mean(errors <= 1.594)),
                         "final_full_gradient_inf": r["final_full_gradient_inf"], "active_stationary": control.get("active_stationary"),
                         "full_stationary": control.get("full_stationary"), "optimizer_status": control["status"],
                         "objective_calls": control.get("actual_objective_evaluations", 0),
                         "QPU_objective_cost_known": r["VQE_objective_QPU_shots"] is not None,
                         "known_branch_shots": r["known_per_branch_model_budget"], "inherited_source_pilot_shots": r["inherited_pilot_shots"],
                         "new_coordinate_pilot_shots": r["new_QSCI_pilot_shots"], "final_shots_per_rep": r["final_shots_per_rep"],
                         "final_repeats": r["final_repeats"], "new_actual_model_shots": r["new_actual_model_shots"],
                         "mean_final_actual_d": float(np.mean(d)), "mean_final_unique_observed": float(np.mean(unique)),
                         "mean_final_HF_fraction": float(np.mean(hf)), "sampling_CNOT": res["cnot_count"],
                         "sampling_CNOT_depth": res["cnot_depth"], "sampling_total_depth": res["total_depth"],
                         "actual_wall_seconds": r["actual_wall_seconds"], "peak_RSS_bytes": r["peak_process_RSS_bytes"],
                         "source_result_sha256": file_digest(Path(item["source_cell"])/"result.json"),
                         "result_sha256": proof["result_sha256"]})
    frame = pd.DataFrame(rows)
    baseline = frame[frame["mode"] == "source_qsci_grid"].set_index(["case", "source_seed"])
    for metric in ("QSCI_mean_error_mHa", "physical_VQE_error_mHa", "sampling_CNOT", "sampling_CNOT_depth", "mean_final_HF_fraction"):
        frame[f"delta_{metric}_vs_source"] = [r[metric]-baseline.loc[(r["case"], r["source_seed"]), metric] for _, r in frame.iterrows()]
    frame.to_csv(out/"all-cells.csv", index=False)
    fields = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and c not in ("source_seed", "distance_A")]
    cases = frame.groupby(["case", "molecule", "distance_A", "mode"], sort=True)[fields].mean().reset_index()
    cases.to_csv(out/"case-seed-means.csv", index=False)
    molecule = frame.groupby(["molecule", "mode"])[fields].mean().reset_index()
    molecule.to_csv(out/"molecule-cluster-means.csv", index=False)
    summaries = []
    for mode in ORDER:
        sub = molecule[molecule["mode"] == mode]
        delta = sub["delta_QSCI_mean_error_mHa_vs_source"].to_numpy()
        rng, _ = rng_for("parameter-controls-exploratory-molecule-bootstrap", cfg["mode"], mode)
        boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
        summaries.append({"mode": mode, "molecule_clusters": len(delta), "mean_QSCI_delta_vs_source_mHa": float(delta.mean()),
                          "exploratory_cluster90_low_mHa": float(np.quantile(boot, .05)), "exploratory_cluster90_high_mHa": float(np.quantile(boot, .95)),
                          "mean_physical_VQE_delta_mHa": float(sub["delta_physical_VQE_error_mHa_vs_source"].mean()),
                          "mean_HF_fraction_delta": float(sub["delta_mean_final_HF_fraction_vs_source"].mean()),
                          "QPU_objective_cost_known": mode not in ("one_parameter_vqe", "blockwise_vqe", "full_vqe")})
    pd.DataFrame(summaries).to_csv(out/"exploratory-molecule-cluster-comparison.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
    molecules = sorted(frame["molecule"].unique()); nrows = (len(molecules)+2)//3
    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "svg", "pdf"): fig.savefig(out/f"{name}.{suffix}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    for metric, label, fname, scale in [
        ("QSCI_mean_error_mHa", "Independent QSCI error (mHa)", "01-QSCI-error-vs-distance", True),
        ("physical_VQE_error_mHa", "Physical preparation energy error (mHa)", "02-physical-energy-vs-distance", True),
        ("delta_QSCI_mean_error_mHa_vs_source", "QSCI error minus source (mHa)", "03-angle-control-difference", True),
        ("mean_final_HF_fraction", "Measured HF fraction", "04-HF-duplication", False)]:
        fig, axes = plt.subplots(nrows, 3, figsize=(13, 3.4*nrows), squeeze=False)
        for ax, mol in zip(axes.flat, molecules):
            for mode, labelmode, color in zip(ORDER, LABELS, COLORS):
                g = cases[(cases["molecule"] == mol) & (cases["mode"] == mode)].sort_values("distance_A")
                ax.plot(g["distance_A"], g[metric], "o-", ms=3, lw=1.2, label=labelmode, color=color)
            ax.set(title=mol, xlabel="Bond distance / equal spacing (angstrom)", ylabel=label)
            if scale: ax.set_yscale("symlog", linthresh=.1)
            ax.axhline(0, lw=.5, color="grey"); ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(molecules):]: ax.set_visible(False)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle("Fixed inherited topology; all source seeds; 10,000 known model shots + UNKNOWN VQE cost", fontsize=11)
        save(fig, fname)
    fig, ax = plt.subplots(figsize=(10, 4.3))
    for i, (mode, color) in enumerate(zip(ORDER, COLORS)):
        g = frame[frame["mode"] == mode]
        ax.scatter([i]*len(g), g["objective_calls"], s=9, alpha=.45, color=color)
    ax.set(xticks=range(6), xticklabels=LABELS, ylabel="Actual classical objective + full-gradient calls",
           title="Classical VQE work (not QPU shots); bounded BFGS, every saved trajectory")
    ax.tick_params(axis="x", labelrotation=17); ax.grid(axis="y", alpha=.2)
    save(fig, "05-classical-objective-cost")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for mode, label, color in zip(ORDER, LABELS, COLORS):
        sub = molecule[molecule["mode"] == mode]
        for ax, resource in zip(axes, ("sampling_CNOT", "sampling_CNOT_depth")):
            ax.scatter(sub[resource], sub["QSCI_mean_error_mHa"], label=label, color=color, s=27)
            ax.set(xlabel=resource.replace("_", " "), ylabel="QSCI error (mHa); molecule mean")
            ax.set_yscale("symlog", linthresh=.1)
    axes[0].legend(fontsize=7)
    fig.suptitle("Conditional final readout/resource comparison; NOT a complete quantum-cost frontier", fontsize=11)
    save(fig, "06-final-circuit-resource-diagnostic")
    totals = {"new_raw_model_shots": int(frame["new_actual_model_shots"].sum()),
              "inherited_source_pilot_shots_unique_not_new": sum(acquisition_counts.values()),
              "unique_source_constructions": len(acquisition_counts), "new_classical_objective_calls": int(frame["objective_calls"].sum()),
              "new_coordinate_pilot_shots": int(frame["new_coordinate_pilot_shots"].sum()),
              "new_final_model_shots": validation["totals"]["new_final_shots"],
              "VQE_QPU_objective_cost": None, "hardware_shots": 0}
    assert totals["new_raw_model_shots"] == validation["totals"]["new_pilot_shots"]+validation["totals"]["new_final_shots"]
    table = "\n".join(f"| {s['mode']} | {s['mean_QSCI_delta_vs_source_mHa']:+.4f} | [{s['exploratory_cluster90_low_mHa']:+.4f}, {s['exploratory_cluster90_high_mHa']:+.4f}] | {s['mean_physical_VQE_delta_mHa']:+.4f} | {'known model' if s['QPU_objective_cost_known'] else 'UNKNOWN objective shots'} |" for s in summaries)
    text = f"""# 固定演算子列のパラメータ対照 — {cfg['mode']}

## 問いと検証範囲

同じRAACS演算子列で、角度の更新だけを変えると、物理状態のenergyと独立QSCIのenergyはどう変わるか。
{len(frame)} / {validation['planned']}セルを実行し、全{validation['totals']['draws_rebuilt']}drawのseed・counts、
{validation['totals']['physical_QPY_vectors_checked']}実QPY状態/測定分布、
{validation['totals']['objective_trace_calls_checked']}objective trace評価を再構成した。
小規模独立auditorのみ<=16q全vectorを使用。数値最適化は回路が実際に到達する<=512配置に限定。
新FCI solveなし。既存FCI値をこの外部レポートの誤差計算だけに使用した。

これはpost-constructionの条件付き診断。後から角度を変えた場合のoperator再選択は調べていない。
**D5のonlineパラメータ更新・選択相互作用は未完了で、この実験だけでは充足しない。**

## 結果（負の差＝sourceより改善）

全seed・全距離を使用。距離を独立分子として数えず、分子内平均後に分子を等重みとした。
区間は分子cluster bootstrapの探索的90%区間。少数cluster・事後設計でありconfirmatory検定ではない。
各セルの20（quickは4）drawのt90区間は条件付きreadout変動だけを示す。

| 方法 | ΔQSCI (mHa) | 探索的cluster90%区間 | Δ物理VQE (mHa) | 量子費用 |
|---|---:|---:|---:|---|
{table}

物理VQEの改善がQSCIの改善を保証するとは扱わない。部分変数でstationaryでも全変数stationaryとは限らない。
stationarityとoptimizer終了、global optimumの証明は別。全係数・評価trace・active/full勾配を保持した。

## 費用・公平性・限界

- 既知モデル予算10,000 = inherited source pilots + 新coordinate pilots + 独立final draw。
- 原構築{len(acquisition_counts)}本のpilot合計{sum(acquisition_counts.values()):,}は既存の費用。6方式・20final rep分へ増殖して実消費と呼ばない。
- 新たに生成したraw model shotsは{totals['new_raw_model_shots']:,}。repeatごとに実際に別抽選したfinal countsを一度ずつ計上。
- VQEの{totals['new_classical_objective_calls']:,}joint objective/gradient callsは**古典simulation**。そのQPU測定shot費用は未知（0やcall数に置換しない）。
- 従ってVQE方式を完全なend-to-end量子予算一致の優越性証拠とはしない。
- 元演算子列はsource full-cost探索から継承。fixed/random角度もその構造発見費用を無料にしない。
- 物理supportは回路の深さとともに指数増大し得る。浅い回路でのbounded sparse成功から一般的深回路scalabilityを主張しない。
- QSCIの観測配置+既知HFだけを採用。backendの到達可能配置を無料でCIへ挿入しない。
- 回路count/depthは同じ論理分解規約。native hardwareは未設定。実機shotsは0。
- 構造/seed固定の診断であり、強化LUCJやnative ADAPT-QSCIへの優越性の主張ではない。

## 再現

```bash
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmark.parameter_controls --mode {cfg['mode']}
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmark.parameter_controls_audit {root.relative_to(ROOT)}
```

CSV: all-cells / case-seed-means / molecule-cluster-means / exploratory-molecule-cluster-comparison。
6図をPNG/SVG/PDFで保存。データsource・audit・外部参照はreport-manifest.jsonに結合。
全研究は未完了。
"""
    (out/"REPORT_JA.md").write_text(text)
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Parameter controls</title><h1>Fixed-operator parameter controls</h1><p>Conditional diagnostic; not an online discovery algorithm. VQE QPU objective cost UNKNOWN.</p>'+"".join(f'<h2>{p.stem}</h2><img src="{p.name}" style="max-width:100%"><p><a href="{p.stem}.svg">SVG</a> / <a href="{p.stem}.pdf">PDF</a></p>' for p in sorted(out.glob("*.png"))))
    manifest = {"status": "VALIDATED_CONDITIONAL_PARAMETER_CONTROL_REPORT", "cells": len(frame), "totals": totals,
                "summaries": summaries, "experiment_manifest_sha256": file_digest(root/"manifest.json"),
                "audit_sha256": file_digest(root/"validation.json"), "report_code_sha256": file_digest(Path(__file__)),
                "external_reference_json_sha256": {case: pair[1] for case, pair in refs.items()},
                "artifacts": {p.name: file_digest(p) for p in sorted(out.iterdir()) if p.is_file()},
                "whole_study_complete": False, "D5_online_requirement_completed": False}
    save_json(out/"report-manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("root"); p.add_argument("--output")
    args = p.parse_args(); r = report(args.root, args.output)
    print(json.dumps({"status": r["status"], "cells": r["cells"], "totals": r["totals"], "summaries": r["summaries"]}))
