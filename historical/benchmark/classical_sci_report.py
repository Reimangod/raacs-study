"""Independent numerical audit and small-system SCI-versus-sampler report."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import ROOT, file_digest, save_json
from .chemistry import load_reference
from .study_figures import energy_limits


def report(root, b1):
    root = Path(root); b1 = Path(b1)
    mf = json.loads((root / "manifest.json").read_text()); source = Path(mf["source"])
    for name, sha in mf["implementation"].items():
        if file_digest(ROOT / name) != sha:
            raise RuntimeError("SCI frozen code drift: "+name)
    data = pd.read_csv(root / "summary.csv")
    cells = pd.read_csv(root / "all-cell-status.csv")
    failures = []; checked = 0; started = perf_counter()
    with threadpool_limits(limits=1):
        for case in mf["cases"]:
            ref = load_reference(source / "references" / case)
            for criterion in cells.criterion.unique():
                directory = root / "cases" / case / criterion
                result = json.loads((directory / "result.json").read_text())
                previous = set(); energy_previous = np.inf
                for record in result["records"]:
                    with np.load(directory / f"step-{record['iteration']:04d}.npz") as z:
                        ids, c = z["ids"], z["coefficients"]
                    checked += 1
                    assert ids.tolist() == record["selected"] and len(ids) == record["dimension"]
                    assert previous.issubset(set(ids)) and len(set(ids)) == len(ids)
                    assert set(ids).issubset(set(ref["ids"]))
                    hs = ref["h"][ids][:, ids].toarray()
                    independent = float(eigh(hs, subset_by_index=(0, 0), eigvals_only=True)[0])
                    assert abs(independent-record["energy"]) < 1e-9
                    assert abs(np.linalg.norm(c)-1.) < 1e-9
                    assert np.linalg.norm(hs @ c-record["energy"]*c) < 1e-8
                    outside = ref["h"][:, ids] @ c
                    outside[ids] = 0
                    assert abs(np.linalg.norm(outside)-record["external_residual_norm"]) < 1e-9
                    assert record["energy"] <= energy_previous+1e-9
                    assert record["energy"] >= ref["meta"]["fci_energy"]-1e-9
                    previous, energy_previous = set(ids), record["energy"]
                rows = data[(data.case_id == case) & (data.criterion == criterion)]
                assert len(rows) == len(mf["R_grid"])
                for row in rows.itertuples():
                    record = max((r for r in result["records"] if r["dimension"] <= row.R_cap), key=lambda r: r["dimension"])
                    assert row.actual_d == record["dimension"] and abs(row.variational_energy-record["energy"]) < 1e-10
    out = root / "derived-report-v1"; out.mkdir(exist_ok=True)
    audit = {"status": "ALL_SMALL_SYSTEM_SCI_RECORDS_NUMERICALLY_VALIDATED", "cells": len(cells), "summary_rows": len(data),
             "independent_saved_space_eigensolves": checked, "all_external_residuals_checked": True,
             "audit_seconds": perf_counter()-started, "failures": failures, "full_SHCI": False,
             "practical_large_system_validation": False, "hardware_shots": 0, "whole_study_complete": False}
    save_json(out / "validation.json", audit)
    b1data = pd.read_csv(b1 / "comparison-with-recovery.csv")
    q = b1data[(b1data.shots.isin([1000, 100000])) & b1data.method.isin(["source_final_ceo_ovp", "source_final_ceo_mvp", "lucj_ccsd_2_vqe", "lucj_mp2_2_vqe"])]
    q = q[["case_id", "method", "shots", "R_cap", "mean_actual_d", "mean_error_mHa", "new_full_construction_seconds", "historical_trajectory_seconds_as_recorded"]]
    q.to_csv(out / "quantum-common-R-comparison.csv", index=False)
    figs = out / "figures"; figs.mkdir(exist_ok=True); names = []
    for case, g in data.groupby("case_id"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for criterion, part in g.groupby("criterion"):
            for ax in axes:
                ax.plot(part.R_cap, part.variational_error_mHa, "o-", markersize=3, label=criterion)
        for ax, shots in zip(axes, [1000, 100000]):
            s = q[(q.case_id == case) & (q.shots == shots)]
            for method, part in s.groupby("method"):
                ax.plot(part.R_cap, part.mean_error_mHa, ".--", label=method)
            ax.set_xscale("log", base=2); ax.set_yscale("symlog", linthresh=.01)
            ax.set(xlabel="Common CI cardinality cap R (actual d may differ)", ylabel="Variational error (mHa)", title=f"Quantum readout M{shots}; classical shots0")
            energy_limits(ax, list(g.variational_error_mHa)+list(s.mean_error_mHa))
            ax.axhline(1.5936014376, color="gray", lw=.8, ls="--")
        fig.suptitle(f"{case} — ranked SCI controls vs bounded sampler readouts\nSmall source Hamiltonian backend; NOT optimized SHCI / quantum advantage")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=7)
        name = case+"-SCI-and-quantum-common-R"; names.append(name)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(figs / f"{name}.{ext}", dpi=220)
        plt.close(fig)
    lines = ["# 古典SCI順位付け対照：実行・独立検証", "", f"全{len(mf['cases'])}条件、{len(cells)}セル、{len(data)}共通R結果。保存部分空間{checked}個を独立に再対角化・外部残差照合。実機shots0。", "",
             "HCIのmax |Hxi ci|、CIPSI係数型 |rx/(E−Hxx)|、EN energy型 |rx|²/|E−Hxx|を分けて実装した。次元倍増・R≤512の順位付け対照であり、fixed-epsilon HCIや最適化SHCIの完全再現ではない。", "",
             "## R128の全条件", "", "| 条件 | criterion | 実d | 誤差 mHa | 外部残差 norm |", "|---|---|---:|---:|---:|"]
    for row in data[data.R_cap == 128].itertuples():
        lines.append(f"| {row.case_id} | {row.criterion} | {row.actual_d} | {row.variational_error_mHa:.6g} | {row.external_residual_norm:.6g} |")
    lines += ["", "## 何が分かったか／何を言えないか", "",
              "- 小さい部分空間では古典的に接続配置を探すだけでも大きな精度改善が起こる。QSCI samplerがHFや固定LUCJより良いことだけでは量子的な追加価値の証明にならない。", "- 共通Rと実CI次元は別。図ではactual dが異なり得る。量子側pairwise-d対照はB1/B2に保存しており、この図をdimension完全一致の因果比較とは呼ばない。", "- CI内部Ritz残差と、外部配置へ漏れる残差は別。小さい内部残差は小さいenergy errorの証明ではない。", "- PT2は推定値。near-zero denominatorがあればraw PT2はnull、正則化値を別列に保存。variational energyと足し合わせた値を厳密上限・精度証明と呼ばない。", "- 本試験は既存の小系Fock疎行列を読み込む。新たなFCIは行わず、FCI正解は評価のみ。ただし、このbackendは20–40qubitの実用SCI実証ではない。", "- 古典計算のfull prefix-to512費用を各Rに重複加算しない。source Hamiltonian構築費用はshared historical setupで比較外・未知。QPU gate費用はN/Aであり、無料の量子状態準備を意味しない。", "- optimized heat-bath generator、semistochastic PT2、production SHCI/DMRG参照は未導入。今回のwallをそれらより速い／遅いと一般化しない。", ""]
    (out / "REPORT_JA.md").write_text("\n".join(lines))
    save_json(out / "report-status.json", {"figures": len(names), "formats": ["png", "svg", "pdf"], "audit": audit,
                                         "implementation_sha256": file_digest(__file__), "whole_study_complete": False})
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True)
    parser.add_argument("--b1", default=str(ROOT / "results/sampler-baselines-B1-standard-v1"))
    args = parser.parse_args()
    print(json.dumps(report(args.output, args.b1)))
