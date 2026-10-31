"""Outer full-SQD qualification figures and explicit complete-sector caveat."""
import json
from math import comb
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import ROOT, file_digest, save_json


def report():
    root = ROOT/"results/full-fermionic-SQD-quick-v1"
    valid = json.loads((root/"validation.json").read_text())
    assert valid["status"] == "FULL_FERMIONIC_SQD_QUICK_RAW_AND_PROJECTIONS_VALIDATED"
    frame = pd.read_csv(root/"qualification-summary.csv"); scopes = []
    for case in frame.case.unique():
        meta = json.loads((ROOT/"results/standard-v1/references"/case/"reference.json").read_text())
        ix = frame.case == case; dim = comb(meta["n_orbitals"], meta["nalpha"])*comb(meta["n_orbitals"], meta["nbeta"])
        frame.loc[ix, "QSCI_error_mHa"] = 1000*(frame.loc[ix, "QSCI_energy_Ha"]-meta["fci_energy"])
        frame.loc[ix, "SQD_error_mHa"] = 1000*(frame.loc[ix, "SQD_energy_Ha"]-meta["fci_energy"])
        frame.loc[ix, "registered_full_sector_dimension"] = dim
        frame.loc[ix, "SQD_selected_space_spans_full_registered_sector"] = frame.loc[ix, "SQD_d"] == dim
        scopes.append({"case": case, "registered_sector_dimension": dim,
                       "SQD_rows_spanning_entire_sector": int((frame.loc[ix, "SQD_d"] == dim).sum()),
                       "rows": int(ix.sum())})
    output = root/"derived-report-v1"; output.mkdir(exist_ok=True)
    frame.to_csv(output/"posthoc-reference-and-sector-scope.csv", index=False)
    save_json(output/"selected-sector-scope.json", {"cases": scopes,
        "meaning_of_legacy_full_FCI_solve_false_flag": "No separate full-FCI reference solver is invoked by the workflow; it does not assert selected SCI never spans the complete small sector",
        "full_selected_sector_is_mathematically_an_exact_diagonalization_in_that_sector": True})
    figures = []
    for case, group in frame.groupby("case"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        methods = sorted(group.method.unique()); x = np.arange(len(methods)); width = .18
        for j, (noise, label) in enumerate([(0., "ideal"), (.01, "1% synthetic readout")]):
            g = group[group.readout_flip_probability == noise].groupby("method").mean(numeric_only=True).reindex(methods)
            for k, (column, name) in enumerate([("QSCI", "explicit QSCI"), ("SQD", "full fermionic SQD")]):
                offset = (2*j+k-1.5)*width
                axes[0].bar(x+offset, g[column+"_error_mHa"], width, label=f"{name}, {label}")
                axes[1].bar(x+offset, g[column+"_d"], width)
        short = [m.replace("B1:source_final_", "").replace("B1:", "").replace("RAACS:full_cost-qsci_local_grid-last_only-s101-exact_small", "RAACS full cost") for m in methods]
        for ax in axes:
            ax.set_xticks(x, short, rotation=18, ha="right"); ax.grid(axis="y", alpha=.2)
        axes[0].set_ylabel("Error against post-hoc FCI reference (mHa)"); axes[0].set_yscale("symlog", linthresh=.001)
        axes[1].set_ylabel("Selected determinant dimension")
        axes[1].axhline(group.registered_full_sector_dimension.iloc[0], color="black", linestyle="--", linewidth=1)
        fig.suptitle(case+": reused M1000 acquisitions, two repeats; small-sector qualification")
        handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
        fig.tight_layout(rect=(0, .1, 1, .94)); name = "full-SQD-"+case
        for suffix in ("png", "svg", "pdf"): fig.savefig(output/f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig); figures.append(name)
    text = f"""# Full fermionic SQD：初回の実行・独立監査\n\n
H2/H4、4 sampler、2 readout repeat、ideal/合成1%readoutの32セルを実行。
32/32成功。192個の実際のCartesian選択空間を、独立Slater–Condon積分行列
で再対角化し、最大差は{valid['max_projection_energy_discrepancy_Ha']:.3g} Ha。
32 recovery出力記録とraw countsも照合済み。

これは公式SDKのfull fermionic SQD workflowであり、旧explicit determinant
recoveryとは異なる。raw観測配置、古典的recoveryで生まれた配置、α×βの
Cartesian展開で追加された配置は別々に保存した。SDKのbatch/iteration最良点
の選択を、独立な次回読出しの成績とは呼ばない。

## 特に重要な限界

小さいH2/H4では選択SCI空間が物理sector全体へ到達し得る。その場合は
数学的にそのsectorの完全対角化であり、量子優位性・大規模性能ではない。
元recordの`full_FCI_solve=false`は「別のfull-FCI参照solverを呼ばない」
という実装上の意味に限定し、完全sectorに届かなかったという意味には使わない。
各セルの該当状況をposthoc-reference-and-sector-scope.csvに明示した。

14個の既存acquisitionを再利用。corruption armは同じraw dataを古典的に
変換した対応比較であり、新しい独立量子標本ではない。新規量子acquisition0、
実機shots0。FCI参照はこの外側レポートのみで使い、SDKの選択へ渡していない。

合成readoutだけの32セルでPhase Fは完了しない。全分子、gate depolarizing、
relaxation/dephasing、routing/SWAP、資源を揃えた比較が残る。S²の値は
測るが、今回はS²を強制していない。論文の実機設定の完全再現ではない。
"""
    (output/"REPORT_JA.md").write_text(text)
    result = {"figures": figures, "formats": ["png", "svg", "pdf"], "posthoc_FCI_reporting_only": True,
              "reporter_sha256": file_digest(__file__), "whole_study_complete": False}
    save_json(output/"manifest.json", result); return result


if __name__ == "__main__": print(json.dumps(report()))
