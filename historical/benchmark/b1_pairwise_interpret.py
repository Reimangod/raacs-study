"""Complete-case, actual-d paired descriptive interpretation of sealed B1 data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from .common import ROOT, file_digest, save_json


def analyze(b1, output):
    b1, output = Path(b1), Path(output); output.mkdir(exist_ok=True, parents=True)
    mf = json.loads((b1 / "manifest.json").read_text())
    data = pd.read_csv(b1 / "comparison-with-recovery.csv")
    pair_rows = []
    for case in mf["case_ids"]:
        pairs = pd.read_parquet(b1 / "readouts" / case / "pairwise-ceo-matched-d.parquet")
        pairs = pairs[pairs.reference_method == "source_final_ceo_mvp"].copy()
        pairs["delta_error_mHa_actual_d_matched"] = (pairs.energy-pairs.reference_energy)*1000
        for (method, M, R), g in pairs.groupby(["method", "shots", "cap"]):
            a = data[(data.case_id == case) & (data.method == "source_final_ceo_mvp") & (data.shots == M) & (data.R_cap == R)]
            b = data[(data.case_id == case) & (data.method == method) & (data.shots == M) & (data.R_cap == R)]
            assert len(a) == len(b) == 1
            a, b = a.iloc[0], b.iloc[0]
            delta = g.delta_error_mHa_actual_d_matched
            pair_rows.append({"case_id": case, "molecule": case.split("_R")[0], "method": method, "shots": M, "R_cap": R,
                              "CEO_common_R_error_mHa": a.mean_error_mHa, "method_common_R_error_mHa": b.mean_error_mHa,
                              "delta_error_mHa_common_R": b.mean_error_mHa-a.mean_error_mHa,
                              "delta_error_mHa_actual_d_matched": float(delta.mean()), "mean_matched_d": float(g.common_d.mean()),
                              "paired_readout_t90_halfwidth_mHa": float(t.ppf(.95, len(g)-1)*delta.std(ddof=1)/np.sqrt(len(g))),
                              "repeats": len(g), "method_cnot": b.cnot_count, "CEO_cnot": a.cnot_count,
                              "method_cnot_depth": b.cnot_depth, "CEO_cnot_depth": a.cnot_depth,
                              "CNOT_reduction_percent": 100*(1-b.cnot_count/a.cnot_count) if a.cnot_count > 0 else np.nan,
                              "CNOT_depth_reduction_percent": 100*(1-b.cnot_depth/a.cnot_depth) if a.cnot_depth > 0 else np.nan,
                              "method_new_full_construction_seconds": b.new_full_construction_seconds,
                              "CEO_historical_trajectory_seconds": a.historical_trajectory_seconds_as_recorded,
                              "cost_comparability": "NOT_MATCHED_DEVICE_OR_FULL_WALL_SCOPE; do not divide for speedup"})
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(output / "B1-actual-d-vs-common-R.csv", index=False)
    selected = pairs[(pairs.shots == 1000) & (pairs.R_cap == 128)]
    methods = ["lucj_ccsd_2_vqe", "lucj_mp2_2_vqe", "lucj_random_2_vqe_s101", "lucj_random_2_vqe_s211", "lucj_random_2_vqe_s307", "lucj_ccsd_2_qsci"]
    rows = []
    for mol in sorted(set(x.split("_R")[0] for x in mf["case_ids"])):
        planned = [x for x in mf["case_ids"] if x.startswith(mol+"_R")]
        for method in methods:
            s = selected[(selected.molecule == mol) & (selected.method == method)]
            # Each comparison uses the SAME available geometries on both arms.
            row = {"molecule": mol, "method": method, "planned_conditions": len(planned), "available_paired_conditions": len(s),
                   "missing_conditions": json.dumps(sorted(set(planned)-set(s.case_id))),
                   "unit": "paired geometry-level descriptive median; NOT independent molecule replicates"}
            for column in ("CEO_common_R_error_mHa", "method_common_R_error_mHa", "delta_error_mHa_common_R",
                           "delta_error_mHa_actual_d_matched", "CNOT_reduction_percent", "CNOT_depth_reduction_percent"):
                row["median_"+column] = float(s[column].median()) if len(s) else None
            rows.append(row)
    pd.DataFrame(rows).to_csv(output / "B1-paired-geometry-descriptive-medians.csv", index=False)
    lines = ["# CEO-MVPと強化LUCJ：実CI次元と費用の解釈", "",
             "以下はM1000・R128の記述的集計。CCSD欠測では両手法とも同じavailable geometryへ制限したペア中央値を使用し、予定・欠測を併記する。分子間の一般優越性や有意差検定の主張ではない。", "",
             "差分ΔE＝LUCJ−CEO。負ならLUCJ側の誤差が低い。削減率の負値はLUCJ側で資源が増えたことを意味する。CNOT減とDepth減を同一視しない。", "",
             "| 分子 | 手法 | ペア / 予定 | 共通R ΔE mHa | 実d一致 ΔE mHa | CNOT削減 % | depth削減 % |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["method"] not in ("lucj_ccsd_2_vqe", "lucj_mp2_2_vqe"):
            continue
        lines.append(f"| {row['molecule']} | {row['method']} | {row['available_paired_conditions']} / {row['planned_conditions']} | {row['median_delta_error_mHa_common_R']:.6g} | {row['median_delta_error_mHa_actual_d_matched']:.6g} | {row['median_CNOT_reduction_percent']:.5g} | {row['median_CNOT_depth_reduction_percent']:.5g} |")
    lines += ["", "## 何を結論にできるか", "",
              "- 共通R上限は、同じ実CI次元の保証ではない。sample supportが広い方法は同じRでも多くの配置を使うことがある。actual-d対照はこの違いを切り分けるが、小さいdに切り詰めるので唯一の実用性能指標でもない。両方を見る。",
              "- H4_R1.1では共通Rでの見え方と、実dを合わせたLUCJ−CEO誤差差分が異なる。全行の値はCSVに保存しており、好きな方だけ採用しない。",
              "- LUCJは深さが小さくてもCNOT総数が多い場合がある。CEOが低誤差でも深い場合がある。精度・count・depth・shotsのトレードオフを保持し、任意の重みで一つの『得点』へ潰さない。",
              "- 新LUCJは上限付きL-BFGS-B等の結果であり、任意optimizerでの最良LUCJとは言わない。新cohortの全wallと、旧CPU/A100 historical trajectoryはscopeが違い、その比を費用優位と呼ばない。",
              "- 相関の強さについては結合距離だけで断定せず、HF weight/correlation energy等をposthoc診断として併記する。FCIの診断量は構築・選択・停止へ入れない。",
              "- 近いgeometryを独立分子として数えた有意差・標本数の水増しはしない。分子cluster推論は独立した評価単位・複数分子で別途実施する。", ""]
    (output / "REPORT_JA.md").write_text("\n".join(lines))
    status = {"paired_rows": len(pairs), "descriptive_rows": len(rows), "source_dataset": json.loads((b1 / "REPRODUCIBILITY.json").read_text())["dataset_id"],
              "source_comparison_sha256": file_digest(b1 / "comparison-with-recovery.csv"), "implementation_sha256": file_digest(__file__),
              "whole_study_complete": False}
    save_json(output / "status.json", status)
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--b1", default=str(ROOT / "results/sampler-baselines-B1-standard-v1"))
    p.add_argument("--output", default=str(ROOT / "results/sampler-baseline-interpretation-v1")); args = p.parse_args()
    print(json.dumps(analyze(args.b1, args.output)))
