"""Read-only, separate N and Sz diagnostics of final saved samplers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def analyze(root):
    root = Path(root)
    if (root / "REPRODUCIBILITY.json").exists():
        raise RuntimeError("Sealed output; use a successor derivative")
    source = Path(json.loads((root / "manifest.json").read_text())["source"])
    rows = []
    for case in sorted((source / "checkpoints").iterdir()):
        meta = json.loads((source / "references" / case.name / "reference.json").read_text())
        n = meta["n_qubits"]
        alpha_mask = sum(1 << (n-1-q) for q in range(0, n, 2))
        beta_mask = sum(1 << (n-1-q) for q in range(1, n, 2))
        a = np.array([(i & alpha_mask).bit_count() for i in range(1 << n)])
        b = np.array([(i & beta_mask).bit_count() for i in range(1 << n)])
        for method in sorted(case.iterdir()):
            checkpoints = sorted(method.glob("step-*.json"))
            if not checkpoints:
                continue
            cp = json.loads(checkpoints[-1].read_text())
            with np.load(cp["distribution"]) as data:
                p = data["probability"]
            rows.append({"case_id": case.name, "method": method.name, "step": cp["step"],
                         "checkpoint_id": cp["checkpoint_id"],
                         "valid_N_probability": float(p[a+b == meta["n_electrons"]].sum()),
                         "valid_Sz_probability": float(p[a-b == meta["nalpha"]-meta["nbeta"]].sum()),
                         "valid_joint_probability": float(p[(a == meta["nalpha"]) & (b == meta["nbeta"])].sum()),
                         "mean_N": float(p @ (a+b)), "mean_Sz": float(p @ (a-b)/2),
                         "scope": "saved_final_Z_distribution; N_and_Sz_not_total_S2"})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "analysis/final-sector-diagnostic.csv", index=False)
    lines = ["# NとSzを分けた最終保存点の診断", "",
             "全324 available method/caseを保持。粒子数Nの一致とSzの一致、全スピンS²は異なる検査である。元の回路・収束ラベルは変更していない。", "",
             "| 条件 | 手法 | step | P(N正しい) | P(Sz正しい) | P(両方) |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["valid_joint_probability"] < .99:
            lines.append(f"| {row['case_id']} | {row['method']} | {row['step']} | {row['valid_N_probability']:.10g} | {row['valid_Sz_probability']:.10g} | {row['valid_joint_probability']:.10g} |")
    lines += ["", "H4_R2.2/qubitはN=4を保存するが、平均Sz=-2で、要求したSz=0にはほぼ確率がない。これを粒子数漏れと呼んではいけない。N2_R2.2/qubitでは主にSzの不一致に加え、小さいN不一致も観測される。", "",
              "変分勾配収束は要求sectorのsamplerとしての適合性を保証しない。QSCIでは要求sectorの観測配置のみを使用するため、これらの確率損失はraw-shot費用に反映される。事後に無償で条件付けて有利なsamplerへ置換しない。新しいsector-preserving手法は別methodとして評価する。", "",
              "小さなsectorでRがsector全体以上になるケースは、広いランダムsamplerでも良いCIを得られうる。低HF重複だけでは重要配置の発見効率やエネルギー改善を保証しない。これは量子優位性の証拠ではない。", ""]
    (root / "SECTOR_DIAGNOSTIC_JA.md").write_text("\n".join(lines))
    return {"rows": len(rows), "joint_below_099": int((frame.valid_joint_probability < .99).sum())}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    print(json.dumps(analyze(p.parse_args().output)))
