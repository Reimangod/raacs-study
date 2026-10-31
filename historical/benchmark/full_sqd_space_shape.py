"""Explain non-nested explicit vs Cartesian selected spaces without FCI labels."""
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .common import file_digest, save_json
from .fermionic_sqd_durable import decode
from .resource_adaptive_sampler import integral_problem


def spin_shapes(ids, n):
    alpha = set(); beta = set()
    for x in ids:
        alpha.add(sum(((x >> (n-1-2*j)) & 1) << j for j in range(n//2)))
        beta.add(sum(((x >> (n-2-2*j)) & 1) << j for j in range(n//2)))
    return len(alpha), len(beta)


def diagnose(root):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text()); validation = json.loads((root/"validation.json").read_text())
    assert file_digest(root/"events.sqlite") == validation["database_sha256"]
    destination = root/"space-shape-diagnostic-v1"; destination.mkdir(exist_ok=True)
    db = sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro", uri=True)
    rows = []
    with threadpool_limits(limits=1):
        for raw, in db.execute("SELECT result FROM cells WHERE status='COMPLETED'"):
            r = decode(raw); task = r["task"]; ci, sqd = r["QSCI"], r["SQD"]
            n = r["problem"]["n_qubits"]; aa, bb = spin_shapes(ci["selected_ids"], n)
            q, s = set(ci["selected_ids"]), set(sqd["selected_interleaved_ids"])
            delta = sqd["energy_Ha"]-ci["energy_Ha"]
            if q <= s: assert delta < 1e-7
            # Bounded diagnostic union: no unobserved FCI determinants are added.
            union = sorted(q | s); assert len(union) <= 128
            backend, info = integral_problem(Path(cfg["source"])/"references"/task["case"])
            union_energy = float(np.linalg.eigvalsh(backend.matrix(union, union))[0])
            assert union_energy <= min(sqd["energy_Ha"], ci["energy_Ha"])+1e-7
            rows.append({**task, "molecule": task["case"].split("_R")[0], "QSCI_d": len(q), "SQD_d": len(s),
                "QSCI_alpha_strings": aa, "QSCI_beta_strings": bb, "QSCI_Cartesian_closure_dimension": aa*bb,
                "QSCI_space_cannot_fit_registered_spin_cap": max(aa, bb) > cfg["SQD_settings"]["max_spin_strings"],
                "QSCI_Cartesian_closure_exceeds_total_cap": aa*bb > cfg["R"],
                "QSCI_subset_of_returned_SQD": q <= s, "QSCI_configurations_not_in_returned_SQD": len(q-s),
                "SQD_configurations_not_in_QSCI": len(s-q), "SQD_minus_QSCI_mHa": delta*1000,
                "union_diagnostic_dimension": len(union), "union_energy_Ha": union_energy,
                "union_minus_best_existing_mHa": 1000*(union_energy-min(sqd["energy_Ha"], ci["energy_Ha"])),
                "union_diagnostic_NOT_primary_algorithm_or_free_work": True})
    db.close(); frame = pd.DataFrame(rows)
    frame.to_csv(destination/"all-completed-space-shapes.csv", index=False)
    molecule = frame.groupby("molecule", as_index=False).agg(cells=("case", "size"),
        fraction_QSCI_exceeds_spin_cap=("QSCI_space_cannot_fit_registered_spin_cap", "mean"),
        fraction_closure_exceeds_R=("QSCI_Cartesian_closure_exceeds_total_cap", "mean"),
        fraction_QSCI_subset=("QSCI_subset_of_returned_SQD", "mean"),
        mean_QSCI_ids_lost=("QSCI_configurations_not_in_returned_SQD", "mean"),
        mean_union_minus_best_mHa=("union_minus_best_existing_mHa", "mean"))
    molecule.to_csv(destination/"molecule-space-shape-means.csv", index=False)
    text = """# なぜfull SQDがexplicit QSCIより悪化し得るか

今回のfull SQDは各spin最大8strings、そのCartesian積で最大64determinants。
explicit QSCIは同じ総R64でも、64個の任意の観測determinantsを保持できる。
例えば64配置を保つためにalpha20strings、beta20stringsが必要なら、Cartesian
方式では400配置が必要になる。同じ64上限は同じ表現能力ではない。

従って選択空間は一般に包含関係がなく、SQDが既存の低energy QSCI配置を
落とすことがある。recoveryの追加はenergyを単調に下げる定理ではない。
実際の配置集合・必要alpha/beta数・落とした配置数を全成功セルで数えた。
QSCIがSQD空間に含まれるセルでは、変分原理のenergy不等式を検査済み。

QSCIとSQDの実配置集合のunion（最大128）も外側の診断として対角化した。
これはFCIから補完しておらず、両空間を包含するため悪化しない。ただし
主実験のR64を超える場合があり、追加古典計算も発生する。主手法の成績へ
混ぜず、無料の修正や量子サンプラーの改善とは呼ばない。

今回の差には空間形状・batch/subsampling・iteration上限・係数選択が含まれる。
shape統計だけで差の全てを因果的に説明した、とは言わない。小さいspin capの
結果から『full SQD一般に劣る』『CEO一般に有利』とは結論しない。
"""
    (destination/"EXPLANATION_JA.md").write_text(text)
    result = {"status": "BOUNDED_SELECTED_SPACE_SHAPE_DIAGNOSTIC_COMPLETE", "cells": len(frame),
        "database_sha256": validation["database_sha256"], "script_sha256": file_digest(__file__),
        "QSCI_subsets": int(frame.QSCI_subset_of_returned_SQD.sum()),
        "QSCI_spaces_exceeding_spin_cap": int(frame.QSCI_space_cannot_fit_registered_spin_cap.sum()),
        "diagnostic_union_not_primary_R64": True, "new_quantum_acquisitions": 0, "new_FCI_reference_solves": 0,
        "artifacts": {p.name: file_digest(p) for p in destination.iterdir() if p.is_file() and p.name != "manifest.json"}}
    save_json(destination/"manifest.json", result); return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); a = p.parse_args(); print(json.dumps(diagnose(a.output)))
