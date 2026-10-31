"""Post-audit all-molecule SQD report; FCI appears only in outer diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import file_digest, save_json
from .fermionic_sqd_durable import decode, read_trace
from .resource_adaptive_sampler import integral_problem

LABELS = {"B1:source_final_ceo_ovp": "CEO OVP", "B1:source_final_ceo_mvp": "CEO MVP",
          "B1:lucj_ccsd_2_vqe": "CCSD2-VQE LUCJ", "RAACS:full_cost-qsci_local_grid-last_only-s101-exact_small": "RAACS full cost"}
COLORS = dict(zip(LABELS, ("#3465a4", "#9c4a93", "#d17a22", "#2c8263")))


def source_resources(cfg, case, method):
    source = Path(cfg["comparison"])/"readouts"/case/"readouts.sqlite"
    db = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    metadata = json.loads(db.execute("SELECT metadata FROM cells WHERE method=?", (method,)).fetchone()[0]); db.close()
    path = Path(metadata["result_path"])
    assert file_digest(path) == metadata["result_sha256"]
    result = json.loads(path.read_text())
    if "resources" in result:
        return result["resources"], {"source_result": str(path), "sha256": metadata["result_sha256"],
            "construction_seconds_scope_specific": result.get("construction_wall_seconds", result.get("trajectory_seconds")),
            "construction_model_shots": None, "quantum_construction_shots_status": "UNKNOWN_NOT_ZERO"}
    dbpath = path.parent/"events.sqlite"
    assert file_digest(dbpath) == result["event_database_sha256"]
    db = sqlite3.connect(f"file:{dbpath}?mode=ro", uri=True)
    cid = result["final_circuit_allocations"][0]["circuit_id"]
    assert len(result["final_circuit_allocations"]) == 1
    resource = json.loads(db.execute("SELECT resources FROM circuits WHERE id=?", (cid,)).fetchone()[0]); db.close()
    return resource, {"source_result": str(path), "sha256": metadata["result_sha256"],
        "construction_seconds_scope_specific": result["actual_wall_seconds"], "construction_model_shots": result["pilot_raw_model_shots"],
        "quantum_construction_shots_status": "KNOWN_SIMULATED_PILOTS_NOT_HARDWARE"}


def report(root):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text()); validation = json.loads((root/"validation.json").read_text())
    assert validation["status"] == "FULL_SQD_DURABLE_COUNTS_TRACES_AND_SPACES_VALIDATED"
    assert file_digest(root/"events.sqlite") == validation["database_sha256"]
    output = root/"report-v1"; output.mkdir(exist_ok=True)
    db = sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro", uri=True)
    rows = []; diagnoses = []; provenance = {}; cache = {}
    with threadpool_limits(limits=1):
        for tid, result_raw in db.execute("SELECT id,result FROM cells ORDER BY id"):
            r = decode(result_raw); task = r["task"]; case = task["case"]
            meta = json.loads((Path(cfg["source"])/"references"/case/"reference.json").read_text())
            a = {"id": tid, **task, "status": r["status"], "molecule": meta["case"]["molecule"], "distance_A": meta["case"]["distance"],
                 "new_quantum_acquisitions": 0, "hardware_shots": 0, "error": r.get("error"),
                 "new_cell_classical_seconds": r.get("cell_actual_classical_wall_seconds"), "full_scope_QPU_cost": None}
            trace = read_trace(db, tid)
            if r["status"] == "ENGINEERING_OR_RESOURCE_FAILURE_PRESERVED":
                backend, info = integral_problem(Path(cfg["source"])/"references"/case)
                for event in trace:
                    if event["kind"] != "SOLVE_RESULT": continue
                    p = event["payload"]; spectrum = np.linalg.eigvalsh(backend.matrix(p["selected_ids"], p["selected_ids"]))
                    diagnoses.append({"id": tid, "case": case, "method": task["method"], "rep": task["rep"], "readout_flip_probability": task["readout_flip_probability"],
                        "event_seq": event["seq"], "dimension": len(spectrum), "returned_energy_Ha": p["energy_Ha"],
                        "selected_ground_energy_Ha": float(spectrum[0]), "returned_minus_ground_mHa": float(1000*(p["energy_Ha"]-spectrum[0])),
                        "distance_to_nearest_selected_eigenvalue_Ha": float(min(abs(spectrum-p["energy_Ha"]))),
                        "spin_square": p["spin_square"], "diagnostic_only_no_retry": True})
            if r["status"] == "COMPLETED":
                ci, sqd, md = r["QSCI"], r["SQD"], r["same_actual_d"]
                denominator = meta["rhf_energy"]-meta["fci_energy"]
                raw_ids = set(sqd["raw_observed_ids"])
                recovered = set(x for p in sqd["recovery_trace"] for x in p["new_recovered_ids_not_raw_observed"])
                cartesian = set(x for p in sqd["solver_trace"] for x in p["cartesian_ids_not_raw_observed"])
                a.update(QSCI_error_mHa=1000*(ci["energy_Ha"]-meta["fci_energy"]), SQD_error_mHa=1000*(sqd["energy_Ha"]-meta["fci_energy"]),
                    SQD_minus_QSCI_mHa=1000*(sqd["energy_Ha"]-ci["energy_Ha"]), QSCI_d=ci["actual_dimension"], SQD_d=sqd["dimension"],
                    QSCI_correlation_recovered_percent=100*(meta["rhf_energy"]-ci["energy_Ha"])/denominator,
                    SQD_correlation_recovered_percent=100*(meta["rhf_energy"]-sqd["energy_Ha"])/denominator,
                    matched_d=md["d"], matched_SQD_minus_QSCI_mHa=1000*(md["SQD_energy_Ha"]-md["QSCI_energy_Ha"]),
                    same_sector_full_dimension=r["registered_sector_dimension"], selected_space_spans_full_sector=r["SQD_selected_space_spans_full_sector"],
                    raw_unique_ids=len(raw_ids), raw_valid_sector_fraction=sqd["raw_valid_sector_fraction"], recovery_new_ids_unique=len(recovered),
                    cartesian_new_ids_unique=len(cartesian), recovery_classical_calls=len(sqd["recovery_trace"]),
                    selected_solver_calls=len(sqd["solver_trace"]), SQD_iterations=sqd["iterations_used"],
                    SQD_max_iterations_reached=sqd["max_iterations_reached"], SQD_solver_wall_seconds=sqd["actual_wall_seconds"],
                    process_RSS_HWM_bytes=r["process_RSS_HWM_bytes"])
                key = (case, task["method"])
                if key not in cache: cache[key] = source_resources(cfg, *key)
                resources, p = cache[key]; provenance[str(key)] = p
                a.update(sampling_CNOT=resources["cnot_count"], sampling_CNOT_depth=resources["cnot_depth"], sampling_total_depth=resources["total_depth"],
                         sampling_CNOT_times_M=resources["cnot_count"]*cfg["M"], sampling_native_2q_count=resources.get("native_2q_count"))
            rows.append(a)
    db.close()
    frame = pd.DataFrame(rows); frame.to_csv(output/"all-planned-cells.csv", index=False)
    pd.DataFrame(diagnoses).to_csv(output/"failed-solver-selected-space-diagnostics.csv", index=False)
    save_json(output/"source-resource-cost-provenance.json", provenance)
    completed = frame[frame.status == "COMPLETED"].copy()
    by_case = completed.groupby(["case", "molecule", "distance_A", "method", "readout_flip_probability"], as_index=False).mean(numeric_only=True)
    by_case.to_csv(output/"all-case-means-available-with-missingness.csv", index=False)
    mol = by_case.groupby(["molecule", "method", "readout_flip_probability"], as_index=False).mean(numeric_only=True)
    mol.to_csv(output/"molecule-means-available-conditions.csv", index=False)
    paired_rows = []
    # Only identical case/repeat/noise cells in each two-method contrast.
    for baseline in ("B1:source_final_ceo_ovp", "B1:source_final_ceo_mvp", "B1:lucj_ccsd_2_vqe"):
        primary = completed[completed.method.str.startswith("RAACS:")]
        control = completed[completed.method == baseline]
        joined = primary.merge(control, on=["case", "rep", "readout_flip_probability", "molecule"], suffixes=("_RAACS", "_baseline"))
        joined["delta_mHa"] = joined.SQD_error_mHa_RAACS-joined.SQD_error_mHa_baseline
        for (molecule, noise), group in joined.groupby(["molecule", "readout_flip_probability"]):
            geometry_means = group.groupby("case").delta_mHa.mean()
            paired_rows.append({"molecule": molecule, "baseline": baseline, "readout_flip_probability": noise,
                "available_paired_processing_cells": len(group), "available_paired_geometries": len(geometry_means),
                "mean_RAACS_minus_baseline_full_SQD_mHa": float(geometry_means.mean()),
                "conditional_on_available_integrity_valid_pairs": True,
                "not_complete_case_success_only_population_claim": True})
    pairs = pd.DataFrame(paired_rows); pairs.to_csv(output/"RAACS-vs-strong-controls-paired-molecule-means.csv", index=False)
    figures = []
    for molecule in sorted(completed.molecule.unique()):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
        for method in cfg["methods"]:
            for noise, style in ((0., "-"), (.01, "--")):
                g = by_case[(by_case.molecule == molecule) & (by_case.method == method) & (by_case.readout_flip_probability == noise)].sort_values("distance_A")
                if g.empty: continue
                name = LABELS[method]+(" / readout1%" if noise else " / ideal")
                axes[0].plot(g.distance_A, g.SQD_error_mHa, style, color=COLORS[method], marker="o", markersize=3, label=name)
                axes[1].plot(g.distance_A, g.SQD_minus_QSCI_mHa, style, color=COLORS[method], marker="o", markersize=3)
                axes[2].plot(g.distance_A, g.matched_SQD_minus_QSCI_mHa, style, color=COLORS[method], marker="o", markersize=3)
        axes[0].set_ylabel("Full SQD error vs reporting-only FCI (mHa)"); axes[0].set_yscale("symlog", linthresh=.01)
        axes[1].set_ylabel("Full SQD minus QSCI, common R64 (mHa)")
        axes[2].set_ylabel("Full SQD minus QSCI, matched actual d (mHa)")
        for ax in axes:
            ax.set_xlabel("Distance / equal spacing (Å)"); ax.grid(alpha=.2); ax.axhline(0, linewidth=.7, color="gray")
        fig.suptitle(f"{molecule}: existing M1000, three paired raw repeats; failures remain missing")
        handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7)
        fig.tight_layout(rect=(0, .14, 1, .94)); name = "full-SQD-durable-"+molecule
        for fmt in ("png", "svg", "pdf"): fig.savefig(output/(name+"."+fmt), dpi=180, bbox_inches="tight")
        plt.close(fig); figures.append(name)
    note = f"""# Full fermionic SQD：全分子への拡張と独立監査

{len(frame)}予定セルを全件terminal化。完了{len(completed)}、元データ欠測{int((frame.status=='SOURCE_INITIALIZATION_MISSING').sum())}、
処理失敗{int((frame.status=='ENGINEERING_OR_RESOURCE_FAILURE_PRESERVED').sum())}。H2/H4/H6/LiH/BeH2/N2、各5距離。
各失敗と欠測はall-planned-cells.csvに残し、成功だけで全予定集合の優位性を主張しない。
公式SDKのfull loopを使用し、32セルquickで旧版とのenergy・選択空間・全iterationが一致。
失敗注入testでは既実行solver/recoveryの記録が別connectionから読み戻せた。

## 実行結果をどう読むか

比較するのは同じ既存raw M1000のQSCIとfull fermionic SQD。raw観測、古典recovery生成、
α×β Cartesian生成を分離。総R64と、各cellで小さい方の実dに揃えた診断を両方出力。
SQD側の実d診断は自身の最良batch係数で並べたもので、FCI係数は使わない。
実dを揃えても総古典作業や選択規則は同じにならない。
full loopの改善を『recoveryだけの因果効果』とは呼ばない。Cartesian空間の展開や
best-batch選択も同時に変わる。最良batchは独立な次回測定の結果でもない。

選択空間がsector全体を含んだセルは{validation['selected_spaces_spanning_full_sector']}件。
それは小さい物理sectorの完全対角化であり、量子優位や大規模性能の証拠ではない。
Nalpha/Nbetaを固定するがS²は強制していない。spin-squareをsolver traceに記録。

## 8セルの検査失敗を隠さない

伸長H6/N2で公式PySCF/Davidson出力が選択空間の最低Ritz値に一致しないケースを検出。
独立に保存済みselected-space全固有値へ照合し、差と近い固有値までの距離をCSVへ保存。
一部は高い根に非常に近く、ほかに非収束の可能性もある。初期guess/反復設定に関する
実装上の診断であり、SDK全般の不具合、CEOの性能低下、真の基底状態の不存在とは言わない。
これら8件の部分traceは保存され、黙って再試行したり成功に置換していない。

## 費用と限界

新規量子acquisition0・実機shots0。既存の{validation['unique_preexisting_acquisitions']}個の独立raw
acquisition（{validation['unique_preexisting_model_shots_referenced']} model shots）を参照。
noise armやrecovery iterationは独立な新規量子データではない。今回の古典処理時間と
元回路の構築費用は別記。samplerのCNOT/DepthはQSCI/SQDで同じで、古典後処理が
小さくした量子回路ではない。CNOT×MをQPU全費用と呼ばない。
FCI参照はこの外側レポートだけに使用し、新FCI計算なし。3 repeatから90%成功shotsを
捏造しない。欠測/失敗のため手法対比較は同じcase/rep/noiseのavailable pairsに限定。

PhaseF全体は未完了：gate depolarization、T1/T2、routing/native resourceが残る。
真の複数orbital-basis CI、大規模強baseline、最終10指標/18図/15質問の統合も残る。
"""
    (output/"REPORT_JA.md").write_text(note)
    result = {"status": "FULL_SQD_DURABLE_ALL_MOLECULE_REPORT_WRITTEN", "all_planned_cells": len(frame),
        "completed": len(completed), "figures": figures, "formats": ["png", "svg", "pdf"],
        "validation_sha256": file_digest(root/"validation.json"), "database_sha256": validation["database_sha256"],
        "reporter_sha256": file_digest(__file__), "posthoc_FCI_labels_only": True,
        "artifacts": {p.name: file_digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"},
        "whole_phase_F_complete": False, "whole_study_complete": False}
    save_json(output/"manifest.json", result); return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); a = p.parse_args(); print(json.dumps(report(a.output)))
