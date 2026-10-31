"""Versioned bounded-ground SQD execution/audit/report; parent outputs immutable."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
from unittest.mock import patch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyscf.fci.selected_ci as pyscf_sci
from .common import ROOT, digest, file_digest, save_json
from . import full_sqd_durable_cohort as parent
from . import full_sqd_durable_audit as parent_audit
from .fermionic_sqd_durable import decode, read_trace
from .bounded_selected_sci import projected_sqd
from .full_sqd_durable_report import LABELS, COLORS, source_resources

NAMESPACE = "full-fermionic-SQD-bounded-projected-ground-v1"
BASE_SETTINGS = parent.settings


def settings(mode):
    cfg = BASE_SETTINGS(mode)
    cfg.update(schema=NAMESPACE, mode=mode+"-projected", execution_mode=mode,
        selected_solver="official_PySCF_selected_action_bounded_projected_ground_v1",
        selected_dimension_cap=64,
        parent_scope="uniform development successor, not a new independent acquisition or eight-cell status patch",
        pyscf_selected_ci_source_sha256=file_digest(pyscf_sci.__file__),
        spec_sha256=file_digest(ROOT/"study-v1/FULL_SQD_PROJECTED_GROUND.md"))
    for p in ("bounded_selected_sci.py", "full_sqd_projected_study.py"):
        cfg["implementation"]["benchmark/"+p] = file_digest(ROOT/"benchmark"/p)
    return cfg


def run(root, mode):
    if mode == "standard":
        q = json.loads((ROOT/"results/full-fermionic-SQD-projected-quick-v1/selected-ground-validation.json").read_text())
        assert q["status"] == "BOUNDED_SELECTED_GROUND_ALL_EVENTS_VALIDATED" and q["quick_final_energy_parity_cells"] == 32
    # Explicit profile binding to immutable parent, confined to this one process.
    with patch.object(parent, "settings", settings), patch.object(parent, "NAMESPACE", NAMESPACE), patch.object(parent, "full_sqd_durable", projected_sqd):
        return parent.run(root, mode)


def audit(root):
    root = Path(root)
    with patch.object(parent_audit, "NAMESPACE", NAMESPACE):
        result = parent_audit.audit(root)
    cfg = json.loads((root/"manifest.json").read_text())
    assert file_digest(pyscf_sci.__file__) == cfg["pyscf_selected_ci_source_sha256"]
    db = sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro", uri=True)
    quick_legacy = {}
    if cfg["execution_mode"] == "quick":
        for p in (ROOT/"results/full-fermionic-SQD-quick-v1/cells").glob("*/*/result.json"):
            r = json.loads(p.read_text()); quick_legacy[digest(r["task"])] = r["SQD"]
    solve_count = products = parity = 0; maximum = 0; requests = 0
    for tid, raw in db.execute("SELECT id,result FROM cells WHERE status='COMPLETED'"):
        r = decode(raw); sqd = r["SQD"]; trace = read_trace(db, tid)
        req = [x["payload"] for x in trace if x["kind"] == "PROJECTED_GROUND_REQUEST"]
        res = [x["payload"] for x in trace if x["kind"] == "PROJECTED_GROUND_RESULT"]
        out = [x["payload"] for x in trace if x["kind"] == "SOLVE_RESULT"]
        assert len(req) == len(res) == len(out) == sqd["selected_solver_work"]["calls"]
        work = 0
        for q, s, o in zip(req, res, out):
            d = len(o["selected_ids"])
            assert 1 <= d == q["dimension"] == s["selected_dimension"] <= cfg["selected_dimension_cap"]
            assert q["planned_selected_H_vector_products"] == s["selected_H_vector_products"] == d
            assert abs(s["electronic_ground_energy_Ha"]+r["problem"]["physical_integral_binding"]["core"]-o["energy_Ha"]) < 1e-7
            assert s["ground_Ritz_residual"] <= 1e-8 and s["max_antihermitian_error"] <= 1e-9
            assert not s["full_Fock_matrix_or_vector"] and s["new_quantum_shots"] == 0
            work += d; maximum = max(maximum, d)
        assert work == sqd["selected_solver_work"]["selected_H_vector_products"]
        solve_count += len(res); products += work; requests += len(req)
        if quick_legacy:
            assert abs(sqd["energy_Ha"]-quick_legacy[digest(r["task"])]["energy_Ha"]) < 1e-7
            parity += 1
    db.close()
    out = {"status": "BOUNDED_SELECTED_GROUND_ALL_EVENTS_VALIDATED", "completed": result["completed"],
        "projected_ground_requests": requests, "projected_ground_results": solve_count, "selected_H_vector_products": products,
        "maximum_selected_dimension": maximum, "quick_final_energy_parity_cells": parity,
        "parent_validation_sha256": file_digest(root/"validation.json"), "auditor_adapter_sha256": file_digest(__file__),
        "database_sha256": result["database_sha256"], "new_quantum_acquisitions": 0, "hardware_shots": 0,
        "full_Fock_vector_or_matrix": False, "whole_study_complete": False}
    save_json(root/"selected-ground-validation.json", out); return out


def report(root):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text())
    validation = json.loads((root/"selected-ground-validation.json").read_text())
    parent_validation = json.loads((root/"validation.json").read_text())
    assert file_digest(root/"events.sqlite") == validation["database_sha256"]
    destination = root/"report-v1"; destination.mkdir(exist_ok=True)
    db = sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro", uri=True)
    oldroot = ROOT/"results/full-fermionic-SQD-durable-standard-v1"
    olddb = sqlite3.connect(f"file:{oldroot/'events.sqlite'}?mode=ro", uri=True)
    oldstatus = {digest(json.loads(t)): (status, decode(raw)) for t, status, raw in olddb.execute("SELECT task,status,result FROM cells")}; olddb.close()
    rows = []; old_failed = []; resource_cache = {}
    for tid, raw in db.execute("SELECT id,result FROM cells"):
        r = decode(raw); task = r["task"]
        meta = json.loads((Path(cfg["source"])/"references"/task["case"]/"reference.json").read_text())
        a = {"id": tid, **task, "status": r["status"], "molecule": meta["case"]["molecule"], "distance_A": meta["case"]["distance"],
            "classical_seconds_new_processing": r.get("cell_actual_classical_wall_seconds"), "new_quantum_acquisitions": 0,
            "QPU_construction_cost": None, "hardware_shots": 0}
        old = oldstatus.get(digest(task))
        if old: a["previous_durable_v1_status"] = old[0]
        if r["status"] == "COMPLETED":
            ci, sqd, md = r["QSCI"], r["SQD"], r["same_actual_d"]
            a.update(QSCI_error_mHa=1000*(ci["energy_Ha"]-meta["fci_energy"]), SQD_error_mHa=1000*(sqd["energy_Ha"]-meta["fci_energy"]),
                SQD_minus_QSCI_mHa=1000*(sqd["energy_Ha"]-ci["energy_Ha"]),
                matched_d=md["d"], matched_SQD_minus_QSCI_mHa=1000*(md["SQD_energy_Ha"]-md["QSCI_energy_Ha"]),
                QSCI_d=ci["actual_dimension"], SQD_d=sqd["dimension"],
                SQD_correlation_recovered_percent=100*(meta["rhf_energy"]-sqd["energy_Ha"])/(meta["rhf_energy"]-meta["fci_energy"]),
                selected_H_vector_products=sqd["selected_solver_work"]["selected_H_vector_products"],
                SQD_solver_calls=sqd["selected_solver_work"]["calls"], selected_space_full_sector=r["SQD_selected_space_spans_full_sector"],
                process_RSS_HWM_bytes=r["process_RSS_HWM_bytes"], SQD_iterations=sqd["iterations_used"])
            if old and old[0] == "COMPLETED":
                a["new_minus_old_SQD_mHa"] = 1000*(sqd["energy_Ha"]-old[1]["SQD"]["energy_Ha"])
            key = (task["case"], task["method"])
            if key not in resource_cache: resource_cache[key] = source_resources(cfg, *key)
            res, provenance = resource_cache[key]
            a.update(sampling_CNOT=res["cnot_count"], sampling_CNOT_depth=res["cnot_depth"],
                sampling_total_depth=res["total_depth"], source_construction_seconds_scope_specific=provenance["construction_seconds_scope_specific"])
        if old and old[0] == "ENGINEERING_OR_RESOURCE_FAILURE_PRESERVED": old_failed.append(a.copy())
        rows.append(a)
    db.close()
    frame = pd.DataFrame(rows); frame.to_csv(destination/"all-planned-cells.csv", index=False)
    pd.DataFrame(old_failed).to_csv(destination/"original-eight-failures-uniform-successor-status.csv", index=False)
    complete = frame[frame.status == "COMPLETED"]
    by_case = complete.groupby(["case", "molecule", "distance_A", "method", "readout_flip_probability"], as_index=False).mean(numeric_only=True)
    by_case.to_csv(destination/"case-means.csv", index=False)
    by_case.groupby(["molecule", "method", "readout_flip_probability"], as_index=False).mean(numeric_only=True).to_csv(destination/"molecule-means.csv", index=False)
    comparisons = []
    for baseline in cfg["methods"][:3]:
        a = complete[complete.method.str.startswith("RAACS:")]; b = complete[complete.method == baseline]
        joined = a.merge(b, on=["case", "rep", "readout_flip_probability", "molecule"], suffixes=("_RAACS", "_baseline"))
        joined["delta_mHa"] = joined.SQD_error_mHa_RAACS-joined.SQD_error_mHa_baseline
        for (molecule, noise), g in joined.groupby(["molecule", "readout_flip_probability"]):
            comparisons.append({"molecule": molecule, "baseline": baseline, "readout_flip_probability": noise,
                "paired_cells": len(g), "paired_geometries": g.case.nunique(),
                "mean_RAACS_minus_baseline_SQD_mHa": float(g.groupby("case").delta_mHa.mean().mean())})
    pd.DataFrame(comparisons).to_csv(destination/"paired-molecule-comparisons.csv", index=False)
    figures = []
    for molecule in sorted(complete.molecule.unique()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
        for method in cfg["methods"]:
            for noise, ls in ((0., "-"), (.01, "--")):
                g = by_case[(by_case.molecule == molecule) & (by_case.method == method) & (by_case.readout_flip_probability == noise)].sort_values("distance_A")
                if not len(g): continue
                axes[0].plot(g.distance_A, g.SQD_error_mHa, ls, marker="o", ms=3, color=COLORS[method], label=LABELS[method]+(" / readout1%" if noise else " / ideal"))
                axes[1].plot(g.distance_A, g.SQD_minus_QSCI_mHa, ls, marker="o", ms=3, color=COLORS[method])
        axes[0].set_ylabel("SQD error vs reporting-only FCI (mHa)"); axes[0].set_yscale("symlog", linthresh=.01)
        axes[1].set_ylabel("Full SQD minus explicit QSCI at R64 (mHa)")
        for ax in axes: ax.set_xlabel("Distance / equal spacing (Å)"); ax.grid(alpha=.2); ax.axhline(0, color="gray", lw=.7)
        fig.suptitle(molecule+": bounded selected-ground successor; reused M1000, 3 paired repeats")
        handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7)
        fig.tight_layout(rect=(0, .15, 1, .94)); name = "selected-ground-SQD-"+molecule
        for fmt in ("png", "svg", "pdf"): fig.savefig(destination/(name+"."+fmt), dpi=180, bbox_inches="tight")
        plt.close(fig); figures.append(name)
    note = f"""# Full SQD：選択空間の最低根を返す修正版

予定{len(frame)}セル、完了{len(complete)}、元の欠測{len(parent_validation['missing_preserved'])}、
新処理失敗{len(parent_validation['failures_preserved'])}。旧8失敗の今回の状態は専用CSVに全件保存。
旧v1の700成功/12欠測/8失敗は上書きしていない。今回は全手法・全セルへ一律の
solver変更を適用したdevelopment successorであり、8件だけの都合のよい救済や
新しい独立confirmatoryデータではない。

## 何を修正したか

公式SQDのrecovery/batch/iteration、PySCFの選択配置/RDMを保ち、SelectedCIの
固有値kernelだけを変更した。既に選択された最大64次元に対して各単位ベクトルへ
H*vを適用し、Hermitian射影行列の最低根を返す。初期guessが別の固有空間に入る
問題を回避する。これは大規模FCIでなく、選択CIの小行列対角化であり、その費用
{validation['selected_H_vector_products']}回のH*vを追加古典作業として記録。
全{validation['projected_ground_results']}solver結果を独立Slater–Condon行列と照合。
全Fock空間に対する基底状態保証や、大CI空間でのscalable solverとは主張しない。

同じM1000/R64/raw acquisitionを再利用。新しい量子acquisition0・実機shots0。
noise armsは同じ元標本に対する対応変換。FCI参照は外側の誤差表示だけで、選択に
渡していない。小さいselected spaceがsector全体へ届いた件数は
{parent_validation['selected_spaces_spanning_full_sector']}。その場合は数学的完全sector対角化。

## 残る解釈上の注意

explicit QSCIの任意64配置と、SQDのalpha8×beta8は同じ表現能力でない。
full SQDがQSCIの配置を落とすこともあり、recoveryの追加が必ず改善するとは限らない。
同じ実dへ切り詰めた診断も別列に保存し、総古典費用一致とは呼ばない。
3repeatだけからshots-to90%を保証しない。複数距離を独立な分子数と数えない。
源LUCJ欠測は保持し、available pairsのみを比較。QPU構築費用の未知はゼロにしない。

今回の修正はgate noise/T1T2/routing、真のmultiple-orbital-basis CI、LevelBの
残り強baselineや最終統合を完了させるものではない。研究全体は未完了。
"""
    (destination/"REPORT_JA.md").write_text(note)
    out = {"status": "SELECTED_GROUND_SQD_REPORT_WRITTEN", "planned": len(frame), "completed": len(complete),
        "original_failed_tasks": len(old_failed), "original_failed_tasks_now_completed": sum(x["status"] == "COMPLETED" for x in old_failed),
        "figures": figures, "formats": ["png", "svg", "pdf"], "reporter_sha256": file_digest(__file__),
        "database_sha256": validation["database_sha256"],
        "artifacts": {p.name: file_digest(p) for p in destination.iterdir() if p.is_file() and p.name != "manifest.json"},
        "whole_study_complete": False}
    save_json(destination/"manifest.json", out); return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="quick")
    p.add_argument("--stage", choices=["run", "audit", "report", "all"], default="all"); p.add_argument("--output")
    a = p.parse_args(); root = Path(a.output) if a.output else ROOT/f"results/full-fermionic-SQD-projected-{a.mode}-v1"
    with threadpool_limits(limits=1):
        for name, function in (("run", lambda: run(root, a.mode)), ("audit", lambda: audit(root)), ("report", lambda: report(root))):
            if a.stage in ("all", name): print(json.dumps({"stage": name, "result": function()}), flush=True)
