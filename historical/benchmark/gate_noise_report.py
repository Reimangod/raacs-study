"""Post-audit molecular noise/resource/cost report. FCI is reporting-only."""
import argparse
import json
from math import comb
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import file_digest, save_json
from .gate_noise_study import unpack
from .full_sqd_durable_report import LABELS, COLORS, source_resources
from .raacs_independent_report import wilson


def report(root):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text()); valid = json.loads((root/"validation.json").read_text())
    assert valid["status"] == "GATE_NOISE_COUNTS_RESOURCES_AND_SELECTED_SPACES_VALIDATED"
    assert file_digest(root/"events.sqlite") == valid["database_sha256"]
    output = root/"report-v1"; output.mkdir(exist_ok=True)
    db = sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro", uri=True)
    circuits = {cid: (case, method, target, available, unpack(raw)) for cid, case, method, target, available, raw in db.execute("SELECT id,case_id,method,target,status,record FROM circuits")}
    rows = []; source_costs = {}; compiled = []
    for cid, (case, method, target, available, record) in circuits.items():
        a = {"circuit": cid, "case": case, "method": method, "target": target, "status": available}
        if available == "AVAILABLE":
            a.update(record["resources"]); a["compile_and_parity_seconds"] = record["compile_and_independent_parity_seconds"]
            key = (case, method)
            if key not in source_costs:
                # Comparator origin includes the immutable comparison cohort path.
                inherited_cfg = {"comparison": str(Path(cfg["source"]).parent/"RAACS-independent-comparison-standard-sharded-v1")}
                resource, cost = source_resources(inherited_cfg, case, method)
                source_costs[key] = {"case": case, "method": method, **cost, "original_logical_resources": resource}
        compiled.append(a)
    for tid, raw_task, terminal, raw in db.execute("SELECT * FROM tasks"):
        task = json.loads(raw_task); r = unpack(raw)
        meta = json.loads((Path(cfg["source"])/"references"/task["case"]/"reference.json").read_text())
        a = {"id": tid, **task, "status": terminal, "molecule": meta["case"]["molecule"], "distance_A": meta["case"]["distance"],
            "known_committed_model_shots": r["known_committed_model_shots"], "hardware_shots": 0,
            "QPU_construction_cost": None, "error": r.get("error")}
        if "simulation_runtime" in r: a.update(r["simulation_runtime"])
        record = circuits[task["circuit"]][4]
        if "resources" in record:
            res = record["resources"]
            a.update(native_CX_count=res["native_2q_count"], native_CX_depth=res["native_2q_depth"],
                native_total_gate_depth=res["native_total_gate_depth_without_measurement"],
                inserted_swaps=res["inserted_explicit_swaps_before_decomposition"],
                model_CX_times_shots=res["native_2q_count"]*r["known_committed_model_shots"])
        if "QSCI" in r:
            ci = r["QSCI"]; a.update(QSCI_energy_Ha=ci["energy_Ha"], QSCI_error_mHa=1000*(ci["energy_Ha"]-meta["fci_energy"]),
                QSCI_d=ci["actual_dimension"], QSCI_seconds=r["QSCI_seconds"])
        if terminal == "COMPLETED":
            sqd = r["SQD"]; md = r["same_actual_d"]
            a.update(SQD_energy_Ha=sqd["energy_Ha"], SQD_error_mHa=1000*(sqd["energy_Ha"]-meta["fci_energy"]), SQD_d=sqd["dimension"],
                SQD_minus_QSCI_mHa=1000*(sqd["energy_Ha"]-r["QSCI"]["energy_Ha"]),
                matched_d=md["d"], matched_SQD_minus_QSCI_mHa=1000*(md["SQD_energy_Ha"]-md["QSCI_energy_Ha"]),
                raw_valid_sector_fraction=sqd["raw_valid_sector_fraction"], SQD_classical_seconds=sqd["actual_wall_seconds"],
                recovery_calls=len(sqd["recovery_trace"]), selected_solver_calls=len(sqd["solver_trace"]),
                selected_H_vector_products=sqd["selected_solver_work"]["selected_H_vector_products"],
                SQD_selected_space_spans_full_sector=sqd["dimension"] == comb(meta["n_orbitals"], meta["nalpha"])*comb(meta["n_orbitals"], meta["nbeta"]),
                recovery_new_ids_unique=len(set(x for p in sqd["recovery_trace"] for x in p["new_recovered_ids_not_raw_observed"])),
                Cartesian_new_ids_unique=len(set(x for p in sqd["solver_trace"] for x in p["cartesian_ids_not_raw_observed"])))
            denominator = meta["rhf_energy"]-meta["fci_energy"]
            for name in ("QSCI", "SQD"):
                a[name+"_correlation_recovered_percent"] = 100*(meta["rhf_energy"]-a[name+"_energy_Ha"])/denominator
            error_before = a["QSCI_error_mHa"]
            a["full_SQD_error_improvement_percent"] = 100*(error_before-a["SQD_error_mHa"])/error_before if abs(error_before) > 1e-7 else None
            ep = r["explicit_last_fullSQD_recovery_projection"]
            if ep is not None:
                a.update(explicit_recovery_projection_error_mHa=1000*(ep["energy_Ha"]-meta["fci_energy"]), explicit_recovery_projection_d=ep["dimension"])
        rows.append(a)
    db.close()
    frame = pd.DataFrame(rows); frame.to_csv(output/"all-planned-cells.csv", index=False)
    pd.DataFrame(compiled).to_csv(output/"native-circuit-resources-once.csv", index=False)
    save_json(output/"inherited-construction-costs-once.json", list(source_costs.values()))
    done = frame[frame.status == "COMPLETED"].copy()
    groupkeys = ["case", "molecule", "distance_A", "method", "target", "profile"]
    means = done.groupby(groupkeys, as_index=False).mean(numeric_only=True)
    means.to_csv(output/"case-noise-target-means.csv", index=False)
    molecule = means.groupby(["molecule", "method", "target", "profile"], as_index=False).mean(numeric_only=True)
    molecule.to_csv(output/"molecule-means.csv", index=False)
    success = []
    for key, g in done.groupby(groupkeys):
        for energy in ("QSCI", "SQD"):
            n = len(g); k = int((g[energy+"_error_mHa"] <= 1.5936014376).sum()); low, high = wilson(k, n)
            success.append({**dict(zip(groupkeys, key)), "readout": energy, "successes": k, "trials": n,
                "observed_success_fraction": k/n, "Wilson90_low": low, "Wilson90_high": high,
                "shots_to90": None, "shots_to90_status": "NOT_IDENTIFIABLE_SINGLE_M_GRID_AND_THREE_REPEATS"})
    pd.DataFrame(success).to_csv(output/"success-fractions-not-shots90-certificates.csv", index=False)
    contrasts = []
    for baseline in cfg["methods"][:3]:
        a, b = done[done.method.str.startswith("RAACS:")], done[done.method == baseline]
        paired = a.merge(b, on=["case", "seed", "profile", "target", "molecule"], suffixes=("_RAACS", "_baseline"))
        paired["delta_SQD_mHa"] = paired.SQD_error_mHa_RAACS-paired.SQD_error_mHa_baseline
        paired["delta_QSCI_mHa"] = paired.QSCI_error_mHa_RAACS-paired.QSCI_error_mHa_baseline
        for (mol, profile, target), g in paired.groupby(["molecule", "profile", "target"]):
            case_means = g.groupby("case")[["delta_SQD_mHa", "delta_QSCI_mHa"]].mean()
            contrasts.append({"molecule": mol, "baseline": baseline, "profile": profile, "target": target,
                "paired_completed_cells": len(g), "paired_available_geometries": len(case_means),
                "RAACS_minus_baseline_SQD_mHa": float(case_means.delta_SQD_mHa.mean()),
                "RAACS_minus_baseline_QSCI_mHa": float(case_means.delta_QSCI_mHa.mean()),
                "noise_realizations_not_shared": True, "conditional_available_pairs_missingness_visible_in_full_table": True})
    pd.DataFrame(contrasts).to_csv(output/"paired-molecule-contrasts.csv", index=False)
    figures = []
    for mol in sorted(done.molecule.unique()):
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True)
        for i, target in enumerate(cfg["targets"]):
            for j, profile in enumerate(cfg["profiles"]):
                ax = axes[i, j]
                for method in cfg["methods"]:
                    g = means[(means.molecule == mol) & (means.method == method) & (means.target == target) & (means.profile == profile)].sort_values("distance_A")
                    if len(g): ax.plot(g.distance_A, g.SQD_error_mHa, "o-", ms=3, color=COLORS[method], label=LABELS[method])
                ax.set_title(target+" / "+profile, fontsize=10); ax.set_xlabel("Distance (Å)"); ax.set_ylabel("Full SQD error (mHa)")
                ax.set_yscale("symlog", linthresh=.01); ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=4)
        fig.suptitle(mol+": synthetic gate noise, fixed final samplers; M250/R64, missing/capped cells retained")
        fig.tight_layout(rect=(0, .06, 1, .95)); name = "gate-noise-"+mol
        for fmt in ("png", "svg", "pdf"): fig.savefig(output/(name+"."+fmt), dpi=160, bbox_inches="tight")
        plt.close(fig); figures.append(name)
    text = f"""# Phase F：合成ゲートノイズ・配線制約の本比較

全予定{len(frame)}セル、SQDまで完了{len(done)}。状態の全内訳はall-planned-cells.csv。
新しく保存したモデルshotsは{valid['known_committed_new_model_shots']}、実機shots0。
同じ元の最終回路、M250/R64でnoise/targetを比較し、最良checkpointを選び直していない。
失敗/cap/元CCSD欠測を保持。条件別の成功率の分母はCSVで明示し、成功例だけの
available-pair比較を全予定集合の優位性とは呼ばない。

## 比較の範囲

all-to-all/lineのsynthetic nativeCX target、ideal/readout/depolarizing2q/
gate-localT1T2/combined。実機較正値でなく、idleやmeasurement relaxationは含まない。
測定bitはrouting前に固定し、SWAP前後の論理配置を独立検証した。旧ECR資源とは別規約。
QSCI/fullSQD/同じ実d/最後のrecovery分布の明示projectionを区別した。後者は
独立なstandalone回復法ではない。raw観測配置と古典recovery/Cartesian追加を分離。
任意R64とalpha8×beta8は同じ表現能力でなく、fullSQDの改善を当然視しない。
小さな完全sectorの対角化も明記し、量子優位性とは呼ばない。

親samplerの構築費用は共有元ごとに一度だけ別保存。今回のcompile/physical-noise
simulation/CI/SQD時間は新しい古典作業。CNOT×shotsは実行量proxyでQPU全費用でない。
FCIはこの外側表示だけ、新FCI計算なし。noise armsは独立RNGで同一noise実現ではない。
分子内geometry/seedを独立な分子と数えない。単一M/3seedから90%成功shotsを捏造しない。

監査は全chunk/seed/整数counts/資源/選択空間とrecoveryを照合し、各分子で事前指定の
1chunkだけをAerで決定論的再実行。その他全drawを再実行したとは主張しない。
決定論的再実行は新しい独立acquisitionでなく、追加監査CPU作業として別記。
完全な研究目標にはmulti-orbital-basis CI、大規模強baseline、最終統合などが残る。
"""
    (output/"REPORT_JA.md").write_text(text)
    manifest = {"status": "GATE_NOISE_FULL_COHORT_REPORTED", "planned": len(frame), "completed": len(done),
        "figures": figures, "formats": ["png", "svg", "pdf"], "database_sha256": valid["database_sha256"],
        "reporter_sha256": file_digest(__file__), "validation_sha256": file_digest(root/"validation.json"),
        "artifacts": {p.name: file_digest(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"},
        "whole_study_complete": False}
    save_json(output/"manifest.json", manifest); return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); a = p.parse_args(); print(json.dumps(report(a.output)))
