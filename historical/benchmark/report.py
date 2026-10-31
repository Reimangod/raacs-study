from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def make_report(root):
    root=Path(root)
    from .analysis import analyze
    audit=analyze(root)
    cps=[json.loads(p.read_text()) for p in (root/"checkpoints").glob("*/*/step-*.json")]
    native_path=root/"native_resources/ecr-line/summary.json"
    native={c["checkpoint_id"]:{"native_2q_count":c["native_2q_count"],"native_2q_depth":c["native_2q_depth"],
                                "native_total_depth":c["native_total_depth"],"native_target":c["target"],
                                "native_status":"SYNTHETIC_PROFILE_NOT_HARDWARE"}
            for c in json.loads(native_path.read_text())} if native_path.exists() else {}
    flat=[{"case_id":c["case_id"],"method":c["method"],"step":c["step"],"checkpoint_id":c["checkpoint_id"],
           "vqe_energy":c["vqe_energy"],"vqe_error_mHa":1000*c["vqe_energy_error"],
           "resource_scope":"Z-sampling-equivalent circuit; VQE energy belongs to preparation",
           **{"preparation_"+k:c["preparation_resources"][k] for k in ("cnot_count","cnot_depth","total_depth")},
           **{k:v for k,v in c["resources"].items() if k!="compiler"},
           **native.get(c["checkpoint_id"],{}),
           "objective_evaluations":c["objective_evaluations"],"optimizer_seconds":c["optimization_seconds"]}
          for c in cps]
    table=pd.DataFrame(flat)
    table.to_csv(root/"summary.csv",index=False)
    table.to_parquet(root/"summary.parquet",index=False)
    qsci=[row for p in (root/"qsci").glob("*/summary.json") for row in json.loads(p.read_text())]
    if qsci:
        pd.DataFrame(qsci).to_csv(root/"qsci-summary.csv",index=False)
        pd.DataFrame(qsci).to_parquet(root/"qsci-summary.parquet",index=False)
    lines=["# CEO sampler benchmark — current evidence", "",
           "This is an automatically generated progress report, not a presupposed CEO superiority claim.",
           "",f"Saved checkpoints: {len(cps)}. Cases with checkpoints: {table.case_id.nunique() if len(table) else 0}/30.",
           "", "## Final saved checkpoints", "",
           "Preparation VQE energy and Z-sampler resources are intentionally distinct: terminal diagonal-gate removal preserves Z probabilities, not the molecular VQE state. Do not interpret this table as a same-state VQE compression claim.","",
           "| Case | Method | Step | Preparation VQE error mHa | Z-sampler CNOT | Z-sampler CNOT depth | Registered parameters |",
           "|---|---|---:|---:|---:|---:|---:|"]
    if len(table):
        for _,r in table.sort_values("step").groupby(["case_id","method"]).tail(1).sort_values(["case_id","method"]).iterrows():
            lines.append(f"| {r.case_id} | {r.method} | {r.step} | {r.vqe_error_mHa:.6f} | {r.cnot_count} | {r.cnot_depth} | {r.parameter_count} |")
    utilization=root/"gpu-production-v1/observed-utilization.json"
    if utilization.exists():
        hardware=json.loads(utilization.read_text())
        lines += ["","## Observed compute use","",
                  f"Primary N2 geometries used actual A100 numerical kernels. Configured concurrency2; observed primary concurrency{hardware['observed_max_concurrent_primary_A100']}. A pre-existing unrelated job consumed one of the normal-QOS two-GPU user quota; it was not stopped. Two independent full64-step qualification trajectories did overlap on two A100 allocations. Chemistry, LUCJ, QSCI/CI, recovery and rendering stayed CPU-side. No QPU or hardware shots."]
    lines += ["", "## Status and missingness", ""]
    request=json.loads((root/"request.json").read_text())
    methods=request["options"]["methods"]
    status_rows=[]
    for case in request.get("cohort_cases",request["cases"]):
        name=f"{case['molecule']}_R{case['distance']:.1f}"
        for method in methods:
            path=root/"checkpoints"/name/method/"status.json"
            failure=root/"failures"/name/f"{method}.json"
            status=json.loads(path.read_text())["status"] if path.exists() else ("FAILED_BEFORE_CHECKPOINT" if failure.exists() else "NOT_STARTED")
            status_rows.append({"case_id":name,"method":method,"status":status,
                                "failure":json.loads(failure.read_text()).get("error") if failure.exists() else None,
                                "failure_is_historical_superseded":bool(failure.exists() and path.exists() and status not in ("FAILED","RUNNING"))})
            lines.append(f"- {name}/{method}: {status}")
    pd.DataFrame(status_rows).to_csv(root/"all-cell-status.csv",index=False)
    from .common import save_json
    save_json(root/"all-cell-status.json",status_rows)
    comparisons=pd.DataFrame(json.loads((root/"analysis/resource_constrained_comparisons.json").read_text()))
    optima=pd.DataFrame(json.loads((root/"analysis/checkpoint_optima.json").read_text()))
    costs=pd.DataFrame(json.loads((root/"analysis/construction_costs.json").read_text()))
    lines += ["", "## Eight research questions — current quantitative evidence", "",
              "All checkpoint optima below are retrospective. Finite-QSCI choices use the same evaluation draws, not independent policy validation. FCI-weight choices are ORACLE diagnostics. Geometry points are not independent molecules.","",
              "### 1. Important non-HF configuration discovery", "",
              "See analysis/analytic_metrics.csv for every step, all four shot settings and all four top-K values. analysis/checkpoint_optima.csv criterion D reports the FCI-oracle non-HF coverage/CNOT optimum; it is not a learned or executable online selection policy.","",
              "### 2. QSCI-optimal versus final VQE checkpoint", ""]
    metric_lookup={(c["case_id"],c["method"],c["step"],m["shots"]):m for c in cps for m in c["analytic_metrics"]}
    coverage=[]
    for p in comparisons.to_dict("records"):
        a=metric_lookup[(p["case_id"],p["ceo"],p["ceo_step"],p["shots"])]["nonhf_weighted_coverage"]
        b=metric_lookup[(p["case_id"],p["baseline"],p["baseline_step"],p["shots"])]["nonhf_weighted_coverage"]
        coverage.append({**p,"delta_expected_nonhf_coverage":None if a is None or b is None else a-b})
    if coverage:
        pd.DataFrame(coverage).to_csv(root/"analysis/coverage_at_qsci_selected_points.csv",index=False)
        pd.DataFrame(coverage).to_parquet(root/"analysis/coverage_at_qsci_selected_points.parquet",index=False)
        marker=lines.index("### 2. QSCI-optimal versus final VQE checkpoint")
        extra=["At the SAME retrospective resource-constrained QSCI comparison points (not a new oracle coverage optimum), M=1000:","",
               "| Baseline | CEO | Observed pairs | Median expected non-HF coverage difference |","|---|---|---:|---:|"]
        for (baseline,ceo),g in pd.DataFrame(coverage).query("shots == 1000").groupby(["baseline","ceo"]):
            extra.append(f"| {baseline} | {ceo} | {len(g)} | {g.delta_expected_nonhf_coverage.median():.8g} |")
        lines[marker:marker]=extra+[""]
    early=total=0
    if len(optima):
        for key,g in optima[(optima.shots==1000)&optima.method.isin(["ceo_ovp","ceo_mvp","qeb","qubit","gsd"])].groupby(["case_id","method"]):
            c=g[g.criterion=="C_min_qsci_error"];e=g[g.criterion=="E_final_saved_vqe_checkpoint"]
            if len(c) and len(e) and c.iloc[0].attained and e.iloc[0].attained:
                total+=1;early+=int(c.iloc[0].step<e.iloc[0].step)
    lines.append(f"At M=1000, the empirical best-QSCI checkpoint was earlier than the final saved checkpoint in {early}/{total} observed case-pool pairs. Ties are broken by CNOT then step; this can reflect a readout plateau, not a mechanism or a free stopping rule.")
    lines += ["", "### 3. CEO versus LUCJ, QEB, qubit and GSD", "",
              "Same M and R cap. The CEO choice is constrained to no more CNOT AND CNOT depth than the comparator. Realized d is recorded and may differ. Counts below are observed-mean comparisons, not statistical dominance.","",
              "| Baseline | CEO | Lower error | Numerical tie | Higher error | Observed feasible cases |",
              "|---|---|---:|---:|---:|---:|"]
    if len(comparisons):
        for (baseline,ceo),g in comparisons[comparisons.shots==1000].groupby(["baseline","ceo"]):
            delta=g.delta_error_mHa
            lines.append(f"| {baseline} | {ceo} | {sum(delta < -1e-5)} | {sum(abs(delta)<=1e-5)} | {sum(delta>1e-5)} | {len(g)} |")
    lines += ["", "### 4. CNOT/depth and shots to chemical accuracy", "",
              "analysis/checkpoint_optima.csv A/B contains minimum CNOT/depth checkpoints meeting the observed mean-error threshold, with success rate and actual d. analysis/shots_to_chemical_accuracy.csv records the smallest TESTED shot grid value and explicit non-attainment. This is not a continuous shot threshold or confidence guarantee.","",
              "### 5. Distance and HF-weight dependence", "",
              f"{audit['case_count']}/30 conditions have saved checkpoints. references/*/reference.json contains HF weight and correlation energy. Cross-case heatmaps retain unmeasured cells as missing; correlation_dependence shows continuous reference diagnostics, not a universal weak/strong-correlation classification. No significance claim pools geometries as independent molecules.","",
              "### 6. Construction-inclusive cost", "",
              f"Saved optimizer objective calls: {audit['construction_objective_calls']}. Per-method optimizer seconds, gradient-vector calls and CCSD initialization are in analysis/construction_costs.csv. They are classical simulation work, not actual quantum measurement shots. A single common end-to-end quantum-shot superiority conclusion is therefore NOT_ESTABLISHED.","",
              "### 7. Does the result survive official SQD recovery?", ""]
    if len(optima):
        marker=lines.index("### 5. Distance and HF-weight dependence")
        extra=["Observed mean chemical-accuracy attainment at ANY saved checkpoint, M=1000, R cap32:","",
               "| Method | Attained | Available conditions | Planned conditions |","|---|---:|---:|---:|"]
        selected=optima[(optima.shots==1000)&(optima.criterion=="A_min_cnot_mean_chemical")]
        for method,g in selected.groupby("method"):
            extra.append(f"| {method} | {sum(g.attained)} | {len(g)} | 30 |")
        lines[marker:marker]=extra+[""]
    sqd_path=root/"sqd/summary.json"
    if sqd_path.exists():
        sqd=pd.DataFrame(json.loads(sqd_path.read_text()))
        lines += [f"Official recovery readouts: {len(sqd)}. Same original raw counts, same HF occupancy initialization, last iteration rather than best iteration; extra quantum shots=0.","",
                  "| Method | M | Observed readouts | Mean recovery energy change (mHa) | Mean valid raw rate |",
                  "|---|---:|---:|---:|---:|"]
        if len(sqd):
            for (method,shots),g in sqd.groupby(["method","shots"]):
                delta=1000*(g.energy_with_recovery-g.energy_without_recovery)
                lines.append(f"| {method} | {shots} | {len(g)} | {delta.mean():.8g} | {g.valid_raw_rate.mean():.6f} |")
        lines.append("These pooled method summaries are descriptive; per-case records retain every result. This is explicit-determinant qubit-H SQD recovery, not full alpha-beta Cartesian fermionic SQD. No hardware-noise advantage follows from noiseless data.")
    else:
        lines.append("NOT_YET_EVALUATED in this result directory. Installed dependencies alone are not execution evidence.")
    selected_path=root/"sqd-selected-v1/pair-summary.json"
    if selected_path.exists():
        selected=pd.DataFrame(json.loads(selected_path.read_text()))
        lines += ["","The previously selected early-QSCI comparison points also receive the SAME official recovery, reusing existing counts and any identical final-readout recovery. These are the same retrospective pairs, not independent confirmation. M=1000:","",
                  "| Baseline | CEO | Pairs | Median delta before (mHa) | Median delta after (mHa) | Pairs with equal d in all20 draws after |",
                  "|---|---|---:|---:|---:|---:|"]
        if len(selected):
            for (baseline,ceo),g in selected[selected.shots==1000].groupby(["baseline","ceo"]):
                lines.append(f"| {baseline} | {ceo} | {len(g)} | {g.delta_error_mHa.median():.8g} | {g.delta_after_recovery_mHa.median():.8g} | {sum(g.equal_realized_d_repeats_after==20)} |")
    matched=root/"matched-dimension/summary.json"
    if matched.exists():
        data=pd.DataFrame(json.loads(matched.read_text()))
        lines += ["","### Exact-realized-d secondary comparisons","",
                  "The two readouts reuse their original raw counts and truncate both sets to the smaller realized dimension. This removes a CI-size difference, NOT retrospective selection bias. Negative delta favors CEO. No new shots.","",
                  "| Baseline | CEO | M | Pairs | Median matched-d delta error (mHa) |",
                  "|---|---|---:|---:|---:|"]
        if len(data):
            for (b,c,m),g in data.groupby(["baseline","ceo","shots"]):
                lines.append(f"| {b} | {c} | {m} | {len(g)} | {g.matched_d_delta_error_mHa.median():.8g} |")
    matched_recovery=root/"sqd-matched-dimension/summary.json"
    if matched_recovery.exists():
        data=pd.DataFrame(json.loads(matched_recovery.read_text()))
        lines += ["","### Four-way identical final CI dimension, before/after recovery","",
                  "Every replicate uses min(dCEO-before,dBase-before,dCEO-after,dBase-after), retaining the original HF/frequency ordering. This avoids conditioning on only the favorable equal-d subset. Recovery's intermediate occupancy updates still used the common registered R cap32; final-d matching is NOT a matched-total-classical-work claim. Same retrospective pairs and counts; no new shots. M=1000:","",
                  "| Baseline | CEO | Pairs | Median delta before (mHa) | Median delta after (mHa) |",
                  "|---|---|---:|---:|---:|"]
        for (b,c),g in data[data.shots==1000].groupby(["baseline","ceo"]):
            lines.append(f"| {b} | {c} | {len(g)} | {g.matched_delta_before_mHa.median():.8g} | {g.matched_delta_after_mHa.median():.8g} |")
    noisy=[r for p in (root/"readout-noise-v1").glob("*/*.json") for r in json.loads(p.read_text())["rows"]]
    if noisy:
        data=pd.DataFrame(noisy)
        lines += ["","### Secondary synthetic readout errors","",
                  "Independent symmetric1% bit-flip readout error, identical for every arm. NOT gate noise, Aer, hardware calibration or hardware shots. Official recovery uses the same noisy counts and HF occupancy initialization. Recovery can create configurations classically, so better energies alone do not prove a quantum sampler advantage; HF/random controls remain in the dataset.","",
                  "| Method | M | Readouts | Mean valid raw fraction | Median error before (mHa) | Median error after (mHa) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for (method,m),g in data.groupby(["method","shots"]):
            lines.append(f"| {method} | {m} | {len(g)} | {g.valid_raw_rate.mean():.6f} | {g.error_before_mHa.median():.8g} | {g.error_after_mHa.median():.8g} |")
        lines.append("Descriptive medians across current cases, not independent molecule-level inference. Before/after CI dimensions may differ; all per-readout dimensions and failure/missing conditions are retained.")
    lines += ["", "### 8. Next sampling-aware CEO change", "",
              "Use the observed gap between coverage, QSCI energy and final VQE quality to choose at most one follow-up objective. Do not infer that lower VQE energy, more entropy or more unique configurations alone is better. Any new selection policy needs separate selection/evaluation seeds, common resource/readout budgets and all exploration cost recorded. No new policy superiority is inferred from retrospective optima.",
              "", "## Limits", "",
              "Common full-ansatz BFGS tests pool effects, not full CEO* HR/OGM/TETRIS reproduction. Fixed1/2-layer CCSD initialization does not represent all optimized or deeper LUCJ protocols. Native device resources are NOT_CONFIGURED unless a derived target profile is supplied; the supplied ECR-line profile is synthetic, not calibrated hardware. Analytic FCI-weight coverage is an expectation/oracle diagnostic, not measured QSCI energy. MAX_STEPS and TIME_CENSORED are not convergence. Missing comparisons are not zero effects. Input VQE energy belongs to the preparation state; QSCI resources belong to its exactly Z-distribution-equivalent sampling circuit.",
              "The common primary R=32 cap can impose a selected-CI capacity floor even when the preparation VQE energy improves. Failure to reach chemical accuracy under this cap is not proof that the ansatz cannot sample a useful larger subspace. CCSD is shared across variants: the per-method initialization field is a standalone proxy and must not be summed as actual newly executed work. Historical failed initialization runtimes without timing evidence remain unknown, not zero."]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    return len(cps)


def main():
    import sys
    if '--scalable-study' in sys.argv and '--current-evidence' in sys.argv:
        from .current_evidence import main as current_main
        old=sys.argv;sys.argv=[x for x in old if x not in ('--scalable-study','--current-evidence')]
        try:return current_main()
        finally:sys.argv=old
    if '--scalable-study' in sys.argv:
        from .research_synthesis_v2 import main as synthesis_main
        old=sys.argv;sys.argv=[x for x in old if x!='--scalable-study']
        try:return synthesis_main()
        finally:sys.argv=old
    p=argparse.ArgumentParser();p.add_argument("--output",default="results/quick")
    output=p.parse_args().output
    manifest=Path(output)/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("schema")=="sampler-study-A-v1":
        from .study_report import make_report as study_report
        return study_report(output)
    make_report(output)


if __name__=="__main__":
    main()
