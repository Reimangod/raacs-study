"""Declared retrospective checkpoint optima, resource frontiers and all-case analyses.

No selection here changes/trains a sampler. FCI-weight and same-sample finite-QSCI
choices are labelled as oracle/retrospective diagnostics, not deployable policies.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import save_json
from .qsci import CHEMICAL_ACCURACY_HA

ADAPT_METHODS=("ceo_ovp","ceo_mvp","qeb","qubit","gsd")


def finite_or_none(value):
    if isinstance(value,dict):
        return {str(k):finite_or_none(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [finite_or_none(x) for x in value]
    if isinstance(value,np.generic):
        return finite_or_none(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def nondominated_mask(frame,columns):
    values=frame[columns].to_numpy(dtype=float)
    # Numerical Ritz/FCI roundoff must not manufacture a resource tradeoff.
    # This is not a statistical equivalence margin. Raw energies are retained.
    tolerance=np.array([1e-5 if c=="mean_energy_error_mHa" else 0. for c in columns])
    mask=np.ones(len(frame),dtype=bool)
    for i,row in enumerate(values):
        # Minimize all objectives; exact duplicates are retained, not invented wins.
        mask[i]=not np.any(np.all(values<=row+tolerance,axis=1)&np.any(values<row-tolerance,axis=1))
    return mask


def table_export(directory,name,rows):
    directory.mkdir(parents=True,exist_ok=True)
    clean=finite_or_none(rows)
    save_json(directory/f"{name}.json",clean)
    if clean:
        df=pd.DataFrame(clean)
        df.to_csv(directory/f"{name}.csv",index=False)
        df.to_parquet(directory/f"{name}.parquet",index=False)


def analyze(root):
    root=Path(root);out=root/"analysis"
    checkpoints=[json.loads(p.read_text()) for p in (root/"checkpoints").glob("*/*/step-*.json")]
    summaries=[x for p in (root/"qsci").glob("*/summary.json") for x in json.loads(p.read_text())]
    if not checkpoints:
        return {"checkpoints":0,"finite_groups":0,"conclusions":[]}
    by_id={c["checkpoint_id"]:c for c in checkpoints}
    lookup={(c["case_id"],c["method"],c["step"]):c for c in checkpoints}
    q=pd.DataFrame(summaries)
    archived_fixed_groups=0
    if len(q):
        latest={key:max(c["step"] for c in items) for key,items in _checkpoint_groups(checkpoints)}
        keep=[r.method in ADAPT_METHODS or r.step==latest[(r.case_id,r.method)] for r in q.itertuples()]
        archived_fixed_groups=len(q)-sum(keep);q=q[keep].copy()
    optima=[];costs=[];analytic=[]
    for (case_id,method),items in _checkpoint_groups(checkpoints):
        items.sort(key=lambda c:c["step"])
        final=items[-1]
        costs.append({"case_id":case_id,"method":method,"checkpoint_count":len(items),
                      "last_step":final["step"],"final_vqe_error_mHa":1000*final["vqe_energy_error"],
                      "objective_calls":sum(c.get("objective_evaluations",0) for c in items),
                      "gradient_vector_calls":sum(c.get("gradient_evaluations",0) for c in items),
                      "optimizer_seconds":sum(c.get("optimization_seconds",0) for c in items),
                      "ccsd_initialization_seconds":final.get("ccsd_initialization_seconds",0),
                      "numerical_backend":final.get("execution_backend","numpy/scipy CPU"),
                      "device":final.get("execution_device","CPU"),
                      "construction_cost_units":"classical simulation calls/time; quantum-shots conversion NOT_ESTABLISHED"})
        for c in items:
            for row in c["analytic_metrics"]:
                analytic.append({"case_id":case_id,"method":method,"step":c["step"],
                                 "cnot_count":c["resources"]["cnot_count"],
                                 "cnot_depth":c["resources"]["cnot_depth"],**row})
        if q.empty:
            continue
        groups=q[(q.case_id==case_id)&(q.method==method)&q.include_hf]
        for shots,g in groups.groupby("shots"):
            eligible=g[g.mean_energy_error_mHa<=1000*CHEMICAL_ACCURACY_HA]
            for code,column in (("A_min_cnot_mean_chemical","cnot_count"),("B_min_depth_mean_chemical","cnot_depth")):
                chosen=eligible.sort_values([column,"step"]).iloc[0] if len(eligible) else None
                optima.append(_optimum(case_id,method,shots,code,chosen,"RETROSPECTIVE_FINITE_SAMPLE_MEAN;not_success_probability_certificate"))
            best=g.sort_values(["mean_energy_error_mHa","cnot_count","step"]).iloc[0]
            optima.append(_optimum(case_id,method,shots,"C_min_qsci_error",best,"RETROSPECTIVE_SAME_EVALUATION_SAMPLE_OPTIMUM"))
            candidates=[]
            for c in (items if method in ADAPT_METHODS else [final]):
                metric=next((x for x in c["analytic_metrics"] if x["shots"]==shots),None)
                if metric and metric["nonhf_weighted_coverage"] is not None and c["resources"]["cnot_count"]>0:
                    candidates.append((metric["nonhf_weighted_coverage"]/c["resources"]["cnot_count"],c["step"]))
            if candidates:
                score,step=max(candidates,key=lambda x:(x[0],-x[1]))
                selected=g[g.step==step]
                optima.append({**_optimum(case_id,method,shots,"D_max_nonhf_coverage_per_cnot",
                                   selected.iloc[0] if len(selected) else None,"ORACLE_FCI_WEIGHT_DIAGNOSTIC"),
                               "score":score})
            final_row=g[g.step==final["step"]]
            optima.append(_optimum(case_id,method,shots,"E_final_saved_vqe_checkpoint",
                                  final_row.iloc[0] if len(final_row) else None,"LAST_SAVED_CHECKPOINT;may_be_step_or_time_censored"))
    fronts=[];comparisons=[]
    if len(q):
        primary=q[q.include_hf&q.mean_energy_error_mHa.notna()].copy()
        for case_id,g in primary.groupby("case_id"):
            g=g.copy()
            g["pareto_cnot_shots_error"]=nondominated_mask(g,["cnot_count","shots","mean_energy_error_mHa"])
            g["pareto_depth_shots_error"]=nondominated_mask(g,["cnot_depth","shots","mean_energy_error_mHa"])
            g["selection_status"]="RETROSPECTIVE_EMPIRICAL_FRONTIER;point_estimates_not_statistical_dominance"
            fronts.extend(g.to_dict("records"))
            for shots,s in g.groupby("shots"):
                for baseline in ("lucj_1","lucj_2","qeb","qubit","gsd"):
                    b=s[s.method==baseline]
                    if b.empty:
                        continue
                    b=b.sort_values(["mean_energy_error_mHa","cnot_count","step"]).iloc[0]
                    for ceo in ("ceo_ovp","ceo_mvp"):
                        c=s[s.method==ceo]
                        if c.empty:
                            continue
                        # Constrain BOTH logical resources before comparing energy.
                        feasible=c[(c.cnot_count<=b.cnot_count)&(c.cnot_depth<=b.cnot_depth)]
                        if feasible.empty:
                            continue
                        chosen=feasible.sort_values(["mean_energy_error_mHa","cnot_count","step"]).iloc[0]
                        comparisons.append({"case_id":case_id,"shots":int(shots),"R_cap":int(b.R_cap),
                                            "ceo":ceo,"baseline":baseline,"ceo_step":int(chosen.step),
                                            "baseline_step":int(b.step),
                                            "ceo_error_mHa":chosen.mean_energy_error_mHa,
                                            "baseline_error_mHa":b.mean_energy_error_mHa,
                                            "delta_error_mHa":chosen.mean_energy_error_mHa-b.mean_energy_error_mHa,
                                            "ceo_cnot":int(chosen.cnot_count),"baseline_cnot":int(b.cnot_count),
                                            "ceo_depth":int(chosen.cnot_depth),"baseline_depth":int(b.cnot_depth),
                                            "ceo_dimension_mean":chosen.actual_dimension_mean,
                                            "baseline_dimension_mean":b.actual_dimension_mean,
                                            "selection_status":"RETROSPECTIVE_SAME_SAMPLE;not_independently_validated_policy"})
    table_export(out,"checkpoint_optima",optima)
    table_export(out,"pareto_frontiers",fronts)
    table_export(out,"resource_constrained_comparisons",comparisons)
    table_export(out,"construction_costs",costs)
    table_export(out,"analytic_metrics",analytic)
    # Shot-to-quality: smallest tested grid M for the observed mean criterion.
    shots_to=[]
    if len(q):
        for key,g in q[q.include_hf].groupby(["case_id","method","step"]):
            attained=g[g.mean_energy_error_mHa<=1000*CHEMICAL_ACCURACY_HA].sort_values("shots")
            row=attained.iloc[0] if len(attained) else g.sort_values("shots").iloc[-1]
            shots_to.append({"case_id":key[0],"method":key[1],"step":int(key[2]),
                             "minimum_tested_shots_mean_chemical":int(row.shots) if len(attained) else None,
                             "status":"ATTAINED_ON_TESTED_GRID" if len(attained) else "NOT_ATTAINED_UP_TO_MAX_TESTED_SHOTS",
                             "max_tested_shots":int(g.shots.max()),"cnot_depth":int(row.cnot_depth),
                             "cnot_count":int(row.cnot_count),"success_rate_at_selected_grid":row.chemical_accuracy_success_rate,
                             "note":"Not exact shots-to-target; sampled grid, retrospective observed-mean criterion"})
    table_export(out,"shots_to_chemical_accuracy",shots_to)
    result={"checkpoints":len(checkpoints),"finite_groups":len(summaries),"optimum_rows":len(optima),
            "comparison_rows":len(comparisons),"case_count":len({c["case_id"] for c in checkpoints}),
            "archived_fixed_control_groups_excluded_from_primary_selection":archived_fixed_groups,
            "frontier_numerical_energy_tolerance_mHa":1e-5,
            "construction_objective_calls":sum(c["objective_calls"] for c in costs),
            "limits":["same R cap not necessarily same realized d","only latest fixed-control revision enters primary frontier/selection; all old raw data retained",
                      "oracle and retrospective selections are not trained policies",
                      "sample variability is not molecular independence","native hardware metrics require a declared target"]}
    save_json(out/"analysis-status.json",result)
    return result


def _checkpoint_groups(checkpoints):
    groups={}
    for c in checkpoints:
        groups.setdefault((c["case_id"],c["method"]),[]).append(c)
    return groups.items()


def _optimum(case,method,shots,criterion,row,label):
    result={"case_id":case,"method":method,"shots":int(shots),"criterion":criterion,
            "selection_status":label,"attained":row is not None}
    for name in ("step","cnot_count","cnot_depth","mean_energy_error_mHa","chemical_accuracy_success_rate","actual_dimension_mean"):
        result[name]=None if row is None else row[name]
    return finite_or_none(result)
