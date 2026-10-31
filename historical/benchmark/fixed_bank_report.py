"""Outer allocation report: final-resource savings do not erase growth cost."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import ROOT,file_digest,rng_for,save_json
from .resource_adaptive_sampler import integral_problem
from .local_sampler_primitives import unpack_counts64
from .fixed_bank_allocation import METHODS
from .fixed_bank_audit import SUCCESS

LABELS=("Last shallow (pilot charged)","Uniform","Greedy pilot energy","Exclusive discovery","Energy weighted","Deep, no new pilot")
COLORS=("#2b6c9c","#15968b","#8666ac","#cf8034","#bb4b65","#333333")


def report(root):
    root=Path(root).resolve();cfg=json.loads((root/"manifest.json").read_text());validation=json.loads((root/"validation.json").read_text())
    assert validation["status"]==SUCCESS and file_digest(root/"events.sqlite")==validation["events_sha256"]
    out=root/"report-v1"
    if out.exists():raise RuntimeError("Use additive report version")
    db=sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro",uri=True);rows=[];readouts=[];refs={};source_costs=[]
    for item in cfg["queue"]:
        case=item["case"];refpath=Path(cfg["source"])/"references"/case/"reference.json";ref=json.loads(refpath.read_text());refs[case]=file_digest(refpath)
        backend,info=integral_problem(refpath.parent);cached={}
        def solve(ids):
            ids=tuple(sorted(ids))
            if ids not in cached:cached[ids]=float(np.linalg.eigvalsh(backend.matrix(ids,ids))[0])
            return cached[ids]
        parent=json.loads((Path(item["source_cell"])/"result.json").read_text())
        selected=[json.loads(v[0]) for v in db.execute("SELECT result FROM bases WHERE status='COMPLETED'")]
        base=next(v for v in selected if v["case"]==case and v["source_seed"]==item["seed"]);bid=base["base_id"]
        original=sqlite3.connect(f"file:{Path(item['source_cell'])/'events.sqlite'}?mode=ro",uri=True)
        historical_gate_shots=0;historical_depth_shots=0
        for role,n,raw in original.execute("SELECT role,raw_shots,metadata FROM draws"):
            if role=="independent_final":continue
            res=json.loads(raw)["sampling_resources"];historical_gate_shots+=n*res["cnot_count"];historical_depth_shots+=n*res["cnot_depth"]
        original.close()
        source_costs.append({"case":case,"seed":item["seed"],"historical_pilot_shots":parent["pilot_raw_model_shots"],"historical_pilot_gate_shots":historical_gate_shots,
                             "historical_pilot_depth_shots":historical_depth_shots,"historical_total_wall_seconds_as_recorded":parent["actual_wall_seconds"],
                             "new_all_six_policies_and_readouts_wall_seconds":base["classical_wall_seconds"],"not_six_new_growth_runs":True})
        orders={};observations={}
        for mode in METHODS:
            rec=json.loads(db.execute("SELECT record FROM policies WHERE base=? AND method=?",(bid,mode)).fetchone()[0]);values=[];dims=[];rawcounts=[]
            for rep,raw in db.execute("SELECT rep,record FROM finals WHERE base=? AND method=? ORDER BY rep",(bid,mode)):
                f=json.loads(raw);counts={}
                for did in f["component_draws"]:
                    blob=db.execute("SELECT counts FROM draws WHERE id=?",(did,)).fetchone()[0];ids,ns=unpack_counts64(blob)
                    for x,n in zip(ids,ns):counts[int(x)]=counts.get(int(x),0)+int(n)
                order=[info["hf_id"]]+sorted((x for x in counts if x!=info["hf_id"]),key=lambda x:(-counts[x],x))
                orders[(mode,rep)]=order;observations[(mode,rep)]=counts;E=solve(order[:cfg["R"]]);assert abs(E-f["energy_Ha"])<1e-9
                values.append(1000*(E-ref["fci_energy"]));dims.append(f["actual_d"]);rawcounts.append(counts.get(info["hf_id"],0)/sum(counts.values()))
            params=[]
            for cid in rec["members"]:
                params.append(json.loads(db.execute("SELECT resources FROM circuits WHERE id=?",(cid,)).fetchone()[0])["parameter_count"])
            rows.append({"case":case,"molecule":ref["case"]["molecule"],"distance_A":ref["case"]["distance"],"seed":item["seed"],"method":mode,
                         "mean_error_mHa":float(np.mean(values)),"readout_t90_halfwidth_mHa":float(t.ppf(.95,len(values)-1)*np.std(values,ddof=1)/np.sqrt(len(values))),
                         "observed_chemical_success":float(np.mean(np.array(values)<=1.594)),"mean_actual_d":float(np.mean(dims)),"mean_HF_fraction":float(np.mean(rawcounts)),
                         "max_final_CNOT":rec["maximum_final_CNOT"],"max_final_depth":rec["maximum_final_depth"],"weighted_CNOT":rec["weighted_final_CNOT"],"weighted_depth":rec["weighted_final_depth"],
                         "maximum_parameters_per_used_circuit":max(p for p,n in zip(params,rec["allocation"]) if n),"final_switches":rec["final_switches"],
                         "new_pilot_charge":rec["new_pilot_charge"],"final_shots":rec["final_shots"],"historical_source_pilot_charge":rec["source_pilot_charge"],
                         "final_gate_shots":rec["final_gate_shots"],"new_pilot_gate_shots_charge":rec["pilot_gate_shots_charged"],
                         "known_branch_gate_shots_including_source":rec["final_gate_shots"]+rec["pilot_gate_shots_charged"]+historical_gate_shots,
                         "known_branch_model_shots":cfg["M"],"no_historical_growth_avoidance_claim":True})
        # Pair each rule with the stronger, zero-new-pilot deep control at the
        # same construction seed / final-replicate index. Dimension matching
        # is a diagnostic, not an extra acquisition or shared-randomness claim.
        for mode in METHODS:
            for rep in range(cfg["repeats"]):
                a=orders[(mode,rep)][:cfg["R"]];b=orders[("single_deep_no_pilot",rep)][:cfg["R"]];d=min(len(a),len(b))
                readouts.append({"case":case,"molecule":ref["case"]["molecule"],"seed":item["seed"],"method":mode,"rep":rep,
                                 "common_R_delta_mHa":1000*(solve(a)-solve(b)),"paired_actual_d":d,
                                 "matched_d_delta_mHa":1000*(solve(a[:d])-solve(b[:d]))})
    db.close();frame=pd.DataFrame(rows);rd=pd.DataFrame(readouts);base=frame[frame.method=="single_deep_no_pilot"].set_index(["case","seed"])
    for metric in ("mean_error_mHa","max_final_CNOT","max_final_depth","known_branch_gate_shots_including_source"):
        frame[f"delta_{metric}_vs_deep"]=[r[metric]-base.loc[(r["case"],r["seed"]),metric] for _,r in frame.iterrows()]
    assert len(frame)==validation["policy_cells"] and len(rd)==len(frame)*cfg["repeats"]
    out.mkdir();frame.to_csv(out/"all-policy-cells.csv",index=False);rd.to_csv(out/"common-R-and-paired-d.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    pd.DataFrame(source_costs).to_csv(out/"unique-source-costs-NOT-new-acquisitions.csv",index=False)
    metrics=["mean_error_mHa","max_final_CNOT","max_final_depth","weighted_CNOT","weighted_depth","maximum_parameters_per_used_circuit","known_branch_gate_shots_including_source"]+[c for c in frame if c.startswith("delta_")]
    mol=frame.groupby(["molecule","method"])[metrics].mean().reset_index();mol.to_csv(out/"molecule-means.csv",index=False)
    summaries=[]
    for mode in METHODS:
        g=mol[mol.method==mode];delta=g.delta_mean_error_mHa_vs_deep.to_numpy();rng,_=rng_for("fixed-bank-cluster-bootstrap",cfg["mode"],mode)
        boot=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(axis=1)
        summaries.append({"method":mode,"molecule_clusters":len(delta),"mean_QSCI_delta_vs_deep_mHa":float(delta.mean()),"cluster90_low":float(np.quantile(boot,.05)),
                          "cluster90_high":float(np.quantile(boot,.95)),"mean_max_final_CNOT":float(g.max_final_CNOT.mean()),"mean_max_final_depth":float(g.max_final_depth.mean()),
                          "mean_known_branch_gate_shots_including_source":float(g.known_branch_gate_shots_including_source.mean())})
    pd.DataFrame(summaries).to_csv(out/"exploratory-cluster-summary.csv",index=False)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False});mols=sorted(frame.molecule.unique());nr=(len(mols)+2)//3
    def save(fig,name):
        fig.tight_layout()
        for ext in ("png","svg","pdf"):fig.savefig(out/f"{name}.{ext}",dpi=190,bbox_inches="tight")
        plt.close(fig)
    for metric,label,name in (("mean_error_mHa","Independent QSCI error (mHa)","01-error-vs-distance"),("weighted_CNOT","Shot-weighted final CNOT count","02-final-weighted-CNOT"),("weighted_depth","Shot-weighted final CNOT depth","03-final-weighted-depth")):
        fig,axes=plt.subplots(nr,3,figsize=(12,3.2*nr),squeeze=False)
        for ax,m in zip(axes.flat,mols):
            for mode,title,color in zip(METHODS,LABELS,COLORS):
                g=frame[(frame.molecule==m)&(frame.method==mode)].groupby("distance_A")[metric].mean();ax.plot(g.index,g.values,"o-",ms=3,lw=1.1,label=title,color=color)
            ax.set(title=m,xlabel="Distance / equal spacing (angstrom)",ylabel=label);ax.grid(alpha=.2)
            if metric=="mean_error_mHa":ax.set_yscale("symlog",linthresh=.1)
        for ax in list(axes.flat)[len(mols):]:ax.set_visible(False)
        axes.flat[0].legend(fontsize=6.5);fig.suptitle("Fixed inherited shallow bank; source growth + pilots + final M=10,000",fontsize=11);save(fig,name)
    table="\n".join(f"| {s['method']} | {s['mean_QSCI_delta_vs_deep_mHa']:+.4f} | [{s['cluster90_low']:+.4f},{s['cluster90_high']:+.4f}] | {s['mean_max_final_CNOT']:.2f} | {s['mean_max_final_depth']:.2f} |" for s in summaries)
    text=f"""# Fixed-bank allocation diagnostic — {cfg['mode']}

{validation['bases']} inherited source trajectories, {len(frame)}policy cells,
{validation['raw_component_and_pilot_draws_rebuilt']}raw component/pilot draws rebuilt,
{validation['physical_QPY_states_and_sampling_distributions_checked']}physical QPY state/distribution checks.
New actual model shots: {validation['new_pilot_shots_actual_once']+validation['new_final_model_shots']:,};
shared pilots counted ONCE ({validation['new_pilot_shots_actual_once']:,}).
Inherited sourcepilot shots ({validation['inherited_pilot_shots_unique_NOT_new']:,}) are historical, not new.
NoFCIselection/newFCIsolve; existingFCIlabels loaded only by this outer report.

## All-case exploratory result

Negative Δ means smaller error than the stronger deep control with no NEW pilot.
Average all seeds/distances within molecule, then equal-weight molecules.
Intervals are exploratory molecule-cluster bootstrap, not confirmatory inference.

| Rule | ΔQSCI vsdeep (mHa) | cluster90 | mean maximum finalCNOT | mean maximum finalDepth |
|---|---:|---:|---:|---:|
{table}

## What is and is not isolated

The shallow bank is identical across five allocation rules, with identical
sharedpilot information/cost. Their primary comparison isolates allocation
conditional on that bank and pilot. Uniform/last-only were deliberately
charged these pilots; these are NOT cheapest implementations of fixed rules.
The deep control instead uses no new pilot and gets correspondingly more
finalshots at the SAME M10000 known construction+sampling budget. Final
counts are independent of pilot selection. Component counts were genuinely
drawn with fixed per-circuit allocations, then summed, never probability-averaged.

Five-mode results are conditional diagnostics, not new operator discoveries.
Full historical source growth is retained/costed; lower FINAL circuit depth
does not prove that deep construction was avoided. Report maximum per-circuit
resources and shot-weighted averages separately. Parameters are maximum
registered slots in a used circuit, not zero because numerical angles are fixed.
Circuit-switch count is measured; hardware latency/native target unknown.
SectorNalpha/Nbeta is notS² certification. Hardware shots0.

common-R-and-paired-d.csv.gz also truncates each policy/deep pair to the smaller
actually observed dimension. This uses the same counts, adds CI work not shots,
and is not an identical-noise or identical-distribution causal experiment.
No bestseed filtering; distances are not independent molecules. New classical
walltime is stored once for a whole source bank+all6shadow policies/readouts,
not fabricated as six separately measured construction times. Historical
source walltime includes its own recorded scope; no device-matched speedup claim.

Three figures×PNG/SVG/PDF, all-policy-cells.csv, commonR/paired-d records,
unique source costs and molecule-cluster summary are bound in report-manifest.
This does not complete Bmultibasis/Estronglarge/Fnoise or the whole study.
"""
    (out/"REPORT.md").write_text(text)
    manifest={"status":"FIXED_BANK_ALLOCATION_EXECUTED_AUDITED_REPORTED","policy_cells":len(frame),"readout_diagnostic_rows":len(rd),"summaries":summaries,
              "validation_sha256":file_digest(root/"validation.json"),"events_sha256":validation["events_sha256"],"reference_hashes_outer_only":refs,
              "report_code_sha256":file_digest(Path(__file__)),"artifact_sha256":{p.name:file_digest(p) for p in out.iterdir() if p.is_file()},"whole_study_complete":False}
    save_json(out/"report-manifest.json",manifest);return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");a=p.parse_args();r=report(a.root);print(json.dumps({k:r[k] for k in ("status","policy_cells","summaries")}))
