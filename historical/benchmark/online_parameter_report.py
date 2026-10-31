"""Post-construction online-control report with common-R and common-d readouts."""
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
from .common import ROOT,digest,file_digest,rng_for,save_json
from .local_sampler_primitives import unpack_counts64
from .resource_adaptive_sampler import integral_problem
from .online_parameter_controls import decoded
from .online_parameter_audit import SUCCESS,ops_from
from .sparse_qe_circuit import SparseQECircuit

METHODS=("fixed_pi4","one_parameter_vqe","blockwise_vqe","full_vqe","qsci_local_grid")
LABELS=("Fixed pi/4","One-angle VQE","Last-two VQE","Full VQE","QSCI angle grid")
COLORS=("#8666ac","#2b6c9c","#15968b","#cf8034","#333333")


def report(root):
    root=Path(root).resolve();out=root/"report-v1"
    if out.exists():raise RuntimeError("Additive report directory required")
    cfg=json.loads((root/"manifest.json").read_text());audit=json.loads((root/"validation.json").read_text())
    assert audit["status"]==SUCCESS and not audit["failures"]
    assert file_digest(root/"events.sqlite")==audit["events_sha256"]
    db=sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro",uri=True)
    rows=[];readouts=[];refs={};source_readout_hashes={}
    for case in cfg["cases"]:
        # Existing FCI labels are first read by this outer report, never policy.
        refpath=Path(cfg["source"])/"references"/case/"reference.json"
        ref=json.loads(refpath.read_text());refs[case]=file_digest(refpath)
        backend,info=integral_problem(refpath.parent);eigen={}
        def energy(ids):
            key=tuple(sorted(ids))
            if key not in eigen:eigen[key]=float(np.linalg.eigvalsh(backend.matrix(key,key))[0])
            return eigen[key]
        cells=[]
        for (raw,) in db.execute("SELECT result FROM cells WHERE status='COMPLETED'"):
            r=json.loads(raw)
            if r["case"]==case:cells.append(r)
        orders={};cellmap={}
        for r in cells:
            tid=r["task_id"];cellmap[(r["seed"],r["mode"])]=r;final=[];HF=[]
            for did,raw in db.execute("SELECT id,counts FROM draws WHERE cell=? AND role='final' ORDER BY rowid",(tid,)):
                ids,ns=unpack_counts64(raw);obs=dict(zip(map(int,ids),map(int,ns)))
                ordered=[info["hf_id"]]+sorted((x for x in obs if x!=info["hf_id"]),key=lambda x:(-obs[x],x))
                orders[did]=ordered;final.append(did);HF.append(obs.get(info["hf_id"],0)/sum(obs.values()))
            assert final==[x["draw"] for x in r["final"]]
            acts=decoded(db.execute("SELECT actions FROM circuits WHERE id=?",(r["final_circuit"],)).fetchone()[0])
            sim=SparseQECircuit(info["n_qubits"],info["hf_id"],ops_from(acts),cfg["support_cap"],backend)
            physicalE,grad=sim.energy_gradient([a["theta"] for a in acts])
            res=r["sampling_resources"]
            vals=1000*(np.array([x["energy_Ha"] for x in r["final"]])-ref["fci_energy"])
            rows.append({"case":case,"molecule":ref["case"]["molecule"],"distance_A":ref["case"]["distance"],"seed":r["seed"],"mode":r["mode"],
                         "task_id":tid,"construction_stop":r["construction_stop"],"accepted_steps":len(acts),"decisions":r["decisions"],
                         "n_qubits":info["n_qubits"],"registered_parameters":len(acts),"sampling_CNOT":res["cnot_count"],
                         "sampling_CNOT_depth":res["cnot_depth"],"sampling_total_depth":res["total_depth"],
                         "physical_energy_error_mHa":1000*(physicalE-ref["fci_energy"]),"physical_full_gradient_inf":max(abs(grad),default=0.),
                         "QSCI_error_mHa":float(vals.mean()),"QSCI_t90_halfwidth_mHa":float(t.ppf(.95,len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals))),
                         "observed_chemical_success":float(np.mean(vals<=1.594)),"mean_actual_d_R64":float(np.mean([x["actual_d"] for x in r["final"]])),
                         "mean_measured_HF_fraction":float(np.mean(HF)),"pilot_shots":r["pilot_shots"],"final_shots_per_rep":r["final_shots_per_rep"],
                         "actual_new_model_shots":r["new_actual_model_shots"],"objective_calls_all_candidates":r["actual_classical_objective_calls"],
                         "QPU_objective_cost_known":r["VQE_QPU_objective_shots"] is not None,"construction_wall_seconds":r["wall_seconds"],
                         "peak_RSS_bytes":r["peak_RSS_bytes"],"result_sha256":digest(r)})
        for seed in cfg["seeds"]:
            group=[cellmap[(seed,m)] for m in METHODS]
            for rep in range(cfg["final_repeats"]):
                for R in cfg["readout_R"]:
                    d=min(min(len(orders[r["final"][rep]["draw"]]),R) for r in group)
                    for r in group:
                        did=r["final"][rep]["draw"]
                        for scope,cap in (("common_R",R),("all_five_methods_actual_d_matched",d)):
                            selected=orders[did][:cap];E=energy(selected)
                            assert E>=ref["fci_energy"]-1e-8
                            if scope=="common_R" and R==64:assert abs(E-r["final"][rep]["energy_Ha"])<1e-9
                            readouts.append({"case":case,"molecule":ref["case"]["molecule"],"distance_A":ref["case"]["distance"],"seed":seed,
                                             "mode":r["mode"],"rep":rep,"R":R,"scope":scope,"actual_d":len(selected),"error_mHa":1000*(E-ref["fci_energy"])})
        print(json.dumps({"reported_case":case,"cells":len(rows),"readouts":len(readouts)}),flush=True)
    db.close();frame=pd.DataFrame(rows);rd=pd.DataFrame(readouts)
    expected=len(cfg["cases"])*len(cfg["seeds"])*len(METHODS)
    assert len(frame)==expected==audit["validated"]
    assert len(rd)==expected*cfg["final_repeats"]*len(cfg["readout_R"])*2
    base=frame[frame["mode"]=="qsci_local_grid"].set_index(["case","seed"])
    for metric in ("QSCI_error_mHa","sampling_CNOT","sampling_CNOT_depth","physical_energy_error_mHa"):
        frame[f"delta_{metric}_vs_online_grid"]=[r[metric]-base.loc[(r["case"],r["seed"]),metric] for _,r in frame.iterrows()]
    out.mkdir();frame.to_csv(out/"all-constructed-cells.csv",index=False)
    rd.to_csv(out/"all-R-and-matched-d-readouts.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    metrics=["QSCI_error_mHa","physical_energy_error_mHa","sampling_CNOT","sampling_CNOT_depth","registered_parameters","mean_measured_HF_fraction","pilot_shots","objective_calls_all_candidates","construction_wall_seconds","accepted_steps"]
    metrics += [c for c in frame if c.startswith("delta_")]
    means=frame.groupby(["molecule","mode"])[metrics].mean().reset_index()
    means.to_csv(out/"molecule-means.csv",index=False)
    summaries=[]
    for mode in METHODS:
        g=means[means["mode"]==mode];delta=g.delta_QSCI_error_mHa_vs_online_grid.to_numpy()
        rng,_=rng_for("online-parameter-cluster-bootstrap-v1",cfg["mode"],mode)
        boot=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(axis=1)
        summaries.append({"mode":mode,"molecule_clusters":len(delta),"QSCI_delta_mHa":float(delta.mean()),
                          "cluster90_low":float(np.quantile(boot,.05)),"cluster90_high":float(np.quantile(boot,.95)),
                          "mean_CNOT":float(g.sampling_CNOT.mean()),"mean_CNOT_depth":float(g.sampling_CNOT_depth.mean()),
                          "mean_accepted_steps":float(g.accepted_steps.mean()),"QPU_objective_cost_known":not mode.endswith("_vqe")})
    pd.DataFrame(summaries).to_csv(out/"exploratory-cluster-comparison.csv",index=False)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    mols=sorted(frame.molecule.unique());nr=(len(mols)+2)//3
    def save(fig,name):
        fig.tight_layout()
        for ext in ("png","svg","pdf"):fig.savefig(out/f"{name}.{ext}",dpi=190,bbox_inches="tight")
        plt.close(fig)
    for metric,ylabel,name in (("QSCI_error_mHa","Independent QSCI error (mHa)","01-QSCI-error"),
                               ("sampling_CNOT","Sampling CNOT count","02-CNOT-count"),
                               ("sampling_CNOT_depth","Sampling CNOT depth","03-CNOT-depth"),
                               ("accepted_steps","Accepted physical QE blocks","04-accepted-blocks")):
        fig,axes=plt.subplots(nr,3,figsize=(12,3.1*nr),squeeze=False)
        for ax,mol in zip(axes.flat,mols):
            for mode,label,color in zip(METHODS,LABELS,COLORS):
                x=frame[(frame.molecule==mol)&(frame["mode"]==mode)].groupby("distance_A")[metric].mean()
                ax.plot(x.index,x.values,"o-",ms=3,lw=1.2,label=label,color=color)
            ax.set(title=mol,xlabel="Distance / equal spacing (angstrom)",ylabel=ylabel);ax.grid(alpha=.2)
            if metric=="QSCI_error_mHa":ax.set_yscale("symlog",linthresh=.1)
        for ax in list(axes.flat)[len(mols):]:ax.set_visible(False)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle("Fresh online construction: known sampling M=10,000; VQE objective QPU cost UNKNOWN",fontsize=10)
        save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for mode,label,color in zip(METHODS,LABELS,COLORS):
        a=means[means["mode"]==mode]
        for ax,x in zip(axes,("sampling_CNOT","sampling_CNOT_depth")):
            ax.scatter(a[x],a.QSCI_error_mHa,label=label,color=color,s=25)
            ax.set(xlabel=x.replace("_"," "),ylabel="QSCI error (mHa), molecule mean");ax.set_yscale("symlog",linthresh=.1)
    axes[0].legend(fontsize=7);fig.suptitle("Conditional resource tradeoffs, NOT a complete quantum-cost frontier",fontsize=10)
    save(fig,"05-resource-quality")
    srows="\n".join(f"| {r['mode']} | {r['QSCI_delta_mHa']:+.3f} | [{r['cluster90_low']:+.3f},{r['cluster90_high']:+.3f}] | {r['mean_CNOT']:.1f} | {r['mean_CNOT_depth']:.1f} |" for r in summaries)
    totals={"new_pilot_model_shots":int(frame.pilot_shots.sum()),"new_model_shots":int(frame.actual_new_model_shots.sum()),
            "actual_classical_objective_calls_all_candidates":int(frame.objective_calls_all_candidates.sum()),"hardware_shots":0,"VQE_QPU_objective_cost":None}
    assert totals["new_model_shots"]==audit["new_pilot_model_shots"]+audit["new_final_model_shots"]
    text=f"""# Online parameter/discovery controls — {cfg['mode']}

## Actual execution and validation

{len(frame)}/{expected} fresh HF-started constructions, all seeds retained.
Unlike the earlier fixed-topology diagnostic, operator lists were NOT inherited.
Each proposed candidate underwent the registered angle treatment before its
paid32-shot pilot, acceptance, and next-round residual proposal construction.
All rejected-candidate optimization work is retained. All{audit['raw_draws_rebuilt']}draws,
{audit['final_QPY_states_and_distributions_checked']}final QPY state/distribution checks,
{audit['objective_trace_calls_recomputed']}objective trace evaluations were validated.
FCI labels were read only for THIS outer error report, no newFCI solve.

## Exploratory all-case result

Negative delta means smaller QSCI error than online QSCI-grid control.
Average all construction seeds and distances within molecule, then equal-weight
molecules. Cluster90 intervals are exploratory bootstrap, not confirmatory tests.

| Method | ΔQSCI mHa | cluster90 interval | meanCNOT | meanDepth |
|---|---:|---:|---:|---:|
{srows}

construction_stop distribution (capped/no-improvement are NOT convergence):
```
{frame.groupby(['mode','construction_stop']).size().to_string()}
```

## Cost and inference boundaries

- Known-model M10000 includes all pilots and ONE independent final draw. There
  are{cfg['final_repeats']}actual final readout repetitions; construction is paid once.
- Total NEW model shots{totals['new_model_shots']:,}, including{totals['new_pilot_model_shots']:,}pilots.
- Actual classical objective/gradient calls{totals['actual_classical_objective_calls_all_candidates']:,}, including rejected candidates.
  Their QPU objective cost is UNKNOWN, not0/callcount. VQE rows are NOT complete
  end-to-end quantum-cost-matched results. No hardware shots.
- CommonR8..512 AND all-five-methods minimum realized-d diagnostics use the SAME
  final counts. They add CI computation, not shots. d-matching is a diagnostic,
  not a claim of identical distributions or a different operational algorithm.
- Independent readout replicates are conditional on constructed circuit; three
  seeds are not fifteen independent molecules. No bestseed outcome filtering.
- These are end-to-end stochastic policy variants. Different mode-bound RNG
  paths mean operator-path differences are NOT isolated causal estimates of
  coefficient optimization alone. The prior fixed-topology controls supply
  the separate conditional angle diagnostic; neither design replaces the other.
- Exact sparse physical support <=512 can grow exponentially with depth.
  The policy receives counts, not the physical amplitudes; no fullFock vector/H
  internally. Full<=16q vectors used only for independent small-case auditing.
- Nalpha/Nbeta checks are NOT a total-spin S² certificate. Native hardware
  target/measurement-objective shot model remain unconfigured.
- Read-only old source preserved. Initial storage refusal and audit-qualification
  near-tie failures preserved. Producer's candidates/angles/counts were not
  changed to make the auditor agree. Numerical audit compares independent values
  with tolerance and replays exact recorded scores for bit-sensitive ties.
- This is not yet a fresh strong-LUCJ/nativeADAPT-QSCI comparison for every new
  online cell, nor completion of B multi-basis/E stronglarge/Fnoise/fullstudy.

## Data / reproduction

all-constructed-cells.csv; all-R-and-matched-d-readouts.csv.gz;
molecule-means.csv; exploratory-cluster-comparison.csv;5figures×PNG/SVG/PDF.

```bash
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmark.online_parameter_controls --mode {cfg['mode']} --min-free-gib 2.75
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmark.online_parameter_audit {root.relative_to(ROOT)}
```

Do not rerun completed draws. Completed cells return recorded results.
Partial cells preserve evidence, no automatic free retry. Whole goal remains active.
"""
    (out/"REPORT.md").write_text(text)
    manifest={"status":"ONLINE_PARAMETER_INTERACTION_EXECUTED_AUDITED_REPORTED","cells":len(frame),"readout_rows":len(rd),"totals":totals,
              "summaries":summaries,"events_sha256":audit["events_sha256"],"validation_sha256":file_digest(root/"validation.json"),
              "reference_json_hashes_outer_evaluation_only":refs,"report_code_sha256":file_digest(Path(__file__)),
              "artifact_sha256":{p.name:file_digest(p) for p in out.iterdir() if p.is_file()},"whole_study_complete":False}
    save_json(out/"report-manifest.json",manifest);return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");a=p.parse_args();r=report(a.root)
    print(json.dumps({k:r[k] for k in ("status","cells","readout_rows","totals","summaries")}))
