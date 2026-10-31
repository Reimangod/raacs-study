"""Independently check plotted tables; additive color-correct Fig15 export.

V1's seed-range fills used an independent Matplotlib color cycle. Preserve v1,
but use explicitly matched line/fill colors in this separate presentation v2.
No scientific values or existing figures/manifests are edited.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
POOLS=("ceo_ovp","ceo_mvp","qeb","qubit","gsd")


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a,b):
    if not np.allclose(np.asarray(a,float),np.asarray(b,float),rtol=0,atol=1e-9,equal_nan=True):
        raise AssertionError("Independent plotted-value reconstruction mismatch")


def run(root):
    root=Path(root);mf=json.loads((root/"manifest.json").read_text())
    for name,digest in mf["source_sha256"].items():assert sha(ROOT/name)==digest
    for name,digest in mf["output_sha256"].items():assert sha(root/name)==digest
    source=ROOT/"results/standard-v1"; A=ROOT/"results/sampler-study-standard-v1"; B1=ROOT/"results/sampler-baselines-B1-standard-v1"; B4=ROOT/"results/random-B4-standard-v1"
    inventory=pd.DataFrame(json.loads((A/"checkpoint-inventory.json").read_text()))
    cp=inventory[inventory.primary & inventory.case_id.isin(mf["cases"])].copy()
    ref={c:json.loads((source/"references"/c/"reference.json").read_text()) for c in mf["cases"]}
    rows=pd.read_parquet(A/"analysis/readout-summary.parquet")
    rows=rows[rows.primary_record & rows.complete_group & (rows.selection_rule=="frequency") & (rows.shots==1000)]
    expected=rows[rows.R_cap==32].merge(cp,on=["case_id","method","step","checkpoint_id"],validate="one_to_one").set_index("checkpoint_id").sort_index()
    curves=pd.read_csv(root/"growth-curves.csv").set_index("checkpoint_id").sort_index()
    assert expected.index.equals(curves.index)
    close(expected.mean_error_mHa,curves.mean_error_mHa)
    close(expected.vqe_energy,curves.vqe_energy)
    close([(r.vqe_energy-ref[r.case_id]["fci_energy"])*1000 for r in expected.itertuples()],curves.vqe_error_mHa)
    transition=pd.read_csv(root/"addition-reoptimization-transitions.csv")
    for r in transition.itertuples():
        a,b=curves.loc[r.prior_checkpoint_id],curves.loc[r.checkpoint_id]
        assert a.case_id==b.case_id==r.case_id and a.method==b.method==r.method and b.step==a.step+1
        close([r.qsci_gain_mHa,r.delta_cnot,r.delta_depth],[a.mean_error_mHa-b.mean_error_mHa,b.cnot-a.cnot,b.depth-a.depth])
        assert json.loads(r.added_group_json)==cp[cp.checkpoint_id==r.checkpoint_id].iloc[0].added_group
    discovery=pd.read_parquet(A/"analysis/discovery.parquet")
    discovery=discovery[discovery.primary_record & discovery.case_id.isin(mf["cases"]) & (discovery.shots==1000)].set_index("checkpoint_id").sort_index()
    coverage=pd.read_csv(root/"HF-vs-nonHF-coverage.csv").set_index("checkpoint_id").sort_index()
    assert coverage.index.equals(discovery.index)
    close(coverage.hf_probability,discovery.hf_probability)
    close(coverage.nonhf_weighted_coverage,discovery.nonhf_weighted_coverage)
    heat=pd.read_csv(root/"unified-method-heatmap.csv")
    b1rows=pd.read_csv(B1/"readout-summary.csv")
    last=cp.groupby(["case_id","method"]).step.max().to_dict()
    for r in heat.itertuples():
        if r.method in POOLS:
            q=rows[(rows.case_id==r.case_id)&(rows.method==r.method)&(rows.R_cap==64)&(rows.step==last[r.case_id,r.method])]
            assert len(q)==1;close([r.mean_error_mHa,r.mean_actual_d],[q.iloc[0].mean_error_mHa,q.iloc[0].mean_dimension])
        else:
            q=b1rows[(b1rows.case_id==r.case_id)&(b1rows.method==r.method)&(b1rows.shots==1000)&(b1rows.R_cap==64)]
            if len(q):
                assert len(q)==1;close([r.mean_error_mHa,r.mean_actual_d],[q.iloc[0].mean_error_mHa,q.iloc[0].mean_actual_d])
            else:assert pd.isna(r.mean_error_mHa) and pd.isna(r.mean_actual_d)
    oracle=pd.read_csv(root/"oracle-random-trained-discovery.csv")
    assert len(oracle)==len(mf["cases"])*9*9
    for case,g in oracle.groupby("case_id"):
        with np.load(source/"references"/case/"reference.npz",allow_pickle=False) as z:
            ids=z["ids"];w=z["weights"].astype(np.longdouble)
        keep=ids!=ref[case]["hf_id"];denom=w[keep].sum()
        for (method,seed),arm in g.groupby(["method","construction_seed"],dropna=False):
            if method=="FCI_Born_oracle":p=w
            else:
                if method in POOLS:
                    row=cp[(cp.case_id==case)&(cp.method==method)].sort_values("step").iloc[-1]
                    file=source/row.distribution_path
                else:file=B4/"cells"/case/f"{method}_s{int(seed)}"/"final.npz"
                with np.load(file,allow_pickle=False) as z:p=z["probability"][ids].astype(np.longdouble)
            for r in arm.itertuples():
                # Independent long-double direct powers, not expm1 implementation.
                value=float(np.sum(w[keep]*(1-np.power(1-np.clip(p[keep],0,1),int(r.M))))/denom) if denom>1e-12 else np.nan
                close(value,r.nonhf_expected_weighted_coverage)
    expected_stems=[f"{c}_Fig{n}_{suffix}" for c in mf["cases"] for n,suffix in [("05","growth"),("12","operator_gain"),("13","HF_coverage"),("15","random_trained_oracle")]]+["Fig09_unified_method_heatmap"]
    assert set(mf["output_sha256"]) >= {s+"."+e for s in expected_stems for e in ("png","svg","pdf")}
    out=root/"presentation-v2";out.mkdir(exist_ok=True)
    colors={"FCI_Born_oracle":"#2563a6","ceo_mvp":"#c06b27","ceo_ovp":"#278452","random_ceo":"#b04266","trained_ceo_random_angles":"#7857a6"}
    for case,g in oracle.groupby("case_id"):
        fig,ax=plt.subplots(figsize=(7.5,5));plt.rcParams["svg.fonttype"]="none"
        for method,arm in g.groupby("method"):
            a=arm.groupby("M").nonhf_expected_weighted_coverage.agg(["mean","min","max"])
            ax.plot(a.index,100*a["mean"],"o--" if method=="FCI_Born_oracle" else "o-",ms=3,label=method,color=colors[method])
            if arm.construction_seed.notna().any():ax.fill_between(a.index,100*a["min"],100*a["max"],alpha=.12,color=colors[method])
        ax.set(xlabel="Hypothetical computational-basis draws",ylabel="Expected non-HF weighted coverage (%)",title=case,xscale="log")
        ax.legend(fontsize=7);ax.grid(alpha=.15)
        fig.text(.015,.015,"Saved FCI Born distribution: oracle analysis, not practical preparation, not an optimal-discovery bound.\nRandom: all3 seeds, range (NOT CI); final CEO length/order is retrospective. Analytic expectations, no new draws.",fontsize=7)
        fig.tight_layout(rect=(0,.14,1,.97))
        for ext in ("png","svg","pdf"):fig.savefig(out/f"{case}_Fig15_random_trained_oracle.{ext}",dpi=160,facecolor="white")
        plt.close(fig)
    grouped=oracle[oracle.M==1000].groupby(["case_id","method"]).nonhf_expected_weighted_coverage.mean().unstack()
    (out/"M1000-case-means.csv").write_text(grouped.to_csv())
    molecule=grouped.copy();molecule["molecule"]=[x.split("_R")[0] for x in molecule.index]
    means=molecule.groupby("molecule").mean(numeric_only=True).mean().to_dict()
    record={"status":"ALL_PLOTTED_TABLES_RECONSTRUCTED_PRESENTATION_V2_EXPORTED", "source_manifest_sha256":sha(root/"manifest.json"),"auditor_sha256":sha(__file__),
        "growth_points_checked":len(curves),"adjacent_growth_changes_checked":len(transition),"coverage_points_checked":len(coverage),
        "heatmap_planned_cells_checked":len(heat),"oracle_analytic_values_checked":len(oracle),"figure_sets_verified":len(expected_stems),
        "M1000_six_molecule_equally_weighted_expected_coverage":means,
        "M1000_cases_exceeding_FCI_Born_oracle":int((grouped.max(axis=1)>grouped.FCI_Born_oracle+1e-10).sum()),
        "descriptive_only_geometry_points_not_independent_molecules":True,"new_quantum_draws":0,"new_reference_solves":0,"whole_goal_complete":False,
        "presentation_correction":"Range fill now matches line color; original v1 output and numerical tables unchanged",
        "presentation_output_sha256":{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="validation-and-style-v2.json"}}
    (out/"validation-and-style-v2.json").write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();r=run(a.root)
    print(json.dumps({k:r[k] for k in ("status","growth_points_checked","adjacent_growth_changes_checked","oracle_analytic_values_checked","whole_goal_complete")}))
