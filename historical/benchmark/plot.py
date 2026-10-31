from __future__ import annotations
import argparse
import json
from itertools import product
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS={"ceo_mvp":"#1f77b4","ceo_ovp":"#ff7f0e","qeb":"#2ca02c","qubit":"#d62728",
        "gsd":"#9467bd","lucj_1":"#8c564b","lucj_2":"#e377c2","lucj_truncated":"#bcbd22",
        "lucj_random_101":"#7f7f7f","hf":"#111111","ceo_random_101":"#17becf"}


def save_figure(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if len(fig.axes)==1:
        ax=fig.axes[0];handles,labels=ax.get_legend_handles_labels()
        if len(labels)>6:
            fig.set_size_inches(fig.get_figwidth()+1.6,fig.get_figheight())
            ax.legend(handles,labels,fontsize=6,loc="upper left",bbox_to_anchor=(1.02,1))
    fig.tight_layout()
    for extension in ("png","svg","pdf"):
        fig.savefig(path.with_suffix("."+extension),dpi=200)
    plt.close(fig)


def trajectory(ax,g,x,y,method,**kwargs):
    g=g.sort_values("step")
    fixed=method not in ("ceo_ovp","ceo_mvp","qeb","qubit","gsd")
    if fixed and x=="step":
        # Random-control revisions follow their source CEO cap, NOT ADAPT growth.
        g=g[g.step==g.step.max()].copy();g["step"]=0
    ax.plot(g[x],g[y],marker=".",linestyle="None" if fixed else "-",label=method,color=COLORS[method],**kwargs)


def plot_all(root):
    root = Path(root)
    checkpoints = [json.loads(p.read_text()) for p in (root/"checkpoints").glob("*/*/step-*.json")]
    if not checkpoints:
        return
    analytic = pd.DataFrame([{**a,"case_id":c["case_id"],"method":c["method"],"step":c["step"],
                              "cnot_count":c["resources"]["cnot_count"],
                              "vqe_error_mHa":1000*c["vqe_energy_error"]}
                             for c in checkpoints for a in c["analytic_metrics"]])
    qsci_rows = [row for p in (root/"qsci").glob("*/summary.json") for row in json.loads(p.read_text())]
    finite = pd.DataFrame(qsci_rows)
    from .analysis import analyze
    analyze(root)
    for case_id,group in analytic.groupby("case_id"):
        for y,x,name,label in [
            ("nonhf_weighted_coverage","cnot_count","nonhf_coverage","Expected non-HF FCI-weight coverage"),
            ("hf_probability","step","hf_probability","HF probability"),
            ("vqe_error_mHa","step","vqe_error","VQE error (mHa)"),
            ("recall_top10","step","top10_recall","Expected top-10 non-HF determinant recall")]:
            fig,ax = plt.subplots(figsize=(5.5,3.8))
            for method,g in group[group.shots==1000].groupby("method"):
                trajectory(ax,g,x,y,method)
            if y=="vqe_error_mHa":ax.set_yscale("symlog",linthresh=1e-6)
            ax.set(xlabel=x,ylabel=label,title=f"{case_id}: analytic distribution diagnostic")
            ax.legend(fontsize=7); ax.grid(alpha=.2)
            save_figure(fig,root/"figures"/case_id/name)
        if len(finite):
            selected = finite[(finite.case_id==case_id)&finite.include_hf]
            for (x,stem),shots in product([( "cnot_count","qsci_vs_cnot"),("cnot_depth","qsci_vs_depth"),("step","qsci_vs_iteration")],(1000,10000,100000)):
                name=stem if shots==1000 else stem+f"_M{shots}"
                fig,ax=plt.subplots(figsize=(5.5,3.8))
                for method,g in selected[selected.shots==shots].groupby("method"):
                    trajectory(ax,g,x,"mean_energy_error_mHa",method)
                ax.set_yscale("symlog",linthresh=1e-6)
                ax.axhline(1.59360143764,color="gray",ls=":",label="1 kcal/mol")
                ax.set(xlabel=x,ylabel="QSCI mean error (mHa)",title=f"{case_id}: finite {shots} raw shots")
                ax.legend(fontsize=7);ax.grid(alpha=.2)
                save_figure(fig,root/"figures"/case_id/name)
            fig,ax=plt.subplots(figsize=(5.5,3.8))
            for method,g in selected.groupby("method"):
                last = g[g.step==g.step.max()].sort_values("shots")
                ax.plot(last.shots,last.mean_energy_error_mHa,".-",label=method,color=COLORS[method])
            ax.set_xscale("log")
            ax.set_yscale("symlog",linthresh=1e-6)
            ax.set(xlabel="Raw shots",ylabel="QSCI mean error (mHa)",title=f"{case_id}: final checkpoint, finite samples")
            ax.legend(fontsize=7);ax.grid(alpha=.2)
            save_figure(fig,root/"figures"/case_id/"qsci_vs_shots")
        # All requested top-K diagnostics, not merely K=10.
        fig,axes=plt.subplots(2,2,figsize=(9,6))
        for k,ax in zip((5,10,20,50),axes.ravel()):
            for method,g in group[group.shots==1000].groupby("method"):
                trajectory(ax,g,"step",f"recall_top{k}",method,markersize=3)
            ax.set(xlabel="Saved growth step (fixed ansatz:0)",ylabel=f"Expected top-{k} recall",ylim=(-.03,1.03))
            ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=5,ncol=2)
        fig.suptitle(f"{case_id}: analytic FCI-reference recall, M=1000")
        save_figure(fig,root/"figures"/case_id/"topk_recall")
        shots_file=root/"analysis/shots_to_chemical_accuracy.json"
        if shots_file.exists():
            shots=pd.DataFrame(json.loads(shots_file.read_text()))
            if len(shots):
                selected=shots[shots.case_id==case_id]
                fig,ax=plt.subplots(figsize=(6,4))
                for method,g in selected.groupby("method"):
                    good=g[g.minimum_tested_shots_mean_chemical.notna()]
                    ax.scatter(good.cnot_depth,good.minimum_tested_shots_mean_chemical,label=method,s=20,color=COLORS[method])
                ax.set_yscale("log");ax.set_ylim(700,140000)
                ax.set(xlabel="Sampler CNOT depth",ylabel="Smallest tested M meeting mean-error criterion",
                       title=f"{case_id}: finite grid; non-attainment omitted from dots")
                ax.text(.02,.02,f"Not attained: {sum(selected.status!='ATTAINED_ON_TESTED_GRID')}/{len(selected)} checkpoints",
                        transform=ax.transAxes,fontsize=7)
                ax.legend(fontsize=5,ncol=2);ax.grid(alpha=.2)
                save_figure(fig,root/"figures"/case_id/"shots_to_chemical_vs_depth")
    plot_cross_case(root,finite,checkpoints)


def plot_cross_case(root,finite,checkpoints):
    import numpy as np
    from .config import DISTANCES
    if finite.empty:
        return
    directory=root/"figures/cross_case"
    primary=finite[finite.include_hf].copy()
    at_m=primary[primary.shots==1000].copy()
    last=at_m.sort_values("step").groupby(["case_id","method"]).tail(1)
    references={p.parent.name:json.loads(p.read_text()) for p in (root/"references").glob("*/reference.json")}
    # Predefined full grid: unexecuted/unavailable cells remain NaN (grey), never0.
    distances=sorted({r for values in DISTANCES.values() for r in values})
    molecules=list(DISTANCES)
    for method in ("ceo_ovp","ceo_mvp","qeb","qubit","gsd","lucj_1","lucj_2"):
        matrix=np.full((len(molecules),len(distances)),np.nan)
        for _,r in last[last.method==method].iterrows():
            molecule,bond=r.case_id.split("_R");bond=float(bond)
            matrix[molecules.index(molecule),distances.index(bond)]=np.log10(max(r.mean_energy_error_mHa,1e-6))
        fig,ax=plt.subplots(figsize=(10,3.5))
        color=plt.get_cmap("viridis").copy();color.set_bad("#dddddd")
        image=ax.imshow(np.ma.masked_invalid(matrix),aspect="auto",cmap=color,vmin=-6,vmax=3)
        ax.set_xticks(range(len(distances)),[f"{r:g}" for r in distances],rotation=45)
        ax.set_yticks(range(len(molecules)),molecules)
        ax.set(xlabel="Bond distance (Å)",title=f"{method}: final checkpoint QSCI, M=1000; grey=not measured/not in grid")
        fig.colorbar(image,ax=ax,label="log10 mean error (mHa), display floor1e-6")
        save_figure(fig,directory/f"heatmap_{method}")
    comparisons=pd.DataFrame(json.loads((root/"analysis/resource_constrained_comparisons.json").read_text()))
    for baseline,name in (("lucj_1","ceo_lucj1_difference"),("lucj_2","ceo_lucj2_difference"),("qeb","ceo_qeb_cnot_reduction")):
        fig,ax=plt.subplots(figsize=(9,4))
        if len(comparisons):
            data=comparisons[(comparisons.shots==1000)&(comparisons.baseline==baseline)]
            cases=sorted(data.case_id.unique())
            for method,g in data.groupby("ceo"):
                if baseline=="qeb":
                    g=g[(g.delta_error_mHa<=1e-5)&(g.baseline_cnot>0)]
                    y=100*(1-g.ceo_cnot/g.baseline_cnot)
                else:
                    y=g.delta_error_mHa
                ax.scatter([cases.index(c) for c in g.case_id],y,label=method,color=COLORS[method])
            ax.set_xticks(range(len(cases)),cases,rotation=70,fontsize=6)
        ax.axhline(0,color="gray",ls=":")
        ax.set(ylabel="CNOT reduction (%) at no worse observed mean energy" if baseline=="qeb" else "CEO minus LUCJ error (mHa)",
               title="Retrospective; same M/R cap; CEO CNOT/depth constrained; actual d may differ")
        handles,_=ax.get_legend_handles_labels()
        if handles:ax.legend(fontsize=7)
        ax.grid(alpha=.2);save_figure(fig,directory/name)
    optima=pd.DataFrame(json.loads((root/"analysis/checkpoint_optima.json").read_text()))
    fig,ax=plt.subplots(figsize=(6,4))
    if len(optima):
        for method,g in optima[(optima.criterion=="C_min_qsci_error")&(optima.shots==1000)&optima.attained].groupby("method"):
            if method in ("ceo_ovp","ceo_mvp","qeb","qubit","gsd"):
                ax.hist(g.step,bins=np.arange(-.5,max(1,optima.step.max())+1.5),histtype="step",label=method,color=COLORS[method])
    ax.set(xlabel="Retrospective best-QSCI growth step",ylabel="Case-method count",
           title="Geometry points are not independent molecules")
    ax.legend(fontsize=7);save_figure(fig,directory/"optimal_checkpoint_distribution")
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for method,g in last.groupby("method"):
        x=[references[c]["hf_weight_fci"] for c in g.case_id]
        axes[0].scatter(x,g.mean_energy_error_mHa,s=18,label=method,color=COLORS[method])
        axes[1].scatter([-1000*references[c]["correlation_energy"] for c in g.case_id],g.mean_energy_error_mHa,s=18,label=method,color=COLORS[method])
    axes[0].set_xlabel("FCI HF determinant weight")
    axes[1].set_xlabel("RHF-FCI correlation energy magnitude (mHa)")
    for ax in axes:
        ax.set_ylabel("Final-checkpoint QSCI mean error (mHa)");ax.grid(alpha=.2)
    axes[0].legend(fontsize=5,ncol=2)
    fig.suptitle("Correlation diagnostics, not universal weak/strong regime certificates")
    save_figure(fig,directory/"correlation_dependence")
    front=pd.DataFrame(json.loads((root/"analysis/pareto_frontiers.json").read_text()))
    # Separate systems; no averaging unlike molecules' energies/resources together.
    for case,g in front[front.shots==1000].groupby("case_id"):
        fig,ax=plt.subplots(figsize=(6,4))
        for method,arm in g.groupby("method"):
            ax.scatter(arm.cnot_count,arm.mean_energy_error_mHa,s=12,alpha=.5,label=method,color=COLORS[method])
        edge=g[g.pareto_cnot_shots_error].sort_values("cnot_count")
        ax.scatter(edge.cnot_count,edge.mean_energy_error_mHa,facecolors="none",edgecolors="black",s=48,label="empirical nondominated")
        ax.set(xlabel="Sampler CNOT",ylabel="QSCI mean error (mHa)",title=f"{case}: M1000 observed frontier, not statistical dominance")
        ax.legend(fontsize=5,ncol=2);ax.grid(alpha=.2)
        save_figure(fig,root/"figures"/case/"pareto_frontier")


def main():
    import sys
    if '--scalable-study' in sys.argv and '--current-evidence' in sys.argv:
        from .current_figure_index import main as current_main
        old=sys.argv;sys.argv=[x for x in old if x not in ('--scalable-study','--current-evidence')]
        try:return current_main()
        finally:sys.argv=old
    if '--scalable-study' in sys.argv:
        from .research_figure_cli_v2 import main as research_main
        old=sys.argv;sys.argv=[x for x in old if x!='--scalable-study']
        try:return research_main()
        finally:sys.argv=old
    parser=argparse.ArgumentParser()
    parser.add_argument("--all",action="store_true")
    parser.add_argument("--output",default="results/quick")
    args=parser.parse_args()
    manifest=Path(args.output)/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("schema")=="sampler-study-A-v1":
        from .study_figures import plot_all as study_plots
        return study_plots(args.output)
    plot_all(args.output)


if __name__=="__main__":
    main()

