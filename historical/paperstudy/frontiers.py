"""Retrospective observed-grid frontiers; no invented interpolated circuits."""
from collections import defaultdict
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from .io import ROOT,save_json
from .analysis_products import write_csv

METRICS=('CNOT','CNOT_depth','total_depth','coefficient_slots_or_member_mean')
RESOURCE_CI_CAPS=(8,16,32,64)
AXIS_LABELS={'CNOT':'CNOT count','CNOT_depth':'CNOT depth','total_depth':'Total depth',
             'coefficient_slots_or_member_mean':'Parameter coordinate slots / member mean'}


def nondominated(points,metric):
    return [p for p in points if not any(q['energy']<=p['energy']and q[metric]<=p[metric]
        and(q['energy']<p['energy']or q[metric]<p[metric])for q in points)]


def short_label(method):
    if method=='A_RAACS_CURRENT':return 'A'
    if method=='D_RUSTIQ_RESOURCE_CONTROL':return 'D-R'
    if method.startswith('D_SQDRIFT_N'):return 'D'+method.split('_N',1)[1].replace('-M','/')
    if method.startswith(('B_LUCJ','C_LUCJ','E_LUCJ')):return method[0]+method.rsplit('_L',1)[1]
    return method


def place_labels(ax,specs,renderer):
    """Greedy display-only label placement. Never changes points or limits."""
    bounds=ax.get_window_extent(renderer).padded(-2);occupied=[];annotations=[]
    markers=[]
    for _,xy in specs:
        x,y=ax.transData.transform(xy);markers.append(Bbox.from_extents(x-4,y-4,x+4,y+4))
    positions=[(dx,dy)for dy in(3,-3,12,-12,22,-22,32,-32)for dx in(3,-3)]
    def area(box):return max(0,box.width)*max(0,box.height)
    def intersection(a,b):
        return max(0,min(a.x1,b.x1)-max(a.x0,b.x0))*max(0,min(a.y1,b.y1)-max(a.y0,b.y0))
    for label,xy in specs:
        ann=ax.annotate(label,xy,xytext=(3,3),textcoords='offset points',fontsize=6,
            arrowprops={'arrowstyle':'-','color':'#777777','linewidth':.4,'shrinkA':1,'shrinkB':3})
        ann.set_in_layout(False)
        candidates=[]
        for order,(dx,dy)in enumerate(positions):
            ann.set_position((dx,dy));ann.set_ha('left'if dx>0 else'right');ann.set_va('bottom'if dy>0 else'top')
            ann.update_positions(renderer);box=Text.get_window_extent(ann,renderer).padded(1)
            score=(10*max(0,area(box)-intersection(box,bounds))+sum(intersection(box,b)for b in occupied)
                +sum(intersection(box,b)for b in markers))
            candidates.append((score,order,dx,dy))
        _,_,dx,dy=min(candidates)
        ann.set_position((dx,dy));ann.set_ha('left'if dx>0 else'right');ann.set_va('bottom'if dy>0 else'top')
        ann.update_positions(renderer);ann.arrow_patch.set_visible(abs(dy)>8)
        occupied.append(Text.get_window_extent(ann,renderer).padded(1));annotations.append(ann)
    return annotations


def run():
    root=ROOT/'analysis-v1';source=root/'all-group-statistics.csv'
    if not source.exists():return
    with source.open()as f:rows=list(csv.DictReader(f))
    groups=defaultdict(list)
    for r in rows:
        if int(r['observed_runs'])!=int(r['planned_runs']):continue
        p={'case_id':r['case_id'],'budget':int(r['budget']),'r_cap':int(r['r_cap']),'method_id':r['method_id'],
            'lane':r['lane'],'energy':float(r['mean_energy_Ha']),'energy_minus_HF_mHa':float(r['mean_energy_minus_HF_mHa']),
            'energy_SD_Ha':float(r['std_energy_Ha'])if r.get('std_energy_Ha')else None,'n':int(r['observed_runs'])}
        p.update({m:float(r['mean_'+m])for m in METRICS})
        groups[(p['case_id'],p['budget'],p['r_cap'])].append(p)
        if r['method_id']=='D_SQDRIFT_N20'and int(r.get('Rustiq_observed_runs')or 0)==int(r['planned_runs'])and r.get('mean_Rustiq_cnot_count'):
            alternative={**p,'method_id':'D_RUSTIQ_RESOURCE_CONTROL','lane':'S_RESOURCE_ONLY_COMPILER_CONTROL',
                'CNOT':float(r['mean_Rustiq_cnot_count']),'CNOT_depth':float(r['mean_Rustiq_cnot_depth']),
                'total_depth':float(r['mean_Rustiq_total_unitary_depth']),
                'same_outcomes_as':'D_SQDRIFT_N20','new_model_shots':0,'not_an_additional_independent_experiment':True}
            groups[(p['case_id'],p['budget'],p['r_cap'])].append(alternative)
    frontier=[];thresholds=[]
    for(case,B,r),points in sorted(groups.items()):
        for metric in METRICS:
            frontier +=[{**p,'objective':metric,'retrospective_grid_only':True,
                'statistical_superiority_or_energy_equivalence_certified':False}for p in nondominated(points,metric)]
            # Every observed mean-energy target is shown. Each feasible choice
            # is an ACTUALLY evaluated setting, not a point interpolated between
            # circuits. This is an oracle descriptive lookup, not a stopping rule.
            for target in sorted(set(p['energy']for p in points)):
                for method in sorted(set(p['method_id']for p in points)):
                    candidates=[p for p in points if p['method_id']==method and p['energy']<=target]
                    best=min(candidates,key=lambda p:(p[metric],p['energy'],p['method_id']))if candidates else None
                    thresholds.append({'case_id':case,'budget':B,'r_cap':r,'target_mean_energy_Ha':target,'method_id':method,
                        'resource_objective':metric,'observed_feasible':best is not None,'selected_actual_resource':None if best is None else best[metric],
                        'selected_actual_mean_energy_Ha':None if best is None else best['energy'],'retrospective_oracle_not_prospective_rule':True,
                        'measurement_budget_shared_in_this_comparison':True,'same_target_not_statistically_proven_equal_energy':True})
    write_csv(root/'retrospective-observed-resource-frontiers.csv',frontier)
    write_csv(root/'same-energy-target-observed-feasible-lookup.csv',thresholds)
    colors={'A':'#1b6ca8','B':'#b7472a','C':'#26834a','D':'#864b98','E':'#d18b00'};figures=[]
    for case in sorted(set(k[0]for k in groups)):
        for r in sorted(set(k[2]for k in groups if k[0]==case)):
            if r not in RESOURCE_CI_CAPS:continue
            fig,axes=plt.subplots(3,4,figsize=(13,9),layout='constrained')
            label_specs=[]
            for i,B in enumerate((10000,100000,1000000)):
                points=groups.get((case,B,r),[])
                for j,metric in enumerate(METRICS):
                    ax=axes[i,j];ids={id(p)for p in nondominated(points,metric)}
                    specs=[]
                    for p in points:
                        ax.scatter(p[metric],p['energy_minus_HF_mHa'],color=colors[p['method_id'][0]],
                            marker='o'if p['lane']=='C_PRIMARY'else'^',s=35 if id(p)in ids else 20,
                            edgecolor='black'if id(p)in ids else'none',linewidth=.9)
                        specs.append((short_label(p['method_id']),(p[metric],p['energy_minus_HF_mHa'])))
                    ax.set_xlabel(AXIS_LABELS[metric],fontsize=8);ax.set_ylabel('CI energy - HF (mHa)',fontsize=8)
                    ax.set_title('B='+str(B)+', observed settings='+str(len(points)),fontsize=9);ax.grid(alpha=.2)
                    if points:ax.margins(x=.12,y=.12)
                    else:
                        ax.text(.5,.5,'No evaluated full-seed grid point\nSee planned status; not energy zero',
                            transform=ax.transAxes,ha='center',va='center',fontsize=9,color='#555555')
                        ax.set_xticks([]);ax.set_yticks([])
                    label_specs.append((ax,specs))
            fig.suptitle(case+' | r='+str(r)+' | evaluated means only; black edge=observed 2D frontier\nA RAACS; B naive LUCJ; C compressed LUCJ; D SqDRIFT; D-R same counts with Rustiq resources; E TN. Not statistical superiority.',fontsize=9)
            fig.canvas.draw();renderer=fig.canvas.get_renderer()
            for ax,specs in label_specs:place_labels(ax,specs,renderer)
            name=case+'-observed-frontier-r'+str(r)
            for ext in('png','svg'):fig.savefig(root/(name+'.'+ext),dpi=180)
            plt.close(fig);figures.append(name+'.png')
    save_json(root/'frontier-status.json',{'observed_grid_groups':len(groups),'frontier_points':len(frontier),
        'resource_plot_CI_caps':list(RESOURCE_CI_CAPS),
        'all_target_lookup_rows':len(thresholds),'figures':figures,'no_interpolated_circuits':True,
        'only_full_planned_seed_groups_in_frontier':True,'all_partial_or_missing_cells_remain_in_original_coverage':True,
        'statistical_equivalence_or_global_optimality_claimed':False,'whole_experiment_complete':False})
    print(json.dumps({'frontier_points':len(frontier),'figures':len(figures),'whole_experiment_complete':False}),flush=True)


if __name__=='__main__':run()
