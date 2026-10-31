"""V2 construction-aware Fig8: empty-panel safe, original numerical tables unchanged.

Only completed, independently audited RAACS/B1/B2/B3 comparison is read. No
solver, state preparation, optimization or draws. Fig18 is NOT substituted by
a partial-cost chart: total classical time remains unknown when logs cannot
prove it. All registered seeds are aggregated, never best-seed selected.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def family(method):
    method=re.sub(r'-s\d+-(exact_small|mps)$',r'-\1',method)
    return re.sub(r'[_-]s\d+$','',method)


def build_budget(method,result):
    """Model shot semantics, not actual QPU usage or simulator objective calls."""
    if result.get('config',{}).get('parent_checkpoint_id') is not None:
        return None,'UNKNOWN_UPSTREAM_TRAINED_TOPOLOGY_COST'
    if method.startswith('RAACS:'):
        return int(result['pilot_raw_model_shots']),'ALL_REGISTERED_CONSTRUCTION_PILOTS'
    if result.get('spec',{}).get('objective')=='vqe':
        return None,'UNKNOWN_VQE_OBJECTIVE_SHOTS'
    if ':source_final_' in method:
        name=method.split(':source_final_')[1]
        if name in ('hf','lucj_1','lucj_2','lucj_truncated','lucj_random_101'):
            return 0,'FIXED_CLASSICAL_INITIALIZER_NO_BUILD_MEASUREMENTS'
        return None,'UNKNOWN_ADAPT_OR_TRAINED_TOPOLOGY_SHOTS'
    value=result.get('construction_simulated_raw_shots',result.get('simulated_construction_shots'))
    if value is None:return None,'NO_MEASUREMENT_ACCOUNTING_RECORD'
    if isinstance(value,bool) or int(value)!=value or value<0:raise ValueError('Invalid construction model budget')
    return int(value),'RECORDED_CONSTRUCTION_MODEL_SHOTS'


def budget_view(scope,M,final,build):
    if not np.isfinite(final) or final<=0:return None,'NO_VALID_FINAL_DRAW'
    if scope=='readout_only':
        if final!=M:raise ValueError('Readout budget mismatch')
    elif scope=='known_build_plus_readout':
        if build is None:return None,'UNKNOWN_BUILD_CANNOT_ENTER_FULL_BUDGET_VIEW'
        if final+build!=M:raise ValueError('Build plus final budget mismatch')
    else:raise ValueError('Unknown comparison scope')
    return (None,'UNKNOWN_BUILD_MEASUREMENT_BUDGET') if build is None else (int(final+build),'KNOWN_MODEL_BUILD_PLUS_FINAL')


def nondominated(values):
    """Strict empirical minimization; missing coordinates never become zero."""
    a=np.asarray(values,float)
    if a.ndim!=2:raise ValueError('Matrix of objectives required')
    ok=np.isfinite(a).all(axis=1);out=np.zeros(len(a),bool)
    for i in np.flatnonzero(ok):
        out[i]=not np.any(np.all(a[ok]<=a[i],axis=1)&np.any(a[ok]<a[i],axis=1))
    return out


def constructor_time(method,result):
    """Recorded scope only. No unknown dependency is synthesized as zero."""
    if method.startswith('RAACS:'):
        return result['actual_wall_seconds'],'RAACS_RECORDED_RUN_INCLUDING_ORIGINAL_FINAL_DIAGNOSTIC'
    if ':source_final_' in method:
        return result.get('trajectory_seconds'),'HISTORICAL_TRAJECTORY_UNMATCHED_DEVICE_SCOPE'
    seconds=result.get('construction_wall_seconds')
    scope=method.split(':')[0]+'_RECORDED_CONSTRUCTION'
    init=result.get('initialization') or {}
    if method.startswith('B1:') and init:
        if seconds is None or not all(k in init for k in ('dependency_total_seconds','dependency_new_seconds')):
            return None,'B1_STANDALONE_DEPENDENCY_COST_UNKNOWN'
        seconds=seconds+init['dependency_total_seconds']-init['dependency_new_seconds']
        scope='B1_RECORDED_STANDALONE_DEPENDENCY_ESTIMATE'
    if seconds is not None and (not np.isfinite(seconds) or seconds<0):
        raise ValueError('Invalid recorded constructor time')
    return seconds,scope


def complete_mean(values):
    """No favorable mean from only the observed subset of registered costs."""
    return float(values.mean()) if values.notna().all() else None


def build(output,mode='standard',R=64):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():
        old=json.loads((output/'manifest.json').read_text())
        assert old['script_sha256']==sha(__file__)
        for p,v in old['source_sha256'].items():assert sha(ROOT/p)==v
        for p,v in old['output_sha256'].items():assert sha(output/p)==v
        return old
    start=time.perf_counter();inputs={}
    # Read-only derivative budget; does NOT authorize production below its floor.
    if shutil.disk_usage(output).free<64*2**20:raise RuntimeError('Report needs64MiB free; no source deletion permitted')
    def bind(p):p=Path(p);inputs[str(p.relative_to(ROOT))]=sha(p);return p
    def read(p):return json.loads(bind(p).read_text())
    root=ROOT/'results/RAACS-independent-comparison-standard-sharded-v1'
    cfg=read(root/'manifest.json');validation=read(root/'validation.json')
    assert validation['status']=='RAACS_INDEPENDENT_COMPARISON_COUNTS_AND_PAIRS_VALIDATED'
    correction=read(root/'cost-corrected-report-v2/cost-correction.json')
    assert correction['raw_data_or_energy_changed'] is False
    frame=pd.read_csv(bind(root/'cost-corrected-report-v2/all-cell-common-R-summary-cost-corrected.csv'))
    frame=frame[frame.R==R].copy()
    cases=[c for c in cfg['cases'] if mode=='standard' or c in ('H2_R0.7','H4_R1.1')]
    frame=frame[frame.case_id.isin(cases)]
    registry=sorted(pd.read_csv(bind(root/'cost-corrected-report-v2/all-cell-common-R-summary-cost-corrected.csv'),usecols=['method']).method.unique())
    attrs=[];timings=[];statuses=[]
    for case in cases:
        dbpath=bind(root/'readouts'/case/'readouts.sqlite')
        with sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True) as db:
            cells={m:(kind,json.loads(meta)) for m,kind,meta in db.execute('SELECT method,kind,metadata FROM cells')}
            timing=db.execute('''SELECT l.method,l.scope,l.total_M,avg(d.seconds),avg(o.lookup_seconds)
              FROM links l JOIN draws d ON d.id=l.draw JOIN outcomes o ON o.draw=d.id
              WHERE o.cap=? AND o.rule='frequency' AND l.status='AVAILABLE'
              GROUP BY l.method,l.scope,l.total_M''',(R,)).fetchall()
        timings.extend(dict(case_id=case,method=m,scope=s,M=M,recorded_readout_access_seconds=t,recorded_CI_lookup_or_solve_seconds=ci) for m,s,M,t,ci in timing)
        records={r['method']:r for r in read(root/'readouts'/case/'source-records.json')}
        for method in registry:
            path=Path(cells[method][1]['result_path']) if method in cells else Path(records[method]['result_path']) if method in records else None
            if path is None:
                statuses.append(dict(case_id=case,method=method,family=family(method),status='SOURCE_UNAVAILABLE_PRESERVED',scientific_failure=False))
                continue
            r=read(path)
            expected=cells[method][1].get('result_sha256') if method in cells else records[method]['sha256']
            assert sha(path)==expected
            status=r.get('status','SOURCE_SAVED_FINAL')
            statuses.append(dict(case_id=case,method=method,family=family(method),status=status,scientific_failure=False))
            if method not in cells:continue
            build_shots,reason=build_budget(method,r)
            resources=[]
            if method.startswith('RAACS:'):
                events=bind(path.parent/'events.sqlite')
                assert sha(events)==r['event_database_sha256']
                with sqlite3.connect(f'file:{events}?mode=ro',uri=True) as db:
                    resources=[json.loads(db.execute('SELECT resources FROM circuits WHERE id=?',(a['circuit_id'],)).fetchone()[0]) for a in r['final_circuit_allocations'] if a['shots']>0]
            else:
                resources=[r['resources']]
            seconds,cost_scope=constructor_time(method,r)
            assert resources,'A saved sampler must have an actual registered component'
            attrs.append(dict(case_id=case,method=method,family=family(method),
                construction_model_shots=build_shots,construction_budget_status=reason,
                construction_recorded_seconds=seconds,classical_cost_scope=cost_scope,
                component_cnot_max=max(x['cnot_count'] for x in resources),
                component_cnot_depth_max=max(x['cnot_depth'] for x in resources),
                component_total_depth_max=max(x['total_depth'] for x in resources),
                component_registered_parameters_max=max(x['parameter_count'] for x in resources),
                final_component_count=len(resources),peak_process_RSS_bytes=r.get('peak_process_rss_bytes',r.get('peak_process_rss')),
                native_2Q_count=None,native_2Q_depth=None,native_status='NOT_RECONSTRUCTED_FOR_THIS_COMPARISON',
                total_classical_wall_seconds=None,total_classical_status='NO_COMPLETE_SAME_SCOPE_END_TO_END_WALL_TRACE',
                historical_build_incurred=True,result_path=str(path.relative_to(ROOT))))
    attributes=pd.DataFrame(attrs);attributes.to_csv(output/'sampler-resource-and-cost-records.csv',index=False)
    status=pd.DataFrame(statuses);status.to_csv(output/'all-planned-sampler-statuses.csv',index=False)
    data=frame.drop(columns=['family']).merge(attributes,on=['case_id','method'],how='left',validate='many_to_one')
    assert data.result_path.notna().all(),'Every plotted row must bind an available source result'
    timing=pd.DataFrame(timings)
    if len(timing):data=data.merge(timing,on=['case_id','method','scope','M'],how='left',validate='one_to_one')
    totals=[];budget_status=[]
    for r in data.itertuples():
        build_shots=int(r.construction_model_shots) if pd.notna(r.construction_model_shots) else None
        total,why=budget_view(r.scope,int(r.M),r.mean_final_shots,build_shots)
        totals.append(total);budget_status.append(why)
    data['total_model_shots_per_application']=totals;data['total_model_budget_status']=budget_status
    data['known_model_shot_lower_bound']=data.mean_final_shots+data.construction_model_shots.fillna(0)
    data['hardware_shots']=0
    data['cost_application_scope']='one saved-constructor application plus one conditional final draw; NOT whole-study sum'
    data.to_csv(output/'all-budget-resource-metrics.csv',index=False)
    # Group every registered construction seed before any frontier filtering.
    expected=status.groupby(['case_id','family']).size().to_dict()
    groups=[]
    for (case,fam,scope,M),g in data.groupby(['case_id','family','scope','M']):
        complete=len(g)==expected[(case,fam)]
        known=complete and g.total_model_shots_per_application.notna().all()
        groups.append(dict(case_id=case,molecule=g.molecule.iloc[0],family=fam,scope=scope,M=int(M),R=R,
            registered_samplers_expected=expected[(case,fam)],available_samplers=len(g),all_registered_seeds_available=complete,
            mean_error_mHa=g.mean_error_mHa.mean(),mean_conditional_success=g.success_fraction.mean(),
            total_model_shots_seed_mean=g.total_model_shots_per_application.mean() if known else None,
            readout_model_shots_seed_mean=g.mean_final_shots.mean(),
            worst_registered_component_depth=g.component_cnot_depth_max.max(),
            worst_registered_component_cnot=g.component_cnot_max.max(),
            recorded_construction_seconds_seed_mean=complete_mean(g.construction_recorded_seconds) if complete else None,
            classical_cost_scope=g.classical_cost_scope.iloc[0],full_model_budget_known=known,
            empirical_p90_feasible=bool(complete and g.success_fraction.mean()>=.9),
            total_classical_wall_seconds=None,
            all_seed_mean_is_not_unconditional_success_certificate=True))
        assert g.classical_cost_scope.nunique()==1,'Seed family mixes incompatible constructor-time scopes'
    grouped=pd.DataFrame(groups);grouped['depth_budget_empirical_nondominated']=False;grouped['scoped_construction_budget_empirical_nondominated']=False
    feasible=grouped.empirical_p90_feasible & grouped.full_model_budget_known
    for _,g in grouped[feasible].groupby(['case_id','scope']):
        grouped.loc[g.index,'depth_budget_empirical_nondominated']=nondominated(g[['total_model_shots_seed_mean','worst_registered_component_depth']])
    for _,g in grouped[feasible].groupby(['case_id','scope','classical_cost_scope']):
        grouped.loc[g.index,'scoped_construction_budget_empirical_nondominated']=nondominated(g[['total_model_shots_seed_mean','recorded_construction_seconds_seed_mean']])
    grouped.to_csv(output/'all-seed-family-frontiers.csv',index=False)
    missing=pd.read_csv(bind(root/'derived-report-v1/all-budget-missingness.csv'))
    missing[missing.case_id.isin(cases)].to_csv(output/'preserved-budget-missingness.csv',index=False)
    images=[];palette={'RAACS':'#2563a6','B1':'#b05c23','B2exact':'#278452','B3':'#7857a6'}
    for case in cases:
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file())>32*2**20:
            raise RuntimeError('Read-only derivative32MiB cap reached; partial files preserved')
        # Full-budget plots: exclude unknown and infeasible rows, retain in CSV.
        g=grouped[(grouped.case_id==case)&(grouped.scope=='known_build_plus_readout')]
        fig,axes=plt.subplots(1,2,figsize=(10.5,5))
        for cohort,color in palette.items():
            q=g[g.family.str.startswith(cohort+':') & g.full_model_budget_known & g.empirical_p90_feasible]
            for ax,col,front in [(axes[0],'worst_registered_component_depth','depth_budget_empirical_nondominated'),
                                  (axes[1],'recorded_construction_seconds_seed_mean','scoped_construction_budget_empirical_nondominated')]:
                ax.scatter(q.total_model_shots_seed_mean,q[col],s=25,color=color,alpha=.7,label=cohort)
                e=q[q[front]]
                ax.scatter(e.total_model_shots_seed_mean,e[col],s=65,facecolors='none',edgecolors=color,linewidths=1.5)
        for ax,label in zip(axes,('Worst registered component CNOT depth','Recorded construction seconds (scope-stratified)')):
            ax.set(xlabel='Known construction + final MODEL shots / application',ylabel=label,xscale='log',title=case)
            ax.set_xlim(min(cfg['M'])*.8,max(cfg['M'])*1.25)
            ax.grid(alpha=.15)
            if ax.collections and any(len(x.get_offsets()) for x in ax.collections):ax.legend(fontsize=7)
            else:ax.text(.5,.5,'No qualifying known-budget point\nSee all-status / missingness tables',transform=ax.transAxes,ha='center',va='center')
        note=f'R{R}; feasibility = mean conditional observed success >=90%, NOT a confidence guarantee. All registered seeds averaged; no best seed.\nRings: empirical nondominance; right panel computed WITHIN construction-time scope only. Unknown VQE/trained-topology budgets excluded, not zero.\nThese are hypothetical per-application model budgets, not hardware shots or total project cost. Right panel is NOT full classical wall time.'
        fig.text(.015,.01,note,fontsize=7);fig.tight_layout(rect=(0,.17,1,.97))
        for ext in ('png','svg','pdf'):
            p=output/f'{case}_Fig08_known_model_budget_frontier.{ext}';fig.savefig(p,dpi=160,facecolor='white');images.append(p.name)
        plt.close(fig)
    # No invented Fig18: complete classical end-to-end timing is not recoverable.
    unknown_count=int(grouped.total_classical_wall_seconds.isna().sum())
    report=['# Construction-aware model-budget frontier and remaining total-cost gap','',
        f'Cases{len(cases)}; planned sampler-case identities{len(status)}; grouped R{R} budget endpoints{len(grouped)}. No new draws/states/optimizer/reference solves.',
        '', '## What this proves and does not prove',
        '- Fig8 plots genuine completed-study data under the registered chemical-accuracy success label. At least90% observed conditional success is a descriptive feasibility rule, not certified90% algorithmic success.',
        '- All registered construction seeds are aggregated before filtering. A missing seed prevents family-level eligibility. No best seed, geometry or unregistered layer is selected.',
        '- Known construction pilots plus final readout are counted once per hypothetical application. R variants do not multiply physical shots. These rows are NOT a whole-study sum or additional acquisitions.',
        '- Historical parent-trained CEO random topology remains UNKNOWN upstream measurement cost. VQE calls/time are not converted into QPU shots. Unknown budget rows never enter a known-total-budget frontier.',
        '- CNOT/Depth are logical Z-sampling resources; a multi-circuit sampler reports the worst registered component for depth, not a fabricated fused circuit. All raw component source identities remain available.',
        '- The right panel is recorded construction time, stratified by recording scope. It includes environment/cache/simulator effects, and is not controlled speedup or an end-to-end QPU/classical-time comparison.',
        f'- Full total-classical-wall is UNKNOWN for all{unknown_count} family endpoints: recorded draw access can include cache retrieval rather than original acquisition, CI lookup can reuse old diagonalization, and construction logs have different scope. These measured components cannot establish complete from-scratch workflow time.',
        '- Therefore requested Fig18 is STILL NOT PRODUCED/CERTIFIED. A partial construction-cost plot is not renamed total classical+quantum Pareto. This file explicitly preserves the missing requirement.',
        '- Original source artifacts, failed cells and budget-infeasible results are unchanged. Hardware shots0. No universal sampler superiority or whole-study completion.','', '## Files',
        '- all-budget-resource-metrics.csv: source-level readout/error/success/dimension/resources/cost fields; unknowns remain empty.',
        '- all-seed-family-frontiers.csv: every seed group, including nonqualifying and unknown-budget endpoints.',
        '- all-planned-sampler-statuses.csv and preserved-budget-missingness.csv: failures, unavailable sources and infeasible budgets.',
        '- sampler-resource-and-cost-records.csv: measured construction scopes and source paths.','']
    for stem in sorted({Path(p).stem for p in images}):report.append(f'- {stem}: [PNG]({stem}.png) / [SVG]({stem}.svg) / [PDF]({stem}.pdf)')
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')
    result={'schema':'construction-aware-known-budget-frontier-v2','producer_file':'benchmark/final_cost_frontiers_v2.py','parent_producer_sha256':sha(ROOT/'benchmark/final_cost_frontiers.py'),'mode':mode,'R':R,'cases':cases,'planned_sampler_case_identities':len(status),
        'source_level_budget_rows':len(data),'all_seed_family_endpoints':len(grouped),'fig08_sets':len(images)//3,'Fig18_status':'MISSING_COMPLETE_TOTAL_CLASSICAL_WALL_TRACE',
        'new_draws':0,'hardware_shots':0,'new_reference_solves':0,'whole_goal_complete':False,
        'new_reporting_seconds':time.perf_counter()-start,'source_sha256':dict(sorted(inputs.items())),'script_sha256':sha(__file__),
        'output_sha256':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='manifest.json'}}
    (output/'manifest.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--mode',choices=('quick','standard'),default='standard');a=p.parse_args();r=build(a.output,a.mode)
    print(json.dumps({k:r[k] for k in ('mode','planned_sampler_case_identities','source_level_budget_rows','all_seed_family_endpoints','fig08_sets','Fig18_status','whole_goal_complete')}))
