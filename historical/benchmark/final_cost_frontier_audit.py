"""Independent read-only reconstruction of the final construction-cost view.

No producer helpers, new quantum sampling, optimizer, solver or reference work.
Checks actual per-repeat energies and dimensions as well as raw source costs,
seed completeness, empirical nondominance, all output/source hashes and formats.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def equal(actual,expected,atol=1e-8):
    if expected is None:
        assert pd.isna(actual),(actual,expected)
    else:
        assert np.isclose(actual,expected,rtol=1e-11,atol=atol),(actual,expected)


def label(method):
    return re.sub(r'[_-]s\d+$','',re.sub(r'-s\d+-(exact_small|mps)$',r'-\1',method))


def direct_frontier(matrix):
    # Pairwise check, deliberately separate from the producer's vectorization.
    rows=np.asarray(matrix,float)
    return [bool(np.isfinite(a).all() and not any(
        np.isfinite(b).all() and all(y<=x for x,y in zip(a,b)) and any(y<x for x,y in zip(a,b))
        for b in rows)) for a in rows]


def audit(output):
    output=Path(output);m=json.loads((output/'manifest.json').read_text())
    destination=output/'audit-v1';destination.mkdir(exist_ok=True)
    target=destination/'validation.json'
    if target.exists():
        old=json.loads(target.read_text())
        assert old['auditor_sha256']==digest(__file__)
        assert old['input_manifest_sha256']==digest(output/'manifest.json')
        for p,s in old['additional_source_sha256'].items():assert digest(ROOT/p)==s
        for p,s in m['source_sha256'].items():assert digest(ROOT/p)==s
        for p,s in m['output_sha256'].items():assert digest(output/p)==s
        return old
    assert digest(ROOT/m.get('producer_file','benchmark/final_cost_frontiers.py'))==m['script_sha256']
    if 'parent_producer_sha256' in m:assert digest(ROOT/'benchmark/final_cost_frontiers.py')==m['parent_producer_sha256']
    for p,s in m['source_sha256'].items():assert digest(ROOT/p)==s
    for p,s in m['output_sha256'].items():assert digest(output/p)==s
    sources={}
    def read(p):
        p=Path(p);sources[str(p.relative_to(ROOT))]=digest(p);return json.loads(p.read_text())
    source=ROOT/'results/RAACS-independent-comparison-standard-sharded-v1'
    cfg=read(source/'manifest.json')
    costs=pd.read_csv(output/'sampler-resource-and-cost-records.csv')
    rows=pd.read_csv(output/'all-budget-resource-metrics.csv')
    grouped=pd.read_csv(output/'all-seed-family-frontiers.csv')
    statuses=pd.read_csv(output/'all-planned-sampler-statuses.csv')
    assert len(rows)==m['source_level_budget_rows']
    assert len(grouped)==m['all_seed_family_endpoints']
    assert len(statuses)==m['planned_sampler_case_identities']
    assert not rows.duplicated(['case_id','method','scope','M','R']).any()
    assert len(statuses)==82*len(m['cases']) and not statuses.duplicated(['case_id','method']).any()
    assert set(rows.case_id)==set(m['cases'])
    resource_fields={'component_cnot_max':'cnot_count','component_cnot_depth_max':'cnot_depth',
                     'component_total_depth_max':'total_depth','component_registered_parameters_max':'parameter_count'}
    for c in costs.itertuples():
        r=read(ROOT/c.result_path);method=c.method
        if method.startswith('RAACS:'):
            build=r['pilot_raw_model_shots'];rs=[]
            p=ROOT/c.result_path
            with sqlite3.connect(f'file:{p.parent/"events.sqlite"}?mode=ro',uri=True) as db:
                byid=dict(db.execute('SELECT id,resources FROM circuits'))
                for allocation in r['final_circuit_allocations']:
                    if allocation['shots']>0:rs.append(json.loads(byid[allocation['circuit_id']]))
            seconds=r['actual_wall_seconds']
        else:
            rs=[r['resources']]
            if r.get('config',{}).get('parent_checkpoint_id') is not None or r.get('spec',{}).get('objective')=='vqe':build=None
            elif ':source_final_' in method:
                name=method.split(':source_final_')[-1]
                build=0 if name in ('hf','lucj_1','lucj_2','lucj_truncated','lucj_random_101') else None
            else:build=r.get('construction_simulated_raw_shots',r.get('simulated_construction_shots'))
            if ':source_final_' in method:seconds=r.get('trajectory_seconds')
            else:
                seconds=r.get('construction_wall_seconds')
                init=r.get('initialization') or {}
                if method.startswith('B1:') and init:
                    seconds=None if seconds is None or not {'dependency_total_seconds','dependency_new_seconds'}.issubset(init) else seconds+init['dependency_total_seconds']-init['dependency_new_seconds']
        equal(c.construction_model_shots,build,0)
        equal(c.construction_recorded_seconds,seconds)
        for field,key in resource_fields.items():equal(getattr(c,field),max(x[key] for x in rs),0)
        assert c.final_component_count==len(rs)
    repeat_records=0;checked_rows=0
    for case in m['cases']:
        p=source/'readouts'/case/'individual-readouts.parquet';sources[str(p.relative_to(ROOT))]=digest(p)
        raw=pd.read_parquet(p);raw=raw[raw.cap==m['R']].copy()
        ref=read(Path(cfg['source'])/'references'/case/'reference.json')
        keycols=['method','scope','total_M']
        originals={key:g for key,g in raw.groupby(keycols)}
        for row in rows[rows.case_id==case].itertuples():
            g=originals[(row.method,row.scope,row.M)]
            assert len(g)==row.repeats==20 and g.rep.nunique()==20
            error=1000*(g.energy-ref['fci_energy'])
            equal(row.mean_error_mHa,error.mean())
            equal(row.success_fraction,(error<=1.5936014376).mean(),1e-12)
            equal(row.mean_actual_d,g.dimension.mean())
            equal(row.mean_final_shots,g.final_shots.mean(),0)
            equal(row.correlation_energy_recovery_percent,100*(ref['rhf_energy']-g.energy.mean())/(ref['rhf_energy']-ref['fci_energy']),1e-7)
            if row.scope=='known_build_plus_readout':
                assert pd.notna(row.construction_model_shots)
                equal(row.mean_final_shots+row.construction_model_shots,row.M,0)
            else:equal(row.mean_final_shots,row.M,0)
            equal(row.total_model_shots_per_application,None if pd.isna(row.construction_model_shots) else row.mean_final_shots+row.construction_model_shots,0)
            assert row.hardware_shots==0 and pd.isna(row.total_classical_wall_seconds)
            assert pd.isna(row.native_2Q_count) and pd.isna(row.native_2Q_depth)
            repeat_records+=len(g);checked_rows+=1
        # Reconstruct timing components directly, not from producer summaries.
        with sqlite3.connect(f'file:{source/"readouts"/case/"readouts.sqlite"}?mode=ro',uri=True) as db:
            observed=pd.read_sql_query('''SELECT l.method,l.scope,l.total_M,d.seconds,o.lookup_seconds FROM links l
                JOIN draws d ON l.draw=d.id JOIN outcomes o ON o.draw=l.draw
                WHERE l.status='AVAILABLE' AND o.rule='frequency' AND o.cap=?''',db,params=(m['R'],))
        for (method,scope,M),g in observed.groupby(['method','scope','total_M']):
            q=rows[(rows.case_id==case)&(rows.method==method)&(rows.scope==scope)&(rows.M==M)]
            if not len(q):continue # e.g. intentionally superseded unknown-cost full-budget row.
            assert len(q)==1
            equal(q.iloc[0].recorded_readout_access_seconds,g.seconds.mean())
            equal(q.iloc[0].recorded_CI_lookup_or_solve_seconds,g.lookup_seconds.mean())
    statuses['reconstructed_family']=statuses.method.map(label)
    assert (statuses.family==statuses.reconstructed_family).all()
    expected=statuses.groupby(['case_id','reconstructed_family']).size()
    for t in grouped.itertuples():
        part=rows[(rows.case_id==t.case_id)&(rows.family==t.family)&(rows.scope==t.scope)&(rows.M==t.M)]
        assert len(part)==t.available_samplers
        n=int(expected[(t.case_id,t.family)]);complete=len(part)==n
        assert t.registered_samplers_expected==n and t.all_registered_seeds_available==complete
        known=complete and part.total_model_shots_per_application.notna().all()
        assert t.full_model_budget_known==known
        equal(t.mean_error_mHa,part.mean_error_mHa.mean())
        equal(t.mean_conditional_success,part.success_fraction.mean())
        equal(t.total_model_shots_seed_mean,part.total_model_shots_per_application.mean() if known else None)
        equal(t.worst_registered_component_depth,part.component_cnot_depth_max.max(),0)
        equal(t.worst_registered_component_cnot,part.component_cnot_max.max(),0)
        equal(t.recorded_construction_seconds_seed_mean,part.construction_recorded_seconds.mean() if complete and part.construction_recorded_seconds.notna().all() else None)
        assert t.empirical_p90_feasible==bool(complete and part.success_fraction.mean()>=.9)
    for front,coordinates,keys in [
        ('depth_budget_empirical_nondominated',['total_model_shots_seed_mean','worst_registered_component_depth'],['case_id','scope']),
        ('scoped_construction_budget_empirical_nondominated',['total_model_shots_seed_mean','recorded_construction_seconds_seed_mean'],['case_id','scope','classical_cost_scope'])]:
        eligible=grouped.empirical_p90_feasible&grouped.full_model_budget_known
        assert not grouped.loc[~eligible,front].any()
        for _,part in grouped[eligible].groupby(keys):assert part[front].tolist()==direct_frontier(part[coordinates])
    figure_sets=0
    for case in m['cases']:
        for ext,signature in [('png',b'\x89PNG'),('svg',b'<?xml'),('pdf',b'%PDF')]:
            p=output/f'{case}_Fig08_known_model_budget_frontier.{ext}'
            with p.open('rb') as f:assert f.read(5).startswith(signature)
        figure_sets+=1
    assert figure_sets==m['fig08_sets']
    assert m['Fig18_status']=='MISSING_COMPLETE_TOTAL_CLASSICAL_WALL_TRACE'
    assert m['new_draws']==m['hardware_shots']==m['new_reference_solves']==0 and not m['whole_goal_complete']
    result={'status':'RAW_COSTS_REPEATS_SEED_COMPLETENESS_FRONTIERS_AND_HASHES_VALIDATED',
        'input_manifest_sha256':digest(output/'manifest.json'),'auditor_sha256':digest(__file__),
        'additional_source_sha256':dict(sorted(sources.items())),
        'raw_sampler_resources_and_costs_checked':len(costs),'planned_sampler_cases_checked':len(statuses),
        'raw_repeat_records_referenced':repeat_records,'source_level_budget_rows_checked':checked_rows,
        'all_seed_family_frontiers_checked':len(grouped),'figure_sets_checked':figure_sets,
        'new_draws':0,'hardware_shots':0,'new_solver_calls':0,'Fig18_still_missing':True,'whole_goal_complete':False}
    target.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    r=audit(a.output);print(json.dumps({k:v for k,v in r.items() if k!='additional_source_sha256'}))
