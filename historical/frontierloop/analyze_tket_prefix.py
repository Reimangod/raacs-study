"""Derived N37 audit, no recompilation or probability/energy rerun."""
import gzip
from io import BytesIO
import json
import sqlite3
from time import process_time,perf_counter
import numpy as np
from qiskit import qpy
from .paths import ROOT
from .io import load,save,sha256

RUN=ROOT/'runs/N37-TKET-FIXED-PREFIX-v1'
FIELDS=['cnot_count','cnot_depth','total_unitary_depth']


def main():
    cpu,wall=process_time(),perf_counter()
    terminal=load(RUN/'terminal.json');spec=load(RUN/'freeze.json')
    d=sqlite3.connect('file:'+str(RUN/'compiled.sqlite')+'?mode=ro',uri=True)
    if d.execute('pragma integrity_check').fetchall()!=[('ok',)]:raise ValueError('SQLite integrity')
    rows=[json.loads(r[0]) for r in d.execute('select record from members order by case_id,idx')]
    expected={(r['case'],r['index']):r for r in spec['identities']}
    checked=0;maxerr=0.
    for r in rows:
        if r['member_id']!=expected[r['case'],r['index']]['member_id']:raise ValueError('Source identity')
        if not r['policy']['no_increase_vs_strong_generic'] or r['policy']['selection_uses_energy_or_counts']:
            raise ValueError('Frozen selection guard')
        for variant,b in d.execute('select variant,qpy_gz from circuits where case_id=? and idx=?',(r['case'],r['index'])):
            if sha256(b)!=r['qpy_gz_SHA256'][variant]:raise ValueError('Compiled QPY hash')
            q=qpy.load(BytesIO(gzip.decompress(b)))[0]
            actual=[int(q.count_ops().get('cx',0)),int(q.depth(filter_function=lambda x:x.operation.name=='cx')),int(q.depth())]
            if actual!=[r['resources'][variant][k] for k in FIELDS]:raise ValueError('Resource recount mismatch')
            c=r['independent_MPS_check']['circuits'][variant]
            if c['status']!='INDEPENDENT_UNTRUNCATED_MPS_PROBABILITY_PASSED':raise ValueError('Missing probability proof')
            maxerr=max(maxerr,c['probability_L1_with_leakage_bound'])
            if c['probability_L1_with_leakage_bound']>spec['probability_L1_tolerance']:raise ValueError('Probability tolerance')
            checked+=1
    d.close();cases=[]
    for case in [x['case'] for x in spec['sources']]:
        rr=[r for r in rows if r['case']==case]
        if not rr:
            cases.append({'case':case,'completed':0,'planned':4,'status':'NOT_EXECUTED_NOT_SCIENTIFIC_FAILURE'});continue
        old=[]
        for r in rr:
            safe=[n for n in ('NATIVE','RUSTIQ') if all(r['resources'][n][k]<=r['resources']['NATIVE'][k] for k in FIELDS)]
            n=min(safe,key=lambda n:tuple(r['resources'][n][k] for k in FIELDS)+(n,));old.append(r['resources'][n])
        means={label:{k:float(np.mean([a[k] for a in values])) for k in FIELDS}
               for label,values in [('NATIVE',[r['resources']['NATIVE'] for r in rr]),('OLD_SAFE_NATIVE_RUSTIQ',old),
                                     ('STRONG_GENERIC',[r['policy']['control_resources'] for r in rr]),
                                     ('SELECTED_WITH_PROJECTED',[r['policy']['selected_resources'] for r in rr])]}
        reductions={base:{k:100*(1-means['SELECTED_WITH_PROJECTED'][k]/means[base][k]) if means[base][k] else None
                          for k in FIELDS} for base in ('NATIVE','OLD_SAFE_NATIVE_RUSTIQ','STRONG_GENERIC')}
        cases.append({'case':case,'completed':len(rr),'planned':4,'means':means,'reduction_percent':reductions,
                      'selected':[{ 'index':r['index'],'control':r['policy']['control'],'selected':r['policy']['selected']} for r in rr],
                      'compile_CPU_seconds':sum(sum(x['CPU_seconds'] for x in r['compile_cost'].values()) for r in rr)})
    summary={'status':terminal['status'],'terminal':terminal,'planned_members':16,'completed_members':len(rows),
             'actual_saved_QPYs_independently_recounted':checked,'max_probability_L1_including_leakage':maxerr,
             'cases':cases,'new_model_QPU_shots':[0,0],'new_CI_energy_optimizer':[0,0,0],
             'parameter_rank':'UNKNOWN_NO_PARAMETER_REDUCTION_CLAIM',
             'scope':'FIRST4_ONLY_EQUAL_MEMBER_MEAN_NOT_WHOLE783BANK_OR_SHOT_WEIGHTED_METRICS;LOGICAL_ALLTOALL_OPT1',
             'cost_boundary':'PARENT_PROCESS_CPU_PLUS_INTERNAL_MPS_CPU_RECORDED;COLD_IMPORT_TRANSFER_ALL_BUILD_DEVELOPMENT_COST_NOT_COMPLETE',
             'new_energy_or_recovery_improvement_claimed':False,'best_paper_superiority_proven':False,
             'run_bytes':sum(p.stat().st_size for p in RUN.iterdir() if p.is_file()),
             'audit_CPU_seconds':process_time()-cpu,'audit_wall_seconds':perf_counter()-wall}
    save(ROOT/'analysis/N37-tket-prefix-summary.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':main()
