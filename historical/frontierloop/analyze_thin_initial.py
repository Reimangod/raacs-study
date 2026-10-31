"""N39 actual artifact audit and resource/cost comparison, no rerun."""
import gzip
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from time import process_time,perf_counter
import numpy as np
from qiskit import qpy
from .paths import ROOT
from .io import load,save,sha256
from .tket_prefix_study import FIELDS,select

RUN=ROOT/'runs/N39-THIN-INITIAL-COMPILER-v1'


def main():
    cpu,wall=process_time(),perf_counter();spec=load(RUN/'freeze.json');terminal=load(RUN/'terminal.json')
    if terminal['failure'] or terminal['completed_members']!=16:raise ValueError('Incomplete N39')
    for p,expected in spec['old_db_SHA256'].items():
        if sha256(Path(p).read_bytes())!=expected:raise ValueError('Preserved heavy database changed')
    expected={(s['case'],m['index']):m['member_id'] for s in spec['sources'] for m in s['members']}
    d=sqlite3.connect('file:'+str(RUN/'compiled.sqlite')+'?mode=ro',uri=True)
    if d.execute('pragma integrity_check').fetchall()!=[('ok',)]:raise ValueError('SQLite integrity')
    rows=[];seen=set();nq=0;maxerr=0.;heavy_zero_cpu=0.
    old=[sqlite3.connect('file:'+p+'?mode=ro',uri=True) for p in spec['old_db_SHA256']]
    for case,idx,text in d.execute('select * from members order by case_id,idx'):
        r=json.loads(text);key=(case,idx)
        if key in seen or expected.get(key)!=r['member_id']:raise ValueError('Source identity/duplication')
        seen.add(key)
        if r['policy']!=select(r['resources']):raise ValueError('Policy changed')
        heavy=[x for db in old if (x:=db.execute('select record from members where case_id=? and idx=?',key).fetchone())]
        if len(heavy)!=1:raise ValueError('Heavy same-member identity missing/duplicated')
        h=json.loads(heavy[0][0])
        if h['policy']!=r['heavy_policy'] or h['member_id']!=r['member_id']:raise ValueError('Wrong heavy comparator')
        heavy_zero_cpu+=sum(v['CPU_seconds'] for k,v in h['compile_cost'].items() if k.endswith('_ZERO_INPUT'))
        blobs=list(d.execute('select variant,qpy_gz from circuits where case_id=? and idx=?',key))
        if len(blobs)!=6:raise ValueError('Missing actual circuits')
        for name,b in blobs:
            if sha256(b)!=r['qpy_gz_SHA256'][name]:raise ValueError('QPY mismatch')
            q=qpy.load(BytesIO(gzip.decompress(b)))[0]
            metrics=[int(q.count_ops().get('cx',0)),int(q.depth(filter_function=lambda x:x.operation.name=='cx')),int(q.depth())]
            if metrics!=[r['resources'][name][k] for k in FIELDS]:raise ValueError('Actual resource count mismatch')
            check=r['independent_MPS_check']['circuits'][name]
            if check['status']!='INDEPENDENT_UNTRUNCATED_MPS_PROBABILITY_PASSED' or check['probability_L1_with_leakage_bound']>1e-8:
                raise ValueError('Actual source probability check failed')
            maxerr=max(maxerr,check['probability_L1_with_leakage_bound']);nq+=1
        rows.append(r)
    d.close()
    for db in old:db.close()
    if set(expected)!=seen or nq!=96:raise ValueError('All16/96 coverage failed')
    cases=[]
    for s in spec['sources']:
        rr=[r for r in rows if r['case']==s['case']]
        means={label:{k:float(np.mean([r[p][v][k] for r in rr])) for k in FIELDS}
               for label,p,v in [('thin_control','policy','control_resources'),('thin_selected','policy','selected_resources'),
                                 ('heavy_control','heavy_policy','control_resources'),('heavy_selected','heavy_policy','selected_resources')]}
        changes={base:{k:100*(means['thin_selected'][k]/means[base][k]-1) for k in FIELDS}
                 for base in ('thin_control','heavy_control','heavy_selected')}
        cases.append({'case':s['case'],'members':len(rr),'means':means,'thin_selected_change_percent':changes,
                      'thin_strict_CNOT_improvements_vs_thin_control':sum(r['policy']['selected_resources']['cnot_count']<r['policy']['control_resources']['cnot_count'] for r in rr),
                      'thin_no_metric_worse_vs_heavy_selected_members':sum(all(r['policy']['selected_resources'][k]<=r['heavy_policy']['selected_resources'][k] for k in FIELDS) for r in rr),
                      'thin_compile_CPU_seconds':sum(sum(v['CPU_seconds'] for v in r['compile_cost'].values()) for r in rr)})
    summary={'status':'ALL16_THIN_MEMBERS_96_ACTUAL_QPY_RESOURCES_PROBABILITIES_AUDITED',
             'terminal':terminal,'cases':cases,'completed_members':16,'actual_QPY_recounts':nq,
             'max_probability_L1_including_leakage':maxerr,
             'thin_48_successful_compile_CPU_seconds':sum(sum(v['CPU_seconds'] for v in r['compile_cost'].values()) for r in rows),
             'heavy_48_same_zero_input_compile_CPU_seconds':heavy_zero_cpu,
             'heavy_96_all_compile_CPU_seconds':sum(r['heavy_compile_CPU_seconds'] for r in rows),
             'scope':'SAME_FIRST4_OF783_EACH4CASE_NOT_WHOLEBANK;LOGICAL_ALLTOALL_OPT1;NO_NEW_ENERGY_RECOVERY',
             'parameter_rank':'UNKNOWN_NO_PARAMETER_REDUCTION_CLAIM','new_model_QPU_shots':[0,0],
             'new_energy_CI_optimizer':[0,0,0],'best_paper_superiority_proven':False,
             'run_bytes':sum(p.stat().st_size for p in RUN.iterdir() if p.is_file()),
             'audit_CPU_seconds':process_time()-cpu,'audit_wall_seconds':perf_counter()-wall}
    save(ROOT/'analysis/N39-thin-compiler-summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
