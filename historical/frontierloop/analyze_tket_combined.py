"""Independent artifact/resource audit of immutable N37 + exact SWAP successor N38."""
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
from .tket_prefix_study import select,FIELDS


def main():
    cpu,wall=process_time(),perf_counter()
    runs=[ROOT/'runs/N37-TKET-FIXED-PREFIX-v1',ROOT/'runs/N38-TKET-FE-SUCCESSOR-v1']
    specs=[load(p/'freeze.json') for p in runs];terms=[load(p/'terminal.json') for p in runs]
    if terms[0]['completed_members']!=12 or terms[1]['completed_members']!=4 or terms[1]['failure']:
        raise RuntimeError('All16 are not verified yet')
    if sha256((runs[0]/'compiled.sqlite').read_bytes())!=specs[1]['old_N37_db_SHA256']:
        raise ValueError('Immutable old evidence changed')
    if sha256((runs[0]/'terminal.json').read_bytes())!=specs[1]['old_N37_terminal_SHA256']:
        raise ValueError('Old failure/cost erased')
    expected={(s['case'],m['index']):m for s in specs[0]['sources'] for m in s['members']}
    for s in specs[0]['sources']:
        if sha256(Path(s['meta_path']).read_bytes())!=s['meta_SHA256'] or sha256(Path(s['source_result']).read_bytes())!=s['source_result_SHA256']:
            raise ValueError('Actual physical problem/source binding drift')
    rows=[];seen=set();nq=0;maxerr=0.
    for run in runs:
        d=sqlite3.connect('file:'+str(run/'compiled.sqlite')+'?mode=ro',uri=True)
        if d.execute('pragma integrity_check').fetchall()!=[('ok',)]:raise ValueError('Database integrity')
        for case,idx,text in d.execute('select * from members order by case_id,idx'):
            key=(case,idx);r=json.loads(text)
            if key in seen or key not in expected or r['member_id']!=expected[key]['member_id']:
                raise ValueError('Duplicated/unplanned/mismatched member')
            seen.add(key)
            if select(r['resources'])!=r['policy']:raise ValueError('Frozen selection mismatch')
            variants=list(d.execute('select variant,qpy_gz from circuits where case_id=? and idx=?',(case,idx)))
            if len(variants)!=9:raise ValueError('Missing circuit variants')
            if r['independent_MPS_check']['versions']!={'qiskit':'2.6.0','qiskit_aer':'0.18.0'}:
                raise ValueError('Independent verifier version drift')
            for name,b in variants:
                if sha256(b)!=r['qpy_gz_SHA256'][name]:raise ValueError('Actual artifact hash')
                q=qpy.load(BytesIO(gzip.decompress(b)))[0]
                actual=[int(q.count_ops().get('cx',0)),int(q.depth(filter_function=lambda i:i.operation.name=='cx')),int(q.depth())]
                if actual!=[r['resources'][name][k] for k in FIELDS]:raise ValueError('Independent resource recount')
                c=r['independent_MPS_check']['circuits'][name]
                if c['status']!='INDEPENDENT_UNTRUNCATED_MPS_PROBABILITY_PASSED':raise ValueError('Probability proof absent')
                if c['probability_L1_with_leakage_bound']>1e-8:raise ValueError('Probability mismatch')
                maxerr=max(maxerr,c['probability_L1_with_leakage_bound']);nq+=1
            r['run']=run.name;rows.append(r)
        d.close()
    if seen!=set(expected) or nq!=144:raise ValueError('Incomplete planned cohort')
    cases=[]
    for s in specs[0]['sources']:
        rr=[r for r in rows if r['case']==s['case']]
        means={label:{k:float(np.mean([v[k] for v in vv])) for k in FIELDS}
               for label,vv in [('NATIVE',[r['resources']['NATIVE'] for r in rr]),
                                ('STRONG_GENERIC',[r['policy']['control_resources'] for r in rr]),
                                ('SELECTED_WITH_PROJECTED',[r['policy']['selected_resources'] for r in rr])]}
        cases.append({'case':s['case'],'completed':len(rr),'means':means,
                      'reduction_percent_vs_strong_generic':{k:100*(1-means['SELECTED_WITH_PROJECTED'][k]/means['STRONG_GENERIC'][k]) for k in FIELDS},
                      'reduction_percent_vs_native':{k:100*(1-means['SELECTED_WITH_PROJECTED'][k]/means['NATIVE'][k]) for k in FIELDS},
                      'projected_used':sum(r['policy']['selected'].startswith('PROJECTED') for r in rr),
                      'strict_CNOT_improved_members':sum(r['policy']['selected_resources']['cnot_count']<r['policy']['control_resources']['cnot_count'] for r in rr),
                      'compile_CPU_seconds':sum(sum(x['CPU_seconds'] for x in r['compile_cost'].values()) for r in rr),
                      'members':[{'index':r['index'],'control':r['policy']['control'],'selected':r['policy']['selected'],
                                  'control_resources':r['policy']['control_resources'],
                                  'selected_resources':r['policy']['selected_resources']} for r in rr]})
    summary={'status':'ALL16_FIXED_PREFIX_MEMBERS_AND144_ACTUAL_QPYS_AUDITED',
             'planned_members':16,'completed_members':16,'actual_saved_QPYs_independently_recounted':144,
             'successful_new_compiles':96,'old_failed_member_attempts_at_least':1,
             'max_probability_L1_including_leakage':maxerr,'cases':cases,
             'parent_CPU_seconds_including_preserved_failure':sum(t['CPU_seconds_parent'] for t in terms),
             'wall_seconds_including_preserved_failure':sum(t['wall_seconds'] for t in terms),
             'verifier_internal_CPU_seconds':sum(t['verifier_CPU_seconds'] for t in terms),
             'N38_children_CPU_including_startup':terms[1]['CPU_seconds_children_including_startup'],
             'N37_children_startup_CPU':'UNKNOWN_NOT_ZERO_NOT_IN_INTERNAL_COUNTER',
             'old_failure_unchanged':True,'completed_old_members_rerun':0,
             'new_model_QPU_shots':[0,0],'new_CI_energy_optimizer':[0,0,0],
             'new_energy_or_recovery_improvement_claimed':False,'parameter_rank':'UNKNOWN_NO_PARAMETER_REDUCTION_CLAIM',
             'best_paper_superiority_proven':False,
             'scope':'FIXED_FIRST4_OF783_EACH4CASES_EQUAL_MEMBER_MEAN_NOT_WHOLEBANK;LOGICAL_ALLTOALL_OPT1;SAMPLING_ONLY',
             'run_bytes':sum(p.stat().st_size for r in runs for p in r.iterdir() if p.is_file()),
             'audit_CPU_seconds':process_time()-cpu,'audit_wall_seconds':perf_counter()-wall}
    save(ROOT/'analysis/N37-N38-tket-combined-summary.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':main()
