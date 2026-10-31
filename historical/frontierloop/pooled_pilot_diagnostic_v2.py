"""N40P-v2: same six predeclared pooled endpoints, deduplicated five bases."""
from collections import Counter
from pathlib import Path
from time import process_time,perf_counter
import numpy as np
from . import pooled_pilot_diagnostic as prior
from . import direct_sample_pilot as src
from .paths import ROOT
from .io import load,save,write,arrays,sha256
from .finite_source_readout import rank_record,basis_key
from .joint_ci_sparse import solve
from .analyze_finite_source_readout import audit_root

RUN=ROOT/'runs/N40P-PAID-SAMPLE-POOL-DIAGNOSTIC-v2'


def tasks():
    p=src.verified();old=load(src.RUN/'terminal.json')
    d=next(x['physical'] for x in p['catalog_all4_definitions'] if x['source']['case']==src.CASE)
    points={src.point_id(x['point']):x for x in p['points']};out=[]
    for rep in src.REPS:
        for arm in ('TRAIN_POOL_4500','ALL_PAID_POOL_5500'):
            ds=[r for r in old['draws'] if r['replicate']==rep and r['role'] in (('TRAIN',) if arm=='TRAIN_POOL_4500' else ('TRAIN','FINAL'))]
            if len({r['draw_id'] for r in ds})!=len(ds):raise ValueError('Double counted raw event')
            counts=Counter();weighted={k:0. for k in ('cnot_count','cnot_depth','total_unitary_depth')}
            for row in ds:
                if sha256(Path(row['path']).read_bytes())!=row['SHA256']:raise ValueError('Preserved counts changed')
                with np.load(row['path'],allow_pickle=False) as z:
                    for x,c in zip(z['blocked_ids'],z['counts']):counts[int(x)]+=int(c)
                for k in weighted:weighted[k]+=row['shots']*points[src.point_id(row['point'])]['resources'][k]
            ids=np.array(sorted(counts),np.uint64);n=np.array([counts[int(x)] for x in ids],np.uint64);shots=int(n.sum())
            if shots!=(4500 if arm=='TRAIN_POOL_4500' else 5500):raise ValueError('Registered pool budget')
            rank=rank_record(ids,n,d['norb'],d['nelec'][0]);R=rank['R32'];strings=sorted(rank['rank'][:R])
            pairs=np.array([(a,b) for a in strings for b in strings],np.uint64);key=basis_key(d,pairs,p['library_SHA256'])
            baseline=next(x for x in old['states'] if x['replicate']==rep and x['role']=='BASELINE' and x['arm']=='R32_CAP')
            out.append({'case':src.CASE,'replicate':rep,'arm':arm,'R':R,'actual_D':R*R,'half_strings':strings,'basis_id':key,
                        'source_draw_ids':[x['draw_id'] for x in ds],'pooled_model_samples':shots,
                        'shot_weighted_resources':{k:v/shots for k,v in weighted.items()},
                        'different_source_circuits':len({src.point_id(x['point']) for x in ds}),
                        'original_5500_baseline_energy_Ha':baseline['result']['energy_Ha'],
                        'original_5500_baseline_penalized_energy_Ha':baseline['result']['penalized_energy_Ha']})
    return p,d,out


def binding():
    if (prior.RUN/'started.json').exists():raise RuntimeError('v1 unexpectedly executed')
    if prior.binding()!=load(prior.RUN/'freeze.json'):raise ValueError('Previous diagnostic/source drift')
    _,_,rows=tasks()
    return {'implementation_SHA256':sha256(Path(__file__).read_bytes()),
            'prior_v1_freeze_SHA256':sha256((prior.RUN/'freeze.json').read_bytes()),
            'correction':'PRE_OUTCOME_BASIS_DEDUP_TWO_ENDPOINTS_SHARE_A_BASIS_NO_REPEATED_SOLVE_OR_OVERWRITE',
            'v1_execution':'SUPERSEDED_BEFORE_ANY_NEW_CI_NOT_A_SCIENTIFIC_NEGATIVE',
            'tasks':rows,'unique_basis_ids':sorted({r['basis_id'] for r in rows}),
            'new_model_QPU_shots':[0,0],'max_new_roots':5,'root_seconds':20,'global_wall_seconds':60,
            'scope':'SAME_SIX_RETROSPECTIVE_ENDPOINTS_NOT_CONFIRMATORY'}


def run():
    spec=load(RUN/'freeze.json')
    if binding()!=spec:raise ValueError('N40P-v2 drift')
    if (RUN/'started.json').exists():raise RuntimeError('No duplicate v2')
    p,d,records=tasks();cpu,wall=process_time(),perf_counter()
    save(RUN/'started.json',{'freeze_SHA256':sha256((RUN/'freeze.json').read_bytes())})
    with np.load(d['local_integrals'],allow_pickle=False) as z:h,g=z['h1'],z['eri']
    rows=[];attempts=[];cache={}
    for task in records:
        if perf_counter()-wall>60:raise TimeoutError('No-shot diagnostic cap')
        key=task['basis_id'];new=False
        if key not in cache:
            old=src.RUN/'roots'/(key+'.json')
            if old.exists():value=load(old)
            else:
                pairs=np.array([(a,b) for a in task['half_strings'] for b in task['half_strings']],np.uint64)
                tick=process_time();result,v=solve(h,g,d['core_energy'],tuple(d['nelec']),pairs,p['library'],seconds=20)
                file=RUN/'roots'/(key+'.npz');write(file,arrays(joint_pairs=pairs,coefficients=v))
                value={'status':'COMPLETED','basis_id':key,'result':result,'root_path':str(file),
                       'root_SHA256':sha256(file.read_bytes()),'attempt_CPU_seconds':process_time()-tick}
                save(file.with_suffix('.json'),value);attempts.append(value);new=True
            cache[key]=value
        value=cache[key];row=dict(task,**{k:v for k,v in value.items() if k!='basis_id'},new_CI_attempt=new,new_quantum_samples=0)
        row['saved_vector_check']=audit_root(row,d,p['library'])
        row['delta_vs_original_baseline_mHa']=1000*(value['result']['energy_Ha']-task['original_5500_baseline_energy_Ha'])
        row['delta_penalized_vs_original_baseline_mHa']=1000*(value['result']['penalized_energy_Ha']-task['original_5500_baseline_penalized_energy_Ha'])
        rows.append(row);save(RUN/'progress.json',{'rows':rows,'actual_new_roots':len(attempts)})
    summary={'status':'ALL6_RETROSPECTIVE_POOLED_ENDPOINTS_FIVE_BASES_AUDITED','rows':rows,
             'unique_bases':len(cache),'actual_new_CI_roots':len(attempts),'new_model_QPU_shots':[0,0],
             'old_model_samples_reused_not_new':33000,'CPU_seconds_including_root_audit':process_time()-cpu,'wall_seconds':perf_counter()-wall,
             'means_delta_vs_5500baseline_mHa':{a:float(np.mean([r['delta_vs_original_baseline_mHa'] for r in rows if r['arm']==a]))
                                               for a in ('TRAIN_POOL_4500','ALL_PAID_POOL_5500')},
             'retrospective_not_confirmatory':True,'original_negative_result_changed':False,
             'best_paper_superiority_proven':False,'parameter_rank':'UNKNOWN','v1_new_energy_evaluations':0}
    save(RUN/'terminal.json',summary);save(ROOT/'analysis/N40P-pooled-diagnostic-summary-v2.json',summary)
    print({k:v for k,v in summary.items() if k!='rows'})


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['freeze']:
        if (RUN/'freeze.json').exists():raise RuntimeError('Already frozen')
        save(RUN/'freeze.json',binding());print('N40P v2 same6 endpoints/five unique bases frozen, v1 not executed')
    else:run()
