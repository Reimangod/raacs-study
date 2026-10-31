"""N40P retrospective all-paid-samples diagnostic. No changes to original N40."""
import argparse
from collections import Counter
from pathlib import Path
from time import process_time,perf_counter
import numpy as np
from .paths import ROOT
from .io import load,save,write,arrays,sha256
from . import direct_sample_pilot as src
from .finite_source_readout import rank_record,basis_key
from .joint_ci_sparse import solve
from .analyze_finite_source_readout import audit_root

RUN=ROOT/'runs/N40P-PAID-SAMPLE-POOL-DIAGNOSTIC-v1'


def binding():
    p=src.verified();t=load(src.RUN/'terminal.json');a=ROOT/'analysis/N40-direct-sample-summary.json'
    if t['failure'] or t['actual_model_shots']!=33000 or load(a)['distinct_roots_audited']!=33:
        raise RuntimeError('Original full pilot must be preserved and audited')
    return {'implementation_SHA256':sha256(Path(__file__).read_bytes()),
            'original_protocol_SHA256':sha256((src.RUN/'protocol.json').read_bytes()),
            'original_terminal_SHA256':sha256((src.RUN/'terminal.json').read_bytes()),
            'original_negative_summary_SHA256':sha256(a.read_bytes()),
            'arms':['TRAIN_POOL_4500','ALL_PAID_POOL_5500'],'max_R':32,'max_D':1024,
            'draw_ids_SHA256':{d['draw_id']:d['SHA256'] for d in t['draws']},
            'selection':'UNCHANGED_FREQUENCY_SYMMETRIZED_HALF_RANK_NO_ENERGY_OR_FCI_RANKING',
            'new_model_QPU_shots':[0,0],'maximum_new_roots':6,'root_seconds':20,'global_wall_seconds':60,
            'maximum_new_storage_bytes':2**20,'scope':'RETROSPECTIVE_HYPOTHESIS_GENERATION_NOT_NEW_INDEPENDENT_CONFIRMATION'}


def run():
    spec=load(RUN/'freeze.json')
    if spec!=binding():raise ValueError('Pooled diagnostic freeze drift')
    if (RUN/'started.json').exists():raise RuntimeError('No duplicate diagnostic')
    p=src.verified();old=load(src.RUN/'terminal.json');d=next(x['physical'] for x in p['catalog_all4_definitions'] if x['source']['case']==src.CASE)
    cpu,wall=process_time(),perf_counter();save(RUN/'started.json',{'freeze_SHA256':sha256((RUN/'freeze.json').read_bytes())})
    with np.load(d['local_integrals'],allow_pickle=False) as z:h,g=z['h1'],z['eri']
    points={src.point_id(x['point']):x for x in p['points']};rows=[];attempts=[]
    for rep in src.REPS:
        baseline=next(x for x in old['states'] if x['role']=='BASELINE' and x['arm']=='R32_CAP' and x['replicate']==rep)
        for arm in spec['arms']:
            if perf_counter()-wall>60:raise TimeoutError('Bounded no-shot pool analysis')
            selected=[x for x in old['draws'] if x['replicate']==rep and x['role'] in (('TRAIN',) if arm=='TRAIN_POOL_4500' else ('TRAIN','FINAL'))]
            if len({x['draw_id'] for x in selected})!=len(selected):raise ValueError('Double-counted draws')
            pooled=Counter();weighted={k:0. for k in ('cnot_count','cnot_depth','total_unitary_depth')}
            for draw in selected:
                if sha256(Path(draw['path']).read_bytes())!=draw['SHA256']:raise ValueError('Bound raw draw changed')
                with np.load(draw['path'],allow_pickle=False) as z:
                    for x,n in zip(z['blocked_ids'],z['counts']):pooled[int(x)]+=int(n)
                for k in weighted:weighted[k]+=draw['shots']*points[src.point_id(draw['point'])]['resources'][k]
            ids=np.array(sorted(pooled),np.uint64);counts=np.array([pooled[int(x)] for x in ids],np.uint64)
            shots=int(counts.sum());expected=4500 if arm=='TRAIN_POOL_4500' else 5500
            if shots!=expected:raise ValueError('Pool budget/duplicate mismatch')
            record=rank_record(ids,counts,d['norb'],d['nelec'][0]);R=record['R32'];strings=sorted(record['rank'][:R])
            pairs=np.array([(a,b) for a in strings for b in strings],np.uint64);key=basis_key(d,pairs,p['library_SHA256'])
            path=RUN/'roots'/(key+'.npz');previous=src.RUN/'roots'/(key+'.json')
            if previous.exists():value=load(previous);new=False
            else:
                tick=process_time();result,v=solve(h,g,d['core_energy'],tuple(d['nelec']),pairs,p['library'],seconds=20)
                write(path,arrays(joint_pairs=pairs,coefficients=v));value={'status':'COMPLETED','basis_id':key,'result':result,
                    'root_path':str(path),'root_SHA256':sha256(path.read_bytes()),'attempt_CPU_seconds':process_time()-tick}
                save(path.with_suffix('.json'),value);attempts.append(value);new=True
            row=dict(value,case=src.CASE,replicate=rep,arm=arm,R=R,actual_D=R*R,half_strings=strings,
                     source_draw_ids=[x['draw_id'] for x in selected],pooled_model_samples=shots,
                     new_quantum_samples=0,new_CI_attempt=new,shot_weighted_resources={k:v/shots for k,v in weighted.items()},
                     different_source_circuits=len({src.point_id(x['point']) for x in selected}),
                     original_5500_baseline_energy_Ha=baseline['result']['energy_Ha'],
                     delta_vs_original_baseline_mHa=1000*(value['result']['energy_Ha']-baseline['result']['energy_Ha']),
                     delta_penalized_vs_original_baseline_mHa=1000*(value['result']['penalized_energy_Ha']-baseline['result']['penalized_energy_Ha']))
            row['saved_vector_check']=audit_root(row,d,p['library']);rows.append(row)
            save(RUN/'progress.json',{'rows':rows,'attempts':attempts})
    if len(rows)!=6:raise ValueError('All predeclared diagnostic points required')
    summary={'status':'ALL6_RETROSPECTIVE_POOLED_RESULTS_AND_ROOTS_AUDITED','rows':rows,'actual_new_CI_roots':len(attempts),
             'new_model_QPU_shots':[0,0],'old_quantum_model_samples_reused_not_new':33000,
             'CPU_seconds_including_saved_vector_audit':process_time()-cpu,'wall_seconds':perf_counter()-wall,
             'means_delta_vs_5500baseline_mHa':{a:float(np.mean([r['delta_vs_original_baseline_mHa'] for r in rows if r['arm']==a])) for a in spec['arms']},
             'selection_and_final_data_not_independent':True,'retrospective_not_confirmatory':True,
             'same_training_data_can_define_a_valid_variational_operational_output':True,
             'performance_requires_fresh_full_pipeline_repetitions':True,'original_negative_result_changed':False,
             'best_paper_superiority_proven':False,'parameter_rank':'UNKNOWN'}
    if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file())>2**20:raise OSError('Bounded new pool evidence')
    save(RUN/'terminal.json',summary);save(ROOT/'analysis/N40P-pooled-diagnostic-summary.json',summary)
    print({k:v for k,v in summary.items() if k!='rows'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args()
    if a.action=='freeze':
        if (RUN/'freeze.json').exists():raise RuntimeError('Already frozen')
        save(RUN/'freeze.json',binding());print('N40P both pools/all3reps frozen before new CI')
    else:run()
