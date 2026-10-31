"""T3 fixed alternative classical partitions; no new quantum observations."""
import argparse,json,os,traceback
from pathlib import Path
from time import process_time,perf_counter
import xml.etree.ElementTree as ET
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,sha256,digest,space
from .native_trim_screen import binding as t1_binding
from .pooled_selection_screen import binding as t2_binding
from .readout_seed_plan import make,evaluation_key
from .pool_ranking import selections
from .trim_readout import Solver
from .root_catalog import Catalog

PROTOCOL=ROOT/'protocol-T3-READOUT-SEEDS.json'
FREEZE=ROOT/'freeze-T3-READOUT-SEEDS.json'
RUN=ROOT/'runs/T3-READOUT-SEEDS-v1'


def plans(settings,t1):
    result=[]
    kernels={name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in
        ['lowest_penalized.py','lowest_orbital.py','hole_rdm.py','energy_cache.py']}
    for seed in settings['seeds']:
        for source in t1['sources']:
            meta=load(OLD/'data/physical'/source['case']/'problem.json')
            old_request=load(ROOT/'runs/T1-NATIVE-TRIM-v1'/source['case']/source['method_id']/'request.json')
            hf=old_request['HF_single_spin_string'];n=meta['norb'];ne=(meta['nalpha'],meta['nbeta'])
            with np.load(source['payload'],allow_pickle=False) as a:
                request=make(a['blocked_ids'],a['counts'],n,ne,hf,seed)
            context=dict(problem_id=source['expected_problem_id'],physical_SHA256=source['physical_SHA256'],
                norb=n,nelec=list(ne),spin_penalty_Ha=.1,spin_target_S_squared=0,
                orbital_optimization=False,solver_kernels_SHA256=kernels,solver_seed=20261025)
            cell_id=digest([source['case'],source['method_id'],seed,source['payload_SHA256'],context,request])
            result.append(dict(cell_id=cell_id,source=source,context=context,HF_string=hf,request=request))
    return result


def binding():
    t1=load(ROOT/'freeze-T1-NATIVE-TRIM.json');t2=load(ROOT/'freeze-T2-POOLED-SELECTION.json')
    if t1_binding()!=t1 or t2_binding()!=t2:raise RuntimeError('Old native/kernel/source evidence drift')
    qualifiers={}
    for filename,count in [('qualification-readout-seed-plan-v1-tests.xml',5),('qualification-root-catalog-v1-tests.xml',3),
                           ('qualification-readout-seed-integration-v1-tests.xml',1)]:
        p=ROOT/filename;suites=list(ET.parse(p).getroot().iter('testsuite'))
        if sum(int(x.get('tests',0)) for x in suites)!=count or any(int(x.get(k,0)) for x in suites for k in ['errors','failures','skipped']):
            raise RuntimeError('Native request/cache qualification incomplete')
        qualifiers[filename]=sha256(p.read_bytes())
    return dict(protocol_SHA256=sha256(PROTOCOL.read_bytes()),T1_freeze_SHA256=sha256((ROOT/'freeze-T1-NATIVE-TRIM.json').read_bytes()),
        T2_freeze_SHA256=sha256((ROOT/'freeze-T2-POOLED-SELECTION.json').read_bytes()),qualifiers=qualifiers,
        implementation_SHA256={str(p):sha256(p.read_bytes()) for p in [Path(__file__),ROOT/'frontierloop/readout_seed_plan.py',
            ROOT/'frontierloop/root_catalog.py',ROOT/'test_readout_seed_plan.py',ROOT/'test_root_catalog.py']},
        plans=plans(load(PROTOCOL),t1),new_model_shots=0)


def register_legacy(catalog,frozen):
    """Previously paid exact same-basis roots, never oracle FCI information."""
    sources={p['source']['case']+'/'+p['source']['method_id']:p for p in frozen['plans']}
    for p in sources.values():
        source,context=p['source'],p['context']
        path=ROOT/'runs/T1-NATIVE-TRIM-v1'/source['case']/source['method_id']/'result.json'
        for i,row in enumerate(load(path)['calls']):
            catalog.put(context,row,str(path)+'#calls/'+str(i),row['CPU_seconds'])
    for p in sources.values():
        s,ctx=p['source'],p['context'];row=load(s['R2_baseline'])
        catalog.put(ctx,row,s['R2_baseline'],row['incremental_CPU_seconds'])
    for p in load(ROOT/'freeze-T2-POOLED-SELECTION.json')['plans']:
        s=p['source'];ctx=sources[s['case']+'/'+s['method_id']]['context']
        endpoint=ROOT/'runs/T2-POOLED-SELECTION-v1'/s['case']/s['method_id']/'terminal.json'
        for arm in load(endpoint)['rows']:
            if arm['status']!='COMPLETED':continue
            origin=arm['evaluation_origin'];path,sep,index=origin.partition('#calls/')
            data=load(path);row=data['calls'][int(index)] if sep else data
            catalog.put(ctx,row,origin,arm['CI_numeric_stage_CPU_seconds_inherited_or_measured'])


def evaluate(plan,settings,catalog):
    source=plan['source'];case=source['case'];method=source['method_id'];seed=plan['request']['seed']
    target=RUN/'cells'/plan['cell_id'];context=plan['context'];requests=plan['request']
    save(target/'request.json',plan);save(target/'started.json',dict(pid=os.getpid(),cell_id=plan['cell_id']))
    cpu,wall=process_time(),perf_counter();calls=[];rows=[];new_count=0
    physical=OLD/'data/physical'/case
    with np.load(physical/'integrals.npz',allow_pickle=False) as a:h,g=a['h1'],a['eri']
    meta=load(physical/'problem.json');ne=tuple(context['nelec']);n=context['norb']
    solver=Solver(h,g,meta['core_energy'],ne,dict(r_cap=128,spin_penalty_Ha=.1,
        per_CI_wall_cap_seconds=settings['per_CI_wall_cap_seconds'],
        total_wall_cap_seconds=settings['per_cell_wall_cap_seconds'],total_CPU_cap_seconds=settings['per_cell_CPU_cap_seconds']))

    def get(strings,purpose,need_weights=False):
        nonlocal new_count
        space();key=evaluation_key(context,strings);found=catalog.find(key,need_weights)
        reused=found is not None
        if not reused:
            remaining=min(settings['per_cell_CPU_cap_seconds']-(process_time()-cpu),settings['per_cell_wall_cap_seconds']-(perf_counter()-wall))
            if remaining<=0:raise TimeoutError('Fixed T3 source-seed work cap')
            solver.settings['per_CI_wall_cap_seconds']=min(settings['per_CI_wall_cap_seconds'],remaining)
            tick,start=process_time(),perf_counter()
            cur=catalog.db.execute('INSERT INTO attempts(lookup,cell,status) VALUES(?,?,?)',(key,plan['cell_id'],'STARTED'));aid=cur.lastrowid
            catalog.db.commit()
            try:
                ss=np.asarray(strings,np.int64);solver([(ss,ss)],h,g,n,ne)
                row=solver.calls[-1];origin='T3_NEW_ATTEMPT:'+str(aid)
                catalog.put(context,row,origin,row['CPU_seconds'],True);new_count+=1
                catalog.db.execute('UPDATE attempts SET status=?,cpu=?,wall=? WHERE seq=?',
                    ('COMPLETED',process_time()-tick,perf_counter()-start,aid));catalog.db.commit()
            except Exception as exc:
                catalog.db.execute('UPDATE attempts SET status=?,cpu=?,wall=?,error=? WHERE seq=?',
                    ('FAILED_ENGINEERING_PRESERVED',process_time()-tick,perf_counter()-start,traceback.format_exc(),aid));catalog.db.commit()
                raise
            found=catalog.find(key,need_weights)
        calls.append(dict(purpose=purpose,key=key,observation_id=found['id'],origin=found['origin'],
            actual_new_solve=not reused,needs_marginals=need_weights,CI_strings=strings))
        save(target/'progress.json',dict(calls=calls,new_CI_solves=new_count,CPU_seconds=process_time()-cpu))
        return key,found

    try:
        screens=[];screen_keys=[]
        if requests['screening_required']:
            for i,strings in enumerate(requests['screens']):
                key,entry=get(strings,'SCREEN_'+str(i),True);screen_keys.append(key)
                screens.append({**entry['record'],'round':0,'batch':i})
            with np.load(source['payload'],allow_pickle=False) as a:
                selection=selections(screens,a['blocked_ids'],a['counts'],n,plan['HF_string'])
            arms=selection['arms']
            if arms['SAME_UNION_FREQUENCY']!=requests['same_union_frequency']:raise RuntimeError('Outcome-free frequency request drift')
        else:
            arms={name:requests['universe'] for name in settings['arms']}
            selection=dict(arms=arms,universe=requests['universe'],screening_skipped=True,
                reason='ALL_UNION_STRINGS_FIT_R128_NO_CI_SCORES_NEEDED',no_invented_marginal_weights=True)
        # Endpoint spaces fixed before the new endpoint energy observations.
        save(target/'selection.json',selection);rows=[]
        baseline=load(source['R2_baseline']);oldkey=evaluation_key(context,baseline['CI_strings'])
        for arm in settings['arms']:
            strings=arms[arm]
            if len(strings)**2!=requests['final_actual_CI_dimension']:raise RuntimeError('Matched actual CI dimension violated')
            key,entry=get(strings,'FINAL_'+arm);root=entry['record']
            required=[key]+(screen_keys if arm!='SAME_UNION_FREQUENCY' else [])
            rows.append(dict(arm=arm,status='COMPLETED',CI_strings=strings,energy_Ha=root['energy_Ha'],
                penalized_energy_Ha=root['penalized_energy_Ha'],S2=root['spin_square'],
                actual_CI_dimension=len(strings)**2,observation_id=entry['id'],evaluation_origin=entry['origin'],
                required_work=catalog.work(required),required_screen_keys=list(dict.fromkeys(screen_keys)) if arm!='SAME_UNION_FREQUENCY' else [],
                all_source_observations_reused=True,new_model_shots=0,new_QPU_shots=0))
            save(target/'endpoint-progress.json',dict(rows=rows,new_CI_solves=new_count))
        result=dict(status='COMPLETED',case=case,method=method,seed=seed,cell_id=plan['cell_id'],
            request_digest=requests['request_digest'],physical_problem_id=context['problem_id'],
            source_counts_SHA256=source['payload_SHA256'],rows=rows,calls=calls,
            screening_required=requests['screening_required'],screening_unique_basis_count=len(set(screen_keys)),
            all_arms_identical_basis=len({tuple(x) for x in arms.values()})==1,
            old_R2_energy_Ha=baseline['energy_Ha'],old_R2_S2=baseline['spin_square'],
            old_R2_actual_CI_dimension=baseline['actual_CI_dimension'],old_R2_historical_CPU_seconds=baseline['incremental_CPU_seconds'],
            old_R2_canonical_work=catalog.work([oldkey]),new_CI_solves=new_count,
            new_actual_CPU_seconds=process_time()-cpu,new_actual_wall_seconds=perf_counter()-wall,
            same_counts_not_independent_quantum_confirmation=True,goal_complete=False)
        save(target/'result.json',result)
    except Exception as exc:
        result=dict(status='CI_OR_CELL_CAP_PRESERVED' if isinstance(exc,TimeoutError) else 'FAILED_ENGINEERING_PRESERVED',
            case=case,method=method,seed=seed,cell_id=plan['cell_id'],error=repr(exc),traceback=traceback.format_exc(),
            calls=calls,rows=rows,new_CI_solves=new_count,new_actual_CPU_seconds=process_time()-cpu,
            new_actual_wall_seconds=perf_counter()-wall,new_model_shots=0,not_negative_quantum_energy=True)
        save(target/'failure.json',result)
    print('T3',case,method,seed,result['status'],'NEW_CI',new_count,flush=True)
    return {k:result[k] for k in ['case','method','seed','cell_id','status','new_CI_solves']}


def run():
    frozen=load(FREEZE)
    if binding()!=frozen:raise RuntimeError('T3 freeze changed')
    space();RUN.mkdir(exist_ok=False);cpu,wall=process_time(),perf_counter()
    save(RUN/'started.json',dict(pid=os.getpid(),freeze_SHA256=sha256(FREEZE.read_bytes()),new_model_shots=0))
    catalog=Catalog(RUN/'evaluations.sqlite');states=[];settings=load(PROTOCOL)
    try:
        register_legacy(catalog,frozen)
        for p in frozen['plans']:
            space()
            if process_time()-cpu>settings['max_new_CPU_seconds'] or perf_counter()-wall>settings['max_new_wall_seconds']:
                raise TimeoutError('T3 global cap; preserve partial evidence, no blind retry')
            if sum(x.stat().st_size for x in RUN.rglob('*') if x.is_file())>settings['output_cap_bytes']:
                raise OSError('T3 compact storage cap')
            states.append(evaluate(p,settings,catalog));save(RUN/'progress.json',dict(states=states,
                planned=48,terminal=len(states),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall))
        attempts=catalog.db.execute('SELECT status,count(*),sum(cpu) FROM attempts GROUP BY status').fetchall()
        new_roots=catalog.db.execute('SELECT count(*) FROM observations WHERE is_new=1').fetchone()[0]
    finally:catalog.close()
    save(RUN/'terminal.json',dict(status='ALL48_FIXED_CLASSICAL_SEED_CELLS_TERMINAL',states=states,
        new_completed_CI_roots=new_roots,attempts=attempts,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        catalog_SHA256=sha256((RUN/'evaluations.sqlite').read_bytes()),new_model_shots=0,new_QPU_shots=0,
        independent_quantum_confirmation=False,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:
        value=binding()
        if FREEZE.exists() and load(FREEZE)!=value:raise RuntimeError('Never replace T3 freeze')
        save(FREEZE,value);print('T3_48_NATIVE_REQUEST_CELLS_FROZEN_NO_ENERGY_EVALUATION')
    else:run()
