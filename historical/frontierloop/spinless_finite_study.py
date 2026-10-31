"""N35 all4 spinless source finite draws and all8-method common CI."""
from pathlib import Path
from time import process_time,perf_counter
import os,xml.etree.ElementTree as ET
import numpy as np
from .paths import ROOT
from .io import load,save,write,arrays,sha256,digest,space
from . import spinless_source_preflight_v2 as preflight
from . import analytical_source_readout as oldread
from . import analytical_source_sampling as oldsource
from . import finite_source_study as n29
from . import optimized_control_sampling as n30
from . import finite_source_readout as parent
from .joint_ci_sparse import solve

LOCAL=ROOT/'runs/N35-SPINLESS-FINITE-v1'
METHOD=preflight.METHOD
REPLICATES=(290101,290102,290103)


def seed(source_sha,case,rep):return int(digest(['N35_FRESH_SPINLESS_SEED_V1',source_sha,case,rep])[:16],16)%(1<<63)


def prepare():
    preflight.verified();op,oq,sub=oldread.verified();space()
    if LOCAL.exists():raise RuntimeError('Existing N35; preserve/reconcile')
    terminal=load(preflight.LOCAL/'terminal.json')
    if terminal['status']!='N34_ALL3_FIXTURES_AND4_SOURCE_PREFLIGHTS_PASSED':raise RuntimeError('All4 source preflights required')
    test=ROOT/'qualification-N34-terminal-tests.xml';suites=list(ET.parse(test).getroot().iter('testsuite'))
    if sum(int(s.get('tests',0)) for s in suites)!=3 or any(int(s.get('failures',0)) or int(s.get('errors',0)) for s in suites):raise RuntimeError('Actual three N34 tests required')
    oldrows=oldread.inventory(op,sub);requests=[]
    for s in terminal['sources']:
        d=op['physical'][s['case']]
        if s['definition']['problem_id']!=d['problem_id'] or s['definition']['integrals_SHA256']!=d['integrals_SHA256']:raise RuntimeError('Same physical problem/orbital gauge required')
        if sha256(Path(s['state']).read_bytes())!=s['state_SHA256'] or sha256(Path(s['operator']).read_bytes())!=s['operator_SHA256']:raise RuntimeError('Qualified source drift')
        for rep in REPLICATES:
            request=dict(case=s['case'],method=METHOD,replicate=rep,seed=seed(s['state_SHA256'],s['case'],rep),shots=1000,
                source_state_SHA256=s['state_SHA256'],physical_problem_id=d['problem_id'])
            request['draw_id']=digest(['N35_DRAW',request]);request['cell_id']=digest(['N35_CELL',request]);requests.append(request)
    previous={r['seed'] for folder in (n29.LUCJ,n30.SAMPLING,oldsource.SAMPLING) for c in load(folder/'protocol.json')['circuits'] for r in c['sampling_requests']}
    previous|={r['seed'] for r in load(n29.LOCAL/'protocol.json')['cpu_requests']}
    if len({r['seed'] for r in requests})!=12 or previous&{r['seed'] for r in requests}:raise RuntimeError('Fresh seed collision')
    newrows=[dict(case=r['case'],method=METHOD,replicate=r['replicate'],draw_id=r['draw_id'],cell_id=r['cell_id'],available=True,origin='N35_NEW_SPINLESS_MODEL_DRAW') for r in requests]
    rows=oldrows+newrows;tasks=[dict(row,arm=arm) for row in rows for arm in parent.ARMS]
    if len(rows)!=96 or sum(r['available'] for r in rows)!=87 or len(tasks)!=192:raise RuntimeError('All96source/192endpoint coverage required')
    LOCAL.mkdir(parents=True)
    save(LOCAL/'protocol.json',dict(version='N35_ALL4_SDK_SPINLESS_L1_FRESH_FINITE_AND_COMMON_CI_V1',method=METHOD,
        sources=terminal['sources'],requests=requests,tasks=tasks,physical=op['physical'],
        code_SHA256=sha256(Path(__file__).read_bytes()),preflight_terminal_SHA256=sha256((preflight.LOCAL/'terminal.json').read_bytes()),
        old_summary_SHA256=sha256((ROOT/'analysis/N33-ALL-summary.json').read_bytes()),
        old_source_protocol_SHA256=sha256((oldsource.LOCAL/'protocol.json').read_bytes()),
        helpers={name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in ('analytical_source_readout.py','finite_source_readout.py','joint_ci_sparse.py','joint_ci_sparse.c')},
        plan_SHA256=sha256((ROOT/'analysis/N35_SPINLESS_FINITE_PLAN_JA.md').read_bytes()),test_SHA256=sha256(test.read_bytes()),
        library=op['library'],library_SHA256=op['library_SHA256'],new_model_samples_cap=12000,new_QPU_shots=0,
        max_R=32,max_D=1024,root_seconds=20,global_seconds=600,new_run_cap_bytes=32*2**20,
        old_source_and_preflight_costs_retained=True,original_paper_initializer_reproduction=False,goal_complete=False))
    print('N35_FROZEN_ALL4_12NEW1K_DRAWS_192ENDPOINTS_18OLDMISSING',flush=True)


def verified():
    p=load(LOCAL/'protocol.json');preflight.verified();oldread.verified()
    if p['code_SHA256']!=sha256(Path(__file__).read_bytes()) or p['preflight_terminal_SHA256']!=sha256((preflight.LOCAL/'terminal.json').read_bytes()):raise RuntimeError('Frozen N35 code/source drift')
    if p['old_summary_SHA256']!=sha256((ROOT/'analysis/N33-ALL-summary.json').read_bytes()) or p['old_source_protocol_SHA256']!=sha256((oldsource.LOCAL/'protocol.json').read_bytes()):raise RuntimeError('Old evidence context changed')
    if sha256(Path(p['library']).read_bytes())!=p['library_SHA256']:raise RuntimeError('Qualified library changed')
    for name,h in p['helpers'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=h:raise RuntimeError('Frozen helper changed')
    return p


def bound(p,start):
    space()
    if perf_counter()-start>p['global_seconds']:raise TimeoutError('N35 stage wall cap')
    if sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())>p['new_run_cap_bytes']:raise OSError('N35 bounded storage')


def sample():
    p=verified();space()
    if (LOCAL/'sample-started.json').exists():raise RuntimeError('Existing sample intent; inspect, no blind restart')
    save(LOCAL/'sample-started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    cpu,wall=process_time(),perf_counter();rows=[]
    try:
        for s in p['sources']:
            bound(p,wall)
            if sha256(Path(s['state']).read_bytes())!=s['state_SHA256']:raise RuntimeError('Actual source probability changed')
            with np.load(s['state'],allow_pickle=False) as z:ids,v,prob=z['half_strings'],z['amplitudes'],z['probabilities']
            d=p['physical'][s['case']];n,k=d['norb'],d['nelec'][0]
            if not np.array_equal(prob,abs(v)**2) or abs(prob.sum()-1)>1e-8 or any(int(x).bit_count()!=k for x in ids):raise RuntimeError('Wrong actual half-label source')
            # Normalize only floating roundoff, already bounded by 1e-8 above.
            prob=prob/prob.sum()
            for req in [r for r in p['requests'] if r['case']==s['case']]:
                bound(p,wall);folder=LOCAL/'draws'/req['draw_id'];save(folder/'started.json',req)
                half=np.random.default_rng(req['seed']).choice(ids,size=req['shots'],p=prob)
                full=half|(half<<np.uint64(n));values,counts=np.unique(full,return_counts=True);counts=counts.astype(np.uint64)
                n29.validate_counts(values,counts,n,d['nelec']);file=folder/'counts.npz';write(file,arrays(blocked_ids=values,counts=counts))
                row=dict(**req,status='COMPLETED',path=str(file),payload_SHA256=sha256(file.read_bytes()),new_model_shots=1000,new_QPU_shots=0,
                    source_role='QUALIFIED_EXACT_PAIR_LABEL_SPINLESS_UCJ_MODEL_EQUIVALENT',source_simulation_paid_in_N34=True,half_sample_counted_once=True)
                save(folder/'result.json',row);rows.append(row);save(LOCAL/'sample-progress.json',dict(rows=rows))
            print('N35_NEW_MODEL_DRAWS',s['case'],3,flush=True)
        save(LOCAL/'sample-terminal.json',dict(status='N35_ALL12_NEW_DRAWS_TERMINAL',rows=rows,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
            protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),new_model_shots=12000,new_QPU_shots=0,new_optimizers=0,goal_complete=False))
    except Exception as exc:
        save(LOCAL/'sample-failed.json',dict(error=repr(exc),completed=len(rows),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise


def all_records(p):
    op,oq,sub=oldread.verified();oldrows=oldread.inventory(op,sub);out=oldread.records(op,sub,oldrows)
    for req in p['requests']:
        folder=LOCAL/'draws'/req['draw_id'];r=load(folder/'result.json');d=p['physical'][req['case']]
        if load(folder/'started.json')!=req or any(r[k]!=v for k,v in req.items()) or r['status']!='COMPLETED':raise RuntimeError('Actual draw request/result binding mismatch')
        file=Path(r['path'])
        if file!=folder/'counts.npz' or sha256(file.read_bytes())!=r['payload_SHA256']:raise RuntimeError('Actual raw counts drift')
        with np.load(file,allow_pickle=False) as z:ids,counts=z['blocked_ids'],z['counts']
        n29.validate_counts(ids,counts,d['norb'],d['nelec'])
        if any((int(x)&((1<<d['norb'])-1))!=(int(x)>>d['norb']) for x in ids):raise RuntimeError('Incorrect pair-to-spinful label isometry')
        source=next(s for s in p['sources'] if s['case']==req['case'])
        if sha256(Path(source['state']).read_bytes())!=source['state_SHA256']:raise RuntimeError('Bound probability source changed')
        with np.load(source['state'],allow_pickle=False) as z:
            probability=z['probabilities'];probability=probability/probability.sum()
            expected=np.random.default_rng(req['seed']).choice(z['half_strings'],size=req['shots'],p=probability)
        values,c=np.unique(expected|(expected<<np.uint64(d['norb'])),return_counts=True)
        if not np.array_equal(values,ids) or not np.array_equal(c,counts):raise RuntimeError('Independent actual seed/count verification failed')
        out[req['draw_id']]=dict(status='COMPLETED',**parent.rank_record(ids,counts,d['norb'],d['nelec'][0]),
            payload_SHA256=r['payload_SHA256'],seed=req['seed'],sampling_role=r['source_role'],source_state_SHA256=source['state_SHA256'])
    return out


def readout():
    p=verified();space();counts=all_records(p)
    if (LOCAL/'readout-started.json').exists():raise RuntimeError('Existing CI intent; inspect, no blind restart')
    save(LOCAL/'readout-started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    cpu,wall=process_time(),perf_counter();states=[];attempts=[];roots=LOCAL/'roots'
    try:
        for task in p['tasks']:
            bound(p,wall);row=dict(task)
            if not task['available']:row.update(status='MISSING_SAME_PROBLEM_SOURCE',energy_Ha=None)
            else:
                rec=counts[task['draw_id']];d=p['physical'][task['case']]
                R=rec['R32'] if task['arm']=='R32_CAP' else min(counts[t['draw_id']]['R32'] for t in p['tasks'] if t['available'] and t['case']==task['case'] and t['replicate']==task['replicate'])
                strings=sorted(rec['rank'][:R]);pairs=np.array([(a,b) for a in strings for b in strings],np.uint64)
                key=parent.basis_key(d,pairs,p['library_SHA256']);root=roots/key;meta=root.with_suffix('.json')
                candidates=[meta,oldread.LOCAL/'roots'/(key+'.json'),oldread.oldread.LOCAL/'roots'/(key+'.json'),parent.LOCAL/'roots'/(key+'.json')]
                reuse=next((f for f in candidates if f.exists()),None)
                if reuse:
                    cache=load(reuse);new=False;origin='N35_SAME_BASIS' if reuse==meta else 'OLD_SAME_PHYSICAL_BASIS'
                    if cache['basis_id']!=key or cache['status'] not in ('COMPLETED','FAILED_PRESERVED'):raise RuntimeError('Root cache identity mismatch')
                else:
                    if root.with_suffix('.started.json').exists():raise RuntimeError('Partial root; no blind rerun')
                    file=Path(d['local_integrals'])
                    if sha256(file.read_bytes())!=d['integrals_SHA256']:raise RuntimeError('Physical integral drift')
                    save(root.with_suffix('.started.json'),dict(case=task['case'],basis_id=key,pid=os.getpid()));tick=process_time()
                    try:
                        with np.load(file,allow_pickle=False) as z:result,v=solve(z['h1'],z['eri'],d['core_energy'],tuple(d['nelec']),pairs,p['library'],seconds=p['root_seconds'],triangle_cap=8000000)
                        path=root.with_suffix('.npz');write(path,arrays(joint_pairs=pairs,coefficients=v))
                        cache=dict(status='COMPLETED',basis_id=key,result=result,root_path=str(path),root_SHA256=sha256(path.read_bytes()))
                    except Exception as exc:cache=dict(status='FAILED_PRESERVED',basis_id=key,result=None,error=repr(exc))
                    cache['attempt_CPU_seconds']=process_time()-tick;save(meta,cache);attempts.append(dict(basis_id=key,status=cache['status'],CPU_seconds=cache['attempt_CPU_seconds']));new=True;origin='N35_NEW_ROOT'
                if cache['status']=='COMPLETED' and sha256(Path(cache['root_path']).read_bytes())!=cache['root_SHA256']:raise RuntimeError('Actual root bytes differ')
                row.update(cache,R=R,actual_D=R*R,half_strings=strings,source_evidence=rec,new_CI_attempt=new,cache_origin=origin)
            states.append(row);save(LOCAL/'readout-progress.json',dict(states=states,attempts=attempts))
        save(LOCAL/'readout-terminal.json',dict(status='N35_ALL192_ENDPOINTS_TERMINAL',states=states,attempts=attempts,
            actual_new_CI_roots=sum(a['status']=='COMPLETED' for a in attempts),failed_CI_attempts=sum(a['status']!='COMPLETED' for a in attempts),
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),new_QPU_shots=0,new_optimizers=0,goal_complete=False))
        print('N35_ALL192_ENDPOINTS_TERMINAL',len(attempts),'actual new CI roots',flush=True)
    except Exception as exc:
        save(LOCAL/'readout-failed.json',dict(error=repr(exc),completed=len(states),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise


if __name__=='__main__':
    import sys
    {'prepare':prepare,'sample':sample,'readout':readout}[sys.argv[1]]()
