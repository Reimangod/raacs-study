"""N26 zero-shot classical pair-amplitude control; never a circuit with zero CX."""
from pathlib import Path
from time import process_time,perf_counter
import xml.etree.ElementTree as ET
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,write,arrays,sha256,digest,space
from .prefix_performance_sampling import capacity
from .joint_ci_sparse import solve
from paperstudy.lucj import amplitudes_from_saved

LOCAL=ROOT/'runs/N26-CLASSICAL-PAIR-CONTROL-v1'
PARENT=ROOT/'runs/N25-PAIRED-LOWSHOT-v1'


def rank_pair_amplitudes(n,k,t2):
    t2=np.asarray(t2)
    if t2.shape!=(k,k,n-k,n-k) or np.iscomplexobj(t2) or not np.isfinite(t2).all():raise ValueError('Finite pinned real restricted amplitudes required')
    hf=(1<<k)-1;scores={hf:1.}
    for i in range(k):
        for a in range(k,n):
            weight=float(t2[i,i,a-k,a-k])**2
            if not np.isfinite(weight):raise ValueError('Nonfinite squared pair amplitude')
            if weight>0:scores[hf^(1<<i)^(1<<a)]=weight
    order=[hf]+sorted((x for x in scores if x!=hf),key=lambda x:(-scores[x],x))
    return order,[dict(half_string=x,weight=scores[x]) for x in order]


def requested_plans(case,draws):
    out=[]
    for rep in (250101,250102,250103):
        d=next(r for r in draws if (r['case'],r['method'],r['replicate'])==(case,'PAIR_T2',rep))
        candidates=[r for r in draws if r['case']==case and r['replicate']==rep]
        R=min(32,*(len(r['rank']) for r in candidates))
        out.append(dict(case=case,arm='MATCHED_N25_COMMON_R',replicate=rep,R=R,source_draw_SHA256=d['SHA256']))
    out.append(dict(case=case,arm='CLASSICAL_R32_CAP',replicate=None,R=32))
    return out


def prepare():
    c=capacity()
    if LOCAL.exists():raise RuntimeError('Existing N26; preserve and reconcile, never overwrite')
    q=ROOT/'qualification-N26-preexecution-v1-tests.xml';suites=list(ET.parse(q).getroot().iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=3 or any(int(s.get('failures','0')) or int(s.get('errors','0')) for s in suites):raise RuntimeError('All3preexecution tests required')
    parent=load(PARENT/'protocol.json');t=load(PARENT/'terminal.json');summary=load(ROOT/'analysis/N25-lowshot-pair-summary.json')
    if summary['terminal_SHA256']!=sha256((PARENT/'terminal.json').read_bytes()) or summary['protocol_SHA256']!=sha256((PARENT/'protocol.json').read_bytes()):raise RuntimeError('Actual audited parent required')
    plans=[r for s in parent['sources'] for r in requested_plans(s['case'],t['draws'])]
    if len(plans)!=16:raise RuntimeError('All16 planned points required')
    LOCAL.mkdir()
    save(LOCAL/'protocol.json',dict(version='N26_CLASSICAL_ONLY_PAIR_AMPLITUDE_RANK_CONTROL_V1',plans=plans,sources=parent['sources'],
        parent_protocol_SHA256=sha256((PARENT/'protocol.json').read_bytes()),parent_terminal_SHA256=sha256((PARENT/'terminal.json').read_bytes()),parent_summary_SHA256=sha256((ROOT/'analysis/N25-lowshot-pair-summary.json').read_bytes()),
        code_SHA256=sha256(Path(__file__).read_bytes()),helper_SHA256={name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in ('joint_ci_sparse.py','joint_ci_sparse.c')},
        initializer_helper_SHA256=sha256((OLD/'paperstudy/lucj.py').read_bytes()),library=parent['library'],library_SHA256=parent['library_SHA256'],
        qualification_SHA256=sha256(q.read_bytes()),plan_SHA256=sha256((ROOT/'analysis/N26_CLASSICAL_PAIR_CONTROL_PLAN_JA.md').read_bytes()),
        initial_capacity=c,extra_run_bytes_cap=16*2**20,max_wall_seconds=120,max_CI_wall_seconds=30,max_D=1024,
        numerical_equation='H_PLUS_0p1_S2_PHYSICAL_H_REPORTED_SEPARATELY',quantum_resources=None,
        new_model_shots=0,new_QPU_shots=0,new_classical_random_draws=0,GPU_allocations=0,goal_complete=False))
    print('N26_FROZEN16_ZERO_SHOT_CLASSICAL_CONTROL_POINTS',flush=True)


def run():
    p=load(LOCAL/'protocol.json');initial=capacity()
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing intent; never restart blindly')
    if sha256(Path(__file__).read_bytes())!=p['code_SHA256'] or sha256(Path(p['library']).read_bytes())!=p['library_SHA256']:raise RuntimeError('Frozen code/kernel changed')
    if any(sha256((ROOT/'frontierloop'/name).read_bytes())!=h for name,h in p['helper_SHA256'].items()) or sha256((OLD/'paperstudy/lucj.py').read_bytes())!=p['initializer_helper_SHA256']:raise RuntimeError('Helper changed')
    if sha256((PARENT/'protocol.json').read_bytes())!=p['parent_protocol_SHA256'] or sha256((PARENT/'terminal.json').read_bytes())!=p['parent_terminal_SHA256']:raise RuntimeError('Parent changed')
    cpu,wall=process_time(),perf_counter();last_scan=wall;rows=[];cache={};attempts=[];rankings=[]
    save(LOCAL/'started.json',dict(pid=__import__('os').getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),capacity=initial))
    def bound():
        nonlocal last_scan
        space()
        if perf_counter()-wall>p['max_wall_seconds']:raise TimeoutError('N26global cap')
        if sum(x.stat().st_size for x in LOCAL.rglob('*') if x.is_file())>p['extra_run_bytes_cap']:raise OSError('Bounded new-run output cap')
        if perf_counter()-last_scan>60:capacity();last_scan=perf_counter()
    try:
        for source in p['sources']:
            case=source['case'];d=source['definition']
            for field,hkey in [('amplitudes','amplitudes_SHA256'),('initialization_record_path','initialization_record_SHA256')]:
                if sha256(Path(source[field]).read_bytes())!=source[hkey]:raise RuntimeError('Initializer binding changed')
            n=d['norb'];k=d['nelec'][0];tick=process_time();_,t2,kind=amplitudes_from_saved(source['amplitudes'],n,k)
            rank,scores=rank_pair_amplitudes(n,k,t2);ranking=dict(case=case,rank=rank,scores=scores,amplitude_kind=kind,CPU_seconds=process_time()-tick)
            rankings.append(ranking);save(LOCAL/'rankings.json',rankings)
            file=Path(source['local_integrals'])
            if sha256(file.read_bytes())!=d['integrals_SHA256']:raise RuntimeError('Physical integral mismatch')
            with np.load(file,allow_pickle=False) as a:h=a['h1'];g=a['eri']
            for plan in [r for r in p['plans'] if r['case']==case]:
                bound();R=plan['R'] if plan['arm']=='MATCHED_N25_COMMON_R' else min(32,len(rank));r=dict(plan,actual_R=R,quantum_resources=None)
                if len(rank)<R:r.update(status='INSUFFICIENT_REGISTERED_T2_SUPPORT',result=None);rows.append(r);continue
                strings=sorted(rank[:R]);pairs=np.array([(a,b) for a in strings for b in strings],np.uint64);key=digest([case,pairs.tolist()]);r.update(half_strings=strings,basis_id=key,actual_D=R*R)
                if key in cache:r.update(cache[key],new_CI_solve=False)
                else:
                    tick=process_time();a=dict(case=case,basis_id=key,status='STARTED');attempts.append(a);save(LOCAL/'CI-attempts.json',attempts)
                    try:
                        result,v=solve(h,g,d['core_energy'],tuple(d['nelec']),pairs,p['library'],seconds=p['max_CI_wall_seconds'])
                        file=LOCAL/'roots'/(key+'.npz');write(file,arrays(joint_pairs=pairs,coefficients=v))
                        cache[key]=dict(status='COMPLETED',result=result,root_path=str(file),root_SHA256=sha256(file.read_bytes()))
                        r.update(cache[key],new_CI_solve=True);a['status']='COMPLETED'
                    except Exception as exc:
                        cache[key]=dict(status='FAILED_PRESERVED',result=None,error=repr(exc));r.update(cache[key],new_CI_solve=False);a.update(status='FAILED_PRESERVED',error=repr(exc))
                    finally:a['CPU_seconds']=process_time()-tick;save(LOCAL/'CI-attempts.json',attempts)
                rows.append(r);save(LOCAL/'progress.json',dict(rows=rows));print('N26',case,plan['arm'],plan['replicate'],R,r['status'],flush=True)
        bound();final=capacity()
        save(LOCAL/'terminal.json',dict(status='N26_ALL16_CLASSICAL_CONTROL_POINTS_TERMINAL',rows=rows,rankings=rankings,
            actual_new_CI_roots=sum(a['status']=='COMPLETED' for a in attempts),failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,CI_attempt_CPU_seconds=sum(a['CPU_seconds'] for a in attempts),
            protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),final_capacity=final,
            new_model_shots=0,new_QPU_shots=0,new_classical_random_draws=0,GPU_allocations=0,goal_complete=False))
    except Exception as exc:
        save(LOCAL/'failed.json',dict(status='FAILED_PRESERVED',error=repr(exc),completed_states=len(rows),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise


if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()
