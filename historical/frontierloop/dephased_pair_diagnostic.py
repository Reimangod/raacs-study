"""N27 per-gate dephasing diagnostic, exact probabilities on BOTH sides."""
from pathlib import Path
from time import process_time,perf_counter
import xml.etree.ElementTree as ET
import numpy as np
from .paths import ROOT
from .io import load,save,write,arrays,sha256,digest,space
from .prefix_performance_sampling import capacity
from .joint_ci_sparse import solve

LOCAL=ROOT/'runs/N27-DEPHASED-PAIR-DIAGNOSTIC-v1'
PARENT=ROOT/'runs/N25-PAIRED-LOWSHOT-v1'


def dephased_distribution(ids,k,angles):
    ids=np.asarray(ids);lookup={int(x):i for i,x in enumerate(ids)}
    if len(lookup)!=len(ids) or any(int(x).bit_count()!=k for x in ids):raise ValueError('Unique same-sector states required')
    p=np.zeros(len(ids));p[lookup[(1<<k)-1]]=1.
    for e in angles:
        i,a,theta=e['i'],e['a'],e['theta']
        if i==a or not np.isfinite(theta):raise ValueError('Finite distinct gate wires required')
        sel=[j for j,x in enumerate(map(int,ids)) if x>>i&1 and not x>>a&1]
        target=[lookup[int(ids[j])^(1<<i)^(1<<a)] for j in sel]
        x=p[sel].copy();y=p[target].copy();c,s=np.cos(theta)**2,np.sin(theta)**2
        p[sel]=c*x+s*y;p[target]=s*x+c*y
    if p.min()<0 or abs(float(p.sum())-1)>1e-10:raise ValueError('Classical Markov norm/positivity failure')
    return p


def top_half(ids,p,hf,R):
    if len(ids)!=len(p) or not np.isfinite(p).all() or np.min(p)<0:raise ValueError('Finite probability table required')
    weights=dict(zip(map(int,ids),map(float,p)))
    if hf not in weights:raise ValueError('Registered HF missing')
    order=[hf]+sorted((x for x in weights if x!=hf and weights[x]>0),key=lambda x:(-weights[x],x))
    if len(order)<R:raise ValueError('Insufficient nonzero support; no padding')
    margin=weights[order[R-1]]-(weights[order[R]] if len(order)>R else 0.)
    return sorted(order[:R]),float(margin)


def prepare():
    initial=capacity()
    if LOCAL.exists():raise RuntimeError('Existing N27; never overwrite/restart')
    q=ROOT/'qualification-N27-preexecution-v1-tests.xml';ss=list(ET.parse(q).getroot().iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in ss)!=3 or any(int(s.get('failures','0')) or int(s.get('errors','0')) for s in ss):raise RuntimeError('All3preexecution qualification tests required')
    p=load(PARENT/'protocol.json');summary=load(ROOT/'analysis/N25-lowshot-pair-summary.json')
    if summary['terminal_SHA256']!=sha256((PARENT/'terminal.json').read_bytes()):raise RuntimeError('Audited original source required')
    sources=[]
    for s in p['sources']:
        case=s['case'];state=PARENT/'sources'/case/'pair-state.npz';angles=PARENT/'sources'/case/'angles.json'
        sources.append(dict(case=case,definition=s['definition'],integrals=s['local_integrals'],state=str(state),state_SHA256=sha256(state.read_bytes()),angles=str(angles),angles_SHA256=sha256(angles.read_bytes())))
    LOCAL.mkdir()
    save(LOCAL/'protocol.json',dict(version='N27_COHERENT_VS_PER_GATE_DEPHASED_EXACT_PROBABILITY_DIAGNOSTIC_V1',sources=sources,
        R_values=[16,32],methods=['COHERENT_EXACT_PROB','DEPHASED_EXACT_PROB'],planned=16,
        code_SHA256=sha256(Path(__file__).read_bytes()),helper_SHA256={name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in ('paired_lowshot.py','joint_ci_sparse.py','joint_ci_sparse.c')},
        library=p['library'],library_SHA256=p['library_SHA256'],initial_capacity=initial,
        qualification_SHA256=sha256(q.read_bytes()),plan_SHA256=sha256((ROOT/'analysis/N27_DEPHASED_PAIR_PLAN_JA.md').read_bytes()),
        max_wall_seconds=60,max_CI_wall_seconds=20,max_run_bytes=8*2**20,near_tie_margin_design_flag=1e-12,
        exact_distribution_oracle_both_methods_not_finite_shot_performance=True,new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,goal_complete=False))
    print('N27_FROZEN_ALL4_EXACT_INFORMATION_PARITY16READOUTS',flush=True)


def run():
    capacity();p=load(LOCAL/'protocol.json')
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing intent; no blind restart')
    if sha256(Path(__file__).read_bytes())!=p['code_SHA256'] or sha256(Path(p['library']).read_bytes())!=p['library_SHA256'] or any(sha256((ROOT/'frontierloop'/name).read_bytes())!=h for name,h in p['helper_SHA256'].items()):raise RuntimeError('Frozen implementation/math changed')
    cpu,wall=process_time(),perf_counter();save(LOCAL/'started.json',dict(pid=__import__('os').getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    rows=[];diagnostics=[];attempts=[];cache={}
    def bound():
        space()
        if perf_counter()-wall>p['max_wall_seconds']:raise TimeoutError('N27global cap')
        if sum(x.stat().st_size for x in LOCAL.rglob('*') if x.is_file())>p['max_run_bytes']:raise OSError('N27bounded output cap')
    try:
        for source in p['sources']:
            bound();case=source['case'];d=source['definition'];k=d['nelec'][0];hf=(1<<k)-1
            for field,hfield in [('state','state_SHA256'),('angles','angles_SHA256')]:
                if sha256(Path(source[field]).read_bytes())!=source[hfield]:raise RuntimeError('Actual source changed')
            with np.load(source['state'],allow_pickle=False) as z:ids=z['half_strings'];amplitudes=z['amplitudes']
            angles=load(source['angles'])['angles'];coherent=amplitudes**2;dephased=dephased_distribution(ids,k,angles)
            if abs(float(coherent.sum())-1)>1e-10:raise RuntimeError('Original coherent norm differs')
            payload=LOCAL/case/'probabilities.npz';write(payload,arrays(half_strings=ids,coherent=coherent,dephased=dephased))
            high=np.array([k-(int(x)&hf).bit_count()>=2 for x in ids])
            diag=dict(case=case,probabilities_path=str(payload),probabilities_SHA256=sha256(payload.read_bytes()),
                single_draw_total_variation=float(np.sum(abs(coherent-dephased)))/2,
                coherent_HF_probability=float(coherent[list(map(int,ids)).index(hf)]),dephased_HF_probability=float(dephased[list(map(int,ids)).index(hf)]),
                coherent_multiple_pair_mass=float(coherent[high].sum()),dephased_multiple_pair_mass=float(dephased[high].sum()),
                probability_dimension=len(ids),finite_shot_performance_not_measured=True)
            diagnostics.append(diag);save(LOCAL/'diagnostics.json',diagnostics)
            path=Path(source['integrals'])
            if sha256(path.read_bytes())!=d['integrals_SHA256']:raise RuntimeError('Physical integrals changed')
            with np.load(path,allow_pickle=False) as z:h=z['h1'];g=z['eri']
            for method,prob in zip(p['methods'],(coherent,dephased)):
                for R in p['R_values']:
                    bound();r=dict(case=case,method=method,R=R,actual_D=R*R)
                    strings,margin=top_half(ids,prob,hf,R);pairs=np.array([(a,b) for a in strings for b in strings],np.uint64);key=digest([case,pairs.tolist()])
                    r.update(half_strings=strings,basis_id=key,probability_rank_margin=margin,near_tie=margin<=p['near_tie_margin_design_flag'])
                    if key in cache:r.update(cache[key],new_CI_solve=False)
                    else:
                        at=dict(case=case,basis_id=key,status='STARTED');attempts.append(at);save(LOCAL/'CI-attempts.json',attempts);tick=process_time()
                        try:
                            result,v=solve(h,g,d['core_energy'],tuple(d['nelec']),pairs,p['library'],seconds=p['max_CI_wall_seconds'])
                            path=LOCAL/'roots'/(key+'.npz');write(path,arrays(joint_pairs=pairs,coefficients=v))
                            cache[key]=dict(status='COMPLETED',result=result,root_path=str(path),root_SHA256=sha256(path.read_bytes()));at['status']='COMPLETED';r.update(cache[key],new_CI_solve=True)
                        except Exception as exc:
                            cache[key]=dict(status='FAILED_PRESERVED',result=None,error=repr(exc));at.update(status='FAILED_PRESERVED',error=repr(exc));r.update(cache[key],new_CI_solve=False)
                        finally:at['CPU_seconds']=process_time()-tick;save(LOCAL/'CI-attempts.json',attempts)
                    rows.append(r);save(LOCAL/'progress.json',dict(rows=rows));print('N27',case,method,R,r['status'],flush=True)
        bound();capacity()
        save(LOCAL/'terminal.json',dict(status='N27_ALL16_EXACT_PROBABILITY_READOUTS_TERMINAL',rows=rows,diagnostics=diagnostics,
            actual_new_CI_roots=sum(a['status']=='COMPLETED' for a in attempts),failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,CI_CPU_seconds=sum(a['CPU_seconds'] for a in attempts),
            protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,
            quantum_advantage_or_hardware_noise_performance_proved=False,goal_complete=False))
    except Exception as exc:
        save(LOCAL/'failed.json',dict(status='FAILED_PRESERVED',error=repr(exc),completed_states=len(rows),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise


if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()
