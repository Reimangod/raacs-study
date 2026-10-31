"""N28 fixed complex-phase/common-scale paired source feasibility experiment."""
from io import BytesIO
from pathlib import Path
from time import process_time,perf_counter
import importlib.metadata,xml.etree.ElementTree as ET
import numpy as np
from qiskit import QuantumCircuit,qpy
from qiskit.circuit.library import XXPlusYYGate
from .paths import ROOT
from .io import load,save,write,arrays,sha256,digest,space
from .prefix_performance_sampling import capacity
from .slater_prefix_preflight import compile_views
from .dephased_pair_diagnostic import dephased_distribution,top_half
from .joint_ci_sparse import solve

LOCAL=ROOT/'runs/N28-PHASED-PAIR-STRESS-v1'
PARENT=ROOT/'runs/N27-DEPHASED-PAIR-DIAGNOSTIC-v1'


def angles_for(angles,k,scale,pattern):
    if scale not in (1,2,4) or pattern not in ('REAL','FLUX','GAUGE_ROW'):raise ValueError('Unregistered scale/phase')
    return [dict(i=e['i'],a=e['a'],theta=scale*e['theta'],beta=float(np.pi/2+
        (np.pi/2*((e['i']*(e['a']-k))%2) if pattern=='FLUX' else .37*e['i'] if pattern=='GAUGE_ROW' else 0.))) for e in angles]


def evolve(ids,k,angles):
    lookup={int(x):i for i,x in enumerate(ids)}
    if len(lookup)!=len(ids) or any(int(x).bit_count()!=k for x in ids):raise ValueError('Unique same-sector basis required')
    v=np.zeros(len(ids),complex);v[lookup[(1<<k)-1]]=1.
    for e in angles:
        i,a=e['i'],e['a']
        if i==a or not np.isfinite([e['theta'],e['beta']]).all():raise ValueError('Finite distinct-wire gate required')
        sel=[j for j,b in enumerate(map(int,ids)) if b>>i&1 and not b>>a&1]
        target=[lookup[int(ids[j])^(1<<i)^(1<<a)] for j in sel]
        # Use actual pinned Qiskit convention, not a recalled beta sign formula.
        U=XXPlusYYGate(2*e['theta'],e['beta']).to_matrix()[1:3,1:3]
        x,y=v[sel].copy(),v[target].copy()
        v[sel]=U[0,0]*x+U[0,1]*y;v[target]=U[1,0]*x+U[1,1]*y
    if abs(float(np.vdot(v,v).real)-1)>1e-10:raise RuntimeError('Complex pair norm failed')
    return v


def circuit(n,k,angles,full):
    q=QuantumCircuit(2*n if full else n)
    for i in range(k):q.x(i)
    for e in angles:q.append(XXPlusYYGate(2*e['theta'],e['beta']),[e['i'],e['a']])
    if full:
        for i in range(n):q.cx(i,n+i)
    return q


def prepare():
    initial=capacity()
    if LOCAL.exists():raise RuntimeError('Existing N28; never overwrite/restart')
    test=ROOT/'qualification-N28-preexecution-v1-tests.xml';ss=list(ET.parse(test).getroot().iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in ss)!=4 or any(int(s.get('failures','0')) or int(s.get('errors','0')) for s in ss):raise RuntimeError('All4complex/gauge tests required')
    parent=load(PARENT/'protocol.json');summary=load(ROOT/'analysis/N27-dephased-pair-summary.json')
    if summary['terminal_SHA256']!=sha256((PARENT/'terminal.json').read_bytes()) or summary['protocol_SHA256']!=sha256((PARENT/'protocol.json').read_bytes()):raise RuntimeError('Audited original N27 required')
    LOCAL.mkdir()
    save(LOCAL/'protocol.json',dict(version='N28_COMMON_SCALE_FLUX_AND_MARKOV_SOURCE_STRESS_V1',sources=parent['sources'],scales=[1,2,4],patterns=['REAL','FLUX'],
        planned_readouts=36,planned_coherent_sources=24,planned_resource_points=96,R=32,actual_D=1024,
        code_SHA256=sha256(Path(__file__).read_bytes()),helper_SHA256={n:sha256((ROOT/'frontierloop'/n).read_bytes()) for n in ('slater_prefix_preflight.py','dephased_pair_diagnostic.py','joint_ci_sparse.py','joint_ci_sparse.c')},
        qualification_SHA256=sha256(test.read_bytes()),plan_SHA256=sha256((ROOT/'analysis/N28_SOURCE_STRESS_PLAN_JA.md').read_bytes()),
        parent_summary_SHA256=sha256((ROOT/'analysis/N27-dephased-pair-summary.json').read_bytes()),
        library=parent['library'],library_SHA256=parent['library_SHA256'],packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','qiskit')},
        max_wall_seconds=180,max_CI_wall_seconds=20,max_run_bytes=24*2**20,near_tie_margin=1e-12,
        initial_capacity=initial,exact_probabilities_both_quantum_and_classical_controls=True,finite_shot_performance=False,
        phase_and_scale_fixed_conditions_not_fitted_ansatz_variables=True,new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,goal_complete=False))
    print('N28_FROZEN36READOUTS24COHERENT96RESOURCEPOINTS',flush=True)


def run():
    capacity();p=load(LOCAL/'protocol.json')
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing execution intent; no blind restart')
    if sha256(Path(__file__).read_bytes())!=p['code_SHA256'] or sha256(Path(p['library']).read_bytes())!=p['library_SHA256'] or any(sha256((ROOT/'frontierloop'/n).read_bytes())!=h for n,h in p['helper_SHA256'].items()):raise RuntimeError('Frozen implementation/math changed')
    if {n:importlib.metadata.version(n) for n in p['packages']}!=p['packages']:raise RuntimeError('Pinned runtime changed')
    cpu,wall=process_time(),perf_counter();last_scan=wall;rows=[];sources=[];attempts=[];cache={}
    save(LOCAL/'started.json',dict(pid=__import__('os').getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    def bound():
        nonlocal last_scan
        space()
        if perf_counter()-wall>p['max_wall_seconds']:raise TimeoutError('N28global cap')
        if sum(x.stat().st_size for x in LOCAL.rglob('*') if x.is_file())>p['max_run_bytes']:raise OSError('Bounded new output cap')
        if perf_counter()-last_scan>60:capacity();last_scan=perf_counter()
    try:
        for original in p['sources']:
            case=original['case'];d=original['definition'];n=d['norb'];k=d['nelec'][0];hf=(1<<k)-1
            for field,hkey in [('state','state_SHA256'),('angles','angles_SHA256')]:
                if sha256(Path(original[field]).read_bytes())!=original[hkey]:raise RuntimeError('Original physical source changed')
            with np.load(original['state'],allow_pickle=False) as z:ids=z['half_strings'];oldv=z['amplitudes']
            oldangles=load(original['angles'])['angles'];file=Path(original['integrals'])
            if sha256(file.read_bytes())!=d['integrals_SHA256']:raise RuntimeError('Same physical integrals required')
            with np.load(file,allow_pickle=False) as z:h=z['h1'];g=z['eri']
            for scale in p['scales']:
                distributions={};scaled=angles_for(oldangles,k,scale,'REAL');markov=dephased_distribution(ids,k,scaled)
                for pattern in p['patterns']:
                    bound();angles=angles_for(oldangles,k,scale,pattern);v=evolve(ids,k,angles);prob=abs(v)**2
                    if scale==1 and pattern=='REAL' and np.max(abs(v-oldv))>1e-12:raise RuntimeError('Original real source parity failed')
                    folder=LOCAL/case/('s'+str(scale)+'-'+pattern);file=folder/'source.npz';write(file,arrays(half_strings=ids,amplitudes=v,probabilities=prob))
                    points=[]
                    for full in (False,True):
                        qc=circuit(n,k,angles,full)
                        for level in (1,3):
                            prep,samp,res=compile_views(qc,level);files={}
                            for role,q in [('preparation',prep),('sampling',samp)]:
                                stream=BytesIO();qpy.dump(q,stream);blob=stream.getvalue();qpath=folder/('FULL_SPIN_COPY' if full else 'PAIR_ENCODING')/f'opt{level}-{role}.qpy';write(qpath,blob)
                                files[role]=dict(path=str(qpath),SHA256=sha256(blob))
                            points.append(dict(encoding='FULL_SPIN_COPY' if full else 'PAIR_ENCODING',n_qubits=qc.num_qubits,resources=res,circuits=files))
                    record=dict(case=case,scale=scale,pattern=pattern,source_file=str(file),source_SHA256=sha256(file.read_bytes()),
                        angles=angles,declared_real_pair_variables=k*(n-k),all_compiler_points=points,
                        HF_probability=float(prob[list(map(int,ids)).index(hf)]),single_draw_TV_vs_same_scale_Markov=float(np.sum(abs(prob-markov)))/2)
                    sources.append(record);save(LOCAL/'sources.json',sources);distributions['COHERENT_'+pattern]=(prob,record)
                file=LOCAL/case/('s'+str(scale)+'-MARKOV.npz');write(file,arrays(half_strings=ids,probabilities=markov))
                distributions['MARKOV']=(markov,dict(source_file=str(file),source_SHA256=sha256(file.read_bytes()),quantum_resources=None))
                for method,(prob,source) in distributions.items():
                    bound();strings,margin=top_half(ids,prob,hf,p['R']);pairs=np.array([(a,b) for a in strings for b in strings],np.uint64);key=digest([case,pairs.tolist()])
                    row=dict(case=case,scale=scale,method=method,actual_D=len(pairs),R=p['R'],half_strings=strings,basis_id=key,
                        source_file=source['source_file'],source_SHA256=source['source_SHA256'],probability_rank_margin=margin,near_tie=margin<=p['near_tie_margin'])
                    if key in cache:row.update(cache[key],new_CI_solve=False)
                    else:
                        at=dict(case=case,basis_id=key,status='STARTED');attempts.append(at);save(LOCAL/'CI-attempts.json',attempts);tick=process_time()
                        try:
                            result,c=solve(h,g,d['core_energy'],tuple(d['nelec']),pairs,p['library'],seconds=p['max_CI_wall_seconds'])
                            path=LOCAL/'roots'/(key+'.npz');write(path,arrays(joint_pairs=pairs,coefficients=c))
                            cache[key]=dict(status='COMPLETED',result=result,root_path=str(path),root_SHA256=sha256(path.read_bytes()));at['status']='COMPLETED';row.update(cache[key],new_CI_solve=True)
                        except Exception as exc:
                            cache[key]=dict(status='FAILED_PRESERVED',result=None,error=repr(exc));at.update(status='FAILED_PRESERVED',error=repr(exc));row.update(cache[key],new_CI_solve=False)
                        finally:at['CPU_seconds']=process_time()-tick;save(LOCAL/'CI-attempts.json',attempts)
                    rows.append(row);save(LOCAL/'progress.json',dict(rows=rows));print('N28',case,scale,method,row['status'],flush=True)
        bound();capacity()
        save(LOCAL/'terminal.json',dict(status='N28_ALL36_SOURCE_STRESS_READOUTS_TERMINAL',rows=rows,sources=sources,
            actual_new_CI_roots=sum(a['status']=='COMPLETED' for a in attempts),failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,CI_CPU_seconds=sum(a['CPU_seconds'] for a in attempts),
            protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,goal_complete=False))
    except Exception as exc:
        save(LOCAL/'failed.json',dict(status='FAILED_PRESERVED',error=repr(exc),completed_states=len(rows),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise


if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()
