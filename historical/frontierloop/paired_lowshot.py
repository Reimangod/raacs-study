"""N25 bounded, explicitly exploratory known-pair source/readout experiment."""
from itertools import combinations
from io import BytesIO
from pathlib import Path
from time import process_time,perf_counter
import importlib.metadata,xml.etree.ElementTree as ET
import numpy as np
from qiskit import QuantumCircuit,qpy
from qiskit.circuit.library import XXPlusYYGate
from .paths import ROOT,OLD
from .io import load,save,write,arrays,sha256,digest,space
from .paired_resource_preflight import schedule
from .slater_prefix_preflight import compile_views
from .joint_ci_sparse import solve
from .prefix_performance_sampling import capacity
from paperstudy.lucj import amplitudes_from_saved

LOCAL=ROOT/'runs/N25-PAIRED-LOWSHOT-v1'
N17=ROOT/'runs/N17-INDEPENDENT-PREFIX-PERFORMANCE-v1'
N23=ROOT/'runs/N23-BACKBONE-UNIFORM-CI-v1'


def paired_sector(n,k,t2):
    t2=np.asarray(t2)
    if t2.shape!=(k,k,n-k,n-k) or np.iscomplexobj(t2) or not np.isfinite(t2).all():raise ValueError('Pinned real restricted t2 required')
    if __import__('math').comb(n,k)>20000:raise ValueError('Bounded pair-sector only')
    ids=np.array(sorted(sum(1<<i for i in row) for row in combinations(range(n),k)),dtype=np.uint64)
    lookup={int(x):i for i,x in enumerate(ids)};v=np.zeros(len(ids));v[lookup[(1<<k)-1]]=1.;angles=[]
    for i,a in (edge for layer in schedule(n,k) for edge in layer):
        theta=float(t2[i,i,a-k,a-k]);angles.append(dict(i=i,a=a,theta=theta))
        selected=[j for j,b in enumerate(map(int,ids)) if b>>i&1 and not b>>a&1]
        target=[lookup[int(ids[j])^(1<<i)^(1<<a)] for j in selected]
        x=v[selected].copy();y=v[target].copy();c,s=np.cos(theta),np.sin(theta)
        v[selected]=c*x-s*y;v[target]=s*x+c*y
    if abs(float(v@v)-1)>1e-10:raise ValueError('Pair norm failed')
    return ids,v,angles


def paired_circuit(n,k,angles,full):
    q=QuantumCircuit(2*n if full else n)
    for i in range(k):q.x(i)
    for e in angles:q.append(XXPlusYYGate(2*e['theta'],np.pi/2),[e['i'],e['a']])
    if full:
        for i in range(n):q.cx(i,n+i)
    return q


def thin_counts(ids,counts,shots,seed):
    ids=np.asarray(ids);counts=np.asarray(counts)
    if ids.ndim!=1 or len(ids)!=len(counts) or len(set(map(int,ids)))!=len(ids) or counts.dtype.kind not in 'iu' or any(counts<=0) or int(counts.sum())<shots:raise ValueError('Valid existing counts required')
    selected=np.random.default_rng(seed).multivariate_hypergeometric(counts.astype(np.int64),shots)
    keep=selected>0
    return ids[keep],selected[keep]


def ranked_half(ids,counts,n,hf):
    weights={hf:0.};mask=(1<<n)-1
    for b,c in zip(map(int,ids),map(float,counts)):
        for x in (b&mask,b>>n):weights[x]=weights.get(x,0.)+c
    return [hf]+sorted((x for x in weights if x!=hf),key=lambda x:(-weights[x],x))


def seed(case,method,replica,source_sha):return int(digest(['N25',case,method,replica,source_sha])[:16],16)%(1<<63)


def prepare():
    capacity()
    if LOCAL.exists():raise RuntimeError('Existing N25; no overwrite or blind restart')
    test=ROOT/'qualification-N25-preexecution-v1-tests.xml';suites=list(ET.parse(test).getroot().iter('testsuite'))
    if sum(int(x.get('tests','0')) for x in suites)!=4 or any(int(x.get('failures','0')) or int(x.get('errors','0')) for x in suites):raise RuntimeError('All4 qualification tests required')
    physical=load(N23/'protocol.json')['physical'];sources=[]
    for case,d in sorted(physical.items()):
        folder=OLD/'data/initialization-fresh'/case;record=load(folder/'result.json');amp=folder/'amplitudes.npz'
        if record['problem_id']!=d['problem_id'] or record['status']!='FRESH_INITIALIZATION_COMPLETED' or sha256(amp.read_bytes())!=record['amplitudes_sha256']:raise RuntimeError('Bound initialization required')
        file=OLD/'data/physical'/case/'integrals.npz'
        if sha256(file.read_bytes())!=d['integrals_SHA256']:raise RuntimeError('Physical integrals differ')
        sources.append(dict(case=case,definition=d,local_integrals=str(file),amplitudes=str(amp),amplitudes_SHA256=record['amplitudes_sha256'],
            initialization_record=record,initialization_record_path=str(folder/'result.json'),initialization_record_SHA256=sha256((folder/'result.json').read_bytes())))
    inventory=load(ROOT/'analysis/N17-sampling-stage-inventory.json');originals=[]
    for r in inventory['rows']:
        if r['method']!='ORIGINAL':continue
        row=dict(r)
        if r['status']=='COMPLETED':
            path=N17/'decoded'/(r['draw_id']+'.npz');proof=load(path.with_suffix('.json'))
            if sha256(path.read_bytes())!=proof['decoded_SHA256'] or proof['actual_shots']!=100000:raise RuntimeError('Old actual100k draw required')
            row.update(payload=str(path),payload_SHA256=proof['decoded_SHA256'])
        originals.append(row)
    if len(originals)!=36 or sum(r['status']=='COMPLETED' for r in originals)!=30:raise RuntimeError('All36original rows incl6missing required')
    obs=load(N23/'collected/observations.json');control=[]
    for r in load(ROOT/'analysis/N23-backbone-uniform-summary.json')['rows']:
        if r['method']!='UNRESTRICTED_HF_HCI':continue
        o=next(o for o in obs if o['id']==r['observation_id']);file=N23/'collected'/o['vector_file']
        if sha256(file.read_bytes())!=o['vector_SHA256']:raise RuntimeError('Bound classical HCI root required')
        control.append(dict(case=r['case'],vector=str(file),vector_SHA256=o['vector_SHA256'],observation_id=o['id'],
            actual_old_CI_dimension=r['actual_CI_dimension'],old_CPU_seconds=r['standalone_native_plus_CI_CPU_seconds'],
            semantics='RETROSPECTIVE_CLASSICAL_HF_HCI_MARGINAL_CONTROL_NOT_MATCHED_COLD_WORK'))
    library=ROOT/'runs/N17-M4-CI-WITNESS-v1/library/libjoint_ci.dylib';build=load(library.parent/'build.json')
    if sha256(library.read_bytes())!=build['binary_SHA256'] or sha256((ROOT/'frontierloop/joint_ci_sparse.c').read_bytes())!=build['source_SHA256']:raise RuntimeError('Qualified Mac kernel changed')
    LOCAL.mkdir()
    save(LOCAL/'protocol.json',dict(version='N25_REAL_PAIR_T2_LOWSHOT_COMMON_ACTUAL_D_EXPLORATORY_V1',sources=sources,originals=originals,classical_controls=control,
        code_SHA256=sha256(Path(__file__).read_bytes()),helper_SHA256={name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in ('paired_resource_preflight.py','slater_prefix_preflight.py','joint_ci_sparse.py','joint_ci_sparse.c')},
        initializer_helper_SHA256=sha256((OLD/'paperstudy/lucj.py').read_bytes()),qualification_SHA256=sha256(test.read_bytes()),
        library=str(library),library_SHA256=build['binary_SHA256'],replicates=[250101,250102,250103],shots=1000,R_cap=32,
        packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','qiskit','ffsim','pyscf')},
        plan_SHA256=sha256((ROOT/'analysis/N25_PAIR_LOW_SHOT_PLAN_JA.md').read_bytes()),
        max_wall_seconds=600,max_CI_wall_seconds=30,triangle_cap=8000000,max_joint_D=1024,
        maximum_new_model_equivalent_samples=12000,maximum_new_classical_uniform_samples=12000,
        old_100k_source_costs_not_erased=True,no_molecular_VQE_or_FCI=True,independent_confirmation=False,goal_complete=False))
    print('N25_FROZEN4CASES_12PAIR12UNIFORM30SUBSETS6MISSING_CLASSICALCONTROLS',flush=True)


def run():
    capacity();p=load(LOCAL/'protocol.json')
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing execution intent; reconcile, never duplicate')
    if sha256(Path(__file__).read_bytes())!=p['code_SHA256'] or sha256(Path(p['library']).read_bytes())!=p['library_SHA256']:raise RuntimeError('Implementation or kernel changed')
    for name,h in p['helper_SHA256'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=h:raise RuntimeError('Frozen helpers changed')
    if sha256((OLD/'paperstudy/lucj.py').read_bytes())!=p['initializer_helper_SHA256'] or {n:importlib.metadata.version(n) for n in p['packages']}!=p['packages']:raise RuntimeError('Runtime changed')
    cpu,wall=process_time(),perf_counter();save(LOCAL/'started.json',dict(pid=__import__('os').getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    draws=[];resources=[];attempts=[];states=[];roots=[];cache={};phase={}
    def bound():
        capacity()
        if perf_counter()-wall>p['max_wall_seconds']:raise TimeoutError('N25 global resource cap')
    def counts_record(case,method,replica,ids,counts,role,meta):
        d=next(s['definition'] for s in p['sources'] if s['case']==case);n=d['norb'];ne=d['nelec']
        if any((int(x)&((1<<n)-1)).bit_count()!=ne[0] or (int(x)>>n).bit_count()!=ne[1] for x in ids):raise ValueError('Wrong actual particle sector')
        if int(np.sum(counts))!=p['shots']:raise ValueError('Wrong actual sample count')
        path=LOCAL/'draws'/case/method/(str(replica)+'.npz');write(path,arrays(blocked_ids=ids,counts=counts))
        r=dict(case=case,method=method,replicate=replica,sample_role=role,path=str(path),SHA256=sha256(path.read_bytes()),
            rank=ranked_half(ids,counts,n,(1<<ne[0])-1),readout_samples=p['shots'],**meta)
        draws.append(r);save(LOCAL/'draws.json',draws)
    try:
        tick=process_time()
        for source in p['sources']:
            bound();case=source['case'];d=source['definition'];n=d['norb'];k=d['nelec'][0]
            for path,key in [('amplitudes','amplitudes_SHA256'),('initialization_record_path','initialization_record_SHA256')]:
                if sha256(Path(source[path]).read_bytes())!=source[key]:raise RuntimeError('Initialization changed')
            t1,t2,kind=amplitudes_from_saved(source['amplitudes'],n,k);ids,v,angles=paired_sector(n,k,t2)
            write(LOCAL/'sources'/case/'pair-state.npz',arrays(half_strings=ids,amplitudes=v))
            save(LOCAL/'sources'/case/'angles.json',dict(angles=angles,amplitude_kind=kind,registered_initialization=source['initialization_record']['selected_method']))
            for full in (False,True):
                qc=paired_circuit(n,k,angles,full);views=[]
                for level in (1,3):
                    prep,samp,res=compile_views(qc,level);files={}
                    for role,q in [('preparation',prep),('sampling',samp)]:
                        stream=BytesIO();qpy.dump(q,stream);blob=stream.getvalue();path=LOCAL/'sources'/case/('FULL_SPIN_COPY' if full else 'PAIR_ENCODING')/f'opt{level}-{role}.qpy';write(path,blob)
                        files[role]=dict(path=str(path),SHA256=sha256(blob))
                    views.append(dict(resources=res,circuits=files))
                resources.append(dict(case=case,view='FULL_SPIN_COPY' if full else 'PAIR_ENCODING',n_qubits=qc.num_qubits,
                    declared_parameters=k*(n-k),nonzero_initialized_angles=sum(e['theta']!=0 for e in angles),compiler_points=views))
            state_sha=sha256((LOCAL/'sources'/case/'pair-state.npz').read_bytes())
            for replica in p['replicates']:
                bound()
                for method in ('PAIR_T2','UNIFORM_SECTOR'):
                    rngseed=seed(case,method,replica,state_sha if method=='PAIR_T2' else d['problem_id']);rng=np.random.default_rng(rngseed)
                    a=rng.choice(ids,size=p['shots'],p=v*v if method=='PAIR_T2' else None)
                    b=a if method=='PAIR_T2' else rng.choice(ids,size=p['shots'])
                    raw=a|(b<<np.uint64(n));values,counts=np.unique(raw,return_counts=True)
                    counts_record(case,method,replica,values,counts,'NEW_EXACT_PAIR_MODEL_EQUIVALENT' if method=='PAIR_T2' else 'NEW_UNIFORM_CLASSICAL',dict(seed=rngseed,new_model_equivalent_samples=p['shots'] if method=='PAIR_T2' else 0,new_uniform_classical_samples=p['shots'] if method=='UNIFORM_SECTOR' else 0,new_QPU_shots=0))
        phase['pair_sources_compile_and24_new_draws_CPU_seconds']=process_time()-tick
        tick=process_time()
        for old in p['originals']:
            replica=p['replicates'][old['replicate']-160101];method=old['source']
            if old['status']!='COMPLETED':
                for arm in ('R32_CAP','COMMON_R'):states.append(dict(case=old['case'],method=method,replicate=replica,arm=arm,status='MISSING_SAME_PROBLEM_SOURCE',energy_Ha=None))
                continue
            if sha256(Path(old['payload']).read_bytes())!=old['payload_SHA256']:raise RuntimeError('Parent100k counts changed')
            rngseed=seed(old['case'],method,replica,old['payload_SHA256'])
            with np.load(old['payload'],allow_pickle=False) as a:ids,counts=thin_counts(a['blocked_ids'],a['counts'],p['shots'],rngseed)
            counts_record(old['case'],method,replica,ids,counts,'RETROSPECTIVE1000_OF100000_NOT_NEW_INDEPENDENT_SOURCE',dict(seed=rngseed,new_model_equivalent_samples=0,new_uniform_classical_samples=0,new_QPU_shots=0,old_actual_samples=100000,old_draw_id=old['draw_id'],old_source=old))
        phase['retrospective30_subsets_CPU_seconds']=process_time()-tick
        save(LOCAL/'resources.json',resources)
        controls={}
        for c in p['classical_controls']:
            path=Path(c['vector'])
            if sha256(path.read_bytes())!=c['vector_SHA256']:raise RuntimeError('Classical root changed')
            with np.load(path,allow_pickle=False) as a:
                pairs=a['joint_pairs'];v=a['coefficients'];n=next(s['definition']['norb'] for s in p['sources'] if s['case']==c['case'])
                if abs(v@v-1)>1e-10:raise ValueError('Classical control norm differs')
                ids=pairs[:,0]|(pairs[:,1]<<np.uint64(n));hf=(1<<next(s['definition']['nelec'][0] for s in p['sources'] if s['case']==c['case']))-1
                controls[c['case']]=ranked_half(ids,v*v,n,hf)
        for s in p['sources']:
            case=s['case'];d=s['definition'];file=Path(s['local_integrals'])
            if sha256(file.read_bytes())!=d['integrals_SHA256']:raise RuntimeError('Physical integrals changed')
            with np.load(file,allow_pickle=False) as a:h=a['h1'];g=a['eri']
            for replica in p['replicates']:
                local=[r for r in draws if r['case']==case and r['replicate']==replica]
                local.append(dict(case=case,replicate=replica,method='CLASSICAL_HF_HCI_MARGINAL_ORACLE',rank=controls[case],sample_role='PRECOMPUTED_CLASSICAL_CONTROL_NOT_FREE_OR_EQUAL_COLD_WORK'))
                common=min(p['R_cap'],*(len(r['rank']) for r in local))
                for source in local:
                    for arm,R in [('R32_CAP',min(p['R_cap'],len(source['rank']))),('COMMON_R',common)]:
                        bound();strings=sorted(source['rank'][:R]);pairs=np.array([(a,b) for a in strings for b in strings],np.uint64)
                        key=digest([case,pairs.tolist()]);out=dict(case=case,method=source['method'],replicate=replica,arm=arm,R=R,actual_D=R*R,half_strings=strings,source_role=source['sample_role'])
                        if key in cache:out.update(cache[key],new_CI_solve=False)
                        else:
                            attempt=dict(case=case,basis_id=key,status='STARTED',R=R);attempts.append(attempt);save(LOCAL/'CI-attempts.json',attempts);tick=process_time()
                            try:
                                result,v=solve(h,g,d['core_energy'],tuple(d['nelec']),pairs,p['library'],seconds=p['max_CI_wall_seconds'],triangle_cap=p['triangle_cap'])
                                path=LOCAL/'roots'/(key+'.npz');write(path,arrays(joint_pairs=pairs,coefficients=v))
                                cache[key]=dict(status='COMPLETED',result=result,root_path=str(path),root_SHA256=sha256(path.read_bytes()),basis_id=key)
                                out.update(cache[key],new_CI_solve=True);attempt.update(status='COMPLETED',root_path=str(path));roots.append(key)
                            except Exception as exc:
                                cache[key]=dict(status='FAILED_PRESERVED',error=repr(exc),basis_id=key,result=None)
                                out.update(cache[key],new_CI_solve=False);attempt.update(status='FAILED_PRESERVED',error=repr(exc))
                            finally:attempt['CPU_seconds']=process_time()-tick;save(LOCAL/'CI-attempts.json',attempts)
                        states.append(out);save(LOCAL/'progress.json',dict(states=states,actual_new_CI_roots=len(roots)))
                print('N25',case,replica,'common_R',common,'terminal',len(states),flush=True)
        save(LOCAL/'terminal.json',dict(status='N25_ALL_FROZEN_LOW_SHOT_STATES_TERMINAL',states=states,draws=draws,resources=resources,
            actual_new_CI_roots=len(roots),failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),
            actual_new_model_equivalent_samples=sum(r['new_model_equivalent_samples'] for r in draws),actual_new_uniform_classical_samples=sum(r['new_uniform_classical_samples'] for r in draws),
            retrospective_old_counts_subsets=sum('RETROSPECTIVE' in r['sample_role'] for r in draws),
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,phase_costs=phase,
            protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),new_QPU_shots=0,GPU_allocations=0,independent_confirmation=False,goal_complete=False))
        print('N25_TERMINAL',len(states),len(roots),flush=True)
    except Exception as exc:
        save(LOCAL/'failed.json',dict(status='FAILED_PRESERVED_DO_NOT_BLIND_RESTART',error=repr(exc),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
            completed_draws=len(draws),completed_states=len(states),actual_new_CI_roots=len(roots)))
        raise


if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()
