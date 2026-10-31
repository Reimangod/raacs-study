"""Bounded execution of N76 frozen sources. No new source/policy search."""
from pathlib import Path
from time import process_time, perf_counter
import gzip, json, os, signal, sys, traceback
import numpy as np
from .paths import ROOT, OLD
from .io import load, save, write, arrays, sha256, digest, space
from .fixed_confirmation_inventory import LOCAL as INVENTORY

LOCAL = ROOT/'runs/N76-FIXED-CONFIRMATION-EXECUTION-v1'
SCIENCE = INVENTORY/'confirmation-science-freeze-v1.json'
SCIENCE_SHA = 'ebd4c4b50b4da7049ae015a70118ca7ab15532ea8fcff60cc21e699e1bc93dab'

def split(total, n):
    if total < 0 or n < 1: raise ValueError('Invalid allocation')
    return [total//n + (i < total % n) for i in range(n)]

def seed(problem, method, member, replicate):
    return int(digest(['CONFIRM20261027', problem, method, member, replicate])[:8], 16)

def member_id(method, member):
    if method == 'ACTIVE_GROUP_FIXED_LIBRARY': return member['source_SHA256']
    if method == 'B4_FIXED_SELECTED_ENSEMBLE': return member['id']
    if method.startswith('C5_C_LUCJ'): return member['circuit_id']
    return member['member_id']

def members(cell):
    row=cell['source_input'];method=cell['method']
    return [row['circuit']] if method.startswith('C5_C_LUCJ') else row['members']

def allocations(cell):
    mm=members(cell);total=cell['planned_final_model_samples_per_replicate']
    if cell['method'].startswith('SQDRIFT') or cell['method']=='ACTIVE_GROUP_FIXED_LIBRARY':
        times=sorted({m['time'] for m in mm});counts={}
        for t,budget in zip(times,split(total,len(times))):
            indices=[i for i,m in enumerate(mm) if m['time']==t]
            for i,n in zip(indices,split(budget,len(indices))):counts[i]=n
        return [counts[i] for i in range(len(mm))]
    return split(total,len(mm))

def plan(science):
    requests=[]
    for cell in science['cells']:
        if cell['status']=='MISSING_SAME_PROBLEM_SOURCE' or cell['method']=='CLASSICAL_HF_HCI':continue
        method=cell['method'];mm=members(cell)
        for member,shots in zip(mm,allocations(cell)):
            mid=member_id(method,member)
            backend='A100' if method.startswith('C5_C_LUCJ') and cell['case']!='NAPH20-BLOCK-BOYS' else 'M4'
            for rep in science['replicates']:
                request=dict(case=cell['case'],method=method,problem_id=cell['problem_id'],member_id=mid,replicate=rep,
                    seed=seed(cell['problem_id'],method,mid,rep),shots=shots,backend=backend)
                request['id']=digest(request);requests.append(request)
    assert len(requests)==556 and len({r['id'] for r in requests})==556
    assert sum(r['shots'] for r in requests)==sum(c['planned_new_model_samples'] for c in science['cells'])
    return requests

def check(science):
    if sha256(SCIENCE.read_bytes())!=SCIENCE_SHA:raise RuntimeError('Frozen science changed')
    for file,h in science['bindings'].items():
        if sha256(Path(file).read_bytes())!=h:raise RuntimeError('Frozen input drift '+file)

def capacity(extra=0):
    space();science=load(SCIENCE)
    local=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())
    root=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    if local+extra>science['storage']['new_bytes'] or root+extra>science['storage']['root_logical_bytes']:
        raise OSError('Confirmation storage cap; preserve evidence, no deletion')

def checked_write(path,blob):capacity(len(blob));write(path,blob)

def bind():
    if (LOCAL/'execution-binding.json').exists():raise RuntimeError('Already bound; inspect existing work')
    science=load(SCIENCE);check(science);requests=plan(science);capacity()
    from .prefix_performance_sampling import qualified_module,QUAL,LOCAL as N17
    qualified_module()
    code=[Path(__file__),OLD/'paperstudy/sqdrift_execution.py',OLD/'paperstudy/sparse_raacs.py',OLD/'paperstudy/ci.py',
        QUAL/'primitive_plan.py',N17/'worker.py',N17/'base_kernel.py',ROOT/'tests/test_fixed_confirmation_execution.py']
    import xml.etree.ElementTree as ET
    test=ROOT/'qualification-fixed-confirmation-execution-v1.xml'
    suites=list(ET.parse(test).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)>=6 and not any(int(s.get(k,0)) for s in suites for k in ('errors','failures'))
    code.append(test)
    save(LOCAL/'execution-binding.json',dict(science_file=str(SCIENCE),science_SHA256=SCIENCE_SHA,
        code_bindings={str(f):sha256(f.read_bytes()) for f in code},requests=requests,
        source_policies_changed=False,new_source_search_authorized=False,actual_new_model_samples=0,
        RNG_M4='numpy.default_rng(seed).multinomial(shots, p); sorted blocked IDs',
        RNG_A100='qualified byte-identical N17 worker CuPy RandomState(seed) inverse CDF',
        replicas_independent_readout_given_frozen_source_not_independent_construction=True))
    print('EXECUTION_BOUND556_REQUESTS_5146624_NEW_MODEL',flush=True)

def bound():
    b=load(LOCAL/'execution-binding.json');science=load(SCIENCE);check(science)
    for f,h in b['code_bindings'].items():
        if sha256(Path(f).read_bytes())!=h:raise RuntimeError('Execution code drift '+f)
    return science,b

def sample(ids,amplitudes,shots,rng_seed,n,nelec):
    ids=np.asarray(ids,np.uint64);v=np.asarray(amplitudes,complex)
    if ids.ndim!=1 or v.shape!=ids.shape or len(np.unique(ids))!=len(ids) or not np.isfinite(v).all():raise ValueError('Invalid source')
    mask=(1<<n)-1
    if any((int(b)&mask).bit_count()!=nelec[0] or (int(b)>>n).bit_count()!=nelec[1] or int(b)>>2*n for b in ids):
        raise ValueError('Source outside bound spin particle sector')
    p=abs(v)**2;norm=float(p.sum())
    if abs(norm-1)>1e-8:raise ValueError('Source lost normalization')
    order=np.argsort(ids);ids=ids[order];p=p[order]
    cc=np.random.default_rng(rng_seed).multinomial(shots,p/p.sum());keep=cc>0
    return ids[keep],cc[keep].astype(np.uint64),norm

def source(cell,member,meta):
    from paperstudy.ci import interleaved_to_blocked
    method=cell['method'];n=meta['norb']
    if method=='ACTIVE_GROUP_FIXED_LIBRARY':
        with np.load(member['source_file'],allow_pickle=False) as a:
            return np.array([interleaved_to_blocked(int(x),n) for x in a['source_mode_ids']],np.uint64),a['amplitudes'].copy(),{'source_file_SHA256':member['source_SHA256']}
    if method=='B4_FIXED_SELECTED_ENSEMBLE':
        from paperstudy.sparse_raacs import evolve
        state={int(meta['hf_id']):1+0j}
        for action in member['actions']:state=evolve(state,action,meta['n_qubits'])
        raw=sorted(state)
        return np.array([interleaved_to_blocked(x,n) for x in raw],np.uint64),np.array([state[x] for x in raw]),{'existing_author_QE_engine':True,'amplitude_truncation':False}
    if method.startswith('SQDRIFT'):
        from paperstudy.sqdrift_execution import member_state
        packet=json.loads(gzip.decompress(Path(member['source_packet']).read_bytes()));state,proof=member_state(meta,packet);raw=sorted(state)
        return np.array([interleaved_to_blocked(x,n) for x in raw],np.uint64),np.array([state[x] for x in raw]),proof
    if method.startswith('C5_C_LUCJ'):
        from .prefix_performance_sampling import qualified_module
        module=qualified_module();row=cell['source_input'];config=load(row['config_file'])
        with np.load(row['arrays_file'],allow_pickle=False) as a:aa={k:a[k] for k in a.files}
        v=module.numpy_state(config,aa);sa,sb=[module.strings(n,e) for e in config['nelec']]
        ids=np.repeat(sa.astype(np.uint64),len(sb))|(np.tile(sb.astype(np.uint64),len(sa))<<np.uint64(n))
        return ids,v,{'qualified_primitive_numpy_engine':True,'source_QPY_SHA256':member['original_QPY_SHA256']}
    raise ValueError('Unknown source method')

def run_m4():
    science,b=bound();start=LOCAL/'m4-started.json'
    if start.exists():raise RuntimeError('Existing intent; reconcile process and partial evidence, no blind restart')
    capacity();cpu,wall=process_time(),perf_counter();save(start,dict(pid=os.getpid(),binding_SHA256=sha256((LOCAL/'execution-binding.json').read_bytes())))
    # One member per method/case before a second: deterministic breadth-first scheduling.
    cells=[c for c in science['cells'] if any(r['case']==c['case'] and r['method']==c['method'] and r['backend']=='M4' for r in b['requests'])]
    tasks=[(c,mm[i]) for i in range(max(len(members(c)) for c in cells)) for c in cells for mm in [members(c)] if i<len(mm)]
    records=[];current=None;fatal=None
    def timeout(signum,frame):raise TimeoutError('Frozen M4 source CPU/wall cap reached')
    alarm=signal.signal(signal.SIGALRM,timeout);prof=signal.signal(signal.SIGPROF,timeout)
    signal.setitimer(signal.ITIMER_REAL,science['source']['M4_wall_cap']);signal.setitimer(signal.ITIMER_PROF,science['source']['M4_CPU_cap'])
    try:
        for cell,member in tasks:
            capacity();mid=member_id(cell['method'],member);reqs=[r for r in b['requests'] if r['case']==cell['case'] and r['method']==cell['method'] and r['member_id']==mid]
            current=dict(case=cell['case'],method=cell['method'],member_id=mid,requests=reqs)
            save(LOCAL/'m4-current.json',current);tick,w=process_time(),perf_counter()
            meta=load(next(c['meta'] for c in science['cases'] if c['case']==cell['case']))
            try:
                ids,v,proof=source(cell,member,meta);build=process_time()-tick
                for r in reqs:
                    out=LOCAL/'member-draws'/(r['id']+'.npz');intent=out.with_suffix('.intent.json')
                    if intent.exists() or out.exists():raise RuntimeError('Duplicate request intent')
                    save(intent,dict(request=r,stage='DRAW_STARTED'))
                    xx,cc,norm=sample(ids,v,r['shots'],r['seed'],meta['norb'],(meta['nalpha'],meta['nbeta']))
                    blob=arrays(blocked_ids=xx,counts=cc);checked_write(out,blob)
                    rec=dict(request=r,status='COMPLETED',counts_file=str(out),counts_SHA256=sha256(blob),actual_new_model_samples=int(cc.sum()),new_QPU_shots=0,norm_squared=norm,source_proof=proof)
                    save(out.with_suffix('.json'),rec);records.append(rec)
                save(LOCAL/'member-builds'/(digest([cell['case'],cell['method'],mid])+'.json'),dict(current,source_CPU_seconds=build,total_CPU_seconds=process_time()-tick,wall_seconds=perf_counter()-w,status='COMPLETED'))
                print('MEMBER_COMPLETED',cell['case'],cell['method'],mid[:12],len(records),flush=True)
            except TimeoutError:raise
            except Exception as exc:
                rec=dict(current,status='FAILED_SOURCE_PRESERVED',error=repr(exc),traceback=traceback.format_exc(),CPU_seconds=process_time()-tick)
                save(LOCAL/'member-builds'/(digest([cell['case'],cell['method'],mid])+'.json'),rec);records.append(rec)
            save(LOCAL/'m4-progress.json',dict(records=records,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall))
    except Exception as exc:fatal=dict(error=repr(exc),traceback=traceback.format_exc(),interrupted_member=current)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.setitimer(signal.ITIMER_PROF,0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGPROF,prof)
        complete=[r for r in records if r['status']=='COMPLETED']
        save(LOCAL/'m4-terminal.json',dict(status='BOUNDED_M4_STAGE_TERMINAL' if fatal is None else 'BOUNDED_M4_STAGE_STOPPED_PRESERVED',fatal=fatal,records=records,
            actual_new_model_samples=sum(r['actual_new_model_samples'] for r in complete),completed_member_requests=len(complete),planned_member_requests=sum(r['backend']=='M4' for r in b['requests']),
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,new_QPU_shots=0,goal_complete=False))
        print('M4_TERMINAL',len(complete),sum(r['actual_new_model_samples'] for r in complete),fatal and fatal['error'],flush=True)

if __name__=='__main__':{'bind':bind,'m4':run_m4}[sys.argv[1]]()
