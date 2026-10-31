"""N64B: separate diagnostic of numerical positive-support artifacts."""
from pathlib import Path
from time import process_time,perf_counter
import gzip,json,os,sys
import numpy as np
from .paths import ROOT
from .io import load,save,sha256,digest,write,arrays,space
from .support_saturation import LOCAL as PRIOR,half_distribution,member_state,interleaved_to_blocked
from .joint_ci_sparse import solve
from .contiguous_CI_audit import audit
LOCAL=ROOT/'runs/N64B-NUMERICAL-TAIL-SENSITIVITY-v1'

def selected(weights,hf,epsilon):
    if not 0<epsilon<1:raise ValueError('Diagnostic epsilon must be positive')
    return sorted({hf}|{int(x) for x,w in weights.items() if w>epsilon})

def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N64B; no replace')
    p=load(PRIOR/'protocol.json');t=load(PRIOR/'terminal.json');space()
    assert len(t['rows'])==24
    for f,h in p['bindings'].items():assert sha256(Path(f).read_bytes())==h
    save(LOCAL/'protocol.json',dict(prior_protocol_SHA256=sha256((PRIOR/'protocol.json').read_bytes()),
        prior_terminal_SHA256=sha256((PRIOR/'terminal.json').read_bytes()),code_SHA256=sha256(Path(__file__).read_bytes()),
        plan_SHA256=sha256((ROOT/'analysis/N64B_TAIL_SENSITIVITY_PLAN_JA.md').read_bytes()),
        epsilon=1e-12,CPU_cap=20,wall_cap=60,root_wall_cap=5,new_bytes=2*2**20,
        scope='POSTHOC_NUMERICAL_SENSITIVITY_NOT_EQUIVALENT_SOURCE_NOT_NEW_SAMPLER_NOT_EXACT_SUPPORT_PROOF',
        new_model_samples=0,new_QPU_shots=0,goal_complete=False))

def run():
    p=load(LOCAL/'protocol.json');q=load(PRIOR/'protocol.json');t=load(PRIOR/'terminal.json');space()
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing attempt no repeat')
    assert sha256(Path(__file__).read_bytes())==p['code_SHA256']
    assert sha256((PRIOR/'protocol.json').read_bytes())==p['prior_protocol_SHA256'] and sha256((PRIOR/'terminal.json').read_bytes())==p['prior_terminal_SHA256']
    for f,h in q['bindings'].items():assert sha256(Path(f).read_bytes())==h
    save(LOCAL/'started.json',dict(pid=os.getpid()));cpu,wall=process_time(),perf_counter();rows=[];cache={};attempts=[]
    for file in q['cache_files']+[str(PRIOR/'terminal.json')]:
        z=load(file)
        entries=z.get('states',[]) if 'states' in z else [dict(basis_id=e['basis_id'],root=e['root'],status=e['root']['status']) for r in z['rows'] for e in r['evaluations']]
        for e in entries:
            if e['status']=='COMPLETED':cache[e['basis_id']]=e['root']
    for item in q['draws']:
        d=item['draw'];base=Path(item['base']);c=next(c for c in q['cases'] if c['case']==d['case']);meta=load(c['meta']);row=dict(case=d['case'],seed=d['seed'],arm=d['arm'],draw_id=d['draw_id'])
        try:
            if process_time()-cpu+6>p['CPU_cap'] or perf_counter()-wall+6>p['wall_cap']:raise TimeoutError('Global admission')
            prob={}
            for m in d['members']:
                packet=json.loads(gzip.decompress((base/'exports'/d['case']/'members'/m['member_id']/'member.json.gz').read_bytes()));state,_=member_state(meta,packet)
                for bit,amp in state.items():
                    bit=interleaved_to_blocked(int(bit),meta['norb']);prob[bit]=prob.get(bit,0.)+m['shots']/1000*float(abs(amp)**2)
            hf=(1<<meta['nalpha'])-1;weights,order=half_distribution(prob,meta['norb'],hf);strings=selected(weights,hf,p['epsilon']);key=digest([meta['problem_id'],strings,q['common_equation'],q['library_SHA256']]);fresh=key not in cache
            if fresh:
                tick=process_time()
                with np.load(c['integrals'],allow_pickle=False) as a:h,g=a['h1'],a['eri']
                pairs=np.asarray([(a,b) for a in strings for b in strings],np.uint64)
                result,v=solve(h,g,meta['core_energy'],(meta['nalpha'],meta['nbeta']),pairs,Path(q['library']),seconds=p['root_wall_cap'],triangle_cap=8000000)
                check=audit(h,g,meta['core_energy'],(meta['nalpha'],meta['nbeta']),strings,v,result);f=LOCAL/'roots'/(key+'.npz');blob=arrays(joint_pairs=pairs,coefficients=v)
                if sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())+len(blob)>p['new_bytes']:raise OSError('Output cap')
                write(f,blob);cache[key]=dict(status='COMPLETED',result=result,root_file=str(f),root_SHA256=sha256(blob),independent_RDM_audit=check)
                attempts.append(dict(basis_id=key,CPU_seconds=process_time()-tick,status='COMPLETED'))
            entry=cache[key];assert sha256(Path(entry['root_file']).read_bytes())==entry['root_SHA256']
            prior=next(r for r in t['rows'] if r['draw_id']==d['draw_id']);full=next(e for e in prior['evaluations'] if e['R_cap']==128)
            row.update(status='COMPLETED',actual_R=len(strings),prior_numerical_R=prior['half_support'],removed_half_mass=sum(w for x,w in weights.items() if x not in strings),
                root=entry,basis_id=key,new_CI_root=fresh,delta_vs_all_numerical_support_mHa=1000*(entry['result']['energy_Ha']-full['root']['result']['energy_Ha']))
        except Exception as exc:row.update(status='FAILED_OR_CAP_PRESERVED',error=repr(exc))
        rows.append(row);save(LOCAL/'progress.json',dict(rows=rows,attempts=attempts))
    save(LOCAL/'terminal.json',dict(status='ALL24_TAIL_SENSITIVITY_STATUSES_RECORDED',rows=rows,attempts=attempts,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,new_model_samples=0,new_QPU_shots=0,goal_complete=False))
    print('N64B_TERMINAL',len(rows),len(attempts),flush=True)

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
