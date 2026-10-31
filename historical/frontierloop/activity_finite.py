"""N74 stratified common-random-number paired ablation, actual model draws."""
from pathlib import Path
from time import process_time,perf_counter
import os,sys,signal,traceback
import numpy as np
from .paths import ROOT
from .io import load,save,write,arrays,sha256,digest,space
from .activity_resource import LOCAL as N73
from .activity_resource_portfolio import LOCAL as N73B
from .state_activity_preflight import LOCAL as N72
from .correlated_source_qualification import LOCAL as N69
from .pair_group_finite import LOCAL as N70
from .paired_lowshot import paired_sector,ranked_half
from .joint_ci_sparse import solve
from .contiguous_CI_audit import audit
from paperstudy.ci import interleaved_to_blocked
from paperstudy.lucj import amplitudes_from_saved

LOCAL=ROOT/'runs/N74-ACTIVITY-FINITE-v1'
ARMS=['PAIR_ONLY','RANDOM_GROUP','ACTIVE_GROUP']

def common_seed(problem,rep,time):return int(digest(['N74_CRN',problem,rep,time])[:16],16)%(1<<63)

def inverse_cdf(ids,amplitudes,uniforms):
    ids=np.asarray(ids,np.uint64);v=np.asarray(amplitudes,complex);u=np.asarray(uniforms,float)
    if ids.ndim!=1 or v.shape!=ids.shape or len(set(map(int,ids)))!=len(ids) or not np.isfinite(v).all():raise ValueError('Unique finite source required')
    prob=abs(v)**2
    if abs(float(prob.sum())-1)>1e-10 or not np.isfinite(u).all() or np.any(u<0) or np.any(u>=1):raise ValueError('Normalized source and valid uniforms required')
    order=np.argsort(ids);ids=ids[order];prob=prob[order];cdf=np.cumsum(prob/prob.sum());cdf[-1]=1.
    raw=ids[np.searchsorted(cdf,u,side='right')]
    return np.unique(raw,return_counts=True)

def prepare():
    if (LOCAL/'protocol.json').exists():raise RuntimeError('Existing freeze; no replace')
    space();p73=load(N73/'protocol.json');t73=load(N73B/'terminal.json');p70=load(N70/'protocol.json');t72=load(N72/'terminal.json')
    assert len(t73['rows'])==12 and all(c['proceed_to_finite_screen'] for c in t73['cases'])
    bindings=dict(t73['bindings']);states=[]
    for arm,records in [('RANDOM_GROUP',p70['states']),('ACTIVE_GROUP',t72['selected'])]:
        for r in records:
            f=Path(r['state_file']);assert sha256(f.read_bytes())==r['state_SHA256'];bindings[str(f)]=r['state_SHA256']
            states.append(dict(case=r['case'],arm=arm,time=r['time'],state_file=str(f),state_SHA256=r['state_SHA256']))
    for c in p70['cases']:
        bindings[c['integrals']]=c['integrals_SHA256'];bindings[c['meta']]=c['meta_SHA256']
    import xml.etree.ElementTree as ET
    tests=ROOT/'qualification-N74-policy-v2.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==4 and not any(int(s.get(k,0)) for s in suites for k in ('failures','errors'))
    for f in [Path(__file__),N73/'protocol.json',N73/'terminal.json',N73B/'terminal.json',N72/'terminal.json',N70/'protocol.json',
        ROOT/'frontierloop/contiguous_CI_audit.py',ROOT/'analysis/N74_ACTIVITY_FINITE_PLAN_JA.md',tests,Path(p70['library'])]:bindings[str(f)]=sha256(f.read_bytes())
    for f,h in bindings.items():assert sha256(Path(f).read_bytes())==h,'Input drift '+f
    used=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    if used+8*2**20>2**30:raise OSError('Research logical quota')
    save(LOCAL/'protocol.json',dict(version='N74_STATE_ACTIVITY_MATCHED_THREE_ARM_CRN_READOUT',bindings=bindings,sources=p73['sources'],states=states,cases=p70['cases'],
        arms=ARMS,replicates=[740101,740102,740103],time_shots=p70['time_shots'],R_caps=[32,128],shots_per_draw=1000,
        common_equation=p70['common_equation'],library=p70['library'],library_SHA256=p70['library_SHA256'],
        source_CPU_cap=30,CI_CPU_cap=180,CPU_cap=210,wall_cap=300,root_wall_cap=15,triangle_cap=8000000,new_bytes=8*2**20,root_bytes=2**30,
        planned_new_model_samples=36000,new_QPU_shots=0,independent_source_confirmation=False,matched_cold_work=False,goal_complete=False))
    print('N74_FROZEN36DRAWS_36000MODEL_108CI_FIELDS',flush=True)

def run():
    p=load(LOCAL/'protocol.json');space()
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing execution intent; inspect process, no duplicate')
    for f,h in p['bindings'].items():assert sha256(Path(f).read_bytes())==h,'Input drift '+f
    save(LOCAL/'started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    cpu,wall=process_time(),perf_counter();draws=[];rows=[];attempts=[];cache={}
    initial=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file());base=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    def timeout(signum,frame):raise TimeoutError('N74 frozen CPU/wall cap')
    alarm=signal.signal(signal.SIGALRM,timeout);prof=signal.signal(signal.SIGPROF,timeout)
    signal.setitimer(signal.ITIMER_REAL,p['wall_cap']);signal.setitimer(signal.ITIMER_PROF,p['CPU_cap'])
    def bound(extra=0):
        space();size=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())
        if size+extra>p['new_bytes'] or base+size-initial+extra>p['root_bytes']:raise OSError('N74 storage cap')
    def artifact(f,blob):bound(len(blob));write(f,blob)
    try:
        for s in p['sources']:
            bound();meta=load(s['meta']);n,k=meta['norb'],meta['nalpha'];_,t2,kind=amplitudes_from_saved(s['amplitudes'],n,k);half,v,_=paired_sector(n,k,t2)
            source={('PAIR_ONLY',time):(half|(half<<np.uint64(n)),v,sha256(v.tobytes())) for time,_ in p['time_shots']}
            for r in [r for r in p['states'] if r['case']==s['case']]:
                with np.load(r['state_file'],allow_pickle=False) as a:
                    ids=np.array([interleaved_to_blocked(int(b),n) for b in a['source_mode_ids']],np.uint64);amps=a['amplitudes']
                source[(r['arm'],r['time'])]=(ids,amps,r['state_SHA256'])
            for rep in p['replicates']:
                uniforms={time:np.random.default_rng(common_seed(meta['problem_id'],rep,time)).random(shots) for time,shots in p['time_shots']}
                for arm in p['arms']:
                    pooled={};members=[]
                    for time,shots in p['time_shots']:
                        ids,vv,source_sha=source[(arm,time)];xx,cc=inverse_cdf(ids,vv,uniforms[time]);f=LOCAL/'draws'/s['case']/arm/f'{rep}-T{time}.npz';blob=arrays(blocked_ids=xx,counts=cc);artifact(f,blob)
                        for b,c in zip(xx,cc):pooled[int(b)]=pooled.get(int(b),0)+int(c)
                        members.append(dict(time=time,shots=shots,common_readout_seed=common_seed(meta['problem_id'],rep,time),source_SHA256=source_sha,
                            counts_file=str(f),counts_SHA256=sha256(blob)))
                    ids=np.array(sorted(pooled),np.uint64);counts=np.array([pooled[int(b)] for b in ids],np.int64);assert counts.sum()==1000
                    f=LOCAL/'draws'/s['case']/arm/f'{rep}.npz';blob=arrays(blocked_ids=ids,counts=counts);artifact(f,blob)
                    draws.append(dict(case=s['case'],arm=arm,replicate=rep,amplitude_kind=kind,rank=ranked_half(ids,counts,n,(1<<k)-1),members=members,
                        counts_file=str(f),counts_SHA256=sha256(blob),actual_new_model_samples=1000,new_QPU_shots=0))
                    save(LOCAL/'draws.json',draws)
                    if process_time()-cpu>p['source_CPU_cap']:raise TimeoutError('N74 source CPU cap')
        source_cpu=process_time()-cpu;plans=[]
        for s in p['sources']:
            for rep in p['replicates']:
                local=[d for d in draws if d['case']==s['case'] and d['replicate']==rep];common=min(32,*(len(d['rank']) for d in local))
                for d in local:
                    for context,R in [('R32_CAP',min(32,len(d['rank']))),('R128_CAP',min(128,len(d['rank']))),('COMMON_ACTUAL_R',common)]:
                        plans.append(dict(case=d['case'],arm=d['arm'],replicate=rep,context=context,strings=sorted(map(int,d['rank'][:R]))))
        assert len(draws)==36 and len(plans)==108;save(LOCAL/'CI-plan.json',plans);ci_cpu=process_time()
        for r in plans:
            bound();c=next(c for c in p['cases'] if c['case']==r['case']);meta=load(c['meta']);strings=r['strings'];key=digest([meta['problem_id'],strings,p['common_equation'],p['library_SHA256']]);fresh=key not in cache
            if fresh:
                tick=process_time();entry=dict(status='NOT_RUN_GLOBAL_CI_CAP');remaining=p['CI_CPU_cap']-(tick-ci_cpu)
                if remaining>0:
                    try:
                        with np.load(c['integrals'],allow_pickle=False) as a:h,g=a['h1'],a['eri']
                        pairs=np.array([(a,b) for a in strings for b in strings],np.uint64)
                        result,v=solve(h,g,meta['core_energy'],(meta['nalpha'],meta['nbeta']),pairs,Path(p['library']),seconds=min(remaining,p['root_wall_cap']),triangle_cap=p['triangle_cap'])
                        check=audit(h,g,meta['core_energy'],(meta['nalpha'],meta['nbeta']),strings,v,result);f=LOCAL/'roots'/(key+'.npz');blob=arrays(joint_pairs=pairs,coefficients=v);artifact(f,blob)
                        entry=dict(status='COMPLETED',result=result,independent_RDM_audit=check,root_file=str(f),root_SHA256=sha256(blob))
                    except Exception as exc:entry=dict(status='FAILED_PRESERVED',error=repr(exc),traceback=traceback.format_exc())
                entry['CPU_seconds']=process_time()-tick;cache[key]=entry;attempts.append(dict(basis_id=key,status=entry['status'],CPU_seconds=entry['CPU_seconds']));save(LOCAL/'roots'/(key+'.json'),entry)
            root=cache[key];rows.append(dict(r,basis_id=key,status=root['status'],root=root,new_CI_attempt=fresh,actual_R=len(strings),actual_D=len(strings)**2,energy_Ha=root.get('result',{}).get('energy_Ha')))
            save(LOCAL/'CI-progress.json',dict(rows=rows,attempts=attempts))
        save(LOCAL/'terminal.json',dict(status='ALL108_CI_STATUSES_RECORDED',draws=draws,rows=rows,attempts=attempts,
            actual_new_model_samples=36000,new_QPU_shots=0,actual_new_CI_roots=sum(a['status']=='COMPLETED' for a in attempts),failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),
            source_CPU_seconds=source_cpu,CI_CPU_seconds=process_time()-ci_cpu,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
            independent_source_confirmation=False,goal_complete=False));print('N74_TERMINAL108',len(attempts),flush=True)
    except Exception as exc:
        save(LOCAL/'failed.json',dict(error=repr(exc),draws=draws,rows=rows,attempts=attempts,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.setitimer(signal.ITIMER_PROF,0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGPROF,prof)

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
