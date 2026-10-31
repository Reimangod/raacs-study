"""N70 fixed all-four paired-source ablation; measured draws and common CI."""
from pathlib import Path
from time import process_time,perf_counter
import os,sys,signal,traceback
import numpy as np
from .paths import ROOT
from .io import load,save,write,arrays,sha256,digest,space
from .correlated_source_qualification import LOCAL as N69
from .pair_group_preflight import LOCAL as N67
from .exact_group_synthesis import LOCAL as N68
from .full20_breadth_plan import LOCAL as N66
from .paired_lowshot import paired_sector,ranked_half
from .joint_ci_sparse import solve
from .contiguous_CI_audit import audit
from paperstudy.ci import interleaved_to_blocked
from paperstudy.lucj import amplitudes_from_saved

LOCAL=ROOT/'runs/N70-PAIR-GROUP-FINITE-v1'
ARMS=['PAIR_ONLY','PAIR_PLUS_GROUP']

def sample(ids,amplitudes,shots,seed):
    ids=np.asarray(ids,np.uint64);v=np.asarray(amplitudes,complex);prob=abs(v)**2
    if len(ids)!=len(set(map(int,ids))) or not np.isfinite(prob).all() or abs(float(prob.sum())-1)>1e-10:raise ValueError('Verified normalized unique source required')
    raw=np.random.default_rng(seed).choice(ids,size=shots,p=prob/prob.sum())
    return np.unique(raw,return_counts=True)

def seed(problem,arm,time,source):return int(digest(['N70',700101,problem,arm,time,source])[:16],16)%(1<<63)

def prepare():
    space()
    if (LOCAL/'protocol.json').exists():raise RuntimeError('Existing freeze; no replace')
    q=load(N69/'protocol.json');qt=load(N69/'terminal.json');p66=load(N66/'protocol.json')
    assert qt['status']=='ALL12_CORRELATED_SOURCE_TWO_ROUTE_CHECKS_PASSED' and len(qt['rows'])==12
    bindings=dict(q['bindings']);controls=[]
    for r in qt['rows']:
        f=Path(r['state_file']);assert sha256(f.read_bytes())==r['state_SHA256'];bindings[str(f)]=r['state_SHA256']
    for d in p66['draws']:
        f=N66/'draws'/d['draw_id']/'counts.npz';z=load(f.parent/'result.json');assert sha256(f.read_bytes())==z['counts_SHA256']
        controls.append(dict(case=d['case'],arm='REFERENCE_FULL_N20_30',path=str(f),SHA256=z['counts_SHA256'],role='OLD1000_MODEL_NOT_NEW_OR_INDEPENDENT_CONFIRMATION'))
    previous=load(ROOT/'runs/N25-PAIRED-LOWSHOT-v1/draws.json')
    for r in previous:
        if r['replicate']==250101 and r['method'].startswith('C5_C_LUCJ_'):
            controls.append(dict(case=r['case'],arm=r['method'],path=r['path'],SHA256=r['SHA256'],role='OLD1000_SUBSET_FROM100000_COLD_COST_NOT_MATCHED'))
    assert len(controls)==14
    for r in controls:
        assert sha256(Path(r['path']).read_bytes())==r['SHA256'];bindings[r['path']]=r['SHA256']
    for c in p66['cases']:
        bindings[c['integrals']]=c['integrals_SHA256'];bindings[c['meta']]=c['meta_SHA256']
    import xml.etree.ElementTree as ET
    tests=ROOT/'qualification-N70-policy-v1.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==3 and not any(int(s.get(k,0)) for s in suites for k in ('failures','errors'))
    for f in [Path(__file__),N69/'protocol.json',N69/'terminal.json',N67/'terminal.json',N68/'terminal.json',
        N66/'protocol.json',ROOT/'frontierloop/contiguous_CI_audit.py',ROOT/'analysis/N70_PAIR_GROUP_FINITE_PLAN_JA.md',tests,Path(p66['library'])]:
        bindings[str(f)]=sha256(f.read_bytes())
    for f,h in bindings.items():assert sha256(Path(f).read_bytes())==h,'Input drift '+f
    used=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    if used+4*2**20>2**30:raise OSError('Research quota')
    save(LOCAL/'protocol.json',dict(version='N70_PAIR_ONLY_VS_ONE_GROUP_FIXED_EXPLORATORY',bindings=bindings,sources=q['sources'],states=qt['rows'],cases=p66['cases'],
        controls=controls,arms=ARMS,seed_id=700101,shots_per_arm=1000,time_shots=[[1,334],[2,333],[3,333]],
        R_caps=[32,128],common_equation=p66['common_equation'],library=p66['library'],library_SHA256=p66['library_SHA256'],
        source_CPU_cap=30,CI_CPU_cap=60,CI_wall_per_root=15,wall_cap=180,new_bytes=4*2**20,root_bytes=2**30,
        actual_expected_new_model_samples=8000,new_QPU_shots=0,independent_confirmation=False,matched_cold_work=False,matched_LUCJ_compiler=False,goal_complete=False))
    print('N70_FROZEN8_NEW_DRAWS14_OLD_REFERENCES',flush=True)

def run():
    p=load(LOCAL/'protocol.json');space()
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing intent; no repeat')
    for f,h in p['bindings'].items():assert sha256(Path(f).read_bytes())==h,'Input drift '+f
    save(LOCAL/'started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    cpu,wall=process_time(),perf_counter();draws=[];rows=[];attempts=[];cache={}
    initial=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file());base=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    def timeout(signum,frame):raise TimeoutError('N70 frozen wall cap')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,p['wall_cap'])
    def bound():
        space();size=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())
        if size>p['new_bytes'] or base+size-initial>p['root_bytes']:raise OSError('N70 storage cap')
        if perf_counter()-wall>p['wall_cap']:raise TimeoutError('N70 wall budget')
    try:
        for s in p['sources']:
            bound();meta=load(s['meta']);n,k=meta['norb'],meta['nalpha'];_,t2,kind=amplitudes_from_saved(s['amplitudes'],n,k)
            half,v,_=paired_sector(n,k,t2);pairids=half|(half<<np.uint64(n));pairsha=digest([sha256(pairids.tobytes()),sha256(v.tobytes())])
            for arm in p['arms']:
                pooled={};members=[]
                for time,shots in p['time_shots']:
                    if arm=='PAIR_ONLY':ids,amps,sourceid=pairids,v,pairsha
                    else:
                        r=next(r for r in p['states'] if r['case']==s['case'] and r['time']==time)
                        with np.load(r['state_file'],allow_pickle=False) as a:
                            ids=np.array([interleaved_to_blocked(int(b),n) for b in a['source_mode_ids']],np.uint64);amps=a['amplitudes']
                        sourceid=r['state_SHA256']
                    rngseed=seed(meta['problem_id'],arm,time,sourceid);values,counts=sample(ids,amps,shots,rngseed)
                    for b,c in zip(values,counts):pooled[int(b)]=pooled.get(int(b),0)+int(c)
                    members.append(dict(time=time,shots=shots,seed=rngseed,source_SHA256=sourceid))
                ids=np.array(sorted(pooled),np.uint64);counts=np.array([pooled[int(b)] for b in ids],np.int64)
                assert counts.sum()==1000
                f=LOCAL/'draws'/f'{s["case"]}-{arm}.npz';blob=arrays(blocked_ids=ids,counts=counts);write(f,blob)
                rank=ranked_half(ids,counts,n,(1<<k)-1)
                draws.append(dict(case=s['case'],arm=arm,path=str(f),SHA256=sha256(blob),rank=rank,members=members,amplitude_kind=kind,
                    role='NEW_VERIFIED_SOURCE_MODEL_SAMPLES_NOT_QPU',actual_new_model_samples=1000,new_QPU_shots=0))
                save(LOCAL/'draws.json',draws);bound()
                if process_time()-cpu>p['source_CPU_cap']:raise TimeoutError('N70 source CPU cap')
        source_cpu=process_time()-cpu
        for r in p['controls']:
            c=next(c for c in p['cases'] if c['case']==r['case']);meta=load(c['meta'])
            with np.load(r['path'],allow_pickle=False) as a:
                assert a['counts'].sum()==1000;rank=ranked_half(a['blocked_ids'],a['counts'],meta['norb'],(1<<meta['nalpha'])-1)
            draws.append(dict(r,rank=rank,actual_new_model_samples=0,new_QPU_shots=0))
        assert len(draws)==22
        save(LOCAL/'draws.json',draws);plans=[]
        for c in p['cases']:
            local=[d for d in draws if d['case']==c['case']];common=min(32,*(len(d['rank']) for d in local))
            for d in local:
                for context,R in [('R32_CAP',min(32,len(d['rank']))),('R128_CAP',min(128,len(d['rank']))),('COMMON_ACTUAL_R',common)]:
                    plans.append(dict(case=d['case'],arm=d['arm'],context=context,strings=sorted(map(int,d['rank'][:R])),role=d['role']))
        assert len(plans)==66;save(LOCAL/'CI-plan.json',plans);ci_cpu=process_time()
        for r in plans:
            bound();c=next(c for c in p['cases'] if c['case']==r['case']);meta=load(c['meta']);strings=r['strings']
            key=digest([meta['problem_id'],strings,p['common_equation'],p['library_SHA256']]);fresh=key not in cache
            if fresh:
                tick=process_time();entry=dict(status='NOT_RUN_CI_GLOBAL_CAP');remaining=p['CI_CPU_cap']-(tick-ci_cpu)
                if remaining>0:
                    try:
                        with np.load(c['integrals'],allow_pickle=False) as a:h,g=a['h1'],a['eri']
                        pairs=np.array([(a,b) for a in strings for b in strings],np.uint64)
                        result,vector=solve(h,g,meta['core_energy'],(meta['nalpha'],meta['nbeta']),pairs,Path(p['library']),seconds=min(remaining,p['CI_wall_per_root']),triangle_cap=8000000)
                        check=audit(h,g,meta['core_energy'],(meta['nalpha'],meta['nbeta']),strings,vector,result)
                        f=LOCAL/'roots'/(key+'.npz');blob=arrays(joint_pairs=pairs,coefficients=vector);write(f,blob)
                        entry=dict(status='COMPLETED',result=result,independent_RDM_audit=check,root_file=str(f),root_SHA256=sha256(blob))
                    except Exception as exc:entry=dict(status='FAILED_PRESERVED',error=repr(exc),traceback=traceback.format_exc())
                entry['CPU_seconds']=process_time()-tick;cache[key]=entry;attempts.append(dict(basis_id=key,status=entry['status'],CPU_seconds=entry['CPU_seconds']))
                save(LOCAL/'roots'/(key+'.json'),entry)
            root=cache[key];rows.append(dict(r,basis_id=key,status=root['status'],root=root,new_CI_attempt=fresh,actual_R=len(strings),actual_D=len(strings)**2,
                energy_Ha=root.get('result',{}).get('energy_Ha')))
            save(LOCAL/'CI-progress.json',dict(rows=rows,attempts=attempts))
        save(LOCAL/'terminal.json',dict(status='ALL66_CI_STATUSES_RECORDED',draws=draws,rows=rows,attempts=attempts,
            source_CPU_seconds=source_cpu,CI_CPU_seconds=process_time()-ci_cpu,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
            actual_new_model_samples=8000,new_QPU_shots=0,actual_new_CI_roots=sum(a['status']=='COMPLETED' for a in attempts),
            failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),independent_confirmation=False,goal_complete=False))
        print('N70_TERMINAL66',len(attempts),flush=True)
    except Exception as exc:
        save(LOCAL/'failed.json',dict(error=repr(exc),draws=draws,rows=rows,attempts=attempts,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
