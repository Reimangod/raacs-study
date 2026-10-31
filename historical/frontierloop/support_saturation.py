"""N64: privileged fixed-source probability diagnosis, NOT new sampling."""
from collections import defaultdict
from pathlib import Path
from time import process_time,perf_counter
import gzip,json,os,sys
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,write,arrays,sha256,digest,space
from .full_N5_plan import LOCAL as N63,verified as verified63
from .filtered_SqDRIFT_plan import LOCAL as N57,verified as verified57
from .joint_ci_sparse import solve
from .contiguous_CI_audit import audit
from paperstudy.sqdrift_execution import member_state
from paperstudy.ci import interleaved_to_blocked

LOCAL=ROOT/'runs/N64-FIXED-SOURCE-SUPPORT-v1'

def half_distribution(probabilities,n,hf):
    weights=defaultdict(float);mask=(1<<n)-1
    for b,w in probabilities.items():
        if not np.isfinite(w) or w<0:raise ValueError('Invalid probability')
        if w>0:
            weights[b&mask]+=w/2;weights[b>>n]+=w/2
    if abs(sum(weights.values())-1)>1e-8:raise ValueError('Distribution not normalized')
    weights.setdefault(hf,0.)
    order=[hf]+sorted((x for x in weights if x!=hf),key=lambda x:(-weights[x],x))
    return dict(weights),order

def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N64; no replace')
    space();p=verified63();old=verified57();draws=[];bindings={}
    for base,pp,arm in [(N63,p,'FULL_N5'),(N57,old,'FILTERED_SHORT')]:
        bindings[str(base/'protocol.json')]=sha256((base/'protocol.json').read_bytes())
        for d in pp['draws']:
            if d['arm']!=arm:continue
            draws.append(dict(draw=d,base=str(base)))
            for f in [base/'draws'/d['draw_id']/'counts.npz',base/'draws'/d['draw_id']/'result.json']:
                bindings[str(f)]=sha256(f.read_bytes())
            for m in d['members']:
                f=base/'exports'/d['case']/'members'/m['member_id']/'member.json.gz'
                bindings[str(f)]=sha256(f.read_bytes())
    assert len(draws)==24
    caches=[N63/'CI-terminal.json',N57/'partial-CI-terminal-v3.json',ROOT/'runs/N62-FILTER-ONLY-N20-v1/targeted-successor-v2/CI-terminal.json']
    for c in p['cases']:
        for k in ['meta','integrals']:bindings[c[k]]=sha256(Path(c[k]).read_bytes())
    import xml.etree.ElementTree as ET
    test=ROOT/'qualification-N64-preexecution-v1.xml'
    suites=list(ET.parse(test).getroot().iter('testsuite'))
    if sum(int(s.get('tests',0)) for s in suites)!=7 or any(int(s.get(k,0)) for s in suites for k in ['failures','errors']):raise ValueError('Seven preexecution checks required')
    for f in caches+[Path(__file__),ROOT/'frontierloop/joint_ci_sparse.py',ROOT/'frontierloop/contiguous_CI_audit.py',OLD/'paperstudy/sqdrift_execution.py',Path(p['library']),ROOT/'analysis/N64_SUPPORT_SATURATION_PLAN_JA.md',test]:
        bindings[str(f)]=sha256(f.read_bytes())
    save(LOCAL/'protocol.json',dict(version='N64_PRIVILEGED_FIXED_SOURCE_SUPPORT_DIAGNOSTIC',draws=draws,cases=p['cases'],R_caps=[32,128],
        CPU_cap=90,wall_cap=180,root_wall_cap=10,new_output_bytes=2*2**20,root_cap=2**30,
        library=p['library'],library_SHA256=p['library_SHA256'],common_equation=p['common_equation'],
        cache_files=list(map(str,caches)),bindings=bindings,new_model_samples=0,new_QPU_shots=0,
        probability_cutoff=0,independent_confirmation=False,operational_algorithm=False,goal_complete=False))
    print('N64_FROZEN24',flush=True)

def run():
    p=load(LOCAL/'protocol.json');space()
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing attempt; do not repeat')
    for f,h in p['bindings'].items():
        if sha256(Path(f).read_bytes())!=h:raise ValueError('Input drift '+f)
    save(LOCAL/'started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    cpu,wall=process_time(),perf_counter();rows=[];attempts=[];cache={}
    for f in p['cache_files']:
        for r in load(f)['states']:
            if r['status']=='COMPLETED':cache[r['basis_id']]=r['root']
    for item in p['draws']:
        d=item['draw'];base=Path(item['base']);c=next(c for c in p['cases'] if c['case']==d['case']);meta=load(c['meta'])
        row=dict(case=d['case'],seed=d['seed'],arm=d['arm'],draw_id=d['draw_id'],status='NOT_RUN_BUDGET',evaluations=[])
        try:
            if process_time()-cpu+11>p['CPU_cap'] or perf_counter()-wall+11>p['wall_cap']:raise TimeoutError('Global admission cap')
            prob=defaultdict(float);norm_errors=[]
            for m in d['members']:
                member=json.loads(gzip.decompress((base/'exports'/d['case']/'members'/m['member_id']/'member.json.gz').read_bytes()))
                assert member['spec']==m['spec']
                state,detail=member_state(meta,member);norm_errors.append(abs(detail['norm_squared']-1))
                for x,a in state.items():prob[interleaved_to_blocked(int(x),meta['norb'])]+=m['shots']/1000*float(abs(a)**2)
            hf=(1<<meta['nalpha'])-1;weights,order=half_distribution(prob,meta['norb'],hf)
            with np.load(base/'draws'/d['draw_id']/'counts.npz',allow_pickle=False) as a:
                observed=set()
                for x in a['blocked_ids']:
                    observed.update([int(x)&((1<<meta['norb'])-1),int(x)>>meta['norb']])
            row.update(status='DISTRIBUTION_COMPLETED',half_support=len(order),full_support=sum(v>0 for v in prob.values()),
                observed_half_count=len(observed),observed_half_probability_mass=sum(weights.get(x,0) for x in observed),
                HF_probability=prob.get(hf|(hf<<meta['norb']),0.),max_member_norm_error=max(norm_errors))
            with np.load(c['integrals'],allow_pickle=False) as a:h,g=a['h1'],a['eri']
            for cap in p['R_caps']:
                strings=sorted(order[:cap]);key=digest([meta['problem_id'],strings,p['common_equation'],p['library_SHA256']]);fresh=key not in cache
                if fresh:
                    tick=process_time();entry=dict(status='NOT_RUN_BUDGET')
                    if tick-cpu+11<=p['CPU_cap'] and perf_counter()-wall+11<=p['wall_cap']:
                        try:
                            pairs=np.asarray([(a,b) for a in strings for b in strings],np.uint64)
                            result,v=solve(h,g,meta['core_energy'],(meta['nalpha'],meta['nbeta']),pairs,Path(p['library']),seconds=p['root_wall_cap'],triangle_cap=8000000)
                            check=audit(h,g,meta['core_energy'],(meta['nalpha'],meta['nbeta']),strings,v,result)
                            blob=arrays(joint_pairs=pairs,coefficients=v);f=LOCAL/'roots'/(key+'.npz')
                            if sum(x.stat().st_size for x in LOCAL.rglob('*') if x.is_file())+len(blob)>p['new_output_bytes']:raise OSError('Output cap')
                            write(f,blob);entry=dict(status='COMPLETED',result=result,independent_RDM_audit=check,root_file=str(f),root_SHA256=sha256(blob))
                        except Exception as exc:entry=dict(status='FAILED_PRESERVED',error=repr(exc))
                    cache[key]=entry;attempts.append(dict(basis_id=key,status=entry['status'],CPU_seconds=process_time()-tick))
                entry=cache[key]
                if entry['status']=='COMPLETED':assert sha256(Path(entry['root_file']).read_bytes())==entry['root_SHA256']
                row['evaluations'].append(dict(R_cap=cap,actual_R=len(strings),actual_D=len(strings)**2,basis_id=key,root=entry,new_root=fresh,
                    full_model_half_support_used=len(strings)==len(order),selected_half_probability_mass=sum(weights[x] for x in strings)))
        except Exception as exc:row.update(status='FAILED_OR_CAP_PRESERVED',error=repr(exc))
        rows.append(row);save(LOCAL/'progress.json',dict(rows=rows,attempts=attempts,CPU_seconds=process_time()-cpu))
        print('N64',len(rows),d['case'],d['arm'],row['status'],row.get('half_support'),flush=True)
    save(LOCAL/'terminal.json',dict(status='ALL24_DIAGNOSTIC_STATUSES_RECORDED',rows=rows,attempts=attempts,
        CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,new_model_samples=0,new_QPU_shots=0,
        privileged_probability=True,independent_confirmation=False,goal_complete=False))

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
