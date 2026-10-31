"""N65 same-equation R-cap and pairwise actual-R readout; no new sampling."""
from pathlib import Path
from time import process_time,perf_counter
import os,traceback
import numpy as np
from .paths import ROOT
from .io import load,save,write,arrays,sha256,digest,space
from .ensemble_breadth_plan import LOCAL,verified
from .full_N5_plan import LOCAL as N63,verified as verified63
from .filtered_SqDRIFT_plan import LOCAL as N57,verified as verified57
from .paired_lowshot import ranked_half
from .joint_ci_sparse import solve
from .contiguous_CI_audit import audit

def run():
    p=verified();p63=verified63();p57=verified57();space()
    if (LOCAL/'CI-started.json').exists():raise RuntimeError('Existing readout; do not repeat')
    cost=load(LOCAL/'final-accounting.json');assert cost['process_children_reaped']
    sources={d['draw_id']:(d,LOCAL) for d in p['draws']}
    for d in p['draws']:
        pp,base=(p63,N63) if d['arm']=='FULL_N5' else (p57,N57)
        old=next(x for x in pp['draws'] if x['draw_id']==d['previous_9_member_draw_id'])
        sources[old['draw_id']]=(old,base)
    ranks={};cache={};plans=[]
    for id,(d,base) in sources.items():
        c=next(c for c in p['cases'] if c['case']==d['case']);meta=load(c['meta']);f=base/'draws'/id
        ranks[id]=None
        if (f/'result.json').exists():
            z=load(f/'result.json');assert z['problem_id']==d['problem_id'] and sha256((f/'counts.npz').read_bytes())==z['counts_SHA256']
            with np.load(f/'counts.npz',allow_pickle=False) as a:ranks[id]=list(map(int,ranked_half(a['blocked_ids'],a['counts'],meta['norb'],(1<<meta['nalpha'])-1)))
    reuse=[N63/'CI-terminal.json',N57/'partial-CI-terminal-v3.json',ROOT/'runs/N62-FILTER-ONLY-N20-v1/targeted-successor-v2/CI-terminal.json']
    for f in reuse:
        for r in load(f)['states']:
            if r['status']=='COMPLETED':cache[r['basis_id']]=r['root']
    def plan(d,context,cap,strings,pair=None):
        return dict(case=d['case'],seed=d['seed'],arm=d['arm'],draw_id=d['draw_id'],ensemble_size=30 if d['draw_id'] in {x['draw_id'] for x in p['draws']} else 9,
            context=context,R_cap=cap,strings=strings,pair=pair)
    for d in p['draws']:
        for cap in p['R_caps']:
            rr=ranks[d['draw_id']];plans.append(plan(d,'MAIN',cap,sorted(rr[:cap]) if rr is not None else None))
    for c in p['cases']:
        n5,short=[next(d for d in p['draws'] if d['case']==c['case'] and d['arm']==arm) for arm in ['FULL_N5','FILTERED_SHORT']]
        pairs=[('N5_9_TO30',[sources[n5['previous_9_member_draw_id']][0],n5]),
            ('SHORT_9_TO30',[sources[short['previous_9_member_draw_id']][0],short]),('BOTH_30',[n5,short])]
        for label,dd in pairs:
            rr=[ranks[d['draw_id']] for d in dd];size=min(32,*(len(r) for r in rr)) if all(r is not None for r in rr) else None
            for d,rank in zip(dd,rr):plans.append(plan(d,'COMMON_R',32,sorted(rank[:size]) if size is not None else None,label))
    assert len(plans)==40
    save(LOCAL/'CI-started.json',dict(pid=os.getpid(),plans=plans,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
        cache_files_SHA256={str(f):sha256(f.read_bytes()) for f in reuse}))
    cpu,wall=process_time(),perf_counter();states=[];attempts=[]
    for r in plans:
        c=next(c for c in p['cases'] if c['case']==r['case']);meta=load(c['meta']);strings=r['strings'];row=dict(r,status='SOURCE_MISSING',energy_Ha=None,actual_R=None,actual_D=None)
        if strings is not None:
            key=digest([meta['problem_id'],strings,p['common_equation'],p['library_SHA256']]);fresh=key not in cache
            row.update(basis_id=key,actual_R=len(strings),actual_D=len(strings)**2,new_CI_attempt=fresh)
            if fresh:
                tick=process_time();entry=dict(status='NOT_RUN_GLOBAL_CAP');remaining=p['CI_total_CPU_seconds']-(tick-cpu)
                if remaining>0:
                    try:
                        space()
                        with np.load(c['integrals'],allow_pickle=False) as a:h,g=a['h1'],a['eri']
                        pairs=np.asarray([(a,b) for a in strings for b in strings],np.uint64)
                        result,v=solve(h,g,meta['core_energy'],(meta['nalpha'],meta['nbeta']),pairs,Path(p['library']),seconds=min(remaining,p['CI_wall_seconds_per_root']),triangle_cap=8000000)
                        check=audit(h,g,meta['core_energy'],(meta['nalpha'],meta['nbeta']),strings,v,result);f=LOCAL/'roots'/(key+'.npz');blob=arrays(joint_pairs=pairs,coefficients=v)
                        if sum(x.stat().st_size for x in LOCAL.rglob('*') if x.is_file())+len(blob)>p['new_output_bytes']:raise OSError('Output cap')
                        write(f,blob);entry=dict(status='COMPLETED',root_file=str(f),root_SHA256=sha256(blob),result=result,independent_RDM_audit=check)
                    except Exception as exc:entry=dict(status='FAILED_PRESERVED',error=repr(exc),traceback=traceback.format_exc())
                entry['attempt_CPU_seconds']=process_time()-tick;cache[key]=entry
                attempts.append(dict(basis_id=key,status=entry['status'],CPU_seconds=entry['attempt_CPU_seconds']));save(LOCAL/'roots'/(key+'.json'),entry)
            root=cache[key];row.update(status=root['status'],root=root)
            if root['status']=='COMPLETED':
                assert sha256(Path(root['root_file']).read_bytes())==root['root_SHA256'];row['energy_Ha']=root['result']['energy_Ha']
        states.append(row);save(LOCAL/'CI-progress.json',dict(states=states,attempts=attempts))
    save(LOCAL/'CI-terminal.json',dict(status='ALL40_CI_STATUSES_RECORDED',states=states,attempts=attempts,
        actual_new_CI_roots=sum(a['status']=='COMPLETED' for a in attempts),failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),
        CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,new_model_samples=0,new_QPU_shots=0,goal_complete=False))
    print('N65_CI_TERMINAL',len(states),len(attempts),flush=True)

if __name__=='__main__':run()
