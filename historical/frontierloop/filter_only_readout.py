"""N62 common physical-H CI, exact-basis reuse and common-R diagnostics."""
from pathlib import Path
from time import process_time,perf_counter
import os,traceback
import numpy as np
from .filter_only_plan import LOCAL,verified
from .filtered_SqDRIFT_plan import LOCAL as OLD_LOCAL,verified as old_verified
from .paths import ROOT
from .io import load,save,write,sha256,arrays,digest,space
from .paired_lowshot import ranked_half
from .joint_ci_sparse import solve
from .contiguous_CI_audit import audit


def run():
    p=verified();old=old_verified();space()
    if (LOCAL/'CI-started.json').exists():raise RuntimeError('Existing CI run; reconcile no repeat')
    accounting=load(LOCAL/'final-accounting.json')
    if not accounting['process_children_reaped']:raise RuntimeError('Source exporter still live')
    byid={d['draw_id']:(d,LOCAL) for d in p['draws']}
    byid.update({d['draw_id']:(d,OLD_LOCAL) for d in old['draws']})
    ranks={};cache={};plans=[]
    for d,base in byid.values():
        c=next(c for c in p['cases'] if c['case']==d['case']);meta=load(c['meta']);f=base/'draws'/d['draw_id']
        if (f/'result.json').exists():
            r=load(f/'result.json');assert r['counts_SHA256']==sha256((f/'counts.npz').read_bytes()) and r['problem_id']==d['problem_id']
            with np.load(f/'counts.npz',allow_pickle=False) as a:ranks[d['draw_id']]=list(map(int,ranked_half(a['blocked_ids'],a['counts'],meta['norb'],(1<<meta['nalpha'])-1)))
        else:ranks[d['draw_id']]=None
    old_ci=load(OLD_LOCAL/'partial-CI-terminal-v3.json')
    reuse_files=[OLD_LOCAL/'partial-CI-terminal-v3.json',ROOT/'runs/N59-LARGEST-COMMON-R-DIAGNOSTIC-v1/terminal.json',ROOT/'runs/N61-FE-COMMON-R-DIAGNOSTIC-v1/terminal.json']
    for file in reuse_files:
        for r in load(file)['states']:
            if r['status']=='COMPLETED':cache[r['basis_id']]=r['root']
    for d in p['draws']:
        for cap in p['R_caps']:
            rank=ranks[d['draw_id']];strings=sorted(rank[:cap]) if rank is not None else None
            plans.append(dict(context='MAIN',case=d['case'],seed=d['seed'],arm=d['arm'],draw_id=d['draw_id'],R_cap=cap,strings=strings))
    for c in p['cases']:
        for seed in p['seeds']:
            new=next(d for d in p['draws'] if d['case']==c['case'] and d['seed']==seed)
            for comparator in ['FULL_N20','FILTERED_SHORT']:
                other=next(d for d in old['draws'] if d['case']==c['case'] and d['seed']==seed and d['arm']==comparator)
                rr=[ranks[d['draw_id']] for d in [new,other]]
                size=min(32,*(len(r) for r in rr)) if all(r is not None for r in rr) else None
                for d,rank in zip([new,other],rr):
                    plans.append(dict(context='COMMON_R',pair=comparator,case=d['case'],seed=seed,arm=d['arm'],draw_id=d['draw_id'],
                        R_cap=32,strings=sorted(rank[:size]) if size is not None else None))
    save(LOCAL/'CI-started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
        readout_SHA256=sha256(Path(__file__).read_bytes()),plans=plans,CI_CPU_cap=p['CI_total_CPU_seconds'],common_R_uses_counts_only=True,
        source_accounting_SHA256=sha256((LOCAL/'final-accounting.json').read_bytes()),old_reuse_artifacts={str(f):sha256(f.read_bytes()) for f in reuse_files}))
    cpu,wall=process_time(),perf_counter();states=[];attempts=[]
    for plan in plans:
        c=next(c for c in p['cases'] if c['case']==plan['case']);meta=load(c['meta']);strings=plan['strings'];row=dict(plan)
        row.update(status='SOURCE_MISSING',actual_R=None,actual_D=None,basis_id=None,energy_Ha=None)
        if strings is not None:
            key=digest([meta['problem_id'],strings,p['common_equation'],p['library_SHA256']]);new=key not in cache
            row.update(basis_id=key,actual_R=len(strings),actual_D=len(strings)**2)
            if new:
                tick=process_time();remaining=p['CI_total_CPU_seconds']-(tick-cpu);entry=dict(status='NOT_RUN_GLOBAL_CI_CAP')
                if remaining>0:
                    try:
                        space()
                        with np.load(c['integrals'],allow_pickle=False) as a:h,g=a['h1'],a['eri']
                        ne=(meta['nalpha'],meta['nbeta']);pairs=np.asarray([(a,b) for a in strings for b in strings],np.uint64)
                        result,v=solve(h,g,meta['core_energy'],ne,pairs,Path(p['library']),seconds=min(p['CI_wall_seconds_per_root'],remaining),triangle_cap=8000000)
                        independent=audit(h,g,meta['core_energy'],ne,strings,v,result)
                        file=LOCAL/'roots'/(key+'.npz');blob=arrays(joint_pairs=pairs,coefficients=v)
                        if sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())+len(blob)>p['new_output_bytes']:raise OSError('Readout output cap')
                        write(file,blob);entry=dict(status='COMPLETED',result=result,independent_RDM_audit=independent,root_file=str(file),root_SHA256=sha256(blob))
                    except Exception as exc:entry=dict(status='FAILED_PRESERVED',error=repr(exc),traceback=traceback.format_exc())
                entry['attempt_CPU_seconds']=process_time()-tick;cache[key]=entry
                attempts.append(dict(basis_id=key,status=entry['status'],CPU_seconds=entry['attempt_CPU_seconds']));save(LOCAL/'roots'/(key+'.json'),entry)
            entry=cache[key];row.update(status=entry['status'],root=entry,new_CI_attempt=new)
            if entry['status']=='COMPLETED':
                if sha256(Path(entry['root_file']).read_bytes())!=entry['root_SHA256']:raise ValueError('Cached evaluation changed')
                row['energy_Ha']=entry['result']['energy_Ha']
        states.append(row);save(LOCAL/'CI-progress.json',dict(states=states,attempts=attempts,CPU_seconds=process_time()-cpu))
    save(LOCAL/'CI-terminal.json',dict(status='ALL72_NEW_ARM_AND_PAIRWISE_COMMON_R_STATUSES_RECORDED',states=states,attempts=attempts,
        main_planned=24,common_R_planned=48,actual_new_CI_roots=sum(a['status']=='COMPLETED' for a in attempts),
        failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),
        CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,new_model_samples=0,new_QPU_shots=0,goal_complete=False))
    print('N62_CI_TERMINAL',len(states),len(attempts),flush=True)


if __name__=='__main__':run()
