"""N57 same-equation R32/R128 CI, exact-basis dedup and independent RDM audit."""
from pathlib import Path
from time import process_time,perf_counter
import os,sys,traceback
import numpy as np
from pyscf.fci import selected_ci
from .filtered_SqDRIFT_plan import LOCAL,verified
from .paths import ROOT
from .io import load,save,write,arrays,sha256,digest,space
from .paired_lowshot import ranked_half
from .joint_ci_sparse import solve
from .portable_penalized_ci import expectation
from .fe_native_readout_diagnostic import LIBRARY


def audit(h,g,core,ne,strings,v,result):
    cpu=process_time();s=np.asarray(strings,np.int64)
    ci=selected_ci._as_SCIvector(v.reshape(len(s),len(s)),(s,s));solver=selected_ci.SelectedCI();solver.verbose=0
    one,two=solver.make_rdm12(ci,len(h),ne);energy=expectation(h,g,core,one,two);spin=float(solver.spin_square(ci,len(h),ne)[0])
    number=sum(ne);errors=dict(RDM_energy_Ha=abs(energy-result['energy_Ha']),S2=abs(spin-result['spin_square']),
        particle_number=abs(float(np.trace(one))-number),pair_number=abs(float(np.einsum('pprr->',two))-number*(number-1)))
    if max(errors.values())>1e-7:raise ValueError('Independent per-root physical RDM/sector mismatch')
    return dict(errors=errors,CPU_seconds=process_time()-cpu)


def run():
    p=verified();source=load(LOCAL/'source-terminal.json')
    if source['status']!='ALL24_FRESH_SOURCE_DRAWS_COMPLETED':raise ValueError('All frozen source draws required')
    if (LOCAL/'CI-started.json').exists():raise RuntimeError('Existing readout; reconcile not repeat')
    save(LOCAL/'CI-started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),library_SHA256=sha256(LIBRARY.read_bytes())))
    cpu,wall=process_time(),perf_counter();cache={};states=[];attempts=[]
    for c in p['cases']:
        meta=load(c['meta']);ne=(meta['nalpha'],meta['nbeta']);n=meta['norb']
        with np.load(c['integrals'],allow_pickle=False) as a:h,g=a['h1'],a['eri']
        for draw in [d for d in p['draws'] if d['case']==c['case']]:
            folder=LOCAL/'draws'/draw['draw_id'];s=load(folder/'result.json')
            if s['draw_id']!=draw['draw_id'] or s['problem_id']!=draw['problem_id'] or s['counts_SHA256']!=sha256((folder/'counts.npz').read_bytes()):raise ValueError('Source/count identity differs')
            with np.load(folder/'counts.npz',allow_pickle=False) as a:rank=ranked_half(a['blocked_ids'],a['counts'],n,(1<<ne[0])-1)
            for cap in p['R_caps']:
                space();strings=sorted(rank[:cap]);key=digest([draw['problem_id'],strings,p['common_equation'],sha256(LIBRARY.read_bytes())])
                row=dict(draw_id=draw['draw_id'],case=c['case'],arm=draw['arm'],seed=draw['seed'],R_cap=cap,
                    actual_R=len(strings),actual_D=len(strings)**2,basis_id=key,energy_Ha=None)
                new=key not in cache
                if new:
                    entry=dict(status='NOT_RUN_GLOBAL_CI_CAP',result=None)
                    if process_time()-cpu<p['CI_total_CPU_seconds']:
                        attempt=dict(basis_id=key,case=c['case'],status='STARTED');attempts.append(attempt);save(LOCAL/'CI-attempts.json',attempts);tick=process_time()
                        try:
                            pairs=np.asarray([(a,b) for a in strings for b in strings],np.uint64)
                            seconds=min(p['CI_wall_seconds_per_root'],p['CI_total_CPU_seconds']-(process_time()-cpu))
                            result,v=solve(h,g,meta['core_energy'],ne,pairs,LIBRARY,seconds=seconds,triangle_cap=8000000)
                            independent=audit(h,g,meta['core_energy'],ne,strings,v,result)
                            file=LOCAL/'roots'/(key+'.npz');blob=arrays(joint_pairs=pairs,coefficients=v)
                            if sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())+len(blob)>p['new_output_bytes']:raise OSError('N57 output cap before root save')
                            if sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())+len(blob)>p['research_root_bytes']:raise OSError('Research root cap before root save')
                            write(file,blob)
                            entry=dict(status='COMPLETED',result=result,independent_RDM_audit=independent,root_file=str(file),root_SHA256=sha256(blob))
                            attempt.update(status='COMPLETED')
                        except Exception as exc:
                            entry=dict(status='FAILED_PRESERVED',result=None,error=repr(exc),traceback=traceback.format_exc());attempt.update(status='FAILED_PRESERVED',error=repr(exc))
                        entry['attempt_CPU_seconds']=process_time()-tick;attempt['CPU_seconds']=entry['attempt_CPU_seconds'];save(LOCAL/'CI-attempts.json',attempts)
                    cache[key]=entry;save(LOCAL/'roots'/(key+'.json'),entry)
                entry=cache[key];row.update(status=entry['status'],root=entry,new_CI_attempt=new and entry['status']!='NOT_RUN_GLOBAL_CI_CAP')
                if entry['status']=='COMPLETED':
                    row['energy_Ha']=entry['result']['energy_Ha']
                    earlier=next((x for x in states if x['draw_id']==draw['draw_id'] and x['status']=='COMPLETED'),None)
                    if earlier and entry['result']['penalized_energy_Ha']>earlier['root']['result']['penalized_energy_Ha']+1e-8:
                        raise ValueError('Nested R128 penalized objective violates feasible R32 witness')
                states.append(row);save(LOCAL/'CI-progress.json',dict(states=states,attempts=attempts,CPU_seconds=process_time()-cpu))
            print('N57_CI',c['case'],draw['arm'],draw['seed'],len(states),flush=True)
    save(LOCAL/'CI-terminal.json',dict(status='ALL48_FROZEN_READOUT_ENDPOINTS_TERMINAL',states=states,attempts=attempts,
        completed_endpoints=sum(x['status']=='COMPLETED' for x in states),actual_new_CI_roots=sum(a['status']=='COMPLETED' for a in attempts),
        failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        new_model_samples=0,new_QPU_shots=0,goal_complete=False))


if __name__=='__main__':run()
