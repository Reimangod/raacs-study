"""Evaluate all available frozen draws; explicitly retain Fe's 12 missing endpoints."""
from time import process_time,perf_counter
import os,traceback
import numpy as np
from .filtered_SqDRIFT_plan import LOCAL,verified
from .filtered_SqDRIFT_readout import audit
from .paths import ROOT
from .io import load,save,write,arrays,sha256,digest,space
from .paired_lowshot import ranked_half
from .joint_ci_sparse import solve
from .fe_native_readout_diagnostic import LIBRARY


def run(version='v3',audit_function=None):
    p=verified()
    from .contiguous_CI_audit import audit as contiguous_audit
    audit_function=audit_function or contiguous_audit
    combined=load(LOCAL/'source-frame-successor-v1/combined-source-terminal.json')
    if len(combined['completed'])!=24:raise ValueError('All sources required before Fe CI')
    source_manifest=LOCAL/'source-frame-successor-v1/manifest.json'
    binding=load(source_manifest)
    for field,name in [('checker_SHA256','clifford_frame_checker_v3.py'),('batch_worker_SHA256','frame_batch_worker.py'),('source_worker_SHA256','frame_source_successor.py')]:
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=binding[field]:raise ValueError('Source successor implementation changed')
    prior=load(LOCAL/'partial-CI-terminal-v2.json')
    prior_states=[r for r in prior['states'] if r['case']!='FE2S2-40-P2']
    assert len(prior_states)==36 and all(r['status']=='COMPLETED' for r in prior_states)
    roots=LOCAL/('roots' if version=='v1' else 'roots-'+version)
    if (LOCAL/('partial-CI-started-'+version+'.json')).exists():raise RuntimeError('Existing partial reader; reconcile no rerun')
    failure=load(LOCAL/'source-failure.json');tail=load(LOCAL/'source-tail-v1/failure.json')
    assert len(failure['completed'])==18 and not tail['completed']
    save(LOCAL/('partial-CI-started-'+version+'.json'),dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
        source_successor_manifest_SHA256=sha256(source_manifest.read_bytes()),
        source_combined_terminal_SHA256=sha256((LOCAL/'source-frame-successor-v1/combined-source-terminal.json').read_bytes()),
        implementation_SHA256=sha256(__import__('pathlib').Path(__file__).read_bytes()),
        source_failure_SHA256=sha256((LOCAL/'source-failure.json').read_bytes()),
        tail_failure_SHA256=sha256((LOCAL/'source-tail-v1/failure.json').read_bytes()),
        source_unavailable_is_not_a_scientific_energy_rejection=True))
    cpu,wall=process_time(),perf_counter();cache={};states=list(prior_states);attempts=[]
    prior_CPU=load(LOCAL/'partial-CI-terminal-v1.json')['CPU_seconds']+prior['CPU_seconds']
    CPU_budget=p['CI_total_CPU_seconds']-prior_CPU
    for c in p['cases'][-1:]:
        meta=load(c['meta']);ne=(meta['nalpha'],meta['nbeta']);n=meta['norb']
        with np.load(c['integrals'],allow_pickle=False) as a:h,g=a['h1'],a['eri']
        for draw in [d for d in p['draws'] if d['case']==c['case']]:
            folder=LOCAL/'draws'/draw['draw_id']
            if not (folder/'result.json').exists():
                for cap in p['R_caps']:states.append(dict(draw_id=draw['draw_id'],case=c['case'],arm=draw['arm'],seed=draw['seed'],R_cap=cap,
                    actual_R=None,actual_D=None,basis_id=None,energy_Ha=None,status='SOURCE_UNAVAILABLE_VERIFIER_CPU_CAP_BEFORE_MODEL_DRAW'))
                continue
            s=load(folder/'result.json');assert s['counts_SHA256']==sha256((folder/'counts.npz').read_bytes()) and s['problem_id']==draw['problem_id']
            with np.load(folder/'counts.npz',allow_pickle=False) as a:rank=ranked_half(a['blocked_ids'],a['counts'],n,(1<<ne[0])-1)
            for cap in p['R_caps']:
                space();strings=sorted(rank[:cap]);key=digest([draw['problem_id'],strings,p['common_equation'],p['library_SHA256']])
                row=dict(draw_id=draw['draw_id'],case=c['case'],arm=draw['arm'],seed=draw['seed'],R_cap=cap,
                    actual_R=len(strings),actual_D=len(strings)**2,basis_id=key,energy_Ha=None)
                new=key not in cache
                if new:
                    entry=dict(status='NOT_RUN_GLOBAL_CI_CAP',result=None)
                    if process_time()-cpu<CPU_budget:
                        attempt=dict(basis_id=key,case=c['case'],status='STARTED');attempts.append(attempt);save(LOCAL/('partial-CI-attempts-'+version+'.json'),attempts);tick=process_time()
                        try:
                            pairs=np.asarray([(a,b) for a in strings for b in strings],np.uint64)
                            seconds=min(p['CI_wall_seconds_per_root'],CPU_budget-(process_time()-cpu))
                            result,v=solve(h,g,meta['core_energy'],ne,pairs,LIBRARY,seconds=seconds,triangle_cap=8000000)
                            check=audit_function(h,g,meta['core_energy'],ne,strings,v,result)
                            file=roots/(key+'.npz');blob=arrays(joint_pairs=pairs,coefficients=v)
                            if sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())+len(blob)>p['new_output_bytes']:raise OSError('Frozen output cap')
                            write(file,blob);entry=dict(status='COMPLETED',result=result,independent_RDM_audit=check,root_file=str(file),root_SHA256=sha256(blob));attempt['status']='COMPLETED'
                        except Exception as exc:
                            entry=dict(status='FAILED_PRESERVED',result=None,error=repr(exc),traceback=traceback.format_exc());attempt.update(status='FAILED_PRESERVED',error=repr(exc))
                        entry['attempt_CPU_seconds']=process_time()-tick;attempt['CPU_seconds']=entry['attempt_CPU_seconds'];save(LOCAL/('partial-CI-attempts-'+version+'.json'),attempts)
                    cache[key]=entry;save(roots/(key+'.json'),entry)
                entry=cache[key];row.update(status=entry['status'],root=entry,new_CI_attempt=new and entry['status']!='NOT_RUN_GLOBAL_CI_CAP')
                if entry['status']=='COMPLETED':
                    row['energy_Ha']=entry['result']['energy_Ha'];earlier=next((r for r in states if r['draw_id']==draw['draw_id'] and r['status']=='COMPLETED'),None)
                    if earlier and entry['result']['penalized_energy_Ha']>earlier['root']['result']['penalized_energy_Ha']+1e-8:raise ValueError('Nested objective violated')
                states.append(row);save(LOCAL/('partial-CI-progress-'+version+'.json'),dict(states=states,CPU_seconds=process_time()-cpu))
            print('N57_PARTIAL_CI',c['case'],draw['arm'],draw['seed'],len(states),flush=True)
    save(LOCAL/('partial-CI-terminal-'+version+'.json'),dict(status='ALL48_STATUSES_RECORDED_ALL24_SOURCES_MIXED_NUMERICAL_VERIFIERS',states=states,attempts=attempts,
        completed_endpoints=sum(r['status']=='COMPLETED' for r in states),source_missing_endpoints=sum(r['actual_R'] is None for r in states),
        actual_new_CI_roots=sum(a['status']=='COMPLETED' for a in attempts),
        predecessor_unique_roots_reused=18,total_unique_roots_in_combined_report=18+sum(a['status']=='COMPLETED' for a in attempts),
        original_source_MPS_protocol_declared_passed=False,failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),
        CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,new_model_samples=0,new_QPU_shots=0,goal_complete=False))


if __name__=='__main__':run()
