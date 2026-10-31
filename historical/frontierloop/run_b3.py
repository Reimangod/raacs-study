import argparse
import fcntl
from io import BytesIO
from time import process_time,perf_counter,time
import numpy as np
from qiskit import qpy
from paperstudy.ci import solve,interleaved_to_blocked
from raacsnext.run import physical,summarize_members
from raacsnext.sampler import merge_counts
from .paths import ROOT
from .io import load,save,write,arrays,sha256,digest,space
from .journal import Journal
from .beam import BeamEngine


def code_hashes():
    return {name:sha256((ROOT/'frontierloop'/name).read_bytes())for name in('bank.py','beam.py','journal.py','run_b3.py')}


def run(case,seed):
    spec=load(ROOT/'protocol-B3.json')
    if case not in spec['cases']or seed not in spec['seeds']:raise ValueError('Unregistered B3 cell')
    m,h,g=physical(case);budget=spec['total_model_shots_per_run']
    request={'cycle':'B3','variant':'RESIDUAL_BEAM','case_id':case,'problem_id':m['problem_id'],'seed':seed,'budget':budget,
        'protocol_SHA256':sha256((ROOT/'protocol-B3.json').read_bytes()),'code_SHA256':code_hashes()}
    identity=digest(request);directory=ROOT/'runs/B3'/case/str(seed);directory.mkdir(parents=True,exist_ok=True)
    with(directory/'owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(directory/'request.json').exists()and load(directory/'request.json')!=request:raise RuntimeError('B3 code/request drift')
        if(directory/'result.json').exists():return load(directory/'result.json')
        if(directory/'started.json').exists():raise RuntimeError('Partial B3 evidence; inspect before any restart')
        space();save(directory/'request.json',request);save(directory/'started.json',{'pid':__import__('os').getpid(),'unix':time()})
        cpu,tick=process_time(),perf_counter();j=Journal(m,directory,identity,budget)
        try:
            output=BeamEngine(m,h,g,j,directory,budget,spec['construction_CPU_seconds']).run()
            final=output.pop('final_members');observed=merge_counts([x['member']['observed']for x in final])
            ids=np.array([interleaved_to_blocked(x,m['norb'])for x in sorted(observed)],np.uint64)
            counts=np.array([observed[x]for x in sorted(observed)],np.uint64)
            payload=arrays(blocked_ids=ids,counts=counts);write(directory/'final-counts.npz',payload)
            cis=[]
            for r in spec['final_r_caps']:
                c,t=process_time(),perf_counter();value=solve(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),ids,counts,r)
                value.update(CPU_seconds=process_time()-c,wall_seconds=perf_counter()-t,new_model_shots=0,
                    source_counts_SHA256=sha256(payload));save(directory/('CI-r'+str(r)+'.json'),value);cis.append(value)
            ledger=j.ledger.snapshot(identity)
            if ledger['known_actual_shots']!=budget or ledger['unknown_or_inflight_draws']or counts.sum()+output['pilot_model_shots']!=budget:
                raise RuntimeError('B3 physical shot accounting mismatch')
            members=[]
            for x in final:
                stream=BytesIO();qpy.dump(x['member']['circuit'],stream)
                write(directory/'final-circuits'/(x['member']['id']+'.qpy'),stream.getvalue())
                members.append({'id':x['member']['id'],'actions':x['member']['actions'],'shots':x['shots'],
                    'sampling_resources':x['member']['metadata']['sampling_resources'],
                    'preparation_resources':x['member']['metadata']['preparation_resources']})
            result={**request,**output,**summarize_members(final),'run_id':identity,'status':'COMPLETED',
                'members':members,'ledger':ledger,'final_CI':cis,'final_model_shots':int(counts.sum()),
                'final_counts_SHA256':sha256(payload),'cost':j.cost(),'actual_QPU_shots':0,
                'run_body_CPU_seconds':process_time()-cpu,'run_body_wall_seconds':perf_counter()-tick,
                'independent_final_not_pilot_counts':True,'performance_claim_authorized':False}
            save(directory/'result.json',result)
            print(case,'B3',seed,'E64',cis[-1]['energy_Ha'],'depth',output['completed_depth'],'bank',len(final),'stop',output['construction_stop_reason'],flush=True)
            return result
        except Exception as exc:
            save(directory/'failure.json',{'error':repr(exc),'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,'ledger':j.ledger.snapshot(identity)})
            raise
        finally:j.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--seed',required=True,type=int)
    a=p.parse_args();run(a.case,a.seed)
