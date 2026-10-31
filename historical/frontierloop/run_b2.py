"""Matched30 CPU-second /100k-shot bank and objective development controls."""
import argparse
import fcntl
from io import BytesIO
from pathlib import Path
from time import process_time,perf_counter,time
import numpy as np
from qiskit import qpy
from paperstudy.ci import solve,interleaved_to_blocked
from raacsnext.run import physical,summarize_members
from raacsnext.sampler import merge_counts
from .paths import ROOT,V2
from .io import load,save,write,arrays,sha256,digest,space
from .journal import Journal
from .bank import BankEngine


def code_hashes():
    return {name:sha256((ROOT/'frontierloop'/name).read_bytes())for name in('bank.py','journal.py','run_b2.py')}


def make_engine(meta,h,g,journal,directory,budget,settings,cap):
    engine=BankEngine(meta,h,g,journal,directory,budget,cap,settings['screen_r'],settings['confirmation_r'])
    if not settings['bank']:
        # PreviousEngine D2 means Cartesian objective with child-only replace;
        # the complete-pass runner and all other budgets remain identical.
        engine.variant='D2'
    return engine


def run(case,seed,variant):
    spec=load(ROOT/'protocol-B2.json')
    if case not in spec['cases']or seed not in spec['seeds']or variant not in spec['variants']:
        raise ValueError('Unregistered B2 condition')
    m,h,g=physical(case);budget=spec['total_model_shots_per_run'];settings=spec['settings'][variant]
    binding={'cycle':'B2','case_id':case,'seed':seed,'variant':variant,'budget':budget,'problem_id':m['problem_id'],
        'code_SHA256':code_hashes(),'protocol_SHA256':sha256((ROOT/'protocol-B2.json').read_bytes()),
        'CPU_cap_seconds':spec['construction_CPU_seconds'],'settings':settings}
    identity=digest(binding);directory=ROOT/'runs/B2'/case/variant/str(seed);directory.mkdir(parents=True,exist_ok=True)
    with(directory/'owner.lock').open('a')as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=directory/'request.json'
        if path.exists()and load(path)!=binding:raise RuntimeError('Code/settings drift; no silent replacement')
        if(directory/'result.json').exists():return load(directory/'result.json')
        if(directory/'started.json').exists():raise RuntimeError('Partial run; preserve evidence before any retry')
        space();save(path,binding);save(directory/'started.json',{'unix':time(),'pid':__import__('os').getpid()})
        cpu,tick=process_time(),perf_counter();journal=Journal(m,directory,identity,budget)
        try:
            engine=make_engine(m,h,g,journal,directory,budget,settings,spec['construction_CPU_seconds'])
            output=engine.run();final=output.pop('final_members')
            observed=merge_counts([x['member']['observed']for x in final])
            ids=np.array([interleaved_to_blocked(x,m['norb'])for x in sorted(observed)],np.uint64)
            counts=np.array([observed[x]for x in sorted(observed)],np.uint64)
            payload=arrays(blocked_ids=ids,counts=counts);write(directory/'final-counts.npz',payload)
            ci=[]
            for r in spec['final_r_caps']:
                c,t=process_time(),perf_counter();value=solve(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),ids,counts,r)
                value.update(CPU_seconds=process_time()-c,wall_seconds=perf_counter()-t,
                    source_counts_SHA256=sha256(payload),new_model_shots=0)
                save(directory/('CI-r'+str(r)+'.json'),value);ci.append(value)
            for member in final:
                stream=BytesIO();qpy.dump(member['member']['circuit'],stream)
                write(directory/'final-circuits'/(member['member']['id']+'.qpy'),stream.getvalue())
            ledger=journal.ledger.snapshot(identity)
            if ledger['known_actual_shots']!=budget or ledger['unknown_or_inflight_draws']:
                raise RuntimeError('Actual paid-shot budget mismatch')
            if counts.sum()+output['pilot_model_shots']!=budget:raise RuntimeError('Pilot/final count mismatch')
            result={**binding,**output,**summarize_members(final),'run_id':identity,'status':'COMPLETED',
                'ledger':ledger,'final_model_shots':int(counts.sum()),'actual_QPU_shots':0,
                'final_CI':ci,'final_counts_SHA256':sha256(payload),'cost':journal.cost(),
                'run_body_CPU_seconds':process_time()-cpu,'run_body_wall_seconds':perf_counter()-tick,
                'members':[{'id':x['member']['id'],'actions':x['member']['actions'],'shots':x['shots'],
                    'sampling_resources':x['member']['metadata']['sampling_resources'],
                    'preparation_resources':x['member']['metadata']['preparation_resources']}for x in final],
                'independent_final_not_pilot_counts':True,'performance_claim_authorized':False}
            save(directory/'result.json',result)
            print(case,variant,seed,'E64',ci[-1]['energy_Ha'],'bank',len(final),'stop',output['construction_stop_reason'],flush=True)
            return result
        except Exception as exc:
            save(directory/'failure.json',{'error':repr(exc),'CPU_seconds':process_time()-cpu,
                'wall_seconds':perf_counter()-tick,'ledger':journal.ledger.snapshot(identity)})
            raise
        finally:journal.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--variant',required=True);a=p.parse_args();run(a.case,a.seed,a.variant)
