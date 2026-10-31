"""Isolated molecular runner. Evaluation references are not selection inputs."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import sqlite3
from time import perf_counter, process_time, time
from unittest.mock import patch

import numpy as np
from benchmark import resource_adaptive_sampler as native
from benchmark.local_sampler_primitives import LocalQE, unpack_counts64
from benchmark.circuit_sampling_runtime import SamplingSpec
from paperstudy.io import assert_safe_physical_input
from paperstudy.ci import solve, interleaved_to_blocked
from paperstudy.sparse_raacs import traced_append
from .paths import ROOT,OLD
from .io import save,load,write_bytes,array_bytes,digest,sha256,space_check
from .sampler import Journal,provider,merge_counts,circuit_from_actions
from .policy import Engine,normalized_resources


def physical(case):
    folder=OLD/'data/physical'/case
    meta=load(folder/'problem.json')
    assert_safe_physical_input(meta)
    with np.load(folder/'integrals.npz',allow_pickle=False)as a:
        h,g=a['h1'].copy(),a['eri'].copy()
    if {k:sha256(v.astype('<f8').tobytes())for k,v in(('h1',h),('eri',g))} != meta['physical_arrays_sha256']:
        raise RuntimeError('Physical integral bytes changed')
    return meta,h,g


def science_hashes():
    names=('policy.py','sampler.py','run.py')
    return {name:sha256((ROOT/'raacsnext'/name).read_bytes())for name in names}


def native_d0(meta,journal,directory,budget,seed,physical_folder):
    cpu=process_time();phase={}
    phase_path=directory/'native-construction-cost.json'
    original=LocalQE.append
    allocate=native.allocation_counts
    def timed_allocation(*args,**kwargs):
        phase['construction_CPU_seconds']=process_time()-cpu
        save(phase_path,phase)
        return allocate(*args,**kwargs)
    with patch.object(native,'CircuitCountSampler',lambda spec:journal),patch.object(
        LocalQE,'append',lambda self,parent,theta:traced_append(original,self,parent,theta)),patch.object(
        native,'allocation_counts',timed_allocation):
        result=native.run_sampler(physical_folder,directory/'native',native.RAACSSpec(
            total_model_shots=budget,max_steps=32,candidates_measured_per_step=4,CI_cap=64,seed=seed),
            SamplingSpec(backend='exact_sparse_QE',cache_entries=2))
    if not phase:
        # A completed native task can be recovered after a later reporting
        # failure. Its original construction cost must not become a cache hit.
        if not phase_path.exists():raise RuntimeError('Original D0 construction timing missing')
        phase=load(phase_path)
    with sqlite3.connect(directory/'native/events.sqlite')as db:
        rows=db.execute("SELECT counts64,metadata FROM draws WHERE role='independent_final'").fetchall()
    if len(rows)!=1:raise RuntimeError('D0 must preserve native last_only')
    ids,counts=unpack_counts64(rows[0][0]);metadata=json.loads(rows[0][1])
    actions=result['accepted_operators']
    member=normalized_resources({'id':digest(['physical-QE-actions',meta['hf_id'],meta['n_qubits'],actions]),
        'actions':actions,'observed':{int(x):int(n)for x,n in zip(ids,counts)},'metadata':metadata})
    save(directory/'native-summary.json',result)
    return {'final_members':[{'member':member,'shots':int(counts.sum())}],
        'pilot_model_shots':result['pilot_raw_model_shots'],
        'construction_stop_reason':result['construction_stop_reason'],
        'construction_CPU_seconds':phase['construction_CPU_seconds'],
        'accepted_decisions':result['accepted_steps'],'named_H_requests':result['classical_named_H_requests'],
        'bank_size':1,'decision_count':result['accepted_steps'],
        'reference_information_used':False,'molecule_label_used':False}


def summarize_members(final):
    total=sum(m['shots']for m in final)
    answer={}
    for kind in('sampling_resources','preparation_resources'):
        answer[kind]={k:sum(x['shots']*x['member']['metadata'][kind][k]for x in final)/total
            for k in('cnot_count','cnot_depth','total_depth')}
        answer[kind]['maximum']={k:max(x['member']['metadata'][kind][k]for x in final)
            for k in('cnot_count','cnot_depth','total_depth')}
    answer['coefficient_slots_shot_weighted_mean']=sum(x['shots']*len(x['member']['actions'])for x in final)/total
    answer['coefficient_slots_distinct_member_sum']=sum(len(x['member']['actions'])for x in final)
    answer['nonredundant_independent_parameters']=None
    answer['VQE_optimized_variables']=0
    return answer


def run(case,variant,budget,seed,stage='DEVELOPMENT',CPU_cap=None):
    protocol=load(ROOT/'protocol.json')
    if case not in protocol['cases'] or variant not in protocol['variants']:
        raise ValueError('Unregistered case/variant')
    meta,h,g=physical(case)
    hashes=science_hashes()
    config={'schema':'raacs-practical-run-2','stage':stage,'problem_id':meta['problem_id'],
        'case_id':case,'variant':variant,'budget':int(budget),'seed':int(seed),
        'protocol_SHA256':sha256((ROOT/'protocol.json').read_bytes()),'code_SHA256':hashes,
        'construction_CPU_cap':CPU_cap,'physical_input_path':str(OLD/'data/physical'/case)}
    identity=digest(config)
    directory=ROOT/'runs'/stage/case/variant/(str(budget)+'-'+str(seed))
    directory.mkdir(parents=True,exist_ok=True)
    with(directory/'owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        request=directory/'request.json'
        if request.exists()and load(request)!=config:
            raise RuntimeError('Run code/settings changed; preserve prior evidence, do not silently rerun')
        if(directory/'result.json').exists():return load(directory/'result.json')
        if stage=='DEVELOPMENT'and time()>load(ROOT/'campaign-start.json')['pilot_deadline_unix']:
            raise TimeoutError('Registered 24-hour development deadline')
        if variant!='D0'and CPU_cap is None:raise ValueError('D0-derived construction cap required')
        space_check(2*1024**2)
        save(request,config)
        attempts=len(list((directory/'attempts').glob('*/started.json')))+1
        attempt=directory/'attempts'/str(attempts)
        tick,cpu=perf_counter(),process_time()
        save(attempt/'started.json',{'pid':__import__('os').getpid(),'started_unix':time(),'run_id':identity})
        journal=Journal(provider(meta),directory,ROOT/'shots.sqlite',identity,budget)
        try:
            if variant=='D0':
                output=native_d0(meta,journal,directory,budget,seed,OLD/'data/physical'/case)
            else:
                output=Engine(meta,h,g,journal,directory,variant,budget,CPU_cap).run()
            final=output.pop('final_members')
            observed=merge_counts([x['member']['observed']for x in final])
            ids=np.array([interleaved_to_blocked(x,meta['norb'])for x in sorted(observed)],np.uint64)
            counts=np.array([observed[x]for x in sorted(observed)],np.uint64)
            payload=array_bytes(blocked_ids=ids,counts=counts)
            write_bytes(directory/'final-counts.npz',payload)
            assert int(counts.sum())+output['pilot_model_shots']==budget
            ci=[]
            for r in(8,16,32,64):
                target=directory/('CI-r'+str(r)+'.json')
                if target.exists():c=load(target)
                else:
                    cc,tt=process_time(),perf_counter()
                    c=solve(h,g,meta['core_energy'],(meta['nalpha'],meta['nbeta']),ids,counts,r)
                    c.update(CPU_seconds=process_time()-cc,wall_seconds=perf_counter()-tt,
                        problem_id=meta['problem_id'],source_counts_SHA256=sha256(payload),new_model_shots=0)
                    save(target,c)
                if c['source_counts_SHA256']!=sha256(payload):raise RuntimeError('CI cache counts mismatch')
                ci.append(c)
            snapshot=journal.ledger.snapshot(identity)
            if snapshot['known_actual_shots']!=budget or snapshot['unknown_or_inflight_draws']:
                raise RuntimeError('Draw ledger does not reconcile')
            members=[{'circuit_id':x['member']['id'],'actions':x['member']['actions'],'shots':x['shots'],
                'preparation_resources':x['member']['metadata']['preparation_resources'],
                'sampling_resources':x['member']['metadata']['sampling_resources']}for x in final]
            record={**config,**output,**summarize_members(final),'run_id':identity,'status':'COMPLETED',
                'members':members,'final_model_shots':int(counts.sum()),'final_counts_SHA256':sha256(payload),
                'final_CI':ci,'ledger':snapshot,'actual_QPU_shots':0,
                'attempt_CPU_seconds':process_time()-cpu,'attempt_wall_seconds':perf_counter()-tick,
                'new_draws_this_attempt':journal.new_draws,'replayed_draws_this_attempt':journal.replayed_draws,
                'all_model_CNOT_applications':sum(x['actual_shots']*x['metadata']['sampling_resources']['cnot_count']for x in journal.records),
                'mixed_sampling_not_a_prepared_CI_state':True,'final_is_independent_of_pilot':True,
                'all_attempt_work_retained':True,'performance_claim_authorized':False}
            save(attempt/'completed.json',{'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick})
            save(directory/'result.json',record)
            print(case,variant,budget,seed,record['construction_stop_reason'],
                'E64',ci[-1]['energy_Ha'],'bank',len(final),'pilot',record['pilot_model_shots'],flush=True)
            return record
        except Exception as exc:
            save(attempt/'failed.json',{'error':repr(exc),'CPU_seconds':process_time()-cpu,
                'wall_seconds':perf_counter()-tick,'ledger':journal.ledger.snapshot(identity)})
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--variant',required=True)
    p.add_argument('--budget',type=int,default=100000);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--CPU-cap',type=float,default=None)
    a=p.parse_args();run(a.case,a.variant,a.budget,a.seed,CPU_cap=a.CPU_cap)
