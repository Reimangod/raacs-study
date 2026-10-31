import argparse
import fcntl
import os
from pathlib import Path
from time import time,process_time,perf_counter
import xml.etree.ElementTree as ET
import numpy as np
from paperstudy.ci import common_strings
from raacsnext.run import physical
from .paths import ROOT
from .io import load,save,write,arrays,sha256,space
from .classical_cisd import solve
from .cisd_sparse import marginal_strings
from .fast_orbital import optimize as fast
from .orbital import optimize as unpenalized
from .spin_orbital import optimize as penalized
from .run_singlet_orbital import freeze as readout_dependencies


def freeze():
    paths=[ROOT/'frontierloop'/n for n in ('classical_cisd.py','cisd_sparse.py','fast_orbital.py',
        'hole_rdm.py','energy_cache.py','run_classical_control.py')]
    paths += [ROOT/'protocol-C6-CLASSICAL-CISD.json',ROOT/'test_classical_cisd.py',ROOT/'test_cisd_sparse.py',
        ROOT/'test_fast_orbital.py',ROOT/'qualification-fast-orbital-krylov-v2-tests.xml']
    paths += [ROOT/('protocol-'+mode+'.json') for mode in ('C2-OO','C3-SPIN-OO','C4-SINGLET-OO')]
    return {'files':{str(p):sha256(p.read_bytes()) for p in paths},'readout_dependencies':readout_dependencies(),
        'new_quantum_model_shots':0,'goal_complete':False}


def guard():
    if load(ROOT/'freeze-C6-CLASSICAL-CISD.json')!=freeze():
        raise RuntimeError('C6 code/protocol/qualification drift')
    return load(ROOT/'protocol-C6-CLASSICAL-CISD.json')


def prepare(case):
    spec=guard()
    if case not in spec['cases']:
        raise ValueError('Unregistered physical case')
    directory=ROOT/'runs/C6-CLASSICAL-CISD'/case/'source';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (directory/'result.json').exists():return load(directory/'result.json')
        if (directory/'started.json').exists():raise RuntimeError('Partial C6 source preserved, no blind retry')
        m,h,g=physical(case);space();save(directory/'started.json',{'pid':os.getpid(),'unix':time()})
        save(directory/'request.json',{'case':case,'problem_id':m['problem_id'],'freeze':freeze()})
        cpu,tick=process_time(),perf_counter()
        try:
            result=solve(m,h,g,spec['source_CPU_seconds'],spec['source_wall_seconds'])
            ids,coeff=result.pop('ids'),result.pop('coefficients')
            payload=arrays(blocked_ids=ids,coefficients=coeff)
            write(directory/'sparse-CISD-state.npz',payload)
            weights=abs(coeff)**2;weights/=weights.sum()
            sampled=np.random.default_rng(spec['classical_draw_seed']).choice(ids,size=spec['classical_draws'],p=weights)
            observed,counts=np.unique(sampled,return_counts=True)
            draws=arrays(blocked_ids=observed,counts=counts)
            write(directory/'classical-counts.npz',draws)
            result.update(status='COMPLETED',case=case,physical_problem_id=m['problem_id'],freeze=freeze(),
                sparse_state_SHA256=sha256(payload),classical_counts_SHA256=sha256(draws),
                classical_draws=spec['classical_draws'],classical_draw_seed=spec['classical_draw_seed'],
                sparse_nonzero_configurations=int(np.count_nonzero(coeff)),CISD_coefficient_slots=len(coeff),
                whole_body_CPU_seconds=process_time()-cpu,whole_body_wall_seconds=perf_counter()-tick,
                not_a_quantum_state_preparation=True)
            save(directory/'result.json',result)
            print('C6 source',case,'CISD_E',result['energy_Ha'],'CPU',result['CPU_seconds'],flush=True)
            return result
        except Exception as exc:
            save(directory/'failure.json',{'error':repr(exc),'status':'SOURCE_FAILED_PRESERVED',
                'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,'new_quantum_model_shots':0})
            raise


def run(case,variant,mode):
    spec=guard()
    if case not in spec['cases'] or variant not in spec['variants'] or mode not in spec['readouts']:
        raise ValueError('Unregistered C6 cell')
    source_dir=ROOT/'runs/C6-CLASSICAL-CISD'/case/'source';source=load(source_dir/'result.json')
    m,h,g=physical(case)
    if source['physical_problem_id']!=m['problem_id'] or source['freeze']!=freeze():
        raise RuntimeError('C6 source binding failed')
    directory=ROOT/'runs/C6-CLASSICAL-CISD'/case/variant/mode;directory.mkdir(parents=True,exist_ok=True)
    with (directory/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (directory/'result.json').exists():return load(directory/'result.json')
        if (directory/'started.json').exists():raise RuntimeError('Partial C6 OO preserved, no blind retry')
        if variant=='DETERMINISTIC_CISD_MARGINAL':
            payload=source_dir/'sparse-CISD-state.npz'
            assert sha256(payload.read_bytes())==source['sparse_state_SHA256']
            with np.load(payload,allow_pickle=False) as a:
                strings=marginal_strings(a['blocked_ids'],a['coefficients'],m['norb'],m['nalpha'],spec['r_cap'])
            classical_draws=0
        else:
            payload=source_dir/'classical-counts.npz'
            assert sha256(payload.read_bytes())==source['classical_counts_SHA256']
            with np.load(payload,allow_pickle=False) as a:
                strings,_=common_strings(a['blocked_ids'],a['counts'],m['norb'],(m['nalpha'],m['nbeta']),spec['r_cap'])
            classical_draws=spec['classical_draws']
        request={'case_id':case,'method':variant,'readout':mode,'physical_problem_id':m['problem_id'],
            'source_result_SHA256':sha256((source_dir/'result.json').read_bytes()),'freeze':freeze(),
            'CI_strings':strings.tolist(),'classical_draws':classical_draws,'r_cap':spec['r_cap'],
            'quantum_sampler_used':False,'quantum_sampler_resources':None,
            'new_quantum_model_shots':0,'actual_QPU_shots':0}
        space();save(directory/'request.json',request)
        save(directory/'started.json',{'pid':os.getpid(),'unix':time()})
        cpu,tick=process_time(),perf_counter()
        def checkpoint(history,rotations):
            save(directory/'checkpoint.json',{'history':history,'new_quantum_model_shots':0})
            write(directory/'rotations-checkpoint.npz',arrays(rotations=np.array(rotations)))
        try:
            ospec=load(ROOT/('protocol-'+mode+'.json'))
            if mode=='C4-SINGLET-OO':
                result=fast(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),strings,ospec,mode,checkpoint)
            else:
                fn=unpenalized if mode=='C2-OO' else penalized
                result=fn(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),strings,ospec,checkpoint)
            rotations,hh,gg=(result.pop(k) for k in ('rotations','final_h1','final_eri'))
            payload=arrays(rotations=np.array(rotations),h1=hh,eri=gg,CI_strings=strings)
            write(directory/'optimized-representation.npz',payload)
            result.update(**request,status='COMPLETED',representation_SHA256=sha256(payload),
                whole_body_CPU_seconds=process_time()-cpu,whole_body_wall_seconds=perf_counter()-tick,
                correlated_CI_state_not_claimed_prepared_with_zero_gates=True,goal_complete=False)
            save(directory/'result.json',result)
            print('C6',case,variant,mode,'E',result['final_energy_Ha'],'CPU',result['CPU_seconds'],flush=True)
            return result
        except Exception as exc:
            save(directory/'failure.json',{'error':repr(exc),'status':'FAILED_ENGINEERING_PRESERVED',
                'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,'new_quantum_model_shots':0})
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--prepare',action='store_true')
    p.add_argument('--case');p.add_argument('--variant');p.add_argument('--readout');a=p.parse_args()
    if a.freeze:
        suites=ET.parse(ROOT/'qualification-fast-orbital-krylov-v2-tests.xml').getroot().iter('testsuite')
        values=list(suites)
        if sum(int(s.get('tests','0')) for s in values)!=6 or any(int(s.get(k,'0')) for s in values for k in ('failures','errors')):
            raise RuntimeError('C6 fast S0 backend not qualified')
        path=ROOT/'freeze-C6-CLASSICAL-CISD.json';value=freeze()
        if path.exists() and load(path)!=value:raise RuntimeError('Never replace C6 freeze')
        save(path,value)
    elif a.prepare:prepare(a.case)
    else:run(a.case,a.variant,a.readout)
