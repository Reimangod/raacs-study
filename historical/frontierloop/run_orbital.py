import argparse
import fcntl
import inspect
from importlib.metadata import version
from pathlib import Path
from time import time,process_time,perf_counter
import ffsim
import numpy as np
from pyscf.fci import cistring
from paperstudy.ci import common_strings
from raacsnext.run import physical
from .paths import ROOT,V2
from .io import load,save,write,arrays,digest,sha256,space
from .orbital import optimize


def source(case,method,meta):
    if method=='RAACS_B3':path=ROOT/'runs/B3'/case/'73001'
    elif method=='LUCJ_COMPRESSED_L1':
        path=ROOT/'Fe40-C1-v1/71001'if case=='FE2S2-40-P2'else V2/'runs/DEVELOPMENT'/case/'LUCJ_COMPRESSED_L1/100000-51001'
    else:
        n=meta['norb'];ne=(meta['nalpha'],meta['nbeta']);rng=np.random.default_rng(75001)
        strings=cistring.make_strings(range(n),ne[0])
        alpha=rng.choice(strings,100000);beta=rng.choice(strings,100000)
        values=alpha.astype(np.uint64)|(beta.astype(np.uint64)<<n)
        ids,counts=np.unique(values,return_counts=True)
        return ids,counts,{'kind':'CLASSICAL_UNIFORM_SPIN_SECTOR','classical_draws':100000,'new_quantum_model_shots':0,
            'full_Fock_enumeration':False,'spin_string_count':len(strings),'sampler_CNOT':0,'not_a_strong_SCI_baseline':True}
    record=load(path/'result.json')
    if record['problem_id']!=meta['problem_id']:raise RuntimeError('OO source problem mismatch')
    with np.load(path/'final-counts.npz',allow_pickle=False)as a:ids,counts=a['blocked_ids'],a['counts']
    return ids,counts,{'kind':'REUSE_EXISTING_FIXED_MEASUREMENT_RECORD','source_result':str(path/'result.json'),
        'source_result_SHA256':sha256((path/'result.json').read_bytes()),'source_counts_SHA256':sha256((path/'final-counts.npz').read_bytes()),
        'source_total_budget':record['budget'],'source_final_model_shots':int(counts.sum()),
        'source_energy_r64':record['final_CI'][-1]['energy_Ha'],'new_quantum_model_shots':0,
        'sampler_resources':record['sampling_resources'],
        'sampler_coefficient_slots':record.get('coefficient_slots_shot_weighted_mean',record.get('coefficient_slots'))}


def code_hashes():
    libfile=Path(inspect.getfile(ffsim.optimize_orbitals))
    return {'orbital.py':sha256((ROOT/'frontierloop/orbital.py').read_bytes()),
        'run_orbital.py':sha256(Path(__file__).read_bytes()),'ffsim_optimizer_source':sha256(libfile.read_bytes())}


def run(case,method):
    spec=load(ROOT/'protocol-C2-OO.json')
    if case not in spec['cases']or method not in spec['methods']:raise ValueError('Unregistered OO cell')
    directory=ROOT/'runs/C2-OO'/case/method;directory.mkdir(parents=True,exist_ok=True)
    with(directory/'owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m,h,g=physical(case);ids,counts,provenance=source(case,method,m)
        strings,info=common_strings(ids,counts,m['norb'],(m['nalpha'],m['nbeta']),64)
        binding={'case_id':case,'method':method,'cycle':'C2-OO','physical_problem_id':m['problem_id'],
            'source':provenance,'code_SHA256':code_hashes(),'protocol_SHA256':sha256((ROOT/'protocol-C2-OO.json').read_bytes()),
            'runtime':{p:version(p)for p in('numpy','scipy','pyscf','ffsim','jax')},
            'CI_strings':strings.tolist(),'original_counts_context':info}
        if(directory/'request.json').exists()and load(directory/'request.json')!=binding:raise RuntimeError('OO source/code drift')
        if(directory/'result.json').exists():return load(directory/'result.json')
        if(directory/'started.json').exists():raise RuntimeError('Partial OO work must be preserved/reconciled')
        space();save(directory/'request.json',binding)
        save(directory/'started.json',{'unix':time(),'pid':__import__('os').getpid()})
        cpu,tick=process_time(),perf_counter()
        def checkpoint(history,rotations):
            save(directory/'checkpoint.json',{'history':history,'new_quantum_model_shots':0})
            write(directory/'rotations-checkpoint.npz',arrays(rotations=np.array(rotations)))
        try:
            result=optimize(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),strings,spec,checkpoint)
            if provenance.get('source_energy_r64')is not None and abs(result['initial_energy_Ha']-provenance['source_energy_r64'])>1e-8:
                raise RuntimeError('Initial OO solve does not match saved readout')
            rotations=result.pop('rotations');hh=result.pop('final_h1');gg=result.pop('final_eri')
            payload=arrays(rotations=np.array(rotations),h1=hh,eri=gg,CI_strings=strings)
            write(directory/'optimized-representation.npz',payload)
            result.update(**binding,status='COMPLETED',representation_SHA256=sha256(payload),
                whole_body_CPU_seconds=process_time()-cpu,whole_body_wall_seconds=perf_counter()-tick,
                final_state_is_classical_CI_with_variational_orbitals_not_original_sampler_state=True,
                physical_sampling_in_new_basis_performed=False,world_best_claim_authorized=False)
            save(directory/'result.json',result)
            print('C2',case,method,'before',result['initial_energy_Ha'],'after',result['final_energy_Ha'],flush=True)
            return result
        except Exception as exc:
            save(directory/'failure.json',{'error':repr(exc),'CPU_seconds':process_time()-cpu,
                'wall_seconds':perf_counter()-tick,'new_quantum_model_shots':0});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--method',required=True)
    a=p.parse_args();run(a.case,a.method)
