import argparse
import fcntl
import inspect
import os
from pathlib import Path
from time import time,process_time,perf_counter
import ffsim
import numpy as np
from pyscf.fci import selected_ci
from paperstudy.ci import common_strings
from raacsnext.run import physical
from .paths import ROOT
from .io import load,save,write,arrays,sha256,space
from .run_orbital import source
from .run_spin_orbital import freeze as prior_freeze
from .singlet import optimize


def freeze():
    paths = [ROOT/'protocol-C4-SINGLET-OO.json',ROOT/'frontierloop/singlet.py',
        Path(__file__),ROOT/'test_singlet.py']
    return {'files':{str(p):sha256(p.read_bytes()) for p in paths},
        'C3_frozen_dependencies':prior_freeze(), 'new_quantum_model_shots':0}


def run(case,method):
    spec = load(ROOT/'protocol-C4-SINGLET-OO.json')
    if case not in spec['cases'] or method not in spec['methods']:
        raise ValueError('Unregistered C4 cell')
    frozen = freeze()
    if load(ROOT/'freeze-C4-SINGLET-OO.json')!=frozen:
        raise RuntimeError('C4 frozen source drift')
    directory = ROOT/'runs/C4-SINGLET-OO'/case/method
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m,h,g = physical(case)
        ids,counts,provenance = source(case,method,m)
        strings,info = common_strings(ids,counts,m['norb'],(m['nalpha'],m['nbeta']),64)
        previous = load(ROOT/'runs/C3-SPIN-OO'/case/method/'request.json')
        if previous['source']!=provenance or previous['CI_strings']!=strings.tolist():
            raise RuntimeError('C4 changed paired inputs')
        binding = {'case_id':case,'method':method,'cycle':'C4-SINGLET-OO',
            'physical_problem_id':m['problem_id'],'source':provenance,'code_freeze':frozen,
            'CI_strings':strings.tolist(),'original_counts_context':info,
            'C3_request_SHA256':sha256((ROOT/'runs/C3-SPIN-OO'/case/method/'request.json').read_bytes())}
        if (directory/'request.json').exists() and load(directory/'request.json')!=binding:
            raise RuntimeError('C4 request drift')
        if (directory/'result.json').exists():
            return load(directory/'result.json')
        if (directory/'started.json').exists():
            raise RuntimeError('C4 partial work preserved; no blind retry')
        space()
        save(directory/'request.json',binding)
        save(directory/'started.json',{'pid':os.getpid(),'unix':time()})
        cpu,tick = process_time(),perf_counter()
        def checkpoint(history,rotations):
            save(directory/'checkpoint.json',{'history':history,'new_quantum_model_shots':0})
            write(directory/'rotations-checkpoint.npz',arrays(rotations=np.array(rotations)))
        try:
            result = optimize(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),strings,spec,checkpoint)
            rotations,hh,gg = (result.pop(k) for k in ('rotations','final_h1','final_eri'))
            payload = arrays(rotations=np.array(rotations),h1=hh,eri=gg,CI_strings=strings)
            write(directory/'optimized-representation.npz',payload)
            result.update(**binding,status='COMPLETED',representation_SHA256=sha256(payload),
                whole_body_CPU_seconds=process_time()-cpu,whole_body_wall_seconds=perf_counter()-tick,
                final_state_is_classical_CI_with_variational_orbitals_not_original_sampler_state=True,
                physical_sampling_in_new_basis_performed=False,world_best_claim_authorized=False)
            save(directory/'result.json',result)
            print(case,method,'E',result['final_energy_Ha'],'S2',result['history'][-1]['spin_square'],
                'S0_dim',result['spin_basis']['singlet_dimension'],flush=True)
            return result
        except Exception as exc:
            save(directory/'failure.json',{'error':repr(exc),'CPU_seconds':process_time()-cpu,
                'wall_seconds':perf_counter()-tick,'new_quantum_model_shots':0})
            raise


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--freeze',action='store_true')
    p.add_argument('--case')
    p.add_argument('--method')
    a = p.parse_args()
    if a.freeze:
        path = ROOT/'freeze-C4-SINGLET-OO.json'
        if path.exists() and load(path)!=freeze():
            raise RuntimeError('Never replace C4 freeze')
        save(path,freeze())
    else:
        run(a.case,a.method)
