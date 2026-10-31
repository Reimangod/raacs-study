"""Immutable-input paired C3 experiment. No new sampling, no reference use."""
import argparse
import fcntl
import inspect
import os
from importlib.metadata import version
from pathlib import Path
from time import time, process_time, perf_counter
import ffsim
import numpy as np
from pyscf.fci import addons, selected_ci
from paperstudy.ci import common_strings
from raacsnext.run import physical
from .paths import ROOT
from .io import load, save, write, arrays, sha256, space
from .run_orbital import source
from .spin_orbital import optimize


def freeze():
    paths = [ROOT/'protocol-C3-SPIN-OO.json', ROOT/'frontierloop/spin_orbital.py',
        Path(__file__), ROOT/'frontierloop/run_orbital.py', ROOT/'test_spin_orbital.py']
    paths += [Path(inspect.getfile(f)) for f in
        [ffsim.optimize_orbitals, addons.fix_spin_, selected_ci.contract_ss]]
    return {'files':{str(p):sha256(p.read_bytes()) for p in paths},
        'runtime':{p:version(p) for p in ('numpy','scipy','pyscf','ffsim','jax')},
        'new_model_shots_authorized':0, 'world_best_claim_authorized':False}


def run(case, method):
    spec = load(ROOT/'protocol-C3-SPIN-OO.json')
    if case not in spec['cases'] or method not in spec['methods']:
        raise ValueError('Unregistered C3 cell')
    current = freeze()
    if load(ROOT/'freeze-C3-SPIN-OO.json') != current:
        raise RuntimeError('C3 code/protocol/runtime changed after freeze')
    directory = ROOT/'runs/C3-SPIN-OO'/case/method
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        m, h, g = physical(case)
        ids, counts, provenance = source(case, method, m)
        strings, info = common_strings(ids, counts, m['norb'], (m['nalpha'],m['nbeta']), spec['r_cap'])
        previous = load(ROOT/'runs/C2-OO'/case/method/'request.json')
        if previous['source'] != provenance or previous['CI_strings'] != strings.tolist():
            raise RuntimeError('C3 changed the paired source or CI subspace')
        binding = {'case_id':case, 'method':method, 'cycle':'C3-SPIN-OO',
            'physical_problem_id':m['problem_id'], 'source':provenance,
            'code_freeze':current, 'CI_strings':strings.tolist(), 'original_counts_context':info,
            'C2_request_SHA256':sha256((ROOT/'runs/C2-OO'/case/method/'request.json').read_bytes())}
        if (directory/'request.json').exists() and load(directory/'request.json') != binding:
            raise RuntimeError('C3 request drift')
        if (directory/'result.json').exists():
            return load(directory/'result.json')
        if (directory/'started.json').exists():
            raise RuntimeError('Preserve partial C3 run; no blind retry')
        space()
        save(directory/'request.json', binding)
        save(directory/'started.json', {'unix':time(), 'pid':os.getpid()})
        cpu, tick = process_time(), perf_counter()
        def checkpoint(history, rotations):
            save(directory/'checkpoint.json', {'history':history, 'new_quantum_model_shots':0})
            write(directory/'rotations-checkpoint.npz', arrays(rotations=np.array(rotations)))
        try:
            result = optimize(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),strings,spec,checkpoint)
            rotations, hh, gg = (result.pop(k) for k in ('rotations','final_h1','final_eri'))
            payload = arrays(rotations=np.array(rotations),h1=hh,eri=gg,CI_strings=strings)
            write(directory/'optimized-representation.npz', payload)
            result.update(**binding, status='COMPLETED', representation_SHA256=sha256(payload),
                whole_body_CPU_seconds=process_time()-cpu, whole_body_wall_seconds=perf_counter()-tick,
                final_state_is_classical_CI_with_variational_orbitals_not_original_sampler_state=True,
                physical_sampling_in_new_basis_performed=False, world_best_claim_authorized=False)
            save(directory/'result.json',result)
            print(case,method,'physical',result['final_energy_Ha'],
                'S2',result['history'][-1]['spin_square'],flush=True)
            return result
        except Exception as exc:
            save(directory/'failure.json', {'error':repr(exc),'CPU_seconds':process_time()-cpu,
                'wall_seconds':perf_counter()-tick,'new_quantum_model_shots':0})
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--freeze',action='store_true')
    p.add_argument('--case')
    p.add_argument('--method')
    a = p.parse_args()
    if a.freeze:
        target = ROOT/'freeze-C3-SPIN-OO.json'
        if target.exists() and load(target)!=freeze():
            raise RuntimeError('Never overwrite a C3 freeze')
        save(target,freeze())
    else:
        run(a.case,a.method)
