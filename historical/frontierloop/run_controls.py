"""C5 applies every existing spin/OO rule equally to stronger controls."""
import argparse
import fcntl
import os
from time import time,process_time,perf_counter
import numpy as np
from paperstudy.ci import common_strings
from raacsnext.run import physical
from .paths import ROOT
from .io import load,save,write,arrays,sha256,space
from .control_sources import build_manifest,load_source
from .run_singlet_orbital import freeze as prior_freeze
from .orbital import optimize as unconstrained
from .spin_orbital import optimize as penalized
from .singlet import optimize as singlet


def freeze():
    names=['control_sources.py','run_controls.py','orbital.py','spin_orbital.py','singlet.py']
    paths=[ROOT/'frontierloop'/name for name in names]
    paths += [ROOT/'protocol-C5-CONTROLS.json',ROOT/'test_control_sources.py',ROOT/'manifest-C5-CONTROLS.json']
    paths += [ROOT/('protocol-'+name+'.json') for name in ('C2-OO','C3-SPIN-OO','C4-SINGLET-OO')]
    return {'files':{str(p):sha256(p.read_bytes()) for p in paths},
        'readout_dependencies':prior_freeze(), 'new_model_shots':0}


def run(case,method,readout):
    spec=load(ROOT/'protocol-C5-CONTROLS.json')
    if case not in spec['cases'] or method not in spec['methods'] or readout not in spec['readouts']:
        raise ValueError('Unregistered C5 cell')
    current=freeze()
    if load(ROOT/'freeze-C5-CONTROLS.json')!=current:
        raise RuntimeError('C5 code/protocol/source manifest changed')
    cell=next(x for x in load(ROOT/'manifest-C5-CONTROLS.json')['cells'] if x['case']==case and x['method']==method)
    directory=ROOT/'runs/C5-CONTROLS'/case/method/readout
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (directory/'result.json').exists():
            return load(directory/'result.json')
        if cell['status']!='AVAILABLE':
            result={'case_id':case,'method':method,'readout':readout,'status':cell['status'],
                'source':cell,'freeze':current,'new_quantum_model_shots':0,'world_best_claim_authorized':False}
            save(directory/'result.json',result)
            return result
        if (directory/'started.json').exists():
            raise RuntimeError('Partial C5 work preserved; no blind retry')
        m,h,g=physical(case)
        if m['problem_id']!=cell['expected_problem_id']:
            raise RuntimeError('C5 physical problem changed')
        ids,counts=load_source(cell)
        strings,info=common_strings(ids,counts,m['norb'],(m['nalpha'],m['nbeta']),spec['r_cap'])
        request={'case_id':case,'method':method,'readout':readout,'physical_problem_id':m['problem_id'],
            'source':cell,'freeze':current,'CI_strings':strings.tolist(),'original_counts_context':info}
        space();save(directory/'request.json',request)
        save(directory/'started.json',{'pid':os.getpid(),'unix':time()})
        cpu,tick=process_time(),perf_counter()
        def checkpoint(history,rotations):
            save(directory/'checkpoint.json',{'history':history,'new_quantum_model_shots':0})
            write(directory/'rotations-checkpoint.npz',arrays(rotations=np.array(rotations)))
        try:
            fn={'C2-OO':unconstrained,'C3-SPIN-OO':penalized,'C4-SINGLET-OO':singlet}[readout]
            oo_spec=load(ROOT/('protocol-'+readout+'.json'))
            result=fn(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),strings,oo_spec,checkpoint)
            rotations,hh,gg=(result.pop(k) for k in ('rotations','final_h1','final_eri'))
            payload=arrays(rotations=np.array(rotations),h1=hh,eri=gg,CI_strings=strings)
            write(directory/'optimized-representation.npz',payload)
            result.update(**request,status='COMPLETED',representation_SHA256=sha256(payload),
                whole_body_CPU_seconds=process_time()-cpu,whole_body_wall_seconds=perf_counter()-tick,
                final_state_is_classical_CI_with_variational_orbitals_not_original_sampler_state=True,
                physical_sampling_in_new_basis_performed=False,world_best_claim_authorized=False)
            save(directory/'result.json',result)
            print(case,method,readout,'E',result['final_energy_Ha'],'CPU',result['CPU_seconds'],flush=True)
            return result
        except Exception as exc:
            failure={**request,'status':'FAILED_ENGINEERING_PRESERVED','error':repr(exc),
                'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,'new_quantum_model_shots':0}
            save(directory/'failure.json',failure)
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true')
    p.add_argument('--case');p.add_argument('--method');p.add_argument('--readout');a=p.parse_args()
    if a.freeze:
        manifest=ROOT/'manifest-C5-CONTROLS.json';value=build_manifest()
        if manifest.exists() and load(manifest)!=value:
            raise RuntimeError('Never replace a frozen source manifest')
        save(manifest,value)
        path=ROOT/'freeze-C5-CONTROLS.json';value=freeze()
        if path.exists() and load(path)!=value:
            raise RuntimeError('Never replace a C5 freeze')
        save(path,value)
        print({'available':load(manifest)['available_sources'],'missing':load(manifest)['missing_sources']})
    else:
        run(a.case,a.method,a.readout)
