"""Fair no-shot readout expansion: strong controls/RAACS/classical, fixed sources."""
import fcntl
import os
from collections import Counter
from pathlib import Path
from time import process_time,perf_counter
import numpy as np
import ffsim
from .paths import ROOT,OLD
from .io import load,save,sha256,space,digest
from .control_sources import load_source
from .cisd_sparse import marginal_strings
from .lowest_orbital import Reader
from .marginal_remote_worker import select_half_pool

DIRECTORY=ROOT/'runs/R2-COMMON-CI-v1'


def source_manifest():
    cells=[];old=load(ROOT/'manifest-C5-CONTROLS.json')
    for cell in old['cells']:
        row={**cell,'family':'C5','kind':'QUANTUM_FULL_COUNTS','method_id':'C5_'+cell['method']}
        if cell['status']=='AVAILABLE':
            load_source(cell)
            row.update(payload=str(Path(cell['source_result']).parent/'final-counts.npz'),payload_SHA256=cell['source_counts_SHA256'])
        cells.append(row)
    for case in load(ROOT/'protocol-R2-COMMON-CI.json')['cases']:
        for cycle,seed in [('B3',73001),('B4',74001)]:
            p=ROOT/'runs'/cycle/case/str(seed)/'result.json';r=load(p);payload=p.parent/'final-counts.npz'
            if sha256(payload.read_bytes())!=r['final_CI'][0]['source_counts_SHA256']:raise RuntimeError('RAACS final counts drift')
            cells.append({'case':case,'method_id':'RAACS_'+cycle,'family':cycle,'kind':'QUANTUM_FULL_COUNTS','status':'AVAILABLE',
                'expected_problem_id':r['problem_id'],'source_result':str(p),'source_result_SHA256':sha256(p.read_bytes()),
                'payload':str(payload),'payload_SHA256':sha256(payload.read_bytes()),'source_total_budget':r['budget'],
                'source_final_model_shots':r['final_model_shots'],'source_pilot_model_shots':r['pilot_model_shots'],
                'source_construction_CPU_seconds':r['construction_CPU_seconds'],'sampler_resources':r['sampling_resources'],
                'sampler_coefficient_slots_mean':r['coefficient_slots_shot_weighted_mean']})
        p=ROOT/'runs/C6-CLASSICAL-CISD'/case/'source/result.json';r=load(p)
        for variant,name,key in [('DETERMINISTIC_CISD_MARGINAL','sparse-CISD-state.npz','sparse_state_SHA256'),('CLASSICAL_CISD_100K_DRAWS','classical-counts.npz','classical_counts_SHA256')]:
            payload=p.parent/name
            if sha256(payload.read_bytes())!=r[key]:raise RuntimeError('Classical source payload drift')
            cells.append({'case':case,'method_id':variant,'family':'C6','kind':'CLASSICAL_COEFFICIENT_MARGINAL' if variant.startswith('DETERMINISTIC') else 'CLASSICAL_FULL_COUNTS',
                'status':'AVAILABLE','expected_problem_id':r['physical_problem_id'],'source_result':str(p),'source_result_SHA256':sha256(p.read_bytes()),
                'payload':str(payload),'payload_SHA256':sha256(payload.read_bytes()),'source_total_budget':0,
                'source_final_model_shots':0,'classical_draws':0 if variant.startswith('DETERMINISTIC') else r['classical_draws'],
                'source_construction_CPU_seconds':r['whole_body_CPU_seconds'],'sampler_resources':None,'sampler_coefficient_slots_mean':None})
    return cells


def prepare():
    spec=load(ROOT/'protocol-R2-COMMON-CI.json');cells=source_manifest()
    if sum(c['status']=='AVAILABLE' for c in cells)!=spec['expected_available_sources'] or sum(c['status']!='AVAILABLE' for c in cells)!=spec['expected_missing_same_problem_sources']:
        raise RuntimeError('Unexpected source coverage, do not silently drop/add cases')
    names=['common_ci_cap.py','lowest_orbital.py','lowest_penalized.py','hole_rdm.py','energy_cache.py','cisd_sparse.py','marginal_remote_worker.py','control_sources.py']
    b={'code_SHA256':{n:sha256((ROOT/'frontierloop'/n).read_bytes()) for n in names},
        'protocol_SHA256':sha256((ROOT/'protocol-R2-COMMON-CI.json').read_bytes()),
        'settings_SHA256':sha256((ROOT/'protocol-C3-SPIN-OO.json').read_bytes()),
        'qualification_SHA256':sha256((ROOT/'qualification-lowest-orbital-tests.xml').read_bytes()),
        'physical_SHA256':{c:{n:sha256((OLD/'data/physical'/c/n).read_bytes()) for n in ['problem.json','integrals.npz']} for c in spec['cases']},'sources':cells}
    DIRECTORY.mkdir(exist_ok=True);p=DIRECTORY/'freeze.json'
    if p.exists():
        if load(p)!=b:raise RuntimeError('R2 frozen source/code changed')
    else:save(p,b)
    print('R2_FROZEN',len(cells),'available',sum(c['status']=='AVAILABLE' for c in cells),flush=True)
    return spec,b


def run(cell,spec,b):
    case=cell['case'];d=DIRECTORY/case/cell['method_id'];d.mkdir(parents=True,exist_ok=True)
    if cell['status']!='AVAILABLE':
        save(d/'missing.json',{'status':'MISSING_SAME_PROBLEM_SOURCE','new_model_shots':0});return 'MISSING_SAME_PROBLEM_SOURCE'
    if (d/'result.json').exists():return 'COMPLETED'
    if (d/'started.json').exists():return 'PARTIAL_OR_FAILED_PRESERVED_NO_RETRY'
    space();physical=OLD/'data/physical'/case;m=load(physical/'problem.json');n=m['norb'];ne=m['nalpha'],m['nbeta']
    if m['problem_id']!=cell['expected_problem_id']:raise RuntimeError('R2 physical problem mismatch')
    if sha256(Path(cell['source_result']).read_bytes())!=cell['source_result_SHA256'] or sha256(Path(cell['payload']).read_bytes())!=cell['payload_SHA256']:raise RuntimeError('R2 fixed source drift')
    if cell['kind']=='CLASSICAL_COEFFICIENT_MARGINAL':
        with np.load(cell['payload'],allow_pickle=False) as a:strings=marginal_strings(a['blocked_ids'],a['coefficients'],n,ne[0],spec['r_cap'])
    else:
        frequency=Counter();mask=(1<<n)-1
        with np.load(cell['payload'],allow_pickle=False) as a:
            for x,c in zip(a['blocked_ids'],a['counts']):
                if (int(x)&mask).bit_count()!=ne[0] or (int(x)>>n).bit_count()!=ne[1]:raise RuntimeError('R2 source outside particle sector')
                frequency[int(x)&mask]+=int(c);frequency[int(x)>>n]+=int(c)
        ids=np.array(sorted(frequency),np.uint64);counts=np.array([frequency[int(x)] for x in ids],np.uint64)
        strings=select_half_pool(ids,counts,n,ne[0],spec['r_cap'])
    request={'case':case,'method':cell['method_id'],'physical_problem_id':m['problem_id'],'source':cell,
        'CI_strings':strings.tolist(),'r_cap':spec['r_cap'],'freeze_digest':digest(b),'new_model_shots':0}
    save(d/'request.json',request);save(d/'started.json',{'pid':os.getpid(),'request_SHA256':sha256((d/'request.json').read_bytes())})
    cpu,tick=process_time(),perf_counter()
    try:
        with np.load(physical/'integrals.npz',allow_pickle=False) as a:ham=ffsim.MolecularHamiltonian(a['h1'],a['eri'],constant=m['core_energy'])
        settings={**load(ROOT/'protocol-C3-SPIN-OO.json'),'per_CI_wall_cap_seconds':spec['per_CI_wall_cap_seconds']}
        r,_=Reader(n,ne,strings,settings).diagonalize(ham)
        r.update(status='COMPLETED',case=case,method=cell['method_id'],physical_problem_id=m['problem_id'],
            CI_strings=strings.tolist(),r_cap=spec['r_cap'],actual_CI_dimension=len(strings)**2,
            source_payload_SHA256=cell['payload_SHA256'],incremental_CPU_seconds=process_time()-cpu,incremental_wall_seconds=perf_counter()-tick,
            orbital_optimization_performed=False,new_model_shots=0,actual_QPU_shots=0,goal_complete=False)
        save(d/'result.json',r);print('R2_CI',case,cell['method_id'],r['energy_Ha'],len(strings)**2,flush=True);return 'COMPLETED'
    except Exception as exc:
        save(d/'failure.json',{'status':'FAILED_ENGINEERING_PRESERVED','error':repr(exc),'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,'new_model_shots':0})
        print('R2_FAILED',case,cell['method_id'],repr(exc),flush=True);return 'FAILED_ENGINEERING_PRESERVED'


def main(prepare_only=False):
    spec,b=prepare()
    if prepare_only:return
    with (DIRECTORY/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(DIRECTORY/'owner.json',{'pid':os.getpid(),'authority':'EXISTING_DATA_CI_ONLY_NO_SHOT_EXECUTION'})
        states=[]
        for cell in b['sources']:
            status=run(cell,spec,b);states.append({'case':cell['case'],'method':cell['method_id'],'status':status})
            save(DIRECTORY/'progress.json',{'states':states,'completed':sum(s['status']=='COMPLETED' for s in states),'planned_available':49,'planned_missing':7,'new_model_shots':0,'goal_complete':False})
        save(DIRECTORY/'terminal.json',{'states':states,'completed':sum(s['status']=='COMPLETED' for s in states),'new_model_shots':0,'goal_complete':False})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();main(args.prepare_only)
