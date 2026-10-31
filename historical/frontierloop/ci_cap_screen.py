"""All fixed cases/methods at R64/128 in unchanged orbitals, zero new shots."""
import fcntl
from collections import Counter
from pathlib import Path
from time import process_time,perf_counter
import numpy as np
import ffsim
from .paths import ROOT,OLD
from .io import load,save,sha256,space,digest
from .lowest_orbital import Reader
from .lowest_readouts import source as source_path
from .marginal_remote_worker import select_half_pool
from paperstudy.budget import BudgetLedger

DIRECTORY=ROOT/'runs/R1-CI-CAP-SCREEN-v1'


def binding():
    names=['ci_cap_screen.py','lowest_orbital.py','lowest_penalized.py','hole_rdm.py','energy_cache.py','marginal_remote_worker.py']
    spec=load(ROOT/'protocol-R1-CI-CAP-SCREEN.json');sources={}
    for case in spec['cases']:
        for method in spec['methods']:
            d,half=source_path(case,method);p=d/('half-counts.npz' if half else 'final-counts.npz')
            sources[case+'/'+method]={'counts_SHA256':sha256(p.read_bytes()),'draw_SHA256':sha256((d/'draw-result.json').read_bytes())}
    return {'protocol_SHA256':sha256((ROOT/'protocol-R1-CI-CAP-SCREEN.json').read_bytes()),
        'code_SHA256':{n:sha256((ROOT/'frontierloop'/n).read_bytes()) for n in names},'sources':sources,
        'original_C3_settings_SHA256':sha256((ROOT/'protocol-C3-SPIN-OO.json').read_bytes()),
        'qualification_SHA256':sha256((ROOT/'qualification-lowest-orbital-tests.xml').read_bytes())}


def main():
    DIRECTORY.mkdir(exist_ok=True);b=binding();f=DIRECTORY/'freeze.json'
    if f.exists():
        if load(f)!=b:raise RuntimeError('R1 freeze changed')
    else:save(f,b)
    with (DIRECTORY/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spec=load(ROOT/'protocol-R1-CI-CAP-SCREEN.json');states=[]
        settings={**load(ROOT/'protocol-C3-SPIN-OO.json'),'per_CI_wall_cap_seconds':spec['per_CI_wall_cap_seconds']}
        for case in spec['cases']:
            physical=OLD/'data/physical'/case;m=load(physical/'problem.json');n=m['norb'];ne=m['nalpha'],m['nbeta']
            with np.load(physical/'integrals.npz',allow_pickle=False) as a:ham=ffsim.MolecularHamiltonian(a['h1'],a['eri'],constant=m['core_energy'])
            for method in spec['methods']:
                src,half=source_path(case,method);draw=load(src/'draw-result.json');frequency=Counter()
                p=src/('half-counts.npz' if half else 'final-counts.npz')
                if sha256(p.read_bytes())!=draw['counts_SHA256']:raise RuntimeError('R1 count binding drift')
                with np.load(p,allow_pickle=False) as a:
                    if half:
                        for x,c in zip(a['half_strings'],a['counts']):frequency[int(x)]+=int(c)
                    else:
                        mask=(1<<n)-1
                        for x,c in zip(a['blocked_ids'],a['counts']):frequency[int(x)&mask]+=int(c);frequency[int(x)>>n]+=int(c)
                runid=draw['run_id'] if half else load(src/'request.json')['run_id']
                ids=np.array(sorted(frequency),np.uint64);counts=np.array([frequency[int(x)] for x in ids],np.uint64)
                previous=None;previous_objective=None
                for cap in spec['R_caps']:
                    d=DIRECTORY/case/method/str(cap);d.mkdir(parents=True,exist_ok=True)
                    strings=select_half_pool(ids,counts,n,ne[0],cap)
                    if previous is not None and not set(previous)<=set(strings):raise RuntimeError('Frequency-selected R spaces not nested')
                    previous=strings
                    if (d/'result.json').exists():
                        r=load(d/'result.json');previous_objective=r['penalized_energy_Ha'];states.append({'case':case,'method':method,'R':cap,'status':'COMPLETED'});continue
                    if (d/'started.json').exists():states.append({'case':case,'method':method,'R':cap,'status':'PARTIAL_OR_FAILED_PRESERVED'});continue
                    space();request={'case':case,'method':method,'r_cap':cap,'source_run_id':runid,'counts_SHA256':draw['counts_SHA256'],
                        'CI_strings':strings.tolist(),'physical_problem_id':m['problem_id'],'integrals_SHA256':sha256((physical/'integrals.npz').read_bytes()),
                        'freeze_digest':digest(b),'new_model_shots':0}
                    save(d/'request.json',request);BudgetLedger(ROOT/'production-shots.sqlite').record_readout(runid,'R1-C3-INITIAL-R'+str(cap),'final',request)
                    save(d/'started.json',{'request_SHA256':sha256((d/'request.json').read_bytes())});cpu,tick=process_time(),perf_counter()
                    try:
                        c7=ROOT/'runs/C7-LOWEST-ROOT-v1'/case/method/'result.json';reuse=False
                        if cap==64 and c7.exists():
                            old=load(c7);old_request=load(c7.parent/'request.json')
                            if old['CI_strings']!=strings.tolist() or old['source_counts_SHA256']!=draw['counts_SHA256'] or old['physical_problem_id']!=m['problem_id']:
                                raise RuntimeError('R64 C7 reuse context mismatch')
                            if old_request['integrals_SHA256']!=request['integrals_SHA256']:raise RuntimeError('R64 physical integrals mismatch')
                            r=dict(old['history'][0]);reuse=True
                        else:r,_=Reader(n,ne,strings,settings).diagonalize(ham)
                        if previous_objective is not None and r['penalized_energy_Ha']>previous_objective+1e-8:
                            raise RuntimeError('Increasing R exceeded nested-subspace variational bound')
                        previous_objective=r['penalized_energy_Ha']
                        r.update(status='COMPLETED',case=case,method=method,r_cap=cap,actual_CI_dimension=len(strings)**2,
                            source_counts_SHA256=draw['counts_SHA256'],physical_problem_id=m['problem_id'],CI_strings=strings.tolist(),
                            orbital_optimization_performed=False,C7_initial_reused=reuse,
                            inherited_initial_CI_CPU_seconds=None if reuse else process_time()-cpu,
                            reused_result_SHA256=sha256(c7.read_bytes()) if reuse else None,
                            incremental_CPU_seconds=process_time()-cpu,incremental_wall_seconds=perf_counter()-tick,
                            new_model_shots=0,actual_QPU_shots=0,goal_complete=False)
                        save(d/'result.json',r);states.append({'case':case,'method':method,'R':cap,'status':'COMPLETED'})
                        print('R1_CI',case,method,cap,r['energy_Ha'],'reused',reuse,flush=True)
                    except Exception as exc:
                        save(d/'failure.json',{'status':'FAILED_ENGINEERING_PRESERVED','error':repr(exc),'new_model_shots':0,
                            'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick})
                        states.append({'case':case,'method':method,'R':cap,'status':'FAILED_ENGINEERING_PRESERVED'});print('R1_FAILED',case,method,cap,repr(exc),flush=True)
                    save(DIRECTORY/'progress.json',{'states':states,'completed':sum(x['status']=='COMPLETED' for x in states),'planned':24,'new_model_shots':0,'goal_complete':False})
        save(DIRECTORY/'terminal.json',{'states':states,'completed':sum(x['status']=='COMPLETED' for x in states),'planned':24,'new_model_shots':0,'goal_complete':False})


if __name__=='__main__':main()
