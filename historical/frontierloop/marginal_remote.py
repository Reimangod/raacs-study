"""Freeze/dispatch/collect a distinct F2 CPU study; never modify D2."""
import argparse
import fcntl
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from .paths import ROOT,OLD
from .io import load,save,write,sha256,digest,space
from paperstudy.budget import BudgetLedger
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST
from paperstudy.gpu_qualification import PYTHON,REMOTE as OLD_REMOTE

DIRECTORY=ROOT/'runs/F2-POSITIVE-MARGINAL-v1'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/F2-positive-marginal-v1'
RUNTIME={"numpy":"2.4.6","scipy":"1.17.1","ffsim":"0.0.84","pyscf":"2.14.0","qiskit":"2.6.0","jax":"0.10.2"}


def input_paths():
    spec=load(ROOT/'protocol-F2-POSITIVE-MARGINAL.json');paths={
        'worker.py':ROOT/'frontierloop/marginal_remote_worker.py',
        'frontierloop/__init__.py':ROOT/'frontierloop/remote_package_init.py',
        'paperstudy/__init__.py':ROOT/'frontierloop/remote_package_init.py',
        'paperstudy/ci.py':OLD/'paperstudy/ci.py'}
    for name in ['conditional_marginal.py','primitive_plan.py','orbital.py','spin_orbital.py',
        'fast_orbital.py','energy_cache.py','hole_rdm.py','singlet.py']:
        paths['frontierloop/'+name]=ROOT/'frontierloop'/name
    for case in spec['cases']:
        paths['inputs/'+case+'/operator.npz']=OLD/'prepared'/case/'C_LUCJ_COMPRESSED_L1-L1/operator.npz'
        paths['inputs/'+case+'/integrals.npz']=OLD/'data/physical'/case/'integrals.npz'
    return paths


def binding():
    spec=load(ROOT/'protocol-F2-POSITIVE-MARGINAL.json')
    return {'protocol_SHA256':sha256((ROOT/'protocol-F2-POSITIVE-MARGINAL.json').read_bytes()),
        'input_SHA256':{name:sha256(p.read_bytes()) for name,p in input_paths().items()},
        'physical_metadata_SHA256':{case:sha256((OLD/'data/physical'/case/'problem.json').read_bytes()) for case in spec['cases']},
        'readout_specs_SHA256':{mode:sha256((ROOT/('protocol-'+mode+'.json')).read_bytes()) for mode in spec['readouts']},
        'standalone_qualification_SHA256':sha256((ROOT/'qualification-F2-standalone-tests.xml').read_bytes()),
        'fast_backend_qualification_SHA256':sha256((ROOT/'qualification-fast-orbital-krylov-v2-tests.xml').read_bytes()),
        'dispatcher_SHA256':sha256(Path(__file__).read_bytes()),'expected_runtime':RUNTIME}


def prepare():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    value=binding()
    if (DIRECTORY/'manifest.json').exists():
        manifest=load(DIRECTORY/'manifest.json')
        if manifest['freeze']!=value:raise RuntimeError('Never change an F2 freeze')
        return manifest
    suites=list(ET.parse(ROOT/'qualification-F2-standalone-tests.xml').getroot().iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=2 or any(int(s.get(k,'0')) for s in suites for k in ('failures','errors')):
        raise RuntimeError('Standalone F2 worker not qualified')
    space();spec=load(ROOT/'protocol-F2-POSITIVE-MARGINAL.json')
    spec['readout_specs']={mode:load(ROOT/('protocol-'+mode+'.json')) for mode in spec['readouts']}
    cases={}
    for case in spec['cases']:
        m=load(OLD/'data/physical'/case/'problem.json')
        physical={k:m[k] for k in ['norb','nalpha','nbeta','core_energy','problem_id']}
        source=OLD/'prepared'/case/'C_LUCJ_COMPRESSED_L1-L1/construction.json'
        cases[case]={'physical':physical,'operator_file':'inputs/'+case+'/operator.npz',
            'integrals_file':'inputs/'+case+'/integrals.npz',
            'run_id':digest(['F2-positive-marginal-v1',physical['problem_id'],value]),
            'source_initialization_record':{'path':str(source),'SHA256':sha256(source.read_bytes())}}
    manifest={'freeze':value,'protocol':spec,'cases':cases,'input_SHA256':value['input_SHA256'],
        'expected_runtime':RUNTIME,'source_energies_or_FCI_not_provided':True,
        'D2_results_not_provided':True,'GPU_requested':False,'goal_complete':False}
    save(DIRECTORY/'manifest.json',manifest);return manifest


def submit():
    manifest=prepare()
    with (DIRECTORY/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (DIRECTORY/'job.json').exists():return load(DIRECTORY/'job.json')
        if (DIRECTORY/'submission-intent.json').exists():raise RuntimeError('Uncertain F2 Slurm submission, no duplicate')
        remote_free=ssh('df -Pk /share/reiwatanabe',timeout=45)
        fields=remote_free.strip().splitlines()[-1].split();free=int(fields[3])*1024
        if free<manifest['protocol']['remote_free_floor_bytes']+manifest['protocol']['new_namespace_bytes_cap']:
            raise OSError('Remote free reserve insufficient; do not lower floor')
        save(DIRECTORY/'remote-capacity.json',{'df':remote_free,'available_bytes':free,'local_molecular_numeric_work':False})
        runtime=__import__('json').loads(ssh('cat '+OLD_REMOTE+'/cpu-runtime-v1/runtime.json',timeout=45))
        if runtime['runtime']!=RUNTIME:raise RuntimeError('Remote CPU package runtime changed')
        ssh('mkdir -p '+shlex.quote(REMOTE))
        for name,path in input_paths().items():
            ssh('mkdir -p '+shlex.quote(str(Path(REMOTE+'/'+name).parent)))
            send(path,REMOTE+'/'+name)
        send(DIRECTORY/'manifest.json',REMOTE+'/manifest.json')
        script='\n'.join(['#!/bin/bash','set -euo pipefail',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_NUM_PROCS=1',
            'export JAX_PLATFORMS=cpu',
            'export PYTHONPATH='+REMOTE+':'+OLD_REMOTE+'/cpu-runtime-v1/packages',
            shlex.join([PYTHON,'-u',REMOTE+'/worker.py',REMOTE])])+'\n'
        write(DIRECTORY/'job.sbatch',script.encode());send(DIRECTORY/'job.sbatch',REMOTE+'/job.sbatch')
        ledger=BudgetLedger(ROOT/'production-shots.sqlite')
        for case,row in manifest['cases'].items():
            cap=manifest['protocol']['model_shots_per_case'];ledger.register(row['run_id'],cap)
            request={'kind':'F2-half-register-positive-mixture','physical_problem_id':row['physical']['problem_id'],
                'freeze_digest':digest(manifest['freeze']),'half_register_shots':cap,'source_operator_SHA256':
                    manifest['input_SHA256'][row['operator_file']]}
            if ledger.reserve(row['run_id'],'final',request,cap)!='NEW_RESERVATION':raise RuntimeError('Existing F2 draw reservation')
        command=shlex.join(['sbatch','--parsable','--partition=gpu-short','--cpus-per-task=1','--mem=8G',
            '--time=00:30:00','--job-name=frontier-F2-positive-CPU-v1','--output='+REMOTE+'/slurm.out',
            '--error='+REMOTE+'/slurm.err',REMOTE+'/job.sbatch'])
        intent={'remote':REMOTE,'command':command,'model_shots_reserved':400000,'GPU_requested':False,
            'manifest_SHA256':sha256((DIRECTORY/'manifest.json').read_bytes()),'freeze':manifest['freeze']}
        save(DIRECTORY/'submission-intent.json',intent)
        reply=ssh(command).strip().split(';')[0]
        if not reply.isdigit():raise RuntimeError('Uncertain F2 sbatch outcome')
        job={**intent,'job_id':reply};save(DIRECTORY/'job.json',job);print('F2_CPU_SUBMITTED',reply,flush=True);return job


def collect():
    job=load(DIRECTORY/'job.json');manifest=load(DIRECTORY/'manifest.json')
    if manifest['freeze']!=binding():raise RuntimeError('F2 source code changed after freeze')
    if (DIRECTORY/'collected.json').exists():return load(DIRECTORY/'collected.json')
    live=ssh('squeue -h -j '+job['job_id']+' -o "%i %T %j %R"',timeout=45)
    accounting=ssh('sacct -X -n -j '+job['job_id']+' --format=JobID,State,Start,End,Elapsed,ExitCode,AllocTRES -P',timeout=45)
    save(DIRECTORY/'observed.json',{'live':live,'accounting':accounting})
    if live.strip():print('F2_CPU_LIVE',live.strip(),flush=True);return None
    if not any(s in accounting for s in ['COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED']):return None
    # Copy only a bounded, finished namespace; no environment or statevector.
    remote_size=ssh('du -sk '+shlex.quote(REMOTE),timeout=45).split()[0]
    if int(remote_size)*1024>manifest['protocol']['new_namespace_bytes_cap']:
        raise RuntimeError('Remote namespace unexpectedly large, preserve without bulk download')
    space();out=DIRECTORY/'remote-output';out.mkdir(exist_ok=True)
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/.',str(out)],check=True,timeout=180)
    ledger=BudgetLedger(ROOT/'production-shots.sqlite');checks=[]
    for case,row in manifest['cases'].items():
        d=out/'outputs'/case
        if not (d/'draw-result.json').exists():
            checks.append({'case':case,'draw_status':'MISSING_PRESERVED','known_model_shots':None});continue
        r=load(d/'draw-result.json');payload=(d/'half-counts.npz').read_bytes()
        if r['run_id']!=row['run_id'] or sha256(payload)!=r['counts_SHA256'] or r['known_model_shots']!=100000:
            raise RuntimeError('F2 raw count/request mismatch')
        ledger.complete(row['run_id'],'final',r['known_model_shots'],sha256(payload))
        checks.append({'case':case,'draw_status':'COMPLETED','known_model_shots':r['known_model_shots'],
            'readouts':{mode:'COMPLETED' if (d/mode/'result.json').exists() else 'NOT_COMPLETED' for mode in manifest['protocol']['readouts']}})
    result={'status':'REMOTE_F2_PREFIX_COLLECTED_NOT_SUPERIORITY','checks':checks,'accounting':accounting,
        'all_four_draws_and_twelve_readouts_complete':all(c.get('known_model_shots')==100000 and all(v=='COMPLETED' for v in c['readouts'].values()) for c in checks),
        'manifest_SHA256':job['manifest_SHA256'],'goal_complete':False}
    save(DIRECTORY/'collected.json',result);print(result,flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','collect']);a=p.parse_args();globals()[a.action]()
