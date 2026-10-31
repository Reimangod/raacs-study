"""Additive one-A100 qualification of the actual compressed-circuit adapter."""
import fcntl
from io import BytesIO
import json
from pathlib import Path
import shlex
import subprocess
from time import perf_counter,process_time
import numpy as np
import ffsim
from qiskit import qpy
from .paths import ROOT, OLD
from .io import load,save,write,arrays,sha256,digest,space
from .primitive_plan import plan,expand_for_execution
from .merged_compression import materialize
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST
from paperstudy.gpu_qualification import PYTHON, ENV

DIRECTORY=ROOT/'qualification/D2-primitive-GPU-v2'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/D2-primitive-GPU-v2'
BASE=OLD/'paperstudy/gpu_lucj_engine.py'
WORKER=ROOT/'frontierloop/primitive_gpu_worker.py'


def build_input(qc,n,nelec,directory,expected,problem='synthetic_qualification',sampling_requests=None):
    directory.mkdir(parents=True,exist_ok=True)
    qc=expand_for_execution(qc)
    conf,values=plan(qc,n,nelec)
    if expected is not None:values['expected_sector_vector']=expected
    blob=arrays(**values);write(directory/'arrays.npz',blob)
    stream=BytesIO();qpy.dump(qc,stream);qblob=stream.getvalue();write(directory/'source.qpy',qblob)
    conf.update(arrays_SHA256=sha256(blob),worker_SHA256=sha256(WORKER.read_bytes()),
        base_kernel_SHA256=sha256(BASE.read_bytes()),source_QPY_SHA256=sha256(qblob),
        physical_problem_id=problem,sampling_requests=sampling_requests or [])
    conf['input_id']=digest(conf);save(directory/'config.json',conf)
    return conf


def prepare():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    if (DIRECTORY/'prepared.json').exists():return load(DIRECTORY/'prepared.json')
    if (DIRECTORY/'started.json').exists():raise RuntimeError('Partial qualification inputs preserved')
    import xml.etree.ElementTree as ET
    suites=list(ET.parse(ROOT/'qualification-primitive-QPY-remediation-v2.xml').getroot().iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=7 or any(int(s.get(k,'0')) for s in suites for k in ('failures','errors')):
        raise RuntimeError('Actual QPY roundtrip and primitive semantics not qualified')
    space();save(DIRECTORY/'started.json',{'purpose':'actual compressed circuit GPU parity; zero new shots'})
    inputs=[];cpu,tick=process_time(),perf_counter()
    # Every qualification includes independent full-Fock small-circuit tests
    # already passed on CPU; these references use ffsim's primitive evolution.
    for seed,ne in [(4401,(2,2)),(4402,(3,3))]:
        rng=np.random.default_rng(seed);n=4
        aa=rng.normal(0,.2,(n,n));aa=(aa+aa.T)/2
        ab=rng.normal(0,.2,(n,n));ab=(ab+ab.T)/2
        op=ffsim.UCJOpSpinBalanced(np.array([[aa,ab]]),np.array([ffsim.random.random_unitary(n,seed=seed)]),
            ffsim.random.random_unitary(n,seed=seed+1))
        q,_=materialize(op,ne,1);expected=ffsim.qiskit.final_state_vector(q,norb=n,nelec=ne).vec
        name='small-'+str(seed);conf=build_input(q,n,ne,DIRECTORY/name,expected)
        inputs.append({'name':name,'input_id':conf['input_id'],'kind':'independent_small_sector_parity'})
    case='NAPH20-BLOCK-BOYS';d=ROOT/'runs/D1-MERGED-RESOURCE'/case/'8'
    r=load(d/'result.json');p=d/'native-materialized.qpy'
    if sha256(p.read_bytes())!=r['QPY_SHA256']['native-materialized']:raise RuntimeError('D1 QPY binding changed')
    q=qpy.load(BytesIO(p.read_bytes()))[0];m=load(OLD/'data/physical'/case/'problem.json');ne=m['nalpha'],m['nbeta']
    expected=ffsim.qiskit.final_state_vector(expand_for_execution(q),norb=m['norb'],nelec=ne).vec
    name='physical-20q';conf=build_input(q,m['norb'],ne,DIRECTORY/name,expected,m['problem_id'])
    inputs.append({'name':name,'input_id':conf['input_id'],'kind':'actual_materialized_physical_20q_state_parity',
        'source_D1_result_SHA256':sha256((d/'result.json').read_bytes())})
    value={'inputs':inputs,'new_model_shots':0,'CPU_reference_preparation_seconds':process_time()-cpu,
        'wall_seconds':perf_counter()-tick,'qualification_not_production':True,
        'source_SHA256':{str(p):sha256(p.read_bytes()) for p in (WORKER,BASE,ROOT/'frontierloop/primitive_plan.py',Path(__file__))},
        'independent_full_Fock_CPU_tests':sha256((ROOT/'qualification-merged-compression-tests.xml').read_bytes()),
        'CPU_serialization_tests':sha256((ROOT/'qualification-primitive-QPY-remediation-v2.xml').read_bytes())}
    save(DIRECTORY/'prepared.json',value);return value


def submit():
    prepared=prepare()
    with (DIRECTORY/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (DIRECTORY/'job.json').exists():return load(DIRECTORY/'job.json')
        if (DIRECTORY/'submission-intent.json').exists():raise RuntimeError('Uncertain sbatch: reconcile instead of resubmit')
        state=ssh('squeue -h -u reiwatanabe -o "%i %T %j %b %R"',timeout=45)
        save(DIRECTORY/'pre-dispatch-queue.json',{'raw':state,'other_jobs_not_modified':True,'one_A100_requested':True})
        ssh('mkdir -p '+shlex.quote(REMOTE));send(WORKER,REMOTE+'/worker.py');send(BASE,REMOTE+'/base_kernel.py')
        commands=[]
        for item in prepared['inputs']:
            name=item['name'];remote=REMOTE+'/'+name;ssh('mkdir -p '+shlex.quote(remote))
            for fn in ('config.json','arrays.npz'):send(DIRECTORY/name/fn,remote+'/'+fn)
            commands.append(shlex.join([PYTHON,'-u',REMOTE+'/worker.py',remote+'/config.json',remote+'/arrays.npz',remote+'/output']))
        script='\n'.join(['#!/bin/bash','set -euo pipefail',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1',
            'export CUDA_PATH='+ENV+'/lib/python3.12/site-packages/nvidia/cuda_runtime',
            'export LD_LIBRARY_PATH="$CUDA_PATH/lib:'+ENV+'/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"',
            'export CUPY_ACCELERATORS=""',*commands])+'\n'
        write(DIRECTORY/'job.sbatch',script.encode());send(DIRECTORY/'job.sbatch',REMOTE+'/job.sbatch')
        command=shlex.join(['sbatch','--parsable','--partition=gpu-short','--gres=gpu:a100:1',
            '--cpus-per-task=1','--mem=8G','--time=00:05:00','--job-name=frontier-D2-primitive-qual-v2',
            '--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err',REMOTE+'/job.sbatch'])
        intent={'remote':REMOTE,'command':command,'prepared_SHA256':sha256((DIRECTORY/'prepared.json').read_bytes()),
            'model_shots':0,'purpose':'CUDA primitive adapter numerical qualification, not performance'}
        save(DIRECTORY/'submission-intent.json',intent)
        reply=ssh(command).strip().split(';')[0]
        if not reply.isdigit():raise RuntimeError('Uncertain Slurm response')
        job={**intent,'job_id':reply};save(DIRECTORY/'job.json',job);print(job,flush=True);return job


def collect():
    job=load(DIRECTORY/'job.json')
    if (DIRECTORY/'collected.json').exists():return load(DIRECTORY/'collected.json')
    live=ssh('squeue -h -j '+job['job_id']+' -o "%i %T %j %R"',timeout=45)
    accounting=ssh('sacct -X -n -j '+job['job_id']+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=45)
    save(DIRECTORY/'observed.json',{'live':live,'accounting':accounting})
    if live.strip():print('QUALIFICATION_LIVE',live.strip(),flush=True);return
    if not any(s in accounting for s in ('COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED')):
        print('NOT_YET_AUTHORITATIVELY_TERMINAL',accounting,flush=True);return
    prepared=load(DIRECTORY/'prepared.json');checks=[]
    for item in prepared['inputs']:
        name=item['name'];out=DIRECTORY/name/'output';out.mkdir(exist_ok=True)
        remote=REMOTE+'/'+name+'/output/result.json'
        exists=ssh('test -f '+shlex.quote(remote)+' && echo exists || true',timeout=45).strip()=='exists'
        if not exists:
            checks.append({'name':name,'status':'MISSING_RESULT','qualified':False});continue
        subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,HOST+':'+remote,str(out/'result.json')],check=True,timeout=90)
        r=load(out/'result.json');parity=r.get('independent_ffsim_parity')
        good=(r['input_id']==item['input_id'] and r['status']=='EXACT_MATERIALIZED_CIRCUIT_GPU_COMPLETED'
            and parity is not None and max(parity.values())<1e-8 and r['sampling_results']==[])
        checks.append({'name':name,'qualified':bool(good),'parity':parity,'source_SHA256':sha256((out/'result.json').read_bytes())})
    for name in ('slurm.out','slurm.err'):
        subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(DIRECTORY/name)],check=True,timeout=90)
    value={'status':'QUALIFIED_THREE_EXACT_CIRCUITS' if all(c['qualified'] for c in checks) else 'QUALIFICATION_FAILED_PRESERVED',
        'checks':checks,'new_model_shots':0,'accounting':accounting,'goal_complete':False}
    save(DIRECTORY/'collected.json',value);print(value,flush=True);return value


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','collect']);a=p.parse_args()
    globals()[a.action]()
