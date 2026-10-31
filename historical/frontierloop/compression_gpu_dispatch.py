"""One fresh case per A100, bound to the frozen D2 requests and shot ledger."""
import fcntl
from pathlib import Path
import shlex
import subprocess
from .paths import ROOT
from .io import load,save,write,sha256
from .compression_trial import DIRECTORY,guard,validate_row,draw_request
from .primitive_gpu import DIRECTORY as QUALIFICATION,WORKER,BASE
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST
from paperstudy.gpu_qualification import PYTHON,ENV
from paperstudy.budget import BudgetLedger

REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/D2-production-v1'


def study_gpu_jobs(queue):
    """The two-job cap is for this study; Slurm owns shared-node allocation.

    Do not stop, relabel or take ownership of unrelated chat jobs. Counting
    every user job caused false waiting despite free physical GPUs.
    """
    return [line for line in queue.splitlines() if 'gpu:' in line and 'frontier-D2-' in line]


def submit(case):
    spec=guard()
    if case not in spec['cases'] or case=='NAPH20-BLOCK-BOYS':raise ValueError('Only frozen32/40q GPU cases')
    qualified=load(QUALIFICATION/'collected.json');prepared=load(QUALIFICATION/'prepared.json')
    if qualified['status']!='QUALIFIED_THREE_EXACT_CIRCUITS':raise RuntimeError('CUDA adapter not qualified')
    if any(sha256(Path(p).read_bytes())!=h for p,h in prepared['source_SHA256'].items()):
        raise RuntimeError('Qualified CPU/GPU source changed')
    d=DIRECTORY/case/'GPU-dispatch';d.mkdir(parents=True,exist_ok=True)
    with (d/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (d/'job.json').exists():return load(d/'job.json')
        if (d/'submission-intent.json').exists():raise RuntimeError('Unknown Slurm submission; never duplicate')
        queue=ssh('squeue -h -u reiwatanabe -o "%i %T %j %b"',timeout=45)
        gpu_jobs=study_gpu_jobs(queue)
        save(d/'queue-before-submit.json',{'raw':queue,'other_jobs_unchanged':True,
            'study_scope_job_cap':2,'study_gpu_jobs':gpu_jobs,
            'Slurm_allocates_only_available_GRES_no_preemption_requested':True})
        if len(gpu_jobs)>=2:
            print('WAITING_FOR_USER_GPU_SLOT',case,flush=True);return None
        remote=REMOTE+'/'+case;ssh('mkdir -p '+shlex.quote(remote))
        send(WORKER,remote+'/worker.py');send(BASE,remote+'/base_kernel.py');commands=[];rows=[]
        for point in spec['points']:
            row=validate_row(case,point);cell=DIRECTORY/case/point
            config=load(cell/'config.json')
            if config['input_id']!=row['execution_input_id'] or sha256((cell/'arrays.npz').read_bytes())!=config['arrays_SHA256']:
                raise RuntimeError('D2 execution input mismatch')
            target=remote+'/'+point;ssh('mkdir -p '+shlex.quote(target))
            for name in ('arrays.npz','config.json'):send(cell/name,target+'/'+name)
            commands.append(shlex.join([PYTHON,'-u',remote+'/worker.py',target+'/config.json',target+'/arrays.npz',target+'/output']))
            rows.append(row)
        script='\n'.join(['#!/bin/bash','set -euo pipefail',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1',
            'export CUDA_PATH='+ENV+'/lib/python3.12/site-packages/nvidia/cuda_runtime',
            'export LD_LIBRARY_PATH="$CUDA_PATH/lib:'+ENV+'/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"',
            'export CUPY_ACCELERATORS=""',*commands])+'\n'
        write(d/'job.sbatch',script.encode());send(d/'job.sbatch',remote+'/job.sbatch')
        ledger=BudgetLedger(ROOT/'production-shots.sqlite')
        for row in rows:
            status=ledger.reserve(row['run_id'],'final',draw_request(row),row['shots'])
            if status!='NEW_RESERVATION':raise RuntimeError('Draw already reserved: reconcile, not resubmit')
        command=shlex.join(['sbatch','--parsable','--partition=gpu-short','--gres=gpu:a100:1',
            '--cpus-per-task=1','--mem=16G','--time=00:25:00','--job-name=frontier-D2-'+case,
            '--output='+remote+'/slurm.out','--error='+remote+'/slurm.err',remote+'/job.sbatch'])
        intent={'remote':remote,'case':case,'command':command,'new_model_shots_reserved':sum(x['shots'] for x in rows),
            'worker_SHA256':sha256(WORKER.read_bytes()),'dispatcher_SHA256':sha256(Path(__file__).read_bytes()),
            'qualification_SHA256':sha256((QUALIFICATION/'collected.json').read_bytes())}
        save(d/'submission-intent.json',intent)
        reply=ssh(command).strip().split(';')[0]
        if not reply.isdigit():raise RuntimeError('Uncertain sbatch outcome')
        job={**intent,'job_id':reply};save(d/'job.json',job);print('D2_GPU_SUBMITTED',case,reply,flush=True);return job


def collect(case):
    guard();d=DIRECTORY/case/'GPU-dispatch';job=load(d/'job.json')
    if (d/'collected.json').exists():return load(d/'collected.json')
    live=ssh('squeue -h -j '+job['job_id']+' -o "%i %T %j %R"',timeout=45)
    accounting=ssh('sacct -X -n -j '+job['job_id']+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=45)
    save(d/'observed.json',{'live':live,'accounting':accounting})
    if live.strip():print('D2_GPU_LIVE',live.strip(),flush=True);return None
    if not any(s in accounting for s in ('COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED')):return None
    ledger=BudgetLedger(ROOT/'production-shots.sqlite');checks=[]
    for point in guard()['points']:
        row=validate_row(case,point);cell=DIRECTORY/case/point;remote=job['remote']+'/'+point+'/output'
        exists=ssh('test -f '+shlex.quote(remote+'/result.json')+' && echo yes || true',timeout=45).strip()=='yes'
        if not exists:
            checks.append({'point':point,'status':'MISSING_RESULT_PRESERVED','known_model_shots':None});continue
        for name,target in [('result.json','remote-result.json'),(row['run_id']+'.npz','final-counts.npz')]:
            subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,HOST+':'+remote+'/'+name,str(cell/target)],check=True,timeout=90)
        r=load(cell/'remote-result.json')
        if r['input_id']!=row['execution_input_id'] or r['source_QPY_SHA256']!=row['exact_execution_QPY_SHA256']:
            raise RuntimeError('GPU result not bound to frozen actual circuit')
        if r['status']!='EXACT_MATERIALIZED_CIRCUIT_GPU_COMPLETED' or len(r['sampling_results'])!=1:
            raise RuntimeError('GPU result status/draw count mismatch')
        draw=r['sampling_results'][0];counts_hash=sha256((cell/'final-counts.npz').read_bytes())
        if draw['id']!=row['run_id'] or draw['counts_SHA256']!=counts_hash or draw['actual_model_shots']!=row['shots']:
            raise RuntimeError('GPU raw counts evidence mismatch')
        ledger.complete(row['run_id'],'final',draw['actual_model_shots'],counts_hash)
        save(cell/'draw-result.json',{**r,'input_id':row['execution_input_id'],'run_id':row['run_id'],
            'counts_SHA256':counts_hash,'actual_model_shots':draw['actual_model_shots'],'GPU_used':True,
            'ledger':ledger.snapshot(row['run_id'])})
        checks.append({'point':point,'status':'COMPLETED','known_model_shots':draw['actual_model_shots']})
    for name in ('slurm.out','slurm.err'):
        subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,HOST+':'+job['remote']+'/'+name,str(d/name)],check=True,timeout=90)
    result={'case':case,'job_id':job['job_id'],'checks':checks,'accounting':accounting,
        'status':'ALL_FOUR_DRAWS_COLLECTED' if all(c['status']=='COMPLETED' for c in checks) else 'ENGINEERING_FAILURE_PREFIX_PRESERVED',
        'goal_complete':False}
    save(d/'collected.json',result);print(result,flush=True);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['submit','collect']);p.add_argument('--case',required=True);a=p.parse_args()
    globals()[a.action](a.case)
