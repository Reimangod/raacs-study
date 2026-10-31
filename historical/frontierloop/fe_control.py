"""New independent Fe40 LUCJ readouts using the already-qualified CPU kernel.

This uses a CPU-only high-RAM Slurm allocation, not unqualified GPU results.
Old remote data/runtime are read-only; one new namespace and one job only.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
from time import time,process_time,perf_counter
import numpy as np
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST
from paperstudy.budget import BudgetLedger
from paperstudy.ci import solve
from paperstudy.gpu_qualification import REMOTE as OLD_REMOTE,PYTHON
from raacsnext.run import physical
from .paths import ROOT,OLD
from .io import load,save,write,sha256,digest,space

REMOTE='/share/reiwatanabe/raacs-frontier-research-loop-20261025/Fe40-C1-v1'


def submit():
    root=ROOT/'Fe40-C1-v1';root.mkdir(parents=True,exist_ok=True)
    if(root/'job.json').exists():return load(root/'job.json')
    if(root/'submission-intent.json').exists():raise RuntimeError('Uncertain submission: reconcile exact job, never resend')
    m,_,_=physical('FE2S2-40-P2')
    construction=load(OLD/'prepared/FE2S2-40-P2/C_LUCJ_COMPRESSED_L1-L1/construction.json')
    operator=OLD/'prepared/FE2S2-40-P2/C_LUCJ_COMPRESSED_L1-L1/operator.npz'
    worker=OLD/'paperstudy/remote_cpu_lucj_worker.py'
    old=load(OLD/'remote-CPU-LUCJ-v1/100000/C_LUCJ_COMPRESSED_L1/request.json')
    if sha256(worker.read_bytes())!=old['worker_sha256']or sha256(operator.read_bytes())!=construction['operator_sha256']:
        raise RuntimeError('Qualified molecular executor/operator changed')
    requests=[{'run_id':digest(['new-frontier-C1-Fe40',m['problem_id'],s,100000,construction['operator_sha256']]),
        'seed':s,'shots':100000}for s in(71001,71011,71019)]
    config={'case_id':'FE2S2-40-P2','problem_id':m['problem_id'],'method_id':'LUCJ_COMPRESSED_L1',
        'norb':20,'nelec':[15,15],'operator_sha256':construction['operator_sha256'],
        'CPU_parity_reference':old['CPU_parity_reference'],'worker_sha256':sha256(worker.read_bytes()),
        'runtime':old['runtime'],'requests':requests,'exploratory_control_not_paper_exact_reproduction':True}
    config['config_id']=digest(config);save(root/'request.json',config);save(root/'construction.json',construction)
    ssh('mkdir -p '+shlex.quote(REMOTE),timeout=30)
    send(worker,REMOTE+'/worker.py');send(operator,REMOTE+'/operator.npz');send(root/'request.json',REMOTE+'/request.json')
    script='\n'.join(['#!/bin/bash','set -euo pipefail',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1',
        'export PYTHONPATH='+OLD_REMOTE+'/cpu-runtime-v1/packages',
        shlex.join([PYTHON,REMOTE+'/worker.py',REMOTE+'/request.json',REMOTE+'/operator.npz',REMOTE+'/output'])])+'\n'
    write(root/'job.sbatch',script.encode());send(root/'job.sbatch',REMOTE+'/job.sbatch')
    ledger=BudgetLedger(ROOT/'production-shots.sqlite')
    for r in requests:
        ledger.register(r['run_id'],r['shots'])
        request={'config_id':config['config_id'],'operator_SHA256':config['operator_sha256'],'shots':r['shots'],'seed':r['seed']}
        if ledger.reserve(r['run_id'],'independent-final',request,r['shots'])!='NEW_RESERVATION':
            raise RuntimeError('Remote draw already reserved; no duplicate dispatch')
    command=shlex.join(['sbatch','--parsable','--partition=gpu-short','--cpus-per-task=1','--mem=24G',
        '--time=00:10:00','--job-name=raacs-next-Fe40-C1-20261025',
        '--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err',REMOTE+'/job.sbatch'])
    intent={'remote':REMOTE,'config_id':config['config_id'],'command':command,'submitted_unix':time(),
        'GPU_requested':False,'new_reserved_model_shots':300000}
    save(root/'submission-intent.json',intent)
    answer=ssh(command,timeout=30).strip().split(';')[0]
    if not answer.isdigit():raise RuntimeError('Uncertain submission reply '+answer)
    result={**intent,'job_id':answer};save(root/'job.json',result);print(result,flush=True);return result


def collect():
    root=ROOT/'Fe40-C1-v1';job=load(root/'job.json')
    state=ssh('squeue -h -j '+job['job_id']+' -o "%i %T %R"',timeout=30).strip()
    if state:print('LIVE',state,flush=True);return
    accounting=ssh('sacct -n -j '+job['job_id']+' --format=JobID,State,Elapsed,ExitCode,TotalCPU',timeout=30)
    save(root/'terminal-accounting.json',{'accounting':accounting,'GPU_requested':False})
    r=__import__('json').loads(ssh('cat '+REMOTE+'/output/result.json',timeout=30))
    config=load(root/'request.json')
    if r['config_id']!=config['config_id']or r['status']!='EXACT_REMOTE_CPU_LUCJ_COMPLETED':raise RuntimeError('Wrong remote result')
    save(root/'remote-result.json',r)
    m,h,g=physical('FE2S2-40-P2');construction=load(root/'construction.json')
    ledger=BudgetLedger(ROOT/'production-shots.sqlite')
    for result in r['sampling_results']:
        directory=root/str(result['seed']);directory.mkdir(parents=True,exist_ok=True)
        path=directory/'final-counts.npz';space()
        if not path.exists():
            subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/output/'+result['run_id']+'.npz',str(path)],check=True,timeout=30)
        if sha256(path.read_bytes())!=result['counts_sha256']:raise RuntimeError('Count transfer mismatch')
        with np.load(path,allow_pickle=False)as a:ids,counts=a['blocked_ids'],a['counts']
        if int(counts.sum())!=100000:raise RuntimeError('Remote shot totals mismatch')
        ledger.complete(result['run_id'],'independent-final',100000,result['counts_sha256'])
        if(directory/'result.json').exists():continue
        cis=[]
        for cap in(8,16,32,64):
            cpu,t=process_time(),perf_counter();c=solve(h,g,m['core_energy'],(15,15),ids,counts,cap)
            c.update(CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-t,new_model_shots=0,
                source_counts_SHA256=result['counts_sha256']);cis.append(c)
        summary={'case_id':'FE2S2-40-P2','variant':'LUCJ_COMPRESSED_L1','seed':result['seed'],
            'budget':100000,'run_id':result['run_id'],'problem_id':m['problem_id'],'final_CI':cis,'ledger':ledger.snapshot(result['run_id']),
            'construction':construction,'shared_remote_preparation':r['preparation'],'sampling_cost':result,
            'sampling_resources':construction['resources']['Z_sampling_equivalent'],
            'coefficient_slots':construction['coefficient_slots'],'actual_QPU_shots':0,'GPU_used':False,
            'independent_fixed_operator_readout_not_optimizer_replication':True,'status':'COMPLETED'}
        save(directory/'result.json',summary)
        print('CONTROL',result['seed'],'E64',cis[-1]['energy_Ha'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['submit','collect']);a=p.parse_args()
    submit()if a.action=='submit'else collect()
