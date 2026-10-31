"""One fixed-source, post-start approximation diagnostic; not a new sampler."""
import json,shlex,subprocess
from pathlib import Path
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST
from .gpu_qualification import REMOTE,PYTHON
from .io import ROOT,save_json,atomic_bytes,sha256,digest


def run():
    root=ROOT/'TN-MPS-diagnostic-v1';root.mkdir(exist_ok=True)
    if(root/'collected.json').exists():return json.loads((root/'collected.json').read_text())
    if not(root/'job.json').exists():
        if(root/'submission-intent.json').exists():raise RuntimeError('Reconcile unknown diagnostic submission')
        source=ROOT/'TN-controls-v1/NAPH20-P1/optimization-result.json';v=json.loads(source.read_text())
        op=source.with_name('optimized-operator.npz');worker=Path(__file__).with_name('mps_state_diagnostic_worker.py')
        if sha256(op.read_bytes())!=v['operator_sha256']:raise RuntimeError('Diagnostic source operator mismatch')
        request={'version':1,'case_id':'NAPH20-P1','problem_id':v['problem_id'],'norb':10,'nelec':[5,5],
            'operator_sha256':v['operator_sha256'],'bond_grid':[50,128,256,512],'cutoff':1e-10,'per_chi_seconds':120,
            'full_state_maximum_qubits':20,'fixed_source_not_optimized_or_selected_by_diagnostic':True,
            'post_start_sensitivity_not_confirmatory':True,'source_result_sha256':sha256(source.read_bytes()),
            'worker_sha256':sha256(worker.read_bytes()),'new_model_shots':0,'new_optimizer_calls':0}
        request['request_id']=digest(request);save_json(root/'request.json',request)
        remote=REMOTE+'/TN-MPS-diagnostic-v1';ssh('mkdir -p '+remote)
        for p,name in((worker,'worker.py'),(op,'operator.npz'),(root/'request.json','request.json')):send(p,remote+'/'+name)
        script='\n'.join(['#!/bin/bash','set -euo pipefail',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1',
            'export PYTHONPATH='+REMOTE+'/tn-runtime-v1/packages:'+REMOTE+'/cpu-runtime-v1/packages',
            shlex.join([PYTHON,'-u',remote+'/worker.py',remote+'/request.json'])])+'\n'
        atomic_bytes(root/'job.sbatch',script.encode());send(root/'job.sbatch',remote+'/job.sbatch')
        command=shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=4G',
            '--time=00:10:00','--job-name=paper-MPS-fixed20q','--output='+remote+'/slurm.out','--error='+remote+'/slurm.err',remote+'/job.sbatch'])
        intent={'remote':remote,'command':command,'request_id':request['request_id']}
        save_json(root/'submission-intent.json',intent);reply=ssh(command).strip().split(';')[0]
        if not reply.isdigit():raise RuntimeError('Unknown diagnostic sbatch response')
        save_json(root/'job.json',{**intent,'job_id':reply});print('MPS_DIAGNOSTIC_SUBMITTED',reply,flush=True);return
    job=json.loads((root/'job.json').read_text());state=ssh('sacct -X -n -j '+job['job_id']+' --format=JobID,State,Elapsed,ExitCode -P')
    if not any(s in state for s in('COMPLETED','TIMEOUT','FAILED','CANCELLED','OUT_OF_MEMORY')):
        print('MPS_DIAGNOSTIC_LIVE',state.strip(),flush=True);return
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,HOST+':'+job['remote']+'/.',str(root/'output')],check=True,timeout=120)
    result=root/'output/result.json'
    if result.exists()and json.loads(result.read_text())['request']['request_id']!=job['request_id']:
        raise RuntimeError('Diagnostic result belongs to different source')
    record={'actual_Slurm_accounting':state,'result_available':result.exists(),'other_cases_certified':False,
        'files_sha256':{str(p.relative_to(root/'output')):sha256(p.read_bytes())for p in(root/'output').rglob('*')if p.is_file()}}
    save_json(root/'collected.json',record);print(json.dumps(record),flush=True);return record


if __name__=='__main__':run()
