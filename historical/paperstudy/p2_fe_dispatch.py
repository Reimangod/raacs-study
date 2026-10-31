"""Fe P2 author-amplitude anchors in a separate bounded CPU namespace."""
import json,shlex,subprocess
from pathlib import Path
from time import sleep
import ffsim,numpy as np
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST
from .author_compressed import load_author
from .common_execution import load_physical
from .gpu_qualification import REMOTE,PYTHON
from .input_adapter import array_bytes
from .lucj import amplitudes_from_saved,interaction_pairs,make_circuit,circuit_views
from .io import ROOT,save_json,atomic_bytes,digest,sha256


def submit(kind):
    case='FE2S2-40-P2';root=ROOT/'paper-anchors/P2-Fe-remote-v1'/kind;root.mkdir(parents=True,exist_ok=True)
    if(root/'job.json').exists():return json.loads((root/'job.json').read_text())
    if(root/'submission-intent.json').exists():raise RuntimeError('Uncertain P2 Fe submission')
    runtime=json.loads((ROOT/'hardware-runtime-v1/qualification.json').read_text())
    if runtime['version']!='0.14.0':raise RuntimeError('Unqualified current SDK')
    m,_,_=load_physical(case);n=m['norb'];ne=(m['nalpha'],m['nbeta'])
    if kind=='naive':
        t1,t2,source=amplitudes_from_saved(ROOT/'data/initialization'/case/'amplitudes.npz',n,ne[0])
        op,_,_=load_author().from_t_amplitudes_compressed(t2,t1=t1,n_reps=1,interaction_pairs=interaction_pairs(n),optimize=False)
    else:
        source_dir=ROOT/'qualification'/(case+'-author-compressed')
        source=json.loads((source_dir/'result.json').read_text())
        if source['status']!='AUTHOR_INITIALIZER_RETURNED':raise RuntimeError('Source initializer incomplete')
        with np.load(source_dir/'operator.npz',allow_pickle=False)as z:op=ffsim.UCJOpSpinBalanced(z['diag_coulomb_mats'],z['orbital_rotations'],z['final_orbital_rotation'])
    atomic_bytes(root/'operator.npz',array_bytes({'diag_coulomb_mats':op.diag_coulomb_mats,'orbital_rotations':op.orbital_rotations,'final_orbital_rotation':op.final_orbital_rotation}))
    _,_,views=circuit_views(make_circuit(op,n,ne));save_json(root/'construction.json',{'source':source,'resource_views':views,
        'coefficient_slots':len(op.to_parameters(interaction_pairs=interaction_pairs(n))),'fresh_C_lane_amplitudes_used':False})
    remote=REMOTE+'/P2-Fe-source-v1/'+kind;ssh('mkdir -p '+remote)
    worker=Path(__file__).with_name('p2_fe_worker.py');physical=ROOT/'data/physical'/case/'integrals.npz'
    for path,name in((root/'operator.npz','operator.npz'),(physical,'integrals.npz'),(worker,'worker.py')):send(path,remote+'/'+name)
    config={'case_id':case,'kind':kind,'problem_id':m['problem_id'],'norb':n,'nelec':list(ne),'core_energy':m['core_energy'],
        'model_shots':100000,'seed':0,'samples_per_batch':4000,'num_batches':10,'max_dim_per_spin':4000,'max_iterations':1,
        'source_initializer':source,'worker_sha256':sha256(worker.read_bytes()),'historical_Dice_solver_reproduced':False,
        'input_sha256':{'operator.npz':sha256((root/'operator.npz').read_bytes()),'integrals.npz':sha256(physical.read_bytes())}}
    config['config_id']=digest(config);save_json(root/'request.json',config);send(root/'request.json',remote+'/request.json')
    script='\n'.join(['#!/bin/bash','set -euo pipefail','export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1',
        'export PYTHONPATH='+REMOTE+'/H-runtime-v1/packages:'+REMOTE+'/cpu-runtime-v1/packages',
        shlex.join([PYTHON,'-u',remote+'/worker.py',remote+'/request.json'])])+'\n'
    atomic_bytes(root/'job.sbatch',script.encode());send(root/'job.sbatch',remote+'/job.sbatch')
    command=shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=24G','--time=00:55:00',
        '--job-name=paper-P2-Fe-'+kind,'--output='+remote+'/slurm.out','--error='+remote+'/slurm.err',remote+'/job.sbatch'])
    intent={'remote':remote,'config_id':config['config_id'],'kind':kind,'command':command};save_json(root/'submission-intent.json',intent)
    reply=ssh(command).strip().split(';')[0]
    if not reply.isdigit():raise RuntimeError('Uncertain P2 Fe sbatch result')
    job={**intent,'job_id':reply};save_json(root/'job.json',job);print('P2_FE_SUBMITTED',kind,reply,flush=True);return job


def collect(kind,job):
    root=ROOT/'paper-anchors/P2-Fe-remote-v1'/kind
    if(root/'collected.json').exists():return True
    state=ssh('sacct -X -n -j '+job['job_id']+' --format=JobID,State,Elapsed,ExitCode -P')
    if not any(x in state for x in('COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED')):return False
    names=ssh('find '+job['remote']+" -maxdepth 1 -type f -name '*.json' -printf '%f\\n'").splitlines()
    names += ['slurm.out','slurm.err']
    if ssh('test -f '+job['remote']+'/raw-counts.npz && echo yes || true').strip():names+=['raw-counts.npz']
    out=root/'output';out.mkdir(exist_ok=True)
    for name in names:
        if '/'in name or name.startswith('.'):raise ValueError('Invalid result path')
        subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,HOST+':'+job['remote']+'/'+name,str(out/name)],check=True,timeout=120)
    result=json.loads((out/'result.json').read_text())if(out/'result.json').exists()else None
    if result and result['configuration']['config_id']!=job['config_id']:raise RuntimeError('P2 Fe config mismatch')
    save_json(root/'collected.json',{'state':state,'result_status':result['status']if result else'NO_FINAL_RESULT_PREFIX_PRESERVED',
        'reports_sha256':{p.name:sha256(p.read_bytes())for p in out.iterdir()},'whole_experiment_complete':False})
    print('P2_FE_COLLECTED',kind,flush=True);return True


if __name__=='__main__':
    while True:
        done=sum(collect(k,submit(k))for k in('naive','compressed'))
        if done==2:break
        sleep(45)
