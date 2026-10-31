"""Two bounded A100 lanes; exact byte-identical qualified N17 workers."""
from pathlib import Path
from datetime import datetime, timezone
import json, shlex, subprocess, sys, tarfile
from .io import load, save, write, sha256, digest
from .fixed_confirmation_execution import LOCAL, bound, capacity
from .prefix_performance_sampling import LOCAL as N17, decode
from benchmark.gpu_remote import ssh, send, OPTIONS, HOST
from paperstudy.gpu_qualification import PYTHON

REMOTE='/share/reiwatanabe/raacs-fixed-confirmation-20261027-v1'
GPU=LOCAL/'gpu'

def prepare():
    science,b=bound();target=GPU/'binding.json'
    if target.exists():raise RuntimeError('Existing GPU binding; do not replace')
    jobs=[]
    for cell in science['cells']:
        req=[r for r in b['requests'] if r['case']==cell['case'] and r['method']==cell['method'] and r['backend']=='A100']
        if not req:continue
        row=cell['source_input'];circuit=row['circuit'];cfg=load(row['config_file']);cfg['sampling_requests']=[dict(id=r['id'],seed=r['seed'],shots=r['shots'],replicate_label=r['replicate']) for r in req]
        cfg['physical_problem_id']=cell['problem_id'];cfg['source_QPY_SHA256']=circuit['original_QPY_SHA256']
        cfg['worker_SHA256']=sha256((N17/'worker.py').read_bytes());cfg['base_kernel_SHA256']=sha256((N17/'base_kernel.py').read_bytes())
        cfg.pop('input_id',None);cfg['input_id']=digest(cfg)
        d=GPU/'inputs'/cfg['input_id'];save(d/'config.json',cfg);write(d/'arrays.npz',Path(row['arrays_file']).read_bytes())
        jobs.append(dict(case=cell['case'],method=cell['method'],input_id=cfg['input_id'],config_file=str(d/'config.json'),config_SHA256=sha256((d/'config.json').read_bytes()),arrays_SHA256=sha256((d/'arrays.npz').read_bytes()),
            estimated_work=int(circuit['sector_dimension'])*len(cfg['ops']),requests=req,decoder=circuit['decoder'],norb=cfg['norb'],nelec=cfg['nelec']))
    lanes=[[],[]];weights=[0,0]
    for job in sorted(jobs,key=lambda j:(-j['estimated_work'],j['input_id'])):
        lane=min(range(2),key=lambda i:(weights[i],i));lanes[lane].append(job);weights[lane]+=job['estimated_work']
    assert len(jobs)==9 and sum(len(j['requests']) for j in jobs)==18
    for name in ['worker.py','base_kernel.py']:write(GPU/name,(N17/name).read_bytes())
    env=PYTHON.rsplit('/bin/python',1)[0]
    for i,lane in enumerate(lanes):
        lines=['#!/bin/bash','set -euo pipefail','export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1',
            'export CUDA_PATH='+env+'/lib/python3.12/site-packages/nvidia/cuda_runtime',
            'export LD_LIBRARY_PATH="$CUDA_PATH/lib:'+env+'/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"','export CUPY_ACCELERATORS=""']
        for j in lane:
            rd=REMOTE+'/inputs/'+j['input_id'];lines.append(shlex.join([PYTHON,'-u',REMOTE+'/worker.py',rd+'/config.json',rd+'/arrays.npz',REMOTE+'/outputs/'+j['input_id']]))
        write(GPU/f'lane{i}.sbatch', ('\n'.join(lines)+'\n').encode())
    files=[Path(__file__),GPU/'worker.py',GPU/'base_kernel.py',GPU/'lane0.sbatch',GPU/'lane1.sbatch']
    save(target,dict(execution_binding_SHA256=sha256((LOCAL/'execution-binding.json').read_bytes()),code_and_scripts={str(f):sha256(f.read_bytes()) for f in files},lanes=lanes,remote=REMOTE,max_jobs=2,wall_seconds_per_job=900))
    print('GPU_BOUND9_INPUTS18_DRAWS_TWO_LANES',weights,flush=True)

def verified():
    bound();p=load(GPU/'binding.json')
    if p['execution_binding_SHA256']!=sha256((LOCAL/'execution-binding.json').read_bytes()):raise RuntimeError('Execution binding drift')
    for f,h in p['code_and_scripts'].items():
        if sha256(Path(f).read_bytes())!=h:raise RuntimeError('GPU code drift')
    for lane in p['lanes']:
        for j in lane:
            d=Path(j['config_file']).parent
            for name,key in [('config.json','config_SHA256'),('arrays.npz','arrays_SHA256')]:
                if sha256((d/name).read_bytes())!=j[key]:raise RuntimeError('GPU input drift')
    return p

def submit():
    p=verified();capacity()
    if (GPU/'stage-intent.json').exists():raise RuntimeError('Existing remote staging intent; reconcile, no duplicate submission')
    paths={str(N17/'worker.py'):sha256((N17/'worker.py').read_bytes())}
    oldremote='/share/reiwatanabe/raacs-frontier-loop-20261025/N17-independent-prefix-v1'
    program="import json,pathlib,hashlib,subprocess; paths="+repr({oldremote+'/'+n:sha256((GPU/n).read_bytes()) for n in ['worker.py','base_kernel.py']})+"; print(json.dumps({'files':{f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest() for f in paths},'queue':subprocess.check_output(['squeue','-h','-u','reiwatanabe','-o','%i|%T|%j|%R'],text=True),'expected':paths}))"
    observation=json.loads(ssh(shlex.join([PYTHON,'-c',program]),timeout=40));assert observation['files']==observation['expected']
    save(GPU/'pre-dispatch-remote-observation.json',dict(observed_UTC=datetime.now(timezone.utc).isoformat(),**observation,other_jobs_changed=False))
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('Existing remote namespace, reconcile')
    save(GPU/'stage-intent.json',dict(remote=REMOTE,bindings_SHA256=sha256((GPU/'binding.json').read_bytes())))
    ssh('mkdir '+shlex.quote(REMOTE),timeout=35)
    payload=GPU/'inputs-transfer.tar'
    with tarfile.open(payload,'w') as tf:
        for f in [GPU/'worker.py',GPU/'base_kernel.py',GPU/'lane0.sbatch',GPU/'lane1.sbatch',*sorted((GPU/'inputs').rglob('*'))]:
            if f.is_file():tf.add(f,arcname=str(f.relative_to(GPU)),recursive=False)
    send(payload,REMOTE+'/inputs-transfer.tar');ssh('tar -xf '+shlex.quote(REMOTE+'/inputs-transfer.tar')+' -C '+shlex.quote(REMOTE),timeout=40)
    verifyprogram="import pathlib,hashlib,json; r=pathlib.Path("+repr(REMOTE)+"); print(json.dumps({str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in r.rglob('*') if f.is_file() and f.name!='inputs-transfer.tar'}))"
    remotehash=json.loads(ssh(shlex.join([PYTHON,'-c',verifyprogram]),timeout=40))
    expected={str(f.relative_to(GPU)):sha256(f.read_bytes()) for f in [GPU/'worker.py',GPU/'base_kernel.py',GPU/'lane0.sbatch',GPU/'lane1.sbatch',*sorted((GPU/'inputs').rglob('*'))] if f.is_file()}
    assert remotehash==expected;save(GPU/'remote-upload-verification.json',dict(files=remotehash,all_match=True))
    for i in range(2):
        command=shlex.join(['sbatch','--parsable','--partition=gpu-short','--gres=gpu:a100:1','--cpus-per-task=1','--mem=8G','--time=00:15:00',
            '--job-name=frontier-fixed-confirm-lane'+str(i),'--output='+REMOTE+f'/lane{i}.out','--error='+REMOTE+f'/lane{i}.err',REMOTE+f'/lane{i}.sbatch'])
        save(GPU/f'lane{i}-submission-intent.json',dict(command=command))
        reply=ssh(command,timeout=40).strip().split(';')[0]
        if not reply.isdigit():raise RuntimeError('Uncertain sbatch; reconcile exact name, no retry')
        save(GPU/f'lane{i}-job.json',dict(job_id=reply,remote=REMOTE,lane=i));print('A100_SUBMITTED',i,reply,flush=True)

def collect():
    p=verified();capacity();observations=[]
    for i in range(2):
        job=load(GPU/f'lane{i}-job.json');jid=job['job_id']
        live=ssh('squeue -h -j '+jid+' -o "%i|%T|%j|%R"',timeout=35).strip()
        account=ssh('sacct -X -n -j '+jid+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=35).strip()
        observations.append(dict(job_id=jid,lane=i,live=live,accounting=account));print(jid,live or account,flush=True)
    save(GPU/'latest-job-observation.json',dict(observed_UTC=datetime.now(timezone.utc).isoformat(),jobs=observations))
    if any(j['live'] for j in observations):return
    program="import pathlib,tarfile; r=pathlib.Path("+repr(REMOTE)+"); files=[f for f in r.rglob('*') if f.is_file() and ('outputs' in f.relative_to(r).parts or f.name in ['lane0.out','lane0.err','lane1.out','lane1.err'])]; assert sum(f.stat().st_size for f in files)<16*2**20; t=tarfile.open(r/'results.tar.gz','w:gz'); [t.add(f,arcname=str(f.relative_to(r)),recursive=False) for f in files]; t.close()"
    ssh(shlex.join([PYTHON,'-c',program]),timeout=40)
    payload=GPU/'results.tar.gz'
    subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/results.tar.gz',str(payload)],check=True,timeout=45)
    capacity(payload.stat().st_size)
    with tarfile.open(payload) as tf:
        for member in tf.getmembers():
            if not member.isfile() or '..' in Path(member.name).parts or Path(member.name).is_absolute():raise RuntimeError('Invalid remote archive')
            write(GPU/'collected'/member.name,tf.extractfile(member).read())
    records=[]
    for lane in p['lanes']:
        for j in lane:
            d=GPU/'collected/outputs'/j['input_id'];resultfile=d/'result.json'
            if not resultfile.exists():
                records.append(dict(input_id=j['input_id'],status='MISSING_GPU_RESULT_PRESERVED',started=(d/'started.json').exists()));continue
            result=load(resultfile);assert result['input_id']==j['input_id'] and result['physical_problem_id']==j['requests'][0]['problem_id']
            cfg=load(j['config_file']);assert result['source_QPY_SHA256']==cfg['source_QPY_SHA256'] and abs(result['norm_squared']-1)<1e-8
            assert len(result['sampling_results'])==2
            for req in j['requests']:
                r=next(r for r in result['sampling_results'] if r['id']==req['id']);file=d/(req['id']+'.npz')
                assert r['seed']==req['seed'] and r['shots']==r['actual_model_shots']==req['shots'] and sha256(file.read_bytes())==r['counts_SHA256']
                import numpy as np
                with np.load(file,allow_pickle=False) as a:
                    assert a['counts'].sum()==req['shots'];ids=decode(a['blocked_ids'],j['decoder'],j['norb'],j['nelec'])
                    from .io import arrays
                    blob=arrays(blocked_ids=ids,counts=a['counts'])
                out=LOCAL/'member-draws'/(req['id']+'.npz')
                if out.exists():raise RuntimeError('Already collected request; inspect, no overwrite')
                write(out,blob);rec=dict(request=req,status='COMPLETED',counts_file=str(out),counts_SHA256=sha256(blob),actual_new_model_samples=req['shots'],new_QPU_shots=0,GPU_result_file=str(resultfile))
                save(out.with_suffix('.json'),rec);records.append(rec)
    save(GPU/'terminal.json',dict(status='BOUNDED_GPU_LANES_TERMINAL',records=records,new_QPU_shots=0,actual_new_model_samples=sum(r.get('actual_new_model_samples',0) for r in records)))
    print('GPU_COLLECTED',len(records),flush=True)

if __name__=='__main__':{'prepare':prepare,'submit':submit,'collect':collect}[sys.argv[1]]()
