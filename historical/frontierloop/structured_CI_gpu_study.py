"""N53 frozen GPU action qualification; no CI root or source optimization."""
from pathlib import Path
from time import process_time,perf_counter
import json,shlex,subprocess,sys,tarfile
import xml.etree.ElementTree as ET
import numpy as np
from pyscf.fci import selected_ci,direct_spin1,cistring
from .paths import ROOT,OLD
from .io import load,save,write,arrays,sha256,space
from .structured_CI_arrays import make_arrays
from .structured_CI_device import dense_probes
from .energy_cache import hole_integrals
from .joint_ci_sparse import matrices
from .fe_native_readout_diagnostic import LIBRARY
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST

LOCAL=ROOT/'runs/N53-STRUCTURED-CI-CUDA-QUALIFICATION-v1'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/N53-structured-CI-CUDA-qualification-v1'


def cpu_actions(h,g,k,s):
    n=len(h);ne=(k,k)
    if len(s)<=16:
        H,S,_=matrices(h,g,ne,[(int(a),int(b)) for a in s for b in s],LIBRARY)
        return lambda x: H@x,lambda x:S@x
    links=selected_ci._all_linkstr_index((s,s),n,ne);h2=direct_spin1.absorb_h1e(h,g,n,ne,.5)
    solver=selected_ci.SelectedCI();solver.verbose=0
    ci=lambda x:selected_ci._as_SCIvector(x.reshape(len(s),len(s)),(s,s))
    return (lambda x:np.asarray(selected_ci.contract_2e(h2,ci(x),n,ne,link_index=links)).ravel(),
            lambda x:np.asarray(solver.contract_ss(ci(x),n,ne)).ravel())


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N53 namespace; reconcile, do not recreate')
    space();tests=ROOT/'qualification-N53-device-math-v1.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=4 or any(int(s.get('failures','0')) or int(s.get('errors','0')) for s in suites):raise RuntimeError('Four device-algorithm tests must pass first')
    cpu,wall=process_time(),perf_counter();cases=[];inputs=[]
    for n,k in [(3,1),(4,2),(5,3)]:
        rng=np.random.default_rng(5300+n);h=rng.normal(size=(n,n));h=(h+h.T)/2
        f=rng.normal(size=(3,n,n));f=(f+f.transpose(0,2,1))/2
        g=np.einsum('xpq,xrs->pqrs',f[:2],f[:2])-np.einsum('xpq,xrs->pqrs',f[2:],f[2:])
        s=np.asarray(cistring.make_strings(range(n),k)[::2],np.int64)
        inputs.append((f'TOY_n{n}_k{k}',h,g,k,s,0.,dict(kind='SYNTHETIC_SIGNED_ERI_NO_MOLECULE')))
    source=load(ROOT/'runs/N51-FULL-BATCH-CI-v1/protocol.json')
    with np.load(OLD/'data/physical/FE2S2-40-P2/integrals.npz',allow_pickle=False) as a:hh,gg,shift=hole_integrals(a['h1'],a['eri'],0.)
    first=next(t for t in source['tasks'] if t['method']=='C5_C_LUCJ_COMPRESSED_L1')
    small=np.sort(((1<<20)-1)^np.asarray(first['strings'],np.int64))[:12]
    inputs.append(('FE_R12',hh,gg,5,small,shift,dict(problem_id=source['problem_id'],subset_qualification_only=True)))
    for task in source['tasks']:
        s=np.sort(((1<<20)-1)^np.asarray(task['strings'],np.int64))
        inputs.append((task['method'],hh,gg,5,s,shift,dict(problem_id=source['problem_id'],N51_task_id=task['task_id'],same_N51_full_basis=True)))
    for ci,(name,h,g,k,s,core,identity) in enumerate(inputs):
        t=process_time();payload,math=make_arrays(h,g,k,s,LIBRARY);H,S=cpu_actions(h,g,k,s)
        D=len(s)**2;seed=53100+ci;X=dense_probes(D,seed)
        sparse=np.zeros(D);rng=np.random.default_rng(53200+ci);idx=rng.choice(D,size=min(10,D),replace=False);sparse[idx]=rng.normal(size=len(idx))
        sh=H(sparse);ss=S(sparse)
        t0=process_time();hx=H(X[0]);h0=process_time()-t0
        t0=process_time();hy=H(X[1]);h1=process_time()-t0
        t0=process_time();sx=S(X[0]);s0=process_time()-t0
        probes=np.sort(np.random.default_rng(53300+ci).choice(D,size=min(32,D),replace=False))
        payload.update(core_shift=np.float64(core),sparse_vector=sparse,sparse_H=sh,sparse_S2=ss,
             sparse_total=sh+.1*ss+core*sparse,probe_indices=probes,
             dense_H_components=hx[probes],dense_S2_components=sx[probes])
        if D<=4225:payload.update(dense_H_full=hx,dense_S2_full=sx)
        file='inputs/'+name+'.npz';blob=arrays(**payload);write(LOCAL/file,blob)
        oracle=dict(x_H_x=float(X[0]@hx),x_H_y=float(X[0]@hy),y_H_x=float(X[1]@hx),
              H_x_norm=float(np.linalg.norm(hx)),S_x_norm=float(np.linalg.norm(sx)),x_S_x=float(X[0]@sx))
        cases.append(dict(id=name,file=file,SHA256=sha256(blob),R=len(s),D=D,dense_seed=seed,
             dense_input_SHA256=sha256(X.astype('<f8').tobytes()),dense_oracle=oracle,source_problem=identity,
             CPU_reference=dict(dense_H_CPU_seconds=[h0,h1],dense_S2_CPU_seconds=s0,
                                whole_case_preparation_CPU_seconds=process_time()-t,math=math)))
        print('N53_PREPARED',name,'R',len(s),'D',D,'bytes',len(blob),flush=True)
        if process_time()-cpu>120 or perf_counter()-wall>180:raise RuntimeError('Finite N53 CPU preparation cap exceeded; preserve partials')
    for src,name in [(ROOT/'frontierloop/structured_CI_gpu_worker.py','worker.py'),(ROOT/'frontierloop/structured_CI_device.py','structured_CI_device.py')]:write(LOCAL/name,src.read_bytes())
    save(LOCAL/'protocol.json',dict(version='N53_EXACT_STRUCTURED_CI_CUDA_ACTION_ONLY',cases=cases,
         input_SHA256={**{c['file']:c['SHA256'] for c in cases},**{name:sha256((LOCAL/name).read_bytes()) for name in ['worker.py','structured_CI_device.py']}},
         expected_cupy='13.6.0',tile=8,action_tolerance=1e-8,memory_pool_cap_bytes=8*2**30,
         body_wall_cap_seconds=600,new_CI_roots=0,new_model_samples=0,new_QPU_shots=0,
         prepare_CPU_seconds=process_time()-cpu,prepare_wall_seconds=perf_counter()-wall,
         original_N51_protocol_SHA256=sha256((ROOT/'runs/N51-FULL-BATCH-CI-v1/protocol.json').read_bytes()),
         no_factor_truncation=True,no_source_or_basis_changes=True,
         original_scientific_outputs_unchanged=True,goal_complete=False))
    print('N53_ALL8_INPUTS_FROZEN',flush=True)


def verified():
    p=load(LOCAL/'protocol.json')
    for name,h in p['input_SHA256'].items():
        if sha256((LOCAL/name).read_bytes())!=h:raise ValueError('Frozen N53 input changed')
    return p


def submit():
    from paperstudy.gpu_qualification import PYTHON,ENV
    p=verified();space()
    if (LOCAL/'staging-intent.json').exists():raise RuntimeError('Existing dispatch intent; do not retry sbatch')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('Remote namespace exists')
    save(LOCAL/'pre-dispatch-queue.json',dict(raw=ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%R"',timeout=35),other_jobs_not_modified=True))
    script='\n'.join(['#!/bin/bash','set -euo pipefail',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1',
        'export CUDA_PATH='+ENV+'/lib/python3.12/site-packages/nvidia/cuda_runtime',
        'export LD_LIBRARY_PATH="$CUDA_PATH/lib:'+ENV+'/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"',
        'export CUPY_ACCELERATORS=""','export CUPY_CACHE_DIR='+REMOTE+'/cuda-cache',
        shlex.join([PYTHON,'-u',REMOTE+'/worker.py',REMOTE])])+'\n'
    write(LOCAL/'job.sbatch',script.encode())
    files=list(p['input_SHA256'])+['protocol.json','job.sbatch'];bundle=LOCAL/'staging.tar.gz'
    with tarfile.open(bundle,'w:gz') as tar:
        for name in files:tar.add(LOCAL/name,arcname=name,recursive=False)
    if bundle.stat().st_size>64*2**20:raise ValueError('Bounded upload cap exceeded')
    save(LOCAL/'staging-intent.json',dict(remote=REMOTE,archive_SHA256=sha256(bundle.read_bytes()),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    ssh('mkdir '+shlex.quote(REMOTE),timeout=35);send(bundle,REMOTE+'/staging.tar.gz')
    script="import pathlib,tarfile,hashlib;r=pathlib.Path("+repr(REMOTE)+");f=r/'staging.tar.gz';assert hashlib.sha256(f.read_bytes()).hexdigest()=="+repr(sha256(bundle.read_bytes()))+";t=tarfile.open(f);assert all(m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts for m in t.getmembers());t.extractall(r);print('N53_STAGED')"
    print(ssh(shlex.join([PYTHON,'-c',script]),timeout=35),flush=True)
    command=shlex.join(['sbatch','--parsable','--partition=gpu-short','--gres=gpu:a100:1','--cpus-per-task=1','--mem=8G','--time=00:12:00',
          '--job-name=frontier-N53-CI-CUDA-qual','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err',REMOTE+'/job.sbatch'])
    save(LOCAL/'submission-intent.json',dict(command=command))
    job=ssh(command,timeout=35).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Unknown sbatch result; reconcile exact name rather than resubmit')
    save(LOCAL/'job.json',dict(job_id=job,remote=REMOTE,GPU_allocations=1));print('N53_A100_SUBMITTED',job,flush=True)


def status():
    job=load(LOCAL/'job.json')['job_id']
    result=ssh('squeue -h -j '+job+' -o "%i|%T|%j|%R"; sacct -X -n -j '+job+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=40)
    save(LOCAL/'last-status.json',dict(job_id=job,raw=result));print(result,flush=True)


def collect():
    job=load(LOCAL/'job.json')['job_id'];target=LOCAL/'collected'
    account=ssh('sacct -X -n -j '+job+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=35)
    state=next((line.split('|')[1] for line in account.splitlines() if line.split('|')[0]==job),None)
    if state not in ['COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED']:raise RuntimeError('Job live or unresolved; no restart')
    if target.exists():raise RuntimeError('Existing partial collection; reconcile without recopy')
    target.mkdir();subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/out',str(target)],check=True,timeout=120)
    save(LOCAL/'collection.json',dict(accounting=account,remote_evidence_deleted=False,
           files={str(f.relative_to(target)):sha256(f.read_bytes()) for f in target.rglob('*') if f.is_file()}))
    print('N53_COLLECTED',state)


if __name__=='__main__':{'prepare':prepare,'submit':submit,'status':status,'collect':collect}[sys.argv[1]]()
