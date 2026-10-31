"""N54 actual hybrid SCI roots, two bounded one-A100 jobs, no new sampling."""
from pathlib import Path
from time import process_time,perf_counter
import importlib.metadata as metadata
import shlex,subprocess,sys,tarfile
import xml.etree.ElementTree as ET
import numpy as np
from .paths import ROOT
from .io import load,save,write,arrays,sha256,space
from .hybrid_CI_solver import exact_diagonal
from .joint_ci_sparse import matrices
from .fe_native_readout_diagnostic import LIBRARY
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST

LOCAL=ROOT/'runs/N54-HYBRID-GPU-FULL-BATCH-CI-v1'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/N54-hybrid-GPU-full-batch-CI-v1'
JOB_TAG='N54'
N53=ROOT/'runs/N53-STRUCTURED-CI-CUDA-QUALIFICATION-v1'
N51=ROOT/'runs/N51-FULL-BATCH-CI-v1'


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N54 namespace; reconcile, do not recreate')
    space();cpu,wall=process_time(),perf_counter()
    audit=load(ROOT/'analysis/N53-CUDA-action-successor-audit.json')
    assert audit['status']=='N53_ALL8_ACTION_CASES_VERIFIED_WITH_PRESERVED_V1_INCIDENT'
    tests=ROOT/'qualification-N54-hybrid-solver-v1.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
    assert sum(int(x.get('tests','0')) for x in suites)==5
    assert not any(int(x.get('failures','0')) or int(x.get('errors','0')) for x in suites)
    p53=load(N53/'protocol.json');p51=load(N51/'protocol.json');qual=[];production={};hashes={}
    chosen=p53['cases'][:3]+[p53['cases'][4]]+p53['cases'][6:]
    keys=['n','k','strings','one_spin_H','factors','factor_signs','pq','rows','cols','signs','occupancies']
    for i,c in enumerate(chosen):
        source=N53/c['file'];assert sha256(source.read_bytes())==c['SHA256']
        with np.load(source,allow_pickle=False) as a:data={k:a[k] for k in keys};shift=float(a['core_shift'])
        molecular=i>=3;n=int(data['n']);k=int(data['k']);D=c['D']
        rng=np.random.default_rng(20261025)
        for _ in range(6):rng.normal(size=D)
        initial=rng.normal(size=D);initial/=np.linalg.norm(initial)
        data.update(initial_vector=initial,exact_diagonal=exact_diagonal(data),
            core_shift=np.float64(shift+(p51['core'] if molecular else 0.)),
            electron_strings=np.sort(((1<<n)-1)^data['strings']) if molecular else data['strings'],
            electron_nelec=np.array([15,15] if molecular else [k,k]))
        new=dict(id=c['id'],R=c['R'],D=D,file=c['file'],molecular=molecular,
            N53_packet_SHA256=c['SHA256'],initial_vector_SHA256=sha256(initial.astype('<f8').tobytes()),
            initial_rule='EXACT_N51_RNG_AFTER_SIX_PROBE_DRAWS_NOT_FINAL_ROOT')
        if molecular:
            task=next(x for x in p51['tasks'] if x['method']==c['id'])
            assert np.array_equal(data['electron_strings'],task['strings'])
            new.update(problem_id=p51['problem_id'],N51_task_id=task['task_id'])
        if i<3:
            with np.load(N53/'inputs'/f"{c['id']}.npz",allow_pickle=False) as a:pass
            # Independent Slater-Condon matrices, used only as qualification labels.
            # Rebuild h/g from the same frozen synthetic random input recipe.
            rng2=np.random.default_rng(5300+n);h=rng2.normal(size=(n,n));h=(h+h.T)/2
            f=rng2.normal(size=(3,n,n));f=(f+f.transpose(0,2,1))/2
            g=np.einsum('xpq,xrs->pqrs',f[:2],f[:2])-np.einsum('xpq,xrs->pqrs',f[2:],f[2:])
            H,S,_=matrices(h,g,(k,k),[(int(a),int(b)) for a in data['strings'] for b in data['strings']],LIBRARY)
            e,v=np.linalg.eigh((H+.1*S).toarray());v=v[:,0]
            new['oracle']=dict(energy_Ha=float(v@(H@v)),penalized_energy_Ha=float(e[0]),spin_square=float(v@(S@v)))
        elif i==3:
            r=load(N51/'out'/new['N51_task_id']/'result.json')
            new['oracle']={k:r['row'][k] for k in ['energy_Ha','penalized_energy_Ha','spin_square']}
        blob=arrays(**data);write(LOCAL/new['file'],blob);hashes[new['file']]=sha256(blob);new['SHA256']=sha256(blob)
        if i<4:qual.append(new)
        else:production['lane'+str(i-4)]=new
    for src,target in [(ROOT/'frontierloop/hybrid_CI_gpu_worker.py','worker.py'),
            (ROOT/'frontierloop/hybrid_CI_solver.py','hybrid_CI_solver.py'),
            (N53/'structured_CI_device.py','structured_CI_device.py')]:
        blob=src.read_bytes();write(LOCAL/target,blob);hashes[target]=sha256(blob)
    p=dict(version='N54_BOUNDED_HYBRID_FULL_BATCH_SCI_SAME_N51_BASES',qualifications=qual,production=production,
        input_SHA256=hashes,common_runtime=p51['common_runtime'],cupy_version='13.6.0',
        numerical_packages={k:metadata.version(k) for k in ['numpy','scipy','pyscf']},
        max_H_actions=300,case_CPU_seconds=600,case_wall_seconds=180,RSS_cap_bytes=2*2**30,GPU_pool_cap_bytes=8*2**30,
        equation=p51['equation'],original_N51_protocol_SHA256=sha256((N51/'protocol.json').read_bytes()),
        N53_audit_SHA256=sha256((ROOT/'analysis/N53-CUDA-action-successor-audit.json').read_bytes()),
        qualification_tests_SHA256=sha256(tests.read_bytes()),prepare_CPU_seconds=process_time()-cpu,
        prepare_wall_seconds=perf_counter()-wall,new_model_samples=0,new_QPU_shots=0,
        no_final_roots_or_reference_energies_used_by_solver=True,qualification_labels_not_solver_inputs=True,
        same_N51_Davidson_settings_and_initialization=True,same_CI_space_and_equation=True,
        changed_backend_and_wall_CPU_resources_not_matched_end_to_end_work=True,
        independent_confirmation=False,full_paper_reproduction=False,goal_complete=False)
    save(LOCAL/'protocol.json',p);print('N54_FROZEN_TWO_MOLECULAR_SPACES_AND_FOUR_QUALIFICATIONS',flush=True)


def verified():
    p=load(LOCAL/'protocol.json')
    for file,h in p['input_SHA256'].items():
        if sha256((LOCAL/file).read_bytes())!=h:raise ValueError('Frozen N54 input changed')
    return p


def submit():
    from paperstudy.gpu_qualification import PYTHON,ENV
    p=verified();space()
    if (LOCAL/'staging-intent.json').exists():raise RuntimeError('Existing dispatch intent; no blind retry')
    queue=ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%R"',timeout=35)
    save(LOCAL/'pre-dispatch-queue.json',dict(raw=queue,other_jobs_not_modified=True))
    if any('|frontier-' in line for line in queue.splitlines()):raise RuntimeError('Another frontier job is live; reconcile own max-two budget')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('Remote namespace exists')
    for lane in p['production']:
        script='\n'.join(['#!/bin/bash','set -euo pipefail',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1',
            'export PYTHONPATH='+p['common_runtime']+'/site',
            'export CUDA_PATH='+ENV+'/lib/python3.12/site-packages/nvidia/cuda_runtime',
            'export LD_LIBRARY_PATH="$CUDA_PATH/lib:'+ENV+'/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"',
            'export CUPY_ACCELERATORS=""','export CUPY_CACHE_DIR='+REMOTE+'/cuda-cache-'+lane,
            shlex.join([PYTHON,'-u',REMOTE+'/worker.py',REMOTE,lane])])+'\n'
        write(LOCAL/(lane+'.sbatch'),script.encode())
    files=list(p['input_SHA256'])+['protocol.json','lane0.sbatch','lane1.sbatch'];bundle=LOCAL/'staging.tar.gz'
    with tarfile.open(bundle,'w:gz') as tar:
        for name in files:tar.add(LOCAL/name,arcname=name,recursive=False)
    if bundle.stat().st_size>64*2**20:raise ValueError('Bounded transfer cap exceeded')
    save(LOCAL/'staging-intent.json',dict(remote=REMOTE,archive_SHA256=sha256(bundle.read_bytes()),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    ssh('mkdir '+shlex.quote(REMOTE),timeout=35);send(bundle,REMOTE+'/staging.tar.gz')
    unpack="import pathlib,tarfile,hashlib;r=pathlib.Path("+repr(REMOTE)+");f=r/'staging.tar.gz';assert hashlib.sha256(f.read_bytes()).hexdigest()=="+repr(sha256(bundle.read_bytes()))+";t=tarfile.open(f);assert all(m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts for m in t.getmembers());t.extractall(r);print('N54_STAGED')"
    print(ssh(shlex.join([PYTHON,'-c',unpack]),timeout=35),flush=True)
    for lane in p['production']:
        command=shlex.join(['sbatch','--parsable','--partition=gpu-short','--gres=gpu:a100:1','--cpus-per-task=1','--mem=8G','--time=00:08:00',
            '--job-name=frontier-'+JOB_TAG+'-'+lane,'--output='+REMOTE+'/'+lane+'.out','--error='+REMOTE+'/'+lane+'.err',REMOTE+'/'+lane+'.sbatch'])
        save(LOCAL/(lane+'-intent.json'),dict(command=command))
        job=ssh(command,timeout=35).strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('Unknown submission; inspect exact lane name, no retry')
        save(LOCAL/(lane+'-job.json'),dict(job_id=job,remote=REMOTE,GPU_allocations=1));print('N54_A100_SUBMITTED',lane,job,flush=True)


def status():
    ids=','.join(load(f)['job_id'] for f in sorted(LOCAL.glob('lane*-job.json')))
    if not ids:raise RuntimeError('No receipt; reconcile intent first')
    raw=ssh('squeue -h -j '+ids+' -o "%i|%T|%j|%R"; sacct -X -n -j '+ids+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=40)
    save(LOCAL/'last-status.json',dict(job_ids=ids,raw=raw));print(raw,flush=True)


def progress():
    from paperstudy.gpu_qualification import PYTHON
    script='import pathlib,json;r=pathlib.Path('+repr(REMOTE)+');print(json.dumps({str(p.relative_to(r)):json.loads(p.read_text()) for p in (r/"out").glob("lane*/*/progress.json")},sort_keys=True))'
    raw=ssh(shlex.join([PYTHON,'-c',script]),timeout=35)
    save(LOCAL/'last-progress-observation.json',dict(raw=raw));print(raw,flush=True)


def collect():
    verified();jobs=[load(f) for f in sorted(LOCAL.glob('lane*-job.json'))]
    if len(jobs)!=2:raise RuntimeError('Expected two durable receipts; reconcile partial dispatch')
    ids=','.join(j['job_id'] for j in jobs)
    raw=ssh('sacct -X -n -j '+ids+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=35)
    states={line.split('|')[0]:line.split('|')[1] for line in raw.splitlines() if '|' in line}
    if any(states.get(j['job_id']) not in ['COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED'] for j in jobs):raise RuntimeError('Job live or unresolved; no copy/restart')
    target=LOCAL/'collected'
    if target.exists():raise RuntimeError('Existing collection; reconcile without recopy')
    target.mkdir();subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/out',str(target)],check=True,timeout=120)
    save(LOCAL/'collection.json',dict(accounting=raw,remote_evidence_deleted=False,
        files={str(f.relative_to(target)):sha256(f.read_bytes()) for f in target.rglob('*') if f.is_file()}))
    print('N54_ALL_REMOTE_RAW_COLLECTED',flush=True)


if __name__=='__main__':{'prepare':prepare,'submit':submit,'status':status,'progress':progress,'collect':collect}[sys.argv[1]]()
