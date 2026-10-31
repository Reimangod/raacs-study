"""N51 controller: fixed full-batch Fe pilot, one small M4 then two CPU lanes."""
from pathlib import Path
import importlib.metadata as metadata
import json,shlex,subprocess,sys
from time import process_time,perf_counter
import xml.etree.ElementTree as ET
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,write,arrays,sha256,digest,space
from . import native_batch_contract as batches
from .joint_ci_sparse import matrices
from .fe_native_readout_diagnostic import LIBRARY
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as LAUNCHER

LOCAL=ROOT/'runs/N51-FULL-BATCH-CI-v1'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/N51-full-batch-CI-v1'
WORKER=ROOT/'frontierloop/full_batch_CI_worker.py'


def prepare():
    if LOCAL.exists():raise RuntimeError('N51 exists; reconcile not overwrite')
    cpu,wall=process_time(),perf_counter();space()
    test=ROOT/'qualification-N51-kernel-v1.xml'; suites=list(ET.parse(test).getroot().iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=3 or any(int(s.get('errors','0')) or int(s.get('failures','0')) for s in suites):raise RuntimeError('All three kernel parity tests must pass first')
    inventory=load(batches.LOCAL/'terminal.json');tasks=[]
    for method in ['C5_D_SQDRIFT_N5','C5_D_SQDRIFT_N20','C5_C_LUCJ_COMPRESSED_L1','C5_E_LUCJ_TN_COMMON_L1']:
        source=next(s for s in inventory['rows'] if s['method']==method)
        b=next(b for b in source['batches'] if b['arm']==batches.ARMS[1] and b['batch']==0)
        task=dict(method=method,strings=b['half_strings'],R=b['R_alpha'],D=b['D'],source=source['source'],source_cost=source['cost'],batch=0)
        task['task_id']=digest(task);tasks.append(task)
    physical=OLD/'data/physical/FE2S2-40-P2';d=load(physical/'problem.json')
    write(LOCAL/'worker.py',WORKER.read_bytes());write(LOCAL/'integrals.npz',(physical/'integrals.npz').read_bytes())
    for name in ['full_batch_CI_kernel.py','portable_penalized_ci.py','hole_rdm.py']:
        write(LOCAL/'pipeline'/name,(ROOT/'frontierloop'/name).read_bytes())
    write(LOCAL/'pipeline/__init__.py',b'')
    vals={};oracles=[]
    for i,(n,k) in enumerate([(4,2),(5,3)]):
        from pyscf.fci import cistring
        rng=np.random.default_rng(5100+n);h=rng.normal(size=(n,n));h=(h+h.T)/2
        f=rng.normal(size=(3,n,n))*.2;f=(f+f.transpose(0,2,1))/2;g=np.einsum('xpq,xrs->pqrs',f,f)
        s=np.asarray(cistring.make_strings(range(n),k)[::2],np.int64)
        H,S,_=matrices(h,g,(k,k),[(a,b) for a in s for b in s],LIBRARY)
        e,v=np.linalg.eigh((H+.1*S).toarray());v=v[:,0]
        oracles.append(dict(core=.17,nelec=[k,k],energy_Ha=float(v@(H@v)+.17),penalized_energy_Ha=float(e[0]+.17),spin_square=float(v@(S@v))))
        vals.update({f'h{i}':h,f'g{i}':g,f's{i}':s})
    write(LOCAL/'qualification-inputs.npz',arrays(**vals))
    old=load(ROOT/'runs/N21-NESTED-CI-DAVIDSON-v1/protocol.json')
    files=['worker.py','integrals.npz','qualification-inputs.npz']+['pipeline/'+p.name for p in (LOCAL/'pipeline').iterdir() if p.is_file()]
    save(LOCAL/'protocol.json',dict(version='N51_FIXED_SOURCE_FULL_BATCH_PILOT',tasks=tasks,problem_id=d['problem_id'],
         nelec=[15,15],core=d['core_energy'],common_runtime=old['common_runtime'],
         numerical_packages={k:metadata.version(k) for k in ['numpy','scipy','pyscf']},qualification_oracles=oracles,
         input_SHA256={name:sha256((LOCAL/name).read_bytes()) for name in files},
         source_inventory_SHA256=sha256((batches.LOCAL/'terminal.json').read_bytes()),
         qualification_tests_SHA256=sha256(test.read_bytes()),prepare_CPU_seconds=process_time()-cpu,prepare_wall_seconds=perf_counter()-wall,
         CPU_cap_seconds=600,wall_cap_seconds=720,RSS_cap_bytes=2*2**30,max_H_actions=300,
         equation='PROJECTED_PHYSICAL_H_PLUS_0p1_S2_NOT_EXACT_S0_NO_OO',
         partitions={'M4':[0],'lane0':[2],'lane1':[3,1]},new_model_samples=0,new_QPU_shots=0,
         no_new_sources_or_optimization=True,independent_confirmation=False,full_paper_reproduction=False,
         controller_SHA256=sha256(Path(__file__).read_bytes()),goal_complete=False))
    print('N51_FROZEN4_FIXED_BATCHES_UNIFORM_CAPS',flush=True)


def verified():
    p=load(LOCAL/'protocol.json')
    for name,h in p['input_SHA256'].items():
        if sha256((LOCAL/name).read_bytes())!=h:raise RuntimeError('Frozen N51 input differs')
    return p


def m4():
    p=verified()
    command=['rtk','proxy',sys.executable,str(LOCAL/'worker.py'),str(LOCAL),'run','M4','0']
    subprocess.run(command,check=True)


def submit():
    p=verified();space()
    result=load(LOCAL/'out'/'supervisor-M4-terminal.json')
    if len(result['rows'])!=1 or result['rows'][0]['status']!='COMPLETED':raise RuntimeError('Saved small root not complete; no large dispatch')
    if (LOCAL/'staging-intent.json').exists():raise RuntimeError('Existing staging/submission; reconcile without duplicate')
    save(LOCAL/'staging-intent.json',dict(remote=REMOTE,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('Remote namespace exists; no overwrite')
    ssh('mkdir '+shlex.quote(REMOTE),timeout=35)
    for name in ['protocol.json','worker.py','integrals.npz','qualification-inputs.npz']:send(LOCAL/name,REMOTE+'/'+name)
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(LOCAL/'pipeline'),HOST+':'+REMOTE+'/'],check=True,timeout=60)
    for lane,indices in [('lane0',p['partitions']['lane0']),('lane1',p['partitions']['lane1'])]:
        wrap=shlex.join(['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','RAYON_NUM_THREADS=1',
             'PYTHONPATH='+p['common_runtime']+'/site',LAUNCHER+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE,'run',lane,*map(str,indices)])
        command=shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=4G','--time=00:26:00',
             '--job-name=frontier-N51-'+lane,'--output='+REMOTE+'/'+lane+'.out','--error='+REMOTE+'/'+lane+'.err','--wrap='+wrap])
        save(LOCAL/(lane+'-intent.json'),dict(command=command,indices=indices))
        jid=ssh(command,timeout=35).strip().split(';')[0]
        if not jid.isdigit():raise RuntimeError('Unknown submission; reconcile exact lane job name, no retry')
        save(LOCAL/(lane+'-job.json'),dict(job_id=jid,remote=REMOTE,indices=indices,GPU_allocations=0))
        print('N51_CPU_ONLY_SUBMITTED',lane,jid,flush=True)


def status():
    jobs=[load(f) for f in sorted(LOCAL.glob('lane*-job.json'))]
    if not jobs:raise RuntimeError('No dispatch receipt; inspect intent rather than submit again')
    ids=','.join(j['job_id'] for j in jobs)
    command='squeue -h -j '+ids+' -o "%i|%T|%j|%R"; sacct -X -n -j '+ids+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P'
    result=ssh(command,timeout=40);save(LOCAL/'last-remote-status.json',dict(job_ids=ids,raw=result));print(result,flush=True)


def collect():
    verified();jobs=[load(f) for f in sorted(LOCAL.glob('lane*-job.json'))]
    ids=','.join(j['job_id'] for j in jobs)
    account=ssh('sacct -X -n -j '+ids+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=35)
    states={line.split('|')[0]:line.split('|')[1] for line in account.splitlines() if '|' in line}
    if len(jobs)!=2 or any(states.get(j['job_id']) not in ['COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY'] for j in jobs):raise RuntimeError('Owned job still live/unresolved; do not collect/restart')
    target=LOCAL/'collected'
    if target.exists():raise RuntimeError('Existing collection; inspect hashes, do not blindly recopy')
    target.mkdir()
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/out',str(target)],check=True,timeout=120)
    save(LOCAL/'collection.json',dict(accounting=account,remote_evidence_deleted=False,
          files={str(f.relative_to(target)):sha256(f.read_bytes()) for f in target.rglob('*') if f.is_file()}))
    print('N51_RAW_COLLECTED',len(list(target.rglob('root.npz'))),'ROOTS')


if __name__=='__main__':{'prepare':prepare,'m4':m4,'submit':submit,'status':status,'collect':collect}[sys.argv[1]]()
