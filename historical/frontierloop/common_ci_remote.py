"""One bounded private CPU setup, no modification of existing runtimes."""
import argparse,shlex,shutil,subprocess,json
from pathlib import Path
from .paths import ROOT,OLD
from .io import load,save,sha256
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as BASE

LOCAL=ROOT/'runs/N3-PRIVATE-COMMON-CI-v1'
REMOTE='/share/reiwatanabe/ceo-native-common-ci-20261025-v1'
WORKER=ROOT/'frontierloop/common_ci_remote_worker.py'


def submit():
    if LOCAL.exists():raise RuntimeError('Existing private CPU attempt; reconcile instead of duplicate')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('Private CPU target exists')
    LOCAL.mkdir(exist_ok=False);cases=[]
    for case in load(ROOT/'protocol-T3-READOUT-SEEDS.json').get('cases',[]) or ['NAPH20-BLOCK-BOYS','N2-32-R1p2-P2','N2-32-R2p4-P2','FE2S2-40-P2']:
        p=next(p for p in load(ROOT/'freeze-T3-READOUT-SEEDS.json')['plans'] if p['source']['case']==case and p['source']['method_id']=='RAACS_B4')
        source=OLD/'data/physical'/case;meta=load(source/'problem.json');prior=load(p['source']['R2_baseline'])
        folder=LOCAL/'inputs'/case;folder.mkdir(parents=True);shutil.copyfile(source/'integrals.npz',folder/'integrals.npz')
        if sha256((folder/'integrals.npz').read_bytes())!=p['context']['physical_SHA256']['integrals.npz']:raise RuntimeError('Physical fixture mismatch')
        cases.append(dict(case=case,problem_id=p['context']['problem_id'],nelec=[meta['nalpha'],meta['nbeta']],
            core_energy=meta['core_energy'],CI_strings=prior['CI_strings'],integrals_SHA256=sha256((folder/'integrals.npz').read_bytes()),
            reference_origin=p['source']['R2_baseline'],reference_SHA256=p['source']['R2_baseline_SHA256'],
            expected={k:prior[k] for k in ['energy_Ha','penalized_energy_Ha','spin_square']}))
    kernels=LOCAL/'kernels';kernels.mkdir()
    for name in ['portable_penalized_ci.py','hole_rdm.py']:shutil.copyfile(ROOT/'frontierloop'/name,kernels/name)
    packages=dict(numpy='2.4.6',scipy='1.17.1',pyscf='2.14.0',h5py='3.16.0',setuptools='84.2.1')
    spec=dict(remote_root=REMOTE,worker_SHA256=sha256(WORKER.read_bytes()),packages=packages,
        requirements=[k+'=='+v for k,v in packages.items()],cases=cases,
        kernel_SHA256={'kernels/'+p.name:sha256(p.read_bytes()) for p in kernels.iterdir()},
        scope='FOUR_ALREADY_OBSERVED_ROOTS_COMMON_FINITE_0p1_S2_NO_OO_PLATFORM_QUALIFICATION_NO_NEW_SCIENTIFIC_CANDIDATES',
        new_model_shots=0,GPU_allocations=0)
    save(LOCAL/'protocol.json',spec);ssh('mkdir '+shlex.quote(REMOTE),timeout=35)
    send(WORKER,REMOTE+'/worker.py');send(LOCAL/'protocol.json',REMOTE+'/protocol.json')
    for path in [LOCAL/'inputs',kernels]:
        subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(path),HOST+':'+REMOTE+'/'],check=True,timeout=45)
    command=shlex.join(['sbatch','--parsable','--partition=gpu-short','--cpus-per-task=1','--mem=4G',
        '--time=00:10:00','--job-name=ceo-private-common-ci-v1','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err',
        '--wrap='+shlex.join([BASE+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE+'/protocol.json'])])
    save(LOCAL/'submission-started.json',dict(command=command,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    reply=ssh(command,timeout=35).strip().split(';')[0]
    if not reply.isdigit():raise RuntimeError('Uncertain CPU setup submission; reconcile exact name')
    save(LOCAL/'job.json',dict(job_id=reply,remote=REMOTE,GPU_allocations=0));print('SUBMITTED_PRIVATE_CPU_SETUP',reply,flush=True)


def status(collect=False):
    jid=load(LOCAL/'job.json')['job_id'];live=ssh('squeue -h -j '+jid+' -o "%i %T %j %R"',timeout=35).strip()
    accounting=ssh('sacct -X -n -j '+jid+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=35).strip()
    save(LOCAL/'status.json',dict(job_id=jid,live=live,accounting=accounting));print(live or accounting,flush=True)
    if not collect or live:return
    for name in ['started.json','install-started.json','pip-install.log','pip-report.json','runtime.json','progress.json','result.json','slurm.out','slurm.err']:
        if ssh('test -f '+shlex.quote(REMOTE+'/'+name)+' && echo yes || true',timeout=35).strip()!='yes':continue
        subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(LOCAL/name)],check=True,timeout=45)
    if (LOCAL/'result.json').exists():print('COLLECTED',load(LOCAL/'result.json')['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['submit','status','collect']);a=p.parse_args()
    if a.action=='submit':submit()
    else:status(a.action=='collect')
