"""One private CPU-only native DICE build; reconcile uncertain submission."""
import argparse,json,shlex,subprocess
from pathlib import Path
from .paths import ROOT,PROJECT
from .io import load,save,sha256,write
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as BASE

REMOTE='/share/reiwatanabe/ceo-native-dice-20261025-v1'
LOCAL=ROOT/'runs/N1-NATIVE-DICE-BUILD-v1'
WORKER=ROOT/'frontierloop/native_dice_build_worker.py'


def submit():
    if LOCAL.exists():raise RuntimeError('Existing attempt; reconcile status, never duplicate')
    check=ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW_PRIVATE_ROOT',timeout=35)
    if check.strip()!='NEW_PRIVATE_ROOT':raise RuntimeError('Remote target already exists')
    LOCAL.mkdir(exist_ok=False)
    protocol=dict(remote_root=REMOTE,worker_SHA256=sha256(WORKER.read_bytes()),
        dice_commit='3198db1f8e0066489d173d30a52bc21967969a0f',
        wrapper_commit='58c45f4106396914ce496c09e21dcb2be073dc34',
        wrapper_url='https://raw.githubusercontent.com/Qiskit/qiskit-addon-dice-solver/58c45f4106396914ce496c09e21dcb2be073dc34/qiskit_addon_dice_solver/dice_solver.py',
        wrapper_SHA256='abbe282f5fec903b384ec048a553a134c1d2033dfa28efa8519565ad20bed512',
        boost_url='https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz',
        boost_SHA256='be0d91732d5b0cc6fbb275c7939974457e79b54d6f07ce2e3dfdd68bef883b0b',
        max_wall_seconds=840,runtime_cap_bytes=512*1024**2,disk_floor_bytes=3*1024**3,
        CPU_cores=1,GPU_allocations=0,new_model_shots=0,new_molecular_energy_evaluations=0,
        scope='BUILD_ONLY_NATIVE_SERIAL_UNQUALIFIED_NO_SYSTEM_INSTALL_OR_SHARED_ENV_CHANGE')
    save(LOCAL/'protocol.json',protocol)
    ssh('mkdir '+shlex.quote(REMOTE),timeout=35)
    send(WORKER,REMOTE+'/worker.py');send(LOCAL/'protocol.json',REMOTE+'/protocol.json')
    command=shlex.join(['sbatch','--parsable','--partition=gpu-short','--cpus-per-task=1','--mem=4G',
        '--time=00:15:00','--job-name=ceo-native-dice-build-v1','--output='+REMOTE+'/slurm.out',
        '--error='+REMOTE+'/slurm.err','--wrap='+shlex.join([BASE+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE+'/protocol.json'])])
    save(LOCAL/'submission-started.json',dict(command=command,remote_root=REMOTE,
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),GPU_allocations=0))
    reply=ssh(command,timeout=35).strip().split(';')[0]
    if not reply.isdigit():raise RuntimeError('Uncertain sbatch response; inspect exact job name before any retry')
    save(LOCAL/'job.json',dict(job_id=reply,remote=REMOTE,GPU_allocations=0));print('SUBMITTED_CPU_BUILD',reply,flush=True)


def status(collect=False):
    job=load(LOCAL/'job.json');jid=job['job_id']
    live=ssh('squeue -h -j '+jid+' -o "%i %T %j %R"',timeout=35).strip()
    accounting=ssh('sacct -X -n -j '+jid+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=35).strip()
    value=dict(live=live,accounting=accounting,job_id=jid)
    save(LOCAL/'status.json',value);print(json.dumps(value),flush=True)
    if not collect or live:return
    result=ssh('test -f '+shlex.quote(REMOTE+'/result.json')+' && echo RESULT_PRESENT || true',timeout=35).strip()
    names=['slurm.out','slurm.err','progress.json','started.json']+(['result.json'] if result=='RESULT_PRESENT' else [])
    for name in names:
        present=ssh('test -f '+shlex.quote(REMOTE+'/'+name)+' && echo yes || true',timeout=35).strip()
        if present!='yes':continue
        subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(LOCAL/name)],check=True,timeout=45)
    if result=='RESULT_PRESENT':
        value=load(LOCAL/'result.json')
        for event in value['events']:
            name=event['name']+'.log'
            if Path(name).name!=name or not name.endswith('.log'):raise RuntimeError('Unsafe event log path')
            subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(LOCAL/name)],check=True,timeout=45)
        print('COLLECTED',value['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['submit','status','collect']);a=p.parse_args()
    if a.action=='submit':submit()
    else:status(a.action=='collect')
