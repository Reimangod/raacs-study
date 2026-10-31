"""CPU-only tiny native qualification, not molecular performance dispatch."""
import argparse,shlex,subprocess,json
from pathlib import Path
from .paths import ROOT
from .io import load,save,sha256
from .native_dice_toys import RUN
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as BASE

REMOTE='/share/reiwatanabe/ceo-native-dice-toys-20261025-v1'
WORKER=ROOT/'frontierloop/native_dice_toy_worker.py'


def submit():
    if (RUN/'submission-started.json').exists() or (RUN/'job.json').exists():raise RuntimeError('Existing tiny submission; reconcile,never duplicate')
    build=load(ROOT/'runs/N1-NATIVE-DICE-BUILD-repair-v2/result.json')
    if build['status']!='NATIVE_DICE_SERIAL_BUILT_UNQUALIFIED_NO_SCIENTIFIC_OUTCOMES' or not build['prior_evidence_unchanged']:
        raise RuntimeError('Pinned source build required')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('Tiny target already exists')
    fixtures=load(RUN/'fixtures.json')
    if len(fixtures['fixtures'])!=7:raise RuntimeError('All7 synthetic controls required')
    spec=dict(remote_root=REMOTE,binary=build['binary'],binary_SHA256=build['binary_SHA256'],
        worker_SHA256=sha256(WORKER.read_bytes()),fixtures_SHA256=sha256((RUN/'fixtures.json').read_bytes()),
        scope='ONLY_SYNTHETIC_FIXEDSPACE_AND_HCI_EXPANSION_SPIN_CONTRACT_NO_MOLECULAR_PERFORMANCE',
        max_cell_wall_seconds=20,new_model_shots=0,new_QPU_shots=0)
    save(RUN/'protocol.json',spec);ssh('mkdir '+shlex.quote(REMOTE),timeout=35)
    send(WORKER,REMOTE+'/worker.py');send(RUN/'protocol.json',REMOTE+'/protocol.json');send(RUN/'fixtures.json',REMOTE+'/fixtures.json')
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(RUN/'inputs'),HOST+':'+REMOTE+'/'],check=True,timeout=45)
    command=shlex.join(['sbatch','--parsable','--partition=gpu-short','--cpus-per-task=1','--mem=2G',
        '--time=00:05:00','--job-name=ceo-native-dice-toys-v1','--output='+REMOTE+'/slurm.out',
        '--error='+REMOTE+'/slurm.err','--wrap='+shlex.join([BASE+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE+'/protocol.json'])])
    save(RUN/'submission-started.json',dict(command=command,protocol_SHA256=sha256((RUN/'protocol.json').read_bytes())))
    reply=ssh(command,timeout=35).strip().split(';')[0]
    if not reply.isdigit():raise RuntimeError('Uncertain tiny submission; inspect exact job name')
    save(RUN/'job.json',dict(job_id=reply,remote=REMOTE,GPU_allocations=0));print('SUBMITTED_TINY_CPU',reply,flush=True)


def status(collect=False):
    job=load(RUN/'job.json');jid=job['job_id']
    live=ssh('squeue -h -j '+jid+' -o "%i %T %j %R"',timeout=35).strip()
    accounting=ssh('sacct -X -n -j '+jid+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=35).strip()
    save(RUN/'status.json',dict(job_id=jid,live=live,accounting=accounting));print(live or accounting,flush=True)
    if not collect or live:return
    for name in ['started.json','progress.json','terminal.json','slurm.out','slurm.err']:
        if ssh('test -f '+shlex.quote(REMOTE+'/'+name)+' && echo yes || true',timeout=35).strip()!='yes':continue
        subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(RUN/name)],check=True,timeout=45)
    if (RUN/'terminal.json').exists():
        subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/inputs',str(RUN/'collected')],check=True,timeout=45)
        print('TINY_RAW_OUTPUTS_COLLECTED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['submit','status','collect']);a=parser.parse_args()
    if a.action=='submit':submit()
    else:status(a.action=='collect')
