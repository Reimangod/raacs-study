"""Additive dependency repair, not a retry of scientific outcomes."""
import argparse,shlex
from .paths import ROOT
from .io import load,save,sha256
from . import native_dice_dispatch as base

PRIOR=ROOT/'runs/N1-NATIVE-DICE-BUILD-v1'
PRIOR_REMOTE='/share/reiwatanabe/ceo-native-dice-20261025-v1'
base.REMOTE=PRIOR_REMOTE+'-repair-v2'
base.LOCAL=ROOT/'runs/N1-NATIVE-DICE-BUILD-repair-v2'
WORKER=ROOT/'frontierloop/native_dice_build_repair.py'


def submit():
    if base.LOCAL.exists():raise RuntimeError('Prior repair intent; reconcile exact job before any retry')
    previous=load(PRIOR/'job.json');old=load(PRIOR/'result.json')
    if 'FAILURE' not in old['status']:raise RuntimeError('Only the diagnosed failed build may be repaired')
    if base.ssh('squeue -h -j '+previous['job_id']+' -o %T',timeout=35).strip():raise RuntimeError('Previous job still live')
    if base.ssh('test ! -e '+shlex.quote(base.REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':
        raise RuntimeError('Repair target already exists')
    base.LOCAL.mkdir(exist_ok=False)
    proof={name:sha256((PRIOR/name).read_bytes()) for name in ['protocol.json','result.json','10-serialization.log','progress.json','started.json']}
    proof['worker.py']=load(PRIOR/'protocol.json')['worker_SHA256']
    protocol=dict(remote_root=base.REMOTE,prior_root=PRIOR_REMOTE,prior_evidence_SHA256=proof,
        worker_SHA256=sha256(WORKER.read_bytes()),max_wall_seconds=840,
        repair='ONLY_MISSING_OFFICIAL_BOOST_TOOLS_BOOST_INSTALL_FROM_SAME_VERIFIED_ARCHIVE_THEN_UNCHANGED_NATIVE_SERIAL_BUILD',
        native_sources_changed=False,GPU_allocations=0,new_model_shots=0,new_molecular_energy_evaluations=0)
    save(base.LOCAL/'protocol.json',protocol)
    base.ssh('mkdir '+shlex.quote(base.REMOTE),timeout=35)
    base.send(WORKER,base.REMOTE+'/worker.py');base.send(base.LOCAL/'protocol.json',base.REMOTE+'/protocol.json')
    command=shlex.join(['sbatch','--parsable','--partition=gpu-short','--cpus-per-task=1','--mem=4G',
        '--time=00:15:00','--job-name=ceo-native-dice-repair-v2','--output='+base.REMOTE+'/slurm.out',
        '--error='+base.REMOTE+'/slurm.err','--wrap='+shlex.join([base.BASE+'/venv/bin/python','-u',
            base.REMOTE+'/worker.py',base.REMOTE+'/protocol.json'])])
    save(base.LOCAL/'submission-started.json',dict(command=command,prior_job_id=previous['job_id'],
        protocol_SHA256=sha256((base.LOCAL/'protocol.json').read_bytes()),GPU_allocations=0))
    reply=base.ssh(command,timeout=35).strip().split(';')[0]
    if not reply.isdigit():raise RuntimeError('Uncertain submission; reconcile exact job name')
    save(base.LOCAL/'job.json',dict(job_id=reply,remote=base.REMOTE,GPU_allocations=0))
    print('SUBMITTED_CPU_REPAIR',reply,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['submit','status','collect']);a=parser.parse_args()
    if a.action=='submit':submit()
    else:base.status(a.action=='collect')
