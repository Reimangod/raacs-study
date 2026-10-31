"""Retry only the N4 pre-molecular mkdir failure in a new artifact workspace.

The native/CI kernels, all20 plans, parameters and budgets stay byte-identical.
Original failed workspace/Slurm logs are untouched. No scientific outcome was
available in the original attempt; this is not favorable-outcome selection.
"""
import argparse,shlex,shutil,subprocess
from pathlib import Path
from .paths import ROOT
from .io import load,save,sha256
from . import native_hci_dispatch as base

OLD_LOCAL=base.LOCAL;OLD_REMOTE=base.REMOTE
LOCAL=ROOT/'runs/N4-NATIVE-HCI-PROPOSALS-bootstrap-v2'
REMOTE=OLD_REMOTE+'-bootstrap-v2'


def submit():
    if LOCAL.exists():raise RuntimeError('Existing bootstrap intent; reconcile instead of duplicate')
    oldjob=load(OLD_LOCAL/'job.json');original=load(OLD_LOCAL/'protocol.json')
    if base.ssh('squeue -h -j '+oldjob['job_id']+' -o %T',timeout=35).strip():raise RuntimeError('Original job live')
    proof=base.ssh('test ! -e '+shlex.quote(OLD_REMOTE+'/cells')+' && test ! -e '+shlex.quote(OLD_REMOTE+'/common-ci-attempts.json')+
        ' && test ! -e '+shlex.quote(OLD_REMOTE+'/common-ci-observations.json')+' && echo PRE_MOLECULAR_EMPTY',timeout=35).strip()
    if proof!='PRE_MOLECULAR_EMPTY':raise RuntimeError('Do not bootstrap a run that may contain molecular outcomes')
    if base.ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('New bootstrap target exists')
    LOCAL.mkdir(exist_ok=False)
    for name in ['physical','pipeline','qualification']:shutil.copytree(OLD_LOCAL/name,LOCAL/name)
    spec={**original,'remote_root':REMOTE,'bootstrap_parent_protocol_SHA256':sha256((OLD_LOCAL/'protocol.json').read_bytes()),
        'bootstrap_helper_SHA256':sha256(Path(__file__).read_bytes()),'bootstrap_parent_job':oldjob,
        'bootstrap_change':'CREATE_EMPTY_CELLS_PARENT_BEFORE_UNCHANGED_FROZEN_WORKER_NO_SCIENTIFIC_SETTING_CHANGE'}
    if {k:v for k,v in spec.items() if not k.startswith('bootstrap_') and k!='remote_root'}!={k:v for k,v in original.items() if k!='remote_root'}:
        raise RuntimeError('Scientific protocol changed during bootstrap')
    save(LOCAL/'protocol.json',spec)
    base.ssh('mkdir '+shlex.quote(REMOTE)+' '+shlex.quote(REMOTE+'/cells'),timeout=35)
    base.send(base.WORKER,REMOTE+'/worker.py');base.send(LOCAL/'protocol.json',REMOTE+'/protocol.json')
    for name in ['physical','pipeline','qualification']:
        subprocess.run(['rtk','proxy','scp','-r','-P','2222',*base.OPTIONS,str(LOCAL/name),base.HOST+':'+REMOTE+'/'],check=True,timeout=45)
    command=shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=8G',
        '--time=01:05:00','--job-name=ceo-native-hci-common-bootstrap-v2','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err',
        '--wrap='+shlex.join([base.BASE+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE+'/protocol.json'])])
    save(LOCAL/'submission-started.json',dict(command=command,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    reply=base.ssh(command,timeout=35).strip().split(';')[0]
    if not reply.isdigit():raise RuntimeError('Uncertain bootstrap submission; inspect exact name')
    save(LOCAL/'job.json',dict(job_id=reply,remote=REMOTE,GPU_allocations=0));print('SUBMITTED_SAME_PROTOCOL_BOOTSTRAP',reply,flush=True)


def status(collect=False):
    oldlocal,oldremote=base.LOCAL,base.REMOTE
    try:
        base.LOCAL,base.REMOTE=LOCAL,REMOTE;base.status(collect)
    finally:base.LOCAL,base.REMOTE=oldlocal,oldremote


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['submit','status','collect']);a=p.parse_args()
    if a.action=='submit':submit()
    else:status(a.action=='collect')
