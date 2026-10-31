"""Single additive N9 source-staging repair, preserving failed v1 evidence."""
import argparse,shlex,shutil,subprocess
from .paths import ROOT
from .io import load,save,sha256,space
from . import paired_dice_qualification as original
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as BASE

LOCAL=ROOT/'runs/N9-PAIRED-INPUT-QUALIFICATION-bootstrap-v2'
REMOTE='/share/reiwatanabe/ceo-paired-input-qualification-20261026-bootstrap-v2'
BOOT=ROOT/'frontierloop/paired_dice_bootstrap.py'


def submit():
    if LOCAL.exists():raise RuntimeError('Existing repair intent; reconcile exact job')
    space();prior=original.LOCAL;oldspec=load(prior/'protocol.json');jid=load(prior/'job.json')['job_id']
    acc=ssh('sacct -X -n -j '+jid+' --format=JobID,State,Elapsed,ExitCode -P',timeout=35)
    if '|FAILED|' not in acc:raise RuntimeError('Prior exact job not failed/terminal')
    text=ssh('cat '+shlex.quote(original.REMOTE+'/terminal.json'),timeout=35)
    import json
    terminal=json.loads(text)
    if terminal['rows'] or terminal['new_molecular_evaluations'] or not terminal['original_source_unchanged'] or not terminal['original_binary_unchanged']:
        raise RuntimeError('Not an outcome-free staging repair')
    if "FileExistsError" not in terminal['error']:raise RuntimeError('Different incident; investigate before repair')
    LOCAL.mkdir();save(LOCAL/'prior-terminal.json',terminal);save(LOCAL/'prior-accounting.json',dict(accounting=acc))
    shutil.copyfile(prior/'fixtures.json',LOCAL/'fixtures.json');shutil.copytree(prior/'inputs',LOCAL/'inputs')
    if sha256((LOCAL/'fixtures.json').read_bytes())!=oldspec['fixtures_SHA256']:raise RuntimeError('Original synthetic fixtures changed')
    evidence={}
    for name in ['protocol.json','terminal.json']:
        evidence[name]=ssh('sha256sum '+shlex.quote(original.REMOTE+'/'+name),timeout=35).split()[0]
    spec=dict(oldspec,version='N9_TINY_QUALIFICATION_STAGING_REPAIR_V2',remote_root=REMOTE,
        bootstrap_SHA256=sha256(BOOT.read_bytes()),prior_failed_root=original.REMOTE,prior_failed_evidence_SHA256=evidence,
        source_staging='ONLY_CACHED_SHCI_EIGEN_ROOT_CPP_H_MAKEFILE_LICENSE_NO_LAZY_FETCH',
        previous_staging_incident='FULL_GIT_ARCHIVE_CREATED_OBJ_DIRECTORY_AND_UNNECESSARY_DATA_579172KiB_PRESERVED_NO_NUMERIC_OUTCOME')
    save(LOCAL/'protocol.json',spec)
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('Repair remote root exists')
    ssh('mkdir '+shlex.quote(REMOTE),timeout=35)
    for p,name in [(BOOT,'bootstrap.py'),(original.WORKER,'base_worker.py'),(original.PATCH,'paired_dice_patch.py'),
                   (LOCAL/'protocol.json','protocol.json'),(LOCAL/'fixtures.json','fixtures.json')]:send(p,REMOTE+'/'+name)
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(LOCAL/'inputs'),HOST+':'+REMOTE+'/'],check=True,timeout=50)
    wrap=shlex.join(['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',BASE+'/venv/bin/python','-u',REMOTE+'/bootstrap.py',REMOTE+'/protocol.json'])
    cmd=shlex.join(['sbatch','--parsable','--partition=gpu-short','--cpus-per-task=1','--mem=4G','--time=00:15:00',
        '--job-name=ceo-paired-input-tiny-bootstrap-v2','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err','--wrap='+wrap])
    save(LOCAL/'submission-started.json',dict(command=cmd,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    job=ssh(cmd,timeout=35).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Unknown submit result; do not retry blindly')
    save(LOCAL/'job.json',dict(job_id=job,remote=REMOTE,GPU_allocations=0));print('N9_ADDITIVE_STAGING_REPAIR_SUBMITTED_CPU_ONLY',job,flush=True)


def status():
    original.LOCAL=LOCAL;original.REMOTE=REMOTE;original.status(False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['submit','status']);a=p.parse_args();globals()[a.action]()
