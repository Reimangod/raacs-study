"""Fresh classical control, private qualified native DICE. No GPU allocated."""
from pathlib import Path
import json,shlex,subprocess,sys,tarfile
from .io import load,save,write,sha256
from .paths import ROOT
from .fixed_confirmation_execution import LOCAL,bound,capacity
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST
from paperstudy.gpu_qualification import PYTHON

HCI=LOCAL/'hci';REMOTE='/share/reiwatanabe/raacs-fixed-confirmation-hci-20261027-v1'

def prepare():
    science,b=bound();target=HCI/'protocol.json'
    if target.exists():raise RuntimeError('Existing HCI protocol; no replacement')
    n4=ROOT/'runs/N4-NATIVE-HCI-PROPOSALS-v1';old=load(n4/'protocol.json');build=load(ROOT/'runs/N1-NATIVE-DICE-BUILD-repair-v2/result.json');cases=[]
    for c in science['cases']:
        meta=load(c['meta']);p=old['physical'][c['case']];file=n4/'physical'/c['case']/'fcidump.txt'
        assert sha256(file.read_bytes())==p['fcidump_SHA256'] and p['integrals_SHA256']==c['integrals_SHA256']
        write(HCI/'physical'/c['case']/'fcidump.txt',file.read_bytes())
        cases.append(dict(case=c['case'],problem_id=meta['problem_id'],norb=meta['norb'],nelec=[meta['nalpha'],meta['nbeta']],HF_string=(1<<meta['nalpha'])-1,fcidump_SHA256=p['fcidump_SHA256']))
    files={'worker.py':ROOT/'frontierloop/fixed_confirmation_hci_worker.py','native_dice_binary.py':ROOT/'frontierloop/native_dice_binary.py','native_hci_plan.py':ROOT/'frontierloop/native_hci_plan.py'}
    for name,file in files.items():write(HCI/name,file.read_bytes())
    bindings={name:sha256((HCI/name).read_bytes()) for name in files};bindings[build['binary']]=build['binary_SHA256']
    save(target,dict(remote=REMOTE,execution_binding_SHA256=sha256((LOCAL/'execution-binding.json').read_bytes()),dispatcher_SHA256=sha256(Path(__file__).read_bytes()),
        runtime_bindings=bindings,binary=build['binary'],native_lock=old['native_lock'],cases=cases,native_wall_cap_per_case=60,final_CI_performed_here=False,new_model_samples=0))
    print('HCI_FROZEN4_FRESH_HF_SOURCE_CONTROLS',flush=True)

def submit():
    bound();capacity();p=load(HCI/'protocol.json')
    assert sha256(Path(__file__).read_bytes())==p['dispatcher_SHA256']
    if (HCI/'submission-intent.json').exists():raise RuntimeError('Existing HCI intent; reconcile')
    program="import pathlib,hashlib,json,subprocess; p="+repr(p['binary'])+"; print(json.dumps({'binary_SHA256':hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest(),'queue':subprocess.check_output(['squeue','-h','-u','reiwatanabe','-o','%i|%T|%j|%R'],text=True)}))"
    obs=json.loads(ssh(shlex.join([PYTHON,'-c',program]),timeout=40));assert obs['binary_SHA256']==p['runtime_bindings'][p['binary']]
    save(HCI/'pre-dispatch-observation.json',obs)
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('Existing remote namespace; reconcile')
    save(HCI/'stage-intent.json',dict(remote=REMOTE));ssh('mkdir '+shlex.quote(REMOTE),timeout=35)
    archive=HCI/'inputs.tar'
    with tarfile.open(archive,'w') as tf:
        for f in [HCI/'protocol.json',HCI/'worker.py',HCI/'native_dice_binary.py',HCI/'native_hci_plan.py',*sorted((HCI/'physical').rglob('fcidump.txt'))]:tf.add(f,arcname=str(f.relative_to(HCI)),recursive=False)
    send(archive,REMOTE+'/inputs.tar');ssh('tar -xf '+shlex.quote(REMOTE+'/inputs.tar')+' -C '+shlex.quote(REMOTE),timeout=35)
    command=shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=8G','--time=00:06:00','--job-name=frontier-fixed-confirm-HCI','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err','--wrap='+shlex.join([PYTHON,'-u',REMOTE+'/worker.py',REMOTE+'/protocol.json'])])
    save(HCI/'submission-intent.json',dict(command=command));reply=ssh(command,timeout=40).strip().split(';')[0]
    if not reply.isdigit():raise RuntimeError('Uncertain HCI sbatch; reconcile')
    save(HCI/'job.json',dict(job_id=reply,remote=REMOTE,GPU_allocations=0));print('HCI_CPU_SUBMITTED',reply,flush=True)

def collect():
    job=load(HCI/'job.json');jid=job['job_id'];live=ssh('squeue -h -j '+jid+' -o "%i|%T|%j|%R"',timeout=35).strip()
    account=ssh('sacct -X -n -j '+jid+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=35).strip();save(HCI/'latest-job-observation.json',dict(live=live,accounting=account));print(live or account,flush=True)
    if live:return
    # Small terminal source records only: native determinant archives retained remotely.
    for name in ['started.json','progress.json','terminal.json','slurm.out','slurm.err']:
        if ssh('test -f '+shlex.quote(REMOTE+'/'+name)+' && echo YES || true',timeout=35).strip()=='YES':
            subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(HCI/name)],check=True,timeout=40)
    print('HCI_COMPACT_EVIDENCE_COLLECTED',flush=True)

if __name__=='__main__':{'prepare':prepare,'submit':submit,'collect':collect}[sys.argv[1]]()
