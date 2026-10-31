"""Read-only remote prerequisite snapshot; never submits or changes jobs."""
import json,shlex
from pathlib import Path
from datetime import datetime,timezone
from .paths import ROOT
from .io import load,save,sha256
from .fixed_confirmation_inventory import LOCAL
from .prefix_performance_sampling import LOCAL as N17,REMOTE as OLD_REMOTE
from paperstudy.gpu_qualification import PYTHON
from benchmark.gpu_remote import ssh

def run():
    out=LOCAL/'remote-prerequisites-v1.json'
    if out.exists():raise RuntimeError('Existing observation; use recorded snapshot, not overwrite')
    inventory=load(LOCAL/'inventory.json');p17=load(N17/'protocol.json');build=load(ROOT/'runs/N1-NATIVE-DICE-BUILD-repair-v2/result.json')
    paths={OLD_REMOTE+'/worker.py':p17['worker_SHA256'],OLD_REMOTE+'/base_kernel.py':p17['base_SHA256'],build['binary']:build['binary_SHA256']}
    program="import json,pathlib,hashlib,subprocess,importlib.metadata; paths="+repr(paths)+"; q=subprocess.run(['squeue','-h','-u','reiwatanabe','-o','%i|%T|%j|%R'],capture_output=True,text=True); print(json.dumps({'files':{f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest() if pathlib.Path(f).is_file() else None for f in paths},'queue_returncode':q.returncode,'queue':q.stdout,'queue_stderr':q.stderr,'packages':{n:importlib.metadata.version(n) for n in ['numpy','scipy','cupy-cuda12x']}}))"
    row=dict(observed_UTC=datetime.now(timezone.utc).isoformat(),inventory_SHA256=sha256((LOCAL/'inventory.json').read_bytes()),
        checker_SHA256=sha256(Path(__file__).read_bytes()),expected=paths,job_submitted=False,other_jobs_changed=False,new_model_samples=0,new_CI_roots=0)
    try:
        response=json.loads(ssh(shlex.join([PYTHON,'-c',program]),timeout=40));row['response']=response
        row['all_expected_files_match']=response['files']==paths
        row['status']='READ_ONLY_REMOTE_PREREQUISITES_PRESENT' if row['all_expected_files_match'] and response['queue_returncode']==0 else 'REMOTE_PREREQUISITES_INCOMPLETE'
    except Exception as exc:row.update(status='REMOTE_OBSERVATION_FAILED_PRESERVED_NOT_JOB_TERMINAL',error=repr(exc))
    row['GPU_runtime_executed_now']=False;row['confirmation_execution_authorized']=False
    save(out,row);print(json.dumps(row,ensure_ascii=False),flush=True)

if __name__=='__main__':run()
