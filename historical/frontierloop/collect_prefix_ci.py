"""N17 split-batch CI status and bounded terminal evidence transfer."""
import argparse
import types
from pathlib import Path
from .paths import ROOT
from .io import load,save,sha256
from . import prefix_performance_ci as study
from .prefix_performance_sampling import capacity
from benchmark.gpu_remote import ssh


def status():
    job=load(study.LOCAL/'job.json')['job_id']
    queue=ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%R"',timeout=30)
    live='\n'.join(line for line in queue.splitlines() if line.split('|')[0]==job)
    account=ssh('sacct -X -n -j '+job+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=30).strip()
    result=dict(job_id=job,live=live,accounting=account)
    save(study.LOCAL/'status.json',result);print(live or account or 'UNRESOLVED_NO_RESTART',flush=True)


def collector_module():
    # The mature hash/path/gzip/terminal collector is reused, changing only
    # its selected-evidence byte ceiling and cosmetic stdout label.
    parent=ROOT/'frontierloop/collect_small_pool_bundle.py';text=parent.read_text()
    before='total>32*1024**2'
    if text.count(before)!=1:raise RuntimeError('Pinned collection bound context changed')
    text=text.replace(before,'total>128*1024**2').replace("'N7_EVIDENCE_ONLY_BUNDLE_COLLECTED'","'N17_CI_EVIDENCE_ONLY_BUNDLE_COLLECTED'")
    module=types.ModuleType('frontierloop.n17_ci_evidence_collector');module.__package__='frontierloop';module.__file__=str(parent)
    exec(compile(text,str(parent),'exec'),module.__dict__)
    module.LOCAL=study.LOCAL;module.REMOTE=study.REMOTE
    return module,parent,text


def collect():
    capacity();spec=load(study.LOCAL/'protocol.json')
    if sha256((study.LOCAL/'worker.py').read_bytes())!=spec['worker_SHA256']:
        raise RuntimeError('Frozen actual CI worker changed')
    module,parent,text=collector_module();module.collect()
    save(study.LOCAL/'N17-collector-binding.json',dict(parent_collector_SHA256=sha256(parent.read_bytes()),
        generated_collector_SHA256=sha256(text.encode()),adapter_SHA256=sha256(Path(__file__).read_bytes()),
        selected_bytes_cap=128*2**20,raw_remote_deleted=False,scientific_execution_restarted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','collect']);p.add_argument('--batch',choices=['M4','A100','ALL'],default='M4')
    a=p.parse_args();study.configure(a.batch);globals()[a.action]()
