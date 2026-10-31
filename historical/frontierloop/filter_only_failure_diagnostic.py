"""One bounded re-inspection of the first failed member; no sampling rescue.

The old wrapper retained only a stdout digest on a returned FAILED response.
This successor retains the full numerical error, without changing old records.
"""
import json,os,sys,subprocess,resource
from time import process_time,perf_counter
from .filter_only_plan import LOCAL,verified
from .paths import ROOT
from .io import load,save,sha256
from .strong_bank_prepare import prepare,semantics

OUT=ROOT/'runs/N62-VERIFIER-FAILURE-DIAG-v1'


def run():
    p=verified()
    if OUT.exists():raise RuntimeError('Existing diagnostic; no repeat')
    final=load(LOCAL/'final-accounting.json')
    if not final['process_children_reaped']:raise RuntimeError('Wait for exact source process termination')
    failures=[d for d in p['draws'] if (LOCAL/'draws'/d['draw_id']/'failure.json').exists()]
    if not failures:raise RuntimeError('No saved source failures')
    d=failures[0];folder=LOCAL/'draws'/d['draw_id'];records=list((folder/'verification-processes').glob('checker-process-*.json'))
    index=max(int(f.stem.rsplit('-',1)[-1]) for f in records);m=d['members'][index];c=next(c for c in p['cases'] if c['case']==d['case'])
    meta=load(c['meta']);meta['case_id']=c['case'];target=LOCAL/'exports'/c['case']/'members'/m['member_id']
    save(OUT/'manifest.json',dict(draw_id=d['draw_id'],case=d['case'],seed=d['seed'],member_index=index,member_id=m['member_id'],
        reason='FIRST_FAILED_DRAW_FIRST_FAILED_MEMBER_DIAGNOSTIC_NOT_OUTCOME_SELECTION',
        original_failure_SHA256=sha256((folder/'failure.json').read_bytes()),
        original_checker_process=load(folder/'verification-processes'/('checker-process-'+str(index)+'.json')),
        source_packet_SHA256=sha256((target/'member.json.gz').read_bytes()),
        sampler_and_source_status_will_not_change=True,new_model_samples=0,new_CI_roots=0,
        original_intermediate_QPY_blobs_not_saved=True,semantic_reconstruction_not_byte_identical_claim=True,
        worker_CPU_limit_unchanged=30,wall_limit=60))
    cpu,wall=process_time(),perf_counter();artifacts={name:dict(path=str(target/name),sha256=sha256((target/name).read_bytes())) for name in ['member.json.gz','circuit.qpy.gz']}
    item=prepare(meta,dict(index=index,member_id=m['member_id'],spec=m['spec'],historical_member_shots=m['shots'],artifacts=artifacts),
        {str(target/name):(target/name).read_bytes() for name in artifacts})
    request=dict(jobs=[item['request']],tolerance=1e-8)
    save(OUT/'reconstructed-semantics.json',{name:semantics(b) for name,b in item['qpy_blobs'].items()})
    env=dict(os.environ,PYTHONPATH=str(ROOT));r=subprocess.run([sys.executable,'-m','frontierloop.frame_batch_worker'],
        input=json.dumps(request),text=True,capture_output=True,timeout=60,env=env)
    child=resource.getrusage(resource.RUSAGE_CHILDREN)
    save(OUT/'terminal.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
        parsed=json.loads(r.stdout) if r.returncode==0 else None,
        CPU_seconds_parent_and_reaped_children=process_time()-cpu+child.ru_utime+child.ru_stime,
        wall_seconds=perf_counter()-wall,new_model_samples=0,new_CI_roots=0,original_failure_preserved=True))
    print('N62_DIAGNOSTIC',r.returncode,r.stdout[-3500:],flush=True)


if __name__=='__main__':run()
