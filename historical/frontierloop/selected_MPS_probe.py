"""Fixed failed Fe member: verify actual resource-selected circuit only.

Diagnostic only. Does not authorize sampling under the old all-three contract.
"""
import json,os,sys,subprocess
from time import perf_counter
from .filtered_SqDRIFT_plan import LOCAL,verified
from .io import load,save,sha256
from .paths import ROOT
from .strong_bank_prepare import prepare
from .strong_projection import OVERLAY


def run():
    out=LOCAL/'qualification/Fe-selected-MPS-v1'
    if out.exists(): raise RuntimeError('Do not repeat existing probe')
    p=verified();c=p['cases'][-1];d=next(x for x in p['draws'] if x['case']==c['case']);m=d['members'][1]
    f=LOCAL/'exports'/c['case']/'members'/m['member_id'];meta=load(c['meta']);meta['case_id']=c['case']
    artifacts={n:dict(path=str(f/n),sha256=sha256((f/n).read_bytes())) for n in ['member.json.gz','circuit.qpy.gz']}
    q=prepare(meta,dict(index=m['index'],member_id=m['member_id'],spec=m['spec'],historical_member_shots=m['shots'],artifacts=artifacts),
        {str(f/n):(f/n).read_bytes() for n in artifacts})
    chosen=q['row']['policy']['selected']
    request=dict(circuits={chosen:q['request']['circuits'][chosen]},tolerance=1e-8)
    save(out/'manifest.json',dict(member_id=m['member_id'],selected=chosen,policy=q['row']['policy'],
        source_and_selected_QPY_unchanged=True,request_SHA256=sha256(json.dumps(request,sort_keys=True).encode()),
        worker_SHA256=sha256((ROOT/'frontierloop/stable_MPS_worker.py').read_bytes()),
        original_all_three_requirement_not_satisfied_by_this_probe=True,
        CPU_limit=30,wall_limit=60,new_model_samples=0,new_CI_roots=0))
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(OVERLAY),str(ROOT)]));t=perf_counter()
    r=subprocess.run([sys.executable,'-m','frontierloop.stable_MPS_worker'],input=json.dumps(request),
        text=True,capture_output=True,timeout=60,env=env)
    save(out/'terminal.json',dict(returncode=r.returncode,wall_seconds=perf_counter()-t,stdout=r.stdout,stderr=r.stderr,
        passed=r.returncode==0,new_model_samples=0,new_CI_roots=0,old_protocol_completion_claimed=False))
    print('SELECTED_MPS_RETURN',r.returncode,flush=True)


if __name__=='__main__': run()
