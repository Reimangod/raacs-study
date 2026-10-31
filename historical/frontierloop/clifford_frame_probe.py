"""N60 fixed Fe gate-level verifier probe; no samples or CI outcomes."""
import json,os,sys,subprocess
from pathlib import Path
from time import perf_counter,process_time
from .filtered_SqDRIFT_plan import LOCAL,verified
from .io import load,save,sha256
from .paths import ROOT
from .strong_bank_prepare import prepare

OUT=LOCAL/'qualification/Fe-Clifford-frame-v1'


def run():
    if OUT.exists():raise RuntimeError('Existing fixed probe; inspect no repeat')
    import xml.etree.ElementTree as ET
    suites=list(ET.parse(ROOT/'qualification-N60-frame-v1.xml').getroot().iter('testsuite'))
    if sum(int(x.get('tests',0)) for x in suites)!=5 or any(int(x.get(k,0)) for x in suites for k in ['failures','errors']):raise ValueError('Small qualification not passed')
    p=verified();cpu,wall=process_time(),perf_counter()
    save(OUT/'manifest.json',dict(original_protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
        checker_SHA256=sha256((ROOT/'frontierloop/clifford_frame_checker.py').read_bytes()),
        qualifier_SHA256=sha256((ROOT/'qualification-N60-frame-v1.xml').read_bytes()),
        scope='INDEPENDENT_ACTUAL_QPY_GATE_PROBABILITY_NOT_SOURCE_Pauli_REPLAY',
        model_samples=0,CI_roots=0,error_tolerance=1e-8,CPU_limit=30,
        approximate_amplitude_cutoff=1e-14,discarded_L2_sum_limit=1e-10,
        original_protocol_not_declared_complete=True))
    c=p['cases'][-1];d=next(x for x in p['draws'] if x['case']==c['case']);m=d['members'][1]
    f=LOCAL/'exports'/c['case']/'members'/m['member_id'];meta=load(c['meta']);meta['case_id']=c['case']
    artifacts={n:dict(path=str(f/n),sha256=sha256((f/n).read_bytes())) for n in ['member.json.gz','circuit.qpy.gz']}
    q=prepare(meta,dict(index=m['index'],member_id=m['member_id'],spec=m['spec'],historical_member_shots=m['shots'],artifacts=artifacts),
        {str(f/n):(f/n).read_bytes() for n in artifacts})
    request=dict(circuits=q['request']['circuits'],tolerance=1e-8)
    save(OUT/'request-binding.json',dict(member_id=m['member_id'],QPY_SHA256=q['row']['qpy_gz_SHA256'],
        request_SHA256=sha256(json.dumps(request,sort_keys=True).encode())))
    env=dict(os.environ,PYTHONPATH=str(ROOT));t=perf_counter()
    r=subprocess.run([sys.executable,'-m','frontierloop.clifford_frame_checker'],input=json.dumps(request),
        text=True,capture_output=True,timeout=60,env=env)
    save(OUT/'terminal.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,wall_seconds=perf_counter()-t,
        controller_CPU_seconds=process_time()-cpu,passed=r.returncode==0,
        new_model_samples=0,new_CI_roots=0,original_MPS_protocol_passed=False))
    print('N60_FRAME_PROBE',r.returncode,flush=True)


if __name__=='__main__':run()
