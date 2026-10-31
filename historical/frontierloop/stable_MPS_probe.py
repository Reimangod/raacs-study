"""One fixed failed Fe member plus statevector-qualified small circuits."""
import base64
import gzip
from io import BytesIO
import json
import os
import subprocess
import sys
from time import process_time, perf_counter
from .paths import ROOT
from .io import load, save, sha256, space
from .filtered_SqDRIFT_plan import LOCAL, verified
from .strong_bank_prepare import prepare
from .strong_projection import OVERLAY

OUT = LOCAL/'qualification/Fe-stable-MPS-v1'


def invoke(request, name):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(OVERLAY),str(ROOT)]))
    wall = perf_counter()
    r = subprocess.run([sys.executable,'-m','frontierloop.stable_MPS_worker'],
        input=json.dumps(request), text=True, capture_output=True, timeout=60, env=env)
    evidence = dict(returncode=r.returncode, wall_seconds=perf_counter()-wall,
        request_SHA256=sha256(json.dumps(request,sort_keys=True).encode()),
        stderr=r.stderr, stdout=r.stdout)
    save(OUT/(name+'.json'),evidence)
    if r.returncode:
        raise RuntimeError('Numerical MPS failed: '+str(r.returncode))
    return json.loads(r.stdout)


def run():
    from qiskit import QuantumCircuit,qpy
    from qiskit.quantum_info import Statevector
    if OUT.exists():
        raise RuntimeError('Probe already attempted; inspect existing evidence')
    p=verified();space();cpu,wall=process_time(),perf_counter()
    save(OUT/'manifest.json',dict(original_protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
        new_worker_SHA256=sha256((ROOT/'frontierloop/stable_MPS_worker.py').read_bytes()),
        old_worker_SHA256=sha256((ROOT/'frontierloop/mps_distribution_check.py').read_bytes()),
        threshold_before=0.,threshold_after=1e-16,error_tolerance_unchanged=1e-8,
        finite_bond_cap=None,CPU_limit_unchanged=30,wall_limit=60,
        source_policy_unchanged=True,new_model_samples=0,new_CI_roots=0,
        reason='One fixed failed Fe member, no energy-driven selection; standard numerical SVD threshold',
        primary_documentation='https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html'))
    circuits={}
    for n in [2,4,6]:
        q=QuantumCircuit(n)
        for i in range(n): q.ry(.17*(i+1),i)
        for i in range(n-1): q.cx(i,i+1)
        for i in range(n): q.rz(.31*(i+1),i)
        probs=Statevector(q).probabilities_dict()
        stream=BytesIO();qpy.dump(q,stream)
        circuits[str(n)]=dict(qpy_gz_base64=base64.b64encode(gzip.compress(stream.getvalue(),mtime=0)).decode(),
            expected_probabilities={int(k,2):float(v) for k,v in probs.items()})
    small=invoke(dict(circuits=circuits,tolerance=1e-8),'small-statevector-parity')
    c=p['cases'][-1];draw=next(d for d in p['draws'] if d['case']==c['case'])
    meta=load(c['meta']);meta['case_id']=c['case'];m=draw['members'][1]
    folder=LOCAL/'exports'/c['case']/'members'/m['member_id']
    plan=dict(index=m['index'],member_id=m['member_id'],spec=m['spec'],historical_member_shots=m['shots'],
        artifacts={name:dict(path=str(folder/name),sha256=sha256((folder/name).read_bytes())) for name in ['member.json.gz','circuit.qpy.gz']})
    item=prepare(meta,plan,{str(folder/name):(folder/name).read_bytes() for name in ['member.json.gz','circuit.qpy.gz']})
    request=dict(circuits=item['request']['circuits'],tolerance=1e-8)
    save(OUT/'fixed-member.json',dict(member_id=m['member_id'],draw_id=draw['draw_id'],
        source_sparse_support=item['row']['source_sparse_support'],qpy_SHA256=item['row']['qpy_gz_SHA256']))
    real=invoke(request,'failed-Fe-member')
    save(OUT/'terminal.json',dict(status='SMALL_STATEVECTOR_AND_FIXED_FAILED_FE_ALL3_VARIANTS_PASSED',
        small=small,Fe=real,new_model_samples=0,new_CI_roots=0,
        controller_CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        numerical_truncation_explicit=True,whole_study_goal_complete=False))
    print('N57_STABLE_MPS_PROBE_PASSED',real['CPU_seconds'],flush=True)


if __name__=='__main__': run()
