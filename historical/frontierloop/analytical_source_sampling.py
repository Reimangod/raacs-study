"""N33 new finite draws from ALL4 paid N32 operators, qualified N17 engine."""
from pathlib import Path
from io import BytesIO
from time import process_time,perf_counter
import math
import xml.etree.ElementTree as ET
import numpy as np
import ffsim
from qiskit import qpy
from paperstudy.lucj import make_circuit
from .paths import ROOT
from .io import load,save,write,arrays,sha256,digest,space
from . import analytical_source_optimization as n32
from . import finite_source_study as n29
from . import optimized_control_sampling as n30
from . import prefix_performance_sampling as legacy

LOCAL=ROOT/'runs/N33-ANALYTICAL-L1-FINITE-v1'
SAMPLING=LOCAL/'sampling'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/N33-analytical-l1-finite-v1'
METHOD='ANALYTICAL_OPT_L1_8EVAL'
REPLICATES=(290101,290102,290103)


def seed(cid,rep):return int(digest(['N33_NEW_MODEL_DRAW_V1',cid,rep])[:8],16)


def configured():
    legacy.LOCAL=SAMPLING;legacy.REMOTE=REMOTE;legacy.SHOTS=1000
    return legacy


def prepare():
    storage=legacy.capacity();n32p=n32.verified();t=load(n32.LOCAL/'terminal.json');old=load(n29.LUCJ/'protocol.json')
    if len(t['rows'])!=4 or any(r['exitcode']!=0 for r in t['rows']):raise RuntimeError('All4 actual N32 outcomes required')
    if LOCAL.exists():raise RuntimeError('Existing N33; preserve/reconcile')
    test=ROOT/'qualification-N32-terminal-tests-v2.xml';suites=list(ET.parse(test).getroot().iter('testsuite'))
    if sum(int(s.get('tests',0)) for s in suites)!=6 or any(int(s.get('failures',0)) or int(s.get('errors',0)) for s in suites):raise RuntimeError('Six scoped N31/N32 evidence tests required')
    cpu,wall=process_time(),perf_counter();module=legacy.qualified_module();LOCAL.mkdir();SAMPLING.mkdir()
    for name in ('worker.py','base_kernel.py','primitive_plan.py'):write(SAMPLING/name,(n29.LUCJ/name).read_bytes())
    circuits=[];cells=[];sources=[]
    for item in n32p['sources']:
        s=item['source'];case=s['case'];resultfile=n32.LOCAL/'cases'/case/'result.json';r=load(resultfile)
        if r['problem_id']!=s['problem_id'] or sha256(Path(r['operator']).read_bytes())!=r['operator_SHA256']:raise RuntimeError('Actual optimized source drift')
        with np.load(r['operator'],allow_pickle=False) as z:op=ffsim.UCJOpSpinBalanced(z['diag_coulomb_mats'],z['orbital_rotations'],z['final_orbital_rotation'])
        qc=ffsim.qiskit.PRE_INIT.run(make_circuit(op,s['norb'],tuple(s['nelec'])))
        stream=BytesIO();qpy.dump(qc,stream);blob=stream.getvalue();path=LOCAL/'sources'/case/'source.qpy';write(path,blob)
        restored=qpy.load(BytesIO(blob))[0];cfg,values=module.plan(restored,s['norb'],tuple(s['nelec']))
        cid=digest(dict(problem_id=s['problem_id'],QPY_SHA256=sha256(blob),decoder=list(range(2*s['norb'])),qualified_plan_SHA256=old['plan_SHA256']))
        folder=SAMPLING/'circuits'/cid;folder.mkdir(parents=True);payload=arrays(**values);write(folder/'arrays.npz',payload)
        requests=[dict(id=digest(['N33',cid,rep]),replicate_label=rep,seed=seed(cid,rep),shots=1000) for rep in REPLICATES]
        cfg.update(arrays_SHA256=sha256(payload),worker_SHA256=old['worker_SHA256'],base_kernel_SHA256=old['base_SHA256'],
            source_QPY_SHA256=sha256(blob),physical_problem_id=s['problem_id'],sampling_requests=requests)
        cfg['input_id']=digest(cfg);save(folder/'config.json',cfg)
        dim=math.prod(math.comb(s['norb'],e) for e in s['nelec'])
        circuit=dict(circuit_id=cid,case=case,source=METHOD,problem_id=s['problem_id'],norb=s['norb'],nelec=s['nelec'],physical_SHA256=s['physical_SHA256'],
            original_QPY=str(path),original_QPY_SHA256=sha256(blob),decoder=list(range(2*s['norb'])),config_SHA256=sha256((folder/'config.json').read_bytes()),
            arrays_SHA256=sha256(payload),input_id=cfg['input_id'],sampling_requests=requests,sector_dimension=dim,
            primitive_count=len(cfg['ops']),estimated_work=dim*len(cfg['ops']),backend='M4' if s['norb']==10 else 'A100')
        circuits.append(circuit)
        sources.append(dict(case=case,problem_id=s['problem_id'],result=str(resultfile),result_SHA256=sha256(resultfile.read_bytes()),
            operator=r['operator'],operator_SHA256=r['operator_SHA256'],resources=r['optimized_resources'],
            optimizer_coordinate_slots=r['optimizer_coordinate_slots'],independent_parameter_rank=None,costs=r['costs'],old_initialization_cost=r['old_initialization_cost']))
        for req in requests:
            cell=dict(case=case,source=METHOD,method='ORIGINAL',problem_id=s['problem_id'],replicate=req['replicate_label'],status='PLANNED',
                circuit_id=cid,draw_id=req['id'],selected_resources=r['optimized_resources'][0]['resources'])
            cell['cell_id']=digest(['N33',cell]);cells.append(cell)
    previous={r['seed'] for c in old['circuits'] for r in c['sampling_requests']}
    previous|={r['seed'] for c in load(n30.SAMPLING/'protocol.json')['circuits'] for r in c['sampling_requests']}
    seeds=[r['seed'] for c in circuits for r in c['sampling_requests']]
    if len(seeds)!=12 or len(set(seeds))!=12 or previous&set(seeds):raise RuntimeError('New seed collision')
    lanes=[[],[]];loads=[0,0]
    for c in sorted((c for c in circuits if c['backend']=='A100'),key=lambda c:(-c['estimated_work'],c['circuit_id'])):
        lane=min(range(2),key=lambda j:loads[j]);lanes[lane].append(c['circuit_id']);loads[lane]+=c['estimated_work']
    sub=dict(old,version='N33_PAID_ANALYTICAL_L1_NEW1000_DRAWS_V1',cells=cells,circuits=circuits,GPU_lanes=lanes,
        maximum_new_model_shots=12000,maximum_logical_model_shots=12000,selection='ALL4_PHYSICAL_VQE_SELECTED_NOT_SQD_SELECTED',
        namespace_override=str(SAMPLING),remote_override=REMOTE,readouts=['N33_R32_CAP','N33_COMMON_R_ALL7METHODS'])
    sub.pop('N29_adapter_SHA256',None);sub['N33_adapter_SHA256']=sha256(Path(__file__).read_bytes())
    legacy.init_ledger(SAMPLING/'draw-ledger.sqlite',cells,circuits);save(SAMPLING/'protocol.json',sub)
    save(LOCAL/'protocol.json',dict(version='N33_ALL4_OPTIMIZED_L1_FRESH_FINITE_V1',sources=sources,
        physical={x['source']['case']:x['physical'] for x in n32p['sources']},
        sampling_protocol_SHA256=sha256((SAMPLING/'protocol.json').read_bytes()),code_SHA256=sha256(Path(__file__).read_bytes()),
        N32_terminal_SHA256=sha256((n32.LOCAL/'terminal.json').read_bytes()),N32_protocol_SHA256=sha256((n32.LOCAL/'protocol.json').read_bytes()),
        test_SHA256=sha256(test.read_bytes()),plan_SHA256=sha256((ROOT/'analysis/N33_FINITE_SAMPLING_PLAN_JA.md').read_bytes()),
        prepare_CPU_seconds=process_time()-cpu,prepare_wall_seconds=perf_counter()-wall,new_run_cap_bytes=32*2**20,
        new_model_shots_cap=12000,new_QPU_shots=0,new_optimizer_calls=0,expected_CUDA_cupy='13.6.0',
        library=load(n29.LOCAL/'protocol.json')['library'],library_SHA256=load(n29.LOCAL/'protocol.json')['library_SHA256'],
        storage_before=storage,goal_complete=False))
    print('N33_FROZEN_ALL4_12_FRESH1000_DRAWS_MAX2_ONEA100',flush=True)


def verified():
    n32.verified();p=load(LOCAL/'protocol.json')
    if p['code_SHA256']!=sha256(Path(__file__).read_bytes()) or p['sampling_protocol_SHA256']!=sha256((SAMPLING/'protocol.json').read_bytes()):raise RuntimeError('N33 frozen drift')
    if p['N32_terminal_SHA256']!=sha256((n32.LOCAL/'terminal.json').read_bytes()) or p['N32_protocol_SHA256']!=sha256((n32.LOCAL/'protocol.json').read_bytes()):raise RuntimeError('N32 outcome context drift')
    for s in p['sources']:
        if sha256(Path(s['result']).read_bytes())!=s['result_SHA256'] or sha256(Path(s['operator']).read_bytes())!=s['operator_SHA256']:raise RuntimeError('Selected output drift')
    configured().verified_protocol();return p


def m4():verified();configured().m4()
def submit():verified();configured().submit()
def status():verified();configured().status()
def collect():
    p=verified()
    from . import collect_prefix_performance as c
    c.LOCAL=SAMPLING;c.REMOTE=REMOTE;c.collect()


if __name__=='__main__':
    import sys
    {'prepare':prepare,'m4':m4,'submit':submit,'status':status,'collect':collect}[sys.argv[1]]()
