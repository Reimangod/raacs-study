"""N30 existing tuned LUCJ source: fresh counts, unchanged qualified kernel."""
from pathlib import Path
from io import BytesIO
from time import process_time, perf_counter
import math
import xml.etree.ElementTree as ET
import ffsim
import numpy as np
from qiskit import qpy
from .paths import ROOT, OLD
from .io import load, save, write, arrays, sha256, digest, space
from . import prefix_performance_sampling as legacy
from . import finite_source_study as n29
from .slater_prefix_preflight import compile_views
from paperstudy.lucj import make_circuit

LOCAL=ROOT/'runs/N30-EXISTING-TUNED-LUCJ-v1'
SAMPLING=LOCAL/'sampling'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/N30-existing-tuned-lucj-v1'
METHOD='EXISTING_TN_COMMON_L1'
REPLICATES=(290101,290102,290103)


def source_inventory(physical):
    specfile=OLD/'TN-high-final-v1/spec.json';old=load(specfile);rows=[]
    for case,d in sorted(physical.items()):
        s=old['sources'].get(case)
        if s is None or s['problem_id']!=d['problem_id']:
            rows.append(dict(case=case,status='MISSING_SAME_PROBLEM_SOURCE',problem_id=d['problem_id'],
                reason='NO_TUNED_OPERATOR_WITH_SAME_PROBLEM_AND_ORBITAL_ID;CANONICAL_NAPH_NOT_BLOCK_BOYS'))
            continue
        file=OLD/s['operator'];construction=OLD/s['construction']
        if sha256(file.read_bytes())!=s['operator_sha256'] or sha256(construction.read_bytes())!=s['construction_sha256']:
            raise RuntimeError('Existing selected tuned source changed')
        c=load(construction)
        if c['operator_sha256']!=s['operator_sha256']:
            raise RuntimeError('Historical construction mismatch')
        meta=load(OLD/'data/physical'/case/'problem.json')
        integrals=OLD/'data/physical'/case/'integrals.npz'
        request=OLD/'TN-controls-v1'/case/'final-readout/request.json'
        if meta['problem_id']!=d['problem_id'] or load(request)['problem_id']!=d['problem_id'] or sha256(integrals.read_bytes())!=d['integrals_SHA256']:
            raise RuntimeError('Tuned physical source mismatch')
        costfile=OLD/c['optimization_source']
        if not costfile.is_file():raise RuntimeError('Actual old training record unavailable')
        rows.append(dict(case=case,status='AVAILABLE',problem_id=d['problem_id'],norb=d['norb'],nelec=d['nelec'],
            operator=str(file),operator_SHA256=s['operator_sha256'],construction=str(construction),
            construction_SHA256=s['construction_sha256'],physical_SHA256=sha256((OLD/'data/physical'/case/'problem.json').read_bytes()),
            old_optimization_record=str(costfile),old_optimization_record_SHA256=sha256(costfile.read_bytes()),
            old_surrogate_coordinate_slots=c['surrogate_optimized_coordinate_slots'],
            nonredundant_independent_parameters=c['nonredundant_independent_parameters']))
    if len(rows)!=4 or sum(r['status']=='AVAILABLE' for r in rows)!=3:
        raise RuntimeError('Expected all4 cases/3same-problem tuned controls; inspect rather than silently expand')
    return rows


def seed(cid,rep):return int(digest(['N30_EXISTING_TUNED_NEW_MODEL_DRAW_V1',cid,rep])[:8],16)


def configured():
    legacy.LOCAL=SAMPLING;legacy.REMOTE=REMOTE;legacy.SHOTS=1000
    return legacy


def prepare():
    storage=legacy.capacity();p,_=n29.verified();old=load(n29.LUCJ/'protocol.json')
    if LOCAL.exists():raise RuntimeError('Existing N30; no overwrite/restart')
    test=ROOT/'qualification-N30-preexecution-v1-tests.xml'
    suites=list(ET.parse(test).getroot().iter('testsuite'))
    if sum(int(x.get('tests',0)) for x in suites)!=3 or any(int(x.get('failures',0)) or int(x.get('errors',0)) for x in suites):
        raise RuntimeError('All3 N30 qualification tests required')
    sources=source_inventory(p['physical']);module=legacy.qualified_module();cpu,wall=process_time(),perf_counter()
    LOCAL.mkdir();SAMPLING.mkdir()
    for name in ('worker.py','primitive_plan.py','base_kernel.py'):write(SAMPLING/name,(n29.LUCJ/name).read_bytes())
    circuits=[];cells=[];resources=[]
    for s in sources:
        if s['status']!='AVAILABLE':
            for rep in REPLICATES:
                cell=dict(case=s['case'],source=METHOD,method='ORIGINAL',problem_id=s['problem_id'],replicate=rep,status='MISSING_SAME_PROBLEM_SOURCE')
                cell['cell_id']=digest(['N30',cell]);cells.append(cell)
            continue
        with np.load(s['operator'],allow_pickle=False) as z:
            op=ffsim.UCJOpSpinBalanced(z['diag_coulomb_mats'],z['orbital_rotations'],z['final_orbital_rotation'])
        if op.n_reps!=1:raise RuntimeError('Existing tuned L1 required')
        qc=ffsim.qiskit.PRE_INIT.run(make_circuit(op,s['norb'],tuple(s['nelec'])))
        stream=BytesIO();qpy.dump(qc,stream);blob=stream.getvalue()
        path=LOCAL/'sources'/s['case']/'source.qpy';write(path,blob)
        restored=qpy.load(BytesIO(blob))[0]
        cfg,values=module.plan(restored,s['norb'],tuple(s['nelec']))
        cid=digest(dict(problem_id=s['problem_id'],QPY_SHA256=sha256(blob),decoder=list(range(2*s['norb'])),qualified_plan_SHA256=old['plan_SHA256']))
        folder=SAMPLING/'circuits'/cid;folder.mkdir(parents=True)
        payload=arrays(**values);write(folder/'arrays.npz',payload)
        requests=[dict(id=digest(['N30',cid,rep]),replicate_label=rep,seed=seed(cid,rep),shots=1000) for rep in REPLICATES]
        cfg.update(arrays_SHA256=sha256(payload),worker_SHA256=old['worker_SHA256'],base_kernel_SHA256=old['base_SHA256'],
            source_QPY_SHA256=sha256(blob),physical_problem_id=s['problem_id'],sampling_requests=requests)
        cfg['input_id']=digest(cfg);save(folder/'config.json',cfg)
        dim=math.prod(math.comb(s['norb'],e) for e in s['nelec'])
        circuit=dict(circuit_id=cid,case=s['case'],source=METHOD,problem_id=s['problem_id'],norb=s['norb'],nelec=s['nelec'],physical_SHA256=s['physical_SHA256'],
            original_QPY=str(path),original_QPY_SHA256=sha256(blob),decoder=list(range(2*s['norb'])),config_SHA256=sha256((folder/'config.json').read_bytes()),
            arrays_SHA256=sha256(payload),input_id=cfg['input_id'],sampling_requests=requests,sector_dimension=dim,
            primitive_count=len(cfg['ops']),estimated_work=dim*len(cfg['ops']),backend='A100')
        circuits.append(circuit);points=[]
        for level in (1,3):
            prep,samp,res=compile_views(restored,level);files={}
            for role,c in [('preparation',prep),('sampling',samp)]:
                out=BytesIO();qpy.dump(c,out);path=LOCAL/'sources'/s['case']/f'opt{level}-{role}.qpy';write(path,out.getvalue())
                files[role]=dict(path=str(path),SHA256=sha256(out.getvalue()))
            points.append(dict(resources=res,circuits=files))
        resources.append(dict(case=s['case'],method=METHOD,compiler_points=points,
            surrogate_coordinate_slots=s['old_surrogate_coordinate_slots'],independent_rank=s['nonredundant_independent_parameters']))
        for req in requests:
            cell=dict(case=s['case'],source=METHOD,method='ORIGINAL',problem_id=s['problem_id'],replicate=req['replicate_label'],status='PLANNED',
                circuit_id=cid,draw_id=req['id'],selected_resources=points[0]['resources'])
            cell['cell_id']=digest(['N30',cell]);cells.append(cell)
    old_seeds={r['seed'] for r in p['cpu_requests']}|{r['seed'] for c in old['circuits'] for r in c['sampling_requests']}
    seeds=[r['seed'] for c in circuits for r in c['sampling_requests']]
    if len(seeds)!=9 or len(set(seeds))!=9 or old_seeds&set(seeds):raise RuntimeError('N30 new seed collision')
    lanes=[[],[]];loads=[0,0]
    for c in sorted(circuits,key=lambda c:(-c['estimated_work'],c['circuit_id'])):
        lane=min(range(2),key=lambda j:loads[j]);lanes[lane].append(c['circuit_id']);loads[lane]+=c['estimated_work']
    sub=dict(old,version='N30_EXISTING_TUNED_LUCJ_NEW1000_DRAWS_V1',cells=cells,circuits=circuits,GPU_lanes=lanes,
        maximum_new_model_shots=9000,maximum_logical_model_shots=9000,selection='PREVIOUSLY_SELECTED_TUNED_SOURCE_NO_NEW_OPTIMIZATION',
        namespace_override=str(SAMPLING),remote_override=REMOTE,readouts=['N30_R32_CAP','N30_COMMON_R_ALL6METHODS'])
    sub.pop('N29_adapter_SHA256',None)
    sub['N30_adapter_SHA256']=sha256(Path(__file__).read_bytes())
    legacy.init_ledger(SAMPLING/'draw-ledger.sqlite',cells,circuits);save(SAMPLING/'protocol.json',sub)
    save(LOCAL/'protocol.json',dict(version='N30_ADD_MISSING_TUNED_CONTROL_V1',sources=sources,physical=p['physical'],resources=resources,
        source_protocol_SHA256=sha256((n29.LOCAL/'protocol.json').read_bytes()),N29_readout_summary_SHA256=sha256((ROOT/'analysis/N29-readout-120.json').read_bytes()),
        code_SHA256=sha256(Path(__file__).read_bytes()),sampling_protocol_SHA256=sha256((SAMPLING/'protocol.json').read_bytes()),
        plan_SHA256=sha256((ROOT/'analysis/N30_OPTIMIZED_CONTROL_PLAN_JA.md').read_bytes()),tests_SHA256=sha256(test.read_bytes()),
        helper_SHA256={name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in ('prefix_performance_sampling.py','finite_source_readout.py','collect_prefix_performance.py','slater_prefix_preflight.py')},
        library=p['library'],library_SHA256=p['library_SHA256'],prepare_CPU_seconds=process_time()-cpu,prepare_wall_seconds=perf_counter()-wall,
        new_model_samples_cap=9000,QPU_shots=0,new_optimizers=0,readout_endpoints=144,max_R=32,max_D=1024,root_seconds=20,batch_seconds=180,
        expected_CUDA_cupy='13.6.0',new_run_cap_bytes=32*2**20,storage_before=storage,
        original_paper_native_TN_reproduction=False,goal_complete=False))
    print('N30_FROZEN3_SOURCES9_NEW_DRAWS3_MISSING_NO_RETRAINING',flush=True)


def verified():
    p=load(LOCAL/'protocol.json')
    if sha256(Path(__file__).read_bytes())!=p['code_SHA256'] or sha256((SAMPLING/'protocol.json').read_bytes())!=p['sampling_protocol_SHA256']:
        raise RuntimeError('Frozen N30 drift')
    if sha256((ROOT/'analysis/N29-readout-120.json').read_bytes())!=p['N29_readout_summary_SHA256']:
        raise RuntimeError('Bound N29 baseline changed')
    for name,h in p['helper_SHA256'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=h:raise RuntimeError('Frozen helper drift')
    for s in p['sources']:
        if s['status']=='AVAILABLE':
            for name,key in [('operator','operator_SHA256'),('construction','construction_SHA256'),('old_optimization_record','old_optimization_record_SHA256')]:
                if sha256(Path(s[name]).read_bytes())!=s[key]:raise RuntimeError('Historical source/cost record changed')
    configured().verified_protocol()
    return p


def submit():verified();configured().submit()
def status():verified();configured().status()


def collect():
    p=verified()
    from . import collect_prefix_performance as collector
    collector.LOCAL=SAMPLING;collector.REMOTE=REMOTE;collector.collect()
    for c in load(SAMPLING/'protocol.json')['circuits']:
        f=SAMPLING/'collected/circuits'/c['circuit_id']/'output/result.json'
        if f.exists() and load(f)['cupy']!=p['expected_CUDA_cupy']:raise RuntimeError('Unqualified CUDA runtime')


if __name__=='__main__':
    import sys
    {'prepare':prepare,'submit':submit,'status':status,'collect':collect}[sys.argv[1]]()
