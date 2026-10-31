"""N18: connect existing half-register observations to the qualified HCI readout.

No fabricated joint bitstrings, no new shots, no modification to F2/H1/N17.
This is a historical single-draw screen, not independent confirmation.
"""
import argparse
from io import BytesIO
from pathlib import Path
import shlex
import shutil
import subprocess
import numpy as np
from qiskit import qpy
from .paths import ROOT
from .io import load,save,write,sha256,digest
from .prefix_performance_sampling import capacity
from .native_source_control_plan import initial_strings
from .sample_universe_audit import allowed_pool
from .analyze_marginal import audit_case
from .prefix_performance_ci import PARENT
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as LAUNCHER

LOCAL=ROOT/'runs/N18-HALF-SOURCE-NATIVE-SCREEN-v1'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/N18-half-source-native-v1'
HALF_ROOTS={'F2':ROOT/'runs/F2-POSITIVE-MARGINAL-v1','H1':ROOT/'runs/H1-HALF-REMAP-v1'}


def half_pool(ids,counts,n,electrons,hf):
    ids=np.asarray(ids);counts=np.asarray(counts)
    if ids.ndim!=1 or counts.ndim!=1 or len(ids)!=len(counts) or ids.dtype.kind not in 'iu' or counts.dtype.kind not in 'iu':
        raise ValueError('Explicit half-register integer observations required')
    if len(set(map(int,ids)))!=len(ids) or int(counts.sum())!=100000 or any(int(c)<=0 for c in counts):
        raise ValueError('Unique positive half counts with actual100k shots required')
    if any(int(x)<0 or int(x)>=1<<n or int(x).bit_count()!=electrons for x in ids):
        raise ValueError('Half counts outside registered sector')
    scores={int(x):int(c) for x,c in zip(ids,counts)}
    ranked=sorted([hf]+sorted((x for x in scores if x!=hf),key=lambda x:(-scores[x],x))[:127])
    return ranked,sorted(set(scores)|{hf})


def audit_qpy(path,expected):
    q=qpy.load(BytesIO(path.read_bytes()))[0]
    counters=dict(n_qubits=q.num_qubits,cnot_count=int(q.count_ops().get('cx',0)),
        cnot_depth=int(q.depth(filter_function=lambda x:x.operation.name=='cx')),total_depth=int(q.depth()),
        numeric_gate_coefficient_slots_not_rank=sum(len(x.operation.params) for x in q))
    if counters!=expected:raise RuntimeError('Actual stored half QPY resource recount mismatch')
    return dict(path=str(path),SHA256=sha256(path.read_bytes()),resources=counters)


def half_source(method,case,definition,source_binding):
    root=HALF_ROOTS[method];manifest=load(root/'manifest.json');record=manifest['cases'][case]
    d=root/'remote-output/outputs'/case;draw=load(d/'draw-result.json');payload=d/'half-counts.npz'
    binding=record['source_initialization_record']
    if binding['SHA256']!=source_binding['construction_SHA256'] or binding['path']!=source_binding['construction'] or sha256(Path(binding['path']).read_bytes())!=binding['SHA256']:
        raise RuntimeError('Half/full source initialization is not the same registered parent')
    if manifest['freeze']['input_SHA256']['inputs/'+case+'/operator.npz']!=source_binding['operator_SHA256']:
        raise RuntimeError('Half/full source operator mismatch')
    if manifest['freeze']['input_SHA256']['inputs/'+case+'/integrals.npz']!=definition['integrals_SHA256']:
        raise RuntimeError('Half/full Hamiltonian or orbital gauge mismatch')
    if draw['physical_problem_id']!=definition['problem_id'] or draw['run_id']!=record['run_id'] or draw['actual_QPU_shots']!=0 or draw['known_model_shots']!=100000:
        raise RuntimeError('Actual half draw physical problem/work identity mismatch')
    if sha256(payload.read_bytes())!=draw['counts_SHA256']:raise RuntimeError('Original half observation changed')
    with np.load(payload,allow_pickle=False) as a:
        if set(a.files)!={'half_strings','counts'}:raise RuntimeError('Do not infer a joint-data schema from half counts')
        strings,pool=half_pool(a['half_strings'],a['counts'],definition['norb'],definition['nelec'][0],(1<<definition['nelec'][0])-1)
    evidence=[]
    if method=='F2':
        audit=audit_case(case,record,manifest)
        if not audit['draw_verified']:raise RuntimeError('All conditional members/counts must be verified')
        members=load(d/'members.json')
        for member in members['members']:
            for role in ['preparation','sampling']:
                path=d/'circuits'/f"{member['member']:03d}-{role}.qpy"
                if sha256(path.read_bytes())!=member['QPY_SHA256'][role]:raise RuntimeError('F2 conditional member changed')
                evidence.append(dict(path=str(path),SHA256=sha256(path.read_bytes()),resources=member['resources'][role]))
        resources=draw['sampling_resources_shot_weighted_mean'];members_count=len(members['members'])
        construction_CPU=draw['pattern_and_all_compilation_CPU_seconds']
    else:
        for role in ['preparation','sampling']:
            path=d/(role+'.qpy')
            if sha256(path.read_bytes())!=draw['QPY_SHA256'][role]:raise RuntimeError('H1 QPY changed')
            evidence.append(audit_qpy(path,draw['resources'][role]))
        resources=draw['resources']['sampling'];members_count=1;construction_CPU=draw['compilation_CPU_seconds']
    result=dict(status='AVAILABLE',kind='QUANTUM_HALF_COUNTS',method=method,expected_problem_id=definition['problem_id'],
        payload=str(payload),payload_SHA256=sha256(payload.read_bytes()),source_result=str(d/'draw-result.json'),source_result_SHA256=sha256((d/'draw-result.json').read_bytes()),
        source_initialization_record=binding,operator_SHA256=source_binding['operator_SHA256'],
        actual_historical_model_shots=100000,half_configuration_count=100000,sampler_qubits=definition['norb'],
        half_counts_not_quantum_joint_data=True,all_member_QPY_evidence=evidence,member_count=members_count,
        sampler_resources=resources,source_model_input_coordinates_retained_not_independent_rank=True,independent_ansatz_parameter_count=None,
        source_pattern_or_compilation_CPU_seconds=construction_CPU,source_sampling_CPU_seconds=draw['sampling_CPU_seconds'],
        new_model_shots=0,new_QPU_shots=0)
    return result,strings,pool


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N18 source screen; no overwrite')
    capacity();n17=load(ROOT/'analysis/N17-whole-study-summary.json')
    if n17['status']!='N17_ALL108_CELLS_216_READOUTS_4_HF_CONTROLS_REVIEWED':raise RuntimeError('Close the whole N17 first')
    parent=load(PARENT/'protocol.json');n15=load(ROOT/'runs/N15-SLATER-PREFIX-ROUNDING-PREFLIGHT-v1/protocol.json')
    source_binding={s['case']:s for s in n15['sources'] if s['status']=='AVAILABLE' and s['method'].endswith('L1')}
    plans=[]
    for case,definition in sorted(parent['physical'].items()):
        full=next(p['source'] for p in parent['plans'] if p['case']==case and p['method']=='C5_C_LUCJ_COMPRESSED_L1')
        for key,name in [('payload','payload_SHA256'),('source_result','source_result_SHA256')]:
            if sha256(Path(full[key]).read_bytes())!=full[name]:raise RuntimeError('Original full source changed')
        if source_binding[case]['source_result_SHA256']!=full['source_result_SHA256']:
            raise RuntimeError('Full count source and verified operator differ')
        full=dict(full,actual_historical_model_shots=100000,half_configuration_count=200000,sampler_qubits=2*definition['norb'],
                  source_model_input_coordinates_retained_not_independent_rank=True,independent_ansatz_parameter_count=None)
        sources=[('FULL_LUCJ_L1',full,initial_strings(full,definition['norb'],tuple(definition['nelec'])),
                  sorted(allowed_pool(full,definition['norb'],tuple(definition['nelec']))))]
        for method in ['F2','H1']:sources.append((method,*half_source(method,case,definition,source_binding[case])))
        hf=(1<<definition['nelec'][0])-1
        for method,source,strings,pool in sources:
            row=dict(case=case,method=method,source=source,selection_mode='SAMPLE_CONSTRAINED',initial_strings=[hf],
                allowed_strings=pool,frequency_strings=strings,HF_string=hf,requested_readouts=['SOURCE_FREQUENCY_R128','JOINT_TOP_D16384'])
            row['task_id']=digest(row);plans.append(row)
        row=dict(case=case,method='UNRESTRICTED_HF_HCI',source=dict(status='AVAILABLE',kind='CLASSICAL_HF_ONLY',expected_problem_id=definition['problem_id']),
            selection_mode='UNRESTRICTED_HF_CONTROL',initial_strings=[hf],allowed_strings=None,frequency_strings=None,HF_string=hf,requested_readouts=['JOINT_TOP_D16384'])
        row['task_id']=digest(row);plans.append(row)
    if len(plans)!=16 or len({p['task_id'] for p in plans})!=16:raise RuntimeError('All4x3sources plus4HF controls required')
    LOCAL.mkdir();(LOCAL/'pipeline').mkdir()
    actual=ROOT/'runs/N17-INDEPENDENT-PREFIX-CI-A100-v1';previous=load(actual/'protocol.json')
    worker=(actual/'worker.py').read_bytes()
    if sha256(worker)!=previous['worker_SHA256']:raise RuntimeError('Qualified common readout changed')
    write(LOCAL/'worker.py',worker)
    for name,value in previous['pipeline_SHA256'].items():
        file=actual/'pipeline'/name
        if sha256(file.read_bytes())!=value:raise RuntimeError('Qualified numerical helper changed')
        shutil.copyfile(file,LOCAL/'pipeline'/name)
    spec={k:previous[k] for k in ['physical','common_runtime','packages','native_binary','native_lock','qualified_external_SHA256','minimal_qualification','pipeline_SHA256','sparse_library']}
    spec.update(version='N18_EXISTING_HALF_FULL_SOURCES_COMMON_HCI_V1',remote_root=REMOTE,plans=plans,
        policy=dict(previous['policy'],global_wall_seconds=1800),worker_SHA256=sha256(worker),
        source_N17_audit_SHA256=sha256((ROOT/'analysis/N17-whole-study-summary.json').read_bytes()),
        preparer_SHA256=sha256(Path(__file__).read_bytes()),tests_SHA256=sha256((ROOT/'qualification-N18-half-source-v1-tests.xml').read_bytes()),
        planned=16,available=16,missing=0,source_draws_per_method_per_case=1,readout_endpoints=28,
        observations='EXISTING100K_COUNTS_PER_SOURCE_REUSED_NOT_NEW_INDEPENDENT_DRAWS',
        same_quantum_shots_not_same_half_configuration_count=True,
        source_model_parameters_not_claimed_reduced=True,
        half_counts_not_encoded_as_quantum_joint_observations=True,
        actual_work_not_matched=True,new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,goal_complete=False)
    save(LOCAL/'protocol.json',spec);print('N18_FROZEN16TASKS28READOUTS_NO_NEW_SHOTS',flush=True)


def submit():
    p=load(LOCAL/'protocol.json');capacity()
    if (LOCAL/'submission-intent.json').exists():raise RuntimeError('Existing N18 dispatch intent; no duplicate')
    if sha256((LOCAL/'worker.py').read_bytes())!=p['worker_SHA256']:raise RuntimeError('Frozen worker changed')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=30).strip()!='NEW':raise RuntimeError('Existing remote N18 work')
    ssh('mkdir '+shlex.quote(REMOTE),timeout=30)
    for name in ['worker.py','protocol.json']:send(LOCAL/name,REMOTE+'/'+name)
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(LOCAL/'pipeline'),HOST+':'+REMOTE+'/'],check=True,timeout=50)
    wrap=shlex.join(['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',LAUNCHER+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE+'/protocol.json'])
    command=shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=8G','--time=00:35:00',
        '--job-name=frontier-N18-half-source-HCI','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err','--wrap='+wrap])
    save(LOCAL/'submission-intent.json',dict(command=command,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    job=ssh(command,timeout=30).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Uncertain dispatch response; no blind re-submit')
    save(LOCAL/'job.json',dict(job_id=job,remote=REMOTE,GPU_allocations=0));print('N18_CPU_ONLY_SUBMITTED',job,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit']);a=p.parse_args();globals()[a.action]()
