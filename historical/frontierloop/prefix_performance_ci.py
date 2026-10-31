"""N17 common readouts, all sources and replicates, not a policy search.

The original N13 worker/numerical kernels are preserved. This mechanical
successor evaluates observed-frequency CI BEFORE and independently of HCI.
"""
import argparse
import difflib
from pathlib import Path
import shlex
import shutil
import subprocess
from .paths import ROOT
from .io import load,save,write,sha256,digest,space
from .prefix_performance_sampling import LOCAL as SAMPLES,verified_protocol,db,capacity
from .native_hci_plan import frequency_strings
from .sample_universe_audit import allowed_pool
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as LAUNCHER

LOCAL=ROOT/'runs/N17-INDEPENDENT-PREFIX-CI-v1'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/N17-independent-prefix-ci-v1'
PARENT=ROOT/'runs/N13-SAMPLE-CONSTRAINED-ALL-SOURCES-v1'
BASE_LOCAL=LOCAL
BASE_REMOTE=REMOTE


def configure(batch):
    global LOCAL,REMOTE
    if batch not in ['ALL','M4','A100']:raise ValueError('Unknown fixed execution batch')
    LOCAL=BASE_LOCAL if batch=='ALL' else ROOT/('runs/N17-INDEPENDENT-PREFIX-CI-'+batch+'-v1')
    REMOTE=BASE_REMOTE if batch=='ALL' else BASE_REMOTE+'-'+batch.lower()


def batch_cells(sampling,batch):
    if batch=='ALL':return sampling['cells']
    case_backend={}
    for circuit in sampling['circuits']:
        case=circuit['case'];backend=circuit['backend']
        if case in case_backend and case_backend[case]!=backend:raise ValueError('Inconsistent physical execution assignment')
        case_backend[case]=backend
    # Execution assignment, not energy/ranking. Missing variants of a case stay
    # in that same batch; waiting is never reclassified as scientific missingness.
    return [c for c in sampling['cells'] if case_backend[c['case']]==batch]


def replace(text,before,after):
    if text.count(before)!=1:raise ValueError('Qualified N13 worker context drift')
    return text.replace(before,after)


def generate():
    base=(PARENT/'worker.py').read_text();old=load(PARENT/'protocol.json')
    if sha256(base.encode())!=old['worker_SHA256']:raise RuntimeError('N13 parent changed')
    text=replace(base,"            try:\n                definition=spec['physical'][plan['case']]",
        "            if plan.get('frequency_strings') is not None:\n"
        "                try:\n"
        "                    observed=plan['frequency_strings']\n"
        "                    row['arms']['SOURCE_FREQUENCY_R128']=evaluate(plan['case'],[(a,b) for a in observed for b in observed],plan['task_id']+':SOURCE_FREQUENCY_R128')\n"
        "                except Exception as exc:row['errors']['SOURCE_FREQUENCY_R128']=dict(error=repr(exc),traceback=traceback.format_exc())\n"
        "                save(folder/'progress.json',row)\n"
        "            try:\n                definition=spec['physical'][plan['case']]")
    text=replace(text,"len(row['arms'])==len(spec['policy']['readout_arms'])","set(row['arms'])==set(plan['requested_readouts'])")
    text=replace(text,'>128*1024**2','>512*1024**2')
    text=replace(text,"print('N13',","print('N17_CI',")
    text=replace(text,"status='ALL64_SAMPLE_CONSTRAINED_AND_UNRESTRICTED_HF_PLANS_TERMINAL_WITH_MISSINGNESS'",
        "status='N17_ALL_FROZEN_DRAW_READOUTS_AND4_HF_CONTROLS_TERMINAL_WITH_MISSINGNESS'")
    return text


def draw_source(draw_id,case,definition):
    path=SAMPLES/'decoded'/(draw_id+'.npz');record=load(path.with_suffix('.json'))
    if sha256(path.read_bytes())!=record['decoded_SHA256'] or record['actual_shots']!=100000:
        raise ValueError('Actual completed observation required')
    source=dict(status='AVAILABLE',kind='QUANTUM_FULL_COUNTS',expected_problem_id=definition['problem_id'],
        payload=str(path),payload_SHA256=record['decoded_SHA256'],draw_id=draw_id,
        sampling_evidence_SHA256=sha256(path.with_suffix('.json').read_bytes()))
    import numpy as np
    with np.load(path,allow_pickle=False) as a:
        strings=frequency_strings(a['blocked_ids'],a['counts'],definition['norb'],(1<<definition['nelec'][0])-1)
    return source,strings,sorted(allowed_pool(source,definition['norb'],tuple(definition['nelec'])))


def prepare(batch='ALL'):
    if LOCAL.exists():raise RuntimeError('Existing N17 CI freeze, never overwrite')
    capacity();sampling=verified_protocol();parent=load(PARENT/'protocol.json')
    cells=batch_cells(sampling,batch);wanted={c['draw_id'] for c in cells if c.get('draw_id')}
    if batch=='ALL' and any((ROOT/('runs/N17-INDEPENDENT-PREFIX-CI-'+b+'-v1/protocol.json')).exists() for b in ['M4','A100']):
        raise RuntimeError('Split batch already frozen; do not duplicate its CI work through ALL')
    with db() as c:
        states=dict(c.execute('SELECT draw_id,status FROM draws'))
        if any(states.get(draw)!='COMPLETED' for draw in wanted):
            raise RuntimeError('All independent draws in this predetermined execution batch must be complete')
    LOCAL.mkdir();(LOCAL/'pipeline').mkdir()
    for name,value in parent['pipeline_SHA256'].items():
        file=PARENT/'pipeline'/name
        if sha256(file.read_bytes())!=value:raise RuntimeError('Qualified numerical helper changed')
        shutil.copyfile(file,LOCAL/'pipeline'/name)
    worker=generate();write(LOCAL/'worker.py',worker.encode())
    write(LOCAL/'worker-diff.patch',''.join(difflib.unified_diff((PARENT/'worker.py').read_text().splitlines(True),worker.splitlines(True),fromfile='qualified-N13',tofile='N17-frequency-before-native')).encode())
    plans=[];lookup={};logical=[]
    for cell in cells:
        definition=parent['physical'][cell['case']];hf=(1<<definition['nelec'][0])-1
        if definition['problem_id']!=cell['problem_id']:raise RuntimeError('Physical source mismatch')
        if cell['status']=='MISSING_SAME_PROBLEM_SOURCE':
            source=dict(status='MISSING_SAME_PROBLEM_SOURCE',expected_problem_id=cell['problem_id'])
            row=dict(case=cell['case'],method=cell['source']+':'+cell['method']+':'+str(cell['replicate']),source=source,
                selection_mode='MISSING',initial_strings=None,allowed_strings=None,frequency_strings=None,
                HF_string=hf,requested_readouts=['SOURCE_FREQUENCY_R128','JOINT_TOP_D16384'])
            row['task_id']=digest(row);plans.append(row)
        elif cell['draw_id'] not in lookup:
            source,strings,pool=draw_source(cell['draw_id'],cell['case'],definition)
            row=dict(case=cell['case'],method='DRAW:'+cell['draw_id'],source=source,selection_mode='SAMPLE_CONSTRAINED',
                initial_strings=[hf],allowed_strings=pool,frequency_strings=strings,HF_string=hf,
                requested_readouts=['SOURCE_FREQUENCY_R128','JOINT_TOP_D16384'])
            row['task_id']=digest(row);plans.append(row);lookup[cell['draw_id']]=row
        else:row=lookup[cell['draw_id']]
        logical.append(dict(cell_id=cell['cell_id'],case=cell['case'],source=cell['source'],method=cell['method'],
            replicate=cell['replicate'],draw_id=cell.get('draw_id'),task_id=row['task_id']))
    cases={cell['case'] for cell in cells}
    for case,d in sorted(parent['physical'].items()):
        if case not in cases:continue
        hf=(1<<d['nelec'][0])-1
        row=dict(case=case,method='UNRESTRICTED_HF_HCI',source=dict(status='AVAILABLE',kind='CLASSICAL_HF_ONLY',expected_problem_id=d['problem_id']),
            selection_mode='UNRESTRICTED_HF_CONTROL',initial_strings=[hf],allowed_strings=None,frequency_strings=None,
            HF_string=hf,requested_readouts=['JOINT_TOP_D16384'])
        row['task_id']=digest(row);plans.append(row)
    policy=dict(parent['policy'],global_wall_seconds=7200,readout_arms=['JOINT_TOP_D16384'],
        source_frequency_readout='INDEPENDENT_PRE_NATIVE_OBSERVED_MARGINAL_R128',
        readout_order='FREQUENCY_BEFORE_NATIVE_SO_NATIVE_FAILURE_CANNOT_HIDE_BASELINE',workspace_cap_bytes=512*2**20)
    spec={k:parent[k] for k in ['physical','common_runtime','packages','native_binary','native_lock','qualified_external_SHA256',
        'minimal_qualification','pipeline_SHA256','sparse_library']}
    spec.update(version='N17_INDEPENDENT_DRAWS_COMMON_TWO_READOUTS_V1',remote_root=REMOTE,plans=plans,logical_cells=logical,
        policy=policy,worker_SHA256=sha256(worker.encode()),preparer_SHA256=sha256(Path(__file__).read_bytes()),
        parent_protocol_SHA256=sha256((PARENT/'protocol.json').read_bytes()),sampling_protocol_SHA256=sha256((SAMPLES/'protocol.json').read_bytes()),
        tests_SHA256=sha256((ROOT/'qualification-N17-ci-batches-v2-tests.xml').read_bytes()),new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,
        actual_final_D_equal=False,actual_CPU_equal=False,final_CI_state_not_prepared_by_sampler=True,
        execution_batch=batch,full_logical_inventory_SHA256=digest(sampling['cells']),full_logical_inventory_count=108,
        other_batch_cells_waiting_or_separately_accounted=len(sampling['cells'])-len(cells),
        common_worker_logic_SHA256=sha256(worker.encode()),
        planned=len(plans),available=sum(p['source']['status']=='AVAILABLE' for p in plans),
        missing=sum(p['source']['status']!='AVAILABLE' for p in plans),goal_complete=False)
    save(LOCAL/'protocol.json',spec);print('N17_CI_FROZEN',len(plans),'plans',len(logical),'logical cells',flush=True)


def submit():
    p=load(LOCAL/'protocol.json');capacity()
    if (LOCAL/'submission-intent.json').exists():raise RuntimeError('Existing CI submission intent; reconcile exact job')
    if sha256((LOCAL/'worker.py').read_bytes())!=p['worker_SHA256']:raise RuntimeError('Frozen CI worker changed')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=30).strip()!='NEW':raise RuntimeError('Existing remote CI work')
    ssh('mkdir '+shlex.quote(REMOTE),timeout=30)
    for name in ['worker.py','protocol.json']:send(LOCAL/name,REMOTE+'/'+name)
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(LOCAL/'pipeline'),HOST+':'+REMOTE+'/'],check=True,timeout=50)
    wrap=shlex.join(['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',LAUNCHER+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE+'/protocol.json'])
    cmd=shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=8G','--time=02:05:00',
        '--job-name=frontier-N17-common-CI','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err','--wrap='+wrap])
    save(LOCAL/'submission-intent.json',dict(command=cmd,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    job=ssh(cmd,timeout=30).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Uncertain CI submit response; no blind retry')
    save(LOCAL/'job.json',dict(job_id=job,remote=REMOTE,GPU_allocations=0));print('N17_CI_SUBMITTED_CPU_ONLY',job,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit']);p.add_argument('--batch',choices=['ALL','M4','A100'],default='ALL')
    a=p.parse_args();configure(a.batch)
    prepare(a.batch) if a.action=='prepare' else submit()
