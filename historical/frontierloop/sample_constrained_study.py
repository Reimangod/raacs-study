"""N13: all existing sources with the qualified observed-pool restriction.

All molecular starts are HF. Four fresh unrestricted HF controls use the same
private binary with the opt-in unset. Source amplitudes only define membership;
they are not injected as HCI coefficients. No new quantum sampling is involved.
"""
import argparse,difflib,shlex,shutil,subprocess
from pathlib import Path
from .paths import ROOT
from .io import load,save,write,sha256,digest,space
from .sample_universe_audit import allowed_pool
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as BASE

LOCAL=ROOT/'runs/N13-SAMPLE-CONSTRAINED-ALL-SOURCES-v1'
REMOTE='/share/reiwatanabe/ceo-sample-constrained-all-sources-20261026-v1'
PARENT=ROOT/'frontierloop/small_initial_pool_worker.py'


def replace(text,before,after):
    if text.count(before)!=1:raise ValueError('Qualified executor context drift')
    return text.replace(before,after)


def generate(base):
    text=replace(base,'def native(folder,fcidump,nelec,strings):','def native(folder,fcidump,nelec,strings,allowed=None):')
    text=replace(text,"        lines=['nroots 1'", "        child_env=os.environ.copy();child_env.pop('CEO_DICE_SAMPLE_POOL_V1',None)\n"
        "        if allowed is not None:\n"
        "            if allowed!=sorted(set(allowed)) or not set(strings)<=set(allowed):raise RuntimeError('Invalid prepared pool')\n"
        "            (folder/'AllowedHalfDets.bin').write_bytes(b''.join(int(x).to_bytes(16,'big') for x in allowed))\n"
        "            child_env['CEO_DICE_SAMPLE_POOL_V1']='1'\n"
        "        lines=['nroots 1'")
    text=replace(text,'env=os.environ.copy(),stdout=log','env=child_env,stdout=log')
    text=replace(text,"        if len(energy)!=8:raise RuntimeError('One diagnostic native energy required')", 
        "        if len(energy)!=8:raise RuntimeError('One diagnostic native energy required')\n"
        "        if allowed is not None:\n"
        "            if [int(x) for x in (folder/'CEO-sample-pool-v1.txt').read_text().split()]!=allowed:raise RuntimeError('Actual pool differs')\n"
        "            if any(a not in allowed or b not in allowed for a,b,_ in pairs):raise RuntimeError('Native outside observed pool')\n"
        "            attempt['allowed_pool_SHA256']=sha(folder/'AllowedHalfDets.bin')\n"
        "            attempt['allowed_pool_size']=len(allowed)\n"
        "        elif (folder/'CEO-sample-pool-v1.txt').exists():raise RuntimeError('Opt-in leaked into unrestricted control')")
    text=replace(text,"native(root/'qualification',q['fcidump_path'],(1,1),[1])","native(root/'qualification',q['fcidump_path'],(1,1),[1],allowed=[1,2])")
    text=replace(text,"new_model_shots=0,new_QPU_shots=0)\n        if plan", "new_model_shots=0,new_QPU_shots=0,selection_mode=plan['selection_mode'])\n        if plan")
    text=replace(text,"native(folder,definition['fcidump_path'],tuple(definition['nelec']),plan['initial_strings'])",
        "native(folder,definition['fcidump_path'],tuple(definition['nelec']),plan['initial_strings'],allowed=plan['allowed_strings'])")
    text=text.replace("print('N7',","print('N13',")
    text=replace(text,"status='ALL_SMALL_INITIAL_POOL_SOURCES_TERMINAL_WITH_MISSINGNESS'","status='ALL64_SAMPLE_CONSTRAINED_AND_UNRESTRICTED_HF_PLANS_TERMINAL_WITH_MISSINGNESS'")
    return text


def plans_from_existing(prior,policy,binary_sha):
    source_plans=[p for p in prior['plans'] if p['initialization_arm']=='PRODUCT_R8']
    if len(source_plans)!=60 or sum(p['source']['status']=='AVAILABLE' for p in source_plans)!=53:raise ValueError('All60 source inventory required')
    plans=[]
    for old in source_plans:
        case=old['case'];source=old['source'];d=prior['physical'][case];hf=(1<<d['nelec'][0])-1
        if source['expected_problem_id']!=d['problem_id']:raise ValueError('Source/physical mismatch')
        pool=None
        if source['status']=='AVAILABLE':
            for name,key in [('payload','payload_SHA256'),('source_result','source_result_SHA256')]:
                if name in source and sha256(Path(source[name]).read_bytes())!=source[key]:raise ValueError('Old source changed')
            pool=sorted(allowed_pool(source,d['norb'],tuple(d['nelec'])))
            if len(pool)>100000 or hf not in pool:raise ValueError('Unqualified pool')
        row=dict(case=case,method=old['method'],source=source,selection_mode='SAMPLE_CONSTRAINED',
            initial_strings=[hf] if pool is not None else None,allowed_strings=pool,HF_string=hf,
            initial_joint_dimension=1 if pool is not None else None,
            observed_half_pool_size=len(pool) if pool is not None else None)
        row['task_id']=digest([row,policy,d,binary_sha]);plans.append(row)
    for old in source_plans:
        if old['method']!='CLASSICAL_HF_ONLY':continue
        d=prior['physical'][old['case']];hf=(1<<d['nelec'][0])-1
        row=dict(case=old['case'],method='UNRESTRICTED_HF_HCI',source=old['source'],selection_mode='UNRESTRICTED_HF_CONTROL',
            initial_strings=[hf],allowed_strings=None,HF_string=hf,initial_joint_dimension=1,observed_half_pool_size=None)
        row['task_id']=digest([row,policy,d,binary_sha]);plans.append(row)
    if len(plans)!=64 or len({p['task_id'] for p in plans})!=64:raise ValueError('64 unique plans required')
    return plans


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N13 intent; do not overwrite')
    space();p6=load(ROOT/'runs/N6-NATIVE-SOURCE-CONTROLS-v1/protocol.json')
    qfile=ROOT/'analysis/N12-sample-constraint-qualification-summary.json';q=load(qfile)
    if q['status']!='N12_ALL24_AUDITED_19_PHYSICS_5_EXPECTED_REJECTIONS':raise RuntimeError('Full N12 qualification required')
    prior=load(ROOT/'runs/N10-PAIRED-INITIAL-POOL-v1/protocol.json')
    old=load(ROOT/'runs/N7-SMALL-INITIAL-POOL-v1/protocol.json')
    if sha256(PARENT.read_bytes())!=old['worker_SHA256']:raise RuntimeError('Parent executor drift')
    policy=dict(p6['policy'],initial_marginal_R_cap=None,initial_joint_D_cap=1,
        global_wall_seconds=3900,readout_arms=['JOINT_TOP_D16384'],
        source_universe='FULL_SOURCE_ALPHA_BETA_UNION_PLUS_HF_NO_R_TRUNCATION',
        native_start='HF_ONLY_FOR_ALL_SOURCES',source_coefficient_initialization=False,
        final_readout='SAME_H_PLUS_0p1_S2_JOINT_TOP_D16384_NO_OO')
    plans=plans_from_existing(prior,policy,q['binary_SHA256'])
    LOCAL.mkdir();(LOCAL/'pipeline').mkdir()
    for name,value in p6['pipeline_SHA256'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=value:raise RuntimeError('Qualified helper drift')
        shutil.copyfile(ROOT/'frontierloop'/name,LOCAL/'pipeline'/name)
    base=PARENT.read_text();worker=generate(base);write(LOCAL/'worker.py',worker.encode())
    write(LOCAL/'worker-diff.patch',''.join(difflib.unified_diff(base.splitlines(True),worker.splitlines(True),fromfile='qualified-N7',tofile='N13-observed-pool')).encode())
    external=dict(p6['qualified_external_SHA256']);external[q['binary']]=q['binary_SHA256']
    external['/share/reiwatanabe/ceo-sample-constraint-qualification-20261026-v1/terminal.json']=q['terminal_SHA256']
    spec=dict(p6,version='N13_ALL60_SAMPLE_CONSTRAINED_PLUS4_MATCHED_HF_CONTROLS_V1',remote_root=REMOTE,
        plans=plans,policy=policy,native_binary=q['binary'],qualified_external_SHA256=external,
        worker_SHA256=sha256((LOCAL/'worker.py').read_bytes()),mechanical_parent_SHA256=sha256(PARENT.read_bytes()),
        preparer_SHA256=sha256(Path(__file__).read_bytes()),pool_builder_SHA256=sha256((ROOT/'frontierloop/sample_universe_audit.py').read_bytes()),
        qualification_SHA256=sha256(qfile.read_bytes()),source_protocol_SHA256=sha256((ROOT/'runs/N10-PAIRED-INITIAL-POOL-v1/protocol.json').read_bytes()),
        tests_SHA256=sha256((ROOT/'qualification-sample-constrained-study-v1-tests.xml').read_bytes()),
        planned=64,available=57,missing=7,
        scope='QUALIFIED_EQ9_UNIVERSE_COMPONENT_COMMON_ADAPTATION_NOT_FULL_HCI_HSQD_PAPER_OR_QUANTUM_ADVANTAGE',
        source_observations='EXISTING_COUNTS_REUSED_NOT_INDEPENDENT_DRAWS',
        actual_initial_D_equal=True,actual_final_D_equal=False,actual_CPU_equal=False,
        new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,goal_complete=False)
    save(LOCAL/'protocol.json',spec);print('N13_FROZEN64_PLANS57_AVAILABLE7_MISSING',flush=True)


def submit():
    if (LOCAL/'submission-started.json').exists():raise RuntimeError('Existing submission intent; reconcile exact handle')
    p=load(LOCAL/'protocol.json')
    if sha256((LOCAL/'worker.py').read_bytes())!=p['worker_SHA256']:raise RuntimeError('Frozen executor changed')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('Remote root exists')
    ssh('mkdir '+shlex.quote(REMOTE),timeout=35)
    for name in ['worker.py','protocol.json']:send(LOCAL/name,REMOTE+'/'+name)
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(LOCAL/'pipeline'),HOST+':'+REMOTE+'/'],check=True,timeout=50)
    wrap=shlex.join(['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',BASE+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE+'/protocol.json'])
    cmd=shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=8G','--time=01:10:00',
        '--job-name=ceo-sample-constrained-all-sources-v1','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err','--wrap='+wrap])
    save(LOCAL/'submission-started.json',dict(command=cmd,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    job=ssh(cmd,timeout=35).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Unknown submission; no blind retry')
    save(LOCAL/'job.json',dict(job_id=job,remote=REMOTE,GPU_allocations=0));print('N13_SUBMITTED_CPU_ONLY',job,flush=True)


def status():
    from . import paired_pool_study as old
    old.LOCAL=LOCAL;old.REMOTE=REMOTE;old.status()


def collect():
    from . import collect_small_pool_bundle as old
    old.LOCAL=LOCAL;old.REMOTE=REMOTE;old.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','submit','status','collect'])
    a=parser.parse_args();globals()[a.action]()
