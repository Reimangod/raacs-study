"""N10: both equal-initial-dimension arms on ALL existing source variants."""
import argparse,difflib,shlex,shutil,subprocess
from pathlib import Path
from .paths import ROOT
from .io import load,save,write,sha256,digest,space
from .paired_pool_plan import source_pairs
from .paired_pool_executor import generate
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as BASE

LOCAL=ROOT/'runs/N10-PAIRED-INITIAL-POOL-v1'
REMOTE='/share/reiwatanabe/ceo-paired-initial-pool-20261026-v1'
PARENT=ROOT/'frontierloop/small_initial_pool_worker.py'


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N10; no overwrite or requeue')
    space();p6=load(ROOT/'runs/N6-NATIVE-SOURCE-CONTROLS-v1/protocol.json')
    n9file=ROOT/'analysis/N9-paired-input-qualification-summary.json';n9=load(n9file)
    if n9['status']!='ALL24_TINY_ADAPTER_CASES_AUDITED_20_PHYSICS_4_EXPECTED_REJECTIONS':raise RuntimeError('All N9 tiny qualifications required')
    oldp=load(ROOT/'runs/N7-SMALL-INITIAL-POOL-v1/protocol.json')
    if sha256(PARENT.read_bytes())!=oldp['worker_SHA256']:raise RuntimeError('Mechanical parent changed')
    r2file=ROOT/'runs/R2-COMMON-CI-v1/freeze.json';r2=load(r2file)
    n5=load(ROOT/'runs/N5-JOINT-SPARSE-CI-v1/protocol.json')
    sources=[dict(s) for s in r2['sources']]
    for p in n5['plans']:
        if p['method']=='CLASSICAL_HF_ONLY':
            sources.append(dict(p['source'],status='AVAILABLE',case=p['case'],method_id=p['method'],
                                expected_problem_id=p6['physical'][p['case']]['problem_id']))
    if len(sources)!=60 or len({(s['case'],s['method_id']) for s in sources})!=60 or sum(s['status']=='AVAILABLE' for s in sources)!=53:
        raise RuntimeError('All60 source inventory with53available7missing required')
    policy=dict(p6['policy'],initial_marginal_R_cap=8,initial_joint_D_cap=64,
        joint_input_rule='HF_MANDATORY_OBSERVED_PAIR_WEIGHT_DESC_LEX_TIE_PRODUCT_LEX_BACKFILL_EQUAL_ACTUAL_INITIAL_DIMENSION',
        global_wall_seconds=3900,readout_arms=['JOINT_TOP_D16384'],
        equal_initial_dimension=True,spin_flip_closure_not_guaranteed=True)
    plans=[]
    for s in sources:
        d=p6['physical'][s['case']];strings=product=paired=info=None
        if s['expected_problem_id']!=d['problem_id']:raise RuntimeError('Problem mismatch')
        if s['status']=='AVAILABLE':
            for name,key in [('payload','payload_SHA256'),('source_result','source_result_SHA256')]:
                if name in s and sha256(Path(s[name]).read_bytes())!=s[key]:raise RuntimeError('Source drift')
            strings,product,paired,info=source_pairs(s,d['norb'],tuple(d['nelec']))
        for arm in ['PRODUCT_R8','OBSERVED_JOINT_EQUAL_D']:
            row=dict(case=s['case'],method=s['method_id'],source=s,HF_string=(1<<d['nelec'][0])-1,
                initialization_arm=arm,initial_strings=strings,
                initial_pairs=paired if arm=='OBSERVED_JOINT_EQUAL_D' else None,
                expected_initial_pairs=paired if arm=='OBSERVED_JOINT_EQUAL_D' else product,
                initial_joint_dimension=len(product) if product is not None else None,selection_diagnostics=info)
            row['task_id']=digest([row,policy,d,n9['binary_SHA256']]);plans.append(row)
    LOCAL.mkdir();(LOCAL/'pipeline').mkdir()
    for name,value in p6['pipeline_SHA256'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=value:raise RuntimeError('Qualified kernel/helper drift')
        shutil.copyfile(ROOT/'frontierloop'/name,LOCAL/'pipeline'/name)
    base=PARENT.read_text();worker=generate(base)
    write(LOCAL/'worker.py',worker.encode());write(LOCAL/'worker-diff.patch',''.join(difflib.unified_diff(base.splitlines(True),worker.splitlines(True),fromfile='qualified-N7',tofile='N10-paired-input')).encode())
    external=dict(p6['qualified_external_SHA256']);external[n9['binary']]=n9['binary_SHA256']
    qualified='/share/reiwatanabe/ceo-paired-input-qualification-20261026-bootstrap-v2/terminal.json'
    external[qualified]=n9['terminal_SHA256']
    spec=dict(p6,version='N10_ALL_SOURCE_EQUAL_INITIAL_DIMENSION_PAIR_INFORMATION_V1',remote_root=REMOTE,
        worker_SHA256=sha256((LOCAL/'worker.py').read_bytes()),mechanical_parent_SHA256=sha256(PARENT.read_bytes()),
        preparer_SHA256=sha256(Path(__file__).read_bytes()),plans=plans,policy=policy,
        input_rule_SHA256=sha256((ROOT/'frontierloop/paired_pool_plan.py').read_bytes()),
        executor_generator_SHA256=sha256((ROOT/'frontierloop/paired_pool_executor.py').read_bytes()),
        qualification_tests_SHA256=sha256((ROOT/'qualification-paired-pool-v1-tests.xml').read_bytes()),
        N9_qualification_SHA256=sha256(n9file.read_bytes()),R2_source_freeze_SHA256=sha256(r2file.read_bytes()),
        native_binary=n9['binary'],qualified_external_SHA256=external,
        scope='ALL60_SOURCE_VARIANTS_120PLANNED106AVAILABLE14MISSING_BOTH_ARMS_FRESH_HCI_NO_NEW_QUANTUM_SHOTS',
        reused_executor_legacy_labels='N7/small initial pool labels are cosmetic; protocol and actual paired bytes authoritative',
        molecular_performance_verified=False,original_source_or_binary_modified=False,goal_complete=False)
    save(LOCAL/'protocol.json',spec);print('N10_FROZEN120_ARMS106AVAILABLE14MISSING',flush=True)


def submit():
    if (LOCAL/'submission-started.json').exists():raise RuntimeError('Existing N10 submit intent; reconcile exact job')
    spec=load(LOCAL/'protocol.json')
    if sha256((LOCAL/'worker.py').read_bytes())!=spec['worker_SHA256']:raise RuntimeError('Frozen worker changed')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('Remote root exists')
    ssh('mkdir '+shlex.quote(REMOTE),timeout=35)
    for name in ['worker.py','protocol.json']:send(LOCAL/name,REMOTE+'/'+name)
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(LOCAL/'pipeline'),HOST+':'+REMOTE+'/'],check=True,timeout=50)
    wrap=shlex.join(['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',BASE+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE+'/protocol.json'])
    cmd=shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=8G','--time=01:10:00',
        '--job-name=ceo-paired-initial-pool-v1','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err','--wrap='+wrap])
    save(LOCAL/'submission-started.json',dict(command=cmd,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    job=ssh(cmd,timeout=35).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Unknown submission; no blind retry')
    save(LOCAL/'job.json',dict(job_id=job,remote=REMOTE,GPU_allocations=0));print('N10_SUBMITTED_CPU_ONLY',job,flush=True)


def status():
    job=load(LOCAL/'job.json')['job_id'];q=ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%R"',timeout=35)
    live='\n'.join(x for x in q.splitlines() if x.split('|')[0]==job)
    acc=ssh('sacct -X -n -j '+job+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=35).strip()
    save(LOCAL/'status.json',dict(job_id=job,live=live,accounting=acc));print(live or acc or 'UNRESOLVED_NO_RESTART',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','status']);a=p.parse_args();globals()[a.action]()
