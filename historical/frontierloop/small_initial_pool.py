"""All-source small initial-pool diagnostic; frozen old runs are never changed."""
import argparse,shlex,shutil,subprocess
from pathlib import Path
from .paths import ROOT
from .io import load,save,sha256,digest,space
from .joint_ci_dispatch import LOCAL as N5
from .native_source_controls import LOCAL as N6
from .small_initial_pool_plan import initial_strings
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as BASE

LOCAL=ROOT/'runs/N7-SMALL-INITIAL-POOL-v1'
REMOTE='/share/reiwatanabe/ceo-small-initial-pool-20261025-v1'
WORKER=ROOT/'frontierloop/small_initial_pool_worker.py'


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N7 intent/freeze; do not overwrite')
    space();p5=load(N5/'protocol.json');p6=load(N6/'protocol.json')
    audit=load(ROOT/'analysis/N6-source-controls-summary.json')
    if audit['completed_readouts']!=14 or audit['status']!='ALL8_SOURCE_CONTROLS_AND16_ARMS_AUDITED':raise RuntimeError('Complete all N6 available controls first')
    if sha256((ROOT/'frontierloop/native_source_control_worker.py').read_bytes())!=p6['worker_SHA256']:raise RuntimeError('Mechanical parent worker drift')
    policy=dict(p6['policy'],initial_marginal_R_cap=8,global_wall_seconds=3900,readout_arms=['JOINT_TOP_D16384'])
    entries=[('N5',p) for p in p5['plans']]+[('N6',p) for p in p6['plans']]
    if len(entries)!=28 or len({(p['case'],p['method']) for _,p in entries})!=28:raise RuntimeError('All4x7 source classes required')
    plans=[]
    for stage,parent in entries:
        case=parent['case'];definition=p6['physical'][case];source=dict(parent['source'])
        source.setdefault('status','AVAILABLE');source.setdefault('expected_problem_id',definition['problem_id'])
        if source['expected_problem_id']!=definition['problem_id']:raise RuntimeError('Source problem mismatch')
        strings=None;prior_count=None
        if source['status']=='AVAILABLE':
            for name,key in [('payload','payload_SHA256'),('source_result','source_result_SHA256')]:
                if name in source and sha256(Path(source[name]).read_bytes())!=source[key]:raise RuntimeError('Source artifact drift')
            old=initial_strings(source,definition['norb'],tuple(definition['nelec']),r=128)
            strings=initial_strings(source,definition['norb'],tuple(definition['nelec']),r=8)
            if len(strings)>8 or not set(strings)<=set(old):raise RuntimeError('Small pool not nested in source-ranked pool')
            prior_count=len(old)
        row=dict(case=case,method=parent['method'],source=source,HF_string=(1<<definition['nelec'][0])-1,
            initial_strings=strings,initial_joint_dimension=len(strings)**2 if strings is not None else None,
            prior_initial_string_count=prior_count,parent_stage=stage,parent_task_id=parent['task_id'])
        row['task_id']=digest([row,policy,definition,sha256((ROOT/'frontierloop/small_initial_pool_plan.py').read_bytes())]);plans.append(row)
    if sum(p['source']['status']=='AVAILABLE' for p in plans)!=27:raise RuntimeError('Expected27 available+1 missing')
    LOCAL.mkdir();(LOCAL/'pipeline').mkdir()
    for name,value in p6['pipeline_SHA256'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=value:raise RuntimeError('Qualified native/CI helper changed')
        shutil.copyfile(ROOT/'frontierloop'/name,LOCAL/'pipeline'/name)
    spec=dict(p6,version='N7_ALL_SOURCE_SMALL_INITIAL_POOL_V1',remote_root=REMOTE,policy=policy,plans=plans,
        worker_SHA256=sha256(WORKER.read_bytes()),preparer_SHA256=sha256(Path(__file__).read_bytes()),
        initial_rule_SHA256=sha256((ROOT/'frontierloop/small_initial_pool_plan.py').read_bytes()),
        source_tests_SHA256=sha256((ROOT/'qualification-small-initial-pool-v1-tests.xml').read_bytes()),
        mechanical_parent_worker_SHA256=p6['worker_SHA256'],
        N6_protocol_SHA256=sha256((N6/'protocol.json').read_bytes()),N6_terminal_SHA256=audit['terminal_SHA256'],
        scope='ALL4X7_SOURCES_INITIAL_POOL8_OR_HF1_SAME_NATIVE_AND_JOINT_ONLY_NOT_NEW_QUANTUM_DATA')
    save(LOCAL/'protocol.json',spec);print('N7_FROZEN28_SOURCE_CELLS_27AVAILABLE_1MISSING_JOINT_ONLY',flush=True)


def submit():
    if (LOCAL/'job.json').exists() or (LOCAL/'submission-started.json').exists():raise RuntimeError('Existing N7 submit intent; reconcile exact job')
    spec=load(LOCAL/'protocol.json')
    if sha256(WORKER.read_bytes())!=spec['worker_SHA256']:raise RuntimeError('Frozen N7 worker changed')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('Remote N7 root exists')
    ssh('mkdir '+shlex.quote(REMOTE),timeout=35);send(WORKER,REMOTE+'/worker.py');send(LOCAL/'protocol.json',REMOTE+'/protocol.json')
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(LOCAL/'pipeline'),HOST+':'+REMOTE+'/'],check=True,timeout=50)
    wrap=shlex.join(['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',BASE+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE+'/protocol.json'])
    cmd=shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=8G','--time=01:10:00',
        '--job-name=ceo-small-initial-pool-v1','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err','--wrap='+wrap])
    save(LOCAL/'submission-started.json',dict(command=cmd,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    jid=ssh(cmd,timeout=35).strip().split(';')[0]
    if not jid.isdigit():raise RuntimeError('Unknown submit result; inspect exact name, never resubmit blindly')
    save(LOCAL/'job.json',dict(job_id=jid,remote=REMOTE,GPU_allocations=0));print('N7_SUBMITTED_CPU_ONLY',jid,flush=True)


def status(collect=False):
    jid=load(LOCAL/'job.json')['job_id'];queue=ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%R"',timeout=35)
    live='\n'.join(line for line in queue.splitlines() if line.split('|')[0]==jid)
    accounting=ssh('sacct -X -n -j '+jid+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=35).strip()
    save(LOCAL/'status.json',dict(job_id=jid,live=live,accounting=accounting));print(live or accounting or 'UNRESOLVED_NO_RESTART',flush=True)
    if not collect or live:return
    if not accounting:raise RuntimeError('Need authoritative accounting before terminal collection')
    if int(ssh('du -sk '+shlex.quote(REMOTE),timeout=35).split()[0])>32*1024:raise RuntimeError('Need compact collection; remote evidence preserved')
    space();target=LOCAL/'collected';target.mkdir(exist_ok=True)
    for name in ['started.json','qualification','progress.json','terminal.json','observations.json','ci-attempts.json','slurm.out','slurm.err','cells','roots']:
        if ssh('test -e '+shlex.quote(REMOTE+'/'+name)+' && echo yes || true',timeout=35).strip()!='yes':continue
        subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(target)],check=True,timeout=50)
    print('N7_BOUNDED_EVIDENCE_COLLECTED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','status','collect']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='submit':submit()
    else:status(a.action=='collect')
