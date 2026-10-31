"""N8: complete the existing source inventory, no new shots or R tuning.

Reuse the byte-identical qualified N7 executor with R128 in the *prepared*
input. Its legacy N7/small-pool log labels are not this experiment's identity.
The protocol, source hashes and actual Alpha/Beta bytes define N8.
"""
import argparse, shlex, shutil, subprocess
from pathlib import Path
from .paths import ROOT
from .io import load, save, sha256, digest, space
from .native_source_controls import LOCAL as N6
from .native_source_control_plan import initial_strings
from benchmark.gpu_remote import ssh, send, OPTIONS, HOST, REMOTE as BASE

LOCAL=ROOT/'runs/N8-REMAINING-SOURCE-CONTROLS-v1'
REMOTE='/share/reiwatanabe/ceo-remaining-source-controls-20261026-v1'
WORKER=ROOT/'frontierloop/small_initial_pool_worker.py'
METHODS=('C5_B_LUCJ_NAIVE_L1','C5_B_LUCJ_NAIVE_L2','C5_B_LUCJ_NAIVE_L4',
         'C5_C_LUCJ_COMPRESSED_L2','C5_C_LUCJ_COMPRESSED_L4',
         'C5_D_SQDRIFT_N5','C5_D_SQDRIFT_N50','RAACS_B3')


def choose_sources(r2, covered, physical):
    """Select by source identity alone, never outcome/energy/FCI."""
    selected=[s for s in r2['sources'] if s['method_id'] in METHODS]
    keys={(s['case'],s['method_id']) for s in selected}
    if len(selected)!=32 or len(keys)!=32 or keys & covered:
        raise ValueError('Exactly32 previously uncovered source cells required')
    if keys!={(c,m) for c in physical for m in METHODS}:
        raise ValueError('All4 physical cases and all8 source methods required')
    if sum(s['status']=='AVAILABLE' for s in selected)!=26:
        raise ValueError('Preserve all26 available and6 missing source cells')
    for s in selected:
        if s['expected_problem_id']!=physical[s['case']]['problem_id']:
            raise ValueError('Wrong physical problem')
    return selected


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N8 intent; do not overwrite/resubmit')
    space();p6=load(N6/'protocol.json');p7=load(ROOT/'runs/N7-SMALL-INITIAL-POOL-v1/protocol.json')
    if sha256(WORKER.read_bytes())!=p7['worker_SHA256']:raise RuntimeError('Qualified reused executor changed')
    n7=load(ROOT/'analysis/N7-small-initial-pool-summary.json')
    if n7['completed']!=27:raise RuntimeError('Reconcile all prior N7 outcomes first')
    previous=ROOT/'analysis/N6-combined-joint-comparison.json'
    prior=load(previous)['rows'];covered={(r['case'],r['method']) for r in prior}
    if len(prior)!=28 or len(covered)!=28:raise RuntimeError('N5/N6 coverage invalid')
    r2file=ROOT/'runs/R2-COMMON-CI-v1/freeze.json';r2=load(r2file)
    selected=choose_sources(r2,covered,p6['physical'])
    policy=dict(p6['policy'],initial_marginal_R_cap=128,global_wall_seconds=3900,
                readout_arms=['JOINT_TOP_D16384'])
    plans=[]
    for s in selected:
        d=p6['physical'][s['case']];strings=None
        if s['status']=='AVAILABLE':
            for name,key in [('payload','payload_SHA256'),('source_result','source_result_SHA256')]:
                if sha256(Path(s[name]).read_bytes())!=s[key]:raise RuntimeError('Frozen source changed')
            strings=initial_strings(s,d['norb'],tuple(d['nelec']))
            old=load(ROOT/'runs/R2-COMMON-CI-v1'/s['case']/s['method_id']/'request.json')
            if strings!=old['CI_strings']:raise RuntimeError('R128 ranking differs from original source')
        row=dict(case=s['case'],method=s['method_id'],source=s,initial_strings=strings,
                 HF_string=(1<<d['nelec'][0])-1,
                 initial_joint_dimension=len(strings)**2 if strings is not None else None)
        row['task_id']=digest([row,policy,d,sha256(WORKER.read_bytes())]);plans.append(row)
    LOCAL.mkdir();(LOCAL/'pipeline').mkdir()
    for name,value in p6['pipeline_SHA256'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=value:raise RuntimeError('Qualified helper changed')
        shutil.copyfile(ROOT/'frontierloop'/name,LOCAL/'pipeline'/name)
    spec=dict(p6,version='N8_REMAINING_ALL_SOURCE_COMMON_JOINT_CONTROLS_V1',remote_root=REMOTE,
        plans=plans,policy=policy,worker_SHA256=sha256(WORKER.read_bytes()),
        preparer_SHA256=sha256(Path(__file__).read_bytes()),
        source_tests_SHA256=sha256((ROOT/'qualification-remaining-source-controls-v1-tests.xml').read_bytes()),
        reused_executor_legacy_labels='N7/small-pool prose only; actual prepared inputs are R128',
        R2_source_freeze_SHA256=sha256(r2file.read_bytes()),
        prior_28_comparison_SHA256=sha256(previous.read_bytes()),
        N7_summary_SHA256=sha256((ROOT/'analysis/N7-small-initial-pool-summary.json').read_bytes()),
        combined_planned_source_cells=60,combined_available_source_cells=53,
        scope='ALL_EXISTING_AVAILABLE_SOURCE_VARIANTS_SAME_R128_HCI_JOINT_OPPORTUNITY_NOT_HIGHEST_PAPER_REPRODUCTION')
    save(LOCAL/'protocol.json',spec);print('N8_FROZEN32_ADDITIONAL_26AVAILABLE_6MISSING',flush=True)


def submit():
    if (LOCAL/'job.json').exists() or (LOCAL/'submission-started.json').exists():raise RuntimeError('Existing submit intent; reconcile without resubmit')
    spec=load(LOCAL/'protocol.json')
    if sha256(WORKER.read_bytes())!=spec['worker_SHA256']:raise RuntimeError('Executor drift')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('Remote root exists')
    ssh('mkdir '+shlex.quote(REMOTE),timeout=35);send(WORKER,REMOTE+'/worker.py');send(LOCAL/'protocol.json',REMOTE+'/protocol.json')
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(LOCAL/'pipeline'),HOST+':'+REMOTE+'/'],check=True,timeout=50)
    wrap=shlex.join(['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',BASE+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE+'/protocol.json'])
    cmd=shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=8G','--time=01:10:00',
        '--job-name=ceo-remaining-source-controls-v1','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err','--wrap='+wrap])
    save(LOCAL/'submission-started.json',dict(command=cmd,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    jid=ssh(cmd,timeout=35).strip().split(';')[0]
    if not jid.isdigit():raise RuntimeError('Unknown submission; reconcile exact job name')
    save(LOCAL/'job.json',dict(job_id=jid,remote=REMOTE,GPU_allocations=0));print('N8_SUBMITTED_CPU_ONLY',jid,flush=True)


def status():
    jid=load(LOCAL/'job.json')['job_id'];queue=ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%R"',timeout=35)
    live='\n'.join(x for x in queue.splitlines() if x.split('|')[0]==jid)
    a=ssh('sacct -X -n -j '+jid+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=35).strip()
    save(LOCAL/'status.json',dict(job_id=jid,live=live,accounting=a));print(live or a or 'UNRESOLVED_NO_RESTART',flush=True)


def collect():
    # Reuse a completed evidence-only tar/hash collector, not a numeric executor.
    from . import collect_small_pool_bundle as collector
    collector.LOCAL=LOCAL;collector.REMOTE=REMOTE
    collector.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','status','collect']);a=p.parse_args()
    globals()[a.action]()
