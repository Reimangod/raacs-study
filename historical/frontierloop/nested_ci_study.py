"""N20 fixed-source nested CI dimension/cost experiment, no new circuit policy."""
import argparse
import math
from pathlib import Path
import shlex
import subprocess
import shutil
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256,digest
from .prefix_performance_sampling import capacity
from .nested_ci_plan import frequency_basis
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as LAUNCHER

LOCAL=ROOT/'runs/N20-NESTED-CI-SCALING-v1'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/N20-nested-ci-scaling-v1'
WORKER=ROOT/'frontierloop/nested_ci_worker.py'


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N20; no overwrite')
    capacity()
    if load(ROOT/'analysis/N19-tn-objective-feasibility-summary.json')['state_counts']!={'COMPLETED':4}:
        raise RuntimeError('Close all4 prior feasibility attempts first')
    prior=load(ROOT/'runs/N18-HALF-SOURCE-NATIVE-SCREEN-v1/protocol.json')
    old=load(ROOT/'analysis/N18-half-source-native-summary.json');tasks={};points=[];sources={}
    for r in (128,256,512,1024):
        for case,d in sorted(prior['physical'].items()):
            source=next(p['source'] for p in prior['plans'] if p['case']==case and p['method']=='FULL_LUCJ_L1')
            if source['kind']!='QUANTUM_FULL_COUNTS' or source['expected_problem_id']!=d['problem_id']:
                raise RuntimeError('Same actual full source required')
            for path,key in [('payload','payload_SHA256'),('source_result','source_result_SHA256')]:
                if sha256(Path(source[path]).read_bytes())!=source[key]:raise RuntimeError('Original source changed')
            with np.load(source['payload'],allow_pickle=False) as a:
                if int(a['counts'].sum())!=100000:raise RuntimeError('Same100k observation required')
                strings=frequency_basis(a['blocked_ids'],a['counts'],d['norb'],d['nelec'],r)
            if len(strings)==math.comb(d['norb'],d['nelec'][0]):
                raise RuntimeError('Full active-space solve not in this source diagnostic')
            expected=next(x for x in old['rows'] if (x['case'],x['method'],x['arm'])==(case,'FULL_LUCJ_L1','SOURCE_FREQUENCY_R128'))
            task_id=digest([d['problem_id'],strings])
            if task_id not in tasks:
                tasks[task_id]=dict(task_id=task_id,case=case,problem_id=d['problem_id'],requested_R=r,strings=strings,
                    expected_R128={k:expected[k] for k in ('energy_Ha','penalized_energy_Ha','spin_square')})
            points.append(dict(case=case,requested_R=r,task_id=task_id,actual_half_dimension=len(strings),actual_CI_dimension=len(strings)**2))
            sources[case]=dict(source=source,original_full_compiler_points=expected['all_source_compiler_points'])
    if len(points)!=16 or len(tasks)!=11:raise RuntimeError('All16 points/11 unique bases required')
    for case in sources:
        sequence=[tasks[p['task_id']]['strings'] for p in points if p['case']==case]
        if any(not set(a)<=set(b) for a,b in zip(sequence,sequence[1:])):raise RuntimeError('CI spaces not nested')
    LOCAL.mkdir();(LOCAL/'pipeline').mkdir()
    write(LOCAL/'worker.py',WORKER.read_bytes())
    names=['nested_ci_kernel.py','portable_penalized_ci.py','hole_rdm.py']
    for name in names:shutil.copyfile(ROOT/'frontierloop'/name,LOCAL/'pipeline'/name)
    write(LOCAL/'pipeline/__init__.py',b'')
    qualified=load(ROOT/'runs/N3-PRIVATE-COMMON-CI-v1/protocol.json')
    for name in ('portable_penalized_ci.py','hole_rdm.py'):
        if sha256((LOCAL/'pipeline'/name).read_bytes())!=qualified['kernel_SHA256']['kernels/'+name]:
            raise RuntimeError('Qualified shared math changed')
    save(LOCAL/'protocol.json',dict(version='N20_ALL4_FIXED_SOURCE_NESTED_CI_SCALING_V1',remote_root=REMOTE,
        points=points,tasks=list(tasks.values()),sources=sources,physical=prior['physical'],
        common_runtime=prior['common_runtime'],packages=prior['packages'],
        worker_SHA256=sha256(WORKER.read_bytes()),preparer_SHA256=sha256(Path(__file__).read_bytes()),
        input_SHA256={'worker.py':sha256((LOCAL/'worker.py').read_bytes()),**{'pipeline/'+p.name:sha256(p.read_bytes()) for p in (LOCAL/'pipeline').iterdir()}},
        rule_SHA256=sha256((ROOT/'frontierloop/nested_ci_plan.py').read_bytes()),
        tests_SHA256=sha256((ROOT/'qualification-N20-preexecution-v1-tests.xml').read_bytes()),
        prior_N18_summary_SHA256=sha256((ROOT/'analysis/N18-half-source-native-summary.json').read_bytes()),
        root_wall_seconds=75,child_hard_wall_seconds=110,child_virtual_memory_bytes=6*2**30,
        global_wall_seconds=1800,common_equation='H_PLUS_0p1_S2_NOT_EXACT_S0_NO_OO',
        eigenpairs=1,warm_start='PREVIOUS_VERIFIED_ROOT_PLUS_FIXED1e-3_RANDOM_COMPONENT_WITH_EMBEDDED_WITNESS_CHECK',
        stop_rule='ALL4_R128_PARITY_BEFORE_ANY_EXPANSION_STOP_CASE_AFTER_FIRST_SCALING_LIMIT',
        global_point_order='R_ASCENDING_THEN_CASE_LEXICOGRAPHIC',
        new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,
        actual_work_matched=False,independent_confirmation=False,full_paper_reproduction=False,
        source_or_circuit_resource_improvement_claimed=False,goal_complete=False))
    print('N20_FROZEN_ALL16_POINTS11_UNIQUE_BASES_NO_NEW_SHOTS',flush=True)


def submit():
    spec=load(LOCAL/'protocol.json');capacity()
    if (LOCAL/'submission-intent.json').exists() or (LOCAL/'job.json').exists():raise RuntimeError('Existing N20 dispatch; no duplicate')
    if sha256(WORKER.read_bytes())!=spec['worker_SHA256'] or sha256(Path(__file__).read_bytes())!=spec['preparer_SHA256']:
        raise RuntimeError('Frozen worker/preparer changed')
    for name,h in spec['input_SHA256'].items():
        if sha256((LOCAL/name).read_bytes())!=h:raise RuntimeError('Frozen N20 input changed')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=30).strip()!='NEW':raise RuntimeError('Existing remote work')
    ssh('mkdir '+shlex.quote(REMOTE),timeout=30)
    for name in ('worker.py','protocol.json'):send(LOCAL/name,REMOTE+'/'+name)
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(LOCAL/'pipeline'),HOST+':'+REMOTE+'/'],check=True,timeout=50)
    wrap=shlex.join(['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','RAYON_NUM_THREADS=1',
        'PYTHONPATH='+spec['common_runtime']+'/site',LAUNCHER+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE])
    command=shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=8G','--time=00:30:00',
        '--job-name=frontier-N20-nested-CI','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err','--wrap='+wrap])
    save(LOCAL/'submission-intent.json',dict(command=command,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    job=ssh(command,timeout=30).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Uncertain sbatch; reconcile exact name, no retry')
    save(LOCAL/'job.json',dict(job_id=job,remote=REMOTE,GPU_allocations=0));print('N20_CPU_ONLY_SUBMITTED',job,flush=True)


def status():
    job=load(LOCAL/'job.json')['job_id'];q=ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%R"',timeout=30)
    live='\n'.join(s for s in q.splitlines() if s.split('|')[0]==job)
    account=ssh('sacct -X -n -j '+job+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=30)
    save(LOCAL/'status.json',dict(job_id=job,live=live,accounting=account));print(live or account or 'UNRESOLVED_NO_RESTART',flush=True)


def collect():
    from . import collect_small_pool_bundle as collector
    collector.LOCAL=LOCAL;collector.REMOTE=REMOTE;collector.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','status','collect']);a=p.parse_args();globals()[a.action]()
