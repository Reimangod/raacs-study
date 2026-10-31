"""Freeze, run and collect the eight N6 source-control cells, without retries."""
import argparse,shlex,shutil,subprocess
from pathlib import Path
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,sha256,digest,space
from .native_source_control_plan import METHODS,initial_strings
from .native_hci_bootstrap import LOCAL as N4
from .joint_ci_dispatch import LOCAL as N5
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as BASE

LOCAL=ROOT/'runs/N6-NATIVE-SOURCE-CONTROLS-v1'
REMOTE='/share/reiwatanabe/ceo-native-source-controls-20261025-v1'
WORKER=ROOT/'frontierloop/native_source_control_worker.py'


def prepare():
    if LOCAL.exists():raise RuntimeError('N6 freeze/intent exists; never overwrite/resubmit')
    space();r2=load(ROOT/'runs/R2-COMMON-CI-v1/freeze.json');p4=load(N4/'protocol.json');p5=load(N5/'protocol.json')
    if load(ROOT/'analysis/N5-joint-CI-summary.json')['completed_readouts']!=40:raise RuntimeError('Complete qualified N5 required')
    q5=N5/'collected/qualification.json';t5=N5/'collected/terminal.json';library=p5['remote_root']+'/library/libjoint_ci.so'
    if load(q5)['status']!='ALL4_REAL_FULL_PRODUCT_C7_PARITY_PASSED':raise RuntimeError('Four-real-case numerical parity required')
    build=load(N5/'collected/library/build.json')
    if sha256((N5/'collected/library/libjoint_ci.so').read_bytes())!=build['binary_SHA256']:raise RuntimeError('Qualified sparse binary changed')
    physical={}
    for case,definition in p4['physical'].items():
        for name,value in r2['physical_SHA256'][case].items():
            if sha256((OLD/'data/physical'/case/name).read_bytes())!=value:raise RuntimeError('R2 physical input changed')
        physical[case]=dict(definition,fcidump_path=p4['remote_root']+'/physical/'+case+'/fcidump.txt')
    policy=dict(native_cutoff=.0005,native_maxiter=10,native_energy_tol=1e-10,native_davidson_tol=1e-8,
        native_wall_seconds=60,native_RLIMIT_AS_bytes=6*1024**3,native_RLIMIT_FSIZE_bytes=64*1024**2,
        initial_marginal_R_cap=128,product_R_cap=128,joint_D_cap=16384,spin_penalty_Ha=.1,
        root_wall_seconds=75,triangle_entry_cap=8000000,all_threads=1,new_model_shots=0,new_QPU_shots=0,
        same_dimension_cap_not_equal_actual_dimension=True,spinfree_proposal_not_CI_energy=True,
        native_RNG='UNMODIFIED_UPSTREAM_DEFAULT_NOT_EXPLICITLY_SEEDED',no_orbital_optimization=True)
    selected=[s for s in r2['sources'] if s['method_id'] in METHODS]
    if len(selected)!=8 or sum(s['status']=='AVAILABLE' for s in selected)!=7:raise RuntimeError('Eight planned/seven available source controls required')
    LOCAL.mkdir();(LOCAL/'pipeline').mkdir();plans=[]
    for case,definition in physical.items():
        ne=tuple(definition['nelec']);hf=(1<<ne[0])-1
        for method in METHODS:
            source=next(s for s in selected if s['case']==case and s['method_id']==method)
            if source['expected_problem_id']!=definition['problem_id']:raise RuntimeError('Source physical identity differs')
            strings=None
            if source['status']=='AVAILABLE':
                for path,key in [('payload','payload_SHA256'),('source_result','source_result_SHA256')]:
                    if sha256(Path(source[path]).read_bytes())!=source[key]:raise RuntimeError('Actual source artifact changed')
                strings=initial_strings(source,definition['norb'],ne)
                prior=load(ROOT/'runs/R2-COMMON-CI-v1'/case/method/'request.json')
                if strings!=prior['CI_strings']:raise RuntimeError('Source-only initial space differs from registered R2')
            row=dict(case=case,method=method,source=source,initial_strings=strings,HF_string=hf,
                initial_joint_dimension=len(strings)**2 if strings is not None else None)
            row['task_id']=digest([row,policy,definition,sha256((ROOT/'frontierloop/native_source_control_plan.py').read_bytes())]);plans.append(row)
    names=['native_dice_binary.py','native_hci_plan.py','joint_ci_plan.py','joint_ci_sparse.py','joint_ci_sparse.c']
    for name in names:shutil.copyfile(ROOT/'frontierloop'/name,LOCAL/'pipeline'/name)
    for name in ['joint_ci_sparse.py','joint_ci_sparse.c','joint_ci_plan.py']:
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=p5['task_implementation_SHA256'][name]:raise RuntimeError('Qualified N5 evaluator changed')
    mini=p4['minimal_qualification']
    spec=dict(version='N6_NATIVE_SOURCE_CONTROLS_COMMON_PRODUCT_AND_JOINT_V1',remote_root=REMOTE,
        worker_SHA256=sha256(WORKER.read_bytes()),preparer_SHA256=sha256(Path(__file__).read_bytes()),
        initial_rule_SHA256=sha256((ROOT/'frontierloop/native_source_control_plan.py').read_bytes()),
        source_tests_SHA256=sha256((ROOT/'qualification-native-source-control-v1-tests.xml').read_bytes()),
        pipeline_SHA256={name:sha256((LOCAL/'pipeline'/name).read_bytes()) for name in names},
        common_runtime=p5['common_runtime'],packages=p5['packages'],sparse_library=library,
        native_binary=p4['binary'],native_lock=p4['native_lock'],
        qualified_external_SHA256={p4['binary']:p4['binary_SHA256'],library:build['binary_SHA256'],
            p5['remote_root']+'/qualification.json':sha256(q5.read_bytes()),p5['remote_root']+'/terminal.json':sha256(t5.read_bytes())},
        minimal_qualification=dict(mini,fcidump_path=p4['remote_root']+'/qualification/fcidump.txt'),
        R2_source_freeze_SHA256=sha256((ROOT/'runs/R2-COMMON-CI-v1/freeze.json').read_bytes()),
        N5_protocol_SHA256=sha256((N5/'protocol.json').read_bytes()),plans=plans,policy=policy,physical=physical,
        new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,
        scope='ALL4_CASES_TWO_ADDITIONAL_SOURCE_CONTROLS_ONE_EXPLICIT_MISSING_TN_NOT_FULL_HIGHEST_PAPER',
        independent_quantum_confirmation=False,final_CI_state_not_prepared_by_sampler=True,
        independent_RDM_for_each_joint_root=False,goal_complete=False)
    save(LOCAL/'protocol.json',spec);print('N6_FROZEN8_SOURCE_CELLS_7AVAILABLE_1MISSING_16PLANNED_ARMS',flush=True)


def submit():
    if (LOCAL/'job.json').exists() or (LOCAL/'submission-started.json').exists():raise RuntimeError('N6 submit intent exists; reconcile exact job')
    spec=load(LOCAL/'protocol.json')
    if sha256(WORKER.read_bytes())!=spec['worker_SHA256']:raise RuntimeError('Frozen N6 worker changed')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('N6 remote root exists')
    ssh('mkdir '+shlex.quote(REMOTE),timeout=35);send(WORKER,REMOTE+'/worker.py');send(LOCAL/'protocol.json',REMOTE+'/protocol.json')
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(LOCAL/'pipeline'),HOST+':'+REMOTE+'/'],check=True,timeout=50)
    wrap=shlex.join(['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',BASE+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE+'/protocol.json'])
    command=shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=8G','--time=00:35:00',
        '--job-name=ceo-native-source-controls-v1','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err','--wrap='+wrap])
    save(LOCAL/'submission-started.json',dict(command=command,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    jid=ssh(command,timeout=35).strip().split(';')[0]
    if not jid.isdigit():raise RuntimeError('Unknown submission; inspect exact job name instead of resubmit')
    save(LOCAL/'job.json',dict(job_id=jid,remote=REMOTE,GPU_allocations=0));print('N6_SUBMITTED_CPU_ONLY',jid,flush=True)


def status(collect=False):
    jid=load(LOCAL/'job.json')['job_id']
    # Query the user queue, not -j: Slurm ages finished jobs out of squeue and
    # -j then returns an error despite valid sacct terminal evidence.
    queue=ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%R"',timeout=35)
    live='\n'.join(line for line in queue.splitlines() if line.split('|')[0]==jid)
    account=ssh('sacct -X -n -j '+jid+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=35).strip()
    save(LOCAL/'status.json',dict(job_id=jid,live=live,accounting=account));print(live or account or 'UNRESOLVED_NO_RESTART',flush=True)
    if not collect or live:return
    if not account:raise RuntimeError('No authoritative terminal accounting; never infer completion from empty queue')
    if int(ssh('du -sk '+shlex.quote(REMOTE),timeout=35).split()[0])>32*1024:raise RuntimeError('Bounded collector needs compact alternative; remote evidence retained')
    space();target=LOCAL/'collected';target.mkdir(exist_ok=True)
    for name in ['started.json','qualification','progress.json','terminal.json','observations.json','ci-attempts.json','slurm.out','slurm.err','cells','roots']:
        if ssh('test -e '+shlex.quote(REMOTE+'/'+name)+' && echo yes || true',timeout=35).strip()!='yes':continue
        subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(target)],check=True,timeout=50)
    print('N6_BOUNDED_EVIDENCE_COLLECTED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','status','collect']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='submit':submit()
    else:status(a.action=='collect')
