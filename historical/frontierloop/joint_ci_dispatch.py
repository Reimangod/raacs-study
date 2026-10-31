"""Freeze/dispatch N5, no native DICE restart and no new quantum sampling."""
import argparse,gzip,itertools,shlex,shutil,subprocess
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256,digest,arrays,space
from .native_hci_bootstrap import LOCAL as PARENT
from .native_dice_binary import decode
from .joint_ci_plan import joint_pairs
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as BASE

LOCAL=ROOT/'runs/N5-JOINT-SPARSE-CI-v1'
REMOTE='/share/reiwatanabe/ceo-joint-sparse-ci-20261025-v1'
WORKER=ROOT/'frontierloop/joint_ci_worker.py'


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N5 freeze/intent; reconcile instead of overwrite')
    space();parent=load(PARENT/'protocol.json');audit=load(ROOT/'analysis/N4-native-HCI-summary.json')
    if audit['status']!='ALL20_FROZEN_NATIVE_HCI_SOURCE_CELLS_AUDITED' or audit['protocol_SHA256']!=sha256((PARENT/'protocol.json').read_bytes()):
        raise RuntimeError('Audited exact parent source required')
    test=ROOT/'qualification-joint-CI-sparse-current-tests.xml'
    import xml.etree.ElementTree as ET
    suites=ET.parse(test).getroot().iter('testsuite')
    if any(int(s.attrib.get('failures','0'))+int(s.attrib.get('errors','0')) for s in suites):raise RuntimeError('Small joint-matrix qualification not passed')
    terminal=load(PARENT/'terminal.json');states={s['task_id']:s for s in terminal['states']}
    policy=dict(joint_selection='HF_ALWAYS_THEN_NATIVE_AMPLITUDE_SQUARED_TIE_ALPHA_BETA_INTEGER_LEX',
        final_joint_dimension_cap=16384,product_string_R_cap=128,spin_penalty_Ha=.1,root_wall_seconds=75,
        triangle_entry_cap=8000000,parent_and_child_threads=1,exact_matrix_element_threshold=0.,
        no_orbital_optimization=True,ARPACK_k=2,ARPACK_tol=1e-12,ARPACK_maxiter=1000,random_seed=20261025,
        native_HCI_proposals_reused_not_newly_computed=True,same_dimension_cap_not_equal_actual_dimension=True,
        inherited_native_proposal_CPU_charged_in_standalone_comparisons=True)
    implementation={name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in ['joint_ci_sparse.py','joint_ci_sparse.c','joint_ci_plan.py']}
    LOCAL.mkdir();(LOCAL/'bases').mkdir();(LOCAL/'pipeline').mkdir();plans=[];qualification=[]
    for plan in parent['plans']:
        state=states[plan['task_id']];archive=PARENT/'collected'/plan['task_id']/'dets.bin.gz';raw=gzip.decompress(archive.read_bytes())
        if sha256(raw)!=state['proposal']['raw_determinants_SHA256']:raise RuntimeError('Native coefficient export changed')
        n,native=decode(raw);definition=parent['physical'][plan['case']]
        if n!=definition['norb']:raise RuntimeError('Physical orbital mismatch')
        pairs,selection=joint_pairs(native,n,definition['nelec'],plan['HF_string'])
        product=list(itertools.product(state['proposal']['selected_strings'],repeat=2))
        if len(product)>16384:raise RuntimeError('Parent product exceeds common dimension cap')
        record=dict(case=plan['case'],method=plan['method'],source=plan['source'],parent_task_id=plan['task_id'],
            native_raw_SHA256=sha256(raw),selection=selection,
            inherited_native_CPU_seconds=state['native_attempt_CPU_seconds'],
            inherited_native_wall_seconds=state['native_attempt_wall_seconds'])
        task=digest([record,policy,implementation]);name='bases/'+task+'.npz'
        write(LOCAL/name,arrays(joint=np.array(pairs,np.uint64),product=np.array(product,np.uint64)))
        plans.append(dict(record,task_id=task,basis_file=name,basis_SHA256=sha256((LOCAL/name).read_bytes())))
        if plan['method']=='CLASSICAL_HF_ONLY':
            qualification.append(dict(case=plan['case'],basis_file=name,
                expected={k:state['final']['row'][k] for k in ['energy_Ha','penalized_energy_Ha','spin_square']},
                expected_origin=state['final']['evaluation_origin'],purpose='COMMON_KERNEL_QUALIFICATION_ONLY_NOT_PRODUCTION_SELECTION'))
    if len(plans)!=20 or len(qualification)!=4:raise RuntimeError('All4x5 sources and full product qualifiers required')
    for name in ['joint_ci_sparse.py','joint_ci_sparse.c']:shutil.copyfile(ROOT/'frontierloop'/name,LOCAL/'pipeline'/name)
    spec=dict(version='N5_JOINT_VS_PRODUCT_COMMON_EXPLICIT_SPARSE_V1',remote_root=REMOTE,
        worker_SHA256=sha256(WORKER.read_bytes()),preparer_SHA256=sha256(Path(__file__).read_bytes()),
        task_implementation_SHA256=implementation,pipeline_SHA256={p.name:sha256(p.read_bytes()) for p in (LOCAL/'pipeline').iterdir()},
        parent_protocol_SHA256=sha256((PARENT/'protocol.json').read_bytes()),parent_terminal_SHA256=sha256((PARENT/'terminal.json').read_bytes()),
        parent_analysis_SHA256=sha256((ROOT/'analysis/N4-native-HCI-summary.json').read_bytes()),
        small_matrix_tests_SHA256=sha256(test.read_bytes()),policy=policy,plans=plans,qualification=qualification,
        legacy_kernel_SHA256=parent['kernel_SHA256'],
        common_runtime=parent['common_runtime'],packages=parent['packages'],physical=parent['physical'],
        upstream_formula_source='https://github.com/pyscf/naive-hci/tree/0c28d6eddf745c9900869332fb74e1ac490a1975',
        new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,
        scope='COMMON_D16384_CAP_OLD_NATIVE_PROPOSALS_ALL4X5_SOURCES_NOT_NEW_QUANTUM_ANSATZ_OR_PAPER_HIGHEST_REPRODUCTION',
        final_CI_state_not_prepared_by_sampler=True,goal_complete=False)
    save(LOCAL/'protocol.json',spec);print('N5_FROZEN_ALL20_40_READOUTS_WITH4_PREEXECUTION_QUALIFIERS',flush=True)


def submit():
    if (LOCAL/'submission-started.json').exists() or (LOCAL/'job.json').exists():raise RuntimeError('N5 intent exists; never duplicate')
    spec=load(LOCAL/'protocol.json')
    if sha256(WORKER.read_bytes())!=spec['worker_SHA256']:raise RuntimeError('Frozen N5 worker changed')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('N5 remote target exists')
    ssh('mkdir '+shlex.quote(REMOTE),timeout=35);send(WORKER,REMOTE+'/worker.py');send(LOCAL/'protocol.json',REMOTE+'/protocol.json')
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(LOCAL/'pipeline'),str(LOCAL/'bases'),HOST+':'+REMOTE+'/'],check=True,timeout=50)
    wrap=shlex.join(['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',BASE+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE+'/protocol.json'])
    cmd=shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=8G','--time=01:05:00',
        '--job-name=ceo-joint-sparse-ci-v1','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err','--wrap='+wrap])
    save(LOCAL/'submission-started.json',dict(command=cmd,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    jid=ssh(cmd,timeout=35).strip().split(';')[0]
    if not jid.isdigit():raise RuntimeError('Unknown submit result; inspect exact name, never blindly resubmit')
    save(LOCAL/'job.json',dict(job_id=jid,remote=REMOTE,GPU_allocations=0));print('N5_SUBMITTED_CPU_ONLY',jid,flush=True)


def status(collect=False):
    jid=load(LOCAL/'job.json')['job_id'];live=ssh('squeue -h -j '+jid+' -o "%i %T %j %R"',timeout=35).strip()
    account=ssh('sacct -X -n -j '+jid+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=35).strip()
    save(LOCAL/'status.json',dict(job_id=jid,live=live,accounting=account));print(live or account,flush=True)
    if not collect or live:return
    total=ssh('du -sk '+shlex.quote(REMOTE),timeout=35).split()[0]
    if int(total)>32*1024:raise RuntimeError('N5 collection beyond32MiB; retain remote evidence')
    space();target=LOCAL/'collected';target.mkdir(exist_ok=True)
    for name in ['started.json','qualification.json','legacy-comparisons.json','progress.json','terminal.json','observations.json','attempts.json','slurm.out','slurm.err','cells','roots','library']:
        if ssh('test -e '+shlex.quote(REMOTE+'/'+name)+' && echo yes || true',timeout=35).strip()!='yes':continue
        subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(target)],check=True,timeout=50)
    print('N5_COLLECTED_BOUNDED_TERMINAL_EVIDENCE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','submit','status','collect']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='submit':submit()
    else:status(args.action=='collect')
