"""Twenty fixed source cells: native HCI proposals + common C7 equation.

Native proposals are spin-free; final comparisons all use finite0.1Ha S².
This is a bounded exploratory common-setup study, not original-paper highest
HCI performance, equal actual CPU, or fresh independent quantum sampling.
"""
import argparse,shlex,shutil,subprocess,json
from pathlib import Path
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,sha256,digest
from .native_hci_plan import frequency_strings
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as BASE

LOCAL=ROOT/'runs/N4-NATIVE-HCI-PROPOSALS-v1'
REMOTE='/share/reiwatanabe/ceo-native-hci-proposals-20261025-v1'
CI_REMOTE='/share/reiwatanabe/ceo-native-common-ci-20261025-v1'
WORKER=ROOT/'frontierloop/native_hci_worker.py'


def prepare():
    from pyscf import tools
    if LOCAL.exists():raise RuntimeError('Existing N4 plan/intent; never overwrite/restart')
    build=load(ROOT/'runs/N1-NATIVE-DICE-BUILD-repair-v2/result.json')
    qualification=load(ROOT/'analysis/N2-native-Dice-toy-summary.json')
    common_path=ROOT/'runs/N3-PRIVATE-COMMON-CI-v1/result.json';common=load(common_path)
    if qualification['passed']!=7 or common['status']!='PRIVATE_CPU_RUNTIME_FOUR_CASE_PARITY_PASSED':
        raise RuntimeError('Actual native physics and common-runtime parity required')
    LOCAL.mkdir(exist_ok=False);frozen=load(ROOT/'freeze-T3-READOUT-SEEDS.json');physical={};plans=[]
    common_spec=load(ROOT/'runs/N3-PRIVATE-COMMON-CI-v1/protocol.json')
    implementation=dict(worker_SHA256=sha256(WORKER.read_bytes()),
        proposal_rule_SHA256=sha256((ROOT/'frontierloop/native_hci_plan.py').read_bytes()),
        decoder_SHA256=sha256((ROOT/'frontierloop/native_dice_binary.py').read_bytes()),
        common_kernel_SHA256=common_spec['kernel_SHA256'],packages=common_spec['packages'],
        planned_attempt_protocol_id='N4_NATIVE_HCI_PROPOSALS_COMMON_CI_V1')
    unique=list({(p['source']['case'],p['source']['method_id']):p for p in frozen['plans']}.values())
    if len(unique)!=16:raise RuntimeError('All16 fixed sources required')
    policy=dict(native_select_cutoff=.0005,native_maxiter=10,native_energy_tol=1e-10,native_davidson_tol=1e-8,
        initial_frequency_R_cap=128,final_R_cap=128,final_spin_penalty_Ha=.1,native_wall_cap_seconds=60,
        common_CI_wall_cap_seconds=60,native_memory_limit_bytes=6*1024**3,
        actual_native_joint_dimension_may_exceed_final_R_squared=True,
        common_cap_enforcement='MATRIX_VECTOR_BOUNDARY_SOFT_WALL_CAP_SAME_AS_C7',
        native_cap_enforcement='OWN_SUBPROCESS_WALL60_SECONDS_PLUS_TERMINATION_GRACE',
        native_initialization='UPSTREAM_UNMODIFIED_DEFAULT_NOT_EXPLICIT_USER_SEED_OR_RECORDED_RNG_STATE')
    for case in dict.fromkeys(p['source']['case'] for p in unique):
        case_plans=[p for p in unique if p['source']['case']==case];example=case_plans[0]
        meta_path=OLD/'data/physical'/case/'problem.json';a_path=meta_path.parent/'integrals.npz';meta=load(meta_path)
        for name,value in example['context']['physical_SHA256'].items():
            if sha256((meta_path.parent/name).read_bytes())!=value:raise RuntimeError('Physical source drift')
        folder=LOCAL/'physical'/case;folder.mkdir(parents=True)
        with np.load(a_path,allow_pickle=False) as a:
            tools.fcidump.from_integrals(folder/'fcidump.txt',a['h1'],a['eri'],meta['norb'],(meta['nalpha'],meta['nbeta']),nuc=meta['core_energy'])
        physical[case]=dict(problem_id=example['context']['problem_id'],norb=meta['norb'],nelec=[meta['nalpha'],meta['nbeta']],
            core_energy=meta['core_energy'],integrals_path=CI_REMOTE+'/inputs/'+case+'/integrals.npz',
            integrals_SHA256=sha256(a_path.read_bytes()),fcidump_SHA256=sha256((folder/'fcidump.txt').read_bytes()))
        hf=example['HF_string']
        entries=[dict(method='CLASSICAL_HF_ONLY',initial_strings=[hf],source=dict(kind='CLASSICAL_HF_ONLY',
            source_total_budget=0,source_final_model_shots=0,sampler_resources=None,source_payload_SHA256=None,
            not_a_zero_CNOT_preparation_of_the_final_CI_state=True))]
        for p in case_plans:
            source=p['source'];payload=Path(source['payload'])
            if sha256(payload.read_bytes())!=source['payload_SHA256']:raise RuntimeError('Frozen observed counts changed')
            with np.load(payload,allow_pickle=False) as a:
                strings=frequency_strings(a['blocked_ids'],a['counts'],meta['norb'],hf)
            entries.append(dict(method=source['method_id'],initial_strings=strings,source=source))
        for entry in entries:
            row=dict(case=case,HF_string=hf,**entry)
            row['task_id']=digest([row,policy,physical[case],build['binary_SHA256'],implementation]);plans.append(row)
    if len(plans)!=20 or len({p['task_id'] for p in plans})!=20:raise RuntimeError('Twenty unique source cells required')
    pipeline=LOCAL/'pipeline';pipeline.mkdir()
    for name in ['native_hci_plan.py','native_dice_binary.py']:shutil.copyfile(ROOT/'frontierloop'/name,pipeline/name)
    toy=next(c for c in load(ROOT/'runs/N2-NATIVE-DICE-TOYS-v1/fixtures.json')['fixtures'] if c['name']=='NATIVE_HCI_EXPANSION')
    folder=LOCAL/'qualification';folder.mkdir();shutil.copyfile(ROOT/'runs/N2-NATIVE-DICE-TOYS-v1/inputs/NATIVE_HCI_EXPANSION/fcidump.txt',folder/'fcidump.txt')
    spec=dict(version='N4_NATIVE_HCI_PROPOSALS_COMMON_CI_V1',remote_root=REMOTE,common_runtime=CI_REMOTE,
        preparer_SHA256=sha256(Path(__file__).read_bytes()),task_identity_implementation=implementation,
        worker_SHA256=sha256(WORKER.read_bytes()),binary=build['binary'],binary_SHA256=build['binary_SHA256'],
        common_qualification_SHA256=sha256(common_path.read_bytes()),kernel_SHA256=common_spec['kernel_SHA256'],
        packages=common_spec['packages'],local_module_SHA256={p.name:sha256(p.read_bytes()) for p in pipeline.iterdir()},
        native_lock='/share/reiwatanabe/ceo-native-dice-20261025-v1/native-serial.lock',
        minimal_qualification=dict(fcidump_SHA256=sha256((folder/'fcidump.txt').read_bytes()),
            H_full=toy['H_full'],expected_energy_Ha=toy['free_minimum_energy_Ha']),
        plans=plans,physical=physical,policy=policy,new_model_shots=0,new_QPU_shots=0,
        final_energy_not_prepared_by_sampler=True,independent_quantum_confirmation=False,
        old_reference_energy_or_FCI_information_used_in_policy=False,
        comparison_scope='SAME_SOURCE_SAME_FINAL_MAX_R_AND_COMMON_FINITE_S2_NOT_EQUAL_ACTUAL_CPU_NOT_NATIVE_SPIN_FREE_ENERGY_IN_C3_TABLE')
    save(LOCAL/'protocol.json',spec);print('N4_FROZEN20_OUTCOME_FREE_SOURCE_PLANS',flush=True)


def submit():
    if (LOCAL/'submission-started.json').exists() or (LOCAL/'job.json').exists():raise RuntimeError('Existing N4 intent; reconcile exact job')
    spec=load(LOCAL/'protocol.json')
    if sha256(WORKER.read_bytes())!=spec['worker_SHA256']:raise RuntimeError('Frozen pipeline changed')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('N4 remote target exists')
    ssh('mkdir '+shlex.quote(REMOTE),timeout=35);send(WORKER,REMOTE+'/worker.py');send(LOCAL/'protocol.json',REMOTE+'/protocol.json')
    for name in ['physical','pipeline','qualification']:
        subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(LOCAL/name),HOST+':'+REMOTE+'/'],check=True,timeout=45)
    command=shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=8G',
        '--time=01:05:00','--job-name=ceo-native-hci-common-v1','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err',
        '--wrap='+shlex.join([BASE+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE+'/protocol.json'])])
    save(LOCAL/'submission-started.json',dict(command=command,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    reply=ssh(command,timeout=35).strip().split(';')[0]
    if not reply.isdigit():raise RuntimeError('Uncertain N4 submission; inspect exact job name')
    save(LOCAL/'job.json',dict(job_id=reply,remote=REMOTE,GPU_allocations=0));print('SUBMITTED_N4_CPU',reply,flush=True)


def status(collect=False):
    jid=load(LOCAL/'job.json')['job_id'];live=ssh('squeue -h -j '+jid+' -o "%i %T %j %R"',timeout=35).strip()
    accounting=ssh('sacct -X -n -j '+jid+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=35).strip()
    save(LOCAL/'status.json',dict(job_id=jid,live=live,accounting=accounting));print(live or accounting,flush=True)
    if not collect or live:return
    # Compact records first. Raw native files remain on AIC; only the lossless
    # coefficient archives and small input/log files are copied on demand.
    for name in ['started.json','progress.json','terminal.json','common-ci-observations.json','common-ci-attempts.json','slurm.out','slurm.err']:
        if ssh('test -f '+shlex.quote(REMOTE+'/'+name)+' && echo yes || true',timeout=35).strip()!='yes':continue
        subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(LOCAL/name)],check=True,timeout=45)
    print('N4_COMPACT_EVIDENCE_COLLECTED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','status','collect']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='submit':submit()
    else:status(a.action=='collect')
