"""N19 one original-style MPS draw per all four fixed sources, not an optimizer."""
import argparse
from pathlib import Path
import shlex
import subprocess
import shutil
from .paths import ROOT, OLD
from .io import load, save, write, sha256
from .prefix_performance_sampling import capacity
from benchmark.gpu_remote import ssh, send, OPTIONS, HOST

LOCAL = ROOT/'runs/N19-TN-OBJECTIVE-FEASIBILITY-v1'
REMOTE = '/share/reiwatanabe/raacs-frontier-loop-20261025/N19-tn-objective-feasibility-v1'
WORKER = ROOT/'frontierloop/tn_objective_feasibility_worker.py'
REFERENCE = ROOT/'reference/sqd-0.12.0-subsampling.py'
PYTHON = '/share/reiwatanabe/ceo-qsci-engine-20261022-v2/venv/bin/python'
PACKAGES = ['/share/reiwatanabe/raacs-paper-aligned-20261025-v1/tn-runtime-v1/packages',
            '/share/reiwatanabe/raacs-paper-aligned-20261025-v1/cpu-runtime-v1/packages']


def validated_cases(sources, physical):
    cases = []
    for case, definition in sorted(physical.items()):
        s = next(s for s in sources if s['case'] == case and s['method'].endswith('L1') and s['status'] == 'AVAILABLE')
        metadata = OLD/'data/physical'/case/'problem.json'
        integrals = OLD/'data/physical'/case/'integrals.npz'
        if s['problem_id'] != definition['problem_id'] or s['norb'] != definition['norb'] or s['nelec'] != definition['nelec']:
            raise RuntimeError('Physical operator or orbital identity differs')
        # N15 physical_SHA256 names problem.json, not the integral NPZ hash.
        if sha256(metadata.read_bytes()) != s['physical_SHA256'] or sha256(integrals.read_bytes()) != definition['integrals_SHA256']:
            raise RuntimeError('Physical metadata or numerical integral bytes changed')
        for path, key in [('operator', 'operator_SHA256'), ('construction', 'construction_SHA256')]:
            if sha256(Path(s[path]).read_bytes()) != s[key]:
                raise RuntimeError('Old source changed')
        cases.append(dict(case=case, problem_id=s['problem_id'], norb=s['norb'], nelec=s['nelec'],
            operator=s['operator'], operator_SHA256=s['operator_SHA256'],
            construction=s['construction'], construction_SHA256=s['construction_SHA256'],
            problem_metadata_SHA256=s['physical_SHA256'], integrals_SHA256=definition['integrals_SHA256']))
    if len(cases) != 4:
        raise RuntimeError('All four fixed physical conditions required')
    return cases


def prepare():
    if LOCAL.exists():
        raise RuntimeError('N19 exists; preserve and inspect, do not overwrite')
    capacity()
    if load(ROOT/'analysis/N18-half-source-native-summary.json')['status'] != 'N18_ALL16_TASKS28_READOUTS_AUDITED':
        raise RuntimeError('Close previous whole screen before new attempt')
    sources = load(ROOT/'runs/N15-SLATER-PREFIX-ROUNDING-PREFLIGHT-v1/protocol.json')['sources']
    physical = load(ROOT/'runs/N18-HALF-SOURCE-NATIVE-SCREEN-v1/protocol.json')['physical']
    cases = validated_cases(sources, physical)
    LOCAL.mkdir(); (LOCAL/'inputs').mkdir()
    write(LOCAL/'worker.py', WORKER.read_bytes())
    write(LOCAL/'sqd-subsampling.py', REFERENCE.read_bytes())
    for c in cases:
        shutil.copyfile(c['operator'], LOCAL/'inputs'/(c['case']+'-operator.npz'))
    files = ['worker.py', 'sqd-subsampling.py'] + ['inputs/'+c['case']+'-operator.npz' for c in cases]
    save(LOCAL/'protocol.json', dict(version='N19_ALL4_TN_SOURCE_COST_AND_ORIGINAL_SUBSPACE_FEASIBILITY_V1',
        cases=cases, MPS_max_bond=50, MPS_cutoff=1e-10, shots=10000, sampling_seed=190101,
        batch_seed=0, samples_per_batch=4000, num_batches=10, max_half_dimension=4000,
        per_case_wall_seconds=240, maximum_campaign_wall_seconds=1200,
        input_SHA256={name:sha256((LOCAL/name).read_bytes()) for name in files},
        reference_SHA256=sha256(REFERENCE.read_bytes()),
        reference_url='https://raw.githubusercontent.com/Qiskit/qiskit-addon-sqd/0.12.0/qiskit_addon_sqd/subsampling.py',
        author_commit='b52bca05650e60d890b06ac0a3b6d167303d0a90',
        worker_SHA256=sha256(WORKER.read_bytes()), preparer_SHA256=sha256(Path(__file__).read_bytes()),
        tests_SHA256=sha256((ROOT/'qualification-N19-preexecution-v1-tests.xml').read_bytes()),
        Python=PYTHON, packages=PACKAGES, explicit_threads=1, allocated_CPUs=1,
        expected_classical_surrogate_samples=40000, new_exact_model_shots=0, actual_QPU_shots=0,
        molecular_energy_evaluations=0, optimizer_calls=0, GPU_allocations=0,
        scope='ONE_UNCHANGED_L1_PER_ALL4_MPS_COST_SECTOR_AND_AUTHOR_BATCH_SIZE_SCREEN_NO_CI_ENERGY_NO_OPTIMIZATION',
        full_paper_reproduction=False, independent_quantum_confirmation=False, goal_complete=False))
    print('N19_FROZEN_ALL4_MPS10K_10BATCHES_NO_ENERGY_NO_OPTIMIZER', flush=True)


def submit():
    spec = load(LOCAL/'protocol.json'); capacity()
    if (LOCAL/'submission-intent.json').exists() or (LOCAL/'job.json').exists():
        raise RuntimeError('Existing N19 dispatch; no blind retry')
    if sha256(WORKER.read_bytes()) != spec['worker_SHA256'] or sha256(Path(__file__).read_bytes()) != spec['preparer_SHA256']:
        raise RuntimeError('Frozen executable changed')
    for name, expected in spec['input_SHA256'].items():
        if sha256((LOCAL/name).read_bytes()) != expected:
            raise RuntimeError('Frozen payload changed')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW', timeout=30).strip() != 'NEW':
        raise RuntimeError('Existing remote N19 work')
    ssh('mkdir -p '+shlex.quote(REMOTE+'/inputs'), timeout=30)
    for name in ['worker.py', 'protocol.json', 'sqd-subsampling.py']:
        send(LOCAL/name, REMOTE+'/'+name)
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(LOCAL/'inputs')+'/.',HOST+':'+REMOTE+'/inputs/'], check=True, timeout=50)
    wrap = shlex.join(['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','RAYON_NUM_THREADS=1',
        'QISKIT_NUM_PROCS=1','JAX_PLATFORMS=cpu','PYTHONPATH='+':'.join(PACKAGES),PYTHON,'-u',REMOTE+'/worker.py',REMOTE])
    command = shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=1','--mem=8G','--time=00:20:00',
        '--job-name=frontier-N19-TN-feasibility','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err','--wrap='+wrap])
    save(LOCAL/'submission-intent.json', dict(command=command, protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    jid = ssh(command, timeout=30).strip().split(';')[0]
    if not jid.isdigit():
        raise RuntimeError('Uncertain submission; reconcile exact job, do not resubmit')
    save(LOCAL/'job.json', dict(job_id=jid, remote=REMOTE, GPU_allocations=0))
    print('N19_CPU_ONLY_SUBMITTED', jid, flush=True)


def status():
    job = load(LOCAL/'job.json')['job_id']
    q = ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%R"', timeout=30)
    live = '\n'.join(x for x in q.splitlines() if x.split('|')[0] == job)
    account = ssh('sacct -X -n -j '+job+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P', timeout=30)
    save(LOCAL/'status.json', dict(job_id=job, live=live, accounting=account))
    print(live or account or 'UNRESOLVED_NO_RESTART', flush=True)


def collect():
    from . import collect_small_pool_bundle as collector
    collector.LOCAL=LOCAL; collector.REMOTE=REMOTE; collector.collect()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','submit','status','collect'])
    args=parser.parse_args(); globals()[args.action]()
