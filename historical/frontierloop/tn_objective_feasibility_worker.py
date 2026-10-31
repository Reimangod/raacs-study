"""N19 bounded MPS/source and paper-type subspace feasibility, no CI energy.

Independent tiny full-Fock parity precedes all molecular draws. Production
states are never densified. SQD 0.12.0 subsample is executed unmodified as an
isolated, inspected function; no dependency monkeypatch or shared-env change.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import process_time, perf_counter
import numpy as np


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def upstream_subsample(path):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'subsample']
    if len(nodes) != 1 or nodes[0].decorator_list:
        raise ValueError('Only the inspected unmodified public subsample function is allowed')
    namespace = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['subsample']


def batch_spaces(ids, counts, n, nelec, sampler, seed):
    ids = np.asarray(ids, dtype=np.uint64); counts = np.asarray(counts)
    if ids.ndim != 1 or counts.shape != ids.shape or len(set(map(int, ids))) != len(ids) or np.any(counts <= 0):
        raise ValueError('Unique full-register counts required')
    mask = (1 << n) - 1
    valid = np.array([(int(i) & mask).bit_count() == nelec[0] and (int(i) >> n).bit_count() == nelec[1] for i in ids])
    rejected = int(counts[~valid].sum())
    if not valid.any():
        raise ValueError('No correct-sector MPS samples')
    good = ids[valid]; probabilities = counts[valid] / counts[valid].sum()
    # Same first-iteration, no-recovery input convention as SQD 0.12.0.
    bits = ((good[:, None] >> np.arange(2*n-1, -1, -1, dtype=np.uint64)) & 1).astype(bool)
    batches = sampler(bits, probabilities, 4000, 10, rand_seed=seed)
    result = []; arrays = {}
    for index, batch in enumerate(batches):
        powers = 1 << np.arange(n-1, -1, -1, dtype=np.uint64)
        alpha, ca = np.unique(batch[:, n:].astype(np.uint64) @ powers, return_counts=True)
        beta, cb = np.unique(batch[:, :n].astype(np.uint64) @ powers, return_counts=True)
        merged = np.concatenate((alpha, beta)); frequencies = np.concatenate((ca, cb))
        ranked = merged[np.argsort(frequencies)[::-1]]
        _, first = np.unique(ranked, return_index=True)
        strings = np.sort(ranked[np.sort(first)][:4000])
        arrays[f'half_strings_{index}'] = strings
        result.append(dict(batch=index, selected_unique_full_configurations=len(batch),
                           half_configurations=len(strings), joint_dimension=len(strings)**2,
                           coefficient_vector_complex128_bytes=16*len(strings)**2))
    return result, arrays, rejected


def circuit_from_operator(operator, n, nelec):
    import ffsim
    from qiskit import QuantumCircuit
    q = QuantumCircuit(2*n)
    q.append(ffsim.qiskit.PrepareHartreeFockJW(n, tuple(nelec)), range(2*n))
    q.append(ffsim.qiskit.UCJOpSpinBalancedJW(operator), range(2*n))
    return q.decompose(reps=2)


def mps_from_circuit(q, max_bond=50, cutoff=1e-10):
    import quimb.tensor as qtn
    from qiskit_quimb import quimb_circuit
    return quimb_circuit(q, quimb_circuit_class=qtn.CircuitMPS,
                         max_bond=max_bond, cutoff=cutoff, progbar=False)


def qualify(root, spec):
    cpu, start = process_time(), perf_counter()
    import ffsim
    from qiskit.quantum_info import Statevector
    op = ffsim.random.random_ucj_op_spin_balanced(2, n_reps=1, with_final_orbital_rotation=True, seed=190001)
    q = circuit_from_operator(op, 2, (1, 1)); mps = mps_from_circuit(q)
    actual = np.asarray(mps.to_dense()).reshape(-1)
    reverse = np.array([int(f'{i:04b}'[::-1], 2) for i in range(16)])
    expected = Statevector.from_instruction(q).data[reverse]
    error = float(np.max(np.abs(actual-expected)))
    if error > 1e-10 or abs(np.vdot(actual, actual)-1) > 1e-10:
        raise RuntimeError('Tiny independent full-Fock MPS/Qiskit parity failed')
    sampler = upstream_subsample(root/'sqd-subsampling.py')
    rows, _, rejected = batch_spaces(np.array([5, 6, 9, 10], np.uint64), np.array([40, 30, 20, 10]), 2, (1, 1), sampler, 0)
    if rejected or len(rows) != 10 or any(r['joint_dimension'] != 4 for r in rows):
        raise RuntimeError('Tiny actual author batch semantics failed')
    save(root/'qualification/result.json', dict(status='TINY_MPS_FULL_FOCK_AND_AUTHOR_BATCH_PARITY_PASSED',
        maximum_state_error=error, full_Fock_qubits=4, molecular_full_state_certified=False,
        reference_SHA256=spec['reference_SHA256'], new_molecular_energy_evaluations=0,
        CPU_seconds=process_time()-cpu, wall_seconds=perf_counter()-start))


def one_case(root, case):
    import ffsim
    spec = json.loads((root/'protocol.json').read_text())
    entry = next(x for x in spec['cases'] if x['case'] == case)
    folder = root/'cells'/case; folder.mkdir(parents=True, exist_ok=True)
    cpu, start = process_time(), perf_counter()
    if (folder/'started.json').exists():
        raise RuntimeError('No duplicate or partial MPS attempt')
    save(folder/'started.json', dict(case=case, problem_id=entry['problem_id'], operator_SHA256=entry['operator_SHA256'], pid=os.getpid()))
    with np.load(root/'inputs'/(case+'-operator.npz'), allow_pickle=False) as a:
        op = ffsim.UCJOpSpinBalanced(a['diag_coulomb_mats'], a['orbital_rotations'], a['final_orbital_rotation'])
    q = circuit_from_operator(op, entry['norb'], entry['nelec'])
    build_start = perf_counter(); mps = mps_from_circuit(q)
    build_seconds = perf_counter()-build_start
    norm = float(abs(mps.psi.norm()))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError('Invalid approximate MPS norm')
    tick = perf_counter(); observed = []; sample_path = folder/'samples-prefix.json'
    # Stream counted progress, so a timeout never converts an incomplete draw to zero.
    for sample in mps.sample(spec['shots'], seed=spec['sampling_seed']):
        if len(sample) != 2*entry['norb'] or set(sample)-{'0', '1'}:
            raise RuntimeError('Unexpected MPS bitstring convention')
        observed.append(int(sample[::-1], 2))
        if len(observed) % 1000 == 0:
            ids, counts = np.unique(np.array(observed, np.uint64), return_counts=True)
            np.savez_compressed(folder/'partial-counts.npz', blocked_ids=ids, counts=counts.astype(np.uint64))
            save(sample_path, dict(generated=len(observed), complete=False, actual_QPU_shots=0))
    sampling_seconds = perf_counter()-tick
    ids, counts = np.unique(np.array(observed, np.uint64), return_counts=True)
    np.savez_compressed(folder/'counts.npz', blocked_ids=ids, counts=counts.astype(np.uint64))
    sampler = upstream_subsample(root/'sqd-subsampling.py')
    spaces, arrays, rejected = batch_spaces(ids, counts, entry['norb'], entry['nelec'], sampler, spec['batch_seed'])
    np.savez_compressed(folder/'batch-spaces.npz', **arrays)
    result = dict(case=case, problem_id=entry['problem_id'], status='COMPLETED',
        operator_SHA256=entry['operator_SHA256'], MPS_max_bond=50, MPS_cutoff=1e-10,
        approximate_MPS_norm=norm, actual_max_bond=int(mps.psi.max_bond()),
        norm_is_not_exact_state_fidelity=True, sampling_seed=spec['sampling_seed'], batch_seed=spec['batch_seed'],
        actual_classical_surrogate_samples=len(observed), wrong_sector_samples=rejected,
        new_exact_model_shots=0, actual_QPU_shots=0, unique_full_configurations=len(ids),
        counts_SHA256=sha((folder/'counts.npz').read_bytes()),
        spaces_SHA256=sha((folder/'batch-spaces.npz').read_bytes()), batch_spaces=spaces,
        MPS_build_wall_seconds=build_seconds, sampling_wall_seconds=sampling_seconds,
        CPU_seconds=process_time()-cpu, wall_seconds=perf_counter()-start,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        CI_energy_evaluations=0, independent_quantum_confirmation=False, full_paper_reproduction=False)
    save(sample_path, dict(generated=len(observed), complete=True, actual_QPU_shots=0))
    save(folder/'result.json', result)


def run(root):
    spec = json.loads((root/'protocol.json').read_text()); start = perf_counter()
    if (root/'started.json').exists():
        raise RuntimeError('Existing N19 attempt; no blind rerun')
    for name, expected in spec['input_SHA256'].items():
        if sha((root/name).read_bytes()) != expected:
            raise RuntimeError('Frozen N19 file changed: '+name)
    save(root/'started.json', dict(protocol_SHA256=sha((root/'protocol.json').read_bytes()),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), thread_env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        molecular_energy_evaluations=0, GPU_allocations=0))
    qualify(root, spec)
    results = []; child_usage = resource.getrusage(resource.RUSAGE_CHILDREN); cpu = process_time()
    for entry in spec['cases']:
        folder = root/'cells'/entry['case']; folder.mkdir(parents=True, exist_ok=True)
        timed_out = False
        with (folder/'stdout.txt').open('wb') as out, (folder/'stderr.txt').open('wb') as err:
            try:
                completed = subprocess.run([sys.executable, str(root/'worker.py'), str(root), '--case', entry['case']],
                    stdout=out, stderr=err, timeout=spec['per_case_wall_seconds'], check=False)
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True; exit_code = None
        if (folder/'result.json').exists() and exit_code == 0:
            result = json.loads((folder/'result.json').read_text())
        else:
            progress = json.loads((folder/'samples-prefix.json').read_text()) if (folder/'samples-prefix.json').exists() else None
            result = dict(case=entry['case'], status='TIME_LIMIT_PRESERVED' if timed_out else 'FAILED_PRESERVED',
                exit_code=exit_code, partial_sampling_progress=progress, unknown_unrecorded_partial_draws_not_zero=True)
            save(folder/'failure.json', result)
        results.append(result); save(root/'progress.json', dict(states=results))
    final = resource.getrusage(resource.RUSAGE_CHILDREN)
    save(root/'terminal.json', dict(status='N19_ALL4_FEASIBILITY_ATTEMPTS_TERMINAL', states=results,
        CPU_seconds_children=final.ru_utime+final.ru_stime-child_usage.ru_utime-child_usage.ru_stime,
        CPU_seconds_supervisor_after_qualification=process_time()-cpu,
        wall_seconds_including_qualification=perf_counter()-start, new_CI_energy_evaluations=0,
        new_exact_model_shots=0, actual_QPU_shots=0, GPU_allocations=0, goal_complete=False))


if __name__ == '__main__':
    root = Path(sys.argv[1])
    if '--case' in sys.argv:
        one_case(root, sys.argv[sys.argv.index('--case')+1])
    else:
        run(root)
