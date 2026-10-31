"""External-process total workflow timing of unchanged native samplers.

Only physical integrals/HF enter the numerical worker. Ground references and
historical trained/optimized circuits are absent from its input. Serial, durable,
bounded execution; no automatic replay of partial work.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('RAACS-full-cost', 'LUCJ-CCSD-2-fixed', 'LUCJ-CCSD-2-QSCI4')
FREE_FLOOR = int(2.75 * 2**30)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def frozen(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError('Immutable workflow configuration/receipt differs: ' + str(path))
    else:
        save(path, value)


def configuration(mode):
    source = ROOT / 'results/standard-v1/references'
    cases = sorted(p.name for p in source.iterdir() if (p / 'reference.json').exists())
    if len(cases) != 30:
        raise RuntimeError('The complete original30-case physical cohort is required')
    if mode == 'quick':
        cases = ['H2_R0.7', 'H4_R1.1']
    files = ['workflow_cost_study.py', 'workflow_cost_analysis.py',
             'molecular_strong_controls.py', 'resource_adaptive_sampler.py',
             'molecular_lucj_primitives.py', 'spin_blocked_sampling.py',
             'mps_layout_qualification.py', 'mps_sampling.py',
             'circuit_sampling_runtime.py', 'integral_hamiltonian.py',
             'integral_connections.py', 'local_sampler_primitives.py', 'ansatz.py',
             'molecular_mps_scaling.py', 'common.py', 'study/core.py']
    return {'schema': 'given-integrals-total-workflow-v1', 'mode': mode,
            'queue': [{'case': c, 'method': m} for c in cases for m in METHODS],
            'physical_source_hashes': {c: {name: sha(source / c / name) for name in
                                       ('reference.json', 'reference.npz')} for c in cases},
            'total_model_shots': 2000, 'pilot_shots': 64, 'QSCI_LUCJ_points': 4,
            'R_grid': [8, 32, 64], 'RAACS_seed': 80111 if mode == 'quick' else 80113,
            'sampling': {'backend': 'mps', 'max_bond': 256, 'cutoff': 0.,
                         'construction_seconds_cap': 120., 'sampling_seconds_cap': 120.,
                         'RSS_cap_bytes': 2 * 2**30, 'cache_entries': 2},
            'free_floor_bytes': FREE_FLOOR, 'output_cap_bytes': 2**30,
            'RSS_cap_bytes': 2 * 2**30, 'cell_wall_cap_seconds': 900.,
            'classical_SCI_seconds': 60., 'CCSD_seconds': 180., 'CCSD_max_cycles': 150,
            'implementation': {p: sha(ROOT / 'benchmark' / p) for p in files},
            'vendor_circuits_sha256': sha(ROOT / 'vendor/ceo/adaptvqe/circuits.py'),
            'spec_sha256': sha(ROOT / 'study-v1/TOTAL_WORKFLOW_COST_V1.md'),
            'packages': {p: importlib.metadata.version(p) for p in
                         ('numpy', 'scipy', 'qiskit', 'ffsim', 'pyscf', 'quimb')},
            'timing_scope': 'Popen_to_child_exit_including_child_result_persistence_given_saved_integrals',
            'excludes': ['shared physical-input export', 'RHF and integral generation',
                         'parent receipt persistence', 'post-run audit and figures'],
            'fresh_interpreter_and_private_amplitudes': True,
            'sampling_provider': 'SpinBlockedCountSampler; logical circuits unchanged',
            'native_provider_substitutions': ['native.CircuitCountSampler', 'adaptive.CircuitCountSampler',
                                              'native.RAACSSpec.seed_only'],
            'OS_and_JIT_caches_controlled': False, 'hardware_shots': 0, 'FCI_in_policy': False}


def task_id(item, cfg):
    return digest({'item': item, 'configuration': cfg})


def stage_input(case, root):
    """No FCI arrays/labels copied; this shared export is separately timed."""
    import numpy as np
    source = ROOT / 'results/standard-v1/references' / case
    destination = root / 'physical-inputs' / case
    old = json.loads((source / 'reference.json').read_text())
    metadata = {k: old[k] for k in ('n_qubits', 'nalpha', 'nbeta', 'hf_id')}
    metadata['hf_energy_Ha'] = old['rhf_energy']
    metadata['schema'] = 'physical-integrals-HF-only-v1'
    if metadata['n_qubits'] > 16:
        raise ValueError('This timing cohort is the original small-system cohort, not new scaling')
    with np.load(source / 'reference.npz') as archive:
        h1, eri = archive['h1'].copy(), archive['eri'].copy()
    binding = {'metadata_sha256': sha(source / 'reference.json'),
               'integrals_archive_sha256': sha(source / 'reference.npz'),
               'h1_sha256': hashlib.sha256(h1.astype('<f8').tobytes()).hexdigest(),
               'eri_sha256': hashlib.sha256(eri.astype('<f8').tobytes()).hexdigest(),
               'FCI_fields_exported': False, 'coefficient_or_probability_arrays_exported': False}
    destination.mkdir(parents=True, exist_ok=True)
    frozen(destination / 'problem.json', metadata)
    frozen(destination / 'source-binding.json', binding)
    path = destination / 'integrals.npz'
    if path.exists():
        with np.load(path) as archive:
            assert set(archive.files) == {'h1', 'eri'}
            assert np.array_equal(archive['h1'], h1) and np.array_equal(archive['eri'], eri)
    else:
        with path.open('xb') as stream:
            np.savez_compressed(stream, h1=h1, eri=eri)
    return destination


def events(folder, method, result):
    """Reconstruct all paid raw samples and circuit proxies from durable events."""
    import sqlite3
    from .local_sampler_primitives import unpack_counts64
    adaptive = method == 'RAACS-full-cost'
    base = folder / 'native-RAACS' if adaptive else folder
    path = base / 'events.sqlite'
    rows, final, circuits = [], {}, {}
    if not path.exists():
        return {'draws': [], 'final_counts': {}, 'circuits': {}, 'known_model_shots': 0,
                'unknown_inflight': result.get('status') != 'COMPLETED', 'database': None,
                'database_sha256': None, 'gate_shot_proxy': 0, 'depth_shot_proxy': 0,
                'final_circuit_id': None}
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        resource_column = 'resources' if adaptive else 'payload'
        for cid, payload in db.execute(f'SELECT id,{resource_column} FROM circuits'):
            value = json.loads(payload)
            circuits[cid] = value if adaptive else value['resources']
        if adaptive:
            native_path = base / 'result.json'
            native = json.loads(native_path.read_text()) if native_path.exists() else {}
            final_ids = set(native.get('final_draw_ids', []))
            query = 'SELECT id,circuit,role,requested,counts64,metadata FROM draws'
        else:
            final_ids = {result.get('final_draw_id')}
            query = 'SELECT id,circuit,role,requested,counts,metadata FROM draws'
        gate, depth, known, inflight, final_cid = 0, 0, 0, False, None
        for did, cid, role, requested, blob, rawmeta in db.execute(query):
            counts, metadata, raw = {}, None, None
            if blob is None:
                inflight = True
            else:
                ids, nn = unpack_counts64(blob)
                counts = dict(zip(map(int, ids), map(int, nn)))
                metadata = json.loads(rawmeta)
                raw = sum(counts.values())
                assert raw == metadata['actual_raw_model_shots'] <= requested
                known += raw
                gate += raw * circuits[cid]['cnot_count']
                depth += raw * circuits[cid]['cnot_depth']
                if did in final_ids:
                    if final_cid is not None:
                        assert final_cid == cid
                    final_cid = cid
                    for x, n in counts.items():
                        final[x] = final.get(x, 0) + n
            rows.append({'id': did, 'circuit': cid, 'role': role, 'requested': requested,
                         'raw': raw, 'metadata': metadata})
    # RAACS writes a draw only after it returns; absent in-flight work on failure
    # cannot be established to be zero from that schema.
    inflight = inflight or (adaptive and result.get('status') != 'COMPLETED')
    return {'draws': rows, 'final_counts': final, 'circuits': circuits,
            'known_model_shots': known, 'unknown_inflight': inflight,
            'database': str(path.relative_to(folder)), 'database_sha256': sha(path),
            'gate_shot_proxy': gate, 'depth_shot_proxy': depth, 'final_circuit_id': final_cid}


def worker(root, index):
    # Everything below is inside the externally observed child process timer.
    from dataclasses import replace
    from threadpoolctl import threadpool_limits
    from . import molecular_strong_controls as native
    from . import resource_adaptive_sampler as adaptive
    from .spin_blocked_sampling import SpinBlockedCountSampler
    from .molecular_mps_scaling import no_full_state_guards
    from .resource_adaptive_sampler import integral_problem
    cfg = json.loads((root / 'manifest.json').read_text())
    if cfg != configuration(cfg['mode']):
        raise RuntimeError('Workflow numerical identity drift')
    item = cfg['queue'][index]
    folder = root / 'cases' / item['case'] / item['method']
    with (folder / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'worker-start.json').exists():
            raise RuntimeError('Started workflow may not be replayed for free')
        frozen(folder / 'worker-start.json', {'task_id': task_id(item, cfg), 'PID': os.getpid()})
        original_spec = native.RAACSSpec
        original_native_sampler, original_adaptive_sampler = native.CircuitCountSampler, adaptive.CircuitCountSampler
        try:
            native.RAACSSpec = lambda **kwargs: replace(original_spec(**kwargs), seed=cfg['RAACS_seed'])
            native.CircuitCountSampler = adaptive.CircuitCountSampler = SpinBlockedCountSampler
            with threadpool_limits(limits=1), no_full_state_guards():
                problem = root / 'physical-inputs' / item['case']
                result = native.cell(problem, folder, item['method'], cfg, root, folder / 'private-amplitudes')
                evidence = events(folder, item['method'], result)
                readouts = []
                if evidence['final_counts']:
                    if item['method'] != 'RAACS-full-cost':
                        # The existing native LUCJ workflow already persisted
                        # this complete R-grid. Do not repeat its diagonalizations.
                        readouts = json.loads((folder / 'readouts.json').read_text())
                    else:
                        backend, info = integral_problem(problem)
                        readouts = native.readouts(backend, info, evidence['final_counts'], [8, 32])
                        actual = json.loads((folder / 'native-RAACS/result.json').read_text())
                        readouts.append({'R': 64, 'energy_Ha': actual['final_QSCI_energy_Ha'],
                                         'actual_dimension': actual['actual_final_CI_dimension'],
                                         'selected_ids': actual['final_selected_ids'],
                                         'internal_residual': None,
                                         'source': 'existing native final R64 solve; residual independently audited later'})
                # Raw counts already persist in SQLite; no second copy in JSON.
                evidence.pop('final_counts')
                evidence['readouts'] = readouts
                evidence['result_sha256'] = sha(folder / 'result.json') if (folder / 'result.json').exists() else None
                evidence['peak_RSS_bytes'] = native.rss_peak()
                evidence['hardware_shots'] = 0
                save(folder / 'workflow-summary.json', evidence)
        finally:
            native.RAACSSpec = original_spec
            native.CircuitCountSampler, adaptive.CircuitCountSampler = original_native_sampler, original_adaptive_sampler


def observe_rss(pid):
    observed = subprocess.run(['ps', '-p', str(pid), '-o', 'rss='], capture_output=True, text=True)
    return int(observed.stdout.strip()) * 1024 if observed.returncode == 0 and observed.stdout.strip() else 0


def observe_process(proc, cfg, root, started):
    if time.monotonic() - started > cfg['cell_wall_cap_seconds']:
        return 'PROCESS_WALL_CAP'
    if observe_rss(proc.pid) > cfg['RSS_cap_bytes']:
        return 'PROCESS_RSS_CAP'
    if shutil.disk_usage(root).free < cfg['free_floor_bytes']:
        return 'PROCESS_FREE_FLOOR_BOUNDARY'
    return None


def prepare(root, mode):
    root = Path(root).resolve()
    if root.parent != ROOT / 'results' or root.name != f'total-workflow-{mode}-v1':
        raise ValueError('Use the separate versioned workflow result root')
    root.mkdir(exist_ok=True)
    cfg = configuration(mode)
    frozen(root / 'manifest.json', cfg)
    setup = root / 'input-export.json'
    if not setup.exists():
        tick = time.monotonic()
        for case in sorted({i['case'] for i in cfg['queue']}):
            stage_input(case, root)
        frozen(setup, {'wall_seconds': time.monotonic() - tick,
                      'scope': 'shared physical-only input export; excluded from method timer',
                      'new_quantum_shots': 0, 'new_FCI_solves': 0})
    return cfg


def run(root, mode):
    root = Path(root).resolve()
    if root.parent != ROOT / 'results' or root.name != f'total-workflow-{mode}-v1':
        raise ValueError('Use the separate versioned workflow result root')
    root.mkdir(exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = prepare(root, mode)
        terminals = []
        for index, item in enumerate(cfg['queue']):
            folder = root / 'cases' / item['case'] / item['method']
            folder.mkdir(parents=True, exist_ok=True)
            receipt_path = folder / 'process-receipt.json'
            expected = task_id(item, cfg)
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                assert receipt['task_id'] == expected
                for name, value in receipt.get('files', {}).items():
                    assert sha(folder / name) == value
                terminals.append(receipt)
                continue
            if (folder / 'process-start.json').exists():
                # This invocation owns no earlier process handle. A successor
                # may reconcile evidence, but may not execute the same item.
                save(root / 'execution-status.json', {'status': 'UNRECONCILED_STARTED_TASK_PRESERVED',
                     'index': index, 'terminal': len(terminals), 'planned': len(cfg['queue'])})
                return
            if shutil.disk_usage(root).free < cfg['free_floor_bytes']:
                save(root / 'execution-status.json', {'status': 'PENDING_CAPACITY_BOUNDARY', 'terminal': len(terminals),
                     'planned': len(cfg['queue']), 'next_index': index})
                return
            frozen(folder / 'process-start.json', {'task_id': expected, 'item': item, 'index': index})
            environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            command = [sys.executable, '-m', 'benchmark.workflow_cost_study', '--worker-root', str(root), '--worker-index', str(index)]
            with (folder / 'worker.log').open('xb') as log:
                tick = time.monotonic()
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=log, start_new_session=True)
                save(root / 'execution-status.json', {'status': 'RUNNING_ONE_FRESH_PROCESS', 'PID': process.pid,
                     'terminal': len(terminals), 'planned': len(cfg['queue']), 'index': index, 'task_id': expected})
                boundary = None
                while process.poll() is None:
                    boundary = observe_process(process, cfg, root, tick)
                    if boundary:
                        # Only this exact newly-created private group is stopped.
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait()
                        break
                    # Return promptly when the child exits, rather than adding
                    # a full polling sleep to short workflows.
                    try:
                        process.wait(timeout=.5)
                    except subprocess.TimeoutExpired:
                        pass
                wall = time.monotonic() - tick
            evidence = json.loads((folder / 'workflow-summary.json').read_text()) if (folder / 'workflow-summary.json').exists() else None
            result = json.loads((folder / 'result.json').read_text()) if (folder / 'result.json').exists() else None
            status = boundary or (result['status'] if result else 'CHILD_FAILURE_PRESERVED')
            receipt = {'task_id': expected, 'item': item, 'index': index, 'status': status,
                       'returncode': process.returncode, 'PID': process.pid, 'total_workflow_wall_seconds': wall,
                       'timing_scope': cfg['timing_scope'], 'hardware_shots': 0,
                       'all_model_work_known': evidence is not None and not evidence['unknown_inflight'],
                       'total_model_shots': None if evidence is None or evidence['unknown_inflight'] else evidence['known_model_shots'],
                       'files': {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name not in
                                 ('process-receipt.json', 'worker.lock') and p.suffix != '.tmp'}}
            frozen(receipt_path, receipt)
            terminals.append(receipt)
            save(root / 'execution-status.json', {'status': 'RUNNING', 'terminal': len(terminals), 'planned': len(cfg['queue'])})
            print(index, item['case'], item['method'], status, f'{wall:.3f}s', flush=True)
            if boundary or sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > cfg['output_cap_bytes']:
                save(root / 'execution-status.json', {'status': 'PENDING_RESOURCE_BOUNDARY_AFTER_PRESERVED_TERMINAL',
                     'terminal': len(terminals), 'planned': len(cfg['queue'])})
                return
        save(root / 'execution-status.json', {'status': 'ALL_WORKFLOW_TASKS_TERMINAL_PENDING_AUDIT',
             'terminal': len(terminals), 'planned': len(cfg['queue']), 'whole_goal_complete': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['quick', 'standard'], default='quick')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker-root', type=Path)
    parser.add_argument('--worker-index', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.worker_root is not None:
        worker(args.worker_root, args.worker_index)
    elif args.prepare_only:
        prepare(args.output or ROOT / f'results/total-workflow-{args.mode}-v1', args.mode)
    else:
        run(args.output or ROOT / f'results/total-workflow-{args.mode}-v1', args.mode)
