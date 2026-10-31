"""Sixteen paired native Trim readouts, no new sampling or orbital changes."""
import argparse
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import process_time, perf_counter
import traceback
import xml.etree.ElementTree as ET
import numpy as np
import qiskit_addon_sqd.fermion as native_fermion
import qiskit_addon_sqd.trim as native_trim
import qiskit_addon_sqd.subsampling as native_subsampling
from .paths import ROOT, OLD
from .io import load, save, write, arrays, sha256, space
from .trim_readout import run as trim_run

PROTOCOL = ROOT/'protocol-T1-NATIVE-TRIM.json'
DIRECTORY = ROOT/'runs/T1-NATIVE-TRIM-v1'
FREEZE = ROOT/'freeze-T1-NATIVE-TRIM.json'


def binding():
    settings = load(PROTOCOL)
    suites = list(ET.parse(ROOT/'qualification-native-trim-v2-tests.xml').getroot().iter('testsuite'))
    if sum(int(x.get('tests', '0')) for x in suites) != 5 or any(
            int(x.get(k, '0')) for x in suites for k in ['errors', 'failures', 'skipped']):
        raise RuntimeError('Native Trim/electron-vector qualification incomplete')
    old_path = ROOT/'runs/R2-COMMON-CI-v1/freeze.json'
    old = load(old_path)
    names = ['lowest_penalized.py', 'lowest_orbital.py', 'hole_rdm.py', 'energy_cache.py']
    for name in names:
        if sha256((ROOT/'frontierloop'/name).read_bytes()) != old['code_SHA256'][name]:
            raise RuntimeError('Unchanged qualified C7/R2 kernel has drifted')
    sources = []
    for case in settings['cases']:
        for method in settings['methods']:
            cell = next(x for x in old['sources'] if x['case'] == case and x['method_id'] == method)
            if cell['status'] != 'AVAILABLE':
                raise RuntimeError('Do not silently omit a fixed source')
            for key, checksum in [('payload', 'payload_SHA256'), ('source_result', 'source_result_SHA256')]:
                if sha256(Path(cell[key]).read_bytes()) != cell[checksum]:
                    raise RuntimeError('Immutable source bytes changed')
            baseline = ROOT/'runs/R2-COMMON-CI-v1'/case/method/'result.json'
            row = load(baseline)
            if row['status'] != 'COMPLETED' or row['source_payload_SHA256'] != cell['payload_SHA256'] or row['r_cap'] != 128:
                raise RuntimeError('Fixed R2 frequency endpoint/context missing')
            physical = OLD/'data/physical'/case
            sources.append({**cell, 'R2_baseline': str(baseline), 'R2_baseline_SHA256': sha256(baseline.read_bytes()),
                'physical_SHA256': {name: sha256((physical/name).read_bytes()) for name in ['problem.json', 'integrals.npz']}})
    if len(sources) != settings['expected_sources']:
        raise ValueError('Unexpected T1 source count')
    return {'protocol_SHA256': sha256(PROTOCOL.read_bytes()), 'R2_freeze_SHA256': sha256(old_path.read_bytes()),
        'qualification_SHA256': sha256((ROOT/'qualification-native-trim-v2-tests.xml').read_bytes()),
        'implementation_SHA256': {str(p): sha256(p.read_bytes()) for p in [Path(__file__),
            ROOT/'frontierloop/trim_readout.py', ROOT/'test_trim_readout.py']+[ROOT/'frontierloop'/x for x in names]},
        'native_package_sources_SHA256': {str(Path(module.__file__)): sha256(Path(module.__file__).read_bytes())
            for module in [native_fermion, native_trim, native_subsampling]},
        'versions': {k: version(k) for k in ['qiskit-addon-sqd', 'qiskit', 'pyscf', 'ffsim', 'scipy', 'numpy']},
        'sources': sources, 'new_model_shots': 0, 'FCI_reference_in_selection': False}


def run_cell(cell, settings):
    target = DIRECTORY/cell['case']/cell['method_id']
    if (target/'started.json').exists():
        raise RuntimeError('Cell already started; never blind-retry partial/terminal evidence')
    physical = OLD/'data/physical'/cell['case']
    meta = load(physical/'problem.json')
    if meta['problem_id'] != cell['expected_problem_id']:
        raise ValueError('Source physical identity mismatch')
    n, ne = meta['norb'], (meta['nalpha'], meta['nbeta'])
    with np.load(cell['payload'], allow_pickle=False) as a:
        ids, counts = a['blocked_ids'].copy(), a['counts'].copy()
    if int(counts.sum()) != cell['source_final_model_shots'] and cell['kind'] != 'CLASSICAL_FULL_COUNTS':
        raise ValueError('Original model shot count changed')
    if cell['kind'] == 'CLASSICAL_FULL_COUNTS' and int(counts.sum()) != cell['classical_draws']:
        raise ValueError('Original classical draw count changed')
    hf = int(meta['hf_id'])
    ha = sum(((hf >> (2*n-1-2*q)) & 1) << q for q in range(n))
    hb = sum(((hf >> (2*n-2-2*q)) & 1) << q for q in range(n))
    if ha != hb:
        raise ValueError('Registered spin-balanced HF mismatch')
    save(target/'request.json', {'source': cell, 'settings': settings, 'HF_single_spin_string': ha,
        'original_final_observations': int(counts.sum()), 'new_model_shots': 0,
        'not_new_independent_sampler_evidence': True})
    save(target/'started.json', {'pid': os.getpid(), 'request_SHA256': sha256((target/'request.json').read_bytes())})
    cpu, wall = process_time(), perf_counter()
    def observed(calls):
        save(target/'CI-history.json', {'calls': calls, 'CPU_seconds': process_time()-cpu,
            'wall_seconds': perf_counter()-wall, 'new_model_shots': 0})
        print('T1_CI', cell['case'], cell['method_id'], calls[-1]['round'], calls[-1]['batch'],
              calls[-1]['actual_CI_dimension'], calls[-1]['energy_Ha'], flush=True)
    try:
        with np.load(physical/'integrals.npz', allow_pickle=False) as a:
            record, state = trim_run(a['h1'], a['eri'], meta['core_energy'], ne, ids, counts, ha, settings, observed)
        write(target/'final-selected-CI-state.npz', arrays(amplitudes=state.amplitudes,
            strings_a=state.ci_strs_a, strings_b=state.ci_strs_b, norb=np.array(n), nelec=np.array(ne)))
        record.update(case=cell['case'], method=cell['method_id'], physical_problem_id=meta['problem_id'],
            source_payload_SHA256=cell['payload_SHA256'],
            final_state_SHA256=sha256((target/'final-selected-CI-state.npz').read_bytes()),
            source_kind=cell['kind'], source_total_model_shot_budget=cell['source_total_budget'],
            source_final_model_shots=cell['source_final_model_shots'],
            original_classical_draws=cell.get('classical_draws', 0),
            full_attempt_CPU_seconds=process_time()-cpu, full_attempt_wall_seconds=perf_counter()-wall,
            source_construction_and_initialization_cost_not_in_new_readout_cost=True,
            old_frequency_endpoint=load(cell['R2_baseline']),
            R2_baseline_SHA256=cell['R2_baseline_SHA256'],
            selected_CI_vector_saved=True, full_Fock_statevector_saved=False,
            actual_QPU_shots=0, goal_complete=False)
        save(target/'result.json', record)
        return 'COMPLETED'
    except Exception as exc:
        status = 'CI_OR_TOTAL_WORK_CAP_PRESERVED' if isinstance(exc, TimeoutError) else 'FAILED_ENGINEERING_PRESERVED'
        history = load(target/'CI-history.json') if (target/'CI-history.json').exists() else {'calls': []}
        save(target/'failure.json', {'status': status, 'error': repr(exc), 'traceback': traceback.format_exc(),
            'last_completed_CI_count': len(history['calls']), 'unfinished_CI_matvec_count': None,
            'full_attempt_CPU_seconds': process_time()-cpu, 'full_attempt_wall_seconds': perf_counter()-wall,
            'new_model_shots': 0, 'not_a_negative_quantum_energy_result': True})
        print('T1_FAILURE', cell['case'], cell['method_id'], repr(exc), flush=True)
        return status


def main(freeze_only=False):
    b = binding()
    if freeze_only:
        if FREEZE.exists() and load(FREEZE) != b:
            raise RuntimeError('Never replace T1 freeze')
        save(FREEZE, b); print('T1_FROZEN', len(b['sources']), 'sources; new model shots0'); return
    if load(FREEZE) != b:
        raise RuntimeError('T1 frozen setup changed')
    if (DIRECTORY/'started.json').exists():
        raise RuntimeError('T1 already started; preserve and inspect instead of blind restart')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (DIRECTORY/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        settings = load(PROTOCOL); space(); cpu, wall = process_time(), perf_counter()
        save(DIRECTORY/'started.json', {'pid': os.getpid(), 'freeze_SHA256': sha256(FREEZE.read_bytes()),
            'authority': 'SIXTEEN_EXISTING_SOURCE_NATIVE_TRIM_READOUTS_NO_NEW_SHOTS'})
        states = []
        for cell in b['sources']:
            space()
            if sum(p.stat().st_size for p in DIRECTORY.rglob('*') if p.is_file()) > settings['output_cap_bytes']:
                raise OSError('Fixed T1 output cap; preserve progress')
            status = run_cell(cell, settings)
            states.append({'case': cell['case'], 'method': cell['method_id'], 'status': status})
            save(DIRECTORY/'progress.json', {'states': states, 'terminal_cells': len(states),
                'planned': settings['expected_sources'], 'new_model_shots': 0,
                'CPU_seconds': process_time()-cpu, 'wall_seconds': perf_counter()-wall})
        save(DIRECTORY/'terminal.json', {'status': 'ALL_FIXED_SOURCES_TERMINAL', 'states': states,
            'CPU_seconds': process_time()-cpu, 'wall_seconds': perf_counter()-wall,
            'new_model_shots': 0, 'goal_complete': False})
        print('T1_TERMINAL', len(states), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--freeze', action='store_true'); args = p.parse_args(); main(args.freeze)
