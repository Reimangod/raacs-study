"""T2: all three budget-filling bases frozen BEFORE their new root energies."""
import argparse
from importlib.metadata import version
from pathlib import Path
from time import process_time, perf_counter
import xml.etree.ElementTree as ET
import traceback
import numpy as np
import ffsim
from .paths import ROOT, OLD
from .io import load, save, sha256, digest, space
from .pool_ranking import selections
from .lowest_orbital import Reader
from .native_trim_screen import binding as t1_binding

DIRECTORY = ROOT/'runs/T2-POOLED-SELECTION-v1'
FREEZE = ROOT/'freeze-T2-POOLED-SELECTION.json'
PROTOCOL = ROOT/'protocol-T2-POOLED-SELECTION.json'


def plans():
    frozen = load(ROOT/'freeze-T1-NATIVE-TRIM.json')
    if t1_binding() != frozen or load(ROOT/'runs/T1-NATIVE-TRIM-v1/terminal.json')['status'] != 'ALL_FIXED_SOURCES_TERMINAL':
        raise RuntimeError('Complete unchanged T1 required')
    result = []
    for source in frozen['sources']:
        p = ROOT/'runs/T1-NATIVE-TRIM-v1'/source['case']/source['method_id']/'result.json'
        old = load(p); request = load(p.parent/'request.json')
        n = load(OLD/'data/physical'/source['case']/'problem.json')['norb']
        with np.load(source['payload'], allow_pickle=False) as a:
            selected = selections(old['calls'], a['blocked_ids'], a['counts'], n, request['HF_single_spin_string'])
        result.append({'source': source, 'T1_result_path': str(p), 'T1_result_SHA256': sha256(p.read_bytes()),
            'selection': selected, 'selection_SHA256': digest(selected)})
    return result


def binding():
    xml = ROOT/'qualification-pool-ranking-v1-tests.xml'
    suites = list(ET.parse(xml).getroot().iter('testsuite'))
    if sum(int(s.get('tests', '0')) for s in suites) != 4 or any(int(s.get(k, '0')) for s in suites for k in ['errors', 'failures', 'skipped']):
        raise RuntimeError('Pooled/native-parity/metamorphic qualifications incomplete')
    return {'protocol_SHA256': sha256(PROTOCOL.read_bytes()), 'qualifier_SHA256': sha256(xml.read_bytes()),
        'code_SHA256': {str(p): sha256(p.read_bytes()) for p in [Path(__file__), ROOT/'frontierloop/pool_ranking.py', ROOT/'test_pool_ranking.py']},
        'T1_freeze_SHA256': sha256((ROOT/'freeze-T1-NATIVE-TRIM.json').read_bytes()),
        'versions': {k: version(k) for k in ['qiskit-addon-sqd', 'qiskit', 'pyscf', 'ffsim', 'numpy', 'scipy']},
        'plans': plans(), 'new_model_shots': 0}


def evaluate_cell(plan, spec):
    source = plan['source']; case, method = source['case'], source['method_id']
    target = DIRECTORY/case/method
    if (target/'started.json').exists():
        raise RuntimeError('Preserve prior T2 cell instead of blind retry')
    save(target/'started.json', {'frozen_selection_SHA256': plan['selection_SHA256'], 'new_model_shots': 0})
    old = load(plan['T1_result_path']); baseline = load(source['R2_baseline'])
    cache = {tuple(baseline['CI_strings']): (baseline, source['R2_baseline'], baseline['incremental_CPU_seconds'])}
    for index, row in enumerate(old['calls']):
        cache.setdefault(tuple(row['CI_strings']), (row, plan['T1_result_path']+'#calls/'+str(index), row['CPU_seconds']))
    meta = load(OLD/'data/physical'/case/'problem.json')
    with np.load(OLD/'data/physical'/case/'integrals.npz', allow_pickle=False) as a:
        ham = ffsim.MolecularHamiltonian(a['h1'], a['eri'], constant=meta['core_energy'])
    screen_cost = sum(x['CPU_seconds'] for x in old['calls'] if x['round'] == 0)
    wall, cpu = perf_counter(), process_time(); rows = []; new_roots = 0
    for arm in spec['arms']:
        basis = tuple(plan['selection']['arms'][arm]); event_id = digest([meta['problem_id'], basis, .1, 'unchangedC7-SA-k2-seed20261025'])
        reused = basis in cache
        try:
            if reused:
                root, origin, cost = cache[basis]
            else:
                remaining = min(spec['per_source_new_wall_cap_seconds']-(perf_counter()-wall),
                    spec['per_source_new_CPU_cap_seconds']-(process_time()-cpu), spec['per_CI_wall_cap_seconds'])
                if remaining <= 0:
                    raise TimeoutError('T2 source-wide new-work cap')
                tick, start_cpu = perf_counter(), process_time()
                root, _ = Reader(meta['norb'], (meta['nalpha'], meta['nbeta']), np.array(basis, np.int64),
                    {'spin_target_S_squared': 0, 'spin_penalty_Ha': .1, 'per_CI_wall_cap_seconds': remaining}).diagonalize(ham)
                root.update(CI_strings=list(basis), CPU_seconds=process_time()-start_cpu,
                    wall_seconds=perf_counter()-tick, new_model_shots=0, event_id=event_id)
                save(target/('root-'+event_id+'.json'), root)
                origin = str(target/('root-'+event_id+'.json')); cost = root['CPU_seconds']
                cache[basis] = (root, origin, cost); new_roots += 1
            attributed_screen = screen_cost if arm != 'SAME_UNION_FREQUENCY' and plan['selection']['screening_CI_required_operationally'] else 0.
            row = {'arm': arm, 'status': 'COMPLETED', 'energy_Ha': root['energy_Ha'],
                'penalized_energy_Ha': root['penalized_energy_Ha'], 'S2': root['spin_square'],
                'actual_CI_dimension': len(basis)**2, 'CI_strings': list(basis), 'event_id': event_id,
                'evaluation_reused': reused, 'evaluation_origin': origin,
                'CI_numeric_stage_CPU_seconds_inherited_or_measured': cost,
                'required_screen_CI_CPU_seconds_attributed': attributed_screen,
                'accounted_operational_CI_CPU_seconds_not_cold_total': cost+attributed_screen,
                'quantum_source_build_cost_and_driver_overhead_not_in_numeric_proxy': True,
                'same_physical_basis_as_R2': basis == tuple(baseline['CI_strings']),
                'new_model_shots': 0, 'independent_quantum_evidence': False}
        except Exception as exc:
            row = {'arm': arm, 'status': 'CI_OR_TOTAL_WORK_CAP_PRESERVED' if isinstance(exc, TimeoutError) else 'FAILED_ENGINEERING_PRESERVED',
                'error': repr(exc), 'traceback': traceback.format_exc(), 'new_model_shots': 0,
                'not_negative_quantum_energy_evidence': True}
        rows.append(row); save(target/'progress.json', {'rows': rows, 'new_CI_roots': new_roots,
            'new_CPU_seconds': process_time()-cpu, 'new_wall_seconds': perf_counter()-wall})
        print('T2', case, method, arm, row['status'], row.get('energy_Ha'), 'REUSED', reused, flush=True)
    save(target/'terminal.json', {'case': case, 'method': method, 'rows': rows, 'new_CI_roots': new_roots,
        'new_CPU_seconds': process_time()-cpu, 'new_wall_seconds': perf_counter()-wall,
        'old_T1_screen_CPU_seconds': screen_cost, 'old_T1_result_SHA256': plan['T1_result_SHA256'],
        'frozen_selection_SHA256': plan['selection_SHA256'], 'new_model_shots': 0})
    return {'case': case, 'method': method, 'new_CI_roots': new_roots,
        'statuses': [r['status'] for r in rows]}


def main(freeze_only=False):
    b = binding()
    if freeze_only:
        if FREEZE.exists() and load(FREEZE) != b:
            raise RuntimeError('Never replace T2 freeze')
        save(FREEZE, b); print('T2_FROZEN', len(b['plans'])*3, 'arm bases; new roots not evaluated'); return
    if load(FREEZE) != b:
        raise RuntimeError('T2 input or basis drift')
    if (DIRECTORY/'started.json').exists():
        raise RuntimeError('T2 already started; inspect exact live process/terminal, never blind restart')
    spec = load(PROTOCOL); space(); wall, cpu = perf_counter(), process_time()
    save(DIRECTORY/'started.json', {'freeze_SHA256': sha256(FREEZE.read_bytes()), 'new_model_shots': 0})
    states = []
    for p in b['plans']:
        space()
        if sum(x.stat().st_size for x in DIRECTORY.rglob('*') if x.is_file()) > spec['output_cap_bytes']:
            raise OSError('T2 output budget')
        states.append(evaluate_cell(p, spec)); save(DIRECTORY/'progress.json', {'states': states})
    save(DIRECTORY/'terminal.json', {'status': 'ALL_FIXED_ARM_BASES_TERMINAL', 'states': states,
        'new_CI_roots': sum(x['new_CI_roots'] for x in states),
        'new_CPU_seconds': process_time()-cpu, 'new_wall_seconds': perf_counter()-wall,
        'new_model_shots': 0, 'goal_complete': False})
    print('T2_TERMINAL', len(states), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--freeze', action='store_true'); a = p.parse_args(); main(a.freeze)
