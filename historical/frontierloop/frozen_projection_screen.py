"""Bounded, zero-shot G2 preflight; old raw archives are opened read-only."""
import argparse
import gzip
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
import sqlite3
import tarfile
from time import perf_counter, process_time
import traceback
import xml.etree.ElementTree as ET
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector
from .paths import ROOT, OLD
from .io import load, save, write, sha256, digest, space
from .frozen_projection import restrict, verify, restore, synthesize, fermionic_relabel
from paperstudy.lucj import circuit_views
from paperstudy.sqdrift_execution import member_state


RUN = ROOT/'runs/G2-FROZEN-OCCUPATION-v1'
PROTOCOL = ROOT/'protocol-G2-FROZEN-OCCUPATION.json'
FREEZE = ROOT/'freeze-G2-FROZEN-OCCUPATION.json'


def resolve_inputs():
    spec = load(PROTOCOL)
    source = load(ROOT/spec['source_manifest'])
    cases = []
    db = sqlite3.connect('file:'+str(OLD/'raw-archives-v1/path-index.sqlite')+'?mode=ro', uri=True)
    try:
        for case in spec['cases']:
            cell = next(x for x in source['cells'] if x['case'] == case and x['method'] == spec['method'])
            row = {'case': case, 'problem_id': cell['expected_problem_id'], 'status': cell['status']}
            if cell['status'] != 'AVAILABLE':
                cases.append(row)
                continue
            path = Path(cell['source_result'])
            if sha256(path.read_bytes()) != cell['source_result_SHA256']:
                raise RuntimeError('Frozen C5 source changed')
            original = load(path)
            if original['problem_id'] != row['problem_id'] or original['n_excitations'] != 5:
                raise ValueError('Problem/N5 mismatch')
            members = []
            for index, record in enumerate(original['member_records'][:spec['members_per_available_source']]):
                mid = digest(record['spec'])
                if mid != record['member_id']:
                    raise ValueError('Member/spec ID mismatch')
                artifacts = {}
                for filename in ['member.json.gz', 'circuit.qpy.gz']:
                    relative = 'native-SqDRIFT/'+case+'/members/'+mid+'/'+filename
                    p = OLD/relative
                    if p.exists():
                        blob = p.read_bytes()
                        entry = {'path': relative, 'sha256': sha256(blob), 'bytes': len(blob), 'bundle': None}
                    else:
                        item = db.execute('SELECT bundle,sha256,bytes FROM members WHERE path=?', (relative,)).fetchone()
                        if item is None:
                            raise FileNotFoundError(relative)
                        entry = {'path': relative, 'bundle': item[0], 'sha256': item[1], 'bytes': item[2]}
                    artifacts[filename] = entry
                members.append({'index': index, 'member_id': mid, 'spec': record['spec'],
                    'historical_member_shots': record['shots'], 'artifacts': artifacts,
                    'historical_native_resources': record['resources']['Z_sampling_equivalent']})
            meta_path = OLD/'data/physical'/case/'problem.json'
            row.update(source_result=str(path), source_result_SHA256=sha256(path.read_bytes()),
                source_seed=cell['source_seed'], source_final_model_shots=cell['source_final_model_shots'],
                meta_path=str(meta_path), meta_SHA256=sha256(meta_path.read_bytes()), members=members)
            cases.append(row)
    finally:
        db.close()
    return cases


def binding():
    qualifier = ROOT/'qualification-frozen-projection-v2-tests.xml'
    suites = list(ET.parse(qualifier).getroot().iter('testsuite'))
    if sum(int(x.get('tests', '0')) for x in suites) != 10 or any(
            int(x.get(k, '0')) for x in suites for k in ['failures', 'errors', 'skipped']):
        raise RuntimeError('Independent projection/relabel qualification not passed')
    return {'protocol_SHA256': sha256(PROTOCOL.read_bytes()),
        'qualification_SHA256': sha256(qualifier.read_bytes()),
        'source_manifest_SHA256': sha256((ROOT/'manifest-C5-CONTROLS.json').read_bytes()),
        'implementation_SHA256': {str(p): sha256(p.read_bytes()) for p in
            [Path(__file__), ROOT/'frontierloop/frozen_projection.py', ROOT/'test_frozen_projection.py',
             OLD/'paperstudy/lucj.py', OLD/'paperstudy/sqdrift_execution.py']},
        'versions': {k: version(k) for k in ['qiskit', 'ffsim', 'numpy', 'scipy', 'pyscf']},
        'sources': resolve_inputs(), 'new_model_shots': 0, 'energy_evaluations': 0}


def read_selected(entries):
    """One streaming pass per archive, only selected bytes retained in RAM."""
    blobs = {}
    bundles = {}
    expected = {x['path']: x for x in entries}
    for e in entries:
        if e['bundle'] is None:
            blobs[e['path']] = (OLD/e['path']).read_bytes()
        else:
            bundles.setdefault(e['bundle'], set()).add(e['path'])
    for bundle, wanted in bundles.items():
        with tarfile.open(OLD/bundle, 'r|gz') as archive:
            for item in archive:
                if item.name in wanted:
                    blobs[item.name] = archive.extractfile(item).read()
                    wanted.remove(item.name)
                    if not wanted:
                        break
        if wanted:
            raise FileNotFoundError(sorted(wanted))
    for path, e in expected.items():
        if len(blobs[path]) != e['bytes'] or sha256(blobs[path]) != e['sha256']:
            raise RuntimeError('Raw archived bytes changed: '+path)
    return blobs


def primitive_resources(q):
    prep, sampling, resources = circuit_views(q)
    return sampling, resources['Z_sampling_equivalent']


def evaluate(meta, member_input, blobs, spec, target):
    import json
    cpu, wall = process_time(), perf_counter()
    entries = member_input['artifacts']
    member = json.loads(gzip.decompress(blobs[entries['member.json.gz']['path']]))
    original_qpy = blobs[entries['circuit.qpy.gz']['path']]
    if member['spec'] != member_input['spec'] or sha256(original_qpy) != member['circuit_file_sha256']:
        raise ValueError('Native member/QPY identity mismatch')
    n, hf = int(meta['n_qubits']), int(meta['hf_id'])
    seq = member['source_mode_group_Pauli_sequence']
    perm = member['permutation']
    # Match strong native fermionic layout; a qubit-only swap is not sufficient.
    relabeled, new_hf = fermionic_relabel(seq, n, hf, perm)
    cert = restrict(relabeled, n, new_hf)
    verify(cert, relabeled, n, new_hf)
    circuit = synthesize(cert['groups'], cert['active_modes'], new_hf, n, cert['global_phase'])
    full = synthesize(relabeled, list(range(n)), new_hf, n)
    original = qpy.load(BytesIO(gzip.decompress(original_qpy)))[0]
    if original.metadata.get('permutation') != perm:
        raise ValueError('Original native QPY lost fermionic layout binding')
    tick = process_time()
    actual, projected_resource = primitive_resources(circuit)
    _, rustiq_resource = primitive_resources(full)
    _, native_resource = primitive_resources(original)
    compiler_cpu = process_time()-tick
    active = len(cert['active_modes'])
    probability = {'status': 'NOT_CHECKED_ACTIVE_REGISTER_ABOVE_PREREGISTERED_CAP', 'active_qubits': active}
    if active <= spec['actual_member_full_Fock_check_max_active_qubits']:
        expected, numerical = member_state(meta, member)
        actual_probs = Statevector.from_instruction(actual).probabilities() if active else np.array([1.])
        rebuilt = {}
        for address, p in enumerate(actual_probs):
            if p == 0:
                continue
            physical = restore(address, cert)
            old = sum(((physical >> (n-1-perm[q])) & 1) << (n-1-q) for q in range(n))
            rebuilt[old] = rebuilt.get(old, 0.)+float(p)
        union = set(rebuilt) | set(expected)
        error = float(sum(abs(rebuilt.get(x, 0.)-abs(expected.get(x, 0j))**2) for x in union))
        if error > spec['actual_member_probability_L1_tolerance']:
            raise RuntimeError('Actual molecular independent Qiskit probability mismatch: '+str(error))
        probability = {'status': 'INDEPENDENT_REDUCED_QISKIT_VS_ORIGINAL_SPARSE_PASSED',
            'active_qubits': active, 'probability_L1': error, 'original_state_numerics': numerical,
            'statevector_saved': False, 'sampling_shots': 0}
    cert.update(original_source_sequence_SHA256=digest(seq), relabeled_sequence_SHA256=digest(relabeled),
        original_native_permutation=perm, original_member_id=member_input['member_id'])
    write(target/'certificate.json.gz', gzip.compress((json.dumps(cert, sort_keys=True)+'\n').encode(), mtime=0))
    stream = BytesIO(); qpy.dump(actual, stream)
    write(target/'restricted-sampling.qpy.gz', gzip.compress(stream.getvalue(), mtime=0))
    fields = ['cnot_count', 'cnot_depth', 'total_unitary_depth']
    envelope = {k: min(native_resource[k], rustiq_resource[k]) for k in fields}
    return {'status': 'RESOURCE_PREFLIGHT_COMPLETE', 'member_id': member_input['member_id'],
        'index': member_input['index'], 'spec': member_input['spec'],
        'original_qubits': n, 'active_qubits': active, 'frozen_qubits': n-active,
        'registered_sector_restriction_certificate_passed': True, 'independent_probability': probability,
        'projected_resources': projected_resource, 'native_recount_resources': native_resource,
        'same_layout_Rustiq_resources': rustiq_resource,
        'historical_native_resources': member_input['historical_native_resources'],
        'best_control_per_metric_envelope': envelope,
        'envelope_may_combine_different_control_circuits': True,
        'strict_improvement_over_both_controls': {k: projected_resource[k] < envelope[k] for k in fields},
        'source_Pauli_coefficient_slots': sum(len(g) for g in seq),
        'restricted_Pauli_coefficient_slots': sum(len(g) for g in cert['groups']),
        'optimized_variables_before_and_after': 0,
        'coefficient_slots_not_independent_parameter_dimension': True,
        'common_counter_CPU_seconds': compiler_cpu,
        'incremental_CPU_seconds': process_time()-cpu, 'incremental_wall_seconds': perf_counter()-wall,
        'new_model_shots': 0, 'new_QPU_shots': 0, 'energy_Ha': None, 'CI_evaluations': 0,
        'ideal_probability_preservation_not_arbitrary_noise_channel_equivalence': True}


def run():
    frozen = load(FREEZE)
    if frozen != binding():
        raise RuntimeError('G2 outcome-free freeze drift')
    if (RUN/'started.json').exists():
        raise RuntimeError('Preserve partial/terminal G2; no blind retry')
    spec = load(PROTOCOL)
    cpu, wall = process_time(), perf_counter()
    space(); save(RUN/'started.json', {'freeze_SHA256': sha256(FREEZE.read_bytes()), 'sources': frozen['sources']})
    rows = []
    for source in frozen['sources']:
        case = source['case']
        if source['status'] != 'AVAILABLE':
            rows.append({'case': case, 'status': source['status'], 'new_model_shots': 0})
            continue
        entries = [e for m in source['members'] for e in m['artifacts'].values()]
        blobs = read_selected(entries)
        meta = load(source['meta_path'])
        if meta['problem_id'] != source['problem_id']:
            raise ValueError('Physical source identity mismatch')
        for member in source['members']:
            space()
            if process_time()-cpu > spec['max_incremental_CPU_seconds'] or perf_counter()-wall > spec['max_incremental_wall_seconds']:
                raise TimeoutError('Pre-registered G2 global budget; preserve partial evidence')
            if sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()) > spec['max_new_directory_bytes']:
                raise OSError('Pre-registered G2 output cap')
            target = RUN/case/f"{member['index']:03d}"
            try:
                row = evaluate(meta, member, blobs, spec, target)
            except Exception as exc:
                row = {'status': 'FAILED_ENGINEERING_PRESERVED', 'member_id': member['member_id'],
                    'index': member['index'], 'error': repr(exc), 'traceback': traceback.format_exc(), 'new_model_shots': 0}
            row.update(case=case, problem_id=source['problem_id'])
            save(target/'result.json', row); rows.append(row)
            print(case, member['index'], row['status'], row.get('projected_resources'), flush=True)
    save(RUN/'terminal.json', {'status': 'PREFLIGHT_TERMINAL_NOT_WHOLE_BANK_PERFORMANCE', 'rows': rows,
        'new_model_shots': 0, 'molecular_energy_evaluations': 0, 'CI_evaluations': 0,
        'incremental_CPU_seconds': process_time()-cpu, 'incremental_wall_seconds': perf_counter()-wall,
        'freeze_SHA256': sha256(FREEZE.read_bytes()), 'goal_complete': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--freeze', action='store_true'); args = p.parse_args()
    if args.freeze:
        record = binding()
        if FREEZE.exists() and load(FREEZE) != record:
            raise RuntimeError('Never overwrite frozen G2')
        save(FREEZE, record)
        print('G2_FROZEN', len(record['sources']), 'cases; no resource/energy outcome used')
    else:
        run()
