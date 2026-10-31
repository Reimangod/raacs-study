"""G4: stronger N20 fixed-prefix, no outcomes, no original kernel changes."""
import argparse
import base64
from collections import Counter
import gzip
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import resource
import sqlite3
import subprocess
import sys
from time import perf_counter, process_time
import traceback
import xml.etree.ElementTree as ET
from qiskit import qpy
from .paths import ROOT, OLD, PROJECT
from .io import load, save, digest, sha256, space
from .frozen_projection import fermionic_relabel, restrict, verify, restore, synthesize
from .frozen_projection_screen import read_selected
from paperstudy.lucj import circuit_views
from paperstudy.sqdrift_execution import member_state

PROTOCOL = ROOT/'protocol-G4-STRONG-PROJECTION.json'
FREEZE = ROOT/'freeze-G4-STRONG-PROJECTION.json'
RUN = ROOT/'runs/G4-STRONG-PROJECTION-v1'
OVERLAY = PROJECT/'.runtime-overlays/aer-noise-v1/lib/python3.11/site-packages'
FIELDS = ['cnot_count', 'cnot_depth', 'total_unitary_depth']


def dominates(a, b):
    return all(a[k] <= b[k] for k in FIELDS) and any(a[k] < b[k] for k in FIELDS)


def choose_safe(resources):
    # Unlike frozen G3's CNOT-first control, guarantee no original-native depth
    # increase even when the two compiler controls trade off count and depth.
    base = 'RUSTIQ' if dominates(resources['RUSTIQ'], resources['NATIVE']) else 'NATIVE'
    selected = 'PROJECTED' if dominates(resources['PROJECTED'], resources[base]) else base
    return {'baseline': base, 'selected': selected, 'baseline_resources': resources[base],
        'selected_resources': resources[selected], 'uses_energy_or_counts': False,
        'no_increase_vs_native': all(resources[selected][k] <= resources['NATIVE'][k] for k in FIELDS),
        'projected_accepted': selected == 'PROJECTED',
        'rustiq_count_depth_tradeoff_excluded': resources['RUSTIQ']['cnot_count'] < resources['NATIVE']['cnot_count']
            and not all(resources['RUSTIQ'][k] <= resources['NATIVE'][k] for k in FIELDS)}


def physical_address(old, n, permutation):
    return sum(((old >> (n-1-q)) & 1) << (n-1-permutation[q]) for q in range(n))


def expected_addresses(expected, n, permutation, certificate, reduced):
    answer = {}
    for old, amplitude in expected.items():
        physical = physical_address(old, n, permutation)
        if reduced:
            address = sum(((physical >> (n-1-q)) & 1) << j
                for j, q in enumerate(certificate['active_modes']))
            if restore(address, certificate) != physical:
                raise ValueError('Original source violates registered fixed occupations')
        else:
            address = sum(((physical >> (n-1-q)) & 1) << q for q in range(n))
        if address in answer:
            raise ValueError('Decoder failed to be injective on original support')
        answer[address] = float(abs(amplitude)**2)
    return answer


def source_plans(spec):
    manifest = load(ROOT/'manifest-C5-CONTROLS.json')
    plans = []
    db = sqlite3.connect('file:'+str(OLD/'raw-archives-v1/path-index.sqlite')+'?mode=ro', uri=True)
    try:
        for case in spec['cases']:
            source = next(x for x in manifest['cells'] if x['case'] == case and x['method'] == 'D_SQDRIFT_N20')
            if source['status'] != 'AVAILABLE':
                plans.append({'case': case, 'status': source['status']}); continue
            path = Path(source['source_result']); result = load(path)
            if sha256(path.read_bytes()) != source['source_result_SHA256'] or result['method_id'] != 'D_SQDRIFT_N20':
                raise ValueError('Frozen stronger N20 source mismatch')
            if result['problem_id'] != source['expected_problem_id']:
                raise ValueError('Same-problem source mismatch')
            records = result['member_records']
            if len(records) != 783 or sum(x['shots'] for x in records) != 100000 or any(
                    x['spec']['num_terms'] != 20 for x in records):
                raise ValueError('Original N20 bank changed')
            members = []
            for i, row in enumerate(records[:spec['members_per_case']]):
                mid = digest(row['spec'])
                if mid != row['member_id']:
                    raise ValueError('Original spec/member ID mismatch')
                entries = {}
                for name in ['member.json.gz', 'circuit.qpy.gz']:
                    relative = 'native-SqDRIFT/'+case+'/members/'+mid+'/'+name
                    p = OLD/relative
                    if p.exists():
                        blob = p.read_bytes()
                        entry = dict(path=relative, bundle=None, sha256=sha256(blob), bytes=len(blob))
                    else:
                        e = db.execute('SELECT bundle,sha256,bytes FROM members WHERE path=?', (relative,)).fetchone()
                        if e is None: raise FileNotFoundError(relative)
                        entry = dict(path=relative, bundle=e[0], sha256=e[1], bytes=e[2])
                    entries[name] = entry
                members.append(dict(index=i, member_id=mid, spec=row['spec'], artifacts=entries,
                    historical_member_shots=row['shots']))
            meta = OLD/'data/physical'/case/'problem.json'
            plans.append(dict(case=case, status='AVAILABLE', source_result=str(path),
                source_result_SHA256=sha256(path.read_bytes()), problem_id=result['problem_id'],
                original_bank_members=len(records), meta_path=str(meta), meta_SHA256=sha256(meta.read_bytes()),
                members=members))
    finally:
        db.close()
    return plans


def binding():
    qualifiers = {}
    for name, count in [('qualification-mps-distribution-v2-tests.xml', 5),
                        ('qualification-strong-projection-v2-tests.xml', 5)]:
        p = ROOT/name; suites = list(ET.parse(p).getroot().iter('testsuite'))
        if sum(int(x.get('tests', 0)) for x in suites) != count or any(
                int(x.get(k, 0)) for x in suites for k in ['failures', 'errors', 'skipped']):
            raise RuntimeError('G4 independent qualifier not passed: '+name)
        qualifiers[name] = sha256(p.read_bytes())
    g3 = load(ROOT/'freeze-G3-FULLBANK-PROJECTION.json')
    for p in [ROOT/'frontierloop/frozen_projection.py', OLD/'paperstudy/lucj.py', OLD/'paperstudy/sqdrift_execution.py']:
        frozen = load(ROOT/'freeze-G2-FROZEN-OCCUPATION.json')['implementation_SHA256'][str(p)]
        if sha256(p.read_bytes()) != frozen: raise RuntimeError('Original qualified kernel changed')
    return {'protocol_SHA256': sha256(PROTOCOL.read_bytes()), 'qualifiers': qualifiers,
        'source_manifest_SHA256': sha256((ROOT/'manifest-C5-CONTROLS.json').read_bytes()),
        'implementation_SHA256': {str(p): sha256(p.read_bytes()) for p in [Path(__file__),
            ROOT/'frontierloop/mps_distribution_check.py', ROOT/'test_strong_projection.py',
            ROOT/'test_mps_distribution_check.py', ROOT/'frontierloop/frozen_projection_screen.py']},
        'Aer_overlay': {str(p.relative_to(OVERLAY)): sha256(p.read_bytes()) for p in sorted(OVERLAY.rglob('*'))
            if p.is_file() and p.suffix in ['.py', '.so', '.dylib']},
        'dependencies': {k: version(k) for k in g3['dependencies']},
        'sources': source_plans(load(PROTOCOL)), 'new_model_shots': 0}


def compile_and_verify(meta, plan, blobs, spec):
    cpu, wall = process_time(), perf_counter()
    raw = blobs[plan['artifacts']['member.json.gz']['path']]
    source_qpy = blobs[plan['artifacts']['circuit.qpy.gz']['path']]
    member = json.loads(gzip.decompress(raw))
    if member['spec'] != plan['spec'] or sha256(source_qpy) != member['circuit_file_sha256']:
        raise ValueError('Native source binding mismatch')
    n, hf = int(meta['n_qubits']), int(meta['hf_id'])
    sequence, perm = member['source_mode_group_Pauli_sequence'], member['permutation']
    relabeled, physical_hf = fermionic_relabel(sequence, n, hf, perm)
    certificate = restrict(relabeled, n, physical_hf)
    verify(certificate, relabeled, n, physical_hf)
    original = qpy.load(BytesIO(gzip.decompress(source_qpy)))[0]
    if original.metadata.get('permutation') != perm:
        raise ValueError('Native physical layout missing')
    candidates = {'NATIVE': original, 'RUSTIQ': synthesize(relabeled, list(range(n)), physical_hf, n),
        'PROJECTED': synthesize(certificate['groups'], certificate['active_modes'], physical_hf, n,
            certificate['global_phase'])}
    resources, qpy_blobs, request_circuits = {}, {}, {}
    expected, numerics = member_state(meta, member)
    for name, qc in candidates.items():
        _, sample, views = circuit_views(qc)
        resources[name] = views['Z_sampling_equivalent']
        stream = BytesIO(); qpy.dump(sample, stream)
        qpy_blobs[name] = gzip.compress(stream.getvalue(), mtime=0)
        request_circuits[name] = {'qpy_gz_base64': base64.b64encode(qpy_blobs[name]).decode(),
            'expected_probabilities': expected_addresses(expected, n, perm, certificate, name == 'PROJECTED')}
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(OVERLAY), str(ROOT)]))
    child = subprocess.run([sys.executable, '-m', 'frontierloop.mps_distribution_check'],
        input=json.dumps({'circuits': request_circuits, 'tolerance': spec['probability_L1_tolerance']}),
        text=True, capture_output=True, timeout=spec['max_verifier_wall_seconds_per_member'], env=env)
    if child.returncode:
        raise RuntimeError('Independent MPS verifier failed: '+child.stderr[-6000:])
    check = json.loads(child.stdout)
    if check['versions'] != {'qiskit': '2.6.0', 'qiskit_aer': '0.18.0'}:
        raise RuntimeError('Independent verifier version mismatch')
    policy = choose_safe(resources)
    if not policy['no_increase_vs_native']: raise RuntimeError('Resource safety guard failed')
    certificate.update(original_member_id=plan['member_id'], original_source_sequence_SHA256=digest(sequence),
        relabeled_sequence_SHA256=digest(relabeled), original_native_permutation=perm)
    cert_blob = gzip.compress(json.dumps(certificate, sort_keys=True).encode(), mtime=0)
    record = dict(status='CERTIFIED_RESOURCE_POINT', case=meta['case_id'], problem_id=meta['problem_id'],
        index=plan['index'], member_id=plan['member_id'], spec=plan['spec'], source_artifacts=plan['artifacts'],
        original_member_shot_weight=plan['historical_member_shots'], original_qubits=n,
        active_qubits=len(certificate['active_modes']), source_sparse_support=len(expected),
        original_source_numerics=numerics, resources=resources, policy=policy, independent_check=check,
        coefficient_slots_original=sum(len(g) for g in sequence),
        coefficient_slots_restricted=sum(len(g) for g in certificate['groups']),
        optimized_variables_before_and_after=0, independent_parameter_reduction_claimed=False,
        certificate_SHA256=sha256(cert_blob), qpy_gz_SHA256={k: sha256(v) for k,v in qpy_blobs.items()},
        parent_CPU_seconds=process_time()-cpu, wall_seconds=perf_counter()-wall,
        new_model_shots=0, new_QPU_shots=0, new_energy_evaluations=0, new_CI_evaluations=0,
        ideal_probability_not_arbitrary_noise_channel_equivalence=True)
    return record, cert_blob, qpy_blobs


def total_cpu():
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return process_time()+c.ru_utime+c.ru_stime


def run():
    frozen = load(FREEZE)
    if binding() != frozen: raise RuntimeError('G4 freeze drift')
    if (RUN/'started.json').exists(): raise RuntimeError('Do not restart partial/terminal G4')
    spec = load(PROTOCOL); space(); cpu, wall = total_cpu(), perf_counter()
    save(RUN/'started.json', {'freeze_SHA256': sha256(FREEZE.read_bytes()), 'new_model_shots': 0})
    db = sqlite3.connect(RUN/'compiled-members.sqlite')
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE members(case_id TEXT,idx INTEGER,member_id TEXT,record TEXT,certificate_gz BLOB,native_gz BLOB,rustiq_gz BLOB,projected_gz BLOB,PRIMARY KEY(case_id,idx))')
    db.commit(); statuses = Counter(); cases = []
    try:
        for source in frozen['sources']:
            if source['status'] != 'AVAILABLE':
                cases.append(dict(case=source['case'], status=source['status'])); continue
            blobs = read_selected([e for m in source['members'] for e in m['artifacts'].values()])
            meta = load(source['meta_path']); meta['case_id'] = source['case']
            case_status = Counter()
            for plan in source['members']:
                space()
                if total_cpu()-cpu > spec['max_CPU_seconds'] or perf_counter()-wall > spec['max_wall_seconds']:
                    raise TimeoutError('G4 frozen total cost cap')
                if sum(p.stat().st_size for p in RUN.iterdir() if p.is_file()) > spec['max_new_directory_bytes']:
                    raise OSError('G4 compact storage cap')
                try:
                    row, cert, compiled = compile_and_verify(meta, plan, blobs, spec)
                except Exception as exc:
                    row = dict(status='FAILED_ENGINEERING_PRESERVED', case=source['case'], index=plan['index'],
                        member_id=plan['member_id'], error=repr(exc), traceback=traceback.format_exc(),
                        new_model_shots=0, new_CI_evaluations=0)
                    cert, compiled = None, {k: None for k in ['NATIVE','RUSTIQ','PROJECTED']}
                db.execute('INSERT INTO members VALUES(?,?,?,?,?,?,?,?)', (source['case'], plan['index'],
                    plan['member_id'], json.dumps(row, sort_keys=True, allow_nan=False), cert,
                    compiled['NATIVE'], compiled['RUSTIQ'], compiled['PROJECTED'])); db.commit()
                statuses[row['status']] += 1; case_status[row['status']] += 1
                print(source['case'], plan['index'], row['status'], row.get('active_qubits'), flush=True)
            cases.append(dict(case=source['case'], statuses=dict(case_status)))
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]: raise RuntimeError('G4 database integrity')
    finally:
        db.close()
    save(RUN/'terminal.json', dict(status='FIXED_PREFIX_RESOURCE_TERMINAL_NOT_WHOLE_BANK', cases=cases,
        statuses=dict(statuses), CPU_seconds_including_children=total_cpu()-cpu,
        wall_seconds=perf_counter()-wall, database_SHA256=sha256((RUN/'compiled-members.sqlite').read_bytes()),
        freeze_SHA256=sha256(FREEZE.read_bytes()), new_model_shots=0, new_QPU_shots=0,
        new_energy_evaluations=0, new_CI_evaluations=0, energy_result_join_authorized=False, goal_complete=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--freeze', action='store_true'); args = p.parse_args()
    if args.freeze:
        b = binding()
        if FREEZE.exists() and load(FREEZE) != b: raise RuntimeError('Never replace G4 freeze')
        save(FREEZE, b); print('G4_FIXED_PREFIX_FROZEN', sum(len(s.get('members', [])) for s in b['sources']))
    else:
        run()
