"""Pure compile/evaluate successor; artifacts returned for compact SQLite storage."""
import gzip
import json
from io import BytesIO
from time import process_time, perf_counter
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector
from .paths import ROOT
from .io import digest, sha256
from .frozen_projection import restrict, verify, restore, synthesize, fermionic_relabel
from paperstudy.lucj import circuit_views
from paperstudy.sqdrift_execution import member_state

FIELDS = ['cnot_count', 'cnot_depth', 'total_unitary_depth']


def choose(native, rustiq, projected):
    controls = [('NATIVE', native), ('RUSTIQ', rustiq)]
    name, base = min(controls, key=lambda x: tuple(x[1][k] for k in FIELDS)+(x[0],))
    accept = all(projected[k] <= base[k] for k in FIELDS) and any(projected[k] < base[k] for k in FIELDS)
    return {'baseline': name, 'selected': 'PROJECTED' if accept else name,
        'projected_accepted': accept, 'baseline_resources': base,
        'selected_resources': projected if accept else base,
        'uses_energy_or_counts': False, 'objective': 'CNOT then CNOT depth then total depth; no metric increase'}


def compile_member(meta, plan, blobs, max_active=16):
    wall, cpu = perf_counter(), process_time()
    member_blob = blobs[plan['artifacts']['member.json.gz']['path']]
    qpy_blob = blobs[plan['artifacts']['circuit.qpy.gz']['path']]
    member = json.loads(gzip.decompress(member_blob))
    if member['spec'] != plan['spec'] or sha256(qpy_blob) != member['circuit_file_sha256']:
        raise ValueError('Native source binding mismatch')
    n, hf = int(meta['n_qubits']), int(meta['hf_id'])
    sequence = member['source_mode_group_Pauli_sequence']
    permutation = member['permutation']
    relabeled, new_hf = fermionic_relabel(sequence, n, hf, permutation)
    cert = restrict(relabeled, n, new_hf)
    verify(cert, relabeled, n, new_hf)
    projected = synthesize(cert['groups'], cert['active_modes'], new_hf, n, cert['global_phase'])
    full = synthesize(relabeled, list(range(n)), new_hf, n)
    original = qpy.load(BytesIO(gzip.decompress(qpy_blob)))[0]
    if original.metadata.get('permutation') != permutation:
        raise ValueError('Saved native QPY metadata mismatch')
    circuits = {}
    resources = {}
    for name, qc in [('NATIVE', original), ('RUSTIQ', full), ('PROJECTED', projected)]:
        _, sampling, views = circuit_views(qc)
        circuits[name] = sampling
        resources[name] = views['Z_sampling_equivalent']
    active = len(cert['active_modes'])
    probability = {'status': 'SYMBOLIC_ONLY_ABOVE_FIXED_ACTIVE_CAP', 'active_qubits': active}
    if active <= max_active:
        expected, numerics = member_state(meta, member)
        probs = Statevector.from_instruction(circuits['PROJECTED']).probabilities() if active else np.array([1.])
        restored = {}
        for address, p in enumerate(probs):
            if p == 0:
                continue
            physical = restore(address, cert)
            old = sum(((physical >> (n-1-permutation[q])) & 1) << (n-1-q) for q in range(n))
            restored[old] = restored.get(old, 0.)+float(p)
        error = float(sum(abs(restored.get(x, 0.)-abs(expected.get(x, 0j))**2)
                          for x in set(restored) | set(expected)))
        if error > 1e-8:
            raise RuntimeError('Independent actual-member distribution parity failed: '+str(error))
        probability = {'status': 'INDEPENDENT_REDUCED_QISKIT_VS_ORIGINAL_SPARSE_PASSED',
            'active_qubits': active, 'probability_L1': error, 'original_state_numerics': numerics}
    policy = choose(resources['NATIVE'], resources['RUSTIQ'], resources['PROJECTED'])
    cert.update(original_source_sequence_SHA256=digest(sequence), relabeled_sequence_SHA256=digest(relabeled),
        original_native_permutation=permutation, original_member_id=plan['member_id'])
    cert_blob = gzip.compress((json.dumps(cert, sort_keys=True)+'\n').encode(), mtime=0)
    stream = BytesIO(); qpy.dump(circuits[policy['selected']], stream)
    compiled_blob = gzip.compress(stream.getvalue(), mtime=0)
    record = {'status': 'CERTIFIED_RESOURCE_POINT', 'case': meta['case_id'], 'problem_id': meta['problem_id'],
        'index': plan['index'], 'member_id': plan['member_id'], 'spec': plan['spec'],
        'source_artifacts': plan['artifacts'], 'original_member_shot_weight': plan['historical_member_shots'],
        'original_qubits': n, 'projected_active_qubits': active,
        'selected_qubits': active if policy['selected'] == 'PROJECTED' else n,
        'resources': resources, 'policy': policy, 'independent_probability': probability,
        'symbolic_restriction_passed': True,
        'coefficient_slots_original': sum(len(g) for g in sequence),
        'coefficient_slots_restricted': sum(len(g) for g in cert['groups']),
        'optimized_variables_before_and_after': 0, 'independent_parameter_reduction_claimed': False,
        'certificate_SHA256': sha256(cert_blob), 'selected_QPY_gz_SHA256': sha256(compiled_blob),
        'incremental_CPU_seconds': process_time()-cpu, 'incremental_wall_seconds': perf_counter()-wall,
        'new_model_shots': 0, 'new_energy_evaluations': 0, 'new_CI_evaluations': 0,
        'ideal_source_distribution_not_arbitrary_noise_channel_equivalence': True}
    return record, cert_blob, compiled_blob
