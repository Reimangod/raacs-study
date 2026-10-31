"""G5 pure preparation, qualified against all frozen G4 members before use.

G4 scientific kernels/guard remain immutable. Splitting preparation from the
child call merely amortizes interpreter/import costs across up to16 members.
"""
import base64
import gzip
from io import BytesIO
import json
import os
import subprocess
import sys
from time import process_time, perf_counter
from qiskit import qpy
from .paths import ROOT
from .io import digest, sha256
from .strong_projection import OVERLAY, choose_safe, expected_addresses
from .frozen_projection import fermionic_relabel, restrict, verify, synthesize
from paperstudy.lucj import circuit_views
from paperstudy.sqdrift_execution import member_state


def semantics(blob):
    """Ordered primitive fingerprint, excluding nonphysical circuit names."""
    q = qpy.load(BytesIO(gzip.decompress(blob)))[0]
    return digest({'qubits': q.num_qubits, 'clbits': q.num_clbits,
        'global_phase': float(q.global_phase).hex(),
        'gates': [[x.operation.name, [q.find_bit(b).index for b in x.qubits],
                   [float(p).hex() for p in x.operation.params]] for x in q.data]})


def prepare(meta, plan, blobs):
    cpu, wall = process_time(), perf_counter()
    raw = blobs[plan['artifacts']['member.json.gz']['path']]
    source_qpy = blobs[plan['artifacts']['circuit.qpy.gz']['path']]
    member = json.loads(gzip.decompress(raw))
    if member['spec'] != plan['spec'] or sha256(source_qpy) != member['circuit_file_sha256']:
        raise ValueError('Native source binding mismatch')
    if digest(member['spec']) != plan['member_id']:
        raise ValueError('Native member ID mismatch')
    n, hf = int(meta['n_qubits']), int(meta['hf_id'])
    sequence, perm = member['source_mode_group_Pauli_sequence'], member['permutation']
    relabeled, physical_hf = fermionic_relabel(sequence, n, hf, perm)
    certificate = restrict(relabeled, n, physical_hf)
    verify(certificate, relabeled, n, physical_hf)
    original = qpy.load(BytesIO(gzip.decompress(source_qpy)))[0]
    if original.metadata.get('permutation') != perm: raise ValueError('Native physical layout missing')
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
    policy = choose_safe(resources)
    if not policy['no_increase_vs_native']: raise RuntimeError('Resource guard failed')
    certificate.update(original_member_id=plan['member_id'], original_source_sequence_SHA256=digest(sequence),
        relabeled_sequence_SHA256=digest(relabeled), original_native_permutation=perm)
    cert_blob = gzip.compress(json.dumps(certificate, sort_keys=True).encode(), mtime=0)
    row = dict(status='PREPARED_NOT_YET_INDEPENDENTLY_CHECKED', case=meta['case_id'], problem_id=meta['problem_id'],
        index=plan['index'], member_id=plan['member_id'], spec=plan['spec'], source_artifacts=plan['artifacts'],
        original_member_shot_weight=plan['historical_member_shots'], original_qubits=n,
        active_qubits=len(certificate['active_modes']), source_sparse_support=len(expected),
        original_source_numerics=numerics, resources=resources, policy=policy,
        coefficient_slots_original=sum(len(g) for g in sequence),
        coefficient_slots_restricted=sum(len(g) for g in certificate['groups']),
        optimized_variables_before_and_after=0, independent_parameter_reduction_claimed=False,
        certificate_SHA256=sha256(cert_blob), qpy_gz_SHA256={k: sha256(v) for k,v in qpy_blobs.items()},
        preparation_CPU_seconds=process_time()-cpu, preparation_wall_seconds=perf_counter()-wall,
        new_model_shots=0, new_QPU_shots=0, new_energy_evaluations=0, new_CI_evaluations=0,
        ideal_probability_not_arbitrary_noise_channel_equivalence=True)
    key = meta['case_id']+':'+str(plan['index'])+':'+plan['member_id']
    return dict(row=row, certificate=cert_blob, qpy_blobs=qpy_blobs,
        request=dict(key=key, circuits=request_circuits))


def verify_batch(items, wall_limit=45):
    if not items or len(items) > 16: raise ValueError('Batch size')
    request = {'jobs': [x['request'] for x in items], 'tolerance': 1e-8}
    from .mps_batch_worker import validate
    validate(request)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(OVERLAY), str(ROOT)]))
    r = subprocess.run([sys.executable, '-m', 'frontierloop.mps_batch_worker'],
        input=json.dumps(request), text=True, capture_output=True, timeout=wall_limit, env=env)
    if r.returncode: raise RuntimeError('Batch process failed: '+r.stderr[-6000:])
    result = json.loads(r.stdout)
    if set(result['results']) != {x['request']['key'] for x in items}: raise RuntimeError('Batch result alias/missingness')
    for entry in result['results'].values():
        if entry['status'] == 'PASSED' and entry['check']['versions'] != {'qiskit': '2.6.0', 'qiskit_aer': '0.18.0'}:
            raise RuntimeError('Pinned checker version drift')
    return result
