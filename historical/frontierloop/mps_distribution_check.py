"""Independent ideal sampling check; Aer lives in an existing isolated overlay.

No statevector, sample, CI, or energy is generated. The MPS is kept only in RAM.
Querying ALL original sparse support plus contracting the MPS norm checks
leakage too. A probability check is not a full unitary/channel certificate.
"""
import base64
import gzip
from io import BytesIO
import json
import sys
from time import process_time, perf_counter
import numpy as np


def amplitude(mps, address):
    gamma, lam = mps
    if address < 0 or address >= 1 << len(gamma):
        raise ValueError('Address outside MPS register')
    v = np.array([1.+0j])
    for i, pair in enumerate(gamma):
        v = v @ np.asarray(pair[(address >> i) & 1])
        if i < len(lam):
            v = v * np.asarray(lam[i])
    if v.shape != (1,):
        raise ValueError('MPS terminal bond mismatch')
    return complex(v[0])


def norm_squared(mps):
    gamma, lam = mps
    e = np.ones((1, 1), dtype=complex)
    for i, pair in enumerate(gamma):
        factors = [np.asarray(a) for a in pair]
        if i < len(lam):
            factors = [a*np.asarray(lam[i])[None, :] for a in factors]
        e = sum(a.conj().T @ e @ a for a in factors)
    z = complex(e[0, 0])
    if e.shape != (1, 1) or abs(z.imag) > 1e-9:
        raise ValueError('Invalid contracted MPS norm')
    return float(z.real)


def compare(mps, expected, tolerance=1e-8):
    if not expected or len(expected) > 100000:
        raise ValueError('Sparse verification support cap')
    expected = {int(k): float(p) for k, p in expected.items()}
    if any(not np.isfinite(p) or p < 0 for p in expected.values()):
        raise ValueError('Invalid source probabilities')
    if abs(sum(expected.values())-1) > tolerance:
        raise ValueError('Original sparse distribution is not normalized')
    norm = norm_squared(mps)
    probs = {k: abs(amplitude(mps, k))**2 for k in expected}
    mass = sum(probs.values())
    # Absolute difference also detects a numerically inconsistent mass > norm.
    leakage = abs(norm-mass)
    l1 = float(sum(abs(probs[k]-p) for k, p in expected.items())+leakage)
    if not np.isfinite(l1) or l1 > tolerance or abs(norm-1) > tolerance:
        raise ValueError(f'MPS/source distribution mismatch: L1={l1}, norm={norm}')
    return {'probability_L1_with_leakage_bound': l1, 'MPS_norm_squared': norm,
        'queried_support_probability': float(mass), 'outside_support_mass_bound': leakage,
        'expected_support_size': len(expected),
        'max_final_bond_dimension': max((len(x) for x in mps[1]), default=1)}


def check_request(request):
    from qiskit import qpy
    import qiskit
    import qiskit_aer
    from qiskit_aer import AerSimulator
    cpu, wall = process_time(), perf_counter()
    options = dict(method='matrix_product_state', precision='double',
        matrix_product_state_truncation_threshold=0.,
        matrix_product_state_max_bond_dimension=None, zero_threshold=0.,
        enable_truncation=False, max_parallel_threads=1, max_parallel_experiments=1,
        max_parallel_shots=1, max_memory_mb=2048)
    simulator = AerSimulator(**options)
    rows = {}
    for name, item in request['circuits'].items():
        qc = qpy.load(BytesIO(gzip.decompress(base64.b64decode(item['qpy_gz_base64']))))[0]
        if qc.num_qubits > 40 or qc.num_clbits or len(qc.data) > 100000:
            raise ValueError('Verification circuit scope exceeded')
        if qc.num_qubits == 0:
            mps, metadata = ([], []), {'method': 'trivial_zero_qubit'}
        else:
            qc.save_matrix_product_state()
            result = simulator.run(qc, shots=0, memory=False).result()
            if not result.success:
                raise RuntimeError('Independent Aer execution failed')
            metadata = result.results[0].metadata
            if metadata['matrix_product_state_truncation_threshold'] != 0:
                raise RuntimeError('MPS truncation was enabled')
            if metadata['matrix_product_state_max_bond_dimension'] < 2**40:
                raise RuntimeError('Finite MPS bond cap was enabled')
            if metadata['num_qubits'] != qc.num_qubits:
                raise RuntimeError('Aer changed physical wire addressing')
            mps = result.data()['matrix_product_state']
        rows[name] = {**compare(mps, item['expected_probabilities'], request['tolerance']),
            'qubits': qc.num_qubits, 'Aer_method': metadata['method'],
            'status': 'INDEPENDENT_UNTRUNCATED_MPS_PROBABILITY_PASSED'}
    return {'circuits': rows, 'versions': {'qiskit': qiskit.__version__, 'qiskit_aer': qiskit_aer.__version__},
        'options': options, 'CPU_seconds': process_time()-cpu, 'wall_seconds': perf_counter()-wall,
        'new_model_shots': 0, 'new_QPU_shots': 0, 'state_saved': False}


if __name__ == '__main__':
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    print(json.dumps(check_request(json.load(sys.stdin)), allow_nan=False))
