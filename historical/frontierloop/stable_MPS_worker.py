"""Additive numerical verifier: standard Aer SVD threshold, unchanged error test.

Not untruncated, not a unitary/noise-channel certificate. No source is sampled.
The probability comparison includes every expected address and outside mass.
"""
import base64
import gzip
from io import BytesIO
import json
import sys
from time import process_time, perf_counter
from .mps_distribution_check import compare


def check(request):
    from qiskit import qpy
    import qiskit
    import qiskit_aer
    from qiskit_aer import AerSimulator
    if request['tolerance'] != 1e-8:
        raise ValueError('Original probability tolerance required')
    cpu, wall = process_time(), perf_counter()
    options = dict(method='matrix_product_state', precision='double',
        matrix_product_state_truncation_threshold=1e-16,
        matrix_product_state_max_bond_dimension=None, zero_threshold=0.,
        enable_truncation=False, max_parallel_threads=1, max_parallel_experiments=1,
        max_parallel_shots=1, max_memory_mb=2048)
    simulator = AerSimulator(**options)
    rows = {}
    for name, item in request['circuits'].items():
        qc = qpy.load(BytesIO(gzip.decompress(base64.b64decode(item['qpy_gz_base64']))))[0]
        if qc.num_qubits > 40 or qc.num_clbits or len(qc.data) > 100000:
            raise ValueError('Verification scope exceeded')
        if qc.num_qubits:
            qc.save_matrix_product_state()
            result = simulator.run(qc, shots=0, memory=False).result()
            if not result.success:
                raise RuntimeError('Aer execution failed')
            metadata = result.results[0].metadata
            if metadata['matrix_product_state_truncation_threshold'] != 1e-16:
                raise RuntimeError('Numerical threshold drift')
            if metadata['matrix_product_state_max_bond_dimension'] < 2**40:
                raise RuntimeError('Finite bond cap')
            if metadata['num_qubits'] != qc.num_qubits:
                raise RuntimeError('Address mapping changed')
            mps = result.data()['matrix_product_state']
        else:
            mps = ([], [])
        rows[name] = dict(compare(mps, item['expected_probabilities'], 1e-8),
            qubits=qc.num_qubits,
            status='NUMERICALLY_TRUNCATED_MPS_ALL_SUPPORT_PROBABILITY_PASSED')
    return dict(circuits=rows, options=options,
        versions=dict(qiskit=qiskit.__version__, qiskit_aer=qiskit_aer.__version__),
        CPU_seconds=process_time()-cpu, wall_seconds=perf_counter()-wall,
        new_model_shots=0, new_QPU_shots=0, state_saved=False,
        untruncated=False, unitary_or_noise_channel_equivalence=False)


if __name__ == '__main__':
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    print(json.dumps(check(json.load(sys.stdin)), allow_nan=False))
