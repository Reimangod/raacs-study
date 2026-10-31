import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from benchmark.mps_layout_qualification import spin_blocked_mapping,permute_sampling,logical_id
from benchmark.mps_sampling import construct_mps
from benchmark.molecular_mps_scaling import no_full_state_guards


def test_literal_permutation_preserves_exact_distribution_and_gate_count():
    qc=QuantumCircuit(8)
    for i in range(8):qc.ry(.23*(i+1),i)
    for a,b in [(0,7),(4,2),(1,5)]:qc.cx(a,b)
    blocked,mapping=permute_sampling(qc)
    assert mapping==spin_blocked_mapping(8)
    original=Statevector.from_instruction(qc).probabilities()
    new=Statevector.from_instruction(blocked).probabilities()
    restored=np.zeros_like(original)
    for b,p in enumerate(new):restored[logical_id(b,mapping)]=p
    assert np.max(abs(restored-original))<1e-12
    assert dict(blocked.count_ops())==dict(qc.count_ops())
    with no_full_state_guards():
        m,meta=construct_mps(blocked,max_bond=16)
    assert abs(meta['retained_norm_squared']-1)<1e-10
