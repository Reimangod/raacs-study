import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from benchmark.ansatz import measurement_equivalent_circuit
from benchmark.config import Case
from benchmark.chemistry import build_reference
from benchmark.lucj import run_lucj


def test_measurement_compiler():
    qc=QuantumCircuit(3);qc.h(0);qc.cx(0,1);qc.ry(.3,2);qc.rz(.8,0);qc.cz(1,2)
    sampler,removed=measurement_equivalent_circuit(qc)
    assert removed==2
    assert np.allclose(Statevector(qc).probabilities(),Statevector(sampler).probabilities(),atol=1e-12)


def test_lucj_ffsim_pyscf_qiskit_parity(tmp_path):
    reference=build_reference(Case("H4",1.1),tmp_path)
    result=run_lucj(Case("H4",1.1),reference,"lucj_1",tmp_path)
    assert result["status"]=="FIXED_CCSD_INITIALIZED"
