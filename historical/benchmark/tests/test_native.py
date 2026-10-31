from qiskit import QuantumCircuit
from benchmark.native import native_resources


def test_native_contract_is_explicit_not_hardware():
    qc=QuantumCircuit(3);qc.h(0);qc.cx(0,2)
    record=native_resources(qc)
    assert record["native_2q_count"]>=1
    assert record["native_2q_depth"]>=1
    assert record["hardware_execution"] is False
    assert set(record["gate_counts"])<={"rz","sx","x","ecr"}
