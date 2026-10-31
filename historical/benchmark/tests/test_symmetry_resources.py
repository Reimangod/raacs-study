import numpy as np
import pytest
from qiskit import QuantumCircuit
from benchmark.ansatz import make_pool,resources


@pytest.mark.parametrize("method",["ceo_ovp","ceo_mvp","qeb","qubit","gsd"])
def test_expected_number_and_spin_projection_conservation(method):
    pool=make_pool(method,4)
    alpha=np.array([int(i&8>0)+int(i&2>0) for i in range(16)])
    beta=np.array([int(i&4>0)+int(i&1>0) for i in range(16)])
    violations=[]
    for index in range(pool.size):
        matrix=pool.get_imp_op(index).tocoo()
        nonzero=np.abs(matrix.data)>1e-12
        row,column=matrix.row[nonzero],matrix.col[nonzero]
        violations.append(bool(np.any(alpha[row]!=alpha[column]) or np.any(beta[row]!=beta[column])))
    if method=="qubit":
        assert any(violations)  # Do not incorrectly declare all Pauli rotations number-preserving.
    else:
        assert not any(violations)


def test_logical_cnot_depth_ignores_one_qubit_layers_not_two_qubit_dependencies():
    qc=QuantumCircuit(4)
    qc.cx(0,1);qc.cx(2,3);qc.h(0);qc.cx(0,2)
    data=resources(qc,7,3)
    assert data["cnot_count"]==3
    assert data["cnot_depth"]==2
    assert data["total_depth"]==3
    assert data["parameter_count"]==7
