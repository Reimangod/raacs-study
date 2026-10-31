from copy import deepcopy
import csv
import pytest
from qiskit import QuantumCircuit

from benchmark.molecular_postprocess_review import check_compiled, check_csv


def resource(qc, target='line_cx'):
    return dict(target=target, native_2q_family='CX', synthetic_not_calibrated_hardware=True,
        measurement_physical_to_logical_clbit=[(i, i) for i in range(qc.num_qubits)],
        gate_counts=dict(qc.count_ops()), native_2q_count=int(qc.count_ops().get('cx', 0)),
        native_2q_depth=qc.depth(filter_function=lambda g:g.operation.num_qubits == 2),
        native_total_gate_depth_without_measurement=qc.remove_final_measurements(inplace=False).depth())


def circuit():
    qc = QuantumCircuit(2); qc.x(0); qc.cx(0, 1); qc.measure_all()
    return qc


def test_actual_qiskit_counts_depth_and_barrier():
    qc = circuit(); check_compiled(qc, resource(qc), 2)


@pytest.mark.parametrize('field', ['native_2q_count', 'native_2q_depth', 'native_total_gate_depth_without_measurement'])
def test_wrong_resource_is_rejected(field):
    qc = circuit(); r = resource(qc); r[field] += 1
    with pytest.raises(ValueError, match='mismatch'):
        check_compiled(qc, r, 2)


def test_logical_readout_binding_is_not_just_a_bijection():
    qc = circuit(); r = resource(qc); r['measurement_physical_to_logical_clbit'] = [(0, 1), (1, 0)]
    with pytest.raises(ValueError, match='Physical/logical'):
        check_compiled(qc, r, 2)


def test_nonlocal_cx_is_rejected_even_if_counts_match():
    qc = QuantumCircuit(3); qc.cx(0, 2); qc.measure_all()
    with pytest.raises(ValueError, match='Line connectivity'):
        check_compiled(qc, resource(qc), 3)


def test_not_actually_native_gate_is_rejected():
    qc = QuantumCircuit(2); qc.h(0); qc.measure_all()
    with pytest.raises(ValueError, match='Non-native gate'):
        check_compiled(qc, resource(qc), 2)


def test_csv_reconciliation_preserves_unknowns_and_all_planned_rows(tmp_path):
    rows = [dict(case='a', method='m', value=None), dict(case='a', method='failed', value=2.0)]
    p = tmp_path / 'x.csv'
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    assert check_csv(p, rows, ('case', 'method')) == 2
    missing = deepcopy(rows); missing.append(dict(case='b', method='m', value=None))
    with pytest.raises(ValueError, match='dropped'):
        check_csv(p, missing, ('case', 'method'))
    changed = deepcopy(rows); changed[0]['value'] = 0
    with pytest.raises(ValueError, match='value mismatch'):
        check_csv(p, changed, ('case', 'method'))
