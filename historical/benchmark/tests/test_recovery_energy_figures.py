"""Pure Fig14 projection tests; no external services or numerical experiments."""
from copy import deepcopy
import pytest
from benchmark.recovery_energy_figures import project


def fixture():
    cfg = dict(cases=['X_R1.0'], methods=['present','missing'], reps=[0], readout_probabilities=[0.,.01], M=1000, R=64)
    references = {'X_R1.0':dict(case=dict(molecule='X',distance=1.),fci_energy=-2.)}
    records = []
    for method in cfg['methods']:
        for noise in cfg['readout_probabilities']:
            task = dict(case='X_R1.0',method=method,rep=0,readout_flip_probability=noise)
            status = 'COMPLETED' if method == 'present' else 'SOURCE_INITIALIZATION_MISSING'
            result = dict(task=task,status=status)
            if method == 'present':
                # Preserve a worsening result and a distinct noisy baseline.
                qenergy = -1.99 if noise == 0 else -1.98
                result.update(same_raw_counts=True,new_quantum_sample_acquisitions=0,hardware_shots=0,
                              QSCI=dict(energy_Ha=qenergy,actual_dimension=4),
                              SQD=dict(energy_Ha=-1.985,dimension=6),
                              same_actual_d=dict(d=4,QSCI_energy_Ha=qenergy,SQD_energy_Ha=-1.975))
            records.append((method+str(noise),task,status,result))
    return cfg,records,references


def test_all_planned_missingness_not_zero_and_no_outcome_selection():
    cfg,records,refs = fixture(); rows = project(cfg,records,refs)
    assert len(rows) == 4
    assert all(r['QSCI_energy_Ha'] is None and r['matched_d'] is None for r in rows if r['method']=='missing')
    present = [r for r in rows if r['method']=='present']
    assert present[0]['SQD_minus_QSCI_mHa'] > 0 and present[1]['SQD_minus_QSCI_mHa'] < 0
    assert present[0]['QSCI_energy_Ha'] != present[1]['QSCI_energy_Ha']
    assert present[0]['SQD_d'] == 6 and present[0]['matched_d'] == 4
    assert all(r['new_CI_solves'] == 0 and r['work_matched'] is False for r in rows)


def test_order_independent():
    cfg,records,refs = fixture()
    assert project(cfg,records,refs) == project(cfg,list(reversed(records)),refs)


@pytest.mark.parametrize('defect',['duplicate','omitted','task','running','nan','dimension','matched','pairing','variational','restriction'])
def test_invalid_or_incomplete_input_rejected(defect):
    cfg,records,refs = deepcopy(fixture())
    if defect=='duplicate': records.append(records[0])
    elif defect=='omitted': records.pop()
    elif defect=='task': records[0][3]['task'] = dict(records[0][1],rep=99)
    elif defect=='running': records[0] = (records[0][0],records[0][1],'RUNNING',dict(records[0][3],status='RUNNING'))
    elif defect=='nan': records[0][3]['QSCI']['energy_Ha'] = float('nan')
    elif defect=='dimension': records[0][3]['SQD']['dimension'] = 65
    elif defect=='matched': records[0][3]['same_actual_d']['d'] = 3
    elif defect=='pairing': records[0][3]['same_raw_counts'] = False
    elif defect=='variational': records[0][3]['QSCI']['energy_Ha'] = -3.
    elif defect=='restriction': records[0][3]['same_actual_d']['SQD_energy_Ha'] = -1.999
    with pytest.raises(ValueError): project(cfg,records,refs)


def test_tiny_negative_roundoff_is_retained_not_clipped():
    cfg,records,refs = fixture()
    records[0][3]['QSCI']['energy_Ha'] = -2.-1e-13
    records[0][3]['same_actual_d']['QSCI_energy_Ha'] = -2.-1e-13
    row = next(r for r in project(cfg,records,refs) if r['id']==records[0][0])
    assert row['QSCI_error_mHa'] < 0
