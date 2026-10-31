"""Synthetic read-only review tests, not numerical-production qualification."""
from copy import deepcopy
import hashlib
import json
import pytest
from benchmark.hci_paired_review import check_body, expected_subset, identity


def fixture():
    old = dict(case='toy',method='LUCJ',R_cap=8,epsilon1_Ha=.01,sampler_status='COMPLETED',
        HCI_status='DIMENSION_CAP',comparison_status='AVAILABLE_DESCRIPTIVE_PAIR',
        sampler_energy_Ha=-3.,HCI_last_solved_energy_Ha=-4.)
    source = dict(problem_id='physical',hf_id=7,counts={1:2,2:1,3:100},
        sampler=dict(actual_dimension=4,selected_ids=[1,2,3,7]),
        hci=dict(dimension=2,selected_ids=[3,7],coefficients=[-.9,.1]))
    selected=[3,7]; key=hashlib.sha256(json.dumps(['physical',selected]).encode()).hexdigest()
    space=dict(problem_id='physical',selected_ids=selected,energy_Ha=-2.,
        independent_recheck_energy_Ha=-2.,internal_residual=0.,new_eigensolver_calls=2,analysis_seconds=.01)
    row={k:old[k] for k in ('case','method','R_cap','epsilon1_Ha','sampler_status','HCI_status')}
    row.update(status='PAIRED_DIMENSION_DIAGNOSTIC',matched_d=2,trivial_HF_space=False,work_matched=False,
        sampler_space_key=key,HCI_coefficient_truncated_space_key=key,
        sampler_restricted_energy_Ha=-2.,HCI_coefficient_truncated_energy_Ha=-2.,
        sampler_minus_truncated_HCI_mHa=0.,original_sampler_energy_Ha=-3.,original_HCI_energy_Ha=-4.)
    result=dict(status='ALL_PAIRED_DIAGNOSTIC_SPACES_RECHECKED',rows=[row],unique_spaces={key:space},
        new_quantum_acquisitions=0,new_FCI_solves=0,whole_goal_complete=False,
        new_eigensolver_calls=2,maximum_eigensolver_disagreement_Ha=0.,analysis_wall_seconds=.02)
    return dict(rows=[old],planned_pairs=1),result,{identity(old):source}


def test_all_planned_arithmetic_with_shared_space_charged_once():
    config,result,sources=fixture(); rows,status=check_body(config,result,sources)
    assert rows[0]['matched_d']==2 and rows[0]['ground_state_error_Ha'] is None
    assert status=={'PAIRED_DIMENSION_DIAGNOSTIC':1}
    assert rows[0]['original_HCI_energy_Ha']==-4.


def test_missingness_and_original_native_endpoint_retained():
    config,result,sources=fixture()
    old=deepcopy(config['rows'][0]);old.update(method='failed',comparison_status='MISSING_PAIR_PRESERVED',
        sampler_status='ENGINEERING_FAILURE',sampler_energy_Ha=None)
    config['rows'].append(old);config['planned_pairs']=2
    result['rows'].append({k:old[k] for k in ('case','method','R_cap','epsilon1_Ha','sampler_status','HCI_status')})
    result['rows'][-1]['status']='MISSING_PAIR_PRESERVED'
    rows,_=check_body(config,result,sources)
    assert rows[1]['matched_d'] is None and rows[1]['sampler_restricted_energy_Ha'] is None
    assert rows[1]['original_HCI_energy_Ha']==-4.


@pytest.mark.parametrize('change', ['duplicate','drop','dimension','bool_dimension','work','nan',
    'difference','original','residual','recheck','problem','selection','calls','unknown_acquisition',
    'summary','unreferenced','variational'])
def test_tampering_rejected(change):
    config,result,sources=fixture();row=result['rows'][0];space=next(iter(result['unique_spaces'].values()))
    if change=='duplicate':result['rows'].append(deepcopy(row))
    elif change=='drop':result['rows']=[]
    elif change=='dimension':row['matched_d']=3
    elif change=='bool_dimension':row['matched_d']=True
    elif change=='work':row['work_matched']=True
    elif change=='nan':space['energy_Ha']=float('nan')
    elif change=='difference':row['sampler_minus_truncated_HCI_mHa']=1.
    elif change=='original':row['original_HCI_energy_Ha']=-3.
    elif change=='residual':space['internal_residual']=.1
    elif change=='recheck':space['independent_recheck_energy_Ha']=-1.
    elif change=='problem':space['problem_id']='other'
    elif change=='selection':space['selected_ids']=[1,7]
    elif change=='calls':result['new_eigensolver_calls']=4
    elif change=='unknown_acquisition':result['new_quantum_acquisitions']=None
    elif change=='summary':result['maximum_eigensolver_disagreement_Ha']=.1
    elif change=='unreferenced':result['unique_spaces']['extra']=deepcopy(space)
    elif change=='variational':config['rows'][0]['sampler_energy_Ha']=-1.;row['original_sampler_energy_Ha']=-1.
    with pytest.raises((ValueError,KeyError)):check_body(config,result,sources)


def test_frequency_and_coefficient_ranking_not_numeric_prefix():
    assert expected_subset([1,2,3,7],7,2,{1:3,2:3,3:100})==[3,7]
    assert expected_subset([1,2,3,7],7,3,{1:.2,2:.2,3:.1})==[1,2,7]


def test_unobserved_config_not_inserted():
    config,result,sources=fixture();next(iter(sources.values()))['counts'].pop(3)
    with pytest.raises(ValueError,match='Unobserved'):check_body(config,result,sources)


def test_absent_production_result_is_not_success(tmp_path):
    from benchmark.hci_paired_review import review
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(FileNotFoundError):review(tmp_path)
