"""Interpretation tests; no quantum module or solver required."""
import json
import pytest

from benchmark.molecular_failure_diagnostics import classify


def failed(error):
    return {'status': 'ENGINEERING_OR_INITIALIZATION_FAILURE_PRESERVED', 'error': error,
            'new_primary_model_shots': None, 'unreported_inflight_model_shots': None}


def test_mps_limit_does_not_become_physical_method_loss_or_zero_cost():
    row = classify(failed("RuntimeError('MPS construction resource cap at gate boundary')"), 1-1e-6)
    assert row['category'] == 'MPS_CONSTRUCTION_RESOURCE_CAP'
    assert row['original_new_primary_model_shots'] is None
    assert row['rejected_operation_new_model_shots'] is None
    assert not row['whole_cell_physical_method_loss']
    assert not row['whole_cell_quantum_cost_reconciled_by_this_report']


def test_only_structured_rejected_operation_has_zero_samples():
    certificate = {'build': {'retained_norm_squared': .99}, 'confirmed_new_model_shots': 0}
    error = "BlockedApproximationUnqualified('resource cap BEFORE_SAMPLING " + json.dumps(certificate) + "')"
    row = classify(failed(error), 1-1e-6)
    assert row['category'] == 'MPS_APPROXIMATION_UNQUALIFIED'
    assert row['rejected_operation_new_model_shots'] == 0
    assert row['original_new_primary_model_shots'] is None
    assert row['original_unreported_inflight_model_shots'] is None
    assert not row['whole_cell_quantum_cost_reconciled_by_this_report']
    certificate['confirmed_new_model_shots'] = False
    bad = error[:error.index('{')] + json.dumps(certificate) + "')"
    assert classify(failed(bad), 1-1e-6)['category'] == 'UNCLASSIFIED_ENGINEERING_FAILURE'


@pytest.mark.parametrize('error', ["BlockedApproximationUnqualified('bad')", 'RuntimeError(unknown)',
    "BlockedApproximationUnqualified('bad {\"build\": {\"retained_norm_squared\": 1}, \"confirmed_new_model_shots\": 0}')"])
def test_unknown_or_inconsistent_evidence_is_not_silently_qualified(error):
    row = classify(failed(error), 1-1e-6)
    assert row['category'] == 'UNCLASSIFIED_ENGINEERING_FAILURE'
    assert row['rejected_operation_new_model_shots'] is None


def test_shared_initialization_failure_is_separate_from_first_rejection():
    first = failed("RuntimeError('CCSD did not converge; do not initialize LUCJ from failed amplitudes')")
    shared = failed("RuntimeError('Preserved amplitude initialization failure: /case/amplitude-dependencies/ccsd.json')")
    assert classify(first, .999999)['category'] == 'CCSD_INITIALIZATION_REJECTED'
    assert classify(shared, .999999)['category'] == 'PRESERVED_INITIALIZATION_DEPENDENCY_FAILURE'
    complete = classify({'status': 'COMPLETED'}, .999999)
    assert complete['category'] == 'COMPLETED_NOT_AUTOMATICALLY_ACCURATE'
    assert not complete['whole_cell_physical_method_loss']


def test_all_planned_cells_and_physical_failure_are_kept_without_source_writes(tmp_path):
    from benchmark.molecular_failure_diagnostics import build, sha
    source, output = tmp_path/'source', tmp_path/'derived'
    source.mkdir()
    config = {'schema':'molecular-strong-controls-v2-spin-blocked','implementation':{},
              'queue':[{'id':'broken'},{'id':'pending'}],'methods':['A','B'],
              'minimum_retained_norm_squared':1-1e-6}
    (source/'manifest.json').write_text(json.dumps(config))
    bad=source/'cases/broken/physical-failure.json'
    bad.parent.mkdir(parents=True);bad.write_text(json.dumps({'error':'input construction failed'}))
    before={p:sha(p) for p in source.rglob('*') if p.is_file()}
    record,folder=build(source,output)
    assert record['planned']==4 and record['terminal_at_snapshot']==2
    assert record['category_counts']['PHYSICAL_INPUT_FAILURE_PRESERVED']==2
    assert record['category_counts']['NOT_TERMINAL_AT_SNAPSHOT']==2
    assert before=={p:sha(p) for p in source.rglob('*') if p.is_file()}
    assert (folder/'all-planned-cells.csv').exists()
    assert record['new_model_shots']==0 and record['new_ground_state_solves']==0
    with pytest.raises(ValueError,match='outside'):
        build(source,source/'unsafe')
