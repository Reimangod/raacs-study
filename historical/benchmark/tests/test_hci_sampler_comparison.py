"""Pure endpoint/accounting tests; no quantum backend or eigensolve imports."""
from copy import deepcopy
import csv
import hashlib
import json

import pytest

from benchmark.hci_sampler_comparison import (
    build, canonical, endpoint_row, require_same_problem, sha, unique_index,
)


def problem():
    return dict(problem_id='same-physical-problem', n_qubits=4, nalpha=1, nbeta=1,
                hf_id=5, HF_energy_Ha=-1., physical_integral_binding={'h1': 'h', 'eri': 'v'})


def endpoints():
    q = dict(status='COMPLETED', readouts=[dict(R=8, actual_dimension=4, energy_Ha=-1.2)],
             sampling_resources=dict(cnot_count=12, cnot_depth=8), unknown_inflight_draws=None)
    h = dict(status='DIMENSION_CAP_BEFORE_COMPLETE_THRESHOLD_EXPANSION', epsilon1_Ha=.01,
             final_energy_Ha=-1.25, final_dimension=8, quantum_shots=0)
    return q, h


def test_sign_and_unequal_dimension_are_not_a_matched_work_win():
    q, h = endpoints()
    row = endpoint_row('x', 'RAACS-full-cost', 8, q, h)
    assert row['sampler_minus_HCI_mHa'] == pytest.approx(50.)
    assert row['dimension_comparison'] == 'NOT_MATCHED'
    assert row['HCI_threshold_set_closed'] is False
    assert row['work_matched_superiority_established'] is False
    assert row['sampler_unknown_inflight_draws'] is None
    assert row['HCI_CNOT'] is None


def test_equal_dimension_does_not_erase_cost_or_reference_unknowns():
    q, h = endpoints(); h['final_dimension'] = 4
    row = endpoint_row('x', 'RAACS-full-cost', 8, q, h)
    assert row['equal_actual_dimension'] is True
    assert row['timing_scope'].startswith('UNMATCHED')
    assert row['sampler_ground_state_error_Ha'] is None
    assert row['HCI_ground_state_error_Ha'] is None
    assert row['work_matched_superiority_established'] is False


@pytest.mark.parametrize('status', ['ENGINEERING_OR_INITIALIZATION_FAILURE_PRESERVED', 'NOT_STARTED'])
def test_failed_q_is_retained_even_if_an_energy_is_present(status):
    q, h = endpoints(); q['status'] = status
    row = endpoint_row('x', 'RAACS-full-cost', 8, q, h)
    assert row['comparison_status'] == 'MISSING_PAIR_PRESERVED'
    assert row['sampler_minus_HCI_mHa'] is None
    assert row['equal_actual_dimension'] is False


def test_unavailable_r_or_nonfinite_energy_is_not_zero():
    q, h = endpoints()
    assert endpoint_row('x', 'm', 64, q, h)['sampler_energy_Ha'] is None
    h['final_energy_Ha'] = float('nan')
    assert endpoint_row('x', 'm', 8, q, h)['HCI_last_solved_energy_Ha'] is None
    q['readouts'] = None
    assert endpoint_row('x', 'm', 8, q, h)['comparison_status'] == 'MISSING_PAIR_PRESERVED'


@pytest.mark.parametrize('key', ['problem_id', 'physical_integral_binding', 'n_qubits', 'nalpha', 'nbeta', 'hf_id'])
def test_physical_identity_mismatch_is_rejected(key):
    a = problem(); b = deepcopy(a); b[key] = None
    with pytest.raises(ValueError, match='Physical problem mismatch'):
        require_same_problem(a, b)


def test_duplicate_endpoint_or_readout_is_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        unique_index([{'id': 1}, {'id': 1}], ('id',))
    q, h = endpoints(); q['readouts'] *= 2
    with pytest.raises(ValueError, match='Duplicate'):
        endpoint_row('x', 'm', 8, q, h)


def fixture(tmp):
    e, h, physical = tmp / 'E', tmp / 'HCI', tmp / 'physical'
    def save(path, record):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(canonical(record))
        return sha(path)
    p = problem()
    meta = save(physical / 'problem.json', p)
    archive = physical / 'integrals.npz'; archive.write_bytes(b'fixture-integral-bytes-not-an-executed-problem')
    es = dict(path=str(physical), meta_sha256=meta, integrals_sha256=sha(archive))
    hs = dict(path=str(physical), metadata_sha256=meta, archive_sha256=sha(archive), problem=p)
    save(e / 'cases/x/physical-source.json', es)
    save(h / 'cases/E-x/source.json', hs)
    em = dict(queue=[{'id': 'x'}], methods=['RAACS-full-cost', 'failed'])
    hm = dict(queue=[dict(id='E-x', source_cohort='E26')], thresholds=[.01])
    emsha, hmsha = save(e / 'manifest.json', em), save(h / 'manifest.json', hm)
    q, hc = endpoints()
    q.update(case='x', method='RAACS-full-cost', problem=p)
    q['result_sha256'] = save(e / 'cases/x/RAACS-full-cost/result.json', q)
    failed = dict(case='x', method='failed', status='ENGINEERING_OR_INITIALIZATION_FAILURE_PRESERVED')
    failed['result_sha256'] = save(e / 'cases/x/failed/result.json', failed)
    hc.update(case='E-x')
    hc['source_result_sha256'] = save(h / 'cases/E-x/epsilon-0.01/result.json', hc)
    save(e / 'validation.json', dict(status='MOLECULAR_STRONG_ALL_RECORDS_AUDITED', manifest_sha256=emsha, rows=[q, failed]))
    save(h / 'validation.json', dict(status='THRESHOLD_HCI_STANDARD_INDEPENDENTLY_AUDITED', manifest_sha256=hmsha, rows=[hc]))
    return e, h


def test_full_denominator_determinism_source_preservation_and_export_hashes(tmp_path):
    e, h = fixture(tmp_path)
    before = {str(p): sha(p) for root in (e, h) for p in root.rglob('*') if p.is_file()}
    body, folder = build(e, h, tmp_path / 'out')
    assert body['planned_pairs'] == 6
    assert body['counts'] == {'AVAILABLE_DESCRIPTIVE_PAIR': 1, 'MISSING_PAIR_PRESERVED': 5}
    again, same = build(e, h, tmp_path / 'out')
    assert body == again and folder == same
    assert all(sha(p) == digest for p, digest in before.items())
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert all(sha(folder / name) == digest for name, digest in manifest['files'].items())
    assert sha(folder / 'comparison.json') == folder.name
    with (folder / 'all-planned-pairs.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6 and sum(r['comparison_status'] == 'MISSING_PAIR_PRESERVED' for r in rows) == 5
    assert body['new_model_shots'] == body['new_CI_solves'] == 0


def test_changed_integrals_fail_closed(tmp_path):
    e, h = fixture(tmp_path)
    (tmp_path / 'physical/integrals.npz').write_bytes(b'different')
    with pytest.raises(ValueError, match='Integral archive changed'):
        build(e, h, tmp_path / 'out')


def test_independent_review_reconstructs_saved_pairs(tmp_path):
    from benchmark.hci_sampler_comparison_review import review
    e, h = fixture(tmp_path)
    _, folder = build(e, h, tmp_path / 'out')
    result = review(folder)
    assert result['planned_pairs'] == 6
    assert result['available_pairs'] == 1 and result['missing_pairs'] == 5
    assert result['new_CI_solves'] == result['new_model_shots'] == 0


def test_independent_review_rejects_changed_csv(tmp_path):
    from benchmark.hci_sampler_comparison_review import review
    e, h = fixture(tmp_path)
    _, folder = build(e, h, tmp_path / 'out')
    path = folder / 'all-planned-pairs.csv'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(AssertionError):
        review(folder)
