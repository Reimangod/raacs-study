import csv
import json
import pytest
from benchmark.current_evidence import terminal_counts, hci_summary, workflow_summary, update_questions, save_once


def test_running_does_not_inflate_terminal_prefix():
    assert terminal_counts({'COMPLETED':4,'RUNNING':1,'PENDING':2,'SOURCE_INITIALIZATION_MISSING':3})==7


def test_hci_missing_and_hf_only_are_not_nontrivial_wins():
    rows=[dict(method='X',R_cap='8',epsilon1_Ha='.01',status='PAIRED_DIMENSION_DIAGNOSTIC',
               matched_d='1',sampler_minus_truncated_HCI_mHa='0'),
          dict(method='X',R_cap='8',epsilon1_Ha='.01',status='MISSING_PAIR_PRESERVED',
               matched_d='',sampler_minus_truncated_HCI_mHa='')]
    r=hci_summary(rows)[0]
    assert (r['planned'],r['available'],r['nontrivial'],r['trivial_HF'],r['missing'])==(2,1,0,1,1)
    assert r['median_sampler_minus_truncated_HCI_mHa'] is None and r['ground_state_error'] is None
    assert not r['work_matched']


def test_all_hci_thresholds_retained_and_delta_sign_explicit():
    rows=[dict(method='X',R_cap='8',epsilon1_Ha=e,status='PAIRED_DIMENSION_DIAGNOSTIC',
               matched_d='2',sampler_minus_truncated_HCI_mHa=d) for e,d in [('.01','2'),('.001','-3')]]
    r=hci_summary(rows)
    assert len(r)==2 and sum(v['positive_favors_truncated_HCI'] for v in r)==1
    assert sum(v['negative_favors_sampler'] for v in r)==1


def test_cost_means_molecules_equally_without_erasing_failures():
    def row(mol,t,status='COMPLETED'):
        return dict(method='X',molecule=mol,status=status,total_workflow_wall_seconds=t,
                    all_draw_CNOT_times_shots_proxy=t*10)
    data={'rows':[row('A',1),row('A',1),row('B',9),row('B',100,'FAIL')]}
    r=workflow_summary(data)[0]
    assert r['planned']==4 and r['completed']==3 and r['missing_or_failed']==1
    assert r['equal_molecule_observed_total_wall_seconds']==5
    assert r['historical_CEO_training_shot_cost'] is None


def test_questions_update_without_mutating_historical_answers():
    old=[dict(number=i,status='old',answer='old',title=str(i),evidence=['table.csv']) for i in range(1,16)]
    new=update_questions(old)
    assert all(q['status']=='old' for q in old)
    assert new[10]['status']=='GIVEN_INTEGRALS_TOTAL_WORKFLOW_MEASURED_NOT_HISTORICAL_CEO_TOTAL'
    assert new[14]['status'].startswith('THRESHOLD_HCI_')
    with pytest.raises(ValueError):update_questions(old[:-1])


def test_save_once_is_non_destructive(tmp_path):
    p=tmp_path/'output';save_once(p,'known');t=p.stat().st_mtime_ns;save_once(p,'known')
    assert p.stat().st_mtime_ns==t
    with pytest.raises(ValueError):save_once(p,'different')
