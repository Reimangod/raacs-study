import pytest
from benchmark.current_evidence_review import check_hci, check_cost, validate_links


def test_real_hci_sign_and_missingness_are_checked_independently():
    rows=[dict(method='A',R_cap='8',epsilon1_Ha='.01',status='PAIRED_DIMENSION_DIAGNOSTIC',matched_d='2',sampler_minus_truncated_HCI_mHa='3')]
    summary=[dict(method='A',R_cap='8',epsilon1_Ha='.01',planned='1',available='1',nontrivial='1',trivial_HF='0',missing='0',
                  median_sampler_minus_truncated_HCI_mHa='3',negative_favors_sampler='0',within_numerical_1e_8_mHa='0',positive_favors_truncated_HCI='1',ground_state_error='',work_matched='False')]
    check_hci(rows,summary)
    summary[0]['missing']='1'
    with pytest.raises(ValueError):check_hci(rows,summary)


def test_independent_cost_review_refuses_zero_fill_and_geometry_pooling():
    rows=[dict(method='A',molecule=m,status='COMPLETED',total_workflow_wall_seconds=t,all_draw_CNOT_times_shots_proxy=10*t)
          for m,t in [('X',1),('X',1),('Y',9)]]
    summary=[dict(method='A',planned='3',completed='3',missing_or_failed='0',historical_CEO_training_shot_cost='',
                  equal_molecule_observed_total_wall_seconds='5',equal_molecule_all_paid_CNOT_times_model_samples='50')]
    check_cost(rows,summary)
    summary[0]['historical_CEO_training_shot_cost']='0'
    with pytest.raises(ValueError):check_cost(rows,summary)
    summary[0]['historical_CEO_training_shot_cost']='';summary[0]['equal_molecule_observed_total_wall_seconds']=str(11/3)
    with pytest.raises(ValueError):check_cost(rows,summary)


def test_report_links_must_resolve(tmp_path):
    p=tmp_path/'README.md';p.write_text('[real](target.csv)');(tmp_path/'target.csv').touch()
    assert validate_links(p)==1
    p.write_text('[missing](missing.csv)')
    with pytest.raises(ValueError):validate_links(p)


def test_current_report_dispatch_keeps_legacy_arguments(monkeypatch):
    import sys
    from benchmark import report,current_evidence
    original=['benchmark.report','--scalable-study','--current-evidence','--mode','quick']
    monkeypatch.setattr(sys,'argv',original);seen=[]
    monkeypatch.setattr(current_evidence,'main',lambda:seen.append(list(sys.argv)))
    report.main();assert seen==[['benchmark.report','--mode','quick']] and sys.argv is original


def test_current_plot_dispatch_is_read_only_index(monkeypatch):
    import sys
    from benchmark import plot,current_figure_index
    original=['benchmark.plot','--all','--scalable-study','--current-evidence','--mode','standard']
    monkeypatch.setattr(sys,'argv',original);seen=[]
    monkeypatch.setattr(current_figure_index,'main',lambda:seen.append(list(sys.argv)))
    plot.main();assert seen==[['benchmark.plot','--all','--mode','standard']] and sys.argv is original
