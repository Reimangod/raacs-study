import pandas as pd
import pytest
from benchmark.research_synthesis_v2 import question_titles,paired_cases,equal_molecule_pairs,trajectory_associations


def row(case,method,error,complete=True):
    return dict(case_id=case,family=method,mean_error_mHa=error,worst_registered_component_cnot=10,
        worst_registered_component_depth=5,mean_conditional_success=.9,full_model_budget_known=False,
        all_registered_seeds_available=complete)


def test_all_cases_and_incomplete_seeds_preserved():
    d=pd.DataFrame([row('A_R1','CEO',2),row('A_R1','X',1,False),row('A_R2','CEO',1)])
    pairs=paired_cases(d,'CEO',['A_R1','A_R2'])
    assert len(pairs)==4
    assert (pairs[pairs.family=='X'].status=='MISSING_PAIR_PRESERVED').all()
    with pytest.raises(ValueError):paired_cases(pd.concat([d,d]),'CEO',['A_R1'])


def test_molecule_equal_weight_is_not_geometry_pooling():
    d=pd.DataFrame([row('A_R1','CEO',0),row('A_R1','X',1),row('A_R2','CEO',0),row('A_R2','X',1),
                    row('B_R1','CEO',0),row('B_R1','X',9)])
    pairs=paired_cases(d,'CEO',['A_R1','A_R2','B_R1']);mol,means=equal_molecule_pairs(pairs)
    assert means.set_index('family').loc['X','equal_molecule_delta_error_mHa']==5
    assert means.set_index('family').loc['X','paired_conditions']==3


def test_trajectory_rank_not_cross_problem_absolute_energy():
    a=pd.DataFrame(dict(case_id=['A_R1']*3+['B_R1'],method=['CEO']*4,
        checkpoint_id=list('abcd'),mean_error_mHa=[3,2,1,0]))
    b=pd.DataFrame(dict(checkpoint_id=list('abcd'),nonhf_weighted_coverage=[.1,.2,.3,1.]))
    r=trajectory_associations(a,b).set_index('case_id')
    assert r.loc['A_R1','spearman_coverage_vs_QSCI_error']==-1
    assert r.loc['B_R1','status']=='CONSTANT_OR_TOO_FEW'
    assert not r.isolated_causal_effect.any()


def test_fifteen_questions_come_from_full_objective():
    text='# 11. 最終レポートで答える研究上の質問\n'+'\n'.join(f'{i}. Question{i}' for i in range(1,16))+'\n# 12. End'
    assert len(question_titles(text))==15
    with pytest.raises(ValueError):question_titles(text.replace('15. Question15',''))


def test_report_opt_in_dispatch_restores_legacy_argv(monkeypatch):
    import sys
    from benchmark import report,research_synthesis_v2 as research_synthesis
    original=['benchmark.report','--scalable-study','--mode','quick','--output','unused']
    seen=[];monkeypatch.setattr(sys,'argv',original)
    monkeypatch.setattr(research_synthesis,'main',lambda:seen.append(list(sys.argv)))
    report.main()
    assert seen==[['benchmark.report','--mode','quick','--output','unused']]
    assert sys.argv is original


def test_plot_opt_in_dispatch_restores_legacy_argv(monkeypatch):
    import sys
    from benchmark import plot,research_figure_cli_v2 as research_figure_cli
    original=['benchmark.plot','--all','--scalable-study','--mode','quick']
    seen=[];monkeypatch.setattr(sys,'argv',original)
    monkeypatch.setattr(research_figure_cli,'main',lambda:seen.append(list(sys.argv)))
    plot.main()
    assert seen==[['benchmark.plot','--all','--mode','quick']]
    assert sys.argv is original


def test_direct_RAACS_pairing_does_not_subtract_different_geometry_means():
    from benchmark.research_synthesis_review import direct_comparisons,PRIMARY
    d=pd.DataFrame([row('A_R1',PRIMARY,10),row('A_R2',PRIMARY,100),row('A_R1','X',1)])
    d['scope']='readout_only';d['M']=1000;d['total_model_shots_seed_mean']=float('nan')
    r=direct_comparisons(d,['A_R1','A_R2'])
    q=r[(r.baseline_family=='X')&(r.scope=='readout_only')&(r.M==1000)]
    assert q.status.tolist()==['AVAILABLE_COMPLETE_SEED_PAIR','BASELINE_NO_BUDGET_ENDPOINT']
    assert q.RAACS_minus_baseline_mHa.mean()==9 # not mean([10,100])-1=54


def test_direct_known_budget_refuses_unknown_or_partial_family():
    from benchmark.research_synthesis_review import direct_comparisons,PRIMARY
    d=pd.DataFrame([row('A_R1',PRIMARY,1),row('A_R1','X',0)])
    d['scope']='known_build_plus_readout';d['M']=1000;d['total_model_shots_seed_mean']=float('nan')
    r=direct_comparisons(d,['A_R1'])
    q=r[(r.baseline_family=='X')&(r.scope=='known_build_plus_readout')&(r.M==1000)]
    assert q.iloc[0].status=='INCOMPLETE_OR_UNKNOWN_BUDGET_PRESERVED'
    assert pd.isna(q.iloc[0].get('RAACS_minus_baseline_mHa'))


def test_relationship_verification_never_renders_or_writes(tmp_path,monkeypatch):
    import json
    from benchmark import read_only_relationship_validation as v
    monkeypatch.setattr(v,'ROOT',tmp_path)
    (tmp_path/'benchmark').mkdir();(tmp_path/'benchmark/final_relationship_followup.py').write_text('fixture auditor')
    (tmp_path/'source.csv').write_text('source')
    root=tmp_path/'result';out=root/'presentation-v2';out.mkdir(parents=True)
    (root/'table.csv').write_text('numbers');(out/'image.png').write_bytes(b'fixture image')
    m={'source_sha256':{'source.csv':v.digest(tmp_path/'source.csv')},'output_sha256':{'table.csv':v.digest(root/'table.csv')}}
    (root/'manifest.json').write_text(json.dumps(m))
    record={'status':'ALL_PLOTTED_TABLES_RECONSTRUCTED_PRESENTATION_V2_EXPORTED',
        'source_manifest_sha256':v.digest(root/'manifest.json'),'auditor_sha256':v.digest(tmp_path/'benchmark/final_relationship_followup.py'),
        'presentation_output_sha256':{'image.png':v.digest(out/'image.png')}}
    (out/'validation-and-style-v2.json').write_text(json.dumps(record))
    before={str(p):(v.digest(p),p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    assert v.validate(root)['new_renderings']==0
    assert v.validate(root)['new_renderings']==0
    after={str(p):(v.digest(p),p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    assert before==after


def test_legacy_report_default_unchanged(tmp_path,monkeypatch):
    import sys
    from benchmark import report
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(sys,'argv',['benchmark.report'])
    seen=[];monkeypatch.setattr(report,'make_report',lambda x:seen.append(str(x)))
    report.main();assert seen==['results/quick']


def test_legacy_plot_default_unchanged(tmp_path,monkeypatch):
    import sys
    from benchmark import plot
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(sys,'argv',['benchmark.plot','--all'])
    seen=[];monkeypatch.setattr(plot,'plot_all',lambda x:seen.append(str(x)))
    plot.main();assert seen==['results/quick']
