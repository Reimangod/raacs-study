"""Tests of report-only calculations, not new sampler or solver experiments."""
import numpy as np
import pandas as pd
import pytest

from benchmark.final_relationship_figures import adjacent_improvements, primary_readouts, weighted_nonhf_discovery


def test_marginal_coverage_matches_small_explicit_outcomes():
    ids, w, p = [0, 1, 2], [.6, .3, .1], [.2, .5, .3]
    expected = sum(p[i]*p[j]*sum(w[k] for k in (1,2) if k in (i,j))/.4 for i in range(3) for j in range(3))
    assert weighted_nonhf_discovery(ids,w,0,p,2) == pytest.approx(expected)
    assert weighted_nonhf_discovery(ids,w,0,p,0) == 0
    assert weighted_nonhf_discovery([0],[1.],0,[1.],100) is None


def test_oracle_born_distribution_is_not_a_discovery_upper_bound():
    oracle = weighted_nonhf_discovery([0,1],[.9,.1],0,[.9,.1],1)
    alternative = weighted_nonhf_discovery([0,1],[.9,.1],0,[0.,1.],1)
    assert alternative > oracle
    with pytest.raises(ValueError): weighted_nonhf_discovery([0,1],[.9,.1],0,[.9,.9],1)
    with pytest.raises(ValueError): weighted_nonhf_discovery([0,0],[.9,.1],0,[.9,.1],1)
    with pytest.raises(ValueError): weighted_nonhf_discovery([0,1],[.9,.1],0,[.9,.1],.5)


def test_growth_changes_retain_worsening_and_refuse_missing_steps():
    frame = pd.DataFrame([dict(case_id="toy",method="ceo_ovp",step=i,checkpoint_id=str(i),
        cnot=c,depth=d,mean_error_mHa=e) for i,c,d,e in [(0,0,0,10.),(1,9,7,4.),(2,18,14,5.)]])
    result=adjacent_improvements(frame)
    assert [r["qsci_gain_mHa"] for r in result] == [6.,-1.]
    assert all(r["delta_cnot"] == 9 for r in result)
    assert all("REOPTIMIZATION" in r["interpretation"] for r in result)
    with pytest.raises(ValueError):adjacent_improvements(frame.iloc[[0,2]])


def test_readout_selector_never_uses_archive_incomplete_or_different_budget():
    row=dict(primary_record=True,complete_group=True,selection_rule="frequency",shots=1000,R_cap=32)
    frame=pd.DataFrame([row,{**row,"primary_record":False},{**row,"complete_group":False},
        {**row,"shots":10000},{**row,"R_cap":64},{**row,"selection_rule":"uniform_observed"}])
    assert primary_readouts(frame).index.tolist() == [0]
