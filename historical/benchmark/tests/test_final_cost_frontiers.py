import numpy as np
import pandas as pd
import pytest
from benchmark.final_cost_frontiers_v2 import build_budget,budget_view,complete_mean,constructor_time,family,nondominated
from benchmark.final_cost_frontier_audit import direct_frontier,equal


def test_unknown_vqe_and_parent_topology_are_not_free():
    assert build_budget('B1:lucj_ccsd_2_vqe',{'spec':{'objective':'vqe'},'simulated_construction_shots':0})[0] is None
    assert build_budget('B1:source_final_ceo_random_101',{'config':{'parent_checkpoint_id':42}})[0] is None
    assert build_budget('B1:source_final_ceo_mvp',{})[0] is None
    assert build_budget('B1:source_final_hf',{})[0]==0
    assert build_budget('RAACS:x',{'pilot_raw_model_shots':4128})[0]==4128
    assert build_budget('B2exact:x',{'construction_simulated_raw_shots':6000})[0]==6000
    with pytest.raises(ValueError):build_budget('B3:x',{'simulated_construction_shots':-1})


def test_both_budget_views_and_unknown_are_explicit():
    assert budget_view('readout_only',10000,10000,4128)[0]==14128
    assert budget_view('known_build_plus_readout',10000,5872,4128)[0]==10000
    assert budget_view('readout_only',1000,1000,None)[0] is None
    assert budget_view('known_build_plus_readout',10000,10000,None)[0] is None
    with pytest.raises(ValueError):budget_view('known_build_plus_readout',10000,10000,4128)


def test_time_scope_dependency_not_double_charged_or_assumed_for_b2():
    r={'construction_wall_seconds':10,'initialization':{'dependency_total_seconds':3,'dependency_new_seconds':2}}
    assert constructor_time('B1:x',r)[0]==11
    assert constructor_time('B2exact:x',{'construction_wall_seconds':10,'initialization':{'kind':'exact physical'}})==(10,'B2exact_RECORDED_CONSTRUCTION')
    assert constructor_time('B1:source_final_hf',{})[0] is None
    assert constructor_time('B1:x',{'construction_wall_seconds':10,'initialization':{'kind':'unknown'}})[0] is None
    assert complete_mean(pd.Series([1.,np.nan])) is None
    assert complete_mean(pd.Series([1.,3.]))==2.


def test_seed_grouping_and_strict_pareto_unknowns():
    assert family('RAACS:full_cost-s101-exact_small')==family('RAACS:full_cost-s211-exact_small')
    assert family('RAACS:full_cost-s101-mps')!=family('RAACS:full_cost-s101-exact_small')
    assert family('B3:Pauli_qDRIFT_t1_steps256_s101')==family('B3:Pauli_qDRIFT_t1_steps256_s211')
    points=[[1,3],[2,2],[3,3],[np.nan,1],[1,3]]
    assert nondominated(points).tolist()==[True,True,False,False,True]
    assert direct_frontier(points)==[True,True,False,False,True]


def test_independent_audit_rejects_fabricated_unknown_and_frontier():
    with pytest.raises(AssertionError):equal(0,None)
    with pytest.raises(AssertionError):equal(3,4)
    equal(np.nan,None)
    assert direct_frontier([[1,1],[2,2],[np.inf,0],[2,1]])==[True,False,False,False]


def test_empty_log_budget_panel_has_explicit_positive_domain():
    import io
    import inspect
    import matplotlib.pyplot as plt
    from benchmark.final_cost_frontiers_v2 import build
    assert "ax.set_xlim(min(cfg['M'])*.8,max(cfg['M'])*1.25)" in inspect.getsource(build)
    fig,ax=plt.subplots()
    ax.scatter([],[])
    ax.set_xscale('log')
    ax.set_xlim(1000*.8,100000*1.25)
    fig.tight_layout()
    dest=io.BytesIO();fig.savefig(dest,format='png');plt.close(fig)
    assert dest.getvalue().startswith(b'\x89PNG')
