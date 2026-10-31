"""All-K projection tests; no new quantum/CI work or existing-output edits."""
import pandas as pd
import pytest
from benchmark.topk_discovery_figures import KS,project


def fixture():
    inventory=pd.DataFrame([dict(case_id='X',method='ceo',step=0,checkpoint_id='old',primary=True),
        dict(case_id='X',method='ceo',step=1,checkpoint_id='last',primary=True),
        dict(case_id='X',method='ceo',step=2,checkpoint_id='superseded',primary=False)])
    rows=[]
    for cp,value in (('old',.9),('last',.2)):
        for M in (10,100):
            row=dict(checkpoint_id=cp,shots=M,primary_record=True)
            for k in KS:row.update({f'recall_top{k}':value,f'recall_top{k}_actual_targets':min(2,k)})
            rows.append(row)
    planned=pd.DataFrame([dict(case_id='X',method='ceo',status='MAX_STEPS'),
                          dict(case_id='X',method='missing',status='INITIALIZATION_FAILURE')])
    return inventory,pd.DataFrame(rows),planned


def test_last_primary_not_best_or_superseded():
    inv,met,plan=fixture();t=project(inv,met,plan,[10,100]);a=t[t.method=='ceo']
    assert len(t)==20 and set(t.K_requested)==set(KS)
    assert set(a.checkpoint_id)=={'last'} and set(a.expected_recall)=={.2}
    assert set(a.source_terminal_status)=={'MAX_STEPS'}


def test_missing_values_are_not_zero_or_success():
    inv,met,plan=fixture();t=project(inv,met,plan,[10,100]);m=t[t.method=='missing']
    assert set(m.status)=={'MISSING_SOURCE_PRESERVED'}
    assert m.expected_recall.isna().all() and m.actual_targets.isna().all() and m.step.isna().all()


def test_actual_targets_can_be_less_than_requested():
    inv,met,plan=fixture();t=project(inv,met,plan,[10,100]);a=t[t.method=='ceo']
    assert set(a.actual_targets)=={2}


@pytest.mark.parametrize('kind',['duplicate_planned','duplicate_primary','duplicate_metric','missing_metric','nan_recall','target_count'])
def test_ambiguous_or_invalid_input_is_rejected(kind):
    inv,met,plan=fixture()
    if kind=='duplicate_planned':plan=pd.concat([plan,plan.iloc[[0]]])
    elif kind=='duplicate_primary':inv=pd.concat([inv,inv.iloc[[0]]])
    elif kind=='duplicate_metric':met=pd.concat([met,met.iloc[[0]]])
    elif kind=='missing_metric':met=met[~((met.checkpoint_id=='last')&(met.shots==10))]
    elif kind=='nan_recall':met.loc[met.checkpoint_id=='last','recall_top3']=float('nan')
    elif kind=='target_count':met.loc[met.checkpoint_id=='last','recall_top3_actual_targets']=4
    with pytest.raises((ValueError,KeyError)):project(inv,met,plan,[10,100])


def test_no_target_remains_undefined():
    inv,met,plan=fixture()
    for k in KS:
        met[f'recall_top{k}_actual_targets']=0;met[f'recall_top{k}']=float('nan')
    t=project(inv,met,plan,[10,100]);a=t[t.method=='ceo']
    assert a.expected_recall.isna().all() and set(a.status)=={'NO_NONHF_TARGETS'}
