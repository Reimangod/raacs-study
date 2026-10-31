"""Finite confirmation scientific contract, not another source-search loop."""
from pathlib import Path
from .paths import ROOT,OLD
from .io import load,save,sha256,digest,space
from .fixed_confirmation_inventory import LOCAL

def build():
    inventory=load(LOCAL/'inventory.json');closed=load(ROOT/'analysis/EXPLORATION_SEARCH_CLOSED_2026-10-27.json')
    assert not closed['new_source_search_authorized'] and closed['exploration_loops_consumed']==2
    p74=load(ROOT/'runs/N74-ACTIVITY-FINITE-v1/protocol.json');cells=[];endpoints=[];bindings=dict(inventory['bindings'])
    bindings.update(p74['bindings'])
    for f in [Path(__file__),LOCAL/'inventory.json',ROOT/'EXPLORATION_CLOSEOUT_DIRECTIVE_2026-10-27.md',
        ROOT/'analysis/EXPLORATION_SEARCH_CLOSED_2026-10-27.json',ROOT/'analysis/FIXED_CONFIRMATION_AND_CONCLUSION_PLAN_2026-10-27_JA.md']:
        bindings[str(f)]=sha256(f.read_bytes())
    for c in p74['cases']:
        bindings[c['integrals']]=c['integrals_SHA256'];bindings[c['meta']]=c['meta_SHA256']
        f=OLD/'data/evaluation-only'/(c['case']+'.json');bindings[str(f)]=sha256(f.read_bytes())
    for r in inventory['rows']:
        meta=load(next(c['meta'] for c in p74['cases'] if c['case']==r['case']));missing=r['status']=='MISSING_SAME_PROBLEM_SOURCE';classical=r['method']=='CLASSICAL_HF_HCI'
        pilot=int(r.get('historical_construction_model_shots',0));campaign=0 if classical or missing else 200000
        final=campaign-pilot if campaign else 0;assert final>=0 and final%2==0
        cell=dict(case=r['case'],method=r['method'],problem_id=meta['problem_id'],source_input=r,
            status='MISSING_SAME_PROBLEM_SOURCE' if missing else 'FROZEN_NOT_STARTED',
            campaign_model_cost_including_one_off_pilot=campaign,historical_one_off_pilot_model_cost=pilot,
            planned_new_model_samples=final,planned_final_model_samples_per_replicate=final//2,
            final_model_samples_are_new_not_old_subsets=True,new_QPU_shots=0,
            independent_parameter_rank=None if not classical else 'NOT_APPLICABLE',
            fixed_source_only_not_new_adaptive_construction=True)
        cells.append(cell)
        for rep in [760101,760102]:
            for label,R in [('R8',8),('R16',16),('R32',32),('R64',64),('COMMON_ACTUAL_R32',32)]:
                row=dict(case=r['case'],method=r['method'],replicate=rep,context=label,R_cap=R,
                    status='MISSING_SAME_PROBLEM_SOURCE' if missing else 'FROZEN_NOT_STARTED')
                row['endpoint_id']=digest(row);endpoints.append(row)
    assert len(cells)==32 and len(endpoints)==320 and len({r['endpoint_id'] for r in endpoints})==320
    assert sum(c['campaign_model_cost_including_one_off_pilot'] for c in cells)==5200000
    assert sum(c['planned_new_model_samples'] for c in cells)==5146624
    assert sum(c['historical_one_off_pilot_model_cost'] for c in cells)==53376
    assert sum(e['status']=='MISSING_SAME_PROBLEM_SOURCE' for e in endpoints)==20
    for f,h in bindings.items():assert sha256(Path(f).read_bytes())==h,'Frozen confirmation input drift '+f
    return dict(version='FIXED_CONFIRMATION_SCIENCE_V1_NO_NEW_SOURCE_SEARCH',bindings=bindings,cases=p74['cases'],cells=cells,endpoints=endpoints,
        proposals=inventory['proposals'],maximum_proposals=2,replicates=[760101,760102],
        new_source_search_authorized=False,source_policy_or_case_specific_fallback_changes_authorized=False,
        extra_human_approval_required=False,execution_binding_may_be_created_autonomously=True,
        runner_status='PENDING_IMPLEMENTATION_AND_BINDING_NOT_A_SCIENTIFIC_NO_GO',
        seed_rule='SHA256([CONFIRM20261027,problem_id,method,member_id,replicate])_FIRST32_BITS',
        common_equation=p74['common_equation'],library=p74['library'],library_SHA256=p74['library_SHA256'],
        CI=dict(R_caps=[8,16,32,64],primary_R_cap=32,include_all_method_common_actual_R=True,spin_penalty_Ha=.1,OO=False,
            CPU_cap=600,wall_cap=900,root_wall_cap=30,triangle_cap=10000000,R128_not_in_this_confirmation=True),
        HCI=dict(native_wall_cap_per_case=60,memory_bytes=6*2**30,use_existing_qualified_native_parameters=True,deterministic_root_not_independent_two_replicates=True),
        source=dict(M4_CPU_cap=300,M4_wall_cap=600,max_own_one_A100_jobs=2,A100_lane_wall_seconds=900,refresh_remote_identity_and_queue_before_dispatch=True),
        compiler=dict(common_opt_levels=[1,3],preserve_source_native_exact_portfolio=True,additional_CPU_cap=30,additional_wall_cap=90,missing_qualification_is_missing_not_source_substitution=True),
        storage=dict(new_bytes=24*2**20,root_logical_bytes=2**30,free_floor_bytes=3*2**30),
        confirmation_scope='NEW_INDEPENDENT_READOUT_DATA_CONDITIONAL_ON_FROZEN_SOURCE_NOT_NEW_ADAPTIVE_CONSTRUCTION_OR_SOURCE_ENSEMBLES',
        confidence_rule='TWO_REPLICATES_DESCRIPTIVE_MEAN_RANGE_AND_ALL_POINTS_NO_PVALUE_SUPERIORITY',
        stronger_paper_highest_budget_reproduction=False,goal_complete=False)

def run():
    space();target=LOCAL/'confirmation-science-freeze-v1.json'
    if target.exists():raise RuntimeError('Existing confirmation science freeze; no replace or seed extension')
    for name in ['confirmation-started.json','execution-binding.json']:
        if (LOCAL/name).exists():raise RuntimeError('Unexpected prior execution; reconcile, no new freeze')
    import xml.etree.ElementTree as ET
    test=ROOT/'qualification-fixed-confirmation-freeze-v1.xml';suites=list(ET.parse(test).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==2 and not any(int(s.get(k,0)) for s in suites for k in ('failures','errors'))
    value=build();value['bindings'][str(test)]=sha256(test.read_bytes())
    source=ROOT/'tests/test_fixed_confirmation_freeze.py';value['bindings'][str(source)]=sha256(source.read_bytes())
    save(target,value)
    print('CONFIRMATION_SCIENCE_FROZEN32_CELLS320_FIELDS_5146624_NEW_MODEL_NO_EXECUTION_YET',flush=True)

if __name__=='__main__':run()
