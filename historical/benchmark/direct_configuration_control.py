"""Three exploratory direct-CI forks; reuse nine immutable RAACS trajectories.

Same initial information/local proposal kernel; independent post-fork spaces.
Not optimized HCI, matched-work superiority, or new quantum execution.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import sqlite3
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
CASES=('H4_R1.1','LiH_R1.5','N2_R1.4')
SEEDS=(101,211,307)
SPEC=ROOT/'study-v1/OBJECTIVE_ADDITION_SCOPE.md'
MODULE_ENTRY_TIME=perf_counter()


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(x): return json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n'
def require(ok,message):
    if not ok:raise ValueError(message)
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(canonical(value));temp.replace(path)
def immutable(path,value):
    if path.exists():require(json.loads(path.read_text())==value,'Immutable binding drift')
    else:save(path,value)
def rss():
    import sys
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def ranked_direct_targets(proposals,external,selected,cap):
    """All eligible bank targets can be added; no arbitrary two-target handicap."""
    require(type(cap) is int and 1<=cap<=64 and len(set(selected))==len(selected)<=cap,'CI cap/IDs')
    union={int(x) for p in proposals for x in p['proposal_target_ids']}-set(selected)
    scores={int(x):float(v) for x,v in zip(external['ids'],external['en_energy_importance'])}
    require(len(scores)==len(external['ids']),'Duplicate external ID')
    require(union<=set(scores),'Operator target absent from external information')
    require(all(math.isfinite(v) and v>=0 for v in scores.values()),'Invalid importance')
    return sorted((x for x in union if scores[x]>0),key=lambda x:(-scores[x],x))[:cap-len(selected)]


def source_paths(case,seed):
    return ROOT/'results/RAACS-standard-v1/cells'/case/f'full_cost-qsci_local_grid-last_only-s{seed}-exact_small'


def prepare(output):
    output=Path(output).resolve()
    require(not any(output.is_relative_to(ROOT/'results'/name) for name in ('standard-v1','RAACS-standard-v1')),
            'Never write inside immutable source datasets')
    output.mkdir(parents=True,exist_ok=True)
    bindings={};cases=[]
    for case in CASES:
        algorithms=[]
        for seed in SEEDS:
            folder=source_paths(case,seed)
            for name in ('config.json','result.json','events.sqlite','input-provenance.json'):
                bindings[str(folder/name)]=sha(folder/name)
            cfg=json.loads((folder/'config.json').read_text());r=json.loads((folder/'result.json').read_text())
            require(r['status']=='COMPLETED' and r['event_database_sha256']==sha(folder/'events.sqlite'),'Unqualified source')
            require(r['algorithm']==cfg['algorithm'] and r['problem']==cfg['problem'],'Source result/config mismatch')
            require(r['FCI_used'] is False,'FCI source policy')
            algorithm=dict(cfg['algorithm']);algorithm.pop('seed');algorithms.append(algorithm)
            for name,value in cfg['implementation'].items():
                p=ROOT/'vendor/ceo/adaptvqe/circuits.py' if name=='circuits.py' else ROOT/'benchmark'/name
                require(sha(p)==value,'Frozen source kernel changed: '+name)
                bindings[str(p)]=value
        require(all(a==algorithms[0] for a in algorithms),'Source seed settings differ')
        require((algorithms[0]['CI_cap'],algorithms[0]['max_steps'],algorithms[0]['parent_connection_cap'],
            algorithms[0]['target_cap'],algorithms[0]['candidate_bank_cap'],algorithms[0]['maximum_named_H_requests'])
            ==(64,8,128,128,32,20000000),'Unregistered source bounds')
        physical=ROOT/'results/standard-v1/references'/case
        for name in ('reference.json','reference.npz'):bindings[str(physical/name)]=sha(physical/name)
        cases.append(dict(case=case,physical=str(physical),algorithm=algorithms[0]))
    bindings[str(SPEC)]=sha(SPEC);bindings[str(Path(__file__))]=sha(__file__)
    cfg=dict(schema='direct-configuration-fork-v1',cases=cases,source_seeds=list(SEEDS),inputs=bindings,
        free_floor_bytes=int(2.75*2**30),rss_cap_bytes=2*2**30,case_seconds=120,
        selection='ALL_POSITIVE_EN_TARGETS_IN_SAME_LOCAL_BANK_UNTIL_R_CAP',
        timing_scope='NEW_CLASSICAL_GIVEN_INTEGRALS_FORK_VS_LINKED_HISTORICAL_QUANTUM;NOT_CONTROLLED_SPEEDUP',
        exploratory_previously_studied_conditions=True,new_quantum_acquisitions=0,
        same_final_dimension_or_work_required=False,whole_goal_complete=False)
    immutable(output/'manifest.json',cfg)
    return cfg


def solve(backend,ids):
    import numpy as np
    require(len(ids)<=64 and len(ids)==len(set(ids)),'Bounded unique selected space')
    tick=perf_counter();matrix=backend.matrix(ids,ids);values,vectors=np.linalg.eigh(matrix)
    energy=float(values[0]);coeff=vectors[:,0]
    residual=float(np.linalg.norm(matrix@coeff-energy*coeff))
    require(residual<1e-8,'Internal Ritz residual')
    return dict(selected_ids=list(ids),coefficients=coeff,energy_Ha=energy,
        actual_dimension=len(ids),internal_Ritz_residual=residual,solve_seconds=perf_counter()-tick)


def initial_evidence(case,cfg,info,ci,scope):
    from .local_sampler_primitives import unpack_counts64
    linked=[]
    for seed in SEEDS:
        folder=source_paths(case,seed);source=json.loads((folder/'result.json').read_text())
        require(source['problem']==info,'Physical problem mismatch')
        with sqlite3.connect((folder/'events.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
            rows=list(db.execute('SELECT role,raw_shots,counts64 FROM draws WHERE role="initial_pilot"'))
            require(len(rows)==1,'Initial pilot missing/ambiguous')
            ids,counts=unpack_counts64(rows[0][2]);observed={int(x):int(n) for x,n in zip(ids,counts)}
            require(observed=={info['hf_id']:cfg['probe_shots']} and rows[0][1]==cfg['probe_shots'],'Initial CI not common HF')
            decision=db.execute('SELECT metadata FROM decisions WHERE step=1').fetchone()
            require(decision is not None,'Source has no first decision')
            require(json.loads(decision[0])['proposal_scope']==scope,'Initial proposal scope mismatch')
            resources=[]
            for allocation in source['final_circuit_allocations']:
                if allocation['shots']:
                    value=db.execute('SELECT resources FROM circuits WHERE id=?',(allocation['circuit_id'],)).fetchone()
                    require(value is not None,'Missing source final circuit');resources.append(json.loads(value[0]))
        require(abs(ci['energy_Ha']-info['HF_energy_Ha'])<1e-8,'HF initial energy mismatch')
        linked.append(dict(seed=seed,source=str(folder/'result.json'),source_sha256=sha(folder/'result.json'),
            native_final_energy_Ha=source['final_QSCI_energy_Ha'],native_final_dimension=source['actual_final_CI_dimension'],
            historical_total_model_shots=source['actual_total_model_shots'],historical_pilot_model_shots=source['pilot_raw_model_shots'],
            historical_named_H_requests=source['classical_named_H_requests'],historical_wall_seconds=source['actual_wall_seconds'],
            final_resources=resources,initial_HF_pilot_verified=True))
    return linked


def execute_case(output,configuration,item):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .resource_adaptive_sampler import integral_problem
    from .local_sampler_primitives import local_proposals,valid_sector
    from .integral_connections import IntegralConnections
    from .molecular_mps_scaling import no_full_state_guards
    output=Path(output);case=item['case'];folder=output/case;folder.mkdir(exist_ok=True)
    with (folder/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result_path=folder/'result.json'
        if result_path.exists():
            r=json.loads(result_path.read_text());require(r['manifest_sha256']==sha(output/'manifest.json'),'Result binding drift')
            return r
        require(not (folder/'started.json').exists(),'Partial attempt preserved; no free automatic retry')
        require(all(sha(p)==value for p,value in configuration['inputs'].items()),'Source/code drift')
        # Cost coordinator has not started while F is live. Refuse if its real numerical child is running.
        cost_path=ROOT/'results/total-workflow-sequence-v1/execution-status.json'
        if cost_path.exists():
            cost=json.loads(cost_path.read_text())
            require(cost['status'] in ('WAITING_ON_VERIFIED_NUMERICAL_DEPENDENCIES','WORKFLOW_TOTAL_COST_COHORTS_AUDITED_REPORTED'),
                    'Do not overlap isolated total-cost timing')
        require(shutil.disk_usage(ROOT).free>=configuration['free_floor_bytes'],'Capacity boundary')
        save(folder/'started.json',dict(manifest_sha256=sha(output/'manifest.json'),case=case))
        started=perf_counter();history=[]
        try:
            class CapturedConnections(IntegralConnections):
                def external(self,*args,**kwargs):
                    self.last=super().external(*args,**kwargs);return self.last
            with threadpool_limits(limits=1),no_full_state_guards():
                tick=perf_counter();backend,info=integral_problem(item['physical']);setup=perf_counter()-tick
                a=item['algorithm'];connections=CapturedConnections(backend,parent_cap=a['parent_connection_cap'])
                current=solve(backend,[info['hf_id']]);history.append(dict(iteration=0,**{k:v.tolist() if hasattr(v,'tolist') else v for k,v in current.items()}))
                linked=None;stop='EXPANSION_CAP';first_bank=None
                for iteration in range(1,a['max_steps']+1):
                    if perf_counter()-started>configuration['case_seconds']:stop='WALL_CAP_AT_ITERATION_BOUNDARY';break
                    if rss()>configuration['rss_cap_bytes']:stop='RSS_CAP_AT_ITERATION_BOUNDARY';break
                    if backend.matrix_entry_requests>=a['maximum_named_H_requests']:stop='NAMED_H_CAP_AT_ITERATION_BOUNDARY';break
                    if current['actual_dimension']>=a['CI_cap']:stop='DIMENSION_CAP';break
                    tick=perf_counter();proposals,scope=local_proposals(backend,connections,current,a['target_cap'],a['candidate_bank_cap'])
                    proposal_seconds=perf_counter()-tick;external=connections.last
                    if iteration==1:
                        linked=initial_evidence(case,a,info,current,scope)
                        first_bank=dict(selected_ids=current['selected_ids'],scope=scope,
                            operators=[dict(source=list(p['operator'].source),target=list(p['operator'].target),
                                proposal_target_ids=p['proposal_target_ids']) for p in proposals],
                            external_ids=list(map(int,external['ids'])),EN_importance=list(map(float,external['en_energy_importance'])),
                            proof_scope='REPLAY_SAME_FROZEN_KERNEL_INPUTS_AND_SAVED_FIRST_SCOPE;NOT_ORIGINAL_STORED_TARGET_HASH')
                        save(folder/'initial-common-information.json',first_bank)
                    additions=ranked_direct_targets(proposals,external,current['selected_ids'],a['CI_cap'])
                    if not additions:stop='NO_POSITIVE_TARGET_IN_BOUNDED_LOCAL_BANK';break
                    ids=sorted(current['selected_ids']+additions)
                    require(all(valid_sector(x,info['n_qubits'],info['nalpha'],info['nbeta']) for x in ids),'Candidate sector mismatch')
                    previous=current['energy_Ha'];current=solve(backend,ids)
                    require(current['energy_Ha']<=previous+1e-8,'Nested-space variational inequality')
                    history.append(dict(iteration=iteration,added_ids=additions,proposal_scope=scope,
                        external_residual_norm_of_returned_terms=external['external_residual_norm_of_returned_terms'],
                        external_residual_scope=external['residual_scope'],proposal_seconds=proposal_seconds,
                        named_H_requests_cumulative=backend.matrix_entry_requests,
                        **{k:v.tolist() if hasattr(v,'tolist') else v for k,v in current.items()}))
                    save(folder/'checkpoint.json',dict(history=history,whole_goal_complete=False))
                    if current['actual_dimension']>=a['CI_cap']:stop='DIMENSION_CAP';break
                require(linked is not None,'Initial fork not qualified')
                # Recheck every saved projected energy with a separate eigvalsh call, charged separately.
                tick=perf_counter();delta=0.
                for row in history:
                    energy=float(np.linalg.eigvalsh(backend.matrix(row['selected_ids'],row['selected_ids']))[0])
                    delta=max(delta,abs(energy-row['energy_Ha']))
                review_seconds=perf_counter()-tick;require(delta<1e-8,'Saved-space energy recheck')
                sector=math.comb(info['n_qubits']//2,info['nalpha'])*math.comb(info['n_qubits']//2,info['nbeta'])
                result=dict(status='CLASSICAL_DIRECT_CONFIGURATION_FORK_EXECUTED',case=case,
                    manifest_sha256=sha(output/'manifest.json'),problem=info,stop_reason=stop,history=history,
                    source_RAACS=linked,initial_common_information_sha256=sha(folder/'initial-common-information.json'),
                    final_energy_Ha=current['energy_Ha'],final_dimension=current['actual_dimension'],
                    small_sector_dimension=sector,selected_space_covers_small_sector=current['actual_dimension']==sector,
                    actual_named_H_requests_including_rechecks=backend.matrix_entry_requests,
                    connection_statistics=asdict(connections.statistics),source_load_seconds=setup,
                    new_classical_invocation_seconds=perf_counter()-started,energy_recheck_seconds=review_seconds,
                    process_startup_seconds=None,module_entry_to_result_elapsed_seconds=perf_counter()-MODULE_ENTRY_TIME,
                    classical_timer_boundary='AFTER_NUMERICAL_IMPORTS_AND_SOURCE_BINDING_BEFORE_PHYSICAL_LOAD;NOT_FRESH_PROCESS_TOTAL',
                    new_projected_solver_calls=len(history),new_energy_recheck_calls=len(history),
                    maximum_saved_energy_disagreement_Ha=delta,peak_process_RSS=rss(),
                    new_quantum_samples=0,hardware_shots=0,new_FCI_reference_calculations=0,
                    classical_CNOT=None,classical_CNOT_depth=None,ground_state_error_Ha=None,
                    controlled_speedup_established=False,same_final_dimension_or_work=False,
                    whole_goal_complete=False)
            require(all(sha(p)==value for p,value in configuration['inputs'].items()),'Source changed during fork')
            save(result_path,result);return result
        except Exception as exc:
            save(folder/'failure.json',dict(status='FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',error=repr(exc),history=history))
            raise


def report(output):
    import csv
    output=Path(output);rows=[];statuses=[]
    for case in CASES:
        path=output/case/'result.json'
        if not path.exists():statuses.append(dict(case=case,status='NOT_EXECUTED_OR_FAILURE_PRESERVED'));continue
        r=json.loads(path.read_text());statuses.append(dict(case=case,status=r['status'],classical_cost_group=case))
        for q in r['source_RAACS']:
            rows.append(dict(case=case,RAACS_seed=q['seed'],classical_cost_group=case,
                direct_energy_Ha=r['final_energy_Ha'],RAACS_energy_Ha=q['native_final_energy_Ha'],
                direct_minus_RAACS_mHa=1000*(r['final_energy_Ha']-q['native_final_energy_Ha']),
                direct_d=r['final_dimension'],RAACS_d=q['native_final_dimension'],common_R_cap=64,
                RAACS_historical_total_model_shots=q['historical_total_model_shots'],
                same_final_dimension=False,work_matched=False,exploratory=True))
    if rows:
        with (output/'comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(output/'status.json',dict(cases=statuses,available_display_pairs=len(rows),
        classical_executions=sum(r['status']=='CLASSICAL_DIRECT_CONFIGURATION_FORK_EXECUTED' for r in statuses),
        repeated_RAACS_seeds_are_not_independent_classical_replicates=True,whole_goal_complete=False))
    return statuses


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/RAACS-direct-configuration-fork-v1')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--case',choices=CASES);p.add_argument('--all',action='store_true')
    a=p.parse_args();cfg=prepare(a.output)
    if a.prepare_only:print(canonical(dict(prepared_cases=list(CASES),new_outcomes=0,manifest_sha256=sha(a.output/'manifest.json'))))
    else:
        require(bool(a.case) != bool(a.all),'Choose --case or --all')
        for item in cfg['cases']:
            if a.all or item['case']==a.case:
                r=execute_case(a.output,cfg,item);print(item['case'],r['status'],r['final_dimension'],flush=True)
        print(canonical(report(a.output)))
