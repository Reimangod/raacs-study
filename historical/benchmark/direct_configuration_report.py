"""Additive corrected report; verify roots with pre-existing sparse-H entries.

The numerical producer is frozen. Its legacy CSV 'same_final_dimension=False'
means matching was not enforced, but is misleading as an actual-d indicator.
This successor independently computes actual equality and support inclusion.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n'


def pair_fields(case,d,q):
    ds=set(d['history'][-1]['selected_ids']);qs=set(q['final_selected_ids'])
    if len(ds)!=d['final_dimension'] or len(qs)!=q['actual_final_CI_dimension']:
        raise ValueError('Saved dimension/ID mismatch')
    return dict(case=case,RAACS_seed=q['algorithm']['seed'],classical_cost_group=case,
        direct_energy_Ha=d['final_energy_Ha'],RAACS_energy_Ha=q['final_QSCI_energy_Ha'],
        direct_minus_RAACS_mHa=1000*(d['final_energy_Ha']-q['final_QSCI_energy_Ha']),
        direct_d=len(ds),RAACS_d=len(qs),equal_actual_dimension=len(ds)==len(qs),
        identical_selected_space=ds==qs,RAACS_space_subset_of_direct=qs<=ds,
        intersecting_configurations=len(ds&qs),common_R_cap=64,
        same_final_dimension_enforced=False,work_matched=False,
        direct_quantum_shots=0,RAACS_historical_total_model_shots=q['actual_total_model_shots'],
        classical_CNOT=None,classical_depth=None,ground_state_error_Ha=None,exploratory=True)


def build(output):
    import numpy as np
    from scipy.sparse import load_npz
    from threadpoolctl import threadpool_limits
    output=Path(output).resolve();cfg=json.loads((output/'manifest.json').read_text())
    if any(sha(p)!=h for p,h in cfg['inputs'].items()):raise ValueError('Frozen inputs/producer drift')
    inputs=dict(cfg['inputs']);inputs[str(output/'manifest.json')]=sha(output/'manifest.json')
    inputs[str(Path(__file__))]=sha(__file__)
    rows=[];costs=[];checks=0;maximum=0.;began=perf_counter()
    with threadpool_limits(limits=1):
        for item in cfg['cases']:
            case=item['case'];path=output/case/'result.json';d=json.loads(path.read_text());inputs[str(path)]=sha(path)
            if d['status']!='CLASSICAL_DIRECT_CONFIGURATION_FORK_EXECUTED' or d['new_quantum_samples']!=0:
                raise ValueError('Incomplete or unexpected fork')
            if d['manifest_sha256']!=sha(output/'manifest.json'):raise ValueError('Result binding drift')
            hpath=Path(item['physical'])/'hamiltonian.npz';inputs[str(hpath)]=sha(hpath)
            h=load_npz(hpath)  # Existing <=16q evaluation artifact, never new full-H construction.
            def check(ids,energy):
                nonlocal checks,maximum
                if not ids or len(ids)>64 or len(ids)!=len(set(ids)):raise ValueError('Selected-space bound')
                independent=float(np.linalg.eigvalsh(h[ids][:,ids].toarray())[0]);checks+=1
                gap=abs(independent-energy);maximum=max(maximum,gap)
                if gap>1e-8:raise ValueError('Independent saved sparse-H energy parity')
            previous=float('inf')
            for step in d['history']:
                check(step['selected_ids'],step['energy_Ha'])
                if step['energy_Ha']>previous+1e-8:raise ValueError('Nested energy increased')
                previous=step['energy_Ha']
            bankpath=output/case/'initial-common-information.json';inputs[str(bankpath)]=sha(bankpath)
            if inputs[str(bankpath)]!=d['initial_common_information_sha256']:raise ValueError('Initial fork drift')
            for link in d['source_RAACS']:
                qp=Path(link['source']);q=json.loads(qp.read_text());inputs[str(qp)]=sha(qp)
                if inputs[str(qp)]!=link['source_sha256'] or q['problem']!=d['problem']:raise ValueError('Linked quantum problem drift')
                check(q['final_selected_ids'],q['final_QSCI_energy_Ha'])
                rows.append(pair_fields(case,d,q))
            costs.append(dict(case=case,classical_cost_group=case,
                direct_stop_reason=d['stop_reason'],direct_d=d['final_dimension'],
                direct_named_H_including_producer_rechecks=d['actual_named_H_requests_including_rechecks'],
                direct_invocation_seconds=d['new_classical_invocation_seconds'],
                source_load_seconds=d['source_load_seconds'],direct_peak_RSS=d['peak_process_RSS'],
                process_startup_seconds=d['process_startup_seconds'],
                direct_quantum_shots=0,classical_cost_is_shared_across_three_quantum_seed_rows=True,
                same_small_sector_covered=d['selected_space_covers_small_sector'],controlled_speedup=False))
            del h
    if len(rows)!=3*len(cfg['cases']) or any(sha(p)!=value for p,value in inputs.items()):raise ValueError('Coverage/source drift')
    prefix=output/'comparison-H4-prefix.csv'
    if prefix.exists() and not (output/'comparison.csv').read_bytes().startswith(prefix.read_bytes()):
        raise ValueError('Original H4 partial CSV was altered')
    def csvtext(values):
        stream=StringIO();w=csv.DictWriter(stream,fieldnames=list(values[0]));w.writeheader();w.writerows(values);return stream.getvalue()
    summary=dict(status='ALL_THREE_FORKS_AND_NINE_REUSED_RAACS_SELECTED_ENERGIES_INDEPENDENTLY_CHECKED',
        input_sha256=inputs,classical_executions=len(costs),linked_quantum_trajectories=len(rows),
        actual_equal_dimension_pairs=sum(r['equal_actual_dimension'] for r in rows),
        identical_selected_space_pairs=sum(r['identical_selected_space'] for r in rows),
        RAACS_spaces_contained_in_direct=sum(r['RAACS_space_subset_of_direct'] for r in rows),
        independent_old_sparse_H_projected_checks=checks,maximum_energy_difference_Ha=maximum,
        review_seconds=perf_counter()-began,new_quantum_samples=0,new_FCI_reference_calculations=0,
        legacy_dimension_flag_is_not_actual_equality=True,whole_goal_complete=False)
    lines=['# Common-initial-information direct-CI / RAACS diagnostic','',
        'Exploratory three previously studied conditions. Classical branch computed once per condition; '
        'nine quantum trajectories are REUSED, not nine new quantum experiments.','',
        'The legacy parent CSV hardcoded `same_final_dimension=False` to indicate no forced matching; '
        'that is not an actual equality test. Use this corrected additive `equal_actual_dimension` field. '
        'Original numerical results, producer code and partial CSV are preserved.','',
        '|Case|Direct d|RAACS d (seed101/211/307)|Direct minus RAACS mHa (same order)|','|---|---:|---|---|']
    for item in cfg['cases']:
        part=[r for r in rows if r['case']==item['case']]
        lines.append('|'+item['case']+'|'+str(part[0]['direct_d'])+'|'+', '.join(str(r['RAACS_d']) for r in part)+'|'+', '.join(f"{r['direct_minus_RAACS_mHa']:.6g}" for r in part)+'|')
    lines+=['','Negative differences favor the direct classical selected-CI branch. '
        'Unequal d is exposed, not hidden or forced to match. Same d/space is a descriptive fact, '
        'not matched work, a new control selected prospectively, or quantum advantage.',
        'The identical initial HF and frozen local proposal inputs are replay-qualified against saved '
        'initial pilots/scopes. Original quantum runs did not log an independent full candidate-ID hash. '
        'After the fork, different trajectories and spaces are allowed.',
        'Classical direct addition may explain energy improvements obtainable without quantum sampling '
        'in these conditions. This limits a claim that inserting quantum sampling was necessary here; '
        'it does not establish general classical/quantum dominance or invalidate the rest of the study.',
        'Unknown process startup remains null. New classical seconds and inherited RAACS seconds have '
        'different boundaries and simulation overheads: no controlled speedup. Classical CNOT/depth are '
        'N/A, not a free quantum preparation. External residuals are screened/capped, not convergence proofs.',
        f'Independent old sparse-H selected-space rechecks: {checks}; maximum discrepancy {maximum:.4g} Ha. '
        'No new full Hamiltonian/FCI reference, quantum draws, optimizer or large statevectors. '
        'Validation time/calls are separate from the production fork costs. Whole study remains incomplete.']
    dest=output/'report-v2';dest.mkdir(exist_ok=True)
    texts={'comparison-corrected.csv':csvtext(rows),'classical-costs-once-per-case.csv':csvtext(costs),
        'REPORT.md':'\n'.join(lines)+'\n','validation.json':canonical(summary)}
    # Measured review time is run-specific; never silently overwrite a previous report.
    if any((dest/name).exists() for name in texts):raise ValueError('Report already exists; preserve it and use a successor for a new review')
    for name,value in texts.items():(dest/name).write_bytes(value.encode())
    return {k:v for k,v in summary.items() if k!='input_sha256'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'results/RAACS-direct-configuration-fork-v1')
    print(canonical(build(p.parse_args().root)))
