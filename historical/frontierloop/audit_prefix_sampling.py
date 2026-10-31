"""Bind N17 counts, declared coordinates and every compiler control point.

This is a sampling-stage inventory, NOT a molecular energy result. Completed
counts are actually rechecked, while pending/missing rows stay explicitly so.
"""
import csv
from collections import Counter
from io import StringIO
from pathlib import Path
from time import process_time,perf_counter
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from .prefix_performance_sampling import LOCAL,N15,db,verified_protocol,decode


def parameter_row(cell,bindings):
    policy='STATE_OVERLAP_ROUND' if cell['method']=='ORIGINAL' else cell['method']
    choices=[r for r in bindings if (r['case'],r['method'],r['policy'],r['tier'])==
             (cell['case'],cell['source'],policy,.01)]
    if len(choices)!=1:raise ValueError('Unique registered parameter family required')
    row=choices[0]
    return row['matched_compiled_parent_declared_coordinates'] if cell['method']=='ORIGINAL' else row['rounded_declared_coordinates']


def all_control_outputs(case,source,raw):
    # Some sources already have a rounded candidate at the smallest tier.
    # Therefore use the separately audited full EXACT portfolio, not candidate
    # rows as an incomplete proxy for its opt1/opt3 controls.
    portfolio=load(ROOT/'analysis/N15-control-portfolio-audit.json')
    if portfolio['original_result_SHA256']!=sha256((N15/'result.json').read_bytes()):
        raise ValueError('Original portfolio/result identity mismatch')
    matches=[r['all_exact_controls'] for r in portfolio['rows'] if (r['case'],r['method'])==(case,source)]
    if not matches or any(c!=matches[0] for c in matches):raise ValueError('Inconsistent exact source portfolio')
    result=[dict(resources=o['resources'],
        preparation=dict(path=o['preparation'],SHA256=o['preparation_SHA256']),
        sampling=dict(path=o['sampling'],SHA256=o['sampling_SHA256'])) for o in matches[0]]
    if len(result)!=2 or [o['resources']['optimization_level'] for o in result]!=[1,3]:
        raise ValueError('Both actual source opt1/opt3 controls required')
    for output in result:
        for role in ['preparation','sampling']:
            if sha256((N15/output[role]['path']).read_bytes())!=output[role]['SHA256']:
                raise ValueError('Original compiler resource evidence changed')
        from .prefix_portfolio_audit import counters
        measured=counters(N15/output['sampling']['path'])
        if any(output['resources'][k]!=v for k,v in measured.items()):
            raise ValueError('Direct actual compiler control recount mismatch')
    return result


def run():
    cpu,tick=process_time(),perf_counter();spec=verified_protocol()
    binding_file=ROOT/'analysis/N15-declared-coordinate-bindings.json'
    if sha256(binding_file.read_bytes())!=spec['parameters_binding_SHA256']:raise RuntimeError('Registered parameter bindings changed')
    bindings=load(binding_file)['rows'];raw=load(N15/'result.json')['rows']
    with db() as c:
        c.row_factory=__import__('sqlite3').Row
        draws={r['draw_id']:dict(r) for r in c.execute('SELECT * FROM draws')}
        cells={r['cell_id']:dict(r) for r in c.execute('SELECT * FROM cells')}
    circuits={c['circuit_id']:c for c in spec['circuits']};rechecked={};controls={};rows=[]
    actual_sim_CPU=0.;actual_sim_wall=0.;completed_circuits=0
    for circuit in spec['circuits']:
        base=LOCAL if circuit['backend']=='M4' else LOCAL/'collected'
        output=base/'circuits'/circuit['circuit_id']/'output'
        if not (output/'result.json').exists():continue
        result=load(output/'result.json')
        if result['input_id']!=circuit['input_id'] or result['source_QPY_SHA256']!=circuit['original_QPY_SHA256']:
            raise RuntimeError('Wrong actual sampler result')
        actual_sim_CPU+=result['whole_body_CPU_seconds'];actual_sim_wall+=result['whole_body_wall_seconds'];completed_circuits+=1
        for request in circuit['sampling_requests']:
            draw=draws[request['id']]
            if draw['status']!='COMPLETED':continue
            physical=output/(request['id']+'.npz');logical=LOCAL/'decoded'/(request['id']+'.npz')
            evidence=load(logical.with_suffix('.json'))
            if sha256(physical.read_bytes())!=draw['raw_sha'] or sha256(logical.read_bytes())!=draw['decoded_sha']:
                raise RuntimeError('Raw or decoded actual counts changed')
            if evidence['result_SHA256']!=sha256((output/'result.json').read_bytes()) or evidence['decoder']!=circuit['decoder'] or evidence['draw_id']!=request['id']:
                raise RuntimeError('Decoder/evaluation evidence binding failed')
            with np.load(physical,allow_pickle=False) as a:pids=a['blocked_ids'];pc=a['counts']
            ids=decode(pids,circuit['decoder'],circuit['norb'],circuit['nelec']);order=np.argsort(ids)
            with np.load(logical,allow_pickle=False) as a:
                if not np.array_equal(a['blocked_ids'],ids[order]) or not np.array_equal(a['counts'],pc[order]):
                    raise RuntimeError('Decoded counts do not equal the registered wire map')
            actual=next(r for r in result['sampling_results'] if r['id']==request['id'])
            if any(actual[k]!=v for k,v in request.items()) or actual['actual_model_shots']!=100000 or int(pc.sum())!=100000 or draw['actual_shots']!=100000:
                raise RuntimeError('Actual Shot/RNG accounting differs')
            rechecked[request['id']]=dict(raw_SHA256=draw['raw_sha'],decoded_SHA256=draw['decoded_sha'],actual_shots=100000)
    for cell in spec['cells']:
        ledger=cells[cell['cell_id']]
        row=dict(case=cell['case'],source=cell['source'],method=cell['method'],replicate=cell['replicate'],
            status=ledger['status'],cell_id=cell['cell_id'],draw_id=cell.get('draw_id'),
            energy_Ha=None,recovery_pct=None,energy_missing_reason='CI_NOT_STARTED',actual_QPU_shots=0)
        if cell['status']=='MISSING_SAME_PROBLEM_SOURCE':
            if ledger['status']!=cell['status'] or ledger['draw_id'] is not None:raise RuntimeError('Missing source fabricated')
            row.update(energy_missing_reason='MISSING_SAME_PROBLEM_SOURCE',CNOT=None,CNOT_Depth=None,Total_Depth=None,declared_parameters=None)
        else:
            if ledger['draw_id']!=cell['draw_id']:raise RuntimeError('Logical cell/draw link changed')
            if ledger['status']=='COMPLETED' and cell['draw_id'] not in rechecked:raise RuntimeError('Completed cell has no actual verified counts')
            circuit=circuits[cell['circuit_id']];key=(cell['case'],cell['source'])
            if key not in controls:controls[key]=all_control_outputs(*key,raw)
            resource=cell['selected_resources']
            row.update(CNOT=resource['CNOT'],CNOT_Depth=resource['CNOT_Depth'],Total_Depth=resource['Total_Depth'],
                declared_parameters=parameter_row(cell,bindings),backend=circuit['backend'],
                preparation_infidelity=cell['preparation_infidelity'],
                original_opt1_resources=controls[key][0]['resources'],original_opt3_resources=controls[key][1]['resources'],
                actual_sample_shots=100000 if cell['draw_id'] in rechecked else None,
                not_independent_method_draw=sum(c.get('draw_id')==cell['draw_id'] for c in spec['cells'])>1)
        rows.append(row)
    if len(rows)!=108 or set(cells)!={c['cell_id'] for c in spec['cells']}:
        raise RuntimeError('All 108 logical states required')
    ledger_complete={k for k,v in draws.items() if v['status']=='COMPLETED'}
    if ledger_complete!=set(rechecked):raise RuntimeError('Unattributed completed physical draws')
    result=dict(status='N17_SAMPLING_STAGE_ALL108_LOGICAL_STATES_INVENTORIED_NOT_ENERGY_RESULTS',rows=rows,
        logical_status_counts=dict(Counter(r['status'] for r in rows)),unique_draw_status_counts=dict(Counter(d['status'] for d in draws.values())),
        actual_new_model_shots=sum(d['actual_shots'] for d in rechecked.values()),maximum_new_model_shots=spec['maximum_new_model_shots'],
        actual_new_QPU_shots=0,completed_unique_circuits=completed_circuits,verified_unique_draws=len(rechecked),
        simulation_CPU_seconds_sum=actual_sim_CPU,simulation_wall_seconds_sum_not_campaign_elapsed=actual_sim_wall,
        audit_CPU_seconds=process_time()-cpu,audit_wall_seconds=perf_counter()-tick,
        CI_energy_verified=False,full_paper_superiority=False,goal_complete=False,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
        audit_SHA256=sha256(Path(__file__).read_bytes()))
    save(ROOT/'analysis/N17-sampling-stage-inventory.json',result)
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)
    write(ROOT/'analysis/N17-all108-sampling-stage.csv',stream.getvalue().encode('utf-8-sig'))
    print(result['status'],result['logical_status_counts'],result['actual_new_model_shots'],flush=True)


if __name__=='__main__':run()
