"""Add executed Fe2 qualification to the preserved full-objective audit.

The local-input absence assessment in v2 is not proof of public unavailability.
No old audit or numerical result is edited; this only supersedes applicability.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .current_evidence import ROOT,OBJECTIVE,canonical,csv_text,save_once,sha


def build(output=None):
    old=ROOT/'results/full-objective-audit-2026-10-23-v2'
    out=Path(output or ROOT/'results/full-objective-audit-2026-10-23-v3');out.mkdir(exist_ok=True)
    parent=json.loads((old/'audit.json').read_text())
    if sha(OBJECTIVE)!=parent['original_objective_sha256']:raise ValueError('Original goal changed')
    for path,expected in parent['evidence_sha256'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Parent scientific evidence drift: '+path)
    with (old/'original-objective-items.csv').open() as f:items=list(csv.DictReader(f))
    with (old/'requirement-judgments.csv').open() as f:judgments=list(csv.DictReader(f))
    fe=ROOT/'results/FeS-feasibility-pilot-v1'
    result=json.loads((fe/'completion.json').read_text())
    check=json.loads((fe/'RAACS-full-cost/independent-validation.json').read_text())
    source=json.loads((fe/'public-inputs/provenance.json').read_text())
    problem=json.loads((fe/'physical-problem/problem.json').read_text())
    cfg=json.loads((fe/'manifest.json').read_text())
    if result['status']!='QUALIFICATION_EXECUTED_AND_VALIDATED' or result['actual_model_samples']!=2000:
        raise ValueError('Actual Fe pilot not completed')
    if check['maximum_energy_difference_Ha']>1e-8 or problem['n_qubits']!=40:
        raise ValueError('Missing Fe physical qualification')
    if source['Fe4']['header']['n_qubits']!=72 or source['Fe4']['active_space_truncated']:
        raise ValueError('Fe4 applicability misrepresented')
    for path,expected in cfg['implementation'].items():
        if sha(ROOT/'benchmark'/path)!=expected:raise ValueError('Bound Fe numerical code drift')
    bound=['study-v1/FE_S_BOUNDED_FEASIBILITY.md']
    bound += ['results/FeS-feasibility-pilot-v1/'+p for p in (
        'manifest.json','completion.json','execution-binding.json','public-inputs/provenance.json',
        'public-inputs/fcidump_Fe2S2_MO.txt','public-inputs/Fe4S4-header.txt',
        'physical-problem/problem.json','physical-problem/integrals.npz',
        'RAACS-full-cost/config.json','RAACS-full-cost/input-provenance.json',
        'RAACS-full-cost/events.sqlite','RAACS-full-cost/result.json',
        'RAACS-full-cost/independent-validation.json','same-circuit-bonds/result.json','report-v1/manifest.json')]
    evidence=dict(parent['evidence_sha256'])
    for path in bound:evidence[path]=sha(ROOT/path)
    group=next(j for j in judgments if j['group']=='E1C')
    group.update(status='FE2_BOUNDED_QUALIFICATION_EXECUTED_FE4_OUTSIDE_CURRENT_REPRESENTATION',
        evidence=';'.join(bound),interpretation_scope=(
        'Public author physical inputs exist. Fe2 CAS(30e,20o)/40q:2000total model samples,4operators,'
        'd8,48CNOT/29depth,independent64-entry root check andsamecircuit64/128/256MPS convergence. '
        'One occupation-reference exploratory pilot,notoptimizedRHF/groundaccuracy/S2/quantumadvantage. '
        'Fe4 header CAS(54e,36o)/72q exceeds qualified62-bit route;notexecuted,notinputabsence.'))
    changed=[]
    for row in items:
        if row['group']=='E1C':
            row.update(assessment=group['status'],evidence=group['evidence']);changed.append(int(row['line']))
    applicability=dict(parent['conditional_applicability'])
    applicability.update(qualified_Fe_cluster_input_and_backend_available='Fe2_BOUNDED_SHALLOW_ONLY',
        registered_Fe_bearing_problems=1,
        reason='Pinned public Fe2 integrals and actual bounded40q circuit/count/CI qualification now available. Fe4 is72q, outside current62-bit route.',
        Fe2={'qubits':40,'status':result['status'],'reference_semantics':problem['reference_semantics'],
            'ground_state_reference':None,'S_squared_certified':False},
        Fe4=source['Fe4'],scope='Availability assessed against actual public pinned physical data and executed constrained route; not generic impossibility.')
    record=dict(schema='full-objective-evidence-audit-v3',
        status='FE_PUBLIC_INPUT_APPLICABILITY_CORRECTED_AND_FE2_EXECUTION_BOUND',
        original_objective_sha256=sha(OBJECTIVE),parent_audit_sha256=sha(old/'audit.json'),
        original_items=len(items),updated_lines=changed,unmapped_items=0,
        sealed_preservation_audit_reused_not_reexecuted=parent['sealed_preservation_audit_reused_not_reexecuted'],
        evidence_sha256=evidence,conditional_applicability=applicability,
        scientific_claims_not_established=parent['scientific_claims_not_established'],
        script_sha256=sha(__file__),new_experiments_by_auditor=0,new_eigensolver_calls_by_auditor=0,
        Fe2_experiment_and_quick_are_separate_recorded_work=True,hardware_shots=0,
        whole_goal_complete=False,remaining_work='Final reproducibility code/data entry packaging and completion decision; no automatic research-performance success assertion.')
    save_once(out/'audit.json',canonical(record));save_once(out/'original-objective-items.csv',csv_text(items))
    save_once(out/'requirement-judgments.csv',csv_text(judgments))
    save_once(out/'READ_FIRST_JA.md',
        '# 全目的監査v3：公開Fe入力と実試行を反映\n\n'
        '旧v1/v2を保持し、E1Cの入力・実行可能性の判断を更新した。'
        '公開Fe2積分は存在し、40q/2000model samplesの上限付き実行と独立検算が完了。'
        'Fe4入力も存在するが72qで現在の認定62-bit経路を超えるため未実施。'
        '入力不在と表現上限を混同しない。\n\n'
        'Fe2は4演算子/d8/48CNOT/29Depthの浅い単一seed feasibility。'
        '初期占有配置からの低下であり、optimizedRHF、singlet ground accuracy、'
        'chemical accuracy、CEO/LUCJ優位の証明ではない。\n\n'
        '367元項目と旧科学結果を保持。既存親evidenceのSHAを再確認したが、'
        '全45,831旧artifactの再ハッシュを再実行したとは数えない。'
        '本auditorは新しい実験・eigensolveを行わず、別記録の実行結果を結合した。'
        '全目標完了の最終判定と再現用code/data入口の整理はまだ残る。\n')
    return {'status':record['status'],'original_items':len(items),'updated_lines':changed,'whole_goal_complete':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path)
    print(json.dumps(build(p.parse_args().output)))
