"""N76 final tables: all fixed endpoints, actual dimensions, costs, missingness."""
from collections import Counter
from io import StringIO
from pathlib import Path
import csv,sys
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256,digest
from .fixed_confirmation_execution import LOCAL,bound,capacity
from .analyze import reference

NAMES={'ACTIVE_GROUP_FIXED_LIBRARY':'相関seed＋活性group（固定）','B4_FIXED_SELECTED_ENSEMBLE':'B4残差適応bank（固定）',
    'C5_C_LUCJ_COMPRESSED_L1':'Compressed LUCJ 1層','C5_C_LUCJ_COMPRESSED_L2':'Compressed LUCJ 2層','C5_C_LUCJ_COMPRESSED_L4':'Compressed LUCJ 4層',
    'SQDRIFT_N5_FIXED30':'SqDRIFT N5・30回路','SQDRIFT_N20_FIXED30':'SqDRIFT N20・30回路','CLASSICAL_HF_HCI':'古典HF-HCI'}
CASES={'NAPH20-BLOCK-BOYS':'ナフタレン 20q','N2-32-R1p2-P2':'N₂ 1.2 Å・32q','N2-32-R2p4-P2':'N₂ 2.4 Å・32q','FE2S2-40-P2':'Fe₂S₂ 40q'}

def stats(values):
    good=[float(x) for x in values if x is not None]
    return dict(n=len(good),mean=float(np.mean(good)) if good else None,minimum=min(good) if good else None,maximum=max(good) if good else None)

def number(value,digits=3):return '未取得' if value is None else f'{value:.{digits}f}'

def compare(proposal,control):
    """Record component differences; never infer strongest-paper/matched-work win."""
    fields=['energy_Ha','CNOT','CNOT_Depth','Total_Depth','actual_D']
    differences={key:(proposal[key]-control[key]) if proposal.get(key) is not None and control.get(key) is not None else None for key in fields}
    comparable=all(differences[k] is not None for k in fields)
    if not comparable:status='INCOMPLETE_OR_CLASSICAL_RESOURCE_NOT_APPLICABLE'
    elif differences['energy_Ha']<=1e-8 and all(differences[k]<=0 for k in fields[1:]) and any(differences[k]<0 for k in fields[1:]):status='DESCRIPTIVE_ENERGY_RESOURCE_AND_DIMENSION_NONINFERIOR_NOT_TOTAL_WORK_SUPERIORITY'
    else:status='TRADEOFF_OR_NO_COMPONENTWISE_ADVANTAGE'
    return dict(status=status,differences=differences,complete_matched_work_or_highest_paper_claim=False)

def run():
    science,b=bound();capacity();terminal=load(LOCAL/'CI-terminal.json');p=load(LOCAL/'CI-binding.json');compiler=load(LOCAL/'compiler-qualified-point-selection-v2.json')
    if terminal['status']!='ALL320_ENDPOINT_STATUSES_TERMINAL' or len(terminal['rows'])!=320:raise RuntimeError('Final CI is not terminal; no final report')
    output=ROOT/'analysis/N76-fixed-confirmation-summary-v2.json'
    if output.exists():raise RuntimeError('Existing final summary; inspect without overwrite')
    for file,h in p['bindings'].items():
        if sha256(Path(file).read_bytes())!=h:raise RuntimeError('Confirmation input drift '+file)
    cells={(c['case'],c['method']):c for c in science['cells']};rr={(r['case'],r['method']):r for r in compiler['aggregate']}
    flat=[];seen=set();roots=set()
    assert {e['endpoint_id'] for e in science['endpoints']}=={e['endpoint_id'] for e in terminal['rows']}
    for row in terminal['rows']:
        key=(row['case'],row['method']);cell=cells[key];res=rr.get(key);energy=row.get('energy_Ha');hf,ref=reference(row['case']);root=row.get('root',{})
        if row['status']=='COMPLETED':
            assert energy is not None and root['result']['residual_norm']<=1e-8
            assert max(root['independent_RDM_audit']['errors'].values())<=1e-7
            f=Path(root['root_file']);assert sha256(f.read_bytes())==root['root_SHA256']
            assert row['actual_D']==row['actual_R']**2 and row['basis_id']==digest([cell['problem_id'],row['strings'],science['common_equation'],science['library_SHA256']])
            roots.add(row['basis_id'])
        if row.get('basis_id'):
            assert row['new_CI_attempt']==(row['basis_id'] not in seen);seen.add(row['basis_id'])
        flat.append(dict(case=row['case'],method=row['method'],replicate=row['replicate'],context=row['context'],status=row['status'],
            energy_Ha=energy,Recovery_pct=100*(hf-energy)/(hf-ref) if energy is not None else None,
            penalized_energy_Ha=root.get('result',{}).get('penalized_energy_Ha'),spin_square=root.get('result',{}).get('spin_square'),
            actual_R=row.get('actual_R'),actual_D=row.get('actual_D'),
            CNOT=res['mean']['CNOT'] if res else None,CNOT_Depth=res['mean']['CNOT_Depth'] if res else None,Total_Depth=res['mean']['Total_Depth'] if res else None,
            numeric_primitive_coefficient_slots_weighted=res['coefficient_slots_weighted_mean'] if res else None,independent_parameter_rank=None,
            VQE_optimized_variables=0 if res else None,
            planned_new_final_model_samples_per_replicate=cell['planned_final_model_samples_per_replicate'],historical_one_off_B4_pilot_model_samples=cell['historical_one_off_pilot_model_cost'],
            campaign_model_samples_including_pilot=cell['campaign_model_cost_including_one_off_pilot'],actual_QPU_shots=0,
            basis_id=row.get('basis_id'),new_CI_attempt=row.get('new_CI_attempt',False),
            reference_scope='UNVERIFIED_SCI_NOT_FCI_RECOVERY_IS_PROVISIONAL' if row['case']=='FE2S2-40-P2' else 'EXISTING_SAME_PROBLEM_REFERENCE',
            deterministic_classical_display_not_independent_replicate=row['method']=='CLASSICAL_HF_HCI'))
    groups=[]
    for cell in science['cells']:
        for context in ['R8','R16','R32','R64','COMMON_ACTUAL_R32']:
            rows=[r for r in flat if (r['case'],r['method'],r['context'])==(cell['case'],cell['method'],context)]
            groups.append(dict(case=cell['case'],method=cell['method'],context=context,status_counts=dict(Counter(r['status'] for r in rows)),
                energy_Ha=stats([r['energy_Ha'] for r in rows]),Recovery_pct=stats([r['Recovery_pct'] for r in rows]),actual_R=[r['actual_R'] for r in rows],actual_D=[r['actual_D'] for r in rows],
                source_fixed_not_independent_trajectory_replicates=True))
    contrasts=[]
    for prop in science['proposals']:
        for r in [r for r in flat if r['method']==prop]:
            for c in [c for c in flat if c['case']==r['case'] and c['replicate']==r['replicate'] and c['context']==r['context'] and c['method'] not in science['proposals']]:
                contrasts.append(dict(case=r['case'],proposal=prop,control=c['method'],replicate=r['replicate'],context=r['context'],**compare(r,c)))
    m4=load(LOCAL/'m4-terminal.json');hci=load(LOCAL/'hci/terminal.json');gpu=load(LOCAL/'gpu/terminal.json')
    lineagefile=LOCAL/'source-cost-lineage-v1.json';lineage=load(lineagefile)
    for file,h in lineage['bindings'].items():
        if sha256(Path(file).read_bytes())!=h:raise RuntimeError('Source cost lineage drift '+file)
    assert len(lineage['rows'])==32
    gpucost=[]
    for f in sorted((LOCAL/'gpu/collected/outputs').glob('*/result.json')):
        rec=load(f);gpucost.append(dict(input_id=rec['input_id'],CPU_seconds=rec['whole_body_CPU_seconds'],wall_seconds=rec['whole_body_wall_seconds'],job_id=rec['job_id']))
    costs=dict(M4_source_CPU_seconds=m4['CPU_seconds'],M4_source_wall_seconds=m4['wall_seconds'],M4_CPU_cap_deviation_seconds=max(0,m4['CPU_seconds']-science['source']['M4_CPU_cap']),
        GPU_per_source=gpucost,GPU_summed_CPU_seconds=sum(r['CPU_seconds'] for r in gpucost),GPU_summed_source_wall_seconds_not_campaign_elapsed=sum(r['wall_seconds'] for r in gpucost),
        HCI_native_CPU_seconds=sum(r.get('native_attempt_CPU_seconds',0) for r in hci['rows']),HCI_worker_CPU_seconds=hci['worker_CPU_seconds'],HCI_wall_seconds=hci['wall_seconds'],
        compiler_CPU_seconds=compiler['CPU_seconds'],CI_CPU_seconds=terminal['CPU_seconds'],CI_wall_seconds=terminal['wall_seconds'],
        actual_new_model_samples=m4['actual_new_model_samples']+gpu['actual_new_model_samples'],historical_one_off_pilot=sum(c['historical_one_off_pilot_model_cost'] for c in science['cells']),
        development_source_initialization_tests_transfer_and_full_cold_cost_not_complete=True,
        known_historical_source_cost_lineage=lineage,source_cost_lineage_SHA256=sha256(lineagefile.read_bytes()))
    assert costs['actual_new_model_samples']==p['actual_persisted_new_model_samples']
    summary=dict(status='FINITE_FIXED_SOURCE_CONFIRMATION_TERMINAL_NOT_STRONGEST_PAPER_GOAL_COMPLETE',science_SHA256=sha256(Path(b['science_file']).read_bytes()),endpoint_status_counts=dict(Counter(r['status'] for r in flat)),
        actual_unique_successful_CI_roots=len(roots),rows=flat,groups=groups,contrasts=contrasts,costs=costs,new_source_search_authorized=False,goal_complete=False,
        resource_correction='V2_EXCLUDES_OPAQUE_UNDECOMPOSED_GATES_NOT_ZERO_CNOT_PRESERVES_V1_ERROR_SOURCES_AND_CI_UNCHANGED',
        qualified_resource_file_SHA256=sha256((LOCAL/'compiler-qualified-point-selection-v2.json').read_bytes()),
        limitations=['ONLY_TWO_READOUT_REPLICATES_GIVEN_FIXED_SOURCES','B4_OLD_SELECTION_PILOT_NOT_FRESH_CONSTRUCTION','TWO_NAPH_LUCJ_SOURCE_CELLS_MISSING','UNKNOWN_INDEPENDENT_PARAMETER_RANK','NOT_PAPER_HIGHEST_ENSEMBLE_AND_CI_DIMENSION','INCOMPLETE_COLD_COST','LUCJ_INITIALIZER_FIT_HAD_BOUNDED_NONCONVERGED_STAGES_NOT_FULLY_CONVERGED_STRONGEST_BASELINE','CROSS_FRAMEWORK_COMPILER_PARITY_INCOMPLETE','FE_REFERENCE_UNVERIFIED_SCI','S2_PENALTY_NOT_EXACT_SINGLET'])
    save(output,summary)
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat);write(ROOT/'analysis/N76-fixed-confirmation-all320-v2.csv',stream.getvalue().encode('utf-8-sig'))
    lines=['# 固定確認N76：全条件・全対照・全欠測の結果（資源計数修正版v2）','',
        '初版のLUCJ CNOT=0は、分解前の複合ゲートを基底ゲートとして数えた誤りだった。初版を保全し、このv2を有効な資源比較表とする。既存の同じopt1/3出力を基底ゲート規約で再確認しただけで、source・readout・CI・seed・予算は変更していない。追加の確認CPUも元のcompiler予算内で計上した。','',
        '新しい方策の探索は終了している。以下は固定sourceに対する新しい読出し2反復であり、新しいadaptive構築を2回成功させた実験ではない。全指標・最高論文への総合優越は未証明。','',
        '主表示はR上限32。同じR上限でも実R/Dは違い得る。全R8/16/32/64・共通実R・各反復は付属CSV/JSONに残す。係数slotは独立Parametersではない。']
    for case in CASES:
        lines+=['',f'## {CASES[case]}','',
            '| 方法 | 完了反復/2 | Energy平均 Ha | Energy範囲 Ha | Recovery平均 % | 平均CNOT | 平均CX Depth | 平均Total Depth | 係数slot平均 | 実R（2反復） |',
            '|---|---:|---:|---|---:|---:|---:|---:|---:|---|']
        for cell in [c for c in science['cells'] if c['case']==case]:
            method=cell['method'];g=next(g for g in groups if (g['case'],g['method'],g['context'])==(case,method,'R32'));res=rr.get((case,method));mean=res['mean'] if res else {};e=g['energy_Ha']
            lines.append(f"| {NAMES[method]} | {e['n']} | {number(e['mean'],9)} | {number(e['minimum'],9)} ～ {number(e['maximum'],9)} | {number(g['Recovery_pct']['mean'])} | {number(mean.get('CNOT'),2)} | {number(mean.get('CNOT_Depth'),2)} | {number(mean.get('Total_Depth'),2)} | {number(res['coefficient_slots_weighted_mean'] if res else None,2)} | {g['actual_R']} |")
        valid=[g for g in groups if g['case']==case and g['context']=='R32' and g['energy_Ha']['n']==2]
        if valid:
            best=min(valid,key=lambda g:g['energy_Ha']['mean']);lines+=['',f"この条件で今回の平均energyが最も低かったのは **{NAMES[best['method']]}**。ただしこれだけでは、回路の小ささ・実CI次元・古典構築費用まで含む総合勝利を意味しない。古典HCIの2欄は同じ決定論的sourceの表示で、独立した2実験ではない。"]
    lines+=['','## 費用・欠測・判断の範囲','',f"全320欄の状態：`{summary['endpoint_status_counts']}`。新しい成功CI根は{len(roots)}個で、同じbasisの再表示は新計算に数えていない。",'',
        f"新規model samplesは{costs['actual_new_model_samples']:,}。旧B4構築pilotは{costs['historical_one_off_pilot']:,}を別途一回分として計上。実QPU Shotは0。",'',
        f"Mac source CPU {costs['M4_source_CPU_seconds']:.6f}秒、GPU source合計wall {costs['GPU_summed_source_wall_seconds_not_campaign_elapsed']:.3f}秒（並列campaign elapsedではない）、HCI native CPU {costs['HCI_native_CPU_seconds']:.3f}秒、compiler CPU {costs['compiler_CPU_seconds']:.3f}秒、CI CPU {costs['CI_CPU_seconds']:.3f}秒。",'',
        '既知の旧source初期化費用は[元sourceに対応付けた費用表](N76_SOURCE_COST_LINEAGE_JA.md)とJSONへ別記した。LUCJには回数上限で終わったfit stageがあり、収束した最高性能baselineの証明にはならない。amplitude生成の収束とLUCJ圧縮fitの収束を混同しない。', '',
        '開発・CCSD等の初期構築・tests・転送をすべて含むcold費用は未完成であり、完全matched-workとは呼ばない。既知の共有初期化artifactを複数methodの行数で重複加算しない。Mac CPU300秒枠の軽微な超過もJSONに残した。', '',
        '独立Parameters、最高原著予算、全cross-framework compiler、Fe参照、欠測が残るため、「全指標で最強論文に勝った」という目標は達成認定しない。2反復からp値による一般的優越を作らない。', '',
        '有限確認の結果が弱くても、新しいsource・角度・seed・第3探索ループは自動で追加しない。改善点とtrade-offを限定して報告し、確認・結論の納品でこの探索を閉じる。','']
    write(ROOT/'analysis/N76_FIXED_CONFIRMATION_RESULTS_V2_JA.md','\n'.join(lines).encode());print('N76_FINAL_REPORT_V2_ALL320_FIELDS',summary['endpoint_status_counts'],flush=True)

if __name__=='__main__':run()
