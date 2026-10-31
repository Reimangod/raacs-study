"""N76 historical source costs joined by physical bytes, not method names alone."""
from pathlib import Path
from .paths import ROOT
from .io import load,save,write,sha256,digest
from .fixed_confirmation_execution import LOCAL,bound,capacity
from .source_cost_audit import unique_subtotal

def build():
    science,b=bound();oldfile=ROOT/'analysis/SOURCE-cost-components-v1.json';old=load(oldfile)
    for file,h in old['evidence_SHA256'].items():
        if sha256(Path(file).read_bytes())!=h:raise RuntimeError('Historical cost evidence drift '+file)
    components={c['id']:c for c in old['components']};used={};rows=[];bindings={str(oldfile):sha256(oldfile.read_bytes())}
    s15=load(ROOT/'runs/N15-SLATER-PREFIX-ROUNDING-PREFLIGHT-v1/protocol.json')
    s74=load(ROOT/'runs/N74-ACTIVITY-FINITE-v1/protocol.json')
    for cell in science['cells']:
        case,method=cell['case'],cell['method'];selected=[]
        row=dict(case=case,method=method,source_status=cell['status'],historical_initializer_components=[],
            known_initializer_CPU_seconds=None,historical_adaptive_construction_CPU_seconds=None,
            historical_B4_pilot_model_samples=cell['historical_one_off_pilot_model_cost'],
            all_LUCJ_initializer_stages_converged=None,initializer_failed_stages=None,
            full_cold_source_CPU_seconds=None,full_cold_source_cost_reconstructed=False,
            new_molecular_evaluations=0,new_model_samples=0)
        if cell['status']=='MISSING_SAME_PROBLEM_SOURCE':row['cost_scope']='MISSING_SOURCE_NOT_ZERO_COST'
        elif method.startswith('C5_C_LUCJ'):
            source=next(s for s in s15['sources'] if s['case']==case and s['method']==method)
            cost=next(r for r in old['rows'] if r['case']==case and r['method']==method)
            assert source['source_result_SHA256']==cost['source_result_SHA256'] and source['source_result']==cost['source_result']
            selected=[components[c] for c in cost['initializer_component_ids']]
            row.update(all_LUCJ_initializer_stages_converged=cost['all_initializer_stages_converged'],initializer_failed_stages=cost['initializer_failed_stages'],
                old_source_result=cost['source_result'],old_source_result_SHA256=cost['source_result_SHA256'],
                known_compilation_or_preparation_CPU_seconds=cost['compilation_or_preparation_CPU_seconds'],
                cost_scope='OLD_INITIALIZER_SUBTOTAL_NOT_OLD_INCLUSIVE_ATTEMPT_NOT_FULL_COLD_NOT_NEW_SAMPLING_COST')
        elif method=='ACTIVE_GROUP_FIXED_LIBRARY':
            source=next(s for s in s74['sources'] if s['case']==case);amplitude=Path(source['amplitudes']);file=amplitude.with_name('result.json');r=load(file);h=sha256(file.read_bytes())
            assert r['problem_id']==cell['problem_id'] and r['amplitudes_sha256']==sha256(amplitude.read_bytes())
            c=dict(id=digest([str(file.resolve()),h,'CPU_process_seconds']),artifact=str(file),artifact_SHA256=h,field='CPU_process_seconds',phase='fresh_amplitudes',CPU_seconds=r['CPU_process_seconds'])
            selected=[c];row.update(amplitude_initializer_method=r['selected_method'],
                cost_scope='OLD_FRESH_AMPLITUDE_SUBTOTAL_ONLY_NOT_ACTIVITY_LIBRARY_SEARCH_OR_QPY_COMPILATION_OR_COMPLETE_COLD_COST')
            bindings[str(file)]=h;bindings[str(amplitude)]=r['amplitudes_sha256']
        elif method=='B4_FIXED_SELECTED_ENSEMBLE':
            row.update(historical_adaptive_construction_CPU_seconds=cell['source_input']['historical_construction_CPU_seconds'],
                cost_scope='OLD_ONE_OFF_ADAPTIVE_BANK_CONSTRUCTION_INCLUSIVE_OF_PILOT_WORK_NOT_ADDED_TO_NESTED_PILOT_CPU')
        elif method.startswith('SQDRIFT'):
            row['cost_scope']='FIXED_HAMILTONIAN_PACKET_SOURCE_OLD_EXPORT_FACTOR_GROUP_AND_COMPILER_COMPLETE_COLD_COST_NOT_RECONSTRUCTED_NOT_ASSUMED_ZERO'
        else:row['cost_scope']='FRESH_NATIVE_HCI_BUILD_CHARGED_FROM_HCI_TERMINAL_SUPPLIED_HAMILTONIAN_AND_ORBITAL_PREPROCESS_COST_UNKNOWN'
        for c in selected:
            assert sha256(Path(c['artifact']).read_bytes())==c['artifact_SHA256']
            assert load(c['artifact'])[c['field']]==c['CPU_seconds']
            if c['id'] in used:assert used[c['id']]==c
            used[c['id']]=c
        row['historical_initializer_components']=[c['id'] for c in selected]
        row['known_initializer_CPU_seconds']=unique_subtotal(selected);rows.append(row)
    assert len(rows)==32
    return dict(status='N76_SOURCE_COST_LINEAGE_RECOVERED_NOT_FULL_COLD_OR_MATCHED_WORK',rows=rows,unique_initializer_components=list(used.values()),
        unique_initializer_CPU_seconds_not_sum_of_method_rows=unique_subtotal(list(used.values())),bindings=bindings,
        historical_components_not_new_CPU_executions=True,new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,goal_complete=False)

def run():
    capacity();target=LOCAL/'source-cost-lineage-v1.json'
    if target.exists():raise RuntimeError('Existing cost supplement; no overwrite')
    result=build();save(target,result)
    names={'ACTIVE_GROUP_FIXED_LIBRARY':'相関seed＋活性group','B4_FIXED_SELECTED_ENSEMBLE':'B4固定bank','CLASSICAL_HF_HCI':'古典HF-HCI'}
    lines=['# N76：過去source構築費用の対応付け','',
        '新しい分子計算は行わず、確認対象と一致する元artifactへ費用を結び付けた。金額・CPU総量が不明な欄を0としていない。','',
        '| 条件 | 方法 | 既知の初期化小計 CPU秒 | 旧adaptive構築 CPU秒 | LUCJ初期化で上限等に達したstage数 |','|---|---|---:|---:|---:|']
    def fmt(x):return '不明／該当なし' if x is None else f'{x:.6f}'
    for r in result['rows']:lines.append(f"| {r['case']} | {names.get(r['method'],r['method'])} | {fmt(r['known_initializer_CPU_seconds'])} | {fmt(r['historical_adaptive_construction_CPU_seconds'])} | {r['initializer_failed_stages'] if r['initializer_failed_stages'] is not None else '—'} |")
    lines+=['',f"同じ初期化artifactを一度だけ数えた小計：{result['unique_initializer_CPU_seconds_not_sum_of_method_rows']:.6f} CPU秒。表の行を単純に合計すると、複数methodが共有するamplitude生成を二重計上する。",'',
        'LUCJの初期化には最適化反復上限で終了したstageがある。保存された固定回路を無効とするものではないが、「収束した最強LUCJ baseline」の主張はできない。CCSD等のamplitude生成の収束と、LUCJへの圧縮fitの収束も区別する。','',
        'ACTIVEの既知小計は元amplitude生成だけで、古いactivity ranking・source library検査・回路合成を含む完全費用ではない。SqDRIFTの古いexport/group/compile等の全費用も0とは置かない。B4の旧構築CPUにはpilotに伴う作業が含まれるため、その内訳を再加算しない。','',
        'これは過去費用のprovenanceであり、新しいCPU計算を実行した回数ではない。新sampling、HCI、compiler、最終CIの実測費用とは範囲を分けて報告する。完全cold費用および完全matched-workは未認定。','']
    write(ROOT/'analysis/N76_SOURCE_COST_LINEAGE_JA.md','\n'.join(lines).encode());print('N76_COST_LINEAGE32_ROWS',result['unique_initializer_CPU_seconds_not_sum_of_method_rows'],flush=True)

if __name__=='__main__':run()
