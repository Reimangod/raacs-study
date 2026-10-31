"""Audit all N20 frozen points, actual saved roots, aliases, limits and costs."""
import csv
from io import StringIO
from collections import Counter
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from .nested_ci_study import LOCAL
from .nested_ci_kernel import embed_root
from .nested_ci_plan import frequency_basis
from .analyze import reference


def check_root(path,task,row,physical):
    if sha256(path.read_bytes())!=row['root_SHA256']:raise RuntimeError('Saved actual root changed')
    with np.load(path,allow_pickle=False) as a:saved={k:a[k] for k in a.files}
    expected=np.array(task['strings'],np.int64); n=physical['norb']; ne=tuple(physical['nelec'])
    holes=all(e>n/2 for e in ne)
    solver_strings=np.sort(((1<<n)-1)^expected) if holes else expected
    # Explicit branches avoid mixing electron and particle-hole dimensions.
    if str(saved['problem_id'])!=task['problem_id'] or str(saved['task_id'])!=task['task_id'] or int(saved['norb'])!=n or bool(saved['particle_hole_used'])!=holes:
        raise RuntimeError('Root context differs')
    if tuple(saved['electron_nelec'])!=ne or tuple(saved['solver_nelec'])!=(tuple(n-e for e in ne) if holes else ne):
        raise RuntimeError('Root electron/hole sector differs')
    if not np.array_equal(saved['electron_strings'],expected) or not np.array_equal(saved['solver_strings'],solver_strings):
        raise RuntimeError('Saved root basis differs from frozen source space')
    v=saved['coefficients']
    if v.shape!=(len(expected)**2,) or not np.isfinite(v).all() or abs(v@v-1)>1e-10:
        raise RuntimeError('Actual root dimensions/norm differ')
    result=row['row']
    if (result['actual_CI_dimension']!=len(v) or result['actual_half_dimension']!=len(expected)
        or result['residual_norm']>1e-8 or max(result['random_bilinear_hermiticity_errors'])>1e-10
        or max(result['independent_RDM_expectation_errors'])>1e-8
        or abs(result['penalized_energy_Ha']-result['energy_Ha']-.1*result['spin_square'])>1e-8
        or abs(float(saved['penalized_energy_Ha'])-result['penalized_energy_Ha'])>1e-12):
        raise RuntimeError('Root numerical/physical certificate differs')
    return saved


def run(local=LOCAL,label='N20'):
    LOCAL=local
    spec=load(LOCAL/'protocol.json');c=LOCAL/'collected';terminal=load(c/'terminal.json');started=load(c/'started.json')
    if started['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()) or started['pyscf_threads']!=1 or set(started['thread_env'].values())!={'1'} or started['versions']!=spec['packages']:
        raise RuntimeError('Actual protocol/runtime/thread binding differs')
    for name,h in spec['input_SHA256'].items():
        if sha256((LOCAL/name).read_bytes())!=h:raise RuntimeError('Frozen N20 input changed')
    if sha256((ROOT/'frontierloop/nested_ci_plan.py').read_bytes())!=spec['rule_SHA256']:
        raise RuntimeError('Source-only selection rule changed')
    tasks={t['task_id']:t for t in spec['tasks']};actual={t['task_id']:t for t in terminal['actual_tasks']}
    for case,d in spec['physical'].items():
        src=spec['sources'][case]['source']
        for key,h in (('payload','payload_SHA256'),('source_result','source_result_SHA256')):
            if sha256(Path(src[key]).read_bytes())!=src[h]:raise RuntimeError('Original source evidence changed')
        with np.load(src['payload'],allow_pickle=False) as a:
            if int(a['counts'].sum())!=100000:raise RuntimeError('Original observation budget differs')
            for p in (p for p in spec['points'] if p['case']==case):
                expected=frequency_basis(a['blocked_ids'],a['counts'],d['norb'],d['nelec'],p['requested_R'])
                if tasks[p['task_id']]['strings']!=expected:raise RuntimeError('Frozen basis not reproduced from actual source')
    if len(tasks)!=11 or len(actual)!=len(terminal['actual_tasks']) or len(actual)!=terminal['attempted_unique_tasks'] or not set(actual)<=set(tasks):
        raise RuntimeError('Actual attempt/root identity mismatch')
    if len(terminal['points'])!=16 or [{k:r[k] for k in p} for r,p in zip(terminal['points'],spec['points'])]!=spec['points']:
        raise RuntimeError('Full frozen16 point order must remain intact')
    roots={};body_CPU=0.;complete=0
    for tid,r in actual.items():
        folder=c/'cells'/tid
        if r['status']=='COMPLETED':
            if load(folder/'result.json')!=r:raise RuntimeError('Actual task result differs')
            roots[tid]=check_root(folder/'root.npz',tasks[tid],r,spec['physical'][r['case']]);complete+=1
            body_CPU+=r['CPU_seconds']
            if tasks[tid]['requested_R']==128 and (not r['R128_parity_errors'] or max(r['R128_parity_errors'].values())>1e-8):
                raise RuntimeError('R128 prior-root parity missing')
        else:
            file=folder/('result.json' if r['status']=='FAILED_PRESERVED' else 'process-failure.json')
            if load(file)!=r:raise RuntimeError('Preserved failure missing')
            if r.get('CPU_seconds') is not None:body_CPU+=r['CPU_seconds']
    if complete!=terminal['completed_new_roots'] or body_CPU>terminal['CPU_seconds_children']+1e-5:
        raise RuntimeError('Actual work/unique completed root mismatch')
    for tid,r in actual.items():
        warm=r.get('warm_task_id')
        if r['status']=='COMPLETED' and warm:
            if warm not in roots or tasks[warm]['problem_id']!=tasks[tid]['problem_id']:raise RuntimeError('Invalid warm predecessor')
            embedded=embed_root(roots[warm],roots[tid]['solver_strings'])
            if abs(embedded@embedded-1)>1e-10 or r['row']['embedded_parent_witness_error_Ha']>1e-8 or r['row']['penalized_energy_Ha']>actual[warm]['row']['penalized_energy_Ha']+1e-8:
                raise RuntimeError('Nested parent witness/variational objective differs')
    prior=load(ROOT/'analysis/N18-half-source-native-summary.json')
    if sha256((ROOT/'analysis/N18-half-source-native-summary.json').read_bytes())!=spec['prior_N18_summary_SHA256']:
        raise RuntimeError('Frozen comparator results changed')
    declared=load(ROOT/'analysis/N15-declared-coordinate-bindings.json')['rows']
    rows=[];seen=set();cumulative={case:0. for case in spec['sources']}
    for p in terminal['points']:
        case=p['case'];tid=p['task_id'];r=actual.get(tid);good=p['status']=='COMPLETED'
        if p['execution_origin_task_id'] is not None and (r is None or p['status']!=r['status']):raise RuntimeError('Point/attempt status differs')
        if p['new_CI_solve']:
            if tid in seen:raise RuntimeError('Duplicate numerical attempt accounting')
            seen.add(tid)
            if r.get('CPU_seconds') is None:cumulative[case]=None
            elif cumulative[case] is not None:cumulative[case]+=r['CPU_seconds']
        src=spec['sources'][case]['source'];metrics=src['sampler_resources']
        previous=next(x for x in prior['rows'] if (x['case'],x['method'],x['arm'])==(case,'FULL_LUCJ_L1','SOURCE_FREQUENCY_R128'))
        strong=next(x for x in prior['rows'] if (x['case'],x['method'],x['arm'])==(case,'FULL_LUCJ_L1','JOINT_TOP_D16384'))
        hf,ref=reference(case);energy=r['row']['energy_Ha'] if good else None
        row=dict(case=case,requested_R=p['requested_R'],actual_R=p['actual_half_dimension'],actual_CI_dimension=p['actual_CI_dimension'],
            task_id=tid,status=p['status'],new_numerical_attempt=p['new_CI_solve'],alias_of_same_task=not p['new_CI_solve'] and p['execution_origin_task_id'] is not None,
            energy_Ha=energy,penalized_energy_Ha=r['row']['penalized_energy_Ha'] if good else None,spin_square=r['row']['spin_square'] if good else None,
            energy_delta_vs_R128_mHa=1000*(energy-previous['energy_Ha']) if good else None,
            energy_delta_vs_same_source_N18_HCI_mHa=1000*(energy-strong['energy_Ha']) if good else None,
            same_source_N18_HCI_actual_D=strong['actual_CI_dimension'],
            reference_gap_mHa=1000*(energy-ref) if good else None,recovery_pct=100*(hf-energy)/(hf-ref) if good else None,
            reference_kind=previous['reference_kind'],reference_precision=previous['reference_precision'],
            root_CPU_seconds=r.get('CPU_seconds') if r is not None else None,
            cumulative_known_case_body_CPU_seconds=cumulative[case],
            root_matvec_calls=r['row']['matvec_calls'] if good else None,peak_RSS_KiB=r.get('peak_RSS_KiB') if r else None,
            error=r.get('error') if r else None,root_SHA256=r.get('root_SHA256') if r else None,
            model_shots_reused=100000,new_model_shots=0,new_QPU_shots=0,
            CNOT=metrics['cnot_count'],CNOT_Depth=metrics['cnot_depth'],Total_Depth=metrics['total_unitary_depth'],
            source_declared_ffsim_coordinates=next(x['original_source_ffsim_declared_coordinates'] for x in declared if x['case']==case and x['method'].endswith('L1')),
            independent_ansatz_parameters=None,source_and_circuit_resources_changed=False,
            same_frozen_full_LUCJ_source=True,statistical_superiority_or_equivalence_certified=False)
        rows.append(row)
    if len(seen)!=len(actual):raise RuntimeError('Missing or doubled attempted task')
    out=dict(status=label+'_ALL16_POINTS_AND_ALL_ACTUAL_SAVED_ROOTS_AUDITED',rows=rows,
        status_counts=dict(Counter(r['status'] for r in rows)),attempted_unique_tasks=len(actual),completed_new_roots=complete,
        CPU_seconds_children=terminal['CPU_seconds_children'],CPU_seconds_supervisor=terminal['CPU_seconds_supervisor'],
        wall_seconds=terminal['wall_seconds'],new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,
        source_construction_sampling_and_analysis_costs_additional=True,full_paper_reproduction=False,
        matched_work_between_methods=False,source_improvement_or_best_paper_superiority_established=False,goal_complete=False,
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),terminal_SHA256=sha256((c/'terminal.json').read_bytes()))
    save(ROOT/f'analysis/{label}-nested-CI-summary.json',out)
    stream=StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(ROOT/f'analysis/{label}-all16-nested-CI.csv',stream.getvalue().encode('utf-8-sig'))
    lines=[f'# {label}：同じLUCJ観測のCI空間・精度・費用','',
        '全4条件・16予定点を状態付きで記録。source circuit・100k観測・順位・物理式は固定。R上限だけを変えた。',
        f"状態：{out['status_counts']}。実attempt {out['attempted_unique_tasks']}、完了した新root {complete}。",
        f"子process CPU {out['CPU_seconds_children']:.3f}秒、全wall {out['wall_seconds']:.3f}秒。追加ShotとGPU割当は0。",'',
        '| 条件 | R上限→実R | 実CI D | 状態 | energy Ha | R128との差 mHa | 同source HCIとの差 mHa | Recovery % | S² | root CPU秒* | 累積case body CPU秒* |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    def f(x,n=3):return '未取得' if x is None else f'{x:.{n}f}'
    for r in rows:
        status=r['status']+('（同じroot再利用）' if r['alias_of_same_task'] else '')
        lines.append('| '+' | '.join([r['case'],f"{r['requested_R']}→{r['actual_R']}",str(r['actual_CI_dimension']),status,
            f(r['energy_Ha'],9),f(r['energy_delta_vs_R128_mHa']),f(r['energy_delta_vs_same_source_N18_HCI_mHa']),f(r['recovery_pct']),f(r['spin_square'],6),
            f(r['root_CPU_seconds']),f(r['cumulative_known_case_body_CPU_seconds'])])+' |')
    lines+=['','## 読み方と限界','',
        '- 負のenergy差はR128より低いphysical H energy。nested witnessで検査した単調性はH+0.1S²の最適化目的であり、H単独の一般的保証ではない。',
        '- 弱いfrequency R128だけでなく、同sourceへ既に適用した強いN18 HCI jointのenergy差も示した。R128との差をすべて新しいCI方式の固有利益としない。大きいCIの利益と選択方式の利益を区別する。',
        '- * root CPUは当該rootを得たtask bodyの記録。alias行には元のrootの費用が再掲されるが、再計算費用として加算しない。warm-startは前段を要するため累積case body CPUも示した。全子process CPUにはimport/失敗も含む。',
        '- sourceの初期化・過去のsampling/compile・検証/回収/解析は別であり、上表だけをcold end-to-end総費用としない。未知CPUを0へ補完しない。',
        '- 同じcountsの事後readout診断で、独立quantum confirmationや新しいsamplerではない。回路資源・ansatz Parameterは変えていない。Parameter数が未認証でも、CI root係数数をansatz Parameterへ読み替えない。',
        '- CSVのsource_declared_ffsim_coordinatesは元の宣言chartの座標数（Naph212、N₂531、Fe824）。最小独立rankではなく、N15の別chartの162/385/574と混ぜて削減率を作らない。readout中の最適化変数0を回路Parameters0と呼ばない。',
        '- 失敗・時間切れ・停止後の未実行も16点の分母に残す。root数とlogical点数、量子Shotと古典CI作業を区別する。',
        '- source pool飽和で同じ基底になるRはalias。同じenergyを別の独立成功としない。',
        '- 今回はfull LUCJのみの診断。他source/強い古典対照にも同じ大きい空間・費用機会を与えるまでは、方法間の優越を主張できない。',
        '- Fe参照は精度未認証のSCI、N₂ archive FCIの精度認証も未完。referenceの種別・検証状態はCSVに保持し、選択・順位には使っていない。',
        '- 原著の最高TN sourceやrandom batch/Diceの完全再現ではない。最強原著への総合優越、実機Shot優位性、全研究goal達成は未証明。']
    write(ROOT/f'analysis/{label}_NESTED_CI_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(out['status'],out['status_counts'],flush=True)


if __name__=='__main__':run()
